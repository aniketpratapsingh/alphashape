#include "common/algorithm/markers/track_markers_task.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <future>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "base/utils/omp.hpp"
#include "common/algorithm/markers/detect_markers_task.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/core/marker.hpp"
#include "common/core/types.hpp"
#include "image/image.hpp"
#include "image/io/exif.hpp"
#include "image/io/image_io.hpp"
#include "image/operators/downsample.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

constexpr double kMinCorrelation = 0.6;
constexpr double kMinPatchStdDev = 0.02;
constexpr double kMinGradientEnergy = 1e-6;
constexpr std::size_t kBudgetBytes = 1500ull * 1024ull * 1024ull;

bool contains(const std::vector<std::string>& list, const std::string& value) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

const Chunk* find_chunk(const Document& document, std::int32_t chunk_id) {
    for (const auto& workspace : document.workspaces()) {
        if (!workspace) {
            continue;
        }
        for (const auto& chunk : workspace->chunks()) {
            if (chunk && chunk->id() == chunk_id) {
                return chunk.get();
            }
        }
    }
    return nullptr;
}

const Frame* frame_of_camera(const Chunk& chunk, std::int32_t camera_id) {
    for (const auto& frame : chunk.frames()) {
        if (frame && frame->camera_id() == camera_id) {
            return frame.get();
        }
    }
    return nullptr;
}

std::filesystem::path resolve_frame_path(const Document& document,
                                         const std::string& path) {
    const std::filesystem::path recorded(path);
    if (recorded.is_absolute() || document.path().empty()) {
        return recorded;
    }
    return std::filesystem::path(document.path()).parent_path() / recorded;
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's
/// civil-from-days inverse). Only the ordering matters, so no timezone is
/// applied: EXIF date-times are local wall-clock.
std::int64_t days_from_civil(int year, unsigned month, unsigned day) {
    year -= month <= 2 ? 1 : 0;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned year_of_era = static_cast<unsigned>(year - era * 400);
    const unsigned day_of_year =
        (153U * (month + (month > 2 ? -3U : 9U)) + 2U) / 5U + day - 1U;
    const unsigned day_of_era =
        year_of_era * 365U + year_of_era / 4U - year_of_era / 100U + day_of_year;
    return static_cast<std::int64_t>(era) * 146097 +
           static_cast<std::int64_t>(day_of_era) - 719468;
}

/// Strict "YYYY:MM:DD HH:MM:SS" (exactly 19 characters), returned as an
/// integer that sorts the same as the real instant. Anything else is nullopt.
std::optional<std::int64_t> parse_datetime(const std::string& text) {
    constexpr const char* kPattern = "dddd:dd:dd dd:dd:dd";
    if (text.size() < 19U) {
        return std::nullopt;
    }
    for (std::size_t i = 0; i < 19U; ++i) {
        if (kPattern[i] == 'd') {
            if (text[i] < '0' || text[i] > '9') {
                return std::nullopt;
            }
        } else if (text[i] != kPattern[i]) {
            return std::nullopt;
        }
    }
    const auto number = [&text](std::size_t at, std::size_t count) {
        int value = 0;
        for (std::size_t i = 0; i < count; ++i) {
            value = value * 10 + (text[at + i] - '0');
        }
        return value;
    };
    const int year = number(0U, 4U);
    const int month = number(5U, 2U);
    const int day = number(8U, 2U);
    const int hour = number(11U, 2U);
    const int minute = number(14U, 2U);
    const int second = number(17U, 2U);
    if (year < 1970 || year > 9999 || month < 1 || month > 12 || day < 1 ||
        day > 31 || hour > 23 || minute > 59 || second > 60) {
        return std::nullopt;
    }
    const std::int64_t days = days_from_civil(
        year, static_cast<unsigned>(month), static_cast<unsigned>(day));
    return ((days * 24 + hour) * 60 + minute) * 60 + second;
}

/// SubSecTimeOriginal as microseconds. A non-numeric field is ignored (zero)
/// rather than invalidating an otherwise good timestamp.
std::int64_t subsecond_micros(const std::optional<std::string>& subsec) {
    if (!subsec) {
        return 0;
    }
    std::int64_t value = 0;
    int digits = 0;
    for (const char c : *subsec) {
        if (digits == 6) {
            break;
        }
        if (c < '0' || c > '9') {
            return 0;
        }
        value = value * 10 + (c - '0');
        ++digits;
    }
    while (digits < 6) {
        value *= 10;
        ++digits;
    }
    return value;
}

/// Capture time as EXIF records it. The model has no time field, so it is read
/// from the frame file. Missing or malformed metadata is "no time", which
/// makes the whole ordering fall back to import order.
std::optional<CaptureTime> capture_time_of(const Document& document,
                                           const Frame& frame) {
    try {
        // Only the metadata is read: JPEG/PNG segments are streamed and a TIFF
        // uses a bounded prefix, so plan time never loads the pixel data.
        return read_capture_time(
            resolve_frame_path(document, frame.path()).string());
    } catch (const std::exception&) {
        // An unreadable file is not an ordering problem; the decode step
        // reports it (and an explicit camera list throws there).
    }
    return std::nullopt;
}

/// One selected camera in the tracking sequence.
struct SequenceCamera {
    const Camera* camera = nullptr;
    const Calibration* calibration = nullptr;
    const Frame* frame = nullptr;
    /// Sortable capture instant (seconds since epoch scaled by 1e6 plus
    /// SubSecTimeOriginal microseconds); nullopt when unusable.
    std::optional<std::int64_t> capture_key;
    /// Set when the file had no timestamp or it did not parse strictly.
    bool capture_time_bad = false;
    /// Position in `Chunk::cameras()`, which is import order.
    std::size_t import_index = 0;
};

/// Resolve the `cameras` parameter. An explicit list is strict about names,
/// frames and sensors; the implicit list skips cameras that cannot be used.
/// The sequence is ordered by capture time only when every selected camera has
/// one, otherwise it stays in import order. stable_sort keeps ties in import
/// order.
std::vector<SequenceCamera> selected_sequence(
    const Document& document, const Chunk& chunk,
    const std::vector<std::string>& filters) {
    for (const std::string& label : filters) {
        bool found = false;
        for (const Camera& camera : chunk.cameras()) {
            found = found || camera.label == label;
        }
        if (!found) {
            throw ValueError("TrackMarkers: no camera with label '" + label + "'");
        }
    }
    std::vector<SequenceCamera> sequence;
    for (std::size_t index = 0; index < chunk.cameras().size(); ++index) {
        const Camera& camera = chunk.cameras()[index];
        const bool selected =
            filters.empty() ? camera.enabled : contains(filters, camera.label);
        if (!selected) {
            continue;
        }
        const Frame* frame = frame_of_camera(chunk, camera.id);
        if (frame == nullptr) {
            if (!filters.empty()) {
                throw ValueError("TrackMarkers: camera '" + camera.label +
                                 "' has no linked frame");
            }
            continue;
        }
        if (camera.sensor_id < 0 ||
            camera.sensor_id >=
                static_cast<std::int32_t>(chunk.sensors().size())) {
            if (!filters.empty()) {
                throw ValueError("TrackMarkers: camera '" + camera.label +
                                 "' has no sensor");
            }
            continue;
        }
        const Calibration& calibration =
            chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)]
                .calibration;
        if (calibration.width <= 0 || calibration.height <= 0) {
            if (!filters.empty()) {
                throw ValueError("TrackMarkers: camera '" + camera.label +
                                 "' has no image size");
            }
            continue;
        }
        SequenceCamera entry;
        entry.camera = &camera;
        entry.calibration = &calibration;
        entry.frame = frame;
        entry.import_index = index;
        sequence.push_back(std::move(entry));
    }
    // Capture times are independent per frame, so read and parse the metadata
    // in parallel: even the bounded read is file I/O per camera.
    const int count = static_cast<int>(sequence.size());
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < count; ++i) {
        SequenceCamera& entry = sequence[static_cast<std::size_t>(i)];
        const std::optional<CaptureTime> time =
            capture_time_of(document, *entry.frame);
        if (!time) {
            entry.capture_time_bad = true;
            continue;
        }
        const std::optional<std::int64_t> seconds =
            parse_datetime(time->datetime_original);
        if (!seconds) {
            entry.capture_time_bad = true;
            continue;
        }
        entry.capture_key =
            *seconds * 1000000 + subsecond_micros(time->subsec_time_original);
    }
    bool all_timed = !sequence.empty();
    for (const SequenceCamera& entry : sequence) {
        if (!entry.capture_key) {
            all_timed = false;
            log(LogLevel::Warn)
                << "TrackMarkers: camera '" << entry.camera->label
                << "' has a missing or malformed capture time; using import "
                   "order";
        }
    }
    if (all_timed) {
        std::stable_sort(sequence.begin(), sequence.end(),
                         [](const SequenceCamera& a, const SequenceCamera& b) {
                             return *a.capture_key < *b.capture_key;
                         });
    }
    return sequence;
}

/// One decoded camera with its luminance pyramid. Full-resolution pixel
/// coordinates are divided by `level_scale` to reach a level; `FloatImage`
/// samples pixel centres at integer coordinates, so the half-pixel offset is
/// applied here exactly once.
struct SequenceView {
    const SequenceCamera* entry = nullptr;
    std::vector<FloatImage> levels;

    int level_count() const { return static_cast<int>(levels.size()); }
    double level_scale(int level) const {
        return std::ldexp(1.0, level);
    }
    const FloatImage& level_image(int level) const {
        return levels[static_cast<std::size_t>(
            std::clamp(level, 0, level_count() - 1))];
    }
    std::size_t bytes() const {
        std::size_t total = 0;
        for (const FloatImage& level : levels) {
            total += level.pixels().size() * sizeof(float);
        }
        return total;
    }
};

/// Thread-safe bounded LRU of decoded pyramids. Markers run in parallel, so a
/// view is decoded once and shared. The budget caps resident luminance; a miss
/// after eviction re-decodes the frame. An explicit camera list throws on an
/// unreadable image; the implicit list logs it and counts it.
class FrameCache {
public:
    FrameCache(const Document& document, const std::vector<SequenceCamera>& sequence,
               const std::vector<std::string>& filters, TrackMarkersStats& stats,
               int pyramid_levels)
        : document_(document), sequence_(sequence), filters_(filters),
          stats_(stats), pyramid_levels_(pyramid_levels) {
        for (const SequenceCamera& entry : sequence_) {
            by_camera_.emplace(entry.camera->id, &entry);
        }
    }

    std::shared_ptr<const SequenceView> get(std::int32_t camera_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto found = entries_.find(camera_id);
        if (found != entries_.end()) {
            order_.splice(order_.begin(), order_, found->second.order);
            return found->second.view;
        }
        if (std::find(failed_.begin(), failed_.end(), camera_id) !=
            failed_.end()) {
            return nullptr;
        }
        // Another thread is already decoding this camera. Wait for its result
        // outside the lock rather than serialising every decode on it.
        const auto in_flight = pending_.find(camera_id);
        if (in_flight != pending_.end()) {
            const std::shared_future<std::shared_ptr<SequenceView>> future =
                in_flight->second;
            lock.unlock();
            return future.get();
        }
        const auto known = by_camera_.find(camera_id);
        if (known == by_camera_.end()) {
            return nullptr;
        }
        const SequenceCamera* entry = known->second;
        std::promise<std::shared_ptr<SequenceView>> promise;
        const std::shared_future<std::shared_ptr<SequenceView>> future =
            promise.get_future().share();
        pending_.emplace(camera_id, future);
        lock.unlock();
        std::shared_ptr<SequenceView> view;
        std::exception_ptr error;
        try {
            view = decode(*entry);
        } catch (...) {
            error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> guard(mutex_);
            pending_.erase(camera_id);
            if (!error) {
                if (!view) {
                    failed_.push_back(camera_id);
                } else {
                    insert_locked(view);
                }
            }
        }
        if (error) {
            promise.set_exception(error);
            std::rethrow_exception(error);
        }
        promise.set_value(view);
        return view;
    }

    std::size_t peak_bytes() const { return peak_bytes_; }

private:
    struct Entry {
        std::shared_ptr<SequenceView> view;
        std::list<std::int32_t>::iterator order;
    };

    std::shared_ptr<SequenceView> decode(const SequenceCamera& entry);

    /// Inserts a freshly decoded view, evicting only views nobody else holds
    /// (use_count 1). Pinned views count against the budget, so a temporary
    /// overshoot is allowed and recorded in `peak_bytes_`. Caller holds mutex_.
    void insert_locked(const std::shared_ptr<SequenceView>& view);

    const Document& document_;
    const std::vector<SequenceCamera>& sequence_;
    const std::vector<std::string>& filters_;
    TrackMarkersStats& stats_;
    int pyramid_levels_;
    std::mutex mutex_;
    std::list<std::int32_t> order_;
    std::unordered_map<std::int32_t, Entry> entries_;
    std::unordered_map<std::int32_t, const SequenceCamera*> by_camera_;
    std::unordered_map<std::int32_t,
                       std::shared_future<std::shared_ptr<SequenceView>>>
        pending_;
    std::vector<std::int32_t> failed_;
    std::size_t bytes_ = 0;
    std::size_t peak_bytes_ = 0;
};

std::shared_ptr<SequenceView> FrameCache::decode(const SequenceCamera& entry) {
    const Camera& camera = *entry.camera;
    try {
        FloatImage full =
            read_image(resolve_frame_path(document_, entry.frame->path())
                           .string())
                .to_luminance();
        auto view = std::make_shared<SequenceView>();
        view->entry = &entry;
        view->levels.push_back(std::move(full));
        const int wanted = std::max(1, pyramid_levels_);
        while (static_cast<int>(view->levels.size()) < wanted) {
            const FloatImage& last = view->levels.back();
            if (last.width() < 8 || last.height() < 8) {
                break;
            }
            view->levels.push_back(downsample_box(last, 2));
        }
        return view;
    } catch (const std::exception& error) {
        const std::string message =
            "TrackMarkers: " + camera.label + ": " + error.what();
        if (!filters_.empty()) {
            throw Error(message);
        }
        log(LogLevel::Warn) << message;
        ++stats_.unreadable_cameras;
        return nullptr;
    }
}

void FrameCache::insert_locked(const std::shared_ptr<SequenceView>& view) {
    const std::int32_t camera_id = view->entry->camera->id;
    const std::size_t incoming = view->bytes();
    // Walk from the least recently used end. A pinned view (use_count > 1) is
    // in use by another tracking thread and must stay resident, so the cache
    // may overshoot the budget until that thread drops its reference.
    auto cursor = order_.end();
    while (bytes_ + incoming > kBudgetBytes && cursor != order_.begin()) {
        --cursor;
        const auto victim = entries_.find(*cursor);
        if (victim == entries_.end()) {
            cursor = order_.erase(cursor);
            continue;
        }
        if (victim->second.view.use_count() > 1) {
            continue;
        }
        bytes_ -= victim->second.view->bytes();
        entries_.erase(victim);
        cursor = order_.erase(cursor);
    }
    entries_[camera_id] = Entry{
        view, order_.insert(order_.begin(), camera_id)};
    bytes_ += incoming;
    peak_bytes_ = std::max(peak_bytes_, bytes_);
}

bool inside_sample(const FloatImage& image, double x, double y,
                   double margin = 1.0) {
    return x >= margin && y >= margin &&
           x <= static_cast<double>(image.width() - 1) - margin &&
           y <= static_cast<double>(image.height() - 1) - margin;
}

struct ShiftResult {
    bool ok = false;
    Eigen::Vector2d shift = Eigen::Vector2d::Zero();
};

/// Translation-only Lucas-Kanade on one pyramid level (Baker & Matthews 2004).
/// `centre` and `shift` are level-pixel coordinates; the window is
/// `window` x `window` samples. A flat template or a flat target patch is "no
/// evidence" and is rejected, which is the first half of the occlusion test.
ShiftResult lk_level(const FloatImage& tmpl, const FloatImage& target,
                     const Eigen::Vector2d& centre, Eigen::Vector2d shift,
                     int window) {
    const int count = std::max(3, window);
    const double c = 0.5 * static_cast<double>(count - 1);
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(count * count));
    double energy = 0.0;
    for (int row = 0; row < count; ++row) {
        const double ty = centre.y() + static_cast<double>(row) - c;
        for (int column = 0; column < count; ++column) {
            const double tx = centre.x() + static_cast<double>(column) - c;
            if (!inside_sample(tmpl, tx, ty)) {
                return ShiftResult{};
            }
            values.push_back(static_cast<double>(tmpl.sample(
                static_cast<float>(tx), static_cast<float>(ty))));
            const double gx =
                static_cast<double>(tmpl.sample(static_cast<float>(tx + 1.0),
                                                static_cast<float>(ty)) -
                                    tmpl.sample(static_cast<float>(tx - 1.0),
                                                static_cast<float>(ty)));
            const double gy =
                static_cast<double>(tmpl.sample(static_cast<float>(tx),
                                                static_cast<float>(ty + 1.0)) -
                                    tmpl.sample(static_cast<float>(tx),
                                                static_cast<float>(ty - 1.0)));
            energy += 0.25 * (gx * gx + gy * gy);
        }
    }
    if (energy / static_cast<double>(values.size()) < kMinGradientEnergy) {
        return ShiftResult{};
    }
    for (int iteration = 0; iteration < 50; ++iteration) {
        Eigen::Matrix2d normal = Eigen::Matrix2d::Zero();
        Eigen::Vector2d gradient = Eigen::Vector2d::Zero();
        std::size_t index = 0;
        for (int row = 0; row < count; ++row) {
            const double qy = centre.y() + shift.y() +
                              static_cast<double>(row) - c;
            for (int column = 0; column < count; ++column) {
                const double qx = centre.x() + shift.x() +
                                  static_cast<double>(column) - c;
                if (!inside_sample(target, qx, qy)) {
                    return ShiftResult{};
                }
                const double gx =
                    static_cast<double>(target.sample(
                        static_cast<float>(qx + 1.0), static_cast<float>(qy)) -
                                        target.sample(static_cast<float>(qx - 1.0),
                                                      static_cast<float>(qy)));
                const double gy =
                    static_cast<double>(target.sample(
                        static_cast<float>(qx), static_cast<float>(qy + 1.0)) -
                                        target.sample(static_cast<float>(qx),
                                                      static_cast<float>(qy - 1.0)));
                const Eigen::Vector2d g(0.5 * gx, 0.5 * gy);
                const double residual = static_cast<double>(target.sample(
                                            static_cast<float>(qx),
                                            static_cast<float>(qy))) -
                                        values[index];
                normal += g * g.transpose();
                gradient += g * residual;
                ++index;
            }
        }
        if (!normal.allFinite() ||
            std::abs(normal.determinant()) < 1e-12) {
            return ShiftResult{};
        }
        const Eigen::Vector2d step = -normal.ldlt().solve(gradient);
        if (!step.allFinite()) {
            return ShiftResult{};
        }
        shift += step;
        if (step.norm() < 0.01) {
            break;
        }
    }
    double final_energy = 0.0;
    for (int row = 0; row < count; ++row) {
        const double qy = centre.y() + shift.y() + static_cast<double>(row) - c;
        for (int column = 0; column < count; ++column) {
            const double qx =
                centre.x() + shift.x() + static_cast<double>(column) - c;
            if (!inside_sample(target, qx, qy)) {
                return ShiftResult{};
            }
            const double gx =
                static_cast<double>(target.sample(static_cast<float>(qx + 1.0),
                                                  static_cast<float>(qy)) -
                                    target.sample(static_cast<float>(qx - 1.0),
                                                  static_cast<float>(qy)));
            const double gy =
                static_cast<double>(target.sample(static_cast<float>(qx),
                                                  static_cast<float>(qy + 1.0)) -
                                    target.sample(static_cast<float>(qx),
                                                  static_cast<float>(qy - 1.0)));
            final_energy += 0.25 * (gx * gx + gy * gy);
        }
    }
    if (final_energy / static_cast<double>(count * count) <
        kMinGradientEnergy) {
        return ShiftResult{};
    }
    ShiftResult result;
    result.ok = true;
    result.shift = shift;
    return result;
}

/// Coarse-to-fine translation. The full-resolution shift is carried down a
/// level by dividing by that level's scale, which doubles it in level pixels.
ShiftResult track_shift(const SequenceView& from, const SequenceView& to,
                        const Eigen::Vector2d& from_pixel,
                        const Eigen::Vector2d& initial, int window,
                        int levels) {
    ShiftResult out;
    out.ok = true;
    out.shift = initial;
    const int available =
        std::min({std::max(1, levels), from.level_count(), to.level_count()});
    for (int level = available - 1; level >= 0; --level) {
        const double scale = from.level_scale(level);
        const Eigen::Vector2d centre =
            from_pixel / scale - Eigen::Vector2d(0.5, 0.5);
        const ShiftResult step =
            lk_level(from.level_image(level), to.level_image(level), centre,
                     out.shift / scale, window);
        if (!step.ok) {
            return ShiftResult{};
        }
        out.shift = step.shift * scale;
    }
    return out;
}

/// Zero-mean normalised cross-correlation between two full-resolution
/// windows. A window with too little intensity variation carries no evidence,
/// so it yields nullopt rather than a misleading score.
std::optional<double> window_ncc(const SequenceView& a,
                                 const Eigen::Vector2d& a_pixel,
                                 const SequenceView& b,
                                 const Eigen::Vector2d& b_pixel, int window) {
    const FloatImage& image_a = a.level_image(0);
    const FloatImage& image_b = b.level_image(0);
    const int count = std::max(3, window);
    const double c = 0.5 * static_cast<double>(count - 1);
    const Eigen::Vector2d centre_a = a_pixel - Eigen::Vector2d(0.5, 0.5);
    const Eigen::Vector2d centre_b = b_pixel - Eigen::Vector2d(0.5, 0.5);
    double sum_a = 0.0;
    double sum_b = 0.0;
    double sum_aa = 0.0;
    double sum_bb = 0.0;
    double sum_ab = 0.0;
    const double n = static_cast<double>(count * count);
    for (int row = 0; row < count; ++row) {
        const double ay = centre_a.y() + static_cast<double>(row) - c;
        const double by = centre_b.y() + static_cast<double>(row) - c;
        for (int column = 0; column < count; ++column) {
            const double ax = centre_a.x() + static_cast<double>(column) - c;
            const double bx = centre_b.x() + static_cast<double>(column) - c;
            if (!inside_sample(image_a, ax, ay) ||
                !inside_sample(image_b, bx, by)) {
                return std::nullopt;
            }
            const double va =
                static_cast<double>(image_a.sample(static_cast<float>(ax),
                                                   static_cast<float>(ay)));
            const double vb =
                static_cast<double>(image_b.sample(static_cast<float>(bx),
                                                   static_cast<float>(by)));
            sum_a += va;
            sum_b += vb;
            sum_aa += va * va;
            sum_bb += vb * vb;
            sum_ab += va * vb;
        }
    }
    const double mean_a = sum_a / n;
    const double mean_b = sum_b / n;
    const double var_a = std::max(0.0, sum_aa / n - mean_a * mean_a);
    const double var_b = std::max(0.0, sum_bb / n - mean_b * mean_b);
    const double std_a = std::sqrt(var_a);
    const double std_b = std::sqrt(var_b);
    if (std_a < kMinPatchStdDev || std_b < kMinPatchStdDev) {
        return std::nullopt;
    }
    const double correlation =
        (sum_ab / n - mean_a * mean_b) / (std_a * std_b);
    return std::clamp(correlation, -1.0, 1.0);
}

struct PairTrack {
    bool ok = false;
    Eigen::Vector2d shift = Eigen::Vector2d::Zero();
};

/// Forward tracking followed by the backward check (Bouguet 2001; Kalal et al.
/// 2010). A match must agree with its own inverse within `max_error_px`, and
/// the two windows must correlate. Either rejection is evidence of occlusion
/// or drift, not a measurement to keep.
PairTrack track_pair(const SequenceView& from, const SequenceView& to,
                     const Eigen::Vector2d& from_pixel, int window,
                     int levels, double max_error_px) {
    const ShiftResult forward = track_shift(
        from, to, from_pixel, Eigen::Vector2d::Zero(), window, levels);
    if (!forward.ok) {
        return PairTrack{};
    }
    const Eigen::Vector2d target_pixel = from_pixel + forward.shift;
    const ShiftResult backward = track_shift(
        to, from, target_pixel, Eigen::Vector2d::Zero(), window, levels);
    if (!backward.ok) {
        return PairTrack{};
    }
    if ((forward.shift + backward.shift).norm() > max_error_px) {
        return PairTrack{};
    }
    const std::optional<double> correlation =
        window_ncc(from, from_pixel, to, target_pixel, window);
    if (!correlation || *correlation < kMinCorrelation) {
        return PairTrack{};
    }
    PairTrack result;
    result.ok = true;
    result.shift = forward.shift;
    return result;
}

/// Coded markers are named `target <code>` by detection, so the code is
/// recoverable from the label without another detection round.
std::optional<int> marker_code(const Marker& marker) {
    constexpr const char* kPrefix = "target ";
    const std::string& label = marker.label;
    if (label.rfind(kPrefix, 0) != 0) {
        return std::nullopt;
    }
    const std::string digits = label.substr(std::string(kPrefix).size());
    if (digits.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    const long value = std::strtol(digits.c_str(), &end, 10);
    if (end == digits.c_str() || end == nullptr || *end != '\0' ||
        value < 0 || value > std::numeric_limits<int>::max()) {
        return std::nullopt;
    }
    return static_cast<int>(value);
}

/// The target style redetection should use. An explicit `target_type` wins.
/// Otherwise a marker may carry its own style, or a coded marker created by
/// DetectMarkers may be resolved through the chunk's recorded run. When none
/// is known, redetection is skipped (and the failure stays a miss).
std::string resolved_target_type(const Chunk& chunk, const Marker& marker,
                                 const std::string& requested) {
    if (!requested.empty()) {
        return requested;
    }
    if (const std::string* stored = marker.meta.find("target_type")) {
        if (!stored->empty()) {
            return *stored;
        }
    }
    if (marker_code(marker)) {
        if (const std::string* detected =
                chunk.meta().find("DetectMarkers/target_type")) {
            if (is_known_target_type(*detected)) {
                return *detected;
            }
        }
    }
    return {};
}

/// Redetect the marker in a `4 x window` search around the last accepted
/// position. A coded marker accepts only its own code; a non-coded marker
/// accepts the nearest detection within twice the window. The crop is copied,
/// so detection never sees the rest of the image and cannot attach to a
/// neighbouring target.
std::optional<Eigen::Vector2d> redetect_marker(
    const SequenceView& view, const Marker& marker, const Chunk& chunk,
    const std::string& requested_type, const Eigen::Vector2d& anchor,
    int window) {
    const std::string type = resolved_target_type(chunk, marker, requested_type);
    if (type.empty()) {
        return std::nullopt;
    }
    const FloatImage& image = view.level_image(0);
    // A projection can be NaN or outside the image when it was imported or
    // carried over from another run. Casting that to int is undefined
    // behaviour, so reject it here like any other failed redetection.
    if (!anchor.allFinite() || image.width() <= 0 || image.height() <= 0) {
        return std::nullopt;
    }
    const double max_x = static_cast<double>(image.width() - 1);
    const double max_y = static_cast<double>(image.height() - 1);
    if (anchor.x() < 0.0 || anchor.y() < 0.0 || anchor.x() > max_x ||
        anchor.y() > max_y) {
        return std::nullopt;
    }
    const int half = std::max(8, 4 * std::max(1, window));
    const int x0 = std::clamp(static_cast<int>(std::floor(anchor.x())) - half,
                              0, std::max(0, image.width() - 1));
    const int y0 = std::clamp(static_cast<int>(std::floor(anchor.y())) - half,
                              0, std::max(0, image.height() - 1));
    const int x1 = std::min(image.width() - 1, x0 + 2 * half);
    const int y1 = std::min(image.height() - 1, y0 + 2 * half);
    const int width = x1 - x0 + 1;
    const int height = y1 - y0 + 1;
    if (width < 8 || height < 8) {
        return std::nullopt;
    }
    FloatImage crop(width, height, 0.0F);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            crop.at(x, y) = image.at(x0 + x, y0 + y);
        }
    }
    std::vector<ImageDetection> detections;
    try {
        detections = detect_targets_in_image(crop, type, 50.0, false, false,
                                             0.0, 5.0);
    } catch (const std::exception&) {
        return std::nullopt;
    }
    const bool coded = is_coded_target_type(type);
    const std::optional<int> code = marker_code(marker);
    if (coded && !code) {
        return std::nullopt;
    }
    const Eigen::Vector2d offset(static_cast<double>(x0),
                                 static_cast<double>(y0));
    double best_distance = coded ? std::numeric_limits<double>::infinity()
                                 : 2.0 * static_cast<double>(window);
    std::optional<Eigen::Vector2d> best;
    for (const ImageDetection& detection : detections) {
        if (coded && (!detection.coded || detection.code != *code)) {
            continue;
        }
        const Eigen::Vector2d pixel = detection.center + offset;
        const double distance = (pixel - anchor).norm();
        if (distance <= best_distance) {
            best_distance = distance;
            best = pixel;
        }
    }
    return best;
}

struct ProjectionAdd {
    std::int32_t camera_id = -1;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
};

struct MarkerTrackResult {
    std::int32_t marker_id = -1;
    std::vector<ProjectionAdd> adds;
    std::size_t tracked = 0;
    std::size_t redetected = 0;
    std::size_t misses = 0;
    bool stopped = false;
    bool skipped = false;
    /// Set when decoding threw inside the parallel region; rethrown serially.
    std::string error;
};

/// A marker's projection state during one run, including additions made by an
/// earlier direction. Tracking never writes the chunk; the command does.
struct LocalProjection {
    std::int32_t camera_id = -1;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    bool enabled = true;
};

const LocalProjection* find_projection(const std::vector<LocalProjection>& state,
                                       std::int32_t camera_id) {
    for (const LocalProjection& projection : state) {
        if (projection.camera_id == camera_id) {
            return &projection;
        }
    }
    return nullptr;
}

struct TrackStep {
    bool ok = false;
    bool redetected = false;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
};

/// One camera transition: Lucas-Kanade first, then redetection when enabled
/// and tracking failed. Redetection also works when the anchor frame cannot
/// be decoded, because it only needs the destination image.
TrackStep track_step(const Chunk& chunk, const Marker& marker,
                     const SequenceCamera& anchor_camera,
                     const Eigen::Vector2d& anchor,
                     const SequenceCamera& next_camera, FrameCache& cache,
                     const TrackMarkersOptions& options) {
    TrackStep step;
    const std::shared_ptr<const SequenceView> from =
        cache.get(anchor_camera.camera->id);
    const std::shared_ptr<const SequenceView> to =
        cache.get(next_camera.camera->id);
    if (from && to) {
        const PairTrack pair =
            track_pair(*from, *to, anchor, options.window,
                       options.pyramid_levels, options.max_error_px);
        if (pair.ok) {
            step.ok = true;
            step.pixel = anchor + pair.shift;
            return step;
        }
    }
    if (!options.redetect || !to) {
        return step;
    }
    const std::optional<Eigen::Vector2d> detected =
        redetect_marker(*to, marker, chunk, options.target_type, anchor,
                        options.window);
    if (detected) {
        step.ok = true;
        step.redetected = true;
        step.pixel = *detected;
    }
    return step;
}

/// One direction over one marker. `order` indexes `sequence` in tracking
/// order. The pass starts at the last position in that order that already has
/// an enabled projection and only ever appends to `state`. Existing enabled
/// projections are re-anchors; an existing disabled projection is never
/// overwritten and never becomes an anchor.
void track_marker_pass(const Chunk& chunk, const Marker& marker,
                       const std::vector<SequenceCamera>& sequence,
                       const std::vector<std::size_t>& order,
                       FrameCache& cache, const TrackMarkersOptions& options,
                       std::vector<LocalProjection>& state,
                       MarkerTrackResult& result) {
    int start = -1;
    for (std::size_t pos = 0; pos < order.size(); ++pos) {
        const LocalProjection* projection = find_projection(
            state, sequence[order[pos]].camera->id);
        if (projection && projection->enabled) {
            start = static_cast<int>(pos);
        }
    }
    if (start < 0) {
        return;
    }
    std::size_t anchor_pos = static_cast<std::size_t>(start);
    Eigen::Vector2d anchor =
        find_projection(state, sequence[order[anchor_pos]].camera->id)->pixel;
    int misses = 0;
    for (std::size_t pos = anchor_pos + 1; pos < order.size(); ++pos) {
        const SequenceCamera& next = sequence[order[pos]];
        const LocalProjection* existing =
            find_projection(state, next.camera->id);
        if (existing) {
            if (existing->enabled) {
                anchor_pos = pos;
                anchor = existing->pixel;
                misses = 0;
            }
            continue;
        }
        const TrackStep step =
            track_step(chunk, marker, sequence[order[anchor_pos]], anchor,
                       next, cache, options);
        if (step.ok) {
            state.push_back({next.camera->id, step.pixel, true});
            result.adds.push_back({next.camera->id, step.pixel});
            if (step.redetected) {
                ++result.redetected;
            } else {
                ++result.tracked;
            }
            anchor_pos = pos;
            anchor = step.pixel;
            misses = 0;
            continue;
        }
        ++result.misses;
        ++misses;
        if (misses > options.max_gap) {
            result.stopped = true;
            break;
        }
    }
}

MarkerTrackResult track_marker(const Chunk& chunk, const Marker& marker,
                               const std::vector<SequenceCamera>& sequence,
                               FrameCache& cache,
                               const TrackMarkersOptions& options) {
    MarkerTrackResult result;
    result.marker_id = marker.id;
    std::vector<LocalProjection> state;
    state.reserve(marker.projections.size());
    for (const MarkerProjection& projection : marker.projections) {
        // A non-finite pixel is unusable as an anchor and unsafe to cast when
        // redetection crops around it. Skipping the marker keeps the same
        // contract as any other marker that cannot be tracked.
        if (projection.enabled && !projection.pixel.allFinite()) {
            result.skipped = true;
            return result;
        }
        state.push_back({projection.camera_id, projection.pixel,
                         projection.enabled});
    }
    bool anchored = false;
    for (const SequenceCamera& entry : sequence) {
        const LocalProjection* projection =
            find_projection(state, entry.camera->id);
        if (projection && projection->enabled) {
            anchored = true;
            break;
        }
    }
    if (!anchored) {
        result.skipped = true;
        return result;
    }
    std::vector<std::size_t> order(sequence.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    if (options.direction == "forward" || options.direction == "both") {
        track_marker_pass(chunk, marker, sequence, order, cache, options,
                          state, result);
    }
    if (options.direction == "backward" || options.direction == "both") {
        std::reverse(order.begin(), order.end());
        track_marker_pass(chunk, marker, sequence, order, cache, options,
                          state, result);
    }
    return result;
}

/// Run every marker, then stage the additions in marker order as ONE
/// transaction. Markers are independent, so the tracking loop is parallel;
/// no chunk mutation happens until the transaction is built.
void run_tracking(const Document& document, const Chunk& chunk,
                  const std::vector<SequenceCamera>& sequence,
                  const TrackMarkersOptions& options, TrackMarkersPlan& plan) {
    FrameCache cache(document, sequence, options.cameras, plan.stats,
                     options.pyramid_levels);
    const int count = static_cast<int>(chunk.markers().size());
    std::vector<MarkerTrackResult> results(chunk.markers().size());
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < count; ++i) {
        const Marker& marker = chunk.markers()[static_cast<std::size_t>(i)];
        try {
            results[static_cast<std::size_t>(i)] =
                track_marker(chunk, marker, sequence, cache, options);
        } catch (const std::exception& error) {
            results[static_cast<std::size_t>(i)].marker_id = marker.id;
            results[static_cast<std::size_t>(i)].error = error.what();
        }
    }
    for (const MarkerTrackResult& result : results) {
        if (!result.error.empty()) {
            throw Error(result.error);
        }
    }
    auto transaction = std::make_unique<CommandTransaction>("Track Markers");
    for (std::size_t i = 0; i < results.size(); ++i) {
        const MarkerTrackResult& result = results[i];
        const Marker& marker = chunk.markers()[i];
        plan.stats.tracked += result.tracked;
        plan.stats.redetected += result.redetected;
        plan.stats.misses += result.misses;
        if (result.stopped) {
            ++plan.stats.stopped_markers;
        }
        if (result.skipped) {
            ++plan.stats.markers_skipped;
        } else if (!result.adds.empty()) {
            ++plan.stats.markers_extended;
        } else {
            ++plan.stats.markers_unchanged;
        }
        for (const ProjectionAdd& add : result.adds) {
            transaction->add(make_add_marker_with_projection_command(
                chunk.id(), marker.id, add.camera_id, add.pixel, false));
        }
    }
    plan.stats.peak_bytes = cache.peak_bytes();
    if (plan.stats.tracked + plan.stats.redetected > 0) {
        plan.command = std::move(transaction);
    }
}

}  // namespace

TrackMarkersPlan plan_track_markers(const Document& document,
                                    std::int32_t chunk_id,
                                    const TrackMarkersOptions& options) {
    TrackMarkersPlan plan;
    if (options.direction != "forward" && options.direction != "backward" &&
        options.direction != "both") {
        throw ValueError(
            "TrackMarkers: direction must be 'forward', 'backward' or 'both'");
    }
    if (!(options.max_error_px > 0.0) || !std::isfinite(options.max_error_px)) {
        throw ValueError("TrackMarkers: max_error_px must be positive");
    }
    if (options.window < 3 || options.window > 255) {
        throw ValueError("TrackMarkers: window must be in [3, 255]");
    }
    if (options.pyramid_levels < 1 || options.pyramid_levels > 8) {
        throw ValueError("TrackMarkers: pyramid_levels must be in [1, 8]");
    }
    if (options.max_gap < 0 || options.max_gap > 10000) {
        throw ValueError("TrackMarkers: max_gap must be in [0, 10000]");
    }
    if (!options.target_type.empty() &&
        !is_known_target_type(options.target_type)) {
        throw ValueError("TrackMarkers: unknown target_type '" +
                         options.target_type + "'; valid values are: " +
                         valid_target_type_names());
    }
    const Chunk* chunk = find_chunk(document, chunk_id);
    if (chunk == nullptr) {
        throw Error("TrackMarkers needs an active chunk");
    }
    const std::vector<SequenceCamera> sequence =
        selected_sequence(document, *chunk, options.cameras);
    run_tracking(document, *chunk, sequence, options, plan);
    return plan;
}

void TrackMarkersTask::declare_params(Params& params) const {
    params.declare({"cameras", ParamType::StringList,
                    std::vector<std::string>{},
                    "Camera labels forming the sequence. Empty means every "
                    "enabled camera; an explicit list also admits unaligned "
                    "cameras."});
    ParamSpec direction{"direction", ParamType::String, std::string{"forward"},
                        "Sequence senses to run: forward, backward or both. "
                        "Both runs forward first, then backward from the "
                        "resulting projections."};
    direction.choices = {"forward", "backward", "both"};
    params.declare(direction);
    ParamSpec error{"max_error_px", ParamType::Float, 1.0,
                    "Maximum forward-backward displacement disagreement, in "
                    "pixels, for a match to be accepted."};
    error.minimum = 1e-6;
    params.declare(error);
    ParamSpec window{"window", ParamType::Int, std::int64_t{21},
                     "Side length of the square Lucas-Kanade window, in "
                     "pixels."};
    window.minimum = std::int64_t{3};
    window.maximum = std::int64_t{255};
    params.declare(window);
    ParamSpec levels{"pyramid_levels", ParamType::Int, std::int64_t{3},
                     "Number of luminance pyramid levels, coarsest first."};
    levels.minimum = std::int64_t{1};
    levels.maximum = std::int64_t{8};
    params.declare(levels);
    params.declare({"redetect", ParamType::Bool, true,
                    "After a failed track, detect the target again in a local "
                    "search window."});
    params.declare({"target_type", ParamType::String, std::string{},
                    "Target style used by redetection, with the same values as "
                    "DetectMarkers. Empty uses the marker's own style when it "
                    "is known, otherwise redetection is skipped."});
    ParamSpec gap{"max_gap", ParamType::Int, std::int64_t{3},
                  "Consecutive missed frames a marker may bridge. One further "
                  "frame is attempted for redetection recovery."};
    gap.minimum = std::int64_t{0};
    gap.maximum = std::int64_t{10000};
    params.declare(gap);
}

std::vector<Stage> TrackMarkersTask::stages() const {
    return {Stage{"track", 0}, Stage{"apply", 0}};
}

void TrackMarkersTask::apply(Document& document, Params& params,
                             const Progress& progress) {
    TrackMarkersOptions options;
    options.cameras = params.get_string_list("cameras");
    options.direction = params.get_string("direction");
    options.max_error_px = params.get_float("max_error_px");
    options.window = static_cast<int>(params.get_int("window"));
    options.pyramid_levels =
        static_cast<int>(params.get_int("pyramid_levels"));
    options.redetect = params.get_bool("redetect");
    options.target_type = params.get_string("target_type");
    options.max_gap = static_cast<int>(params.get_int("max_gap"));

    progress.update("track", 0.0);
    const auto started = std::chrono::steady_clock::now();
    TrackMarkersPlan plan =
        plan_track_markers(document, document.active_chunk_id(), options);
    progress.update("track", 1.0);
    progress.update("apply", 0.0);
    if (plan.command) {
        plan.command->apply(document);
    }
    progress.update("apply", 1.0);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info)
            << "TrackMarkers: tracked " << plan.stats.tracked << ", redetected "
            << plan.stats.redetected << ", misses " << plan.stats.misses
            << ", stopped " << plan.stats.stopped_markers << ", extended "
            << plan.stats.markers_extended << ", unchanged "
            << plan.stats.markers_unchanged << ", skipped "
            << plan.stats.markers_skipped << ", unreadable cameras "
            << plan.stats.unreadable_cameras << ", peak "
            << plan.stats.peak_bytes / (1024 * 1024) << " MiB in " << seconds
            << " s";
    }
}

}  // namespace alphashape
