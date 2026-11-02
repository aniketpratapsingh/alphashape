#include "common/algorithm/markers/detect_markers_task.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "base/utils/omp.hpp"
#include "common/algorithm/markers/apriltag_decoder.hpp"
#include "common/algorithm/markers/circular_decoder.hpp"
#include "common/algorithm/markers/cross_decoder.hpp"
#include "common/algorithm/markers/marker_detector.hpp"
#include "common/algorithm/markers/project_markers.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/core/marker.hpp"
#include "common/core/mask.hpp"
#include "common/core/types.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

enum class MarkerKind { CircularCoded, CircularNonCoded, Cross, AprilTag };

struct TargetSpec {
    MarkerKind kind = MarkerKind::CircularCoded;
    int bits = 12;
    AprilTagFamily family = AprilTagFamily::Tag36h11;
};

struct ParsedTarget {
    TargetSpec spec;
    bool circle21_unsupported = false;
    bool valid = false;
};
ParsedTarget parse_target(const std::string& name) {
    ParsedTarget out;
    out.valid = true;
    if (name == "Circular12" || name == "CircularTarget12bit") {
        out.spec.kind = MarkerKind::CircularCoded;
        out.spec.bits = 12;
        return out;
    }
    if (name == "Circular14" || name == "CircularTarget14bit") {
        out.spec.kind = MarkerKind::CircularCoded;
        out.spec.bits = 14;
        return out;
    }
    if (name == "Circular16" || name == "CircularTarget16bit") {
        out.spec.kind = MarkerKind::CircularCoded;
        out.spec.bits = 16;
        return out;
    }
    if (name == "Circular20" || name == "CircularTarget20bit") {
        out.spec.kind = MarkerKind::CircularCoded;
        out.spec.bits = 20;
        return out;
    }
    if (name == "Circular" || name == "CircularTarget") {
        out.spec.kind = MarkerKind::CircularNonCoded;
        return out;
    }
    if (name == "Cross" || name == "CrossTarget") {
        out.spec.kind = MarkerKind::Cross;
        return out;
    }
    if (name == "AprilTag16h5") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Tag16h5;
        return out;
    }
    if (name == "AprilTag25h9") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Tag25h9;
        return out;
    }
    if (name == "AprilTag36h10") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Tag36h10;
        return out;
    }
    if (name == "AprilTag36h11") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Tag36h11;
        return out;
    }
    if (name == "AprilTagStandard41h12") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Standard41h12;
        return out;
    }
    if (name == "AprilTagStandard52h13") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Standard52h13;
        return out;
    }
    out.valid = false;
    out.circle21_unsupported = name == "AprilTagCircle21h7";
    return out;
}
const std::vector<std::string>& target_choices() {
    static const std::vector<std::string> choices = {
        "Circular12",          "Circular14",          "Circular16",
        "Circular20",          "Circular",            "Cross",
        "AprilTag16h5",        "AprilTag25h9",        "AprilTag36h10",
        "AprilTag36h11",       "AprilTagStandard41h12",
        "AprilTagStandard52h13", "AprilTagCircle21h7"};
    return choices;
}

std::string valid_target_names() {
    std::string text;
    for (const std::string& choice : target_choices()) {
        if (!text.empty()) {
            text += ", ";
        }
        text += choice;
    }
    return text;
}

void invert_in_place(FloatImage& gray) {
    for (float& value : gray.pixels()) {
        value = 1.0F - value;
    }
}

/// Box average. `invert` folds the polarity change into the same pass, so a
/// full-resolution luminance and its inverted copy never coexist.
FloatImage box_downsample(const FloatImage& source, int factor, bool invert = false) {
    if (factor <= 1) {
        return source;
    }
    const int width = source.width() / factor;
    const int height = source.height() / factor;
    FloatImage out(width, height, 0.0F);
    const double total = static_cast<double>(factor * factor);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int dy = 0; dy < factor; ++dy) {
                for (int dx = 0; dx < factor; ++dx) {
                    sum += static_cast<double>(
                        source.at(x * factor + dx, y * factor + dy));
                }
            }
            const double value = sum / total;
            out.at(x, y) = static_cast<float>(invert ? 1.0 - value : value);
        }
    }
    return out;
}
struct Detection {
    int code = -1;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    double radius = 0.0;
    int hamming = 0;
    bool coded = false;
    CrossStyle cross_style = CrossStyle::Saddle;
    // Eigen's default constructor leaves its scalars uninitialised, and a
    // plain `corners{}` therefore still holds indeterminate doubles; GCC's
    // -Wmaybe-uninitialized reports that when a Detection is copied. List all
    // four explicitly so the storage is written on every construction.
    std::array<Eigen::Vector2d, 4> corners{
        Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero()};
    bool has_corners = false;
};

struct DetectorParams {
    int max_hamming = 0;
    bool noparity = false;
    double minimum_size = 0.0;
    double minimum_dist = 5.0;
};

BlobParams blob_params(int width, int height, double minimum_size) {
    BlobParams params;
    params.dark_on_light = true;
    params.min_radius =
        std::max(3, static_cast<int>(std::ceil(std::max(0.0, minimum_size))));
    const int cap = std::max(params.min_radius,
                             std::min(128, std::max(3, std::min(width, height) / 3)));
    params.max_radius = cap;
    params.threshold = 0.02F;
    params.min_axis_ratio = 0.3;
    return params;
}
std::vector<Detection> detect_circular(const FloatImage& image,
                                        const TargetSpec& spec,
                                        const DetectorParams& params) {
    std::vector<Detection> out;
    const BlobParams blob =
        blob_params(image.width(), image.height(), params.minimum_size);
    const std::vector<MarkerCandidate> candidates = detect_blobs(image, blob);
    if (spec.kind == MarkerKind::CircularNonCoded) {
        for (const MarkerCandidate& candidate : candidates) {
            if (candidate.radius_px < params.minimum_size) {
                continue;
            }
            Detection detection;
            detection.center = candidate.center;
            detection.radius = candidate.radius_px;
            out.push_back(detection);
        }
        return out;
    }
    const GradientField gradient = compute_gradient_field(image);
    for (const MarkerCandidate& candidate : candidates) {
        const auto decoded = decode_circular(image, gradient, candidate, spec.bits,
                                             params.max_hamming);
        if (!decoded || decoded->radius_px < params.minimum_size) {
            continue;
        }
        Detection detection;
        detection.code = params.noparity ? (decoded->code / 2) * 2 : decoded->code;
        detection.center = decoded->center;
        detection.radius = decoded->radius_px;
        detection.hamming = decoded->hamming;
        detection.coded = true;
        out.push_back(detection);
    }
    return out;
}

std::vector<Detection> detect_tags(const FloatImage& image, const TargetSpec& spec,
                                    const DetectorParams& params) {
    std::vector<Detection> out;
    QuadParams quad;
    quad.dark_on_light = true;
    for (const CodedTarget& target :
         decode_tags(image, spec.family, params.max_hamming, quad)) {
        if (target.radius_px < params.minimum_size) {
            continue;
        }
        Detection detection;
        detection.code = target.code;
        detection.center = target.center;
        detection.radius = target.radius_px;
        detection.hamming = target.hamming;
        detection.coded = true;
        detection.corners = target.corners;
        detection.has_corners = target.has_corners;
        out.push_back(detection);
    }
    return out;
}
std::vector<Detection> detect_cross_targets(const FloatImage& image,
                                             const DetectorParams& params) {
    CrossParams cross;
    cross.inverted = false;
    cross.minimum_size = params.minimum_size;
    cross.minimum_dist = params.minimum_dist;
    std::vector<Detection> out;
    for (const CrossDetection& found : detect_crosses_styled(image, cross)) {
        if (found.target.radius_px < params.minimum_size) {
            continue;
        }
        Detection detection;
        detection.center = found.target.center;
        detection.radius = found.target.radius_px;
        detection.cross_style = found.style;
        out.push_back(detection);
    }
    return out;
}

std::vector<Detection> detect_in_image(const FloatImage& image,
                                        const TargetSpec& spec,
                                        const DetectorParams& params) {
    if (spec.kind == MarkerKind::AprilTag) {
        return detect_tags(image, spec, params);
    }
    if (spec.kind == MarkerKind::Cross) {
        return detect_cross_targets(image, params);
    }
    return detect_circular(image, spec, params);
}
std::vector<Detection> suppress_duplicates(std::vector<Detection> detections,
                                            double minimum_dist) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  if (a.coded != b.coded) {
                      return a.coded < b.coded;
                  }
                  if (a.code != b.code) {
                      return a.code < b.code;
                  }
                  if (a.hamming != b.hamming) {
                      return a.hamming < b.hamming;
                  }
                  return a.radius > b.radius;
              });
    std::vector<Detection> out;
    for (const Detection& detection : detections) {
        bool suppressed = false;
        for (const Detection& kept : out) {
            if (kept.coded != detection.coded || kept.code != detection.code) {
                continue;
            }
            if ((kept.center - detection.center).norm() < minimum_dist) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            out.push_back(detection);
        }
    }
    std::sort(out.begin(), out.end(), [](const Detection& a, const Detection& b) {
        if (a.center.y() != b.center.y()) {
            return a.center.y() < b.center.y();
        }
        return a.center.x() < b.center.x();
    });
    return out;
}
struct RefinementWindow {
    FloatImage image;
    int x0 = 0;
    int y0 = 0;
};

int window_half(const TargetSpec& spec, double radius) {
    const double margin = spec.kind == MarkerKind::AprilTag
                              ? 2.5 * radius + 10.0
                              : 2.2 * radius + 10.0;
    return static_cast<int>(std::ceil(std::max(10.0, margin)));
}

RefinementWindow crop_refinement_window(const FloatImage& source,
                                        const Eigen::Vector2d& center,
                                        double radius, const TargetSpec& spec,
                                        bool invert) {
    RefinementWindow window;
    const Eigen::Vector2d point = center - Eigen::Vector2d(0.5, 0.5);
    const int half = window_half(spec, radius);
    int x0 = static_cast<int>(std::floor(point.x())) - half;
    int y0 = static_cast<int>(std::floor(point.y())) - half;
    x0 = std::clamp(x0, 0, std::max(0, source.width() - 1));
    y0 = std::clamp(y0, 0, std::max(0, source.height() - 1));
    const int x1 = std::min(source.width() - 1, x0 + 2 * half);
    const int y1 = std::min(source.height() - 1, y0 + 2 * half);
    window.x0 = x0;
    window.y0 = y0;
    window.image = FloatImage(std::max(0, x1 - x0 + 1),
                              std::max(0, y1 - y0 + 1), 0.0F);
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float value = source.at(x, y);
            window.image.at(x - x0, y - y0) = invert ? 1.0F - value : value;
        }
    }
    return window;
}

/// Map reduced-image detections to full resolution and copy only the small
/// windows refinement will read. The caller may then release the full image.
void map_and_crop_refinement(std::vector<Detection>& detections, double scale,
                             const TargetSpec& spec, const FloatImage& full,
                             bool invert,
                             std::vector<RefinementWindow>& windows) {
    windows.clear();
    windows.reserve(detections.size());
    for (Detection& detection : detections) {
        detection.center *= scale;
        detection.radius *= scale;
        if (spec.kind == MarkerKind::AprilTag) {
            for (Eigen::Vector2d& corner : detection.corners) {
                corner *= scale;
            }
        }
        windows.push_back(crop_refinement_window(full, detection.center,
                                                 detection.radius, spec, invert));
    }
}

void refine_from_windows(std::vector<Detection>& detections,
                         const TargetSpec& spec,
                         const std::vector<RefinementWindow>& windows,
                         int max_hamming) {
    for (std::size_t i = 0; i < detections.size(); ++i) {
        Detection& detection = detections[i];
        const RefinementWindow& window = windows[i];
        const Eigen::Vector2d offset(static_cast<double>(window.x0),
                                     static_cast<double>(window.y0));
        const Eigen::Vector2d local = detection.center - offset;
        if (spec.kind == MarkerKind::Cross) {
            const auto refined = detection.cross_style == CrossStyle::Bar
                                     ? refine_bar_cross_center(
                                           window.image, local, detection.radius)
                                     : refine_cross_center(window.image, local,
                                                           detection.radius);
            if (refined) {
                detection.center = *refined + offset;
            }
        } else if (spec.kind == MarkerKind::AprilTag) {
            if (!detection.has_corners) {
                continue;
            }
            MarkerCandidate quad;
            quad.center = local;
            quad.radius_px = detection.radius;
            for (const Eigen::Vector2d& corner : detection.corners) {
                quad.corners.push_back(corner - offset);
            }
            const GradientField gradient = compute_gradient_field(window.image);
            const auto decoded = decode_tag(window.image, gradient, quad,
                                            tag_family(spec.family), max_hamming);
            if (decoded) {
                detection.center = decoded->center + offset;
                detection.radius = decoded->radius_px;
                detection.hamming = decoded->hamming;
                for (std::size_t corner = 0; corner < 4U; ++corner) {
                    detection.corners[corner] =
                        decoded->corners[corner] + offset;
                }
                detection.has_corners = decoded->has_corners;
            }
        } else {
            MarkerCandidate seed;
            seed.center = local;
            seed.radius_px = detection.radius;
            seed.axis_ratio = 1.0;
            const GradientField gradient = compute_gradient_field(window.image);
            if (auto refined = refine_ellipse_center(gradient, seed)) {
                detection.center = *refined + offset;
            }
        }
    }
}
std::filesystem::path resolve_frame_path(const Document& document,
                                         const std::string& path) {
    const std::filesystem::path recorded(path);
    if (recorded.is_absolute() || document.path().empty()) {
        return recorded;
    }
    return std::filesystem::path(document.path()).parent_path() / recorded;
}

const Chunk* find_chunk_by_id(const Document& document, std::int32_t chunk_id) {
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

const Mask* mask_of_camera(const Chunk& chunk, std::int32_t camera_id) {
    for (const Mask& mask : chunk.masks()) {
        if (mask.camera_id == camera_id && mask.is_valid()) {
            return &mask;
        }
    }
    return nullptr;
}

bool mask_allows(const Mask& mask, const Eigen::Vector2d& centre) {
    const int x = static_cast<int>(std::floor(centre.x()));
    const int y = static_cast<int>(std::floor(centre.y()));
    if (x < 0 || y < 0 || x >= mask.width || y >= mask.height) {
        return true;
    }
    const std::uint8_t value =
        mask.data[static_cast<std::size_t>(y * mask.width + x)];
    return mask.invert ? value == 0 : value != 0;
}

bool contains(const std::vector<std::string>& list, const std::string& value) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

bool frame_matches(const Frame& frame,
                   const std::vector<std::string>& filters) {
    if (filters.empty()) {
        return true;
    }
    const std::filesystem::path path(frame.path());
    const std::string filename = path.filename().string();
    const std::string stem = path.stem().string();
    for (const std::string& filter : filters) {
        if (filter == frame.path() || filter == filename || filter == stem) {
            return true;
        }
    }
    return false;
}
struct Assignment {
    std::string label;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    double radius = 0.0;
    int hamming = 0;
    bool existing = false;
    std::int32_t camera_id = -1;
};

struct CameraPlan {
    std::vector<Assignment> assignments;
    std::size_t created = 0;
    std::size_t dropped = 0;
};

struct LabelState {
    std::set<std::string> used;
    int next_point = 1;

    std::string next_point_label() {
        while (true) {
            const std::string label = "point " + std::to_string(next_point);
            ++next_point;
            if (used.insert(label).second) {
                return label;
            }
        }
    }
};

using Predictions = std::vector<std::pair<std::string, Eigen::Vector2d>>;
using ProjectedCameras = std::map<std::string, std::set<std::int32_t>>;
CameraPlan plan_camera(const std::vector<Detection>& detections,
                       std::int32_t camera_id, const Predictions& predictions,
                       const std::map<std::string, std::int32_t>& ids,
                       const ProjectedCameras& projected, double maximum_residual,
                       bool create_markers, LabelState& labels) {
    CameraPlan plan;
    std::map<std::string, std::size_t> best;
    const auto consider = [&](const Assignment& assignment) {
        const auto found = best.find(assignment.label);
        if (found == best.end()) {
            best[assignment.label] = plan.assignments.size();
            plan.assignments.push_back(assignment);
            return;
        }
        Assignment& current = plan.assignments[found->second];
        if (assignment.hamming < current.hamming ||
            (assignment.hamming == current.hamming &&
             assignment.radius > current.radius)) {
            current = assignment;
        }
        ++plan.dropped;  // one of the two competing detections is discarded
    };
    for (const Detection& detection : detections) {
        if (detection.coded) {
            const std::string label = "target " + std::to_string(detection.code);
            const bool exists = ids.count(label) != 0;
            const auto projections = projected.find(label);
            if (exists && projections != projected.end() &&
                projections->second.count(camera_id) != 0) {
                ++plan.dropped;
                continue;  // the camera already has a measurement of this marker
            }
            if (!exists) {
                if (!create_markers) {
                    ++plan.dropped;
                    continue;
                }
                if (labels.used.insert(label).second) {
                    ++plan.created;
                }
            }
            consider(Assignment{label, detection.center, detection.radius,
                                detection.hamming, exists, camera_id});
            continue;
        }
        int nearest = -1;
        double nearest_distance = maximum_residual;
        for (std::size_t i = 0; i < predictions.size(); ++i) {
            const double distance = (predictions[i].second - detection.center).norm();
            if (distance <= nearest_distance) {
                nearest_distance = distance;
                nearest = static_cast<int>(i);
            }
        }
        if (nearest >= 0) {
            const std::string& label = predictions[static_cast<std::size_t>(nearest)].first;
            const auto projections = projected.find(label);
            if (projections != projected.end() &&
                projections->second.count(camera_id) != 0) {
                ++plan.dropped;
                continue;
            }
            consider(Assignment{label, detection.center, detection.radius, 0, true,
                                camera_id});
            continue;
        }
        if (!create_markers) {
            ++plan.dropped;
            continue;
        }
        const std::string label = labels.next_point_label();
        ++plan.created;
        consider(Assignment{label, detection.center, detection.radius, 0, false,
                            camera_id});
    }
    return plan;
}
}  // namespace

int circular_max_hamming(double tolerance) {
    return tolerance >= 75.0 ? 1 : 0;
}

int apriltag_max_hamming(double tolerance, AprilTagFamily family) {
    const int budget = default_max_hamming(family);
    const int cap = (tag_family(family).min_hamming - 1) / 2;
    const double scaled = tolerance / 50.0 * static_cast<double>(budget);
    const int value = static_cast<int>(std::lround(scaled));
    return std::clamp(value, 0, cap);
}

namespace {

struct CameraJob {
    const Camera* camera = nullptr;
    const Frame* frame = nullptr;
    const Calibration* calibration = nullptr;
    int scale = 1;
};

struct CameraResult {
    std::vector<Detection> detections;
    std::string error;
};

int auto_scale(int width, int height) {
    const int long_side = std::max(width, height);
    return std::max(1, (long_side + 4095) / 4096);
}

std::size_t luminance_bytes(int width, int height) {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
           sizeof(float);
}

/// Buffers that coexist while one camera is detected: the full luminance, the
/// reduced detection image (when downscaled), and the detector's own working
/// copies at detection size. Windows used for full-resolution refinement are
/// allocated only after the full image and its reduced copy are released, so
/// they are bounded by a few thousand pixels each and do not add a full image.
std::size_t estimate_peak_working_set(int width, int height, int scale,
                                      MarkerKind kind) {
    const std::size_t full = luminance_bytes(width, height);
    const int detect_width = scale > 1 ? width / scale : width;
    const int detect_height = scale > 1 ? height / scale : height;
    const std::size_t reduced = scale > 1
                                    ? luminance_bytes(detect_width, detect_height)
                                    : 0;
    const std::size_t copies = kind == MarkerKind::Cross
                                   ? 3
                                   : (kind == MarkerKind::AprilTag ? 2 : 1);
    return full + reduced + copies * luminance_bytes(detect_width, detect_height);
}

CameraResult process_camera(const Document& document, const CameraJob& job,
                            const TargetSpec& spec, const DetectorParams& params,
                            bool inverted, bool filter_mask, const Mask* mask) {
    CameraResult result;
    try {
        FloatImage luminance =
            read_image(resolve_frame_path(document, job.frame->path()).string())
                .to_luminance();
        const int width = luminance.width();
        const int height = luminance.height();
        int scale = job.scale > 0 ? job.scale : auto_scale(width, height);
        if (scale > 1 && (width / scale < 8 || height / scale < 8)) {
            scale = 1;
        }
        const std::size_t estimated =
            estimate_peak_working_set(width, height, scale, spec.kind);
        if (Log::enabled(LogLevel::Info)) {
            log(LogLevel::Info)
                << "DetectMarkers: " << job.camera->label << " " << width << "x"
                << height << " estimated peak working set " << estimated
                << " bytes (" << (static_cast<double>(estimated) /
                                  (1024.0 * 1024.0))
                << " MB)";
        }
        // minimum_size and minimum_dist are expressed in full-resolution
        // pixels; the detectors see the reduced image, so scale both.
        DetectorParams detector = params;
        FloatImage reduced;
        FloatImage* detect_image = &luminance;
        if (scale > 1) {
            reduced = box_downsample(luminance, scale, inverted);
            detect_image = &reduced;
            detector.minimum_size = params.minimum_size / static_cast<double>(scale);
            detector.minimum_dist = params.minimum_dist / static_cast<double>(scale);
        } else if (inverted) {
            // Invert the only full-resolution buffer in place: no second copy.
            invert_in_place(luminance);
        }
        std::vector<Detection> detections =
            suppress_duplicates(detect_in_image(*detect_image, spec, detector),
                                detector.minimum_dist);
        if (scale > 1) {
            std::vector<RefinementWindow> windows;
            map_and_crop_refinement(detections, scale, spec, luminance, inverted,
                                    windows);
            luminance = FloatImage();  // release the full-resolution buffer
            reduced = FloatImage();
            refine_from_windows(detections, spec, windows, params.max_hamming);
        }
        if (filter_mask && mask != nullptr) {
            detections.erase(
                std::remove_if(detections.begin(), detections.end(),
                               [mask](const Detection& detection) {
                                   return !mask_allows(*mask, detection.center);
                               }),
                detections.end());
        }
        result.detections = std::move(detections);
    } catch (const std::exception& error) {
        result.error = error.what();
    }
    return result;
}

}  // namespace

void DetectMarkersTask::declare_params(Params& params) const {
    ParamSpec target{"target_type", ParamType::String, std::string{"Circular12"},
                     "Target style. Circular12/14/16/20 are coded ring targets; "
                     "Circular is a non-coded ring or disc; Cross is a checker "
                     "(bowtie) or bar cross; AprilTag names a published family. "
                     "Script aliases CircularTarget12bit/14bit/16bit/20bit, "
                     "CircularTarget and CrossTarget are also accepted."};
    target.choices = target_choices();
    params.declare(target);
    ParamSpec tolerance{"tolerance", ParamType::Float, 50.0,
                        "Error-correction strength 0..100. AprilTag: maps to "
                        "clamp(round(tolerance/50 * family default), 0, (h-1)/2). "
                        "Circular: 1 when tolerance >= 75, else 0."};
    tolerance.minimum = 0.0;
    tolerance.maximum = 100.0;
    params.declare(tolerance);
    params.declare({"inverted", ParamType::Bool, false,
                    "Detect light targets on a dark background (the luminance is "
                    "inverted before detection)."});
    params.declare({"noparity", ParamType::Bool, false,
                    "Circular codes ignore the centre-dot parity bit and report "
                    "code = ring index * 2."});
    ParamSpec minimum_size{"minimum_size", ParamType::Float, 0.0,
                           "Minimum target radius in pixels (0 = decoder default)."};
    minimum_size.minimum = 0.0;
    params.declare(minimum_size);
    ParamSpec minimum_dist{"minimum_dist", ParamType::Float, 5.0,
                           "Non-maximum-suppression distance between detections in "
                           "one image, in pixels."};
    minimum_dist.minimum = 0.0;
    params.declare(minimum_dist);
    ParamSpec maximum_residual{"maximum_residual", ParamType::Float, 5.0,
                               "Attachment radius for non-coded detections to the "
                               "predicted projection of an existing marker, in pixels."};
    maximum_residual.minimum = 0.0;
    params.declare(maximum_residual);
    params.declare({"create_markers", ParamType::Bool, true,
                    "Create a marker for a detection that matches no existing one. "
                    "When false such detections are dropped and counted."});
    params.declare({"cameras", ParamType::StringList, std::vector<std::string>{},
                    "Camera labels to process. Empty means every enabled camera; an "
                    "explicit list also selects unaligned cameras."});
    params.declare({"frames", ParamType::StringList, std::vector<std::string>{},
                    "Frame paths, file names or stems to process. Empty means all."});
    params.declare({"filter_mask", ParamType::Bool, true,
                    "Skip detections whose centre is excluded by the camera mask."});
    ParamSpec downscale{"marker_downscale", ParamType::Int, std::int64_t{0},
                        "Detection downscale factor. 0 selects the smallest integer d "
                        "with long_side / d <= 4096. Every style is then refined at "
                        "full resolution in a small local window: circular centres by "
                        "an ellipse fit, checker crosses by a saddle fit, bar crosses "
                        "by re-fitting both ridge lines, and AprilTags by re-decoding "
                        "the quad."};
    downscale.minimum = std::int64_t{0};
    downscale.maximum = std::int64_t{1024};
    params.declare(downscale);
}

std::vector<Stage> DetectMarkersTask::stages() const {
    return {Stage{"detect", 0}, Stage{"apply", 0}};
}

DetectMarkersPlan plan_detect_markers(const Document& document,
                                      std::int32_t chunk_id,
                                      const DetectMarkersOptions& options) {
    DetectMarkersPlan plan;
    const std::string& target_text = options.target_type;
    const ParsedTarget parsed = parse_target(target_text);
    if (parsed.circle21_unsupported) {
        throw ValueError(
            "DetectMarkers: target_type \"AprilTagCircle21h7\" is not supported "
            "yet: its published tags are circular, so no quad candidate exists "
            "(see LIMITATIONS.md)");
    }
    if (!parsed.valid) {
        throw ValueError("DetectMarkers: unknown target_type '" + target_text +
                         "'; valid values are: " + valid_target_names());
    }
    if (!(options.tolerance >= 0.0 && options.tolerance <= 100.0)) {
        throw ValueError("DetectMarkers: tolerance must be in [0, 100]");
    }
    if (!(options.minimum_size >= 0.0)) {
        throw ValueError("DetectMarkers: minimum_size must be non-negative");
    }
    if (!(options.minimum_dist >= 0.0)) {
        throw ValueError("DetectMarkers: minimum_dist must be non-negative");
    }
    if (!(options.maximum_residual >= 0.0)) {
        throw ValueError("DetectMarkers: maximum_residual must be non-negative");
    }
    if (options.marker_downscale < 0 || options.marker_downscale > 1024) {
        throw ValueError("DetectMarkers: marker_downscale must be in [0, 1024]");
    }

    DetectorParams detector;
    detector.noparity = options.noparity;
    detector.minimum_size = options.minimum_size;
    detector.minimum_dist = options.minimum_dist;
    if (parsed.spec.kind == MarkerKind::AprilTag) {
        detector.max_hamming =
            apriltag_max_hamming(options.tolerance, parsed.spec.family);
    } else if (parsed.spec.kind == MarkerKind::CircularCoded) {
        detector.max_hamming = circular_max_hamming(options.tolerance);
    }
    plan.max_hamming = detector.max_hamming;

    const Chunk* chunk = find_chunk_by_id(document, chunk_id);
    if (chunk == nullptr) {
        throw Error("DetectMarkers needs an active chunk");
    }
    const std::vector<std::string>& camera_filter = options.cameras;
    const std::vector<std::string>& frame_filter = options.frames;
    std::vector<std::string> problems;
    for (const std::string& label : camera_filter) {
        bool found = false;
        for (const Camera& camera : chunk->cameras()) {
            if (camera.label == label) {
                found = true;
                break;
            }
        }
        if (!found) {
            problems.push_back("no camera with label '" + label + "'");
        }
    }
    std::vector<CameraJob> jobs;
    for (const Camera& camera : chunk->cameras()) {
        const bool selected = camera_filter.empty()
                                  ? camera.enabled
                                  : contains(camera_filter, camera.label);
        if (!selected) {
            continue;
        }
        const Frame* frame = frame_of_camera(*chunk, camera.id);
        if (frame == nullptr || !frame_matches(*frame, frame_filter)) {
            if (!camera_filter.empty() && frame == nullptr) {
                problems.push_back("camera '" + camera.label +
                                   "' has no linked frame");
            }
            continue;
        }
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
            if (!camera_filter.empty()) {
                problems.push_back("camera '" + camera.label + "' has no sensor");
            }
            continue;
        }
        CameraJob job;
        job.camera = &camera;
        job.frame = frame;
        job.calibration =
            &chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration;
        job.scale = options.marker_downscale;
        jobs.push_back(job);
    }
    if (!problems.empty()) {
        throw ValueError("DetectMarkers: " + problems.front());
    }
    if (!frame_filter.empty() && jobs.empty()) {
        throw ValueError("DetectMarkers: frames filter matched no camera");
    }

    // Memory rule: a camera above 8 MP is processed alone so only one
    // full-resolution luminance buffer is live. The `if` clause keeps the
    // loop serial in that case, and each camera's internal OpenMP image passes
    // can still use every thread because no outer parallel region is open.
    bool parallel_cameras = true;
    for (const CameraJob& job : jobs) {
        if (job.frame != nullptr &&
            static_cast<std::int64_t>(job.frame->width()) *
                    static_cast<std::int64_t>(job.frame->height()) >
                8ll * 1024ll * 1024ll) {
            parallel_cameras = false;
            break;
        }
    }
    std::vector<CameraResult> results(jobs.size());
    const int count = static_cast<int>(jobs.size());
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) if(parallel_cameras)
#endif
    for (int i = 0; i < count; ++i) {
        const Camera& camera = *jobs[static_cast<std::size_t>(i)].camera;
        const Mask* mask =
            options.filter_mask ? mask_of_camera(*chunk, camera.id) : nullptr;
        results[static_cast<std::size_t>(i)] = process_camera(
            document, jobs[static_cast<std::size_t>(i)], parsed.spec, detector,
            options.inverted, options.filter_mask, mask);
    }
    for (const CameraResult& result : results) {
        if (!result.error.empty()) {
            throw Error("DetectMarkers: " + result.error);
        }
    }
    std::map<std::string, std::int32_t> ids;
    ProjectedCameras projected;
    LabelState labels;
    for (const Marker& marker : chunk->markers()) {
        ids[marker.label] = marker.id;
        labels.used.insert(marker.label);
        for (const MarkerProjection& projection : marker.projections) {
            projected[marker.label].insert(projection.camera_id);
        }
    }
    std::vector<Assignment> commands;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        const CameraJob& job = jobs[i];
        const std::vector<Detection>& detections = results[i].detections;
        plan.stats.detected += detections.size();
        Predictions predictions;
        for (const Marker& marker : chunk->markers()) {
            if (!marker.position.has_value()) {
                continue;
            }
            const auto pixel =
                project_marker(*marker.position, *job.camera, *job.calibration);
            if (pixel) {
                predictions.emplace_back(marker.label, *pixel);
            }
        }
        const CameraPlan camera_plan =
            plan_camera(detections, job.camera->id, predictions, ids, projected,
                        options.maximum_residual, options.create_markers, labels);
        plan.stats.created += camera_plan.created;
        plan.stats.dropped += camera_plan.dropped;
        for (const Assignment& assignment : camera_plan.assignments) {
            if (assignment.existing) {
                ++plan.stats.attached;
            }
            commands.push_back(assignment);
        }
    }
    if (!commands.empty()) {
        auto transaction =
            std::make_unique<CommandTransaction>("Detect Markers");
        for (const Assignment& assignment : commands) {
            transaction->add(make_add_marker_with_projection_command(
                chunk->id(), assignment.label, assignment.camera_id,
                assignment.pixel, false));
        }
        plan.command = std::move(transaction);
    }
    return plan;
}

void DetectMarkersTask::apply(Document& document, Params& params,
                              const Progress& progress) {
    DetectMarkersOptions options;
    options.target_type = params.get_string("target_type");
    options.tolerance = params.get_float("tolerance");
    options.inverted = params.get_bool("inverted");
    options.noparity = params.get_bool("noparity");
    options.minimum_size = params.get_float("minimum_size");
    options.minimum_dist = params.get_float("minimum_dist");
    options.maximum_residual = params.get_float("maximum_residual");
    options.create_markers = params.get_bool("create_markers");
    options.cameras = params.get_string_list("cameras");
    options.frames = params.get_string_list("frames");
    options.filter_mask = params.get_bool("filter_mask");
    const std::int64_t downscale = params.get_int("marker_downscale");
    if (downscale < 0 || downscale > 1024) {
        throw ValueError("DetectMarkers: marker_downscale must be in [0, 1024]");
    }
    options.marker_downscale = static_cast<int>(downscale);

    progress.update("detect", 0.0);
    const auto started = std::chrono::steady_clock::now();
    DetectMarkersPlan plan =
        plan_detect_markers(document, document.active_chunk_id(), options);
    progress.update("detect", 1.0);

    progress.update("apply", 0.0);
    if (plan.command) {
        plan.command->apply(document);
    }
    progress.update("apply", 1.0);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "DetectMarkers: target " << options.target_type
                            << " max_hamming " << plan.max_hamming;
        log(LogLevel::Info) << "DetectMarkers: detected " << plan.stats.detected
                            << ", created " << plan.stats.created << ", attached "
                            << plan.stats.attached << ", dropped "
                            << plan.stats.dropped << " in " << seconds << " s";
    }
}

}  // namespace alphashape
