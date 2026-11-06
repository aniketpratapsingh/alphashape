#include "apps/gui/jobs/estimate.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <system_error>
#include <unordered_set>
#include <utility>

#include "apps/gui/jobs/estimate_model.hpp"
#include "base/platform/memory.hpp"
#include "common/core/chunk.hpp"
#include "common/core/frame.hpp"
#include "common/core/types.hpp"

namespace alphashape::gui {
namespace {

constexpr double kDefaultNeighbours = 8.0;
constexpr double kColdSigma = 0.35;
constexpr double kSigmaFloor = 0.10;
constexpr double kMinSeconds = 0.05;

bool finite_non_negative(double value) {
    return std::isfinite(value) && value >= 0.0;
}

// A declared parameter read that can never escape as an exception.
std::optional<std::int64_t> param_int(const Params& params, const std::string& name) {
    try {
        if (params.declares(name)) {
            return params.get_int(name);
        }
    } catch (...) {
    }
    return std::nullopt;
}

double percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

}  // namespace

std::uint64_t recorded_peak_rss(std::uint64_t before, std::uint64_t after) {
    // The counter is process-wide: a run only gets credit when it lifted it.
    return after > before ? after : 0ULL;
}

std::uint64_t measured_ram_from_history(const std::vector<ThroughputRow>& rows,
                                        double current_units, std::uint64_t fallback) {
    std::vector<std::uint64_t> peaks;
    std::vector<double> units;
    peaks.reserve(rows.size());
    units.reserve(rows.size());
    for (const ThroughputRow& row : rows) {
        if (row.peak_rss_bytes == 0) {
            continue;  // unknown: the run did not raise the high-water mark
        }
        const double row_units = row.cameras * row.megapixels;
        if (!(row_units > 0.0)) {
            continue;
        }
        peaks.push_back(row.peak_rss_bytes);
        units.push_back(row_units);
    }
    if (peaks.size() < 3 || !(current_units > 0.0)) {
        return fallback;
    }
    std::sort(peaks.begin(), peaks.end());
    std::sort(units.begin(), units.end());
    const double median_peak = percentile(
        std::vector<double>(peaks.begin(), peaks.end()), 0.5);
    const double median_units = percentile(units, 0.5);
    if (!(median_units > 0.0)) {
        return fallback;
    }
    return estimates::saturating_bytes(median_peak * (current_units / median_units));
}

void ThroughputHistory::add(ThroughputRow row) {
    if (row.task.empty() || !finite_non_negative(row.cameras) ||
        !finite_non_negative(row.megapixels) || !finite_non_negative(row.seconds)) {
        return;
    }
    std::vector<ThroughputRow>& rows = by_task_[row.task];
    rows.push_back(std::move(row));
    if (rows.size() > kMaxRowsPerTask) {
        rows.erase(rows.begin());
    }
}

const std::vector<ThroughputRow>* ThroughputHistory::rows(const std::string& task) const {
    const auto it = by_task_.find(task);
    return it == by_task_.end() ? nullptr : &it->second;
}

std::vector<ThroughputRow> ThroughputHistory::all() const {
    std::vector<ThroughputRow> out;
    for (const auto& entry : by_task_) {
        out.insert(out.end(), entry.second.begin(), entry.second.end());
    }
    return out;
}

std::size_t ThroughputHistory::size() const {
    std::size_t count = 0;
    for (const auto& entry : by_task_) {
        count += entry.second.size();
    }
    return count;
}

std::string history_path_for_project(const std::string& project_path) {
    if (project_path.empty()) {
        return {};
    }
    // TODO(WP-11.9): use the shared .files helper once it lands (PR #101).
    // A .psz project keeps its directory layout to itself, so its history sits
    // beside the manifest as dir/name.psz.history.tsv rather than creating a
    // dir/name.files/ folder the format does not have.
    std::filesystem::path base(project_path);
    if (base.extension() == ".psz") {
        return project_path + ".history.tsv";
    }
    // ProjectIO stores the side-car folder beside the manifest with the final
    // extension removed: dir/name.psx -> dir/name.files.
    base.replace_extension();
    return base.string() + ".files/job_history.tsv";
}

FileThroughputStore::FileThroughputStore(std::string path) : path_(std::move(path)) {}

ThroughputHistory FileThroughputStore::load() {
    ThroughputHistory history;
    if (path_.empty()) {
        return history;
    }
    std::ifstream input(path_);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream fields(line);
        ThroughputRow row;
        double peak = 0.0;
        std::string extra;
        if (!(fields >> std::quoted(row.task) >> row.cameras >> row.megapixels >>
              row.seconds >> peak) ||
            (fields >> extra)) {
            continue;  // malformed: one bad line must not lose the rest
        }
        if (!finite_non_negative(row.cameras) || !finite_non_negative(row.megapixels) ||
            !finite_non_negative(row.seconds) || !std::isfinite(peak) || peak < 0.0) {
            continue;
        }
        // A finite but enormous peak (1e300) must saturate, not invoke UB.
        row.peak_rss_bytes = estimates::saturating_bytes(peak);
        history.add(std::move(row));
    }
    return history;
}

bool FileThroughputStore::append(const ThroughputRow& row) {
    if (path_.empty()) {
        return false;
    }
    ThroughputHistory history = load();
    history.add(row);
    return write(history);
}

bool FileThroughputStore::write(const ThroughputHistory& history) {
    const std::filesystem::path target(path_);
    std::error_code ec;
    if (target.has_parent_path()) {
        std::filesystem::create_directories(target.parent_path(), ec);
    }
    const std::filesystem::path temp = target.string() + ".tmp";
    {
        std::ofstream output(temp, std::ios::trunc);
        if (!output) {
            return false;
        }
        for (const ThroughputRow& row : history.all()) {
            std::ostringstream line;
            line << std::quoted(row.task) << '\t' << std::setprecision(17) << row.cameras
                 << '\t' << row.megapixels << '\t' << row.seconds << '\t'
                 << static_cast<double>(row.peak_rss_bytes) << '\n';
            output << line.str();
        }
        output.flush();
        if (!output) {
            std::filesystem::remove(temp, ec);
            return false;
        }
    }
    std::filesystem::rename(temp, target, ec);
    if (ec) {
        // A rename onto an existing file is not portable on Windows; replace it.
        std::filesystem::remove(target, ec);
        ec.clear();
        std::filesystem::rename(temp, target, ec);
    }
    if (ec) {
        std::filesystem::remove(temp, ec);
        return false;
    }
    return true;
}

namespace {

using estimates::WorkItem;

/// Tasks whose working resolution is the active depth-map set: they never look
/// at the sensor pixels, so their cost follows whatever downscale produced the
/// maps (typically 4, not the sensor's full resolution).
bool consumes_depth_maps(const std::string& task) {
    return task == "BuildPointCloud" || task == "BuildModel";
}

bool any_camera_enabled(const ChunkFacts& facts) {
    for (const ChunkFacts::CameraFacts& camera : facts.cameras) {
        if (camera.enabled) {
            return true;
        }
    }
    return false;
}

/// Working-resolution pixels of the enabled cameras. The sensor is the
/// authority; a camera whose sensor is missing falls back to its linked frame.
/// Tasks that consume depth maps take their resolution from the active
/// depth-map set instead (see below).
WorkItem work_for(const std::string& task, const Params& params, const ChunkFacts& facts) {
    WorkItem work;
    double total_pixels = 0.0;
    const bool any_aligned = any_camera_enabled(facts);
    for (const ChunkFacts::CameraFacts& camera : facts.cameras) {
        // Before the first alignment no camera is enabled yet, but every image
        // still has to be detected/matched: count them all in that case.
        if (any_aligned && !camera.enabled) {
            continue;
        }
        work.cameras += 1.0;
        total_pixels += camera.pixels;
        // MatchPhotos reuses existing keypoint sets and detects only what is
        // missing, so its detection term covers these cameras alone.
        if (task == "MatchPhotos" && camera.has_frame && !camera.has_keypoints) {
            work.detect_cameras += 1.0;
        }
    }
    if (consumes_depth_maps(task) && facts.depth_map_count > 0.0) {
        // The depth maps ARE the task's input and already carry the builder's
        // downscale, so their per-map size is the working resolution.
        work.cameras = facts.depth_map_count;
        work.megapixels = facts.depth_map_pixels / work.cameras / 1.0e6;
    } else {
        double downscale = 1.0;
        if (const std::optional<std::int64_t> declared = param_int(params, "downscale");
            declared && *declared > 1) {
            downscale = static_cast<double>(*declared);
        }
        if (consumes_depth_maps(task)) {
            // No depth maps yet: assume the default 4x working resolution a
            // BuildDepthMaps run would produce.
            downscale = 4.0;
        }
        work.megapixels = work.cameras > 0.0
                              ? total_pixels / work.cameras / (downscale * downscale) / 1.0e6
                              : 0.0;
    }
    const std::optional<std::int64_t> neighbours = param_int(params, "max_neighbors");
    work.neighbours = neighbours && *neighbours > 0 ? static_cast<double>(*neighbours)
                                                   : kDefaultNeighbours;
    const std::optional<std::int64_t> resolution = param_int(params, "volume_resolution");
    const double volume = resolution && *resolution > 0 ? static_cast<double>(*resolution) : 128.0;
    work.volume_voxels = volume * volume * volume;
    // Output cells. Duplicated camera coverage must not scale the product
    // linearly: prefer an existing grid. An ortho drapes on the DEM, and a DEM
    // rebuild replaces the existing one. Without either, a DEM falls back to
    // the dense cloud's points, then to the coverage proxy divided by the mean
    // overlap (cameras / 4 for a typical nadir block). Other tasks use the
    // coverage proxy.
    const double coverage_pixels = work.cameras * work.megapixels * 1.0e6;
    if (task == "BuildOrthomosaic" && facts.dem_cells > 0) {
        work.product_pixels = static_cast<double>(facts.dem_cells);
    } else if (task == "BuildDem") {
        if (facts.dem_cells > 0) {
            work.product_pixels = static_cast<double>(facts.dem_cells);
        } else if (facts.point_count > 0) {
            work.product_pixels = static_cast<double>(facts.point_count);
        } else if (work.cameras > 0.0) {
            work.product_pixels = coverage_pixels / (work.cameras / 4.0);
        }
    } else {
        work.product_pixels = coverage_pixels;
    }
    // A texture page carries roughly four texels per output cell.
    work.texels = work.product_pixels * 4.0;
    return work;
}

}  // namespace

ChunkFacts chunk_facts(const Chunk& chunk) {
    ChunkFacts facts;
    std::unordered_set<std::int32_t> keypoint_cameras;
    for (const KeypointSet& set : chunk.tie_points().keypoint_sets) {
        keypoint_cameras.insert(set.camera_id);
    }
    for (const Camera& camera : chunk.cameras()) {
        ChunkFacts::CameraFacts entry;
        entry.enabled = camera.enabled;
        entry.has_keypoints = keypoint_cameras.count(camera.id) != 0;
        if (camera.sensor_id >= 0 &&
            camera.sensor_id < static_cast<std::int32_t>(chunk.sensors().size())) {
            const Sensor& sensor =
                chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)];
            if (sensor.width > 0 && sensor.height > 0) {
                entry.pixels = static_cast<double>(sensor.width) *
                               static_cast<double>(sensor.height);
            }
        }
        for (const std::shared_ptr<Frame>& frame : chunk.frames()) {
            if (frame != nullptr && frame->camera_id() == camera.id) {
                entry.has_frame = true;
                if (entry.pixels == 0.0) {
                    entry.pixels = static_cast<double>(frame->width()) *
                                   static_cast<double>(frame->height());
                }
                break;
            }
        }
        facts.cameras.push_back(entry);
    }
    for (const DepthMap& map : chunk.depth_maps()) {
        if (map.width > 0 && map.height > 0) {
            facts.depth_map_pixels += static_cast<double>(map.width) *
                                      static_cast<double>(map.height);
            facts.depth_map_count += 1.0;
        }
    }
    if (const ElevationAsset* active =
            find_asset(chunk.elevations(), chunk.active_elevation_id());
        active != nullptr && active->data != nullptr && !active->data->raster.empty()) {
        facts.dem_cells =
            static_cast<std::uint64_t>(active->data->raster.grid.width) *
            static_cast<std::uint64_t>(active->data->raster.grid.height);
    }
    facts.point_count = static_cast<std::uint64_t>(chunk.point_cloud().points.size());
    return facts;
}

double estimate_camera_count(const std::string& task, const ChunkFacts& facts) {
    if (consumes_depth_maps(task) && facts.depth_map_count > 0.0) {
        return facts.depth_map_count;
    }
    double count = 0.0;
    const bool any_aligned = any_camera_enabled(facts);
    for (const ChunkFacts::CameraFacts& camera : facts.cameras) {
        if (!any_aligned || camera.enabled) {
            count += 1.0;
        }
    }
    return count;
}

ThroughputRow throughput_row_for(const std::string& task, const Params& params,
                                 const Chunk& chunk, double seconds,
                                 std::uint64_t peak_rss_bytes) {
    const WorkItem work = work_for(task, params, chunk_facts(chunk));
    ThroughputRow row;
    row.task = task;
    row.cameras = work.cameras;
    row.megapixels = work.megapixels;
    row.seconds = seconds;
    row.peak_rss_bytes = peak_rss_bytes;
    return row;
}

Estimate estimate_task(const TaskRegistry& registry, const std::string& task,
                       const Params& params, const ChunkFacts& facts,
                       const ThroughputHistory& history) {
    Estimate out;
    const bool known = registry.find(task) != nullptr;
    const estimates::ColdTask kind =
        known ? estimates::cold_task(task) : estimates::ColdTask::Generic;
    const WorkItem work = work_for(task, params, facts);
    const estimates::ColdEstimate cold = estimates::cold_estimate(kind, work);

    out.seconds = cold.seconds;
    out.seconds_sigma_rel = kColdSigma;
    out.peak_ram_bytes = cold.peak_ram_bytes;
    out.output_bytes = cold.output_bytes;
    out.disk_bytes = cold.disk_bytes;
    out.basis = kind == estimates::ColdTask::Generic ? "cold model (generic)" : "cold model";

    const std::vector<ThroughputRow>* rows = history.rows(task);
    if (known && rows != nullptr && rows->size() >= 3) {
        std::vector<double> rates;
        rates.reserve(rows->size());
        for (const ThroughputRow& row : *rows) {
            const double units = row.cameras * row.megapixels;
            if (units > 0.0 && row.seconds > 0.0) {
                rates.push_back(row.seconds / units);
            }
        }
        if (rates.size() >= 3) {
            std::sort(rates.begin(), rates.end());
            const double median = percentile(rates, 0.5);
            const double iqr = percentile(rates, 0.75) - percentile(rates, 0.25);
            const double units = work.cameras * work.megapixels;
            if (median > 0.0) {
                out.seconds = median * units;
                out.seconds_sigma_rel =
                    std::max(kSigmaFloor, iqr / median);
                out.confident = true;
                out.basis = "measured (last " + std::to_string(rows->size()) + " runs)";
                // History beats the cold RAM model too, but only when at least
                // three runs actually recorded a peak (see remembered rule).
                out.peak_ram_bytes =
                    measured_ram_from_history(*rows, units, cold.peak_ram_bytes);
            }
        }
    }
    if (!(out.seconds >= kMinSeconds)) {
        out.seconds = kMinSeconds;
    }
    out.seconds_sigma_rel = std::max(kSigmaFloor, out.seconds_sigma_rel);
    return out;
}

Estimate estimate_task(const TaskRegistry& registry, const std::string& task,
                       const Params& params, const Chunk& chunk,
                       const ThroughputHistory& history) {
    return estimate_task(registry, task, params, chunk_facts(chunk), history);
}

std::uint64_t estimate_memory_budget() {
    const std::uint64_t budget = platform::memory_budget_bytes();
    return budget != 0 ? budget : platform::total_physical_memory_bytes();
}

bool needs_ram_warning(const Estimate& estimate, std::uint64_t budget) {
    if (budget == 0) {
        return false;
    }
    return static_cast<double>(estimate.peak_ram_bytes) >
           0.75 * static_cast<double>(budget);
}

}  // namespace alphashape::gui
