#include "apps/gui/jobs/estimate.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <system_error>
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
    // ProjectIO stores the side-car folder beside the manifest with the final
    // extension removed: dir/name.psx -> dir/name.files.
    std::filesystem::path base(project_path);
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
        row.peak_rss_bytes = static_cast<std::uint64_t>(peak);
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

/// Working-resolution pixels of the enabled cameras. The sensor is the
/// authority; a camera whose sensor is missing falls back to its linked frame.
WorkItem work_for(const Params& params, const Chunk& chunk) {
    WorkItem work;
    double total_pixels = 0.0;
    bool any_aligned = false;
    for (const Camera& camera : chunk.cameras()) {
        any_aligned = any_aligned || camera.enabled;
    }
    for (const Camera& camera : chunk.cameras()) {
        // Before the first alignment no camera is enabled yet, but every image
        // still has to be detected/matched: count them all in that case.
        if (any_aligned && !camera.enabled) {
            continue;
        }
        double pixels = 0.0;
        const std::vector<Sensor>& sensors = chunk.sensors();
        for (std::size_t index = 0; index < sensors.size(); ++index) {
            if (static_cast<std::int32_t>(index) == camera.sensor_id &&
                sensors[index].width > 0 && sensors[index].height > 0) {
                pixels = static_cast<double>(sensors[index].width) *
                         static_cast<double>(sensors[index].height);
                break;
            }
        }
        if (pixels == 0.0) {
            for (const std::shared_ptr<Frame>& frame : chunk.frames()) {
                if (frame != nullptr && frame->camera_id() == camera.id) {
                    pixels = static_cast<double>(frame->width()) *
                             static_cast<double>(frame->height());
                    break;
                }
            }
        }
        work.cameras += 1.0;
        total_pixels += pixels;
    }
    double downscale = 1.0;
    if (const std::optional<std::int64_t> declared = param_int(params, "downscale");
        declared && *declared > 1) {
        downscale = static_cast<double>(*declared);
    }
    work.megapixels = work.cameras > 0.0
                          ? total_pixels / work.cameras / (downscale * downscale) / 1.0e6
                          : 0.0;
    const std::optional<std::int64_t> neighbours = param_int(params, "max_neighbors");
    work.neighbours = neighbours && *neighbours > 0 ? static_cast<double>(*neighbours)
                                                   : kDefaultNeighbours;
    const std::optional<std::int64_t> resolution = param_int(params, "volume_resolution");
    const double volume = resolution && *resolution > 0 ? static_cast<double>(*resolution) : 128.0;
    work.volume_voxels = volume * volume * volume;
    // The output grid's cell count depends on the CRS and the footprint, which
    // are not in the chunk alone. Use the working input pixels as the proxy:
    // an input pixel is at most one output cell after the usual resampling.
    work.product_pixels = work.cameras * work.megapixels * 1.0e6;
    // A texture page carries roughly four texels per input pixel.
    work.texels = work.product_pixels * 4.0;
    return work;
}

}  // namespace

ThroughputRow throughput_row_for(const std::string& task, const Params& params,
                                 const Chunk& chunk, double seconds,
                                 std::uint64_t peak_rss_bytes) {
    const WorkItem work = work_for(params, chunk);
    ThroughputRow row;
    row.task = task;
    row.cameras = work.cameras;
    row.megapixels = work.megapixels;
    row.seconds = seconds;
    row.peak_rss_bytes = peak_rss_bytes;
    return row;
}

Estimate estimate_task(const TaskRegistry& registry, const std::string& task,
                       const Params& params, const Chunk& chunk,
                       const ThroughputHistory& history) {
    Estimate out;
    const bool known = registry.find(task) != nullptr;
    const estimates::ColdTask kind =
        known ? estimates::cold_task(task) : estimates::ColdTask::Generic;
    const WorkItem work = work_for(params, chunk);
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
            }
        }
    }
    if (!(out.seconds >= kMinSeconds)) {
        out.seconds = kMinSeconds;
    }
    out.seconds_sigma_rel = std::max(kSigmaFloor, out.seconds_sigma_rel);
    return out;
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
