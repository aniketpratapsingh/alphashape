// BuildDepthMaps: the dense depth-map stage of the stereo pipeline.
//
// The adapter between the chunk's data model and the stereo:: primitives. For
// every aligned camera that stereo::select_views gives a neighbourhood, the
// camera's luminance image is box-downsampled to the working resolution and its
// reduced calibration derived, stereo::run_patch_match estimates a per-pixel
// depth (distance along the unit bearing) and normals, and
// stereo::filter_depth_map cross-checks the map against its neighbours, leaving
// the retained count as the pixel's confidence.
//
// Everything is staged, then committed with a single assignment: a NEW depth-map
// asset is appended and made active, and every point cloud is dropped (each was
// fused from maps that are no longer active). Images are decoded once and cached
// for the task's lifetime, keyed by camera id.

#include "common/algorithm/stereo/build_depth_maps_task.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/stereo/depth_filter.hpp"
#include "common/algorithm/stereo/patch_match.hpp"
#include "common/algorithm/stereo/patch_match_gpu.hpp"
#include "common/algorithm/stereo/view_selection.hpp"
#include "common/core/assets.hpp"
#include "common/core/frame.hpp"
#include "gpu/gpu.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "image/operators/downsample.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

constexpr int kMaxNeighborsExclusive = 256;

/// Resolve a frame's recorded path against the document's directory (the same
/// rule AddPhotos/DetectFeatures use). A document with no path (unsaved, in
/// memory) has no anchor, so the path is taken as given.
std::filesystem::path resolve_frame_path(const Document& document, const std::string& path) {
    const std::filesystem::path recorded(path);
    if (recorded.is_absolute() || document.path().empty()) {
        return recorded;
    }
    return std::filesystem::path(document.path()).parent_path() / recorded;
}

Eigen::Matrix3d pose_R(const Transform& pose) { return pose.block<3, 3>(0, 0); }
Eigen::Vector3d pose_t(const Transform& pose) { return pose.block<3, 1>(0, 3); }

/// A deterministic per-camera PatchMatch seed: a splitmix64 mix of the camera
/// id, so a run does not depend on thread scheduling.
std::uint64_t seed_for_camera(std::int32_t camera_id) {
    std::uint64_t x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(camera_id));
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/// A camera's luminance image at the working resolution, with the reduced
/// calibration that matches it.
struct LoadedImage {
    FloatImage image;         ///< luminance, downsampled by the task's factor
    Calibration calibration;  ///< scaled_calibration(sensor, downscale)
};

std::size_t valid_count(const DepthMap& map) {
    std::size_t n = 0;
    for (const float d : map.depths) {
        if (d > 0.0F) {
            ++n;
        }
    }
    return n;
}

/// Logs which device the dense stage uses, once per task. The GPU is used only
/// when a context exists and the process-wide switch is on; the kernels then
/// fall back per image through gpu::with_fallback.
void log_device_once() {
    static std::atomic<bool> logged{false};
    if (logged.exchange(true)) {
        return;
    }
    if (gpu::available() && gpu::enabled()) {
        const std::string name = gpu::default_context()->info().name;
        log(LogLevel::Info) << "[GPU 1] " << (name.empty() ? std::string("unknown device") : name);
    } else {
        log(LogLevel::Info) << "[CPU] " << (gpu::enabled()
                                                ? gpu::unavailable_reason()
                                                : std::string("GPU layer disabled"));
    }
}

/// ALPHASHAPE_GPU_SELF_CHECK=1: on the first image, also run the CPU and report
/// gpu::compare stats (relative depth) plus the fraction of pixels within 1%.
bool self_check_enabled() {
    static const bool value = [] {
        const char* text = std::getenv("ALPHASHAPE_GPU_SELF_CHECK");
        return text != nullptr && std::string(text) == "1";
    }();
    return value;
}

void log_self_check(const stereo::PatchMatchResult& gpu_result,
                    const stereo::PatchMatchResult& cpu_result) {
    std::vector<float> relative(gpu_result.depth.size(), 0.0F);
    std::size_t both = 0;
    std::size_t within = 0;
    for (std::size_t p = 0; p < relative.size(); ++p) {
        const float dg = gpu_result.depth[p];
        const float dc = cpu_result.depth[p];
        if (!(dg > 0.0F) || !(dc > 0.0F)) {
            continue;
        }
        ++both;
        const double diff = std::abs(static_cast<double>(dg) - static_cast<double>(dc)) /
                            static_cast<double>(dc);
        relative[p] = static_cast<float>(diff);
        if (diff < 0.01) {
            ++within;
        }
    }
    const std::vector<float> costs(gpu_result.cost.begin(), gpu_result.cost.end());
    const std::vector<float> cpu_costs(cpu_result.cost.begin(), cpu_result.cost.end());
    const gpu::GoldReport depth_report = gpu::compare(
        std::vector<float>(gpu_result.depth.begin(), gpu_result.depth.end()),
        std::vector<float>(cpu_result.depth.begin(), cpu_result.depth.end()));
    const gpu::GoldReport cost_report = gpu::compare(costs, cpu_costs);
    const double fraction = both > 0 ? static_cast<double>(within) / static_cast<double>(both) : 0.0;
    log(LogLevel::Info) << "[GPU self-check] depth mean|d| " << depth_report.mean_abs << ", max|d| "
                        << depth_report.max_abs << "; cost mean|d| " << cost_report.mean_abs
                        << "; within 1% of CPU depth: " << 100.0 * fraction << "% of "
                        << both << " pixels valid in both";
}

}  // namespace

void BuildDepthMapsTask::declare_params(Params& params) const {
    // Defaults follow Metashape's public Python API.
    params.declare({"downscale", ParamType::Int, std::int64_t{4},
                    "Image downscale factor (power of two). Depth maps are built at the "
                    "reduced resolution."});
    params.declare({"filter_mode", ParamType::String, std::string{"mild"},
                    "Depth filter strength: disabled, mild, moderate or aggressive."});
    params.declare({"max_neighbors", ParamType::Int, std::int64_t{16},
                    "Maximum number of neighbour cameras per reference. Must be in [1, 256)."});
    params.declare({"reuse_depth", ParamType::Bool, false,
                    "Keep existing depth maps whose downscale matches for cameras that still "
                    "have usable neighbours."});
    params.declare({"label", ParamType::String, std::string{},
                    "Label for the new depth-map set. Empty selects \"Depth Maps <n>\". Each "
                    "run creates a new set."});
}

std::vector<Stage> BuildDepthMapsTask::stages() const {
    return {Stage{"initialize", 0}, Stage{"buildDepthMaps", 0}, Stage{"filterDepthMaps", 0},
            Stage{"finalize", 0}};
}

void BuildDepthMapsTask::apply(Document& document, Params& params, const Progress& progress) {
    // -- validate every parameter before touching the chunk ------------------
    const std::int64_t downscale_value = params.get_int("downscale");
    const std::string filter_mode_text = params.get_string("filter_mode");
    const std::int64_t max_neighbors = params.get_int("max_neighbors");
    const bool reuse_depth = params.get_bool("reuse_depth");

    if (downscale_value <= 0) {
        throw ValueError("Downscale must be positive");
    }
    if (downscale_value > std::int64_t{1} << 20) {
        throw ValueError("Downscale must be a power of two");
    }
    const int downscale = static_cast<int>(downscale_value);
    if ((downscale & (downscale - 1)) != 0) {
        throw ValueError("Downscale must be a power of two");
    }
    if (max_neighbors < 1 || max_neighbors >= kMaxNeighborsExclusive) {
        throw ValueError("max_neighbors must be in [1, 256)");
    }
    const stereo::FilterMode filter_mode = stereo::parse_filter_mode(filter_mode_text);

    // -- validate the chunk --------------------------------------------------
    progress.update("initialize", 0.0);
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("BuildDepthMaps needs an active chunk");
    }

    std::size_t aligned = 0;
    for (const Camera& camera : chunk->cameras()) {
        if (camera.enabled) {
            ++aligned;
        }
    }
    if (aligned < 2) {
        throw ValueError("BuildDepthMaps: align cameras first");
    }
    const bool have_points =
        std::any_of(chunk->tie_points().tracks.begin(), chunk->tie_points().tracks.end(),
                    [](const Track& track) { return track.point.has_value(); });
    if (!have_points) {
        throw ValueError("BuildDepthMaps: align cameras first");
    }

    // Frame lookup by camera id (first frame wins, as elsewhere in the pipeline)
    // and a sensor lookup, so a camera can be turned into (image, calibration).
    std::unordered_map<std::int32_t, const Frame*> frame_of;
    for (const auto& frame : chunk->frames()) {
        if (frame->camera_id() >= 0) {
            frame_of.emplace(frame->camera_id(), frame.get());
        }
    }

    // -- neighbourhoods and reference-camera centres -------------------------
    stereo::ViewSelectionOptions selection;
    selection.max_neighbors = static_cast<int>(max_neighbors);
    std::vector<stereo::ViewNeighbourhood> neighbourhoods = stereo::select_views(*chunk, selection);
    if (neighbourhoods.empty()) {
        throw ValueError("BuildDepthMaps: no camera has usable neighbour views");
    }

    std::unordered_map<std::int32_t, const Camera*> camera_by_id;
    for (const Camera& camera : chunk->cameras()) {
        camera_by_id[camera.id] = &camera;
    }

    // Existing maps by camera id, for reuse_depth. Read through a CONST view: the
    // non-const accessor would lazily create an empty set on a chunk with none.
    const Chunk& source = *chunk;
    std::unordered_map<std::int32_t, const DepthMap*> existing;
    if (reuse_depth) {
        for (const DepthMap& map : source.depth_maps()) {
            existing.emplace(map.camera_id, &map);
        }
    }
    progress.update("initialize", 1.0);

    // -- build depth maps (sequential over cameras) --------------------------
    // The luminance image of every camera that participates is decoded and
    // downsampled once, cached for the task's lifetime. A missing frame is a
    // hard error: a camera in the neighbourhood graph has a photo by definition.
    std::unordered_map<std::int32_t, std::shared_ptr<LoadedImage>> cache;

    const auto load = [&](std::int32_t camera_id) -> const LoadedImage& {
        const auto found = cache.find(camera_id);
        if (found != cache.end()) {
            return *found->second;
        }
        const auto camera_it = camera_by_id.find(camera_id);
        if (camera_it == camera_by_id.end()) {
            throw ValueError("BuildDepthMaps: depth map references an unknown camera");
        }
        const Camera& camera = *camera_it->second;
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
            throw ValueError("BuildDepthMaps: camera " + std::to_string(camera_id) +
                             " has no sensor");
        }
        const auto frame_it = frame_of.find(camera_id);
        if (frame_it == frame_of.end()) {
            throw ValueError("BuildDepthMaps: camera " + std::to_string(camera_id) +
                             " has no linked frame");
        }
        auto loaded = std::make_shared<LoadedImage>();
        const Image image = read_image(resolve_frame_path(document, frame_it->second->path()).string());
        loaded->image = downsample_box(image.to_luminance(), downscale);
        loaded->calibration = scaled_calibration(
            chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration, downscale);
        cache.emplace(camera_id, loaded);
        return *loaded;
    };

    struct NeighbourhoodViews {
        std::vector<stereo::PMView> views;
        double depth_min = 0.0;
        double depth_max = 0.0;
    };
    const auto build_views = [&](const stereo::ViewNeighbourhood& hood) {
        NeighbourhoodViews out;
        out.depth_min = hood.depth_min;
        out.depth_max = hood.depth_max;
        for (const std::int32_t id : hood.neighbours) {
            const LoadedImage& loaded = load(id);
            const Camera& camera = *camera_by_id.at(id);
            stereo::PMView view;
            view.image = &loaded.image;
            view.calibration = loaded.calibration;
            view.R = pose_R(camera.pose);
            view.t = pose_t(camera.pose);
            out.views.push_back(std::move(view));
        }
        return out;
    };

    std::vector<DepthMap> built;
    built.reserve(neighbourhoods.size());
    std::vector<std::string> errors(neighbourhoods.size());
    std::atomic<bool> self_checked{false};

    progress.update("buildDepthMaps", 0.0);
    for (std::size_t i = 0; i < neighbourhoods.size(); ++i) {
        const stereo::ViewNeighbourhood& hood = neighbourhoods[i];
        try {
            // Reuse an existing map of the right resolution when asked to.
            if (reuse_depth) {
                const auto kept = existing.find(hood.camera_id);
                if (kept != existing.end() && kept->second->downscale == downscale) {
                    built.push_back(*kept->second);
                    progress.update("buildDepthMaps",
                                    static_cast<double>(i + 1) /
                                        static_cast<double>(neighbourhoods.size()));
                    continue;
                }
            }

            const LoadedImage& reference_image = load(hood.camera_id);
            NeighbourhoodViews neighbours = build_views(hood);

            stereo::PMView reference;
            reference.image = &reference_image.image;
            reference.calibration = reference_image.calibration;
            reference.R = pose_R(camera_by_id.at(hood.camera_id)->pose);
            reference.t = pose_t(camera_by_id.at(hood.camera_id)->pose);

            stereo::PatchMatchOptions options;
            options.depth_min = neighbours.depth_min;
            options.depth_max = neighbours.depth_max;
            options.seed = seed_for_camera(hood.camera_id);

            log_device_once();

            // GPU when available and enabled, CPU otherwise or after a device
            // failure (with_fallback logs and counts it once per image).
            const stereo::PatchMatchResult result =
                stereo::run_patch_match_device(reference, neighbours.views, options);

            if (self_check_enabled() && !self_checked.exchange(true)) {
                const stereo::PatchMatchResult cpu_result =
                    stereo::run_patch_match(reference, neighbours.views, options);
                log_self_check(result, cpu_result);
            }

            DepthMap map;
            map.camera_id = hood.camera_id;
            map.width = result.width;
            map.height = result.height;
            map.downscale = downscale;
            map.depths.assign(result.depth.begin(), result.depth.end());
            map.normals.assign(result.normals.begin(), result.normals.end());
            built.push_back(std::move(map));
        } catch (const std::exception& e) {
            errors[i] = "camera " + std::to_string(hood.camera_id) + ": " + e.what();
        }
        progress.update("buildDepthMaps",
                        static_cast<double>(i + 1) / static_cast<double>(neighbourhoods.size()));
        if (progress.is_cancelled()) {
            throw CancelledError("BuildDepthMaps was cancelled");
        }
    }
    for (const std::string& error : errors) {
        if (!error.empty()) {
            throw Error("BuildDepthMaps: " + error);
        }
    }
    progress.update("buildDepthMaps", 1.0);

    // -- filter depth maps against their neighbourhoods ----------------------
    // Maps are looked up by camera id from the staging set, so a reference and
    // its filtered neighbours never disagree about the working resolution.
    std::unordered_map<std::int32_t, const DepthMap*> by_id;
    for (const DepthMap& map : built) {
        by_id[map.camera_id] = &map;
    }
    std::unordered_map<std::int32_t, const stereo::ViewNeighbourhood*> hood_by_id;
    for (const auto& hood : neighbourhoods) {
        hood_by_id[hood.camera_id] = &hood;
    }

    std::vector<DepthMap> filtered;
    filtered.reserve(built.size());
    const std::size_t camera_count = built.size();

    progress.update("filterDepthMaps", 0.0);
    for (std::size_t i = 0; i < built.size(); ++i) {
        const DepthMap& map = built[i];
        const auto hood_it = hood_by_id.find(map.camera_id);
        if (hood_it == hood_by_id.end()) {
            filtered.push_back(map);
            continue;
        }
        const Camera& camera = *camera_by_id.at(map.camera_id);

        stereo::DepthView reference;
        reference.map = &map;
        reference.calibration = scaled_calibration(
            chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration, downscale);
        reference.R = pose_R(camera.pose);
        reference.t = pose_t(camera.pose);

        std::vector<stereo::DepthView> neighbours;
        for (const std::int32_t id : hood_it->second->neighbours) {
            const auto nb = by_id.find(id);
            if (nb == by_id.end()) {
                continue;  // that neighbour produced no map
            }
            const Camera& nb_camera = *camera_by_id.at(id);
            stereo::DepthView view;
            view.map = nb->second;
            view.calibration = scaled_calibration(
                chunk->sensors()[static_cast<std::size_t>(nb_camera.sensor_id)].calibration,
                downscale);
            view.R = pose_R(nb_camera.pose);
            view.t = pose_t(nb_camera.pose);
            neighbours.push_back(std::move(view));
        }

        stereo::FilterOptions filter_options;
        filter_options.mode = filter_mode;

        const std::size_t before = valid_count(map);
        DepthMap result = stereo::filter_depth_map(reference, neighbours, filter_options);
        const std::size_t after = valid_count(result);
        filtered.push_back(std::move(result));

        if (Log::enabled(LogLevel::Info)) {
            const double total = static_cast<double>(map.depths.size());
            log(LogLevel::Info) << "BuildDepthMaps: camera " << map.camera_id << ": "
                                << (total > 0.0 ? 100.0 * static_cast<double>(before) / total : 0.0)
                                << "% valid -> "
                                << (total > 0.0 ? 100.0 * static_cast<double>(after) / total : 0.0)
                                << "% after " << stereo::to_string(filter_mode) << " filtering";
        }
        progress.update("filterDepthMaps",
                        static_cast<double>(i + 1) / static_cast<double>(camera_count));
        if (progress.is_cancelled()) {
            throw CancelledError("BuildDepthMaps was cancelled");
        }
    }
    progress.update("filterDepthMaps", 1.0);

    // -- commit atomically ---------------------------------------------------
    progress.update("finalize", 0.0);
    Chunk staged = std::move(*chunk);
    // Each run appends a NEW depth-map set; earlier sets are kept.
    DepthMapsAsset& set = staged.add_depth_maps(params.get_string("label"));
    *set.data = std::move(filtered);
    // Every point cloud is derived from depth maps, so the ones already in the
    // chunk no longer describe the active maps. Drop them rather than leave a
    // cloud that looks current but was fused from different maps.
    staged.point_clouds().clear();
    staged.set_active_point_cloud_id(-1);
    *chunk = std::move(staged);
    progress.update("finalize", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "BuildDepthMaps: " << chunk->depth_maps().size() << " of "
                            << chunk->cameras().size() << " cameras at downscale " << downscale
                            << ", filter " << stereo::to_string(filter_mode) << " -> asset "
                            << chunk->active_depth_maps_id() << " '"
                            << chunk->depth_map_sets().back().label << "'";
    }
}

}  // namespace alphashape
