// BuildDepthMaps: the dense depth-map stage of the stereo pipeline.
//
// The adapter between the chunk's data model and the stereo:: primitives. For
// every aligned camera that stereo::select_views gives a neighbourhood, the
// camera's luminance image is box-downsampled to the working resolution and its
// reduced calibration derived, stereo::run_patch_match estimates a per-pixel
// depth (distance along the unit bearing) and normals, and
// stereo::filter_depth_map cross-checks the map against its neighbours, leaving
// the retained count as the pixel's confidence and, when repair_holes is on,
// stereo::repair_depth_map refills small holes the filter opened inside an
// otherwise consistent surface.
//
// INCREMENTAL. A camera's search is hours of work at downscale 2, so the run is
// checkpointed in this task's own terms: every finished map stays staged, and a
// cancellation stores the maps finished so far before rethrowing CancelledError.
// Those maps are unfiltered (no cross-view pass has run yet) but complete per
// camera, carry their alignment fingerprint, and are reused by the next run with
// `reuse_depth` — which then filters them properly against the neighbourhood
// that finally exists.
//
// REUSE. `reuse_depth` (default on) skips the PatchMatch search for a camera
// whose existing map matches BOTH the working downscale and the camera's
// current geometry AND neighbourhood: depth_map_fingerprint() over the pose,
// the sensor calibration and the downscale, extended with the neighbour ids,
// their poses and calibrations, the depth range, the reference image file and
// the PatchMatch options. Re-aligning a camera, re-calibrating a sensor or
// changing max_neighbors therefore invalidates every affected map instead of
// silently consuming stale ones. A map with fingerprint 0 (saved by a build
// that did not record one) is never reused.
//
// Progress inside one camera comes from stereo::PatchMatchOptions::on_sweep,
// which the CPU and GPU paths both call after every refinement half-sweep; the
// hook also polls for cancellation, so Cancel takes effect within one sweep
// rather than after the current camera.
//
// Everything else is committed with a single assignment. A finished run appends
// a NEW depth-map asset and makes it active (or completes the "partial" set it
// resumed from), and drops every point cloud (each was fused from maps that are
// no longer active). A CANCELLED run commits depth maps only: point clouds and
// every other product are left untouched. Images are decoded once and cached
// for the task's lifetime, keyed by camera id.

#include "common/algorithm/stereo/build_depth_maps_task.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
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
#include "common/algorithm/stereo/depth_repair.hpp"
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

/// Upper bound on repair_max_area. A hole this large at the working resolution
/// is no longer a filter artefact, and the repair pass must not spend its time
/// on one; 1e6 pixels is a quarter of a 2000x2000 map.
constexpr std::int64_t kMaxRepairArea = 1000000;

/// Set metadata marking a depth-map set a CANCELLED run left behind: every map
/// in it is per-camera complete but has not been through the cross-view filter.
/// It is how a later run recognises the set to merge into (further cancels) or
/// to replace (on success) instead of appending yet another set. Persisted with
/// the asset's <meta>, so it survives a save/load.
constexpr const char* kPartialStatusKey = "status";
constexpr const char* kPartialStatusValue = "partial";

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

/// Counts PatchMatch searches started by this task (test hook). Deliberately
/// not per-task-instance: the tests drive a task through the registry and need
/// to count across several apply() calls.
std::atomic<std::uint64_t> g_search_count{0};

}  // namespace

std::uint64_t depth_map_search_count() { return g_search_count.load(); }

void reset_depth_map_search_count() { g_search_count.store(0); }

void BuildDepthMapsTask::declare_params(Params& params) const {
    ParamSpec downscale{"downscale", ParamType::Int, std::int64_t{4},
                        "Image downscale factor (power of two). Depth maps are built at the "
                        "reduced resolution."};
    downscale.minimum = std::int64_t{1};
    downscale.power_of_two = true;
    params.declare(downscale);

    // "moderate" is the default because it is measurably better on real aerial
    // blocks at no time cost: against "mild" on the aerial-nadir benchmark,
    // moderate left 21% less hole area, with DSM NMAD 0.653 -> 0.574 m and p95
    // 2.51 -> 2.12 m. "mild" restores the previous behaviour.
    ParamSpec filter_mode{"filter_mode", ParamType::String, std::string{"moderate"},
                          "Depth filter strength: \"disabled\", \"mild\", \"moderate\" (default) "
                          "or \"aggressive\". A stronger mode rejects more weakly supported "
                          "pixels, keeping fewer but more reliable ones."};
    filter_mode.choices = {"disabled", "mild", "moderate", "aggressive"};
    params.declare(filter_mode);

    // 8 is the default because a wider neighbourhood costs search time and buys
    // nothing measurable: against 16 on the aerial-nadir benchmark, the depth
    // stage was 30-40% faster and coverage, hole area and DSM accuracy were equal
    // or slightly better. 16 restores the previous behaviour.
    ParamSpec max_neighbors{"max_neighbors", ParamType::Int, std::int64_t{8},
                            "Maximum number of neighbour cameras per reference (default 8). "
                            "More neighbours cost search time without improving the map on "
                            "typical blocks. Must be in [1, 256)."};
    max_neighbors.minimum = std::int64_t{1};
    max_neighbors.maximum = static_cast<std::int64_t>(kMaxNeighborsExclusive) - 1;
    params.declare(max_neighbors);

    // On by default: a finished map is expensive, and a re-run after a cancel
    // should continue rather than start over. Only a map whose downscale AND
    // alignment fingerprint match is reused, so the default cannot consume a
    // stale map after a re-alignment.
    params.declare({"reuse_depth", ParamType::Bool, true,
                    "Reuse existing depth maps for cameras whose map was built at this "
                    "downscale and against the current camera alignment and calibration. A "
                    "map from before a re-alignment is rebuilt rather than reused."});
    params.declare({"label", ParamType::String, std::string{},
                    "Label for the new depth-map set. Empty selects \"Depth Maps <n>\". Each "
                    "run creates a new set."});

    // Coarse-to-fine search (WP-17.14). Automatic is the default because it is
    // what makes the stage faster: the coarse levels decide the basin at a
    // quarter of the pixels each, and the reference-resolution level only has to
    // refine. 1 restores the pre-pyramid search exactly.
    ParamSpec pyramid_levels{"pyramid_levels", ParamType::Int, std::int64_t{0},
                             "Coarse-to-fine depth-search levels: 0 = automatic (halve until the "
                             "shorter side reaches pyramid_min_size), 1 = a single level at the "
                             "working resolution (no pyramid), N = exactly N levels."};
    pyramid_levels.minimum = std::int64_t{0};
    pyramid_levels.maximum = std::int64_t{8};
    params.declare(pyramid_levels);

    ParamSpec pyramid_min_size{"pyramid_min_size", ParamType::Int, std::int64_t{256},
                               "Shortest side, in pixels, an automatic pyramid stops shrinking "
                               "at. Ignored when pyramid_levels is not 0."};
    pyramid_min_size.minimum = std::int64_t{32};
    pyramid_min_size.maximum = std::int64_t{8192};
    params.declare(pyramid_min_size);

    // Photometric cost (DP-03). Census is opt-in because it costs one extra
    // transform per view; it is the more robust of the two on real photo sets
    // (exposure / white-balance differences, low texture), which is where depth
    // map holes come from. "zncc" is the original intensity cost.
    ParamSpec cost_type{"cost_type", ParamType::String, std::string{"zncc"},
                        "Photometric cost the depth search minimises: \"zncc\" (intensity "
                        "normalised cross-correlation) or \"census\" (census descriptors "
                        "correlated by the same ZNCC; more robust to lighting differences and "
                        "low texture)."};
    cost_type.choices = {"zncc", "census"};
    params.declare(cost_type);

    // max_cost is on the selected cost's own scale -- intensity (1 - ZNCC) in
    // [0, 2], census (1 - ZNCC) / 2 in [0, 1] -- and its useful value differs
    // between the two costs. -1 means "whatever this cost's default is"
    // (patch_match_default_max_cost()), which is how a run gets a sensible
    // threshold without the caller having to know either scale.
    ParamSpec max_cost{"max_cost", ParamType::Float, -1.0,
                       "Final pixels whose cost at the selected cost_type's own scale is above "
                       "this are left empty (depth -1). The scale is [0, 2] for zncc "
                       "(1 - ZNCC) and [0, 1] for census ((1 - ZNCC) / 2). -1 selects the cost "
                       "type's own default (0.5 for zncc, 0.6 for census)."};
    max_cost.minimum = -1.0;
    max_cost.maximum = 2.0;
    params.declare(max_cost);

    // Post-filter hole repair. Off by default: it synthesises depth the search
    // never measured, so a run opts in explicitly. Only maps REPAIRED this way
    // take the new fingerprint component (see fingerprint_of), so turning it on
    // and off does not invalidate maps built without it.
    params.declare({"repair_holes", ParamType::Bool, false,
                    "After cross-view filtering, refill small holes left inside an otherwise "
                    "consistent surface. A hole is filled only when one plane explains its "
                    "measured boundary, no coherent second surface owns an arc of that "
                    "boundary, and it does not straddle a depth discontinuity; the filled "
                    "pixels are marked with a reduced confidence. Has no effect when "
                    "filter_mode is \"disabled\": without the filter there are no holes to "
                    "repair, only genuine no-data, and inventing depth there would be wrong."});
    ParamSpec repair_max_area{"repair_max_area", ParamType::Int, std::int64_t{2000},
                              "Largest hole, in pixels at the working resolution, that "
                              "repair_holes may fill (default 2000). Larger holes are left "
                              "empty. Ignored (and not validated) when repair_holes is "
                              "false or filter_mode is \"disabled\"."};
    repair_max_area.minimum = std::int64_t{0};
    repair_max_area.maximum = kMaxRepairArea;
    params.declare(repair_max_area);
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
    const std::int64_t pyramid_levels_value = params.get_int("pyramid_levels");
    const std::int64_t pyramid_min_size = params.get_int("pyramid_min_size");
    const std::string cost_type_text = params.get_string("cost_type");
    const double max_cost_value = params.get_float("max_cost");
    const bool repair_holes = params.get_bool("repair_holes");
    const std::int64_t repair_max_area = params.get_int("repair_max_area");

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
    // The search's own validation also rejects these, but a task must name a bad
    // parameter before it touches the chunk (and before it looks at 300 photos
    // to decide it cannot use them).
    if (pyramid_levels_value < 0 || pyramid_levels_value > 8) {
        throw ValueError("pyramid_levels must be in [0, 8]");
    }
    if (pyramid_min_size < 32 || pyramid_min_size > 8192) {
        throw ValueError("pyramid_min_size must be in [32, 8192]");
    }
    // The cost type is a closed set; the name is validated here rather than
    // defaulted, so a typo fails before the chunk is touched (and before the
    // CLI writes an unknown value into the project file).
    stereo::CostType cost_type = stereo::CostType::Zncc;
    if (cost_type_text == "census") {
        cost_type = stereo::CostType::Census;
    } else if (cost_type_text != "zncc") {
        throw ValueError("cost_type must be \"zncc\" or \"census\"");
    }
    if (max_cost_value < -1.0 || max_cost_value > 2.0) {
        throw ValueError("max_cost must be in [-1, 2]");
    }
    // Only validate repair_max_area when the pass will actually run. With
    // repair off (or the filter disabled) the value is dead configuration, and
    // rejecting a stored project over a parameter nothing reads would be
    // gratuitous. The parse below resolves the filter mode, so validate after.
    const stereo::FilterMode filter_mode = stereo::parse_filter_mode(filter_mode_text);
    if (repair_holes && filter_mode != stereo::FilterMode::Disabled) {
        if (repair_max_area < 0) {
            throw ValueError("repair_max_area must not be negative");
        }
        if (repair_max_area > kMaxRepairArea) {
            throw ValueError("repair_max_area must be at most " + std::to_string(kMaxRepairArea));
        }
    }
    // -1 means "the cost type's own default", resolved once here so every map of
    // the run (and the log line) uses the same number.
    const double max_cost = max_cost_value < 0.0
                                ? stereo::patch_match_default_max_cost(cost_type)
                                : max_cost_value;

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

    // The neighbourhood of every camera, for the fingerprint AND for the reuse
    // test that reads it. select_views above already produced it in camera
    // order; index by id so a map can be fingerprinted where it is stamped.
    std::unordered_map<std::int32_t, const stereo::ViewNeighbourhood*> hood_of;
    for (const stereo::ViewNeighbourhood& hood : neighbourhoods) {
        hood_of[hood.camera_id] = &hood;
    }

    // The PatchMatch options every map of this run is searched with. Fixed here
    // rather than inline so the fingerprint stamps exactly what ran.
    const int kWindowRadius = stereo::PatchMatchOptions{}.window_radius;
    const int kWindowStep = stereo::PatchMatchOptions{}.window_step;
    const int kIterations = stereo::PatchMatchOptions{}.iterations;
    const int kPyramidLevels = static_cast<int>(pyramid_levels_value);
    const int kPyramidMinSize = static_cast<int>(pyramid_min_size);

    // The geometry AND neighbourhood a depth map for this camera at this
    // downscale must have been built from to be reusable. Computed once per
    // camera for the reuse test and stamped on every map this run produces.
    //
    // It deliberately covers the WHOLE neighbourhood: moving a neighbour
    // changes this camera's map (the search compares against that neighbour's
    // pixels), so a re-alignment must invalidate every map that had the moved
    // camera as a neighbour, not just the moved camera's own map.
    const auto fingerprint_of = [&](std::int32_t camera_id) -> std::uint64_t {
        const Camera& camera = *camera_by_id.at(camera_id);
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
            return 0;  // no sensor: nothing can be reused, and the mismatch is caught in load()
        }
        DepthMapFingerprintContext context;
        context.max_neighbors = static_cast<int>(max_neighbors);
        context.window_radius = kWindowRadius;
        context.window_step = kWindowStep;
        context.iterations = kIterations;
        context.seed = seed_for_camera(camera_id);
        // The number of levels this camera's search runs at, resolved from the
        // request and the working image size -- the same plan the search itself
        // uses. A pyramid run and a flat run are different maps, so the level
        // count is part of the map's identity; a single-level run folds nothing
        // (see depth_map_fingerprint), so a project built before the pyramid
        // existed still matches it. The working resolution is what the search
        // sees, so the sensor size is divided by the same downscale the task
        // loads at.
        context.pyramid_levels = [&] {
            stereo::PatchMatchOptions pyramid_options;
            pyramid_options.pyramid_levels = kPyramidLevels;
            pyramid_options.pyramid_min_size = kPyramidMinSize;
            pyramid_options.iterations = kIterations;
            pyramid_options.window_radius = kWindowRadius;
            const std::int32_t sensor_id = camera.sensor_id;
            if (sensor_id < 0 ||
                sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
                return 1;
            }
            const Sensor& sensor = chunk->sensors()[static_cast<std::size_t>(sensor_id)];
            const int width = sensor.width / downscale;
            const int height = sensor.height / downscale;
            if (width <= 0 || height <= 0) {
                return 1;
            }
            return static_cast<int>(
                stereo::patch_match_iteration_plan(width, height, pyramid_options).size());
        }();
        // The cost changes the map as much as the geometry does: the same inputs
        // searched with a different cost find different planes.
        context.cost_type = static_cast<int>(cost_type);
        // And the cut changes which of those planes survive into the map, so the
        // EFFECTIVE threshold (the resolved default included) is folded too.
        context.max_cost = max_cost;
        const auto hood_it = hood_of.find(camera_id);
        if (hood_it != hood_of.end()) {
            const stereo::ViewNeighbourhood& hood = *hood_it->second;
            context.depth_min = hood.depth_min;
            context.depth_max = hood.depth_max;
            context.neighbour_ids = hood.neighbours;
            context.neighbour_poses.reserve(hood.neighbours.size());
            context.neighbour_calibrations.reserve(hood.neighbours.size());
            for (const std::int32_t id : hood.neighbours) {
                const auto nb = camera_by_id.find(id);
                if (nb == camera_by_id.end()) {
                    continue;
                }
                context.neighbour_poses.push_back(nb->second->pose);
                const std::int32_t sensor_id = nb->second->sensor_id;
                // A neighbour with no sensor is a broken chunk; the same
                // condition is caught in load(). Hash a default calibration so
                // the fingerprint stays a function of the model, not of a throw.
                context.neighbour_calibrations.push_back(
                    sensor_id >= 0 &&
                            sensor_id < static_cast<std::int32_t>(chunk->sensors().size())
                        ? chunk->sensors()[static_cast<std::size_t>(sensor_id)].calibration
                        : Calibration{});
            }
        }
        // The reference image, resolved the way load() will resolve it. Absent
        // frames are caught in load(); an unresolvable path hashes as empty.
        const auto frame_it = frame_of.find(camera_id);
        if (frame_it != frame_of.end()) {
            const std::filesystem::path path = resolve_frame_path(document, frame_it->second->path());
            context.image_path = path.string();
            // Size and mtime are a cheap, portable proxy for "the pixels are the
            // same file". Failing to stat is not an error: the map is still a
            // function of the path, and 0/0 hashes consistently.
            std::error_code ec;
            const auto size = std::filesystem::file_size(path, ec);
            if (!ec) {
                context.image_size = static_cast<std::uint64_t>(size);
            }
            ec.clear();
            const auto mtime = std::filesystem::last_write_time(path, ec);
            if (!ec) {
                context.image_mtime = static_cast<std::int64_t>(mtime.time_since_epoch().count());
            }
        }
        // Hole repair changes the final depths, so a repaired map is a different
        // map from an unrepaired one. Folded ONLY when repair_holes is on: a map
        // built before the feature existed hashed neither value, so it keeps
        // matching a repair-off run exactly and existing maps stay reusable.
        std::uint64_t fingerprint = depth_map_fingerprint(
            camera, chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)], downscale,
            &context);
        // The fold follows what the pass actually DOES: with the filter disabled
        // it never runs, so the map is the unrepaired one and must keep the
        // historic fingerprint (and stay reusable).
        if (repair_holes && filter_mode != stereo::FilterMode::Disabled) {
            // A string marker, not a bare 1: no unrepaired fingerprint folds a
            // string here, so enabling repair can never collide with a map built
            // while it was off.
            fingerprint = depth_map_fingerprint_fold_string(fingerprint, "repair_holes");
            fingerprint = depth_map_fingerprint_fold_int(fingerprint, repair_max_area);
        }
        return fingerprint;
    };

    // Existing maps by camera id, for reuse_depth. Read through a CONST view: the
    // non-const accessor would lazily create an empty set on a chunk with none.
    const Chunk& source = *chunk;
    std::unordered_map<std::int32_t, const DepthMap*> existing;
    if (reuse_depth) {
        for (const DepthMap& map : source.depth_maps()) {
            existing.emplace(map.camera_id, &map);
        }
    }
    // A map's SEARCH is reusable only when it was built at this downscale from
    // this geometry, and its stored depths are still the raw search output:
    // filter_mode -1 (never filtered, e.g. checkpointed by a cancelled run) or
    // exactly the mode this run asks for. A map filtered at another strength
    // holds post-filter depths, and the filter does not invert, so its search
    // is repeated rather than re-filtered into something it never was.
    //
    // Fingerprint 0 means "unknown" (a project saved before the field existed),
    // which never matches, so an old project's maps are rebuilt once and carry a
    // fingerprint from then on.
    const int requested_mode = static_cast<int>(filter_mode);
    const auto reusable = [&](const DepthMap& map) {
        return reuse_depth && map.downscale == downscale &&
               map.alignment_fingerprint != 0 &&
               map.alignment_fingerprint == fingerprint_of(map.camera_id) &&
               (map.filter_mode < 0 || map.filter_mode == requested_mode);
    };
    // "This map still describes the current chunk", independent of whether this
    // run IS reusing maps (`reuse_depth` may be off). A checkpoint may carry
    // forward a map it did not search this time, but never one the current
    // geometry has invalidated: a partial set written before a re-alignment must
    // not resurrect those maps when a later cancel merges into it. Also requires
    // the camera to still exist and still be in this run's neighbourhood.
    const auto still_current = [&](const DepthMap& map) {
        return map.downscale == downscale && map.alignment_fingerprint != 0 &&
               map.filter_mode < 0 &&
               camera_by_id.find(map.camera_id) != camera_by_id.end() &&
               hood_of.find(map.camera_id) != hood_of.end() &&
               map.alignment_fingerprint == fingerprint_of(map.camera_id);
    };
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

    // Maps finished so far, in neighbourhood order so the committed set is
    // deterministic.
    std::vector<DepthMap> built;
    built.reserve(neighbourhoods.size());
    std::vector<std::string> errors(neighbourhoods.size());
    std::atomic<bool> self_checked{false};

    // A set an EARLIER run left behind when it was cancelled: its maps are
    // per-camera complete but unfiltered. Resuming from it means the next
    // checkpoint must MERGE into it and a successful run must REPLACE it,
    // rather than piling up a new set per cancel. -1 means "this run started
    // from no partial set" and a cancel simply appends one.
    const std::int32_t resumed_partial_id = [&]() -> std::int32_t {
        const DepthMapsAsset* active =
            find_asset(chunk->depth_map_sets(), chunk->active_depth_maps_id());
        if (active == nullptr) {
            return -1;
        }
        const std::string* status = active->meta.find(kPartialStatusKey);
        return status != nullptr && *status == kPartialStatusValue ? active->id : -1;
    }();
    // The current partial set's id, updated as this run checkpoints: -1 until
    // this run writes one.
    std::int32_t partial_id = resumed_partial_id;

    // Checkpoint the maps finished so far into the chunk, then rethrow. Called
    // only on cancellation. A cancelled run that finished nothing leaves the
    // chunk untouched, so "cancel" on the first camera is still a no-op.
    //
    // WHAT A CANCEL COMMITS: depth maps, and nothing else. Every other product
    // (point clouds, models, DEMs, orthomosaics, earlier depth-map sets) is
    // left exactly as it was: a cancel is not a statement that those products
    // are stale, and destroying hours of unrelated work would make the Cancel
    // button dangerous. (A SUCCESSFUL run still drops the point clouds, because
    // it replaces the maps they were fused from; see the commit block below.)
    //
    // REPEATED CANCELS: the checkpoint merges into the set the run resumed
    // from, or into the set this run already wrote, and marks it "partial".
    // Without that, cancelling N times would leave N nearly-identical sets.
    const auto checkpoint_and_throw = [&](const std::string& message) {
        if (built.empty()) {
            throw CancelledError(message);
        }
        Chunk staged = std::move(*chunk);
        // Union of the previous partial set (if any) and this run's maps, keyed
        // by camera. This run's copies win: a reused map came from that very
        // set, and a fresh search is newer than anything parked there.
        //
        // Only previous maps that are still CURRENT are carried over: a partial
        // set written before a re-alignment must not be merged back into a set
        // produced after it, or the cancel would resurrect maps the whole
        // fingerprint exists to invalidate.
        std::unordered_map<std::int32_t, DepthMap> merged;
        const DepthMapsAsset* previous = find_asset(staged.depth_map_sets(), partial_id);
        if (previous != nullptr && previous->data != nullptr) {
            for (const DepthMap& map : *previous->data) {
                if (still_current(map)) {
                    merged.emplace(map.camera_id, map);
                }
            }
        }
        for (const DepthMap& map : built) {
            merged[map.camera_id] = map;
        }
        std::vector<DepthMap> combined;
        combined.reserve(merged.size());
        for (auto& entry : merged) {
            combined.push_back(std::move(entry.second));
        }
        std::sort(combined.begin(), combined.end(), [](const DepthMap& a, const DepthMap& b) {
            return a.camera_id < b.camera_id;
        });

        DepthMapsAsset* target = find_asset(staged.depth_map_sets(), partial_id);
        if (target == nullptr) {
            target = &staged.add_depth_maps(params.get_string("label"));
            partial_id = target->id;
        }
        const std::size_t kept = combined.size();
        const std::string label = target->label;
        const std::int32_t target_id = target->id;
        *target->data = std::move(combined);
        target->meta.set(kPartialStatusKey, kPartialStatusValue);
        staged.set_active_depth_maps_id(target->id);
        *chunk = std::move(staged);
        // The reuse lookup of a LATER run reads the active set only, so this
        // partial set is exactly what it should continue from; earlier sets
        // stay in the chunk for inspection, as they do after a completed run.
        log(LogLevel::Info) << "BuildDepthMaps: cancelled after " << built.size() << " of "
                            << neighbourhoods.size() << " cameras; kept " << kept
                            << " map(s) as asset " << target_id << " (" << label
                            << "), unfiltered. Re-run with reuse_depth to continue.";
        throw CancelledError(message);
    };

    // Progress within one camera: the search reports 0..1 of ITSELF per sweep;
    // scale that into the camera's slice of the build stage and poll the cancel
    // flag on every tick, so Cancel takes effect inside one sweep rather than
    // after the current camera (which can be hours at downscale 2).
    //
    // The final 1.0 report is NOT a cancellation point even when the user has
    // already asked to stop: the search has finished at that moment, so throwing
    // here would discard a completed camera's map and send the caller into the
    // checkpoint with nothing new to keep. The poll right after the camera is
    // stored catches the same request one instant later and keeps the map.
    const double camera_count = static_cast<double>(neighbourhoods.size());
    std::size_t current_index = 0;
    const auto sweep_progress = [&](double fraction) {
        const double clamped = std::clamp(fraction, 0.0, 1.0);
        progress.update("buildDepthMaps",
                        (static_cast<double>(current_index) + clamped) / camera_count);
        if (clamped < 1.0 && progress.is_cancelled()) {
            throw CancelledError("BuildDepthMaps was cancelled");
        }
    };

    progress.update("buildDepthMaps", 0.0);
    std::size_t reused_count = 0;
    for (std::size_t i = 0; i < neighbourhoods.size(); ++i) {
        current_index = i;
        const stereo::ViewNeighbourhood& hood = neighbourhoods[i];
        try {
            // Reuse an existing map built at this downscale from this geometry.
            // Its search is skipped; filtering still runs unless it was already
            // filtered with the requested mode, so a map checkpointed by an
            // earlier cancelled run gets the cross-view pass it never had.
            if (reuse_depth) {
                const auto kept = existing.find(hood.camera_id);
                if (kept != existing.end() && reusable(*kept->second)) {
                    ++reused_count;
                    built.push_back(*kept->second);
                    progress.update("buildDepthMaps",
                                    (static_cast<double>(i) + 1.0) / camera_count);
                    if (progress.is_cancelled()) {
                        // A reused map is still progress worth keeping.
                        checkpoint_and_throw("BuildDepthMaps was cancelled");
                    }
                    continue;
                }
                // Say WHY a stored map is being rebuilt. After a defaults change
                // (or a re-alignment) every map is rebuilt, which looks like the
                // reuse setting did nothing unless the cause is stated.
                if (kept != existing.end() && Log::enabled(LogLevel::Info)) {
                    const DepthMap& old = *kept->second;
                    const char* reason =
                        old.downscale != downscale
                            ? "built at a different downscale"
                            : (old.alignment_fingerprint == 0
                                   ? "built without a fingerprint"
                                   : (old.alignment_fingerprint != fingerprint_of(hood.camera_id)
                                          ? "built from different settings, alignment or "
                                            "neighbourhood"
                                          : "filtered with a different mode"));
                    log(LogLevel::Info) << "BuildDepthMaps: camera " << hood.camera_id
                                        << ": rebuilding depth map (" << reason << ")";
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
            options.pyramid_levels = kPyramidLevels;
            options.pyramid_min_size = kPyramidMinSize;
            options.cost_type = cost_type;
            options.max_cost = max_cost;
            options.on_sweep = sweep_progress;

            log_device_once();

            const auto started = std::chrono::steady_clock::now();
            // GPU when available and enabled, CPU otherwise or after a device
            // failure (with_fallback logs and counts it once). The device is
            // recorded before the run, because a fallback mid-run is the only
            // case where the answer differs from patch_match_gpu_would_run.
            const bool gpu_path =
                stereo::patch_match_gpu_would_run(reference, neighbours.views, options);
            g_search_count.fetch_add(1);
            const stereo::PatchMatchResult result =
                stereo::run_patch_match_device(reference, neighbours.views, options);
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

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
            map.alignment_fingerprint = fingerprint_of(hood.camera_id);
            // A PatchMatch result is not a final map: the cross-view filter is
            // still to run, so no mode is recorded yet.
            map.filter_mode = -1;
            if (Log::enabled(LogLevel::Info)) {
                log(LogLevel::Info) << "depth map " << (i + 1) << "/" << neighbourhoods.size()
                                    << ": camera " << hood.camera_id << ", " << map.width << "x"
                                    << map.height << ", " << (gpu_path ? "GPU" : "CPU") << ", "
                                    << seconds << " s";
            }
            built.push_back(std::move(map));
        } catch (const CancelledError&) {
            // Keep what finished: those maps are valid and labelled with their
            // downscale and fingerprint, so a re-run skips them.
            checkpoint_and_throw("BuildDepthMaps was cancelled");
        } catch (const std::exception& e) {
            errors[i] = "camera " + std::to_string(hood.camera_id) + ": " + e.what();
        }
        progress.update("buildDepthMaps", (static_cast<double>(i) + 1.0) / camera_count);
        if (progress.is_cancelled()) {
            checkpoint_and_throw("BuildDepthMaps was cancelled");
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
    const std::size_t staged_count = built.size();

    progress.update("filterDepthMaps", 0.0);
    for (std::size_t i = 0; i < built.size(); ++i) {
        const DepthMap& map = built[i];
        // A reused map already filtered with the mode this run asked for is
        // final: running the pass again is not idempotent (it erodes support at
        // the boundaries). Only a map checkpointed unfiltered (mode -1) reaches
        // this loop otherwise, and it is filtered now.
        if (map.filter_mode == requested_mode) {
            filtered.push_back(map);
            progress.update("filterDepthMaps",
                            (static_cast<double>(i) + 1.0) /
                                static_cast<double>(staged_count));
            continue;
        }
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
        // Post-process on the FILTERED map, in both the CPU and GPU depth paths:
        // both converge on `built` before this loop, so there is one repair site.
        //
        // Never repair with the filter disabled: the pass exists to undo holes
        // the cross-view filter opened inside a surface it had already judged
        // consistent. With no filter those holes are genuine no-data (the search
        // found nothing there), and synthesising depth for them would invent
        // surface rather than restore it.
        if (repair_holes && filter_mode != stereo::FilterMode::Disabled) {
            stereo::RepairOptions repair_options;
            repair_options.max_area = static_cast<int>(repair_max_area);
            stereo::RepairStats repair_stats;
            result = stereo::repair_depth_map(result, reference.calibration, repair_options,
                                              &repair_stats);
            if (Log::enabled(LogLevel::Info) && repair_stats.components > 0) {
                log(LogLevel::Info) << "BuildDepthMaps: camera " << map.camera_id << ": repaired "
                                    << repair_stats.filled_pixels << " px in "
                                    << repair_stats.filled_components << " of "
                                    << repair_stats.components << " holes ("
                                    << repair_stats.rejected_area << " too large, "
                                    << repair_stats.rejected_boundary << " open, "
                                    << repair_stats.rejected_plane << " no plane)";
            }
        }
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
                        static_cast<double>(i + 1) / static_cast<double>(staged_count));
        if (progress.is_cancelled()) {
            // The searches are all done, which is the expensive part, so keep
            // them: `built` is the unfiltered set at this point, and the next
            // run filters it. (Filtering cannot be resumed in place: the pass
            // reads neighbour maps, so replacing one mid-loop would make later
            // maps filter against filtered neighbours.)
            checkpoint_and_throw("BuildDepthMaps was cancelled");
        }
    }
    progress.update("filterDepthMaps", 1.0);

    // -- commit atomically ---------------------------------------------------
    progress.update("finalize", 0.0);
    Chunk staged = std::move(*chunk);
    std::string written_label;
    DepthMapsAsset* set = find_asset(staged.depth_map_sets(), partial_id);
    if (set != nullptr) {
        // This run RESUMED from a partial set a cancelled run left behind. A
        // successful run completes that set rather than adding a second one
        // beside it: the partial set and its replacement describe the same
        // cameras, and keeping both would leave the tree with two overlapping
        // sets after a few cancels. The maps are replaced wholesale (the run
        // holds every camera the partial set had, now filtered), and the
        // partial marker is cleared so the set is no longer a resume target.
        set->data->clear();
        *set->data = std::move(filtered);
        set->meta.items.erase(
            std::remove_if(set->meta.items.begin(), set->meta.items.end(),
                           [](const auto& item) { return item.first == kPartialStatusKey; }),
            set->meta.items.end());
        staged.set_active_depth_maps_id(set->id);
    } else {
        // No partial set to complete: append a NEW depth-map set; earlier sets
        // are kept.
        DepthMapsAsset& added = staged.add_depth_maps(params.get_string("label"));
        *added.data = std::move(filtered);
        set = &added;
    }
    written_label = set->label;
    // Every point cloud is derived from depth maps, so the ones already in the
    // chunk no longer describe the active maps. Drop them rather than leave a
    // cloud that looks current but was fused from different maps. (This is
    // SUCCESS-only: a cancelled run keeps them, see checkpoint_and_throw.)
    staged.point_clouds().clear();
    staged.set_active_point_cloud_id(-1);
    *chunk = std::move(staged);
    progress.update("finalize", 1.0);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "BuildDepthMaps: " << chunk->depth_maps().size() << " of "
                            << chunk->cameras().size() << " cameras at downscale " << downscale
                            << ", filter " << stereo::to_string(filter_mode) << ", "
                            << max_neighbors << " neighbours, " << reused_count << " reused"
                            << " -> asset " << chunk->active_depth_maps_id() << " '"
                            << written_label << "'";
    }
}

}  // namespace alphashape
