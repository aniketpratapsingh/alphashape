#include "common/algorithm/stereo/patch_match_gpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "base/platform/env.hpp"
#include "common/algorithm/stereo/depth_pyramid.hpp"
#include "gpu/gpu.hpp"
#include "image/image.hpp"
#include "model/camera/calibration.hpp"

// GPU PatchMatch: the same estimate as run_patch_match(), on the gpu:: layer.
//
// Scheme. Red-black checkerboard propagation with random refinement (Galliani,
// Lasinger & Schindler 2015). Two kernels drive the search:
//   pm_init    initialise one colour from a fresh random plane and record the
//              texture mask in an active flag buffer;
//   pm_refine  one descent half-sweep of one colour: take the incumbent,
//              propagate the 4 adjacent + 4 distance-3 neighbours' planes
//              (always the opposite colour, so the half-sweep is a fixed point),
//              draw a fresh random plane and a perturbed incumbent.
// pm_finalize writes depth/normal/cost and applies the max_cost cut. All of a
// sweep's dispatches go into one gpu::run, which barriers between them.
//
// Precision. float32 throughout (M1 has no fp64): geometry in the reference
// camera frame, ZNCC sums in float. The acceptance rule, window and visibility
// quorum are the CPU's, so the result matches in quality and near-agreement,
// not bit for bit.
//
// Randomness. A PCG hash keyed by (seed, pixel, iteration, stream) plus a draw
// counter, so every draw is a pure function of its coordinates: deterministic
// run to run and independent of scheduling.
//
// Distortion. The forward model of calibration.hpp is reimplemented in GLSL
// (pm_math.glsl, project_point) with the same coefficients, the same Frame /
// Fisheye split and the same undefined cases; there is no undistortion pass, so
// the sampling geometry is the CPU's for every sensor.

namespace alphashape::stereo {
namespace {

namespace g = alphashape::gpu;

constexpr int kMaxSamples = 49;       // matches PM_MAX_SAMPLES in pm_math.glsl
constexpr int kMaxNeighbours = 32;    // matches PM_MAX_NEIGHBOURS
constexpr int kTileX = 8;
constexpr int kTileY = 16;
constexpr int kViewStride = 26;       // fcx(3) b(2) radial(4) tang(4) M(9) c(3) pad(1)
constexpr int kViewIntStride = 4;     // width, height, type, image offset
constexpr std::uint32_t kNoIteration = 0xFFFFFFFFU;

/// PatchMatch keeps its working set on host-visible memory. Every pixel is
/// seeded from the CPU and the depth/normal/cost fields are read back every
/// call, so on a discrete GPU the staging hops would run for the whole grid in
/// both directions; the kernel itself is the same either way. This is the
/// pre-decided fallback from the M10 plan.
/// The memory kind PatchMatch uses for its working set.
g::MemoryKind working_memory() {
    return patch_match_gpu_prefers_host_memory() ? g::MemoryKind::HostVisible
                                                 : g::MemoryKind::DeviceLocal;
}

std::atomic<bool> g_force_failure{false};

/// Push-constant block, byte-for-byte PmParams in pm_math.glsl.
struct PmParams {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t colour;
    std::uint32_t iteration;
    std::uint32_t iteration_hi;
    std::uint32_t n_neighbours;
    std::uint32_t n_samples;
    std::uint32_t best_k;
    float depth_min;
    float depth_max;
    float inv_span;
    float normal_angle;
    float depth_delta;
    float min_texture;
    float max_cost;
    std::int32_t radius;
    std::uint32_t seed_lo;
    std::uint32_t seed_hi;
    std::uint32_t touch;
    std::uint32_t row_offset;
    /// 0 = seed each pixel from a fresh random plane (the coarsest level's
    /// initialisation, and the pre-pyramid behaviour); 1 = seed from the
    /// inherited (depth, normal) already in the state buffers, which is how a
    /// finer pyramid level starts from the upsampled coarse map.
    std::uint32_t seed_mode;
    /// Q2 adaptive min+margin width on the selected cost's interval; 0 = the
    /// plain mean of the best k (bit-identical to the pre-Q2 kernel).
    float cost_margin;
};

static_assert(sizeof(PmParams) == 88, "PmParams must match the GLSL push-constant size");
static_assert(offsetof(PmParams, depth_min) == 32, "PmParams layout drifted");
static_assert(offsetof(PmParams, radius) == 60, "PmParams layout drifted");
static_assert(offsetof(PmParams, seed_lo) == 64, "PmParams layout drifted");
static_assert(offsetof(PmParams, touch) == 72, "PmParams layout drifted");
static_assert(offsetof(PmParams, row_offset) == 76, "PmParams layout drifted");
static_assert(offsetof(PmParams, seed_mode) == 80, "PmParams layout drifted");
static_assert(offsetof(PmParams, cost_margin) == 84, "PmParams layout drifted");

/// Largest pixel count one half-sweep dispatch should cover. A pm_init or
/// pm_refine over a whole large image is a single multi-second dispatch, which
/// on a GPU shared with the display starves the compositor and can hang the
/// machine (#28); the per-submission batcher cannot help, because one half-sweep
/// must see the state the previous one wrote. Whole tile rows are therefore
/// issued as separate dispatches in the same submission.
///
/// Bit-identical by construction: pm_init and pm_refine write only pixels of
/// their own checkerboard colour and read only the opposite colour (all four
/// adjacent and all four distance-3 offsets flip the parity), so a band's
/// sources are never written by any band of the same half-sweep. The tile grid
/// is shifted by whole tiles, so (origin.x + origin.y) & 1 -- which colour a lane
/// owns -- is unchanged, and every barrier a split adds is the same
/// compute->compute dependency the unsplit command buffer already has.
///
/// ALPHASHAPE_GPU_MAX_BAND_PIXELS pins the target (tests use a tiny value to
/// force many bands and compare against the unsplit path); an empty or
/// non-numeric value keeps the default.
constexpr std::size_t kDefaultBandPixels = 250'000;

std::size_t band_pixel_target() {
    const std::string text = platform::get_env("ALPHASHAPE_GPU_MAX_BAND_PIXELS");
    if (!text.empty() && text.find_first_not_of("0123456789") == std::string::npos) {
        const unsigned long long value = std::strtoull(text.c_str(), nullptr, 10);
        if (value > 0) {
            return static_cast<std::size_t>(value);
        }
    }
    return kDefaultBandPixels;
}

/// Window offsets, identical to the CPU's construction (multiples of
/// window_step in [-r, r] including both endpoints and the centre).
std::vector<std::int32_t> window_offsets(const PatchMatchOptions& options, int* radius_out) {
    const int r = options.window_radius;
    std::vector<int> axis;
    for (int v = -r; v <= r; v += options.window_step) {
        axis.push_back(v);
    }
    if (axis.empty() || axis.back() != r) {
        axis.push_back(r);
    }
    if (std::find(axis.begin(), axis.end(), 0) == axis.end()) {
        axis.push_back(0);
        std::sort(axis.begin(), axis.end());
    }
    std::vector<std::int32_t> flat;
    flat.reserve(axis.size() * axis.size() * 2);
    for (int dy : axis) {
        for (int dx : axis) {
            flat.push_back(dx);
            flat.push_back(dy);
        }
    }
    *radius_out = r;
    return flat;
}

/// Validates the inputs exactly as run_patch_match() does, so the GPU path
/// rejects the same bad calls before touching a device.
void validate_inputs(const PMView& reference, const std::vector<PMView>& neighbours,
                     const PatchMatchOptions& options) {
    if (reference.image == nullptr) {
        throw ValueError("stereo::run_patch_match: reference image is null");
    }
    if (reference.image->width() != reference.calibration.width ||
        reference.image->height() != reference.calibration.height) {
        throw ValueError("stereo::run_patch_match: reference calibration does not match image size");
    }
    if (neighbours.empty()) {
        throw ValueError("stereo::run_patch_match: no neighbour views");
    }
    for (std::size_t i = 0; i < neighbours.size(); ++i) {
        if (neighbours[i].image == nullptr) {
            throw ValueError("stereo::run_patch_match: neighbour image is null");
        }
        if (neighbours[i].image->width() != neighbours[i].calibration.width ||
            neighbours[i].image->height() != neighbours[i].calibration.height) {
            throw ValueError(
                "stereo::run_patch_match: neighbour calibration does not match image size");
        }
    }
    if (!(options.depth_min > 0.0) || !(options.depth_max > options.depth_min)) {
        throw ValueError("stereo::run_patch_match: depth range must satisfy 0 < min < max");
    }
    if (options.window_radius < 0) {
        throw ValueError("stereo::run_patch_match: window_radius must be non-negative");
    }
    if (options.window_step < 1) {
        throw ValueError("stereo::run_patch_match: window_step must be at least 1");
    }
    if (options.iterations < 0) {
        throw ValueError("stereo::run_patch_match: iterations must be non-negative");
    }
    if (options.best_k < 0) {
        throw ValueError("stereo::run_patch_match: best_k must be non-negative");
    }
    if (!(options.cost_margin >= 0.0 && options.cost_margin <= 2.0)) {
        throw ValueError("stereo::run_patch_match: cost_margin must be in [0, 2]");
    }
    if (options.pyramid_levels < 0) {
        throw ValueError("stereo::run_patch_match: pyramid_levels must be non-negative");
    }
    if (options.pyramid_levels != 1 && options.pyramid_min_size <= 0) {
        throw ValueError("stereo::run_patch_match: pyramid_min_size must be positive");
    }
    if (options.cost_type == CostType::Census) {
        if (options.census.window_radius < 1) {
            throw ValueError("stereo::run_patch_match: census window_radius must be positive");
        }
        if (options.census.wide && options.census.window_radius > kCensusMaxRadius) {
            throw ValueError(
                "stereo::run_patch_match: a wide census needs census window_radius <= 4 "
                "(four arms of window_radius comparisons, 16 bits)");
        }
        if (!options.census.wide && options.census.window_radius != 1) {
            throw ValueError(
                "stereo::run_patch_match: the 3x3 census requires census window_radius == 1");
        }
        if (!options.census.transform_neighbours) {
            throw ValueError(
                "stereo::run_patch_match: the census cost requires census.transform_neighbours");
        }
    }
}

/// True when one image and its neighbour list fit the kernel limits.
///
/// The census cost is declined here rather than implemented: the descriptors are
/// another whole-image buffer per view, which would take the kernel from 14
/// storage-buffer bindings to 16 and put pm_init/pm_refine over the binding
/// budget the Vulkan layer guarantees (the spec's own risk note leaves the
/// decision to WP-17.15). Recomputing the descriptor inline instead would mean
/// 16 extra image loads per window sample per neighbour -- a multiple of the
/// cost core's work, on shaders that must stay at -O0 to agree with the CPU.
/// Both are worse than running census on the CPU, so the GPU declines and the
/// caller falls back with the log below (a decline is not a device failure:
/// nothing is dispatched and gpu::failure_count() is not advanced).
bool fits_kernel_limits(const PMView& reference, const std::vector<PMView>& neighbours,
                        const PatchMatchOptions& options) {
    if (options.cost_type == CostType::Census) {
        return false;
    }
    if (reference.image == nullptr || neighbours.empty()) {
        return false;
    }
    const std::size_t axis_count =
        static_cast<std::size_t>((2 * options.window_radius) / std::max(1, options.window_step) + 2);
    if (axis_count * axis_count > static_cast<std::size_t>(kMaxSamples)) {
        return false;
    }
    if (neighbours.size() > static_cast<std::size_t>(kMaxNeighbours)) {
        return false;
    }
    // A kernel run needs at least one active pixel per colour on an image at
    // least two window radii across.
    return reference.image->width() > 2 * options.window_radius &&
           reference.image->height() > 2 * options.window_radius;
}

/// One level of the GPU search: everything the dispatches need, with the
/// reference and neighbours already reduced to this level.
struct GpuLevel {
    const PMView* reference = nullptr;
    const std::vector<PMView>* neighbours = nullptr;
    std::uint64_t seed = 1;
    int iterations = 0;
    /// The plane inherited from the level beneath, at this level's resolution.
    /// Null for the coarsest level, which seeds from fresh random planes.
    const DepthHypotheses* warm = nullptr;
    /// The inverse-depth radius the sweep perturbations start from: the
    /// configured span at the coarsest level, the local variation of the
    /// inherited map above it.
    float span = 0.0F;
    /// Where this level sits in the whole run's progress: the fraction already
    /// complete when it starts, and the share of the total its sweeps take.
    double progress_base = 0.0;
    double progress_scale = 1.0;
};

/// The GPU run of one level. Throws gpu::Error on any device failure; the
/// caller wraps the whole pyramid in with_fallback.
PatchMatchResult run_gpu_level(const std::shared_ptr<g::Context>& context, const GpuLevel& level,
                               const PatchMatchOptions& options, double& reported) {
    const PMView& reference = *level.reference;
    const std::vector<PMView>& neighbours = *level.neighbours;

    const FloatImage& ref_image = *reference.image;
    const int width = ref_image.width();
    const int height = ref_image.height();
    const std::size_t count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    int radius = 0;
    const std::vector<std::int32_t> offsets = window_offsets(options, &radius);
    const int n_samples = static_cast<int>(offsets.size() / 2);
    const int n_neighbours = static_cast<int>(neighbours.size());

    // -- reference data ------------------------------------------------------
    // Bearings (centre of pixel (i, j) at (i + 0.5, j + 0.5)) and the reference
    // window values, exactly the CPU's arrays.
    std::vector<float> bearings(count * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t p = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                  static_cast<std::size_t>(x);
            const Eigen::Vector3d b = reference.calibration.unproject(
                Eigen::Vector2d(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5));
            bearings[p * 3 + 0] = static_cast<float>(b.x());
            bearings[p * 3 + 1] = static_cast<float>(b.y());
            bearings[p * 3 + 2] = static_cast<float>(b.z());
        }
    }

    std::vector<float> ref_windows(count * static_cast<std::size_t>(n_samples), 0.0F);
    {
        const std::vector<float>& pixels = ref_image.pixels();
        const int interior_x0 = radius;
        const int interior_x1 = width - 1 - radius;
        const int interior_y0 = radius;
        const int interior_y1 = height - 1 - radius;
        for (int y = interior_y0; y <= interior_y1; ++y) {
            for (int x = interior_x0; x <= interior_x1; ++x) {
                const std::size_t q = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                      static_cast<std::size_t>(x);
                float* const dst = ref_windows.data() + q * static_cast<std::size_t>(n_samples);
                for (int k = 0; k < n_samples; ++k) {
                    const int dx = offsets[static_cast<std::size_t>(2 * k)];
                    const int dy = offsets[static_cast<std::size_t>(2 * k + 1)];
                    dst[static_cast<std::size_t>(k)] =
                        pixels[static_cast<std::size_t>(y + dy) * static_cast<std::size_t>(width) +
                               static_cast<std::size_t>(x + dx)];
                }
            }
        }
    }

    // -- neighbour views -----------------------------------------------------
    // Every image is concatenated into one float buffer; a view records its
    // offset. The transform is X_n = M X_ref + c with M = R_n R_ref^T.
    std::vector<float> view_params(static_cast<std::size_t>(n_neighbours) * kViewStride, 0.0F);
    std::vector<std::uint32_t> view_ints(static_cast<std::size_t>(n_neighbours) * kViewIntStride, 0);
    std::vector<float> neighbour_images;
    std::vector<float> image_scratch;
    const Eigen::Matrix3d rref_t = reference.R.transpose();
    for (int j = 0; j < n_neighbours; ++j) {
        const PMView& view = neighbours[static_cast<std::size_t>(j)];
        const Calibration& c = view.calibration;
        const std::size_t base = static_cast<std::size_t>(j) * kViewStride;
        view_params[base + 0] = static_cast<float>(c.f);
        view_params[base + 1] = static_cast<float>(c.cx);
        view_params[base + 2] = static_cast<float>(c.cy);
        view_params[base + 3] = static_cast<float>(c.b1);
        view_params[base + 4] = static_cast<float>(c.b2);
        view_params[base + 5] = static_cast<float>(c.k1);
        view_params[base + 6] = static_cast<float>(c.k2);
        view_params[base + 7] = static_cast<float>(c.k3);
        view_params[base + 8] = static_cast<float>(c.k4);
        view_params[base + 9] = static_cast<float>(c.p1);
        view_params[base + 10] = static_cast<float>(c.p2);
        view_params[base + 11] = static_cast<float>(c.p3);
        view_params[base + 12] = static_cast<float>(c.p4);
        const Eigen::Matrix3d m = view.R * rref_t;
        for (int r = 0; r < 3; ++r) {
            for (int col = 0; col < 3; ++col) {
                view_params[base + 13 + static_cast<std::size_t>(r * 3 + col)] =
                    static_cast<float>(m(r, col));
            }
        }
        const Eigen::Vector3d centre = view.t - m * reference.t;
        view_params[base + 22] = static_cast<float>(centre.x());
        view_params[base + 23] = static_cast<float>(centre.y());
        view_params[base + 24] = static_cast<float>(centre.z());

        const std::size_t vbase = static_cast<std::size_t>(j) * kViewIntStride;
        view_ints[vbase + 0] = static_cast<std::uint32_t>(c.width);
        view_ints[vbase + 1] = static_cast<std::uint32_t>(c.height);
        view_ints[vbase + 2] = c.type == CameraType::Fisheye ? 1U : 0U;
        view_ints[vbase + 3] = static_cast<std::uint32_t>(neighbour_images.size());
        neighbour_images.insert(neighbour_images.end(), view.image->pixels().begin(),
                                view.image->pixels().end());
    }

    // -- device buffers ------------------------------------------------------
    // All 14 bindings are declared in every kernel (keep_bindings_alive), so
    // one buffer list serves every dispatch. The state buffers start at the
    // CPU's initial values (depth -1, normal zero, cost 2.0, inactive): a pixel
    // the texture mask rejects is never written and must read as "no depth".
    // The read-only inputs and the three outputs follow `kind`; see
    // working_memory() for why that is host-visible by default.
    //
    // A warm-started level starts from the inherited map instead of the empty
    // initial state: pm_init reads it back (seed_mode 1) and evaluates each
    // inherited plane at this level's geometry. Pixels the coarse level left as
    // holes keep depth -1 and fall through to the random draw, which is how the
    // fine level fills a hole the coarse one could not.
    const g::MemoryKind kind = working_memory();
    g::Buffer g_ref = g::Buffer::of(*context, ref_image.pixels(), kind);
    g::Buffer g_nb = g::Buffer::of(*context, neighbour_images, kind);
    g::Buffer g_views = g::Buffer::of(*context, view_params, kind);
    g::Buffer g_view_ints = g::Buffer::of(*context, view_ints, kind);
    g::Buffer g_offsets = g::Buffer::of(*context, offsets, kind);
    std::vector<float> initial_depth(count, -1.0F);
    std::vector<float> initial_normal(count * 3, 0.0F);
    if (level.warm != nullptr) {
        initial_depth = level.warm->depth;
        initial_normal = level.warm->normals;
        initial_depth.resize(count, -1.0F);
        initial_normal.resize(count * 3, 0.0F);
    }
    const std::vector<float> initial_cost(count, 2.0F);
    const std::vector<std::uint32_t> initial_active(count, 0U);
    g::Buffer g_depth = g::Buffer::of(*context, initial_depth, kind);
    g::Buffer g_normal = g::Buffer::of(*context, initial_normal, kind);
    g::Buffer g_cost = g::Buffer::of(*context, initial_cost, kind);
    g::Buffer g_active = g::Buffer::of(*context, initial_active, kind);
    g::Buffer g_ref_windows = g::Buffer::of(*context, ref_windows, kind);
    g::Buffer g_bearings = g::Buffer::of(*context, bearings, kind);
    g::Buffer g_out_depth = g::make_buffer(*context, count * sizeof(float), kind);
    g::Buffer g_out_normals = g::make_buffer(*context, count * 3 * sizeof(float), kind);
    g::Buffer g_out_cost = g::make_buffer(*context, count * sizeof(float), kind);

    std::vector<const g::Buffer*> buffers = {
        &g_ref,   &g_nb,       &g_views,      &g_view_ints,   &g_offsets,
        &g_depth, &g_normal,   &g_cost,       &g_active,      &g_ref_windows,
        &g_bearings, &g_out_depth, &g_out_normals, &g_out_cost};

    const int best_k = std::clamp(
        options.best_k == 0 ? (n_neighbours + 1) / 2 : options.best_k, 1, n_neighbours);
    const float depth_min = static_cast<float>(options.depth_min);
    const float depth_max = static_cast<float>(options.depth_max);
    const float inv_span = 1.0F / depth_min - 1.0F / depth_max;
    // The per-iteration perturbation radius. The coarsest level uses the
    // configured inverse-depth span (the pre-pyramid behaviour); a warm-started
    // level uses the local variation of its inherited map, which the CPU does
    // with pyramid_search_span -- the host computes it once per level so the two
    // paths search from the same radius.
    const float span = level.warm != nullptr ? level.span : inv_span;

    PmParams params{};
    params.width = static_cast<std::uint32_t>(width);
    params.height = static_cast<std::uint32_t>(height);
    params.n_neighbours = static_cast<std::uint32_t>(n_neighbours);
    params.n_samples = static_cast<std::uint32_t>(n_samples);
    params.best_k = static_cast<std::uint32_t>(best_k);
    params.cost_margin = static_cast<float>(options.cost_margin);
    params.seed_lo = static_cast<std::uint32_t>(level.seed & 0xFFFFFFFFULL);
    params.seed_hi = static_cast<std::uint32_t>(level.seed >> 32);
    params.depth_min = depth_min;
    params.depth_max = depth_max;
    params.inv_span = inv_span;
    params.min_texture = static_cast<float>(options.min_texture_stddev);
    params.max_cost = static_cast<float>(options.max_cost);
    params.radius = radius;
    params.touch = 0;
    params.seed_mode = level.warm != nullptr ? 1U : 0U;

    const std::array<std::uint32_t, 3> groups{
        static_cast<std::uint32_t>((width + kTileX - 1) / kTileX),
        static_cast<std::uint32_t>((height + kTileY - 1) / kTileY), 1};

    // One half-sweep is one dispatch per row band: a whole large image in one
    // dispatch is a multi-second job that starves the display GPU (#28), and the
    // per-submission batcher cannot split it because a half-sweep has to see
    // what the previous one wrote. Bands are bit-identical (see
    // band_pixel_target): a band is a shift of the tile grid by whole tiles, and
    // every propagation source is the opposite colour, so no band reads what
    // another band of the same half-sweep writes.
    const std::size_t band_pixels = band_pixel_target();
    const std::uint32_t rows_per_band = [&] {
        // Whole tiles, so the colour a lane owns never changes.
        const std::size_t rows = std::max<std::size_t>(band_pixels / static_cast<std::size_t>(width), 1);
        const std::size_t tiles = std::max<std::size_t>(rows / kTileY, 1);
        return static_cast<std::uint32_t>(tiles * kTileY);
    }();

    // Initialise both colours (the CPU's initial random hypothesis, plus the
    // texture mask), then the red-black sweeps.
    std::vector<g::Dispatch> dispatches;
    // The search fraction after each dispatch has finished. The hook reports
    // the same schedule as the CPU's SweepProgress: the init pair is not
    // reported (0), each refinement half-sweep adds 1/(2 * iterations), and the
    // finalise dispatch reports 1.
    std::vector<double> dispatch_fraction;
    const std::size_t bands =
        (static_cast<std::size_t>(height) + rows_per_band - 1) / rows_per_band;
    dispatches.reserve((static_cast<std::size_t>(2 + 2 * level.iterations)) * bands + 1);
    dispatch_fraction.reserve(dispatches.capacity());
    const auto make_dispatch = [&](const char* kernel) {
        g::Dispatch dispatch;
        dispatch.kernel = kernel;
        dispatch.buffers = buffers;
        dispatch.push_constants = g::push(params);
        dispatch.groups = groups;
        return dispatch;
    };
    // A half-sweep: one dispatch per band of rows, in ascending order. Only the
    // last band's completion advances the reported fraction, so a banded image
    // reports the same milestones as an unbanded one.
    const auto push_half_sweep = [&](const char* kernel, double fraction) {
        if (bands <= 1) {
            params.row_offset = 0;
            dispatches.push_back(make_dispatch(kernel));
        } else {
            for (std::size_t band = 0; band < bands; ++band) {
                const std::size_t row = band * rows_per_band;
                params.row_offset = static_cast<std::uint32_t>(row);
                g::Dispatch dispatch = make_dispatch(kernel);
                dispatch.groups[1] = static_cast<std::uint32_t>(
                    (std::min(static_cast<std::size_t>(height), row + rows_per_band) - row +
                     kTileY - 1) /
                    kTileY);
                dispatches.push_back(std::move(dispatch));
            }
        }
        dispatch_fraction.resize(dispatches.size(), fraction);
    };

    for (std::uint32_t colour = 0; colour < 2; ++colour) {
        params.colour = colour;
        params.iteration = kNoIteration;
        params.iteration_hi = kNoIteration;
        push_half_sweep("pm_init", 0.0);
    }
    params.depth_delta = span * 0.5F;
    params.normal_angle = static_cast<float>(1.0471975511965976 * 0.5);

    const double sweeps_total = static_cast<double>(std::max(2 * level.iterations, 1));
    for (int it = 0; it < level.iterations; ++it) {
        const float s = static_cast<float>(std::pow(0.5, static_cast<double>(it) + 1.0));
        params.iteration = static_cast<std::uint32_t>(it);
        params.iteration_hi = 0U;
        params.depth_delta = span * s;
        params.normal_angle = static_cast<float>(1.0471975511965976 * static_cast<double>(s));
        for (std::uint32_t colour = 0; colour < 2; ++colour) {
            params.colour = colour;
            const double done = static_cast<double>(2 * it + static_cast<int>(colour) + 1);
            // The fraction of the WHOLE run, so a pyramid's progress is monotone
            // across level boundaries just as the CPU's is.
            const double fraction = level.progress_base + level.progress_scale * done / sweeps_total;
            push_half_sweep("pm_refine", fraction);
        }
    }

    // Finalise with the grouped 1D layout (256 wide). It indexes by pixel, not
    // by lane, so it is never banded; row_offset stays 0.
    params.max_cost = static_cast<float>(options.max_cost);
    params.row_offset = 0;
    g::Dispatch finalize;
    finalize.kernel = "pm_finalize";
    finalize.buffers = buffers;
    finalize.push_constants = g::push(params);
    finalize.groups = {static_cast<std::uint32_t>((count + 255) / 256), 1, 1};
    dispatches.push_back(finalize);
    // The finalise dispatch closes this level, which on the last level is the
    // end of the whole run.
    dispatch_fraction.resize(dispatches.size(), level.progress_base + level.progress_scale);

    // Submit in bounded chunks rather than one command buffer for the whole
    // solve. A pixel-heavy reference (480x360, 7 neighbours) is ~15 s of GPU
    // time on the weak AMD test box, which puts a single all-sweeps submission
    // past amdgpu's 10 s job timeout and cancels the context (device lost ->
    // CPU fallback). On a GPU shared with the display (Apple silicon) a job
    // that long starves the compositor and can hang the machine. One dispatch
    // is now one step -- a quarter-sweep row band on a large image, a whole
    // half-sweep otherwise -- because a band writes pixels no other band of the
    // same half-sweep reads, so it is independently schedulable. The first
    // submission carries two (the two pm_init dispatches, exactly the old
    // behaviour when nothing is banded) and the limiter then scales towards the
    // device's target duration. The barriers a split adds are the same ones
    // gpu::run already inserts between dispatches, and neither a band split nor
    // a submission split changes any pixel's inputs, so the result is identical.
    g::BatchLimiter limiter(/*dispatches_per_step=*/1, /*initial_steps=*/2);
    // The caller's per-sweep hook, driven by the batcher: after each submission
    // `done` dispatches have run, and dispatch_fraction says what search
    // fraction that is. Only a fraction that advanced is reported, so a banded
    // image does not tick the bar once per band within one half-sweep.
    const SweepProgress on_sweep = options.on_sweep;
    const auto on_submission = on_sweep
        ? std::function<void(std::size_t, std::size_t)>(
              [&](std::size_t done, std::size_t total) {
                  if (total == 0 || done == 0) {
                      return;
                  }
                  const double fraction =
                      dispatch_fraction[std::min(done, dispatch_fraction.size()) - 1];
                  if (fraction > reported) {
                      reported = fraction;
                      on_sweep(fraction);
                  }
              })
        : std::function<void(std::size_t, std::size_t)>();
    g::run_batched(*context, dispatches, limiter, /*leading=*/0, /*trailing=*/1,
                   on_submission);

    PatchMatchResult result;
    result.width = width;
    result.height = height;
    result.depth = g_out_depth.read<float>(count);
    result.normals = g_out_normals.read<float>(count * 3);
    result.cost = g_out_cost.read<float>(count);
    return result;
}

/// The whole run: one GPU level per pyramid level, coarsest first, with the
/// inherited map upsampled on the HOST between levels. The host hop is
/// deliberate: PatchMatch's working set is host-visible by design (see
/// working_memory()), so the map being carried is already host memory, and
/// pyramid_upsample() is a single linear pass a device dispatch would not beat.
/// It also keeps the kernels untouched apart from pm_init's seed_mode branch.
PatchMatchResult run_gpu(const PMView& reference, const std::vector<PMView>& neighbours,
                         const PatchMatchOptions& options) {
    const std::shared_ptr<g::Context> context = g::default_context();
    if (!context) {
        throw g::Error("no GPU context");
    }
    if (g_force_failure.exchange(false)) {
        throw g::Error("forced failure (test hook)");
    }

    const int width = reference.image->width();
    const int height = reference.image->height();
    const std::vector<int> iterations_by_level = patch_match_iteration_plan(width, height, options);
    const int level_count = static_cast<int>(iterations_by_level.size());

    int total_sweeps = 0;
    for (const int iterations : iterations_by_level) {
        total_sweeps += 2 * std::max(iterations, 0);
    }
    int done_before = 0;
    double reported = 0.0;

    DepthPyramid pyramid;
    pyramid.reference = reference.image;
    pyramid.calibrations.push_back(reference.calibration);
    if (level_count > 1) {
        pyramid = build_depth_pyramid(*reference.image, reference.calibration, level_count);
    }
    DepthHypotheses inherited;
    PatchMatchResult result;
    for (int level = 0; level < level_count; ++level) {
        const int from_finest = level_count - 1 - level;
        const int iterations = iterations_by_level[static_cast<std::size_t>(level)];

        // This level's neighbour images, derived for this level only and freed
        // when the iteration ends. Holding a reduced copy of every neighbour for
        // every level costs sum(f * N) images for the whole run and only one
        // level's worth is ever live. Level f is the original reduced f times,
        // one halving at a time -- the chain the reference pyramid follows, so
        // truncating sizes and the principal-point formula stay consistent
        // between a level's reference and its neighbours. Building it from the
        // original by repeated halving is bit-identical to reusing the level
        // above: a halving is a pure function of its input.
        std::vector<FloatImage> reduced_neighbours;
        std::vector<Calibration> reduced_calibrations;
        if (from_finest > 0) {
            reduced_neighbours.reserve(neighbours.size());
            reduced_calibrations.reserve(neighbours.size());
            for (std::size_t i = 0; i < neighbours.size(); ++i) {
                // from_finest >= 1 here, so the first halving always reads the
                // original and each later one reads the level above it.
                FloatImage reduced = pyramid_downsample(*neighbours[i].image);
                Calibration reduced_calibration = scaled_calibration(neighbours[i].calibration, 2);
                for (int halvings = 1; halvings < from_finest; ++halvings) {
                    reduced = pyramid_downsample(reduced);
                    reduced_calibration = scaled_calibration(reduced_calibration, 2);
                }
                reduced_neighbours.push_back(std::move(reduced));
                reduced_calibrations.push_back(reduced_calibration);
            }
        }

        PMView level_reference;
        level_reference.calibration = pyramid.calibration_from_finest(from_finest);
        level_reference.R = reference.R;
        level_reference.t = reference.t;
        level_reference.image = &pyramid.image(level);

        std::vector<PMView> level_neighbours;
        level_neighbours.reserve(neighbours.size());
        for (std::size_t i = 0; i < neighbours.size(); ++i) {
            PMView view;
            view.calibration = from_finest == 0 ? neighbours[i].calibration
                                                : reduced_calibrations[i];
            view.R = neighbours[i].R;
            view.t = neighbours[i].t;
            view.image = from_finest == 0 ? neighbours[i].image : &reduced_neighbours[i];
            level_neighbours.push_back(std::move(view));
        }

        const double configured_span = 1.0 / options.depth_min - 1.0 / options.depth_max;
        GpuLevel gpu_level;
        gpu_level.reference = &level_reference;
        gpu_level.neighbours = &level_neighbours;
        gpu_level.seed = pyramid_level_seed(options.seed, from_finest);
        gpu_level.iterations = iterations;
        gpu_level.warm = level > 0 && inherited.width > 0 ? &inherited : nullptr;
        // The CPU computes the warm span on the same inherited map with the same
        // function, so both paths search from the same radius and their results
        // stay comparable.
        gpu_level.span = static_cast<float>(
            level > 0 && inherited.width > 0
                ? std::min(configured_span,
                           pyramid_search_span(inherited.depth, inherited.width, inherited.height,
                                               options.window_radius, configured_span / 64.0))
                : configured_span);
        // This level's share of the WHOLE run: the level's own half-sweeps over
        // the run's. run_gpu_level() spreads that share over the level's sweeps
        // (`base + scale * done / sweeps_total`, sweeps_total being this level's
        // own count), so a level contributes exactly local_sweeps / total_sweeps
        // and the levels together add up to 1.0. A scale of 1 / total_sweeps
        // would divide the level's progress by the run's sweep count a second
        // time, leaving the bar at a fraction of the share it should show.
        gpu_level.progress_base = total_sweeps > 0
                                      ? static_cast<double>(done_before) /
                                            static_cast<double>(total_sweeps)
                                      : 0.0;
        gpu_level.progress_scale =
            total_sweeps > 0 ? static_cast<double>(2 * std::max(iterations, 0)) /
                                   static_cast<double>(total_sweeps)
                             : 1.0;

        result = run_gpu_level(context, gpu_level, options, reported);
        done_before += 2 * std::max(iterations, 0);

        if (level + 1 < level_count) {
            const int next_from_finest = level_count - 2 - level;
            const Calibration& next_calibration =
                pyramid.calibration_from_finest(next_from_finest);
            DepthHypotheses coarse;
            coarse.width = result.width;
            coarse.height = result.height;
            coarse.depth = std::move(result.depth);
            coarse.normals = std::move(result.normals);
            inherited = pyramid_upsample(coarse, next_calibration.width, next_calibration.height);
        }
    }
    if (options.on_sweep && reported < 1.0) {
        options.on_sweep(1.0);
    }
    return result;
}

}  // namespace

bool patch_match_gpu_supported(const PMView& reference, const std::vector<PMView>& neighbours,
                               const PatchMatchOptions& options) {
    if (!g::available() || !g::enabled()) {
        return false;
    }
    return fits_kernel_limits(reference, neighbours, options);
}

int patch_match_gpu_max_samples() { return kMaxSamples; }
int patch_match_gpu_max_neighbours() { return kMaxNeighbours; }
bool patch_match_gpu_prefers_host_memory() { return true; }

void patch_match_gpu_force_failure(bool value) { g_force_failure.store(value); }

bool patch_match_gpu_would_run(const PMView& reference, const std::vector<PMView>& neighbours,
                               const PatchMatchOptions& options) {
    return patch_match_gpu_supported(reference, neighbours, options);
}

PatchMatchResult run_patch_match_device(const PMView& reference,
                                        const std::vector<PMView>& neighbours,
                                        const PatchMatchOptions& options) {
    validate_inputs(reference, neighbours, options);
    // The census decline is a decision, not a failure: it is explained once per
    // process (a per-map line on a 300-camera block would drown the log) and
    // never counted, because nothing was dispatched.
    if (options.cost_type == CostType::Census) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true)) {
            log(LogLevel::Info)
                << "PatchMatch: the census cost runs on the CPU (the GPU kernels' "
                   "binding budget cannot hold a per-view census buffer); "
                << (g::available() && g::enabled() ? "the device is left unused for this stage"
                                                  : "no GPU is in use anyway");
        }
        return run_patch_match(reference, neighbours, options);
    }
    const bool gpu_ok = fits_kernel_limits(reference, neighbours, options);
    if (!gpu_ok) {
        return run_patch_match(reference, neighbours, options);
    }
    return g::with_fallback<PatchMatchResult>(
        "PatchMatch", [&] { return run_gpu(reference, neighbours, options); },
        [&] { return run_patch_match(reference, neighbours, options); });
}

}  // namespace alphashape::stereo
