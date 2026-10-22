#include "common/algorithm/mesh/tv_fusion.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape::meshing {
namespace {

/// Minimise 0.5 (u - v)^2 + a * sum_k h_k |u - c_k| over u, clamped to [-1, 1].
///
/// `c` is strictly increasing and `h` non-negative; `a` is the primal step
/// times the data weight (tau * lambda). The objective is strictly convex in u,
/// so the answer is its unique stationary point. The sorted centres split the
/// line into K + 1 intervals; on interval j the subgradient of the data term is
/// the constant a * (L_j - R_j), giving the candidate u = v - a (L_j - R_j).
/// If no interval contains its own candidate the minimiser sits exactly on a
/// knot c_k, where 0 lies in the subgradient interval [-a h_k, a h_k].
double tv_l1_prox(const double v, const double a, const double* h, const std::vector<double>& c) {
    const std::size_t bins = c.size();
    double total = 0.0;
    for (std::size_t k = 0; k < bins; ++k) {
        total += h[k];
    }
    if (total <= 0.0 || a <= 0.0) {
        return std::min(1.0, std::max(-1.0, v));
    }

    const double inf = std::numeric_limits<double>::infinity();
    double left = 0.0;  // sum of h_k over k < j
    for (std::size_t j = 0; j <= bins; ++j) {
        const double right = total - left;
        const double candidate = v - a * (left - right);
        const double lower = (j == 0) ? -inf : c[j - 1];
        const double upper = (j == bins) ? inf : c[j];
        if (candidate > lower && candidate < upper) {
            return std::min(1.0, std::max(-1.0, candidate));
        }
        if (j < bins) {
            left += h[j];
        }
    }

    // Knot fallback: u = c_k with 0 in the subgradient at that knot.
    double lower_weight = 0.0;  // sum of h_i over i < k
    for (std::size_t k = 0; k < bins; ++k) {
        const double upper_weight = total - lower_weight - h[k];
        const double residual = v - c[k] - a * (lower_weight - upper_weight);
        if (std::abs(residual) <= a * h[k]) {
            return std::min(1.0, std::max(-1.0, c[k]));
        }
        lower_weight += h[k];
    }
    return std::min(1.0, std::max(-1.0, v));
}

/// Bin centre c_k uniformly covering [-1, 1].
double bin_centre(const int k, const int bins) {
    if (bins <= 1) {
        return 0.0;
    }
    return -1.0 + 2.0 * static_cast<double>(k) / static_cast<double>(bins - 1);
}

/// The bin a clamped signed distance f in [-1, 1] falls into.
int bin_of(const double f, const int bins) {
    const double scaled = 0.5 * (f + 1.0) * static_cast<double>(bins - 1);
    int index = static_cast<int>(std::lround(scaled));
    index = std::max(0, std::min(bins - 1, index));
    return index;
}

}  // namespace

namespace detail {

TvFusionSetup tv_l1_fusion_setup(const std::vector<FusionView>& views,
                                 const Eigen::Vector3d& bbox_min,
                                 const Eigen::Vector3d& bbox_max,
                                 const TvFusionOptions& options) {
    // ---- Stage 1: validation -------------------------------------------------
    if (views.empty()) {
        throw ValueError("tv_l1_fusion: at least one view is required");
    }
    if (options.histogram_bins < 2) {
        throw ValueError("tv_l1_fusion: histogram_bins must be at least 2");
    }
    if (options.iterations < 0) {
        throw ValueError("tv_l1_fusion: iterations must be non-negative");
    }
    if (!(options.lambda > 0.0)) {
        throw ValueError("tv_l1_fusion: lambda must be positive");
    }
    if (!(options.truncation_voxels > 0.0)) {
        throw ValueError("tv_l1_fusion: truncation_voxels must be positive");
    }
    if (!(options.front_band >= 0.0)) {
        throw ValueError("tv_l1_fusion: front_band must be non-negative");
    }
    if (options.pixel_step < 1) {
        throw ValueError("tv_l1_fusion: pixel_step must be at least 1");
    }
    for (const FusionView& view : views) {
        if (view.map == nullptr) {
            throw ValueError("tv_l1_fusion: view has a null depth map");
        }
        const DepthMap& map = *view.map;
        const std::size_t expected =
            static_cast<std::size_t>(map.width) * static_cast<std::size_t>(map.height);
        if (map.depths.size() != expected) {
            throw ValueError("tv_l1_fusion: depth map size does not match width * height");
        }
        if (view.calibration.width != map.width || view.calibration.height != map.height) {
            throw ValueError("tv_l1_fusion: calibration size does not match the depth map");
        }
    }

    TvFusionSetup setup;
    setup.grid = make_grid(bbox_min, bbox_max, options.max_resolution);
    const GridSpec& grid = setup.grid;
    const std::size_t voxels = grid.size();
    setup.truncation = options.truncation_voxels * grid.voxel;
    const double truncation = setup.truncation;
    setup.bins = options.histogram_bins;
    const int bins = setup.bins;

    std::vector<double>& centres = setup.centres;
    centres.assign(static_cast<std::size_t>(bins), 0.0);
    for (int k = 0; k < bins; ++k) {
        centres[static_cast<std::size_t>(k)] = bin_centre(k, bins);
    }

    // ---- Stage 2: histogram integration --------------------------------------
    // counts[v * bins + k] counts the rays placing voxel v in bin k. Every voxel
    // on a ray between the free-space front band and eta behind the surface gets
    // its signed distance to that surface binned; the data term below is an L1
    // penalty over the resulting per-voxel histogram.
    std::vector<std::uint32_t> counts(voxels * static_cast<std::size_t>(bins), 0U);
    const double step = 0.5 * grid.voxel;

    for (const FusionView& view : views) {
        const DepthMap& map = *view.map;
        const int width = map.width;
        const int height = map.height;
        const Eigen::Vector3d camera_centre = -view.R.transpose() * view.t;
        const Eigen::Matrix3d world_from_camera = view.R.transpose();
        const int row_step = options.pixel_step;
        const double band = options.front_band * truncation;

#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 8)
#endif
        for (int j = 0; j < height; j += row_step) {
            for (int i = 0; i < width; i += row_step) {
                const std::size_t pixel =
                    static_cast<std::size_t>(j) * static_cast<std::size_t>(width) +
                    static_cast<std::size_t>(i);
                const double depth = static_cast<double>(map.depths[pixel]);
                if (!(depth > 0.0)) {
                    continue;  // -1 (and any non-positive value) marks "no depth"
                }
                const Eigen::Vector2d centre_pixel(static_cast<double>(i) + 0.5,
                                                   static_cast<double>(j) + 0.5);
                const Eigen::Vector3d bearing = view.calibration.unproject(centre_pixel);
                const Eigen::Vector3d direction = world_from_camera * bearing;

                const double t_begin = depth - band;
                const double t_end = depth + truncation;
                std::size_t previous = std::numeric_limits<std::size_t>::max();
                for (double t = t_begin; t <= t_end; t += step) {
                    const Eigen::Vector3d point = camera_centre + t * direction;
                    const Eigen::Vector3d local = (point - grid.origin) / grid.voxel;
                    const int vx = static_cast<int>(std::floor(local.x()));
                    const int vy = static_cast<int>(std::floor(local.y()));
                    const int vz = static_cast<int>(std::floor(local.z()));
                    if (!grid.contains(vx, vy, vz)) {
                        continue;
                    }
                    const std::size_t index = grid.index(vx, vy, vz);
                    if (index == previous) {
                        continue;  // same voxel as the previous sample on this ray
                    }
                    previous = index;

                    const double along = (grid.centre(vx, vy, vz) - camera_centre).dot(direction);
                    const double signed_distance = depth - along;
                    const double clamped = std::max(-1.0, std::min(1.0, signed_distance / truncation));
                    const int bin = bin_of(clamped, bins);
                    std::atomic_ref<std::uint32_t> slot(
                        counts[index * static_cast<std::size_t>(bins) + static_cast<std::size_t>(bin)]);
                    slot.fetch_add(1U, std::memory_order_relaxed);
                }
            }
        }
    }

    // ---- Stage 3: observed mask + initial field ------------------------------
    // A voxel is "observed" when at least one ray binned a measurement into it;
    // the mask is then dilated by one voxel so that the surface can be found on
    // the boundary between measured free space and the unobserved interior. The
    // initial field is the weighted median of the bin centres, which is the
    // outlier-resistant starting point the convex problem expects.
    std::vector<std::uint8_t> measured(voxels, 0U);
    std::vector<float>& field = setup.initial_field;
    field.assign(voxels, 0.0F);
    std::vector<double>& weights = setup.weights;
    weights.assign(voxels * static_cast<std::size_t>(bins), 0.0);
    for (std::size_t v = 0; v < voxels; ++v) {
        const std::size_t base = v * static_cast<std::size_t>(bins);
        std::uint64_t weight = 0U;
        for (int k = 0; k < bins; ++k) {
            weight += static_cast<std::uint64_t>(counts[base + static_cast<std::size_t>(k)]);
        }
        if (weight == 0U) {
            continue;
        }
        measured[v] = 1U;
        // The histogram is normalised to a probability distribution, so a voxel
        // seen once weighs the same as one seen ten times. This is deliberate:
        // measured on the seq truth-pose scene (milestone 7), raw counts give 31
        // components / 4.2% floaters against 2 / 0.3% for the normalised form at
        // equal median accuracy, i.e. one lucky outlier view would otherwise
        // outweigh ten agreeing views.
        const double inverse = 1.0 / static_cast<double>(weight);
        std::uint64_t cumulative = 0U;
        bool median_found = false;
        for (int k = 0; k < bins; ++k) {
            const std::uint64_t bin_count =
                static_cast<std::uint64_t>(counts[base + static_cast<std::size_t>(k)]);
            cumulative += bin_count;
            weights[base + static_cast<std::size_t>(k)] =
                static_cast<double>(bin_count) * inverse;
            if (!median_found && cumulative * 2U >= weight) {
                field[v] = static_cast<float>(centres[static_cast<std::size_t>(k)]);
                median_found = true;
            }
        }
    }

    std::vector<std::uint8_t>& observed = setup.observed;
    observed = measured;
    for (int z = 0; z < grid.nz; ++z) {
        for (int y = 0; y < grid.ny; ++y) {
            for (int x = 0; x < grid.nx; ++x) {
                if (measured[grid.index(x, y, z)] != 0U) {
                    continue;
                }
                const int neighbours[6][3] = {{x - 1, y, z}, {x + 1, y, z}, {x, y - 1, z},
                                              {x, y + 1, z}, {x, y, z - 1}, {x, y, z + 1}};
                bool touches = false;
                for (const auto& neighbour : neighbours) {
                    if (grid.contains(neighbour[0], neighbour[1], neighbour[2]) &&
                        measured[grid.index(neighbour[0], neighbour[1], neighbour[2])] != 0U) {
                        touches = true;
                        break;
                    }
                }
                if (touches) {
                    observed[grid.index(x, y, z)] = 1U;
                }
            }
        }
    }

    // Unmeasured voxels left at the initial 0 would sit exactly ON the iso level,
    // so any ray that reaches one registers a spurious crossing (the cage floor
    // under a sphere, for instance). Seed them from the nearest measurement with
    // a breadth-first propagation instead, so the initial field is a clean
    // nearest-neighbour extension with a single zero crossing per ray.
    {
        constexpr int unvisited = -1;
        std::vector<int> distance(voxels, unvisited);
        std::size_t head = 0;
        std::vector<std::size_t> queue;
        queue.reserve(voxels);
        for (std::size_t v = 0; v < voxels; ++v) {
            if (measured[v] != 0U) {
                distance[v] = 0;
                queue.push_back(v);
            }
        }
        const std::size_t stride_y = static_cast<std::size_t>(grid.nx);
        const std::size_t stride_z = stride_y * static_cast<std::size_t>(grid.ny);
        while (head < queue.size()) {
            const std::size_t index = queue[head++];
            const int level = distance[index] + 1;
            const float value = field[index];
            const int x = static_cast<int>(index % stride_y);
            const int y = static_cast<int>((index / stride_y) % static_cast<std::size_t>(grid.ny));
            const int z = static_cast<int>(index / stride_z);
            const std::size_t neighbour_indices[6] = {
                x > 0 ? index - 1U : voxels, x + 1 < grid.nx ? index + 1U : voxels,
                y > 0 ? index - stride_y : voxels,
                y + 1 < grid.ny ? index + stride_y : voxels,
                z > 0 ? index - stride_z : voxels,
                z + 1 < grid.nz ? index + stride_z : voxels};
            for (const std::size_t neighbour : neighbour_indices) {
                if (neighbour >= voxels || distance[neighbour] != unvisited) {
                    continue;
                }
                distance[neighbour] = level;
                field[neighbour] = value;
                queue.push_back(neighbour);
            }
        }
    }

    return setup;
}

}  // namespace detail

VolumeResult tv_l1_fusion(const std::vector<FusionView>& views, const Eigen::Vector3d& bbox_min,
                          const Eigen::Vector3d& bbox_max, const TvFusionOptions& options) {
    return detail::tv_l1_fusion_solve(
        detail::tv_l1_fusion_setup(views, bbox_min, bbox_max, options), options);
}

VolumeResult detail::tv_l1_fusion_solve(TvFusionSetup setup, const TvFusionOptions& options) {
    const GridSpec& grid = setup.grid;
    const std::size_t voxels = grid.size();
    const int bins = setup.bins;
    const std::vector<double>& centres = setup.centres;
    const std::vector<double>& weights = setup.weights;
    const std::vector<float>& field = setup.initial_field;

    VolumeResult result;
    result.field = VoxelGrid<float>(grid, 0.0F);
    result.iso = 0.0F;
    result.observed = std::move(setup.observed);

    // ---- Stage 5: Chambolle-Pock primal-dual ---------------------------------
    // Duality-gap / ergodic step sizes. The forward-difference gradient operator
    // has squared norm 12 on a 3D grid (2^2 per axis), so tau * sigma = 1/12 is
    // the boundary of the stability region; the symmetric choice is 1/sqrt(12).
    const double tau = 1.0 / std::sqrt(12.0);
    const double sigma = tau;
    const double data_weight = tau * options.lambda;

    std::vector<float> u = field;                 // primal variable
    std::vector<float> ubar = field;              // extrapolated primal
    std::vector<float> px(voxels, 0.0F);          // dual field, x component
    std::vector<float> py(voxels, 0.0F);          // dual field, y component
    std::vector<float> pz(voxels, 0.0F);          // dual field, z component

    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        // (1) Dual ascent with projection onto the unit ball: p <- P(p + sigma * grad ubar).
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int z = 0; z < grid.nz; ++z) {
            for (int y = 0; y < grid.ny; ++y) {
                for (int x = 0; x < grid.nx; ++x) {
                    const std::size_t index = grid.index(x, y, z);
                    const float here = ubar[index];
                    const float gx = (x + 1 < grid.nx) ? ubar[index + 1U] - here : 0.0F;
                    const float gy = (y + 1 < grid.ny)
                                         ? ubar[index + static_cast<std::size_t>(grid.nx)] - here
                                         : 0.0F;
                    const float gz = (z + 1 < grid.nz)
                                         ? ubar[index + static_cast<std::size_t>(grid.nx) *
                                                            static_cast<std::size_t>(grid.ny)] -
                                               here
                                         : 0.0F;
                    float ax = px[index] + static_cast<float>(sigma) * gx;
                    float ay = py[index] + static_cast<float>(sigma) * gy;
                    float az = pz[index] + static_cast<float>(sigma) * gz;
                    const float norm = std::sqrt(ax * ax + ay * ay + az * az);
                    if (norm > 1.0F) {
                        const float inverse = 1.0F / norm;
                        ax *= inverse;
                        ay *= inverse;
                        az *= inverse;
                    }
                    px[index] = ax;
                    py[index] = ay;
                    pz[index] = az;
                }
            }
        }

        // (2) Primal descent: u <- prox_{tau f}(u + tau div p), with the
        // extrapolation ubar <- 2u - u_old folded into the same sweep. div is the
        // exact negative adjoint of the forward-difference gradient above: at a
        // voxel the outward face contributes p[i], the inward face p[i - 1],
        // and the missing (Neumann) face contributes nothing.
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int z = 0; z < grid.nz; ++z) {
            const std::size_t stride_y = static_cast<std::size_t>(grid.nx);
            const std::size_t stride_z = stride_y * static_cast<std::size_t>(grid.ny);
            for (int y = 0; y < grid.ny; ++y) {
                for (int x = 0; x < grid.nx; ++x) {
                    const std::size_t index = grid.index(x, y, z);
                    double divergence = static_cast<double>(px[index]) -
                                        (x > 0 ? static_cast<double>(px[index - 1U]) : 0.0);
                    divergence += static_cast<double>(py[index]) -
                                  (y > 0 ? static_cast<double>(py[index - stride_y]) : 0.0);
                    divergence += static_cast<double>(pz[index]) -
                                  (z > 0 ? static_cast<double>(pz[index - stride_z]) : 0.0);

                    const double base = static_cast<double>(u[index]) + tau * divergence;
                    const std::size_t histogram_base = index * static_cast<std::size_t>(bins);
                    const double* histogram = &weights[histogram_base];
                    const float updated =
                        static_cast<float>(tv_l1_prox(base, data_weight, histogram, centres));
                    ubar[index] = static_cast<float>(2.0) * updated - u[index];
                    u[index] = updated;
                }
            }
        }
    }

    result.field.data = u;
    return result;
}

}  // namespace alphashape::meshing
