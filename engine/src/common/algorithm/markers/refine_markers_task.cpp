#include "common/algorithm/markers/refine_markers_task.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "base/utils/omp.hpp"
#include "common/algorithm/markers/cross_decoder.hpp"
#include "common/algorithm/markers/marker_detector.hpp"
#include "common/algorithm/markers/project_markers.hpp"
#include "common/algorithm/sfm/absolute_pose.hpp"
#include "common/algorithm/sfm/triangulation.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/core/marker.hpp"
#include "common/core/types.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "image/operators/downsample.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

/// Patch energy below this standard deviation is "no evidence", not a match
/// (Foerstner & Gulch 1987). A flat patch carries no gradient to align to, so
/// correlating it with anything would produce a meaningless score.
constexpr double kMinPatchStdDev = 0.02;
/// Fraction of the physical target radius used as the depth search step.
constexpr double kDepthStepFraction = 0.5;
/// Half-width, in target radii, of the sampled physical patch.
constexpr double kPatchHalfWidth = 1.5;
/// Samples used to characterise the plane around the initial estimate.
constexpr int kMinPlanePoints = 6;
/// Below this round-to-round movement the target refiner's own noise dominates;
/// another round only adds jitter, so the iteration stops. This is a floor on
/// the caller's `tolerance_px`, never a replacement for it.
constexpr double kRoundStopNoisePx = 0.05;

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

/// One marker's local window in one camera. Only the reduced-resolution crop
/// around that marker and its gradients are kept, never a whole image.
struct CameraView {
    const Camera* camera = nullptr;
    const Calibration* calibration = nullptr;
    /// Crop at `downscale`; pixel (x, y) here is reduced-image index
    /// (offset_x + x, offset_y + y).
    FloatImage image;
    /// dI/d(reduced index), computed once when the crop is cut.
    FloatImage gradient_x;
    FloatImage gradient_y;
    int downscale = 1;
    int offset_x = 0;
    int offset_y = 0;
    int full_width = 0;   ///< reduced-image width (window origin is relative to it)
    int full_height = 0;

    bool aligned() const { return camera != nullptr && camera->enabled; }
    bool loaded() const { return !image.empty(); }

    /// Reduced-image index of a full-resolution pixel coordinate. The reduced
    /// image keeps the calibration convention u' = u / downscale.
    Eigen::Vector2d index_of(const Eigen::Vector2d& pixel) const {
        const double ds = static_cast<double>(downscale);
        return pixel / ds - Eigen::Vector2d(0.5, 0.5);
    }

    /// Reduced-image index relative to this window's top-left pixel.
    Eigen::Vector2d local_of(const Eigen::Vector2d& pixel) const {
        return index_of(pixel) -
               Eigen::Vector2d(static_cast<double>(offset_x),
                               static_cast<double>(offset_y));
    }

    bool contains_pixel(const Eigen::Vector2d& pixel, double margin = 1.0) const {
        if (!loaded()) {
            return false;
        }
        const Eigen::Vector2d p = local_of(pixel);
        return p.x() >= margin && p.y() >= margin &&
               p.x() <= static_cast<double>(image.width() - 1) - margin &&
               p.y() <= static_cast<double>(image.height() - 1) - margin;
    }

    float sample(const Eigen::Vector2d& pixel) const {
        const Eigen::Vector2d p = local_of(pixel);
        return image.sample(static_cast<float>(p.x()), static_cast<float>(p.y()));
    }

    /// Image gradient with respect to a full-resolution pixel shift, from the
    /// gradient stored for this window.
    Eigen::Vector2d gradient(const Eigen::Vector2d& pixel) const {
        const Eigen::Vector2d p = local_of(pixel);
        const double ds = static_cast<double>(std::max(1, downscale));
        const double dx = static_cast<double>(
                              gradient_x.sample(static_cast<float>(p.x()),
                                                static_cast<float>(p.y()))) /
                          ds;
        const double dy = static_cast<double>(
                              gradient_y.sample(static_cast<float>(p.x()),
                                                static_cast<float>(p.y()))) /
                          ds;
        return Eigen::Vector2d(dx, dy);
    }
};

/// Read-only view of the two point sets that describe the scene around a
/// marker: the dense cloud (large; never copied) and the other markers'
/// positions (small, but still not copied).
struct CloudSamples {
    const std::vector<Eigen::Vector3d>* dense = nullptr;
    const std::vector<Eigen::Vector3d>* markers = nullptr;
};

template <typename Function>
void for_each_cloud_point(const CloudSamples& samples, Function&& function) {
    // Non-finite samples are rejected here, before any floor/ceil/lround or
    // sort that would turn them into undefined behaviour.
    if (samples.dense != nullptr) {
        for (const Eigen::Vector3d& point : *samples.dense) {
            if (point.allFinite()) {
                function(point);
            }
        }
    }
    if (samples.markers != nullptr) {
        for (const Eigen::Vector3d& point : *samples.markers) {
            if (point.allFinite()) {
                function(point);
            }
        }
    }
}

/// Chunk-local ray from a camera centre through a pixel direction (unit).
Eigen::Vector3d camera_centre(const Camera& camera) {
    return -camera.pose.block<3, 3>(0, 0).transpose() *
           camera.pose.block<3, 1>(0, 3);
}

std::optional<Eigen::Vector2d> project_view(const CameraView& view,
                                            const Eigen::Vector3d& point) {
    return project_marker(point, *view.camera, *view.calibration, false);
}

struct Observation {
    const CameraView* view = nullptr;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
};

/// Depth of a chunk-local point in one view, in chunk units (distance from the
/// camera centre, which is the depth-map convention and stays valid for fisheye).
double depth_in_view(const CameraView& view, const Eigen::Vector3d& point) {
    const Eigen::Matrix3d rotation = view.camera->pose.block<3, 3>(0, 0);
    const Eigen::Vector3d translation = view.camera->pose.block<3, 1>(0, 3);
    return (rotation * point + translation).norm();
}

/// Least-squares cost over every observation; infinity as soon as one view
/// cannot see the point.
double reprojection_cost(const std::vector<Observation>& observations,
                         const Eigen::Vector3d& point) {
    double cost = 0.0;
    for (const Observation& observation : observations) {
        const auto pixel = project_view(*observation.view, point);
        if (!pixel) {
            return std::numeric_limits<double>::infinity();
        }
        const Eigen::Vector2d difference = *pixel - observation.pixel;
        cost += difference.squaredNorm();
    }
    return cost;
}

/// DLT (Hartley & Zisserman 2004, ch. 12) followed by Gauss-Newton on the
/// reprojection error. The linear solve gives a finite starting point even when
/// the projections are a few pixels off; the refinement removes the bias the
/// algebraic error introduces.
std::optional<Eigen::Vector3d> triangulate_observations(
    const std::vector<Observation>& observations) {
    if (observations.size() < 2) {
        return std::nullopt;
    }
    std::vector<sfm::AbsolutePose> poses;
    std::vector<Eigen::Vector3d> bearings;
    poses.reserve(observations.size());
    bearings.reserve(observations.size());
    for (const Observation& observation : observations) {
        sfm::AbsolutePose pose;
        pose.R = observation.view->camera->pose.block<3, 3>(0, 0);
        pose.t = observation.view->camera->pose.block<3, 1>(0, 3);
        poses.push_back(pose);
        Eigen::Vector3d bearing =
            observation.view->calibration->unproject(observation.pixel);
        if (!bearing.allFinite() || bearing.norm() <= 0.0) {
            return std::nullopt;
        }
        bearings.push_back(bearing.normalized());
    }
    sfm::TriangulationOptions options;
    options.min_parallax_rad = 0.5 * 3.14159265358979323846 / 180.0;
    auto estimate = sfm::triangulate_multiview(poses, bearings, options);
    if (!estimate) {
        return std::nullopt;
    }
    Eigen::Vector3d point = *estimate;
    double cost = reprojection_cost(observations, point);
    if (!std::isfinite(cost)) {
        return std::nullopt;
    }
    for (int iteration = 0; iteration < 10; ++iteration) {
        Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
        Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
        const double h = 1e-6 * std::max(1.0, point.norm());
        for (const Observation& observation : observations) {
            const auto base = project_view(*observation.view, point);
            if (!base) {
                return std::nullopt;
            }
            Eigen::Matrix<double, 2, 3> jacobian;
            for (int axis = 0; axis < 3; ++axis) {
                Eigen::Vector3d delta = Eigen::Vector3d::Zero();
                delta[axis] = h;
                const auto plus = project_view(*observation.view, point + delta);
                const auto minus = project_view(*observation.view, point - delta);
                if (!plus || !minus) {
                    return std::nullopt;
                }
                jacobian.col(axis) = (*plus - *minus) / (2.0 * h);
            }
            const Eigen::Vector2d residual = *base - observation.pixel;
            normal += jacobian.transpose() * jacobian;
            gradient += jacobian.transpose() * residual;
        }
        const double damping = 1e-12 * std::max(1.0, normal.trace());
        const Eigen::Vector3d step =
            -(normal + damping * Eigen::Matrix3d::Identity()).ldlt().solve(gradient);
        if (!step.allFinite()) {
            break;
        }
        double scale = 1.0;
        bool improved = false;
        for (int attempt = 0; attempt < 8; ++attempt) {
            const Eigen::Vector3d candidate = point + scale * step;
            const double candidate_cost = reprojection_cost(observations, candidate);
            if (candidate_cost < cost) {
                point = candidate;
                cost = candidate_cost;
                improved = true;
                break;
            }
            scale *= 0.5;
        }
        if (!improved || step.norm() < 1e-10) {
            break;
        }
    }
    return point;
}

/// Robust triangulation (Fischler & Bolles 1981): try every two-view minimal
/// sample, count the views whose measured projection is within `tolerances`,
/// and re-triangulate from the largest consistent set. A single grossly wrong
/// projection therefore cannot bias the starting point; without this step one
/// 30 px outlier shifts the DLT solution enough to poison the depth search.
std::optional<Eigen::Vector3d> triangulate_robust(
    const std::vector<Observation>& observations,
    const std::vector<double>& tolerances) {
    if (observations.size() <= 2 || tolerances.size() != observations.size()) {
        return triangulate_observations(observations);
    }
    std::vector<std::size_t> best_inliers;
    for (std::size_t i = 0; i < observations.size(); ++i) {
        for (std::size_t j = i + 1; j < observations.size(); ++j) {
            const std::vector<Observation> pair = {observations[i],
                                                   observations[j]};
            const auto estimate = triangulate_observations(pair);
            if (!estimate) {
                continue;
            }
            std::vector<std::size_t> inliers;
            for (std::size_t k = 0; k < observations.size(); ++k) {
                const auto projected =
                    project_view(*observations[k].view, *estimate);
                if (projected &&
                    (*projected - observations[k].pixel).norm() <= tolerances[k]) {
                    inliers.push_back(k);
                }
            }
            if (inliers.size() > best_inliers.size()) {
                best_inliers = std::move(inliers);
            }
        }
    }
    if (best_inliers.size() >= 2) {
        std::vector<Observation> subset;
        subset.reserve(best_inliers.size());
        for (const std::size_t index : best_inliers) {
            subset.push_back(observations[index]);
        }
        if (const auto refined = triangulate_observations(subset)) {
            return refined;
        }
    }
    return triangulate_observations(observations);
}

struct BlobSeed {
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    double radius = 0.0;
};

/// Locate the circular target around `pixel` in the window. The blob's scale
/// supplies the apparent radius in full-resolution pixels; the ellipse refit
/// supplies a seed for the 2-D fallback. Crosses never call this: their radius
/// comes from the cross detector, or from the projected physical radius.
std::optional<BlobSeed> circular_seed(const CameraView& view,
                                      const Eigen::Vector2d& pixel, int patch_size) {
    if (!view.loaded()) {
        return std::nullopt;
    }
    const Eigen::Vector2d local = view.local_of(pixel);
    if (!local.allFinite()) {
        return std::nullopt;
    }
    // The crop covers the whole target even when the measurement is up to one
    // radius from its printed centre.
    const int half = std::max(48, patch_size);
    const int centre_x = static_cast<int>(std::floor(std::clamp(
        local.x(), -1.0e6, 1.0e6)));
    const int centre_y = static_cast<int>(std::floor(std::clamp(
        local.y(), -1.0e6, 1.0e6)));
    const int x0 = std::clamp(centre_x - half, 0,
                              std::max(0, view.image.width() - 2 * half - 1));
    const int y0 = std::clamp(centre_y - half, 0,
                              std::max(0, view.image.height() - 2 * half - 1));
    const int width = std::min(2 * half + 1, view.image.width() - x0);
    const int height = std::min(2 * half + 1, view.image.height() - y0);
    if (width < 16 || height < 16) {
        return std::nullopt;
    }
    FloatImage crop(width, height, 0.0F);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            crop.at(x, y) = view.image.at(x0 + x, y0 + y);
        }
    }
    BlobParams params;
    params.dark_on_light = true;
    params.min_radius = 2;
    params.max_radius = std::max(4, half);
    params.threshold = 0.02F;
    params.min_axis_ratio = 0.3;
    const std::vector<MarkerCandidate> blobs = detect_blobs(crop, params);
    const Eigen::Vector2d seed(local.x() - static_cast<double>(x0) + 0.5,
                               local.y() - static_cast<double>(y0) + 0.5);
    double max_blob_radius = 0.0;
    for (const MarkerCandidate& blob : blobs) {
        if (blob.center.allFinite() && std::isfinite(blob.radius_px)) {
            max_blob_radius = std::max(max_blob_radius, blob.radius_px);
        }
    }
    if (!(max_blob_radius > 0.0)) {
        return std::nullopt;
    }
    // A coded disc breaks into sector blobs. Keep the credible ones (ground
    // texture produces much smaller blobs), take their radius-weighted
    // centroid, then re-centre on the compact ring of target blobs. This lands
    // near the printed centre even when the measurement is most of a radius
    // away.
    std::vector<const MarkerCandidate*> kept;
    for (const MarkerCandidate& blob : blobs) {
        if (!blob.center.allFinite() || !std::isfinite(blob.radius_px) ||
            blob.radius_px < 0.35 * max_blob_radius ||
            (blob.center - seed).norm() > static_cast<double>(half)) {
            continue;
        }
        kept.push_back(&blob);
    }
    if (kept.empty()) {
        return std::nullopt;
    }
    const auto centroid_of =
        [](const std::vector<const MarkerCandidate*>& set)
        -> std::optional<Eigen::Vector2d> {
        double weight_sum = 0.0;
        Eigen::Vector2d sum = Eigen::Vector2d::Zero();
        for (const MarkerCandidate* blob : set) {
            const double weight =
                std::max(blob->response, 1e-6) * blob->radius_px;
            sum += blob->center * weight;
            weight_sum += weight;
        }
        if (!(weight_sum > 0.0) || !sum.allFinite()) {
            return std::nullopt;
        }
        const Eigen::Vector2d centroid = sum / weight_sum;
        return centroid;
    };
    const auto first = centroid_of(kept);
    if (!first) {
        return std::nullopt;
    }
    const auto median_extent =
        [](const std::vector<const MarkerCandidate*>& set,
           const Eigen::Vector2d& centre) {
        std::vector<double> values;
        values.reserve(set.size());
        for (const MarkerCandidate* blob : set) {
            values.push_back((blob->center - centre).norm() +
                             blob->radius_px);
        }
        if (values.empty()) {
            return 0.0;
        }
        const std::size_t middle = values.size() / 2;
        std::nth_element(
            values.begin(),
            values.begin() + static_cast<std::ptrdiff_t>(middle),
            values.end());
        return values[middle];
    };
    const double first_extent = median_extent(kept, *first);
    if (!(first_extent > 0.0)) {
        return std::nullopt;
    }
    std::vector<const MarkerCandidate*> ring;
    for (const MarkerCandidate* blob : kept) {
        if ((blob->center - *first).norm() <= 1.25 * first_extent) {
            ring.push_back(blob);
        }
    }
    if (ring.empty()) {
        return std::nullopt;
    }
    const auto second = centroid_of(ring);
    if (!second) {
        return std::nullopt;
    }
    const Eigen::Vector2d centroid = *second;
    const double outer = std::max(4.0, median_extent(ring, centroid));
    if (!std::isfinite(outer)) {
        return std::nullopt;
    }
    const Eigen::Vector2d window_centre(
        centroid.x() + static_cast<double>(x0),
        centroid.y() + static_cast<double>(y0));
    MarkerCandidate candidate;
    candidate.center = window_centre;
    candidate.radius_px = outer;
    Eigen::Vector2d centre = window_centre;
    if (const auto fitted = refine_ellipse_center(view.image, candidate)) {
        centre = *fitted;
    }
    if (!centre.allFinite() || !(outer > 0.0)) {
        return std::nullopt;
    }
    BlobSeed out;
    out.pixel = (centre + Eigen::Vector2d(
                             static_cast<double>(view.offset_x),
                             static_cast<double>(view.offset_y))) *
                static_cast<double>(view.downscale);
    out.radius = outer * static_cast<double>(view.downscale);
    return out;
}

/// Local target-plane normal. A plane fitted to nearby cloud points follows a
/// tilted target; without enough points the marker is assumed fronto-parallel
/// to the average view (the negated mean viewing direction).
Eigen::Vector3d target_normal(const Eigen::Vector3d& point,
                              const std::vector<Observation>& observations,
                              const CloudSamples& cloud, double radius) {
    std::vector<Eigen::Vector3d> nearby;
    for_each_cloud_point(cloud, [&](const Eigen::Vector3d& sample) {
        if ((sample - point).norm() <= radius) {
            nearby.push_back(sample);
        }
    });
    if (static_cast<int>(nearby.size()) >= kMinPlanePoints) {
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        for (const Eigen::Vector3d& sample : nearby) {
            mean += sample;
        }
        mean /= static_cast<double>(nearby.size());
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
        for (const Eigen::Vector3d& sample : nearby) {
            const Eigen::Vector3d centred = sample - mean;
            covariance += centred * centred.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
        if (solver.info() == Eigen::Success) {
            const Eigen::Vector3d normal = solver.eigenvectors().col(0);
            if (normal.allFinite() && normal.norm() > 0.0) {
                return normal.normalized();
            }
        }
    }
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Observation& observation : observations) {
        const Eigen::Vector3d to_camera =
            camera_centre(*observation.view->camera) - point;
        if (to_camera.norm() > 0.0) {
            mean += to_camera.normalized();
        }
    }
    if (mean.norm() > 1e-12) {
        return mean.normalized();
    }
    return Eigen::Vector3d::UnitY();
}

/// In-plane axes spanning the target plane, orthogonal to `normal`.
std::pair<Eigen::Vector3d, Eigen::Vector3d> plane_axes(
    const Eigen::Vector3d& normal) {
    Eigen::Vector3d reference = Eigen::Vector3d::UnitZ();
    if (std::abs(normal.dot(reference)) > 0.9) {
        reference = Eigen::Vector3d::UnitX();
    }
    const Eigen::Vector3d u = (reference - normal * reference.dot(normal)).normalized();
    const Eigen::Vector3d v = normal.cross(u).normalized();
    return {u, v};
}

/// One affine-sampled square patch: the image of a physical plane patch through
/// a camera (Baker & Matthews 2004, the inverse-compositional template model).
struct Patch {
    std::vector<float> values;
    std::vector<Eigen::Vector2d> pixels;
    double mean = 0.0;
    double stddev = 0.0;
    bool valid = false;
};

/// Sample a `patch_size` square grid on the plane (centre, normal) through
/// `view`. Every view uses the same physical grid, so an oblique view samples
/// the same pattern rather than an axis-aligned image window.
Patch sample_patch(const CameraView& view, const Eigen::Vector3d& centre,
                   const Eigen::Vector3d& u, const Eigen::Vector3d& v,
                   double half_width, int patch_size) {
    Patch patch;
    const int count = std::max(3, patch_size);
    patch.values.reserve(static_cast<std::size_t>(count * count));
    patch.pixels.reserve(static_cast<std::size_t>(count * count));
    const double denominator = static_cast<double>(count - 1);
    for (int row = 0; row < count; ++row) {
        const double sv = 2.0 * static_cast<double>(row) / denominator - 1.0;
        for (int column = 0; column < count; ++column) {
            const double su = 2.0 * static_cast<double>(column) / denominator - 1.0;
            const Eigen::Vector3d point = centre + u * (su * half_width) +
                                          v * (sv * half_width);
            const auto pixel = project_view(view, point);
            if (!pixel || !view.contains_pixel(*pixel)) {
                return patch;
            }
            patch.pixels.push_back(*pixel);
            patch.values.push_back(view.sample(*pixel));
        }
    }
    double sum = 0.0;
    for (const float value : patch.values) {
        sum += static_cast<double>(value);
    }
    patch.mean = sum / static_cast<double>(patch.values.size());
    double variance = 0.0;
    for (const float value : patch.values) {
        const double difference = static_cast<double>(value) - patch.mean;
        variance += difference * difference;
    }
    patch.stddev = std::sqrt(variance / static_cast<double>(patch.values.size()));
    patch.valid = true;
    return patch;
}

/// Zero-mean normalised cross-correlation (Lewis 1995). Both patches must carry
/// enough energy; a flat patch is "no evidence", not a perfect match.
std::optional<double> patch_ncc(const Patch& a, const Patch& b) {
    if (!a.valid || !b.valid || a.values.size() != b.values.size() ||
        a.stddev < kMinPatchStdDev || b.stddev < kMinPatchStdDev) {
        return std::nullopt;
    }
    double covariance = 0.0;
    for (std::size_t i = 0; i < a.values.size(); ++i) {
        covariance += (static_cast<double>(a.values[i]) - a.mean) *
                      (static_cast<double>(b.values[i]) - b.mean);
    }
    const double n = static_cast<double>(a.values.size());
    const double correlation = covariance / (n * a.stddev * b.stddev);
    return std::clamp(correlation, -1.0, 1.0);
}

/// Candidate offsets along the reference ray. The step is proportional to the
/// physical target radius, so the search runs in scale-normalised units
/// (Lindeberg 1998, scale-space). The range comes from robust quantiles of
/// nearby cloud points' depths; a sparse cloud falls back to a symmetric
/// +/- search_steps window.
std::vector<double> depth_candidates(const Eigen::Vector3d& point,
                                     const Eigen::Vector3d& ray,
                                     const CloudSamples& cloud,
                                     double physical_radius, int search_steps) {
    const double step = std::max(kDepthStepFraction * physical_radius, 1e-9);
    const double reach = std::max(8.0 * physical_radius, 0.3);
    if (!point.allFinite() || !ray.allFinite() || !std::isfinite(step)) {
        return {};
    }
    std::vector<double> offsets;
    for_each_cloud_point(cloud, [&](const Eigen::Vector3d& sample) {
        if (!sample.allFinite()) {
            return;
        }
        const Eigen::Vector3d difference = sample - point;
        const double along = difference.dot(ray);
        const double perpendicular = (difference - along * ray).norm();
        if (std::isfinite(along) && std::isfinite(perpendicular) &&
            perpendicular <= reach) {
            offsets.push_back(along);
        }
    });
    const int maximum = std::max(5, 2 * std::max(1, search_steps) + 1);
    std::vector<double> candidates;
    if (offsets.size() >= static_cast<std::size_t>(kMinPlanePoints)) {
        std::sort(offsets.begin(), offsets.end());
        const std::size_t last = offsets.size() - 1;
        const double low = offsets[last / 10];
        const double high = offsets[(last * 9) / 10];
        const double spread = high - low;
        double from = low - 0.5 * spread;
        double to = high + 0.5 * spread;
        if (!std::isfinite(from) || !std::isfinite(to)) {
            from = -static_cast<double>(maximum) * step;
            to = static_cast<double>(maximum) * step;
        }
        if (to - from < step) {
            from -= step;
            to += step;
        }
        const double count_d = std::ceil((to - from) / step) + 1.0;
        const int count = std::clamp(
            std::isfinite(count_d) ? static_cast<int>(std::min(
                count_d, static_cast<double>(maximum)))
                                   : maximum,
            5, maximum);
        for (int i = 0; i < count; ++i) {
            candidates.push_back(from + (to - from) *
                                            static_cast<double>(i) /
                                            static_cast<double>(count - 1));
        }
        return candidates;
    }
    const int steps = std::max(1, search_steps);
    for (int i = -steps; i <= steps; ++i) {
        candidates.push_back(static_cast<double>(i) * step);
    }
    return candidates;
}

/// Hessian determinant of the reduced image at an integer pixel index. A
/// checker cross has a negative determinant at its saddle; a disc or ring does
/// not. Edge pixels return 0.
double hessian_determinant_at(const CameraView& view, int x, int y) {
    if (x < 1 || y < 1 || x + 1 >= view.image.width() ||
        y + 1 >= view.image.height()) {
        return 0.0;
    }
    const float* data = view.image.pixels().data();
    const int width = view.image.width();
    const auto at = [&](int px, int py) {
        return static_cast<double>(
            data[static_cast<std::size_t>(py) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(px)]);
    };
    const double ixx = at(x + 1, y) - 2.0 * at(x, y) + at(x - 1, y);
    const double iyy = at(x, y + 1) - 2.0 * at(x, y) + at(x, y - 1);
    const double ixy = (at(x + 1, y + 1) - at(x + 1, y - 1) -
                        at(x - 1, y + 1) + at(x - 1, y - 1)) / 4.0;
    return ixx * iyy - ixy * ixy;
}

/// Strongest saddle within a small window around `pixel`, in full-resolution
/// pixels. The candidate reprojection can be a few pixels off the printed
/// centre, and the saddle quadratic fit only converges from inside the
/// saddle's basin, so finding the basin first makes the cross refiner robust.
std::optional<Eigen::Vector2d> saddle_seed(const CameraView& view,
                                           const Eigen::Vector2d& pixel) {
    if (!view.loaded()) {
        return std::nullopt;
    }
    const Eigen::Vector2d local = view.local_of(pixel);
    if (!local.allFinite()) {
        return std::nullopt;
    }
    const double max_x = static_cast<double>(std::max(0, view.image.width() - 1));
    const double max_y = static_cast<double>(std::max(0, view.image.height() - 1));
    const int centre_x = static_cast<int>(
        std::lround(std::clamp(local.x(), 0.0, max_x)));
    const int centre_y = static_cast<int>(
        std::lround(std::clamp(local.y(), 0.0, max_y)));
    // Small scan in working-scale pixels: the candidate is a few pixels off at
    // most, and a wider scan would start locking onto unrelated texture.
    constexpr int kScanPx = 6;
    const int scan = kScanPx;
    double best = 0.0;
    int best_x = 0;
    int best_y = 0;
    bool found = false;
    for (int dy = -scan; dy <= scan; ++dy) {
        for (int dx = -scan; dx <= scan; ++dx) {
            const int x = centre_x + dx;
            const int y = centre_y + dy;
            const double determinant = hessian_determinant_at(view, x, y);
            if (determinant < best) {
                best = determinant;
                best_x = x;
                best_y = y;
                found = true;
            }
        }
    }
    if (!found) {
        return std::nullopt;
    }
    const double ds = static_cast<double>(view.downscale);
    return Eigen::Vector2d(
        (static_cast<double>(best_x + view.offset_x) + 0.5) * ds,
        (static_cast<double>(best_y + view.offset_y) + 0.5) * ds);
}

/// 2-D-only refinement for a marker with fewer than two usable views. Both
/// target-specific refiners run; whichever converges with the smaller movement
/// from the seed wins, because the correct refiner locks onto the printed
/// centre in a pixel or two while the wrong one lands on a quadrant or a
/// neighbouring blob far away. An update that would move further than the
/// limit is rejected, which is what keeps a blank patch from wandering onto
/// unrelated texture.
std::optional<Eigen::Vector2d> refined_2d_pixel(
    const CameraView& view, const Eigen::Vector2d& pixel,
    const std::optional<BlobSeed>& seed, bool cross, double hint_work,
    double limit) {
    if (!view.loaded()) {
        return std::nullopt;
    }
    const auto accept = [&](const Eigen::Vector2d& candidate)
        -> std::optional<Eigen::Vector2d> {
        if (!candidate.allFinite() || (candidate - pixel).norm() > limit) {
            return std::nullopt;
        }
        return candidate;
    };
    const double ds = static_cast<double>(view.downscale);
    const Eigen::Vector2d offset(static_cast<double>(view.offset_x),
                                 static_cast<double>(view.offset_y));
    const auto to_pixel = [&](const Eigen::Vector2d& local_calibration) {
        return (local_calibration + offset) * ds;
    };
    const Eigen::Vector2d local_calibration = pixel / ds - offset;
    // Crosses are refined by the saddle/bar refiner using the extent reported
    // by the cross detector, never by the circular blob scale.
    if (cross) {
        Eigen::Vector2d start = local_calibration;
        if (const auto saddle = saddle_seed(view, pixel)) {
            start = *saddle / ds - offset;
        }
        const double hint = std::max(4.0, hint_work);
        if (const auto refined =
                refine_cross_center(view.image, start, hint)) {
            if (const auto accepted = accept(to_pixel(*refined))) {
                return accepted;
            }
        }
        if (const auto refined =
                refine_bar_cross_center(view.image, start, hint)) {
            return accept(to_pixel(*refined));
        }
        return std::nullopt;
    }
    // Circular targets: refit the ellipse at the candidate, using the blob
    // only for its scale. Trusting the blob's centre would accept a nearby
    // texture blob; the printed target is centred at the candidate to within a
    // few pixels, so the radial search converges to the real boundary.
    if (seed) {
        MarkerCandidate ellipse_seed;
        ellipse_seed.center = local_calibration;
        ellipse_seed.radius_px = seed->radius / ds;
        if (const auto fitted =
                refine_ellipse_center(view.image, ellipse_seed)) {
            if (const auto accepted = accept(to_pixel(*fitted))) {
                return accepted;
            }
        }
    }
    // No blob was found: try the cross refiner from the candidate itself.
    const double hint = std::max(4.0, hint_work);
    if (const auto refined =
            refine_cross_center(view.image, local_calibration, hint)) {
        return accept(to_pixel(*refined));
    }
    return std::nullopt;
}

/// Lucas-Kanade translation-only alignment of a view's patch to the mean
/// inlier template (Baker & Matthews 2004). Returns the image shift in
/// full-resolution pixels, or nullopt when a sample leaves the image or the
/// normal matrix is singular.
std::optional<Eigen::Vector2d> lk_shift(const CameraView& view,
                                        const Patch& patch,
                                        const std::vector<double>& mean,
                                        int max_iterations,
                                        double tolerance_px) {
    Eigen::Vector2d shift = Eigen::Vector2d::Zero();
    const int iterations = std::max(1, max_iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        Eigen::Matrix2d normal = Eigen::Matrix2d::Zero();
        Eigen::Vector2d gradient = Eigen::Vector2d::Zero();
        for (std::size_t i = 0; i < patch.pixels.size(); ++i) {
            const Eigen::Vector2d pixel = patch.pixels[i] + shift;
            if (!view.contains_pixel(pixel)) {
                return std::nullopt;
            }
            const Eigen::Vector2d jacobian = view.gradient(pixel);
            const double residual =
                static_cast<double>(view.sample(pixel)) - mean[i];
            normal += jacobian * jacobian.transpose();
            gradient += jacobian * residual;
        }
        if (!normal.allFinite() || std::abs(normal.determinant()) < 1e-12) {
            return std::nullopt;
        }
        const Eigen::Vector2d step = -normal.ldlt().solve(gradient);
        if (!step.allFinite()) {
            return std::nullopt;
        }
        shift += step;
        if (step.norm() < std::max(tolerance_px, 1e-9)) {
            break;
        }
    }
    return shift;
}

struct ProjectionEdit {
    std::int32_t camera_id = -1;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
};

struct MarkerEdit {
    std::int32_t marker_id = -1;
    std::vector<ProjectionEdit> moves;
    std::vector<std::int32_t> removals;
    std::optional<Eigen::Vector3d> position;
    bool failed = false;
    bool touched = false;
    bool skipped = false;
    int rounds = 0;
};

/// One view's data for a marker: its measurement, apparent scale and depth.
struct BlendView {
    const CameraView* view = nullptr;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    std::optional<BlobSeed> seed;
    bool cross = false;
    bool has_scale = false;
    /// Absolute subpixel centre of the printed target in this view, found once
    /// from the measurement. The rounds move towards it; they never re-derive
    /// it from the current 3-D point, which is what lets a common-mode lateral
    /// error be removed.
    bool has_anchor = false;
    Eigen::Vector2d anchor = Eigen::Vector2d::Zero();
    /// Apparent radius in full-resolution pixels.
    double apparent = 1.0;
    /// Cross-refiner radius hint in working-scale pixels.
    double hint_work = 8.0;
    double depth = 1.0;
};

struct CrossSeed {
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();  ///< full-resolution
    double radius = 0.0;                              ///< full-resolution
    bool bar = false;                                 ///< bar style, not saddle
};

/// Locate the cross near `pixel` with the cross detector's own extent, in
/// full-resolution pixels. Nullopt when no cross is detected near the
/// measurement, in which case the caller falls back to the projected physical
/// radius.
std::optional<CrossSeed> cross_extent(const CameraView& view,
                                      const Eigen::Vector2d& pixel,
                                      int patch_size) {
    if (!view.loaded()) {
        return std::nullopt;
    }
    const Eigen::Vector2d local = view.local_of(pixel);
    if (!local.allFinite()) {
        return std::nullopt;
    }
    const int half = std::max(48, 2 * patch_size);
    const int x0 = std::clamp(
        static_cast<int>(std::floor(std::clamp(local.x(), -1.0e6, 1.0e6))) - half,
        0, std::max(0, view.image.width() - 2 * half - 1));
    const int y0 = std::clamp(
        static_cast<int>(std::floor(std::clamp(local.y(), -1.0e6, 1.0e6))) - half,
        0, std::max(0, view.image.height() - 2 * half - 1));
    const int width = std::min(2 * half + 1, view.image.width() - x0);
    const int height = std::min(2 * half + 1, view.image.height() - y0);
    if (width < 16 || height < 16) {
        return std::nullopt;
    }
    FloatImage crop(width, height, 0.0F);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            crop.at(x, y) = view.image.at(x0 + x, y0 + y);
        }
    }
    CrossParams params;
    params.minimum_size = 2.0;
    params.minimum_dist = 8.0;
    const std::vector<CrossDetection> detections =
        detect_crosses_styled(crop, params);
    const Eigen::Vector2d seed(local.x() - static_cast<double>(x0) + 0.5,
                               local.y() - static_cast<double>(y0) + 0.5);
    double best = std::numeric_limits<double>::infinity();
    double radius = 0.0;
    Eigen::Vector2d centre = Eigen::Vector2d::Zero();
    bool bar = false;
    for (const CrossDetection& detection : detections) {
        if (!detection.target.center.allFinite() ||
            !(detection.target.radius_px > 0.0)) {
            continue;
        }
        const double distance = (detection.target.center - seed).norm();
        if (distance < best) {
            best = distance;
            radius = detection.target.radius_px;
            centre = detection.target.center;
            bar = detection.style == CrossStyle::Bar;
        }
    }
    if (!std::isfinite(best) || !(radius > 0.0) ||
        best > static_cast<double>(half) || !centre.allFinite()) {
        return std::nullopt;
    }
    CrossSeed out;
    const double ds = static_cast<double>(view.downscale);
    out.pixel = (centre + Eigen::Vector2d(
                              static_cast<double>(view.offset_x + x0),
                              static_cast<double>(view.offset_y + y0))) *
                ds;
    out.radius = radius * ds;
    out.bar = bar;
    return out;
}

/// Mean NCC of every other view against the reference patch. Views without
/// evidence (no overlap or a flat patch) do not contribute; if none has
/// evidence the candidate cannot be scored at all.
std::optional<double> score_patches(const std::vector<Patch>& patches,
                                    std::size_t reference,
                                    std::vector<std::optional<double>>& scores) {
    scores.assign(patches.size(), std::nullopt);
    double sum = 0.0;
    int count = 0;
    for (std::size_t i = 0; i < patches.size(); ++i) {
        if (i == reference) {
            continue;
        }
        const auto correlation = patch_ncc(patches[reference], patches[i]);
        if (correlation) {
            scores[i] = correlation;
            sum += *correlation;
            ++count;
        }
    }
    if (count == 0) {
        return std::nullopt;
    }
    return sum / static_cast<double>(count);
}

/// Per-view outcome of a candidate: 0 = non-contributor (the patch leaves the
/// window, so the projection is not used but is not contradicted either),
/// 1 = the patch is in the window but its projection is outside the gate,
/// 2 = the patch is usable and its projection agrees with the candidate.
enum class ViewStatus : std::uint8_t { Unused = 0, Gated = 1, Usable = 2 };

struct Winner {
    bool found = false;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    std::size_t reference = 0;
    double score = -2.0;
    std::vector<Patch> patches;
    std::vector<std::optional<double>> scores;
    std::vector<ViewStatus> status;
};

/// A view's measured projection has to agree with the candidate's reprojection
/// within one apparent target radius, and never less than 3 full-resolution
/// pixels (scaled by the working downscale). This is the only signal that can
/// tell a contradicted measurement from a merely noisy one, because NCC itself
/// always sees the printed target. The floor absorbs the reference view's own
/// error and a few pixels of perturbation; the target scale widens it for a
/// large target, where a larger displacement is still a plausible measurement
/// of the same printed target. A projection displaced by more than about one
/// radius is removed, never repaired.
double projection_tolerance(const BlendView& blend) {
    const double floor_px =
        3.0 * static_cast<double>(std::max(1, blend.view->downscale));
    return std::max(floor_px, blend.apparent);
}

/// Search depth for the best (candidate, reference view) pair. Up to three
/// reference views are tried, largest apparent target first, because a larger
/// target carries more texture and less quantisation noise.
std::optional<Winner> search_winner(const Eigen::Vector3d& initial,
                                    const Eigen::Vector3d& u,
                                    const Eigen::Vector3d& v,
                                    const std::vector<BlendView>& blends,
                                    const CloudSamples& cloud,
                                    double physical_radius, int search_steps,
                                    int patch_size) {
    std::vector<std::size_t> references;
    references.reserve(blends.size());
    for (std::size_t i = 0; i < blends.size(); ++i) {
        const auto projected = project_view(*blends[i].view, initial);
        if (projected &&
            (*projected - blends[i].pixel).norm() <=
                projection_tolerance(blends[i])) {
            references.push_back(i);
        }
    }
    std::sort(references.begin(), references.end(),
              [&blends](std::size_t a, std::size_t b) {
                  return blends[a].apparent > blends[b].apparent;
              });
    const std::size_t tries = std::min<std::size_t>(3, references.size());
    Winner best;
    const double half_width = kPatchHalfWidth * physical_radius;
    for (std::size_t attempt = 0; attempt < tries; ++attempt) {
        const std::size_t reference = references[attempt];
        const Eigen::Vector3d centre =
            camera_centre(*blends[reference].view->camera);
        const Eigen::Vector3d ray = initial - centre;
        if (ray.norm() <= 1e-12) {
            continue;
        }
        const Eigen::Vector3d ray_unit = ray.normalized();
        const std::vector<double> offsets = depth_candidates(
            initial, ray_unit, cloud, physical_radius, search_steps);
        for (const double offset : offsets) {
            const Eigen::Vector3d candidate = initial + ray_unit * offset;
            if (!candidate.allFinite()) {
                continue;
            }
            std::vector<Patch> patches;
            std::vector<ViewStatus> status;
            patches.reserve(blends.size());
            status.reserve(blends.size());
            int usable_other = 0;
            for (std::size_t index = 0; index < blends.size(); ++index) {
                const BlendView& blend = blends[index];
                Patch patch = sample_patch(*blend.view, candidate, u, v,
                                           half_width, patch_size);
                if (!patch.valid) {
                    // The patch leaves this view's window: a non-contributor,
                    // not a contradiction. It must not invalidate the
                    // candidate, which only needs the reference view and at
                    // least one other usable view.
                    patches.push_back(Patch{});
                    status.push_back(ViewStatus::Unused);
                    continue;
                }
                // NCC alone cannot detect a contradicted projection: every
                // view still shows the printed target at the candidate's
                // physical patch. A view therefore also has to agree with the
                // candidate geometrically, within one target radius of its
                // measured projection. An unrelated 30 px measurement lands
                // outside this gate and is excluded from the score.
                const auto projected = project_view(*blend.view, candidate);
                if (!projected || (*projected - blend.pixel).norm() >
                                      projection_tolerance(blend)) {
                    patches.push_back(Patch{});
                    status.push_back(ViewStatus::Gated);
                    continue;
                }
                patches.push_back(std::move(patch));
                status.push_back(ViewStatus::Usable);
                if (index != reference) {
                    ++usable_other;
                }
            }
            if (status[reference] != ViewStatus::Usable ||
                patches[reference].stddev < kMinPatchStdDev ||
                usable_other < 1) {
                continue;
            }
            std::vector<std::optional<double>> scores;
            const auto score = score_patches(patches, reference, scores);
            if (!score || *score <= best.score) {
                continue;
            }
            best.found = true;
            best.point = candidate;
            best.reference = reference;
            best.score = *score;
            best.patches = std::move(patches);
            best.scores = std::move(scores);
            best.status = std::move(status);
        }
    }
    if (!best.found) {
        return std::nullopt;
    }
    return best;
}

/// Intersect the viewing ray through `pixel` with the target plane (through
/// `plane_point`, normal `normal`). Used to anchor each view's template at its
/// own 2-D-refined centre instead of the candidate's common offset.
std::optional<Eigen::Vector3d> intersect_plane(const CameraView& view,
                                               const Eigen::Vector2d& pixel,
                                               const Eigen::Vector3d& plane_point,
                                               const Eigen::Vector3d& normal) {
    const Eigen::Matrix3d rotation = view.camera->pose.block<3, 3>(0, 0);
    const Eigen::Vector3d ray = rotation.transpose() *
                                view.calibration->unproject(pixel);
    const double denominator = ray.dot(normal);
    if (std::abs(denominator) < 1e-9) {
        return std::nullopt;
    }
    const Eigen::Vector3d centre = camera_centre(*view.camera);
    const double distance = (plane_point - centre).dot(normal) / denominator;
    if (!std::isfinite(distance) || distance <= 0.0) {
        return std::nullopt;
    }
    return centre + distance * ray;
}

/// Result of one refinement round: the accepted projection of every view
/// (updated for inliers, unchanged otherwise) plus the view statuses of the
/// winning candidate.
struct RoundOutcome {
    std::vector<Eigen::Vector2d> pixels;
    std::vector<bool> inlier;
    std::vector<ViewStatus> status;
    double max_movement = 0.0;
};

/// Consensus (Fischler & Bolles 1981) at the winning candidate, then subpixel
/// refinement of every inlier. A view is an inlier when its correlation is
/// within `correlation_margin` of the best non-reference view and above
/// `min_correlation`; two inliers are required. Each inlier is anchored at the
/// printed centre by the type-specific 2-D refiner, and the movement accepted
/// in this round is clamped relative to this round's anchor.
std::optional<RoundOutcome> apply_winner(
    const std::vector<BlendView>& blends, const Winner& winner,
    const std::vector<Eigen::Vector2d>& current, const Eigen::Vector3d& u,
    const Eigen::Vector3d& v, double half_width,
    const RefineMarkersOptions& options) {
    const std::vector<std::optional<double>>& scores = winner.scores;
    double best_other = -2.0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (i != winner.reference && scores[i]) {
            best_other = std::max(best_other, *scores[i]);
        }
    }
    std::vector<std::size_t> inliers;
    inliers.push_back(winner.reference);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (i == winner.reference || !scores[i]) {
            continue;
        }
        if (*scores[i] >= options.min_correlation &&
            *scores[i] >= best_other - options.correlation_margin) {
            inliers.push_back(i);
        }
    }
    if (inliers.size() < 2) {
        return std::nullopt;
    }
    // A common lateral offset of the physical patch is invisible to NCC (every
    // view samples the same world points), so consistency alone cannot remove
    // the reference view's own bias. Each inlier is therefore anchored at the
    // centre of its printed target by the type-specific 2-D refiner, and the
    // Lucas-Kanade step then aligns those anchored patches to their mean. This
    // keeps the multi-view gate while giving the template an absolute anchor.
    RoundOutcome out;
    out.pixels = current;
    out.inlier.assign(blends.size(), false);
    out.status = winner.status;
    const Eigen::Vector3d normal = u.cross(v).normalized();
    std::vector<Eigen::Vector2d> anchored(blends.size());
    std::vector<Eigen::Vector2d> projected_centres(blends.size());
    std::vector<Patch> anchored_patches(blends.size());
    std::vector<bool> usable(blends.size(), false);
    int projected_inliers = 0;
    for (const std::size_t index : inliers) {
        const BlendView& blend = blends[index];
        const auto projected = project_view(*blend.view, winner.point);
        if (!projected || !projected->allFinite()) {
            continue;
        }
        out.inlier[index] = true;
        ++projected_inliers;
        projected_centres[index] = *projected;
        const double limit = std::max(2.0, blend.apparent);
        Eigen::Vector2d anchor = *projected;
        if (blend.has_anchor && blend.anchor.allFinite()) {
            Eigen::Vector2d delta = blend.anchor - *projected;
            const double distance = delta.norm();
            if (std::isfinite(distance) && distance > limit) {
                delta *= limit / distance;
            }
            anchor = *projected + delta;
        } else {
            std::optional<BlobSeed> seed;
            if (!blend.cross) {
                seed = circular_seed(*blend.view, *projected,
                                     options.patch_size);
            }
            const auto refined =
                refined_2d_pixel(*blend.view, *projected, seed, blend.cross,
                                 blend.hint_work, limit);
            anchor = refined ? *refined : *projected;
        }
        anchored[index] = anchor;
        const auto point = intersect_plane(*blend.view, anchored[index],
                                           winner.point, normal);
        if (point) {
            anchored_patches[index] = sample_patch(
                *blend.view, *point, u, v, half_width, options.patch_size);
            usable[index] = anchored_patches[index].valid;
        }
    }
    if (projected_inliers < 2) {
        return std::nullopt;
    }
    std::vector<double> mean(static_cast<std::size_t>(options.patch_size) *
                                 static_cast<std::size_t>(options.patch_size),
                             0.0);
    int mean_count = 0;
    for (const std::size_t index : inliers) {
        if (!usable[index]) {
            continue;
        }
        const std::vector<float>& values = anchored_patches[index].values;
        if (mean.size() != values.size()) {
            mean.assign(values.size(), 0.0);
        }
        for (std::size_t k = 0; k < mean.size(); ++k) {
            mean[k] += static_cast<double>(values[k]);
        }
        ++mean_count;
    }
    if (mean_count > 0) {
        for (double& value : mean) {
            value /= static_cast<double>(mean_count);
        }
    }
    for (std::size_t i = 0; i < blends.size(); ++i) {
        if (!out.inlier[i]) {
            continue;
        }
        const BlendView& blend = blends[i];
        Eigen::Vector2d refined = anchored[i];
        const double limit = std::max(2.0, blend.apparent);
        if (mean_count > 0 && usable[i]) {
            const auto shift =
                lk_shift(*blend.view, anchored_patches[i], mean,
                         options.max_iterations, options.tolerance_px);
            if (shift &&
                (refined + *shift - projected_centres[i]).norm() <= limit) {
                refined += *shift;
            }
        }
        if (!refined.allFinite()) {
            refined = projected_centres[i];
        }
        out.pixels[i] = refined;
        out.max_movement =
            std::max(out.max_movement, (refined - current[i]).norm());
    }
    return out;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle),
                     values.end());
    return values[middle];
}

/// Detect the target type and fill the apparent radius, in full-resolution
/// pixels. A saddle at the measurement means a cross: its extent comes from the
/// cross detector, and the circular blob scale is never used for it. Circular
/// targets use the blob's semi-major axis. A view whose radius cannot be
/// measured is left for the projected physical radius, filled once the marker
/// has been triangulated.
void fill_scales(std::vector<BlendView>& blends, int patch_size) {
    for (BlendView& blend : blends) {
        if (!blend.view->loaded()) {
            continue;
        }
        const double ds = static_cast<double>(std::max(1, blend.view->downscale));
        if (const auto cross =
                cross_extent(*blend.view, blend.pixel, patch_size)) {
            blend.cross = true;
            blend.apparent = std::max(1.0, cross->radius);
            blend.has_scale = true;
            // The saddle fit only needs to search the basin around the
            // measured centre; a quarter of the arm half-length is the
            // validated working-scale radius.
            blend.hint_work = std::max(4.0, 0.25 * blend.apparent / ds);
            const Eigen::Vector2d offset(
                static_cast<double>(blend.view->offset_x),
                static_cast<double>(blend.view->offset_y));
            Eigen::Vector2d start =
                blend.view->local_of(cross->pixel) +
                Eigen::Vector2d(0.5, 0.5);
            if (!cross->bar) {
                // Prefer the local saddle scan when it finds one: starting the
                // quadratic fit from the strongest nearby saddle is what the
                // 2-D refiner was validated with.
                if (const auto saddle =
                        saddle_seed(*blend.view, blend.pixel)) {
                    start = blend.view->local_of(*saddle) +
                            Eigen::Vector2d(0.5, 0.5);
                }
            }
            blend.anchor = cross->pixel;
            if (const auto refined =
                    cross->bar
                        ? refine_bar_cross_center(blend.view->image, start,
                                                  blend.hint_work)
                        : refine_cross_center(blend.view->image, start,
                                              blend.hint_work)) {
                blend.anchor = (*refined + offset) * ds;
            }
            blend.has_anchor = true;
            continue;
        }
        if (const auto circle =
                circular_seed(*blend.view, blend.pixel, patch_size)) {
            blend.seed = circle;
            blend.apparent = std::max(1.0, circle->radius);
            blend.has_scale = true;
            blend.anchor = circle->pixel;
            blend.has_anchor = true;
            continue;
        }
        blend.hint_work = 8.0;
    }
}

MarkerEdit refine_2d_marker(std::vector<BlendView>& blends) {
    MarkerEdit edit;
    bool any = false;
    bool success = false;
    for (BlendView& blend : blends) {
        any = true;
        const double limit = std::max(2.0, blend.apparent);
        const auto refined = refined_2d_pixel(
            *blend.view, blend.pixel, blend.seed, blend.cross, blend.hint_work,
            limit);
        if (!refined) {
            continue;
        }
        success = true;
        if ((*refined - blend.pixel).norm() > 1e-9) {
            edit.moves.push_back({blend.view->camera->id, *refined});
            edit.touched = true;
        }
    }
    if (any && !success) {
        edit.failed = true;
    }
    return edit;
}

/// Turn the last round into edits. Out-of-window views (`Unused`) are left
/// untouched: they were non-contributors, not contradictions. Every other
/// non-inlier is removed when `remove_invalid` is set. The marker's estimated
/// position is re-triangulated from the accepted refined projections; the
/// reference/source coordinates are never touched.
MarkerEdit finalize_round(const std::vector<BlendView>& aligned,
                          const RoundOutcome& outcome,
                          const RefineMarkersOptions& options) {
    MarkerEdit edit;
    for (std::size_t i = 0; i < aligned.size(); ++i) {
        const BlendView& blend = aligned[i];
        if (outcome.inlier[i]) {
            if ((outcome.pixels[i] - blend.pixel).norm() > 1e-9) {
                edit.moves.push_back(
                    {blend.view->camera->id, outcome.pixels[i]});
                edit.touched = true;
            }
        } else if (outcome.status[i] != ViewStatus::Unused &&
                   options.remove_invalid) {
            edit.removals.push_back(blend.view->camera->id);
            edit.touched = true;
        }
    }
    std::vector<Observation> accepted;
    std::vector<double> tolerances;
    for (std::size_t i = 0; i < aligned.size(); ++i) {
        if (!outcome.inlier[i]) {
            continue;
        }
        accepted.push_back({aligned[i].view, outcome.pixels[i]});
        tolerances.push_back(projection_tolerance(aligned[i]));
    }
    if (accepted.size() >= 2) {
        if (const auto position = triangulate_robust(accepted, tolerances)) {
            edit.position = *position;
        }
    }
    return edit;
}

/// Decide one marker: multi-view consensus when two aligned views exist, a
/// 2-D fallback otherwise. A multi-view attempt that cannot triangulate or
/// agree is "failed" and leaves every projection untouched; it deliberately
/// does not degrade to the per-view refiner, which is exactly the
/// contradictory-projection case the consensus step exists to catch.
MarkerEdit refine_marker(std::vector<BlendView> blends,
                         const CloudSamples& cloud,
                         const RefineMarkersOptions& options) {
    if (blends.empty()) {
        MarkerEdit skipped;
        skipped.skipped = true;
        return skipped;
    }
    fill_scales(blends, options.patch_size);
    std::vector<BlendView> aligned;
    for (const BlendView& blend : blends) {
        if (blend.view->aligned()) {
            aligned.push_back(blend);
        }
    }
    if (aligned.size() >= 2) {
        std::vector<Observation> observations;
        observations.reserve(aligned.size());
        std::vector<double> tolerances;
        tolerances.reserve(aligned.size());
        std::vector<Eigen::Vector2d> current;
        current.reserve(aligned.size());
        for (const BlendView& blend : aligned) {
            observations.push_back({blend.view, blend.pixel});
            const double ds = static_cast<double>(
                std::max(1, blend.view->downscale));
            tolerances.push_back(blend.has_scale ? projection_tolerance(blend)
                                                 : 8.0 * ds);
            current.push_back(blend.pixel);
        }
        auto initial = triangulate_robust(observations, tolerances);
        if (initial) {
            std::vector<double> physical_samples;
            std::vector<double> depths;
            for (BlendView& blend : aligned) {
                blend.depth = depth_in_view(*blend.view, *initial);
                depths.push_back(blend.depth);
                if (!blend.has_scale) {
                    continue;
                }
                const double focal =
                    std::max(blend.view->calibration->f, 1e-6);
                physical_samples.push_back(blend.apparent * blend.depth / focal);
            }
            double physical = median(physical_samples);
            if (!(physical > 0.0) || !std::isfinite(physical)) {
                // No view reported a radius: assume the printed target spans
                // 10 % of the viewing distance (5 % radius). This is a physical
                // length, never a sample count.
                physical = 0.05 * std::max(median(depths), 1e-9);
            }
            // Views without a measured radius use the projected physical radius.
            for (BlendView& blend : aligned) {
                if (blend.has_scale) {
                    continue;
                }
                const double focal =
                    std::max(blend.view->calibration->f, 1e-6);
                blend.apparent = std::max(
                    1.0, physical * focal / std::max(blend.depth, 1e-9));
                blend.hint_work = std::max(
                    4.0, 0.25 * blend.apparent /
                             static_cast<double>(std::max(
                                 1, blend.view->downscale)));
                blend.has_scale = true;
            }
            for (std::size_t i = 0; i < aligned.size(); ++i) {
                tolerances[i] = projection_tolerance(aligned[i]);
            }
            initial = triangulate_robust(observations, tolerances);
            if (initial) {
                const Eigen::Vector3d normal = target_normal(
                    *initial, observations, cloud, std::max(0.15, 6.0 * physical));
                const auto axes = plane_axes(normal);
                Eigen::Vector3d point = *initial;
                std::optional<RoundOutcome> last;
                int rounds_used = 0;
                // Up to three rounds: the depth search and LK are repeated
                // around the point re-triangulated from the previous round's
                // refined centres, which removes a common-mode lateral bias
                // that no single view can see.
                for (int round = 0; round < 3; ++round) {
                    const auto winner = search_winner(
                        point, axes.first, axes.second, aligned, cloud, physical,
                        options.search_steps, options.patch_size);
                    if (!winner) {
                        break;
                    }
                    auto outcome = apply_winner(
                        aligned, *winner, current, axes.first, axes.second,
                        kPatchHalfWidth * physical, options);
                    if (!outcome) {
                        break;
                    }
                    last = std::move(outcome);
                    ++rounds_used;
                    std::vector<Observation> refined;
                    std::vector<double> refined_tolerances;
                    for (std::size_t i = 0; i < aligned.size(); ++i) {
                        if (!last->inlier[i]) {
                            continue;
                        }
                        refined.push_back({aligned[i].view, last->pixels[i]});
                        refined_tolerances.push_back(
                            projection_tolerance(aligned[i]));
                    }
                    current = last->pixels;
                    if (refined.size() < 2) {
                        break;
                    }
                    const auto next =
                        triangulate_robust(refined, refined_tolerances);
                    if (!next) {
                        break;
                    }
                    point = *next;
                    const double stop =
                        std::max(options.tolerance_px, kRoundStopNoisePx);
                    if (last->max_movement < stop) {
                        break;
                    }
                }
                if (last) {
                    MarkerEdit edit = finalize_round(aligned, *last, options);
                    edit.rounds = rounds_used;
                    return edit;
                }
            }
        }
        MarkerEdit failed;
        failed.failed = true;
        return failed;
    }
    return refine_2d_marker(blends);
}

/// Build the pose/calibration/scale record for every selected camera. No image
/// is read here. An explicit camera list is strict: a selected camera with no
/// frame or no sensor is a ValueError, matching DetectMarkers; the implicit
/// "all enabled" list silently skips cameras that cannot contribute.
std::vector<CameraView> build_camera_views(const Chunk& chunk,
                                           const std::vector<std::string>& filters,
                                           int downscale) {
    for (const std::string& label : filters) {
        bool found = false;
        for (const Camera& camera : chunk.cameras()) {
            found = found || camera.label == label;
        }
        if (!found) {
            throw ValueError("RefineMarkers: no camera with label '" + label + "'");
        }
    }
    std::vector<CameraView> views;
    for (const Camera& camera : chunk.cameras()) {
        const bool selected =
            filters.empty() ? camera.enabled : contains(filters, camera.label);
        if (!selected) {
            continue;
        }
        const Frame* frame = frame_of_camera(chunk, camera.id);
        if (frame == nullptr) {
            if (!filters.empty()) {
                throw ValueError("RefineMarkers: camera '" + camera.label +
                                 "' has no linked frame");
            }
            continue;
        }
        (void)frame;
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk.sensors().size())) {
            if (!filters.empty()) {
                throw ValueError("RefineMarkers: camera '" + camera.label +
                                 "' has no sensor");
            }
            continue;
        }
        const Calibration& calibration =
            chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration;
        if (calibration.width <= 0 || calibration.height <= 0) {
            continue;
        }
        CameraView view;
        view.camera = &camera;
        view.calibration = &calibration;
        view.downscale = std::max(1, downscale);
        view.full_width = std::max(1, calibration.width / view.downscale);
        view.full_height = std::max(1, calibration.height / view.downscale);
        views.push_back(std::move(view));
    }
    return views;
}
/// One local window to cut for one projection of one selected marker.
struct WindowRequest {
    std::size_t camera = 0;      ///< index into the camera-view list
    std::size_t marker = 0;      ///< index into the selected-marker list
    std::size_t projection = 0;  ///< index into that marker's projections
    Eigen::Vector2d centre = Eigen::Vector2d::Zero();  ///< full-res pixels
    double half_px = 0.0;                              ///< full-res half extent
};

struct WindowLoadResult {
    bool ok = true;
    std::string error;
};

std::optional<std::size_t> view_index(const std::vector<CameraView>& views,
                                      std::int32_t camera_id) {
    for (std::size_t i = 0; i < views.size(); ++i) {
        if (views[i].camera != nullptr && views[i].camera->id == camera_id) {
            return i;
        }
    }
    return std::nullopt;
}

/// Rough physical target radius, used only to bound a window. The nearest tie
/// point gives the local sampling scale; without a cloud, 5 % of the viewing
/// distance is assumed. Never used as an accuracy scale.
double estimate_physical_radius(const Eigen::Vector3d& point,
                                const std::vector<Observation>& observations,
                                const CloudSamples& cloud) {
    double nearest = std::numeric_limits<double>::infinity();
    for_each_cloud_point(cloud, [&](const Eigen::Vector3d& sample) {
        if (!sample.allFinite() || !point.allFinite()) {
            return;
        }
        nearest = std::min(nearest, (sample - point).norm());
    });
    if (std::isfinite(nearest) && nearest > 0.0) {
        return 2.0 * nearest;
    }
    std::vector<double> depths;
    for (const Observation& observation : observations) {
        depths.push_back(depth_in_view(*observation.view, point));
    }
    const double depth = median(depths);
    return 0.05 * (std::isfinite(depth) && depth > 0.0 ? depth : 1.0);
}

struct WindowPlan {
    std::vector<WindowRequest> requests;
    /// [selected marker][projection] -> request index, or npos when the
    /// projection has no usable window.
    std::vector<std::vector<std::size_t>> marker_windows;
};

/// Plan one window per usable projection. Windows are centred on the measured
/// projection and sized from the projected extent of the whole depth-search
/// range plus the patch footprint, capped at 512 working-scale pixels per side
/// (256 half extent). A candidate whose patch leaves the window is treated as
/// out of the image, never as a contradiction.
WindowPlan build_window_requests(const Chunk& chunk,
                                 const std::vector<std::size_t>& selected,
                                 const std::vector<CameraView>& views,
                                 const CloudSamples& cloud,
                                 const RefineMarkersOptions& options) {
    const std::size_t npos = std::numeric_limits<std::size_t>::max();
    // The floor covers the patch footprint and the 2-D refiner's local window
    // for markers that cannot be triangulated (one view, no search extent).
    const double floor_work =
        std::max(96.0, static_cast<double>(options.patch_size) + 16.0);
    WindowPlan plan;
    plan.marker_windows.resize(selected.size());
    for (std::size_t slot = 0; slot < selected.size(); ++slot) {
        const Marker& marker = chunk.markers()[selected[slot]];
        std::vector<std::size_t>& mapping = plan.marker_windows[slot];
        mapping.assign(marker.projections.size(), npos);
        std::vector<std::size_t> chosen;
        std::vector<Observation> observations;
        for (std::size_t p = 0; p < marker.projections.size(); ++p) {
            const MarkerProjection& projection = marker.projections[p];
            if (!projection.enabled || !projection.pixel.allFinite()) {
                continue;
            }
            const auto index = view_index(views, projection.camera_id);
            if (!index) {
                continue;
            }
            chosen.push_back(p);
            observations.push_back({&views[*index], projection.pixel});
        }
        if (chosen.empty()) {
            continue;
        }
        std::optional<Eigen::Vector3d> initial;
        if (observations.size() >= 2) {
            std::vector<double> tolerances;
            for (const Observation& observation : observations) {
                tolerances.push_back(
                    8.0 * static_cast<double>(std::max(
                              1, observation.view->downscale)));
            }
            initial = triangulate_robust(observations, tolerances);
        }
        const double physical =
            initial ? estimate_physical_radius(*initial, observations, cloud)
                    : 0.05;
        for (std::size_t k = 0; k < chosen.size(); ++k) {
            const MarkerProjection& projection = marker.projections[chosen[k]];
            const CameraView& view = *observations[k].view;
            const double ds =
                static_cast<double>(std::max(1, view.downscale));
            double half_work = floor_work;
            if (initial) {
                const Eigen::Vector3d ray =
                    *initial - camera_centre(*view.camera);
                if (ray.allFinite() && ray.norm() > 1e-12) {
                    const Eigen::Vector3d unit = ray.normalized();
                    const auto offsets = depth_candidates(
                        *initial, unit, cloud, physical, options.search_steps);
                    double extent = 0.0;
                    for (const double offset : offsets) {
                        const auto projected =
                            project_view(view, *initial + unit * offset);
                        if (projected && projected->allFinite()) {
                            extent = std::max(
                                extent, (*projected - projection.pixel).norm());
                        }
                    }
                    const double depth = depth_in_view(view, *initial);
                    const double focal =
                        std::max(view.calibration->f, 1e-6);
                    const double patch_radius =
                        physical * focal / std::max(depth, 1e-9);
                    half_work =
                        std::max(floor_work, extent / ds +
                                                 1.5 * patch_radius / ds + 16.0);
                }
            }
            half_work = std::clamp(half_work, floor_work, 256.0);
            const auto camera_index = view_index(views, projection.camera_id);
            if (!camera_index) {
                continue;
            }
            WindowRequest request;
            request.camera = *camera_index;
            request.marker = slot;
            request.projection = chosen[k];
            request.centre = projection.pixel;
            request.half_px = half_work * ds;
            mapping[chosen[k]] = plan.requests.size();
            plan.requests.push_back(request);
        }
    }
    return plan;
}

/// Copy one crop out of a reduced image and compute dI/d(reduced index) there.
void cut_window(const FloatImage& image, int x0, int y0, int width, int height,
                CameraView& window) {
    window.image = FloatImage(width, height, 0.0F);
    window.gradient_x = FloatImage(width, height, 0.0F);
    window.gradient_y = FloatImage(width, height, 0.0F);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            window.image.at(x, y) = image.at(x0 + x, y0 + y);
        }
    }
    for (int y = 0; y < height; ++y) {
        const int ym = std::max(0, y - 1);
        const int yp = std::min(height - 1, y + 1);
        for (int x = 0; x < width; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(width - 1, x + 1);
            const double dx = static_cast<double>(xp - xm);
            const double dy = static_cast<double>(yp - ym);
            window.gradient_x.at(x, y) =
                dx > 0.0 ? static_cast<float>(
                               (static_cast<double>(window.image.at(xp, y)) -
                                static_cast<double>(window.image.at(xm, y))) /
                               dx)
                         : 0.0F;
            window.gradient_y.at(x, y) =
                dy > 0.0 ? static_cast<float>(
                               (static_cast<double>(window.image.at(x, yp)) -
                                static_cast<double>(window.image.at(x, ym))) /
                               dy)
                         : 0.0F;
        }
    }
    window.offset_x = x0;
    window.offset_y = y0;
    window.full_width = image.width();
    window.full_height = image.height();
}

/// Pass 1 for one camera: decode and reduce the image once, cut every requested
/// window and its gradients, then release the image. Only this camera's crops
/// survive.
WindowLoadResult load_camera_windows(
    const Document& document, const Chunk& chunk, std::size_t camera_index,
    const std::vector<CameraView>& geometry,
    const std::vector<std::size_t>& request_indices,
    const std::vector<WindowRequest>& requests,
    std::vector<CameraView>& windows) {
    const CameraView& view = geometry[camera_index];
    const Frame* frame = frame_of_camera(chunk, view.camera->id);
    if (frame == nullptr) {
        return WindowLoadResult{false, "no linked frame"};
    }
    WindowLoadResult result;
    try {
        FloatImage image =
            read_image(resolve_frame_path(document, frame->path()).string())
                .to_luminance();
        if (view.downscale > 1) {
            image = downsample_box(image, view.downscale);
        }
        const double ds = static_cast<double>(std::max(1, view.downscale));
        if (image.width() < 2 || image.height() < 2) {
            return WindowLoadResult{false, "image is empty"};
        }
        const int max_x = image.width() - 1;
        const int max_y = image.height() - 1;
        for (const std::size_t request_index : request_indices) {
            const WindowRequest& request = requests[request_index];
            CameraView& window = windows[request_index];
            const Eigen::Vector2d centre =
                request.centre / ds - Eigen::Vector2d(0.5, 0.5);
            if (!centre.allFinite() || !std::isfinite(request.half_px)) {
                continue;
            }
            const double half = request.half_px / ds;
            const double left = std::clamp(centre.x() - half, -1.0e9, 1.0e9);
            const double right = std::clamp(centre.x() + half, -1.0e9, 1.0e9);
            const double top = std::clamp(centre.y() - half, -1.0e9, 1.0e9);
            const double bottom = std::clamp(centre.y() + half, -1.0e9, 1.0e9);
            const int x0 = std::clamp(static_cast<int>(std::floor(left)), 0,
                                      max_x);
            const int x1 = std::clamp(static_cast<int>(std::ceil(right)), 0,
                                      max_x);
            const int y0 = std::clamp(static_cast<int>(std::floor(top)), 0,
                                      max_y);
            const int y1 = std::clamp(static_cast<int>(std::ceil(bottom)), 0,
                                      max_y);
            if (x1 <= x0 || y1 <= y0) {
                continue;
            }
            cut_window(image, x0, y0, x1 - x0 + 1, y1 - y0 + 1, window);
        }
    } catch (const std::exception& error) {
        result.ok = false;
        result.error = error.what();
    }
    return result;
}

/// Pass 1 over cameras. Cameras run serially when any frame exceeds 8 MP (the
/// DetectMarkers rule), so at most one full-resolution luminance buffer is
/// live. An unreadable image is fatal only for an explicit camera list;
/// otherwise the camera is logged and skipped.
std::vector<CameraView> load_windows(const Document& document,
                                     const Chunk& chunk,
                                     const std::vector<CameraView>& geometry,
                                     const std::vector<WindowRequest>& requests,
                                     bool explicit_cameras,
                                     RefineMarkersStats& stats) {
    std::vector<std::vector<std::size_t>> groups(geometry.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        groups[requests[i].camera].push_back(i);
    }
    std::vector<CameraView> windows(requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        windows[i] = geometry[requests[i].camera];
    }
    bool serial = false;
    std::size_t max_image_bytes = 0;
    for (std::size_t c = 0; c < geometry.size(); ++c) {
        if (groups[c].empty()) {
            continue;
        }
        const CameraView& view = geometry[c];
        const Frame* frame = frame_of_camera(chunk, view.camera->id);
        const std::int64_t ds = std::max(1, view.downscale);
        const std::int64_t width =
            frame != nullptr && frame->width() > 0 ? frame->width()
                                                   : view.full_width * ds;
        const std::int64_t height =
            frame != nullptr && frame->height() > 0 ? frame->height()
                                                    : view.full_height * ds;
        if (width * height > 8ll * 1024ll * 1024ll) {
            serial = true;
        }
        const std::size_t full = static_cast<std::size_t>(width * height);
        const std::size_t reduced = static_cast<std::size_t>(
            (width / ds) * (height / ds));
        const std::size_t bytes =
            full * 3 + full * sizeof(float) +
            (ds > 1 ? reduced * sizeof(float) : 0);
        max_image_bytes = std::max(max_image_bytes, bytes);
    }
    std::vector<WindowLoadResult> results(geometry.size());
    const int count = static_cast<int>(geometry.size());
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) if(!serial)
#endif
    for (int c = 0; c < count; ++c) {
        if (groups[static_cast<std::size_t>(c)].empty()) {
            continue;
        }
        results[static_cast<std::size_t>(c)] = load_camera_windows(
            document, chunk, static_cast<std::size_t>(c), geometry,
            groups[static_cast<std::size_t>(c)], requests, windows);
    }
    for (std::size_t c = 0; c < geometry.size(); ++c) {
        if (results[c].ok) {
            continue;
        }
        const std::string message =
            "RefineMarkers: " +
            std::string(geometry[c].camera != nullptr ? geometry[c].camera->label
                                                      : std::string("camera")) +
            ": " + results[c].error;
        if (explicit_cameras) {
            throw Error(message);
        }
        log(LogLevel::Warn) << message;
    }
    std::size_t window_bytes = 0;
    for (const CameraView& window : windows) {
        window_bytes += window.image.pixels().size() * sizeof(float) * 3;
    }
    int threads = 1;
#if defined(ALPHASHAPE_HAVE_OPENMP)
    threads = std::max(1, omp_get_max_threads());
#endif
    const std::size_t images = serial || count <= 0
                                   ? 1
                                   : std::min<std::size_t>(
                                         static_cast<std::size_t>(threads),
                                         static_cast<std::size_t>(count));
    stats.peak_bytes = window_bytes + max_image_bytes * images;
    return windows;
}

Marker* find_marker_by_id(Chunk& chunk, std::int32_t marker_id) {
    for (Marker& marker : chunk.markers()) {
        if (marker.id == marker_id) {
            return &marker;
        }
    }
    return nullptr;
}

const Marker* find_marker_by_id(const Chunk& chunk, std::int32_t marker_id) {
    for (const Marker& marker : chunk.markers()) {
        if (marker.id == marker_id) {
            return &marker;
        }
    }
    return nullptr;
}

const MarkerProjection* find_projection(const Marker& marker,
                                        std::int32_t camera_id) {
    for (const MarkerProjection& projection : marker.projections) {
        if (projection.camera_id == camera_id) {
            return &projection;
        }
    }
    return nullptr;
}

/// One staged command for the whole run: its snapshot covers the marker
/// container once, so the undo memory is linear in the number of markers, not
/// quadratic in the number of projections. Revert restores that snapshot
/// exactly, including every projection's order.
class RefineMarkersCommand : public StagedMutation {
public:
    RefineMarkersCommand(std::int32_t chunk_id, std::vector<MarkerEdit> edits)
        : chunk_id_(chunk_id), edits_(std::move(edits)) {}

    std::string name() const override { return "Refine Markers"; }

    std::size_t retained_bytes() const override {
        return snapshot_.capacity() * sizeof(Marker);
    }

protected:
    void validate(Document& document) override {
        const Chunk& chunk = target(document);
        for (const MarkerEdit& edit : edits_) {
            if (edit.skipped || edit.failed) {
                continue;
            }
            const Marker* marker = find_marker_by_id(chunk, edit.marker_id);
            if (marker == nullptr) {
                throw ValueError("RefineMarkers: no marker with id " +
                                 std::to_string(edit.marker_id));
            }
            for (const ProjectionEdit& move : edit.moves) {
                if (!move.pixel.allFinite()) {
                    throw ValueError("RefineMarkers: projection pixel is not finite");
                }
                if (find_projection(*marker, move.camera_id) == nullptr) {
                    throw ValueError("RefineMarkers: projection has no matching camera");
                }
            }
            for (const std::int32_t camera_id : edit.removals) {
                if (find_projection(*marker, camera_id) == nullptr) {
                    throw ValueError("RefineMarkers: projection has no matching camera");
                }
            }
            if (edit.position.has_value() && !edit.position->allFinite()) {
                throw ValueError("RefineMarkers: marker position is not finite");
            }
        }
    }

    void stage(Document& document) override {
        snapshot_ = target(document).markers();
    }

    void commit(Document& document) override {
        Chunk& chunk = target(document);
        for (const MarkerEdit& edit : edits_) {
            if (edit.skipped || edit.failed) {
                continue;
            }
            Marker* marker = find_marker_by_id(chunk, edit.marker_id);
            if (marker == nullptr) {
                throw ValueError("RefineMarkers: no marker with id " +
                                 std::to_string(edit.marker_id));
            }
            for (const ProjectionEdit& move : edit.moves) {
                for (MarkerProjection& projection : marker->projections) {
                    if (projection.camera_id == move.camera_id) {
                        projection.pixel = move.pixel;
                        break;
                    }
                }
            }
            for (const std::int32_t camera_id : edit.removals) {
                auto& projections = marker->projections;
                projections.erase(
                    std::remove_if(projections.begin(), projections.end(),
                                   [camera_id](const MarkerProjection& p) {
                                       return p.camera_id == camera_id;
                                   }),
                    projections.end());
            }
            if (edit.position.has_value()) {
                marker->position = edit.position;
            }
        }
    }

    void restore(Document& document) override {
        target(document).markers() = snapshot_;
    }

private:
    Chunk& target(Document& document) const {
        for (auto& workspace : document.workspaces()) {
            if (!workspace) {
                continue;
            }
            if (Chunk* chunk = workspace->find_chunk(chunk_id_)) {
                return *chunk;
            }
        }
        throw Error("RefineMarkers: no chunk with id " +
                    std::to_string(chunk_id_));
    }

    std::int32_t chunk_id_;
    std::vector<MarkerEdit> edits_;
    std::vector<Marker> snapshot_;
};

}  // namespace

RefineMarkersPlan plan_refine_markers(const Document& document,
                                      std::int32_t chunk_id,
                                      const RefineMarkersOptions& options) {
    RefineMarkersPlan plan;
    if (options.patch_size < 3 || options.patch_size > 255) {
        throw ValueError("RefineMarkers: patch_size must be in [3, 255]");
    }
    if (options.downscale < 1 ||
        (options.downscale & (options.downscale - 1)) != 0) {
        throw ValueError("RefineMarkers: downscale must be a power of two >= 1");
    }
    if (!(options.min_correlation >= -1.0 && options.min_correlation <= 1.0)) {
        throw ValueError("RefineMarkers: min_correlation must be in [-1, 1]");
    }
    if (!(options.correlation_margin >= 0.0)) {
        throw ValueError("RefineMarkers: correlation_margin must be non-negative");
    }
    if (options.search_steps < 1 || options.search_steps > 1000) {
        throw ValueError("RefineMarkers: search_steps must be in [1, 1000]");
    }
    if (options.max_iterations < 1 || options.max_iterations > 10000) {
        throw ValueError("RefineMarkers: max_iterations must be in [1, 10000]");
    }
    if (!(options.tolerance_px > 0.0)) {
        throw ValueError("RefineMarkers: tolerance_px must be positive");
    }
    const Chunk* chunk = find_chunk(document, chunk_id);
    if (chunk == nullptr) {
        throw Error("RefineMarkers needs an active chunk");
    }
    // Geometry only: no image is read for a camera that no selected marker
    // references. Images become per-marker windows in pass 1.
    const std::vector<CameraView> geometry = build_camera_views(
        *chunk, options.cameras, options.downscale);

    std::vector<std::size_t> selected;
    for (std::size_t i = 0; i < chunk->markers().size(); ++i) {
        const Marker& marker = chunk->markers()[i];
        if (options.markers.empty() || contains(options.markers, marker.label)) {
            selected.push_back(i);
        }
    }
    for (const std::string& label : options.markers) {
        bool found = false;
        for (const Marker& marker : chunk->markers()) {
            found = found || marker.label == label;
        }
        if (!found) {
            throw ValueError("RefineMarkers: no marker with label '" + label + "'");
        }
    }
    if (selected.empty()) {
        return plan;
    }
    std::vector<MarkerEdit> edits(selected.size());
    const PointCloud& point_cloud = chunk->point_cloud();
    std::vector<Eigen::Vector3d> marker_positions;
    marker_positions.reserve(chunk->markers().size());
    for (const Marker& marker : chunk->markers()) {
        if (marker.position.has_value()) {
            marker_positions.push_back(*marker.position);
        }
    }
    CloudSamples cloud;
    cloud.dense = &point_cloud.points;
    cloud.markers = &marker_positions;
    const WindowPlan window_plan =
        build_window_requests(*chunk, selected, geometry, cloud, options);
    std::vector<CameraView> windows =
        load_windows(document, *chunk, geometry, window_plan.requests,
                     !options.cameras.empty(), plan.stats);
    const std::size_t npos = std::numeric_limits<std::size_t>::max();
    const int count = static_cast<int>(selected.size());
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < count; ++i) {
        const Marker& marker =
            chunk->markers()[selected[static_cast<std::size_t>(i)]];
        const std::vector<std::size_t>& mapping =
            window_plan.marker_windows[static_cast<std::size_t>(i)];
        std::vector<BlendView> blends;
        blends.reserve(marker.projections.size());
        for (std::size_t p = 0; p < marker.projections.size(); ++p) {
            const MarkerProjection& projection = marker.projections[p];
            if (!projection.enabled || !projection.pixel.allFinite() ||
                p >= mapping.size() || mapping[p] == npos ||
                !windows[mapping[p]].loaded()) {
                continue;
            }
            BlendView blend;
            blend.view = &windows[mapping[p]];
            blend.pixel = projection.pixel;
            blends.push_back(blend);
        }
        try {
            edits[static_cast<std::size_t>(i)] =
                refine_marker(std::move(blends), cloud, options);
        } catch (const std::exception&) {
            // An exception must not escape an OpenMP parallel region, which
            // would terminate the process. A marker that cannot be refined is
            // reported as failed and left untouched, like a failed consensus.
            MarkerEdit failed;
            failed.failed = true;
            edits[static_cast<std::size_t>(i)] = failed;
        }
        edits[static_cast<std::size_t>(i)].marker_id = marker.id;
    }

    // One staged command for the whole run: a single marker-container snapshot
    // keeps undo memory linear in markers rather than quadratic in the number
    // of projections, and revert restores every container exactly.
    std::vector<MarkerEdit> staged;
    for (std::size_t i = 0; i < selected.size(); ++i) {
        const Marker& marker = chunk->markers()[selected[i]];
        MarkerEdit& edit = edits[i];
        if (edit.skipped) {
            ++plan.stats.skipped;
            continue;
        }
        if (edit.failed) {
            ++plan.stats.failed;
            continue;
        }
        if (edit.position.has_value() && marker.position.has_value() &&
            (*edit.position - *marker.position).norm() <= 1e-12) {
            edit.position.reset();
        }
        plan.stats.removed_projections += edit.removals.size();
        if (edit.rounds > 0) {
            plan.stats.rounds += static_cast<std::size_t>(edit.rounds);
        }
        if (edit.touched) {
            ++plan.stats.refined;
        } else {
            ++plan.stats.unchanged;
        }
        if (!edit.moves.empty() || !edit.removals.empty() ||
            edit.position.has_value()) {
            staged.push_back(std::move(edit));
        }
    }
    if (!staged.empty()) {
        plan.command = std::make_unique<RefineMarkersCommand>(
            chunk->id(), std::move(staged));
    }
    return plan;
}

void RefineMarkersTask::declare_params(Params& params) const {
    params.declare({"markers", ParamType::StringList, std::vector<std::string>{},
                    "Marker labels to refine. Empty means every marker."});
    params.declare({"cameras", ParamType::StringList, std::vector<std::string>{},
                    "Camera labels whose projections may be used. Empty means every "
                    "enabled (aligned) camera; an explicit list also admits "
                    "unaligned cameras for 2-D-only refinement."});
    ParamSpec patch{"patch_size", ParamType::Int, std::int64_t{25},
                    "Side length of the square patch sampled on the physical target "
                    "plane, in samples. Odd values put a sample on the centre."};
    patch.minimum = std::int64_t{3};
    patch.maximum = std::int64_t{255};
    params.declare(patch);
    ParamSpec downscale{"downscale", ParamType::Int, std::int64_t{1},
                        "Image downscale factor used for patch sampling "
                        "(a power of two; 1 = full resolution)."};
    downscale.minimum = std::int64_t{1};
    downscale.maximum = std::int64_t{64};
    downscale.power_of_two = true;
    params.declare(downscale);
    ParamSpec correlation{"min_correlation", ParamType::Float, 0.5,
                          "Minimum zero-mean NCC for a view to count as evidence "
                          "and as an inlier."};
    correlation.minimum = -1.0;
    correlation.maximum = 1.0;
    params.declare(correlation);
    ParamSpec margin{"correlation_margin", ParamType::Float, 0.15,
                     "An inlier may be at most this far below the best "
                     "non-reference view's correlation."};
    margin.minimum = 0.0;
    margin.maximum = 1.0;
    params.declare(margin);
    ParamSpec steps{"search_steps", ParamType::Int, std::int64_t{10},
                    "Half-width of the fallback depth search, in steps of the "
                    "physical target radius."};
    steps.minimum = std::int64_t{1};
    steps.maximum = std::int64_t{1000};
    params.declare(steps);
    ParamSpec iterations{"max_iterations", ParamType::Int, std::int64_t{20},
                         "Maximum Lucas-Kanade iterations per inlier view."};
    iterations.minimum = std::int64_t{1};
    iterations.maximum = std::int64_t{10000};
    params.declare(iterations);
    ParamSpec tolerance{"tolerance_px", ParamType::Float, 0.01,
                        "Lucas-Kanade stopping tolerance in pixels."};
    tolerance.minimum = 1e-6;
    params.declare(tolerance);
    params.declare({"remove_invalid", ParamType::Bool, true,
                    "Delete a projection that the multi-view consensus rejects. "
                    "When false the projection is kept unchanged."});
}

std::vector<Stage> RefineMarkersTask::stages() const {
    return {Stage{"refine", 0}, Stage{"apply", 0}};
}

void RefineMarkersTask::apply(Document& document, Params& params,
                              const Progress& progress) {
    RefineMarkersOptions options;
    options.markers = params.get_string_list("markers");
    options.cameras = params.get_string_list("cameras");
    options.patch_size = static_cast<int>(params.get_int("patch_size"));
    options.downscale = static_cast<int>(params.get_int("downscale"));
    options.min_correlation = params.get_float("min_correlation");
    options.correlation_margin = params.get_float("correlation_margin");
    options.search_steps = static_cast<int>(params.get_int("search_steps"));
    options.max_iterations = static_cast<int>(params.get_int("max_iterations"));
    options.tolerance_px = params.get_float("tolerance_px");
    options.remove_invalid = params.get_bool("remove_invalid");

    progress.update("refine", 0.0);
    const auto started = std::chrono::steady_clock::now();
    RefineMarkersPlan plan =
        plan_refine_markers(document, document.active_chunk_id(), options);
    progress.update("refine", 1.0);
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
            << "RefineMarkers: refined " << plan.stats.refined << ", unchanged "
            << plan.stats.unchanged << ", failed " << plan.stats.failed
            << ", skipped " << plan.stats.skipped << ", removed projections "
            << plan.stats.removed_projections << ", rounds "
            << plan.stats.rounds << ", peak "
            << plan.stats.peak_bytes / (1024 * 1024) << " MiB in " << seconds
            << " s";
    }
}

}  // namespace alphashape
