#include "common/algorithm/markers/refine_markers_task.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <list>
#include <memory>
#include <numbers>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

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
#include "common/io/camera_image.hpp"
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

bool contains(const std::vector<std::string>& list, const std::string& value) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

/// True for the coded circular target families. Their even ids print a white
/// centre, so their bright enclosed interior is the only parity-0 seed.
bool is_circular_coded_target_type(const std::string& type) {
    return type == "Circular12" || type == "Circular14" ||
           type == "Circular16" || type == "Circular20" ||
           type == "CircularTarget12bit" || type == "CircularTarget14bit" ||
           type == "CircularTarget16bit" || type == "CircularTarget20bit";
}

/// True for any circular target, coded or not. Both are refined from the
/// printed ring/disc boundary, never from a saddle.
bool is_circular_target_type(const std::string& type) {
    return is_circular_coded_target_type(type) || type == "Circular" ||
           type == "CircularTarget";
}

/// True for the checker/bowtie cross families. Their centre is a saddle where
/// four printed quadrants meet, so the circle/ellipse model does not apply.
bool is_cross_target_type(const std::string& type) {
    return type == "Cross" || type == "CrossTarget";
}

/// Which printed pattern the marker's recorded target type describes.
/// `Unknown` is a marker whose target type was never recorded (a hand-placed
/// or imported survey point): neither model may be assumed, so both are
/// evaluated and the better-supported centre wins. Assuming a circular coded
/// target here used to apply the ring/parity seed search to square survey
/// targets, whose dark quadrant looks exactly like a centre dot.
enum class TargetModel : std::uint8_t { Unknown = 0, Circular = 1, Cross = 2 };

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

/// Mutable counterpart, for publishing the run diagnostics on the chunk meta.
Chunk* find_chunk_mutable(Document& document, std::int32_t chunk_id) {
    for (auto& workspace : document.workspaces()) {
        if (!workspace) {
            continue;
        }
        if (Chunk* chunk = workspace->find_chunk(chunk_id)) {
            return chunk;
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

/// One selected camera with its pose, calibration and reduced-resolution image.
struct CameraView {
    const Camera* camera = nullptr;
    Calibration calibration;
    FloatImage image;
    /// Gradient magnitude of `image`, computed once when the view is decoded
    /// and reused by every marker that refines against this view.
    GradientField gradient_field;
    int downscale = 1;

    bool aligned() const { return camera != nullptr && camera->enabled; }

    /// Image-index coordinate of a full-resolution pixel coordinate. The
    /// reduced image keeps the calibration convention u' = u / downscale.
    Eigen::Vector2d index_of(const Eigen::Vector2d& pixel) const {
        const double ds = static_cast<double>(downscale);
        return pixel / ds - Eigen::Vector2d(0.5, 0.5);
    }

    bool contains_pixel(const Eigen::Vector2d& pixel, double margin = 1.0) const {
        const Eigen::Vector2d p = index_of(pixel);
        return p.x() >= margin && p.y() >= margin &&
               p.x() <= static_cast<double>(image.width() - 1) - margin &&
               p.y() <= static_cast<double>(image.height() - 1) - margin;
    }

    float sample(const Eigen::Vector2d& pixel) const {
        const Eigen::Vector2d p = index_of(pixel);
        return image.sample(static_cast<float>(p.x()), static_cast<float>(p.y()));
    }

    /// Image gradient with respect to a full-resolution pixel shift.
    Eigen::Vector2d gradient(const Eigen::Vector2d& pixel) const {
        const double h = static_cast<double>(downscale);
        const double dx = static_cast<double>(
                              sample(pixel + Eigen::Vector2d(h, 0.0)) -
                              sample(pixel - Eigen::Vector2d(h, 0.0))) /
                          (2.0 * h);
        const double dy = static_cast<double>(
                              sample(pixel + Eigen::Vector2d(0.0, h)) -
                              sample(pixel - Eigen::Vector2d(0.0, h))) /
                          (2.0 * h);
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
            if (!point.allFinite()) {
                continue;
            }
            function(point);
        }
    }
    if (samples.markers != nullptr) {
        for (const Eigen::Vector3d& point : *samples.markers) {
            if (!point.allFinite()) {
                continue;
            }
            function(point);
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
    return project_marker(point, *view.camera, view.calibration, false);
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
        if (!observation.pixel.allFinite()) {
            return std::nullopt;
        }
        sfm::AbsolutePose pose;
        pose.R = observation.view->camera->pose.block<3, 3>(0, 0);
        pose.t = observation.view->camera->pose.block<3, 1>(0, 3);
        poses.push_back(pose);
        Eigen::Vector3d bearing =
            observation.view->calibration.unproject(observation.pixel);
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
    if (!point.allFinite()) {
        return std::nullopt;
    }
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
    /// True when the seed came from the light-on-dark pass, i.e. the target
    /// has no centre dot and its centre must be taken from the outer border.
    bool light_centre = false;
    /// True when the light branch's outer-border fit succeeded, as opposed to
    /// falling back to the interior blob centre.
    bool border_fitted = false;
    /// True when a border fit was found but its radius does not fit inside the
    /// window. The fit is then extrapolated from a clipped arc (the outline
    /// lies outside the crop), which is how a large target's interior edge is
    /// mistaken for its border; the caller retries in a wider window.
    bool border_clipped = false;
    /// True when the seed came from a printed centre dot (parity 1). A seed
    /// that is neither a dot nor a light interior is only a dark-blob guess,
    /// which a coded circular target retries in a wider window.
    bool centre_dot = false;
};

/// Locate the circular target around `pixel` in the (possibly reduced) image.
/// The blob's scale supplies the physical radius; the ellipse refit supplies a
/// seed for the 2-D fallback. A cross has no blob, so this returns nullopt and
/// the caller falls back to patch_size / 4.
std::optional<BlobSeed> circular_seed_window(const CameraView& view,
                                             const Eigen::Vector2d& pixel,
                                             int half, bool coded_circular,
                                             double border_multiple) {
    const Eigen::Vector2d index = view.index_of(pixel);
    if (!index.allFinite()) {
        return std::nullopt;
    }
    const double max_x =
        static_cast<double>(std::max(0, view.image.width() - 1));
    const double max_y =
        static_cast<double>(std::max(0, view.image.height() - 1));
    const int centre_x = static_cast<int>(
        std::floor(std::clamp(index.x(), 0.0, max_x)));
    const int centre_y = static_cast<int>(
        std::floor(std::clamp(index.y(), 0.0, max_y)));
    const int x0 =
        std::clamp(centre_x - half, 0,
                   std::max(0, view.image.width() - 2 * half - 1));
    const int y0 =
        std::clamp(centre_y - half, 0,
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
    // The window already contains the whole printed target, so its half width
    // is the largest blob scale worth searching. A fixed cap would either clip
    // a large target's disc or, when it outgrows a normal target's window,
    // admit a coarse background blob that was never a candidate before.
    params.max_radius = std::max(4, half);
    params.threshold = 0.02F;
    params.min_axis_ratio = 0.3;
    const std::vector<MarkerCandidate> dark_blobs = detect_blobs(crop, params);
    // For a parity-0 circular target the white centre merges with the
    // 0.25..0.5 R white gap, so the dark-on-light pass has no dot to seed
    // from. The light-on-dark pass finds the bright interior disc instead.
    // Both searches are anchored on the expected pixel: a parity-1 dot sits on
    // it, a parity-0 interior is centred on it, while sector wedges (dark or
    // light) are half a radius away and must not be seeded.
    params.dark_on_light = false;
    // Only a coded circular target has the bright enclosed interior this
    // pass responds to. Crosses and other kinds keep the dark-blob path.
    const std::vector<MarkerCandidate> light_blobs =
        coded_circular ? detect_blobs(crop, params)
                       : std::vector<MarkerCandidate>{};
    const Eigen::Vector2d seed(index.x() - static_cast<double>(x0) + 0.5,
                               index.y() - static_cast<double>(y0) + 0.5);
    const double anchor_gate = std::max(4.0, 0.15 * static_cast<double>(half));
    // A printed centre dot is dark with the white gap just outside it. A
    // coarse dark blob formed by the whole target of a parity-0 id also sits
    // on the expected pixel, but its centre is the white inner gap. Sample the
    // centre against an annulus at 1.5x the blob radius to tell them apart.
    const auto looks_like_dot = [&](const MarkerCandidate& blob) {
        if (!(blob.radius_px > 1.0)) {
            return false;
        }
        const int dot_x =
            static_cast<int>(std::lround(blob.center.x() - 0.5));
        const int dot_y =
            static_cast<int>(std::lround(blob.center.y() - 0.5));
        if (dot_x < 0 || dot_y < 0 || dot_x >= crop.width() ||
            dot_y >= crop.height()) {
            return false;
        }
        // Average a small disc rather than one pixel: a 2 px speck or noise
        // spike at the nominal centre must not masquerade as the dot.
        const int sample_radius =
            std::max(1, static_cast<int>(std::lround(0.3 * blob.radius_px)));
        double centre_sum = 0.0;
        int centre_count = 0;
        for (int dy = -sample_radius; dy <= sample_radius; ++dy) {
            for (int dx = -sample_radius; dx <= sample_radius; ++dx) {
                if (dx * dx + dy * dy > sample_radius * sample_radius) {
                    continue;
                }
                const int x = dot_x + dx;
                const int y = dot_y + dy;
                if (x < 0 || y < 0 || x >= crop.width() || y >= crop.height()) {
                    continue;
                }
                centre_sum += static_cast<double>(crop.at(x, y));
                ++centre_count;
            }
        }
        if (centre_count == 0) {
            return false;
        }
        const double centre_value =
            centre_sum / static_cast<double>(centre_count);
        const double ring_radius = 1.5 * blob.radius_px;
        double sum = 0.0;
        int count = 0;
        for (int step = 0; step < 16; ++step) {
            const double angle = 2.0 * std::numbers::pi_v<double> *
                                 static_cast<double>(step) / 16.0;
            const int x = static_cast<int>(std::lround(
                blob.center.x() - 0.5 + ring_radius * std::cos(angle)));
            const int y = static_cast<int>(std::lround(
                blob.center.y() - 0.5 + ring_radius * std::sin(angle)));
            if (x < 0 || y < 0 || x >= crop.width() || y >= crop.height()) {
                continue;
            }
            sum += static_cast<double>(crop.at(x, y));
            ++count;
        }
        return count >= 8 &&
               centre_value + 0.08 < sum / static_cast<double>(count);
    };
    // A printed centre dot sits on the expected pixel and is the smallest dark
    // blob centred there; the sector ring is centred on it too but has a much
    // larger scale, and a sensor speck can pass the dot test a few pixels
    // away. Find the nearest dot-like blob first, then keep the smallest
    // radius within a proximity band of it. A hard minimum radius cannot be
    // used: at a reduced working resolution a genuine dot is only a few pixels
    // across and would fall into the light branch.
    double nearest_dot = std::numeric_limits<double>::infinity();
    for (const MarkerCandidate& blob : dark_blobs) {
        const double distance = (blob.center - seed).norm();
        if (distance > anchor_gate || !looks_like_dot(blob)) {
            continue;
        }
        nearest_dot = std::min(nearest_dot, distance);
    }
    const MarkerCandidate* dot = nullptr;
    if (std::isfinite(nearest_dot)) {
        const double band = nearest_dot + 3.0;
        for (const MarkerCandidate& blob : dark_blobs) {
            const double distance = (blob.center - seed).norm();
            if (distance > anchor_gate || distance > band ||
                !looks_like_dot(blob)) {
                continue;
            }
            if (dot == nullptr || blob.radius_px < dot->radius_px) {
                dot = &blob;
            }
        }
    }
    BlobSeed out;
    Eigen::Vector2d centre = seed;
    double radius = 0.0;
    if (dot != nullptr) {
        // Parity 1: the centre dot is the printed fiducial and supplies the
        // patch scale, unchanged from the dark-only path.
        centre = dot->center;
        radius = dot->radius_px;
        out.centre_dot = true;
        if (const auto fitted = refine_ellipse_center(crop, *dot)) {
            centre = *fitted;
        }
    } else if (coded_circular) {
        // Parity 0: take the largest light blob anchored on the expected
        // pixel (the interior disc) and fit the target's outer border from
        // that pixel. The border is a complete concentric outline, while a
        // first-edge fit mixes the 0.5 R sector ring with the border.
        const MarkerCandidate* light = nullptr;
        for (const MarkerCandidate& blob : light_blobs) {
            if ((blob.center - seed).norm() > anchor_gate) {
                continue;
            }
            if (light == nullptr || blob.radius_px > light->radius_px) {
                light = &blob;
            }
        }
        if (light != nullptr) {
            radius = light->radius_px;
            double fitted_radius = radius;
            if (const auto fitted =
                    refine_border_center(crop, seed, radius, &fitted_radius)) {
                // Reject a fit that left the expected target or whose radius
                // is not a plausible multiple of the interior scale; that is
                // what a neighbouring target's outline looks like.
                const double moved = (*fitted - seed).norm();
                // The border is at most about 2.0x the interior radius by
                // design (the interior reaches down to 0.5 R) and at least
                // about 1.1x it, but the blob's own fit lands anywhere between
                // the 0.5 R sector edge and the border's inner edge at 0.9 R,
                // so the lower bound stays loose: rejecting those fits drops
                // the centre back onto the biased interior blob. The upper
                // bound is looser still in a widened window, where the blob's
                // radius comes from a clipped first-edge measurement.
                const double window_limit =
                    static_cast<double>(half) - 2.0;
                if (fitted_radius > window_limit) {
                    out.border_clipped = true;
                }
                if (fitted_radius >= 0.8 * light->radius_px &&
                    fitted_radius <= border_multiple * light->radius_px &&
                    fitted_radius <= window_limit &&
                    moved <= std::max(4.0, 0.3 * fitted_radius)) {
                    centre = *fitted;
                    radius = fitted_radius;
                    out.border_fitted = true;
                } else {
                    centre = light->center;
                }
            } else {
                centre = light->center;
            }
            out.light_centre = true;
        } else {
            // No target found: fall back to the original nearest dark blob,
            // which is how cross targets and low-contrast cases were seeded.
            const MarkerCandidate* dark = nullptr;
            double best_distance = static_cast<double>(half) * 0.75;
            for (const MarkerCandidate& blob : dark_blobs) {
                const double distance = (blob.center - seed).norm();
                if (distance <= best_distance) {
                    best_distance = distance;
                    dark = &blob;
                }
            }
            if (dark == nullptr) {
                return std::nullopt;
            }
            centre = dark->center;
            radius = dark->radius_px;
            if (const auto fitted = refine_ellipse_center(crop, *dark)) {
                centre = *fitted;
            }
        }
    } else {
        // Non-circular target: keep the original dark-blob path.
        const MarkerCandidate* nearest = nullptr;
        double best_distance = static_cast<double>(half) * 0.75;
        for (const MarkerCandidate& blob : dark_blobs) {
            const double distance = (blob.center - seed).norm();
            if (distance <= best_distance) {
                best_distance = distance;
                nearest = &blob;
            }
        }
        if (nearest == nullptr) {
            return std::nullopt;
        }
        centre = nearest->center;
        radius = nearest->radius_px;
        if (const auto fitted = refine_ellipse_center(crop, *nearest)) {
            centre = *fitted;
        }
    }
    out.pixel = (centre + Eigen::Vector2d(static_cast<double>(x0),
                                          static_cast<double>(y0))) *
                static_cast<double>(view.downscale);
    out.radius = radius * static_cast<double>(view.downscale);
    return out;
}

/// Locate the circular target around `pixel` in the (possibly reduced) image.
/// The crop grows with the expected apparent radius when the caller knows it,
/// but only far enough to hold the printed outline and a little context: an
/// unnecessarily wide crop admits background blobs and edges that a normal
/// target's window never sees. Without an expected radius the 48 px minimum
/// applies, and an outer-border fit that did not fit inside that window is
/// retried once in a window twice as wide, because a border larger than the
/// minimum would otherwise be clipped and read as an inner edge.
std::optional<BlobSeed> circular_seed(const CameraView& view,
                                      const Eigen::Vector2d& pixel,
                                      int patch_size, double expected_radius,
                                      bool coded_circular,
                                      bool retry_when_missing) {
    const double ds = static_cast<double>(std::max(1, view.downscale));
    const double expected_px =
        expected_radius > 0.0 ? expected_radius / ds : 0.0;
    const int extent = std::min(view.image.width(), view.image.height());
    const int max_half = std::max(1, extent / 2);
    int half = std::max(48, patch_size);
    if (expected_px > static_cast<double>(half) - 8.0) {
        // Grow only when the expected radius is close enough to the default
        // half width that the crop could clip the printed outline. The
        // expected radius may be the outer border or a light interior as small
        // as 0.5 R; either way R plus a blur allowance fits inside 1.3x it
        // plus eight pixels. Growing a window that already holds the target
        // only admits background blobs and edges that the normal window never
        // sees.
        half = std::min(max_half,
                        std::max(half, static_cast<int>(
                                           std::ceil(1.3 * expected_px)) + 8));
    }
    half = std::min(half, max_half);
    auto seed = circular_seed_window(view, pixel, half, coded_circular, 2.0);
    // A border fit that needed more room than the window offered means the
    // outline was clipped, so the accepted radius and centre are unreliable; a
    // coded target whose interior also fell outside the window has no light
    // seed at all and only a dark-blob guess. One doubled window covers both
    // cases. A blob radius measured from a clipped first edge is only a lower
    // bound, so the wider window accepts a larger border multiple. Its fit is
    // only used when the outline really was larger than the first window;
    // otherwise the first window already held the target and its own seed is
    // the better one.
    const bool clipped = seed && seed->border_clipped;
    const bool guessed =
        coded_circular && seed && !seed->light_centre && !seed->centre_dot;
    const bool missing = retry_when_missing && !seed;
    if (coded_circular && (clipped || guessed || missing) &&
        half < max_half) {
        const int wider = std::min(2 * half, max_half);
        if (auto retried =
                circular_seed_window(view, pixel, wider, coded_circular, 2.4)) {
            const bool larger =
                retried->border_fitted &&
                retried->radius > static_cast<double>(half) - 2.0;
            if (!seed || larger) {
                return retried;
            }
        }
    }
    return seed;
}

/// Radius of a detected cross near `pixel`, in full-resolution pixels. A cross
/// has no blob scale, so its printed extent is what the physical target radius
/// is derived from. Nullopt when no cross is detected near the seed, which is
/// also how an unpainted region is recognised: it has no printed target.
std::optional<double> cross_radius(const CameraView& view,
                                   const Eigen::Vector2d& pixel,
                                   int patch_size) {
    const Eigen::Vector2d index = view.index_of(pixel);
    if (!index.allFinite()) {
        return std::nullopt;
    }
    const int half = std::max(48, 2 * patch_size);
    const int x0 = std::clamp(
        static_cast<int>(std::floor(std::clamp(index.x(), 0.0, 1.0e6))) - half,
        0, std::max(0, view.image.width() - 2 * half - 1));
    const int y0 = std::clamp(
        static_cast<int>(std::floor(std::clamp(index.y(), 0.0, 1.0e6))) - half,
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
    // A completely flat crop has no cross. It is also the one input that can
    // send the adaptive-threshold contour search through an unbounded number
    // of passes, so the low-texture fixture must exit here.
    double sum = 0.0;
    double sum_squared = 0.0;
    for (const float value : crop.pixels()) {
        const double v = static_cast<double>(value);
        sum += v;
        sum_squared += v * v;
    }
    const double count = static_cast<double>(crop.pixels().size());
    const double variance = sum_squared / count - (sum / count) * (sum / count);
    if (!(variance > 1e-8)) {
        return std::nullopt;
    }
    CrossParams params;
    params.minimum_size = 2.0;
    params.minimum_dist = 8.0;
    const std::vector<CrossDetection> detections =
        detect_crosses_styled(crop, params);
    const Eigen::Vector2d seed(index.x() - static_cast<double>(x0) + 0.5,
                               index.y() - static_cast<double>(y0) + 0.5);
    double best = static_cast<double>(half);
    double radius = 0.0;
    for (const CrossDetection& detection : detections) {
        if (!detection.target.center.allFinite() ||
            !(detection.target.radius_px > 0.0)) {
            continue;
        }
        const double distance = (detection.target.center - seed).norm();
        // The click is the printed target's centre, so a genuine saddle sits
        // on it. A textured background (grass, gravel, foliage) produces
        // spurious saddles everywhere, and without this gate one of those
        // would supply the patch scale for a circular target. The gate is the
        // same fraction-of-scale anchor the circular seed uses.
        const double gate = std::max(4.0, 0.35 * detection.target.radius_px);
        if (distance <= gate && distance < best) {
            best = distance;
            radius = detection.target.radius_px;
        }
    }
    if (!(radius > 0.0)) {
        return std::nullopt;
    }
    return radius * static_cast<double>(std::max(1, view.downscale));
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
            std::isfinite(count_d)
                ? static_cast<int>(std::min(count_d,
                                             static_cast<double>(maximum)))
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
                                           const Eigen::Vector2d& pixel,
                                           int patch_size) {
    const Eigen::Vector2d index = view.index_of(pixel);
    if (!index.allFinite()) {
        return std::nullopt;
    }
    const double max_x =
        static_cast<double>(std::max(0, view.image.width() - 1));
    const double max_y =
        static_cast<double>(std::max(0, view.image.height() - 1));
    const int centre_x = static_cast<int>(
        std::lround(std::clamp(index.x(), 0.0, max_x)));
    const int centre_y = static_cast<int>(
        std::lround(std::clamp(index.y(), 0.0, max_y)));
    // Small window: the candidate is a few pixels off at most, and a wider
    // scan would start locking onto unrelated texture saddles.
    const int scan = std::max(3, std::min(6, patch_size / 4));
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
    return Eigen::Vector2d((static_cast<double>(best_x) + 0.5) * ds,
                           (static_cast<double>(best_y) + 0.5) * ds);
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
    const std::optional<BlobSeed>& seed, int patch_size, double limit,
    double hint_px) {
    if (!pixel.allFinite() || !std::isfinite(limit)) {
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
    // Refit the ellipse at the candidate, using the blob only for its scale.
    // Trusting the blob's centre would accept a nearby texture blob; the
    // printed target is centred at the candidate to within a few pixels, so
    // the radial search converges to the real boundary from there.
    std::optional<Eigen::Vector2d> ellipse;
    if (seed && seed->light_centre) {
        // The crop re-seed already fitted the outer border from a window that
        // holds only this target, so reuse that centre. A border fit on the
        // whole image can lock onto a neighbouring target's outline instead.
        ellipse = accept(seed->pixel);
    } else if (seed) {
        MarkerCandidate ellipse_seed;
        ellipse_seed.center = pixel / ds;
        ellipse_seed.radius_px = seed->radius / ds;
        if (const auto fitted =
                refine_ellipse_center(view.gradient_field, ellipse_seed)) {
            ellipse = accept(*fitted * ds);
        }
    }
    // Both printed patterns are refined: the boundary fit locks onto a ring
    // or disc, the saddle fit onto a checker quadrant crossing. Whichever
    // converged nearer the candidate wins, because the correct refiner locks
    // onto the printed centre in a pixel or two while the wrong one lands on
    // a quadrant, a neighbouring blob or an unrelated saddle far away. On a
    // rendered checker the two disagree by more than a pixel, so keeping both
    // and taking the closer one is measurably more accurate than assuming
    // which pattern the marker carries; a target type only selects the SCALE.
    const double hint =
        hint_px > 0.0
            ? hint_px / ds
            : std::max(4.0, static_cast<double>(patch_size) / 4.0) / ds;
    std::optional<Eigen::Vector2d> cross;
    Eigen::Vector2d start = pixel;
    if (const auto saddle = saddle_seed(view, pixel, patch_size)) {
        start = *saddle;
    }
    if (const auto refined = refine_cross_center(view.image, start / ds, hint)) {
        cross = accept(*refined * ds);
    }
    if (ellipse &&
        (!cross ||
         (*ellipse - pixel).squaredNorm() <= (*cross - pixel).squaredNorm())) {
        return ellipse;
    }
    return cross;
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
    /// Parallel to `removals`: why each projection left the marker.
    std::vector<RefineRemovalReason> removal_reasons;
    /// Re-triangulated estimated position, never the geodetic reference.
    std::optional<Eigen::Vector3d> position;
    /// Projections that reached a decision (moved, kept or removed).
    std::size_t evaluated = 0;
    RefineFailureReason failure = RefineFailureReason::None;
    bool failed = false;
    bool touched = false;
    bool skipped = false;
};

/// One view's data for a marker: its measurement, apparent scale and depth.
struct BlendView {
    const CameraView* view = nullptr;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    std::optional<BlobSeed> seed;
    /// Apparent target radius in full-resolution pixels.
    double apparent = 1.0;
    /// True when `apparent` came from a detected blob; otherwise it is filled
    /// from the projected physical radius after the initial triangulation.
    bool has_scale = false;
    /// True when the scale is a blob radius. A cross-extent scale is real but
    /// carries no blob, so the refiners keep the tighter movement guard.
    bool blob_scale = false;
    /// True when the scale came from the checker-cross detector's target
    /// half-size, so the patch is already in printed-target units.
    bool cross_scale = false;
    /// True when the marker is a coded circular target, the only kind whose
    /// even parity has an enclosed bright interior to seed from.
    bool coded_circular = true;
    /// True when a target kind was recorded for this marker. Without one the
    /// seed search keeps its previous window and only coded kinds widen it.
    bool target_kind_known = false;
    /// The printed pattern the marker declares, or Unknown when none was
    /// recorded. Selects which refiners are allowed to move the projection.
    TargetModel model = TargetModel::Unknown;
    double depth = 1.0;
};

/// Movement guard for the 2-D refiners. A blob-scale view keeps half its
/// apparent radius; a cross-extent (or currently unscaled) view is capped at
/// three full-resolution pixels. The cap is a correction bound, not a target
/// radius: the physical patch scale no longer widens it.
double refinement_limit(const BlendView& blend) {
    const double ds = static_cast<double>(std::max(1, blend.view->downscale));
    const double scale = blend.blob_scale ? blend.apparent : 6.0 * ds;
    return std::max(2.0, 0.5 * scale);
}

/// Hard bound on how far the FINAL refined pixel may sit from the pixel the
/// surveyor supplied, regardless of where the consensus candidate landed.
/// A click on a surveyed target is expected within a couple of pixels, so this
/// is a correction cap, not an accuracy budget: six full-resolution pixels
/// (or half the apparent target radius when the target is larger) is enough
/// for a mis-click to be pulled onto the centre and far too little to wander
/// onto unrelated texture. A refinement that would exceed it is discarded and
/// the measurement is kept unchanged.
double maximum_shift(const BlendView& blend) {
    const double ds = static_cast<double>(std::max(1, blend.view->downscale));
    const double scale = blend.has_scale ? blend.apparent : 6.0 * ds;
    return std::max(6.0 * ds, 0.5 * scale);
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

/// Per-view outcome of a candidate: Unused means the patch leaves the window,
/// so the projection is not evidence either way; Gated means the patch exists
/// but the measured projection is outside the gate; Usable means it scored.
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
/// within one apparent target radius, with a floor of three full-resolution
/// pixels scaled by the working downscale. This is the only signal that can
/// tell a contradicted measurement from a merely noisy one, because NCC itself
/// always sees the printed target. The floor absorbs a few pixels of
/// perturbation plus the reference view's own error; the target scale widens it
/// for a large target. A projection displaced by more than about one radius is
/// therefore removed, not repaired.
double projection_tolerance(const BlendView& blend) {
    const double floor_px =
        3.0 * static_cast<double>(std::max(1, blend.view->downscale));
    return std::max(floor_px, std::max(1.0, blend.apparent));
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
        if (projected && projected->allFinite() &&
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
            bool reference_usable = false;
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
                // measured projection. An unrelated measurement lands outside
                // this gate and is excluded from the score.
                const auto projected = project_view(*blend.view, candidate);
                if (!projected || !projected->allFinite() ||
                    (*projected - blend.pixel).norm() >
                        projection_tolerance(blend)) {
                    patches.push_back(Patch{});
                    status.push_back(ViewStatus::Gated);
                    continue;
                }
                patches.push_back(std::move(patch));
                status.push_back(ViewStatus::Usable);
                if (index == reference) {
                    reference_usable = true;
                } else {
                    ++usable_other;
                }
            }
            if (!reference_usable ||
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
    if (!pixel.allFinite() || !plane_point.allFinite() ||
        !normal.allFinite()) {
        return std::nullopt;
    }
    const Eigen::Matrix3d rotation = view.camera->pose.block<3, 3>(0, 0);
    const Eigen::Vector3d ray = rotation.transpose() *
                                view.calibration.unproject(pixel);
    if (!ray.allFinite()) {
        return std::nullopt;
    }
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

/// Consensus (Fischler & Bolles 1981) at the winning candidate, then subpixel
/// refinement of every inlier. A view is an inlier when its correlation is
/// within `correlation_margin` of the best non-reference view and above
/// `min_correlation`; two inliers are required. A non-inlier is left exactly
/// where it was measured unless `remove_invalid` is set AND its measured
/// pixel geometrically contradicts the refined candidate (disagrees by more
/// than one target radius). A flat patch, a low correlation and a patch that
/// left the image are all statements about the target model, not about the
/// measurement, so they never delete an observation.
std::optional<MarkerEdit> apply_winner(const std::vector<BlendView>& blends,
                                       const Winner& winner,
                                       const Eigen::Vector3d& u,
                                       const Eigen::Vector3d& v,
                                       double half_width,
                                       const RefineMarkersOptions& options) {
    std::vector<std::optional<double>> scores = winner.scores;
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
    const Eigen::Vector3d normal = u.cross(v).normalized();
    std::vector<Eigen::Vector2d> anchored(blends.size());
    std::vector<Eigen::Vector2d> projected_centres(blends.size());
    std::vector<Patch> anchored_patches(blends.size());
    std::vector<bool> usable(blends.size(), false);
    for (const std::size_t index : inliers) {
        const BlendView& blend = blends[index];
        const auto projected = project_view(*blend.view, winner.point);
        if (!projected || !projected->allFinite()) {
            return std::nullopt;
        }
        projected_centres[index] = *projected;
        const double limit = refinement_limit(blend);
        // The blob seed is the ellipse refiner's second opinion and its only
        // anchor for a circular target; it is computed even when the saddle
        // supplied the scale, so both refiners stay available.
        const auto seed = circular_seed(
            *blend.view, *projected, options.patch_size, blend.apparent,
            blend.coded_circular, blend.target_kind_known);
        const auto refined = refined_2d_pixel(
            *blend.view, *projected, seed, options.patch_size, limit,
            blend.has_scale ? blend.apparent : 0.0);
        anchored[index] = refined ? *refined : *projected;
        const auto point = intersect_plane(*blend.view, anchored[index],
                                           winner.point, normal);
        if (point) {
            anchored_patches[index] = sample_patch(
                *blend.view, *point, u, v, half_width, options.patch_size);
            usable[index] = anchored_patches[index].valid;
        }
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
    MarkerEdit edit;
    edit.evaluated = blends.size();
    std::vector<Observation> accepted;
    std::vector<double> tolerances;
    for (std::size_t i = 0; i < blends.size(); ++i) {
        const BlendView& blend = blends[i];
        const bool inlier = std::find(inliers.begin(), inliers.end(), i) !=
                            inliers.end();
        if (!inlier) {
            // A view is only ever deleted for a geometric contradiction: its
            // measured pixel is farther from the refined candidate than one
            // target radius, which no sub-pixel refinement can turn into a
            // useful observation. Every other non-inlier is NOT evidence
            // against the measurement (a flat patch, a low correlation or a
            // patch that left the image all describe the target model, not
            // the click), so it is kept at its measured position. Deleting
            // those observations is what used to halve a real survey's
            // projection count.
            RefineRemovalReason reason = RefineRemovalReason::None;
            const bool unused = i < winner.status.size() &&
                                winner.status[i] == ViewStatus::Unused;
            if (!unused) {
                if (i < winner.status.size() &&
                    winner.status[i] == ViewStatus::Gated) {
                    reason = RefineRemovalReason::GeometricGate;
                } else if (i >= scores.size() || !scores[i]) {
                    reason = RefineRemovalReason::FlatPatch;
                } else if (*scores[i] < options.min_correlation) {
                    reason = RefineRemovalReason::LowCorrelation;
                } else {
                    reason = RefineRemovalReason::CorrelationMargin;
                }
            }
            if (options.remove_invalid &&
                reason == RefineRemovalReason::GeometricGate) {
                edit.removals.push_back(blend.view->camera->id);
                edit.removal_reasons.push_back(reason);
                edit.touched = true;
            }
            continue;
        }
        Eigen::Vector2d refined = anchored[i];
        const double limit = refinement_limit(blend);
        if (mean_count > 0 && usable[i]) {
            const auto shift =
                lk_shift(*blend.view, anchored_patches[i], mean,
                         options.max_iterations, options.tolerance_px);
            if (shift &&
                (refined + *shift - projected_centres[i]).norm() <= limit) {
                refined += *shift;
            }
        }
        // The bound that matters is against the MEASURED pixel, not against
        // the candidate: a bad initial triangulation can put the candidate a
        // whole target radius from the click, and an unbounded correction
        // from there would move a good observation far from where the surveyor
        // clicked. A refinement that would exceed the bound is not trusted;
        // the projection is kept exactly where it was measured.
        if ((refined - blend.pixel).norm() > maximum_shift(blend)) {
            refined = blend.pixel;
        }
        if ((refined - blend.pixel).norm() > 1e-9) {
            edit.moves.push_back({blend.view->camera->id, refined});
            edit.touched = true;
        }
        accepted.push_back({blend.view, refined});
        tolerances.push_back(projection_tolerance(blend));
    }
    // Re-triangulate the estimated position from the accepted refined
    // projections. The marker's geodetic reference is never touched.
    if (accepted.size() >= 2) {
        if (const auto position = triangulate_robust(accepted, tolerances)) {
            edit.position = *position;
        }
    }
    return edit;
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

/// Fill the per-view apparent radius from the blob scale when a circular target
/// yields one. A cross has no blob, so its radius is left for the projected
/// physical radius, filled once the marker has been triangulated: patch_size is
/// a sample count, never a pixel scale.
void fill_scales(std::vector<BlendView>& blends, int patch_size) {
    for (BlendView& blend : blends) {
        // A printed saddle is specific evidence: only a checker pattern (or a
        // genuine bar cross) produces one, and on a survey target the bright
        // background beside a small checker is often a large light blob that
        // the coded-circular light-interior branch would mistake for the
        // target. The saddle's half-size is therefore the patch scale when it
        // exists. The blob seed is still computed: it is the ellipse
        // refiner's independent anchor, and skipping it would hand the whole
        // decision to one refiner.
        std::optional<double> saddle;
        if (blend.model != TargetModel::Circular) {
            saddle = cross_radius(*blend.view, blend.pixel, patch_size);
        }
        blend.seed = circular_seed(*blend.view, blend.pixel, patch_size, 0.0,
                                   blend.coded_circular,
                                   blend.target_kind_known);
        if (saddle.has_value()) {
            blend.apparent = std::max(1.0, *saddle);
            blend.has_scale = true;
            blend.cross_scale = true;
            continue;
        }
        if (blend.seed && std::isfinite(blend.seed->radius) &&
            blend.seed->radius > 0.0) {
            blend.apparent = std::max(1.0, blend.seed->radius);
            blend.has_scale = true;
            blend.blob_scale = true;
            continue;
        }
        if (blend.model == TargetModel::Circular) {
            // A declared circular target with no blob still gets the cross
            // fallback the previous order used.
            if (const auto cross =
                    cross_radius(*blend.view, blend.pixel, patch_size)) {
                blend.apparent = std::max(1.0, *cross);
                blend.has_scale = true;
                blend.cross_scale = true;
                continue;
            }
        }
        blend.apparent = 1.0;
        blend.has_scale = false;
    }
}

MarkerEdit refine_2d_marker(std::vector<BlendView>& blends,
                            const RefineMarkersOptions& options) {
    MarkerEdit edit;
    edit.evaluated = blends.size();
    bool success = false;
    for (BlendView& blend : blends) {
        const double limit = refinement_limit(blend);
        const auto refined = refined_2d_pixel(
            *blend.view, blend.pixel, blend.seed, options.patch_size, limit,
            blend.has_scale ? blend.apparent : 0.0);
        if (!refined) {
            continue;
        }
        success = true;
        if ((*refined - blend.pixel).norm() > 1e-9) {
            edit.moves.push_back({blend.view->camera->id, *refined});
            edit.touched = true;
        }
    }
    if (!blends.empty() && !success) {
        edit.failed = true;
        edit.failure = RefineFailureReason::Consensus;
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
        for (const BlendView& blend : aligned) {
            observations.push_back({blend.view, blend.pixel});
            const double ds =
                static_cast<double>(std::max(1, blend.view->downscale));
            tolerances.push_back(blend.has_scale ? projection_tolerance(blend)
                                                 : 8.0 * ds);
        }
        auto initial = triangulate_robust(observations, tolerances);
        if (!initial || !initial->allFinite()) {
            MarkerEdit failed;
            failed.failed = true;
            failed.failure = RefineFailureReason::Triangulation;
            failed.evaluated = blends.size();
            return failed;
        }
        if (initial && initial->allFinite()) {
            std::vector<double> physical_samples;
            for (BlendView& blend : aligned) {
                blend.depth = depth_in_view(*blend.view, *initial);
                if (!blend.has_scale) {
                    continue;
                }
                const double focal = std::max(blend.view->calibration.f, 1e-6);
                const double sample = blend.apparent * blend.depth / focal;
                if (std::isfinite(sample) && sample > 0.0) {
                    physical_samples.push_back(sample);
                }
            }
            if (physical_samples.empty()) {
                // No view shows a measurable printed target (neither a blob
                // nor a cross), so there is no physical scale to sample. The
                // marker carries no evidence and is left untouched.
                MarkerEdit failed;
                failed.failed = true;
                failed.failure = RefineFailureReason::NoScale;
                failed.evaluated = blends.size();
                return failed;
            }
            const double physical = median(physical_samples);
            if (!(physical > 0.0) || !std::isfinite(physical)) {
                MarkerEdit failed;
                failed.failed = true;
                failed.failure = RefineFailureReason::NoScale;
                failed.evaluated = blends.size();
                return failed;
            }
            // Every view now carries a pixel radius: measured or projected.
            for (BlendView& blend : aligned) {
                if (blend.has_scale) {
                    continue;
                }
                const double focal = std::max(blend.view->calibration.f, 1e-6);
                const double projected_radius =
                    physical * focal / std::max(blend.depth, 1e-9);
                blend.apparent = std::max(1.0, projected_radius);
                blend.has_scale = true;
            }
            for (std::size_t i = 0; i < aligned.size(); ++i) {
                tolerances[i] = projection_tolerance(aligned[i]);
            }
            const auto refined_initial =
                triangulate_robust(observations, tolerances);
            if (refined_initial && refined_initial->allFinite()) {
                initial = refined_initial;
            }
            const Eigen::Vector3d normal = target_normal(
                *initial, observations, cloud, std::max(0.15, 6.0 * physical));
            const auto axes = plane_axes(normal);
            if (const auto winner = search_winner(
                    *initial, axes.first, axes.second, aligned, cloud,
                    physical, options.search_steps, options.patch_size)) {
                if (auto edit =
                        apply_winner(aligned, *winner, axes.first, axes.second,
                                     kPatchHalfWidth * physical, options)) {
                    // Unaligned views were filtered out of the consensus and
                    // are left exactly as measured; they still count as
                    // evaluated so the accounting covers every projection.
                    edit->evaluated = blends.size();
                    return *edit;
                }
            }
        }
        MarkerEdit failed;
        failed.failed = true;
        failed.failure = RefineFailureReason::Consensus;
        failed.evaluated = blends.size();
        return failed;
    }
    return refine_2d_marker(blends, options);
}

/// Validate the selected cameras and keep only those referenced by a selected
/// marker. No image is read here; the cache decodes a view on demand. An
/// explicit camera list is strict about names, frames and sensors.
std::vector<CameraView> build_selected_views(
    const Chunk& chunk, const std::vector<std::string>& filters, int downscale,
    const std::vector<std::int32_t>& needed) {
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
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk.sensors().size())) {
            if (!filters.empty()) {
                throw ValueError("RefineMarkers: camera '" + camera.label +
                                 "' has no sensor");
            }
            continue;
        }
        const Calibration calibration =
            sensor_pipeline_calibration(effective_sensor(chunk, camera));
        if (calibration.width <= 0 || calibration.height <= 0) {
            continue;
        }
        if (std::find(needed.begin(), needed.end(), camera.id) == needed.end()) {
            continue;
        }
        CameraView view;
        view.camera = &camera;
        view.calibration = calibration;
        view.downscale = downscale;
        views.push_back(std::move(view));
    }
    return views;
}

/// Decode one view and compute its gradient field. An implicit camera list
/// logs and counts an unreadable image and returns null; an explicit list
/// throws.
std::shared_ptr<CameraView> decode_view(
    const Document& document, const Chunk& chunk, const CameraView& geometry,
    const std::vector<std::string>& filters, RefineMarkersStats& stats) {
    const Camera& camera = *geometry.camera;
    const Frame* frame = frame_of_camera(chunk, camera.id);
    if (frame == nullptr) {
        return nullptr;
    }
    try {
        FloatImage image =
            load_camera_image(document, chunk, camera).to_luminance();
        if (geometry.downscale > 1) {
            image = downsample_box(image, geometry.downscale);
        }
        auto view = std::make_shared<CameraView>(geometry);
        view->image = std::move(image);
        view->gradient_field = compute_gradient_field(view->image);
        return view;
    } catch (const std::exception& error) {
        const std::string message =
            "RefineMarkers: " + camera.label + ": " + error.what();
        if (!filters.empty()) {
            throw Error(message);
        }
        log(LogLevel::Warn) << message;
        ++stats.unreadable_cameras;
        return nullptr;
    }
}

/// Bounded LRU cache of decoded views. The full image (and its gradient) is
/// kept, never a crop: a crop would change the sampled patch. The byte budget
/// caps how many views are resident at once, and a miss reloads the image.
/// Markers are processed in camera-set order so adjacent markers reuse views.
class ViewCache {
public:
    static constexpr std::size_t kBudgetBytes = 1500ull * 1024ull * 1024ull;

    ViewCache(const Document& document, const Chunk& chunk,
              const std::vector<CameraView>& geometry,
              const std::vector<std::string>& filters,
              RefineMarkersStats& stats)
        : document_(document), chunk_(chunk), geometry_(geometry),
          filters_(filters), stats_(stats) {}

    /// Bytes a resident view occupies: luminance plus gradient magnitude.
    std::size_t bytes_of(std::int32_t camera_id) const {
        const CameraView* view = geometry_of(camera_id);
        if (view == nullptr) {
            return 0;
        }
        const std::size_t ds =
            static_cast<std::size_t>(std::max(1, view->downscale));
        const std::size_t width = static_cast<std::size_t>(std::max(
            1, view->calibration.width / static_cast<int>(ds)));
        const std::size_t height = static_cast<std::size_t>(std::max(
            1, view->calibration.height / static_cast<int>(ds)));
        return width * height * sizeof(float) * 2;
    }

    /// Evict least-recently-used views until `incoming` more bytes fit.
    void make_room(std::size_t incoming) {
        while (!order_.empty() && bytes_ + incoming > kBudgetBytes) {
            const std::int32_t victim = order_.back();
            auto entry = entries_.find(victim);
            if (entry != entries_.end()) {
                bytes_ -= bytes_of(victim);
                entries_.erase(entry);
            }
            order_.pop_back();
        }
    }

    std::shared_ptr<CameraView> get(std::int32_t camera_id) {
        auto entry = entries_.find(camera_id);
        if (entry != entries_.end()) {
            if (entry->second.order != order_.begin()) {
                order_.splice(order_.begin(), order_, entry->second.order);
            }
            return entry->second.view;
        }
        if (std::find(failed_.begin(), failed_.end(), camera_id) !=
            failed_.end()) {
            return nullptr;
        }
        const CameraView* geometry = geometry_of(camera_id);
        if (geometry == nullptr) {
            return nullptr;
        }
        std::shared_ptr<CameraView> view =
            decode_view(document_, chunk_, *geometry, filters_, stats_);
        if (!view) {
            failed_.push_back(camera_id);
            return nullptr;
        }
        const std::size_t bytes = bytes_of(camera_id);
        make_room(bytes);
        entries_[camera_id] =
            Entry{view, order_.insert(order_.begin(), camera_id)};
        bytes_ += bytes;
        peak_bytes_ = std::max(peak_bytes_, bytes_);
        return view;
    }

    std::size_t peak_bytes() const { return peak_bytes_; }

private:
    struct Entry {
        std::shared_ptr<CameraView> view;
        std::list<std::int32_t>::iterator order;
    };

    const CameraView* geometry_of(std::int32_t camera_id) const {
        for (const CameraView& view : geometry_) {
            if (view.camera != nullptr && view.camera->id == camera_id) {
                return &view;
            }
        }
        return nullptr;
    }

    const Document& document_;
    const Chunk& chunk_;
    const std::vector<CameraView>& geometry_;
    const std::vector<std::string>& filters_;
    RefineMarkersStats& stats_;
    std::list<std::int32_t> order_;
    std::unordered_map<std::int32_t, Entry> entries_;
    std::vector<std::int32_t> failed_;
    std::size_t bytes_ = 0;
    std::size_t peak_bytes_ = 0;
};

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

/// One staged command for the whole run. Its snapshot covers the marker
/// container once, so undo memory is linear in markers rather than quadratic in
/// projections, and revert restores the container exactly, order included.
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
                                   [camera_id](const MarkerProjection& projection) {
                                       return projection.camera_id == camera_id;
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

std::string format_number(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return buffer;
}

/// Publish the run's diagnostics on the chunk. A validation protocol reads
/// `chunk.meta` and needs the removal-reason histogram and the shift
/// distribution to tell a target-model mismatch from noisy measurements; the
/// same numbers go to the log. Written even when nothing changed, so a run
/// that refined nothing still explains itself.
void write_refine_meta(Document& document, std::int32_t chunk_id,
                       const RefineMarkersStats& stats) {
    Chunk* chunk = find_chunk_mutable(document, chunk_id);
    if (chunk == nullptr) {
        return;
    }
    const auto put = [&chunk](const std::string& key, std::size_t value) {
        chunk->meta().set("refine_markers." + key, std::to_string(value));
    };
    put("markers_refined", stats.refined);
    put("markers_unchanged", stats.unchanged);
    put("markers_failed", stats.failed);
    put("markers_skipped", stats.skipped);
    put("projections_evaluated", stats.evaluated_projections);
    put("projections_moved", stats.moved_projections);
    put("projections_kept_unrefined", stats.kept_unrefined);
    put("projections_removed", stats.removed_projections);
    put("removed_geometric_gate", stats.removed_geometric_gate);
    put("removed_flat_patch", stats.removed_flat_patch);
    put("removed_low_correlation", stats.removed_low_correlation);
    put("removed_correlation_margin", stats.removed_correlation_margin);
    chunk->meta().set(
        "refine_markers.shift_mean_px",
        format_number(stats.moved_projections == 0
                          ? 0.0
                          : stats.shift_sum_px /
                                static_cast<double>(stats.moved_projections)));
    chunk->meta().set("refine_markers.shift_max_px",
                      format_number(stats.shift_max_px));
    put("shifts_over_2px", stats.shifts_over_2px);
    put("shifts_over_5px", stats.shifts_over_5px);
    put("shifts_over_10px", stats.shifts_over_10px);
    put("failed_no_views", stats.failed_no_views);
    put("failed_no_scale", stats.failed_no_scale);
    put("failed_triangulation", stats.failed_triangulation);
    put("failed_consensus", stats.failed_consensus);
}

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
    // Only a camera that some selected, enabled projection references is worth
    // decoding. Sort the markers so that ones sharing a camera set are
    // adjacent, which lets the view cache reuse them.
    std::vector<std::vector<std::int32_t>> marker_cameras(selected.size());
    std::vector<std::size_t> marker_bytes(selected.size(), 0);
    std::vector<std::string> keys(selected.size());
    std::vector<std::int32_t> needed_cameras;
    for (std::size_t slot = 0; slot < selected.size(); ++slot) {
        std::vector<std::int32_t>& ids = marker_cameras[slot];
        for (const MarkerProjection& projection :
             chunk->markers()[selected[slot]].projections) {
            if (!projection.enabled || !projection.pixel.allFinite()) {
                continue;
            }
            if (std::find(ids.begin(), ids.end(), projection.camera_id) !=
                ids.end()) {
                continue;
            }
            ids.push_back(projection.camera_id);
            if (std::find(needed_cameras.begin(), needed_cameras.end(),
                          projection.camera_id) == needed_cameras.end()) {
                needed_cameras.push_back(projection.camera_id);
            }
        }
        std::sort(ids.begin(), ids.end());
        for (const std::int32_t camera_id : ids) {
            keys[slot] += std::to_string(camera_id) + ",";
        }
    }
    std::vector<std::size_t> order(selected.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&keys](std::size_t a, std::size_t b) {
                         return keys[a] < keys[b];
                     });
    const std::vector<CameraView> geometry = build_selected_views(
        *chunk, options.cameras, options.downscale, needed_cameras);

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
    ViewCache cache(document, *chunk, geometry, options.cameras, plan.stats);
    for (std::size_t slot = 0; slot < selected.size(); ++slot) {
        for (const std::int32_t camera_id : marker_cameras[slot]) {
            marker_bytes[slot] += cache.bytes_of(camera_id);
        }
        // A marker whose own set exceeds the budget cannot be split without a
        // crop; report the unavoidable peak.
        plan.stats.peak_bytes =
            std::max(plan.stats.peak_bytes, marker_bytes[slot]);
    }
    for (const std::size_t slot : order) {
        cache.make_room(marker_bytes[slot]);
        const Marker& marker = chunk->markers()[selected[slot]];
        // A recorded target kind selects the printed pattern. With none (a
        // hand-placed or imported survey point) NEITHER model may be assumed:
        // assuming a circular coded target is what applied the ring/parity
        // seed search to square survey targets. Unknown therefore runs both
        // refiners and keeps the better-supported centre.
        const std::string* target_kind = marker.meta.find("target_type");
        const bool target_kind_known = target_kind != nullptr;
        TargetModel model = TargetModel::Unknown;
        if (target_kind_known) {
            if (is_cross_target_type(*target_kind)) {
                model = TargetModel::Cross;
            } else if (is_circular_target_type(*target_kind)) {
                model = TargetModel::Circular;
            }
        }
        // An unknown kind still gets the light-interior seed: a parity-0
        // circular target has no dark centre, and with both refiners gated by
        // `model` above a spurious seed can only produce a bounded,
        // closest-to-the-click answer, never a deletion.
        const bool coded_circular =
            !target_kind_known || is_circular_coded_target_type(*target_kind);
        std::vector<std::shared_ptr<CameraView>> held;
        std::vector<BlendView> blends;
        blends.reserve(marker.projections.size());
        for (const MarkerProjection& projection : marker.projections) {
            // A disabled projection is not evidence: it is never refined,
            // moved or removed, exactly as if it were absent. A non-finite
            // pixel is equally unusable and never reaches a cast.
            if (!projection.enabled || !projection.pixel.allFinite()) {
                continue;
            }
            std::shared_ptr<CameraView> view = cache.get(projection.camera_id);
            if (!view) {
                continue;
            }
            // `held` keeps the view alive even if the LRU evicts it while the
            // remaining views of this marker are loaded.
            held.push_back(std::move(view));
            BlendView blend;
            blend.view = held.back().get();
            blend.pixel = projection.pixel;
            blend.coded_circular = coded_circular;
            blend.target_kind_known = target_kind_known;
            blend.model = model;
            blends.push_back(blend);
        }
        try {
            edits[slot] = refine_marker(std::move(blends), cloud, options);
        } catch (const std::exception&) {
            MarkerEdit failed;
            failed.failed = true;
            edits[slot] = failed;
        }
        edits[slot].marker_id = marker.id;
    }
    plan.stats.peak_bytes =
        std::max(plan.stats.peak_bytes, cache.peak_bytes());

    // Every selected marker is counted exactly once, and mutations are
    // collected in marker order, so the staged command (and therefore
    // undo/redo) is deterministic.
    std::vector<MarkerEdit> staged;
    for (std::size_t i = 0; i < selected.size(); ++i) {
        MarkerEdit& edit = edits[i];
        if (edit.skipped) {
            ++plan.stats.skipped;
            continue;
        }
        plan.stats.evaluated_projections += edit.evaluated;
        if (edit.failed) {
            ++plan.stats.failed;
            switch (edit.failure) {
                case RefineFailureReason::NoViews:
                    ++plan.stats.failed_no_views;
                    break;
                case RefineFailureReason::NoScale:
                    ++plan.stats.failed_no_scale;
                    break;
                case RefineFailureReason::Triangulation:
                    ++plan.stats.failed_triangulation;
                    break;
                case RefineFailureReason::Consensus:
                case RefineFailureReason::None:
                    ++plan.stats.failed_consensus;
                    break;
            }
            // A failed marker's projections are all left exactly as measured.
            plan.stats.kept_unrefined += edit.evaluated;
            continue;
        }
        if (edit.position.has_value()) {
            const Marker& marker = chunk->markers()[selected[i]];
            if (marker.position.has_value() &&
                (*edit.position - *marker.position).norm() <= 1e-12) {
                edit.position.reset();
            }
        }
        plan.stats.removed_projections += edit.removals.size();
        for (const RefineRemovalReason reason : edit.removal_reasons) {
            switch (reason) {
                case RefineRemovalReason::GeometricGate:
                    ++plan.stats.removed_geometric_gate;
                    break;
                case RefineRemovalReason::FlatPatch:
                    ++plan.stats.removed_flat_patch;
                    break;
                case RefineRemovalReason::LowCorrelation:
                    ++plan.stats.removed_low_correlation;
                    break;
                case RefineRemovalReason::CorrelationMargin:
                    ++plan.stats.removed_correlation_margin;
                    break;
                case RefineRemovalReason::None:
                    break;
            }
        }
        // The shift is measured against the projection the SURVEYOR
        // supplied, which is the only bound a validation protocol can check.
        const Marker& measured = chunk->markers()[selected[i]];
        for (const ProjectionEdit& move : edit.moves) {
            ++plan.stats.moved_projections;
            const MarkerProjection* before =
                find_projection(measured, move.camera_id);
            if (before == nullptr) {
                continue;
            }
            const double shift = (move.pixel - before->pixel).norm();
            plan.stats.shift_sum_px += shift;
            plan.stats.shift_max_px = std::max(plan.stats.shift_max_px, shift);
            if (shift > 2.0) {
                ++plan.stats.shifts_over_2px;
            }
            if (shift > 5.0) {
                ++plan.stats.shifts_over_5px;
            }
            if (shift > 10.0) {
                ++plan.stats.shifts_over_10px;
            }
        }
        const std::size_t decided =
            edit.moves.size() + edit.removals.size();
        plan.stats.kept_unrefined +=
            edit.evaluated > decided ? edit.evaluated - decided : 0;
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
    params.declare({"remove_invalid", ParamType::Bool, false,
                    "Delete a projection whose measured pixel the refined "
                    "candidate contradicts by more than one target radius. "
                    "Off by default: an unconfident refinement always keeps "
                    "the projection where it was measured. Only a geometric "
                    "contradiction is ever removed, never a flat patch or a "
                    "low correlation."});
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
    const std::int32_t chunk_id = document.active_chunk_id();
    RefineMarkersPlan plan =
        plan_refine_markers(document, chunk_id, options);
    progress.update("refine", 1.0);
    progress.update("apply", 0.0);
    if (plan.command) {
        plan.command->apply(document);
    }
    progress.update("apply", 1.0);
    write_refine_meta(document, chunk_id, plan.stats);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info)
            << "RefineMarkers: refined " << plan.stats.refined << ", unchanged "
            << plan.stats.unchanged << ", failed " << plan.stats.failed
            << ", skipped " << plan.stats.skipped
            << ", unreadable cameras " << plan.stats.unreadable_cameras
            << ", removed projections " << plan.stats.removed_projections
            << ", peak " << plan.stats.peak_bytes / (1024 * 1024) << " MiB"
            << " in " << seconds << " s";
        log(LogLevel::Info)
            << "RefineMarkers projections: evaluated "
            << plan.stats.evaluated_projections << ", moved "
            << plan.stats.moved_projections << ", kept unrefined "
            << plan.stats.kept_unrefined << ", removed "
            << plan.stats.removed_projections << " (geometric gate "
            << plan.stats.removed_geometric_gate << ", flat patch "
            << plan.stats.removed_flat_patch << ", low correlation "
            << plan.stats.removed_low_correlation << ", margin "
            << plan.stats.removed_correlation_margin
            << "); shift mean " << format_number(
                   plan.stats.moved_projections == 0
                       ? 0.0
                       : plan.stats.shift_sum_px /
                             static_cast<double>(plan.stats.moved_projections))
            << " px, max " << format_number(plan.stats.shift_max_px)
            << " px, >2px " << plan.stats.shifts_over_2px << ", >5px "
            << plan.stats.shifts_over_5px;
    }
}

}  // namespace alphashape
