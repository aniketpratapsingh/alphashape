// georeference.cpp -- estimate the local -> ENU similarity from GNSS camera
// references and rewrite the chunk into a metric ENU frame. See georeference.hpp
// for the convention and the pose-update derivation.

#include "common/algorithm/sfm/georeference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "common/algorithm/sfm/similarity.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape::sfm {
namespace {

namespace geo = alphashape::geo;

/// Default sigma (metres) used to turn a per-axis accuracy into a horizontal one
/// when a reference carries no usable accuracy.
constexpr double kFallbackHorizontalAccuracy = 10.0;

/// Camera centre C = -R^T t of a world->camera pose.
Eigen::Vector3d camera_centre(const Transform& pose) {
    const Eigen::Matrix3d r = pose.block<3, 3>(0, 0);
    const Eigen::Vector3d t = pose.block<3, 1>(0, 3);
    return -r.transpose() * t;
}

/// Median of a non-empty vector (interpolated for an even count).
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

bool all_finite(const Eigen::Vector3d& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

}  // namespace

GeoreferenceReport georeference_chunk(Chunk& chunk) {
    GeoreferenceReport report;

    // -- which cameras carry a usable reference ------------------------------
    const bool already_georeferenced =
        (chunk.transform() - Transform::Identity()).cwiseAbs().maxCoeff() > 1e-12;

    // Candidates: aligned cameras with an enabled, finite GNSS reference. Their
    // poses are in the chunk's CURRENT local frame.
    std::vector<const Camera*> candidates;
    std::vector<Eigen::Vector3d> candidate_centres;
    for (const Camera& camera : chunk.cameras()) {
        if (!camera.enabled || !camera.reference.enabled ||
            !camera.reference.location.has_value()) {
            continue;
        }
        const Eigen::Vector3d& loc = *camera.reference.location;
        if (!all_finite(loc)) {
            continue;
        }
        candidates.push_back(&camera);
        candidate_centres.push_back(camera_centre(camera.pose));
    }
    report.references_total = candidates.size();
    if (candidates.size() < 3) {
        report.reason = "fewer than 3 aligned cameras with a finite GNSS reference";
        return report;
    }

    // -- origin and reference ENU positions ----------------------------------
    // The origin is the geodetic of the mean ECEF of the references: a point a
    // few metres from the camera swarm, so the ENU linearisation is exact to
    // well under a millimetre over the scene extent.
    Eigen::Vector3d mean_ecef = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> reference_enu;
    reference_enu.reserve(candidates.size());
    for (const Camera* camera : candidates) {
        const Eigen::Vector3d& loc = *camera->reference.location;
        mean_ecef += geo::geodetic_to_ecef(geo::Geodetic{loc.x(), loc.y(), loc.z()});
    }
    mean_ecef /= static_cast<double>(candidates.size());
    const geo::Geodetic origin = geo::ecef_to_geodetic(mean_ecef);
    const Transform ecef_to_local = geo::ecef_to_enu(origin);
    for (const Camera* camera : candidates) {
        const Eigen::Vector3d& loc = *camera->reference.location;
        const Eigen::Vector3d ecef =
            geo::geodetic_to_ecef(geo::Geodetic{loc.x(), loc.y(), loc.z()});
        reference_enu.push_back((ecef_to_local * ecef.homogeneous()).head<3>());
    }

    // -- source points: centres carried into the NEW local frame -------------
    // A fresh chunk (identity transform) has an arbitrary SfM local frame. An
    // already-georeferenced one has a local frame that is ENU at the old origin,
    // so its centres are first taken through the old transform to ECEF and into
    // the new ENU; the composite mapping is then a single similarity applied to
    // the stored coordinates, so repeated calls stay stable.
    Transform map = Transform::Identity();
    if (already_georeferenced) {
        map = ecef_to_local * chunk.transform();
    }
    std::vector<Eigen::Vector3d> source;
    source.reserve(candidate_centres.size());
    for (const Eigen::Vector3d& centre : candidate_centres) {
        source.push_back((map * centre.homogeneous()).head<3>());
    }

    // -- threshold: ours -- max(0.5 m, 5 x median horizontal accuracy) --------
    std::vector<double> horizontal;
    horizontal.reserve(candidates.size());
    for (const Camera* camera : candidates) {
        const Eigen::Vector3d& a = camera->reference.accuracy;
        const double h = std::hypot(a.x(), a.y());
        horizontal.push_back(std::isfinite(h) && h > 0.0 ? h : kFallbackHorizontalAccuracy);
    }
    const double threshold = std::max(0.5, 5.0 * median(horizontal));

    // -- similarity fit ------------------------------------------------------
    const SimilarityRansacResult ransac =
        estimate_similarity_ransac(source, reference_enu, threshold);
    if (!ransac.similarity.has_value()) {
        report.reason = "GNSS references are collinear or the similarity fit failed";
        return report;
    }

    // Composite similarity applied to the STORED local coordinates: T = S o map.
    const Similarity& s = *ransac.similarity;
    const Eigen::Matrix3d map_r = map.block<3, 3>(0, 0);
    const Eigen::Vector3d map_t = map.block<3, 1>(0, 3);
    Similarity t_sim;
    t_sim.scale = s.scale;
    t_sim.R = s.R * map_r;
    t_sim.t = s.scale * (s.R * map_t) + s.t;

    // RMS residual of the camera-centre fit over the inliers.
    double sum_sq = 0.0;
    std::size_t inliers = 0;
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (ransac.inliers[i] != 0) {
            sum_sq += (s.apply(source[i]) - reference_enu[i]).squaredNorm();
            ++inliers;
        }
    }
    if (inliers < 3) {
        report.reason = "fewer than 3 reference inliers after the robust fit";
        return report;
    }

    // -- rewrite the chunk ---------------------------------------------------
    for (Camera& camera : chunk.cameras()) {
        if (!camera.enabled) {
            continue;  // unregistered cameras keep their identity pose
        }
        const Eigen::Matrix3d r_c = camera.pose.block<3, 3>(0, 0);
        const Eigen::Vector3d t_c = camera.pose.block<3, 1>(0, 3);
        const Eigen::Matrix3d r_c_new = r_c * t_sim.R.transpose();
        const Eigen::Vector3d t_c_new = t_sim.scale * t_c - r_c_new * t_sim.t;
        camera.pose.block<3, 3>(0, 0) = r_c_new;
        camera.pose.block<3, 1>(0, 3) = t_c_new;
    }
    for (Track& track : chunk.tie_points().tracks) {
        if (track.point.has_value()) {
            *track.point = t_sim.apply(*track.point);
        }
    }
    // Region: axis-aligned, no rotation -- the centre maps like a point and the
    // size is scaled by the similarity scale.
    chunk.region().centre = t_sim.apply(chunk.region().centre);
    chunk.region().size *= t_sim.scale;

    chunk.transform() = geo::enu_to_ecef(origin);
    chunk.crs().epsg = 4326;
    // The GNSS references this fit consumed are lon/lat/alt in that CRS, and the
    // chunk's local frame is now metric ENU expressible in it, so the camera and
    // world slots name it explicitly (M12 §4.4, WP-12.6). The slots only label
    // the numbers; nothing below converts them, and the marker slot keeps
    // following the chunk CRS because nothing here touched markers.
    chunk.set_camera_crs(chunk.crs());
    chunk.set_world_crs(chunk.crs());

    report.applied = true;
    report.references_used = inliers;
    report.rms_error_m = std::sqrt(sum_sq / static_cast<double>(inliers));
    return report;
}

}  // namespace alphashape::sfm
