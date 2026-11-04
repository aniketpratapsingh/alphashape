// georeference.cpp -- estimate the local -> ENU similarity from GNSS camera
// references and rewrite the chunk into a metric ENU frame. See georeference.hpp
// for the convention and the pose-update derivation.

#include "common/algorithm/sfm/georeference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
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

/// True when a slot's CRS is unset or declares a geographic (lon/lat) datum.
/// update_transform reads references as lon/lat/ellipsoidal height, so a slot
/// naming a projected CRS cannot be consumed. epsg 0 means "unset": nothing
/// contradicts the numbers. 4326 is the canonical geographic code and the
/// 4000-4999 block is the geographic EPSG range that contains it.
bool crs_is_geographic(const auto& crs) {
    const int code = static_cast<int>(crs.epsg);
    if (code != 0) {
        return code == 4326 || (code >= 4000 && code <= 4999);
    }
    // No EPSG code: a WKT that names a projected CRS is still a refusal, and an
    // empty slot declares nothing.
    const std::string& wkt = crs.wkt;
    if (!wkt.empty()) {
        return wkt.find("PROJCRS") == std::string::npos &&
               wkt.find("PROJECTEDCRS") == std::string::npos;
    }
    return true;
}

/// True when the points are not collinear: the farthest pair defines a line and
/// at least one other point deviates from it by more than a relative tolerance.
/// Mirrors the gauge test used by OptimizeCameras.
bool points_not_collinear(const std::vector<Eigen::Vector3d>& points) {
    if (points.size() < 3) {
        return false;
    }
    std::size_t a = 0;
    std::size_t b = 1;
    double best = -1.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = i + 1; j < points.size(); ++j) {
            const double d = (points[i] - points[j]).squaredNorm();
            if (d > best) {
                best = d;
                a = i;
                b = j;
            }
        }
    }
    const Eigen::Vector3d dir = points[a] - points[b];
    const double len = dir.norm();
    if (!(len > 0.0)) {
        return false;
    }
    const Eigen::Vector3d u = dir / len;
    double max_perp = 0.0;
    for (const Eigen::Vector3d& p : points) {
        const Eigen::Vector3d v = p - points[a];
        max_perp = std::max(max_perp, (v - v.dot(u) * u).norm());
    }
    return max_perp > 1e-3 * len;
}

/// Rewrite the chunk by one similarity `t_sim` (already composed with any old
/// transform map) and adopt `origin` as the new ENU origin. Poses, track points
/// and the region always move; marker positions move only when asked, because
/// georeference_chunk historically left the marker slot alone.
///
/// The pose update is derived in georeference.hpp: R_c' = R_c R^T,
/// t_c' = s t_c - R_c' t. The camera centre then maps as C' = s R C + t.
void apply_similarity_to_chunk(Chunk& chunk, const Similarity& t_sim,
                               const geo::Geodetic& origin, bool transform_markers) {
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

    if (transform_markers) {
        for (Marker& marker : chunk.markers()) {
            if (marker.position.has_value()) {
                *marker.position = t_sim.apply(*marker.position);
            }
        }
    }

    chunk.transform() = geo::enu_to_ecef(origin);
    chunk.crs().epsg = 4326;
    // The GNSS references this fit consumed are lon/lat/alt in that CRS, and the
    // chunk's local frame is now metric ENU expressible in it, so the camera and
    // world slots name it explicitly (M12 §4.4, WP-12.6). The slots only label
    // the numbers; nothing below converts them. The marker slot keeps following
    // the chunk CRS unless the caller explicitly transformed marker positions.
    chunk.set_camera_crs(chunk.crs());
    chunk.set_world_crs(chunk.crs());
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
    // Result-identical to the previous inline block; marker positions stay put
    // because this path never touched them.
    apply_similarity_to_chunk(chunk, t_sim, origin, /*transform_markers=*/false);

    report.applied = true;
    report.references_used = inliers;
    report.rms_error_m = std::sqrt(sum_sq / static_cast<double>(inliers));
    return report;
}

GeoreferenceReport update_transform(Chunk& chunk) {
    GeoreferenceReport report;

    struct Sample {
        Eigen::Vector3d local;
        Eigen::Vector3d reference;  ///< lon_deg, lat_deg, alt_m
        double sigma = kFallbackHorizontalAccuracy;
        std::int32_t marker_id = -1;  ///< -1 for a camera reference
    };
    struct CheckSample {
        std::int32_t marker_id = -1;
        Eigen::Vector3d reference;      ///< lon_deg, lat_deg, alt_m
        Eigen::Vector3d reference_enu;  ///< in the new frame
    };
    std::vector<Sample> samples;
    std::vector<CheckSample> checks;

    const bool already_georeferenced =
        (chunk.transform() - Transform::Identity()).cwiseAbs().maxCoeff() > 1e-12;
    const auto sigma_of = [](const Eigen::Vector3d& accuracy) {
        const double s = accuracy.mean();
        return std::isfinite(s) && s > 0.0 ? s : kFallbackHorizontalAccuracy;
    };

    // (a) aligned cameras with a full, finite reference: centre <-> reference.
    for (const Camera& camera : chunk.cameras()) {
        if (!camera.enabled || !camera.reference.enabled ||
            !camera.reference.location.has_value()) {
            continue;
        }
        const Eigen::Vector3d& loc = *camera.reference.location;
        if (!all_finite(loc)) {
            continue;
        }
        const Eigen::Vector3d centre = camera_centre(camera.pose);
        if (!all_finite(centre)) {
            continue;  // a garbage pose contributes no usable source point
        }
        samples.push_back({centre, loc, sigma_of(camera.reference.accuracy), -1});
    }
    // (b) Regular control markers with a position and a full reference. Check
    // points are held out of the fit and measured only after it.
    for (const Marker& marker : chunk.markers()) {
        if (marker.type != MarkerType::Regular || !marker.position.has_value() ||
            !marker.reference.has_value()) {
            continue;
        }
        const Eigen::Vector3d& loc = *marker.reference;
        if (!all_finite(loc) || !all_finite(*marker.position)) {
            continue;  // non-finite source/derived positions poison every fit
        }
        const std::string* role = marker.meta.find("type");
        if (role != nullptr && *role == "check_point") {
            checks.push_back({marker.id, loc, Eigen::Vector3d::Zero()});
            continue;
        }
        samples.push_back(
            {*marker.position, loc, sigma_of(marker.reference_accuracy), marker.id});
    }

    // -- reference CRS guard ------------------------------------------------
    // The numbers gathered above are interpreted as lon/lat/ellipsoidal height.
    // A slot that names a projected CRS cannot be consumed here, so refuse
    // before any mutation. Only the slots that actually contribute are checked:
    // a lone camera reference must not be blocked by a projected marker slot,
    // and vice versa.
    const bool uses_cameras = std::any_of(
        samples.begin(), samples.end(), [](const Sample& s) { return s.marker_id < 0; });
    const bool uses_markers =
        !checks.empty() || std::any_of(samples.begin(), samples.end(),
                                       [](const Sample& s) { return s.marker_id >= 0; });
    if (uses_cameras && !crs_is_geographic(chunk.camera_crs())) {
        report.reason =
            "camera reference CRS is not geographic; references must be lon/lat";
        return report;
    }
    if (uses_markers && !crs_is_geographic(chunk.marker_crs())) {
        report.reason = "marker reference CRS is not geographic; references must be lon/lat";
        return report;
    }

    if (samples.size() < 3) {
        report.reason =
            "need 3 non-collinear reference points (cameras or markers); scale bars alone "
            "cannot georeference";
        return report;
    }

    // -- origin from the fit references (check points never move it) ---------
    Eigen::Vector3d mean_ecef = Eigen::Vector3d::Zero();
    for (const Sample& s : samples) {
        mean_ecef += geo::geodetic_to_ecef(
            geo::Geodetic{s.reference.x(), s.reference.y(), s.reference.z()});
    }
    mean_ecef /= static_cast<double>(samples.size());
    const geo::Geodetic origin = geo::ecef_to_geodetic(mean_ecef);
    const Transform ecef_to_local = geo::ecef_to_enu(origin);

    Transform map = Transform::Identity();
    if (already_georeferenced) {
        map = ecef_to_local * chunk.transform();
    }
    std::vector<Eigen::Vector3d> source;
    std::vector<Eigen::Vector3d> reference_enu;
    std::vector<double> weights;
    source.reserve(samples.size());
    reference_enu.reserve(samples.size());
    weights.reserve(samples.size());
    for (const Sample& s : samples) {
        source.push_back((map * s.local.homogeneous()).head<3>());
        const Eigen::Vector3d ecef = geo::geodetic_to_ecef(
            geo::Geodetic{s.reference.x(), s.reference.y(), s.reference.z()});
        reference_enu.push_back((ecef_to_local * ecef.homogeneous()).head<3>());
        weights.push_back(1.0 / (s.sigma * s.sigma));
    }
    for (CheckSample& c : checks) {
        const Eigen::Vector3d ecef = geo::geodetic_to_ecef(
            geo::Geodetic{c.reference.x(), c.reference.y(), c.reference.z()});
        c.reference_enu = (ecef_to_local * ecef.homogeneous()).head<3>();
    }

    // Both sides must hold a non-degenerate configuration: the sources are the
    // chunk-local points the similarity acts on, the destinations the ENU
    // positions it must reproduce. Either being collinear (or coincident)
    // leaves the rotation/scale under-determined.
    if (!points_not_collinear(source) || !points_not_collinear(reference_enu)) {
        report.reason = "reference positions are degenerate (collinear or coincident)";
        return report;
    }

    // -- initial model: RANSAC with one global band --------------------------
    // The band only seeds the initial inlier set. A single global threshold
    // cannot serve both tight markers and coarse camera references, so the
    // final mask comes from the per-point gate below.
    std::vector<double> sigmas;
    sigmas.reserve(samples.size());
    for (const Sample& s : samples) {
        sigmas.push_back(s.sigma);
    }
    const double threshold = std::max(0.05, 5.0 * median(sigmas));

    constexpr std::uint64_t kSeed = 0x5eed;
    const SimilarityRansacResult ransac =
        estimate_similarity_ransac(source, reference_enu, threshold, kSeed);
    if (!ransac.similarity.has_value()) {
        report.reason = "similarity fit failed";
        return report;
    }

    // Weighted refit on `mask`, rejecting a degenerate result. `reason` lets
    // the caller tell "too few points" from a numerical failure.
    const auto refit_on = [&](const std::vector<std::uint8_t>& mask,
                              std::string& reason) -> std::optional<Similarity> {
        std::vector<Eigen::Vector3d> in_source;
        std::vector<Eigen::Vector3d> in_reference;
        std::vector<double> in_weight;
        for (std::size_t i = 0; i < source.size(); ++i) {
            if (mask[i] == 0) {
                continue;
            }
            in_source.push_back(source[i]);
            in_reference.push_back(reference_enu[i]);
            in_weight.push_back(weights[i]);
        }
        if (in_source.size() < 3) {
            reason = "fewer than 3 reference inliers after the robust fit";
            return std::nullopt;
        }
        const std::optional<Similarity> fit =
            estimate_similarity_weighted(in_source, in_reference, in_weight);
        if (!fit.has_value() || !(fit->scale > 0.0) || !std::isfinite(fit->scale) ||
            !fit->R.allFinite() || !fit->t.allFinite()) {
            reason = "similarity fit failed";
            return std::nullopt;
        }
        return fit;
    };

    // -- per-point gate: refit, reclassify, stop when the mask settles -------
    // Each sample's own sigma decides whether it is an inlier; the 0.05 m floor
    // keeps exact references from being rejected by round-off.
    std::vector<std::uint8_t> mask(source.size(), 0);
    for (std::size_t i = 0; i < source.size(); ++i) {
        mask[i] = ransac.inliers[i] != 0 ? 1 : 0;
    }
    std::string fail_reason;
    std::optional<Similarity> fitted = refit_on(mask, fail_reason);
    if (!fitted.has_value()) {
        report.reason = fail_reason;
        return report;
    }
    for (int iteration = 0; iteration < 5; ++iteration) {
        std::vector<std::uint8_t> next(source.size(), 0);
        std::size_t inliers = 0;
        for (std::size_t i = 0; i < source.size(); ++i) {
            const double gate = std::max(5.0 * samples[i].sigma, 0.05);
            if ((fitted->apply(source[i]) - reference_enu[i]).norm() <= gate) {
                next[i] = 1;
                ++inliers;
            }
        }
        if (inliers < 3) {
            report.reason = "fewer than 3 reference inliers after the robust fit";
            return report;
        }
        if (next == mask) {
            break;  // `fitted` was computed on exactly this mask
        }
        mask = next;
        fitted = refit_on(mask, fail_reason);
        if (!fitted.has_value()) {
            report.reason = fail_reason;
            return report;
        }
    }
    const Similarity& s = *fitted;

    // Composite similarity applied to the STORED local coordinates: T = S o map.
    const Eigen::Matrix3d map_r = map.block<3, 3>(0, 0);
    const Eigen::Vector3d map_t = map.block<3, 1>(0, 3);
    Similarity t_sim;
    t_sim.scale = s.scale;
    t_sim.R = s.R * map_r;
    t_sim.t = s.scale * (s.R * map_t) + s.t;

    // -- report from the FINAL mask ------------------------------------------
    double sum_sq = 0.0;
    std::size_t inlier_count = 0;
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (mask[i] == 0) {
            if (samples[i].marker_id >= 0) {
                report.rejected_markers.push_back(samples[i].marker_id);
            }
            continue;
        }
        sum_sq += (s.apply(source[i]) - reference_enu[i]).squaredNorm();
        if (samples[i].marker_id >= 0) {
            ++report.control_points_used;
        } else {
            ++report.cameras_used;
        }
        ++inlier_count;
    }
    report.rms_control_m = std::sqrt(sum_sq / static_cast<double>(inlier_count));
    report.references_total = samples.size();
    report.references_used = inlier_count;
    report.rms_error_m = report.rms_control_m;

    // -- apply uniformly, including every marker position -------------------
    apply_similarity_to_chunk(chunk, t_sim, origin, /*transform_markers=*/true);

    // -- check-point errors, measured on the transformed positions ----------
    double check_sum = 0.0;
    for (const CheckSample& c : checks) {
        const Marker* marker = nullptr;
        for (const Marker& m : chunk.markers()) {
            if (m.id == c.marker_id) {
                marker = &m;
                break;
            }
        }
        if (marker == nullptr || !marker->position.has_value()) {
            continue;
        }
        const double error = (*marker->position - c.reference_enu).norm();
        report.check_errors.emplace_back(c.marker_id, error);
        check_sum += error * error;
        report.max_check_m = std::max(report.max_check_m, error);
    }
    report.check_points_used = report.check_errors.size();
    if (!report.check_errors.empty()) {
        report.rms_check_m =
            std::sqrt(check_sum / static_cast<double>(report.check_errors.size()));
    }

    report.scale = t_sim.scale;
    report.applied = true;
    return report;
}

}  // namespace alphashape::sfm
