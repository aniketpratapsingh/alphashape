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
#include <Eigen/Geometry>

#include "base/utils/error.hpp"
#include "common/algorithm/chunk/align_chunks_task.hpp"
#include "common/algorithm/sfm/similarity.hpp"
#include "common/core/chunk_geometry.hpp"
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

/// Rotation carried by a 3x3 block, with a uniform scale divided out.
///
/// The chunk transform may map local units to metres (a similarity rather than
/// a rigid transform) and the AlignChunks pose rule may store a scaled rotation
/// in a camera pose. A lever arm expressed in metres must only be rotated, so
/// both scales are removed explicitly instead of being assumed to cancel.
Eigen::Matrix3d pure_rotation(const Eigen::Matrix3d& m) {
    const double determinant = m.determinant();
    if (!std::isfinite(determinant) || determinant <= 0.0) {
        throw ValueError("georeference: camera-shift block carries no valid rotation");
    }
    return m / std::cbrt(determinant);
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

/// Nearest rotation matrix to `m` by Gram-Schmidt on its columns, so the stored
/// orientation stays orthonormal after composing floating-point rotations.
Eigen::Matrix3d orthonormalised_rotation(const Eigen::Matrix3d& m) {
    Eigen::Vector3d x = m.col(0).normalized();
    Eigen::Vector3d y = m.col(1);
    y -= x * x.dot(y);
    y.normalize();
    Eigen::Vector3d z = x.cross(y);
    Eigen::Matrix3d r;
    r.col(0) = x;
    r.col(1) = y;
    r.col(2) = z;
    return r;
}

/// Rewrite the chunk by one similarity `t_sim` (already composed with any old
/// transform map) and adopt `origin` as the new ENU origin. Poses, track points
/// and the region always move; marker positions move only when asked, because
/// georeference_chunk historically left the marker slot alone.
///
/// The pose update is derived in georeference.hpp: R_c' = R_c R^T,
/// t_c' = s t_c - R_c' t. The camera centre then maps as C' = s R C + t.
///
/// `transform_geometry` is the update_transform path: derived geometry (shapes,
/// trajectories, point-cloud and mesh payloads) follows the same similarity,
/// through the AlignChunks helper, and the region carries its orientation
/// through the same similarity. georeference_chunk passes false so its results
/// stay bit-identical.
void apply_similarity_to_chunk(Chunk& chunk, const Similarity& t_sim,
                               const geo::Geodetic& origin, bool transform_markers,
                               bool transform_geometry) {
    if (transform_geometry) {
        // Reuse the AlignChunks transform on a snapshot, then correct the two
        // conventions that differ here: its camera pose rule post-multiplies by
        // S^-1 (scaling the rotation), and its transform is a composition,
        // whereas this path installs a fresh ENU placement.
        ChunkGeometry geometry = capture_geometry(chunk);
        apply_similarity_to_geometry(geometry, t_sim);
        for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
            const Camera& camera = chunk.cameras()[i];
            if (!camera.enabled) {
                geometry.camera_poses[i] = camera.pose;
                // The helper moved every covariance; an unregistered camera
                // keeps its own, matching its untouched pose.
                geometry.camera_location_covariances[i] = camera.location_covariance;
                continue;
            }
            const Eigen::Matrix3d r_c = camera.pose.block<3, 3>(0, 0);
            const Eigen::Vector3d t_c = camera.pose.block<3, 1>(0, 3);
            const Eigen::Matrix3d r_c_new = r_c * t_sim.R.transpose();
            geometry.camera_poses[i].block<3, 3>(0, 0) = r_c_new;
            geometry.camera_poses[i].block<3, 1>(0, 3) =
                t_sim.scale * t_c - r_c_new * t_sim.t;
        }
        // The helper already moved the region's centre, scaled its extents and
        // composed its orientation with the similarity rotation.
        geometry.transform = geo::enu_to_ecef(origin);
        // Camera-centre, tie-point and marker covariances were already moved by
        // apply_similarity_to_geometry above (C' = s^2 R C R^T). The rotation
        // covariance is in each camera's own angle-axis parametrisation and is
        // left unchanged (see the note in the non-geometry path below).
        apply_geometry(chunk, geometry);
        chunk.crs().epsg = 4326;
        chunk.set_camera_crs(chunk.crs());
        chunk.set_world_crs(chunk.crs());
        return;
    }

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
        // The centre C = -R^T t maps as C' = s R C + t, so its covariance maps
        // by s^2 R C R^T. rotation_covariance is expressed in the camera's own
        // angle-axis parametrisation, which this world-frame change leaves
        // untouched, so it is deliberately not transformed.
        camera.location_covariance = t_sim.scale * t_sim.scale * t_sim.R *
                                     camera.location_covariance * t_sim.R.transpose();
    }
    const double track_scale_sq = t_sim.scale * t_sim.scale;
    for (Track& track : chunk.tie_points().tracks) {
        if (track.point.has_value()) {
            *track.point = t_sim.apply(*track.point);
            if (track.covariance.has_value()) {
                // C' = s^2 R C R^T. The stored matrix is float, so the
                // transform runs in double and casts once.
                const Eigen::Matrix3d transformed =
                    track_scale_sq * t_sim.R *
                    (*track.covariance).cast<double>() * t_sim.R.transpose();
                track.covariance = transformed.cast<float>();
            }
        }
    }
    // Region: the centre maps like a point, the extents scale by |s|, and the
    // orientation composes with the similarity rotation (R' = R_sim * R,
    // re-orthonormalised). A rotated region stays rotated rather than being
    // collapsed to an axis-aligned box.
    chunk.region().centre = t_sim.apply(chunk.region().centre);
    chunk.region().size *= std::abs(t_sim.scale);
    chunk.region().rotation =
        orthonormalised_rotation(t_sim.R * chunk.region().rotation);

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
    // A GNSS reference measures the antenna phase centre, not the projection
    // centre. When a sensor records a mount, the fit predicts where the antenna
    // lies for the current pose and targets the centre that puts the antenna on
    // the measured reference. The prediction needs the fit's own rotation, so
    // it is iterated to convergence. Without antenna data only the plain fit
    // runs, exactly as before WP-13.19.
    const auto antenna_of = [&chunk](const Camera& camera) -> const Antenna* {
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk.sensors().size())) {
            return nullptr;
        }
        const Antenna& antenna =
            chunk.sensors()[static_cast<std::size_t>(camera.sensor_id)].antenna;
        return (antenna.is_set() || antenna.boresight_deg.has_value()) ? &antenna
                                                                       : nullptr;
    };
    bool any_antenna = false;
    for (const Camera* camera : candidates) {
        any_antenna = any_antenna || antenna_of(*camera) != nullptr;
    }

    std::vector<Eigen::Vector3d> fit_targets = reference_enu;
    SimilarityRansacResult ransac =
        estimate_similarity_ransac(source, fit_targets, threshold);
    if (!ransac.similarity.has_value()) {
        report.reason = "GNSS references are collinear or the similarity fit failed";
        return report;
    }
    if (any_antenna) {
        // The camera-frame lever arm is already in metres, so only rotations
        // act on it: the chunk transform and the stored pose may each carry a
        // uniform scale (local units -> metres, and the AlignChunks pose rule),
        // and both scales are divided out explicitly. `map_r` carries the
        // stored local frame into the frame the similarity was fitted in.
        //
        // The correction itself needs the rotation it is fitting for, so it is
        // iterated: each pass rebuilds the targets from the ORIGINAL references
        // using the current fit's rotation, then refits. The first fit's
        // rotation is biased by the uncorrected lever arm, so a single
        // correction leaves centimetres of error for a tangential lever arm.
        constexpr int kMaxPasses = 10;
        constexpr double kRotationTolerance = 1e-12;
        const Eigen::Matrix3d map_r = pure_rotation(map.block<3, 3>(0, 0));
        Eigen::Matrix3d enu_from_source = ransac.similarity->R * map_r;
        for (int pass = 0; pass < kMaxPasses; ++pass) {
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                const Antenna* antenna = antenna_of(*candidates[i]);
                if (antenna == nullptr) {
                    fit_targets[i] = reference_enu[i];
                    continue;
                }
                const Eigen::Matrix3d r_c =
                    pure_rotation(candidates[i]->pose.block<3, 3>(0, 0));
                const Eigen::Vector3d shift =
                    enu_from_source * r_c.transpose() * antenna->camera_offset();
                fit_targets[i] = reference_enu[i] - shift;
            }
            ransac = estimate_similarity_ransac(source, fit_targets, threshold);
            if (!ransac.similarity.has_value()) {
                report.reason =
                    "GNSS references are collinear or the antenna-aware fit failed";
                return report;
            }
            const Eigen::Matrix3d next = ransac.similarity->R * map_r;
            const double change = (next - enu_from_source).norm();
            enu_from_source = next;
            if (change < kRotationTolerance) {
                break;
            }
        }
    }

    // Composite similarity applied to the STORED local coordinates: T = S o map.
    // `map` carries a uniform scale when a local unit is not a metre, so the
    // two scales multiply and the composite keeps a pure rotation.
    const Similarity& s = *ransac.similarity;
    const Eigen::Matrix3d map_block = map.block<3, 3>(0, 0);
    const Eigen::Matrix3d map_r = pure_rotation(map_block);
    const Eigen::Vector3d map_t = map.block<3, 1>(0, 3);
    Similarity t_sim;
    t_sim.scale = s.scale * std::cbrt(map_block.determinant());
    t_sim.R = s.R * map_r;
    t_sim.t = s.scale * (s.R * map_t) + s.t;

    // RMS residual of the camera-centre fit over the inliers (against the
    // antenna-corrected targets when one was applied).
    double sum_sq = 0.0;
    std::size_t inliers = 0;
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (ransac.inliers[i] != 0) {
            sum_sq += (s.apply(source[i]) - fit_targets[i]).squaredNorm();
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
    apply_similarity_to_chunk(chunk, t_sim, origin, /*transform_markers=*/false,
                              /*transform_geometry=*/false);

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
    // `map` carries a uniform scale when a local unit is not a metre, so the
    // two scales multiply and the composite keeps a pure rotation.
    const Eigen::Matrix3d map_block = map.block<3, 3>(0, 0);
    const Eigen::Matrix3d map_r = pure_rotation(map_block);
    const Eigen::Vector3d map_t = map.block<3, 1>(0, 3);
    Similarity t_sim;
    t_sim.scale = s.scale * std::cbrt(map_block.determinant());
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
    apply_similarity_to_chunk(chunk, t_sim, origin, /*transform_markers=*/true,
                              /*transform_geometry=*/true);

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
