#include "common/algorithm/reference/reference_table.hpp"

#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <utility>

#include "common/algorithm/reference/reference_orientation.hpp"
#include "model/crs/local_frame.hpp"
#include "model/crs/transform.hpp"
#include "model/geo/geodesy.hpp"

namespace alphashape::reference {
namespace {

using crs::CoordinateSystem;
using crs::CrsPoint;
using crs::LocalFrame;

bool is_identity(const Eigen::Matrix4d& m) {
    return m == Eigen::Matrix4d::Identity();
}

/// Camera centre C = -A^-1 b for a pose written [A | b] with X_cam = A X_world
/// + b. The general inverse (not -R^T b) keeps a similarity from AlignChunks
/// correct.
Eigen::Vector3d camera_centre(const Transform& pose) {
    const Eigen::Matrix3d a = pose.block<3, 3>(0, 0);
    const Eigen::Vector3d b = pose.block<3, 1>(0, 3);
    return -(a.inverse() * b);
}

bool positive_finite(double value) {
    return std::isfinite(value) && value > 0.0;
}

/// The position a camera reference measures: the projection centre plus the
/// mounted antenna's lever arm, in chunk-local units. This is exactly
/// OptimizeCameras' `A = C + R^T lever` (bundle_adjustment.hpp): the lever is
/// in metres in the camera frame, rotated by the pose and divided by the chunk
/// similarity scale. An unset antenna (or a camera without a sensor) leaves the
/// centre unchanged.
Eigen::Vector3d camera_antenna_position(const Chunk& chunk, const Camera& camera,
                                        const Eigen::Vector3d& centre) {
    const std::vector<Sensor>& sensors = chunk.sensors();
    if (camera.sensor_id < 0 ||
        static_cast<std::size_t>(camera.sensor_id) >= sensors.size()) {
        return centre;
    }
    const Eigen::Vector3d lever = effective_antenna(chunk, camera).camera_offset();
    if (lever.isZero()) {
        return centre;
    }
    const double scale =
        std::cbrt(std::abs(chunk.transform().block<3, 3>(0, 0).determinant()));
    if (!positive_finite(scale)) {
        return centre;
    }
    const Eigen::Matrix3d rotation = camera.pose.block<3, 3>(0, 0);
    return centre + rotation.transpose() * lever / scale;
}

/// The per-axis accuracies collapsed to one 1-sigma figure, or -1 when they
/// carry no usable weight.
double mean_accuracy(const Eigen::Vector3d& accuracy) {
    const double mean = (accuracy.x() + accuracy.y() + accuracy.z()) / 3.0;
    return positive_finite(mean) ? mean : -1.0;
}

/// One axis of a per-axis accuracy, or `fallback` when that axis is unusable.
double axis_accuracy(double value, double fallback) {
    return positive_finite(value) ? value : fallback;
}

Eigen::Vector3d per_axis_accuracy(const Eigen::Vector3d& accuracy, double fallback) {
    return Eigen::Vector3d(axis_accuracy(accuracy.x(), fallback),
                           axis_accuracy(accuracy.y(), fallback),
                           axis_accuracy(accuracy.z(), fallback));
}

std::optional<Eigen::Vector3d> as_vector(const std::optional<CrsPoint>& point) {
    if (!point) {
        return std::nullopt;
    }
    return Eigen::Vector3d(point->x, point->y, point->z);
}

}  // namespace

std::size_t ReferenceTable::control_count() const {
    std::size_t count = 0;
    for (const ReferenceRow& row : rows) {
        // A control point needs a surveyed source to constrain anything.
        if (row.kind == ReferenceRowKind::Marker && row.source.has_value()) {
            ++count;
        }
    }
    return count;
}

ReferenceSummary ReferenceTable::summary() const {
    ReferenceSummary result;
    double camera_sum = 0.0;
    double control_sum = 0.0;
    double check_sum = 0.0;
    std::size_t camera_n = 0;
    std::size_t control_n = 0;
    std::size_t check_n = 0;
    for (const ReferenceRow& row : rows) {
        if (row.kind == ReferenceRowKind::Camera) {
            ++result.camera_count;
        } else {
            ++result.marker_count;
        }
        if (!row.error_m.has_value()) {
            continue;
        }
        const double squared = *row.error_m * *row.error_m;
        if (row.kind == ReferenceRowKind::Camera) {
            if (!row.enabled) {
                continue;
            }
            camera_sum += squared;
            ++camera_n;
        } else if (row.kind == ReferenceRowKind::CheckPoint) {
            check_sum += squared;
            ++check_n;
        } else {
            control_sum += squared;
            ++control_n;
        }
    }
    result.control_count = control_n;
    result.check_count = check_n;
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    result.camera_rmse_m =
        camera_n > 0 ? std::sqrt(camera_sum / static_cast<double>(camera_n)) : kNaN;
    result.control_rmse_m =
        control_n > 0 ? std::sqrt(control_sum / static_cast<double>(control_n)) : kNaN;
    result.check_rmse_m =
        check_n > 0 ? std::sqrt(check_sum / static_cast<double>(check_n)) : kNaN;
    return result;
}

ReferenceTable build_reference_table(const Chunk& chunk,
                                     const CoordinateSystem& display_crs) {
    ReferenceTable table;

    // An unreferenced chunk has an identity transform and no CRS: everything
    // stays in chunk-local units.
    const bool transformed = !is_identity(chunk.transform());
    CoordinateSystem chunk_cs;
    bool geo = false;
    if (transformed) {
        chunk_cs = chunk.coordinate_system();
        geo = chunk_cs.is_set() && !chunk_cs.is_local();
    }
    table.georeferenced = geo;

    CoordinateSystem display = chunk_cs;
    if (display_crs.is_set() && !display_crs.is_local()) {
        display = display_crs;
    }

    LocalFrame frame;  // the identity keeps chunk-local == display
    if (geo) {
        frame = LocalFrame::with_rotation(display,
                                          chunk.transform().block<3, 1>(0, 3),
                                          chunk.transform().block<3, 3>(0, 0));
    }

    // A source is converted from its slot's own CRS, which defaults to the
    // chunk CRS but may be set explicitly. This is independent of whether the
    // chunk geometry is georeferenced: an explicitly requested display CRS
    // still has to be honoured for a chunk whose transform is still identity.
    const bool displayable = display.is_set() && !display.is_local();
    const auto slot_source = [&](const CRS& slot) -> std::optional<CoordinateSystem> {
        if (!displayable) {
            return std::nullopt;
        }
        CoordinateSystem source = slot;
        if (!source.is_set() || source.is_local()) {
            return std::nullopt;
        }
        return source;
    };
    const std::optional<CoordinateSystem> camera_cs = slot_source(chunk.camera_crs());
    const std::optional<CoordinateSystem> marker_cs = slot_source(chunk.marker_crs());
    const auto make_source = [&](const std::optional<CoordinateSystem>& source)
        -> std::optional<crs::CrsTransform> {
        if (!source || *source == display) {
            return std::nullopt;
        }
        return crs::CrsTransform::create(*source, display);
    };
    const std::optional<crs::CrsTransform> camera_src = make_source(camera_cs);
    const std::optional<crs::CrsTransform> marker_src = make_source(marker_cs);

    const auto source_of = [&](const std::optional<Eigen::Vector3d>& value,
                               const std::optional<CoordinateSystem>& source_cs,
                               const std::optional<crs::CrsTransform>& xform)
        -> std::optional<Eigen::Vector3d> {
        if (!value) {
            return std::nullopt;
        }
        if (!displayable) {
            return value;  // published in chunk-local units
        }
        if (!source_cs) {
            return std::nullopt;  // the slot CRS is unknown, so no conversion
        }
        if (!xform) {
            return value;  // already in the display CRS
        }
        return as_vector(xform->forward(CrsPoint{value->x(), value->y(), value->z()}));
    };
    const auto estimate_of = [&](const std::optional<Eigen::Vector3d>& local)
        -> std::optional<Eigen::Vector3d> {
        if (!local) {
            return std::nullopt;
        }
        if (!geo) {
            return local;
        }
        const CrsPoint p = frame.to_crs(*local);
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            return std::nullopt;
        }
        return Eigen::Vector3d(p.x, p.y, p.z);
    };
    // A geographic display CRS reports degrees, so its difference is taken in
    // metres in a proper orthonormal ENU frame at the source. Converting both
    // points to ECEF and rotating the difference avoids the similarity scale
    // contaminating the result (the chunk transform's linear part is S*Q, not
    // a rotation, so subtracting local-frame coordinates scales the error).
    // Heights are ellipsoidal, exactly as OptimizeCameras reads them: a vertical
    // datum in the display CRS is not applied.
    const bool geographic_display =
        geo && (display.kind() == crs::CrsKind::Geographic2D ||
                display.kind() == crs::CrsKind::Geographic3D ||
                display.kind() == crs::CrsKind::Compound);
    const Eigen::Matrix4d internal_to_ecef = chunk.transform();
    const auto error_of = [&](const std::optional<Eigen::Vector3d>& estimated_local,
                              const std::optional<Eigen::Vector3d>& estimated,
                              const std::optional<Eigen::Vector3d>& source)
        -> std::optional<Eigen::Vector3d> {
        if (!estimated || !source) {
            return std::nullopt;
        }
        if (geographic_display) {
            if (!estimated_local) {
                return std::nullopt;
            }
            const Eigen::Vector3d estimate_ecef =
                (internal_to_ecef * estimated_local->homogeneous()).head<3>();
            const geo::Geodetic source_geodetic{source->x(), source->y(), source->z()};
            const Eigen::Vector3d source_ecef = geo::geodetic_to_ecef(source_geodetic);
            const Eigen::Matrix3d enu_rotation =
                geo::ecef_to_enu(source_geodetic).block<3, 3>(0, 0);
            const Eigen::Vector3d enu =
                enu_rotation * (estimate_ecef - source_ecef);
            if (!enu.allFinite()) {
                return std::nullopt;
            }
            return enu;
        }
        return Eigen::Vector3d(*estimated - *source);
    };
    const ReferenceSettings settings = chunk.reference_settings();

    // Camera centres and the observation graph, in chunk-local units.
    std::vector<Eigen::Vector3d> centres;
    centres.reserve(chunk.cameras().size());
    std::map<std::int32_t, std::size_t> camera_index;
    for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
        centres.push_back(camera_centre(chunk.cameras()[i].pose));
        camera_index.emplace(chunk.cameras()[i].id, i);
    }
    std::vector<std::vector<std::size_t>> observed_markers(chunk.cameras().size());
    std::vector<std::vector<std::size_t>> observing_cameras(chunk.markers().size());
    for (std::size_t mi = 0; mi < chunk.markers().size(); ++mi) {
        for (const MarkerProjection& projection : chunk.markers()[mi].projections) {
            const auto it = camera_index.find(projection.camera_id);
            if (it == camera_index.end()) {
                continue;
            }
            observed_markers[it->second].push_back(mi);
            observing_cameras[mi].push_back(it->second);
        }
    }

    // Cameras: source from the GNSS reference, estimate from the pose.
    for (std::size_t ci = 0; ci < chunk.cameras().size(); ++ci) {
        const Camera& camera = chunk.cameras()[ci];
        ReferenceRow row;
        row.id = camera.id;
        row.label = camera.label;
        row.kind = ReferenceRowKind::Camera;
        // The checkbox means "use this reference", which is exactly the flag
        // OptimizeCameras filters on -- not the alignment flag.
        row.enabled = camera.reference.enabled;
        if (camera.reference.location) {
            row.source = source_of(camera.reference.location, camera_cs, camera_src);
        }
        const std::optional<Eigen::Vector3d> estimated_local =
            camera.enabled
                ? std::optional<Eigen::Vector3d>(
                      camera_antenna_position(chunk, camera, centres[ci]))
                : std::nullopt;
        row.estimated = estimate_of(estimated_local);
        row.error = error_of(estimated_local, row.estimated, row.source);
        if (row.error) {
            row.error_m = row.error->norm();
        }
        const double sigma = mean_accuracy(camera.reference.accuracy);
        row.sigma = sigma > 0.0 ? sigma : settings.camera_location_accuracy_m;
        row.sigma_xyz = per_axis_accuracy(camera.reference.accuracy, row.sigma);

        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        std::size_t count = 0;
        for (std::size_t mi : observed_markers[ci]) {
            if (chunk.markers()[mi].position) {
                mean += *chunk.markers()[mi].position;
                ++count;
            }
        }
        if (count > 0) {
            mean /= static_cast<double>(count);
            row.capture_distance_m = (centres[ci] - mean).norm();
        }
        table.rows.push_back(std::move(row));
    }

    // Markers and check points: source from the surveyed reference, estimate
    // from the triangulated position.
    for (std::size_t mi = 0; mi < chunk.markers().size(); ++mi) {
        const Marker& marker = chunk.markers()[mi];
        ReferenceRow row;
        row.id = marker.id;
        row.label = marker.label;
        const MarkerReferenceRole role = marker_role(marker);
        row.kind = role == MarkerReferenceRole::CheckPoint ? ReferenceRowKind::CheckPoint
                                                           : ReferenceRowKind::Marker;
        // The checkbox means "control point": checked holds the marker in the
        // adjustment, unchecked marks it as a check point. `valid` is the
        // projection/filter status and is never edited from here.
        row.enabled = role == MarkerReferenceRole::Gcp;
        if (marker.reference) {
            row.source = source_of(marker.reference, marker_cs, marker_src);
        }
        row.estimated = estimate_of(marker.position);
        row.error = error_of(marker.position, row.estimated, row.source);
        if (row.error) {
            row.error_m = row.error->norm();
        }
        row.projection_count = marker.projections.size();
        row.pinned = marker.pinned;
        const double sigma = mean_accuracy(marker.reference_accuracy);
        row.sigma = sigma > 0.0 ? sigma : settings.marker_location_accuracy_m;
        row.sigma_xyz = per_axis_accuracy(marker.reference_accuracy, row.sigma);

        if (marker.position && !observing_cameras[mi].empty()) {
            double sum = 0.0;
            for (std::size_t ci : observing_cameras[mi]) {
                sum += (centres[ci] - *marker.position).norm();
            }
            row.capture_distance_m =
                sum / static_cast<double>(observing_cameras[mi].size());
        }
        table.rows.push_back(std::move(row));
    }
    return table;
}

}  // namespace alphashape::reference
