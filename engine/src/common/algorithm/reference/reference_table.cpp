#include "common/algorithm/reference/reference_table.hpp"

#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <utility>

#include "model/crs/local_frame.hpp"
#include "model/crs/transform.hpp"

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

MarkerReferenceRole marker_role(const Marker& marker) {
    if (const std::string* value = marker.meta.find("type")) {
        if (*value == "check_point") {
            return MarkerReferenceRole::CheckPoint;
        }
    }
    return MarkerReferenceRole::Gcp;
}

void set_marker_role(Marker& marker, MarkerReferenceRole role) {
    marker.meta.set("type", marker_reference_role_name(role));
}

const char* marker_reference_role_name(MarkerReferenceRole role) {
    return role == MarkerReferenceRole::CheckPoint ? "check_point" : "gcp";
}

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
    const auto error_of = [](const std::optional<Eigen::Vector3d>& estimated,
                             const std::optional<Eigen::Vector3d>& source)
        -> std::optional<Eigen::Vector3d> {
        if (!estimated || !source) {
            return std::nullopt;
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
        row.enabled = camera.enabled;
        if (camera.reference.location) {
            row.source = source_of(camera.reference.location, camera_cs, camera_src);
        }
        if (camera.enabled) {
            row.estimated = estimate_of(centres[ci]);
        }
        row.error = error_of(row.estimated, row.source);
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
        row.kind = marker_role(marker) == MarkerReferenceRole::CheckPoint
                       ? ReferenceRowKind::CheckPoint
                       : ReferenceRowKind::Marker;
        row.enabled = marker.valid;
        if (marker.reference) {
            row.source = source_of(marker.reference, marker_cs, marker_src);
        }
        if (marker.position) {
            row.estimated = estimate_of(marker.position);
        }
        row.error = error_of(row.estimated, row.source);
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
