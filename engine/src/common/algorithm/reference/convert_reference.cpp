#include "common/algorithm/reference/convert_reference.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"
#include "common/commands/command.hpp"
#include "common/core/document.hpp"
#include "model/crs/transform.hpp"
#include "model/crs/vertical.hpp"

namespace alphashape {
namespace reference {
namespace {

using crs::CrsPoint;
using crs::CoordinateSystem;

bool is_geographic(const CoordinateSystem& cs) {
    return cs.kind() == crs::CrsKind::Geographic2D || cs.kind() == crs::CrsKind::Geographic3D;
}

/// The store id a datum is looked up under: its grid file stem when the grid is
/// set, otherwise the datum name. `VerticalCrs::grid` is a grid id in the
/// proj-data sense ("us_nga_egm96_15.tif"), while a project store keys on the
/// id `GeoidStore::add()` was given, which for a custom grid is the file stem.
std::string geoid_id(const crs::VerticalCrs& vertical) {
    if (!vertical.grid.empty()) {
        std::string id = vertical.grid;
        const std::size_t dot = id.find_last_of('.');
        const std::size_t slash = id.find_last_of("/\\");
        if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
            id.erase(dot);
        }
        return id;
    }
    return vertical.name;
}

/// The grid for `datum` from `store`, by id first and then by the datum name a
/// store entry carries. A project may key a grid "EGM96" while the datum's PROJ
/// name is "EGM96 height", so the name match is what makes a store the user
/// filled by hand work without renaming entries.
std::shared_ptr<const crs::GeoidGrid> grid_for(const crs::GeoidStore& store,
                                               const crs::VerticalCrs& datum) {
    if (std::shared_ptr<const crs::GeoidGrid> by_id = store.grid(geoid_id(datum))) {
        return by_id;
    }
    for (const crs::GeoidModel& model : store.list()) {
        if (model.name == datum.name) {
            if (std::shared_ptr<const crs::GeoidGrid> by_name = store.grid(model.id)) {
                return by_name;
            }
        }
    }
    return nullptr;
}

/// A place in the chunk, for error messages ("camera 4", "marker \"gcp1\"").
std::string where(const Camera& camera) {
    return "camera " + std::to_string(camera.id);
}

std::string where(const Marker& marker) {
    return "marker \"" + marker.label + "\"";
}

// --- chunk 2
/// Build the horizontal+vertical converter for one slot pair.
///
/// With no vertical datum set on either side this is a plain `CrsTransform`
/// and heights stay ellipsoidal. With a datum, PROJ's compound pipeline is used
/// when every vertical CRS involved is EPSG-defined; only a datum with no EPSG
/// code (a project geoid) is taken from `geoids` and applied by
/// `VerticalTransform::custom` around the horizontal transform.
struct Converter {
    /// The plain horizontal transform; unset when a vertical one handles both.
    std::optional<crs::CrsTransform> horizontal;
    std::optional<crs::VerticalTransform> vertical;

    std::optional<CrsPoint> forward(const CrsPoint& point) const {
        if (vertical) {
            return vertical->forward(point);
        }
        return horizontal->forward(point);
    }
};

Converter build_converter(const CoordinateSystem& source, const CoordinateSystem& target,
                          const char* role, const crs::GeoidStore* geoids) {
    Converter out;
    const std::optional<crs::VerticalCrs> src_v = source.vertical();
    const std::optional<crs::VerticalCrs> dst_v = target.vertical();
    const bool src_set = src_v && src_v->is_set();
    const bool dst_set = dst_v && dst_v->is_set();
    if (!src_set && !dst_set) {
        out.horizontal = crs::CrsTransform::create(source, target);
        return out;
    }
    const crs::VerticalCrs src_datum = src_set ? *src_v : crs::VerticalCrs{};
    const crs::VerticalCrs dst_datum = dst_set ? *dst_v : crs::VerticalCrs{};
    // Which route applies is decided by the description, not by what the store
    // happens to hold. An EPSG vertical CRS has an authority-defined geoid and
    // a PROJ compound pipeline is its exact `cs2cs` answer, so it always takes
    // that route; a store grid never shadows it, because a project grid keyed
    // under the same name may be a different model or a bad copy. Only a datum
    // with no EPSG code (a project geoid, identifiable by name only) needs the
    // store's grid and `VerticalTransform::custom`.
    const bool src_epsg = !src_set || crs::lookup_vertical_crs(src_datum).epsg != 0;
    const bool dst_epsg = !dst_set || crs::lookup_vertical_crs(dst_datum).epsg != 0;
    if (src_epsg && dst_epsg) {
        out.vertical = crs::VerticalTransform::compound(source, target);
        return out;
    }
    std::shared_ptr<const crs::GeoidGrid> stored;
    if (geoids != nullptr) {
        for (const crs::VerticalCrs* datum : {&src_datum, &dst_datum}) {
            if (!datum->is_set()) {
                continue;
            }
            if (std::shared_ptr<const crs::GeoidGrid> grid = grid_for(*geoids, *datum)) {
                stored = std::move(grid);
                break;
            }
        }
    }
    if (stored) {
        out.vertical = crs::VerticalTransform::custom(source, target, src_datum, dst_datum,
                                                     std::move(stored),
                                                     CoordinateSystem::from_epsg(4979));
        return out;
    }
    // No EPSG code and no usable grid: fail by name. A silently skipped geoid
    // would move every height by tens of metres. The datum to name is the one
    // without an EPSG code; at least one side is set here, so when the source is
    // EPSG-defined the target must be the custom one.
    const crs::VerticalCrs& needed =
        (src_set && crs::lookup_vertical_crs(src_datum).epsg == 0) ? src_datum : dst_datum;
    if (geoids == nullptr) {
        throw Error(std::string("convert the ") + role + " references: vertical datum \"" +
                    needed.name + "\" needs the geoid grid \"" + geoid_id(needed) +
                    "\", but no geoid store was supplied");
    }
    throw Error(std::string("convert the ") + role + " references: vertical datum \"" +
                needed.name + "\" needs the geoid grid \"" + geoid_id(needed) +
                "\", which the store does not carry");
}

/// Convert one stored (lon/lat/alt) or (east/north/height) triple. Returns
/// nullopt when the value is entirely absent; a NaN altitude is preserved.
std::optional<Eigen::Vector3d> convert_point(const std::optional<Eigen::Vector3d>& value,
                                             const Converter& converter, const std::string& at) {
    if (!value.has_value()) {
        return std::nullopt;
    }
    if (!std::isfinite(value->x()) || !std::isfinite(value->y())) {
        throw ValueError(at + ": reference has a non-finite horizontal position");
    }
    const bool no_height = std::isnan(value->z());
    // The transform needs a finite z. For a "no height" reference the
    // horizontal answer does not depend on it in any CRS the engine accepts,
    // and the NaN is restored afterwards.
    const CrsPoint in{value->x(), value->y(), no_height ? 0.0 : value->z()};
    const std::optional<CrsPoint> out = converter.forward(in);
    if (!out) {
        throw Error(at + ": the reference cannot be converted to the target CRS "
                          "(outside its area of use, or no geoid coverage)");
    }
    return Eigen::Vector3d(out->x, out->y, no_height ? std::numeric_limits<double>::quiet_NaN()
                                                     : out->z);
}

}  // namespace

crs::RotationOrder rotation_order_of(const Chunk& chunk) {
    const std::string* value = chunk.meta().find(kRotationOrderKey);
    if (value == nullptr) {
        return crs::RotationOrder::ZYX;
    }
    crs::RotationOrder order = crs::RotationOrder::ZYX;
    if (!crs::parse_rotation_order(*value, &order)) {
        throw ValueError(std::string("chunk meta \"") + kRotationOrderKey + "\" is \"" + *value +
                         "\", not a rotation order (XYZ, XZY, YXZ, YZX, ZXY, ZYX)");
    }
    return order;
}

// --- chunk 2
namespace {

/// Central-difference step for one axis, in the CRS's own units for that axis.
/// A geographic CRS has degrees for x/y and metres for z (height), so the two
/// kinds of axis need different steps: 1e-5 deg is about 1.1 m -- well above
/// the 1e-15 deg resolution of a longitude, small enough that curvature stays
/// negligible -- and 1 m is above the 1e-9 m resolution of an ECEF coordinate.
double step_for(const CoordinateSystem& cs, int axis) {
    return is_geographic(cs) && axis < 2 ? 1e-5 : 1.0;
}

/// A right-handed orthonormal basis of one CRS's local axes, in ECEF, at
/// `point` (given in that CRS). Column i is the direction of the CRS's axis i.
Eigen::Matrix3d local_basis(const CoordinateSystem& cs, const crs::CrsPoint& point) {
    const crs::CrsTransform to_ecef =
        crs::CrsTransform::create(cs, CoordinateSystem::from_epsg(4978));
    const auto ecef = [&](const CrsPoint& p) {
        const std::optional<CrsPoint> out = to_ecef.forward(p);
        if (!out) {
            throw Error("the reference point cannot be mapped to ECEF");
        }
        return Eigen::Vector3d(out->x, out->y, out->z);
    };
    std::array<Eigen::Vector3d, 3> columns{};
    for (int axis = 0; axis < 3; ++axis) {
        const double step = step_for(cs, axis);
        CrsPoint plus = point;
        CrsPoint minus = point;
        double* target_plus = axis == 0 ? &plus.x : axis == 1 ? &plus.y : &plus.z;
        double* target_minus = axis == 0 ? &minus.x : axis == 1 ? &minus.y : &minus.z;
        *target_plus += step;
        *target_minus -= step;
        columns[static_cast<std::size_t>(axis)] = (ecef(plus) - ecef(minus)) / (2.0 * step);
    }
    if (columns[2].norm() < 1e-9) {
        // A 2D CRS maps every height to the same point, so its up axis is not
        // observable. Define it as east x north, which is the ellipsoid normal
        // for the geographic case and the grid normal otherwise.
        columns[2] = columns[0].cross(columns[1]);
    }
    Eigen::Matrix3d basis;
    basis.col(0) = columns[0].normalized();
    Eigen::Vector3d second = columns[1] - columns[1].dot(basis.col(0)) * basis.col(0);
    basis.col(1) = second.normalized();
    basis.col(2) = basis.col(0).cross(basis.col(1));
    return basis;
}

}  // namespace

Eigen::Matrix3d frame_rotation(const CoordinateSystem& source, const CoordinateSystem& target,
                               const crs::CrsPoint& point) {
    if (source == target) {
        return Eigen::Matrix3d::Identity();
    }
    if (source.is_local() || target.is_local() || !source.is_set() || !target.is_set()) {
        throw ValueError("frame_rotation: both coordinate systems must be set and not the local "
                         "engineering frame");
    }
    const crs::CrsTransform to_target = crs::CrsTransform::create(source, target);
    const std::optional<CrsPoint> there = to_target.forward(point);
    if (!there) {
        throw Error("frame_rotation: the reference point cannot be converted to the target CRS");
    }
    const Eigen::Matrix3d from_basis = local_basis(source, point);
    const Eigen::Matrix3d to_basis = local_basis(target, *there);
    // d_source = from_basis^T * ecef, d_target = to_basis^T * ecef, hence
    // d_target = to_basis^T * from_basis * d_source.
    return to_basis.transpose() * from_basis;
}

crs::EulerAngles convert_orientation(const crs::EulerAngles& angles,
                                     const CoordinateSystem& source,
                                     const CoordinateSystem& target,
                                     const crs::CrsPoint& point) {
    const Eigen::Matrix3d rotation = frame_rotation(source, target, point);
    const Eigen::Matrix3d in_target = rotation * crs::to_matrix(angles);
    return crs::from_matrix(in_target, angles.order);
}

Eigen::Matrix3d body_to_camera() {
    // Rows are the camera axes written in body coordinates: camera +X (image
    // right) = body -y, camera +Y (image down) = body -z, camera +Z (boresight)
    // = body +x. v_camera = body_to_camera() * v_body.
    Eigen::Matrix3d to_camera;
    to_camera << 0.0, -1.0, 0.0,   //
        0.0, 0.0, -1.0,            //
        1.0, 0.0, 0.0;
    return to_camera;
}

Eigen::Matrix3d body_to_camera_nadir() {
    // A nadir camera is the level camera pitched -90 degrees about body y in
    // the mount: body_to_camera() * Ry(-90 deg). That body-frame rotation is
    // the inverse of the +90 degrees a -90 degree pitch contributes to R_wb
    // (R_wb composes Ry(-pitch)).
    Eigen::Matrix3d mount;
    mount << 0.0, 0.0, -1.0,   //
        0.0, 1.0, 0.0,         //
        1.0, 0.0, 0.0;         // Ry(-90 deg)
    return body_to_camera() * mount;
}

// --- chunk 3
const char* reference_kind_name(ReferenceKind kind) {
    switch (kind) {
        case ReferenceKind::Cameras:
            return "Camera";
        case ReferenceKind::Markers:
            return "Marker";
        case ReferenceKind::Both:
            break;
    }
    return "Camera and Marker";
}

namespace {

/// Convert every camera reference into a detached list; the chunk is not
/// touched. Returns nullopt entries for cameras with no reference.
std::vector<std::optional<Eigen::Vector3d>> converted_camera_locations(
    const Chunk& chunk, const CoordinateSystem& target, const crs::GeoidStore* geoids) {
    const Converter converter = build_converter(chunk.camera_crs(), target, "camera", geoids);
    std::vector<std::optional<Eigen::Vector3d>> converted;
    converted.reserve(chunk.cameras().size());
    for (const Camera& camera : chunk.cameras()) {
        converted.push_back(convert_point(camera.reference.location, converter, where(camera)));
    }
    return converted;
}

std::vector<std::optional<Eigen::Vector3d>> converted_marker_references(
    const Chunk& chunk, const CoordinateSystem& target, const crs::GeoidStore* geoids) {
    const Converter converter = build_converter(chunk.marker_crs(), target, "marker", geoids);
    std::vector<std::optional<Eigen::Vector3d>> converted;
    converted.reserve(chunk.markers().size());
    for (const Marker& marker : chunk.markers()) {
        converted.push_back(convert_point(marker.reference, converter, where(marker)));
    }
    return converted;
}

}  // namespace

void convert_reference(Chunk& chunk, const CoordinateSystem& target, ReferenceKind kind,
                       const crs::GeoidStore* geoids) {
    if (target.is_local() || !target.is_set()) {
        throw ValueError("convert_reference: the target must be a set coordinate system, not the "
                         "local engineering frame");
    }
    // The configured order governs any orientation angle in this chunk, so a
    // corrupt `reference/rotation_order` is reported here rather than much
    // later at an import that would otherwise silently mean another rotation.
    (void)rotation_order_of(chunk);
    const bool cameras = kind == ReferenceKind::Cameras || kind == ReferenceKind::Both;
    const bool markers = kind == ReferenceKind::Markers || kind == ReferenceKind::Both;
    // Compute both slots' new values BEFORE writing either, so a failure in the
    // second slot cannot leave the first one half-converted.
    std::vector<std::optional<Eigen::Vector3d>> camera_values;
    std::vector<std::optional<Eigen::Vector3d>> marker_values;
    if (cameras) {
        camera_values = converted_camera_locations(chunk, target, geoids);
    }
    if (markers) {
        marker_values = converted_marker_references(chunk, target, geoids);
    }
    if (cameras) {
        for (std::size_t i = 0; i < chunk.cameras().size(); ++i) {
            chunk.cameras()[i].reference.location = camera_values[i];
        }
        chunk.set_camera_crs(CRS(target));
    }
    if (markers) {
        for (std::size_t i = 0; i < chunk.markers().size(); ++i) {
            chunk.markers()[i].reference = marker_values[i];
        }
        chunk.set_marker_crs(CRS(target));
    }
}

// --- chunk 4
namespace {

/// One undo entry for a reference conversion: every camera reference, marker
/// reference, and the two CRS slots, restored exactly.
///
/// StagedMutation's contract: stage() captures the pre-state, commit()
/// performs the edit. commit() reuses `convert_reference`, so a redo produces
/// byte-identical values to the first apply (the conversion is deterministic).
class ConvertReferenceCommand : public StagedMutation {
public:
    ConvertReferenceCommand(std::int32_t chunk_id, CoordinateSystem target, ReferenceKind kind,
                            const crs::GeoidStore* geoids)
        : chunk_id_(chunk_id),
          target_(std::move(target)),
          kind_(kind),
          geoids_(geoids) {}

    std::string name() const override {
        return std::string("Convert ") + reference_kind_name(kind_) + " References";
    }

protected:
    void validate(Document& document) override {
        Chunk* chunk = find(document);
        if (chunk == nullptr) {
            throw ValueError("convert references: no chunk with id " +
                             std::to_string(chunk_id_));
        }
        if (target_.is_local() || !target_.is_set()) {
            throw ValueError("convert references: the target must be a set coordinate system");
        }
    }

    void stage(Document& document) override {
        Chunk& chunk = *find(document);
        cameras_ = chunk.cameras();
        markers_ = chunk.markers();
        camera_crs_ = chunk.explicit_camera_crs();
        marker_crs_ = chunk.explicit_marker_crs();
    }

    void commit(Document& document) override {
        convert_reference(*find(document), target_, kind_, geoids_);
    }

    void restore(Document& document) override {
        Chunk& chunk = *find(document);
        chunk.cameras() = cameras_;
        chunk.markers() = markers_;
        if (camera_crs_.has_value()) {
            chunk.set_camera_crs(*camera_crs_);
        } else {
            chunk.clear_camera_crs();
        }
        if (marker_crs_.has_value()) {
            chunk.set_marker_crs(*marker_crs_);
        } else {
            chunk.clear_marker_crs();
        }
    }

private:
    Chunk* find(Document& document) const {
        for (auto& workspace : document.workspaces()) {
            if (Chunk* chunk = workspace->find_chunk(chunk_id_)) {
                return chunk;
            }
        }
        return nullptr;
    }

    std::int32_t chunk_id_;
    CoordinateSystem target_;
    ReferenceKind kind_;
    const crs::GeoidStore* geoids_;
    std::vector<Camera> cameras_;
    std::vector<Marker> markers_;
    std::optional<CRS> camera_crs_;
    std::optional<CRS> marker_crs_;
};

}  // namespace

std::unique_ptr<AbstractCommand> make_convert_reference_command(
    std::int32_t chunk_id, crs::CoordinateSystem target, ReferenceKind kind,
    const crs::GeoidStore* geoids) {
    return std::make_unique<ConvertReferenceCommand>(chunk_id, std::move(target), kind, geoids);
}

}  // namespace reference
}  // namespace alphashape
