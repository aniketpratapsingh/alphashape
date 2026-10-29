// local_frame.cpp -- ECEF-anchored local tangent frame (M12 §4.4, WP-12.6).
//
// Layering note: this file deliberately re-derives the geodetic latitude /
// longitude of the anchor (and the local east-north-up axes) instead of calling
// model/geo/geodesy.hpp. alphashape_crs sits BELOW alphashape_model in the
// dependency order (model links crs PRIVATE), so reaching up for geodesy would
// close a cycle. The formulas are the published ones the geodesy header cites,
// and only two small values are needed here.

#include "model/crs/local_frame.hpp"

#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/LU>

#include "base/utils/error.hpp"

namespace alphashape::crs {
namespace {

/// WGS 84 defining constants (the ellipsoid EPSG:4978 is defined on).
constexpr double kWgs84A = 6378137.0;
constexpr double kWgs84F = 1.0 / 298.257223563;
constexpr double kPi = 3.14159265358979323846;

/// Geocentric ECEF (metres) -> geodetic latitude/longitude (radians).
/// Bowring's iteration on the reduced latitude, the same construction the
/// geodesy header documents; converges to machine precision in a few passes.
void ecef_to_lat_lon(const Eigen::Vector3d& ecef, double* lat_rad, double* lon_rad) {
    const double e2 = kWgs84F * (2.0 - kWgs84F);
    const double p = std::hypot(ecef.x(), ecef.y());
    *lon_rad = std::atan2(ecef.y(), ecef.x());
    if (p == 0.0) {
        *lat_rad = ecef.z() >= 0.0 ? 0.5 * kPi : -0.5 * kPi;
        return;
    }
    double lat = std::atan2(ecef.z(), p * (1.0 - e2));
    for (int i = 0; i < 100; ++i) {
        const double slat = std::sin(lat);
        const double n = kWgs84A / std::sqrt(1.0 - e2 * slat * slat);
        const double next = std::atan2(ecef.z() + n * e2 * slat, p);
        const bool done = std::abs(next - lat) < 1e-15;
        lat = next;
        if (done) {
            break;
        }
    }
    *lat_rad = lat;
}

/// Rotation whose COLUMNS are the east, north and up axes at (lat, lon)
/// expressed in ECEF: enu = R^T * ecef_delta, ecef_delta = R * enu.
Eigen::Matrix3d enu_rotation(double lat_rad, double lon_rad) {
    const double slat = std::sin(lat_rad);
    const double clat = std::cos(lat_rad);
    const double slon = std::sin(lon_rad);
    const double clon = std::cos(lon_rad);

    Eigen::Matrix3d r;
    r.col(0) << -slon, clon, 0.0;                       // east
    r.col(1) << -slat * clon, -slat * slon, clat;       // north
    r.col(2) << clat * clon, clat * slon, slat;         // up
    return r;
}

/// Step (metres) used to linearise to_crs() at the anchor for `transform()`.
/// A central difference makes the result exact for an affine (geocentric)
/// mapping. The step is not 1 m: the CRS values it differences are around the
/// ECEF magnitude (6.4e6 m) or a longitude in degrees (11), where one ULP is
/// about 1e-9 m and 2e-15 deg, so a 1 m step would lose most of the derivative
/// to cancellation. 100 m keeps the difference well above that noise while the
/// curvature it ignores stays far below the documented "near the anchor"
/// accuracy.
constexpr double kJacobianStepM = 100.0;

}  // namespace

LocalFrame::LocalFrame(CoordinateSystem cs, const Eigen::Vector3d& origin_ecef,
                       const Eigen::Matrix3d& rotation, bool identity)
    : cs_(std::move(cs)), origin_ecef_(origin_ecef), rotation_(rotation), identity_(identity) {
    if (!identity_) {
        ecef_to_crs_ =
            CrsTransform::create(CoordinateSystem::from_epsg(4978), cs_, TransformOptions{});
    }
}

LocalFrame LocalFrame::enu_at(const CoordinateSystem& cs, const Eigen::Vector3d& origin_ecef) {
    // The local CRS with a real anchor is a contradiction: a local frame has no
    // geographic origin. Only the identity (zero anchor, identity rotation)
    // local frame is meaningful.
    if (cs.is_local()) {
        throw ValueError(
            "a local frame needs a geographic or projected CRS; use the identity frame");
    }
    if (!cs.is_set()) {
        throw ValueError("cannot anchor a local frame on an unset CRS");
    }
    double lat = 0.0;
    double lon = 0.0;
    ecef_to_lat_lon(origin_ecef, &lat, &lon);
    return with_rotation(cs, origin_ecef, enu_rotation(lat, lon));
}

LocalFrame LocalFrame::with_rotation(const CoordinateSystem& cs,
                                     const Eigen::Vector3d& origin_ecef,
                                     const Eigen::Matrix3d& rotation) {
    if (!cs.is_set()) {
        throw ValueError("cannot anchor a local frame on an unset CRS");
    }
    if (cs.is_local()) {
        throw ValueError(
            "a local frame needs a geographic or projected CRS; use the identity frame");
    }
    return LocalFrame(cs, origin_ecef, rotation, /*identity=*/false);
}

const CoordinateSystem& LocalFrame::coordinate_system() const { return cs_; }

const Eigen::Vector3d& LocalFrame::origin_ecef() const { return origin_ecef_; }

const Eigen::Matrix3d& LocalFrame::rotation() const { return rotation_; }

bool LocalFrame::is_identity() const { return identity_; }

CrsPoint LocalFrame::to_crs(const Eigen::Vector3d& local) const {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (identity_) {
        return CrsPoint{local.x(), local.y(), local.z()};
    }
    const Eigen::Vector3d ecef = origin_ecef_ + rotation_ * local;
    const std::optional<CrsPoint> out = ecef_to_crs_.forward(CrsPoint{ecef.x(), ecef.y(), ecef.z()});
    if (!out.has_value()) {
        return CrsPoint{nan, nan, nan};
    }
    return *out;
}

Eigen::Vector3d LocalFrame::to_local(const CrsPoint& point) const {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (identity_) {
        return Eigen::Vector3d{point.x, point.y, point.z};
    }
    const CrsPoint ecef = ecef_to_crs_.inverse(point);
    if (!std::isfinite(ecef.x) || !std::isfinite(ecef.y) || !std::isfinite(ecef.z)) {
        return Eigen::Vector3d{nan, nan, nan};
    }
    return rotation_.transpose() * (Eigen::Vector3d{ecef.x, ecef.y, ecef.z} - origin_ecef_);
}

Eigen::Matrix4d LocalFrame::transform() const {
    if (identity_) {
        return Eigen::Matrix4d::Identity();
    }
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    const CrsPoint origin = to_crs(Eigen::Vector3d::Zero());
    m(0, 3) = origin.x;
    m(1, 3) = origin.y;
    m(2, 3) = origin.z;
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::Vector3d step = Eigen::Vector3d::Zero();
        step[axis] = kJacobianStepM;
        const CrsPoint plus = to_crs(step);
        const CrsPoint minus = to_crs(-step);
        m(0, axis) = (plus.x - minus.x) / (2.0 * kJacobianStepM);
        m(1, axis) = (plus.y - minus.y) / (2.0 * kJacobianStepM);
        m(2, axis) = (plus.z - minus.z) / (2.0 * kJacobianStepM);
    }
    return m;
}

Eigen::Matrix4d LocalFrame::transformationMatrix() const { return transform().inverse(); }

}  // namespace alphashape::crs
