#pragma once

/// @file local_frame.hpp
/// @brief An ECEF-anchored local tangent (ENU) frame (M12 §4.4, WP-12.6).
///
/// A `LocalFrame` is the bridge between the chunk's internal Cartesian
/// coordinates (metres, right-handed, arbitrary orientation before
/// georeferencing) and a geographic/geocentric/projected CRS. It stores three
/// things:
///
///   * an ECEF anchor (`origin_ecef`, metres, EPSG:4978) -- the local origin;
///   * a rotation whose columns are the local axes expressed in the ECEF axes
///     (the identity for the standard east-north-up tangent frame at the
///     anchor);
///   * the target `CoordinateSystem`.
///
/// The mapping is exact, not a small-angle approximation: a local point is
/// carried to ECEF as `origin_ecef + rotation * local`, ECEF to geodetic by the
/// iterated Bowring inversion (see `ecef_to_geodetic` in the .cpp -- PROJ's
/// closed-form inverse is off by millimetres at these magnitudes), and geodetic
/// to the target projection by PROJ; `to_local` runs that backwards. A
/// geocentric target skips the geodetic step entirely, because its coordinates
/// are the ECEF coordinates. Because every step is done in doubles and iterated
/// where it matters, a local point within 500 km of the anchor round-trips to
/// better than a micrometre for any target CRS PROJ can express.
///
/// `transform()` is the 4x4 affine local -> CRS form. It is exact for a
/// geocentric CRS; for a curved CRS (geographic degrees, a projection) it is
/// the linearisation at the anchor, built from `to_crs` at a 100 m offset, so
/// it is only meaningful for coordinates near the anchor -- which is exactly
/// what a floating-origin render/export path needs.
///
/// A default-constructed frame is the identity: the local coordinates ARE the
/// CRS coordinates. That is the state of a chunk that has not been
/// georeferenced (its CRS is the local engineering frame). For such a frame
/// `to_crs`/`to_local` copy the value and `transform()` is the identity, and no
/// PROJ transform is built.
///
/// Errors: the factories throw `ValueError` when the CRS is unset or the local
/// engineering frame (an identity frame needs neither; default-construct it
/// instead), and whatever `CrsTransform::create` throws when PROJ cannot relate
/// geographic 3D coordinates to the CRS.

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "model/crs/coordinate_system.hpp"
#include "model/crs/transform.hpp"

namespace alphashape::crs {

class LocalFrame {
public:
    /// The identity frame: local coordinates are the CRS coordinates.
    LocalFrame() = default;

    /// A frame anchored at `origin_ecef` whose local axes are the east, north
    /// and up axes at that anchor (ENU), mapped into `cs`.
    ///
    /// @throws ValueError when `cs` is unset (use `local()` for an identity
    ///         frame), or when `cs` is the local CRS together with a non-zero
    ///         anchor or rotation (a local frame has no geographic anchor).
    /// @throws Error when PROJ cannot relate EPSG:4978 to `cs`.
    static LocalFrame enu_at(const CoordinateSystem& cs, const Eigen::Vector3d& origin_ecef);

    /// A frame with an explicit local -> ECEF rotation: column i is the local
    /// axis i expressed in ECEF. Same preconditions as `enu_at`; the rotation
    /// is used as given (it is not orthonormalised).
    static LocalFrame with_rotation(const CoordinateSystem& cs,
                                    const Eigen::Vector3d& origin_ecef,
                                    const Eigen::Matrix3d& rotation);

    /// The target CRS. The local CRS for the identity frame.
    const CoordinateSystem& coordinate_system() const;

    /// The ECEF anchor of the local origin (metres). Zero for the identity
    /// frame.
    const Eigen::Vector3d& origin_ecef() const;

    /// Local axes -> ECEF axes. Identity for the identity frame.
    const Eigen::Matrix3d& rotation() const;

    /// True for a default-constructed frame (or one built on the local CRS
    /// with a zero anchor and identity rotation): `to_crs` is a copy.
    bool is_identity() const;

    /// Local Cartesian (metres) -> CRS point. Returns NaN coordinates when
    /// PROJ reports a failure (a point outside the projection's domain).
    CrsPoint to_crs(const Eigen::Vector3d& local) const;

    /// CRS point -> local Cartesian (metres). The exact inverse of `to_crs`;
    /// NaN when PROJ reports a failure.
    Eigen::Vector3d to_local(const CrsPoint& point) const;

    /// 4x4 affine local -> CRS (row-major semantics of Eigen, i.e. the usual
    /// column-vector convention). Exact for a geocentric CRS, the anchor
    /// linearisation otherwise. Identity for the identity frame.
    Eigen::Matrix4d transform() const;

    /// The inverse of `transform()`: CRS -> local.
    Eigen::Matrix4d transformationMatrix() const;

private:
    LocalFrame(CoordinateSystem cs, const Eigen::Vector3d& origin_ecef,
               const Eigen::Matrix3d& rotation, bool identity);

    CoordinateSystem cs_ = CoordinateSystem::local();
    Eigen::Vector3d origin_ecef_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rotation_ = Eigen::Matrix3d::Identity();
    bool identity_ = true;
    /// True when the target is geocentric (EPSG:4978 and friends), where the
    /// local point IS the ECEF point and no geodetic step is needed.
    bool geocentric_ = false;
    /// EPSG:4979 (geographic 3D, lon/lat/height in visualisation order) -> the
    /// target CRS. Unset for a geocentric target and for the identity frame.
    ///
    /// The ECEF<->geodetic leg is deliberately NOT PROJ's `+inv +proj=cart`:
    /// PROJ's closed-form geodetic inverse loses millimetres at the ECEF
    /// magnitudes this frame handles (2.7 mm round trip at 500 km, measured),
    /// which is the exact trap model/geo/geodesy.hpp documents and avoids by
    /// iterating. The frame does the same iteration itself and asks PROJ only
    /// for geodetic -> projection, which is where it is both exact and needed.
    CrsTransform geo_to_crs_;
};

}  // namespace alphashape::crs
