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
/// carried to ECEF as `origin_ecef + rotation * local` and then through a
/// `CrsTransform` from EPSG:4978 to the frame's CRS; `to_local` runs that
/// backwards. Because every quantity is a double and the anchor is within a
/// few thousand kilometres of the data, a local point within 500 km of the
/// anchor round-trips to better than a micrometre for a WGS 84 geographic CRS.
///
/// `transform()` is the 4x4 affine local -> CRS form. It is exact for a
/// geocentric CRS; for a curved CRS (geographic degrees, a projection) it is
/// the linearisation at the anchor, built from `to_crs` at unit offsets, so it
/// is only meaningful for coordinates near the anchor -- which is exactly what
/// a floating-origin render/export path needs.
///
/// A default-constructed frame is the identity: the local coordinates ARE the
/// CRS coordinates. That is the state of a chunk that has not been
/// georeferenced (its CRS is the local engineering frame). For such a frame
/// `to_crs`/`to_local` copy the value and `transform()` is the identity, and no
/// PROJ transform is built.
///
/// Errors: the factories throw `ValueError` when the CRS is unset (an identity
/// frame needs no CRS; give it `CoordinateSystem::local()` instead), and
/// whatever `CrsTransform::create` throws when PROJ cannot relate EPSG:4978 to
/// the CRS.

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
    CrsTransform ecef_to_crs_;
};

}  // namespace alphashape::crs
