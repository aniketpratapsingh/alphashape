#pragma once

/// @file vertical.hpp
/// @brief Vertical datums: orthometric <-> ellipsoidal heights (M12 §4.5).
///
/// A height is only meaningful with its datum. This module converts between
/// the two the engine deals with:
///
///   * ellipsoidal height h — measured above the reference ellipsoid, which is
///     what a GNSS receiver and a geographic 3D CRS (EPSG:4979) report;
///   * orthometric height H — measured above the geoid (mean sea level), which
///     is what a national vertical datum such as NAVD88 or EGM96 height is.
///
/// The relation is `H = h - N`, with N the geoid undulation at the point. Two
/// paths exist, and the difference matters:
///
///   * When both the horizontal CRS and the vertical datums are EPSG-defined,
///     `VerticalTransform::compound()` hands PROJ the compound pair
///     (e.g. EPSG:4979 -> EPSG:32633+5773) and PROJ builds the whole pipeline,
///     geoid step included. That is exact agreement with `cs2cs` and is what a
///     DEM or ortho written in a national vertical datum uses.
///   * For a custom grid (a project geoid, not an EPSG one) PROJ has no
///     authority code to work with, so `VerticalTransform::custom()` runs the
///     horizontal transform first and then applies the geoid shift in
///     `H = h - N` form. The order matters: N is evaluated at the point's
///     horizontal position, which must be converted to lon/lat first.
///
/// A geoid that cannot evaluate the point (outside its coverage) makes the
/// conversion fail with nullopt rather than return a height that is silently
/// tens of metres off.
///
/// `VerticalCrs::offset_m` is a datum's constant bias, applied around the geoid
/// so it works in both modes and both directions:
///
///   source side, orthometric -> ellipsoidal:  h = H + N + offset_src
///   target side, ellipsoidal -> orthometric:  H = h - N - offset_dst
///
/// A compound pipeline carries the geoid but not a constant, so there the
/// offset is applied before (source) and after (target) PROJ. `offset_m == 0`
/// (the default) is exactly the plain H = h - N relation.

#include <memory>
#include <optional>
#include <string>

#include "model/crs/coordinate_system.hpp"
#include "model/crs/geoid.hpp"
#include "model/crs/transform.hpp"

namespace alphashape::crs {

/// The PROJ vertical CRS code whose grid an `VerticalCrs` names, and the
/// horizontal CRS that grid is referred to. 0 / empty when the description does
/// not identify an EPSG vertical CRS.
struct VerticalCrsLookup {
    int epsg = 0;               ///< 5773 (EGM96), 5703 (NAVD88), ...
    std::string proj_name;      ///< "EGM96 height" when the CRS has no code
    std::string grid;           ///< grid id the vertical CRS needs
};

/// Resolve a `VerticalCrs` against the EPSG database: the vertical CRS code
/// when `grid` or `name` identifies one, otherwise the name alone (PROJ can
/// still build a compound CRS from a name it knows).
///
/// The match is on the authority's own vertical CRS list, so a name only
/// resolves when proj.db lists it; an unknown name yields an empty lookup
/// rather than a guess.
VerticalCrsLookup lookup_vertical_crs(const VerticalCrs& vertical);

/// Ellipsoidal height -> orthometric height at (lon, lat):
/// `H = h - N`. nullopt when the grid cannot evaluate the point.
///
/// This is the bare grid relation: a datum's constant offset is a property of
/// the datum, so `VerticalTransform` applies it, not this function.
std::optional<double> orthometric_height(double ellipsoidal_height_m, double lon_deg,
                                         double lat_deg, const GeoidGrid& geoid);

/// Inverse of `orthometric_height`: `h = H + N`.
std::optional<double> ellipsoidal_height(double orthometric_height_m, double lon_deg,
                                         double lat_deg, const GeoidGrid& geoid);

/// Horizontal + vertical transform, in either of the two modes described in the
/// file comment. Construction throws on a description that cannot be resolved
/// (a missing grid is reported by name, exactly like `CrsTransform`).
class VerticalTransform {
public:
    /// PROJ's own compound pipeline. Both `src` and `dst` must carry their
    /// vertical component (`CoordinateSystem::with_vertical`), and every code
    /// involved must be EPSG-defined.
    /// @throws ValueError when a side has no vertical component or its vertical
    ///         CRS is not EPSG-defined (use `custom()` for those).
    /// @throws Error when PROJ cannot build the compound pipeline, naming the
    ///         grid it needs.
    static VerticalTransform compound(const CoordinateSystem& src, const CoordinateSystem& dst,
                                      const TransformOptions& options = {});

    /// Horizontal transform plus a geoid shift. `src_vertical` is applied at the
    /// source side and `dst_vertical` at the target side; an ellipsoidal side
    /// needs no geoid. `geoid` is asked for the undulation in *geographic*
    /// degrees, so put the horizontal transform's source or target CRS first in
    /// `geographic_crs` — whichever matches the geoid's own datum.
    static VerticalTransform custom(const CoordinateSystem& src, const CoordinateSystem& dst,
                                    const VerticalCrs& src_vertical,
                                    const VerticalCrs& dst_vertical,
                                    std::shared_ptr<const GeoidGrid> geoid,
                                    const CoordinateSystem& geographic_crs,
                                    const TransformOptions& options = {});

    bool is_valid() const { return static_cast<bool>(impl_); }

    /// One point, source -> target. nullopt when the horizontal transform fails
    /// or the geoid does not cover the point.
    std::optional<CrsPoint> forward(const CrsPoint& point) const;
    std::optional<CrsPoint> inverse(const CrsPoint& point) const;

    /// The underlying horizontal transform (the compound pipeline or the
    /// horizontal one, depending on the mode).
    const CrsTransform& horizontal() const;

    /// One-line description for a log line.
    const std::string& description() const;

private:
    struct Impl;
    explicit VerticalTransform(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
};

}  // namespace alphashape::crs
