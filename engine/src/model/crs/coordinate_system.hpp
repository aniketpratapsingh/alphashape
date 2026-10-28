#pragma once

/// @file coordinate_system.hpp
/// @brief PROJ-backed coordinate reference system value type (M12 §4.2).
///
/// A `CoordinateSystem` is a cheap-to-copy value: it holds only the strings
/// that identify the CRS (canonical WKT2:2019, authority name and code, the
/// display name) behind a shared pimpl, never a PROJ object. Equality compares
/// those strings; semantic equivalence is a separate, explicit operation
/// (`is_equivalent_to`) that asks PROJ.
///
/// This header does not include <proj.h>. PROJ is linked PRIVATE into
/// alphashape_crs, so consumers need neither its include directory nor its
/// target (WP-12.1 §4.1).
///
/// Errors: every factory that cannot build a CRS throws
/// `alphashape::ValueError` with a message naming the input. `local()` is the
/// only factory that cannot fail.

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace alphashape::crs {

namespace detail {
/// Identifying strings behind a CoordinateSystem (coordinate_system.cpp).
/// Declared here so the handle can be a shared_ptr to an incomplete type; the
/// definition is private to the implementation. It never holds a PROJ object.
struct CoordinateSystemImpl;
}  // namespace detail

/// The CRS classes the engine distinguishes. Mirrors the PROJ object type for
/// every CRS PROJ can create from an authority code, WKT or PROJ string.
enum class CrsKind {
    Local,        ///< An engineering CRS: chunk/dense space, metres, no datum
    Geographic2D, ///< lon/lat (or lat/lon) on an ellipsoid
    Geographic3D, ///< lon/lat/height on an ellipsoid
    Projected,    ///< Easting/northing in metres on a projection surface
    Geocentric,   ///< Earth-centred, Earth-fixed Cartesian (EPSG:4978)
    Vertical,     ///< Heights only
    Compound,     ///< Horizontal + vertical
};

const char* crs_kind_name(CrsKind kind);

/// Which WKT flavour `to_wkt` produces. WKT2:2019 is the canonical form and the
/// one that round-trips through `from_wkt` without loss.
enum class WktDialect {
    Wkt2_2019,  ///< ISO 19162:2019 (PROJ's default output)
    Wkt2_2015,  ///< ISO 19162:2015
    Wkt1_GDAL,  ///< The OGC WKT1 dialect GDAL writes
    Wkt1_ESRI,  ///< The ESRI-flavoured WKT1
};

const char* wkt_dialect_name(WktDialect dialect);

/// Vertical component of a CRS (M12 §4.2, §4.5). `Ellipsoidal` means "no
/// vertical datum": heights are ellipsoidal and no geoid is applied.
/// The type is defined here because persistence and the DEM/ortho builders
/// carry it; the geoid evaluation itself is WP-12.5.
struct VerticalCrs {
    enum class Kind { Ellipsoidal, GeoidModel, GravityRelated };

    Kind kind = Kind::Ellipsoidal;
    std::string name;      ///< "EGM2008 height", "NAVD88 height"
    std::string grid;      ///< geoid grid id, e.g. "us_nga_egm08_25.tif"
    double offset_m = 0.0; ///< constant offset applied after the grid

    bool is_set() const { return kind != Kind::Ellipsoidal; }
    bool operator==(const VerticalCrs& other) const {
        return kind == other.kind && name == other.name && grid == other.grid &&
               offset_m == other.offset_m;
    }
    bool operator!=(const VerticalCrs& other) const { return !(*this == other); }
};

/// A selectable datum transformation between two CRSs (M12 §4.2, WP-12.4).
///
/// Produced by `CoordinateSystem::available_transformations()` and consumed by
/// `CoordinateSystem::with_transformation()` (datum.hpp). The grid fields are
/// what makes a missing grid diagnosable rather than a silent downgrade to a
/// ballpark: an operation that names a grid PROJ cannot find keeps its place in
/// the list (so the choice stays visible and pinnable), reports the grid under
/// `missing_grids`, and refuses to run with
/// `Error("transformation needs grid <name>; install proj-data")`.
struct DatumTransformation {
    std::string auth_name;   ///< "EPSG" | "DERIVED_FROM(EPSG)" | "PROJ" | ""
    std::string code;        ///< "1314", "7710"; "" for an anonymous operation
    std::string name;        ///< "OSGB36 to WGS 84 (6)", "NTv2", ...
    double accuracy_m = -1;  ///< -1 = unknown
    /// PROJ itself would choose this operation for the CRS pair: its first
    /// candidate, which `available_transformations()` lists first.
    bool preferred = false;
    /// A null (ballpark) datum shift: the CRSs are on different datums and no
    /// transformation is known, so coordinates would carry the datum offset.
    /// `CrsTransform` refuses one unless `allow_ballpark` is set.
    bool ballpark = false;
    /// PROJ pipeline text, in visualisation order (x = east/lon). Empty for an
    /// operation PROJ cannot express as a single PROJ string.
    std::string definition;
    /// Grid ids the operation needs, in the order PROJ reports them.
    std::vector<std::string> grids;
    /// The subset of `grids` that is not installed. Empty when it can run.
    std::vector<std::string> missing_grids;

    /// True when a required grid is absent: applying this operation needs
    /// `proj-data` installed first.
    bool needs_missing_grid() const { return !missing_grids.empty(); }

    /// Identity comparison of every field. Two entries for the same operation
    /// differ in `preferred` depending on the pair they were listed for, which
    /// is why it takes part.
    bool operator==(const DatumTransformation& other) const {
        return auth_name == other.auth_name && code == other.code && name == other.name &&
               accuracy_m == other.accuracy_m && preferred == other.preferred &&
               ballpark == other.ballpark && definition == other.definition &&
               grids == other.grids && missing_grids == other.missing_grids;
    }
    bool operator!=(const DatumTransformation& other) const { return !(*this == other); }
};

class CoordinateSystem {
public:
    /// The unset/local CRS ("LOCAL"): chunk-space metres, no datum. Calling the
    /// default constructor is equivalent to `local()`.
    CoordinateSystem();

    static CoordinateSystem local();

    /// @throws ValueError when PROJ has no CRS with that code.
    static CoordinateSystem from_epsg(int epsg);

    /// Parse WKT in any dialect PROJ understands; the canonical form stored is
    /// always WKT2:2019, so `to_wkt()` never reproduces the input verbatim.
    /// @throws ValueError when the text is not a CRS.
    static CoordinateSystem from_wkt(const std::string& wkt);

    /// Parse a PROJ string ("+proj=utm +zone=33 +datum=WGS84"): `+type=crs` is
    /// added when absent, because without it PROJ yields a coordinate
    /// operation, not a CRS.
    /// @throws ValueError when the text is not a CRS.
    static CoordinateSystem from_proj(const std::string& proj_string);

    /// Look up a name from `builtin_crs_list()` (case-sensitive, exact), plus
    /// "LOCAL" for the local frame.
    /// @throws ValueError when the name is not in the built-in list.
    static CoordinateSystem from_name(const std::string& name);

    /// False for the local CRS and for a default-constructed value; true once a
    /// real CRS was supplied.
    bool is_set() const;
    bool is_local() const;
    CrsKind kind() const;

    /// EPSG code, 0 when the CRS is custom (WKT/PROJ string or local). Codes
    /// from other authorities (ESRI, OGC) are not EPSG codes and report 0;
    /// `auth_name()`/`auth_code()` carry them.
    int epsg() const;

    /// The CRS name as PROJ reports it, e.g. "WGS 84 / UTM zone 33N". "LOCAL"
    /// for the local frame.
    const std::string& name() const;

    /// Canonical WKT. WKT2:2019 is the stored string; other dialects are
    /// generated on demand by PROJ. Single-line output (XML-attribute safe).
    std::string to_wkt(WktDialect dialect = WktDialect::Wkt2_2019) const;

    /// PROJ-string form, for diagnostics and legacy interchange. Lossy for
    /// datums without a PROJ-string equivalent; may be empty.
    std::string to_proj4() const;

    /// "EPSG" for authority-coded CRSs, "LOCAL" for the local frame, "" for a
    /// custom CRS with no identifier.
    std::string auth_name() const;
    /// The authority code as text, e.g. "32633"; "" when there is none.
    std::string auth_code() const;

    /// Equality on the identifying strings (local flag, authority and code,
    /// vertical component, or canonical WKT when no code exists). No PROJ call.
    bool operator==(const CoordinateSystem& other) const;
    bool operator!=(const CoordinateSystem& other) const { return !(*this == other); }

    /// PROJ equivalence: same datum, ellipsoid, axes and units, ignoring
    /// identifier and definition nuances. `ignore_axis_order` compares
    /// geographic CRSs regardless of lat/lon vs lon/lat ordering (the "same
    /// CRS, different axis order" case that string comparison gets wrong).
    bool is_equivalent_to(const CoordinateSystem& other, bool ignore_axis_order = true) const;

    /// Vertical component; nullopt means ellipsoidal heights.
    std::optional<VerticalCrs> vertical() const;

    /// A copy with the vertical component attached. Attaching a geoid does not
    /// change the horizontal definition; applying the shift is WP-12.5.
    CoordinateSystem with_vertical(const VerticalCrs& vertical) const;

    /// Every datum transformation PROJ knows from this CRS to `target`
    /// (WP-12.4, datum.hpp). Ordered preferred-first, then by increasing
    /// accuracy with unknown accuracy last; ties keep PROJ's order.
    ///
    /// A missing grid does not remove an entry — it is reported through
    /// `DatumTransformation::missing_grids` instead, because a silently shorter
    /// list would hide the better operation the user could enable by installing
    /// `proj-data`. Nothing is ever downloaded.
    ///
    /// @throws ValueError when either CRS is the local engineering frame or
    ///         unset, matching `CrsTransform::create`.
    /// @throws Error when PROJ cannot list operations for the pair.
    std::vector<DatumTransformation> available_transformations(
        const CoordinateSystem& target) const;

    /// A copy of this CRS bound to `transformation`'s hub CRS (a WKT2
    /// `BOUNDCRS`), so a later `CrsTransform::create()` with it uses exactly
    /// that operation instead of re-deriving one.
    ///
    /// Only a geographic 2D/3D CRS can be bound: PROJ attaches a datum shift to
    /// a geodetic CRS, so a projected source throws and the caller binds the
    /// underlying geographic CRS instead. A transformation that names a grid
    /// which is not installed throws
    /// `Error("transformation needs grid <name>; install proj-data")` — pinning
    /// is where a missing grid must surface, not at the first coordinate.
    ///
    /// @throws ValueError for the local frame, an unset CRS, a projected CRS or
    ///         an entry that is not one of this CRS's `available_transformations`.
    CoordinateSystem with_transformation(const DatumTransformation& transformation) const;

private:
    explicit CoordinateSystem(std::shared_ptr<const detail::CoordinateSystemImpl> impl);

    /// Identifying strings only: PROJ objects are never stored (WP-12.1 §4.1,
    /// pimpl). Copies share this block, so a Document/Chunk copy is cheap.
    std::shared_ptr<const detail::CoordinateSystemImpl> impl_;
};

/// One entry of the curated built-in list (§4.6, CRS-07).
struct BuiltinCrs {
    int epsg = 0;
    std::string name;  ///< from PROJ (proj_get_name), not hard-coded
    CrsKind kind = CrsKind::Projected;
    bool operator==(const BuiltinCrs& other) const {
        return epsg == other.epsg && name == other.name && kind == other.kind;
    }
};

/// The 48 curated CRSs scripts iterate over (M12 §4.6). Names come from PROJ,
/// resolved on first call. Anything else PROJ knows is reachable through
/// `from_epsg`, which is not limited to this list.
const std::vector<BuiltinCrs>& builtin_crs_list();

/// The names of `builtin_crs_list()` in list order (Python
/// `CoordinateSystem.listBuiltinCRS` order).
std::vector<std::string> builtin_crs_names();

}  // namespace alphashape::crs
