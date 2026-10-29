#include "model/crs/coordinate_system.hpp"

#include <memory>
#include <string>
#include <utility>

#include <proj.h>

#include "base/utils/error.hpp"
#include "model/crs/proj_context.hpp"

namespace alphashape::crs {

/// The data behind a CoordinateSystem: identifying strings only. No PJ* lives
/// here. PROJ is asked for a fresh object whenever an operation needs one, on
/// the calling thread's context (WP-12.1: one context per thread), so a value
/// can be copied across threads and used in both without locking.
struct detail::CoordinateSystemImpl {
    bool local = true;      ///< the "LOCAL" engineering frame
    CrsKind kind = CrsKind::Local;
    std::string name;       ///< proj_get_name(); "LOCAL" for the local frame
    std::string auth_name;  ///< "EPSG" | "LOCAL" | ""
    std::string auth_code;  ///< "32633" | ""
    std::string wkt;        ///< canonical WKT2:2019, single line
    int epsg = 0;           ///< 0 when custom / non-EPSG authority
    std::optional<VerticalCrs> vertical;
};

namespace {

/// The canonical WKT of the local (engineering) CRS. Hand-written so that
/// `local()` never touches PROJ: a default-constructed Chunk must be usable
/// before any PROJ context exists, and the local frame has no datum to look up.
/// WKT2:2019, metres, axes east/north/up.
constexpr const char* kLocalName = "LOCAL";
constexpr const char* kLocalWkt =
    "ENGCRS[\"LOCAL\","
    "EDATUM[\"Local engineering datum\"],"
    "CS[Cartesian,3],"
    "AXIS[\"(E)\",east,ORDER[1],LENGTHUNIT[\"metre\",1]],"
    "AXIS[\"(N)\",north,ORDER[2],LENGTHUNIT[\"metre\",1]],"
    "AXIS[\"(U)\",up,ORDER[3],LENGTHUNIT[\"metre\",1]]]";

struct PjDeleter {
    void operator()(PJ* pj) const {
        if (pj != nullptr) {
            proj_destroy(pj);
        }
    }
};
using PjPtr = std::unique_ptr<PJ, PjDeleter>;

/// PROJ's WKT output options: one line, so the string can sit in an XML
/// attribute and be compared without whitespace normalisation.
const char* const* wkt_options() {
    static const char* const options[] = {"MULTILINE=NO", nullptr};
    return options;
}

std::string proj_error_string(PJ_CONTEXT* ctx) {
    const int err = proj_context_errno(ctx);
    const char* text = err != 0 ? proj_context_errno_string(ctx, err) : nullptr;
    return text != nullptr ? std::string(text) : std::string("unknown PROJ error");
}

/// proj_create() plus a CRS check: PROJ happily returns a coordinate operation
/// for strings that look like a CRS (a PROJ string without "+type=crs", a
/// conversion WKT), and treating one as a CRS would fail later and further away.
PjPtr create_crs_object(const std::string& text) {
    PJ_CONTEXT* ctx = thread_context();
    PjPtr pj{proj_create(ctx, text.c_str())};
    if (!pj || !proj_is_crs(pj.get())) {
        return PjPtr{};
    }
    return pj;
}

CrsKind kind_from_proj(PJ* pj) {
    const PJ_TYPE type = proj_get_type(pj);
    switch (type) {
        case PJ_TYPE_GEOGRAPHIC_2D_CRS:
            return CrsKind::Geographic2D;
        case PJ_TYPE_GEOGRAPHIC_3D_CRS:
            return CrsKind::Geographic3D;
        case PJ_TYPE_PROJECTED_CRS:
            return CrsKind::Projected;
        case PJ_TYPE_GEOCENTRIC_CRS:
            return CrsKind::Geocentric;
        case PJ_TYPE_VERTICAL_CRS:
            return CrsKind::Vertical;
        case PJ_TYPE_COMPOUND_CRS:
            return CrsKind::Compound;
        case PJ_TYPE_ENGINEERING_CRS:
            return CrsKind::Local;
        case PJ_TYPE_BOUND_CRS: {
            // A CRS pinned to a datum by an explicit transformation (WP-12.4):
            // it is the source CRS plus the shift, so its kind is the source's.
            PJ_CONTEXT* ctx = thread_context();
            PjPtr source{proj_get_source_crs(ctx, pj)};
            return source ? kind_from_proj(source.get()) : CrsKind::Geographic2D;
        }
        default:
            // Dynamic geodetic CRSs (ITRF, plate-fixed) report as geocentric or
            // geographic depending on their CS; their CS kind is what callers
            // branch on, so classify by the PROJ type we did not name above.
            return CrsKind::Projected;
    }
}

bool is_epsg_authority(const std::string& auth_name) {
    return auth_name.size() == 4 && (auth_name[0] == 'E' || auth_name[0] == 'e') &&
           (auth_name[1] == 'P' || auth_name[1] == 'p') &&
           (auth_name[2] == 'S' || auth_name[2] == 's') &&
           (auth_name[3] == 'G' || auth_name[3] == 'g');
}

/// A code that is all digits fits `int` for any real authority code; anything
/// else (a PROJ pipeline id, an OGC URN fragment) yields 0 by contract.
int parse_code(const std::string& code) {
    if (code.empty() || code.size() > 9) {
        return 0;
    }
    int value = 0;
    for (char c : code) {
        if (c < '0' || c > '9') {
            return 0;
        }
        value = value * 10 + (c - '0');
    }
    return value > 0 ? value : 0;
}

std::shared_ptr<const detail::CoordinateSystemImpl> make_local() {
    auto impl = std::make_shared<detail::CoordinateSystemImpl>();
    impl->local = true;
    impl->kind = CrsKind::Local;
    impl->name = kLocalName;
    impl->auth_name = kLocalName;
    impl->wkt = kLocalWkt;
    return impl;
}

/// Describe a PROJ CRS object. `pj` is not consumed.
std::shared_ptr<const detail::CoordinateSystemImpl> describe(PJ* pj) {
    PJ_CONTEXT* ctx = thread_context();
    auto impl = std::make_shared<detail::CoordinateSystemImpl>();
    impl->local = false;
    impl->kind = kind_from_proj(pj);

    const char* name = proj_get_name(pj);
    impl->name = name != nullptr ? std::string(name) : std::string();

    const char* auth = proj_get_id_auth_name(pj, 0);
    const char* code = proj_get_id_code(pj, 0);
    impl->auth_name = auth != nullptr ? std::string(auth) : std::string();
    impl->auth_code = code != nullptr ? std::string(code) : std::string();
    if (is_epsg_authority(impl->auth_name)) {
        impl->epsg = parse_code(impl->auth_code);
        if (impl->epsg != 0) {
            // Normalise the spelling so equality does not depend on PROJ's case.
            impl->auth_name = "EPSG";
        }
    }

    const char* wkt = proj_as_wkt(ctx, pj, PJ_WKT2_2019, wkt_options());
    impl->wkt = wkt != nullptr ? std::string(wkt) : std::string();
    return impl;
}

/// Parse `text` as a CRS and describe it; throws ValueError naming the input
/// category and PROJ's reason on failure.
std::shared_ptr<const detail::CoordinateSystemImpl> parse_or_throw(
    const std::string& text, const char* what) {
    PJ_CONTEXT* ctx = thread_context();
    if (text.empty()) {
        throw ValueError(std::string(what) + ": empty input");
    }
    PjPtr pj = create_crs_object(text);
    if (!pj) {
        throw ValueError(std::string(what) + " '" + text + "': " + proj_error_string(ctx) +
                         " (not a coordinate reference system)");
    }
    return describe(pj.get());
}

/// Build a PJ* for `impl` on the calling thread's context. The local frame is
/// parsed from its WKT, which PROJ accepts, so every operation has a real
/// object to work with. Null when PROJ rejects the stored definition (a POJ
/// built by an incompatible PROJ version, say).
PjPtr object_for(const detail::CoordinateSystemImpl& impl) {
    return create_crs_object(impl.wkt);
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructors and factories
// ---------------------------------------------------------------------------

CoordinateSystem::CoordinateSystem() : impl_(make_local()) {}

CoordinateSystem::CoordinateSystem(std::shared_ptr<const detail::CoordinateSystemImpl> impl)
    : impl_(std::move(impl)) {}

CoordinateSystem CoordinateSystem::local() { return CoordinateSystem{make_local()}; }

CoordinateSystem CoordinateSystem::from_epsg(int epsg) {
    if (epsg <= 0) {
        throw ValueError("unknown EPSG code " + std::to_string(epsg));
    }
    PJ_CONTEXT* ctx = thread_context();
    const std::string text = "EPSG:" + std::to_string(epsg);
    PjPtr pj = create_crs_object(text);
    if (!pj) {
        throw ValueError("unknown EPSG code " + std::to_string(epsg) + ": " +
                         proj_error_string(ctx));
    }
    return CoordinateSystem{describe(pj.get())};
}

CoordinateSystem CoordinateSystem::from_wkt(const std::string& wkt) {
    return CoordinateSystem{parse_or_throw(wkt, "invalid WKT")};
}

CoordinateSystem CoordinateSystem::from_proj(const std::string& proj_string) {
    if (proj_string.empty()) {
        throw ValueError("invalid PROJ string: empty input");
    }
    std::string text = proj_string;
    // Without +type=crs PROJ returns a coordinate operation, which
    // create_crs_object() rejects; adding it is what makes a PROJ string name a
    // CRS (§4.2). A string that already carries it is left alone.
    if (text.find("+type=crs") == std::string::npos) {
        text += " +type=crs";
    }
    return CoordinateSystem{parse_or_throw(text, "invalid PROJ string")};
}

CoordinateSystem CoordinateSystem::from_name(const std::string& name) {
    if (name == kLocalName || name == "Local" || name == "local") {
        return local();
    }
    // PROJ names are not unique: EPSG:4326, 4979 and 4978 are all "WGS 84"
    // (geographic 2D, geographic 3D and geocentric). The first entry in list
    // order wins, which puts 4326 — the name everyone means by "WGS 84" — first.
    // Callers that need one of the others use from_epsg.
    for (const BuiltinCrs& entry : builtin_crs_list()) {
        if (entry.name == name) {
            return from_epsg(entry.epsg);
        }
    }
    throw ValueError("unknown coordinate system name '" + name +
                     "' (not in the built-in list; use from_epsg or from_wkt)");
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

bool CoordinateSystem::is_set() const { return impl_ && !impl_->local; }

bool CoordinateSystem::is_local() const { return !impl_ || impl_->local; }

CrsKind CoordinateSystem::kind() const { return impl_ ? impl_->kind : CrsKind::Local; }

int CoordinateSystem::epsg() const { return impl_ ? impl_->epsg : 0; }

const std::string& CoordinateSystem::name() const {
    static const std::string kEmpty;
    return impl_ ? impl_->name : kEmpty;
}

std::string CoordinateSystem::auth_name() const {
    return impl_ ? impl_->auth_name : std::string();
}

std::string CoordinateSystem::auth_code() const {
    return impl_ ? impl_->auth_code : std::string();
}

std::string CoordinateSystem::to_wkt(WktDialect dialect) const {
    if (!impl_) {
        return std::string();
    }
    if (dialect == WktDialect::Wkt2_2019 || impl_->local) {
        // WKT2:2019 is stored; the local frame has no PROJ-defined alternative
        // worth regenerating (PROJ's own WKT1 for an engineering CRS is
        // lossy), so both return the canonical text.
        return impl_->wkt;
    }
    PJ_CONTEXT* ctx = thread_context();
    PjPtr pj = object_for(*impl_);
    if (!pj) {
        return impl_->wkt;
    }
    PJ_WKT_TYPE type = PJ_WKT2_2015;
    switch (dialect) {
        case WktDialect::Wkt2_2015:
            type = PJ_WKT2_2015;
            break;
        case WktDialect::Wkt1_GDAL:
            type = PJ_WKT1_GDAL;
            break;
        case WktDialect::Wkt1_ESRI:
            type = PJ_WKT1_ESRI;
            break;
        case WktDialect::Wkt2_2019:
            type = PJ_WKT2_2019;
            break;
    }
    const char* text = proj_as_wkt(ctx, pj.get(), type, wkt_options());
    return text != nullptr ? std::string(text) : impl_->wkt;
}

std::string CoordinateSystem::to_proj4() const {
    if (!impl_ || impl_->local) {
        return std::string();
    }
    PJ_CONTEXT* ctx = thread_context();
    PjPtr pj = object_for(*impl_);
    if (!pj) {
        return std::string();
    }
    const char* text = proj_as_proj_string(ctx, pj.get(), PJ_PROJ_4, nullptr);
    return text != nullptr ? std::string(text) : std::string();
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

bool CoordinateSystem::operator==(const CoordinateSystem& other) const {
    const bool a_null = !impl_;
    const bool b_null = !other.impl_;
    if (a_null || b_null) {
        return a_null == b_null;
    }
    if (impl_->local != other.impl_->local) {
        return false;
    }
    if (impl_->local) {
        return true;  // one local frame; there is only one
    }
    if (impl_->vertical != other.impl_->vertical) {
        return false;
    }
    // An authority code is the stable identity; the WKT text of the same code
    // differs between PROJ versions (ensemble members, datum names).
    if (!impl_->auth_name.empty() && !impl_->auth_code.empty() && !other.impl_->auth_name.empty() &&
        !other.impl_->auth_code.empty()) {
        return impl_->auth_name == other.impl_->auth_name &&
               impl_->auth_code == other.impl_->auth_code;
    }
    return impl_->wkt == other.impl_->wkt;
}

bool CoordinateSystem::is_equivalent_to(const CoordinateSystem& other,
                                        bool ignore_axis_order) const {
    if (!impl_ || !other.impl_) {
        return !impl_ && !other.impl_;
    }
    if (impl_->local || other.impl_->local) {
        return impl_->local == other.impl_->local;
    }
    if (impl_->vertical != other.impl_->vertical) {
        return false;
    }
    PJ_CONTEXT* ctx = thread_context();
    PjPtr a = object_for(*impl_);
    PjPtr b = object_for(*other.impl_);
    if (!a || !b) {
        return false;
    }
    const PJ_COMPARISON_CRITERION criterion =
        ignore_axis_order ? PJ_COMP_EQUIVALENT_EXCEPT_AXIS_ORDER_GEOGCRS : PJ_COMP_EQUIVALENT;
    return proj_is_equivalent_to_with_ctx(ctx, a.get(), b.get(), criterion) != 0;
}

std::optional<VerticalCrs> CoordinateSystem::vertical() const {
    return impl_ ? impl_->vertical : std::nullopt;
}

CoordinateSystem CoordinateSystem::with_vertical(const VerticalCrs& vertical) const {
    auto impl = std::make_shared<detail::CoordinateSystemImpl>(impl_ ? *impl_ : *make_local());
    impl->vertical = vertical.is_set() ? std::optional<VerticalCrs>(vertical) : std::nullopt;
    return CoordinateSystem{std::move(impl)};
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

const char* crs_kind_name(CrsKind kind) {
    switch (kind) {
        case CrsKind::Local:
            return "local";
        case CrsKind::Geographic2D:
            return "geographic-2d";
        case CrsKind::Geographic3D:
            return "geographic-3d";
        case CrsKind::Projected:
            return "projected";
        case CrsKind::Geocentric:
            return "geocentric";
        case CrsKind::Vertical:
            return "vertical";
        case CrsKind::Compound:
            return "compound";
    }
    return "unknown";
}

const char* wkt_dialect_name(WktDialect dialect) {
    switch (dialect) {
        case WktDialect::Wkt2_2019:
            return "WKT2:2019";
        case WktDialect::Wkt2_2015:
            return "WKT2:2015";
        case WktDialect::Wkt1_GDAL:
            return "WKT1_GDAL";
        case WktDialect::Wkt1_ESRI:
            return "WKT1_ESRI";
    }
    return "unknown";
}

}  // namespace alphashape::crs
