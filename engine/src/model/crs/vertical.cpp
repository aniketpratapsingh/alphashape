#include "model/crs/vertical.hpp"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <proj.h>

#include "base/utils/error.hpp"
#include "model/crs/proj_context.hpp"

namespace alphashape::crs {

namespace {

struct PjDeleter {
    void operator()(PJ* pj) const {
        if (pj != nullptr) {
            proj_destroy(pj);
        }
    }
};
using PjPtr = std::unique_ptr<PJ, PjDeleter>;

std::string proj_error(PJ_CONTEXT* ctx) {
    const int err = proj_context_errno(ctx);
    const char* text = err != 0 ? proj_context_errno_string(ctx, err) : nullptr;
    return text != nullptr && *text != '\0' ? std::string(text) : std::string("unknown PROJ error");
}

/// PROJ's own name for a vertical CRS's datum, used when a `VerticalCrs` has no
/// EPSG code: the names in `VerticalCrs::name` are PROJ/ISO names ("EGM96
/// height", "NAVD88 height"), which is exactly what proj.db's vertical_crs
/// table stores. The list is built once per process.
struct VerticalName {
    std::string name;
    int epsg = 0;
};

const std::vector<VerticalName>& vertical_names() {
    static const std::vector<VerticalName> names = [] {
        std::vector<VerticalName> out;
        PJ_CONTEXT* ctx = thread_context();
        PJ_STRING_LIST* codes =
            proj_get_codes_from_database(ctx, "EPSG", PJ_TYPE_VERTICAL_CRS, 0);
        if (codes == nullptr) {
            return out;
        }
        for (char** it = codes; *it != nullptr; ++it) {
            const int code = std::atoi(*it);
            if (code == 0) {
                continue;
            }
            PjPtr crs{proj_create_from_database(ctx, "EPSG", *it, PJ_CATEGORY_CRS, 0, nullptr)};
            if (!crs) {
                continue;
            }
            const char* name = proj_get_name(crs.get());
            if (name == nullptr) {
                continue;
            }
            out.push_back(VerticalName{name, code});
        }
        proj_string_list_destroy(codes);
        return out;
    }();
    return names;
}

}  // namespace

VerticalCrsLookup lookup_vertical_crs(const VerticalCrs& vertical) {
    VerticalCrsLookup lookup;
    if (!vertical.is_set()) {
        return lookup;
    }
    lookup.grid = vertical.grid;
    if (!vertical.name.empty()) {
        // A `VerticalCrs` may carry the authority code in `grid` ("EPSG:5773")
        // or the PROJ grid file name; neither identifies the vertical CRS, and
        // the name is what does.
        const std::string wanted = vertical.name;
        for (const VerticalName& entry : vertical_names()) {
            // Exact match first: "EGM96 height" must never resolve to
            // "EGM96 height (ftUS)" because of a substring rule.
            if (entry.name == wanted) {
                lookup.epsg = entry.epsg;
                lookup.proj_name = entry.name;
                return lookup;
            }
        }
    }
    // No authority name matched. That is not a failure here: PROJ can still
    // build a compound CRS from a name it knows, and a custom grid is handled
    // by `VerticalTransform::custom()` anyway. Reporting the name lets the
    // caller write the diagnostic.
    lookup.proj_name = vertical.name;
    return lookup;
}

namespace {

/// The grid a `VerticalCrs` needs, as PROJ names it: `grid` when set, otherwise
/// the grid an EPSG vertical CRS's own definition carries (EGM96 height ->
/// us_nga_egm96_15.tif). Empty when neither is available.
std::string required_grid(const VerticalCrs& vertical) {
    if (!vertical.grid.empty()) {
        return vertical.grid;
    }
    const VerticalCrsLookup lookup = lookup_vertical_crs(vertical);
    if (lookup.epsg == 0) {
        return std::string();
    }
    PJ_CONTEXT* ctx = thread_context();
    const std::string code = std::to_string(lookup.epsg);
    PjPtr crs{proj_create_from_database(ctx, "EPSG", code.c_str(), PJ_CATEGORY_CRS, 0, nullptr)};
    if (!crs) {
        return std::string();
    }
    const char* proj4 = proj_as_proj_string(ctx, crs.get(), PJ_PROJ_4, nullptr);
    if (proj4 == nullptr) {
        return std::string();
    }
    const std::string text(proj4);
    const std::string key = "+geoidgrids=";
    const std::size_t at = text.find(key);
    if (at == std::string::npos) {
        return std::string();
    }
    const std::size_t start = at + key.size();
    const std::size_t end = text.find(' ', start);
    return text.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

}  // namespace

std::optional<double> orthometric_height(double ellipsoidal_height_m, double lon_deg,
                                         double lat_deg, const GeoidGrid& geoid) {
    const std::optional<double> undulation = geoid.undulation(lon_deg, lat_deg);
    if (!undulation) {
        return std::nullopt;
    }
    return ellipsoidal_height_m - *undulation;
}

std::optional<double> ellipsoidal_height(double orthometric_height_m, double lon_deg,
                                         double lat_deg, const GeoidGrid& geoid) {
    const std::optional<double> undulation = geoid.undulation(lon_deg, lat_deg);
    if (!undulation) {
        return std::nullopt;
    }
    return orthometric_height_m + *undulation;
}

// ---------------------------------------------------------------------------
// VerticalTransform
// ---------------------------------------------------------------------------

namespace {

/// WKT for a horizontal CRS bound to a vertical one. PROJ's own way to combine
/// them without an authority code for the pair is a compound CRS object; the
/// string form is built here because CrsTransform speaks WKT.
std::string compound_wkt(PJ_CONTEXT* ctx, const CoordinateSystem& horizontal,
                         const VerticalCrs& vertical) {
    const VerticalCrsLookup lookup = lookup_vertical_crs(vertical);
    PjPtr vcrs;
    if (lookup.epsg != 0) {
        const std::string code = std::to_string(lookup.epsg);
        vcrs.reset(proj_create_from_database(ctx, "EPSG", code.c_str(), PJ_CATEGORY_CRS, 0,
                                             nullptr));
    }
    if (!vcrs && !vertical.name.empty()) {
        vcrs.reset(proj_create(ctx, vertical.name.c_str()));
    }
    if (!vcrs || proj_get_type(vcrs.get()) != PJ_TYPE_VERTICAL_CRS) {
        throw ValueError("vertical datum \"" + vertical.name + "\" is not an EPSG vertical CRS");
    }
    PjPtr hcrs{proj_create(ctx, horizontal.to_wkt().c_str())};
    if (!hcrs) {
        throw ValueError("cannot read the horizontal CRS \"" + horizontal.name() + "\": " +
                         proj_error(ctx));
    }
    const std::string name = horizontal.name() + " + " + vertical.name;
    PjPtr compound{proj_create_compound_crs(ctx, name.c_str(), hcrs.get(), vcrs.get())};
    if (!compound) {
        throw ValueError("cannot form the compound CRS \"" + name + "\": " + proj_error(ctx));
    }
    const char* text = proj_as_wkt(ctx, compound.get(), PJ_WKT2_2019, nullptr);
    if (text == nullptr) {
        throw ValueError("PROJ cannot write the compound CRS \"" + name + "\" as WKT");
    }
    return std::string(text);
}

}  // namespace


struct VerticalTransform::Impl {
    enum class Mode { Compound, Custom };

    Mode mode = Mode::Compound;
    CoordinateSystem src;
    CoordinateSystem dst;
    /// Compound mode: the two compound CRSs, resolved at create() so a missing
    /// grid fails there. The transform between them is CrsTransform's.
    CrsTransform compound;
    /// Custom mode: the horizontal transform, the geoid, and the two
    /// ellipsoidal<->geographic bridges used to evaluate it.
    CrsTransform horizontal;
    std::shared_ptr<const GeoidGrid> geoid;
    CrsTransform src_to_geographic;
    CrsTransform dst_to_geographic;
    bool src_orthometric = false;
    bool dst_orthometric = false;
    std::string description;

    /// lon/lat for the geoid at a source-side point, nullopt when the point
    /// cannot be brought to the geographic CRS the geoid is referred to.
    std::optional<std::pair<double, double>> geographic_at(const CrsTransform& to_geo,
                                                          const CrsPoint& point) const {
        const std::optional<CrsPoint> geo = to_geo.forward(point);
        if (!geo) {
            return std::nullopt;
        }
        // Visualisation order throughout: x = lon, y = lat.
        return std::make_pair(geo->x, geo->y);
    }
};

VerticalTransform::VerticalTransform(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

const CrsTransform& VerticalTransform::horizontal() const {
    static const CrsTransform kEmpty;
    if (!impl_) {
        return kEmpty;
    }
    return impl_->mode == Impl::Mode::Compound ? impl_->compound : impl_->horizontal;
}

bool VerticalTransform::is_valid() const { return static_cast<bool>(impl_); }

namespace {

/// The vertical component of a CRS, or a default-constructed one (ellipsoidal)
/// when none is attached.
VerticalCrs vertical_of(const CoordinateSystem& cs) {
    const std::optional<VerticalCrs> vertical = cs.vertical();
    return vertical ? *vertical : VerticalCrs{};
}

/// WKT for `horizontal`, carrying `vertical` when one is set. This is where the
/// vertical component becomes part of the CRS PROJ sees: `CoordinateSystem`
/// keeps it beside the WKT rather than inside it, so a compound pair has to be
/// assembled here.
std::string wkt_with_vertical(PJ_CONTEXT* ctx, const CoordinateSystem& horizontal,
                              const VerticalCrs& vertical) {
    if (vertical.kind == VerticalCrs::Kind::Ellipsoidal) {
        return horizontal.to_wkt();
    }
    return compound_wkt(ctx, horizontal, vertical);
}

/// A pair of CRSs for the compound pipeline, from two horizontal CRSs each
/// carrying (or not) its vertical datum.
std::pair<CoordinateSystem, CoordinateSystem> compound_pair(const CoordinateSystem& src,
                                                            const CoordinateSystem& dst) {
    PJ_CONTEXT* ctx = thread_context();
    const CoordinateSystem src_c{CoordinateSystem::from_wkt(
        wkt_with_vertical(ctx, src, vertical_of(src)))};
    const CoordinateSystem dst_c{CoordinateSystem::from_wkt(
        wkt_with_vertical(ctx, dst, vertical_of(dst)))};
    return {src_c, dst_c};
}

}  // namespace

VerticalTransform VerticalTransform::compound(const CoordinateSystem& src,
                                              const CoordinateSystem& dst,
                                              const TransformOptions& options) {
    const VerticalCrs src_vertical = vertical_of(src);
    const VerticalCrs dst_vertical = vertical_of(dst);
    if (src_vertical.kind == VerticalCrs::Kind::Ellipsoidal &&
        dst_vertical.kind == VerticalCrs::Kind::Ellipsoidal) {
        throw ValueError("a compound transform needs a vertical datum on at least one side");
    }
    // A compound CRS needs an EPSG (or at least an authority) code for its
    // vertical half, because that is the only way PROJ can pick the geoid
    // operation; a custom grid goes through custom().
    for (const CoordinateSystem* side : {&src, &dst}) {
        const VerticalCrs vertical = vertical_of(*side);
        if (vertical.kind == VerticalCrs::Kind::Ellipsoidal) {
            continue;
        }
        if (lookup_vertical_crs(vertical).epsg == 0) {
            throw ValueError("vertical datum \"" + vertical.name +
                             "\" has no EPSG code; use VerticalTransform::custom() with a geoid "
                             "grid");
        }
    }
    auto impl = std::make_shared<Impl>();
    impl->mode = Impl::Mode::Compound;
    impl->src = src;
    impl->dst = dst;
    const auto [src_compound, dst_compound] = compound_pair(src, dst);
    impl->compound = CrsTransform::create(src_compound, dst_compound, options);
    impl->description = src.name() + " -> " + dst.name();
    return VerticalTransform(std::move(impl));
}

VerticalTransform VerticalTransform::custom(const CoordinateSystem& src,
                                            const CoordinateSystem& dst,
                                            const VerticalCrs& src_vertical,
                                            const VerticalCrs& dst_vertical,
                                            std::shared_ptr<const GeoidGrid> geoid,
                                            const CoordinateSystem& geographic_crs,
                                            const TransformOptions& options) {
    const bool src_ortho = src_vertical.kind != VerticalCrs::Kind::Ellipsoidal;
    const bool dst_ortho = dst_vertical.kind != VerticalCrs::Kind::Ellipsoidal;
    if (!src_ortho && !dst_ortho) {
        throw ValueError("a custom vertical transform needs a vertical datum on at least one side");
    }
    if ((src_ortho || dst_ortho) && !geoid) {
        throw ValueError("a custom vertical transform needs a geoid grid");
    }
    auto impl = std::make_shared<Impl>();
    impl->mode = Impl::Mode::Custom;
    impl->src = src;
    impl->dst = dst;
    impl->horizontal = CrsTransform::create(src, dst, options);
    impl->geoid = std::move(geoid);
    impl->src_orthometric = src_ortho;
    impl->dst_orthometric = dst_ortho;
    // The geoid is a function of lon/lat, so each side that needs it gets a
    // transform to the geographic CRS it is referred to. An already-geographic
    // side is its own bridge (CrsTransform short-circuits equal CRSs).
    if (src_ortho) {
        impl->src_to_geographic = CrsTransform::create(src, geographic_crs, options);
    }
    if (dst_ortho) {
        impl->dst_to_geographic = CrsTransform::create(dst, geographic_crs, options);
    }
    impl->description = "custom geoid " + geoid->path();
    return VerticalTransform(std::move(impl));
}

std::optional<CrsPoint> VerticalTransform::forward(const CrsPoint& point) const {
    if (!impl_) {
        return std::nullopt;
    }
    if (impl_->mode == Impl::Mode::Compound) {
        // PROJ's own pipeline already carries the geoid step, so the height it
        // returns is in the target's vertical datum.
        return impl_->compound.forward(point);
    }
    // Custom mode, source -> target:
    //   1. orthometric -> ellipsoidal at the source (h = H + N);
    //   2. horizontal transform (ellipsoidal heights travel through it);
    //   3. ellipsoidal -> orthometric at the target (H = h - N).
    CrsPoint start = point;
    if (impl_->src_orthometric) {
        const auto lonlat = impl_->geographic_at(impl_->src_to_geographic, point);
        if (!lonlat) {
            return std::nullopt;
        }
        const std::optional<double> height =
            ellipsoidal_height(point.z, lonlat->first, lonlat->second, *impl_->geoid);
        if (!height) {
            return std::nullopt;  // outside the geoid's coverage
        }
        start.z = *height;
    }
    const std::optional<CrsPoint> moved = impl_->horizontal.forward(start);
    if (!moved) {
        return std::nullopt;
    }
    if (!impl_->dst_orthometric) {
        return moved;
    }
    const auto lonlat = impl_->geographic_at(impl_->dst_to_geographic, *moved);
    if (!lonlat) {
        return std::nullopt;
    }
    const std::optional<double> height =
        orthometric_height(moved->z, lonlat->first, lonlat->second, *impl_->geoid);
    if (!height) {
        return std::nullopt;
    }
    CrsPoint out = *moved;
    out.z = *height;
    return out;
}

std::optional<CrsPoint> VerticalTransform::inverse(const CrsPoint& point) const {
    if (!impl_) {
        return std::nullopt;
    }
    if (impl_->mode == Impl::Mode::Compound) {
        return impl_->compound.try_inverse(point);
    }
    CrsPoint start = point;
    if (impl_->dst_orthometric) {
        const auto lonlat = impl_->geographic_at(impl_->dst_to_geographic, point);
        if (!lonlat) {
            return std::nullopt;
        }
        const std::optional<double> height =
            ellipsoidal_height(point.z, lonlat->first, lonlat->second, *impl_->geoid);
        if (!height) {
            return std::nullopt;
        }
        start.z = *height;
    }
    const std::optional<CrsPoint> moved = impl_->horizontal.try_inverse(start);
    if (!moved) {
        return std::nullopt;
    }
    if (!impl_->src_orthometric) {
        return moved;
    }
    const auto lonlat = impl_->geographic_at(impl_->src_to_geographic, *moved);
    if (!lonlat) {
        return std::nullopt;
    }
    const std::optional<double> height =
        orthometric_height(moved->z, lonlat->first, lonlat->second, *impl_->geoid);
    if (!height) {
        return std::nullopt;
    }
    CrsPoint out = *moved;
    out.z = *height;
    return out;
}

const std::string& VerticalTransform::description() const {
    static const std::string kInvalid = "invalid";
    return impl_ ? impl_->description : kInvalid;
}
