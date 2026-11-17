#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <proj.h>

#include "model/crs/coordinate_system.hpp"
#include "model/crs/proj_context.hpp"

namespace alphashape::crs {
namespace {

/// The curated list of M12 §4.6 (CRS-07). Codes only: names and kinds come from
/// PROJ (`proj_get_name`, `proj_get_type`), never from a hard-coded table, so a
/// PROJ data update renames a CRS instead of leaving the engine disagreeing
/// with `projinfo`.
///
/// The list is exactly 48 entries (§4.2 acceptance). §4.6's prose enumerates
/// 56 codes, an arithmetic slip in the plan; this list keeps every named
/// category and trims eight codes: the NAD83 / UTM range 11N–19N is represented
/// by its endpoints and middle (26911, 26914, 26919) rather than all nine, and
/// Tokyo (4301) and Pulkovo 1942 (4284) are reachable through `from_epsg` only.
/// Any CRS PROJ knows remains reachable by code; the list is what scripts
/// iterate, not the limit of the engine.
constexpr std::array<int, 48> kBuiltinCodes{
    // WGS 84: geographic 2D, geographic 3D, geocentric.
    4326, 4979, 4978,
    // 24 WGS 84 / UTM zones: 1N–19N and 1S–5S. The other 36 zones (32620–32660,
    // 32706–32760) are reachable by code (§4.6 "the remainder by code").
    32601, 32602, 32603, 32604, 32605, 32606, 32607, 32608, 32609, 32610,
    32611, 32612, 32613, 32614, 32615, 32616, 32617, 32618, 32619,
    32701, 32702, 32703, 32704, 32705,
    // ETRS89 and its projections (Europe).
    4258, 25832, 3035,
    // NAD83, NAD83 / UTM 11N–19N (ends + middle), NAD83 / Conus Albers.
    4269, 26911, 26914, 26919, 5070,
    // GDA2020 (Australia).
    7844, 7855,
    // SWEREF99 TM (Sweden), OSGB36 / British National Grid (UK),
    // CH1903+ / LV95 (Switzerland).
    3006, 27700, 2056,
    // New Zealand: geodetic and NZTM2000.
    4167, 2193,
    // SIRGAS 2000 (South America) and UTM 23S.
    4674, 31983,
    // CGCS2000 (China) and its 3-degree Gauss-Kruger CM 117E.
    4490, 4548,
    // Web Mercator — the raster-tiling default (M20).
    3857,
    // ED50 (European legacy datum, still in many archives).
    4230,
};

/// Build the list once. Resolving 48 EPSG codes costs a few milliseconds; the
/// entries never change for a given PROJ data set, and from_epsg() is the path
/// callers take for anything not in here.
std::vector<BuiltinCrs> build_list() {
    std::vector<BuiltinCrs> list;
    list.reserve(kBuiltinCodes.size());
    for (int code : kBuiltinCodes) {
        const CoordinateSystem cs = CoordinateSystem::from_epsg(code);
        list.push_back(BuiltinCrs{code, cs.name(), cs.kind()});
    }
    return list;
}

}  // namespace

const std::vector<BuiltinCrs>& builtin_crs_list() {
    static const std::vector<BuiltinCrs> list = build_list();
    return list;
}

std::vector<std::string> builtin_crs_names() {
    const std::vector<BuiltinCrs>& list = builtin_crs_list();
    std::vector<std::string> names;
    names.reserve(list.size());
    for (const BuiltinCrs& entry : list) {
        names.push_back(entry.name);
    }
    return names;
}

namespace {

std::vector<CrsCatalogEntry> read_catalog() {
    std::vector<CrsCatalogEntry> catalog;
    PJ_CONTEXT* ctx = thread_context();
    PROJ_CRS_LIST_PARAMETERS* params = proj_get_crs_list_parameters_create();
    if (params == nullptr) {
        return catalog;
    }
    const PJ_TYPE types[] = {PJ_TYPE_GEOGRAPHIC_2D_CRS, PJ_TYPE_GEOGRAPHIC_3D_CRS,
                             PJ_TYPE_GEOCENTRIC_CRS, PJ_TYPE_PROJECTED_CRS};
    params->types = types;
    params->typesCount = sizeof(types) / sizeof(types[0]);
    params->allow_deprecated = 0;
    int count = 0;
    PROJ_CRS_INFO** list = proj_get_crs_info_list_from_database(ctx, "EPSG", params, &count);
    proj_get_crs_list_parameters_destroy(params);
    if (list == nullptr) {
        return catalog;
    }
    catalog.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        const PROJ_CRS_INFO* info = list[i];
        if (info == nullptr || info->code == nullptr || info->name == nullptr) {
            continue;
        }
        char* end = nullptr;
        const long code = std::strtol(info->code, &end, 10);
        if (end == info->code || *end != '\0' || code <= 0) {
            continue;  // EPSG codes are numeric; anything else is not one
        }
        CrsCatalogEntry entry;
        entry.epsg = static_cast<int>(code);
        entry.name = info->name;
        switch (info->type) {
            case PJ_TYPE_GEOGRAPHIC_2D_CRS: entry.kind = CrsKind::Geographic2D; break;
            case PJ_TYPE_GEOGRAPHIC_3D_CRS: entry.kind = CrsKind::Geographic3D; break;
            case PJ_TYPE_GEOCENTRIC_CRS: entry.kind = CrsKind::Geocentric; break;
            default: entry.kind = CrsKind::Projected; break;
        }
        if (info->area_name != nullptr) {
            entry.area = info->area_name;
        }
        catalog.push_back(std::move(entry));
    }
    proj_crs_info_list_destroy(list);
    std::sort(catalog.begin(), catalog.end(),
              [](const CrsCatalogEntry& a, const CrsCatalogEntry& b) { return a.epsg < b.epsg; });
    return catalog;
}

}  // namespace

const std::vector<CrsCatalogEntry>& crs_catalog() {
    static const std::vector<CrsCatalogEntry> catalog = read_catalog();
    return catalog;
}

}  // namespace alphashape::crs
