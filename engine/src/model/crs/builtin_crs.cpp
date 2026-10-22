#include <array>
#include <string>
#include <vector>

#include "model/crs/coordinate_system.hpp"

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

}  // namespace alphashape::crs
