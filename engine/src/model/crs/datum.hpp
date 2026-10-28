#pragma once

/// @file datum.hpp
/// @brief Datum-transformation selection and pinning (M12 §4.4, WP-12.4).
///
/// This header declares no new types: the value type is `DatumTransformation`
/// and the two operations are members of `CoordinateSystem`, as sketched in
/// M12 §4.2. The implementation lives in datum.cpp so the PROJ-facing code for
/// operation selection sits in one translation unit.
///
/// Two ideas:
///
///   * `available_transformations()` enumerates every PROJ operation between
///     two CRSs — Helmert, Molodensky-Badekas, NTv1/NTv2/NADCON/OSTN15 grid
///     shifts, the ballpark fallback — with its accuracy, its grids and a text
///     definition. The list is ordered so that the operation PROJ itself would
///     choose comes first (`preferred`), and the remainder follows in
///     increasing accuracy, unknown accuracy last.
///
///   * `with_transformation()` turns a chosen operation into a CRS: the source
///     CRS bound to the hub CRS by that operation (a WKT2 `BOUNDCRS`). A
///     `CrsTransform` created from the bound CRS to the target then has exactly
///     one candidate operation, so it uses the pinned shift rather than
///     re-deriving one. That is how a caller says "use the 7-parameter Helmert"
///     or "use the NTv2 grid" and gets it.
///
/// Grids are never downloaded. Grids are never downloaded: the context built by
/// `crs::thread_context()` disables network access and the grid cache, and
/// operation selection ignores grid availability so the *list* stays complete —
/// an operation whose grid is missing is reported, and applying it throws
/// `Error("transformation needs grid <name>; install proj-data")` instead of
/// silently degrading to a ballpark.
///
/// `+towgs84` is honoured without special handling here: PROJ models a CRS
/// carrying TOWGS84 as a bound CRS, so its Helmert shift is one of the
/// operations `available_transformations()` lists and the one PROJ marks
/// preferred. `Crs.Towgs84Applied` pins that behaviour.

#include <string>
#include <vector>

#include "model/crs/coordinate_system.hpp"

namespace alphashape::crs {

/// Readable summary of one transformation for a log line or a tooltip, e.g.
/// "OSGB36 to WGS 84 (6) [EPSG:1314, 2 m]". Empty grids are not mentioned.
std::string describe_transformation(const DatumTransformation& transformation);

/// The grid ids an operation needs that are not installed, one per line, or an
/// empty string when every grid is present. The message `CrsTransform::create`
/// throws when a pinned or selected operation cannot run.
std::string missing_grids_summary(const DatumTransformation& transformation);

}  // namespace alphashape::crs
