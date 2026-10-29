#pragma once

/// @file transform.hpp
/// @brief Coordinate transform between two `CoordinateSystem`s (M12 §4.3).
///
/// A `CrsTransform` owns no PROJ object: it records the two CRSs and the
/// options, plus the operation's description and accuracy captured at creation.
/// PROJ operations are bound to a `PJ_CONTEXT`, which is per thread, so the
/// actual pipeline is created lazily on the calling thread and cached in that
/// thread's registry, released at thread exit or when the transform is dropped
/// (see transform.cpp). A transform value is therefore cheap to copy, safe to
/// share across threads, and never touches PROJ twice for the same thread.
///
/// Axis order: every point crossing this API is in *visualisation* order,
/// `x = east/lon`, `y = north/lat`, `z = up` (metres, ellipsoidal height).
/// PROJ's authority axis order for a geographic CRS (e.g. EPSG:4326 is
/// lat, lon) is normalised away with `proj_normalize_for_visualization()`, so
/// `CrsPoint{9.0, 50.0, 0.0}` always means 9°E 50°N, never 9°N 50°E.
///
/// Datum shift policy: `TransformOptions::allow_ballpark` defaults to false and
/// a pipeline that is only a ballpark (a null datum shift, accuracy 0) is
/// rejected at `create()` time. Refusing by default means a coordinate is never
/// silently off by hundreds of metres.
///
/// With no `preferred_operation` the pipeline is PROJ's own: PROJ keeps every
/// candidate operation and picks per point, so an area-limited national
/// operation is applied inside its area of use and skipped outside it. Pinning
/// an operation forces that one everywhere (see WP-12.4 / `datum.hpp`).
///
/// Out of scope here: datum-transformation selection and grid installation
/// (WP-12.4) and geoid evaluation (WP-12.5). `preferred_operation` pins an
/// operation PROJ already knows; vertical components are rejected until the
/// geoid engine can apply them (see `TransformOptions`).

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "model/crs/coordinate_system.hpp"

namespace alphashape::crs {

/// A coordinate in a CRS, always in visualisation order (east/north/up).
struct CrsPoint {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline bool operator==(const CrsPoint& a, const CrsPoint& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

struct TransformOptions {
    /// PROJ operation authority, e.g. "EPSG:1613" (M12 §4.3). Empty lets PROJ
    /// choose per point: an area-limited national operation applies inside its
    /// area of use and another (less accurate, or the ballpark) takes over
    /// outside it — a bare PROJ transform. Setting it pins that one operation
    /// for every point instead.
    ///
    /// The id is matched against each candidate's `AUTH:CODE` after stripping
    /// PROJ's computed wrappers, so "EPSG:1693" finds the listed
    /// "INVERSE(DERIVED_FROM(EPSG)):1693". An unknown code is an error; the
    /// operation actually used is visible through `description()`.
    std::string preferred_operation;

    /// Allow a ballpark (null) datum shift. Default false: refuse rather than
    /// return coordinates that are off by the datum offset.
    bool allow_ballpark = false;

    /// Vertical datums. Only the ellipsoidal kind is accepted: applying a geoid
    /// needs WP-12.5, so a set vertical component is rejected rather than
    /// silently ignored (heights would otherwise be tens of metres wrong).
    std::optional<VerticalCrs> src_vertical;
    std::optional<VerticalCrs> dst_vertical;
};

class CrsTransform {
public:
    /// Result of transforming a batch: one output per input and a validity
    /// flag per output (0 = the point failed, its coordinates are NaN).
    struct Batch {
        std::vector<CrsPoint> points;
        std::vector<unsigned char> valid;

        std::size_t size() const { return points.size(); }
        bool is_valid(std::size_t i) const { return i < valid.size() && valid[i] != 0; }
        /// Number of points PROJ could not transform.
        std::size_t failed() const;
    };

    /// An invalid transform; `is_valid()` is false. Provided so callers can
    /// hold one by value before the CRSs are known.
    CrsTransform();

    /// Resolve the operation between `src` and `dst`.
    /// @throws ValueError when a CRS is the local engineering frame (it needs a
    ///         LocalFrame, CRS-05, not a plain transform), when `src`/`dst`
    ///         is unset, or when a vertical component was supplied.
    /// @throws Error when PROJ has no operation for the pair, when the only
    ///         operation is a ballpark and `allow_ballpark` is false, or when
    ///         `preferred_operation` cannot be used.
    static CrsTransform create(const CoordinateSystem& src, const CoordinateSystem& dst,
                               const TransformOptions& options = {});

    bool is_valid() const { return impl_ != nullptr; }

    const CoordinateSystem& source() const;
    const CoordinateSystem& target() const;
    const TransformOptions& options() const;

    /// Source -> target. nullopt when PROJ reports an error (HUGE_VAL output
    /// or `proj_errno` set), e.g. an input outside the projection's domain.
    std::optional<CrsPoint> forward(const CrsPoint& point) const;

    /// Target -> source, same failure rule. `inverse()` is the unchecked form:
    /// a failed point comes back with NaN coordinates.
    std::optional<CrsPoint> try_inverse(const CrsPoint& point) const;
    CrsPoint inverse(const CrsPoint& point) const;

    /// Batch forms (proj_trans_generic); per-point validity is preserved, so a
    /// single bad point does not discard the rest of the block.
    Batch forward_batch(const std::vector<CrsPoint>& points) const;
    Batch inverse_batch(const std::vector<CrsPoint>& points) const;

    /// The PROJ pipeline text of the operation (logging/diagnostics). Captured
    /// when the transform was created, so it is identical on every thread.
    const std::string& description() const;
    /// Operation accuracy in metres from PROJ, -1 when unknown.
    double accuracy_m() const;
    /// True when the selected operation is a ballpark (null) datum shift. Only
    /// reachable with `allow_ballpark`.
    bool is_ballpark() const;

private:
    struct Impl;
    explicit CrsTransform(std::shared_ptr<const Impl> impl);

    std::shared_ptr<const Impl> impl_;
};

}  // namespace alphashape::crs
