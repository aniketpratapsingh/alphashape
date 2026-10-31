#include "model/crs/transform.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
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

struct ObjListDeleter {
    void operator()(PJ_OBJ_LIST* list) const {
        if (list != nullptr) {
            proj_list_destroy(list);
        }
    }
};
using ObjListPtr = std::unique_ptr<PJ_OBJ_LIST, ObjListDeleter>;

struct FactoryContextDeleter {
    void operator()(PJ_OPERATION_FACTORY_CONTEXT* factory) const {
        if (factory != nullptr) {
            proj_operation_factory_context_destroy(factory);
        }
    }
};
using FactoryContextPtr = std::unique_ptr<PJ_OPERATION_FACTORY_CONTEXT, FactoryContextDeleter>;

std::string proj_context_error(PJ_CONTEXT* ctx) {
    const int err = proj_context_errno(ctx);
    const char* text = err != 0 ? proj_context_errno_string(ctx, err) : nullptr;
    return text != nullptr && *text != '\0' ? std::string(text) : std::string("unknown PROJ error");
}

std::string quoted(const CoordinateSystem& cs) { return "\"" + cs.name() + "\""; }

/// Case-insensitive equality, used for operation identifiers.
bool equals_ignoring_case(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::toupper(static_cast<unsigned char>(a[i])) !=
            std::toupper(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

/// An authority name without PROJ's computed wrappers: "INVERSE(DERIVED_FROM
/// (EPSG))" -> "EPSG", "DERIVED_FROM(EPSG)" -> "EPSG", "EPSG" -> "EPSG".
std::string unwrapped_authority(std::string name) {
    for (;;) {
        const std::size_t open = name.find('(');
        const std::size_t close = name.rfind(')');
        if (open == std::string::npos || close != name.size() - 1 || close <= open + 1) {
            return name;
        }
        name = name.substr(open + 1, close - open - 1);
    }
}

/// Split "AUTH:CODE" into its parts; either is empty when there is no colon.
std::pair<std::string, std::string> split_id(const std::string& id) {
    const std::size_t colon = id.find(':');
    if (colon == std::string::npos) {
        return {std::string(), std::string()};
    }
    return {id.substr(0, colon), id.substr(colon + 1)};
}

/// The operation's authority identifier as "AUTH:CODE", empty when anonymous.
std::string operation_id(PJ* operation) {
    const char* auth = proj_get_id_auth_name(operation, 0);
    const char* code = proj_get_id_code(operation, 0);
    if (auth == nullptr || code == nullptr) {
        return std::string();
    }
    return std::string(auth) + ":" + code;
}

/// True when an operation's identifier matches `wanted` (case-insensitive).
///
/// PROJ reports some operations under a computed authority name, so the
/// reported id of the reverse of NAD 27 to WGS 84 (33) is
/// "INVERSE(DERIVED_FROM(EPSG)):1693". A caller who writes "EPSG:1693" means
/// that same operation, so the wrappers are stripped from both sides before
/// comparing; the code must still match exactly.
bool operation_matches(PJ* operation, const std::string& wanted) {
    const std::string reported = operation_id(operation);
    if (reported.empty() || wanted.empty()) {
        return reported.empty() && wanted.empty();
    }
    if (equals_ignoring_case(reported, wanted)) {
        return true;
    }
    const auto [reported_auth, reported_code] = split_id(reported);
    const auto [wanted_auth, wanted_code] = split_id(wanted);
    if (reported_code.empty() || wanted_code.empty()) {
        return false;
    }
    return equals_ignoring_case(reported_code, wanted_code) &&
           equals_ignoring_case(unwrapped_authority(reported_auth),
                               unwrapped_authority(wanted_auth));
}

/// True when PROJ can be asked for an object's operation data. A per-point
/// pipeline and a CRS report "not a CoordinateOperation", which PROJ prints to
/// stderr, so these questions are only asked of objects that can answer.
bool is_coordinate_operation(PJ* pj) {
    switch (proj_get_type(pj)) {
        case PJ_TYPE_CONVERSION:
        case PJ_TYPE_TRANSFORMATION:
        case PJ_TYPE_CONCATENATED_OPERATION:
        case PJ_TYPE_OTHER_COORDINATE_OPERATION:
            return true;
        default:
            return false;
    }
}

/// True when PROJ can be asked for an object's grids. A transformation, a
/// concatenation, or an "other coordinate operation" carries such a step; a
/// plain conversion cannot, and a per-point pipeline (type 0) is not a
/// coordinate operation at all — asking it makes PROJ print "Object is not a
/// CoordinateOperation". `proj_normalize_for_visualization()` can turn a
/// transformation into the "other" type, so both are accepted.
bool reads_grids(PJ* pj) {
    const PJ_TYPE type = proj_get_type(pj);
    return type == PJ_TYPE_TRANSFORMATION || type == PJ_TYPE_CONCATENATED_OPERATION ||
           type == PJ_TYPE_OTHER_COORDINATE_OPERATION;
}

/// The first grid the operation needs that is not installed, or "" when it can
/// run. PROJ's short name is what `+grids=` and a proj-data package use.
///
/// An optional grid (`+grids=@name`) is skipped: PROJ was told the operation
/// may run without it, so reporting it would refuse a pipeline PROJ is happy
/// to execute. PROJ marks such a grid with a leading '@' in its short name, so
/// no name reported from here carries that marker either.
std::string first_missing_grid(PJ_CONTEXT* ctx, PJ* operation) {
    const int count = proj_coordoperation_get_grid_used_count(ctx, operation);
    for (int i = 0; i < count; ++i) {
        const char* short_name = nullptr;
        const char* full_name = nullptr;
        const char* package_name = nullptr;
        const char* url = nullptr;
        int direct_download = 0;
        int open_license = 0;
        int available = 0;
        if (proj_coordoperation_get_grid_used(ctx, operation, i, &short_name, &full_name,
                                              &package_name, &url, &direct_download, &open_license,
                                              &available) == 0) {
            continue;
        }
        if (short_name == nullptr || *short_name == '\0' || *short_name == '@') {
            continue;
        }
        if (available == 0) {
            return short_name;
        }
    }
    return std::string();
}

/// Candidate operations for a CRS pair, as `available_transformations()` lists
/// them: grid availability ignored (so a missing grid is diagnosed rather than
/// hiding the operation) and areas intersected (so an area-limited national
/// operation stays selectable). Only pinning and diagnosis may use this list;
/// the automatic path must never select from it, because its first entry is not
/// the operation PROJ would apply at a given point.
ObjListPtr list_candidates(PJ_CONTEXT* ctx, PJ* src_pj, PJ* dst_pj, const CoordinateSystem& src,
                           const CoordinateSystem& dst) {
    FactoryContextPtr factory{proj_create_operation_factory_context(ctx, nullptr)};
    if (!factory) {
        throw Error("cannot create a PROJ operation factory context: " + proj_context_error(ctx));
    }
    proj_operation_factory_context_set_grid_availability_use(ctx, factory.get(),
                                                             PROJ_GRID_AVAILABILITY_IGNORED);
    proj_operation_factory_context_set_spatial_criterion(
        ctx, factory.get(), PROJ_SPATIAL_CRITERION_PARTIAL_INTERSECTION);
    ObjListPtr operations{proj_create_operations(ctx, src_pj, dst_pj, factory.get())};
    if (!operations) {
        throw Error("cannot list datum transformations from " + quoted(src) + " to " + quoted(dst) +
                    ": " + proj_context_error(ctx));
    }
    return operations;
}

/// The listed operation whose identifier matches `wanted`, or null.
PjPtr find_operation(PJ_CONTEXT* ctx, PJ_OBJ_LIST* operations, const std::string& wanted) {
    const int count = proj_list_get_count(operations);
    for (int i = 0; i < count; ++i) {
        PjPtr candidate{proj_list_get(ctx, operations, i)};
        if (candidate && operation_matches(candidate.get(), wanted)) {
            return candidate;
        }
    }
    return PjPtr{};
}

/// Throw the "install proj-data" diagnosis when `pj` needs a grid that is not
/// installed. Only objects that can answer the question are inspected (see
/// `reads_grids`); everything else passes.
void diagnose_missing_grid(PJ_CONTEXT* ctx, PJ* pj) {
    if (pj == nullptr || !reads_grids(pj)) {
        return;
    }
    // Without this the pipeline would build and either degrade or fail with
    // "File not found", which says nothing about installing proj-data.
    const std::string missing = first_missing_grid(ctx, pj);
    if (!missing.empty()) {
        throw Error("transformation needs grid " + missing + "; install proj-data");
    }
}

/// The pipeline PROJ builds for a CRS pair with no operation pinned.
///
/// This is deliberately the bare call, with no options: PROJ then keeps every
/// candidate operation and chooses per point, respecting each operation's area
/// of use and the grids that are actually installed. That is the only way an
/// area-limited national operation (`Amersfoort to WGS 84 (4)` over the
/// Netherlands, `OSGB36 to ETRS89 (1)` over Great Britain) is applied where it
/// belongs and skipped where it does not.
PjPtr automatic_pipeline(PJ_CONTEXT* ctx, PJ* src_pj, PJ* dst_pj) {
    PjPtr pipeline{proj_create_crs_to_crs_from_pj(ctx, src_pj, dst_pj, nullptr, nullptr)};
    if (!pipeline) {
        return PjPtr{};
    }
    return PjPtr{proj_normalize_for_visualization(ctx, pipeline.get())};
}

/// The pipeline for one chosen operation object.
///
/// Pinning exists to override PROJ's per-point choice, so the pipeline is the
/// operation itself — normalised to visualisation order — not a re-request of
/// "the operation with this identifier" from the pair. Round-tripping the id
/// through `proj_create_crs_to_crs_from_pj`'s `AUTHORITY=` option is wrong twice
/// over: the option is an authority-name filter (it cannot express an operation
/// id), and an id PROJ computes rather than stores ("DERIVED_FROM(EPSG):1313")
/// is not one proj.db can answer for, so it silently yields a null shift.
PjPtr pinned_pipeline(PJ_CONTEXT* ctx, PJ* operation) {
    PjPtr normalized{proj_normalize_for_visualization(ctx, operation)};
    return normalized;
}

/// Describe the pair for `accuracy_m()` and the ballpark refusal: does any real
/// operation exist for these two CRS extents, and what is the best accuracy
/// stated among them?
///
/// This is a *diagnosis*, not a selection: the pipeline itself is built by
/// `automatic_pipeline`, so each point is transformed by whichever operation
/// PROJ deems right for it. Selecting from this list instead would hand one
/// area-limited national operation to every point in the world.
///
/// The listing is deliberately a *partial-intersection* one (see
/// `list_candidates`), not PROJ's default strict containment. That difference
/// is the whole point. Strict containment asks each operation's own area to
/// contain both CRSs' entire extents; a projection's area of use is routinely
/// larger than the transformation that reaches it — EPSG:3035 covers Europe to
/// 24.6°N, while `ETRS89 to WGS 84 (1)` (EPSG:1149) stops at 33.26°N — so the
/// strict listing offers WGS 84 -> EPSG:3035 nothing but the world ballpark.
/// PROJ's own per-point pipeline still applies the 1 m operation everywhere
/// that operation covers, because per point it only has to contain *that
/// point*. Refusing on the strict list therefore rejected a pair the engine
/// transforms correctly, which is exactly the bug fixed here.
///
/// So the refusal fires only when *no* non-ballpark candidate intersects the
/// pair at all. A datum with a published transformation to WGS 84 is accepted
/// even where that transformation's area is smaller than the target's, and a
/// pair that has nothing but a ballpark (a datum with no published
/// transformation, or a custom `+ellps=` with no datum name) is still refused.
void describe_pair(PJ_CONTEXT* ctx, PJ* src_pj, PJ* dst_pj, const CoordinateSystem& src,
                   const CoordinateSystem& dst, double* accuracy, bool* ballpark) {
    ObjListPtr operations = list_candidates(ctx, src_pj, dst_pj, src, dst);
    const int count = proj_list_get_count(operations.get());
    if (count <= 0) {
        throw Error("no datum transformation from " + quoted(src) + " to " + quoted(dst));
    }
    // Best (smallest positive) accuracy among the non-ballpark candidates, -1
    // when PROJ states none for any of them. A conversion reports 0 m for "not
    // stated", which is not the unknown sentinel the API uses, so 0 never wins.
    // The listing ignores grid availability (`list_candidates`), so this is the
    // best accuracy PROJ can offer for the pair, not necessarily one that runs
    // on a machine missing a grid — the same number `available_transformations`
    // reports for the same candidates.
    bool have_real = false;
    double best_accuracy = -1.0;
    for (int i = 0; i < count; ++i) {
        PjPtr candidate{proj_list_get(ctx, operations.get(), i)};
        if (!candidate) {
            throw Error("no datum transformation from " + quoted(src) + " to " + quoted(dst) +
                        ": " + proj_context_error(ctx));
        }
        if (proj_coordoperation_has_ballpark_transformation(ctx, candidate.get()) == 1) {
            continue;
        }
        have_real = true;
        const double candidate_accuracy = proj_coordoperation_get_accuracy(ctx, candidate.get());
        if (candidate_accuracy > 0.0 &&
            (best_accuracy <= 0.0 || candidate_accuracy < best_accuracy)) {
            best_accuracy = candidate_accuracy;
        }
    }
    *ballpark = !have_real;
    *accuracy = have_real ? best_accuracy : -1.0;
    // No grid diagnosis here. A listed candidate is *not* what a point is
    // transformed by — PROJ keeps every candidate in the automatic pipeline and
    // picks per point — so an absent grid on one of them does not mean this pair
    // cannot run. On a machine without OSTN15, OSGB36 -> WGS 84 still transforms
    // every point (PROJ falls back to a grid-free shift) while a candidate needs
    // `uk_os_OSTN15_NTv2_OSGBtoETRS.tif`; refusing here would reject a transform
    // that works. The diagnosis belongs on the pipeline actually built — see
    // `select_operation`.
}

/// Choose the operation PROJ should use and capture its accuracy/ballpark
/// status. Returns the operation's identifier when the pipeline is pinned to one
/// (empty for the automatic, per-point pipeline), so another thread can rebuild
/// exactly the same choice.
std::string select_operation(PJ_CONTEXT* ctx, PJ* src_pj, PJ* dst_pj, const CoordinateSystem& src,
                             const CoordinateSystem& dst, const TransformOptions& options,
                             double* accuracy, bool* ballpark, PjPtr* pipeline) {
    if (options.preferred_operation.empty()) {
        // PROJ's own pipeline: no options and no pre-selected operation, so it
        // keeps every candidate and chooses per point by area of use and grid
        // availability. This is the same construction the engine has always
        // used, which is what keeps every pair's numbers unchanged.
        *pipeline = automatic_pipeline(ctx, src_pj, dst_pj);
        if (!*pipeline) {
            throw Error("PROJ could not build a transform from " + quoted(src) + " to " +
                        quoted(dst) + ": " + proj_context_error(ctx));
        }
        // The grid diagnosis is asked of the pipeline PROJ actually built, not
        // of a listed candidate: this pipeline is the thing that will run, so
        // its grids are the ones this pair needs. A per-point pipeline (type 0)
        // is not a coordinate operation and cannot be asked — and it only
        // exists when PROJ found something it can always run, so there is
        // nothing to diagnose on it. A fixed pipeline that needs an absent grid
        // (a custom CRS's `+nadgrids=`) is reported by name here rather than
        // failing later with "File not found".
        diagnose_missing_grid(ctx, pipeline->get());
        describe_pair(ctx, src_pj, dst_pj, src, dst, accuracy, ballpark);
        if (*ballpark && !options.allow_ballpark) {
            throw Error("no datum transformation from " + quoted(src) + " to " + quoted(dst) +
                        "; enable allow_ballpark");
        }
        return std::string();
    }

    // A pinned operation is looked up in the pair's candidate list, which keeps
    // every operation (grid availability ignored, areas intersected) so a
    // national operation stays selectable and a missing grid stays diagnosable.
    // The automatic path above never selects from a listing; it only diagnoses
    // with one.
    ObjListPtr operations = list_candidates(ctx, src_pj, dst_pj, src, dst);
    PjPtr chosen = find_operation(ctx, operations.get(), options.preferred_operation);
    if (!chosen) {
        throw Error("preferred operation " + options.preferred_operation +
                    " is not available from " + quoted(src) + " to " + quoted(dst));
    }
    *ballpark = proj_coordoperation_has_ballpark_transformation(ctx, chosen.get()) == 1;
    *accuracy = proj_coordoperation_get_accuracy(ctx, chosen.get());
    if (*accuracy <= 0.0 && !*ballpark) {
        *accuracy = -1.0;  // PROJ reports 0 for "not stated" on conversions
    }
    diagnose_missing_grid(ctx, chosen.get());
    if (*ballpark && !options.allow_ballpark) {
        throw Error("no datum transformation from " + quoted(src) + " to " + quoted(dst) +
                    "; enable allow_ballpark");
    }
    *pipeline = pinned_pipeline(ctx, chosen.get());
    if (!*pipeline) {
        throw Error("PROJ could not build the pinned operation " + options.preferred_operation +
                    " for " + quoted(src) + " to " + quoted(dst) + ": " + proj_context_error(ctx));
    }
    return operation_id(chosen.get());
}

/// Single-point transform. Failure = HUGE_VAL/NaN output or a set proj_errno.
std::optional<CrsPoint> trans_one(PJ* pj, PJ_DIRECTION direction, const CrsPoint& point) {
    if (pj == nullptr) {
        return std::nullopt;
    }
    proj_errno_reset(pj);
    const PJ_COORD in = proj_coord(point.x, point.y, point.z, 0.0);
    const PJ_COORD out = proj_trans(pj, direction, in);
    if (!std::isfinite(out.xyzt.x) || !std::isfinite(out.xyzt.y) || !std::isfinite(out.xyzt.z) ||
        proj_errno(pj) != 0) {
        return std::nullopt;
    }
    return CrsPoint{out.xyzt.x, out.xyzt.y, out.xyzt.z};
}

/// Batch transform via proj_trans_generic. PROJ marks a failed point with
/// HUGE_VAL/NaN; that becomes a validity flag, so one bad point does not
/// poison the block, and failed points carry NaN.
CrsTransform::Batch transform_batch(PJ* pj, PJ_DIRECTION direction,
                                    const std::vector<CrsPoint>& points) {
    CrsTransform::Batch batch;
    batch.points.assign(points.size(), CrsPoint{});
    batch.valid.assign(points.size(), 0);
    if (points.empty() || pj == nullptr) {
        return batch;
    }

    const std::size_t n = points.size();
    std::vector<double> x(n);
    std::vector<double> y(n);
    std::vector<double> z(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = points[i].x;
        y[i] = points[i].y;
        z[i] = points[i].z;
    }

    proj_errno_reset(pj);
    const std::size_t stride = sizeof(double);
    proj_trans_generic(pj, direction, x.data(), stride, n, y.data(), stride, n, z.data(), stride, n,
                       nullptr, 0, 0);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(x[i]) && std::isfinite(y[i]) && std::isfinite(z[i])) {
            batch.points[i] = CrsPoint{x[i], y[i], z[i]};
            batch.valid[i] = 1;
        } else {
            batch.points[i] = CrsPoint{nan, nan, nan};
        }
    }
    return batch;
}

void require_transformable(const CoordinateSystem& cs, const char* role) {
    if (cs.is_local()) {
        throw ValueError(std::string("cannot transform the local engineering frame as the ") + role +
                         "; use LocalFrame instead");
    }
    if (!cs.is_set()) {
        throw ValueError(std::string("the ") + role + " coordinate system is not set");
    }
}

void require_ellipsoidal(const std::optional<VerticalCrs>& vertical, const char* role) {
    if (vertical && vertical->is_set()) {
        throw ValueError(std::string("vertical datum \"") + vertical->name + "\" on the " + role +
                         " CRS is not supported by the transform engine yet; geoid shifts are "
                         "applied by the geoid engine (WP-12.5)");
    }
}

}  // namespace

namespace {

/// Per-thread pipeline cache, keyed by the transform that owns it.
///
/// A PROJ operation belongs to the `PJ_CONTEXT` that created it, and this
/// engine keeps one context per thread (`crs::thread_context()`, destroyed at
/// thread exit). Caching pipelines in the transform would therefore mean one
/// thread destroying another thread's operation against a context that may
/// already be gone, so the cache lives on the thread instead: each pipeline is
/// destroyed either by that thread's later use of the transform, or by this
/// table's destructor at thread exit — before the context it belongs to.
struct ThreadPipelines {
    /// weak_ptr to the owning Impl: an expired entry means the transform died
    /// while this thread still had its pipeline.
    std::unordered_map<const void*, std::pair<std::weak_ptr<const void>, PJ*>> entries;

    /// Force the thread's PJ_CONTEXT to exist before this table does, so the
    /// table (and the pipelines in it) is torn down *before* the context it
    /// belongs to: thread-local destructors run in reverse construction order.
    ThreadPipelines() { (void)thread_context(); }
    ThreadPipelines(const ThreadPipelines&) = delete;
    ThreadPipelines& operator=(const ThreadPipelines&) = delete;

    ~ThreadPipelines() {
        for (auto& entry : entries) {
            proj_destroy(entry.second.second);
        }
    }
};

thread_local ThreadPipelines tl_pipelines;

}  // namespace

/// The transform's immutable definition plus the information needed to build
/// its pipeline on any thread. No PROJ object is stored here: the pipeline
/// lives in `tl_pipelines` (see above).
struct CrsTransform::Impl {
    CoordinateSystem src;
    CoordinateSystem dst;
    TransformOptions options;
    std::string src_wkt;
    std::string dst_wkt;
    bool identity = false;
    /// "AUTH:CODE" of the operation the pipeline is pinned to; empty means
    /// PROJ's own per-point choice, rebuilt by re-asking for the pair.
    std::string operation_authority;
    std::string description;
    double accuracy = -1.0;
    bool ballpark = false;
    /// Set by create(); lets a thread's cache notice this transform is gone.
    std::weak_ptr<const void> token;

    /// Build the axis-normalized pipeline on the calling thread's context.
    PjPtr build() const {
        PJ_CONTEXT* ctx = thread_context();
        PjPtr src_pj{proj_create(ctx, src_wkt.c_str())};
        PjPtr dst_pj{proj_create(ctx, dst_wkt.c_str())};
        if (!src_pj || !dst_pj) {
            throw Error("cannot recreate a stored CRS for " + quoted(src) + " -> " + quoted(dst) +
                        ": " + proj_context_error(ctx));
        }
        if (operation_authority.empty()) {
            PjPtr automatic = automatic_pipeline(ctx, src_pj.get(), dst_pj.get());
            if (!automatic) {
                throw Error("PROJ could not build a transform from " + quoted(src) + " to " +
                            quoted(dst) + ": " + proj_context_error(ctx));
            }
            return automatic;
        }
        // Pinned: find the very operation again and build from the object, so
        // the pipeline is that operation rather than an id looked up in a
        // different context (ids PROJ computes are not answerable by proj.db).
        ObjListPtr operations = list_candidates(ctx, src_pj.get(), dst_pj.get(), src, dst);
        PjPtr chosen = find_operation(ctx, operations.get(), operation_authority);
        if (!chosen) {
            throw Error("pinned operation " + operation_authority +
                        " is no longer available from " + quoted(src) + " to " + quoted(dst));
        }
        PjPtr pinned = pinned_pipeline(ctx, chosen.get());
        if (!pinned) {
            throw Error("PROJ could not build the pinned operation " + operation_authority +
                        " for " + quoted(src) + " to " + quoted(dst) + ": " +
                        proj_context_error(ctx));
        }
        return pinned;
    }

    /// Register `pipeline` as this thread's, replacing one left by a transform
    /// at the same address (an Impl can be freed and another allocated there).
    void store_here(PjPtr pipeline) const {
        auto& slot = tl_pipelines.entries[static_cast<const void*>(this)];
        if (slot.second != nullptr) {
            proj_destroy(slot.second);
        }
        slot.first = token;
        slot.second = pipeline.release();
    }

    /// The calling thread's pipeline, created on first use.
    PJ* pipeline_for_thread() const {
        auto& slot = tl_pipelines.entries[static_cast<const void*>(this)];
        if (slot.second != nullptr) {
            if (!slot.first.expired()) {
                return slot.second;
            }
            // This transform was destroyed; the address now belongs to another
            // (or to nothing). Drop the stale pipeline and rebuild below.
            proj_destroy(slot.second);
            slot.second = nullptr;
        }
        store_here(build());
        return tl_pipelines.entries[static_cast<const void*>(this)].second;
    }
};

std::size_t CrsTransform::Batch::failed() const {
    std::size_t count = 0;
    const std::size_t n = std::min(valid.size(), points.size());
    for (std::size_t i = 0; i < n; ++i) {
        count += valid[i] == 0 ? std::size_t{1} : std::size_t{0};
    }
    return count + (points.size() - n);
}

CrsTransform::CrsTransform() = default;

CrsTransform::CrsTransform(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

CrsTransform CrsTransform::create(const CoordinateSystem& src, const CoordinateSystem& dst,
                                  const TransformOptions& options) {
    require_transformable(src, "source");
    require_transformable(dst, "target");
    require_ellipsoidal(options.src_vertical, "source");
    require_ellipsoidal(options.dst_vertical, "target");

    auto impl = std::make_shared<Impl>();
    impl->token = impl;
    impl->src = src;
    impl->dst = dst;
    impl->options = options;
    impl->src_wkt = src.to_wkt();
    impl->dst_wkt = dst.to_wkt();

    if (src == dst) {
        // The same CRS: once axis order is normalised the operation is a no-op,
        // so short-circuit rather than building a pipeline.
        impl->identity = true;
        impl->description = "identity";
        impl->accuracy = 0.0;
        return CrsTransform(impl);
    }

    PJ_CONTEXT* ctx = thread_context();
    PjPtr src_pj{proj_create(ctx, impl->src_wkt.c_str())};
    PjPtr dst_pj{proj_create(ctx, impl->dst_wkt.c_str())};
    if (!src_pj || !dst_pj) {
        throw Error("cannot recreate the stored CRS for " + quoted(src) + " -> " + quoted(dst) +
                    ": " + proj_context_error(ctx));
    }

    PjPtr pipeline;
    impl->operation_authority = select_operation(ctx, src_pj.get(), dst_pj.get(), src, dst, options,
                                                 &impl->accuracy, &impl->ballpark, &pipeline);

    // Build once here so a broken pipeline fails at create() rather than on
    // first use deep inside a task. The built pipeline is also the authority on
    // what will actually be applied: an operation listed as non-ballpark can
    // still degrade to a ballpark at build time when the grid it needs is not
    // installed, and that must hit the same refusal. A per-point pipeline has no
    // single operation to ask — PROJ decides for each point — so it is skipped.
    if (is_coordinate_operation(pipeline.get()) &&
        proj_coordoperation_has_ballpark_transformation(ctx, pipeline.get()) == 1) {
        impl->ballpark = true;
        if (!options.allow_ballpark) {
            throw Error("no datum transformation from " + quoted(src) + " to " + quoted(dst) +
                        "; enable allow_ballpark");
        }
    }
    // The factory-reported accuracy is the number to publish: a per-point
    // pipeline reports none (it may pick differently at every point), so a 0
    // here must not erase what the selected operation stated.
    const PJ_PROJ_INFO info = proj_pj_info(pipeline.get());
    impl->description = info.definition != nullptr ? std::string(info.definition) : std::string();
    impl->store_here(std::move(pipeline));
    return CrsTransform(impl);
}

const CoordinateSystem& CrsTransform::source() const {
    static const CoordinateSystem kEmpty;
    return impl_ ? impl_->src : kEmpty;
}

const CoordinateSystem& CrsTransform::target() const {
    static const CoordinateSystem kEmpty;
    return impl_ ? impl_->dst : kEmpty;
}

const TransformOptions& CrsTransform::options() const {
    static const TransformOptions kDefault;
    return impl_ ? impl_->options : kDefault;
}

const std::string& CrsTransform::description() const {
    static const std::string kEmpty;
    return impl_ ? impl_->description : kEmpty;
}

double CrsTransform::accuracy_m() const { return impl_ ? impl_->accuracy : -1.0; }

bool CrsTransform::is_ballpark() const { return impl_ ? impl_->ballpark : false; }

std::optional<CrsPoint> CrsTransform::forward(const CrsPoint& point) const {
    if (!impl_) {
        return std::nullopt;
    }
    if (impl_->identity) {
        return point;
    }
    return trans_one(impl_->pipeline_for_thread(), PJ_FWD, point);
}

std::optional<CrsPoint> CrsTransform::try_inverse(const CrsPoint& point) const {
    if (!impl_) {
        return std::nullopt;
    }
    if (impl_->identity) {
        return point;
    }
    return trans_one(impl_->pipeline_for_thread(), PJ_INV, point);
}

CrsPoint CrsTransform::inverse(const CrsPoint& point) const {
    if (const std::optional<CrsPoint> result = try_inverse(point)) {
        return *result;
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return CrsPoint{nan, nan, nan};
}

CrsTransform::Batch CrsTransform::forward_batch(const std::vector<CrsPoint>& points) const {
    if (!impl_ || impl_->identity) {
        Batch batch;
        batch.points = impl_ ? points : std::vector<CrsPoint>(points.size(), CrsPoint{});
        batch.valid.assign(points.size(), impl_ ? 1 : 0);
        return batch;
    }
    return transform_batch(impl_->pipeline_for_thread(), PJ_FWD, points);
}

CrsTransform::Batch CrsTransform::inverse_batch(const std::vector<CrsPoint>& points) const {
    if (!impl_ || impl_->identity) {
        Batch batch;
        batch.points = impl_ ? points : std::vector<CrsPoint>(points.size(), CrsPoint{});
        batch.valid.assign(points.size(), impl_ ? 1 : 0);
        return batch;
    }
    return transform_batch(impl_->pipeline_for_thread(), PJ_INV, points);
}

}  // namespace alphashape::crs
