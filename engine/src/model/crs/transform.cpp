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

/// The operation's authority identifier as "AUTH:CODE", empty when anonymous.
std::string operation_id(PJ* operation) {
    const char* auth = proj_get_id_auth_name(operation, 0);
    const char* code = proj_get_id_code(operation, 0);
    if (auth == nullptr || code == nullptr) {
        return std::string();
    }
    return std::string(auth) + ":" + code;
}

/// True when an operation's identifier matches "AUTH:CODE" (case-insensitive).
bool operation_matches(PJ* operation, const std::string& wanted) {
    const std::string id = operation_id(operation);
    if (id.empty() || id.size() != wanted.size()) {
        return false;
    }
    for (std::size_t i = 0; i < id.size(); ++i) {
        if (std::toupper(static_cast<unsigned char>(id[i])) !=
            std::toupper(static_cast<unsigned char>(wanted[i]))) {
            return false;
        }
    }
    return true;
}

/// Build an axis-normalized pipeline for one CRS pair. With `authority` empty
/// PROJ picks the operation itself; otherwise the pipeline is built from that
/// operation, so the pinned choice is the one actually used.
PjPtr build_pipeline(PJ_CONTEXT* ctx, PJ* src_pj, PJ* dst_pj, const std::string& authority) {
    std::string option;
    const char* option_storage[2] = {nullptr, nullptr};
    if (!authority.empty()) {
        option = "AUTHORITY=" + authority;
        option_storage[0] = option.c_str();
    }
    PjPtr pipeline{proj_create_crs_to_crs_from_pj(ctx, src_pj, dst_pj, nullptr,
                                                  option_storage[0] != nullptr ? option_storage
                                                                               : nullptr)};
    if (!pipeline) {
        return PjPtr{};
    }
    return PjPtr{proj_normalize_for_visualization(ctx, pipeline.get())};
}

/// Choose the operation PROJ should use and capture its accuracy/ballpark
/// The first grid the operation needs that is not installed, or "" when it can
/// run. PROJ's short name is what `+grids=` and a proj-data package use.
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
        if (available == 0 && short_name != nullptr && *short_name != '\0') {
            return short_name;
        }
    }
    return std::string();
}

/// status. Returns the operation's identifier so the pipeline can be built
/// from exactly that operation (empty when the operation is anonymous).
std::string select_operation(PJ_CONTEXT* ctx, PJ* src_pj, PJ* dst_pj, const CoordinateSystem& src,
                             const CoordinateSystem& dst, const TransformOptions& options,
                             double* accuracy, bool* ballpark) {
    // Missing optional grids must not remove an operation from the list: a
    // transform that only needs a grid PROJ cannot find is reported by the
    // pipeline build (WP-12.4 owns the "install proj-data" diagnosis), while a
    // silently shorter list would make `preferred_operation` fail confusingly.
    FactoryContextPtr factory{proj_create_operation_factory_context(ctx, nullptr)};
    if (!factory) {
        throw Error("cannot create a PROJ operation factory context: " + proj_context_error(ctx));
    }
    proj_operation_factory_context_set_grid_availability_use(
        ctx, factory.get(), PROJ_GRID_AVAILABILITY_IGNORED);
    ObjListPtr operations{proj_create_operations(ctx, src_pj, dst_pj, factory.get())};
    if (!operations) {
        throw Error("cannot list datum transformations from " + quoted(src) + " to " +
                    quoted(dst) + ": " + proj_context_error(ctx));
    }
    const int count = proj_list_get_count(operations.get());
    if (count <= 0) {
        throw Error("no datum transformation from " + quoted(src) + " to " + quoted(dst));
    }

    PjPtr chosen;
    if (options.preferred_operation.empty()) {
        chosen.reset(proj_list_get(ctx, operations.get(), 0));
    } else {
        for (int i = 0; i < count && !chosen; ++i) {
            PjPtr candidate{proj_list_get(ctx, operations.get(), i)};
            if (candidate && operation_matches(candidate.get(), options.preferred_operation)) {
                chosen = std::move(candidate);
            }
        }
        if (!chosen) {
            throw Error("preferred operation " + options.preferred_operation +
                        " is not available from " + quoted(src) + " to " + quoted(dst));
        }
    }
    if (!chosen) {
        throw Error("no datum transformation from " + quoted(src) + " to " + quoted(dst) + ": " +
                    proj_context_error(ctx));
    }

    *ballpark = proj_coordoperation_has_ballpark_transformation(ctx, chosen.get()) == 1;
    *accuracy = proj_coordoperation_get_accuracy(ctx, chosen.get());
    if (*accuracy <= 0.0 && !*ballpark) {
        *accuracy = -1.0;  // PROJ reports 0 for "not stated" on conversions
    }

    // A grid this operation needs but PROJ cannot find is diagnosed by name.
    // Without this the pipeline would build and either degrade or fail with
    // "File not found", which says nothing about installing proj-data.
    const std::string missing = first_missing_grid(ctx, chosen.get());
    if (!missing.empty()) {
        throw Error("transformation needs grid " + missing + "; install proj-data");
    }

    if (*ballpark && !options.allow_ballpark) {
        throw Error("no datum transformation from " + quoted(src) + " to " + quoted(dst) +
                    "; enable allow_ballpark");
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
    std::string operation_authority;  ///< "EPSG:16033", empty = PROJ's choice
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
        PjPtr normalized = build_pipeline(ctx, src_pj.get(), dst_pj.get(), operation_authority);
        if (!normalized) {
            throw Error("PROJ could not build a transform from " + quoted(src) + " to " +
                        quoted(dst) + ": " + proj_context_error(ctx));
        }
        return normalized;
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

    impl->operation_authority = select_operation(ctx, src_pj.get(), dst_pj.get(), src, dst, options,
                                                 &impl->accuracy, &impl->ballpark);

    // Build once here so a broken pipeline fails at create() rather than on
    // first use deep inside a task. The built pipeline is also the authority on
    // what will actually be applied: an operation listed as non-ballpark can
    // still degrade to a ballpark at build time when the grid it needs is not
    // installed, and that must hit the same refusal.
    PjPtr pipeline = impl->build();
    if (proj_coordoperation_has_ballpark_transformation(ctx, pipeline.get()) == 1) {
        impl->ballpark = true;
        if (!options.allow_ballpark) {
            throw Error("no datum transformation from " + quoted(src) + " to " + quoted(dst) +
                        "; enable allow_ballpark");
        }
    }
    const double built_accuracy = proj_coordoperation_get_accuracy(ctx, pipeline.get());
    if (built_accuracy > 0.0) {
        impl->accuracy = built_accuracy;
    }
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
