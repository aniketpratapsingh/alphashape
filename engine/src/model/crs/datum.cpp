#include "model/crs/datum.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
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

/// The same guards `CrsTransform::create` applies, so both entry points agree
/// on what is not transformable.
void require_transformable(const CoordinateSystem& cs, const char* role) {
    if (cs.is_local()) {
        throw ValueError(std::string("cannot transform the local engineering frame as the ") + role +
                         "; use LocalFrame instead");
    }
    if (!cs.is_set()) {
        throw ValueError(std::string("the ") + role + " coordinate system is not set");
    }
}

/// PROJ reports some operations under a computed authority name
/// ("DERIVED_FROM(EPSG)", "INVERSE(EPSG)", "INVERSE(DERIVED_FROM(EPSG))") that
/// proj.db cannot answer for. Unwrapping all of them reaches the authority that
/// resolves: the code is the same and the wrapper only says how PROJ derived it.
std::string lookup_authority(std::string auth_name) {
    for (;;) {
        const std::size_t open = auth_name.find('(');
        const std::size_t close = auth_name.rfind(')');
        if (open == std::string::npos || close != auth_name.size() - 1 || close <= open + 1) {
            return auth_name;
        }
        auth_name = auth_name.substr(open + 1, close - open - 1);
    }
}

/// The grid id an operation names, without PROJ's optional-grid marker. PROJ
/// reports `+grids=@name` as the short name "@name" (leading '@' included);
/// the '@' means "use this grid if it is installed, otherwise carry on", so it
/// is a syntax marker rather than part of the file name, and an error message
/// that says "@x" would name no installable grid.
std::string grid_name(const char* short_name) {
    if (short_name == nullptr) {
        return std::string();
    }
    return *short_name == '@' ? std::string(short_name + 1) : std::string(short_name);
}

/// True when PROJ reports this grid as optional (`+grids=@name`). PROJ does not
/// publish the distinction through `proj_coordoperation_get_grid_used`, but it
/// keeps the '@' in the short name.
bool optional_grid(const char* short_name) {
    return short_name != nullptr && *short_name == '@';
}

/// The grid ids an operation requires, split into all and missing. PROJ reports
/// availability per grid; the short name is what `+grids=` names and what a
/// `proj-data` package installs, so it is what an error message must show.
///
/// An optional grid (`+grids=@name`) is never reported missing: PROJ itself
/// runs such a pipeline without it, having been told the absence is acceptable,
/// so treating it as a hard error would refuse a transform main ran fine.
void read_grids(PJ_CONTEXT* ctx, PJ* operation, DatumTransformation* out) {
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
        const std::string name = grid_name(short_name);
        if (name.empty()) {
            continue;
        }
        out->grids.emplace_back(name);
        if (available == 0 && !optional_grid(short_name)) {
            out->missing_grids.emplace_back(name);
        }
    }
}

/// The operation's pipeline text in visualisation order. Taken from a
/// per-operation normalised copy so two entries of the same list cannot
/// collapse to the pair's pipeline.
void read_definition(PJ_CONTEXT* ctx, PJ* operation, DatumTransformation* out) {
    PjPtr normalized{proj_normalize_for_visualization(ctx, operation)};
    const PJ* source = normalized ? normalized.get() : operation;
    const char* text = proj_as_proj_string(ctx, source, PJ_PROJ_5, nullptr);
    if (text != nullptr) {
        out->definition = text;
    }
}

/// Describe one operation of a list. `operation` is borrowed.
void describe_operation(PJ_CONTEXT* ctx, PJ* operation, DatumTransformation* out) {
    const char* auth = proj_get_id_auth_name(operation, 0);
    const char* code = proj_get_id_code(operation, 0);
    out->auth_name = auth != nullptr ? std::string(auth) : std::string();
    out->code = code != nullptr ? std::string(code) : std::string();

    const char* name = proj_get_name(operation);
    out->name = name != nullptr ? std::string(name) : std::string();

    const double accuracy = proj_coordoperation_get_accuracy(ctx, operation);
    out->accuracy_m = accuracy > 0.0 ? accuracy : -1.0;
    out->ballpark = proj_coordoperation_has_ballpark_transformation(ctx, operation) == 1;

    read_grids(ctx, operation, out);
    read_definition(ctx, operation, out);
}

/// Order the tail of the list: increasing accuracy, unknown accuracy last. A
/// stable sort keeps PROJ's own area judgement among equal accuracies.
bool ranks_before(const DatumTransformation& a, const DatumTransformation& b) {
    const bool a_known = a.accuracy_m > 0.0;
    const bool b_known = b.accuracy_m > 0.0;
    if (a_known != b_known) {
        return a_known;
    }
    return a_known && a.accuracy_m < b.accuracy_m;
}

/// Candidate operations for a CRS pair. Grid availability is IGNORED so a
/// missing grid never hides an operation (it is reported per entry instead) and
/// the spatial criterion is PARTIAL_INTERSECTION so every operation whose area
/// overlaps is listed rather than only those covering the whole pair.
ObjListPtr list_operations(PJ_CONTEXT* ctx, PJ* src_pj, PJ* dst_pj, const CoordinateSystem& src,
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
        throw Error("cannot list datum transformations from " + quoted(src) + " to " +
                    quoted(dst) + ": " + proj_context_error(ctx));
    }
    return operations;
}

PjPtr create_crs(PJ_CONTEXT* ctx, const CoordinateSystem& cs) {
    PjPtr pj{proj_create(ctx, cs.to_wkt().c_str())};
    if (!pj) {
        throw Error("cannot recreate the stored CRS for " + quoted(cs) + ": " +
                    proj_context_error(ctx));
    }
    return pj;
}

/// True when PROJ considers two objects the same CRS for coordinate-operation
/// purposes, ignoring identifier and axis order. The axis-order relaxation
/// matters because a listed operation's source CRS is usually in authority
/// (lat/lon) order while the engine's value normalises to visualisation order.
bool equivalent_crs(PJ_CONTEXT* ctx, PJ* a, PJ* b) {
    if (a == nullptr || b == nullptr) {
        return false;
    }
    return proj_is_equivalent_to_with_ctx(ctx, a, b,
                                          PJ_COMP_EQUIVALENT_EXCEPT_AXIS_ORDER_GEOGCRS) == 1;
}

/// The 2D form of a 3D geographic CRS, or null when the CRS is not one.
///
/// A datum shift is stored against the horizontal datum: EPSG:1693's source is
/// NAD27 (a 2D geographic CRS), even when the caller's source is the 3D
/// EPSG:4979. `PJ_COMP_EQUIVALENT_EXCEPT_AXIS_ORDER_GEOGCRS` does not equate a
/// 2D and a 3D CRS, so comparing a 3D base with an operation's source rejected
/// essentially every operation of a 3D-source pair (78 of the 79 listed for
/// 4979 -> 4267, 52 of the 53 for 4979 -> 4269).
///
/// `proj_crs_demote_to_2D` is the PROJ-blessed way to drop the height axis and
/// keep the datum and ellipsoid; the name override is left null so PROJ keeps
/// its own. It arrived in PROJ 6.3, which the build already requires
/// (find_package(PROJ 9.3)), so no version guard is needed.
PjPtr demote_to_2d(PJ_CONTEXT* ctx, PJ* crs) {
    if (crs == nullptr || proj_get_type(crs) != PJ_TYPE_GEOGRAPHIC_3D_CRS) {
        return PjPtr{};
    }
    return PjPtr{proj_crs_demote_to_2D(ctx, nullptr, crs)};
}

/// True when `operation` runs *from* the CRS `base`, comparing the horizontal
/// (2D) identity so a 3D source matches a 2D operation. Used to orient a stored
/// operation (the caller may have listed its inverse) and to refuse one that
/// belongs to a different pair.
bool runs_from(PJ_CONTEXT* ctx, PJ* base, PJ* operation) {
    PjPtr source{proj_get_source_crs(ctx, operation)};
    if (source == nullptr || base == nullptr) {
        return false;
    }
    if (equivalent_crs(ctx, source.get(), base)) {
        return true;
    }
    PjPtr horizontal = demote_to_2d(ctx, base);
    return horizontal != nullptr && equivalent_crs(ctx, source.get(), horizontal.get());
}

/// A datum shift attaches to a geodetic CRS, so only a geographic source can be
/// bound; PROJ rejects a projected or geocentric base.
bool bindable_kind(CrsKind kind) {
    return kind == CrsKind::Geographic2D || kind == CrsKind::Geographic3D;
}

}  // namespace

std::string describe_transformation(const DatumTransformation& transformation) {
    std::string text =
        transformation.name.empty() ? std::string("unnamed operation") : transformation.name;

    std::vector<std::string> facts;
    if (!transformation.auth_name.empty() && !transformation.code.empty()) {
        facts.push_back(transformation.auth_name + ":" + transformation.code);
    }
    if (transformation.accuracy_m > 0.0) {
        // Accuracy is a metre figure; one decimal is what PROJ publishes and
        // what reads well in a log line.
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.1f m", transformation.accuracy_m);
        facts.emplace_back(buffer);
    }
    if (transformation.ballpark) {
        facts.emplace_back("ballpark");
    }
    if (transformation.needs_missing_grid()) {
        facts.emplace_back("missing grid");
    }
    if (facts.empty()) {
        return text;
    }
    text += " [";
    for (std::size_t i = 0; i < facts.size(); ++i) {
        text += i == 0 ? "" : ", ";
        text += facts[i];
    }
    text += ']';
    return text;
}

std::string missing_grids_summary(const DatumTransformation& transformation) {
    std::string text;
    for (const std::string& grid : transformation.missing_grids) {
        if (!text.empty()) {
            text += ", ";
        }
        text += grid;
    }
    return text;
}

std::vector<DatumTransformation> CoordinateSystem::available_transformations(
    const CoordinateSystem& target) const {
    require_transformable(*this, "source");
    require_transformable(target, "target");

    PJ_CONTEXT* ctx = thread_context();
    PjPtr src_pj = create_crs(ctx, *this);
    PjPtr dst_pj = create_crs(ctx, target);

    ObjListPtr operations = list_operations(ctx, src_pj.get(), dst_pj.get(), *this, target);
    const int count = proj_list_get_count(operations.get());
    if (count <= 0) {
        throw Error("no datum transformation from " + quoted(*this) + " to " + quoted(target));
    }

    std::vector<DatumTransformation> transformations;
    transformations.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        PjPtr operation{proj_list_get(ctx, operations.get(), i)};
        if (!operation) {
            throw Error("cannot read datum transformation " + std::to_string(i) + " from " +
                        quoted(*this) + " to " + quoted(target) + ": " + proj_context_error(ctx));
        }
        DatumTransformation entry;
        describe_operation(ctx, operation.get(), &entry);
        transformations.push_back(std::move(entry));
    }

    // Order the list by accuracy, unknown accuracy last; PROJ's own order is
    // kept among equals. No entry is flagged as "the one PROJ would use":
    // PROJ's automatic transform chooses per point (an area-limited national
    // operation applies only inside its area and a less accurate one takes over
    // outside it), so there is no single such operation to name. See the
    // DatumTransformation documentation.
    std::stable_sort(transformations.begin(), transformations.end(), ranks_before);
    return transformations;
}

CoordinateSystem CoordinateSystem::with_transformation(
    const DatumTransformation& transformation) const {
    require_transformable(*this, "source");
    if (!bindable_kind(kind())) {
        // A projected source still has a route: `preferred_operation` matches a
        // listed operation's AUTH:CODE and builds the pipeline from it without
        // binding the CRS, which is the one way a projected CRS can pin a
        // datum shift.
        throw ValueError("cannot bind a datum transformation to " + quoted(*this) + " (a " +
                         crs_kind_name(kind()) +
                         " CRS); bind its geographic CRS instead, or pin the operation with "
                         "TransformOptions::preferred_operation");
    }

    // A missing grid is a hard error here rather than at the first coordinate:
    // pinning is where the caller chose this operation, and PROJ would
    // otherwise fall back to something less accurate without saying so.
    if (transformation.needs_missing_grid()) {
        throw Error("transformation needs grid " + transformation.missing_grids.front() +
                    "; install proj-data");
    }

    if (transformation.auth_name.empty() || transformation.code.empty()) {
        // An anonymous entry has no proj.db object, so its operation cannot be
        // recovered from the entry: `TransformOptions::preferred_operation`
        // matches the same AUTH:CODE and an anonymous entry has none. A
        // ballpark is still reachable through allow_ballpark; a custom CRS's
        // nadgrids shift applies by defining the CRS that way.
        throw ValueError("transformation \"" + transformation.name +
                         "\" has no authority code and cannot be pinned; a ballpark shift is "
                         "available through TransformOptions::allow_ballpark");
    }

    PJ_CONTEXT* ctx = thread_context();
    PjPtr base = create_crs(ctx, *this);

    // Recover the operation object from proj.db. The identifier alone is not
    // enough: PROJ decorates it for operations it derives or inverts
    // ("DERIVED_FROM(EPSG)", "INVERSE(DERIVED_FROM(EPSG))"), and those
    // decorated names are not rows proj.db answers for, so the wrapper is
    // peeled off to reach the stored code. The stored operation is then
    // *oriented*: the caller may have picked the inverse of a stored shift
    // (WGS 84 -> NAD27 is EPSG:1693 NAD27 -> WGS 84 reversed).
    const std::string authority = lookup_authority(transformation.auth_name);
    PjPtr stored{proj_create_from_database(ctx, authority.c_str(), transformation.code.c_str(),
                                           PJ_CATEGORY_COORDINATE_OPERATION,
                                           /*usePROJAlternativeGridNames=*/1, nullptr)};
    if (!stored) {
        throw ValueError("unknown datum transformation " + transformation.auth_name + ":" +
                         transformation.code + ": " + proj_context_error(ctx));
    }
    if (proj_get_type(stored.get()) == PJ_TYPE_CONVERSION) {
        // A projection conversion between two CRSs on the same datum: there is
        // no datum shift to pin, and binding would be a no-op wrapper. The
        // caller wants CrsTransform for this, not a pinned datum.
        throw ValueError("\"" + transformation.name + "\" is a projection conversion, not a "
                         "datum shift from " + quoted(*this) + "; use CrsTransform directly");
    }

    // One of the operation's two ends must be this CRS, or the entry belongs to
    // another pair and pinning it would silently apply the wrong shift.
    PjPtr chosen;
    if (runs_from(ctx, base.get(), stored.get())) {
        chosen = std::move(stored);
    } else {
        PjPtr reversed{proj_coordoperation_create_inverse(ctx, stored.get())};
        if (!reversed || !runs_from(ctx, base.get(), reversed.get())) {
            throw ValueError("datum transformation \"" + transformation.name + "\" (" +
                             transformation.auth_name + ":" + transformation.code +
                             ") does not apply to " + quoted(*this) +
                             "; it is a shift between other CRSs");
        }
        chosen = std::move(reversed);
    }
    PjPtr hub{proj_get_target_crs(ctx, chosen.get())};
    if (!hub) {
        // A datum shift always states the CRS it ends at; without one there is
        // nothing to bind to.
        throw ValueError("datum transformation " + transformation.auth_name + ":" +
                         transformation.code + " (\"" + transformation.name +
                         "\") does not state a target CRS and cannot be bound");
    }

    PjPtr bound{proj_crs_create_bound_crs(ctx, base.get(), hub.get(), chosen.get())};
    if (!bound) {
        throw Error("PROJ could not bind " + quoted(*this) + " by " + transformation.auth_name +
                    ":" + transformation.code + ": " + proj_context_error(ctx));
    }

    // BOUNDCRS WKT is what the value stores; parsing it back turns the result
    // into a normal CoordinateSystem (identifying strings only, no PROJ object).
    // Single line, matching `CoordinateSystem`'s canonical form, so a project
    // file can carry it as an XML attribute.
    const char* single_line[] = {"MULTILINE=NO", nullptr};
    const char* wkt = proj_as_wkt(ctx, bound.get(), PJ_WKT2_2019, single_line);
    if (wkt == nullptr) {
        throw Error("cannot serialise the bound CRS for " + quoted(*this) + ": " +
                    proj_context_error(ctx));
    }
    return CoordinateSystem::from_wkt(wkt);
}

}  // namespace alphashape::crs
