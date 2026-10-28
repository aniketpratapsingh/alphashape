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
/// ("DERIVED_FROM(EPSG)", "INVERSE(EPSG)") that proj.db cannot answer for.
/// The inner authority is the one that resolves, and the code is the same.
std::string lookup_authority(const std::string& auth_name) {
    const std::size_t open = auth_name.find('(');
    const std::size_t close = auth_name.rfind(')');
    if (open != std::string::npos && close == auth_name.size() - 1 && close > open + 1) {
        return auth_name.substr(open + 1, close - open - 1);
    }
    return auth_name;
}

/// The grid ids an operation requires, split into all and missing. PROJ reports
/// availability per grid; the short name is what `+grids=` names and what a
/// `proj-data` package installs, so it is what an error message must show.
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
        if (short_name == nullptr || *short_name == '\0') {
            continue;
        }
        out->grids.emplace_back(short_name);
        if (available == 0) {
            out->missing_grids.emplace_back(short_name);
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

    // PROJ's first candidate is the one it would use, so it is the preferred
    // entry; the rest follow in increasing accuracy behind it.
    DatumTransformation preferred = std::move(transformations.front());
    preferred.preferred = true;
    std::vector<DatumTransformation> rest(transformations.begin() + 1, transformations.end());
    std::stable_sort(rest.begin(), rest.end(), ranks_before);
    transformations.assign(1, preferred);
    transformations.insert(transformations.end(), rest.begin(), rest.end());
    return transformations;
}

CoordinateSystem CoordinateSystem::with_transformation(
    const DatumTransformation& transformation) const {
    require_transformable(*this, "source");
    if (!bindable_kind(kind())) {
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
        // An anonymous entry (the ballpark offset, or a custom CRS's nadgrids
        // shift) has no proj.db object, so its hub cannot be recovered from the
        // entry alone; a ballpark is still reachable through
        // TransformOptions::allow_ballpark.
        throw ValueError("transformation \"" + transformation.name +
                         "\" has no authority code and cannot be pinned; use "
                         "TransformOptions::allow_ballpark for a ballpark shift");
    }

    PJ_CONTEXT* ctx = thread_context();
    PjPtr base = create_crs(ctx, *this);

    // Look the operation up again rather than trusting the entry: this both
    // validates a persisted entry and gives the operation's hub CRS, which the
    // bound CRS needs. Derived names ("DERIVED_FROM(EPSG)") map back to the
    // authority proj.db answers for.
    const std::string authority = lookup_authority(transformation.auth_name);
    PjPtr chosen{proj_create_from_database(ctx, authority.c_str(), transformation.code.c_str(),
                                           PJ_CATEGORY_COORDINATE_OPERATION,
                                           /*usePROJAlternativeGridNames=*/1, nullptr)};
    if (!chosen) {
        throw ValueError("unknown datum transformation " + transformation.auth_name + ":" +
                         transformation.code + ": " + proj_context_error(ctx));
    }
    if (proj_get_type(chosen.get()) == PJ_TYPE_CONVERSION) {
        // A projection conversion between two CRSs on the same datum: there is
        // no datum shift to pin, and binding would be a no-op wrapper. The
        // caller wants CrsTransform for this, not a pinned datum.
        throw ValueError("\"" + transformation.name + "\" is a projection conversion, not a "
                         "datum shift from " + quoted(*this) + "; use CrsTransform directly");
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
