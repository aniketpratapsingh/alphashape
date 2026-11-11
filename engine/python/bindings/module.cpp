// Python front end onto the shared task schema.
//
// The single idea this file expresses: the SAME string-keyed
// parameter schema that the CLI binds against is what a Python keyword argument
// binds against. `doc.run_task("DetectFeatures", keypoint_limit=30)` and
// `--task DetectFeatures --set keypoint_limit=30` reach the engine through the
// identical path (make_params + Params::set + run_task), so the two front ends
// cannot drift from each other or from the project file.
//
// Design notes that are load-bearing:
//
//   * run_task() is entered with the GIL RELEASED and an EMPTY Progress. The
//     detector parallelises across frames with OpenMP; a progress callback that
//     re-entered Python from a worker thread would be a data race on the
//     interpreter, so v0 passes no Python callback at all. The GIL is released
//     so the C++ work actually runs in parallel instead of serialising behind
//     the interpreter lock.
//
//   * Chunk values handed to Python are returned as std::shared_ptr<Chunk>
//     (the same shared_ptr the Workspace holds). Python's shared_ptr holder
//     then co-owns the chunk, so a Chunk that outlives `del doc` stays valid.
//     That is the lifetime guarantee; no keep_alive tricks are needed.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "base/platform/env.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/calibration/calibration_io.hpp"
#include "common/algorithm/markers/marker_sheet.hpp"
#include "common/algorithm/reference/convert_reference.hpp"
#include "common/algorithm/reference/reference_table.hpp"
#include "common/algorithm/photos/camera_groups.hpp"
#include "common/commands/command.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/core/frame_stats.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/calibration.hpp"
#include "model/crs/coordinate_system.hpp"
#if defined(ALPHASHAPE_USE_PROJ)
#include "model/crs/geoid.hpp"
#endif

namespace py = pybind11;

namespace {

using alphashape::Document;
using alphashape::Chunk;

const char* type_name(alphashape::ParamType type) {
    switch (type) {
        case alphashape::ParamType::Bool: return "Bool";
        case alphashape::ParamType::Int: return "Int";
        case alphashape::ParamType::Float: return "Float";
        case alphashape::ParamType::String: return "String";
        case alphashape::ParamType::StringList: return "StringList";
    }
    return "unknown";
}

const char* camera_type_name(alphashape::CameraType type) {
    return type == alphashape::CameraType::Fisheye ? "fisheye" : "frame";
}

// Params::specs() returns a copy, so callers must keep that vector alive while
// holding the returned pointer (a pointer must never outlive the temporary from
// entry.schema.specs()).
const alphashape::ParamSpec* find_spec(const std::vector<alphashape::ParamSpec>& specs,
                                       const std::string& name) {
    for (const alphashape::ParamSpec& spec : specs) {
        if (spec.name == name) {
            return &spec;
        }
    }
    return nullptr;
}

/// Convert one Python keyword value into the variant the declared type wants.
/// The mapping is by declared type, not by Python type alone: an int is legal
/// for a Float parameter (Python has no separate float literal for 1.0 vs 1),
/// but a bool is never silently an Int — bool is a subclass of int in Python and
/// accepting it would turn `flag=True` into `keypoint_limit=1`.
alphashape::ParamValue to_param_value(const alphashape::ParamSpec& spec, py::handle value) {
    const bool is_bool = py::isinstance<py::bool_>(value);
    const bool is_int = py::isinstance<py::int_>(value);
    const bool is_float = py::isinstance<py::float_>(value);
    const bool is_str = py::isinstance<py::str>(value);

    switch (spec.type) {
        case alphashape::ParamType::Bool:
            if (is_bool) {
                return alphashape::ParamValue(py::cast<bool>(value));
            }
            break;
        case alphashape::ParamType::Int:
            if (is_int && !is_bool) {
                return alphashape::ParamValue(py::cast<std::int64_t>(value));
            }
            break;
        case alphashape::ParamType::Float:
            if (is_float || (is_int && !is_bool)) {
                return alphashape::ParamValue(py::cast<double>(value));
            }
            break;
        case alphashape::ParamType::String:
            if (is_str) {
                return alphashape::ParamValue(py::cast<std::string>(value));
            }
            // A Bool is accepted for `color_correction` ONLY, as its text form:
            // that parameter was a Bool in an older release, and the task's own
            // parser reads "true"/"false" as the modes it then selected. Applying
            // this to every String parameter would turn `label=True` into the
            // label "true" and, worse, let a Bool slip past an enum's choices;
            // every other String parameter therefore rejects a Bool below.
            if (is_bool && spec.name == "color_correction") {
                return alphashape::ParamValue(std::string(py::cast<bool>(value) ? "true"
                                                                               : "false"));
            }
            break;
        case alphashape::ParamType::StringList: {
            // A bare str is REJECTED, not iterated: a string IS iterable, so
            // `filenames="a.jpg"` would silently become ['a','.','j',...].
            // This is the single most important check in this function.
            if (is_str) {
                throw py::type_error("parameter '" + spec.name +
                                     "' expects a list of str, not a single str");
            }
            if (!py::isinstance<py::list>(value) && !py::isinstance<py::tuple>(value)) {
                break;
            }
            std::vector<std::string> out;
            for (py::handle item : value) {
                if (!py::isinstance<py::str>(item)) {
                    throw py::type_error("parameter '" + spec.name +
                                         "' expects a list of str, but an element is a " +
                                         std::string(py::cast<std::string>(
                                             py::str(py::type::of(item)))));
                }
                out.push_back(py::cast<std::string>(item));
            }
            return alphashape::ParamValue(std::move(out));
        }
    }
    // A PythonTypeError: the value's Python type is wrong for the declaration.
    // (An unknown *name* is a different failure, thrown as InvalidArgumentError.)
    throw py::type_error("parameter '" + spec.name + "' expects a " + type_name(spec.type));
}

/// A Python exception class for the module, with the given base. Built off
/// PyErr_NewException so it is a real type: `except alphashape.Error` and user
/// subclassing both work.
py::object make_exception(py::module_& m, const char* name, py::handle base) {
    const std::string qualified = std::string(py::cast<std::string>(m.attr("__name__"))) + "." + name;
    py::object obj = py::reinterpret_steal<py::object>(PyErr_NewException(qualified.c_str(), base.ptr(), nullptr));
    m.attr(name) = obj;
    return obj;
}

// pybind11 3.x types ExceptionTranslator as a plain function pointer, not a
// std::function, so the translator cannot capture the exception objects. We
// stash the (owned-by-the-module-dict) class objects in statics and read them
// from a free function. Raw PyObject* rather than py::object because a static
// py::object would be destructed at interpreter shutdown, which is unsafe for
// extension modules; the module dict keeps the real reference alive.
PyObject* g_unsupported = nullptr;
PyObject* g_invalid = nullptr;
PyObject* g_value = nullptr;
PyObject* g_project = nullptr;
PyObject* g_error = nullptr;

/// Raises `type` with `what` as its message. py::set_error() exists only from
/// pybind11 3, while the C API form works with the 2.11 header that Ubuntu
/// ships as well.
void raise_as(PyObject* type, const char* what) {
    PyErr_SetString(type, what);
}

void translate_alphashape(std::exception_ptr ptr) {
    if (!ptr) {
        return;
    }
    try {
        std::rethrow_exception(ptr);
    } catch (const alphashape::UnsupportedTaskError& e) {
        raise_as(g_unsupported, e.what());
    } catch (const alphashape::InvalidArgumentError& e) {
        raise_as(g_invalid, e.what());
    } catch (const alphashape::ValueError& e) {
        raise_as(g_value, e.what());
    } catch (const alphashape::ProjectIOError& e) {
        raise_as(g_project, e.what());
    } catch (const alphashape::Error& e) {
        raise_as(g_error, e.what());
    }
    // Anything else falls through unhandled: pybind11's default translator
    // (or the next one in the chain) still gets its chance.
}

// -- dense array reshaping ----------------------------------------------
// The dense containers (chunk.hpp) store their per-pixel and per-point data as
// flat, row-major std::vector<float>. These helpers reshape that flat storage
// into nested Python lists. The module deliberately does not depend on numpy,
// so the views match the shape convention the rest of the bindings already use
// (Chunk.tracks returns a list of lists of tuples; Camera.pose a list of lists
// of floats). A companion array that a container leaves EMPTY (no normal, no
// confidence) is reported by the caller as None — "absent" — so the shapes
// below only ever see a fully-populated grid.

/// `height` rows of `width` Python floats from a row-major vector of size
/// height*width. A vector whose size does not match yields an empty list, which
/// is how a caller distinguishes a malformed map from a valid one.
py::list reshape_scalar(const std::vector<float>& data, int height, int width) {
    const std::size_t hw = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    py::list rows;
    if (data.size() != hw) {
        return rows;
    }
    for (int r = 0; r < height; ++r) {
        const float* base =
            data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(width);
        py::list row;
        for (int c = 0; c < width; ++c) {
            row.append(py::float_(static_cast<double>(base[c])));
        }
        rows.append(row);
    }
    return rows;
}

/// `height` rows of `width` (x, y, z) tuples from a row-major vector of size
/// 3*height*width with the components interleaved — the DepthMap normal layout.
py::list reshape_vec3_grid(const std::vector<float>& data, int height, int width) {
    const std::size_t hw = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    py::list rows;
    if (data.size() != 3 * hw) {
        return rows;
    }
    for (int r = 0; r < height; ++r) {
        py::list row;
        for (int c = 0; c < width; ++c) {
            const std::size_t i =
                (static_cast<std::size_t>(r) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(c)) *
                3;
            row.append(py::make_tuple(py::float_(static_cast<double>(data[i])),
                                      py::float_(static_cast<double>(data[i + 1])),
                                      py::float_(static_cast<double>(data[i + 2]))));
        }
        rows.append(row);
    }
    return rows;
}

/// One depth map as the full dict both `depth_maps` and `active_depth_maps`
/// expose. A single builder keeps the two properties identical.
py::dict depth_map_entry(const alphashape::DepthMap& map) {
    py::dict entry;
    entry[py::str("camera_id")] = py::int_(map.camera_id);
    entry[py::str("width")] = py::int_(map.width);
    entry[py::str("height")] = py::int_(map.height);
    entry[py::str("downscale")] = py::int_(map.downscale);
    entry[py::str("depth")] = reshape_scalar(map.depths, map.height, map.width);
    entry[py::str("normals")] =
        map.normals.empty()
            ? py::object(py::none())
            : py::object(reshape_vec3_grid(map.normals, map.height, map.width));
    entry[py::str("confidence")] =
        map.confidence.empty()
            ? py::object(py::none())
            : py::object(reshape_scalar(map.confidence, map.height, map.width));
    return entry;
}

/// Bilinear, nodata-aware height sample on a DEM (the orthorectifier's rule).
/// Returns false when the point is off the grid or no valid cell contributes.
bool sample_height(const alphashape::GeoRaster<float>& raster, float nodata, double x, double y,
                   double& out) {
    const alphashape::RasterGrid& grid = raster.grid;
    const Eigen::Vector2d c = grid.to_pixel(x, y);
    const double cf = c.x() - 0.5;
    const double rf = c.y() - 0.5;
    const int c0 = static_cast<int>(std::floor(cf));
    const int r0 = static_cast<int>(std::floor(rf));
    const double fx = cf - static_cast<double>(c0);
    const double fy = rf - static_cast<double>(r0);
    double sum = 0.0;
    double weight = 0.0;
    for (int dr = 0; dr <= 1; ++dr) {
        for (int dc = 0; dc <= 1; ++dc) {
            const int col = c0 + dc;
            const int row = r0 + dr;
            if (!grid.contains_pixel(col, row)) {
                continue;
            }
            const float h = raster.at(col, row);
            if (h == nodata || std::isnan(h)) {
                continue;
            }
            const double w = (dc == 0 ? 1.0 - fx : fx) * (dr == 0 ? 1.0 - fy : fy);
            sum += w * static_cast<double>(h);
            weight += w;
        }
    }
    if (!(weight > 0.0)) {
        return false;
    }
    out = sum / weight;
    return true;
}

/// C++-side state attached to a Python Document under `_undo_stack`.
///
/// The token is deliberately opaque: it exposes no methods a script could
/// call. Holding the Document through a shared_ptr keeps the raw pointer the
/// stack was bound to valid even when a script retains the token after
/// dropping the Document. `cache_pins` keeps every Save As source cache
/// directory alive while retained commands may still spill into it; it is
/// cleared together with the history, exactly like the GUI's pinned_caches_.
struct DocumentHistory {
    std::shared_ptr<alphashape::Document> document;
    std::shared_ptr<alphashape::UndoStack> stack;
    std::vector<std::shared_ptr<void>> cache_pins;
};

namespace {
constexpr const char* kHistoryCapsuleName = "alphashape.document_history";

void delete_history(void* pointer) {
    delete static_cast<DocumentHistory*>(pointer);
}

DocumentHistory* document_history(py::object document) {
    if (!py::hasattr(document, "_undo_stack")) {
        return nullptr;
    }
    return static_cast<DocumentHistory*>(
        document.attr("_undo_stack").cast<py::capsule>().get_pointer());
}

/// Drop the recorded history after a whole-document mutation (a task, or an
/// in-place edit that bypasses the command layer). Tasks mutate in place and
/// cannot be undone, so keeping earlier entries would let an undo restore
/// stale snapshots and delete the task's result.
void clear_document_history(py::object document) {
    DocumentHistory* history = document_history(document);
    if (history != nullptr) {
        history->stack->clear();
        // No history survives, so no spill file in a pinned cache is needed.
        history->cache_pins.clear();
    }
}
}  // namespace
std::shared_ptr<alphashape::UndoStack> document_undo_stack(py::object document);

/// Run a task by name with keyword arguments bound against its declared schema.
/// Shared by Document.run_task and the Chunk convenience methods (buildDem,
/// buildOrthomosaic, ...) so the two cannot drift.
void run_task_kwargs(py::object document, const std::string& name, py::kwargs kwargs) {
    alphashape::Document& d = document.cast<alphashape::Document&>();
    alphashape::TaskRegistry& registry = alphashape::builtin_registry();
    // Throws UnsupportedTaskError (mapped) for an unknown task.
    const alphashape::TaskEntry& entry = registry.require(name);
    alphashape::Params params = alphashape::make_params(entry);
    // Hold the schema copy for the whole binding loop: the ParamSpec pointers
    // below point into it.
    const std::vector<alphashape::ParamSpec> specs = entry.schema.specs();

    // Bind every keyword by NAME against the declared schema. All Python
    // objects are touched here, with the GIL held.
    for (auto item : kwargs) {
        const std::string key = py::cast<std::string>(item.first);
        const alphashape::ParamSpec* spec = find_spec(specs, key);
        if (spec == nullptr) {
            throw alphashape::InvalidArgumentError(key);
        }
        params.set(key, to_param_value(*spec, item.second));
    }

    // Release the GIL: the task computes in C++ and may parallelise across
    // frames. The empty Progress guarantees no worker thread ever calls back
    // into Python.
    {
        py::gil_scoped_release release;
        alphashape::run_task(registry, d, name, params, alphashape::Progress{});
    }
    // A task cannot be undone; discard the history it just invalidated. Only
    // touch the token when one already exists: creating it here would make a
    // document with no edits look like it had a stack.
    clear_document_history(std::move(document));
}

/// Convert a C++ chunk to the Python object the Document bindings hand out,
/// attaching the owning Document as `_document`. Chunk has dynamic_attr(), so
/// the attribute lives in the object's __dict__; pybind11's GC traverses it, so
/// the Document <-> Chunk cycle stays collectable.
py::object attach_document(py::object document, const std::shared_ptr<alphashape::Chunk>& chunk);
py::object owning_document_object(py::object chunk_self);
void execute_command(py::object document,
                     std::unique_ptr<alphashape::AbstractCommand> command);

/// Run a chunk-scoped task on the document that owns `self`.
///
/// Every chunk handed out by the Document bindings carries `_document` (the
/// Python Document object it came from), so `chunk.buildDem(...)` runs through
/// exactly the same run_task path as `doc.run_task("BuildDem", ...)` and
/// records its parameters into the chunk's meta. The active chunk is pointed at
/// `self` first, because the task layer resolves its target that way.
void run_chunk_task(py::object self, const char* name, py::kwargs kwargs) {
    if (!py::hasattr(self, "_document")) {
        throw py::type_error("this Chunk is not attached to a Document");
    }
    py::object document = self.attr("_document");
    const alphashape::Chunk& chunk = self.cast<const alphashape::Chunk&>();
    document.cast<alphashape::Document&>().set_active_chunk_id(chunk.id());
    run_task_kwargs(std::move(document), name, std::move(kwargs));
}

/// The DEM view (milestone 9). Holds the chunk's shared_ptr so the raster it
/// points into cannot be freed while Python still reads it. A copy would double
/// memory on a real DEM; a raw pointer would dangle after `del chunk`.
struct ElevationView {
    std::shared_ptr<alphashape::Chunk> chunk;
    const alphashape::Elevation* elevation = nullptr;
};

/// The orthomosaic view. Same ownership scheme as ElevationView.
struct OrthomosaicView {
    std::shared_ptr<alphashape::Chunk> chunk;
    const alphashape::Orthomosaic* orthomosaic = nullptr;
};

/// Proxy types below refer to an element INSIDE a chunk vector by (chunk, id)
/// and re-resolve on every access. Holding the element's address would dangle
/// as soon as any mutation reallocated or reordered the vector.
struct MarkerRef {
    std::shared_ptr<alphashape::Chunk> chunk;
    py::object document;
    std::int32_t id = -1;

    const alphashape::Marker& get() const {
        for (const alphashape::Marker& marker : chunk->markers()) {
            if (marker.id == id) {
                return marker;
            }
        }
        throw alphashape::ValueError("marker id " + std::to_string(id) +
                                     " no longer exists");
    }
};

struct CameraRef {
    std::shared_ptr<alphashape::Chunk> chunk;
    py::object document;
    std::int32_t id = -1;

    const alphashape::Camera& get() const {
        for (const alphashape::Camera& camera : chunk->cameras()) {
            if (camera.id == id) {
                return camera;
            }
        }
        throw alphashape::ValueError("camera id " + std::to_string(id) +
                                     " no longer exists");
    }
};

struct MarkerGroupRef {
    std::shared_ptr<alphashape::Chunk> chunk;
    py::object document;
    std::int32_t id = -1;

    const alphashape::MarkerGroup& get() const {
        for (const alphashape::MarkerGroup& group : chunk->marker_groups()) {
            if (group.id == id) {
                return group;
            }
        }
        throw alphashape::ValueError("marker group id " + std::to_string(id) +
                                     " no longer exists");
    }
};

struct ScalebarRef {
    std::shared_ptr<alphashape::Chunk> chunk;
    py::object document;
    std::int32_t id = -1;

    const alphashape::Scalebar& get() const {
        for (const alphashape::Scalebar& bar : chunk->scalebars()) {
            if (bar.id == id) {
                return bar;
            }
        }
        throw alphashape::ValueError("scalebar id " + std::to_string(id) +
                                     " no longer exists");
    }
};

/// One projection of one marker, addressed by (marker id, camera id) and
/// re-resolved on every access like the other proxies above.
struct MarkerProjectionRef {
    std::shared_ptr<alphashape::Chunk> chunk;
    py::object document;
    std::int32_t marker_id = -1;
    std::int32_t camera_id = -1;

    const alphashape::Marker& marker() const {
        for (const alphashape::Marker& marker : chunk->markers()) {
            if (marker.id == marker_id) {
                return marker;
            }
        }
        throw alphashape::ValueError("marker id " + std::to_string(marker_id) +
                                     " no longer exists");
    }

    const alphashape::MarkerProjection& get() const {
        for (const alphashape::MarkerProjection& projection :
             marker().projections) {
            if (projection.camera_id == camera_id) {
                return projection;
            }
        }
        throw alphashape::ValueError(
            "marker " + std::to_string(marker_id) + " has no projection in camera " +
            std::to_string(camera_id));
    }
};

/// Shared fields of every AssetSlot instantiation, so an AssetRef can read an
/// asset through one pointer without knowing its payload type.
struct AssetSlotBase {
    std::int32_t id = -1;
    std::string label;
    std::int32_t group_id = -1;
    bool enabled = true;
};

/// One asset of any kind, addressed by (chunk, kind, id) and resolved on
/// every access. `label` is editable through the rename command the GUI uses.
struct AssetRef {
    std::shared_ptr<alphashape::Chunk> chunk;
    py::object document;
    alphashape::AssetKind kind = alphashape::AssetKind::Elevation;
    std::int32_t id = -1;

    /// Copy the common fields out of the addressed slot. Defined after all
    /// the slot types are visible.
    AssetSlotBase fields() const;
    std::int32_t active_id() const;
};

template <class Slot>
AssetSlotBase slot_fields(const Slot& slot) {
    AssetSlotBase fields;
    fields.id = slot.id;
    fields.label = slot.label;
    fields.group_id = slot.group_id;
    fields.enabled = slot.enabled;
    return fields;
}

AssetSlotBase AssetRef::fields() const {
    switch (kind) {
        case alphashape::AssetKind::PointCloud:
            for (const auto& slot : chunk->point_clouds()) {
                if (slot.id == id) return slot_fields(slot);
            }
            break;
        case alphashape::AssetKind::Model:
            for (const auto& slot : chunk->models()) {
                if (slot.id == id) return slot_fields(slot);
            }
            break;
        case alphashape::AssetKind::DepthMaps:
            for (const auto& slot : chunk->depth_map_sets()) {
                if (slot.id == id) return slot_fields(slot);
            }
            break;
        case alphashape::AssetKind::Elevation:
            for (const auto& slot : chunk->elevations()) {
                if (slot.id == id) return slot_fields(slot);
            }
            break;
        case alphashape::AssetKind::Orthomosaic:
            for (const auto& slot : chunk->orthomosaics()) {
                if (slot.id == id) return slot_fields(slot);
            }
            break;
        case alphashape::AssetKind::TiledModel:
            for (const auto& slot : chunk->tiled_models()) {
                if (slot.id == id) return slot_fields(slot);
            }
            break;
        case alphashape::AssetKind::Panorama:
            if (chunk->panorama().has_value() && chunk->panorama()->id == id) {
                return slot_fields(*chunk->panorama());
            }
            break;
    }
    throw alphashape::ValueError("asset id " + std::to_string(id) +
                                 " no longer exists");
}

std::int32_t AssetRef::active_id() const {
    switch (kind) {
        case alphashape::AssetKind::PointCloud:
            return chunk->active_point_cloud_id();
        case alphashape::AssetKind::Model:
            return chunk->active_model_id();
        case alphashape::AssetKind::DepthMaps:
            return chunk->active_depth_maps_id();
        case alphashape::AssetKind::Elevation:
            return chunk->active_elevation_id();
        case alphashape::AssetKind::Orthomosaic:
            return chunk->active_orthomosaic_id();
        case alphashape::AssetKind::TiledModel:
            return chunk->active_tiled_model_id();
        case alphashape::AssetKind::Panorama:
            break;
    }
    return -1;
}

/// One AssetRef per slot, carrying the owning Python Document so a label edit
/// goes through the document's undo stack.
template <class Slot>
py::list asset_refs(py::object self, alphashape::AssetKind kind,
                    const std::vector<Slot>& slots) {
    auto chunk = self.cast<std::shared_ptr<Chunk>>();
    py::object document = owning_document_object(self);
    py::list out;
    for (const Slot& slot : slots) {
        out.append(py::cast(AssetRef{chunk, document, kind, slot.id}));
    }
    return out;
}

/// Defined after the Chunk binding exists: py::cast on a Chunk needs the
/// registered type, and the definition must not run before PYBIND11_MODULE has
/// created it.
py::object attach_document(py::object document, const std::shared_ptr<alphashape::Chunk>& chunk) {
    py::object object = py::cast(chunk);
    object.attr("_document") = std::move(document);
    return object;
}

/// Surface non-fatal calibration-file warnings through the standard Python
/// warnings mechanism, the way a library is expected to report them.
void emit_warnings(const std::vector<std::string>& warnings) {
    if (warnings.empty()) {
        return;
    }
    py::module_ warnings_module = py::module_::import("warnings");
    for (const std::string& message : warnings) {
        // stacklevel=2 reports the caller's line, not this binding's.
        warnings_module.attr("warn")(py::str(message), py::arg("stacklevel") = 2);
    }
}

/// Apply the Set Chunk CRS command on the Document that owns `chunk_self`
/// (every chunk the Document bindings hand out carries `_document`), as one
/// undo entry. Setting the CRS only relabels the chunk; no data is
/// transformed.
void apply_crs_command(py::object chunk_self, alphashape::CRS crs) {
    if (!py::hasattr(chunk_self, "_document")) {
        throw py::type_error("this Chunk is not attached to a Document");
    }
    const std::int32_t id = chunk_self.cast<const alphashape::Chunk&>().id();
    execute_command(
        owning_document_object(std::move(chunk_self)),
        alphashape::make_set_chunk_crs_command(id, std::move(crs)));
}

/// Replace one accuracy in the owning Document's chunk through the undoable
/// "Set Reference Settings" command. `field` names the member so the six
/// public properties share one implementation.
void set_reference_accuracy(py::object chunk_self,
                            double alphashape::ReferenceSettings::*field,
                            double value) {
    if (!(value > 0.0)) {
        throw alphashape::ValueError("reference accuracy must be positive");
    }
    if (!py::hasattr(chunk_self, "_document")) {
        throw py::type_error("this Chunk is not attached to a Document");
    }
    alphashape::ReferenceSettings settings =
        chunk_self.cast<const alphashape::Chunk&>().reference_settings();
    settings.*field = value;
    const std::int32_t id = chunk_self.cast<const alphashape::Chunk&>().id();
    execute_command(
        owning_document_object(std::move(chunk_self)),
        alphashape::make_set_reference_settings_command(id, settings));
}

/// Temporarily set `ReferenceSettings::use_gcp` on a chunk and restore the
/// previous settings when this goes out of scope, including when the task
/// throws. This is a per-call override, not an undoable edit: it issues no
/// command and leaves no history entry. `Chunk.optimizeCameras(use_gcps=...)`
/// is the public caller.
class ReferenceUseGcpOverride {
public:
    ReferenceUseGcpOverride(alphashape::Chunk& chunk, bool value)
        : chunk_(chunk), previous_(chunk.reference_settings()) {
        alphashape::ReferenceSettings settings = previous_;
        settings.use_gcp = value;
        chunk_.set_reference_settings(settings);
    }
    ~ReferenceUseGcpOverride() { chunk_.set_reference_settings(previous_); }
    ReferenceUseGcpOverride(const ReferenceUseGcpOverride&) = delete;
    ReferenceUseGcpOverride& operator=(const ReferenceUseGcpOverride&) = delete;

private:
    alphashape::Chunk& chunk_;
    alphashape::ReferenceSettings previous_;
};

/// The Document that owns `chunk_self`, for the command-backed marker edits.
alphashape::Document& owning_document(py::object chunk_self) {
    if (!py::hasattr(chunk_self, "_document")) {
        throw py::type_error("this Chunk is not attached to a Document");
    }
    return chunk_self.attr("_document").cast<alphashape::Document&>();
}

/// The Python Document object that owns `chunk_self`.
py::object owning_document_object(py::object chunk_self) {
    if (!py::hasattr(chunk_self, "_document")) {
        throw py::type_error("this Chunk is not attached to a Document");
    }
    return chunk_self.attr("_document");
}

/// The undo stack associated with a Python Document, created on first use.
/// The stack lives on the Document object (dynamic_attr), so it is destroyed
/// with it; `Document.undo()` / `Document.redo()` drive this same stack and a
/// scripted edit is therefore an ordinary undo entry.
std::shared_ptr<alphashape::UndoStack> document_undo_stack(py::object document) {
    if (DocumentHistory* history = document_history(document); history != nullptr) {
        return history->stack;
    }
    auto owner = document.cast<std::shared_ptr<alphashape::Document>>();
    auto history = std::make_unique<DocumentHistory>();
    history->document = owner;
    history->stack = std::make_shared<alphashape::UndoStack>(owner.get());
    history->stack->set_cache_dir(owner->cache_dir());
    DocumentHistory* raw = history.release();
    // The token is opaque: no bound class exposes undo/redo on it, so a script
    // cannot keep calling into a stack whose owner it dropped.
    document.attr("_undo_stack") =
        py::capsule(raw, kHistoryCapsuleName, &delete_history);
    return raw->stack;
}

/// Keep a pre-Save-As cache directory alive for as long as the history that
/// may spill into it. Does nothing when no history exists yet: there are then
/// no commands holding a spill file. Pins accumulate so successive Save As
/// calls keep each old cache until the history is cleared.
void pin_document_history_cache(py::object document, std::shared_ptr<void> pin) {
    if (DocumentHistory* history = document_history(document); history != nullptr) {
        history->cache_pins.push_back(std::move(pin));
    }
}

/// Execute one undoable command against the Document that owns `document`.
void execute_command(py::object document,
                     std::unique_ptr<alphashape::AbstractCommand> command) {
    document_undo_stack(std::move(document))->execute(std::move(command));
}

/// Execute one undoable command against the Document that owns `chunk_self`.
void execute_on_chunk(py::object chunk_self,
                      std::unique_ptr<alphashape::AbstractCommand> command) {
    execute_command(owning_document_object(std::move(chunk_self)), std::move(command));
}

py::object attach_document(py::object document,
                           const std::shared_ptr<alphashape::Chunk>& chunk);

namespace {

/// Restore the active chunk as part of the same undo entry as the add, so a
/// redo leaves the active chunk exactly where the original call did.
class RestoreActiveChunkCommand : public alphashape::AbstractCommand {
public:
    RestoreActiveChunkCommand(std::int32_t before, std::int32_t after)
        : before_(before), after_(after) {}

    std::string name() const override { return "Add Chunk"; }

    void apply(alphashape::Document& document) override {
        document.set_active_chunk_id(after_);
    }

    void revert(alphashape::Document& document) override {
        document.set_active_chunk_id(before_);
    }

private:
    std::int32_t before_;
    std::int32_t after_;
};

}  // namespace

/// Add a chunk through the undoable command. The command makes the new chunk
/// active (the GUI wants that); the scripting contract of add_chunk() predates
/// it and leaves the active chunk alone, so the previous id is restored as a
/// second command in the same transaction (redo therefore behaves the same).
py::object document_add_chunk(py::object self, py::object label) {
    Document& document = self.cast<Document&>();
    const std::int32_t previous_active = document.active_chunk_id();
    const std::int32_t created = document.workspace().next_chunk_id();
    std::string resolved;
    if (!label.is_none()) {
        resolved = py::cast<std::string>(label);
    }
    if (resolved.empty()) {
        // "Chunk N" for the smallest unused N, so a second add_chunk() is
        // "Chunk 2" rather than a duplicate "Chunk 1".
        for (int n = 1;; ++n) {
            const std::string candidate = "Chunk " + std::to_string(n);
            bool used = false;
            for (const auto& chunk : document.workspace().chunks()) {
                if (chunk->label() == candidate) {
                    used = true;
                    break;
                }
            }
            if (!used) {
                resolved = candidate;
                break;
            }
        }
    }
    auto transaction =
        std::make_unique<alphashape::CommandTransaction>("Add Chunk");
    transaction->add(alphashape::make_add_chunk_command(
        document.workspace(), resolved, -1));
    transaction->add(
        std::make_unique<RestoreActiveChunkCommand>(created, previous_active));
    execute_command(self, std::move(transaction));
    for (const auto& candidate : document.workspace().chunks()) {
        if (candidate->id() == created) {
            return attach_document(self, candidate);
        }
    }
    throw alphashape::Error("addChunk: the chunk was not created");
}

/// Normalise a chunk-container argument to its elements. A dict — the shape
/// of Document.chunks — contributes its VALUES; a single Chunk or int is one
/// element; anything else iterable contributes its items.
std::vector<py::object> chunk_argument_items(py::object chunks) {
    std::vector<py::object> items;
    if (py::isinstance<Chunk>(chunks) || py::isinstance<py::int_>(chunks)) {
        items.push_back(std::move(chunks));
        return items;
    }
    if (py::isinstance<py::dict>(chunks)) {
        for (auto item : chunks.cast<py::dict>()) {
            items.emplace_back(py::reinterpret_borrow<py::object>(item.second));
        }
        return items;
    }
    for (py::handle item : chunks) {
        items.emplace_back(py::reinterpret_borrow<py::object>(item));
    }
    return items;
}

/// One mask as a dict. `data` is the raw u8 plane, one byte per pixel.
py::dict mask_dict(const alphashape::Mask& mask) {
    py::dict entry;
    entry[py::str("camera_id")] = py::int_(mask.camera_id);
    entry[py::str("width")] = py::int_(mask.width);
    entry[py::str("height")] = py::int_(mask.height);
    entry[py::str("invert")] = py::bool_(mask.invert);
    entry[py::str("data")] = py::bytes(
        reinterpret_cast<const char*>(mask.data.data()), mask.data.size());
    return entry;
}

/// The mask of `camera_id`, or None. Linear scan: a chunk has one mask per
/// camera at most, and the container is sorted by camera id.
const alphashape::Mask* find_mask(const alphashape::Chunk& chunk,
                                  std::int32_t camera_id) {
    for (const alphashape::Mask& mask : chunk.masks()) {
        if (mask.camera_id == camera_id) {
            return &mask;
        }
    }
    return nullptr;
}

/// Resolve a marker argument: an int id or an exact label. Throws ValueError
/// when nothing matches.
std::int32_t marker_id_from_python(const alphashape::Chunk& chunk, py::handle value) {
    if (py::isinstance<py::str>(value)) {
        const std::string label = py::cast<std::string>(value);
        for (const alphashape::Marker& marker : chunk.markers()) {
            if (marker.label == label) {
                return marker.id;
            }
        }
        throw alphashape::ValueError("no marker with label '" + label + "'");
    }
    const std::int32_t id = py::cast<std::int32_t>(value);
    for (const alphashape::Marker& marker : chunk.markers()) {
        if (marker.id == id) {
            return id;
        }
    }
    throw alphashape::ValueError("no marker with id " + std::to_string(id));
}

std::int32_t camera_id_from_python(const alphashape::Chunk& chunk,
                                   py::handle value);

/// Resolve a scalebar endpoint from Python: an int marker id, an exact marker
/// label, or ("marker"|"camera", id-or-marker-label). The command validates
/// that the endpoint exists.
alphashape::ScalebarEndpoint scalebar_endpoint(const alphashape::Chunk& chunk,
                                               py::handle value,
                                               const char* argument) {
    alphashape::ScalebarEndpoint endpoint;
    py::object kind = py::none();
    py::object target = py::reinterpret_borrow<py::object>(value);
    if (py::isinstance<py::tuple>(value) || py::isinstance<py::list>(value)) {
        const auto items = py::cast<std::vector<py::object>>(value);
        if (items.size() != 2) {
            throw alphashape::ValueError(std::string("addScalebar: ") + argument +
                                         " must be (kind, id)");
        }
        kind = items[0];
        target = items[1];
    }
    const bool camera = !kind.is_none() && py::cast<std::string>(kind) == "camera";
    if (!kind.is_none() && !camera &&
        py::cast<std::string>(kind) != "marker") {
        throw alphashape::ValueError(std::string("addScalebar: ") + argument +
                                     " kind must be 'marker' or 'camera'");
    }
    endpoint.kind = camera ? alphashape::ScalebarEndpoint::Kind::Camera
                           : alphashape::ScalebarEndpoint::Kind::Marker;
    if (py::isinstance<py::str>(target)) {
        endpoint.id = camera ? camera_id_from_python(chunk, target)
                             : marker_id_from_python(chunk, target);
        return endpoint;
    }
    endpoint.id = py::cast<std::int32_t>(target);
    return endpoint;
}

/// Resolve a camera argument: an int id or an exact label.
std::int32_t camera_id_from_python(const alphashape::Chunk& chunk, py::handle value) {
    if (py::isinstance<py::str>(value)) {
        const std::string label = py::cast<std::string>(value);
        for (const alphashape::Camera& camera : chunk.cameras()) {
            if (camera.label == label) {
                return camera.id;
            }
        }
        throw alphashape::ValueError("no camera with label '" + label + "'");
    }
    const std::int32_t id = py::cast<std::int32_t>(value);
    for (const alphashape::Camera& camera : chunk.cameras()) {
        if (camera.id == id) {
            return id;
        }
    }
    throw alphashape::ValueError("no camera with id " + std::to_string(id));
}

/// Dense Eigen matrix as nested Python lists, the convention the other dense
/// views use (no numpy dependency). An empty matrix yields an empty list.
template <typename Derived>
py::list nested_matrix(const Eigen::MatrixBase<Derived>& matrix) {
    py::list rows;
    for (Eigen::Index r = 0; r < matrix.rows(); ++r) {
        py::list row;
        for (Eigen::Index c = 0; c < matrix.cols(); ++c) {
            row.append(py::float_(matrix(r, c)));
        }
        rows.append(row);
    }
    return rows;
}

#if defined(ALPHASHAPE_USE_PROJ)
/// (x, y, z) tuple or None, the shape every optional vector in the reference
/// table uses. The table itself needs the CRS engine, so this lives with it.
py::object optional_vec3(const std::optional<Eigen::Vector3d>& value) {
    if (!value) {
        return py::none();
    }
    return py::make_tuple(value->x(), value->y(), value->z());
}
#endif  // ALPHASHAPE_USE_PROJ

#if defined(ALPHASHAPE_USE_PROJ)
/// The process-wide geoid registry behind CoordinateSystem.addGeoid/listGeoids.
/// Seeded with the grids PROJ's data directory has installed; custom grids are
/// referenced where they are (the store has no directory, so nothing is
/// copied).
alphashape::crs::GeoidStore& geoid_store() {
    static alphashape::crs::GeoidStore store =
        alphashape::crs::GeoidStore::from_proj_data();
    return store;
}

/// CoordinateSystem.init(name): an "EPSG:<code>" / "EPSG::<code>" identifier,
/// a WKT definition (anything containing '['), or a built-in name. The
/// engine's from_name() only knows the curated list; this wrapper adds the
/// two script-facing spellings Metashape accepts.
alphashape::crs::CoordinateSystem coordinate_system_from_name(const std::string& name) {
    std::string code;
    if (name.rfind("EPSG::", 0) == 0) {
        code = name.substr(6);
    } else if (name.rfind("EPSG:", 0) == 0) {
        code = name.substr(5);
    }
    if (!code.empty()) {
        const bool digits = code.size() <= 9 &&
                            std::all_of(code.begin(), code.end(), [](unsigned char c) {
                                return std::isdigit(c) != 0;
                            });
        if (!digits) {
            throw alphashape::ValueError("CoordinateSystem.init: '" + name +
                                         "' is not an EPSG code");
        }
        return alphashape::crs::CoordinateSystem::from_epsg(std::stoi(code));
    }
    if (name.find('[') != std::string::npos) {
        return alphashape::crs::CoordinateSystem::from_wkt(name);
    }
    return alphashape::crs::CoordinateSystem::from_name(name);
}
#endif

}  // namespace

PYBIND11_MODULE(alphashape, m) {
    m.doc() = "Python bindings for the alphashape photogrammetry engine.";

    // -- exception types -----------------------------------------------------
    // Order matters for the translator below: most-derived first, and the Python
    // classes mirror the C++ hierarchy so `except alphashape.Error` catches all.
    py::object py_error = make_exception(m, "Error", PyExc_RuntimeError);
    py::object py_unsupported = make_exception(m, "UnsupportedTaskError", py_error);
    py::object py_invalid = make_exception(m, "InvalidArgumentError", py_error);
    py::object py_value = make_exception(m, "ValueError", py_error);
    py::object py_project = make_exception(m, "ProjectIOError", py_error);

    g_error = py_error.ptr();
    g_unsupported = py_unsupported.ptr();
    g_invalid = py_invalid.ptr();
    g_value = py_value.ptr();
    g_project = py_project.ptr();

    py::register_exception_translator(&translate_alphashape);

#if defined(ALPHASHAPE_USE_PROJ)
    // -- CoordinateSystem (WP-12.11) -----------------------------------------
    // A thin value wrapper over crs::CoordinateSystem. Copies are cheap (the
    // pimpl strings are shared), so pybind returns it BY VALUE; `__eq__` is
    // PROJ equivalence, not string identity, which is what a script comparing
    // two definitions of "the same CRS" means.
    using CoordinateSystem = alphashape::crs::CoordinateSystem;

    py::class_<CoordinateSystem> coordinate_system_class(m, "CoordinateSystem");
    coordinate_system_class
        .def(py::init<>(), "The local (unset) engineering CRS.")
        .def(py::init<const CoordinateSystem&>(), py::arg("other"),
             "Copy another coordinate system.")
        .def(py::init([](int epsg) { return CoordinateSystem::from_epsg(epsg); }),
             py::arg("epsg"), "Build from an EPSG code; ValueError when unknown.")
        .def(py::init([](py::kwargs kwargs) {
                 const bool has_wkt = kwargs.contains("wkt");
                 const bool has_proj = kwargs.contains("proj");
                 const bool has_name = kwargs.contains("name");
                 for (auto item : kwargs) {
                     const std::string key = py::cast<std::string>(item.first);
                     if (key != "wkt" && key != "proj" && key != "name") {
                         throw py::type_error("CoordinateSystem: unknown keyword '" + key + "'");
                     }
                     if (!py::isinstance<py::str>(item.second)) {
                         throw py::type_error("CoordinateSystem: '" + key + "' expects a str");
                     }
                 }
                 // Exactly one spelling: the constructor-by-name is name=, so
                 // any two of wkt= / proj= / name= is ambiguous.
                 const int provided = static_cast<int>(has_wkt) +
                                      static_cast<int>(has_proj) +
                                      static_cast<int>(has_name);
                 if (provided != 1) {
                     throw alphashape::ValueError(
                         "CoordinateSystem: pass exactly one of wkt, proj or name");
                 }
                 if (has_wkt) {
                     return CoordinateSystem::from_wkt(py::cast<std::string>(kwargs["wkt"]));
                 }
                 if (has_proj) {
                     return CoordinateSystem::from_proj(py::cast<std::string>(kwargs["proj"]));
                 }
                 if (has_name) {
                     return coordinate_system_from_name(
                         py::cast<std::string>(kwargs["name"]));
                 }
                 throw alphashape::ValueError(
                     "CoordinateSystem: pass exactly one of wkt, proj or name");
             }),
             "Build from WKT (wkt=), a PROJ string (proj=) or a built-in name "
             "(name=). Exactly one is required.")
        .def_property_readonly("name", [](const CoordinateSystem& cs) { return cs.name(); })
        .def_property_readonly("authority",
                               [](const CoordinateSystem& cs) { return cs.auth_name(); })
        .def_property_readonly("epsg", [](const CoordinateSystem& cs) -> py::object {
            const int code = cs.epsg();
            return code == 0 ? py::object(py::none()) : py::object(py::int_(code));
        })
        .def_property_readonly("wkt", [](const CoordinateSystem& cs) { return cs.to_wkt(); })
        .def("toWKT", [](const CoordinateSystem& cs) { return cs.to_wkt(); })
        .def("toProj4", [](const CoordinateSystem& cs) { return cs.to_proj4(); })
        .def_static("listBuiltinCRS",
                    []() { return alphashape::crs::builtin_crs_names(); },
                    "Names of the curated built-in CRS list, in list order.")
        .def_static(
            "init",
            [](const std::string& name) { return coordinate_system_from_name(name); },
            py::arg("name"),
            "Build from \"EPSG:<code>\" / \"EPSG::<code>\", a WKT definition "
            "or a built-in name (the listBuiltinCRS() names).")
        // The geoid registry is process-wide: seeded from PROJ's data
        // directory on first use, then custom grids are referenced where they
        // are. Nothing is copied and nothing is downloaded.
        .def_static(
            "addGeoid",
            [](const std::string& path) {
                // Register the grid at its canonical absolute path, so the store
                // and the project file never depend on the caller's cwd.
                std::error_code ec;
                std::filesystem::path file =
                    std::filesystem::absolute(std::filesystem::path(path), ec);
                if (!ec) {
                    file = std::filesystem::weakly_canonical(file, ec);
                }
                if (ec) {
                    throw alphashape::ValueError("addGeoid: cannot resolve '" + path +
                                                 "': " + ec.message());
                }
                const std::string id = file.stem().string();
                if (id.empty() || !file.has_filename()) {
                    throw alphashape::ValueError("addGeoid: '" + path +
                                                 "' has no file name");
                }
                // Register only a file that really opens as a geoid grid: this
                // turns a late, confusing failure at evaluation time into a
                // ValueError at registration.
                try {
                    (void)alphashape::crs::GeoidGrid::open(file.string());
                } catch (const std::exception&) {
                    throw alphashape::ValueError("addGeoid: '" + path +
                                                 "' is not a readable geoid grid");
                }
                geoid_store().add(id, file.filename().string(), file.string(),
                                  "custom", "");
            },
            py::arg("path"),
            "Register a geoid grid file. Raises when the file is missing or "
            "its id is already registered.")
        .def_static("listGeoids",
                    []() {
                        std::vector<std::string> ids;
                        for (const alphashape::crs::GeoidModel& model :
                             geoid_store().list()) {
                            ids.push_back(model.id);
                        }
                        return ids;
                    },
                    "Ids of the registered geoid grids.")
        // Equality is PROJ equivalence: the same datum, ellipsoid, axes and
        // units. Two different definitions of one CRS compare equal; axis
        // order differences (lat/lon vs lon/lat) do not make them unequal.
        .def("__repr__",
             [](const CoordinateSystem& cs) {
                 return "<CoordinateSystem '" + cs.name() + "'>";
             })
        .def(
            "__eq__",
            [](const CoordinateSystem& a, py::object other) -> py::object {
                if (!py::isinstance<CoordinateSystem>(other)) {
                    return py::reinterpret_borrow<py::object>(Py_NotImplemented);
                }
                const CoordinateSystem b = other.cast<CoordinateSystem>();
                return py::bool_(a.is_equivalent_to(b));
            },
            py::is_operator());
    // Defining __eq__ would otherwise leave pybind's identity __hash__ in
    // place, breaking the hash/eq contract. A value type that compares by PROJ
    // equivalence is unhashable, so hash(cs) raises TypeError like tuple does.
    coordinate_system_class.attr("__hash__") = py::none();
#endif  // ALPHASHAPE_USE_PROJ

    // -- DEM and orthomosaic views (milestone 9) -----------------------------
    // Both are views onto the chunk's own raster, not copies: the chunk is held
    // by shared_ptr so the product stays alive exactly as long as Python needs
    // it. `pixels()` / `heights()` return plain Python lists (the module does
    // not depend on numpy), shaped H x W (heights) or H x W x 4 (pixels). For a
    // 4k x 4k ortho that is ~67M Python ints — a real limitation, noted in
    // LIMITATIONS.md; the raw bytes are also available as `data`.
    py::class_<ElevationView>(m, "Elevation")
        .def_property_readonly("width",
                               [](const ElevationView& v) { return v.elevation->raster.grid.width; })
        .def_property_readonly("height",
                               [](const ElevationView& v) { return v.elevation->raster.grid.height; })
        // (x, y) cell size in projection units.
        .def_property_readonly("resolution", [](const ElevationView& v) {
            return py::make_tuple(v.elevation->raster.grid.pixel_size_x,
                                  v.elevation->raster.grid.pixel_size_y);
        })
        // (x, y) of the outer top-left corner of pixel (0,0).
        .def_property_readonly("origin", [](const ElevationView& v) {
            return py::make_tuple(v.elevation->raster.grid.origin_x,
                                  v.elevation->raster.grid.origin_y);
        })
        // Compatibility accessor over `crs` (M12 §4.9, WP-12.8).
        .def_property_readonly("epsg",
                               [](const ElevationView& v) { return v.elevation->epsg(); })
        .def_property_readonly("nodata", [](const ElevationView& v) {
            return static_cast<double>(v.elevation->nodata);
        })
        .def_property_readonly("source",
                               [](const ElevationView& v) { return v.elevation->source; })
        // The raw float32 samples, little-endian, row-major: width*height*4
        // bytes. The companion of `heights()` for callers that want numpy
        // without the per-value boxing.
        .def_property_readonly("data", [](const ElevationView& v) {
            return py::bytes(reinterpret_cast<const char*>(v.elevation->raster.data.data()),
                             v.elevation->raster.data.size() * sizeof(float));
        })
        // height rows of width floats. Values equal to `nodata` are returned as
        // they are (the sentinel, not None), so a caller can test
        // `value == dem.nodata` without a second pass over the grid.
        .def("heights", [](const ElevationView& v) {
            return reshape_scalar(v.elevation->raster.data, v.elevation->raster.grid.height,
                                  v.elevation->raster.grid.width);
        })
        // Bilinear height at a world (x, y) in the grid's plane frame, or None
        // when the point is off the grid or no valid cell contributes.
        .def(
            "height_at",
            [](const ElevationView& v, double x, double y) -> py::object {
                double h = 0.0;
                if (!sample_height(v.elevation->raster, v.elevation->nodata, x, y, h)) {
                    return py::none();
                }
                return py::float_(h);
            },
            py::arg("x"), py::arg("y"));

    py::class_<OrthomosaicView>(m, "Orthomosaic")
        .def_property_readonly("width", [](const OrthomosaicView& v) {
            return v.orthomosaic->raster.grid.width;
        })
        .def_property_readonly("height", [](const OrthomosaicView& v) {
            return v.orthomosaic->raster.grid.height;
        })
        .def_property_readonly("resolution", [](const OrthomosaicView& v) {
            return py::make_tuple(v.orthomosaic->raster.grid.pixel_size_x,
                                  v.orthomosaic->raster.grid.pixel_size_y);
        })
        .def_property_readonly("origin", [](const OrthomosaicView& v) {
            return py::make_tuple(v.orthomosaic->raster.grid.origin_x,
                                  v.orthomosaic->raster.grid.origin_y);
        })
        // Compatibility accessor over `crs` (M12 §4.9, WP-12.8).
        .def_property_readonly("epsg",
                               [](const OrthomosaicView& v) { return v.orthomosaic->epsg(); })
        .def_property_readonly("surface",
                               [](const OrthomosaicView& v) { return v.orthomosaic->surface; })
        // The raw RGBA bytes, row-major: width*height*4.
        .def_property_readonly("data", [](const OrthomosaicView& v) {
            return py::bytes(reinterpret_cast<const char*>(v.orthomosaic->raster.data.data()),
                             v.orthomosaic->raster.data.size());
        })
        // height rows of width (r, g, b, a) tuples.
        .def("pixels", [](const OrthomosaicView& v) {
            const alphashape::GeoRaster<std::uint8_t>& raster = v.orthomosaic->raster;
            py::list rows;
            for (int r = 0; r < raster.grid.height; ++r) {
                py::list row;
                for (int c = 0; c < raster.grid.width; ++c) {
                    row.append(py::make_tuple(raster.at(c, r, 0), raster.at(c, r, 1),
                                              raster.at(c, r, 2), raster.at(c, r, 3)));
                }
                rows.append(row);
            }
            return rows;
        })
        // height rows of width ints: the source camera id per cell, -1 = none.
        .def("labels", [](const OrthomosaicView& v) {
            const alphashape::RasterGrid& grid = v.orthomosaic->raster.grid;
            py::list rows;
            for (int r = 0; r < grid.height; ++r) {
                py::list row;
                for (int c = 0; c < grid.width; ++c) {
                    row.append(py::int_(v.orthomosaic->labels[
                        static_cast<std::size_t>(r) * static_cast<std::size_t>(grid.width) +
                        static_cast<std::size_t>(c)]));
                }
                rows.append(row);
            }
            return rows;
        });

    // -- read-only views -----------------------------------------------------
    // Calibration, Sensor and Camera are exposed as COPIES. The chunk owns the
    // originals and reallocates its vectors on every AddPhotos, so a reference
    // into chunk.cameras() would dangle the moment another photo is imported.
    // A copy costs a few dozen bytes and removes the whole class of bug; the
    // type names still match what a user would expect to see.
    py::enum_<alphashape::CalibrationFormat>(m, "CalibrationFormat")
        .value("Xml", alphashape::CalibrationFormat::Xml)
        .value("Json", alphashape::CalibrationFormat::Json)
        .value("OpenCvXml", alphashape::CalibrationFormat::OpenCvXml)
        .value("OpenCvYaml", alphashape::CalibrationFormat::OpenCvYaml)
        .export_values();

    py::class_<alphashape::Calibration>(m, "Calibration")
        .def_property_readonly("type",
                               [](const alphashape::Calibration& c) { return camera_type_name(c.type); })
        .def_readonly("width", &alphashape::Calibration::width)
        .def_readonly("height", &alphashape::Calibration::height)
        .def_readonly("f", &alphashape::Calibration::f)
        .def_readonly("cx", &alphashape::Calibration::cx)
        .def_readonly("cy", &alphashape::Calibration::cy)
        .def_readonly("b1", &alphashape::Calibration::b1)
        .def_readonly("b2", &alphashape::Calibration::b2)
        .def_readonly("k1", &alphashape::Calibration::k1)
        .def_readonly("k2", &alphashape::Calibration::k2)
        .def_readonly("k3", &alphashape::Calibration::k3)
        .def_readonly("k4", &alphashape::Calibration::k4)
        .def_readonly("p1", &alphashape::Calibration::p1)
        .def_readonly("p2", &alphashape::Calibration::p2)
        .def_readonly("p3", &alphashape::Calibration::p3)
        .def_readonly("p4", &alphashape::Calibration::p4)
        // Posterior covariance from OptimizeCameras, as nested lists; None when
        // no covariance was computed. Rows follow covariance_params.
        .def_property_readonly("covariance_matrix",
                               [](const alphashape::Calibration& c) -> py::object {
                                   if (c.covariance.size() == 0) {
                                       return py::none();
                                   }
                                   return nested_matrix(c.covariance);
                               })
        .def_property_readonly("covariance_params",
                               [](const alphashape::Calibration& c) -> py::object {
                                   if (c.covariance_params.empty()) {
                                       return py::none();
                                   }
                                   py::list names;
                                   for (const std::string& name : c.covariance_params) {
                                       names.append(py::str(name));
                                   }
                                   return names;
                               })
        // Write this calibration to a file. Warnings (terms the chosen format
        // cannot express) are raised through the standard `warnings` module and
        // also returned as a list of strings.
        .def(
            "save",
            [](const alphashape::Calibration& c, const std::string& path,
               alphashape::CalibrationFormat format) {
                const std::vector<std::string> warnings =
                    alphashape::save_calibration(path, c, format);
                emit_warnings(warnings);
                py::list out;
                for (const std::string& message : warnings) {
                    out.append(py::str(message));
                }
                return out;
            },
            py::arg("path"),
            py::arg("format") = alphashape::CalibrationFormat::Xml,
            "Save the calibration. `width`/`height` in the OpenCV formats come "
            "from the calibration; returns the warning list.")
        // Read a calibration. Warnings are raised through `warnings`; the loaded
        // Calibration is returned.
        .def_static(
            "load",
            [](const std::string& path, alphashape::CalibrationFormat format, int width,
               int height) {
                alphashape::CalibrationFileResult result =
                    alphashape::load_calibration(path, format, width, height);
                emit_warnings(result.warnings);
                return result.calibration;
            },
            py::arg("path"),
            py::arg("format") = alphashape::CalibrationFormat::Xml, py::arg("width") = 0,
            py::arg("height") = 0,
            "Load a calibration. `width`/`height` are the fallback image size "
            "for the OpenCV formats.")
        .def("__repr__", [](const alphashape::Calibration& c) {
            return "<alphashape.Calibration " + std::string(camera_type_name(c.type)) + " " +
                   std::to_string(c.width) + "x" + std::to_string(c.height) + " f=" +
                   std::to_string(c.f) + ">";
        });

    // dynamic_attr: a sensor handed out by Chunk.sensors carries back-pointers
    // (_chunk, _index) so makeMaster() can reach the owning Document's undo
    // stack. The sensor itself is still a value copy.
    py::class_<alphashape::Sensor>(m, "Sensor", py::dynamic_attr())
        .def_property_readonly("label", [](const alphashape::Sensor& s) { return s.label; })
        .def_property_readonly("type", [](const alphashape::Sensor& s) { return camera_type_name(s.type); })
        .def_readonly("width", &alphashape::Sensor::width)
        .def_readonly("height", &alphashape::Sensor::height)
        .def_readonly("pixel_size_um", &alphashape::Sensor::pixel_size_um)
        .def_readonly("focal_length_mm", &alphashape::Sensor::focal_length_mm)
        .def_property_readonly("make", [](const alphashape::Sensor& s) { return s.make; })
        .def_property_readonly("model", [](const alphashape::Sensor& s) { return s.model; })
        .def_property_readonly("serial", [](const alphashape::Sensor& s) { return s.serial; })
        // By value: a reference into the Sensor would outlive the chunk's
        // vector reallocation. See the note above the views.
        .def_property_readonly("calibration",
                               [](const alphashape::Sensor& s) { return s.calibration; })
        .def_property_readonly("fixed_params",
                               [](const alphashape::Sensor& s) { return s.fixed_params; })
        // Index of the master sensor whose calibration this one shares, or -1.
        .def_readonly("master", &alphashape::Sensor::master_sensor_id)
        // Film fiducial layout (WP-13.18a).
        .def_property_readonly("film_enabled",
                               [](const alphashape::Sensor& s) { return s.film.enabled; })
        // Normalised film frame (WP-13.18b): millimetres per pixel, 0 when
        // CalibrateFiducials has not run yet.
        .def_property_readonly("film_resolution_mm_per_px",
                               [](const alphashape::Sensor& s) {
                                   return s.film.resolution_mm_per_px;
                               })
        // (size_x_mm, size_y_mm); (0, 0) means "derive from the fiducials".
        .def_property_readonly("film_size_mm", [](const alphashape::Sensor& s) {
            py::list out;
            out.append(s.film.film_size_mm.x());
            out.append(s.film.film_size_mm.y());
            return out;
        })
        // Detach this sensor from any master link, making it a master in its
        // own right (master_sensor_id = kNoMasterSensor). Goes through the
        // undoable "Set Sensor Master" command; only callable on a sensor
        // obtained from Chunk.sensors, which carries the owning chunk and its
        // index. The index is re-validated here because the chunk may have
        // gained or lost sensors since the value was handed out.
        .def("makeMaster",
             [](py::object self) {
                 if (!py::hasattr(self, "_chunk") || !py::hasattr(self, "_index")) {
                     throw py::type_error(
                         "makeMaster: this Sensor is not attached to a Chunk");
                 }
                 py::object chunk_object = self.attr("_chunk");
                 py::object document = owning_document_object(chunk_object);
                 const auto chunk = chunk_object.cast<std::shared_ptr<Chunk>>();
                 const auto index = py::cast<std::int32_t>(self.attr("_index"));
                 if (index < 0 ||
                     static_cast<std::size_t>(index) >= chunk->sensors().size() ||
                     chunk->sensors()[static_cast<std::size_t>(index)].label !=
                         self.cast<const alphashape::Sensor&>().label) {
                     throw alphashape::ValueError(
                         "makeMaster: the sensor no longer exists in the chunk");
                 }
                 // types.hpp documents -1 as the "no master" sentinel.
                 constexpr std::int32_t kNoMasterSensor = -1;
                 execute_command(
                     document,
                     alphashape::make_set_sensor_master_command(
                         chunk->id(), index, kNoMasterSensor));
             },
             py::is_method(m),
             "Detach this sensor from its master, making it an independent "
             "master sensor (master_sensor_id = -1). Has no effect on a sensor "
             "that is already a master.")
        // (label, film_x_mm, film_y_mm) per calibrated fiducial; y is up.
        .def_property_readonly("fiducials", [](const alphashape::Sensor& s) {
            py::list out;
            for (const alphashape::Fiducial& fiducial : s.film.fiducials) {
                out.append(py::make_tuple(fiducial.label, fiducial.film_mm.x(),
                                          fiducial.film_mm.y()));
            }
            return out;
        })
        // GNSS antenna mount: camera-frame (or body-frame with a boresight)
        // offset in metres, and the optional body->camera boresight in degrees.
        .def_property_readonly("antenna_offset", [](const alphashape::Sensor& s) {
            py::list out;
            out.append(s.antenna.offset_m.x());
            out.append(s.antenna.offset_m.y());
            out.append(s.antenna.offset_m.z());
            return out;
        })
        .def_property_readonly("antenna_boresight", [](const alphashape::Sensor& s) {
            if (!s.antenna.boresight_deg.has_value()) {
                return py::object(py::none());
            }
            py::list out;
            out.append(s.antenna.boresight_deg->x());
            out.append(s.antenna.boresight_deg->y());
            out.append(s.antenna.boresight_deg->z());
            return py::object(out);
        });

    // Frames are held by shared_ptr in Chunk, so the Python object co-owns
    // one and stays valid across workspace edits. Read-only.
    py::class_<alphashape::Frame, std::shared_ptr<alphashape::Frame>>(m, "Frame")
        .def_property_readonly("id", &alphashape::Frame::id)
        .def_property_readonly("path", &alphashape::Frame::path)
        .def_property_readonly("width", &alphashape::Frame::width)
        .def_property_readonly("height", &alphashape::Frame::height)
        .def_property_readonly("camera_id", &alphashape::Frame::camera_id);

    // The mapping protocol the pre-WP-11.11 proxy dicts exposed, shared by the
    // classes that implement keys() and __getitem__. dict(proxy) works too,
    // because Python uses keys() + __getitem__ for that.
    // (Defined as a free function: pybind11 treats a py::object first
    // parameter as the instance.)
    auto mapping_get = [](py::object self, py::object key,
                          py::object fallback) -> py::object {
        for (py::handle known : self.attr("keys")()) {
            if (known.equal(key)) {
                return self.attr("__getitem__")(key);
            }
        }
        return fallback;
    };
    auto mapping_items = [](py::object self) {
        py::list out;
        for (py::handle key : self.attr("keys")()) {
            out.append(py::make_tuple(key, self.attr("__getitem__")(key)));
        }
        return out;
    };
    auto mapping_iter = [](py::object self) {
        return py::iter(self.attr("keys")());
    };

    // Camera is a proxy: it addresses a camera inside Chunk::cameras() by id
    // and re-resolves on every access, so it stays correct across mutations
    // and never holds a pointer into a vector.
    py::class_<CameraRef> camera_class(m, "Camera");
    camera_class.def_property_readonly("id", [](const CameraRef& r) { return r.id; })
        .def_property_readonly("sensor_id",
                               [](const CameraRef& r) { return r.get().sensor_id; })
        .def_property_readonly("label", [](const CameraRef& r) { return r.get().label; })
        .def_property_readonly("orientation",
                               [](const CameraRef& r) { return r.get().orientation; })
        .def_property(
            "enabled",
            py::cpp_function([](const CameraRef& r) { return r.get().enabled; },
                             py::is_method(m)),
            py::cpp_function(
                [](CameraRef& r, bool value) {
                    execute_command(r.document,
                                    alphashape::make_set_camera_enabled_command(
                                        r.chunk->id(), {r.id}, value));
                },
                py::is_method(m)))
        .def_property(
            "selected",
            py::cpp_function([](const CameraRef& r) { return r.get().selected; },
                             py::is_method(m)),
            py::cpp_function(
                [](CameraRef& r, bool value) {
                    execute_command(r.document,
                                    alphashape::make_set_camera_selected_command(
                                        r.chunk->id(), {r.id}, value));
                },
                py::is_method(m)))
        .def_property_readonly("group", [](const CameraRef& r) -> py::object {
            for (const alphashape::CameraGroup& group : r.chunk->camera_groups()) {
                for (const std::int32_t member : group.members) {
                    if (member == r.id) {
                        return py::int_(group.id);
                    }
                }
            }
            return py::none();
        })
        .def_property_readonly("mask", [](const CameraRef& r) -> py::object {
            const alphashape::Mask* mask = find_mask(*r.chunk, r.id);
            if (mask == nullptr) {
                return py::none();
            }
            return mask_dict(*mask);
        });
    // The 4x4 world->camera transform, as a list of four rows of four
    // floats. Identity on a camera that has not been aligned.
    camera_class.def_property_readonly("pose", [](const CameraRef& ref) {
            const alphashape::Camera& c = ref.get();
            py::list rows;
            for (int row_index = 0; row_index < 4; ++row_index) {
                py::list row;
                for (int col = 0; col < 4; ++col) {
                    row.append(py::float_(c.pose(row_index, col)));
                }
                rows.append(row);
            }
            return rows;
        })
        // The camera centre C = -A^-1 b, for a pose written [A | b] with
        // X_cam = A X_world + b. A = R for an ordinary rigid pose, but an
        // AlignChunks similarity scales it (A = R/s), so the general inverse is
        // used rather than the -R^T b shortcut. An unaligned camera has the
        // identity pose, so its centre is (0, 0, 0).
        .def_property_readonly("center", [](const CameraRef& r) {
            const alphashape::Camera& c = r.get();
            const Eigen::Matrix3d A = c.pose.block<3, 3>(0, 0);
            const Eigen::Vector3d b = c.pose.block<3, 1>(0, 3);
            const Eigen::Vector3d centre = -(A.inverse() * b);
            return py::make_tuple(centre.x(), centre.y(), centre.z());
        })
        .def_property_readonly("reference", [](const CameraRef& r) -> py::object {
            const alphashape::Camera& c = r.get();
            // None when there is no measured position; otherwise the
            // (longitude, latitude, altitude) triple, with altitude NaN when
            // the GNSS log had none.
            if (!c.reference.location.has_value()) {
                return py::none();
            }
            const Eigen::Vector3d& p = *c.reference.location;
            return py::make_tuple(p.x(), p.y(), p.z());
        })
        // The per-axis 1-sigma accuracy of the reference position in metres
        // (east, north, up), as a tuple; None when there is no location.
        .def_property_readonly("reference_accuracy", [](const CameraRef& r) -> py::object {
            const alphashape::Camera& c = r.get();
            if (!c.reference.location.has_value()) {
                return py::none();
            }
            const Eigen::Vector3d& a = c.reference.accuracy;
            return py::make_tuple(a.x(), a.y(), a.z());
        })
        // Measured attitude as a (yaw, pitch, roll) tuple in degrees, in the
        // chunk's rotation order; None when the reference carries none.
        .def_property_readonly("reference_orientation",
                               [](const CameraRef& r) -> py::object {
                                   const alphashape::Camera& c = r.get();
                                   if (!c.reference.orientation_deg.has_value()) {
                                       return py::none();
                                   }
                                   const Eigen::Vector3d& a = *c.reference.orientation_deg;
                                   return py::make_tuple(a.x(), a.y(), a.z());
                               })
        // Per-axis 1-sigma attitude accuracy in degrees, positional like
        // reference_orientation; None when the reference does not set its own
        // (the chunk's reference setting then applies).
        .def_property_readonly("reference_rotation_accuracy",
                               [](const CameraRef& r) -> py::object {
                                   const alphashape::Camera& c = r.get();
                                   if (!c.reference.rotation_accuracy_deg.has_value()) {
                                       return py::none();
                                   }
                                   const Eigen::Vector3d& a =
                                       *c.reference.rotation_accuracy_deg;
                                   return py::make_tuple(a.x(), a.y(), a.z());
                               })
        // Whether this camera's measured reference position constrains the fit.
        // The real-block GCP protocol's "GCPs only" variant clears this on every
        // camera so the survey GCPs alone carry the georeference.
        .def_property(
            "reference_enabled",
            py::cpp_function([](const CameraRef& r) { return r.get().reference.enabled; },
                             py::is_method(m)),
            py::cpp_function(
                [](CameraRef& r, bool value) {
                    execute_command(r.document,
                                    alphashape::make_set_camera_reference_enabled_command(
                                        r.chunk->id(), r.id, value));
                },
                py::is_method(m)))
        // Posterior covariance of the centre, chunk-local, 3x3 nested lists.
        // All zeros on a camera whose covariance was not computed.
        .def_property_readonly("location_covariance", [](const CameraRef& r) {
            return nested_matrix(r.get().location_covariance);
        })
        // Posterior covariance of the angle-axis rotation, 3x3 nested lists.
        .def_property_readonly("rotation_covariance", [](const CameraRef& r) {
            return nested_matrix(r.get().rotation_covariance);
        })
        // Film millimetres -> scan pixels, 3x3 nested lists, or None when the
        // marks were not ordered (WP-13.18a).
        .def_property_readonly("film_transform", [](const CameraRef& r) -> py::object {
            const alphashape::Camera& c = r.get();
            if (!c.film_transform.has_value()) {
                return py::none();
            }
            return nested_matrix(*c.film_transform);
        })
        // Measured fiducial pixels, parallel to Sensor.fiducials; None marks a
        // fiducial that was not found on this scan.
        .def_property_readonly("fiducial_pixels", [](const CameraRef& r) {
            const alphashape::Camera& c = r.get();
            py::list out;
            for (const std::optional<Eigen::Vector2d>& pixel : c.fiducial_pixels) {
                if (pixel.has_value()) {
                    out.append(py::make_tuple((*pixel).x(), (*pixel).y()));
                } else {
                    out.append(py::none());
                }
            }
            return out;
        });

    // -- Marker --------------------------------------------------------------
    // A proxy: mutations go through the same undoable commands the GUI uses.
    // The dict protocol is kept alongside the attributes so existing scripts
    // that read marker["label"] keep working.
    py::class_<MarkerRef> marker_class(m, "Marker");
    marker_class.def_property_readonly("id", [](const MarkerRef& r) { return r.id; })
        .def_property(
            "label",
            py::cpp_function([](const MarkerRef& r) { return r.get().label; },
                             py::is_method(m)),
            py::cpp_function(
                [](MarkerRef& r, const std::string& value) {
                    execute_command(r.document,
                                    alphashape::make_set_marker_label_command(
                                        r.chunk->id(), r.id, value));
                },
                py::is_method(m)))
        .def_property_readonly("type", [](const MarkerRef& r) {
            switch (r.get().type) {
                case alphashape::MarkerType::Vertex: return "vertex";
                case alphashape::MarkerType::Fiducial: return "fiducial";
                case alphashape::MarkerType::Regular: break;
            }
            return "regular";
        })
        // The marker group this marker belongs to, or None. Assignment moves
        // it through the undoable "Move to Group" command.
        .def_property(
            "group",
            py::cpp_function([](const MarkerRef& r) -> py::object {
                for (const alphashape::MarkerGroup& group :
                     r.chunk->marker_groups()) {
                    for (const std::int32_t member : group.members) {
                        if (member == r.id) {
                            return py::int_(group.id);
                        }
                    }
                }
                return py::none();
            }, py::is_method(m)),
            py::cpp_function(
                [](MarkerRef& r, py::object value) {
                    const std::int32_t group_id =
                        value.is_none() ? -1 : py::cast<std::int32_t>(value);
                    execute_command(
                        r.document,
                        alphashape::make_move_to_group_command(
                            alphashape::GroupKind::Marker, r.chunk->id(),
                            {r.id}, group_id));
                },
                py::is_method(m)))
        .def_property_readonly("projections", [](const MarkerRef& r) {
            py::list out;
            for (const alphashape::MarkerProjection& projection : r.get().projections) {
                out.append(py::cast(MarkerProjectionRef{
                    r.chunk, r.document, r.id, projection.camera_id}));
            }
            return out;
        })
        .def_property(
            "pinned",
            py::cpp_function([](const MarkerRef& r) { return r.get().pinned; },
                             py::is_method(m)),
            py::cpp_function(
                [](MarkerRef& r, bool value) {
                    execute_command(r.document,
                                    alphashape::make_set_marker_pinned_command(
                                        r.chunk->id(), r.id, value));
                },
                py::is_method(m)))
        .def_property(
            "valid",
            py::cpp_function([](const MarkerRef& r) { return r.get().valid; },
                             py::is_method(m)),
            py::cpp_function(
                [](MarkerRef& r, bool value) {
                    execute_command(r.document,
                                    alphashape::make_set_marker_enabled_command(
                                        r.chunk->id(), r.id, value));
                },
                py::is_method(m)));
    marker_class
        .def_property(
            "position",
            py::cpp_function([](const MarkerRef& r) -> py::object {
                const std::optional<Eigen::Vector3d>& p = r.get().position;
                if (!p.has_value()) {
                    return py::none();
                }
                return py::make_tuple(p->x(), p->y(), p->z());
            }, py::is_method(m)),
            py::cpp_function(
                [](MarkerRef& r, py::object value) {
                    std::optional<Eigen::Vector3d> position;
                    if (!value.is_none()) {
                        const auto values = py::cast<std::vector<double>>(value);
                        if (values.size() != 3) {
                            throw alphashape::ValueError(
                                "marker.position expects an (x, y, z) tuple or None");
                        }
                        position = Eigen::Vector3d(values[0], values[1], values[2]);
                    }
                    execute_command(r.document,
                                    alphashape::make_set_marker_position_command(
                                        r.chunk->id(), r.id, position));
                },
                py::is_method(m)))
        .def_property(
            "reference",
            py::cpp_function([](const MarkerRef& r) -> py::object {
                const std::optional<Eigen::Vector3d>& p = r.get().reference;
                if (!p.has_value()) {
                    return py::none();
                }
                return py::make_tuple(p->x(), p->y(), p->z());
            }, py::is_method(m)),
            py::cpp_function(
                [](MarkerRef& r, py::object value) {
                    std::optional<Eigen::Vector3d> reference;
                    if (!value.is_none()) {
                        const auto values = py::cast<std::vector<double>>(value);
                        if (values.size() != 3) {
                            throw alphashape::ValueError(
                                "marker.reference expects a (lon, lat, alt) tuple "
                                "or None");
                        }
                        reference = Eigen::Vector3d(values[0], values[1], values[2]);
                    }
                    execute_command(r.document,
                                    alphashape::make_set_marker_reference_command(
                                        r.chunk->id(), r.id, reference,
                                        r.get().reference_accuracy));
                },
                py::is_method(m)))
        .def("__getitem__",
             [](const MarkerRef& r, const std::string& key) -> py::object {
                 const alphashape::Marker& marker = r.get();
                 if (key == "id") return py::int_(marker.id);
                 if (key == "label") return py::str(marker.label);
                 if (key == "position") {
                     if (!marker.position.has_value()) return py::none();
                     const Eigen::Vector3d& p = *marker.position;
                     return py::make_tuple(p.x(), p.y(), p.z());
                 }
                 if (key == "reference") {
                     if (!marker.reference.has_value()) return py::none();
                     const Eigen::Vector3d& p = *marker.reference;
                     return py::make_tuple(p.x(), p.y(), p.z());
                 }
                 if (key == "pinned") return py::bool_(marker.pinned);
                 if (key == "valid") return py::bool_(marker.valid);
                 throw py::key_error(key);
             },
             py::arg("key"))
        .def("keys", [](const MarkerRef&) {
            py::list out;
            for (const char* key : {"id", "label", "position", "reference",
                                    "pinned", "valid"}) {
                out.append(py::str(key));
            }
            return out;
        })
        .def("get", mapping_get, py::arg("key"),
             py::arg("default") = py::none())
        .def("items", mapping_items)
        .def("__iter__", mapping_iter);

    // -- MarkerProjection ----------------------------------------------------
    // One image measurement of a marker. The camera is a Camera proxy; pixel
    // is (x, y) in pixels and enabled mirrors `MarkerProjection::enabled`.
    py::class_<MarkerProjectionRef>(m, "MarkerProjection")
        .def_property_readonly("camera_id",
                               [](const MarkerProjectionRef& r) {
                                   return r.camera_id;
                               })
        .def_property_readonly("camera", [](const MarkerProjectionRef& r) {
            return py::cast(
                CameraRef{r.chunk, r.document, r.camera_id});
        })
        .def_property_readonly("pixel", [](const MarkerProjectionRef& r) {
            const Eigen::Vector2d& pixel = r.get().pixel;
            return py::make_tuple(pixel.x(), pixel.y());
        })
        .def_property_readonly("enabled", [](const MarkerProjectionRef& r) {
            return r.get().enabled;
        })
        // The pre-WP-13.20 dict protocol, so scripts that read
        // marker.projections[i]["camera_id"] keep working.
        .def("__getitem__",
             [](const MarkerProjectionRef& r, const std::string& key)
                 -> py::object {
                 if (key == "camera_id") return py::int_(r.camera_id);
                 if (key == "camera") {
                     return py::cast(CameraRef{r.chunk, r.document, r.camera_id});
                 }
                 if (key == "pixel") {
                     const Eigen::Vector2d& pixel = r.get().pixel;
                     return py::make_tuple(pixel.x(), pixel.y());
                 }
                 if (key == "enabled") return py::bool_(r.get().enabled);
                 throw py::key_error(key);
             },
             py::arg("key"))
        .def("keys", [](const MarkerProjectionRef&) {
            py::list out;
            for (const char* key : {"camera_id", "camera", "pixel", "enabled"}) {
                out.append(py::str(key));
            }
            return out;
        })
        // The rest of the mapping protocol, so dict(projection) and
        // `for key in projection` behave like the old plain-dict projection.
        .def("__len__", [](const MarkerProjectionRef&) { return 4; })
        .def("__iter__",
             [](const MarkerProjectionRef&) {
                 py::list out;
                 for (const char* key : {"camera_id", "camera", "pixel", "enabled"}) {
                     out.append(py::str(key));
                 }
                 return py::iter(out);
             })
        .def("get",
             [](const MarkerProjectionRef& r, const std::string& key,
                py::object fallback) -> py::object {
                 if (key == "camera_id") return py::int_(r.camera_id);
                 if (key == "camera") {
                     return py::cast(CameraRef{r.chunk, r.document, r.camera_id});
                 }
                 if (key == "pixel") {
                     const Eigen::Vector2d& pixel = r.get().pixel;
                     return py::make_tuple(pixel.x(), pixel.y());
                 }
                 if (key == "enabled") return py::bool_(r.get().enabled);
                 return fallback;
             },
             py::arg("key"), py::arg("default") = py::none())
        .def("items", [](const MarkerProjectionRef& r) {
            const Eigen::Vector2d& pixel = r.get().pixel;
            py::list out;
            out.append(py::make_tuple(py::str("camera_id"), py::int_(r.camera_id)));
            out.append(py::make_tuple(
                py::str("camera"),
                py::cast(CameraRef{r.chunk, r.document, r.camera_id})));
            out.append(py::make_tuple(py::str("pixel"),
                                      py::make_tuple(pixel.x(), pixel.y())));
            out.append(
                py::make_tuple(py::str("enabled"), py::bool_(r.get().enabled)));
            return out;
        })
        .def("values", [](const MarkerProjectionRef& r) {
            const Eigen::Vector2d& pixel = r.get().pixel;
            py::list out;
            out.append(py::int_(r.camera_id));
            out.append(py::cast(CameraRef{r.chunk, r.document, r.camera_id}));
            out.append(py::make_tuple(pixel.x(), pixel.y()));
            out.append(py::bool_(r.get().enabled));
            return out;
        });

    // -- Asset ---------------------------------------------------------------
    // Every derived product (point cloud, model, DEM, orthomosaic, depth-map
    // set, tiled model, panorama) is addressed by kind and id.
    py::class_<AssetRef> asset_class(m, "Asset");
    asset_class
        .def_property_readonly("id",
                               [](const AssetRef& r) { return r.fields().id; })
        .def_property(
            "label",
            py::cpp_function([](const AssetRef& r) { return r.fields().label; },
                             py::is_method(m)),
            py::cpp_function(
                [](AssetRef& r, const std::string& value) {
                    execute_command(r.document,
                                    alphashape::make_rename_asset_command(
                                        r.kind, r.chunk->id(), r.id, value));
                },
                py::is_method(m)))
        .def_property_readonly("group", [](const AssetRef& r) -> py::object {
            const std::int32_t group = r.fields().group_id;
            return group < 0 ? py::object(py::none()) : py::object(py::int_(group));
        })
        .def_property_readonly(
            "enabled", [](const AssetRef& r) { return r.fields().enabled; })
        .def_property_readonly("active", [](const AssetRef& r) {
            return r.id == r.active_id();
        })
        // The pre-WP-11.11 asset lists were dicts; keep that protocol.
        .def("__getitem__",
             [](const AssetRef& r, const std::string& key) -> py::object {
                 const AssetSlotBase fields = r.fields();
                 if (key == "key" || key == "id") return py::int_(fields.id);
                 if (key == "label") return py::str(fields.label);
                 if (key == "active") return py::bool_(r.id == r.active_id());
                 if (key == "group") {
                     return fields.group_id < 0
                                ? py::object(py::none())
                                : py::object(py::int_(fields.group_id));
                 }
                 if (key == "enabled") return py::bool_(fields.enabled);
                 throw py::key_error(key);
             },
             py::arg("key"))
        .def("keys", [](const AssetRef&) {
            py::list out;
            for (const char* key : {"key", "id", "label", "active", "group",
                                    "enabled"}) {
                out.append(py::str(key));
            }
            return out;
        })
        .def("get", mapping_get, py::arg("key"),
             py::arg("default") = py::none())
        .def("items", mapping_items)
        .def("__iter__", mapping_iter);

    // -- MarkerGroup ---------------------------------------------------------
    py::class_<MarkerGroupRef>(m, "MarkerGroup")
        .def_property_readonly("id", [](const MarkerGroupRef& r) { return r.id; })
        .def_property_readonly("label",
                               [](const MarkerGroupRef& r) { return r.get().label; })
        .def_property_readonly("markers", [](const MarkerGroupRef& r) {
            py::list out;
            for (const std::int32_t member : r.get().members) {
                for (const alphashape::Marker& marker : r.chunk->markers()) {
                    if (marker.id == member) {
                        out.append(py::cast(
                            MarkerRef{r.chunk, r.document, marker.id}));
                        break;
                    }
                }
            }
            return out;
        });

    // -- Scalebar ------------------------------------------------------------
    py::class_<ScalebarRef>(m, "Scalebar")
        .def_property_readonly("id", [](const ScalebarRef& r) { return r.id; })
        .def_property_readonly("label",
                               [](const ScalebarRef& r) { return r.get().label; })
        .def_property_readonly("from_", [](const ScalebarRef& r) {
            return py::make_tuple(
                r.get().from.kind == alphashape::ScalebarEndpoint::Kind::Marker
                    ? "marker" : "camera",
                r.get().from.id);
        })
        .def_property_readonly("to_", [](const ScalebarRef& r) {
            return py::make_tuple(
                r.get().to.kind == alphashape::ScalebarEndpoint::Kind::Marker
                    ? "marker" : "camera",
                r.get().to.id);
        })
        .def_property(
            "reference",
            py::cpp_function(
                [](const ScalebarRef& r) { return r.get().reference_length_m; },
                py::is_method(m)),
            py::cpp_function(
                [](ScalebarRef& r, double value) {
                    execute_command(
                        r.document,
                        alphashape::make_set_scalebar_reference_command(
                            r.chunk->id(), r.id, value,
                            r.get().reference_accuracy_m));
                },
                py::is_method(m)))
        .def_property(
            "accuracy",
            py::cpp_function(
                [](const ScalebarRef& r) { return r.get().reference_accuracy_m; },
                py::is_method(m)),
            py::cpp_function(
                [](ScalebarRef& r, double value) {
                    execute_command(
                        r.document,
                        alphashape::make_set_scalebar_reference_command(
                            r.chunk->id(), r.id, r.get().reference_length_m,
                            value));
                },
                py::is_method(m)))
        .def_property_readonly("estimated_length", [](const ScalebarRef& r) {
            return r.get().estimated_length_m;
        });

    // -- Chunk ---------------------------------------------------------------
    // Held by shared_ptr: values returned from Document methods carry the same
    // shared_ptr the Workspace owns, so Python co-owns the chunk.
    // dynamic_attr() gives each Chunk a __dict__, which the Document bindings
    // use to attach `_document` (the Python Document that handed the chunk
    // out). The task shortcuts need it; pybind11's GC traverses the dict,
    // so the Document <-> Chunk cycle is collectable.
    py::class_<Chunk, std::shared_ptr<Chunk>>(m, "Chunk", py::dynamic_attr())
        .def_property_readonly("id", &Chunk::id)
        // Stable string identity, the key Document.chunks uses.
        .def_property_readonly(
            "key", [](const Chunk& c) { return std::to_string(c.id()); })
        .def_property(
            "label",
            py::cpp_function([](const Chunk& c) { return c.label(); },
                             py::is_method(m)),
            py::cpp_function(
                [](py::object self, const std::string& value) {
                    Chunk& c = self.cast<Chunk&>();
                    execute_on_chunk(
                        self, alphashape::make_set_chunk_label_command(
                                  c.id(), owning_document(self), value));
                },
                py::is_method(m)))
        // `enabled` has no dedicated field on Chunk; it is stored in the
        // chunk meta ("true"/"false", default true) so it persists in doc.xml
        // with the rest of the free-form metadata.
        .def_property(
            "enabled",
            py::cpp_function(
                [](const Chunk& c) {
                    const std::string* value = c.meta().find("enabled");
                    return value == nullptr || *value != "false";
                },
                py::is_method(m)),
            py::cpp_function(
                [](py::object self, bool value) {
                    Chunk& c = self.cast<Chunk&>();
                    execute_on_chunk(
                        self, alphashape::make_set_chunk_meta_command(
                                  c.id(), "enabled", value ? "true" : "false"));
                },
                py::is_method(m)))
        .def_property_readonly(
            "camera_count", [](const Chunk& c) { return static_cast<int>(c.cameras().size()); })
        .def_property_readonly(
            "frame_count", [](const Chunk& c) { return static_cast<int>(c.frames().size()); })
        .def_property_readonly(
            "keypoint_count", [](const Chunk& c) { return c.tie_points().total_keypoints(); })
        .def_property_readonly("match_pair_count", [](const Chunk& c) {
            return static_cast<int>(c.tie_points().matches.size());
        })
        .def_property_readonly(
            "track_count", [](const Chunk& c) { return static_cast<int>(c.tie_points().tracks.size()); })
        .def_property_readonly("aligned_camera_count", [](const Chunk& c) {
            int count = 0;
            for (const alphashape::Camera& camera : c.cameras()) {
                if (camera.enabled) {
                    ++count;
                }
            }
            return count;
        })
        // The number of tracks that carry a triangulated point.
        .def_property_readonly("tie_point_count", [](const Chunk& c) {
            int count = 0;
            for (const alphashape::Track& track : c.tie_points().tracks) {
                if (track.point.has_value()) {
                    ++count;
                }
            }
            return count;
        })
        // The triangulated points, (x, y, z) per track that has one, in track
        // order.
        .def("tie_points",
             [](const Chunk& c) {
                 py::list out;
                 for (const alphashape::Track& track : c.tie_points().tracks) {
                     if (track.point.has_value()) {
                         const Eigen::Vector3d& p = *track.point;
                         out.append(py::make_tuple(p.x(), p.y(), p.z()));
                     }
                 }
                 return out;
             })
        .def("matches",
             [](const Chunk& c, std::int32_t camera_a, std::int32_t camera_b) {
                 // Normalise so the caller may pass either order: the stored
                 // invariant is camera_a < camera_b. An unknown pair is empty.
                 const std::int32_t lo = std::min(camera_a, camera_b);
                 const std::int32_t hi = std::max(camera_a, camera_b);
                 py::list out;
                 for (const alphashape::PairMatches& pair : c.tie_points().matches) {
                     if (pair.camera_a == lo && pair.camera_b == hi) {
                         // `match` rather than `m`: the enclosing module
                         // function's py::module_& is also named m, and
                         // g++ flags the shadow (-Wshadow is an error here).
                         for (const alphashape::KeypointMatch& match : pair.matches) {
                             out.append(py::make_tuple(match.index_a, match.index_b));
                         }
                         break;
                     }
                 }
                 return out;
             },
             py::arg("camera_a"), py::arg("camera_b"))
        .def("tracks",
             [](const Chunk& c) {
                 py::list out;
                 for (const alphashape::Track& track : c.tie_points().tracks) {
                     py::list observations;
                     for (const alphashape::Observation& obs : track.observations) {
                         observations.append(py::make_tuple(obs.camera_id, obs.keypoint_index));
                     }
                     out.append(observations);
                 }
                 return out;
             })
        .def_property_readonly("sensors", py::cpp_function([](py::object self) {
            // Copies, so the list cannot dangle when the chunk grows. Each copy
            // records the owning chunk and its index so Sensor.makeMaster()
            // can issue its undoable command.
            auto chunk = self.cast<std::shared_ptr<Chunk>>();
            py::list out;
            for (std::size_t index = 0; index < chunk->sensors().size(); ++index) {
                py::object item = py::cast(chunk->sensors()[index]);
                item.attr("_chunk") = self;
                item.attr("_index") = py::int_(index);
                out.append(std::move(item));
            }
            return out;
        }, py::is_method(m)))
        .def_property_readonly(
            "cameras",
            py::cpp_function(
                [](py::object self) {
                    auto chunk = self.cast<std::shared_ptr<Chunk>>();
                    py::object document = owning_document_object(self);
                    py::list out;
                    for (const alphashape::Camera& camera : chunk->cameras()) {
                        out.append(py::cast(
                            CameraRef{chunk, document, camera.id}));
                    }
                    return out;
                },
                py::is_method(m)))
        .def_property_readonly(
            "frames",
            py::cpp_function(
                [](const std::shared_ptr<Chunk>& c) {
                    py::list out;
                    for (const std::shared_ptr<alphashape::Frame>& frame :
                         c->frames()) {
                        out.append(py::cast(frame));
                    }
                    return out;
                },
                py::is_method(m)))
        // Camera groups (WP-11.8): the same shape the group commands edit.
        .def_property_readonly("camera_groups", [](const Chunk& c) {
            py::list out;
            for (const alphashape::CameraGroup& group : c.camera_groups()) {
                const char* type = "none";
                switch (group.type) {
                    case alphashape::CameraGroupType::Folder: type = "folder"; break;
                    case alphashape::CameraGroupType::Station: type = "station"; break;
                    case alphashape::CameraGroupType::Fixed: type = "fixed"; break;
                    case alphashape::CameraGroupType::None: break;
                }
                py::list members;
                for (const std::int32_t id : group.members) {
                    members.append(py::int_(id));
                }
                py::dict entry;
                entry[py::str("id")] = py::int_(group.id);
                entry[py::str("label")] = py::str(group.label);
                entry[py::str("type")] = py::str(type);
                entry[py::str("members")] = members;
                entry[py::str("folder")] = py::str(group.folder);
                out.append(entry);
            }
            return out;
        })
        // Rebuild the camera grouping through the undoable command, as one
        // entry on the document's stack.
        .def(
            "group_cameras",
            [](py::object self, const std::string& group_type) {
                const std::int32_t id = self.cast<const alphashape::Chunk&>().id();
                execute_on_chunk(
                    self, alphashape::make_group_cameras_command(
                              id, alphashape::parse_group_type(group_type)));
            },
            py::arg("group_type") = "station",
            "Rebuild camera groups: 'station', 'folder', or 'none' to clear.")
        .def(
            "frame_stats",
            [](const Chunk& c) {
                const alphashape::FrameStats stats = alphashape::compute_frame_stats(c);
                py::dict out;
                if (stats.has_extent) {
                    out[py::str("extent_min")] = py::make_tuple(
                        stats.extent_min.x(), stats.extent_min.y(), stats.extent_min.z());
                    out[py::str("extent_max")] = py::make_tuple(
                        stats.extent_max.x(), stats.extent_max.y(), stats.extent_max.z());
                } else {
                    out[py::str("extent_min")] = py::none();
                    out[py::str("extent_max")] = py::none();
                }
                if (stats.has_world_extent) {
                    out[py::str("world_extent_min")] = py::make_tuple(
                        stats.world_extent_min.x(), stats.world_extent_min.y(),
                        stats.world_extent_min.z());
                    out[py::str("world_extent_max")] = py::make_tuple(
                        stats.world_extent_max.x(), stats.world_extent_max.y(),
                        stats.world_extent_max.z());
                } else {
                    out[py::str("world_extent_min")] = py::none();
                    out[py::str("world_extent_max")] = py::none();
                }
                out[py::str("mean_capture_distance")] =
                    py::float_(stats.mean_capture_distance);
                out[py::str("camera_count")] = py::int_(stats.camera_count);
                out[py::str("enabled_count")] = py::int_(stats.enabled_count);
                out[py::str("aligned_count")] = py::int_(stats.aligned_count);
                out[py::str("frame_count")] = py::int_(stats.frame_count);
                out[py::str("group_count")] = py::int_(stats.group_count);
                return out;
            },
            "Summary of the camera block: extents, mean capture distance and counts.")
        .def("keypoints",
             [](const Chunk& c, std::int32_t camera_id) {
                 py::list out;
                 const alphashape::KeypointSet* set = c.tie_points().find_by_camera(camera_id);
                 if (set != nullptr) {
                     for (const alphashape::Keypoint& kp : set->keypoints) {
                         out.append(py::make_tuple(kp.x, kp.y, kp.scale, kp.orientation));
                     }
                 }
                 return out;
             },
             py::arg("camera_id"))
        .def_property_readonly("meta",
                               [](const Chunk& c) {
                                   py::dict out;
                                   for (const auto& [key, value] : c.meta().items) {
                                       out[py::str(key)] = py::str(value);
                                   }
                                   return out;
                               })
        // The chunk's local -> world (here local ENU -> ECEF) 4x4 transform, as
        // a list of four rows of four floats. Identity until the chunk has been
        // georeferenced by AlignCameras.
        .def_property_readonly("transform", [](const Chunk& c) {
            py::list rows;
            for (int r = 0; r < 4; ++r) {
                py::list row;
                for (int col = 0; col < 4; ++col) {
                    row.append(py::float_(c.transform()(r, col)));
                }
                rows.append(row);
            }
            return rows;
        })
        // The chunk's CRS EPSG code (0 = unset, 4326 = WGS84 geographic once
        // georeferenced). Writable: AlignChunks(method="reference") requires
        // both chunks to be georeferenced, and a script or test needs a way to
        // declare that. Assignment goes through the same "Set Chunk CRS"
        // command as `crs` (EPSG 0 clears to LOCAL), so validation and
        // semantics are identical to the GUI path.
        // The setter is wrapped explicitly with py::is_method: its first
        // parameter is py::object (the Python wrapper that carries
        // `_document`), which def_property's automatic detection does not cover.
        .def_property(
            "crs_epsg",
            py::cpp_function([](const Chunk& c) { return c.crs().epsg; },
                             py::is_method(m)),
            py::cpp_function(
                [](py::object self, int epsg) {
                    if (epsg < 0) {
                        throw alphashape::ValueError("crs_epsg must not be negative");
                    }
                    alphashape::CRS crs;
                    if (epsg != 0) {
                        crs.epsg = epsg;
                        crs.wkt = "EPSG:" + std::to_string(epsg);
                    }
                    apply_crs_command(self, std::move(crs));
                },
                py::is_method(m)))
#if defined(ALPHASHAPE_USE_PROJ)
        // The chunk CRS as a value (WP-12.11). Assignment goes through the
        // "Set Chunk CRS" command on the document's undo stack. It only
        // RELABELS the chunk: no reference, pose, tie point or asset is
        // transformed. Assigning CoordinateSystem() clears to LOCAL.
        // The getter and setter are wrapped explicitly with py::is_method:
        // the setter's first parameter is py::object (the Python wrapper that
        // carries `_document`), which def_property's automatic method
        // detection does not cover.
        .def_property(
            "crs",
            py::cpp_function([](const Chunk& c) { return c.coordinate_system(); },
                             py::is_method(m)),
            py::cpp_function(
                [](py::object self, const CoordinateSystem& cs) {
                    apply_crs_command(self, alphashape::CRS(cs));
                },
                py::is_method(m)),
            "The chunk coordinate system. Assigning one relabels the chunk "
            "only; data are not transformed.")
#endif  // ALPHASHAPE_USE_PROJ
        // The effective camera-reference CRS (WP-12.6/WP-12.7): the explicit
        // slot when set, else the chunk CRS. 0 = unset.
        .def_property_readonly("camera_crs_epsg",
                               [](const Chunk& c) { return c.camera_crs().epsg; })
        .def_property_readonly("marker_crs_epsg",
                               [](const Chunk& c) { return c.marker_crs().epsg; })
        // Reference settings (WP-13.1). The names are the public Python API
        // spellings so scripts that set them keep working. Each assignment goes
        // through the undoable "Set Reference Settings" command on the
        // document's stack, as with `crs`.
        .def_property(
            "camera_location_accuracy",
            py::cpp_function(
                [](const Chunk& c) {
                    return c.reference_settings().camera_location_accuracy_m;
                },
                py::is_method(m)),
            py::cpp_function(
                [](py::object self, double value) {
                    set_reference_accuracy(
                        self, &alphashape::ReferenceSettings::camera_location_accuracy_m, value);
                },
                py::is_method(m)))
        .def_property(
            "camera_rotation_accuracy",
            py::cpp_function(
                [](const Chunk& c) {
                    return c.reference_settings().camera_rotation_accuracy_deg;
                },
                py::is_method(m)),
            py::cpp_function(
                [](py::object self, double value) {
                    set_reference_accuracy(
                        self, &alphashape::ReferenceSettings::camera_rotation_accuracy_deg, value);
                },
                py::is_method(m)))
        .def_property(
            "marker_location_accuracy",
            py::cpp_function(
                [](const Chunk& c) {
                    return c.reference_settings().marker_location_accuracy_m;
                },
                py::is_method(m)),
            py::cpp_function(
                [](py::object self, double value) {
                    set_reference_accuracy(
                        self, &alphashape::ReferenceSettings::marker_location_accuracy_m, value);
                },
                py::is_method(m)))
        .def_property(
            "marker_projection_accuracy",
            py::cpp_function(
                [](const Chunk& c) {
                    return c.reference_settings().marker_projection_accuracy_px;
                },
                py::is_method(m)),
            py::cpp_function(
                [](py::object self, double value) {
                    set_reference_accuracy(
                        self, &alphashape::ReferenceSettings::marker_projection_accuracy_px,
                        value);
                },
                py::is_method(m)))
        .def_property(
            "scalebar_accuracy",
            py::cpp_function(
                [](const Chunk& c) {
                    return c.reference_settings().scalebar_accuracy_m;
                },
                py::is_method(m)),
            py::cpp_function(
                [](py::object self, double value) {
                    set_reference_accuracy(
                        self, &alphashape::ReferenceSettings::scalebar_accuracy_m, value);
                },
                py::is_method(m)))
        .def_property(
            "tiepoint_accuracy",
            py::cpp_function(
                [](const Chunk& c) { return c.reference_settings().tiepoint_accuracy_px; },
                py::is_method(m)),
            py::cpp_function(
                [](py::object self, double value) {
                    set_reference_accuracy(
                        self, &alphashape::ReferenceSettings::tiepoint_accuracy_px, value);
                },
                py::is_method(m)))
        // Whether marker references act as control points. This is the
        // persistent chunk setting; `optimizeCameras(use_gcps=...)` overrides
        // it for one call without changing it. The assignment is undoable,
        // like the accuracies above.
        .def_property(
            "use_gcps",
            py::cpp_function([](const Chunk& c) { return c.reference_settings().use_gcp; },
                             py::is_method(m)),
            py::cpp_function(
                [](py::object self, bool value) {
                    if (!py::hasattr(self, "_document")) {
                        throw py::type_error("this Chunk is not attached to a Document");
                    }
                    alphashape::ReferenceSettings settings =
                        self.cast<const alphashape::Chunk&>().reference_settings();
                    settings.use_gcp = value;
                    const std::int32_t id = self.cast<const alphashape::Chunk&>().id();
                    execute_command(
                        owning_document_object(std::move(self)),
                        alphashape::make_set_reference_settings_command(id, settings));
                },
                py::is_method(m)))
        // The reference table (WP-13.1): one dict per camera and per marker
        // with its measured and adjusted positions in `display_crs` (the
        // chunk's own CRS when omitted), the signed error, the sigma and the
        // capture distance. An unreferenced chunk reports chunk-local units.
#if defined(ALPHASHAPE_USE_PROJ)
        .def(
            "reference_table",
            [](const Chunk& c, py::object display) {
                alphashape::crs::CoordinateSystem display_crs;
                if (!display.is_none()) {
                    display_crs =
                        display.cast<alphashape::crs::CoordinateSystem>();
                }
                const alphashape::reference::ReferenceTable table =
                    alphashape::reference::build_reference_table(c, display_crs);
                py::list out;
                for (const alphashape::reference::ReferenceRow& row : table.rows) {
                    py::dict entry;
                    entry[py::str("id")] = py::int_(row.id);
                    entry[py::str("label")] = py::str(row.label);
                    const char* kind = "camera";
                    if (row.kind == alphashape::reference::ReferenceRowKind::Marker) {
                        kind = "marker";
                    } else if (row.kind ==
                               alphashape::reference::ReferenceRowKind::CheckPoint) {
                        kind = "check_point";
                    }
                    entry[py::str("kind")] = py::str(kind);
                    entry[py::str("enabled")] = py::bool_(row.enabled);
                    entry[py::str("source")] = optional_vec3(row.source);
                    entry[py::str("estimated")] = optional_vec3(row.estimated);
                    entry[py::str("error")] = optional_vec3(row.error);
                    entry[py::str("sigma")] = py::float_(row.sigma);
                    if (std::isfinite(row.capture_distance_m)) {
                        entry[py::str("capture_distance")] =
                            py::float_(row.capture_distance_m);
                    } else {
                        entry[py::str("capture_distance")] = py::none();
                    }
                    out.append(std::move(entry));
                }
                return out;
            },
            py::arg("display_crs") = py::none())
#endif  // ALPHASHAPE_USE_PROJ
        // Markers (M11 §4.1 / WP-11.7). Each entry is a dict with `id`, `label`,
        // `position` (a chunk-local (x, y, z) tuple, or None while unestimated)
        // and `reference` (a (lon, lat, alt) tuple, or None). AlignChunks pairs
        // markers by exact `label`, so two chunks need matching labels, not
        // matching ids.
        .def_property_readonly(
            "markers",
            py::cpp_function(
                [](py::object self) {
                    auto chunk = self.cast<std::shared_ptr<Chunk>>();
                    py::object document = owning_document_object(self);
                    py::list out;
                    for (const alphashape::Marker& marker : chunk->markers()) {
                        out.append(py::cast(
                            MarkerRef{chunk, document, marker.id}));
                    }
                    return out;
                },
                py::is_method(m)))
        .def_property_readonly("marker_groups", py::cpp_function(
            [](py::object self) {
                auto chunk = self.cast<std::shared_ptr<Chunk>>();
                py::object document = owning_document_object(self);
                py::list out;
                for (const alphashape::MarkerGroup& group :
                     chunk->marker_groups()) {
                    out.append(py::cast(
                        MarkerGroupRef{chunk, document, group.id}));
                }
                return out;
            }, py::is_method(m)))
        .def_property_readonly("scalebars", py::cpp_function(
            [](py::object self) {
                auto chunk = self.cast<std::shared_ptr<Chunk>>();
                py::object document = owning_document_object(self);
                py::list out;
                for (const alphashape::Scalebar& bar : chunk->scalebars()) {
                    out.append(py::cast(ScalebarRef{chunk, document, bar.id}));
                }
                return out;
            }, py::is_method(m)))
        // The ground-control view: every marker that carries a measured
        // reference, split by role. `markers` holds all referenced markers,
        // `control_points` the ones the adjustment may use and `check_points`
        // the held-out ones (Marker meta role "check_point"). This is a view
        // over the same Marker proxies as Chunk.markers, not a copy.
        .def_property_readonly("ground_control", py::cpp_function(
            [](py::object self) {
                auto chunk = self.cast<std::shared_ptr<Chunk>>();
                // This view is read-only, so a chunk with no owning Document
                // still reports its referenced markers: the proxies carry no
                // owner and only an edit through them would raise.
                py::object document =
                    py::hasattr(self, "_document") ? self.attr("_document")
                                                   : py::object(py::none());
                py::list all;
                py::list control;
                py::list checks;
                for (const alphashape::Marker& marker : chunk->markers()) {
                    if (!marker.reference.has_value()) {
                        continue;
                    }
                    py::object ref =
                        py::cast(MarkerRef{chunk, document, marker.id});
                    all.append(ref);
                    if (alphashape::marker_role(marker) ==
                        alphashape::MarkerReferenceRole::CheckPoint) {
                        checks.append(ref);
                    } else {
                        control.append(ref);
                    }
                }
                py::dict out;
                out[py::str("markers")] = all;
                out[py::str("control_points")] = control;
                out[py::str("check_points")] = checks;
                return out;
            }, py::is_method(m)))
        // Vector annotations (declared and persisted; the editing UI lands
        // later). One dict per feature, in id order.
        .def_property_readonly("shapes", [](const Chunk& c) {
            py::list out;
            for (const alphashape::Shape& shape : c.shapes().features) {
                py::dict entry;
                entry[py::str("id")] = py::int_(shape.id);
                switch (shape.type) {
                    case alphashape::ShapeType::Polyline:
                        entry[py::str("type")] = py::str("polyline");
                        break;
                    case alphashape::ShapeType::Polygon:
                        entry[py::str("type")] = py::str("polygon");
                        break;
                    case alphashape::ShapeType::Point:
                        entry[py::str("type")] = py::str("point");
                        break;
                }
                entry[py::str("label")] = py::str(shape.label);
                entry[py::str("closed")] = py::bool_(shape.closed);
                entry[py::str("layer")] = py::str(shape.layer);
                entry[py::str("has_altitude")] = py::bool_(shape.has_altitude);
                py::list vertices;
                for (const Eigen::Vector3d& v : shape.vertices) {
                    vertices.append(py::make_tuple(v.x(), v.y(), v.z()));
                }
                entry[py::str("vertices")] = vertices;
                py::dict attributes;
                for (const auto& [key, value] : shape.attributes) {
                    attributes[py::str(key)] = py::str(value);
                }
                entry[py::str("attributes")] = attributes;
                out.append(std::move(entry));
            }
            return out;
        })
        // Timed camera tracks (declared and persisted; populated by future
        // video/import work).
        .def_property_readonly("trajectories", [](const Chunk& c) {
            py::list out;
            for (const alphashape::Trajectory& track : c.trajectories()) {
                py::dict entry;
                entry[py::str("id")] = py::int_(track.id);
                entry[py::str("label")] = py::str(track.label);
                py::list points;
                for (const alphashape::TrajectoryPoint& point : track.points) {
                    py::dict sample;
                    sample[py::str("position")] =
                        py::make_tuple(point.position.x(), point.position.y(),
                                       point.position.z());
                    sample[py::str("timestamp")] = py::float_(point.timestamp_s);
                    if (point.has_orientation) {
                        sample[py::str("orientation")] = py::make_tuple(
                            point.orientation.w(), point.orientation.x(),
                            point.orientation.y(), point.orientation.z());
                    } else {
                        sample[py::str("orientation")] = py::none();
                    }
                    points.append(std::move(sample));
                }
                entry[py::str("points")] = points;
                out.append(std::move(entry));
            }
            return out;
        })
        // One mask per camera (a u8 plane; see mask_dict).
        .def_property_readonly("masks", [](const Chunk& c) {
            py::list out;
            for (const alphashape::Mask& mask : c.masks()) {
                out.append(mask_dict(mask));
            }
            return out;
        })
        .def_property_readonly("region", [](const Chunk& c) {
            py::dict entry;
            const Eigen::Vector3d& centre = c.region().centre;
            const Eigen::Vector3d& size = c.region().size;
            entry[py::str("center")] =
                py::make_tuple(centre.x(), centre.y(), centre.z());
            entry[py::str("size")] = py::make_tuple(size.x(), size.y(), size.z());
            return entry;
        })
        // Manual marker placement (WP-13.4). Each edit goes through the same
        // undoable model command the GUI uses and is recorded on the
        // document's undo stack.
        .def(
            "addMarker",
            [](py::object self, py::object label, py::object position) {
                auto chunk = self.cast<std::shared_ptr<Chunk>>();
                py::object document = owning_document_object(self);
                std::optional<std::string> marker_label;
                if (!label.is_none()) {
                    marker_label = py::cast<std::string>(label);
                }
                std::optional<Eigen::Vector3d> marker_position;
                if (!position.is_none()) {
                    const auto values = py::cast<std::vector<double>>(position);
                    if (values.size() != 3) {
                        throw alphashape::ValueError(
                            "addMarker: position must be an (x, y, z) tuple");
                    }
                    marker_position =
                        Eigen::Vector3d(values[0], values[1], values[2]);
                }
                const std::int32_t new_id = chunk->next_id();
                execute_command(
                    document,
                    alphashape::make_add_marker_command(
                        chunk->id(), std::move(marker_label), marker_position));
                for (const alphashape::Marker& marker : chunk->markers()) {
                    if (marker.id == new_id) {
                        return py::cast(MarkerRef{chunk, document, new_id});
                    }
                }
                throw alphashape::Error("addMarker: the marker was not created");
            },
            py::arg("label") = py::none(), py::arg("position") = py::none(),
            "Create a marker and return it. A missing label generates the "
            "lowest free \"point N\"; a duplicate label raises ValueError. "
            "position is an optional chunk-local (x, y, z) tuple.")
        .def(
            "addMarkerGroup",
            [](py::object self, const std::string& label) -> py::object {
                auto chunk = self.cast<std::shared_ptr<Chunk>>();
                py::object document = owning_document_object(self);
                const std::int32_t id = chunk->next_id();
                execute_command(
                    document,
                    alphashape::make_add_group_command(
                        alphashape::GroupKind::Marker, chunk->id(), label));
                for (const alphashape::MarkerGroup& group :
                     chunk->marker_groups()) {
                    if (group.id == id) {
                        return py::cast(
                            MarkerGroupRef{chunk, document, id});
                    }
                }
                throw alphashape::Error(
                    "addMarkerGroup: the group was not created");
            },
            py::arg("label") = "",
            "Create a marker group. An empty label uses \"Marker Group\".")
        .def(
            "addScalebar",
            [](py::object self, py::object from_, py::object to,
               double reference, double accuracy, const std::string& label)
                -> py::object {
                auto chunk = self.cast<std::shared_ptr<Chunk>>();
                py::object document = owning_document_object(self);
                alphashape::Scalebar bar;
                bar.label = label;
                bar.from = scalebar_endpoint(*chunk, from_, "from_");
                bar.to = scalebar_endpoint(*chunk, to, "to");
                bar.reference_length_m = reference;
                bar.reference_accuracy_m = accuracy;
                const std::int32_t id = chunk->next_id();
                execute_command(
                    document,
                    alphashape::make_add_scalebar_command(chunk->id(), bar));
                for (const alphashape::Scalebar& candidate :
                     chunk->scalebars()) {
                    if (candidate.id == id) {
                        return py::cast(ScalebarRef{chunk, document, id});
                    }
                }
                throw alphashape::Error("addScalebar: the scalebar was not created");
            },
            py::arg("from_"), py::arg("to"), py::arg("reference") = 0.0,
            py::arg("accuracy") = 0.001, py::arg("label") = "",
            "Add a scale bar between two endpoints. Each endpoint is a marker "
            "id, a marker label, or (\"marker\"|\"camera\", id-or-label).")
        .def(
            "copy",
            [](py::object self, const std::string& label) -> py::object {
                alphashape::Document& document = owning_document(self);
                auto chunk = self.cast<std::shared_ptr<Chunk>>();
                const std::int32_t created = document.workspace().next_chunk_id();
                execute_on_chunk(
                    self, alphashape::make_duplicate_chunk_command(
                              chunk->id(), label));
                for (const auto& candidate : document.workspace().chunks()) {
                    if (candidate->id() == created) {
                        return attach_document(owning_document_object(self),
                                               candidate);
                    }
                }
                throw alphashape::Error("copy: the chunk was not duplicated");
            },
            py::arg("label") = "",
            "Duplicate this chunk, ids remapped, and return the new chunk.")
        .def(
            "setMarkerProjection",
            [](py::object self, py::object marker, py::object camera, double u,
               double v) {
                alphashape::Chunk& chunk = self.cast<alphashape::Chunk&>();
                const std::int32_t camera_id = camera_id_from_python(chunk, camera);
                // A string names a marker by label (and may name one that does
                // not exist yet, which the command creates); a Marker object or
                // an integer names it by id and must already exist. The two
                // never shadow each other, so a marker labelled "5" cannot
                // capture an edit meant for marker id 5.
                alphashape::MarkerRef reference =
                    py::isinstance<py::str>(marker)
                        ? alphashape::MarkerRef(py::cast<std::string>(marker))
                        : alphashape::MarkerRef(
                              marker_id_from_python(chunk, marker));
                execute_on_chunk(
                    self, alphashape::make_add_marker_with_projection_command(
                              chunk.id(), std::move(reference), camera_id,
                              Eigen::Vector2d(u, v),
                              /*pinned=*/false));
            },
            py::arg("marker"), py::arg("camera"), py::arg("u"), py::arg("v"),
            "Set or replace one marker projection. A string `marker` is an exact "
            "label (creating the marker when absent); a Marker object or int is "
            "a marker id that must exist. `camera` is an id or exact label.")
        .def(
            "removeMarkerProjection",
            [](py::object self, py::object marker, py::object camera,
               bool delete_marker) {
                alphashape::Chunk& chunk = self.cast<alphashape::Chunk&>();
                const std::int32_t marker_id = marker_id_from_python(chunk, marker);
                const std::int32_t camera_id = camera_id_from_python(chunk, camera);
                execute_on_chunk(
                    self, alphashape::make_remove_marker_projection_command(
                              chunk.id(), marker_id, camera_id, delete_marker));
            },
            py::arg("marker"), py::arg("camera"), py::arg("delete_marker") = false,
            "Remove one projection. With delete_marker a marker whose last "
            "projection this was is removed too.")
        .def(
            "removeMarker",
            [](py::object self, py::object marker) {
                alphashape::Chunk& chunk = self.cast<alphashape::Chunk&>();
                const std::int32_t marker_id =
                    marker_id_from_python(chunk, marker);
                execute_on_chunk(
                    self, alphashape::make_remove_marker_command(chunk.id(),
                                                                 marker_id));
            },
            py::arg("marker"),
            "Remove one marker (by id or exact label) as an undoable step.")
        // Place or move a marker: sets `position` on the marker with `label`,
        // creating the marker (and allocating its id) when the chunk has none
        // with that label. This is the manual placement path WP-11.1 ships
        // (detection is M13); AlignChunks is its first consumer.
        .def(
            "set_marker_position",
            [](py::object self, const std::string& label, double x, double y,
               double z) {
                if (label.empty()) {
                    throw alphashape::ValueError("set_marker_position: empty label");
                }
                const Eigen::Vector3d position(x, y, z);
                if (!position.allFinite()) {
                    throw alphashape::ValueError(
                        "set_marker_position: position must be finite");
                }
                auto chunk = self.cast<std::shared_ptr<Chunk>>();
                for (const alphashape::Marker& marker : chunk->markers()) {
                    if (marker.label == label) {
                        execute_on_chunk(
                            self,
                            alphashape::make_set_marker_position_command(
                                chunk->id(), marker.id, position));
                        return marker.id;
                    }
                }
                const std::int32_t id = chunk->next_id();
                // One call is one undo step: create and enable together.
                auto transaction = std::make_unique<alphashape::CommandTransaction>(
                    "Set Marker Position");
                transaction->add(alphashape::make_add_marker_command(
                    chunk->id(), label, position));
                transaction->add(alphashape::make_set_marker_enabled_command(
                    chunk->id(), id, true));
                execute_on_chunk(self, std::move(transaction));
                return id;
            },
            py::arg("label"), py::arg("x"), py::arg("y"), py::arg("z"),
            "Create or move the marker with this exact label to (x, y, z) in "
            "chunk-local coordinates. Returns its id.")
        // Set or clear one camera's per-axis attitude accuracy in degrees, the
        // sigma trio the adjustment weights that camera's rotation prior by.
        // The edit goes through the undoable command by camera id.
        .def(
            "set_camera_reference_rotation_accuracy",
            [](py::object self, std::int32_t camera_id, py::object accuracy) {
                std::optional<Eigen::Vector3d> sigma;
                if (!accuracy.is_none()) {
                    const auto values = py::cast<std::vector<double>>(accuracy);
                    if (values.size() != 3) {
                        throw alphashape::ValueError(
                            "set_camera_reference_rotation_accuracy: accuracy must "
                            "be a (first, second, third) triple");
                    }
                    sigma = Eigen::Vector3d(values[0], values[1], values[2]);
                }
                const std::int32_t chunk_id =
                    self.cast<const Chunk&>().id();
                execute_on_chunk(
                    self,
                    alphashape::make_set_camera_reference_rotation_accuracy_command(
                        chunk_id, camera_id, sigma));
            },
            py::arg("camera_id"), py::arg("accuracy"),
            "Set one camera's per-axis attitude accuracy in degrees, positional "
            "like reference_orientation; None restores the chunk setting.")
        // Set (or replace) one chunk-meta property.
        .def(
            "set_meta",
            [](py::object self, const std::string& key,
               const std::string& value) {
                if (key.empty()) {
                    throw alphashape::ValueError("set_meta: empty key");
                }
                const std::int32_t chunk_id =
                    self.cast<const Chunk&>().id();
                execute_on_chunk(
                    self, alphashape::make_set_chunk_meta_command(chunk_id, key,
                                                                  value));
            },
            py::arg("key"), py::arg("value"))
        // Replace one sensor's calibration through the undoable "Set Sensor
        // Calibration" command on the document's stack.
        .def(
            "set_sensor_calibration",
            [](py::object self, std::int32_t sensor_index,
               const alphashape::Calibration& calibration) {
                const alphashape::Chunk& chunk = self.cast<const alphashape::Chunk&>();
                execute_on_chunk(
                    self, alphashape::make_set_sensor_calibration_command(
                              chunk.id(), sensor_index, calibration));
            },
            py::arg("sensor_index"), py::arg("calibration"),
            "Replace a sensor's calibration (undoable in the GUI).")
        // Link a sensor to a master (master_index >= 0) or detach it (-1).
        .def(
            "set_sensor_master",
            [](py::object self, std::int32_t sensor_index, std::int32_t master_index) {
                const alphashape::Chunk& chunk = self.cast<const alphashape::Chunk&>();
                execute_on_chunk(
                    self, alphashape::make_set_sensor_master_command(
                              chunk.id(), sensor_index, master_index));
            },
            py::arg("sensor_index"), py::arg("master_index"),
            "Share a sensor's calibration with a master sensor, or detach with -1.")
        // Replace a sensor's film/fiducial layout (WP-13.18a) through the
        // undoable "Set Sensor Film" command.
        .def(
            "set_sensor_film",
            [](py::object self, std::int32_t sensor_index, bool enabled,
               const std::vector<std::tuple<std::string, double, double>>& fiducials,
               const std::string& transform, const std::string& detector,
               bool corners, bool sides, double resolution_mm_per_px,
               std::optional<std::pair<double, double>> film_size_mm) {
                alphashape::FilmSettings film;
                film.enabled = enabled;
                if (!(resolution_mm_per_px >= 0.0)) {
                    throw alphashape::ValueError(
                        "set_sensor_film: resolution must be non-negative");
                }
                film.resolution_mm_per_px = resolution_mm_per_px;
                if (film_size_mm.has_value()) {
                    film.film_size_mm =
                        Eigen::Vector2d(film_size_mm->first, film_size_mm->second);
                    if (!(film.film_size_mm.x() >= 0.0) ||
                        !(film.film_size_mm.y() >= 0.0)) {
                        throw alphashape::ValueError(
                            "set_sensor_film: film size must be non-negative");
                    }
                }
                if (transform == "affine") {
                    film.transform = alphashape::FilmTransformType::Affine;
                } else if (transform == "projective") {
                    film.transform = alphashape::FilmTransformType::Projective;
                } else {
                    throw alphashape::ValueError(
                        "set_sensor_film: transform must be affine or projective");
                }
                if (detector == "circle") {
                    film.detector = alphashape::FiducialDetector::Circle;
                } else if (detector == "corner") {
                    film.detector = alphashape::FiducialDetector::Corner;
                } else if (detector == "frame_corner") {
                    film.detector = alphashape::FiducialDetector::FrameCorner;
                } else {
                    throw alphashape::ValueError(
                        "set_sensor_film: detector must be circle, corner or "
                        "frame_corner");
                }
                film.search_corners = corners;
                film.search_sides = sides;
                for (const auto& entry : fiducials) {
                    alphashape::Fiducial fiducial;
                    fiducial.label = std::get<0>(entry);
                    fiducial.film_mm = Eigen::Vector2d(std::get<1>(entry),
                                                       std::get<2>(entry));
                    film.fiducials.push_back(std::move(fiducial));
                }
                const alphashape::Chunk& chunk = self.cast<const alphashape::Chunk&>();
                execute_on_chunk(
                    self, alphashape::make_set_sensor_film_command(
                              chunk.id(), sensor_index, film));
            },
            py::arg("sensor_index"), py::arg("enabled"), py::arg("fiducials"),
            py::arg("transform") = "affine", py::arg("detector") = "circle",
            py::arg("corners") = true, py::arg("sides") = false,
            py::arg("resolution_mm_per_px") = 0.0,
            py::arg("film_size_mm") = py::none(),
            "Replace a sensor's film fiducial layout (undoable in the GUI).")
        // Set the GNSS antenna mount (undoable in the GUI).
        .def(
            "set_sensor_antenna",
            [](py::object self, std::int32_t sensor_index,
               const Eigen::Vector3d& offset,
               std::optional<Eigen::Vector3d> boresight) {
                const alphashape::Chunk& chunk = self.cast<const alphashape::Chunk&>();
                alphashape::Antenna antenna;
                antenna.offset_m = offset;
                antenna.boresight_deg = boresight;
                execute_on_chunk(
                    self, alphashape::make_set_sensor_antenna_command(
                              chunk.id(), sensor_index, antenna));
            },
            py::arg("sensor_index"), py::arg("offset"), py::arg("boresight") = py::none(),
            "Set a sensor's antenna offset in metres (camera frame) and optional "
            "body-frame boresight angles in degrees. `offset` is an Eigen vector, "
            "so pybind11 needs its Eigen caster and a numpy array; the plain-three-"
            "double spelling is set_sensor_antenna_offset().")
        // The same mount through three scalars. This exists because the Eigen
        // argument above needs pybind11/eigen.h, whose numpy caster is built
        // against the numpy C API: a pybind11 older than 2.12 (Ubuntu 24.04's
        // 2.11.1) with numpy 2.x faults inside the conversion. Three doubles
        // have no numpy dependency at all, so the validation protocol and any
        // other script can set a mount without a numpy ABI coupling.
        .def(
            "set_sensor_antenna_offset",
            [](py::object self, std::int32_t sensor_index, double x, double y,
               double z) {
                const alphashape::Chunk& chunk = self.cast<const alphashape::Chunk&>();
                alphashape::Antenna antenna;
                antenna.offset_m = Eigen::Vector3d(x, y, z);
                execute_on_chunk(
                    self, alphashape::make_set_sensor_antenna_command(
                              chunk.id(), sensor_index, antenna));
            },
            py::arg("sensor_index"), py::arg("x"), py::arg("y"), py::arg("z"),
            "Set a sensor's antenna offset in metres (camera frame x, y, z). "
            "An antenna above a nadir lens is a negative z.")
        // -- dense products (milestone 6) --------------------------------------
        // BuildDepthMaps fills `depth_maps`; BuildPointCloud fuses them into
        // `point_cloud`. Both are read-only views onto the chunk's containers.
        //
        // A depth value is the distance from the camera centre along the pixel's
        // unit bearing in chunk-local units, NOT a camera-frame Z. -1 is the
        // sentinel for "no depth"; depths is always present, while normals and
        // confidence are None when the map carries none (their vectors are empty).
        // ALL depth-map sets flattened, as the public scripting API expects.
        // The ACTIVE set alone is still reachable through `depth_maps_sets`
        // and the `active_depth_maps` convenience below.
        .def_property_readonly("depth_maps", [](const Chunk& c) {
            py::list out;
            for (const alphashape::DepthMapsAsset& set : c.depth_map_sets()) {
                if (!set.data) {
                    continue;
                }
                for (const alphashape::DepthMap& map : *set.data) {
                    out.append(depth_map_entry(map));
                }
            }
            return out;
        })
        // The ACTIVE set's maps, the pre-WP-11.11 meaning of `depth_maps`,
        // kept under its own name for scripts that relied on it. The entries
        // are identical to the corresponding `depth_maps` entries.
        .def_property_readonly("active_depth_maps", [](const Chunk& c) {
            py::list out;
            for (const alphashape::DepthMap& map : c.depth_maps()) {
                out.append(depth_map_entry(map));
            }
            return out;
        })
        // The fused cloud. points is Nx3 float64; normals (Nx3), colours (Nx3
        // uint8) and confidence (N) are None when the task did not store them.
        // point_count is the length of points, so an empty cloud reports 0 even
        // though points is a (rather than a None) empty list.
        .def_property_readonly("point_cloud", [](const Chunk& c) {
            const alphashape::PointCloud& cloud = c.point_cloud();
            const std::size_t n = cloud.points.size();
            py::list points;
            for (const Eigen::Vector3d& p : cloud.points) {
                points.append(py::make_tuple(p.x(), p.y(), p.z()));
            }
            py::list normals;
            for (const Eigen::Vector3f& n_vec : cloud.normals) {
                normals.append(py::make_tuple(static_cast<double>(n_vec.x()),
                                              static_cast<double>(n_vec.y()),
                                              static_cast<double>(n_vec.z())));
            }
            py::list colours;
            for (std::size_t i = 0; i + 2 < cloud.colours.size(); i += 3) {
                colours.append(py::make_tuple(cloud.colours[i], cloud.colours[i + 1],
                                              cloud.colours[i + 2]));
            }
            py::list confidence;
            for (float value : cloud.confidence) {
                confidence.append(py::float_(static_cast<double>(value)));
            }
            py::dict out;
            out[py::str("points")] = std::move(points);
            out[py::str("normals")] =
                cloud.normals.empty() ? py::object(py::none()) : py::object(std::move(normals));
            out[py::str("colors")] =
                cloud.colours.empty() ? py::object(py::none()) : py::object(std::move(colours));
            out[py::str("confidence")] = cloud.confidence.empty()
                                             ? py::object(py::none())
                                             : py::object(std::move(confidence));
            out[py::str("point_count")] = py::int_(n);
            return out;
        })
        // The reconstructed mesh (milestone 7, BuildModel). vertices is Nx3
        // float64; faces is Fx3 int triples indexing into vertices; colours
        // (Nx3 uint8) and confidence (N) are None when the task did not store
        // them. vertex_count / face_count mirror the list lengths, so an empty
        // mesh reports 0 rather than a null view.
        .def_property_readonly("model", [](const Chunk& c) {
            const alphashape::Mesh& mesh = c.model();
            py::list vertices;
            for (const Eigen::Vector3d& v : mesh.vertices) {
                vertices.append(py::make_tuple(v.x(), v.y(), v.z()));
            }
            py::list faces;
            for (const Eigen::Vector3i& f : mesh.faces) {
                faces.append(py::make_tuple(f(0), f(1), f(2)));
            }
            py::list colours;
            for (std::size_t i = 0; i + 2 < mesh.colours.size(); i += 3) {
                colours.append(py::make_tuple(mesh.colours[i], mesh.colours[i + 1],
                                              mesh.colours[i + 2]));
            }
            py::list confidence;
            for (float value : mesh.confidence) {
                confidence.append(py::float_(static_cast<double>(value)));
            }
            // Milestone 8: the UV layout. uvs is the coordinate pool (list of
            // (u, v)); uv_faces is one index triple per face, in the same corner
            // order as faces. Both are None together when the model has no UV
            // layout (BuildUV has not run).
            py::list uvs;
            for (const Eigen::Vector2f& uv : mesh.uvs) {
                uvs.append(py::make_tuple(py::float_(uv.x()), py::float_(uv.y())));
            }
            py::list uv_faces;
            for (const Eigen::Vector3i& uvf : mesh.uv_faces) {
                uv_faces.append(py::make_tuple(uvf(0), uvf(1), uvf(2)));
            }
            py::dict out;
            out[py::str("vertices")] = std::move(vertices);
            out[py::str("faces")] = std::move(faces);
            out[py::str("colors")] =
                mesh.colours.empty() ? py::object(py::none()) : py::object(std::move(colours));
            out[py::str("confidence")] = mesh.confidence.empty()
                                             ? py::object(py::none())
                                             : py::object(std::move(confidence));
            out[py::str("uvs")] =
                mesh.uvs.empty() ? py::object(py::none()) : py::object(std::move(uvs));
            out[py::str("uv_faces")] = mesh.uv_faces.empty()
                                           ? py::object(py::none())
                                           : py::object(std::move(uv_faces));
            out[py::str("vertex_count")] = py::int_(mesh.vertices.size());
            out[py::str("face_count")] = py::int_(mesh.faces.size());
            return out;
        })
        // The baked texture page (milestone 8, BuildTexture), or None. Page 0
        // only for now: the mesh's uv_faces address that single page, so the
        // dict carries the raw RGB bytes plus the dimensions needed to index
        // them.
        .def_property_readonly("model_texture", [](const Chunk& c) -> py::object {
            const std::vector<alphashape::Image>& pages = c.model_textures();
            if (pages.empty() || pages[0].empty()) {
                return py::none();
            }
            const alphashape::Image& page = pages[0];
            py::dict out;
            out[py::str("width")] = py::int_(page.width());
            out[py::str("height")] = py::int_(page.height());
            out[py::str("channels")] = py::int_(page.channels());
            out[py::str("data")] =
                py::bytes(reinterpret_cast<const char*>(page.data()), page.size_bytes());
            return out;
        })
        // The DEM (milestone 9, BuildDem), or None. The view holds the chunk's
        // shared_ptr plus a pointer into the ACTIVE elevation asset, so it
        // outlives `del chunk`; a later buildDem() makes a NEW asset active and
        // this view keeps showing the DEM it was obtained for.
        .def_property_readonly("elevation", [](const std::shared_ptr<Chunk>& c) -> py::object {
            if (c->elevation().empty()) {
                return py::none();
            }
            ElevationView view;
            view.chunk = c;
            view.elevation = &c->elevation();
            return py::cast(std::move(view));
        })
        // The orthomosaic (milestone 9, BuildOrthomosaic), or None.
        .def_property_readonly("orthomosaic",
                               [](const std::shared_ptr<Chunk>& c) -> py::object {
                                   if (c->orthomosaic().empty()) {
                                       return py::none();
                                   }
                                   OrthomosaicView view;
                                   view.chunk = c;
                                   view.orthomosaic = &c->orthomosaic();
                                   return py::cast(std::move(view));
                               })
        // -- multi-instance assets (milestone 11) ------------------------------
        // One asset view per asset of each kind, in id order. The scalar
        // properties above (elevation, orthomosaic, model, point_cloud,
        // model_texture) are views onto the ACTIVE asset, which is the one the
        // last producing task added. `depth_maps` is the exception: it
        // flattens every set, while `active_depth_maps` is the active one.
        .def_property_readonly("elevations", py::cpp_function(
            [](py::object self) {
                auto c = self.cast<std::shared_ptr<Chunk>>();
                return asset_refs(self, alphashape::AssetKind::Elevation,
                                  c->elevations());
            }, py::is_method(m)))
        .def_property_readonly("orthomosaics", py::cpp_function(
            [](py::object self) {
                auto c = self.cast<std::shared_ptr<Chunk>>();
                return asset_refs(self, alphashape::AssetKind::Orthomosaic,
                                  c->orthomosaics());
            }, py::is_method(m)))
        .def_property_readonly("models", py::cpp_function(
            [](py::object self) {
                auto c = self.cast<std::shared_ptr<Chunk>>();
                return asset_refs(self, alphashape::AssetKind::Model,
                                  c->models());
            }, py::is_method(m)))
        .def_property_readonly("point_clouds", py::cpp_function(
            [](py::object self) {
                auto c = self.cast<std::shared_ptr<Chunk>>();
                return asset_refs(self, alphashape::AssetKind::PointCloud,
                                  c->point_clouds());
            }, py::is_method(m)))
        .def_property_readonly("depth_maps_sets", py::cpp_function(
            [](py::object self) {
                auto c = self.cast<std::shared_ptr<Chunk>>();
                return asset_refs(self, alphashape::AssetKind::DepthMaps,
                                  c->depth_map_sets());
            }, py::is_method(m)))
        .def_property_readonly("tiled_models", py::cpp_function(
            [](py::object self) {
                auto c = self.cast<std::shared_ptr<Chunk>>();
                return asset_refs(self, alphashape::AssetKind::TiledModel,
                                  c->tiled_models());
            }, py::is_method(m)))
        .def_property_readonly("panorama", py::cpp_function(
            [](py::object self) -> py::object {
                auto c = self.cast<std::shared_ptr<Chunk>>();
                if (!c->panorama().has_value()) {
                    return py::none();
                }
                return py::cast(AssetRef{
                    c, owning_document_object(self),
                    alphashape::AssetKind::Panorama, c->panorama()->id});
            }, py::is_method(m)))
        // -- milestone 9 task shortcuts ----------------------------------------
        // Metashape-style convenience wrappers over run_task, so
        // `chunk.buildDem(interpolation="extrapolated")` and
        // `doc.run_task("BuildDem", interpolation="extrapolated")` are the same
        // call. They bind against the task's declared schema exactly like
        // run_task, so an unknown keyword is still an InvalidArgumentError.
        // They need the owning Document, which the Document bindings attach as
        // the `_document` attribute on every chunk they hand out.
        .def("buildDem", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "BuildDem", std::move(kwargs));
        })
        .def("exportDem", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "ExportDem", std::move(kwargs));
        })
        .def("buildOrthomosaic", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "BuildOrthomosaic", std::move(kwargs));
        })
        .def("exportOrthomosaic", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "ExportOrthomosaic", std::move(kwargs));
        })
        // Reference import/export (WP-13.2). The public Python API spellings;
        // each is the same run_task call as the task names. In a build without
        // PROJ the tasks are not registered and the call raises
        // UnsupportedTaskError, which is the honest answer: there is no CRS to
        // convert into.
        .def("importReference", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "ImportReference", std::move(kwargs));
        })
        .def("exportReference", [](py::object self, py::kwargs kwargs) {
            // The public spelling `columns` is the task's `columns_order`:
            // both name the ordered list of written fields. Passing both is
            // ambiguous, so refuse it rather than silently preferring one.
            if (kwargs.contains("columns") && kwargs.contains("columns_order")) {
                throw alphashape::ValueError(
                    "exportReference: pass either columns or columns_order, not "
                    "both");
            }
            if (kwargs.contains("columns")) {
                kwargs[py::str("columns_order")] =
                    kwargs.attr("pop")(py::str("columns"));
            }
            run_chunk_task(self, "ExportReference", std::move(kwargs));
        })
        // Marker projection import/export (WP-13.4), the same run_task calls.
        .def("importMarkers", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "ImportMarkers", std::move(kwargs));
        })
        .def("exportMarkers", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "ExportMarkers", std::move(kwargs));
        })
        // Automatic target detection (WP-13.8), the same run_task call.
        .def("detectMarkers", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "DetectMarkers", std::move(kwargs));
        })
        // Chessboard calibration per sensor (WP-13.15a), the same run_task call.
        .def("calibrateCamera", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "CalibrateCamera", std::move(kwargs));
        })
        // Film fiducial detection (WP-13.18a): detect marks on every film
        // camera and fit each scan's film-mm -> pixel transform. The public
        // spelling `fiducial_type` is mapped onto the task's `detector`.
        .def("detectFiducials", [](py::object self, py::kwargs kwargs) {
            // `fiducial_type` and `detector` are two spellings of the same
            // task argument. Refuse both at once instead of silently dropping
            // the one the caller may have intended.
            if (kwargs.contains("fiducial_type") && kwargs.contains("detector")) {
                throw alphashape::ValueError(
                    "detectFiducials: pass either fiducial_type or detector, not "
                    "both");
            }
            if (kwargs.contains("fiducial_type")) {
                py::object value = kwargs.attr("pop")(py::str("fiducial_type"));
                kwargs[py::str("detector")] = value;
            }
            run_chunk_task(self, "DetectFiducials", std::move(kwargs));
        })
        // Derive a sensor's fiducial film coordinates from one reference scan.
        .def("calibrateFiducials", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "CalibrateFiducials", std::move(kwargs));
        })
        // Backward STMap export for a calibrated sensor (WP-13.15b). Both
        // spellings reach the same task; `exportSTMap` is the documented one.
        .def("exportSTMap", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "ExportSTMap", std::move(kwargs));
        })
        .def("exportStmap", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "ExportSTMap", std::move(kwargs));
        })
        // Multi-view consensus refinement (WP-13.9), the same run_task call.
        .def("refineMarkers", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "RefineMarkers", std::move(kwargs));
        })
        // Pyramidal projection tracking (WP-13.10), the same run_task call.
        .def("trackMarkers", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "TrackMarkers", std::move(kwargs));
        })
        // Weighted similarity update from references and control points
        // (WP-13.12): the same run_task call as the UpdateTransform task.
        .def("updateTransform", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "UpdateTransform", std::move(kwargs));
        })
        // Self-calibrating bundle adjustment (WP-13.13). `use_gcps` is not a
        // task parameter: it selects whether control points constrain the
        // adjustment. Passing it overrides the chunk's ReferenceSettings for
        // this call only (default None uses the chunk's value); the previous
        // settings are restored on return, also on failure, with no undo entry.
        .def("optimizeCameras", [](py::object self, py::kwargs kwargs) {
            std::optional<bool> use_gcps;
            if (kwargs.contains("use_gcps")) {
                use_gcps = py::cast<bool>(kwargs.attr("pop")(py::str("use_gcps")));
            }
            if (!use_gcps.has_value()) {
                run_chunk_task(self, "OptimizeCameras", std::move(kwargs));
                return;
            }
            alphashape::Chunk& chunk = self.cast<alphashape::Chunk&>();
            const ReferenceUseGcpOverride guard(chunk, *use_gcps);
            run_chunk_task(self, "OptimizeCameras", std::move(kwargs));
        })
        // Exterior orientation from navigation metadata (WP-13.21): place the
        // selected cameras from their reference position and attitude.
        .def("alignFromReference", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "AlignFromReference", std::move(kwargs));
        })
        // Tie points from the stored matches with the current poses held
        // (WP-13.21): the same run_task call as the task.
        .def("triangulateTiePoints", [](py::object self, py::kwargs kwargs) {
            run_chunk_task(self, "TriangulateTiePoints", std::move(kwargs));
        })
#if defined(ALPHASHAPE_USE_PROJ)
        // Convert the camera/marker reference positions into another CRS
        // (WP-12.7). `target` is an EPSG code; `kind` is "cameras", "markers"
        // or "both". The slots are relabelled, poses and assets are untouched.
        // No geoid store is reachable from a bare Chunk, so a vertical datum
        // that is not an EPSG vertical CRS is refused by name (the disks this
        // front end exposes never carry one).
        .def(
            "convertReference",
            [](py::object self, int target_epsg, const std::string& kind) {
                Chunk& c = self.cast<Chunk&>();
                if (target_epsg <= 0) {
                    throw alphashape::ValueError(
                        "convertReference: target_epsg must be a positive EPSG code");
                }
                alphashape::reference::ReferenceKind which;
                if (kind == "cameras") {
                    which = alphashape::reference::ReferenceKind::Cameras;
                } else if (kind == "markers") {
                    which = alphashape::reference::ReferenceKind::Markers;
                } else if (kind == "both") {
                    which = alphashape::reference::ReferenceKind::Both;
                } else {
                    throw alphashape::ValueError(
                        "convertReference: kind must be 'cameras', 'markers' or 'both'");
                }
                alphashape::reference::convert_reference(
                    c, alphashape::crs::CoordinateSystem::from_epsg(target_epsg), which);
                // An in-place conversion bypasses the command layer, so the
                // history before it no longer describes the document.
                if (py::hasattr(self, "_document")) {
                    clear_document_history(self.attr("_document"));
                }
            },
            py::arg("target_epsg"), py::arg("kind") = "both");
#endif  // ALPHASHAPE_USE_PROJ


    // The per-document undo history is an opaque capsule on the Document
    // object (see document_undo_stack); Document.undo/redo/can_undo/can_redo
    // are the only public surface, so a script cannot retain a stack whose
    // owning Document it dropped.

    // -- Document ------------------------------------------------------------
    // dynamic_attr: the document's undo stack lives on the object (see
    // document_undo_stack), so it is destroyed with the document.
    py::class_<Document, std::shared_ptr<Document>>(m, "Document",
                                                    py::dynamic_attr())
        .def(py::init<>())
        .def_property("path", [](const Document& d) { return d.path(); },
                      [](Document& d, const std::string& path) { d.set_path(path); })
        .def_property_readonly(
            "cache_dir", [](const Document& d) { return d.cache_dir(); },
            "Scratch directory for this project (<base>.files/cache), or '' "
            "when the document has no on-disk location yet. A project.zip "
            "opened directly resolves to <base>.files/cache as well. The "
            "directory is not created until a caller writes into it, and it is "
            "removed when the last Document that shares it closes.")
        .def(
            "save",
            [](py::object self, py::object path, py::object chunks,
               py::object archive) {
                Document& d = self.cast<Document&>();
                // archive=None infers the layout from the extension;
                // archive=True forces the single-file .psz form, False the
                // directory form.
                std::string target = path.is_none() ? d.path()
                                                    : py::cast<std::string>(path);
                if (target.empty()) {
                    throw alphashape::ValueError(
                        "save: the document has no path yet; pass one");
                }
                if (!archive.is_none()) {
                    std::filesystem::path as_path =
                        alphashape::platform::path_from_utf8(target);
                    if (archive.cast<bool>()) {
                        as_path.replace_extension(".psz");
                    } else if (as_path.extension() == ".psz") {
                        as_path.replace_extension(".psx");
                    }
                    target = alphashape::platform::path_to_utf8(as_path);
                }
                // Save As can move the document out of its cache directory.
                // Pin the old one before set_path() so undo spill files that
                // retained commands still reference are not deleted; the GUI
                // does the same (apps/gui/app.cpp save_document_as).
                std::shared_ptr<void> cache_pin;
                if (target != d.path()) {
                    cache_pin = d.pin_cache_directory();
                }
                // Record the path so a later run resolves relative frame paths
                // against the project we just wrote, exactly like load() would.
                d.set_path(target);
                std::vector<std::int32_t> ids;
                if (!chunks.is_none()) {
                    // A bare int is REJECTED, not iterated: `chunks=3` would
                    // otherwise be a TypeError deep inside pybind's caster.
                    if (py::isinstance<py::int_>(chunks)) {
                        throw py::type_error(
                            "save: chunks expects a sequence of chunk ids, not a single int");
                    }
                    for (py::handle item : chunks) {
                        if (py::isinstance<Chunk>(item)) {
                            ids.push_back(item.cast<Chunk*>()->id());
                        } else if (py::isinstance<py::int_>(item)) {
                            ids.push_back(item.cast<std::int32_t>());
                        } else {
                            throw py::type_error(
                                "save: chunks expects chunk ids or Chunk objects");
                        }
                    }
                }
                d.save(target, ids);
                if (py::hasattr(self, "_undo_stack")) {
                    document_undo_stack(self)->set_cache_dir(d.cache_dir());
                }
                if (cache_pin) {
                    pin_document_history_cache(self, std::move(cache_pin));
                }
            },
            py::arg("path") = py::none(), py::arg("chunks") = py::none(),
            py::arg("archive") = py::none(),
            "Write the project. `path=None` writes to the document's current "
            "path (set when it was loaded or last saved). `chunks` restricts "
            "the file to those chunks "
            "(ids or Chunk objects); None writes every chunk. Chunk ids and the "
            "next-chunk counter are preserved so the file can be appended. "
            "`archive` selects the layout: None infers it from `path` (a .psz "
            "path writes the single-file form, anything else the directory "
            "form); True forces a single-file .psz and False the directory "
            "form, rewriting the extension when it does not already match.")
        .def("add_chunk", &document_add_chunk, py::arg("label") = py::none(),
             "Add a chunk and return it. The active chunk is unchanged. With no "
             "label the chunk is named 'Chunk N' for the smallest unused N.")
        .def("addChunk", &document_add_chunk, py::arg("label") = py::none(),
             "Add a chunk and return it (public scripting API spelling).")
        .def("chunk",
             [](py::object self, std::int32_t id) -> py::object {
                 Document& d = self.cast<Document&>();
                 for (auto& chunk : d.workspace().chunks()) {
                     if (chunk->id() == id) {
                         return attach_document(self, chunk);
                     }
                 }
                 return py::none();
             },
             py::arg("id"))
        .def_property_readonly(
            "chunk_ids",
            [](const Document& d) {
                std::vector<std::int32_t> ids;
                ids.reserve(d.workspace().chunks().size());
                for (const auto& chunk : d.workspace().chunks()) {
                    ids.push_back(chunk->id());
                }
                return ids;
            },
            "Ids of the workspace's chunks, in workspace order.")
        // Chunks keyed by Chunk.key, the public scripting API shape. Python
        // 3.7+ dicts preserve insertion order, so iteration is workspace
        // order.
        .def_property_readonly(
            "chunks",
            [](py::object self) {
                Document& d = self.cast<Document&>();
                py::dict out;
                for (const auto& chunk : d.workspace().chunks()) {
                    out[py::str(std::to_string(chunk->id()))] =
                        attach_document(self, chunk);
                }
                return out;
            },
            "Chunks keyed by Chunk.key.")
        .def(
            "remove",
            [](py::object self, py::object chunks) {
                std::vector<std::int32_t> ids;
                auto collect = [&ids](py::handle item) {
                    if (py::isinstance<Chunk>(item)) {
                        ids.push_back(item.cast<Chunk*>()->id());
                    } else if (py::isinstance<py::int_>(item)) {
                        ids.push_back(item.cast<std::int32_t>());
                    } else {
                        throw py::type_error(
                            "remove expects chunk ids or Chunk objects");
                    }
                };
                for (const py::object& item :
                     chunk_argument_items(std::move(chunks))) {
                    collect(item);
                }
                if (ids.empty()) {
                    return;
                }
                auto transaction = std::make_unique<alphashape::CommandTransaction>(
                    "Remove Chunks");
                for (const std::int32_t id : ids) {
                    transaction->add(
                        alphashape::make_remove_chunk_command(id));
                }
                execute_command(self, std::move(transaction));
            },
            py::arg("chunks"),
            "Remove one chunk or a sequence of chunks (ids or Chunk objects) "
            "as one undoable step.")
        .def(
            "append",
            [](py::object self, py::object other) {
                if (!py::isinstance<Document>(other)) {
                    throw py::type_error("append expects a Document");
                }
                auto source = other.cast<std::shared_ptr<Document>>();
                execute_command(
                    self, alphashape::make_append_document_command(
                              source, source->path()));
            },
            py::arg("doc"),
            "Append every chunk of another Document, remapping chunk ids past "
            "this document's next id.")
        .def("undo",
             [](py::object self) { document_undo_stack(self)->undo(); },
             "Undo the last edit recorded on this document's stack.")
        .def("redo",
             [](py::object self) { document_undo_stack(self)->redo(); },
             "Redo the last undone edit.")
        .def_property_readonly("can_undo", [](py::object self) {
            return document_undo_stack(self)->can_undo();
        })
        .def_property_readonly("can_redo", [](py::object self) {
            return document_undo_stack(self)->can_redo();
        })
        .def_property(
            "active_chunk",
            [](py::object self) -> py::object {
                Document& d = self.cast<Document&>();
                Chunk* chunk = d.active_chunk();
                if (chunk == nullptr) {
                    return py::none();
                }
                // Recover the shared_ptr the Workspace holds for this chunk.
                for (auto& candidate : d.workspace().chunks()) {
                    if (candidate.get() == chunk) {
                        return attach_document(self, candidate);
                    }
                }
                return py::none();
            },
            [](Document& d, py::object value) {
                if (value.is_none()) {
                    d.set_active_chunk_id(-1);
                } else if (py::isinstance<Chunk>(value)) {
                    d.set_active_chunk_id(value.cast<Chunk*>()->id());
                } else if (py::isinstance<py::int_>(value)) {
                    d.set_active_chunk_id(value.cast<std::int32_t>());
                } else {
                    throw py::type_error("active_chunk expects a Chunk, an int id, or None");
                }
            })
        .def("run_task",
             [](py::object self, const std::string& name, py::kwargs kwargs) {
                 run_task_kwargs(std::move(self), name, std::move(kwargs));
             },
             py::arg("name"))
        .def(
            "merge_chunks",
            [](py::object self, py::object chunks, const std::string& label) -> py::object {
                Document& d = self.cast<Document&>();
                std::vector<std::string> ids;
                for (const py::object& item :
                     chunk_argument_items(std::move(chunks))) {
                    if (py::isinstance<Chunk>(item)) {
                        ids.push_back(std::to_string(item.cast<Chunk*>()->id()));
                    } else if (py::isinstance<py::int_>(item)) {
                        ids.push_back(std::to_string(item.cast<std::int32_t>()));
                    } else {
                        throw py::type_error(
                            "merge_chunks expects chunk ids or Chunk objects");
                    }
                }
                // The task layer resolves its target through the active chunk;
                // the merged chunk becomes active, so any chunk will do as the
                // starting point when the document has one.
                if (d.active_chunk() == nullptr && !d.workspace().chunks().empty()) {
                    d.set_active_chunk_id(d.workspace().chunks().front()->id());
                }
                py::dict kwargs;
                kwargs[py::str("label")] = py::str(label);
                py::list id_list;
                for (const std::string& id : ids) {
                    id_list.append(py::str(id));
                }
                kwargs[py::str("chunk_ids")] = id_list;
                run_task_kwargs(self, "MergeChunks", kwargs);
                return attach_document(self, d.workspace().chunks().back());
            },
            py::arg("chunks"),
            py::arg("label") = "Merged chunk",
            "Concatenate the given chunks (ids or Chunk objects) into one new "
            "chunk and return it. Refuses chunks with different non-local CRSs.")
        .def(
            "append_project",
            [](py::object self, const std::string& path, py::object chunks,
               bool append_doc) {
                std::vector<std::string> ids;
                if (!chunks.is_none()) {
                    for (py::handle item : chunks) {
                        if (py::isinstance<py::int_>(item)) {
                            ids.push_back(std::to_string(item.cast<std::int32_t>()));
                        } else {
                            throw py::type_error(
                                "append_project: chunk_ids expects ints");
                        }
                    }
                }
                py::dict kwargs;
                kwargs[py::str("path")] = py::str(path);
                kwargs[py::str("append_doc")] = py::bool_(append_doc);
                py::list id_list;
                for (const std::string& id : ids) {
                    id_list.append(py::str(id));
                }
                kwargs[py::str("chunk_ids")] = id_list;
                run_task_kwargs(self, "AppendProject", kwargs);
            },
            py::arg("path"), py::arg("chunk_ids") = py::none(),
            py::arg("append_doc") = true,
            "Append every chunk (or the listed ids) of the project at `path`, "
            "offsetting incoming chunk ids past this document's next id.")
        .def(
            "duplicate_chunk",
            [](py::object self, py::object chunk, const std::string& label) -> py::object {
                Document& d = self.cast<Document&>();
                std::int32_t id = -1;
                if (!chunk.is_none()) {
                    if (py::isinstance<Chunk>(chunk)) {
                        id = chunk.cast<Chunk*>()->id();
                    } else if (py::isinstance<py::int_>(chunk)) {
                        id = chunk.cast<std::int32_t>();
                    } else {
                        throw py::type_error(
                            "duplicate_chunk expects a Chunk, an int id, or None");
                    }
                }
                py::dict kwargs;
                kwargs[py::str("chunk_id")] = py::int_(id);
                if (!label.empty()) {
                    kwargs[py::str("label")] = py::str(label);
                }
                run_task_kwargs(self, "DuplicateChunk", kwargs);
                return attach_document(self, d.workspace().chunks().back());
            },
            py::arg("chunk") = py::none(), py::arg("label") = "",
            "Deep-copy a chunk with every chunk-local id remapped. `chunk` is a "
            "Chunk, an id, or None for the active chunk. Returns the copy.")
        .def(
            "align_chunks",
            [](py::object self, py::object reference, const std::string& method,
               int min_common_markers, double accuracy_m) {
                std::int32_t id = -1;
                if (!reference.is_none()) {
                    if (py::isinstance<Chunk>(reference)) {
                        id = reference.cast<Chunk*>()->id();
                    } else if (py::isinstance<py::int_>(reference)) {
                        id = reference.cast<std::int32_t>();
                    } else {
                        throw py::type_error(
                            "align_chunks expects a Chunk, an int id, or None");
                    }
                }
                py::dict kwargs;
                kwargs[py::str("reference_chunk_id")] = py::int_(id);
                kwargs[py::str("method")] = py::str(method);
                kwargs[py::str("min_common_markers")] = py::int_(min_common_markers);
                kwargs[py::str("accuracy_m")] = py::float_(accuracy_m);
                run_task_kwargs(self, "AlignChunks", kwargs);
            },
            py::arg("reference") = py::none(),
            py::arg("method") = "markers", py::arg("min_common_markers") = 3,
            py::arg("accuracy_m") = 0.0,
            "Align the ACTIVE chunk onto `reference` (a Chunk, an id, or None "
            "for the first other chunk in the workspace). method=\"markers\" "
            "fits a similarity from markers with matching labels and estimated "
            "positions; method=\"reference\" places both chunks by their shared "
            "CRS. The applied 4x4 is recorded in both chunks' meta under "
            "AlignChunks/<other id>/matrix.")
        .def(
            "run_script",
            [](py::object self, const std::string& path, py::object args,
               int timeout_s) {
                py::dict kwargs;
                kwargs[py::str("path")] = py::str(path);
                if (!args.is_none()) {
                    py::list values;
                    for (py::handle item : args) {
                        values.append(py::str(item));
                    }
                    kwargs[py::str("args")] = values;
                }
                kwargs[py::str("timeout_s")] = py::int_(timeout_s);
                run_task_kwargs(self, "RunScript", kwargs);
            },
            py::arg("path"), py::arg("args") = py::none(),
            py::arg("timeout_s") = 600,
            "Run the Python script at `path` as a child process with "
            "--project <path> followed by `args`. The script reads the project "
            "FROM DISK: unsaved in-memory changes are invisible to it, so save "
            "the project first. Output is stored on the active chunk under "
            "RunScript/<n>/output (8 KB).");

    // -- module functions ----------------------------------------------------
    m.def("load_project",
          [](const std::string& path) { return alphashape::ProjectIO::load(path); },
          py::arg("path"));

    m.def("list_tasks", []() {
        alphashape::TaskRegistry& registry = alphashape::builtin_registry();
        return registry.names();
    });

    m.def("make_marker_sheet",
          [](const std::string& path, const std::string& target,
             const std::string& paper, double size_mm, int first_code,
             int count, double dpi) {
              const std::optional<alphashape::SheetTarget> parsed =
                  alphashape::parse_sheet_target(target);
              if (!parsed.has_value()) {
                  throw alphashape::ValueError(
                      "make_marker_sheet: unknown target '" + target + "'");
              }
              alphashape::SheetOptions options;
              options.target = *parsed;
              options.paper = paper;
              options.target_size_mm = size_mm;
              options.first_code = first_code;
              options.count = count;
              const bool png =
                  path.size() >= 4U && path[path.size() - 4U] == '.' &&
                  (path[path.size() - 3U] == 'p' ||
                   path[path.size() - 3U] == 'P') &&
                  (path[path.size() - 2U] == 'n' ||
                   path[path.size() - 2U] == 'N') &&
                  (path[path.size() - 1U] == 'g' ||
                   path[path.size() - 1U] == 'G');
              if (png) {
                  const int pages =
                      alphashape::marker_sheet_page_count(options);
                  if (pages > 1) {
                      throw alphashape::ValueError(
                          "PNG output holds a single page, but this sheet "
                          "needs " +
                          std::to_string(pages) +
                          " pages; write a PDF or reduce the count");
                  }
                  alphashape::write_png(
                      path, alphashape::marker_sheet_raster(options, dpi, 0));
                  return;
              }
              const std::vector<std::uint8_t> data =
                  alphashape::marker_sheet_pdf(options);
              std::ofstream file(path,
                                 std::ios::binary | std::ios::trunc);
              if (!file) {
                  throw alphashape::Error("make_marker_sheet: cannot open '" +
                                          path + "'");
              }
              file.write(reinterpret_cast<const char*>(data.data()),
                         static_cast<std::streamsize>(data.size()));
              file.close();
              if (!file) {
                  throw alphashape::Error("make_marker_sheet: writing '" +
                                          path + "' failed");
              }
          },
          py::arg("path"), py::arg("target"), py::arg("paper") = "A4",
          py::arg("size_mm") = 25.0, py::arg("first_code") = 1,
          py::arg("count") = 0, py::arg("dpi") = 300.0);

    m.def("make_chessboard_sheet",
          [](const std::string& path, int cols, int rows, double square_mm,
             const std::string& paper, double dpi) {
              alphashape::SheetOptions options;
              options.target = alphashape::SheetTarget::Chessboard;
              options.paper = paper;
              options.board_cols = cols;
              options.board_rows = rows;
              options.square_mm = square_mm;
              const bool png =
                  path.size() >= 4U && path[path.size() - 4U] == '.' &&
                  (path[path.size() - 3U] == 'p' ||
                   path[path.size() - 3U] == 'P') &&
                  (path[path.size() - 2U] == 'n' ||
                   path[path.size() - 2U] == 'N') &&
                  (path[path.size() - 1U] == 'g' ||
                   path[path.size() - 1U] == 'G');
              if (png) {
                  const int pages =
                      alphashape::marker_sheet_page_count(options);
                  if (pages > 1) {
                      throw alphashape::ValueError(
                          "PNG output holds a single page, but this sheet "
                          "needs " +
                          std::to_string(pages) +
                          " pages; write a PDF or reduce the count");
                  }
                  alphashape::write_png(
                      path, alphashape::marker_sheet_raster(options, dpi, 0));
                  return;
              }
              const std::vector<std::uint8_t> data =
                  alphashape::marker_sheet_pdf(options);
              std::ofstream file(path,
                                 std::ios::binary | std::ios::trunc);
              if (!file) {
                  throw alphashape::Error(
                      "make_chessboard_sheet: cannot open '" + path + "'");
              }
              file.write(reinterpret_cast<const char*>(data.data()),
                         static_cast<std::streamsize>(data.size()));
              file.close();
              if (!file) {
                  throw alphashape::Error(
                      "make_chessboard_sheet: writing '" + path + "' failed");
              }
          },
          py::arg("path"), py::arg("cols") = 9, py::arg("rows") = 6,
          py::arg("square_mm") = 25.0, py::arg("paper") = "A4",
          py::arg("dpi") = 300.0);

    m.def("task_params",
          [](const std::string& name) {
              const alphashape::TaskEntry& entry = alphashape::builtin_registry().require(name);
              py::dict out;
              for (const alphashape::ParamSpec& spec : entry.schema.specs()) {
                  // (type_str, default_str): the CLI's --list-params columns.
                  out[py::str(spec.name)] =
                      py::make_tuple(py::str(type_name(spec.type)),
                                     py::str(alphashape::format_param_value(spec.default_value)));
              }
              return out;
          },
          py::arg("name"));
}
