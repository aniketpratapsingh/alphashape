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
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "base/utils/error.hpp"
#include "common/algorithm/reference/convert_reference.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/project_io.hpp"
#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"
#include "model/camera/calibration.hpp"
#include "model/crs/coordinate_system.hpp"

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

/// Run a task by name with keyword arguments bound against its declared schema.
/// Shared by Document.run_task and the Chunk convenience methods (buildDem,
/// buildOrthomosaic, ...) so the two cannot drift.
void run_task_kwargs(alphashape::Document& d, const std::string& name, py::kwargs kwargs) {
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
    py::gil_scoped_release release;
    alphashape::run_task(registry, d, name, params, alphashape::Progress{});
}

/// Convert a C++ chunk to the Python object the Document bindings hand out,
/// attaching the owning Document as `_document`. Chunk has dynamic_attr(), so
/// the attribute lives in the object's __dict__; pybind11's GC traverses it, so
/// the Document <-> Chunk cycle stays collectable.
py::object attach_document(py::object document, const std::shared_ptr<alphashape::Chunk>& chunk);

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
    alphashape::Document& document = self.attr("_document").cast<alphashape::Document&>();
    const alphashape::Chunk& chunk = self.cast<const alphashape::Chunk&>();
    document.set_active_chunk_id(chunk.id());
    run_task_kwargs(document, name, std::move(kwargs));
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

/// One entry per asset of one kind: {key: id, label: str, active: bool}. The
/// `key` name and the id semantics follow Metashape's asset lists; the
/// payload itself is reached through the scalar properties, which expose the
/// active asset, so the lists stay cheap however large the rasters are.
template <class Slot>
py::list asset_list(const std::vector<Slot>& slots, std::int32_t active_id) {
    py::list out;
    for (const Slot& slot : slots) {
        py::dict entry;
        entry[py::str("key")] = py::int_(slot.id);
        entry[py::str("label")] = py::str(slot.label);
        entry[py::str("active")] = py::bool_(slot.id == active_id);
        out.append(std::move(entry));
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
        .def("__repr__", [](const alphashape::Calibration& c) {
            return "<alphashape.Calibration " + std::string(camera_type_name(c.type)) + " " +
                   std::to_string(c.width) + "x" + std::to_string(c.height) + " f=" +
                   std::to_string(c.f) + ">";
        });

    py::class_<alphashape::Sensor>(m, "Sensor")
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
                               [](const alphashape::Sensor& s) { return s.fixed_params; });

    py::class_<alphashape::Camera>(m, "Camera")
        .def_readonly("id", &alphashape::Camera::id)
        .def_readonly("sensor_id", &alphashape::Camera::sensor_id)
        .def_property_readonly("label", [](const alphashape::Camera& c) { return c.label; })
        .def_readonly("orientation", &alphashape::Camera::orientation)
        .def_readonly("enabled", &alphashape::Camera::enabled)
        // The 4x4 world->camera transform, as a list of four rows of four
        // floats. Identity on a camera that has not been aligned.
        .def_property_readonly("pose", [](const alphashape::Camera& c) {
            py::list rows;
            for (int r = 0; r < 4; ++r) {
                py::list row;
                for (int col = 0; col < 4; ++col) {
                    row.append(py::float_(c.pose(r, col)));
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
        .def_property_readonly("center", [](const alphashape::Camera& c) {
            const Eigen::Matrix3d A = c.pose.block<3, 3>(0, 0);
            const Eigen::Vector3d b = c.pose.block<3, 1>(0, 3);
            const Eigen::Vector3d centre = -(A.inverse() * b);
            return py::make_tuple(centre.x(), centre.y(), centre.z());
        })
        .def_property_readonly("reference", [](const alphashape::Camera& c) -> py::object {
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
        .def_property_readonly("reference_accuracy", [](const alphashape::Camera& c) -> py::object {
            if (!c.reference.location.has_value()) {
                return py::none();
            }
            const Eigen::Vector3d& a = c.reference.accuracy;
            return py::make_tuple(a.x(), a.y(), a.z());
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
        .def_property("label", &Chunk::label, &Chunk::set_label)
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
        .def_property_readonly("sensors", [](const Chunk& c) {
            // Copies, so the list cannot dangle when the chunk grows.
            py::list out;
            for (const alphashape::Sensor& sensor : c.sensors()) {
                out.append(sensor);
            }
            return out;
        })
        .def_property_readonly("cameras", [](const Chunk& c) {
            py::list out;
            for (const alphashape::Camera& camera : c.cameras()) {
                out.append(camera);
            }
            return out;
        })
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
        // declare that. A non-zero code also writes a minimal WKT descriptor so
        // `is_set()` and CRS equality behave like a project-loaded CRS.
        .def_property(
            "crs_epsg", [](const Chunk& c) { return c.crs().epsg; },
            [](Chunk& c, int epsg) {
                if (epsg < 0) {
                    throw alphashape::ValueError("crs_epsg must not be negative");
                }
                c.crs().epsg = epsg;
                c.crs().wkt =
                    epsg == 0 ? std::string{} : "EPSG:" + std::to_string(epsg);
            })
        // The effective camera-reference CRS (WP-12.6/WP-12.7): the explicit
        // slot when set, else the chunk CRS. 0 = unset.
        .def_property_readonly("camera_crs_epsg",
                               [](const Chunk& c) { return c.camera_crs().epsg; })
        .def_property_readonly("marker_crs_epsg",
                               [](const Chunk& c) { return c.marker_crs().epsg; })
        // Markers (M11 §4.1 / WP-11.7). Each entry is a dict with `id`, `label`,
        // `position` (a chunk-local (x, y, z) tuple, or None while unestimated)
        // and `reference` (a (lon, lat, alt) tuple, or None). AlignChunks pairs
        // markers by exact `label`, so two chunks need matching labels, not
        // matching ids.
        .def_property_readonly("markers", [](const Chunk& c) {
            py::list out;
            for (const alphashape::Marker& marker : c.markers()) {
                py::dict entry;
                entry[py::str("id")] = py::int_(marker.id);
                entry[py::str("label")] = py::str(marker.label);
                if (marker.position.has_value()) {
                    const Eigen::Vector3d& p = *marker.position;
                    entry[py::str("position")] = py::make_tuple(p.x(), p.y(), p.z());
                } else {
                    entry[py::str("position")] = py::none();
                }
                if (marker.reference.has_value()) {
                    const Eigen::Vector3d& r = *marker.reference;
                    entry[py::str("reference")] = py::make_tuple(r.x(), r.y(), r.z());
                } else {
                    entry[py::str("reference")] = py::none();
                }
                out.append(std::move(entry));
            }
            return out;
        })
        // Place or move a marker: sets `position` on the marker with `label`,
        // creating the marker (and allocating its id) when the chunk has none
        // with that label. This is the manual placement path WP-11.1 ships
        // (detection is M13); AlignChunks is its first consumer.
        .def(
            "set_marker_position",
            [](Chunk& c, const std::string& label, double x, double y, double z) {
                if (label.empty()) {
                    throw alphashape::ValueError("set_marker_position: empty label");
                }
                const Eigen::Vector3d position(x, y, z);
                if (!position.allFinite()) {
                    throw alphashape::ValueError(
                        "set_marker_position: position must be finite");
                }
                for (alphashape::Marker& marker : c.markers()) {
                    if (marker.label == label) {
                        marker.position = position;
                        marker.valid = true;
                        return marker.id;
                    }
                }
                alphashape::Marker marker;
                marker.id = c.allocate_id();
                marker.label = label;
                marker.position = position;
                marker.valid = true;
                const std::int32_t id = marker.id;
                c.markers().push_back(std::move(marker));
                return id;
            },
            py::arg("label"), py::arg("x"), py::arg("y"), py::arg("z"),
            "Create or move the marker with this exact label to (x, y, z) in "
            "chunk-local coordinates. Returns its id.")
        // Set (or replace) one chunk-meta property.
        .def(
            "set_meta",
            [](Chunk& c, const std::string& key, const std::string& value) {
                if (key.empty()) {
                    throw alphashape::ValueError("set_meta: empty key");
                }
                c.meta().set(key, value);
            },
            py::arg("key"), py::arg("value"))
        // -- dense products (milestone 6) --------------------------------------
        // BuildDepthMaps fills `depth_maps`; BuildPointCloud fuses them into
        // `point_cloud`. Both are read-only views onto the chunk's containers.
        //
        // A depth value is the distance from the camera centre along the pixel's
        // unit bearing in chunk-local units, NOT a camera-frame Z. -1 is the
        // sentinel for "no depth"; depths is always present, while normals and
        // confidence are None when the map carries none (their vectors are empty).
        .def_property_readonly("depth_maps", [](const Chunk& c) {
            py::list out;
            for (const alphashape::DepthMap& map : c.depth_maps()) {
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
                out.append(std::move(entry));
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
        // One entry per asset of each kind, in id order: {key, label, active}.
        // The scalar properties above (elevation, orthomosaic, model,
        // point_cloud, depth_maps, model_texture) are views onto the ACTIVE
        // asset, which is the one the last producing task added.
        .def_property_readonly("elevations", [](const Chunk& c) {
            return asset_list(c.elevations(), c.active_elevation_id());
        })
        .def_property_readonly("orthomosaics", [](const Chunk& c) {
            return asset_list(c.orthomosaics(), c.active_orthomosaic_id());
        })
        .def_property_readonly("models", [](const Chunk& c) {
            return asset_list(c.models(), c.active_model_id());
        })
        .def_property_readonly("point_clouds", [](const Chunk& c) {
            return asset_list(c.point_clouds(), c.active_point_cloud_id());
        })
        .def_property_readonly("depth_maps_sets", [](const Chunk& c) {
            return asset_list(c.depth_map_sets(), c.active_depth_maps_id());
        })
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
#if defined(ALPHASHAPE_USE_PROJ)
        // Convert the camera/marker reference positions into another CRS
        // (WP-12.7). `target` is an EPSG code; `kind` is "cameras", "markers"
        // or "both". The slots are relabelled, poses and assets are untouched.
        // No geoid store is reachable from a bare Chunk, so a vertical datum
        // that is not an EPSG vertical CRS is refused by name (the disks this
        // front end exposes never carry one).
        .def(
            "convertReference",
            [](Chunk& c, int target_epsg, const std::string& kind) {
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
            },
            py::arg("target_epsg"), py::arg("kind") = "both");
#endif  // ALPHASHAPE_USE_PROJ


    // -- Document ------------------------------------------------------------
    py::class_<Document, std::shared_ptr<Document>>(m, "Document")
        .def(py::init<>())
        .def_property("path", [](const Document& d) { return d.path(); },
                      [](Document& d, const std::string& path) { d.set_path(path); })
        .def(
            "save",
            [](Document& d, const std::string& path, py::object chunks) {
                // Record the path so a later run resolves relative frame paths
                // against the project we just wrote, exactly like load() would.
                d.set_path(path);
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
                d.save(path, ids);
            },
            py::arg("path"), py::arg("chunks") = py::none(),
            "Write the project. `chunks` restricts the file to those chunks "
            "(ids or Chunk objects); None writes every chunk. Chunk ids and the "
            "next-chunk counter are preserved so the file can be appended.")
        .def("add_chunk",
             [](py::object self, const std::string& label) -> py::object {
                 Document& d = self.cast<Document&>();
                 d.workspace().add_chunk(label);
                 return attach_document(self, d.workspace().chunks().back());
             },
             py::arg("label"))
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
             [](Document& d, const std::string& name, py::kwargs kwargs) {
                 run_task_kwargs(d, name, std::move(kwargs));
             },
             py::arg("name"))
        .def(
            "merge_chunks",
            [](py::object self, py::object chunks, const std::string& label) -> py::object {
                Document& d = self.cast<Document&>();
                std::vector<std::string> ids;
                for (py::handle item : chunks) {
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
                run_task_kwargs(d, "MergeChunks", kwargs);
                return attach_document(self, d.workspace().chunks().back());
            },
            py::arg("chunks"),
            py::arg("label") = "Merged chunk",
            "Concatenate the given chunks (ids or Chunk objects) into one new "
            "chunk and return it. Refuses chunks with different non-local CRSs.")
        .def(
            "append_project",
            [](Document& d, const std::string& path, py::object chunks,
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
                run_task_kwargs(d, "AppendProject", kwargs);
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
                run_task_kwargs(d, "DuplicateChunk", kwargs);
                return attach_document(self, d.workspace().chunks().back());
            },
            py::arg("chunk") = py::none(), py::arg("label") = "",
            "Deep-copy a chunk with every chunk-local id remapped. `chunk` is a "
            "Chunk, an id, or None for the active chunk. Returns the copy.")
        .def(
            "align_chunks",
            [](Document& d, py::object reference, const std::string& method,
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
                run_task_kwargs(d, "AlignChunks", kwargs);
            },
            py::arg("reference") = py::none(),
            py::arg("method") = "markers", py::arg("min_common_markers") = 3,
            py::arg("accuracy_m") = 0.0,
            "Align the ACTIVE chunk onto `reference` (a Chunk, an id, or None "
            "for the first other chunk in the workspace). method=\"markers\" "
            "fits a similarity from markers with matching labels and estimated "
            "positions; method=\"reference\" places both chunks by their shared "
            "CRS. The applied 4x4 is recorded in both chunks' meta under "
            "AlignChunks/<other id>/matrix.");

    // -- module functions ----------------------------------------------------
    m.def("load_project",
          [](const std::string& path) { return alphashape::ProjectIO::load(path); },
          py::arg("path"));

    m.def("list_tasks", []() {
        alphashape::TaskRegistry& registry = alphashape::builtin_registry();
        return registry.names();
    });

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
