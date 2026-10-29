#pragma once

/// @file geoid.hpp
/// @brief Geoid grids and undulation evaluation (M12 §4.5).
///
/// A geoid grid stores undulations N — geoid height above the reference
/// ellipsoid, metres — as a single-band GeoTIFF. `GeoidGrid` evaluates such a
/// file two ways:
///
///   * through PROJ (`+proj=vgridshift`), which is the same code `cs2cs
///     EPSG:4979 EPSG:5773` runs, so the engine's numbers match PROJ exactly;
///   * through a direct GeoTIFF read plus Catmull-Rom bicubic interpolation,
///     the fallback for a grid PROJ cannot open.
///
/// The two agree at the grid nodes (Catmull-Rom is interpolating, so a node
/// returns its stored sample and PROJ does the same). Between nodes PROJ
/// interpolates bilinearly; the bicubic differs by the grid's own curvature
/// over half a cell. That gap is a property of the reference data and shrinks
/// quadratically with cell size, so a test pins it on the small 1-degree
/// subset where it is measurable rather than assuming a millimetre.
///
/// Outside a grid's coverage every evaluator returns nullopt; nothing is
/// extrapolated, and no grid is ever downloaded (see proj_context.cpp).
///
/// PROJ resolves a grid by name from its data directories and cannot take a
/// path containing a space through `+grids=`, so a grid is evaluated on a
/// private PROJ context whose search path is the grid file's own directory and
/// the name passed to PROJ is the bare file name. The engine never mutates the
/// thread context's search path.

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "model/crs/coordinate_system.hpp"

namespace alphashape::crs {

/// Geometry of a geoid grid: the node bounding box and the node counts. The
/// bounding box is of the *nodes* (pixel-is-point), matching how PROJ reads the
/// NGA grids.
struct GeoidGridInfo {
    int width = 0;
    int height = 0;
    double min_lon_deg = 0.0;
    double min_lat_deg = 0.0;
    double max_lon_deg = 0.0;
    double max_lat_deg = 0.0;
};

class GeoidStore;

/// One opened geoid grid. Copies share the decoded samples; opening is the
/// expensive step (a full EGM2008 grid is 77 MB of float32).
class GeoidGrid {
public:
    /// Read `path`'s samples and tags.
    /// @throws ProjectIOError when the file is not a readable single-band
    ///         float32 GeoTIFF, or its tags do not describe a north-up grid.
    static std::shared_ptr<const GeoidGrid> open(const std::string& path);

    const GeoidGridInfo& info() const;
    const std::string& path() const;

    /// True when (lon, lat) lies inside the node bounding box.
    bool covers(double lon_deg, double lat_deg) const;

    /// Undulation N in metres at (lon, lat) as PROJ computes it. nullopt
    /// outside coverage and when PROJ cannot open this grid.
    std::optional<double> proj_undulation(double lon_deg, double lat_deg) const;

    /// Undulation N from the GeoTIFF with Catmull-Rom bicubic interpolation.
    /// nullopt outside coverage.
    std::optional<double> grid_undulation(double lon_deg, double lat_deg) const;

    /// `proj_undulation()`, falling back to `grid_undulation()` when PROJ
    /// cannot evaluate the grid. This is what callers should use.
    std::optional<double> undulation(double lon_deg, double lat_deg) const;

private:
    struct Impl;
    explicit GeoidGrid(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
};

/// One geoid definition in a `GeoidStore` (M12 §4.5). `path` is absolute for a
/// grid loaded from disk; a store persisted to `<project>.files/geoids/` stores
/// the bare file name next to its `index.xml` and resolves it on load.
struct GeoidModel {
    std::string id;       ///< "EGM2008" | "EGM96" | a custom grid's file stem
    std::string name;     ///< "EGM2008 height"
    std::string path;     ///< grid file
    std::string source;   ///< "proj-data" | "custom"
    std::string licence;  ///< e.g. "public domain (NGA)"
    /// Node bounding box of the grid (degrees), filled when the grid is added.
    double min_lon_deg = -180.0;
    double min_lat_deg = -90.0;
    double max_lon_deg = 180.0;
    double max_lat_deg = 90.0;

    bool operator==(const GeoidModel& other) const {
        return id == other.id && name == other.name && path == other.path &&
               source == other.source && licence == other.licence &&
               min_lon_deg == other.min_lon_deg && min_lat_deg == other.min_lat_deg &&
               max_lon_deg == other.max_lon_deg && max_lat_deg == other.max_lat_deg;
    }
};

/// The geoid collection of a project (M12 §4.5, §4.8).
///
/// A store holds ready-made entries for the models installed with PROJ
/// (`from_proj_data()` — EGM96 and EGM2008 when their grids are present) plus
/// any custom grid `add()` copies in. `save()` writes the grids and an
/// `index.xml` under `<project>.files/geoids/`; `load()` reads them back and
/// throws when a grid the index names is missing, naming it, so a project never
/// silently loses its vertical datum.
class GeoidStore {
public:
    GeoidStore();

    /// Entries for the geoid grids PROJ's data directory has installed. A model
    /// whose grid is absent is simply not listed; nothing is downloaded.
    static GeoidStore from_proj_data();

    /// Add `grid_path` under `id`, copying the file into the store's own
    /// directory when one is set. Coverage comes from the grid's tags.
    /// @throws Error when `id` is empty or already present.
    bool add(const std::string& id, const std::string& name, const std::string& grid_path,
             const std::string& source, const std::string& licence);

    std::optional<GeoidModel> find(const std::string& id) const;
    std::vector<GeoidModel> list() const;

    /// Undulation N in metres, or nullopt when the id is unknown or the point
    /// is outside the grid.
    std::optional<double> undulation(const std::string& id, double lon_deg,
                                     double lat_deg) const;
    bool covers(const std::string& id, double lon_deg, double lat_deg) const;

    /// Directory `save()` writes to and `load()` reads; empty until either is
    /// called or `set_directory()` is used.
    const std::filesystem::path& directory() const;
    void set_directory(std::filesystem::path dir);

    /// Write every grid plus `index.xml` into `dir` (which becomes
    /// `directory()`). `dir` is normally `<project>.files/geoids/`.
    void save(const std::filesystem::path& dir) const;

    /// Read the store written by `save()`.
    /// @throws Error when the index is missing, unreadable, or names a grid file
    ///         that is not in `dir`.
    static GeoidStore load(const std::filesystem::path& dir);

private:
    struct Entry {
        GeoidModel model;
        /// Opened on first use, so `load()` stays cheap for a project with
        /// several large grids.
        mutable std::shared_ptr<const GeoidGrid> grid;
    };
    std::vector<Entry> entries_;
    std::filesystem::path directory_;

    /// The entry's grid, opened on first use; null when the file cannot be
    /// opened (the caller reports that as "not covered").
    std::shared_ptr<const GeoidGrid> grid_for(const Entry& entry) const;
};

}  // namespace alphashape::crs
