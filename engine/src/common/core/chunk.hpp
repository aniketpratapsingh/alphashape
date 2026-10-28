#pragma once

// Chunk: the unit that owns every per-stage container.
//
// The container set is the `chunk` subtree of doc.xml, cross-checked against
// the public Python Chunk property list. v0 fills cameras,
// sensors, frames and tie points; the remaining containers are declared so the
// serialised schema is complete from the first release and we never have to
// migrate a project format.

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/core/assets.hpp"
#include "common/core/groups.hpp"
#include "common/core/marker.hpp"
#include "common/core/mask.hpp"
#include "common/core/scalebar.hpp"
#include "common/core/shapes.hpp"
#include "common/core/trajectory.hpp"
#include "common/core/types.hpp"
#include "image/image.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/point_cloud.hpp"
#include "model/raster/geo_raster.hpp"

namespace alphashape {

class Frame;

/// The image-record container. Holds the keypoint sets produced by detection.
/// Named after the `tie_points` container, whose `point_cloud.zip`
/// payload (.kpt/.ids/tracks.ply) is its serialised form.
struct TiePoints {
    /// Keypoints and descriptors per camera.
    std::vector<KeypointSet> keypoint_sets;

    /// Verified pairwise matches (MatchPhotos), sorted by (camera_a, camera_b).
    std::vector<PairMatches> matches;

    /// Tracks built from `matches` by transitive closure (MatchPhotos).
    std::vector<Track> tracks;

    KeypointSet* find_by_camera(std::int32_t camera_id);
    const KeypointSet* find_by_camera(std::int32_t camera_id) const;

    std::size_t total_keypoints() const;

    bool operator==(const TiePoints& other) const {
        return keypoint_sets == other.keypoint_sets && matches == other.matches &&
               tracks == other.tracks;
    }
};

/// Depth map container: one depth estimate per pixel of a (possibly
/// downscaled) camera image, produced by the dense-stereo stage (milestone 6).
///
/// DEPTH SEMANTICS: a depth value is the distance from the camera centre along
/// the pixel's UNIT bearing, in chunk-local units — NOT the camera-frame Z
/// coordinate. `depths[i]` is the depth at depth-map pixel i = (u', v') in
/// row-major order; the full-resolution pixel is (u' * downscale, v' * downscale)
/// and the ray through it is `Calibration::unproject(full_pixel)`. Reconstruct
/// the chunk-local point the same way for frame and fisheye sensors:
///
///     ray  = calibration.unproject({u' * downscale + 0.5, v' * downscale + 0.5})
///     p_cam = depth * ray                       // camera frame
///     p_chunk = R^T * (p_cam - t)               // R, t from the camera pose
///
/// Recovering X_cam = depth * ray (rather than reading a stored Z) is what
/// makes the container valid for fisheye cameras, where a "Z" has no single
/// meaning across the field of view.
struct DepthMap {
    std::int32_t camera_id = -1;   ///< the camera this map belongs to
    int width = 0;                 ///< depth-map resolution, = full image / downscale
    int height = 0;
    int downscale = 1;             ///< power of two; full-res u = depth-map u' * downscale

    /// width * height, row-major. -1.0f is the sentinel for "no depth".
    std::vector<float> depths;
    /// Empty, or 3 * width * height: unit normals in the camera frame, each
    /// facing the camera (oriented so the surface normal points toward it).
    std::vector<float> normals;
    /// Empty, or width * height: per-pixel confidence / support weight.
    std::vector<float> confidence;

    /// The cross-view filter this map has been through: -1 when it has not been
    /// filtered at all (a map checkpointed by a cancelled run, or one that
    /// never reached the filter stage), otherwise the FilterMode's enumerator
    /// (0 disabled, 1 mild, 2 moderate, 3 aggressive). A map is reused as-is
    /// only when this already equals the mode the run asked for; anything else
    /// is filtered again, because the filter is not idempotent (each repetition
    /// erodes support at the boundaries) and because a run that asked for
    /// "aggressive" must not silently return "mild" maps.
    int filter_mode = -1;

    /// Hash of everything this map was derived from: the reference camera's
    /// pose and the sensor calibration, the downscale, AND the whole
    /// neighbourhood (neighbour ids, their poses and calibrations, the depth
    /// range, the reference image file and the PatchMatch settings). See
    /// depth_map_fingerprint(). A later BuildDepthMaps run reuses a map only
    /// when this matches the chunk's current geometry, so re-aligning ANY
    /// camera invalidates every map that used it as a neighbour, and a
    /// re-calibration invalidates every map on that sensor. 0 means "unknown"
    /// — written by a build that did not record one — and such a map is never
    /// reused.
    std::uint64_t alignment_fingerprint = 0;

    bool operator==(const DepthMap& other) const {
        return camera_id == other.camera_id && width == other.width &&
               height == other.height && downscale == other.downscale &&
               depths == other.depths && normals == other.normals &&
               confidence == other.confidence && filter_mode == other.filter_mode &&
               alignment_fingerprint == other.alignment_fingerprint;
    }
};

/// Deterministic hash of everything a depth map depends on besides the image
/// pixels: the camera's world->camera pose, the sensor calibration and the
/// working downscale; and, through the optional `context`, the whole
/// NEIGHBOURHOOD the map was estimated from — the neighbour camera ids (sorted),
/// each neighbour's pose and calibration, the depth range, the reference image
/// the pixels came from and the PatchMatch settings that shape the result.
///
/// A map is a function of its reference camera AND its neighbourhood: moving a
/// camera changes every map that used it as a neighbour, not only its own. Two
/// runs that would produce the same map have the same fingerprint; a
/// re-aligned camera, a re-calibrated sensor, a different neighbour set or a
/// different search configuration does not. Never returns 0, which is reserved
/// for "no fingerprint recorded".
///
/// `context` is null for a bare "camera identity" hash (the older, narrower
/// fingerprint); TaskDepthMapFingerprint in build_depth_maps_task.hpp is what
/// the task actually stamps, because only the task knows the neighbourhood.
struct DepthMapFingerprintContext {
    int max_neighbors = 0;
    /// Neighbour camera ids, sorted ascending. Order matters to the caller but
    /// NOT to the hash: it sorts its own copy so a select_views tie ordering
    /// cannot masquerade as a geometric change.
    std::vector<std::int32_t> neighbour_ids;
    /// Sorted parallel data for `neighbour_ids`: pose (16 doubles row-major) and
    /// the sensor calibration. Same order, same size.
    std::vector<Transform> neighbour_poses;
    std::vector<Calibration> neighbour_calibrations;
    double depth_min = 0.0;
    double depth_max = 0.0;
    /// Resolved reference-image path (empty when unknown), with size and mtime
    /// when cheap: 0 means "not stat-able" and is folded as 0, consistently.
    std::string image_path;
    std::uint64_t image_size = 0;
    std::int64_t image_mtime = 0;
    int window_radius = 0;
    int window_step = 0;
    int iterations = 0;
    /// Coarse-to-fine: the number of levels the map's search actually ran at,
    /// resolved from the request and the working image size (1 = no pyramid).
    /// Folded only when it exceeds 1, so a map built before the pyramid existed
    /// -- which recorded nothing here (0), i.e. the flat search -- is still
    /// reused by a run that resolves to a single level.
    int pyramid_levels = 1;
    /// Photometric cost the search minimised (stereo::CostType as an int: 0 =
    /// zncc, 1 = census). Folded only when non-zero, so a map built before the
    /// cost became selectable -- which ran the only cost there was, zncc -- is
    /// still reused by a zncc run.
    int cost_type = 0;
    std::uint64_t seed = 0;
};

std::uint64_t depth_map_fingerprint(const Camera& camera, const Sensor& sensor, int downscale,
                                    const DepthMapFingerprintContext* context = nullptr);

/// Fold one more value into a fingerprint built by depth_map_fingerprint(): the
/// same splitmix64 mixing the fingerprint itself uses, so a caller that knows
/// extra inputs (the task's neighbourhood) can extend it portably and
/// reproducibly across platforms. Exposed for that one use; not a general hash.
std::uint64_t depth_map_fingerprint_fold_double(std::uint64_t h, double value);
std::uint64_t depth_map_fingerprint_fold_int(std::uint64_t h, std::int64_t value);
std::uint64_t depth_map_fingerprint_fold_string(std::uint64_t h, const std::string& text);
/// The seed depth_map_fingerprint() starts from, for a caller that builds one
/// itself (the task folds in a chunk's camera geometry before the per-map part).
std::uint64_t depth_map_fingerprint_seed();

// PointCloud and Mesh are declared in `alphashape_model` (model/mesh/) and
// included above, so the meshing library can consume geometry without
// depending on common/. Their container slots on Chunk are below.

/// Georeferenced elevation model (DEM), milestone 9. The raster's heights are in
/// the plane frame's height unit (ellipsoidal metres for a georeferenced chunk,
/// chunk-local units when unreferenced); `epsg` records which SurfaceFrame
/// projection that grid is expressed in (0 = chunk-local / no geographic
/// meaning).
///
/// Persisted in the project file as a float GeoTIFF side file ("elevation/
/// dem.tif") plus an <elevation> element in doc.xml; exposed to Python as
/// Chunk.elevation (see python/bindings/module.cpp).
struct Elevation {
    GeoRaster<float> raster;   ///< heights; single channel
    int epsg = 0;              ///< SurfaceFrame epsg the grid is in (0 = local)
    float nodata = -32767.0F;  ///< sentinel written for unmeasured cells
    std::string source;        ///< "point_cloud" | "model" | "depth_maps"

    bool empty() const { return raster.empty(); }

    bool operator==(const Elevation&) const = default;
};

/// Orthomosaic (true orthophoto), milestone 9. The raster holds four channels
/// (RGBA); alpha 255 marks a cell that at least one camera observed, alpha 0 is
/// nodata. `labels` records, per cell, the id of the source camera in
/// `Chunk::cameras()` (the seamline labelling), or -1 where no camera covers the
/// cell. `epsg` is the SurfaceFrame projection of the grid the same way
/// `Elevation::epsg` is.
///
/// Persisted in the project file as an RGBA u8 GeoTIFF side file
/// ("orthomosaic/ortho.tif") plus an "AOL1" labels sidecar and an
/// <orthomosaic> element in doc.xml; exposed to Python as Chunk.orthomosaic.
struct Orthomosaic {
    GeoRaster<std::uint8_t> raster;   ///< RGBA, 4 channels; alpha 0 = nodata
    int epsg = 0;                     ///< SurfaceFrame epsg the grid is in (0 = local)
    std::vector<std::int32_t> labels; ///< per-cell source camera id, -1 = none
    std::string surface;              ///< "elevation" | "model"

    bool empty() const { return raster.empty(); }

    bool operator==(const Orthomosaic&) const = default;
};

/// A chunk: cameras, their sensors, the input frames, and every derived product.
class Chunk {
public:
    Chunk() = default;
    explicit Chunk(std::string label);

    std::int32_t id() const { return id_; }
    void set_id(std::int32_t id) { id_ = id; }

    const std::string& label() const { return label_; }
    void set_label(std::string label) { label_ = std::move(label); }

    // -- chunk-local object ids -------------------------------------------
    /// The id the next `allocate_id()` call will return. Every object in a
    /// chunk (marker, scalebar, group, shape, trajectory, asset, ...) draws
    /// from this single counter, so ids are unique across all containers.
    std::int32_t next_id() const { return next_id_; }

    /// Allocate and return a fresh chunk-local id. Ids are never reused, even
    /// after the object is removed.
    std::int32_t allocate_id() { return next_id_++; }

    /// Make the counter at least `id + 1`. Used when restoring a project so
    /// freshly allocated ids never collide with restored ones.
    void reserve_id(std::int32_t id) {
        if (id >= next_id_) {
            next_id_ = id + 1;
        }
    }

    /// Restore the counter verbatim. Project load uses this so a chunk whose
    /// ids are not contiguous keeps its exact allocator state; prefer
    /// `reserve_id` everywhere else.
    void set_next_id(std::int32_t id) { next_id_ = id; }

    // -- containers (populated in v0) --------------------------------------
    std::vector<Sensor>& sensors() { return sensors_; }
    const std::vector<Sensor>& sensors() const { return sensors_; }

    std::vector<Camera>& cameras() { return cameras_; }
    const std::vector<Camera>& cameras() const { return cameras_; }

    std::vector<std::shared_ptr<Frame>>& frames() { return frames_; }
    const std::vector<std::shared_ptr<Frame>>& frames() const { return frames_; }

    TiePoints& tie_points() { return tie_points_; }
    const TiePoints& tie_points() const { return tie_points_; }

    // -- multi-instance asset containers -----------------------------------
    // Every kind is kept sorted by id with unique ids drawn from allocate_id().
    // Exactly one asset per kind is "active": the one the legacy single-slot
    // accessors below return. The active id is -1 iff the vector is empty.

    std::vector<PointCloudAsset>& point_clouds() { return point_clouds_; }
    const std::vector<PointCloudAsset>& point_clouds() const { return point_clouds_; }
    std::int32_t active_point_cloud_id() const { return active_point_cloud_id_; }
    void set_active_point_cloud_id(std::int32_t id) { active_point_cloud_id_ = id; }

    std::vector<ModelAsset>& models() { return models_; }
    const std::vector<ModelAsset>& models() const { return models_; }
    std::int32_t active_model_id() const { return active_model_id_; }
    void set_active_model_id(std::int32_t id) { active_model_id_ = id; }

    std::vector<DepthMapsAsset>& depth_map_sets() { return depth_map_sets_; }
    const std::vector<DepthMapsAsset>& depth_map_sets() const {
        return depth_map_sets_;
    }
    std::int32_t active_depth_maps_id() const { return active_depth_maps_id_; }
    void set_active_depth_maps_id(std::int32_t id) { active_depth_maps_id_ = id; }

    std::vector<ElevationAsset>& elevations() { return elevations_; }
    const std::vector<ElevationAsset>& elevations() const { return elevations_; }
    std::int32_t active_elevation_id() const { return active_elevation_id_; }
    void set_active_elevation_id(std::int32_t id) { active_elevation_id_ = id; }

    std::vector<OrthomosaicAsset>& orthomosaics() { return orthomosaics_; }
    const std::vector<OrthomosaicAsset>& orthomosaics() const {
        return orthomosaics_;
    }
    std::int32_t active_orthomosaic_id() const { return active_orthomosaic_id_; }
    void set_active_orthomosaic_id(std::int32_t id) {
        active_orthomosaic_id_ = id;
    }

    /// Append a new asset of this kind (allocating a fresh id from the shared
    /// counter), make it active, and return it. An empty `label` means
    /// "<Kind> <n>" with the smallest unused n, e.g. "Elevation 1".
    PointCloudAsset& add_point_cloud(std::string label = "");
    ModelAsset& add_model(std::string label = "");
    DepthMapsAsset& add_depth_maps(std::string label = "");
    ElevationAsset& add_elevation(std::string label = "");
    OrthomosaicAsset& add_orthomosaic(std::string label = "");

    // -- legacy single-slot accessors (return the ACTIVE asset) ------------
    // Signatures are unchanged so existing call sites keep compiling. When the
    // container is empty the const overload returns a shared empty singleton;
    // the non-const overload lazily creates a first asset (allocating an id and
    // making it active), so `chunk.point_cloud() = ...` still works. A slot whose
    // payload pointer was cleared (validate() reports it) reads as empty through
    // the const overload and gets a fresh empty payload through the non-const one.

    std::vector<DepthMap>& depth_maps();
    const std::vector<DepthMap>& depth_maps() const;

    PointCloud& point_cloud();
    const PointCloud& point_cloud() const;

    Mesh& model();
    const Mesh& model() const;

    /// Texture pages for the ACTIVE model (milestone 8). Page 0 only for now:
    /// the mesh's uv_faces index the single page baked by the texturing stage,
    /// so every UV in [0,1]^2 addresses model_textures()[0]. Declared as a
    /// vector so multi-page atlases need no format migration later.
    std::vector<Image>& model_textures();
    const std::vector<Image>& model_textures() const;

    Elevation& elevation();
    const Elevation& elevation() const;

    Orthomosaic& orthomosaic();
    const Orthomosaic& orthomosaic() const;

    // -- M11 multi-instance containers (in memory; persisted in WP-11.3) ----
    // Each is kept sorted by id with unique ids; ids come from allocate_id().
    std::vector<Marker>& markers() { return markers_; }
    const std::vector<Marker>& markers() const { return markers_; }

    std::vector<MarkerGroup>& marker_groups() { return marker_groups_; }
    const std::vector<MarkerGroup>& marker_groups() const { return marker_groups_; }

    std::vector<Scalebar>& scalebars() { return scalebars_; }
    const std::vector<Scalebar>& scalebars() const { return scalebars_; }

    std::vector<CameraGroup>& camera_groups() { return camera_groups_; }
    const std::vector<CameraGroup>& camera_groups() const { return camera_groups_; }

    /// One mask per camera; keyed by `Mask::camera_id`, sorted by it.
    std::vector<Mask>& masks() { return masks_; }
    const std::vector<Mask>& masks() const { return masks_; }

    Shapes& shapes() { return shapes_; }
    const Shapes& shapes() const { return shapes_; }

    std::vector<Trajectory>& trajectories() { return trajectories_; }
    const std::vector<Trajectory>& trajectories() const { return trajectories_; }

    /// The single panorama slot (360 imagery); nullopt until one is added.
    std::optional<PanoramaAsset>& panorama() { return panorama_; }
    const std::optional<PanoramaAsset>& panorama() const { return panorama_; }

    std::vector<TiledModelAsset>& tiled_models() { return tiled_models_; }
    const std::vector<TiledModelAsset>& tiled_models() const { return tiled_models_; }

    /// Id of the tiled model shown by default, or -1 when there is none.
    /// Invariant: -1 iff `tiled_models()` is empty; otherwise it names an
    /// existing asset.
    std::int32_t active_tiled_model_id() const { return active_tiled_model_id_; }
    void set_active_tiled_model_id(std::int32_t id) { active_tiled_model_id_ = id; }

    // -- validation ---------------------------------------------------------
    /// Structural invariants. Returns one human-readable message per
    /// violation; an empty vector means the chunk is valid. This is a
    /// read-only check: it never mutates or repairs the chunk.
    std::vector<std::string> validate() const;
    bool is_valid() const { return validate().empty(); }

    // -- document-level attributes ----------------------------------------
    Region& region() { return region_; }
    const Region& region() const { return region_; }

    Transform& transform() { return transform_; }
    const Transform& transform() const { return transform_; }

    Meta& meta() { return meta_; }
    const Meta& meta() const { return meta_; }

    /// The chunk's coordinate reference system, as the engine type.
    /// @throws ValueError when the stored DTO cannot be parsed (a corrupt
    /// project); the DTO accessors below never throw.
    crs::CoordinateSystem coordinate_system() const;
    /// Replace the CRS from the engine type. Does NOT transform any data:
    /// assignment only relabels the chunk (M12 §4.7).
    void set_coordinate_system(const crs::CoordinateSystem& cs);

    /// The project-format DTO (doc.xml `<crs epsg= wkt=/>`). Kept for the
    /// readers/writers and for code that only inspects the identifiers; new
    /// code should use coordinate_system(). Deprecated in M13.
    CRS& crs() { return crs_; }
    const CRS& crs() const { return crs_; }

    /// Structural comparison, used by the project round-trip test.
    bool operator==(const Chunk& other) const;

    /// Deep-value copy and assignment: AssetSlot::data is a shared_ptr, so a
    /// naive memberwise copy would share asset payloads between the two chunks
    /// and an in-place mutation of one would corrupt the other. The copy
    /// constructor and assignment operator therefore clone every asset payload.
    Chunk(const Chunk& other);
    Chunk& operator=(const Chunk& other);
    Chunk(Chunk&&) noexcept = default;
    Chunk& operator=(Chunk&&) noexcept = default;
    ~Chunk() = default;

private:
    void copy_from(const Chunk& other);

    std::int32_t id_ = -1;
    std::string label_;

    std::vector<Sensor> sensors_;
    std::vector<Camera> cameras_;
    std::vector<std::shared_ptr<Frame>> frames_;
    TiePoints tie_points_;

    std::vector<PointCloudAsset> point_clouds_;
    std::int32_t active_point_cloud_id_ = -1;
    std::vector<ModelAsset> models_;
    std::int32_t active_model_id_ = -1;
    std::vector<DepthMapsAsset> depth_map_sets_;
    std::int32_t active_depth_maps_id_ = -1;
    std::vector<ElevationAsset> elevations_;
    std::int32_t active_elevation_id_ = -1;
    std::vector<OrthomosaicAsset> orthomosaics_;
    std::int32_t active_orthomosaic_id_ = -1;

    std::int32_t next_id_ = 0;

    std::vector<Marker> markers_;
    std::vector<MarkerGroup> marker_groups_;
    std::vector<Scalebar> scalebars_;
    std::vector<CameraGroup> camera_groups_;
    std::vector<Mask> masks_;
    Shapes shapes_;
    std::vector<Trajectory> trajectories_;
    std::optional<PanoramaAsset> panorama_;
    std::vector<TiledModelAsset> tiled_models_;
    std::int32_t active_tiled_model_id_ = -1;

    Region region_;
    Transform transform_ = Transform::Identity();
    Meta meta_;
    CRS crs_;
};

}  // namespace alphashape
