// doc.xml format v1: the pre-M11 layout, read verbatim.
//
// v1 stores exactly one asset per kind (the active one) and has no container
// for markers, scale bars, camera groups, masks, shapes, trajectories,
// panoramas or tiled models. Its sidecar entry names are flat
// ("tie_points/...", "depth_maps/...", ...) rather than the v2 per-chunk paths.
//
// This reader exists so every project written before M11 keeps loading exactly
// as it did. It is not a writer: anything saved from now on is v2. The sidecar
// *decoders* are shared with v2 (they are version independent, see
// sidecars.hpp); only the document structure and the entry names differ.

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/io/zip.hpp"
#include "base/utils/error.hpp"
#include "common/core/chunk.hpp"
#include "common/core/frame.hpp"
#include "common/io/doc_xml_common.hpp"
#include "common/io/sidecars.hpp"
#include "image/io/codec.hpp"

namespace alphashape::docxml {
namespace v1_detail {

// -- v1 entry names ---------------------------------------------------------

constexpr const char* kModelTexturePrefix = "model/texture_";

std::vector<std::uint8_t> require_entry(const ZipArchive& archive,
                                       const std::string& path) {
    if (!archive.contains(path)) {
        throw ProjectIOError("project.zip is missing '" + path + "'");
    }
    return archive.get(path);
}

/// The label a stored asset element carries; empty when it predates M11, in
/// which case the caller's add_<kind>() picks the kind's default label.
std::string stored_asset_label(const XmlNode& e) {
    const std::string* label = find_attr(e, "asset_label");
    return label != nullptr ? *label : std::string();
}

/// Apply a stored asset_id to the slot the reader just appended and make that
/// slot active; returns the id now carried by the slot.
///
/// Format v1 stores at most one asset per kind, and the chunk it is read into
/// is freshly restored, so the slot is the only entry: assigning the stored id
/// can never break the vector's sort order. A pre-M11 element carries no id, in
/// which case the freshly allocated one stands and is already active.
template <class Slot>
std::int32_t restore_asset_id(const XmlNode& e, std::vector<Slot>& slots) {
    if (slots.empty()) {
        return -1;
    }
    if (const std::string* id = find_attr(e, "asset_id")) {
        slots.back().id = parse_i32(*id);
    }
    return slots.back().id;
}

void read_sensors(const XmlNode& node, Chunk& chunk, ReadReport* report) {
    const XmlNode* sensors = find_child(node, "sensors");
    if (sensors == nullptr) {
        return;
    }
    std::size_t sensor_index = 0;
    for (const XmlNode& e : sensors->children) {
        warn_unknown_attrs(e,
                           {"id", "label", "type", "width", "height", "pixel_size",
                            "focal_length", "make", "model", "serial", "fixed_params"},
                           report);
        // Sensors are positional: a camera's sensor_id is an index into
        // chunk.sensors(), so the ids must match that order exactly.
        if (parse_int(require_attr(e, "id")) != static_cast<int>(sensor_index)) {
            throw ProjectIOError("doc.xml: sensor ids are not sequential");
        }
        Sensor s;
        s.label = require_attr(e, "label");
        s.type = parse_camera_type(require_attr(e, "type"));
        s.width = parse_int(require_attr(e, "width"));
        s.height = parse_int(require_attr(e, "height"));
        s.pixel_size_um = parse_double(require_attr(e, "pixel_size"));
        s.focal_length_mm = parse_double(require_attr(e, "focal_length"));
        s.make = require_attr(e, "make");
        s.model = require_attr(e, "model");
        s.serial = require_attr(e, "serial");
        s.calibration = read_calibration(find_child(e, "calibration"));
        s.fixed_params = split_strings(require_attr(e, "fixed_params"));
        validate_fixed_params(s.label, s.fixed_params);
        if (const XmlNode* calibration = find_child(e, "calibration")) {
            warn_unknown_calibration_attrs(*calibration, report);
        }
        chunk.sensors().push_back(std::move(s));
        ++sensor_index;
    }
}

void read_cameras(const XmlNode& node, Chunk& chunk, ReadReport* report) {
    const XmlNode* cameras = find_child(node, "cameras");
    if (cameras == nullptr) {
        return;
    }
    for (const XmlNode& e : cameras->children) {
        warn_unknown_attrs(e, {"id", "sensor_id", "label", "enabled", "orientation",
                               "transform"},
                           report);
        Camera c;
        c.id = parse_i32(require_attr(e, "id"));
        c.sensor_id = parse_i32(require_attr(e, "sensor_id"));
        c.label = require_attr(e, "label");
        c.enabled = parse_bool(require_attr(e, "enabled"));
        c.orientation = parse_int(require_attr(e, "orientation"));
        c.pose = parse_matrix4(require_attr(e, "transform"));
        if (const XmlNode* ref = find_child(e, "reference")) {
            c.reference.location = Eigen::Vector3d(
                parse_double(require_attr(*ref, "x")),
                parse_double(require_attr(*ref, "y")),
                parse_double(require_attr(*ref, "z")));
            // The accuracy attribute is required whenever <reference> is
            // present. Every component must be finite and positive.
            c.reference.accuracy = parse_vec3(require_attr(*ref, "accuracy"));
            const Eigen::Vector3d& acc = c.reference.accuracy;
            if (!std::isfinite(acc.x()) || !std::isfinite(acc.y()) ||
                !std::isfinite(acc.z()) || !(acc.x() > 0.0) || !(acc.y() > 0.0) ||
                !(acc.z() > 0.0)) {
                throw ProjectIOError(
                    "doc.xml: <reference> accuracy must be finite and positive, got '" +
                    require_attr(*ref, "accuracy") + "'");
            }
            c.reference.enabled = parse_int(require_attr(*ref, "enabled")) != 0;
        }
        chunk.cameras().push_back(std::move(c));
    }
}

void read_frames(const XmlNode& node, Chunk& chunk, ReadReport* report) {
    const XmlNode* frames = find_child(node, "frames");
    if (frames == nullptr) {
        return;
    }
    for (const XmlNode& e : frames->children) {
        warn_unknown_attrs(e, {"id", "camera_id", "path", "width", "height"}, report);
        auto f = std::make_shared<Frame>(require_attr(e, "path"),
                                        parse_int(require_attr(e, "width")),
                                        parse_int(require_attr(e, "height")));
        f->set_id(parse_i32(require_attr(e, "id")));
        f->set_camera_id(parse_i32(require_attr(e, "camera_id")));
        chunk.frames().push_back(std::move(f));
    }
}

void read_tie_points(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                     ReadReport* report) {
    const XmlNode* tie_points = find_child(node, "tie_points");
    if (tie_points == nullptr) {
        return;
    }
    for (const XmlNode& e : tie_points->children) {
        if (e.name != "keypoint_set" && e.name != "matches" && e.name != "tracks") {
            // v1 is lenient about additions: a child a later revision wrote
            // here is skipped rather than failing an old project's load.
            add_warning(report, "doc.xml: ignoring unknown <tie_points> child <" +
                                    e.name + ">");
            continue;
        }
        if (e.name == "keypoint_set") {
            KeypointSet set;
            set.camera_id = parse_i32(require_attr(e, "camera_id"));
            const std::string path = require_attr(e, "path");
            sidecar::decode_keypoints(require_entry(archive, path), set);
            if (set.keypoints.size() !=
                static_cast<std::size_t>(parse_ll(require_attr(e, "count")))) {
                throw ProjectIOError("keypoint count mismatch for camera " +
                                     std::to_string(set.camera_id));
            }
            chunk.tie_points().keypoint_sets.push_back(std::move(set));
        } else if (e.name == "matches") {
            const std::string path = require_attr(e, "path");
            chunk.tie_points().matches =
                sidecar::decode_matches(require_entry(archive, path));
            if (chunk.tie_points().matches.size() !=
                static_cast<std::size_t>(parse_ll(require_attr(e, "pairs")))) {
                throw ProjectIOError("matches count mismatch in doc.xml");
            }
        } else if (e.name == "tracks") {
            const std::string path = require_attr(e, "path");
            chunk.tie_points().tracks =
                sidecar::decode_tracks(require_entry(archive, path));
            if (chunk.tie_points().tracks.size() !=
                static_cast<std::size_t>(parse_ll(require_attr(e, "count")))) {
                throw ProjectIOError("tracks count mismatch in doc.xml");
            }
        }
    }
    // Cross-check every invariant once the whole container is assembled, using
    // the exact validator the writer runs.
    sidecar::validate_tie_points(chunk);
}

void read_assets(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                 ReadReport* report) {
    if (const XmlNode* depth_maps = find_child(node, "depth_maps")) {
        warn_unknown_attrs(*depth_maps, {"path", "count", "asset_id", "asset_label"},
                           report);
        const std::string path = require_attr(*depth_maps, "path");
        // Each stored product becomes one asset: the legacy accessor would
        // lazily mint an unlabelled one, which would lose the stored identity.
        DepthMapsAsset& asset = chunk.add_depth_maps(stored_asset_label(*depth_maps));
        *asset.data = sidecar::decode_depth_maps(require_entry(archive, path));
        chunk.set_active_depth_maps_id(
            restore_asset_id(*depth_maps, chunk.depth_map_sets()));
        if (chunk.depth_maps().size() !=
            static_cast<std::size_t>(parse_ll(require_attr(*depth_maps, "count")))) {
            throw ProjectIOError("depth map count mismatch in doc.xml");
        }
    }
    if (const XmlNode* point_cloud = find_child(node, "point_cloud")) {
        warn_unknown_attrs(*point_cloud, {"path", "count", "asset_id", "asset_label"},
                           report);
        const std::string path = require_attr(*point_cloud, "path");
        PointCloudAsset& asset = chunk.add_point_cloud(stored_asset_label(*point_cloud));
        *asset.data = sidecar::decode_point_cloud(require_entry(archive, path));
        chunk.set_active_point_cloud_id(
            restore_asset_id(*point_cloud, chunk.point_clouds()));
        if (chunk.point_cloud().points.size() !=
            static_cast<std::size_t>(parse_ll(require_attr(*point_cloud, "count")))) {
            throw ProjectIOError("point cloud count mismatch in doc.xml");
        }
    }
    if (const XmlNode* model = find_child(node, "model")) {
        warn_unknown_attrs(*model,
                           {"path", "faces", "vertices", "textures", "asset_id",
                            "asset_label"},
                           report);
        const std::string path = require_attr(*model, "path");
        ModelAsset& asset = chunk.add_model(stored_asset_label(*model));
        *asset.data = sidecar::decode_mesh(require_entry(archive, path));
        chunk.set_active_model_id(restore_asset_id(*model, chunk.models()));
        if (chunk.model().faces.size() !=
                static_cast<std::size_t>(parse_ll(require_attr(*model, "faces"))) ||
            chunk.model().vertices.size() !=
                static_cast<std::size_t>(parse_ll(require_attr(*model, "vertices")))) {
            throw ProjectIOError("mesh counts mismatch in doc.xml");
        }
        // Texture pages, when announced. The count is authoritative: a page
        // named in doc.xml but absent from the zip, or one that will not decode
        // as an 8-bit image, is a corrupt project.
        if (const std::string* texture_count = find_attr(*model, "textures")) {
            const long long page_count = parse_ll(*texture_count);
            if (page_count < 0) {
                throw ProjectIOError("mesh texture count is negative in doc.xml");
            }
            for (long long page = 0; page < page_count; ++page) {
                const std::string entry =
                    kModelTexturePrefix + std::to_string(page) + ".png";
                const std::vector<std::uint8_t> bytes = require_entry(archive, entry);
                chunk.model_textures().push_back(
                    image_detail::png_decode(bytes.data(), bytes.size()));
            }
        }
    }
    // Milestone 9: the DEM and orthomosaic side files. An absent element means
    // the product is empty, so every earlier project still loads. A named entry
    // that is missing from the zip, or whose grid/metadata disagrees with
    // doc.xml, is a corrupt project.
    if (const XmlNode* elevation = find_child(node, "elevation")) {
        warn_unknown_attrs(*elevation,
                           {"path", "width", "height", "origin", "pixel_size", "epsg",
                            "nodata", "source", "asset_id", "asset_label"},
                           report);
        const std::string path = require_attr(*elevation, "path");
        Elevation dem = sidecar::decode_elevation(require_entry(archive, path));
        dem.source = require_attr(*elevation, "source");
        const RasterGrid& g = dem.raster.grid;
        if (g.width != parse_int(require_attr(*elevation, "width")) ||
            g.height != parse_int(require_attr(*elevation, "height"))) {
            throw ProjectIOError(
                "doc.xml: <elevation> dimensions do not match the GeoTIFF");
        }
        const auto origin =
            parse_pair(require_attr(*elevation, "origin"), "<elevation> origin");
        const auto pixel_size =
            parse_pair(require_attr(*elevation, "pixel_size"), "<elevation> pixel_size");
        if (origin.first != g.origin_x || origin.second != g.origin_y ||
            pixel_size.first != g.pixel_size_x ||
            pixel_size.second != g.pixel_size_y) {
            throw ProjectIOError(
                "doc.xml: <elevation> georeferencing does not match the GeoTIFF");
        }
        if (parse_int(require_attr(*elevation, "epsg")) != dem.epsg) {
            throw ProjectIOError("doc.xml: <elevation> epsg does not match the GeoTIFF");
        }
        dem.nodata = static_cast<float>(parse_double(require_attr(*elevation, "nodata")));
        ElevationAsset& asset = chunk.add_elevation(stored_asset_label(*elevation));
        *asset.data = std::move(dem);
        chunk.set_active_elevation_id(restore_asset_id(*elevation, chunk.elevations()));
    }
    if (const XmlNode* ortho = find_child(node, "orthomosaic")) {
        warn_unknown_attrs(*ortho,
                           {"path", "labels_path", "width", "height", "origin",
                            "pixel_size", "epsg", "surface", "asset_id", "asset_label"},
                           report);
        const std::string path = require_attr(*ortho, "path");
        const std::string labels_path = require_attr(*ortho, "labels_path");
        Orthomosaic mosaic = sidecar::decode_orthomosaic(require_entry(archive, path));
        mosaic.surface = require_attr(*ortho, "surface");
        const RasterGrid& g = mosaic.raster.grid;
        if (g.width != parse_int(require_attr(*ortho, "width")) ||
            g.height != parse_int(require_attr(*ortho, "height"))) {
            throw ProjectIOError(
                "doc.xml: <orthomosaic> dimensions do not match the GeoTIFF");
        }
        const auto origin =
            parse_pair(require_attr(*ortho, "origin"), "<orthomosaic> origin");
        const auto pixel_size =
            parse_pair(require_attr(*ortho, "pixel_size"), "<orthomosaic> pixel_size");
        if (origin.first != g.origin_x || origin.second != g.origin_y ||
            pixel_size.first != g.pixel_size_x ||
            pixel_size.second != g.pixel_size_y) {
            throw ProjectIOError(
                "doc.xml: <orthomosaic> georeferencing does not match the GeoTIFF");
        }
        if (parse_int(require_attr(*ortho, "epsg")) != mosaic.epsg) {
            throw ProjectIOError("doc.xml: <orthomosaic> epsg does not match the GeoTIFF");
        }
        mosaic.labels = sidecar::decode_ortho_labels(
            require_entry(archive, labels_path), g.width, g.height);
        OrthomosaicAsset& asset = chunk.add_orthomosaic(stored_asset_label(*ortho));
        *asset.data = std::move(mosaic);
        chunk.set_active_orthomosaic_id(restore_asset_id(*ortho, chunk.orthomosaics()));
    }
    // Cross-check the dense invariants with the exact validator the writer runs.
    sidecar::validate_dense(chunk);
    sidecar::validate_mesh(chunk);
    sidecar::validate_model_textures(chunk);
    sidecar::validate_products(chunk);
}

void read_chunk(const XmlNode& node, Workspace& workspace, const ZipArchive& archive,
                ReadReport* report) {
    using v1_detail::read_assets;
    using v1_detail::read_cameras;
    using v1_detail::read_frames;
    using v1_detail::read_sensors;
    using v1_detail::read_tie_points;
    // v1 has a fixed, documented chunk vocabulary. An element from a later
    // revision is skipped with a warning rather than rejected: a v1 document is
    // exactly the ones this reader must keep accepting.
    warn_unknown_children(node,
                          {"sensors", "cameras", "frames", "tie_points", "depth_maps",
                           "point_cloud", "model", "elevation", "orthomosaic",
                           "region", "transform", "crs", "meta"},
                          report);
    warn_unknown_attrs(node, {"id", "label", "next_id"}, report);
    Chunk& chunk = workspace.restore_chunk(parse_i32(require_attr(node, "id")),
                                           require_attr(node, "label"));
    v1_detail::read_sensors(node, chunk, report);
    v1_detail::read_cameras(node, chunk, report);
    v1_detail::read_frames(node, chunk, report);
    warn_missing_frame_files(chunk.frames(), report);
    v1_detail::read_tie_points(node, chunk, archive, report);
    v1_detail::read_assets(node, chunk, archive, report);

    if (const XmlNode* region = find_child(node, "region")) {
        chunk.region().centre = parse_vec3(require_attr(*region, "centre"));
        chunk.region().size = parse_vec3(require_attr(*region, "size"));
    }
    if (const XmlNode* transform = find_child(node, "transform")) {
        chunk.transform() = parse_matrix4(require_attr(*transform, "matrix"));
    }
    read_crs(find_child(node, "crs"), chunk.crs());
    read_meta(find_child(node, "meta"), chunk.meta());

    // The id allocator, when the writer recorded it. An older project has no
    // such attribute: the counter stays where adding the assets left it, which
    // is already past every id those assets drew.
    if (const std::string* next_id = find_attr(node, "next_id")) {
        chunk.set_next_id(parse_i32(*next_id));
    }
    // Whatever the stored counter said, it must clear every asset id we have.
    for (const auto& asset : chunk.elevations()) chunk.reserve_id(asset.id);
    for (const auto& asset : chunk.orthomosaics()) chunk.reserve_id(asset.id);
    for (const auto& asset : chunk.models()) chunk.reserve_id(asset.id);
    for (const auto& asset : chunk.point_clouds()) chunk.reserve_id(asset.id);
    for (const auto& asset : chunk.depth_map_sets()) chunk.reserve_id(asset.id);
    sidecar::validate_tie_points(chunk);
    sidecar::validate_dense(chunk);
    sidecar::validate_mesh(chunk);
    sidecar::validate_model_textures(chunk);
    sidecar::validate_products(chunk);
    sidecar::validate_m11_containers(chunk);
}

}  // namespace v1_detail

std::shared_ptr<Document> read_v1(const XmlNode& root, const ZipArchive& archive,
                                  ReadReport* report) {
    auto document = std::make_shared<Document>();
    Workspace& workspace = document->workspace();
    warn_unknown_children(root, {"global_crs", "meta", "chunks"}, report);
    warn_unknown_attrs(root, {"version", "next_id", "active_id"}, report);
    read_crs(find_child(root, "global_crs"), document->global_crs());
    read_meta(find_child(root, "meta"), document->meta());
    if (const XmlNode* chunks = find_child(root, "chunks")) {
        for (const XmlNode& node : chunks->children) {
            if (node.name == "chunk") {
                v1_detail::read_chunk(node, workspace, archive, report);
            }
        }
    }
    // Restore the counter last: it may be ahead of every surviving chunk id
    // (chunks can be deleted), and ids must never be reused.
    const std::int32_t next_id = parse_i32(require_attr(root, "next_id"));
    if (next_id > workspace.next_chunk_id()) {
        workspace.set_next_chunk_id(next_id);
    }
    document->set_active_chunk_id(parse_i32(require_attr(root, "active_id")));
    return document;
}

}  // namespace alphashape::docxml
