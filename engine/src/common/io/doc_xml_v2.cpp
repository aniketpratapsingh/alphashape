// doc.xml format v2: the M11 layout (see docs/plan/M11.md §4.4-4.6 and §4.5).
//
// Writer: every container of every chunk is emitted, all assets of every kind
// (not just the active one), and bulk data lives in per-chunk archive entries
// under "chunks/<chunk_id>/...". Containers are written only when non-empty, so
// a project with nothing in a kind carries no element for it. Attribute and
// child order is fixed, which makes a load/save of an untouched project
// reproduce identical bytes (ProjectV2.UntouchedChunkResaveIsByteStable).
//
// Reader: rebuilds the model from exactly that vocabulary. Unknown *elements*
// inside <chunk> are an error (writer/reader drift must not pass silently);
// unknown *attributes* are ignored. An asset element without a "path" is an
// empty asset of that kind, which is how an in-memory empty asset slot keeps
// its identity across a round-trip.

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/io/zip.hpp"
#include "base/utils/error.hpp"
#include "common/core/chunk.hpp"
#include "common/io/doc_xml.hpp"
#include "common/core/frame.hpp"
#include "common/io/doc_xml_common.hpp"
#include "common/io/sidecars.hpp"
#include "image/io/codec.hpp"
#include "image/io/image_io.hpp"

namespace alphashape::docxml {
namespace docxml_detail {

using sidecar::ByteReader;

// -- entry names ------------------------------------------------------------

constexpr const char* kTexturePrefix = "/texture_";

std::string chunk_prefix(const Chunk& chunk) {
    return "chunks/" + std::to_string(chunk.id());
}

std::string keypoint_entry(const Chunk& chunk, std::int32_t camera_id) {
    return chunk_prefix(chunk) + "/tie_points/camera_" +
           std::to_string(camera_id) + ".kpt";
}

std::string matches_entry(const Chunk& chunk) {
    return chunk_prefix(chunk) + "/tie_points/matches.bin";
}

std::string tracks_entry(const Chunk& chunk) {
    return chunk_prefix(chunk) + "/tie_points/tracks.bin";
}

std::string depth_maps_entry(const Chunk& chunk, std::int32_t asset_id) {
    return chunk_prefix(chunk) + "/depth_maps/" + std::to_string(asset_id) + ".bin";
}

std::string point_cloud_entry(const Chunk& chunk, std::int32_t asset_id) {
    return chunk_prefix(chunk) + "/point_clouds/" + std::to_string(asset_id) + ".bin";
}

std::string mesh_entry(const Chunk& chunk, std::int32_t asset_id) {
    return chunk_prefix(chunk) + "/models/" + std::to_string(asset_id) + "/mesh.bin";
}

std::string texture_entry(const Chunk& chunk, std::int32_t asset_id, std::size_t page) {
    return chunk_prefix(chunk) + "/models/" + std::to_string(asset_id) + kTexturePrefix +
           std::to_string(page) + ".png";
}

std::string elevation_entry(const Chunk& chunk, std::int32_t asset_id) {
    return chunk_prefix(chunk) + "/elevations/" + std::to_string(asset_id) + ".tif";
}

std::string ortho_entry(const Chunk& chunk, std::int32_t asset_id) {
    return chunk_prefix(chunk) + "/orthomosaics/" + std::to_string(asset_id) + ".tif";
}

std::string ortho_labels_entry(const Chunk& chunk, std::int32_t asset_id) {
    return chunk_prefix(chunk) + "/orthomosaics/" + std::to_string(asset_id) +
           ".labels.bin";
}

std::string mask_entry(const Chunk& chunk, int camera_id) {
    return chunk_prefix(chunk) + "/masks/" + std::to_string(camera_id) + ".bin";
}

std::string shapes_entry(const Chunk& chunk) {
    return chunk_prefix(chunk) + "/shapes.shp";
}

std::string trajectory_entry(const Chunk& chunk, std::int32_t asset_id) {
    return chunk_prefix(chunk) + "/trajectories/" + std::to_string(asset_id) + ".bin";
}

std::string frames_entry(const Chunk& chunk) {
    return chunk_prefix(chunk) + "/frames.bin";
}

std::string tiled_model_entry(const Chunk& chunk, std::int32_t asset_id) {
    return chunk_prefix(chunk) + "/tiled_models/" + std::to_string(asset_id) + ".bin";
}

std::string panorama_entry(const Chunk& chunk, std::int32_t asset_id) {
    return chunk_prefix(chunk) + "/panorama_" + std::to_string(asset_id) + ".png";
}

// -- vocabulary -------------------------------------------------------------

std::string marker_type_name(MarkerType type) {
    switch (type) {
        case MarkerType::Regular: return "regular";
        case MarkerType::Vertex: return "vertex";
        case MarkerType::Fiducial: return "fiducial";
    }
    return "regular";
}

MarkerType parse_marker_type(const std::string& s) {
    if (s == "regular") return MarkerType::Regular;
    if (s == "vertex") return MarkerType::Vertex;
    if (s == "fiducial") return MarkerType::Fiducial;
    throw ProjectIOError("doc.xml: unknown marker type '" + s + "'");
}

std::string camera_group_type_name(CameraGroupType type) {
    switch (type) {
        case CameraGroupType::None: return "none";
        case CameraGroupType::Folder: return "folder";
        case CameraGroupType::Station: return "station";
        case CameraGroupType::Fixed: return "fixed";
    }
    return "none";
}

CameraGroupType parse_camera_group_type(const std::string& s) {
    if (s == "none") return CameraGroupType::None;
    if (s == "folder") return CameraGroupType::Folder;
    if (s == "station") return CameraGroupType::Station;
    if (s == "fixed") return CameraGroupType::Fixed;
    throw ProjectIOError("doc.xml: unknown camera group type '" + s + "'");
}

std::string shape_type_name(ShapeType type) {
    switch (type) {
        case ShapeType::Point: return "point";
        case ShapeType::Polyline: return "polyline";
        case ShapeType::Polygon: return "polygon";
    }
    return "point";
}

ShapeType parse_shape_type(const std::string& s) {
    if (s == "point") return ShapeType::Point;
    if (s == "polyline") return ShapeType::Polyline;
    if (s == "polygon") return ShapeType::Polygon;
    throw ProjectIOError("doc.xml: unknown shape type '" + s + "'");
}

std::string endpoint_kind_name(ScalebarEndpoint::Kind kind) {
    return kind == ScalebarEndpoint::Kind::Camera ? "camera" : "marker";
}

ScalebarEndpoint::Kind parse_endpoint_kind(const std::string& s) {
    if (s == "marker") return ScalebarEndpoint::Kind::Marker;
    if (s == "camera") return ScalebarEndpoint::Kind::Camera;
    throw ProjectIOError("doc.xml: unknown scalebar endpoint kind '" + s + "'");
}

std::string id_list(const std::vector<std::int32_t>& ids) {
    std::string out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) out += ' ';
        out += std::to_string(ids[i]);
    }
    return out;
}

std::vector<std::int32_t> parse_id_list(const std::string& s) {
    std::vector<std::int32_t> out;
    for (const std::string& token : split_strings(s)) {
        out.push_back(parse_i32(token));
    }
    return out;
}

/// Space-separated doubles; used for the 3x3 marker position covariance.
std::string covariance9(const Eigen::Matrix3d& m) {
    double values[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) values[r * 3 + c] = m(r, c);
    }
    return join(values, 9);
}

Eigen::Matrix3d parse_covariance9(const std::string& s) {
    const std::vector<double> v = parse_doubles(s);
    if (v.size() != 9) {
        throw ProjectIOError("doc.xml: expected 9 numbers for a covariance");
    }
    Eigen::Matrix3d m;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) m(r, c) = v[static_cast<std::size_t>(r * 3 + c)];
    }
    return m;
}

// -- writer: identity attributes shared by every asset element ---------------

/// Write the slot's identity (id, label, group, enabled, meta) on its element.
/// Every asset element carries these, including the empty ones, so an asset
/// that exists in memory but holds no payload keeps its identity.
template <class Slot>
void write_slot_identity(const Slot& slot, XmlNode& e) {
    set_attr(e, "id", std::to_string(slot.id));
    set_attr(e, "label", slot.label);
    if (slot.group_id >= 0) {
        set_attr(e, "group", std::to_string(slot.group_id));
    }
    if (!slot.enabled) {
        set_attr(e, "enabled", "false");
    }
    if (!slot.meta.items.empty()) {
        write_meta(slot.meta, e);
    }
}

/// Read the identity attributes back onto a slot. `id` is required: an asset
/// element the writer emitted always carries it.
template <class Slot>
void read_slot_identity(const XmlNode& e, Slot& slot) {
    slot.id = parse_i32(require_attr(e, "id"));
    slot.label = attr_or(e, "label", std::string());
    const std::string* group = find_attr(e, "group");
    slot.group_id = group != nullptr ? parse_i32(*group) : -1;
    slot.enabled = attr_or(e, "enabled", "true") != "false";
    read_meta(find_child(e, "meta"), slot.meta);
}

// -- writer: the chunk ------------------------------------------------------

void write_sensors(const Chunk& chunk, XmlNode& node) {
    XmlNode& sensors = add_child(node, "sensors");
    for (std::size_t i = 0; i < chunk.sensors().size(); ++i) {
        const Sensor& s = chunk.sensors()[i];
        XmlNode& e = add_child(sensors, "sensor");
        set_attr(e, "id", std::to_string(i));
        set_attr(e, "label", s.label);
        set_attr(e, "type", camera_type_name(s.type));
        set_attr(e, "width", std::to_string(s.width));
        set_attr(e, "height", std::to_string(s.height));
        set_attr(e, "pixel_size", fmt(s.pixel_size_um));
        set_attr(e, "focal_length", fmt(s.focal_length_mm));
        set_attr(e, "make", s.make);
        set_attr(e, "model", s.model);
        set_attr(e, "serial", s.serial);
        write_calibration(s.calibration, e);
        set_attr(e, "fixed_params", join_strings(s.fixed_params));
    }
}

void write_cameras(const Chunk& chunk, XmlNode& node) {
    XmlNode& cameras = add_child(node, "cameras");
    for (const Camera& c : chunk.cameras()) {
        XmlNode& e = add_child(cameras, "camera");
        set_attr(e, "id", std::to_string(c.id));
        set_attr(e, "sensor_id", std::to_string(c.sensor_id));
        set_attr(e, "label", c.label);
        set_attr(e, "enabled", fmt_bool(c.enabled));
        set_attr(e, "orientation", std::to_string(c.orientation));
        set_attr(e, "transform", matrix4(c.pose));
        // A reference is written only when a location was set. z may be nan
        // (no GNSS altitude); fmt() emits "nan", which parse_double accepts.
        // Every accuracy component must be finite and strictly positive: a
        // zero or negative sigma is not a valid weight, and a NaN would poison
        // the adjustment's information matrix.
        if (c.reference.location.has_value()) {
            const Eigen::Vector3d& acc = c.reference.accuracy;
            if (!std::isfinite(acc.x()) || !std::isfinite(acc.y()) ||
                !std::isfinite(acc.z()) || !(acc.x() > 0.0) || !(acc.y() > 0.0) ||
                !(acc.z() > 0.0)) {
                throw ProjectIOError("camera " + std::to_string(c.id) +
                                     ": reference accuracy must be finite and positive");
            }
            XmlNode& ref = add_child(e, "reference");
            set_attr(ref, "x", fmt((*c.reference.location).x()));
            set_attr(ref, "y", fmt((*c.reference.location).y()));
            set_attr(ref, "z", fmt((*c.reference.location).z()));
            set_attr(ref, "accuracy", vec3(acc));
            set_attr(ref, "enabled", c.reference.enabled ? "1" : "0");
        }
    }
}

void write_camera_groups(const Chunk& chunk, XmlNode& node) {
    if (chunk.camera_groups().empty()) {
        return;
    }
    XmlNode& groups = add_child(node, "camera_groups");
    for (const CameraGroup& g : chunk.camera_groups()) {
        XmlNode& e = add_child(groups, "group");
        set_attr(e, "id", std::to_string(g.id));
        set_attr(e, "label", g.label);
        set_attr(e, "type", camera_group_type_name(g.type));
        set_attr(e, "members", id_list(g.members));
        if (!g.folder.empty()) {
            set_attr(e, "folder", g.folder);
        }
    }
}

void write_frames(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    if (chunk.frames().empty()) {
        return;
    }
    archive.add(frames_entry(chunk), sidecar::encode_frames(chunk.frames()));
    XmlNode& frames = add_child(node, "frames");
    set_attr(frames, "path", frames_entry(chunk));
    for (const auto& f : chunk.frames()) {
        XmlNode& e = add_child(frames, "frame");
        set_attr(e, "id", std::to_string(f->id()));
        set_attr(e, "path", f->path());
        set_attr(e, "width", std::to_string(f->width()));
        set_attr(e, "height", std::to_string(f->height()));
        set_attr(e, "camera_id", std::to_string(f->camera_id()));
    }
}

void write_markers(const Chunk& chunk, XmlNode& node) {
    if (chunk.markers().empty() && chunk.marker_groups().empty()) {
        return;
    }
    XmlNode& markers = add_child(node, "markers");
    for (const Marker& marker : chunk.markers()) {
        XmlNode& e = add_child(markers, "marker");
        set_attr(e, "id", std::to_string(marker.id));
        set_attr(e, "label", marker.label);
        if (marker.group_id >= 0) {
            set_attr(e, "group", std::to_string(marker.group_id));
        }
        set_attr(e, "type", marker_type_name(marker.type));
        // Projections: one <projection> child per camera measurement. The
        // writer emits them in vector order; the validator (run by the caller)
        // has already established they are sorted and unique.
        if (!marker.projections.empty()) {
            set_attr(e, "projections", std::to_string(marker.projections.size()));
            for (const MarkerProjection& p : marker.projections) {
                XmlNode& proj = add_child(e, "projection");
                set_attr(proj, "camera_id", std::to_string(p.camera_id));
                set_attr(proj, "pixel", vec2(p.pixel.x(), p.pixel.y()));
                if (!p.enabled) {
                    set_attr(proj, "enabled", "false");
                }
            }
        }
        if (!marker.meta.items.empty()) {
            write_meta(marker.meta, e);
        }
        if (marker.reference.has_value()) {
            const Eigen::Vector3d& r = *marker.reference;
            set_attr(e, "location", vec3(r));
            set_attr(e, "reference_accuracy", vec3(marker.reference_accuracy));
        }
        if (marker.position.has_value()) {
            set_attr(e, "location_estimate", vec3(*marker.position));
            set_attr(e, "position_covariance", covariance9(marker.position_covariance));
        }
        if (marker.pinned) set_attr(e, "pinned", "1");
        if (marker.valid) set_attr(e, "valid", "1");
        if (marker.internal) set_attr(e, "internal", "1");
    }
    if (!chunk.marker_groups().empty()) {
        XmlNode& groups = add_child(markers, "marker_groups");
        for (const MarkerGroup& g : chunk.marker_groups()) {
            XmlNode& e = add_child(groups, "group");
            set_attr(e, "id", std::to_string(g.id));
            set_attr(e, "label", g.label);
            set_attr(e, "members", id_list(g.members));
        }
    }
}

void write_scalebars(const Chunk& chunk, XmlNode& node) {
    if (chunk.scalebars().empty()) {
        return;
    }
    XmlNode& scalebars = add_child(node, "scalebars");
    for (const Scalebar& bar : chunk.scalebars()) {
        XmlNode& e = add_child(scalebars, "scalebar");
        set_attr(e, "id", std::to_string(bar.id));
        set_attr(e, "label", bar.label);
        set_attr(e, "from_kind", endpoint_kind_name(bar.from.kind));
        set_attr(e, "from_id", std::to_string(bar.from.id));
        set_attr(e, "to_kind", endpoint_kind_name(bar.to.kind));
        set_attr(e, "to_id", std::to_string(bar.to.id));
        set_attr(e, "reference_length", fmt(bar.reference_length_m));
        set_attr(e, "reference_accuracy", fmt(bar.reference_accuracy_m));
        set_attr(e, "estimated_length", fmt(bar.estimated_length_m));
    }
}

void write_tie_points(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    const TiePoints& tie_points = chunk.tie_points();
    // Only when there is something to say: an unpopulated chunk must not carry
    // an empty <tie_points>, so EmptyContainersAreOmitted holds.
    if (tie_points.keypoint_sets.empty() && tie_points.matches.empty() &&
        tie_points.tracks.empty()) {
        return;
    }
    XmlNode& container = add_child(node, "tie_points");
    for (const KeypointSet& set : tie_points.keypoint_sets) {
        if (!set.consistent()) {
            throw ProjectIOError("keypoints and descriptors disagree for camera " +
                                 std::to_string(set.camera_id));
        }
        const std::string entry = keypoint_entry(chunk, set.camera_id);
        if (archive.contains(entry)) {
            throw ProjectIOError("two keypoint sets for camera " +
                                 std::to_string(set.camera_id));
        }
        archive.add(entry, sidecar::encode_keypoints(set));
        XmlNode& e = add_child(container, "keypoint_set");
        set_attr(e, "camera_id", std::to_string(set.camera_id));
        set_attr(e, "count", std::to_string(set.keypoints.size()));
        set_attr(e, "path", entry);
    }
    if (!tie_points.matches.empty()) {
        archive.add(matches_entry(chunk), sidecar::encode_matches(tie_points.matches));
        XmlNode& e = add_child(container, "matches");
        set_attr(e, "pairs", std::to_string(tie_points.matches.size()));
        set_attr(e, "path", matches_entry(chunk));
    }
    if (!tie_points.tracks.empty()) {
        archive.add(tracks_entry(chunk), sidecar::encode_tracks(tie_points.tracks));
        XmlNode& e = add_child(container, "tracks");
        set_attr(e, "count", std::to_string(tie_points.tracks.size()));
        set_attr(e, "path", tracks_entry(chunk));
    }
}

void write_depth_map_sets(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    if (chunk.depth_map_sets().empty()) {
        return;
    }
    XmlNode& container = add_child(node, "depth_map_sets");
    for (const DepthMapsAsset& asset : chunk.depth_map_sets()) {
        XmlNode& e = add_child(container, "depth_maps");
        write_slot_identity(asset, e);
        if (asset.id == chunk.active_depth_maps_id()) {
            set_attr(e, "active", "true");
        }
        if (asset.data != nullptr && !asset.data->empty()) {
            const std::string entry = depth_maps_entry(chunk, asset.id);
            archive.add(entry, sidecar::encode_depth_maps(*asset.data));
            set_attr(e, "count", std::to_string(asset.data->size()));
            set_attr(e, "path", entry);
        }
    }
}

void write_point_clouds(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    if (chunk.point_clouds().empty()) {
        return;
    }
    XmlNode& container = add_child(node, "dense_clouds");
    for (const PointCloudAsset& asset : chunk.point_clouds()) {
        XmlNode& e = add_child(container, "dense_cloud");
        write_slot_identity(asset, e);
        if (asset.id == chunk.active_point_cloud_id()) {
            set_attr(e, "active", "true");
        }
        if (asset.data != nullptr && !asset.data->points.empty()) {
            const std::string entry = point_cloud_entry(chunk, asset.id);
            archive.add(entry, sidecar::encode_point_cloud(*asset.data));
            set_attr(e, "count", std::to_string(asset.data->points.size()));
            set_attr(e, "hasPointColors", asset.data->colours.empty() ? "false" : "true");
            set_attr(e, "hasPointNormals", asset.data->normals.empty() ? "false" : "true");
            set_attr(e, "hasPointConfidence",
                     asset.data->confidence.empty() ? "false" : "true");
            set_attr(e, "path", entry);
        }
    }
}

void write_models(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    if (chunk.models().empty()) {
        return;
    }
    XmlNode& container = add_child(node, "models");
    for (const ModelAsset& asset : chunk.models()) {
        XmlNode& e = add_child(container, "model");
        write_slot_identity(asset, e);
        if (asset.id == chunk.active_model_id()) {
            set_attr(e, "active", "true");
        }
        if (asset.source_id >= 0) {
            set_attr(e, "source_id", std::to_string(asset.source_id));
        }
        if (asset.data != nullptr && !asset.data->empty()) {
            const Mesh& mesh = *asset.data;
            const std::string entry = mesh_entry(chunk, asset.id);
            archive.add(entry, sidecar::encode_mesh(mesh));
            set_attr(e, "faceCount", std::to_string(mesh.faces.size()));
            set_attr(e, "vertexCount", std::to_string(mesh.vertices.size()));
            set_attr(e, "hasVertexColors", mesh.colours.empty() ? "false" : "true");
            set_attr(e, "hasVertexConfidence",
                     mesh.confidence.empty() ? "false" : "true");
            set_attr(e, "hasUV", mesh.uvs.empty() ? "false" : "true");
            set_attr(e, "path", entry);
        }
        // Texture pages live beside the mesh, announced by a count on a
        // <texture> child so a model with no textures is byte-identical to one
        // written before texturing existed.
        if (!asset.textures.empty()) {
            XmlNode& texture = add_child(e, "texture");
            set_attr(texture, "pages", std::to_string(asset.textures.size()));
            for (std::size_t page = 0; page < asset.textures.size(); ++page) {
                const std::string entry = texture_entry(chunk, asset.id, page);
                archive.add(entry, encode_png(asset.textures[page]));
                set_attr(add_child(texture, "page"), "path", entry);
            }
        }
    }
}

void write_elevations(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    if (chunk.elevations().empty()) {
        return;
    }
    XmlNode& container = add_child(node, "elevations");
    for (const ElevationAsset& asset : chunk.elevations()) {
        XmlNode& e = add_child(container, "elevation");
        write_slot_identity(asset, e);
        if (asset.id == chunk.active_elevation_id()) {
            set_attr(e, "active", "true");
        }
        if (asset.data == nullptr || asset.data->empty()) {
            continue;
        }
        const Elevation& elevation = *asset.data;
        const std::string entry = elevation_entry(chunk, asset.id);
        archive.add(entry, sidecar::encode_elevation(elevation));
        const RasterGrid& g = elevation.raster.grid;
        set_attr(e, "path", entry);
        set_attr(e, "width", std::to_string(g.width));
        set_attr(e, "height", std::to_string(g.height));
        set_attr(e, "origin", vec2(g.origin_x, g.origin_y));
        set_attr(e, "pixel_size", vec2(g.pixel_size_x, g.pixel_size_y));
        set_attr(e, "epsg", std::to_string(elevation.epsg()));
        // The canonical WKT is written only when the CRS carries one, so a grid
        // in a plain EPSG code keeps producing the pre-M12 bytes. Reading
        // prefers the code (types.cpp), so an EPSG-only element left by a
        // pre-M12 build loads to exactly the same CRS.
        if (!elevation.crs.wkt.empty()) {
            set_attr(e, "crs_wkt", elevation.crs.wkt);
        }
        set_attr(e, "nodataValue", fmt(static_cast<double>(elevation.nodata)));
        set_attr(e, "source_data", elevation.source);
    }
}

void write_orthomosaics(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    if (chunk.orthomosaics().empty()) {
        return;
    }
    XmlNode& container = add_child(node, "orthomosaics");
    for (const OrthomosaicAsset& asset : chunk.orthomosaics()) {
        XmlNode& e = add_child(container, "orthomosaic");
        write_slot_identity(asset, e);
        if (asset.id == chunk.active_orthomosaic_id()) {
            set_attr(e, "active", "true");
        }
        if (asset.data == nullptr || asset.data->empty()) {
            continue;
        }
        const Orthomosaic& ortho = *asset.data;
        const std::string entry = ortho_entry(chunk, asset.id);
        const std::string labels = ortho_labels_entry(chunk, asset.id);
        archive.add(entry, sidecar::encode_orthomosaic(ortho));
        archive.add(labels, sidecar::encode_ortho_labels(ortho));
        const RasterGrid& g = ortho.raster.grid;
        set_attr(e, "path", entry);
        set_attr(e, "labels_path", labels);
        set_attr(e, "width", std::to_string(g.width));
        set_attr(e, "height", std::to_string(g.height));
        set_attr(e, "origin", vec2(g.origin_x, g.origin_y));
        set_attr(e, "pixel_size", vec2(g.pixel_size_x, g.pixel_size_y));
        set_attr(e, "epsg", std::to_string(ortho.epsg()));
        if (!ortho.crs.wkt.empty()) {
            set_attr(e, "crs_wkt", ortho.crs.wkt);
        }
        set_attr(e, "blendingMode", ortho.surface);
    }
}

void write_masks(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    if (chunk.masks().empty()) {
        return;
    }
    XmlNode& container = add_child(node, "mask_sets");
    XmlNode& masks = add_child(container, "masks");
    for (const Mask& mask : chunk.masks()) {
        const std::string entry = mask_entry(chunk, mask.camera_id);
        archive.add(entry, sidecar::encode_mask(mask));
        XmlNode& e = add_child(masks, "mask");
        set_attr(e, "camera_id", std::to_string(mask.camera_id));
        set_attr(e, "width", std::to_string(mask.width));
        set_attr(e, "height", std::to_string(mask.height));
        if (mask.invert) {
            set_attr(e, "invert", "true");
        }
        set_attr(e, "path", entry);
    }
}

void write_shapes(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    const Shapes& shapes = chunk.shapes();
    if (shapes.features.empty() && shapes.groups.empty()) {
        return;
    }
    XmlNode& container = add_child(node, "shapes");
    if (!shapes.features.empty()) {
        const std::string entry = shapes_entry(chunk);
        archive.add(entry, sidecar::encode_shapes(shapes));
        set_attr(container, "path", entry);
        set_attr(container, "count", std::to_string(shapes.features.size()));
    }
    // Layers and attributes are readable in the XML as well as in "ASH1", so a
    // hand-inspected project shows which layers exist without decoding bytes.
    for (const Shape& shape : shapes.features) {
        XmlNode& e = add_child(container, "shape");
        set_attr(e, "id", std::to_string(shape.id));
        set_attr(e, "type", shape_type_name(shape.type));
        set_attr(e, "label", shape.label);
        set_attr(e, "layer", shape.layer);
        set_attr(e, "closed", fmt_bool(shape.closed));
        if (shape.has_altitude) {
            set_attr(e, "has_altitude", "true");
        }
    }
    if (!shapes.groups.empty()) {
        XmlNode& groups = add_child(container, "shape_groups");
        for (const ShapeGroup& g : shapes.groups) {
            XmlNode& e = add_child(groups, "group");
            set_attr(e, "id", std::to_string(g.id));
            set_attr(e, "label", g.label);
            set_attr(e, "members", id_list(g.members));
        }
    }
}

void write_trajectories(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    if (chunk.trajectories().empty()) {
        return;
    }
    XmlNode& container = add_child(node, "trajectories");
    for (const Trajectory& trajectory : chunk.trajectories()) {
        const std::string entry = trajectory_entry(chunk, trajectory.id);
        archive.add(entry, sidecar::encode_trajectory(trajectory));
        XmlNode& e = add_child(container, "trajectory");
        set_attr(e, "id", std::to_string(trajectory.id));
        set_attr(e, "label", trajectory.label);
        set_attr(e, "count", std::to_string(trajectory.points.size()));
        set_attr(e, "path", entry);
    }
}

void write_panorama(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    if (!chunk.panorama().has_value()) {
        return;
    }
    const PanoramaAsset& panorama = *chunk.panorama();
    XmlNode& e = add_child(node, "panorama");
    write_slot_identity(panorama, e);
    if (panorama.data == nullptr) {
        return;
    }
    const std::string entry = panorama_entry(chunk, panorama.id);
    archive.add(entry, encode_png(*panorama.data));
    set_attr(e, "path", entry);
    set_attr(e, "width", std::to_string(panorama.data->width()));
    set_attr(e, "height", std::to_string(panorama.data->height()));
}

void write_tiled_models(const Chunk& chunk, XmlNode& node, ZipArchive& archive) {
    if (chunk.tiled_models().empty()) {
        return;
    }
    XmlNode& container = add_child(node, "tiled_models");
    for (const TiledModelAsset& asset : chunk.tiled_models()) {
        XmlNode& e = add_child(container, "tiled_model");
        write_slot_identity(asset, e);
        if (asset.id == chunk.active_tiled_model_id()) {
            set_attr(e, "active", "true");
        }
        if (asset.data == nullptr || asset.data->payload.empty()) {
            continue;
        }
        const std::string entry = tiled_model_entry(chunk, asset.id);
        archive.add(entry, sidecar::encode_tiled_model(*asset.data));
        set_attr(e, "path", entry);
        set_attr(e, "size", std::to_string(asset.data->payload.size()));
    }
}

/// The per-object CRS slots (M12 §4.4, CRS-04), written only when a slot was
/// set explicitly. A chunk that follows its own CRS throughout -- every project
/// written before WP-12.6 -- carries no element at all, so its bytes are
/// unchanged. Each slot is its own element rather than an attribute or a child
/// of <crs>, both to keep the persistence of the chunk CRS untouched and so a
/// slot can grow a vertical component later without disturbing its siblings.
void write_crs_slots(const Chunk& chunk, XmlNode& parent) {
    if (chunk.explicit_camera_crs()) {
        write_crs("camera_crs", *chunk.explicit_camera_crs(), parent);
    }
    if (chunk.explicit_marker_crs()) {
        write_crs("marker_crs", *chunk.explicit_marker_crs(), parent);
    }
    if (chunk.explicit_world_crs()) {
        write_crs("world_crs", *chunk.explicit_world_crs(), parent);
    }
}

/// Read the slots back. Absent elements leave the slot unset, which is exactly
/// the "follow the chunk CRS" default; nothing here invents a value, so an old
/// project loads with the same effective CRSs it always had.
void read_crs_slots(const XmlNode& parent, Chunk& chunk, ReadReport* report) {
    if (const XmlNode* camera = find_child(parent, "camera_crs")) {
        warn_unknown_attrs(*camera, {"epsg", "wkt"}, report);
        CRS slot;
        read_crs(camera, slot);
        chunk.set_camera_crs(slot);
    }
    if (const XmlNode* marker = find_child(parent, "marker_crs")) {
        warn_unknown_attrs(*marker, {"epsg", "wkt"}, report);
        CRS slot;
        read_crs(marker, slot);
        chunk.set_marker_crs(slot);
    }
    if (const XmlNode* world = find_child(parent, "world_crs")) {
        warn_unknown_attrs(*world, {"epsg", "wkt"}, report);
        CRS slot;
        read_crs(world, slot);
        chunk.set_world_crs(slot);
    }
}

/// Chunk-level reference accuracies (WP-13.1). Written only when a value
/// differs from the default, so an untouched project carries no <reference>
/// element and re-saves byte-identically. Attribute names are the public Python
/// API spellings, which is what scripts already use.
void write_reference_settings(const Chunk& chunk, XmlNode& parent) {
    const ReferenceSettings& s = chunk.reference_settings();
    if (s.is_default()) {
        return;
    }
    XmlNode& e = add_child(parent, "reference");
    set_attr(e, "camera_location_accuracy", fmt(s.camera_location_accuracy_m));
    set_attr(e, "camera_rotation_accuracy", fmt(s.camera_rotation_accuracy_deg));
    set_attr(e, "marker_location_accuracy", fmt(s.marker_location_accuracy_m));
    set_attr(e, "marker_projection_accuracy", fmt(s.marker_projection_accuracy_px));
    set_attr(e, "scalebar_accuracy", fmt(s.scalebar_accuracy_m));
    set_attr(e, "tiepoint_accuracy", fmt(s.tiepoint_accuracy_px));
    set_attr(e, "use_gcp", fmt_bool(s.use_gcp));
}

/// Read the element back. Absent fields keep the default; an old project has
/// no element at all and therefore loads with the defaults unchanged.
void read_reference_settings(const XmlNode& node, Chunk& chunk, ReadReport* report) {
    const XmlNode* e = find_child(node, "reference");
    if (e == nullptr) {
        return;
    }
    warn_unknown_attrs(*e,
                       {"camera_location_accuracy", "camera_rotation_accuracy",
                        "marker_location_accuracy", "marker_projection_accuracy",
                        "scalebar_accuracy", "tiepoint_accuracy", "use_gcp"},
                       report);
    ReferenceSettings s;
    const auto read = [&](const char* name, double& target) {
        if (const std::string* value = find_attr(*e, name)) {
            target = parse_double(*value);
        }
    };
    read("camera_location_accuracy", s.camera_location_accuracy_m);
    read("camera_rotation_accuracy", s.camera_rotation_accuracy_deg);
    read("marker_location_accuracy", s.marker_location_accuracy_m);
    read("marker_projection_accuracy", s.marker_projection_accuracy_px);
    read("scalebar_accuracy", s.scalebar_accuracy_m);
    read("tiepoint_accuracy", s.tiepoint_accuracy_px);
    s.use_gcp = parse_bool(attr_or(*e, "use_gcp", "true"));
    chunk.set_reference_settings(s);
}

/// The containers every chunk carries, in the fixed order the writer emits
/// them. The reader dispatches over the same set; anything else inside <chunk>
/// is a hard error (v2 is strict about elements, §4.4).
void write_chunk(const Chunk& chunk, XmlNode& parent, ZipArchive& archive) {
    // Enforce every container invariant before a single byte is emitted, so a
    // file the reader would reject can never reach the archive.
    sidecar::validate_tie_points(chunk);
    sidecar::validate_dense(chunk);
    sidecar::validate_mesh(chunk);
    sidecar::validate_model_textures(chunk);
    sidecar::validate_products(chunk);
    sidecar::validate_m11_containers(chunk);

    XmlNode& node = add_child(parent, "chunk");
    set_attr(node, "id", std::to_string(chunk.id()));
    set_attr(node, "label", chunk.label());
    if (chunk.next_id() != 0) {
        set_attr(node, "next_id", std::to_string(chunk.next_id()));
    }

    write_sensors(chunk, node);
    write_cameras(chunk, node);
    write_camera_groups(chunk, node);
    write_frames(chunk, node, archive);
    write_markers(chunk, node);
    write_scalebars(chunk, node);
    write_tie_points(chunk, node, archive);
    write_masks(chunk, node, archive);
    write_depth_map_sets(chunk, node, archive);
    write_point_clouds(chunk, node, archive);
    write_models(chunk, node, archive);
    write_tiled_models(chunk, node, archive);
    write_elevations(chunk, node, archive);
    write_orthomosaics(chunk, node, archive);
    write_trajectories(chunk, node, archive);
    write_shapes(chunk, node, archive);
    write_panorama(chunk, node, archive);

    XmlNode& region = add_child(node, "region");
    set_attr(region, "centre", vec3(chunk.region().centre));
    set_attr(region, "size", vec3(chunk.region().size));
    set_attr(add_child(node, "transform"), "matrix", matrix4(chunk.transform()));
    write_crs("crs", chunk.crs(), node);
    write_crs_slots(chunk, node);
    write_reference_settings(chunk, node);
    write_meta(chunk.meta(), node);
}

}  // namespace docxml_detail

void write_v2(const Document& document, ZipArchive& archive) {
    if (document.workspaces().size() != 1) {
        throw ProjectIOError("project format v2 stores exactly one workspace");
    }
    const Workspace& workspace = document.workspace();
    if (!workspace.meta().items.empty()) {
        throw ProjectIOError("project format v2 cannot store workspace metadata yet");
    }

    XmlNode root{"document", {}, {}, {}};
    set_attr(root, "version", std::to_string(Document::kFormatVersion));
    set_attr(root, "next_id", std::to_string(workspace.next_chunk_id()));
    set_attr(root, "active_id", std::to_string(document.active_chunk_id()));
    write_crs("global_crs", document.global_crs(), root);
    write_meta(document.meta(), root);

    XmlNode& chunks = add_child(root, "chunks");
    for (const auto& chunk : workspace.chunks()) {
        docxml_detail::write_chunk(*chunk, chunks, archive);
    }

    std::string xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    write_node(root, xml, 0);
    archive.add(kDocXmlEntry, xml);
}

// ---------------------------------------------------------------------------
// Reader.
// ---------------------------------------------------------------------------

namespace docxml_detail {

std::vector<std::uint8_t> require_entry(const ZipArchive& archive,
                                        const std::string& path) {
    if (!archive.contains(path)) {
        throw ProjectIOError("project.zip is missing '" + path + "'");
    }
    return archive.get(path);
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
        if (const XmlNode* calibration = find_child(e, "calibration")) {
            warn_unknown_calibration_attrs(*calibration, report);
        }
        // Sensors are positional: a camera's sensor_id is an index into
        // chunk.sensors(), so the stored ids must match that order exactly.
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
        chunk.sensors().push_back(std::move(s));
        ++sensor_index;
    }
}

/// The <reference> child shared by cameras and (in v1) any other element.
CameraReference read_camera_reference(const XmlNode* ref) {
    CameraReference reference;
    if (ref == nullptr) {
        return reference;
    }
    reference.location = Eigen::Vector3d(parse_double(require_attr(*ref, "x")),
                                         parse_double(require_attr(*ref, "y")),
                                         parse_double(require_attr(*ref, "z")));
    // Every component must be finite and positive.
    reference.accuracy = parse_vec3(require_attr(*ref, "accuracy"));
    const Eigen::Vector3d& acc = reference.accuracy;
    if (!std::isfinite(acc.x()) || !std::isfinite(acc.y()) || !std::isfinite(acc.z()) ||
        !(acc.x() > 0.0) || !(acc.y() > 0.0) || !(acc.z() > 0.0)) {
        throw ProjectIOError(
            "doc.xml: <reference> accuracy must be finite and positive, got '" +
            require_attr(*ref, "accuracy") + "'");
    }
    reference.enabled = parse_int(require_attr(*ref, "enabled")) != 0;
    return reference;
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
        c.reference = read_camera_reference(find_child(e, "reference"));
        chunk.cameras().push_back(std::move(c));
    }
}

void read_camera_groups(const XmlNode& node, Chunk& chunk, ReadReport* report) {
    const XmlNode* groups = find_child(node, "camera_groups");
    if (groups == nullptr) {
        return;
    }
    for (const XmlNode& e : groups->children) {
        warn_unknown_attrs(e, {"id", "label", "type", "members", "folder"}, report);
        CameraGroup g;
        g.id = parse_i32(require_attr(e, "id"));
        g.label = require_attr(e, "label");
        g.type = parse_camera_group_type(attr_or(e, "type", "none"));
        g.members = parse_id_list(attr_or(e, "members", std::string()));
        g.folder = attr_or(e, "folder", std::string());
        chunk.camera_groups().push_back(std::move(g));
    }
}

void read_frames(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                 ReadReport* report) {
    const XmlNode* frames = find_child(node, "frames");
    if (frames == nullptr) {
        return;
    }
    warn_unknown_attrs(*frames, {"path"}, report);
    std::vector<std::shared_ptr<Frame>> decoded;
    if (const std::string* path = find_attr(*frames, "path")) {
        decoded = sidecar::decode_frames(require_entry(archive, *path));
    } else {
        // Not fatal: each <frame> element mirrors the fields, so the frames
        // still load. A v2 writer always emits the sidecar, so reaching here
        // means the archive was edited by hand.
        add_warning(report, "doc.xml: <frames> has no frames.bin; using the frame "
                            "elements only");
    }
    // The element list is authoritative for framing; the sidecar holds the
    // same frames, so a mismatch means the project was edited inconsistently.
    const std::vector<XmlNode>& elements = frames->children;
    if (!decoded.empty() && decoded.size() != elements.size()) {
        throw ProjectIOError("doc.xml: <frames> element count does not match frames.bin");
    }
    for (std::size_t i = 0; i < elements.size(); ++i) {
        const XmlNode& e = elements[i];
        warn_unknown_attrs(e, {"id", "path", "width", "height", "camera_id"}, report);
        std::shared_ptr<Frame> f;
        if (i < decoded.size()) {
            f = decoded[i];
            // The element mirror must agree with the binary entry, so a
            // half-edited project cannot pass silently.
            if (f->id() != parse_i32(require_attr(e, "id")) ||
                f->path() != require_attr(e, "path") ||
                f->width() != parse_int(require_attr(e, "width")) ||
                f->height() != parse_int(require_attr(e, "height")) ||
                f->camera_id() != parse_i32(require_attr(e, "camera_id"))) {
                throw ProjectIOError("doc.xml: <frame> does not match frames.bin");
            }
        } else {
            f = std::make_shared<Frame>(require_attr(e, "path"),
                                        parse_int(require_attr(e, "width")),
                                        parse_int(require_attr(e, "height")));
            f->set_id(parse_i32(require_attr(e, "id")));
            f->set_camera_id(parse_i32(require_attr(e, "camera_id")));
        }
        chunk.frames().push_back(std::move(f));
    }
}

void read_markers(const XmlNode& node, Chunk& chunk, ReadReport* report) {
    const XmlNode* markers = find_child(node, "markers");
    if (markers == nullptr) {
        return;
    }
    for (const XmlNode& e : markers->children) {
        if (e.name == "marker_groups") {
            for (const XmlNode& g : e.children) {
                warn_unknown_attrs(g, {"id", "label", "members"}, report);
                MarkerGroup group;
                group.id = parse_i32(require_attr(g, "id"));
                group.label = require_attr(g, "label");
                group.members = parse_id_list(attr_or(g, "members", std::string()));
                chunk.marker_groups().push_back(std::move(group));
            }
            continue;
        }
        if (e.name != "marker") {
            throw ProjectIOError("doc.xml: unknown <markers> child <" + e.name + ">");
        }
        warn_unknown_attrs(e,
                           {"id", "label", "group", "type", "projections", "location",
                            "reference_accuracy", "location_estimate",
                            "position_covariance", "pinned", "valid", "internal"},
                           report);
        Marker marker;
        marker.id = parse_i32(require_attr(e, "id"));
        marker.label = require_attr(e, "label");
        marker.group_id = find_attr(e, "group") != nullptr
                              ? parse_i32(require_attr(e, "group"))
                              : -1;
        marker.type = parse_marker_type(attr_or(e, "type", "regular"));
        for (const XmlNode& proj : e.children) {
            if (proj.name == "meta") {
                read_meta(&proj, marker.meta);
                continue;
            }
            if (proj.name != "projection") {
                continue;
            }
            MarkerProjection p;
            p.camera_id = parse_i32(require_attr(proj, "camera_id"));
            const auto px = parse_pair(require_attr(proj, "pixel"), "<projection> pixel");
            p.pixel = Eigen::Vector2d(px.first, px.second);
            p.enabled = attr_or(proj, "enabled", "true") != "false";
            marker.projections.push_back(p);
        }
        if (const std::string* location = find_attr(e, "location")) {
            marker.reference = parse_vec3(*location);
            marker.reference_accuracy =
                parse_vec3(attr_or(e, "reference_accuracy",
                                   vec3(Eigen::Vector3d::Constant(0.5))));
        }
        if (const std::string* estimate = find_attr(e, "location_estimate")) {
            marker.position = parse_vec3(*estimate);
            marker.position_covariance =
                parse_covariance9(attr_or(e, "position_covariance",
                                          covariance9(Eigen::Matrix3d::Zero())));
        }
        marker.pinned = attr_or(e, "pinned", "0") != "0";
        marker.valid = attr_or(e, "valid", "0") != "0";
        marker.internal = attr_or(e, "internal", "0") != "0";
        chunk.markers().push_back(std::move(marker));
    }
}

void read_scalebars(const XmlNode& node, Chunk& chunk, ReadReport* report) {
    const XmlNode* scalebars = find_child(node, "scalebars");
    if (scalebars == nullptr) {
        return;
    }
    for (const XmlNode& e : scalebars->children) {
        warn_unknown_attrs(e,
                           {"id", "label", "from_kind", "from_id", "to_kind", "to_id",
                            "reference_length", "reference_accuracy",
                            "estimated_length"},
                           report);
        Scalebar bar;
        bar.id = parse_i32(require_attr(e, "id"));
        bar.label = require_attr(e, "label");
        bar.from.kind = parse_endpoint_kind(require_attr(e, "from_kind"));
        bar.from.id = parse_i32(require_attr(e, "from_id"));
        bar.to.kind = parse_endpoint_kind(require_attr(e, "to_kind"));
        bar.to.id = parse_i32(require_attr(e, "to_id"));
        bar.reference_length_m = parse_double(require_attr(e, "reference_length"));
        bar.reference_accuracy_m = parse_double(require_attr(e, "reference_accuracy"));
        bar.estimated_length_m = parse_double(require_attr(e, "estimated_length"));
        chunk.scalebars().push_back(std::move(bar));
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
            throw ProjectIOError("doc.xml: unknown <tie_points> child <" + e.name + ">");
        }
        const std::string path = require_attr(e, "path");
        if (e.name == "keypoint_set") {
            warn_unknown_attrs(e, {"camera_id", "count", "path"}, report);
            KeypointSet set;
            set.camera_id = parse_i32(require_attr(e, "camera_id"));
            sidecar::decode_keypoints(require_entry(archive, path), set);
            if (set.keypoints.size() !=
                static_cast<std::size_t>(parse_ll(require_attr(e, "count")))) {
                throw ProjectIOError("keypoint count mismatch for camera " +
                                     std::to_string(set.camera_id));
            }
            chunk.tie_points().keypoint_sets.push_back(std::move(set));
        } else if (e.name == "matches") {
            warn_unknown_attrs(e, {"pairs", "path"}, report);
            chunk.tie_points().matches =
                sidecar::decode_matches(require_entry(archive, path));
            if (chunk.tie_points().matches.size() !=
                static_cast<std::size_t>(parse_ll(require_attr(e, "pairs")))) {
                throw ProjectIOError("matches count mismatch in doc.xml");
            }
        } else {
            warn_unknown_attrs(e, {"count", "path"}, report);
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

void read_masks(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                ReadReport* report) {
    const XmlNode* sets = find_child(node, "mask_sets");
    if (sets == nullptr) {
        return;
    }
    const XmlNode* masks = find_child(*sets, "masks");
    if (masks == nullptr) {
        return;
    }
    for (const XmlNode& e : masks->children) {
        warn_unknown_attrs(e, {"camera_id", "width", "height", "invert", "path"}, report);
        Mask mask = sidecar::decode_mask(require_entry(archive, require_attr(e, "path")));
        if (mask.camera_id != parse_int(require_attr(e, "camera_id")) ||
            mask.width != parse_int(require_attr(e, "width")) ||
            mask.height != parse_int(require_attr(e, "height"))) {
            throw ProjectIOError("doc.xml: <mask> header does not match its sidecar");
        }
        if (attr_or(e, "invert", "false") == "true") {
            mask.invert = true;
        }
        chunk.masks().push_back(std::move(mask));
    }
}

// -- assets ----------------------------------------------------------------

void read_depth_map_sets(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                         ReadReport* report) {
    const XmlNode* container = find_child(node, "depth_map_sets");
    if (container == nullptr) {
        return;
    }
    for (const XmlNode& e : container->children) {
        warn_unknown_attrs(e,
                           {"id", "label", "group", "enabled", "active", "count",
                            "path"},
                           report);
        DepthMapsAsset slot;
        read_slot_identity(e, slot);
        if (const std::string* path = find_attr(e, "path")) {
            *slot.data = sidecar::decode_depth_maps(require_entry(archive, *path));
            if (slot.data->size() !=
                static_cast<std::size_t>(parse_ll(require_attr(e, "count")))) {
                throw ProjectIOError("depth map count mismatch in doc.xml");
            }
        }
        if (attr_or(e, "active", "false") == "true") {
            chunk.set_active_depth_maps_id(slot.id);
        }
        chunk.depth_map_sets().push_back(std::move(slot));
    }
    sidecar::validate_dense(chunk);
}

void read_point_clouds(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                       ReadReport* report) {
    const XmlNode* container = find_child(node, "dense_clouds");
    if (container == nullptr) {
        return;
    }
    for (const XmlNode& e : container->children) {
        warn_unknown_attrs(e,
                           {"id", "label", "group", "enabled", "active", "count",
                            "path", "hasPointColors", "hasPointNormals",
                            "hasPointConfidence"},
                           report);
        PointCloudAsset slot;
        read_slot_identity(e, slot);
        if (const std::string* path = find_attr(e, "path")) {
            *slot.data = sidecar::decode_point_cloud(require_entry(archive, *path));
            if (slot.data->points.size() !=
                static_cast<std::size_t>(parse_ll(require_attr(e, "count")))) {
                throw ProjectIOError("point cloud count mismatch in doc.xml");
            }
        }
        if (attr_or(e, "active", "false") == "true") {
            chunk.set_active_point_cloud_id(slot.id);
        }
        chunk.point_clouds().push_back(std::move(slot));
    }
    sidecar::validate_dense(chunk);
}

void read_models(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                 ReadReport* report) {
    const XmlNode* container = find_child(node, "models");
    if (container == nullptr) {
        return;
    }
    for (const XmlNode& e : container->children) {
        warn_unknown_attrs(e,
                           {"id", "label", "group", "enabled", "active", "path",
                            "source_id", "faceCount", "vertexCount",
                            "hasVertexColors", "hasVertexConfidence", "hasUV"},
                           report);
        ModelAsset slot;
        read_slot_identity(e, slot);
        if (const std::string* source = find_attr(e, "source_id")) {
            slot.source_id = parse_ll(*source);
        }
        if (const std::string* path = find_attr(e, "path")) {
            *slot.data = sidecar::decode_mesh(require_entry(archive, *path));
            if (slot.data->faces.size() !=
                    static_cast<std::size_t>(parse_ll(require_attr(e, "faceCount"))) ||
                slot.data->vertices.size() !=
                    static_cast<std::size_t>(parse_ll(require_attr(e, "vertexCount")))) {
                throw ProjectIOError("mesh counts mismatch in doc.xml");
            }
        }
        if (const XmlNode* texture = find_child(e, "texture")) {
            warn_unknown_attrs(*texture, {"pages"}, report);
            const long long pages = parse_ll(attr_or(*texture, "pages", "0"));
            if (pages < 0) {
                throw ProjectIOError("mesh texture count is negative in doc.xml");
            }
            std::vector<const XmlNode*> page_nodes = find_children(*texture, "page");
            if (static_cast<long long>(page_nodes.size()) != pages) {
                throw ProjectIOError("doc.xml: <texture> page count does not match");
            }
            for (const XmlNode* page : page_nodes) {
                warn_unknown_attrs(*page, {"path"}, report);
                const std::vector<std::uint8_t> bytes =
                    require_entry(archive, require_attr(*page, "path"));
                slot.textures.push_back(
                    image_detail::png_decode(bytes.data(), bytes.size()));
            }
        }
        if (attr_or(e, "active", "false") == "true") {
            chunk.set_active_model_id(slot.id);
        }
        chunk.models().push_back(std::move(slot));
    }
    sidecar::validate_mesh(chunk);
    sidecar::validate_model_textures(chunk);
}


void read_elevations(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                     ReadReport* report) {
    const XmlNode* container = find_child(node, "elevations");
    if (container == nullptr) {
        return;
    }
    for (const XmlNode& e : container->children) {
        warn_unknown_attrs(e,
                           {"id", "label", "group", "enabled", "active", "path",
                            "width", "height", "origin", "pixel_size", "epsg",
                            "crs_wkt", "nodataValue", "source_data"},
                           report);
        ElevationAsset slot;
        read_slot_identity(e, slot);
        if (const std::string* path = find_attr(e, "path")) {
            Elevation dem = sidecar::decode_elevation(require_entry(archive, *path));
            dem.source = attr_or(e, "source_data", std::string());
            const RasterGrid& g = dem.raster.grid;
            if (g.width != parse_int(require_attr(e, "width")) ||
                g.height != parse_int(require_attr(e, "height"))) {
                throw ProjectIOError(
                    "doc.xml: <elevation> dimensions do not match the GeoTIFF");
            }
            const auto origin =
                parse_pair(require_attr(e, "origin"), "<elevation> origin");
            const auto pixel_size =
                parse_pair(require_attr(e, "pixel_size"), "<elevation> pixel_size");
            if (origin.first != g.origin_x || origin.second != g.origin_y ||
                pixel_size.first != g.pixel_size_x ||
                pixel_size.second != g.pixel_size_y) {
                throw ProjectIOError(
                    "doc.xml: <elevation> georeferencing does not match the GeoTIFF");
            }
            if (parse_int(require_attr(e, "epsg")) != dem.epsg()) {
                throw ProjectIOError(
                    "doc.xml: <elevation> epsg does not match the GeoTIFF");
            }
            // The WKT is optional: an EPSG-only element (any pre-M12 project)
            // leaves `dem.crs` as the decoded code, which is the same CRS.
            dem.crs.wkt = attr_or(e, "crs_wkt", std::string());
            dem.nodata =
                static_cast<float>(parse_double(require_attr(e, "nodataValue")));
            *slot.data = std::move(dem);
        }
        if (attr_or(e, "active", "false") == "true") {
            chunk.set_active_elevation_id(slot.id);
        }
        chunk.elevations().push_back(std::move(slot));
    }
    sidecar::validate_products(chunk);
}

void read_orthomosaics(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                       ReadReport* report) {
    const XmlNode* container = find_child(node, "orthomosaics");
    if (container == nullptr) {
        return;
    }
    for (const XmlNode& e : container->children) {
        warn_unknown_attrs(e,
                           {"id", "label", "group", "enabled", "active", "path",
                            "labels_path", "width", "height", "origin",
                            "pixel_size", "epsg", "crs_wkt", "blendingMode"},
                           report);
        OrthomosaicAsset slot;
        read_slot_identity(e, slot);
        if (const std::string* path = find_attr(e, "path")) {
            Orthomosaic mosaic = sidecar::decode_orthomosaic(require_entry(archive, *path));
            mosaic.surface = attr_or(e, "blendingMode", std::string());
            const RasterGrid& g = mosaic.raster.grid;
            if (g.width != parse_int(require_attr(e, "width")) ||
                g.height != parse_int(require_attr(e, "height"))) {
                throw ProjectIOError(
                    "doc.xml: <orthomosaic> dimensions do not match the GeoTIFF");
            }
            const auto origin =
                parse_pair(require_attr(e, "origin"), "<orthomosaic> origin");
            const auto pixel_size =
                parse_pair(require_attr(e, "pixel_size"), "<orthomosaic> pixel_size");
            if (origin.first != g.origin_x || origin.second != g.origin_y ||
                pixel_size.first != g.pixel_size_x ||
                pixel_size.second != g.pixel_size_y) {
                throw ProjectIOError(
                    "doc.xml: <orthomosaic> georeferencing does not match the GeoTIFF");
            }
            if (parse_int(require_attr(e, "epsg")) != mosaic.epsg()) {
                throw ProjectIOError(
                    "doc.xml: <orthomosaic> epsg does not match the GeoTIFF");
            }
            mosaic.crs.wkt = attr_or(e, "crs_wkt", std::string());
            mosaic.labels = sidecar::decode_ortho_labels(
                require_entry(archive, require_attr(e, "labels_path")), g.width, g.height);
            *slot.data = std::move(mosaic);
        }
        if (attr_or(e, "active", "false") == "true") {
            chunk.set_active_orthomosaic_id(slot.id);
        }
        chunk.orthomosaics().push_back(std::move(slot));
    }
    sidecar::validate_products(chunk);
}

void read_shapes(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                 ReadReport* report) {
    const XmlNode* container = find_child(node, "shapes");
    if (container == nullptr) {
        return;
    }
    warn_unknown_attrs(*container, {"path", "count"}, report);
    if (const std::string* path = find_attr(*container, "path")) {
        chunk.shapes() = sidecar::decode_shapes(require_entry(archive, *path));
        if (chunk.shapes().features.size() !=
            static_cast<std::size_t>(parse_ll(require_attr(*container, "count")))) {
            throw ProjectIOError("doc.xml: <shapes> count does not match shapes.shp");
        }
    }
    // The XML mirror of each shape; the sidecar is authoritative for geometry
    // and attributes, but the element list must agree with it.
    std::vector<const XmlNode*> elements = find_children(*container, "shape");
    if (!elements.empty()) {
        if (elements.size() != chunk.shapes().features.size()) {
            throw ProjectIOError("doc.xml: <shape> element count does not match shapes.shp");
        }
        for (std::size_t i = 0; i < elements.size(); ++i) {
            const Shape& stored = chunk.shapes().features[i];
            const XmlNode& e = *elements[i];
            warn_unknown_attrs(e, {"id", "type", "label", "layer", "closed"}, report);
            if (parse_i32(require_attr(e, "id")) != stored.id ||
                parse_shape_type(require_attr(e, "type")) != stored.type ||
                require_attr(e, "label") != stored.label ||
                attr_or(e, "layer", "Default") != stored.layer) {
                throw ProjectIOError("doc.xml: <shape> element does not match shapes.shp");
            }
        }
    }
    // Groups live in shapes.shp; the XML mirror must agree with it rather than
    // being appended a second time.
    if (const XmlNode* groups = find_child(*container, "shape_groups")) {
        const std::vector<const XmlNode*> group_nodes = find_children(*groups, "group");
        if (group_nodes.size() != chunk.shapes().groups.size()) {
            throw ProjectIOError(
                "doc.xml: <shape_groups> count does not match shapes.shp");
        }
        for (std::size_t i = 0; i < group_nodes.size(); ++i) {
            const ShapeGroup& stored = chunk.shapes().groups[i];
            const XmlNode& g = *group_nodes[i];
            warn_unknown_attrs(g, {"id", "label", "members"}, report);
            if (parse_i32(require_attr(g, "id")) != stored.id ||
                require_attr(g, "label") != stored.label ||
                parse_id_list(attr_or(g, "members", std::string())) != stored.members) {
                throw ProjectIOError(
                    "doc.xml: <shape_groups> entry does not match shapes.shp");
            }
        }
    }
}

void read_trajectories(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                       ReadReport* report) {
    const XmlNode* container = find_child(node, "trajectories");
    if (container == nullptr) {
        return;
    }
    for (const XmlNode& e : container->children) {
        warn_unknown_attrs(e, {"id", "label", "count", "path"}, report);
        Trajectory trajectory =
            sidecar::decode_trajectory(require_entry(archive, require_attr(e, "path")));
        if (trajectory.id != parse_i32(require_attr(e, "id")) ||
            trajectory.label != require_attr(e, "label") ||
            trajectory.points.size() !=
                static_cast<std::size_t>(parse_ll(require_attr(e, "count")))) {
            throw ProjectIOError("doc.xml: <trajectory> does not match its sidecar");
        }
        chunk.trajectories().push_back(std::move(trajectory));
    }
}

void read_panorama(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                   ReadReport* report) {
    const XmlNode* e = find_child(node, "panorama");
    if (e == nullptr) {
        return;
    }
    warn_unknown_attrs(*e, {"id", "label", "group", "enabled", "path"}, report);
    PanoramaAsset panorama;
    read_slot_identity(*e, panorama);
    if (const std::string* path = find_attr(*e, "path")) {
        const std::vector<std::uint8_t> bytes = require_entry(archive, *path);
        panorama.data = std::make_shared<Image>(
            image_detail::png_decode(bytes.data(), bytes.size()));
    } else {
        add_warning(report, "doc.xml: <panorama> entry has no image; it loads empty");
    }
    chunk.panorama() = std::move(panorama);
}

void read_tiled_models(const XmlNode& node, Chunk& chunk, const ZipArchive& archive,
                       ReadReport* report) {
    const XmlNode* container = find_child(node, "tiled_models");
    if (container == nullptr) {
        return;
    }
    for (const XmlNode& e : container->children) {
        warn_unknown_attrs(e,
                           {"id", "label", "group", "enabled", "active", "path",
                            "size"},
                           report);
        TiledModelAsset slot;
        read_slot_identity(e, slot);
        if (const std::string* path = find_attr(e, "path")) {
            *slot.data = sidecar::decode_tiled_model(require_entry(archive, *path));
            if (slot.data->payload.size() !=
                static_cast<std::size_t>(parse_ll(require_attr(e, "size")))) {
                throw ProjectIOError("doc.xml: <tiled_model> size does not match");
            }
        }
        if (attr_or(e, "active", "false") == "true") {
            chunk.set_active_tiled_model_id(slot.id);
        }
        chunk.tiled_models().push_back(std::move(slot));
    }
}

/// The set of element names a v2 <chunk> may carry (§4.4). Anything else is a
/// hard error: v2 is strict inside <chunk> so writer/reader drift cannot pass
/// silently.
void check_chunk_vocabulary(const XmlNode& node) {
    static const char* const kAllowed[] = {
        "sensors",   "cameras",    "camera_groups", "frames",     "markers",
        "scalebars", "tie_points", "mask_sets",     "depth_map_sets",
        "dense_clouds", "models",  "tiled_models",  "elevations", "orthomosaics",
        "trajectories", "shapes",  "panorama",      "region",     "transform",
        "crs",       "meta",       "camera_crs",    "marker_crs", "world_crs",
        "reference",
    };
    for (const XmlNode& child : node.children) {
        bool known = false;
        for (const char* name : kAllowed) {
            if (child.name == name) {
                known = true;
                break;
            }
        }
        if (!known) {
            throw ProjectIOError("doc.xml: unknown element <" + child.name +
                                 "> in chunk (format v2)");
        }
    }
}

void read_chunk(const XmlNode& node, Workspace& workspace, const ZipArchive& archive,
                ReadReport* report) {
    check_chunk_vocabulary(node);
    warn_unknown_attrs(node, {"id", "label", "next_id"}, report);
    Chunk& chunk = workspace.restore_chunk(parse_i32(require_attr(node, "id")),
                                           require_attr(node, "label"));
    docxml_detail::read_sensors(node, chunk, report);
    docxml_detail::read_cameras(node, chunk, report);
    docxml_detail::read_camera_groups(node, chunk, report);
    docxml_detail::read_frames(node, chunk, archive, report);
    warn_missing_frame_files(chunk.frames(), report);
    docxml_detail::read_markers(node, chunk, report);
    docxml_detail::read_scalebars(node, chunk, report);
    docxml_detail::read_tie_points(node, chunk, archive, report);
    docxml_detail::read_masks(node, chunk, archive, report);
    docxml_detail::read_depth_map_sets(node, chunk, archive, report);
    docxml_detail::read_point_clouds(node, chunk, archive, report);
    docxml_detail::read_models(node, chunk, archive, report);
    docxml_detail::read_tiled_models(node, chunk, archive, report);
    docxml_detail::read_elevations(node, chunk, archive, report);
    docxml_detail::read_orthomosaics(node, chunk, archive, report);
    docxml_detail::read_trajectories(node, chunk, archive, report);
    docxml_detail::read_shapes(node, chunk, archive, report);
    docxml_detail::read_panorama(node, chunk, archive, report);

    if (const XmlNode* region = find_child(node, "region")) {
        warn_unknown_attrs(*region, {"centre", "size"}, report);
        chunk.region().centre = parse_vec3(require_attr(*region, "centre"));
        chunk.region().size = parse_vec3(require_attr(*region, "size"));
    }
    if (const XmlNode* transform = find_child(node, "transform")) {
        warn_unknown_attrs(*transform, {"matrix"}, report);
        chunk.transform() = parse_matrix4(require_attr(*transform, "matrix"));
    }
    read_crs(find_child(node, "crs"), chunk.crs());
    read_crs_slots(node, chunk, report);
    read_reference_settings(node, chunk, report);
    read_meta(find_child(node, "meta"), chunk.meta());

    // The id allocator, when the writer recorded it, and then reserve every
    // restored id so a later allocate_id() cannot collide with one of them.
    if (const std::string* next_id = find_attr(node, "next_id")) {
        chunk.set_next_id(parse_i32(*next_id));
    }
    for (const auto& asset : chunk.elevations()) chunk.reserve_id(asset.id);
    for (const auto& asset : chunk.orthomosaics()) chunk.reserve_id(asset.id);
    for (const auto& asset : chunk.models()) chunk.reserve_id(asset.id);
    for (const auto& asset : chunk.point_clouds()) chunk.reserve_id(asset.id);
    for (const auto& asset : chunk.depth_map_sets()) chunk.reserve_id(asset.id);
    for (const auto& asset : chunk.tiled_models()) chunk.reserve_id(asset.id);
    if (chunk.panorama().has_value()) chunk.reserve_id(chunk.panorama()->id);
    for (const auto& marker : chunk.markers()) chunk.reserve_id(marker.id);
    for (const auto& group : chunk.marker_groups()) chunk.reserve_id(group.id);
    for (const auto& bar : chunk.scalebars()) chunk.reserve_id(bar.id);
    for (const auto& group : chunk.camera_groups()) chunk.reserve_id(group.id);
    for (const auto& shape : chunk.shapes().features) chunk.reserve_id(shape.id);
    for (const auto& group : chunk.shapes().groups) chunk.reserve_id(group.id);
    for (const auto& trajectory : chunk.trajectories()) chunk.reserve_id(trajectory.id);

    // A hand-written project may omit the "active" flag; then the first asset
    // of each kind becomes active rather than leaving an invalid -1 behind.
    if (chunk.active_depth_maps_id() == -1 && !chunk.depth_map_sets().empty()) {
        chunk.set_active_depth_maps_id(chunk.depth_map_sets().front().id);
    }
    if (chunk.active_point_cloud_id() == -1 && !chunk.point_clouds().empty()) {
        chunk.set_active_point_cloud_id(chunk.point_clouds().front().id);
    }
    if (chunk.active_model_id() == -1 && !chunk.models().empty()) {
        chunk.set_active_model_id(chunk.models().front().id);
    }
    if (chunk.active_elevation_id() == -1 && !chunk.elevations().empty()) {
        chunk.set_active_elevation_id(chunk.elevations().front().id);
    }
    if (chunk.active_orthomosaic_id() == -1 && !chunk.orthomosaics().empty()) {
        chunk.set_active_orthomosaic_id(chunk.orthomosaics().front().id);
    }
    if (chunk.active_tiled_model_id() == -1 && !chunk.tiled_models().empty()) {
        chunk.set_active_tiled_model_id(chunk.tiled_models().front().id);
    }

    sidecar::validate_m11_containers(chunk);
}

}  // namespace docxml_detail

std::shared_ptr<Document> read_v2(const XmlNode& root, const ZipArchive& archive,
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
                docxml_detail::read_chunk(node, workspace, archive, report);
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
