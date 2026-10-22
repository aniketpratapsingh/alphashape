// tests/v1_writer.hpp -- the pre-M11 doc.xml writer, kept for tests only.
//
// WP-11.4 needs v1 projects to test the versioned reader against, and the
// production writer must never emit v1 again. This header reproduces the writer
// exactly as it stood before the split (c4acca0, when doc.xml.cpp held both
// directions): version="1", the flat sidecar entry names, and one asset per
// kind (the active one). Nothing in src/ includes it, so the "read v1" promise
// is verified against the layout that actually shipped rather than against a
// hand-written guess.
//
// Sidecar payloads come from the shared encoders in common/io/sidecars.hpp,
// which are version independent, so this file only reproduces the *XML*
// structure and the entry names. The one deliberate addition is
// frames.bin: v1 stored frames only as XML elements, so write_project emits
// <frames> without a sidecar path and the v2 reader falls back to the elements.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "base/io/zip.hpp"
#include "common/core/document.hpp"
#include "common/io/doc_xml_common.hpp"
#include "common/io/sidecars.hpp"
#include "image/io/image_io.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/point_cloud.hpp"
#include "model/raster/geotiff.hpp"

namespace alphashape::test_v1 {

namespace detail {

constexpr const char* kMatchesEntry = "tie_points/matches.bin";
constexpr const char* kTracksEntry = "tie_points/tracks.bin";
constexpr const char* kDepthMapsEntry = "depth_maps/depth_maps.bin";
constexpr const char* kPointCloudEntry = "point_cloud/points.bin";
constexpr const char* kMeshEntry = "model/mesh.bin";
constexpr const char* kModelTexturePrefix = "model/texture_";
constexpr const char* kElevationEntry = "elevation/dem.tif";
constexpr const char* kOrthomosaicEntry = "orthomosaic/ortho.tif";
constexpr const char* kOrthoLabelsEntry = "orthomosaic/labels.bin";

void write_sensors(const Chunk& chunk, docxml::XmlNode& node) {
    docxml::XmlNode& sensors = docxml::add_child(node, "sensors");
    for (std::size_t i = 0; i < chunk.sensors().size(); ++i) {
        const Sensor& s = chunk.sensors()[i];
        docxml::XmlNode& e = docxml::add_child(sensors, "sensor");
        docxml::set_attr(e, "id", std::to_string(i));
        docxml::set_attr(e, "label", s.label);
        docxml::set_attr(e, "type", docxml::camera_type_name(s.type));
        docxml::set_attr(e, "width", std::to_string(s.width));
        docxml::set_attr(e, "height", std::to_string(s.height));
        docxml::set_attr(e, "pixel_size", docxml::fmt(s.pixel_size_um));
        docxml::set_attr(e, "focal_length", docxml::fmt(s.focal_length_mm));
        docxml::set_attr(e, "make", s.make);
        docxml::set_attr(e, "model", s.model);
        docxml::set_attr(e, "serial", s.serial);
        docxml::write_calibration(s.calibration, e);
        docxml::set_attr(e, "fixed_params", docxml::join_strings(s.fixed_params));
    }
}

void write_cameras(const Chunk& chunk, docxml::XmlNode& node) {
    docxml::XmlNode& cameras = docxml::add_child(node, "cameras");
    for (const Camera& c : chunk.cameras()) {
        docxml::XmlNode& e = docxml::add_child(cameras, "camera");
        docxml::set_attr(e, "id", std::to_string(c.id));
        docxml::set_attr(e, "sensor_id", std::to_string(c.sensor_id));
        docxml::set_attr(e, "label", c.label);
        docxml::set_attr(e, "enabled", c.enabled ? "true" : "false");
        docxml::set_attr(e, "orientation", std::to_string(c.orientation));
        docxml::set_attr(e, "transform", docxml::matrix4(c.pose));
        if (c.reference.location.has_value()) {
            docxml::XmlNode& ref = docxml::add_child(e, "reference");
            docxml::set_attr(ref, "x", docxml::fmt((*c.reference.location).x()));
            docxml::set_attr(ref, "y", docxml::fmt((*c.reference.location).y()));
            docxml::set_attr(ref, "z", docxml::fmt((*c.reference.location).z()));
            docxml::set_attr(ref, "accuracy", docxml::vec3(c.reference.accuracy));
            docxml::set_attr(ref, "enabled", c.reference.enabled ? "1" : "0");
        }
    }
}

void write_frames(const Chunk& chunk, docxml::XmlNode& node) {
    docxml::XmlNode& frames = docxml::add_child(node, "frames");
    for (const auto& f : chunk.frames()) {
        docxml::XmlNode& e = docxml::add_child(frames, "frame");
        docxml::set_attr(e, "id", std::to_string(f->id()));
        docxml::set_attr(e, "path", f->path());
        docxml::set_attr(e, "width", std::to_string(f->width()));
        docxml::set_attr(e, "height", std::to_string(f->height()));
        docxml::set_attr(e, "camera_id", std::to_string(f->camera_id()));
    }
}

std::string keypoint_entry(std::int32_t camera_id) {
    return "tie_points/" + std::to_string(camera_id) + ".kpt";
}

void write_tie_points(const Chunk& chunk, docxml::XmlNode& node, ZipArchive& archive) {
    docxml::XmlNode& tie_points = docxml::add_child(node, "tie_points");
    for (const KeypointSet& set : chunk.tie_points().keypoint_sets) {
        const std::string entry = keypoint_entry(set.camera_id);
        archive.add(entry, docxml::sidecar::encode_keypoints(set));
        docxml::XmlNode& e = docxml::add_child(tie_points, "keypoint_set");
        docxml::set_attr(e, "camera_id", std::to_string(set.camera_id));
        docxml::set_attr(e, "count", std::to_string(set.keypoints.size()));
        docxml::set_attr(e, "path", entry);
    }
    if (!chunk.tie_points().matches.empty()) {
        archive.add(kMatchesEntry,
                    docxml::sidecar::encode_matches(chunk.tie_points().matches));
        docxml::XmlNode& e = docxml::add_child(tie_points, "matches");
        docxml::set_attr(e, "pairs", std::to_string(chunk.tie_points().matches.size()));
        docxml::set_attr(e, "path", kMatchesEntry);
    }
    if (!chunk.tie_points().tracks.empty()) {
        archive.add(kTracksEntry,
                    docxml::sidecar::encode_tracks(chunk.tie_points().tracks));
        docxml::XmlNode& e = docxml::add_child(tie_points, "tracks");
        docxml::set_attr(e, "count", std::to_string(chunk.tie_points().tracks.size()));
        docxml::set_attr(e, "path", kTracksEntry);
    }
}

/// Dense products, mesh with texture pages, DEM and orthomosaic -- each written
/// only when non-empty, exactly the pre-M11 conditionals.
void write_assets(const Chunk& chunk, docxml::XmlNode& node, ZipArchive& archive) {
    const std::vector<DepthMap>& depth_maps = chunk.depth_maps();
    if (!depth_maps.empty()) {
        archive.add(kDepthMapsEntry, docxml::sidecar::encode_depth_maps(depth_maps));
        docxml::XmlNode& e = docxml::add_child(node, "depth_maps");
        docxml::set_attr(e, "count", std::to_string(depth_maps.size()));
        docxml::set_attr(e, "path", kDepthMapsEntry);
    }
    const PointCloud& cloud = chunk.point_cloud();
    if (!cloud.points.empty()) {
        archive.add(kPointCloudEntry, docxml::sidecar::encode_point_cloud(cloud));
        docxml::XmlNode& e = docxml::add_child(node, "point_cloud");
        docxml::set_attr(e, "count", std::to_string(cloud.points.size()));
        docxml::set_attr(e, "path", kPointCloudEntry);
    }
    const Mesh& mesh = chunk.model();
    if (!mesh.empty()) {
        archive.add(kMeshEntry, docxml::sidecar::encode_mesh(mesh));
        docxml::XmlNode& e = docxml::add_child(node, "model");
        docxml::set_attr(e, "faces", std::to_string(mesh.faces.size()));
        docxml::set_attr(e, "vertices", std::to_string(mesh.vertices.size()));
        docxml::set_attr(e, "path", kMeshEntry);
        const std::vector<Image>& textures = chunk.model_textures();
        docxml::set_attr(e, "textures", std::to_string(textures.size()));
        for (std::size_t page = 0; page < textures.size(); ++page) {
            archive.add(kModelTexturePrefix + std::to_string(page) + ".png",
                        encode_png(textures[page]));
        }
    }
    const Elevation& elevation = chunk.elevation();
    if (!elevation.empty()) {
        archive.add(kElevationEntry, docxml::sidecar::encode_elevation(elevation));
        const RasterGrid& g = elevation.raster.grid;
        docxml::XmlNode& e = docxml::add_child(node, "elevation");
        docxml::set_attr(e, "path", kElevationEntry);
        docxml::set_attr(e, "width", std::to_string(g.width));
        docxml::set_attr(e, "height", std::to_string(g.height));
        docxml::set_attr(e, "origin", docxml::vec2(g.origin_x, g.origin_y));
        docxml::set_attr(e, "pixel_size", docxml::vec2(g.pixel_size_x, g.pixel_size_y));
        docxml::set_attr(e, "epsg", std::to_string(elevation.epsg));
        docxml::set_attr(e, "nodata",
                         docxml::fmt(static_cast<double>(elevation.nodata)));
        docxml::set_attr(e, "source", elevation.source);
    }
    const Orthomosaic& ortho = chunk.orthomosaic();
    if (!ortho.empty()) {
        archive.add(kOrthomosaicEntry, docxml::sidecar::encode_orthomosaic(ortho));
        archive.add(kOrthoLabelsEntry, docxml::sidecar::encode_ortho_labels(ortho));
        const RasterGrid& g = ortho.raster.grid;
        docxml::XmlNode& e = docxml::add_child(node, "orthomosaic");
        docxml::set_attr(e, "path", kOrthomosaicEntry);
        docxml::set_attr(e, "labels_path", kOrthoLabelsEntry);
        docxml::set_attr(e, "width", std::to_string(g.width));
        docxml::set_attr(e, "height", std::to_string(g.height));
        docxml::set_attr(e, "origin", docxml::vec2(g.origin_x, g.origin_y));
        docxml::set_attr(e, "pixel_size", docxml::vec2(g.pixel_size_x, g.pixel_size_y));
        docxml::set_attr(e, "epsg", std::to_string(ortho.epsg));
        docxml::set_attr(e, "surface", ortho.surface);
    }
}

}  // namespace detail

/// Serialise `document` exactly as the pre-M11 writer did, into a new zip.
std::vector<std::uint8_t> write_project(const Document& document) {
    ZipArchive archive;
    const Workspace& workspace = document.workspace();

    docxml::XmlNode root{"document", {}, {}, {}};
    docxml::set_attr(root, "version", "1");
    docxml::set_attr(root, "next_id", std::to_string(workspace.next_chunk_id()));
    docxml::set_attr(root, "active_id", std::to_string(document.active_chunk_id()));
    docxml::write_crs("global_crs", document.global_crs(), root);
    docxml::write_meta(document.meta(), root);

    docxml::XmlNode& chunks = docxml::add_child(root, "chunks");
    for (const auto& chunk_ptr : workspace.chunks()) {
        const Chunk& chunk = *chunk_ptr;
        docxml::XmlNode& node = docxml::add_child(chunks, "chunk");
        docxml::set_attr(node, "id", std::to_string(chunk.id()));
        docxml::set_attr(node, "label", chunk.label());

        detail::write_sensors(chunk, node);
        detail::write_cameras(chunk, node);
        detail::write_frames(chunk, node);
        detail::write_tie_points(chunk, node, archive);
        detail::write_assets(chunk, node, archive);

        docxml::XmlNode& region = docxml::add_child(node, "region");
        docxml::set_attr(region, "centre", docxml::vec3(chunk.region().centre));
        docxml::set_attr(region, "size", docxml::vec3(chunk.region().size));
        docxml::set_attr(docxml::add_child(node, "transform"), "matrix",
                         docxml::matrix4(chunk.transform()));
        docxml::write_crs("crs", chunk.crs(), node);
        docxml::write_meta(chunk.meta(), node);
    }

    std::string xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    docxml::write_node(root, xml, 0);
    archive.add("doc.xml", xml);
    return archive.serialise();
}

}  // namespace alphashape::test_v1
