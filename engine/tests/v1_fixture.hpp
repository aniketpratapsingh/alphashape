#pragma once

// tests/v1_fixture.hpp -- the document behind tests/fixtures/v1_project.zip.
//
// One builder is shared by the generator (tests/make_v1_fixture.cpp, which
// writes the checked-in bytes) and by the reader tests, so the fixture and its
// expectation cannot drift: the test rebuilds this document, re-serialises it
// with the pre-M11 writer, and asserts the result is byte-identical to the file
// in the repository, then loads that file and compares.
//
// Content: one georeferenced chunk with a sensor, three cameras and frames,
// 150 keypoints per camera with descriptors, verified matches and tracks, one
// depth-map set, a point cloud, a textured two-triangle mesh, a 4x3 DEM and a
// 2x2 orthomosaic with seamline labels. Everything v1 could express, nothing
// more. The archive is ~10 KB compressed (~200 KB of sidecar payload before
// deflate), well inside the 20 MB fixture budget.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "image/image.hpp"
#include "model/mesh/mesh.hpp"
#include "model/raster/geotiff.hpp"

namespace alphashape::test_v1 {

inline Keypoint fixture_keypoint(int camera, int index) {
    Keypoint kp;
    kp.x = 4.5F + 3.25F * static_cast<float>(index) + 0.5F * static_cast<float>(camera);
    kp.y = 7.25F + 2.5F * static_cast<float>(index) - 0.25F * static_cast<float>(camera);
    kp.scale = 1.25F + 0.1F * static_cast<float>(index);
    kp.orientation = 0.125F * static_cast<float>(index) + 0.5F * static_cast<float>(camera);
    return kp;
}

inline Descriptor fixture_descriptor(int camera, int index) {
    Descriptor d;
    for (std::size_t i = 0; i < Descriptor::kBytes; ++i) {
        d.bytes[i] = static_cast<std::uint8_t>(
            (static_cast<unsigned>(index) * 7U + static_cast<unsigned>(camera) * 31U + i * 3U) %
            256U);
    }
    return d;
}

/// The fixture document. Deterministic: no clocks, no random draws.
inline Document make_v1_fixture_document() {
    constexpr int kCameras = 3;
    constexpr int kKeypoints = 150;

    Document document;
    Chunk& chunk = document.workspace().add_chunk("M9 aerial block");
    document.set_active_chunk_id(chunk.id());
    document.global_crs().epsg = 4326;
    document.global_crs().wkt = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\"]]";

    Sensor sensor;
    sensor.label = "Aerial 1";
    sensor.type = CameraType::Frame;
    sensor.width = 480;
    sensor.height = 360;
    sensor.pixel_size_um = 2.4;
    sensor.focal_length_mm = 4.5;
    sensor.make = "ACME";
    sensor.model = "Aerial";
    sensor.serial = "aa-001";
    sensor.calibration = Calibration::initial(CameraType::Frame, 480, 360, 500.0);
    sensor.fixed_params = {"k3"};
    chunk.sensors().push_back(sensor);

    for (int c = 0; c < kCameras; ++c) {
        Camera camera;
        camera.id = c;
        camera.sensor_id = 0;
        camera.label = "aerial_r0c" + std::to_string(c);
        camera.enabled = true;
        camera.orientation = 1;
        // Three nadir-ish poses along the east axis, one 60 m above the other.
        camera.pose = Transform::Identity();
        camera.pose(0, 3) = 12.5 * static_cast<double>(c);
        camera.pose(1, 3) = -3.0;
        camera.pose(2, 3) = -60.0 - static_cast<double>(c);
        camera.reference.location = Eigen::Vector3d(9.2 + 1e-5 * c, 48.7, 300.0 + c);
        camera.reference.accuracy = Eigen::Vector3d(2.0, 2.0, 5.0);
        chunk.cameras().push_back(camera);

        auto frame = std::make_shared<Frame>(
            "data/aerial_r0c" + std::to_string(c) + ".jpg", 480, 360);
        frame->set_id(c);
        frame->set_camera_id(c);
        chunk.frames().push_back(frame);

        KeypointSet set;
        set.camera_id = c;
        for (int k = 0; k < kKeypoints; ++k) {
            set.keypoints.push_back(fixture_keypoint(c, k));
            set.descriptors.push_back(fixture_descriptor(c, k));
        }
        chunk.tie_points().keypoint_sets.push_back(std::move(set));
    }

    for (int c = 0; c + 1 < kCameras; ++c) {
        PairMatches pair;
        pair.camera_a = c;
        pair.camera_b = c + 1;
        for (int k = 0; k < kKeypoints; ++k) {
            pair.matches.push_back(
                KeypointMatch{static_cast<std::uint32_t>(k),
                              static_cast<std::uint32_t>((k + 3) % kKeypoints)});
        }
        chunk.tie_points().matches.push_back(std::move(pair));
    }
    for (int k = 0; k < 8; ++k) {
        Track track;
        for (int c = 0; c < kCameras; ++c) {
            track.observations.push_back(
                Observation{c, static_cast<std::uint32_t>((k * 5 + c) % kKeypoints)});
        }
        track.point = Eigen::Vector3d(1.0 * k, -2.0 * k, 3.0 + k);
        chunk.tie_points().tracks.push_back(std::move(track));
    }

    DepthMap map;
    map.camera_id = 0;
    map.width = 4;
    map.height = 3;
    map.downscale = 2;
    map.depths = {30.0F, 30.5F, 31.0F, 31.5F, 32.0F, 32.5F,
                  33.0F, 33.5F, 34.0F, -1.0F, -1.0F, -1.0F};
    chunk.depth_maps().push_back(std::move(map));

    chunk.point_cloud().points = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                  Eigen::Vector3d(1.0, 2.0, 3.0),
                                  Eigen::Vector3d(-1.0, -2.0, -3.0)};
    chunk.point_cloud().colours = {255, 0, 0, 0, 255, 0, 0, 0, 255};

    Mesh mesh;
    mesh.vertices = {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(2.0, 0.0, 0.0),
                     Eigen::Vector3d(0.0, 2.0, 0.0), Eigen::Vector3d(2.0, 2.0, 1.0)};
    mesh.faces = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(1, 3, 2)};
    chunk.model() = mesh;
    Image texture(4, 4, 3);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            texture.at(x, y, 0) = static_cast<std::uint8_t>(x * 60);
            texture.at(x, y, 1) = static_cast<std::uint8_t>(y * 60);
            texture.at(x, y, 2) = static_cast<std::uint8_t>((x + y) * 30);
        }
    }
    chunk.model_textures().push_back(std::move(texture));

    Elevation dem;
    dem.raster.grid.width = 4;
    dem.raster.grid.height = 3;
    dem.raster.grid.origin_x = 500000.0;
    dem.raster.grid.origin_y = 5410002.0;
    dem.raster.grid.pixel_size_x = 0.5;
    dem.raster.grid.pixel_size_y = 0.5;
    dem.raster.channels = 1;
    dem.raster.data = {2.0F, 2.5F, 3.0F, 3.5F,
                       2.25F, 2.75F, 3.25F, 3.75F,
                       2.5F, 3.0F, 3.5F, 4.0F};
    dem.set_epsg(32633);
    dem.nodata = -32767.0F;
    dem.source = "depth_maps";
    chunk.elevation() = std::move(dem);

    Orthomosaic ortho;
    ortho.raster.grid.width = 2;
    ortho.raster.grid.height = 2;
    ortho.raster.grid.origin_x = 500000.0;
    ortho.raster.grid.origin_y = 5410001.0;
    ortho.raster.grid.pixel_size_x = 1.0;
    ortho.raster.grid.pixel_size_y = 1.0;
    ortho.raster.channels = 4;
    ortho.raster.data = {10, 20, 30, 255, 40, 50, 60, 255,
                         70, 80, 90, 255, 100, 110, 120, 0};
    ortho.set_epsg(32633);
    ortho.labels = {0, 1, 2, -1};
    ortho.surface = "elevation";
    chunk.orthomosaic() = std::move(ortho);

    chunk.region().centre = Eigen::Vector3d(12.5, 0.0, -30.0);
    chunk.region().size = Eigen::Vector3d(60.0, 45.0, 20.0);
    chunk.transform() = Transform::Identity();
    chunk.crs().epsg = 32633;
    chunk.crs().wkt = "PROJCS[\"WGS 84 / UTM zone 33N\"]";
    chunk.meta().set("Author", "alphashape fixture");
    chunk.meta().set("Milestone", "M9");
    return document;
}

}  // namespace alphashape::test_v1
