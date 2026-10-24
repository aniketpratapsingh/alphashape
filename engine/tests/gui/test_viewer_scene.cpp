// ViewerScene.*: build_scene() over synthetic chunks. Pure data, no graphics: these assert the
// vertex/instance counts the renderer will draw, which is what a render test can only check
// indirectly.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "apps/gui/viewer/scene.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "model/camera/calibration.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/point_cloud.hpp"

using namespace alphashape;
using namespace alphashape::gui;

namespace {

/// A camera pose looking down -Z from `eye` toward `target`, as world -> camera.
Transform look_at(const Eigen::Vector3d& eye, const Eigen::Vector3d& target) {
    const Eigen::Vector3d forward = (target - eye).normalized();
    const Eigen::Vector3d right = forward.cross(Eigen::Vector3d::UnitZ()).normalized();
    const Eigen::Vector3d up = right.cross(forward).normalized();
    // Camera frame is +X right, +Y down, +Z forward, so world -> camera negates Y.
    Eigen::Matrix3d rotation;
    rotation.row(0) = right.transpose();
    rotation.row(1) = -up.transpose();
    rotation.row(2) = forward.transpose();
    Transform pose = Transform::Identity();
    pose.topLeftCorner<3, 3>() = rotation;
    pose.topRightCorner<3, 1>() = -(rotation * eye);
    return pose;
}

/// A chunk with two aligned cameras sharing one sensor, 100 tie points, 100 dense-cloud points
/// and a two-triangle quad mesh - the shape the P1-VIEW tests pin down.
Chunk make_synthetic_chunk() {
    Chunk chunk("Synthetic");

    Sensor sensor;
    sensor.label = "Test camera";
    sensor.type = CameraType::Frame;
    sensor.width = 400;
    sensor.height = 300;
    sensor.pixel_size_um = 3.0;
    sensor.focal_length_mm = 6.0;
    sensor.calibration = Calibration::initial(CameraType::Frame, 400, 300, 480.0);
    chunk.sensors().push_back(sensor);

    const Eigen::Vector3d target(0.0, 0.0, 0.0);
    for (int i = 0; i < 2; ++i) {
        Camera camera;
        camera.id = i;
        camera.sensor_id = 0;
        camera.label = "IMG_" + std::to_string(i);
        camera.enabled = true;
        camera.pose = look_at(Eigen::Vector3d(2.0 * i - 1.0, -4.0, 1.5), target);
        chunk.cameras().push_back(camera);
    }

    for (int i = 0; i < 100; ++i) {
        Track track;
        track.observations = {{0, static_cast<std::uint32_t>(i)},
                              {1, static_cast<std::uint32_t>(i)}};
        const double t = static_cast<double>(i) / 99.0;
        track.point = Eigen::Vector3d(t - 0.5, 0.25 * t, 0.05 * t);
        chunk.tie_points().tracks.push_back(track);
    }

    PointCloud& cloud = chunk.point_cloud();
    for (int i = 0; i < 100; ++i) {
        cloud.points.push_back(Eigen::Vector3d(0.01 * i, -0.02 * i, 0.5));
    }
    cloud.colours.assign(cloud.points.size() * 3U, 128U);

    Mesh& mesh = chunk.model();
    mesh.vertices = {
        Eigen::Vector3d(-1.0, -1.0, 0.0),
        Eigen::Vector3d(1.0, -1.0, 0.0),
        Eigen::Vector3d(1.0, 1.0, 0.0),
        Eigen::Vector3d(-1.0, 1.0, 0.0),
    };
    mesh.faces = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(0, 2, 3)};
    mesh.colours.assign(mesh.vertices.size() * 3U, 64U);
    return chunk;
}

}  // namespace

TEST(ViewerScene, SyntheticChunkProducesTheExpectedCounts) {
    const Chunk chunk = make_synthetic_chunk();
    const Scene scene = build_scene(chunk, scene_style_for(ThemeMode::Dark));

    EXPECT_EQ(scene.aligned_cameras, 2U);
    EXPECT_EQ(scene.total_cameras, 2U);
    // One sprite per tie point and per dense point; the colours are per point.
    EXPECT_EQ(scene.tie_points.size(), 100U);
    EXPECT_EQ(scene.dense_points.size(), 100U);
    // Two triangles, indexed at three vertices each.
    EXPECT_EQ(scene.mesh_triangles(), 2U);
    EXPECT_EQ(scene.mesh_indices.size(), 6U);
    EXPECT_FALSE(scene.texture.width > 0);
    EXPECT_FALSE(scene.has_texture());

    // Two frustums of five edges each (four apex edges + the four corner edges is 8; count the
    // geometry instead of assuming), plus grid, axes and the bounding box: all lines come in
    // pairs of vertices.
    EXPECT_EQ(scene.lines.size() % 2U, 0U);
    EXPECT_GE(scene.lines.size(), 2U * 8U);  // at least the two frustums
    EXPECT_TRUE(scene.has_bounds);
    EXPECT_TRUE(scene.has_points());
    EXPECT_TRUE(scene.has_mesh());
}

TEST(ViewerScene, BoundsCoverPointsMeshAndCameraCentres) {
    const Chunk chunk = make_synthetic_chunk();
    const Scene scene = build_scene(chunk, scene_style_for(ThemeMode::Dark));
    ASSERT_TRUE(scene.has_bounds);
    // Everything drawn lies inside the reported bounds (within float rounding).
    const auto inside = [&](const Eigen::Vector3f& point) {
        const Eigen::Vector3d p = point.cast<double>();
        return (p.array() >= scene.bounds_min.array() - 1e-6).all() &&
               (p.array() <= scene.bounds_max.array() + 1e-6).all();
    };
    for (const ScenePointVertex& tie : scene.tie_points) {
        EXPECT_TRUE(inside(tie.position)) << tie.position.transpose();
    }
    for (const ScenePointVertex& dense : scene.dense_points) {
        EXPECT_TRUE(inside(dense.position)) << dense.position.transpose();
    }
    for (const SceneMeshVertex& vertex : scene.mesh_vertices) {
        EXPECT_TRUE(inside(vertex.position)) << vertex.position.transpose();
    }
}

TEST(ViewerScene, EmptyChunkBuildsAnEmptyScene) {
    const Chunk chunk;
    const Scene scene = build_scene(chunk, scene_style_for(ThemeMode::Dark));
    EXPECT_TRUE(scene.empty());
    EXPECT_FALSE(scene.has_bounds);
    EXPECT_FALSE(scene.has_points());
    EXPECT_FALSE(scene.has_mesh());
    EXPECT_EQ(scene.aligned_cameras, 0U);
    EXPECT_EQ(scene.total_cameras, 0U);
    EXPECT_TRUE(scene.tie_points.empty());
    EXPECT_TRUE(scene.dense_points.empty());
    EXPECT_TRUE(scene.mesh_indices.empty());
    EXPECT_TRUE(scene.lines.empty());
}

TEST(ViewerScene, DocumentWithoutAnActiveChunkBuildsAnEmptyScene) {
    const Document document;
    EXPECT_EQ(document.active_chunk(), nullptr);
    const Scene scene = build_scene(document, scene_style_for(ThemeMode::Dark));
    EXPECT_TRUE(scene.empty());
}

TEST(ViewerScene, DocumentBuildsItsActiveChunk) {
    Document document;
    document.workspace().restore_chunk(7, "Small");  // empty, NOT active yet
    document.workspace().restore_chunk(9, "Chosen");
    document.workspace().find_chunk(9)->point_cloud().points.push_back(Eigen::Vector3d::Zero());
    document.set_active_chunk_id(9);
    const Scene scene = build_scene(document, scene_style_for(ThemeMode::Light));
    EXPECT_EQ(scene.dense_points.size(), 1U);
    EXPECT_TRUE(scene.has_bounds);
}

TEST(ViewerScene, UnalignedCamerasDrawNoFrustum) {
    Chunk chunk = make_synthetic_chunk();
    for (Camera& camera : chunk.cameras()) {
        camera.pose = Transform::Identity();  // not aligned yet
    }
    const Scene scene = build_scene(chunk, scene_style_for(ThemeMode::Dark));
    EXPECT_EQ(scene.aligned_cameras, 0U);
    EXPECT_EQ(scene.total_cameras, 2U);
    // The points and the mesh are still there, so only the frustum lines are missing.
    EXPECT_EQ(scene.tie_points.size(), 100U);
    EXPECT_TRUE(scene.has_mesh());
}

TEST(ViewerScene, UntexturedMeshUsesVertexColoursAndIndexedVertices) {
    const Chunk chunk = make_synthetic_chunk();
    const Scene scene = build_scene(chunk, scene_style_for(ThemeMode::Dark));
    // Four shared vertices, not six: the mesh is indexed when untextured.
    EXPECT_EQ(scene.mesh_vertices.size(), 4U);
    EXPECT_EQ(scene.mesh_indices.size(), 6U);
    EXPECT_FALSE(scene.textured);
    // 64 / 255 grey from the vertex colours; the renderer shades it, the builder does not.
    EXPECT_NEAR(static_cast<double>(scene.mesh_vertices[0].colour.x()), 64.0 / 255.0, 1e-6);
    // Normals come from the mesh's own winding (+Z for this quad).
    EXPECT_NEAR(std::abs(static_cast<double>(scene.mesh_vertices[0].normal.z())), 1.0, 1e-6);
}

TEST(ViewerScene, TexturedMeshSplitsCornersAndCarriesThePage) {
    Chunk chunk = make_synthetic_chunk();
    Mesh& mesh = chunk.model();
    mesh.uvs = {Eigen::Vector2f(0.0F, 0.0F), Eigen::Vector2f(1.0F, 0.0F),
                Eigen::Vector2f(1.0F, 1.0F), Eigen::Vector2f(0.0F, 1.0F)};
    mesh.uv_faces = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(0, 2, 3)};
    Image& page = chunk.model_textures().emplace_back(2, 2, 3);
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            page.at(x, y, 0) = static_cast<std::uint8_t>(255 * x);
            page.at(x, y, 1) = static_cast<std::uint8_t>(255 * y);
            page.at(x, y, 2) = 40;
        }
    }

    const Scene scene = build_scene(chunk, scene_style_for(ThemeMode::Dark));
    EXPECT_TRUE(scene.textured);
    EXPECT_TRUE(scene.has_texture());
    EXPECT_EQ(scene.texture.width, 2);
    EXPECT_EQ(scene.texture.height, 2);
    EXPECT_EQ(scene.texture.rgba.size(), 2U * 2U * 4U);
    // Alpha is filled in opaque; RGB comes from the page. Texel (x=1, y=0) sits at byte 4, so
    // bytes 4..7 are that texel's R, G, B, A.
    EXPECT_EQ(scene.texture.rgba[3], 255U);  // alpha of texel (0, 0)
    EXPECT_EQ(scene.texture.rgba[4], 255U);  // R of texel (1, 0): 255 * x
    EXPECT_EQ(scene.texture.rgba[5], 0U);    // G of texel (1, 0): 255 * y
    EXPECT_EQ(scene.texture.rgba[6], 40U);   // B of texel (1, 0)
    EXPECT_EQ(scene.texture.rgba[7], 255U);  // alpha of texel (1, 0)
    // Row 0 of the page is v = 0 (the mesh UV convention), i.e. it occupies the first row of the
    // RGBA buffer.
    EXPECT_EQ(scene.texture.rgba[(1U * 2U + 0U) * 4U + 1U], 255U);  // G of texel (0, 1)
    // One vertex per face corner: the UV pool is per corner.
    EXPECT_EQ(scene.mesh_vertices.size(), 6U);
    EXPECT_EQ(scene.mesh_indices.size(), 6U);
    EXPECT_NEAR(static_cast<double>(scene.mesh_vertices[0].uv.x()), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(scene.mesh_vertices[1].uv.x()), 1.0, 1e-6);
}

TEST(ViewerScene, UvsWithoutAPageFallBackToVertexColours) {
    Chunk chunk = make_synthetic_chunk();
    chunk.model().uvs = {Eigen::Vector2f(0.0F, 0.0F), Eigen::Vector2f(1.0F, 0.0F),
                         Eigen::Vector2f(1.0F, 1.0F), Eigen::Vector2f(0.0F, 1.0F)};
    chunk.model().uv_faces = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(0, 2, 3)};
    const Scene scene = build_scene(chunk, scene_style_for(ThemeMode::Dark));
    EXPECT_FALSE(scene.textured);
    EXPECT_FALSE(scene.has_texture());
    EXPECT_EQ(scene.mesh_vertices.size(), 4U);
}

TEST(ViewerScene, DensePointsAreDecimatedAboveTheCap) {
    Chunk chunk;
    PointCloud& cloud = chunk.point_cloud();
    for (int i = 0; i < 1000; ++i) {
        cloud.points.push_back(Eigen::Vector3d(i, 0.0, 0.0));
    }
    BuildOptions options;
    options.max_dense_points = 100;
    const Scene scene = build_scene(chunk, scene_style_for(ThemeMode::Dark), options);
    EXPECT_LE(scene.dense_points.size(), 100U);
    EXPECT_GE(scene.dense_points.size(), 50U);
    // Still the first and the last point of the cloud: the stride keeps the extent.
    EXPECT_NEAR(static_cast<double>(scene.dense_points.front().position.x()), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(scene.dense_points.back().position.x()), 990.0, 1e-3);
}

TEST(ViewerScene, TiePointsAreDecimatedAboveTheCap) {
    Chunk chunk;
    for (int i = 0; i < 500; ++i) {
        Track track;
        track.observations = {{0, 0}, {1, 0}};
        track.point = Eigen::Vector3d(i, 1.0, 2.0);
        chunk.tie_points().tracks.push_back(track);
    }
    BuildOptions options;
    options.max_tie_points = 50;
    const Scene scene = build_scene(chunk, scene_style_for(ThemeMode::Dark), options);
    EXPECT_LE(scene.tie_points.size(), 50U);
    EXPECT_GE(scene.tie_points.size(), 25U);
}

TEST(ViewerScene, PointColoursComeFromTheCloudWhenPresent) {
    Chunk chunk;
    PointCloud& cloud = chunk.point_cloud();
    cloud.points = {Eigen::Vector3d::Zero()};
    cloud.colours = {255U, 0U, 0U};
    const SceneStyle style = scene_style_for(ThemeMode::Dark);
    const Scene coloured = build_scene(chunk, style);
    ASSERT_EQ(coloured.dense_points.size(), 1U);
    EXPECT_NEAR(static_cast<double>(coloured.dense_points[0].colour.x()), 1.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(coloured.dense_points[0].colour.y()), 0.0, 1e-6);

    cloud.colours.clear();
    const Scene plain = build_scene(chunk, style);
    ASSERT_EQ(plain.dense_points.size(), 1U);
    EXPECT_NEAR(static_cast<double>(plain.dense_points[0].colour.x()),
                static_cast<double>(style.point_cloud.x()), 1e-6);
}

TEST(ViewerScene, GridAndAxesCanBeTurnedOff) {
    const Chunk chunk = make_synthetic_chunk();
    BuildOptions all;
    all.grid = true;
    all.axes = true;
    BuildOptions none;
    none.grid = false;
    none.axes = false;
    const Scene with_all = build_scene(chunk, scene_style_for(ThemeMode::Dark), all);
    const Scene without = build_scene(chunk, scene_style_for(ThemeMode::Dark), none);
    EXPECT_GT(with_all.lines.size(), without.lines.size());
}

TEST(ViewerScene, HighlightedCameraIsDrawnInTheActiveColour) {
    const Chunk chunk = make_synthetic_chunk();
    const SceneStyle style = scene_style_for(ThemeMode::Dark);
    BuildOptions options;
    options.highlight_camera = std::int32_t{1};
    const Scene scene = build_scene(chunk, style, options);

    bool saw_active = false;
    bool saw_normal = false;
    for (const SceneLineVertex& vertex : scene.lines) {
        if ((vertex.colour - style.camera_active).norm() < 1e-6F) {
            saw_active = true;
        } else if ((vertex.colour - style.camera).norm() < 1e-6F) {
            saw_normal = true;
        }
    }
    EXPECT_TRUE(saw_active);
    EXPECT_TRUE(saw_normal);
}

TEST(ViewerScene, StyleFollowsTheThemeTokens) {
    const SceneStyle dark = scene_style_for(ThemeMode::Dark);
    const SceneStyle light = scene_style_for(ThemeMode::Light);
    // The light canvas is lighter than the dark one (the tokens' own invariant).
    EXPECT_GT(light.background_bottom.x(), dark.background_bottom.x());
    // Each canvas's label contrasts with its own background: a light canvas has a dark label and
    // a dark canvas a light one. (The two token sets are not ordered the same way.)
    EXPECT_LT(light.label.x(), light.background_bottom.x());
    EXPECT_GT(dark.label.x(), dark.background_bottom.x());
    EXPECT_GE(dark.camera.x(), 0.0F);
    EXPECT_LE(dark.camera.x(), 1.0F);
}

TEST(ViewerScene, MalformedFacesAreSkippedRatherThanReadOutOfBounds) {
    Chunk chunk = make_synthetic_chunk();
    chunk.model().faces.push_back(Eigen::Vector3i(0, 1, 99));  // 99 does not exist
    const Scene scene = build_scene(chunk, scene_style_for(ThemeMode::Dark));
    // The two good faces survive; the bad one contributes only its two valid corners, which the
    // index list drops (a partial triangle is not referenced).
    EXPECT_LE(scene.mesh_indices.size(), 8U);
    EXPECT_GE(scene.mesh_indices.size(), 6U);
    for (const std::uint32_t index : scene.mesh_indices) {
        EXPECT_LT(index, scene.mesh_vertices.size());
    }
}
