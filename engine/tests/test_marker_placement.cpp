// WP-13.4: manual marker placement, the projection API and marker import/export.
//
// The command cases drive the same invariant the model-command suite uses:
// apply changes the chunk, revert restores the exact pre-state (including the
// id counter), and a second apply reproduces the first. The task cases write a
// real file and run it through the registry, so the codec, the label matching
// and the atomic import are exercised together.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/markers/project_markers.hpp"
#if defined(ALPHASHAPE_USE_PROJ)
#include "common/algorithm/reference/reference_table.hpp"
#endif
#include "common/commands/command.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/marker.hpp"
#include "common/core/types.hpp"
#include "common/tasks/registry.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

class TempFile {
public:
    explicit TempFile(const std::string& suffix)
        : path_(platform::unique_temp_path("alphashape_markers_", suffix)) {}
    ~TempFile() {
        std::error_code error;
        fs::remove(path_, error);
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

void write_text(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out) << "cannot create " << path;
    out << text;
    out.close();
    ASSERT_TRUE(out) << "cannot write " << path;
}

Document make_document() {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("Chunk 1");
    document.set_active_chunk_id(chunk.id());
    return document;
}

Camera& add_camera(Chunk& chunk, const std::string& label) {
    chunk.cameras().push_back(Camera{});
    Camera& camera = chunk.cameras().back();
    camera.id = chunk.allocate_id();
    camera.label = label;
    camera.enabled = true;
    return camera;
}

Params params_for(const std::string& task) {
    return make_params(builtin_registry().require(task));
}

void run_task_named(Document& document, const std::string& task, Params& params) {
    run_task(builtin_registry(), document, task, params);
}

Marker* find_marker(Chunk& chunk, const std::string& label) {
    for (Marker& marker : chunk.markers()) {
        if (marker.label == label) {
            return &marker;
        }
    }
    return nullptr;
}

// Eigen's operator== is coefficient-wise, so gtest needs a per-component
// comparison. The default tolerance is far below a pixel.
void expect_vec2(const Eigen::Vector2d& actual, const Eigen::Vector2d& expected,
                 double tolerance = 1e-12) {
    EXPECT_NEAR(actual.x(), expected.x(), tolerance);
    EXPECT_NEAR(actual.y(), expected.y(), tolerance);
}

void expect_vec3(const Eigen::Vector3d& actual, const Eigen::Vector3d& expected) {
    EXPECT_NEAR(actual.x(), expected.x(), 1e-12);
    EXPECT_NEAR(actual.y(), expected.y(), 1e-12);
    EXPECT_NEAR(actual.z(), expected.z(), 1e-12);
}

}  // namespace
}  // namespace alphashape

namespace alphashape {
namespace {

TEST(Markers, PlaceUndoRedo) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "CAM1");
    UndoStack stack(&document);

    const std::int32_t next_id_before = chunk->next_id();
    stack.execute(make_add_marker_command(chunk->id(), std::string("GCP1"),
                                          Eigen::Vector3d(1.0, 2.0, 3.0)));
    ASSERT_TRUE(chunk->validate().empty());
    ASSERT_EQ(chunk->markers().size(), 1U);
    const Marker placed = chunk->markers()[0];
    EXPECT_EQ(placed.label, "GCP1");
    ASSERT_TRUE(placed.position.has_value());
    expect_vec3(*placed.position, Eigen::Vector3d(1.0, 2.0, 3.0));

    stack.undo();
    EXPECT_TRUE(chunk->markers().empty());
    EXPECT_EQ(chunk->next_id(), next_id_before);
    EXPECT_TRUE(chunk->validate().empty());

    stack.redo();
    ASSERT_EQ(chunk->markers().size(), 1U);
    EXPECT_EQ(chunk->markers()[0].id, placed.id);
    EXPECT_EQ(chunk->markers()[0].label, placed.label);
    ASSERT_TRUE(chunk->markers()[0].position.has_value());
    expect_vec3(*chunk->markers()[0].position, *placed.position);
    EXPECT_TRUE(chunk->validate().empty());
}

TEST(Markers, MoveProjectionUndo) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    Camera& camera = add_camera(*chunk, "CAM1");
    UndoStack stack(&document);

    stack.execute(make_add_marker_command(chunk->id(), std::string("M1"),
                                          std::nullopt));
    const std::int32_t marker_id = chunk->markers()[0].id;
    stack.execute(make_move_marker_projection_command(
        chunk->id(), marker_id, camera.id, Eigen::Vector2d(10.0, 20.0)));
    ASSERT_EQ(chunk->markers()[0].projections.size(), 1U);
    expect_vec2(chunk->markers()[0].projections[0].pixel,
                Eigen::Vector2d(10.0, 20.0));

    stack.execute(make_move_marker_projection_command(
        chunk->id(), marker_id, camera.id, Eigen::Vector2d(11.5, 22.5)));
    expect_vec2(chunk->markers()[0].projections[0].pixel,
                Eigen::Vector2d(11.5, 22.5));

    stack.undo();
    ASSERT_EQ(chunk->markers()[0].projections.size(), 1U);
    expect_vec2(chunk->markers()[0].projections[0].pixel,
                Eigen::Vector2d(10.0, 20.0));
    stack.undo();
    EXPECT_TRUE(chunk->markers()[0].projections.empty());
    EXPECT_TRUE(chunk->validate().empty());

    stack.redo();
    stack.redo();
    ASSERT_EQ(chunk->markers()[0].projections.size(), 1U);
    expect_vec2(chunk->markers()[0].projections[0].pixel,
                Eigen::Vector2d(11.5, 22.5));
}

TEST(Markers, DeleteLastProjectionKeepsMarkerWhenNotRequested) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    Camera& camera = add_camera(*chunk, "CAM1");
    UndoStack stack(&document);

    stack.execute(make_add_marker_with_projection_command(
        chunk->id(), "M1", camera.id, Eigen::Vector2d(4.0, 5.0), false));
    ASSERT_EQ(chunk->markers().size(), 1U);
    ASSERT_EQ(chunk->markers()[0].projections.size(), 1U);
    const std::int32_t marker_id = chunk->markers()[0].id;

    stack.execute(make_remove_marker_projection_command(chunk->id(), marker_id,
                                                        camera.id, false));
    ASSERT_EQ(chunk->markers().size(), 1U);
    EXPECT_TRUE(chunk->markers()[0].projections.empty());
    EXPECT_TRUE(chunk->validate().empty());

    stack.undo();
    ASSERT_EQ(chunk->markers().size(), 1U);
    ASSERT_EQ(chunk->markers()[0].projections.size(), 1U);
    expect_vec2(chunk->markers()[0].projections[0].pixel,
                Eigen::Vector2d(4.0, 5.0));
}

TEST(Markers, DeleteLastProjectionDeletesMarkerWhenRequested) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    Camera& camera = add_camera(*chunk, "CAM1");
    UndoStack stack(&document);

    const std::int32_t next_id_before = chunk->next_id();
    stack.execute(make_add_marker_with_projection_command(
        chunk->id(), "M1", camera.id, Eigen::Vector2d(4.0, 5.0), true));
    ASSERT_EQ(chunk->markers().size(), 1U);
    const std::int32_t marker_id = chunk->markers()[0].id;
    EXPECT_TRUE(chunk->markers()[0].pinned);

    stack.execute(make_remove_marker_projection_command(chunk->id(), marker_id,
                                                        camera.id, true));
    EXPECT_TRUE(chunk->markers().empty());
    EXPECT_TRUE(chunk->validate().empty());

    stack.undo();
    ASSERT_EQ(chunk->markers().size(), 1U);
    EXPECT_EQ(chunk->markers()[0].id, marker_id);
    ASSERT_EQ(chunk->markers()[0].projections.size(), 1U);

    // Redo of the delete path removes the marker again, id-for-id.
    stack.redo();
    EXPECT_TRUE(chunk->markers().empty());
    EXPECT_TRUE(chunk->validate().empty());

    stack.undo();
    ASSERT_EQ(chunk->markers().size(), 1U);
    EXPECT_EQ(chunk->markers()[0].id, marker_id);
    ASSERT_EQ(chunk->markers()[0].projections.size(), 1U);
    stack.undo();
    EXPECT_TRUE(chunk->markers().empty());
    EXPECT_EQ(chunk->next_id(), next_id_before);
}

// A marker may legitimately be labelled "5". An edit addressed to marker id 5
// must reach id 5 only, and an edit addressed to label "5" must reach the
// labelled marker only; neither may shadow the other.
TEST(Markers, NumericLabelDoesNotShadowId) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    Camera& camera = add_camera(*chunk, "CAM1");
    // Build ids 0..5 so that one marker is labelled "5" and a DIFFERENT marker
    // has id 5. This is the ambiguous pair the old label-first resolver picked
    // the wrong member of.
    for (int i = 0; i < 3; ++i) {
        make_add_marker_command(chunk->id(), "p" + std::to_string(i),
                                std::nullopt)
            ->apply(document);
    }
    make_add_marker_command(chunk->id(), std::string("5"), std::nullopt)
        ->apply(document);  // id 3
    make_add_marker_command(chunk->id(), std::string("p4"), std::nullopt)
        ->apply(document);  // id 4
    make_add_marker_command(chunk->id(), std::string("target"), std::nullopt)
        ->apply(document);  // id 5
    ASSERT_EQ(chunk->markers().size(), 6U);
    Marker* labelled = find_marker(*chunk, "5");
    ASSERT_NE(labelled, nullptr);
    EXPECT_EQ(labelled->id, 3);
    EXPECT_EQ(find_marker(*chunk, "target")->id, 5);

    // By id 5: only the marker with id 5 gets the projection.
    make_add_marker_with_projection_command(chunk->id(), 5, camera.id,
                                            Eigen::Vector2d(7.0, 8.0), false)
        ->apply(document);
    ASSERT_EQ(find_marker(*chunk, "target")->projections.size(), 1U);
    EXPECT_TRUE(find_marker(*chunk, "5")->projections.empty());

    // By label "5": the marker labelled "5" gets it, id 5 does not.
    make_add_marker_with_projection_command(chunk->id(), std::string("5"),
                                            camera.id, Eigen::Vector2d(1.0, 2.0),
                                            false)
        ->apply(document);
    ASSERT_EQ(find_marker(*chunk, "5")->projections.size(), 1U);
    expect_vec2(find_marker(*chunk, "5")->projections[0].pixel,
                Eigen::Vector2d(1.0, 2.0));
    EXPECT_EQ(find_marker(*chunk, "target")->projections.size(), 1U);
    EXPECT_TRUE(chunk->validate().empty());

    // An id that names no marker is an error, not an implicit creation.
    const std::int32_t missing_id = chunk->next_id();
    EXPECT_THROW(make_add_marker_with_projection_command(
                     chunk->id(), missing_id, camera.id,
                     Eigen::Vector2d(3.0, 4.0), false)
                     ->apply(document),
                 ValueError);
    EXPECT_EQ(chunk->markers().size(), 6U);
}

TEST(Markers, DeleteLastProjectionDetachesGroupMember) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    Camera& camera = add_camera(*chunk, "CAM1");
    UndoStack stack(&document);

    stack.execute(make_add_marker_with_projection_command(
        chunk->id(), "M1", camera.id, Eigen::Vector2d(4.0, 5.0), false));
    const std::int32_t marker_id = chunk->markers()[0].id;
    stack.execute(make_add_group_command(GroupKind::Marker, chunk->id(), "G"));
    const std::int32_t group_id = chunk->marker_groups()[0].id;
    stack.execute(make_move_to_group_command(GroupKind::Marker, chunk->id(),
                                             {marker_id}, group_id));
    ASSERT_EQ(chunk->marker_groups()[0].members.size(), 1U);
    EXPECT_TRUE(chunk->validate().empty());

    stack.execute(make_remove_marker_projection_command(chunk->id(), marker_id,
                                                        camera.id, true));
    ASSERT_TRUE(chunk->markers().empty());
    EXPECT_TRUE(chunk->marker_groups()[0].members.empty());
    EXPECT_TRUE(chunk->validate().empty());

    // Undo restores the marker and its group membership together.
    stack.undo();
    ASSERT_EQ(chunk->markers().size(), 1U);
    ASSERT_EQ(chunk->marker_groups()[0].members.size(), 1U);
    EXPECT_EQ(chunk->marker_groups()[0].members[0], marker_id);
    EXPECT_TRUE(chunk->validate().empty());
}

TEST(Markers, AutoLabelPointN) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();

    // "point 2" taken up front: the generator must fill the lowest free N, not
    // simply count the markers.
    make_add_marker_command(chunk->id(), std::string("point 2"), std::nullopt)
        ->apply(document);
    make_add_marker_command(chunk->id(), std::nullopt, std::nullopt)->apply(document);
    make_add_marker_command(chunk->id(), std::nullopt, std::nullopt)->apply(document);

    ASSERT_EQ(chunk->markers().size(), 3U);
    // Sorted by id: "point 2" was placed first, then the two generated labels.
    EXPECT_EQ(chunk->markers()[0].label, "point 2");
    EXPECT_EQ(chunk->markers()[1].label, "point 1");
    EXPECT_EQ(chunk->markers()[2].label, "point 3");
    EXPECT_TRUE(chunk->validate().empty());
}

TEST(Markers, DuplicateLabelThrows) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    make_add_marker_command(chunk->id(), std::string("GCP1"), std::nullopt)
        ->apply(document);
    ASSERT_EQ(chunk->markers().size(), 1U);

    EXPECT_THROW(make_add_marker_command(chunk->id(), std::string("GCP1"),
                                         std::nullopt)
                     ->apply(document),
                 ValueError);
    EXPECT_THROW(make_add_marker_command(chunk->id(), std::string(), std::nullopt)
                     ->apply(document),
                 ValueError);
    EXPECT_EQ(chunk->markers().size(), 1U);
    EXPECT_TRUE(chunk->validate().empty());
}

Calibration test_calibration() {
    Calibration calibration =
        Calibration::initial(CameraType::Frame, 640, 480, 500.0);
    calibration.cx = 2.0;
    calibration.cy = -3.0;
    return calibration;
}

/// A pose that is neither identity nor axis-aligned: 90 degrees about Z with a
/// translation. The expected pixel below is derived from this matrix by hand.
Camera test_camera() {
    Camera camera;
    camera.id = 0;
    camera.label = "CAM";
    camera.enabled = true;
    camera.pose = Eigen::Matrix4d::Identity();
    camera.pose(0, 0) = 0.0;
    camera.pose(0, 1) = -1.0;
    camera.pose(1, 0) = 1.0;
    camera.pose(1, 1) = 0.0;
    camera.pose(0, 3) = 5.0;
    camera.pose(1, 3) = -2.0;
    camera.pose(2, 3) = 10.0;
    return camera;
}

TEST(Markers, ProjectMatchesKnownPixel) {
    const Calibration calibration = test_calibration();
    const Camera camera = test_camera();
    const Eigen::Vector3d point(1.0, 2.0, 4.0);

    // R * point + t = (3, -1, 14):
    //   x = 0*1 - 1*2 + 0*4 + 5 = 3
    //   y = 1*1 + 0*2 + 0*4 - 2 = -1
    //   z = 0*1 + 0*2 + 1*4 + 10 = 14
    const double expected_u = 640.0 / 2.0 + 2.0 + (3.0 / 14.0) * 500.0;
    const double expected_v = 480.0 / 2.0 - 3.0 + (-1.0 / 14.0) * 500.0;

    const std::optional<Eigen::Vector2d> pixel =
        project_marker(point, camera, calibration);
    ASSERT_TRUE(pixel.has_value());
    EXPECT_NEAR(pixel->x(), expected_u, 1e-9);
    EXPECT_NEAR(pixel->y(), expected_v, 1e-9);

    Marker marker;
    marker.position = point;
    const std::optional<Eigen::Vector2d> via_marker =
        project_marker(marker, camera, calibration);
    ASSERT_TRUE(via_marker.has_value());
    expect_vec2(*via_marker, *pixel);
}

TEST(Markers, ProjectBehindCameraReturnsNullopt) {
    const Calibration calibration = test_calibration();
    Camera camera = test_camera();

    // z = -100 + 10 = -90: at the back of the camera.
    EXPECT_FALSE(project_marker(Eigen::Vector3d(1.0, 2.0, -100.0), camera,
                                calibration)
                     .has_value());
    // The image plane itself (z == 0) is not projected either.
    EXPECT_FALSE(project_marker(Eigen::Vector3d(1.0, 2.0, -10.0), camera,
                                calibration)
                     .has_value());

    // An unaligned camera has no meaningful world -> camera transform.
    camera.enabled = false;
    EXPECT_FALSE(project_marker(Eigen::Vector3d(1.0, 2.0, 4.0), camera,
                                calibration)
                     .has_value());

    // A marker without an estimated position has no projection.
    Marker marker;
    EXPECT_FALSE(project_marker(marker, test_camera(), calibration).has_value());

    // Outside the image: clipped away by default, returned when asked for.
    Camera origin;
    origin.enabled = true;
    origin.pose = Eigen::Matrix4d::Identity();
    const Eigen::Vector3d far_right(1000000.0, 0.0, 10.0);
    EXPECT_FALSE(project_marker(far_right, origin, calibration).has_value());
    const std::optional<Eigen::Vector2d> unclipped =
        project_marker(far_right, origin, calibration, false);
    ASSERT_TRUE(unclipped.has_value());
    EXPECT_GT(unclipped->x(), 640.0);
}

TEST(Markers, FisheyeSeesPointsBehindThePlane) {
    // The equidistant fisheye maps the whole sphere (theta up to pi), so a
    // point at 153 degrees off-axis is representable even though z < 0. Only
    // the Frame model has a z <= 0 cut.
    const Calibration fisheye =
        Calibration::initial(CameraType::Fisheye, 640, 480, 100.0);
    Camera camera;
    camera.enabled = true;
    camera.pose = Eigen::Matrix4d::Identity();
    const Eigen::Vector3d behind(5.0, 0.0, -10.0);

    const std::optional<Eigen::Vector2d> pixel =
        project_marker(behind, camera, fisheye, /*clip_to_image=*/false);
    ASSERT_TRUE(pixel.has_value());
    // theta = atan2(5, -10) = 2.6779 rad; u = 320 + theta * f.
    EXPECT_NEAR(pixel->x(), 320.0 + std::atan2(5.0, -10.0) * 100.0, 1e-9);
    EXPECT_NEAR(pixel->y(), 240.0, 1e-9);

    const Calibration frame =
        Calibration::initial(CameraType::Frame, 640, 480, 100.0);
    EXPECT_FALSE(project_marker(behind, camera, frame, false).has_value());
}

TEST(Markers, ProjectionRoundTripCsv) {
    Document source = make_document();
    Chunk* source_chunk = source.active_chunk();
    const std::int32_t cam0 = add_camera(*source_chunk, "cam0").id;
    const std::int32_t cam1 = add_camera(*source_chunk, "cam1").id;
    make_add_marker_with_projection_command(source_chunk->id(), "A", cam0,
                                            Eigen::Vector2d(12.25, 34.5), false)
        ->apply(source);
    make_add_marker_with_projection_command(source_chunk->id(), "A", cam1,
                                            Eigen::Vector2d(56.75, 78.125), false)
        ->apply(source);

    TempFile file(".csv");
    Params export_params = params_for("ExportMarkers");
    export_params.set("path", file.path());
    run_task_named(source, "ExportMarkers", export_params);

    Document destination = make_document();
    Chunk* destination_chunk = destination.active_chunk();
    add_camera(*destination_chunk, "cam0");
    add_camera(*destination_chunk, "cam1");
    Params import_params = params_for("ImportMarkers");
    import_params.set("path", file.path());
    run_task_named(destination, "ImportMarkers", import_params);

    ASSERT_EQ(destination_chunk->markers().size(), 1U);
    Marker* marker = find_marker(*destination_chunk, "A");
    ASSERT_NE(marker, nullptr);
    ASSERT_EQ(marker->projections.size(), 2U);
    EXPECT_EQ(marker->projections[0].camera_id, 0);
    EXPECT_NEAR(marker->projections[0].pixel.x(), 12.25, 1e-12);
    EXPECT_NEAR(marker->projections[0].pixel.y(), 34.5, 1e-12);
    EXPECT_EQ(marker->projections[1].camera_id, 1);
    EXPECT_NEAR(marker->projections[1].pixel.x(), 56.75, 1e-12);
    EXPECT_NEAR(marker->projections[1].pixel.y(), 78.125, 1e-12);
    EXPECT_TRUE(destination_chunk->validate().empty());
}

TEST(Markers, ProjectionRoundTripXml) {
    Document source = make_document();
    Chunk* source_chunk = source.active_chunk();
    const std::int32_t cam0 = add_camera(*source_chunk, "cam0").id;
    make_add_marker_command(source_chunk->id(), std::string("A"),
                            Eigen::Vector3d(1.25, -2.5, 3.75))
        ->apply(source);
    make_add_marker_with_projection_command(source_chunk->id(), "A", cam0,
                                            Eigen::Vector2d(10.5, 20.25), false)
        ->apply(source);
    make_add_marker_with_projection_command(source_chunk->id(), "B", cam0,
                                            Eigen::Vector2d(30.5, 40.25), false)
        ->apply(source);

    TempFile file(".xml");
    Params export_params = params_for("ExportMarkers");
    export_params.set("path", file.path());
    run_task_named(source, "ExportMarkers", export_params);

    Document destination = make_document();
    Chunk* destination_chunk = destination.active_chunk();
    add_camera(*destination_chunk, "cam0");
    Params import_params = params_for("ImportMarkers");
    import_params.set("path", file.path());
    run_task_named(destination, "ImportMarkers", import_params);

    ASSERT_EQ(destination_chunk->markers().size(), 2U);
    Marker* a = find_marker(*destination_chunk, "A");
    Marker* b = find_marker(*destination_chunk, "B");
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_TRUE(a->position.has_value());
    expect_vec3(*a->position, Eigen::Vector3d(1.25, -2.5, 3.75));
    ASSERT_EQ(a->projections.size(), 1U);
    expect_vec2(a->projections[0].pixel, Eigen::Vector2d(10.5, 20.25));
    EXPECT_FALSE(b->position.has_value());
    ASSERT_EQ(b->projections.size(), 1U);
    expect_vec2(b->projections[0].pixel, Eigen::Vector2d(30.5, 40.25));
    EXPECT_TRUE(destination_chunk->validate().empty());
}

TEST(Markers, ImportUnknownCameraReportsLine) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "cam0");

    TempFile file(".csv");
    write_text(file.path(),
               "marker,camera,u,v\n"
               "A,cam0,1,2\n"
               "B,NOPE,3,4\n");
    Params params = params_for("ImportMarkers");
    params.set("path", file.path());

    bool threw = false;
    try {
        run_task_named(document, "ImportMarkers", params);
    } catch (const ValueError& error) {
        threw = true;
        const std::string message = error.what();
        EXPECT_NE(message.find("line 3"), std::string::npos) << message;
        EXPECT_NE(message.find("NOPE"), std::string::npos) << message;
    }
    EXPECT_TRUE(threw);
    // Nothing was written: the bad row is found before the plan is applied.
    EXPECT_TRUE(chunk->markers().empty());
}

TEST(Markers, ImportIsAtomic) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "cam0");
    make_add_marker_command(chunk->id(), std::string("M1"), std::nullopt)
        ->apply(document);
    const std::int32_t next_id_before = chunk->next_id();

    TempFile file(".csv");
    write_text(file.path(),
               "marker,camera,u,v\n"
               "M1,cam0,1,2\n"
               "M1,NOPE,3,4\n");
    Params params = params_for("ImportMarkers");
    params.set("path", file.path());
    EXPECT_THROW(run_task_named(document, "ImportMarkers", params), ValueError);

    ASSERT_EQ(chunk->markers().size(), 1U);
    EXPECT_TRUE(chunk->markers()[0].projections.empty());
    EXPECT_EQ(chunk->next_id(), next_id_before);
}

TEST(Markers, ExportRejectsDuplicateCameraLabels) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    const std::int32_t first = add_camera(*chunk, "dup").id;
    const std::int32_t second = add_camera(*chunk, "dup").id;
    make_add_marker_with_projection_command(chunk->id(), std::string("M"), first,
                                            Eigen::Vector2d(1.0, 2.0), false)
        ->apply(document);
    make_add_marker_with_projection_command(chunk->id(), std::string("M"), second,
                                            Eigen::Vector2d(3.0, 4.0), false)
        ->apply(document);

    TempFile file(".csv");
    Params params = params_for("ExportMarkers");
    params.set("path", file.path());
    bool threw = false;
    try {
        run_task_named(document, "ExportMarkers", params);
    } catch (const ValueError& error) {
        threw = true;
        const std::string message = error.what();
        EXPECT_NE(message.find("dup"), std::string::npos) << message;
    }
    EXPECT_TRUE(threw);
}

TEST(Markers, ImportRejectsAmbiguousCameraLabel) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "dup");
    add_camera(*chunk, "dup");

    TempFile file(".csv");
    write_text(file.path(),
               "marker,camera,u,v\n"
               "M,dup,1,2\n");
    Params params = params_for("ImportMarkers");
    params.set("path", file.path());

    bool threw = false;
    try {
        run_task_named(document, "ImportMarkers", params);
    } catch (const ValueError& error) {
        threw = true;
        const std::string message = error.what();
        EXPECT_NE(message.find("line 2"), std::string::npos) << message;
        EXPECT_NE(message.find("dup"), std::string::npos) << message;
    }
    EXPECT_TRUE(threw);
    EXPECT_TRUE(chunk->markers().empty());
}

TEST(Markers, ImportStripsUtf8Bom) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    add_camera(*chunk, "cam0");

    TempFile file(".csv");
    write_text(file.path(),
               "\xEF\xBB\xBF"
               "marker,camera,u,v\n"
               "M,cam0,1,2\n");
    Params params = params_for("ImportMarkers");
    params.set("path", file.path());
    run_task_named(document, "ImportMarkers", params);

    ASSERT_EQ(chunk->markers().size(), 1U);
    EXPECT_EQ(chunk->markers()[0].label, "M");
    ASSERT_EQ(chunk->markers()[0].projections.size(), 1U);
    EXPECT_TRUE(chunk->validate().empty());
}

// Leading space, an embedded comma and a double quote: the CSV writer quotes
// the field and the XML writer escapes the attribute, and neither may trim the
// label on the way back in.
TEST(Markers, LabelWhitespaceAndQuotesRoundTrip) {
    const std::string label = " A,\"q\"";
    for (const char* suffix : {".csv", ".xml"}) {
        Document source = make_document();
        Chunk* source_chunk = source.active_chunk();
        const std::int32_t camera_id = add_camera(*source_chunk, "cam0").id;
        make_add_marker_with_projection_command(source_chunk->id(), label,
                                                camera_id,
                                                Eigen::Vector2d(3.0, 4.0), false)
            ->apply(source);

        TempFile file(suffix);
        Params export_params = params_for("ExportMarkers");
        export_params.set("path", file.path());
        run_task_named(source, "ExportMarkers", export_params);

        Document destination = make_document();
        Chunk* destination_chunk = destination.active_chunk();
        add_camera(*destination_chunk, "cam0");
        Params import_params = params_for("ImportMarkers");
        import_params.set("path", file.path());
        run_task_named(destination, "ImportMarkers", import_params);

        SCOPED_TRACE(suffix);
        ASSERT_EQ(destination_chunk->markers().size(), 1U);
        EXPECT_EQ(destination_chunk->markers()[0].label, label);
        ASSERT_EQ(destination_chunk->markers()[0].projections.size(), 1U);
        expect_vec2(destination_chunk->markers()[0].projections[0].pixel,
                    Eigen::Vector2d(3.0, 4.0));
        EXPECT_TRUE(destination_chunk->validate().empty());
    }
}

#if defined(ALPHASHAPE_USE_PROJ)
TEST(Markers, PlacedMarkerAppearsInReferenceTable) {
    Document document = make_document();
    Chunk* chunk = document.active_chunk();
    const std::int32_t camera_id = add_camera(*chunk, "cam0").id;
    const Eigen::Vector3d position(4.0, 5.0, 6.0);
    make_add_marker_command(chunk->id(), std::string("GCP1"), position)
        ->apply(document);
    make_add_marker_with_projection_command(chunk->id(), "GCP1", camera_id,
                                            Eigen::Vector2d(10.0, 20.0), false)
        ->apply(document);

    const reference::ReferenceTable table =
        reference::build_reference_table(*chunk, chunk->coordinate_system());
    const reference::ReferenceRow* row = nullptr;
    for (const reference::ReferenceRow& candidate : table.rows) {
        if (candidate.label == "GCP1") {
            row = &candidate;
            break;
        }
    }
    ASSERT_NE(row, nullptr);
    EXPECT_EQ(row->kind, reference::ReferenceRowKind::Marker);
    ASSERT_TRUE(row->estimated.has_value());
    expect_vec3(*row->estimated, position);
    // Placement alone does not mark a marker as adjusted.
    EXPECT_FALSE(row->enabled);
}
#endif  // ALPHASHAPE_USE_PROJ

}  // namespace
}  // namespace alphashape