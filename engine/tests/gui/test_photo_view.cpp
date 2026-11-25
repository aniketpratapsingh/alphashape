// PhotoView.*: the Photo tab's marker placement (WP-14.25). A real PNG is written into the
// test's scratch directory and linked to a camera, so the view decodes through the engine's
// image IO exactly as it does in the application. Every edit goes through the command layer,
// which is what makes the undo assertions meaningful.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtest/gtest.h>

#include <QPointF>
#include <QString>

#include "apps/gui/app.hpp"
#include "apps/gui/photo_view.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/calibration.hpp"
#include "harness.hpp"

#if defined(ALPHASHAPE_USE_PROJ)
#include "model/crs/coordinate_system.hpp"
#include "model/geo/geodesy.hpp"
#endif

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;

namespace {

constexpr int kWidth = 64;
constexpr int kHeight = 48;

class PhotoViewFixture : public GuiTest {
protected:
    PhotoView& view() { return *window().photo_view(); }

    /// A chunk with one camera whose frame is a real PNG in the scratch directory. The camera
    /// is aligned (identity pose) and has a sensor, so a marker with a position projects.
    Chunk& build_chunk() {
        const QString path = scratch(QStringLiteral("frame.png"));
        Image image(kWidth, kHeight, 3);
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                image.at(x, y, 0) = static_cast<std::uint8_t>(x * 4);
                image.at(x, y, 1) = static_cast<std::uint8_t>(y * 5);
                image.at(x, y, 2) = 128;
            }
        }
        write_png(path.toStdString(), image);

        Document& document = app().document();
        Chunk& chunk = document.workspace().add_chunk("Photo");
        Sensor sensor;
        sensor.width = kWidth;
        sensor.height = kHeight;
        sensor.calibration = Calibration::initial(CameraType::Frame, kWidth, kHeight, 50.0);
        chunk.sensors().push_back(sensor);

        Camera camera;
        camera.id = 0;
        camera.sensor_id = 0;
        camera.label = "cam0";
        camera.enabled = true;
        chunk.cameras().push_back(camera);

        auto frame = std::make_shared<Frame>(path.toStdString(), kWidth, kHeight);
        frame->set_id(0);
        frame->set_camera_id(0);
        chunk.frames().push_back(frame);

        document.set_active_chunk_id(chunk.id());
        return chunk;
    }

    /// The marker with `marker_id`, or nullptr.
    const Marker* marker(std::int32_t marker_id) {
        const Chunk* chunk = app().document().active_chunk();
        if (chunk == nullptr) {
            return nullptr;
        }
        for (const Marker& candidate : chunk->markers()) {
            if (candidate.id == marker_id) {
                return &candidate;
            }
        }
        return nullptr;
    }

#if defined(ALPHASHAPE_USE_PROJ)
    /// Georeference the active chunk with a similarity local -> ECEF and a 4326 CRS, so a
    /// marker's surveyed reference can be predicted. The transform is a scaled, rotated ENU
    /// frame near Adelaide; the camera's identity pose is untouched, so a local point on the
    /// optical axis still projects to the image centre.
    void georeference(Chunk& chunk) {
        namespace geo = alphashape::geo;
        const geo::Geodetic origin{138.6007, -34.9285, 50.0};
        Eigen::Matrix4d local_to_enu = Eigen::Matrix4d::Identity();
        const Eigen::Matrix3d rotation =
            Eigen::AngleAxisd(0.4, Eigen::Vector3d(0.3, 0.8, 0.5).normalized())
                .toRotationMatrix();
        local_to_enu.block<3, 3>(0, 0) = 0.5 * rotation;
        local_to_enu.block<3, 1>(0, 3) = Eigen::Vector3d(3.0, -2.0, 1.5);
        chunk.transform() = geo::enu_to_ecef(origin) * local_to_enu;
        chunk.crs().epsg = 4326;
    }

    /// The geodetic (lon, lat, h) of a local point through the chunk transform.
    static Eigen::Vector3d geodetic_of_local(const Chunk& chunk, const Eigen::Vector3d& local) {
        namespace geo = alphashape::geo;
        const Eigen::Vector3d ecef = (chunk.transform() * local.homogeneous()).head<3>();
        const geo::Geodetic g = geo::ecef_to_geodetic(ecef);
        return Eigen::Vector3d(g.longitude_deg, g.latitude_deg, g.height_m);
    }
#endif
};

using PhotoViewTest = PhotoViewFixture;

TEST_F(PhotoViewTest, OpensTheCameraImage) {
    build_chunk();
    ASSERT_TRUE(view().set_camera(0));
    EXPECT_EQ(view().camera_id(), 0);
    EXPECT_TRUE(view().has_image());
    EXPECT_EQ(view().marker_items_count(), 0);
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(PhotoViewTest, AddMarkerCreatesOneProjectionAndUndoRemovesIt) {
    build_chunk();
    ASSERT_TRUE(view().set_camera(0));
    const std::size_t before = app().undo_stack().size();

    ASSERT_TRUE(view().add_marker_at(QPointF(12.0, 9.0)));
    EXPECT_EQ(app().undo_stack().size(), before + 1);
    ASSERT_EQ(app().document().active_chunk()->markers().size(), 1U);
    const Marker& created = app().document().active_chunk()->markers().front();
    ASSERT_EQ(created.projections.size(), 1U);
    EXPECT_EQ(created.projections.front().camera_id, 0);
    EXPECT_NEAR(created.projections.front().pixel.x(), 12.0, 1e-9);
    EXPECT_NEAR(created.projections.front().pixel.y(), 9.0, 1e-9);
    EXPECT_EQ(view().marker_items_count(), 1);

    app().undo();
    EXPECT_TRUE(app().document().active_chunk()->markers().empty());
    EXPECT_EQ(view().marker_items_count(), 0);
}

TEST_F(PhotoViewTest, MoveProjectionChangesThePixel) {
    build_chunk();
    ASSERT_TRUE(view().set_camera(0));
    ASSERT_TRUE(view().add_marker_at(QPointF(10.0, 10.0)));
    const std::int32_t marker_id = app().document().active_chunk()->markers().front().id;

    ASSERT_TRUE(view().move_projection(marker_id, QPointF(30.0, 20.0)));
    const Marker* moved = marker(marker_id);
    ASSERT_NE(moved, nullptr);
    ASSERT_EQ(moved->projections.size(), 1U);
    EXPECT_NEAR(moved->projections.front().pixel.x(), 30.0, 1e-9);
    EXPECT_NEAR(moved->projections.front().pixel.y(), 20.0, 1e-9);

    app().undo();
    const Marker* restored = marker(marker_id);
    ASSERT_NE(restored, nullptr);
    EXPECT_NEAR(restored->projections.front().pixel.x(), 10.0, 1e-9);
}

TEST_F(PhotoViewTest, PlaceMarkerAddsAProjectionToAnExistingMarker) {
    build_chunk();
    ASSERT_TRUE(view().set_camera(0));
    const std::int32_t chunk_id = app().document().active_chunk()->id();
    app().execute(make_add_marker_command(chunk_id, std::string("GCP 1"), std::nullopt));
    ASSERT_EQ(app().document().active_chunk()->markers().size(), 1U);
    const std::int32_t marker_id = app().document().active_chunk()->markers().front().id;
    ASSERT_TRUE(marker(marker_id)->projections.empty());

    ASSERT_TRUE(view().place_marker_at(marker_id, QPointF(40.0, 30.0)));
    const Marker* placed = marker(marker_id);
    ASSERT_NE(placed, nullptr);
    ASSERT_EQ(placed->projections.size(), 1U);
    EXPECT_EQ(placed->projections.front().camera_id, 0);
    EXPECT_NEAR(placed->projections.front().pixel.x(), 40.0, 1e-9);
    EXPECT_EQ(view().marker_items_count(), 1);
}

TEST_F(PhotoViewTest, PredictedFlagIsDrawnForAMarkerWithAPosition) {
    build_chunk();
    ASSERT_TRUE(view().set_camera(0));
    const std::int32_t chunk_id = app().document().active_chunk()->id();
    // A point on the optical axis projects to the image centre (32, 24).
    app().execute(make_add_marker_command(chunk_id, std::string("Predicted"),
                                          Eigen::Vector3d(0.0, 0.0, 1.0)));
    EXPECT_EQ(view().marker_items_count(), 1);
    const Marker& predicted = app().document().active_chunk()->markers().front();
    EXPECT_TRUE(predicted.projections.empty());
}

TEST_F(PhotoViewTest, RemoveProjectionDeletesTheLastOneWithTheMarker) {
    build_chunk();
    ASSERT_TRUE(view().set_camera(0));
    ASSERT_TRUE(view().add_marker_at(QPointF(5.0, 5.0)));
    const std::int32_t marker_id = app().document().active_chunk()->markers().front().id;

    ASSERT_TRUE(view().remove_projection(marker_id));
    EXPECT_TRUE(app().document().active_chunk()->markers().empty());
    EXPECT_EQ(view().marker_items_count(), 0);
}

TEST_F(PhotoViewTest, PlacedMarkerFlagSitsOnItsStoredPixel) {
    build_chunk();
    ASSERT_TRUE(view().set_camera(0));
    const QPointF pixel(17.0, 23.0);
    ASSERT_TRUE(view().add_marker_at(pixel));
    const std::int32_t marker_id = app().document().active_chunk()->markers().front().id;

    // The flag's scene position is the full-resolution pixel the document stores, so a click
    // at scene (x, y) lands on the marker the engine knows about.
    const Marker* stored = marker(marker_id);
    ASSERT_NE(stored, nullptr);
    ASSERT_EQ(stored->projections.size(), 1U);
    const std::vector<std::pair<std::int32_t, QPointF>> flags = view().marker_flag_positions();
    ASSERT_EQ(flags.size(), 1U);
    EXPECT_EQ(flags.front().first, marker_id);
    EXPECT_NEAR(flags.front().second.x(), stored->projections.front().pixel.x(), 1e-9);
    EXPECT_NEAR(flags.front().second.y(), stored->projections.front().pixel.y(), 1e-9);
    EXPECT_NEAR(flags.front().second.x(), pixel.x(), 1e-9);
    EXPECT_NEAR(flags.front().second.y(), pixel.y(), 1e-9);
}

TEST_F(PhotoViewTest, ReopeningTheSameCameraDoesNotDecodeAgain) {
    build_chunk();
    ASSERT_TRUE(view().set_camera(0));
    const int loads = view().image_loads();
    ASSERT_EQ(loads, 1);

    // Re-opening the same camera (a finished job, a chunk switch) serves the cached pixmap.
    ASSERT_TRUE(view().set_camera(0));
    EXPECT_EQ(view().image_loads(), loads);
    EXPECT_TRUE(view().has_image());
}

TEST_F(PhotoViewTest, UndoOfAMarkerAddRebuildsFlagsWithoutReloading) {
    build_chunk();
    ASSERT_TRUE(view().set_camera(0));
    const int loads = view().image_loads();
    ASSERT_TRUE(view().add_marker_at(QPointF(12.0, 9.0)));
    EXPECT_EQ(view().marker_items_count(), 1);

    // The undo emits undo_changed, which rebuilds the flags only: the image is not decoded.
    app().undo();
    EXPECT_EQ(view().marker_items_count(), 0);
    EXPECT_EQ(view().image_loads(), loads);
    EXPECT_TRUE(view().has_image());
}

TEST_F(PhotoViewTest, UnknownCameraShowsNothing) {
    build_chunk();
    EXPECT_FALSE(view().set_camera(99));
    EXPECT_FALSE(view().has_image());
    EXPECT_EQ(view().marker_items_count(), 0);
}

#if defined(ALPHASHAPE_USE_PROJ)

// A reference-only marker (an imported GCP) on a georeferenced chunk is predicted from its
// surveyed coordinate: the flag is unplaced (blue) and sits where the reference projects.
TEST_F(PhotoViewTest, ReferenceOnlyMarkerIsPredictedOnAGeoreferencedChunk) {
    Chunk& chunk = build_chunk();
    georeference(chunk);
    ASSERT_TRUE(view().set_camera(0));

    // A local point on the optical axis projects to the image centre (32, 24). Give the marker
    // only the reference that maps to that local point, no estimated position.
    const Eigen::Vector3d local(0.0, 0.0, 1.0);
    const Eigen::Vector3d reference = geodetic_of_local(chunk, local);
    const std::int32_t chunk_id = chunk.id();
    app().execute(make_add_marker_command(chunk_id, std::string("GCP 1"), std::nullopt));
    const std::int32_t marker_id = app().document().active_chunk()->markers().front().id;
    app().execute(make_set_marker_reference_command(
        chunk_id, marker_id, std::optional<Eigen::Vector3d>(reference),
        Eigen::Vector3d::Constant(0.5)));

    const Marker* stored = marker(marker_id);
    ASSERT_NE(stored, nullptr);
    ASSERT_TRUE(stored->reference.has_value());
    EXPECT_FALSE(stored->position.has_value());
    EXPECT_TRUE(stored->projections.empty());

    // The flag is drawn (a prediction), at the image centre.
    EXPECT_EQ(view().marker_items_count(), 1);
    const std::vector<std::pair<std::int32_t, QPointF>> flags = view().marker_flag_positions();
    ASSERT_EQ(flags.size(), 1U);
    EXPECT_EQ(flags.front().first, marker_id);
    EXPECT_NEAR(flags.front().second.x(), 32.0, 1e-3);
    EXPECT_NEAR(flags.front().second.y(), 24.0, 1e-3);
}

// Dragging the predicted flag places a projection, exactly as a position-based prediction does.
TEST_F(PhotoViewTest, DraggingAReferencePredictionCreatesAProjection) {
    Chunk& chunk = build_chunk();
    georeference(chunk);
    ASSERT_TRUE(view().set_camera(0));

    const Eigen::Vector3d local(0.0, 0.0, 1.0);
    const Eigen::Vector3d reference = geodetic_of_local(chunk, local);
    const std::int32_t chunk_id = chunk.id();
    app().execute(make_add_marker_command(chunk_id, std::string("GCP 1"), std::nullopt));
    const std::int32_t marker_id = app().document().active_chunk()->markers().front().id;
    app().execute(make_set_marker_reference_command(
        chunk_id, marker_id, std::optional<Eigen::Vector3d>(reference),
        Eigen::Vector3d::Constant(0.5)));
    ASSERT_EQ(view().marker_items_count(), 1);

    const std::size_t before = app().undo_stack().size();
    // The drag handler calls move_projection, which creates the projection for an unplaced flag.
    ASSERT_TRUE(view().move_projection(marker_id, QPointF(40.0, 30.0)));
    EXPECT_EQ(app().undo_stack().size(), before + 1);
    const Marker* placed = marker(marker_id);
    ASSERT_NE(placed, nullptr);
    ASSERT_EQ(placed->projections.size(), 1U);
    EXPECT_EQ(placed->projections.front().camera_id, 0);
    EXPECT_NEAR(placed->projections.front().pixel.x(), 40.0, 1e-9);
    EXPECT_NEAR(placed->projections.front().pixel.y(), 30.0, 1e-9);

    // Undo removes the projection and the flag returns to a prediction.
    app().undo();
    const Marker* restored = marker(marker_id);
    ASSERT_NE(restored, nullptr);
    EXPECT_TRUE(restored->projections.empty());
    EXPECT_EQ(view().marker_items_count(), 1);
}

// A reference-only marker on a chunk that is NOT georeferenced is not predicted: there is no
// transform to carry the reference into the local frame.
TEST_F(PhotoViewTest, ReferenceOnlyMarkerIsNotPredictedWithoutGeoreferencing) {
    Chunk& chunk = build_chunk();
    ASSERT_TRUE(view().set_camera(0));

    const std::int32_t chunk_id = chunk.id();
    app().execute(make_add_marker_command(chunk_id, std::string("GCP 1"), std::nullopt));
    const std::int32_t marker_id = app().document().active_chunk()->markers().front().id;
    app().execute(make_set_marker_reference_command(
        chunk_id, marker_id, std::optional<Eigen::Vector3d>(Eigen::Vector3d(138.6, -34.9, 50.0)),
        Eigen::Vector3d::Constant(0.5)));

    EXPECT_EQ(view().marker_items_count(), 0);
}

#endif  // ALPHASHAPE_USE_PROJ

}  // namespace
