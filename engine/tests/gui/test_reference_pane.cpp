// ReferencePane.*: the Reference dock (WP-14.11) driven offscreen through the
// harness. Every document is built directly in the engine, the way the Photos
// suite builds cameras; the known per-row offsets make the RMSE analytic.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <QAbstractItemModel>
#include <QModelIndex>
#include <QString>
#include <QTableView>
#include <QVariant>

#include "apps/gui/app.hpp"
#include "apps/gui/reference_model.hpp"
#include "apps/gui/reference_pane.hpp"
#include "apps/gui/workspace_pane.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "model/crs/coordinate_system.hpp"
#include "model/crs/transform.hpp"
#include "harness.hpp"
#include "../perf_gate.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;

namespace {

class ReferencePaneFixture : public GuiTest {
protected:
    ReferencePane& pane() { return *widget<ReferencePane>("pane.reference"); }

    /// A chunk with `cameras` cameras and `markers` markers. Camera i's GNSS
    /// source is i-5.5 along x (centre at the origin), and marker i's source is
    /// i+1 below its estimated position, so both RMSE figures are analytic.
    Chunk& build_chunk(int cameras, int markers) {
        Document& document = app().document();
        Chunk& chunk = document.workspace().add_chunk("Reference");
        for (int i = 0; i < cameras; ++i) {
            Camera camera;
            camera.id = i;
            camera.label = "cam" + std::to_string(i);
            camera.enabled = true;
            const double offset = static_cast<double>(i) - 5.5;
            camera.reference.location = Eigen::Vector3d(offset, 0.0, 0.0);
            camera.reference.accuracy = Eigen::Vector3d::Constant(0.02);
            chunk.cameras().push_back(camera);
        }
        for (int i = 0; i < markers; ++i) {
            Marker marker;
            marker.id = 100 + i;
            marker.label = "marker" + std::to_string(i);
            marker.valid = true;
            marker.pinned = i == 0;
            marker.position = Eigen::Vector3d(10.0, 20.0, 30.0);
            marker.reference =
                Eigen::Vector3d(10.0, 20.0, 30.0 - static_cast<double>(i + 1));
            chunk.markers().push_back(marker);
        }
        document.set_active_chunk_id(chunk.id());
        pane().refresh();
        return chunk;
    }

    static double analytic_camera_rmse(int count) {
        double squares = 0.0;
        for (int i = 0; i < count; ++i) {
            const double offset = static_cast<double>(i) - 5.5;
            squares += offset * offset;
        }
        return std::sqrt(squares / static_cast<double>(count));
    }
};

/// The suite name the task registers is ReferencePane.*; the fixture keeps a
/// distinct name so it does not hide the pane type it looks up.
using ReferencePane = ReferencePaneFixture;

TEST_F(ReferencePane, CamerasAndMarkers) {
    build_chunk(12, 4);
    ReferenceModel& cameras = pane().cameras();
    ReferenceModel& markers = pane().markers();
    EXPECT_EQ(cameras.rowCount(), 12);
    EXPECT_EQ(cameras.columnCount(), 10);
    EXPECT_EQ(markers.rowCount(), 4);
    EXPECT_EQ(markers.columnCount(), 11);

    const reference::ReferenceSummary summary = cameras.summary();
    EXPECT_EQ(summary.camera_count, 12U);
    EXPECT_EQ(summary.marker_count, 4U);
    EXPECT_NEAR(summary.camera_rmse_m, analytic_camera_rmse(12), 1e-3);
    // Marker offsets are 1, 2, 3 and 4 metres: sqrt(30 / 4).
    EXPECT_NEAR(summary.control_rmse_m, std::sqrt(30.0 / 4.0), 1e-3);
    EXPECT_EQ(summary.control_count, 4U);
    EXPECT_EQ(summary.check_count, 0U);
    EXPECT_FALSE(std::isfinite(summary.check_rmse_m));
    EXPECT_NE(pane().action("action.reference.settings"), nullptr);
    EXPECT_FALSE(pane().action("action.reference.settings")->isEnabled());
}

TEST_F(ReferencePane, EditEmitsCommand) {
    Chunk& chunk = build_chunk(12, 4);
    ReferenceModel& model = pane().cameras();
    int row = -1;
    for (int i = 0; i < model.rowCount(); ++i) {
        if (model.row_id(i) == 0) {
            row = i;
        }
    }
    ASSERT_GE(row, 0);
    ASSERT_TRUE(model.row(row).source.has_value());
    const double before = model.row(row).source->x();
    const std::size_t before_size = app().undo_stack().size();

    const auto camera_x = [&chunk]() {
        for (const Camera& candidate : chunk.cameras()) {
            if (candidate.id == 0 && candidate.reference.location.has_value()) {
                return candidate.reference.location->x();
            }
        }
        return std::numeric_limits<double>::quiet_NaN();
    };

    const QModelIndex index = model.index(row, ReferenceModel::SourceXColumn);
    ASSERT_TRUE(model.setData(index, 42.0, Qt::EditRole));
    EXPECT_EQ(app().undo_stack().size(), before_size + 1);
    EXPECT_DOUBLE_EQ(camera_x(), 42.0);

    app().undo();
    EXPECT_DOUBLE_EQ(camera_x(), before);
    EXPECT_EQ(app().undo_stack().position(), before_size);
}

TEST_F(ReferencePane, SortByErrorDescending) {
    build_chunk(12, 4);
    ReferenceModel& model = pane().cameras();
    double previous = std::numeric_limits<double>::infinity();
    int measured = 0;
    for (int i = 0; i < model.rowCount(); ++i) {
        const std::optional<double> error = model.error_m(i);
        ASSERT_TRUE(error.has_value());
        EXPECT_LE(*error, previous);
        previous = *error;
        ++measured;
    }
    EXPECT_EQ(measured, 12);
    ASSERT_GE(model.rowCount(), 1);
    EXPECT_NEAR(*model.error_m(0), 5.5, 1e-9);
}

TEST_F(ReferencePane, CheckPointMovesOutOfControlRmse) {
    Chunk& chunk = build_chunk(12, 4);
    ReferenceModel& markers = pane().markers();
    const double before = markers.summary().control_rmse_m;
    ASSERT_GT(before, 0.0);
    // The first row carries the largest error (marker 103, four metres).
    const QModelIndex index = markers.index(0, ReferenceModel::LabelColumn);
    ASSERT_TRUE(markers.setData(
        index, QVariant(static_cast<int>(Qt::Unchecked)), Qt::CheckStateRole));
    const ReferenceSummary& summary = markers.summary();
    EXPECT_NEAR(summary.control_rmse_m, std::sqrt((1.0 + 4.0 + 9.0) / 3.0), 1e-3);
    EXPECT_NEAR(summary.check_rmse_m, 4.0, 1e-3);
    EXPECT_EQ(summary.control_count, 3U);
    EXPECT_EQ(summary.check_count, 1U);
    // The role, not the projection status, changed.
    const auto role_of_103 = [&chunk]() {
        for (const Marker& marker : chunk.markers()) {
            if (marker.id == 103) {
                return std::optional<reference::MarkerReferenceRole>(
                    reference::marker_role(marker));
            }
        }
        return std::optional<reference::MarkerReferenceRole>();
    };
    ASSERT_TRUE(role_of_103().has_value());
    EXPECT_EQ(*role_of_103(), reference::MarkerReferenceRole::CheckPoint);
    for (const Marker& marker : chunk.markers()) {
        if (marker.id == 103) {
            EXPECT_TRUE(marker.valid);
        }
    }
    app().undo();
    ASSERT_TRUE(role_of_103().has_value());
    EXPECT_EQ(*role_of_103(), reference::MarkerReferenceRole::Gcp);
}

TEST_F(ReferencePane, SourceEditIsConvertedBackToTheSlotCrs) {
    Document& document = app().document();
    Chunk& chunk = document.workspace().add_chunk("CRS");
    chunk.set_coordinate_system(crs::CoordinateSystem::from_epsg(32633));
    chunk.set_camera_crs(crs::CoordinateSystem::from_epsg(4979));
    Camera camera;
    camera.id = 0;
    camera.label = "crs";
    camera.enabled = true;
    camera.reference.location = Eigen::Vector3d(15.0, 50.0, 100.0);
    camera.reference.accuracy = Eigen::Vector3d::Constant(0.02);
    chunk.cameras().push_back(camera);
    document.set_active_chunk_id(chunk.id());
    pane().refresh();

    ReferenceModel& model = pane().cameras();
    ASSERT_EQ(model.rowCount(), 1);
    ASSERT_TRUE(model.row(0).source.has_value());
    const Eigen::Vector3d displayed = *model.row(0).source;
    const double typed = displayed.x() + 10.0;  // metres in the projected display
    ASSERT_TRUE(model.setData(model.index(0, ReferenceModel::SourceXColumn), typed,
                              Qt::EditRole));

    const crs::CoordinateSystem display = chunk.coordinate_system();
    const crs::CoordinateSystem slot = crs::CoordinateSystem::from_epsg(4979);
    const crs::CrsTransform inverse = crs::CrsTransform::create(display, slot);
    const crs::CrsPoint expected =
        inverse.forward(crs::CrsPoint{typed, displayed.y(), displayed.z()});
    ASSERT_TRUE(chunk.cameras()[0].reference.location.has_value());
    const Eigen::Vector3d written = *chunk.cameras()[0].reference.location;
    EXPECT_NEAR(written.x(), expected.x, 1e-9);
    EXPECT_NEAR(written.y(), expected.y, 1e-9);
    EXPECT_NEAR(written.z(), expected.z, 1e-6);
}

TEST_F(ReferencePane, SwitchingChunkRefreshesRows) {
    Chunk& first = build_chunk(3, 2);
    Chunk& second = build_chunk(5, 1);
    // The second build_chunk left it active.
    EXPECT_EQ(pane().cameras().rowCount(), 5);
    EXPECT_EQ(pane().cameras().chunk_id(), second.id());

    auto* workspace = widget<WorkspacePane>("pane.workspace");
    ASSERT_NE(workspace, nullptr);
    ASSERT_TRUE(workspace->activate_chunk(first.id()));
    EXPECT_EQ(pane().cameras().chunk_id(), first.id());
    EXPECT_EQ(pane().cameras().rowCount(), 3);
    EXPECT_EQ(pane().markers().rowCount(), 2);

    // Edits go to the chunk that is now active.
    ReferenceModel& cameras = pane().cameras();
    int row = -1;
    for (int i = 0; i < cameras.rowCount(); ++i) {
        if (cameras.row_id(i) == 0) {
            row = i;
        }
    }
    ASSERT_GE(row, 0);
    ASSERT_TRUE(cameras.setData(cameras.index(row, ReferenceModel::SourceXColumn), 7.0,
                                Qt::EditRole));
    EXPECT_DOUBLE_EQ(first.cameras()[0].reference.location->x(), 7.0);
    EXPECT_DOUBLE_EQ(second.cameras()[0].reference.location->x(), -5.5);

    ASSERT_TRUE(workspace->activate_chunk(second.id()));
    EXPECT_EQ(pane().cameras().rowCount(), 5);
    EXPECT_EQ(pane().cameras().chunk_id(), second.id());
}

TEST_F(ReferencePane, ThousandRowsWithinBudget) {
    build_chunk(1000, 0);
    ReferenceModel& model = pane().cameras();
    const auto start = std::chrono::steady_clock::now();
    model.refresh();
    const auto end = std::chrono::steady_clock::now();
    EXPECT_EQ(model.rowCount(), 1000);
    const double millis =
        std::chrono::duration<double, std::milli>(end - start).count();
    if (alphashape::test::perf_asserts_enabled()) {
        EXPECT_LE(millis, 100.0) << "reference pane build took " << millis << " ms";
    }
}

}  // namespace
