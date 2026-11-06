// RegionGizmo.* / TransformDialog.*: the region box gizmo and the numeric object
// transform dialog (WP-14.15). The gizmo core is driven with synthetic rays; the
// undo/escape tests go through the viewport's programmatic drag entry points, so
// they exercise the same command path as a real mouse release.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "apps/gui/app.hpp"
#include "apps/gui/transform_dialog.hpp"
#include "apps/gui/viewer/region_gizmo.hpp"
#include "apps/gui/viewer/viewport_widget.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "harness.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;

namespace {

class RegionGizmoFixture : public GuiTest {
protected:
    Chunk& make_chunk() {
        Document& document = app().document();
        Chunk& chunk = document.workspace().add_chunk("Region");
        chunk.region().centre = Eigen::Vector3d::Zero();
        chunk.region().size = Eigen::Vector3d::Constant(2.0);
        chunk.transform() = Transform::Identity();
        document.set_active_chunk_id(chunk.id());
        return chunk;
    }

    /// Widget pixel of a world point, using the viewport's current camera and size.
    Eigen::Vector2d pixel_of(const Eigen::Vector3d& world) const {
        ViewportWidget* view = window().viewport();
        const double width = static_cast<double>(std::max(view->width(), 1));
        const double height = static_cast<double>(std::max(view->height(), 1));
        const double aspect = width / height;
        const std::optional<Eigen::Vector3d> ndc = view->camera().project(world, aspect);
        EXPECT_TRUE(ndc.has_value());
        if (!ndc) {
            return Eigen::Vector2d::Zero();
        }
        return Eigen::Vector2d((ndc->x() * 0.5 + 0.5) * width,
                               (0.5 - ndc->y() * 0.5) * height);
    }

    /// Drag one face of the box by a pixel offset in Move mode and release it.
    bool drag_move(const Eigen::Vector2d& start, const Eigen::Vector2d& end) {
        ViewportWidget* view = window().viewport();
        view->set_region_mode(RegionGizmoMode::Move);
        if (!view->begin_region_drag(start)) {
            return false;
        }
        view->update_region_drag(end);
        return view->end_region_drag();
    }
};

using RegionGizmo = RegionGizmoFixture;

class TransformDialogFixture : public GuiTest {
protected:
    Chunk& make_chunk_for_dialog() {
        Document& document = app().document();
        Chunk& chunk = document.workspace().add_chunk("Object");
        document.set_active_chunk_id(chunk.id());
        return chunk;
    }
};

using TransformDialog = TransformDialogFixture;

}  // namespace

TEST_F(RegionGizmo, DragFaceChangesOneAxis) {
    GizmoState state;
    state.region.centre = Eigen::Vector3d::Zero();
    state.region.size = Eigen::Vector3d::Constant(2.0);
    state.transform = Transform::Identity();

    GizmoHandle handle;
    handle.mode = RegionGizmoMode::Resize;
    handle.kind = GizmoHandleKind::Face;
    handle.axis = 0;
    handle.sign = 1;

    // Both rays pass near the +X face; the second origin is 0.4 further along +X, so the
    // closest-point parameter on the face-normal line advances by exactly 0.4.
    const GizmoRay first{Eigen::Vector3d(2.0, -1.0, 0.0),
                         Eigen::Vector3d(-1.0, 1.0, 0.0).normalized()};
    const GizmoRay second{Eigen::Vector3d(2.4, -1.0, 0.0),
                          Eigen::Vector3d(-1.0, 1.0, 0.0).normalized()};
    const std::optional<GizmoDrag> drag = begin_drag_with_handle(state, handle, first);
    ASSERT_TRUE(drag.has_value());
    const GizmoState after = update_drag(*drag, second);

    const double expected_world_delta = 0.4;
    const double actual = after.region.size.x() - state.region.size.x();
    EXPECT_NEAR(actual, expected_world_delta, std::abs(expected_world_delta) * 0.02);
    // The other two axes and the rotation are untouched.
    EXPECT_NEAR(after.region.size.y(), 2.0, 1e-12);
    EXPECT_NEAR(after.region.size.z(), 2.0, 1e-12);
    EXPECT_NEAR(after.region.centre.y(), 0.0, 1e-12);
    EXPECT_NEAR(after.region.centre.z(), 0.0, 1e-12);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            EXPECT_NEAR(after.transform(row, col), state.transform(row, col), 1e-12);
        }
    }
}

TEST_F(RegionGizmo, UndoRestores) {
    Chunk& chunk = make_chunk();
    ViewportWidget* view = window().viewport();
    view->resize(800, 600);
    view->set_region_mode(RegionGizmoMode::Move);

    const Region original = chunk.region();
    const Eigen::Vector2d start = pixel_of(Eigen::Vector3d::Zero());
    ASSERT_TRUE(view->begin_region_drag(start));
    view->update_region_drag(start + Eigen::Vector2d(60.0, -20.0));
    ASSERT_TRUE(view->end_region_drag());
    EXPECT_FALSE(chunk.region() == original);

    app().undo();
    EXPECT_TRUE(chunk.region() == original);
}

TEST_F(RegionGizmo, OneCommandPerDrag) {
    Chunk& chunk = make_chunk();
    ViewportWidget* view = window().viewport();
    view->resize(800, 600);
    view->set_region_mode(RegionGizmoMode::Move);

    const std::size_t before = app().undo_stack().size();
    const Eigen::Vector2d start = pixel_of(Eigen::Vector3d::Zero());
    ASSERT_TRUE(view->begin_region_drag(start));
    for (int i = 1; i <= 20; ++i) {
        const double step = static_cast<double>(i);
        view->update_region_drag(start + Eigen::Vector2d(step, -step * 0.5));
    }
    ASSERT_TRUE(view->end_region_drag());
    EXPECT_EQ(app().undo_stack().size(), before + 1);
    EXPECT_FALSE(chunk.region() == Region{});
}

TEST_F(RegionGizmo, EscapeCancels) {
    Chunk& chunk = make_chunk();
    ViewportWidget* view = window().viewport();
    view->resize(800, 600);
    view->set_region_mode(RegionGizmoMode::Move);

    const Region original = chunk.region();
    const std::size_t before = app().undo_stack().size();
    const Eigen::Vector2d start = pixel_of(Eigen::Vector3d::Zero());
    ASSERT_TRUE(view->begin_region_drag(start));
    view->update_region_drag(start + Eigen::Vector2d(80.0, 10.0));
    view->cancel_region_drag();

    EXPECT_TRUE(chunk.region() == original);
    EXPECT_EQ(app().undo_stack().size(), before);
    EXPECT_FALSE(view->region_drag_active());
}

TEST_F(TransformDialog, ApplyAndReset) {
    Chunk& chunk = make_chunk_for_dialog();
    alphashape::gui::TransformDialog dialog(app(), chunk.id());
    dialog.field(QStringLiteral("centre.x"))->setText(QStringLiteral("1.5"));
    dialog.field(QStringLiteral("centre.y"))->setText(QStringLiteral("-2.25"));
    dialog.field(QStringLiteral("centre.z"))->setText(QStringLiteral("3"));
    dialog.field(QStringLiteral("scale"))->setText(QStringLiteral("2"));
    dialog.field(QStringLiteral("yaw"))->setText(QStringLiteral("30"));
    dialog.field(QStringLiteral("pitch"))->setText(QStringLiteral("10"));
    dialog.field(QStringLiteral("roll"))->setText(QStringLiteral("-5"));

    ASSERT_TRUE(dialog.apply());
    const Transform expected = transform_from_values(dialog.values_from_fields());
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            EXPECT_NEAR(chunk.transform()(row, col), expected(row, col), 1e-9);
        }
    }
    EXPECT_NEAR(chunk.transform()(0, 3), 1.5, 1e-9);
    EXPECT_NEAR(chunk.transform()(1, 3), -2.25, 1e-9);
    EXPECT_NEAR(chunk.transform()(2, 3), 3.0, 1e-9);
    EXPECT_NEAR(chunk.transform().topLeftCorner<3, 3>().col(0).norm(), 2.0, 1e-9);

    ASSERT_TRUE(dialog.reset_transform());
    const Transform identity = Transform::Identity();
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            EXPECT_NEAR(chunk.transform()(row, col), identity(row, col), 1e-9);
        }
    }
}

TEST_F(RegionGizmo, RotateRingTurnsAboutZ) {
    GizmoState state;
    state.region.centre = Eigen::Vector3d::Zero();
    state.region.size = Eigen::Vector3d::Constant(2.0);

    GizmoHandle ring;
    ring.mode = RegionGizmoMode::Rotate;
    ring.kind = GizmoHandleKind::Ring;
    ring.axis = 2;
    ring.point = Eigen::Vector3d(1.0, 0.0, 0.0);

    const GizmoRay first{Eigen::Vector3d(10.0, 0.0, 5.0), Eigen::Vector3d(0.0, 0.0, -1.0)};
    const GizmoRay quarter{Eigen::Vector3d(0.0, 10.0, 5.0), Eigen::Vector3d(0.0, 0.0, -1.0)};
    const std::optional<GizmoDrag> drag = begin_drag_with_handle(state, ring, first);
    ASSERT_TRUE(drag.has_value());
    const GizmoState after = update_drag(*drag, quarter);

    // A quarter turn about +Z maps +X to +Y.
    const Eigen::Vector3d turned = after.transform.topLeftCorner<3, 3>() *
                                   Eigen::Vector3d::UnitX();
    EXPECT_NEAR(turned.x(), 0.0, 1e-12);
    EXPECT_NEAR(turned.y(), 1.0, 1e-12);
    EXPECT_NEAR(turned.z(), 0.0, 1e-12);
    EXPECT_TRUE(after.region == state.region);
}

