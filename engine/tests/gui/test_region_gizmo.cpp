// RegionGizmo.* / TransformDialog.*: the region box gizmo and the numeric object
// transform dialog (WP-14.15). The gizmo core is driven with synthetic rays; the
// undo/escape tests go through the viewport's programmatic drag entry points, so
// they exercise the same command path as a real mouse release.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <QLineEdit>
#include <QtTest/QTest>

#include "apps/gui/app.hpp"
#include "apps/gui/transform_dialog.hpp"
#include "apps/gui/viewer/region_gizmo.hpp"
#include "apps/gui/viewer/viewport_widget.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "harness.hpp"

using namespace alphashape;
// Deliberately not `using namespace alphashape::gui`: the gtest fixture must be named
// TransformDialog, and importing the whole namespace would make that name ambiguous with
// gui::TransformDialog. The names used unqualified below are imported one by one instead.
using alphashape::gui::GizmoDrag;
using alphashape::gui::GizmoHandle;
using alphashape::gui::GizmoHandleKind;
using alphashape::gui::GizmoRay;
using alphashape::gui::GizmoState;
using alphashape::gui::RegionGizmoMode;
using alphashape::gui::TransformValues;
using alphashape::gui::ViewportWidget;
using alphashape::gui::transform_from_values;
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
    Eigen::Vector2d pixel_of(const Eigen::Vector3d& world) {
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

class TransformDialogFixture : public GuiTest {
protected:
    Chunk& make_chunk_for_dialog() {
        Document& document = app().document();
        Chunk& chunk = document.workspace().add_chunk("Object");
        document.set_active_chunk_id(chunk.id());
        return chunk;
    }
};

}  // namespace

// Global aliases: a declaration in the global namespace outranks the names a
// `using namespace alphashape::gui` directive brings in, so the gtest fixture
// IDs are exactly RegionGizmo.* / TransformDialog.* without ambiguity with the
// gui::TransformDialog class.
using RegionGizmo = RegionGizmoFixture;
using TransformDialog = TransformDialogFixture;

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
    const double column_norm = chunk.transform().topLeftCorner<3, 3>().col(0).norm();
    EXPECT_NEAR(column_norm, 2.0, 1e-9);

    ASSERT_TRUE(dialog.reset_transform());
    const Transform identity = Transform::Identity();
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            EXPECT_NEAR(chunk.transform()(row, col), identity(row, col), 1e-9);
        }
    }
}

TEST_F(TransformDialog, RejectsNonPositiveScale) {
    Chunk& chunk = make_chunk_for_dialog();
    alphashape::gui::TransformDialog dialog(app(), chunk.id());
    const Transform identity = chunk.transform();
    for (const char* text : {"0", "-2", "not-a-number"}) {
        dialog.field(QStringLiteral("scale"))->setText(QString::fromLatin1(text));
        EXPECT_FALSE(dialog.apply()) << text;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                EXPECT_DOUBLE_EQ(chunk.transform()(row, col), identity(row, col));
            }
        }
    }

    TransformValues values;
    values.scale = 0.0;
    EXPECT_THROW(transform_from_values(values), ValueError);
    values.scale = -1.0;
    EXPECT_THROW(transform_from_values(values), ValueError);
    values.scale = std::numeric_limits<double>::infinity();
    EXPECT_THROW(transform_from_values(values), ValueError);
}

TEST_F(TransformDialog, ParseFailureKeepsLastValidValue) {
    Chunk& chunk = make_chunk_for_dialog();
    alphashape::gui::TransformDialog dialog(app(), chunk.id());
    dialog.field(QStringLiteral("scale"))->setText(QStringLiteral("2"));
    dialog.field(QStringLiteral("yaw"))->setText(QStringLiteral("30"));
    ASSERT_TRUE(dialog.apply());
    const Transform applied = chunk.transform();

    dialog.field(QStringLiteral("scale"))->setText(QStringLiteral("two"));
    const TransformValues values = dialog.values_from_fields();
    EXPECT_NEAR(values.scale, 2.0, 1e-9);
    EXPECT_NEAR(values.yaw_degrees, 30.0, 1e-9);
    // A typo blocks the push and leaves the applied transform alone.
    EXPECT_FALSE(dialog.apply());
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            EXPECT_DOUBLE_EQ(chunk.transform()(row, col), applied(row, col));
        }
    }
}

TEST_F(RegionGizmo, RotateRingTurnsAboutZ) {
    GizmoState state;
    state.region.centre = Eigen::Vector3d(1.0, 2.0, 3.0);
    state.region.size = Eigen::Vector3d::Constant(2.0);
    state.transform = Transform::Identity();
    state.transform(0, 3) = 7.0;  // a placement the gizmo must not disturb

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

    // A quarter turn about the box's local +Z maps its local +X to +Y.
    const Eigen::Vector3d turned = after.region.rotation * Eigen::Vector3d::UnitX();
    EXPECT_NEAR(turned.x(), 0.0, 1e-12);
    EXPECT_NEAR(turned.y(), 1.0, 1e-12);
    EXPECT_NEAR(turned.z(), 0.0, 1e-12);
    EXPECT_FALSE(after.region.rotation == state.region.rotation);
    EXPECT_NEAR(after.region.size.x(), 2.0, 1e-12);
    EXPECT_NEAR(after.region.size.y(), 2.0, 1e-12);
    EXPECT_NEAR(after.region.size.z(), 2.0, 1e-12);
    EXPECT_TRUE(after.region.centre == state.region.centre);
    // The chunk transform is exactly unchanged: the rotation lives in Region.
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            EXPECT_DOUBLE_EQ(after.transform(row, col), state.transform(row, col));
        }
    }
}

TEST_F(RegionGizmo, DragFaceOnRotatedBox) {
    GizmoState state;
    state.region.centre = Eigen::Vector3d(1.0, 2.0, 3.0);
    state.region.size = Eigen::Vector3d::Constant(2.0);
    state.region.rotation =
        Eigen::AngleAxisd(30.0 * std::numbers::pi / 180.0, Eigen::Vector3d::UnitZ())
            .toRotationMatrix();
    state.transform = Transform::Identity();

    GizmoHandle handle;
    handle.mode = RegionGizmoMode::Resize;
    handle.kind = GizmoHandleKind::Face;
    handle.axis = 0;
    handle.sign = 1;

    const Eigen::Vector3d axis = state.region.rotation.col(0);
    const GizmoRay first{Eigen::Vector3d(state.region.centre + state.region.rotation *
                                             Eigen::Vector3d(1.0, -1.0, 0.0)),
                         (state.region.rotation * Eigen::Vector3d(-1.0, 1.0, 0.0)).normalized()};
    GizmoRay second = first;
    second.origin += 0.4 * axis;
    const std::optional<GizmoDrag> drag = begin_drag_with_handle(state, handle, first);
    ASSERT_TRUE(drag.has_value());
    const GizmoState after = update_drag(*drag, second);

    // Only the local X extent changes, by the 0.4 the cursor travelled along
    // that axis.
    EXPECT_NEAR(after.region.size.x(), 2.4, 2.4 * 0.02);
    EXPECT_NEAR(after.region.size.y(), 2.0, 1e-9);
    EXPECT_NEAR(after.region.size.z(), 2.0, 1e-9);
    EXPECT_TRUE(after.region.rotation == state.region.rotation);
    // The opposite face stays fixed, so the centre moves half the growth.
    EXPECT_LT((after.region.centre - (state.region.centre + 0.2 * axis)).norm(), 1e-9);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            EXPECT_DOUBLE_EQ(after.transform(row, col), state.transform(row, col));
        }
    }
}

TEST_F(RegionGizmo, CornerResizeGrowsAndKeepsOppositeCornerFixed) {
    for (int corner = 0; corner < 8; ++corner) {
        GizmoState state;
        state.region.centre = Eigen::Vector3d(1.0, 2.0, 3.0);
        state.region.size = Eigen::Vector3d::Constant(2.0);
        state.region.rotation =
            Eigen::AngleAxisd(20.0 * std::numbers::pi / 180.0, Eigen::Vector3d(0.2, 0.5, 0.8)
                                                                     .normalized())
                .toRotationMatrix();

        GizmoHandle handle;
        handle.mode = RegionGizmoMode::Resize;
        handle.kind = GizmoHandleKind::Corner;
        Eigen::Vector3d signs = Eigen::Vector3d::Zero();
        for (int axis = 0; axis < 3; ++axis) {
            const int sign = (corner & (1 << axis)) != 0 ? 1 : -1;
            handle.axes.push_back(axis);
            handle.signs.push_back(sign);
            signs[axis] = static_cast<double>(sign);
        }

        constexpr double kOutward = 0.3;
        const Eigen::Vector3d outward = state.region.rotation * signs;
        const GizmoRay first{Eigen::Vector3d(5.0, -4.0, 6.0),
                             Eigen::Vector3d(0.31, 0.47, 0.83).normalized()};
        GizmoRay second = first;
        second.origin += kOutward * outward;
        const std::optional<GizmoDrag> drag = begin_drag_with_handle(state, handle, first);
        ASSERT_TRUE(drag.has_value());
        const GizmoState after = update_drag(*drag, second);

        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_NEAR(after.region.size[axis], 2.0 + kOutward, 1e-9) << "corner " << corner;
        }
        // The corner opposite the dragged one must not move.
        Eigen::Vector3d opposite_offset = Eigen::Vector3d::Zero();
        for (int axis = 0; axis < 3; ++axis) {
            opposite_offset[axis] = -signs[axis] * after.region.size[axis] * 0.5;
        }
        const Eigen::Vector3d opposite = after.region.centre +
                                         after.region.rotation * opposite_offset;
        const Eigen::Vector3d expected =
            state.region.centre + state.region.rotation * (-signs);
        EXPECT_LT((opposite - expected).norm(), 1e-9) << "corner " << corner;
    }
}

TEST_F(RegionGizmo, MoveOnNegativeFaceFollowsCursor) {
    for (int axis = 0; axis < 3; ++axis) {
        GizmoState state;
        state.region.centre = Eigen::Vector3d(1.0, 2.0, 3.0);
        state.region.size = Eigen::Vector3d::Constant(2.0);

        GizmoHandle handle;
        handle.mode = RegionGizmoMode::Move;
        handle.kind = GizmoHandleKind::Face;
        handle.axis = axis;
        handle.sign = -1;

        Eigen::Vector3d unit = Eigen::Vector3d::Zero();
        unit[axis] = 1.0;
        const GizmoRay first{Eigen::Vector3d(5.0, -4.0, 6.0),
                             Eigen::Vector3d(0.31, 0.47, 0.83).normalized()};
        GizmoRay second = first;
        second.origin += 0.5 * unit;
        const std::optional<GizmoDrag> drag = begin_drag_with_handle(state, handle, first);
        ASSERT_TRUE(drag.has_value());
        const GizmoState after = update_drag(*drag, second);

        // Grabbing the -face and dragging toward +axis moves the centre +axis.
        for (int other = 0; other < 3; ++other) {
            const double expected = state.region.centre[other] + (other == axis ? 0.5 : 0.0);
            EXPECT_NEAR(after.region.centre[other], expected, 1e-9)
                << "axis " << axis << " component " << other;
        }
        for (int other = 0; other < 3; ++other) {
            EXPECT_NEAR(after.region.size[other], 2.0, 1e-12);
        }
    }
}

TEST_F(RegionGizmo, FunctionKeyMidDragIsIgnored) {
    Chunk& chunk = make_chunk();
    ViewportWidget* view = window().viewport();
    view->resize(800, 600);
    view->set_region_mode(RegionGizmoMode::Move);

    const Region original = chunk.region();
    const Eigen::Vector2d start = pixel_of(Eigen::Vector3d::Zero());
    const Eigen::Vector2d end = start + Eigen::Vector2d(60.0, -20.0);

    ASSERT_TRUE(view->begin_region_drag(start));
    view->update_region_drag(start / 2.0 + end / 2.0);
    view->update_region_drag(end);
    ASSERT_TRUE(view->end_region_drag());
    const Region control = chunk.region();
    app().undo();
    ASSERT_TRUE(chunk.region() == original);

    ASSERT_TRUE(view->begin_region_drag(start));
    view->update_region_drag(start / 2.0 + end / 2.0);
    QTest::keyClick(view, Qt::Key_F);
    view->update_region_drag(end);
    ASSERT_TRUE(view->end_region_drag());
    EXPECT_TRUE(chunk.region() == control);
}

TEST_F(RegionGizmo, DocumentReplacementCancelsDragWithoutCommand) {
    make_chunk();
    ViewportWidget* view = window().viewport();
    view->resize(800, 600);
    view->set_region_mode(RegionGizmoMode::Move);

    const Eigen::Vector2d start = pixel_of(Eigen::Vector3d::Zero());
    ASSERT_TRUE(view->begin_region_drag(start));
    view->update_region_drag(start + Eigen::Vector2d(40.0, 10.0));
    ASSERT_TRUE(view->region_drag_active());

    // Replacing the document destroys the chunk the drag belonged to; the drag
    // is dropped before release, so no command reaches the new document.
    app().new_document(std::string());
    EXPECT_FALSE(view->region_drag_active());
    EXPECT_FALSE(view->end_region_drag());
    EXPECT_EQ(app().undo_stack().size(), 0U);
}

TEST_F(RegionGizmo, ActiveChunkChangeCancelsDragWithoutCommand) {
    Chunk& first = make_chunk();
    Document& document = app().document();
    Chunk& second = document.workspace().add_chunk("Other");
    document.set_active_chunk_id(first.id());

    ViewportWidget* view = window().viewport();
    view->resize(800, 600);
    view->set_region_mode(RegionGizmoMode::Move);

    const Region original = first.region();
    const std::size_t before = app().undo_stack().size();
    const Eigen::Vector2d start = pixel_of(Eigen::Vector3d::Zero());
    ASSERT_TRUE(view->begin_region_drag(start));
    view->update_region_drag(start + Eigen::Vector2d(40.0, 10.0));
    document.set_active_chunk_id(second.id());
    Q_EMIT app().active_chunk_changed(second.id());

    EXPECT_FALSE(view->region_drag_active());
    EXPECT_FALSE(view->end_region_drag());
    EXPECT_EQ(app().undo_stack().size(), before);
    EXPECT_TRUE(first.region() == original);
}

