// ReferenceSettings.*: the reference settings dialog (WP-14.22a). PROJ build
// only, because the dialog hosts the CRS pickers.

#include <cstdint>

#include <gtest/gtest.h>

#include <QDoubleSpinBox>

#include "apps/gui/app.hpp"
#include "apps/gui/modal_hooks.hpp"
#include "apps/gui/reference_settings_dialog.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "harness.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;

namespace {

class ReferenceSettingsFixture : public GuiTest {};

using ReferenceSettings = ReferenceSettingsFixture;

TEST_F(ReferenceSettings, AccuracyPersists) {
    Document& document = app().document();
    Chunk& chunk = document.workspace().add_chunk("Settings");
    document.set_active_chunk_id(chunk.id());
    const std::int32_t chunk_id = chunk.id();
    {
        alphashape::gui::ReferenceSettingsDialog dialog(app());
        QDoubleSpinBox* accuracy =
            dialog.accuracy_editor(QStringLiteral("camera_location"));
        ASSERT_NE(accuracy, nullptr);
        accuracy->setValue(2.5);
        dialog.accept();
    }
    EXPECT_DOUBLE_EQ(chunk.reference_settings().camera_location_accuracy_m, 2.5);
    EXPECT_EQ(app().undo_stack().size(), 1U);

    const QString path = scratch(QStringLiteral("reference.psx"));
    app().save_document_as(path.toStdString());
    app().open_document(path.toStdString());

    const Chunk* reloaded = app().read_document(
        [](const Document& loaded) { return loaded.active_chunk(); });
    ASSERT_NE(reloaded, nullptr);
    EXPECT_EQ(reloaded->id(), chunk_id);
    EXPECT_DOUBLE_EQ(reloaded->reference_settings().camera_location_accuracy_m,
                     2.5);
}

// A failing command must undo the ones before it: the dialog applies its edits
// as one transaction, never one command at a time.
TEST_F(ReferenceSettings, FailedCommandRollsBackTheWholeEdit) {
    modal_hooks().error = [](QWidget*, const QString&, const QString&) {};
    Document& document = app().document();
    Chunk& chunk = document.workspace().add_chunk("Settings");
    document.set_active_chunk_id(chunk.id());
    const alphashape::ReferenceSettings before = chunk.reference_settings();
    ASSERT_FALSE(chunk.explicit_camera_crs().has_value());

    alphashape::gui::ReferenceSettingsDialog dialog(app());
    dialog.set_crs_slot(0, CRS(crs::CoordinateSystem::from_epsg(25832)));
    QDoubleSpinBox* scalebar = dialog.accuracy_editor(QStringLiteral("scalebar"));
    ASSERT_NE(scalebar, nullptr);
    scalebar->setRange(0.0, 1e6);
    scalebar->setValue(0.0);  // the command rejects a non-positive accuracy
    dialog.accept();

    EXPECT_FALSE(chunk.explicit_camera_crs().has_value());
    EXPECT_TRUE(chunk.reference_settings() == before);
    EXPECT_EQ(app().undo_stack().size(), 0U);
}

TEST_F(ReferenceSettings, SlotChangeOverReferencesAsksFirst) {
    Document& document = app().document();
    Chunk& chunk = document.workspace().add_chunk("Settings");
    document.set_active_chunk_id(chunk.id());
    Camera camera;
    camera.id = 0;
    camera.sensor_id = 0;
    camera.reference.location = Eigen::Vector3d(9.0, 48.0, 300.0);
    chunk.cameras().push_back(camera);

    QString asked;
    modal_hooks().choose_reference_change = [&asked](QWidget*, const QString& slot) {
        asked = slot;
        return ReferenceChangeChoice::Cancel;
    };
    alphashape::gui::ReferenceSettingsDialog dialog(app());
    dialog.set_crs_slot(0, CRS(crs::CoordinateSystem::from_epsg(25832)));
    dialog.accept();
    EXPECT_EQ(asked, QStringLiteral("camera"));
    EXPECT_FALSE(chunk.explicit_camera_crs().has_value());
    EXPECT_EQ(app().undo_stack().size(), 0U);

    // Keeping the values relabels the slot without touching them.
    modal_hooks().choose_reference_change = [](QWidget*, const QString&) {
        return ReferenceChangeChoice::Relabel;
    };
    dialog.accept();
    ASSERT_TRUE(chunk.explicit_camera_crs().has_value());
    EXPECT_EQ(chunk.explicit_camera_crs()->epsg, 25832);
    EXPECT_DOUBLE_EQ(chunk.cameras()[0].reference.location->x(), 9.0);
}

}  // namespace
