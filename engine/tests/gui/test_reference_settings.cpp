// ReferenceSettings.*: the reference settings dialog (WP-14.22a). PROJ build
// only, because the dialog hosts the CRS pickers.

#include <cstdint>

#include <gtest/gtest.h>

#include <QDoubleSpinBox>

#include "apps/gui/app.hpp"
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

}  // namespace
