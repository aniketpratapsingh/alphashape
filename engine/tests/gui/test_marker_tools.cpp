// MarkerTools.*: the Detect Markers dialog (a TaskDialog plus a target
// preview) and the Print Markers sheet generator. Both are driven through
// their widgets, and the save path comes from the modal hook so no test opens
// a file dialog.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <variant>

#include <QByteArray>
#include <QCheckBox>
#include <QColor>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QLabel>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QString>

#include "apps/gui/marker_tools_dialogs.hpp"
#include "apps/gui/modal_hooks.hpp"
#include "common/tasks/registry.hpp"
#include "harness.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;

namespace {

class MarkerToolsTest : public GuiTest {
protected:
    QSettings& store() {
        if (!store_) {
            store_ = std::make_unique<QSettings>(scratch(QStringLiteral("markers.ini")),
                                                 QSettings::IniFormat);
        }
        return *store_;
    }

private:
    std::unique_ptr<QSettings> store_;
};

const ParamSpec& spec_of(const std::string& task, const std::string& name) {
    for (const ParamSpec& spec : builtin_registry().require(task).schema.specs()) {
        if (spec.name == name) {
            return spec;
        }
    }
    static const ParamSpec empty;
    return empty;
}

}  // namespace

TEST_F(MarkerToolsTest, DetectMarkersDialogPreviewFollowsTargetType) {
    DetectMarkersDialog dialog(builtin_registry(), store());
    auto* target = qobject_cast<QComboBox*>(dialog.editor("target_type"));
    ASSERT_NE(target, nullptr);
    ASSERT_NE(dialog.preview_label(), nullptr);

    target->setCurrentText(QStringLiteral("Circular12"));
    EXPECT_FALSE(dialog.preview_image().isNull());
    const QImage circular12 = dialog.preview_image();

    target->setCurrentText(QStringLiteral("Circular16"));
    EXPECT_FALSE(dialog.preview_image().isNull());
    EXPECT_TRUE(circular12 != dialog.preview_image())
        << "the two targets must draw different patterns";

    // Circular (uncoded) has no printable sheet: a placeholder, not an exception.
    target->setCurrentText(QStringLiteral("Circular"));
    EXPECT_TRUE(dialog.preview_image().isNull());
    EXPECT_EQ(dialog.preview_label()->text(),
              QStringLiteral("No preview for Circular"));
}

TEST_F(MarkerToolsTest, DetectMarkersDialogInvertedFlipsPreview) {
    DetectMarkersDialog dialog(builtin_registry(), store());
    auto* target = qobject_cast<QComboBox*>(dialog.editor("target_type"));
    auto* inverted = qobject_cast<QCheckBox*>(dialog.editor("inverted"));
    ASSERT_NE(target, nullptr);
    ASSERT_NE(inverted, nullptr);

    target->setCurrentText(QStringLiteral("Circular12"));
    const QImage normal = dialog.preview_image();
    ASSERT_FALSE(normal.isNull());

    inverted->setChecked(true);
    const QImage flipped = dialog.preview_image();
    ASSERT_FALSE(flipped.isNull());
    ASSERT_EQ(flipped.size(), normal.size());

    // Every sample inverts exactly: the sheet is 8-bit grayscale.
    for (int y = 0; y < normal.height(); y += 17) {
        for (int x = 0; x < normal.width(); x += 17) {
            EXPECT_EQ(qRed(flipped.pixel(x, y)), 255 - qRed(normal.pixel(x, y)))
                << "pixel " << x << "," << y;
        }
    }
}

TEST_F(MarkerToolsTest, DetectMarkersDialogToleranceRangeFromSchema) {
    DetectMarkersDialog dialog(builtin_registry(), store());
    const ParamSpec& spec = spec_of("DetectMarkers", "tolerance");
    ASSERT_EQ(spec.name, "tolerance");
    ASSERT_TRUE(std::holds_alternative<double>(spec.minimum));
    ASSERT_TRUE(std::holds_alternative<double>(spec.maximum));

    auto* spin = qobject_cast<QDoubleSpinBox*>(dialog.editor("tolerance"));
    ASSERT_NE(spin, nullptr);
    EXPECT_EQ(spin->minimum(), std::get<double>(spec.minimum));
    EXPECT_EQ(spin->maximum(), std::get<double>(spec.maximum));
    EXPECT_EQ(spin->minimum(), 0.0);
    EXPECT_EQ(spin->maximum(), 100.0);
}

TEST_F(MarkerToolsTest, DetectMarkersDialogParamsReachTask) {
    DetectMarkersDialog dialog(builtin_registry(), store());
    auto* target = qobject_cast<QComboBox*>(dialog.editor("target_type"));
    auto* tolerance = qobject_cast<QDoubleSpinBox*>(dialog.editor("tolerance"));
    auto* inverted = qobject_cast<QCheckBox*>(dialog.editor("inverted"));
    ASSERT_NE(target, nullptr);
    ASSERT_NE(tolerance, nullptr);
    ASSERT_NE(inverted, nullptr);

    target->setCurrentText(QStringLiteral("Circular16"));
    tolerance->setValue(37.5);
    inverted->setChecked(true);
    ASSERT_TRUE(dialog.validate());

    const Params params = dialog.params();
    EXPECT_EQ(std::get<std::string>(params.get("target_type")),
              std::string("Circular16"));
    EXPECT_EQ(std::get<double>(params.get("tolerance")), 37.5);
    EXPECT_EQ(std::get<bool>(params.get("inverted")), true);
}

TEST_F(MarkerToolsTest, PrintMarkersDialogPageCountUpdates) {
    PrintMarkersDialog dialog;
    ASSERT_NE(dialog.count_editor(), nullptr);
    EXPECT_EQ(dialog.page_count(), 1);
    EXPECT_EQ(dialog.page_count_label()->text(), QStringLiteral("1 page"));

    // A hundred Circular12 targets take three A4 pages (35 cells each); the
    // family has 122 codes, so the range stays legal and this exercises pages.
    dialog.count_editor()->setValue(100);
    EXPECT_EQ(dialog.page_count(), 3);
    EXPECT_EQ(dialog.page_count_label()->text(), QStringLiteral("3 pages"));
}

TEST_F(MarkerToolsTest, PrintMarkersDialogSavesPdf) {
    const QString path = scratch(QStringLiteral("markers.pdf"));
    modal_hooks().save_file = [path](QWidget*, const QString&, const QString&,
                                     const QString&) { return path; };
    PrintMarkersDialog dialog;
    ASSERT_TRUE(dialog.layout_valid());
    ASSERT_TRUE(dialog.save_button()->isEnabled());

    dialog.save_pdf();
    QFile file(path);
    ASSERT_TRUE(file.exists());
    ASSERT_TRUE(file.open(QIODevice::ReadOnly));
    EXPECT_EQ(file.read(4), QByteArray("%PDF"));
}

TEST_F(MarkerToolsTest, PrintMarkersDialogInvalidLayoutShowsError) {
    PrintMarkersDialog dialog;
    ASSERT_TRUE(dialog.layout_valid());

    // 400 mm does not fit any edge of A4, so the layout is invalid while the
    // user is typing. The dialog reports it inline and never throws.
    EXPECT_NO_THROW(dialog.target_size_editor()->setValue(400.0));
    EXPECT_FALSE(dialog.layout_valid());
    EXPECT_EQ(dialog.page_count(), 0);
    EXPECT_FALSE(dialog.save_button()->isEnabled());
    EXPECT_FALSE(dialog.error_label()->text().isEmpty());
    EXPECT_NO_THROW(dialog.refresh());

    // Shrinking it back restores a valid sheet.
    dialog.target_size_editor()->setValue(25.0);
    EXPECT_TRUE(dialog.layout_valid());
    EXPECT_TRUE(dialog.save_button()->isEnabled());
    EXPECT_TRUE(dialog.error_label()->text().isEmpty());
}

TEST_F(MarkerToolsTest, PrintMarkersDialogChessboardFieldsToggle) {
    PrintMarkersDialog dialog;
    ASSERT_NE(dialog.target_editor(), nullptr);
    // A coded sheet uses the code fields and not the board geometry.
    EXPECT_TRUE(dialog.target_size_editor()->isEnabled());
    EXPECT_TRUE(dialog.first_code_editor()->isEnabled());
    EXPECT_TRUE(dialog.count_editor()->isEnabled());
    EXPECT_FALSE(dialog.board_cols_editor()->isEnabled());
    EXPECT_FALSE(dialog.board_rows_editor()->isEnabled());
    EXPECT_FALSE(dialog.square_size_editor()->isEnabled());

    const int chessboard =
        dialog.target_editor()->findData(static_cast<int>(SheetTarget::Chessboard));
    ASSERT_GE(chessboard, 0);
    dialog.target_editor()->setCurrentIndex(chessboard);

    EXPECT_FALSE(dialog.target_size_editor()->isEnabled());
    EXPECT_FALSE(dialog.first_code_editor()->isEnabled());
    EXPECT_FALSE(dialog.count_editor()->isEnabled());
    EXPECT_TRUE(dialog.board_cols_editor()->isEnabled());
    EXPECT_TRUE(dialog.board_rows_editor()->isEnabled());
    EXPECT_TRUE(dialog.square_size_editor()->isEnabled());
    EXPECT_TRUE(dialog.layout_valid());
    EXPECT_EQ(dialog.page_count(), 1);
}
