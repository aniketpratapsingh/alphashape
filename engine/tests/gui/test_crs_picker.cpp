// CrsPicker.*: the CRS search widget (WP-14.22a). Built only with the
// PROJ-backed coordinate engine, like the reference pane it serves.

#include <string>

#include <gtest/gtest.h>

#include <QLineEdit>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QTabWidget>

#include "apps/gui/crs_picker.hpp"
#include "harness.hpp"

using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;

namespace {

class CrsPickerFixture : public GuiTest {};

using CrsPicker = CrsPickerFixture;

TEST_F(CrsPicker, SearchAndSelect) {
    alphashape::gui::CrsPicker picker;
    EXPECT_FALSE(picker.is_valid());
    picker.filter_edit()->setText(QStringLiteral("25832"));
    ASSERT_NE(picker.list_widget()->currentItem(), nullptr);
    EXPECT_TRUE(picker.is_valid());
    EXPECT_EQ(picker.coordinate_system().epsg(), 25832);
    EXPECT_NE(picker.coordinate_system().name().find("32N"), std::string::npos);
}

TEST_F(CrsPicker, InvalidWktIsRejected) {
    alphashape::gui::CrsPicker picker;
    picker.tabs()->setCurrentIndex(1);
    picker.wkt_edit()->setPlainText(QStringLiteral("not a coordinate system"));
    EXPECT_FALSE(picker.is_valid());
    EXPECT_FALSE(picker.error().isEmpty());
}

}  // namespace
