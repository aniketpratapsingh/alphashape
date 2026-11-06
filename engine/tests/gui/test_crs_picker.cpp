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

using namespace alphashape;
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

TEST_F(CrsPicker, VerticalDatumRoundTrips) {
    crs::VerticalCrs vertical;
    vertical.kind = crs::VerticalCrs::Kind::GravityRelated;
    vertical.name = "DHHN2016 height";
    const crs::CoordinateSystem with_vertical =
        crs::CoordinateSystem::from_epsg(25832).with_vertical(vertical);
    alphashape::gui::CrsPicker picker;
    picker.set_coordinate_system(with_vertical);
    EXPECT_TRUE(picker.is_valid());

    const crs::CoordinateSystem back = picker.coordinate_system();
    ASSERT_TRUE(back.vertical().has_value());
    EXPECT_EQ(back.vertical()->name, "DHHN2016 height");
    EXPECT_EQ(back.vertical()->kind, crs::VerticalCrs::Kind::GravityRelated);
    // The horizontal component survives too (a compound CRS may carry no EPSG
    // code, so compare its name rather than the code).
    EXPECT_NE(back.name().find("UTM zone 32N"), std::string::npos);
}

TEST_F(CrsPicker, AcceptsCodeOutsideTheCuratedList) {
    alphashape::gui::CrsPicker picker;
    picker.epsg_edit()->setText(QStringLiteral("32644"));
    ASSERT_TRUE(picker.is_valid());
    EXPECT_EQ(picker.coordinate_system().epsg(), 32644);

    picker.epsg_edit()->setText(QStringLiteral("999999"));
    EXPECT_FALSE(picker.is_valid());
}

TEST_F(CrsPicker, CurrentCodeOutsideTheListIsShown) {
    alphashape::gui::CrsPicker picker;
    picker.set_coordinate_system(crs::CoordinateSystem::from_epsg(32644));
    EXPECT_EQ(picker.epsg_edit()->text(), QStringLiteral("32644"));
    EXPECT_TRUE(picker.is_valid());
    EXPECT_EQ(picker.coordinate_system().epsg(), 32644);
}

}  // namespace
