// CalibrationDialog.*: the camera calibration dialog (WP-14.22a) driven
// offscreen. The sensor is built directly in the engine with a known
// calibration and covariance so the Adjusted tab is analytic.

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QTableWidget>

#include "apps/gui/app.hpp"
#include "apps/gui/calibration_dialog.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "model/camera/calibration.hpp"
#include "harness.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;

namespace {

class CalibrationDialogFixture : public GuiTest {
protected:
    /// One sensor with a known calibration, a full-coefficient covariance
    /// (variance 4, f/cx correlation 0.95) and no fixed parameters.
    Chunk& build_sensor() {
        Document& document = app().document();
        Chunk& chunk = document.workspace().add_chunk("Calibration");
        Sensor sensor;
        sensor.label = "Primary";
        sensor.width = 6000;
        sensor.height = 4000;
        sensor.calibration =
            Calibration::initial(CameraType::Frame, 6000, 4000, 5000.0);
        sensor.calibration.cx = 12.5;
        sensor.calibration.cy = -3.25;
        sensor.calibration.k1 = 0.0123;
        sensor.calibration.covariance_params = {
            "f", "cx", "cy", "b1", "b2", "k1", "k2",
            "k3", "k4", "p1", "p2", "p3", "p4"};
        sensor.calibration.covariance =
            Eigen::MatrixXd::Identity(13, 13) * 4.0;
        sensor.calibration.covariance(0, 1) = 3.8;
        sensor.calibration.covariance(1, 0) = 3.8;
        chunk.sensors().push_back(sensor);
        document.set_active_chunk_id(chunk.id());
        return chunk;
    }
};

using CalibrationDialog = CalibrationDialogFixture;

TEST_F(CalibrationDialog, ShowsAdjustedValues) {
    Chunk& chunk = build_sensor();
    alphashape::gui::CalibrationDialog dialog(app());
    dialog.set_chunk(chunk.id());
    ASSERT_EQ(dialog.sensor_list()->count(), 1);
    ASSERT_NE(dialog.adjusted_value("f"), nullptr);
    EXPECT_EQ(dialog.adjusted_value("f")->text(), QStringLiteral("5000.000000"));
    EXPECT_EQ(dialog.adjusted_value("cx")->text(), QStringLiteral("12.500000"));
    EXPECT_EQ(dialog.adjusted_value("k1")->text(), QStringLiteral("0.012300"));
    auto* sigma = dialog.findChild<QLabel*>(QStringLiteral("calibration.sigma.f"));
    ASSERT_NE(sigma, nullptr);
    EXPECT_EQ(sigma->text(), QStringLiteral("2.000000"));
    ASSERT_NE(dialog.correlation_table(), nullptr);
    ASSERT_EQ(dialog.correlation_table()->rowCount(), 13);
    ASSERT_NE(dialog.correlation_table()->item(0, 1), nullptr);
    EXPECT_EQ(dialog.correlation_table()->item(0, 1)->text(),
              QStringLiteral("0.950"));
}

TEST_F(CalibrationDialog, FitFlagEditEmitsCommand) {
    Chunk& chunk = build_sensor();
    alphashape::gui::CalibrationDialog dialog(app());
    dialog.set_chunk(chunk.id());
    QCheckBox* k3 = dialog.fit_check(QStringLiteral("k3"));
    ASSERT_NE(k3, nullptr);
    // "Fit" checked means the coefficient is optimised, i.e. it is NOT in
    // fixed_params. Every coefficient is fitted by default.
    for (int i = 0; i < kCalibrationParamCount; ++i) {
        QCheckBox* check = dialog.fit_check(
            QString::fromLatin1(kCalibrationParamNames[static_cast<std::size_t>(i)]));
        ASSERT_NE(check, nullptr);
        EXPECT_TRUE(check->isChecked());
    }
    ASSERT_TRUE(chunk.sensors()[0].fixed_params.empty());
    const std::size_t before = app().undo_stack().size();

    k3->setChecked(false);

    EXPECT_EQ(app().undo_stack().size(), before + 1);
    ASSERT_EQ(chunk.sensors()[0].fixed_params.size(), 1U);
    EXPECT_EQ(chunk.sensors()[0].fixed_params[0], "k3");

    app().undo();

    EXPECT_TRUE(chunk.sensors()[0].fixed_params.empty());
    EXPECT_EQ(app().undo_stack().position(), before);
}

TEST_F(CalibrationDialog, SlavePointsAtItsMaster) {
    Chunk& chunk = build_sensor();
    Sensor slave;
    slave.label = "Secondary";
    slave.width = 6000;
    slave.height = 4000;
    slave.calibration = chunk.sensors()[0].calibration;
    slave.master_sensor_id = 0;
    chunk.sensors().push_back(slave);
    alphashape::gui::CalibrationDialog dialog(app());
    dialog.set_chunk(chunk.id());
    ASSERT_EQ(dialog.sensor_list()->count(), 2);

    dialog.select_sensor(1);
    ASSERT_EQ(dialog.sensor_index(), 1);
    // A slave's calibration belongs to its master: the editors are read-only
    // and the note points at the master.
    EXPECT_FALSE(dialog.initial_editor(QStringLiteral("f"))->isEnabled());
    EXPECT_FALSE(dialog.fit_check(QStringLiteral("k3"))->isEnabled());
    ASSERT_NE(dialog.slave_note(), nullptr);
    EXPECT_FALSE(dialog.slave_note()->isHidden());
    EXPECT_NE(dialog.slave_note()->text().indexOf(QStringLiteral("Primary")), -1);
    ASSERT_NE(dialog.edit_master_button(), nullptr);
    EXPECT_FALSE(dialog.edit_master_button()->isHidden());

    dialog.edit_master_button()->click();
    EXPECT_EQ(dialog.sensor_index(), 0);
    EXPECT_TRUE(dialog.initial_editor(QStringLiteral("f"))->isEnabled());
    EXPECT_TRUE(dialog.slave_note()->isHidden());
}

TEST_F(CalibrationDialog, EditingCoefficientClearsCovariance) {
    Chunk& chunk = build_sensor();
    alphashape::gui::CalibrationDialog dialog(app());
    dialog.set_chunk(chunk.id());
    auto* sigma = dialog.findChild<QLabel*>(QStringLiteral("calibration.sigma.f"));
    ASSERT_NE(sigma, nullptr);
    ASSERT_EQ(sigma->text(), QStringLiteral("2.000000"));

    dialog.initial_editor(QStringLiteral("k1"))->setValue(0.02);
    dialog.commit_calibration();

    EXPECT_DOUBLE_EQ(chunk.sensors()[0].calibration.k1, 0.02);
    EXPECT_TRUE(chunk.sensors()[0].calibration.covariance_params.empty());
    EXPECT_EQ(sigma->text(), QStringLiteral("n/a"));
}

}  // namespace
