// TaskDialog.*: the parameter dialog generated from the engine's param schema. The drift guard
// builds a dialog for every registered task, so a new task or param type that the dialog cannot
// handle fails here.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QTemporaryDir>

#include "apps/gui/task_dialog.hpp"
#include "common/tasks/registry.hpp"
#include "harness.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;
using EditorKind = TaskDialog::EditorKind;

namespace {

class TaskDialogTest : public GuiTest {
protected:
    QSettings& store() {
        if (!store_) {
            store_ = std::make_unique<QSettings>(scratch("dialog.ini"), QSettings::IniFormat);
        }
        return *store_;
    }
    const TaskRegistry& registry() { return builtin_registry(); }

private:
    std::unique_ptr<QSettings> store_;
};

std::string text_of(const ParamValue& value) { return format_param_value(value); }

}  // namespace

TEST_F(TaskDialogTest, EveryRegisteredTaskBuildsAndReturnsItsDefaults) {
    const std::vector<std::string> names = registry().names();
    ASSERT_FALSE(names.empty());
    for (const std::string& name : names) {
        SCOPED_TRACE(name);
        const TaskEntry& entry = registry().require(name);
        QSettings fresh(scratch(QString::fromStdString(name) + ".ini"), QSettings::IniFormat);
        TaskDialog dialog(name, registry(), fresh);

        // Every declared parameter has an editor.
        for (const ParamSpec& spec : entry.schema.specs()) {
            EXPECT_NE(dialog.editor(spec.name), nullptr) << spec.name;
        }

        // Required values are the only thing that may block OK on untouched defaults.
        if (!dialog.validate()) {
            for (const std::string& bad : dialog.invalid_params()) {
                EXPECT_TRUE(dialog.required(bad)) << bad;
                const ParamSpec* spec = nullptr;
                for (const ParamSpec& candidate : entry.schema.specs()) {
                    if (candidate.name == bad) spec = &candidate;
                }
                ASSERT_NE(spec, nullptr);
                if (spec->type == ParamType::StringList) {
                    dialog.set_value(bad, std::vector<std::string>{"a"});
                } else {
                    dialog.set_value(bad, std::string("a"));
                }
            }
        }
        dialog.accept();
        EXPECT_EQ(dialog.result(), QDialog::Accepted);

        const Params result = dialog.params();
        for (const ParamSpec& spec : entry.schema.specs()) {
            if (dialog.required(spec.name) && std::holds_alternative<std::monostate>(spec.default_value)) {
                continue;
            }
            if (dialog.required(spec.name)) {
                continue;  // "Required" docs on an empty-string default: filled in above
            }
            EXPECT_EQ(text_of(result.get(spec.name)), text_of(spec.default_value)) << spec.name;
            EXPECT_EQ(result.get(spec.name).index(), spec.default_value.index()) << spec.name;
        }
    }
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(TaskDialogTest, WidgetTypeFollowsTheParamType) {
    TaskDialog depth("BuildDepthMaps", registry(), store());
    EXPECT_EQ(depth.kind("reuse_depth"), EditorKind::CheckBox);
    EXPECT_NE(qobject_cast<QCheckBox*>(depth.editor("reuse_depth")), nullptr);
    EXPECT_EQ(depth.kind("downscale"), EditorKind::SpinBox);
    EXPECT_NE(qobject_cast<QSpinBox*>(depth.editor("downscale")), nullptr);
    // No list of accepted values in quotes in its doc: a plain line edit.
    EXPECT_EQ(depth.kind("filter_mode"), EditorKind::LineEdit);
    EXPECT_NE(qobject_cast<QLineEdit*>(depth.editor("filter_mode")), nullptr);

    TaskDialog dem("BuildDem", registry(), store());
    EXPECT_EQ(dem.kind("resolution"), EditorKind::DoubleSpinBox);
    EXPECT_NE(qobject_cast<QDoubleSpinBox*>(dem.editor("resolution")), nullptr);
    EXPECT_EQ(dem.kind("source_data"), EditorKind::ComboBox);
    auto* source = qobject_cast<QComboBox*>(dem.editor("source_data"));
    ASSERT_NE(source, nullptr);
    EXPECT_EQ(source->currentText(), "depth_maps");
    QStringList items;
    for (int i = 0; i < source->count(); ++i) items << source->itemText(i);
    EXPECT_TRUE(items.contains("point_cloud"));
    EXPECT_TRUE(items.contains("model"));

    TaskDialog save("ExportModel", registry(), store());
    EXPECT_EQ(save.kind("path"), EditorKind::PathEdit);
    EXPECT_NE(save.editor("path")->parentWidget()->findChild<QPushButton*>("browse.path"), nullptr);
    EXPECT_EQ(save.kind("format"), EditorKind::ComboBox);

    TaskDialog photos("AddPhotos", registry(), store());
    EXPECT_EQ(photos.kind("filenames"), EditorKind::ListEdit);
    EXPECT_TRUE(photos.required("filenames"));
    EXPECT_FALSE(photos.required("load_reference"));
}

TEST_F(TaskDialogTest, TooltipsComeFromTheSchemaDocs) {
    TaskDialog dialog("BuildDepthMaps", registry(), store());
    const TaskEntry& entry = registry().require("BuildDepthMaps");
    for (const ParamSpec& spec : entry.schema.specs()) {
        EXPECT_EQ(dialog.editor(spec.name)->toolTip().toStdString(), spec.doc) << spec.name;
    }
}

TEST_F(TaskDialogTest, ParametersBeyondTheFirstEightAreAdvanced) {
    const TaskEntry& entry = registry().require("OptimizeCameras");
    TaskDialog dialog("OptimizeCameras", registry(), store());
    std::size_t index = 0;
    for (const ParamSpec& spec : entry.schema.specs()) {
        EXPECT_EQ(dialog.advanced(spec.name), index >= TaskDialog::kGeneralCount) << spec.name;
        ++index;
    }
    EXPECT_GT(index, TaskDialog::kGeneralCount);
}

TEST_F(TaskDialogTest, LastUsedValuesAreRememberedAndResetRestoresDefaults) {
    {
        TaskDialog dialog("BuildDepthMaps", registry(), store());
        auto* downscale = qobject_cast<QSpinBox*>(dialog.editor("downscale"));
        ASSERT_NE(downscale, nullptr);
        EXPECT_EQ(downscale->value(), 4);
        downscale->setValue(8);
        qobject_cast<QCheckBox*>(dialog.editor("reuse_depth"))->setChecked(true);
        qobject_cast<QLineEdit*>(dialog.editor("filter_mode"))->setText("aggressive");
        dialog.accept();
        ASSERT_EQ(dialog.result(), QDialog::Accepted);
        EXPECT_EQ(dialog.params().get_int("downscale"), 8);
    }
    {
        TaskDialog again("BuildDepthMaps", registry(), store());
        EXPECT_EQ(qobject_cast<QSpinBox*>(again.editor("downscale"))->value(), 8);
        EXPECT_TRUE(qobject_cast<QCheckBox*>(again.editor("reuse_depth"))->isChecked());
        const Params remembered = again.params();
        EXPECT_EQ(remembered.get_int("downscale"), 8);
        EXPECT_TRUE(remembered.get_bool("reuse_depth"));
        EXPECT_EQ(remembered.get_string("filter_mode"), "aggressive");

        again.reset_to_defaults();
        EXPECT_EQ(qobject_cast<QSpinBox*>(again.editor("downscale"))->value(), 4);
        EXPECT_FALSE(qobject_cast<QCheckBox*>(again.editor("reuse_depth"))->isChecked());
        EXPECT_EQ(again.params().get_int("downscale"), 4);
        EXPECT_EQ(again.params().get_string("filter_mode"), "mild");
        again.accept();
    }
    TaskDialog third("BuildDepthMaps", registry(), store());
    EXPECT_EQ(third.params().get_int("downscale"), 4);
}

TEST_F(TaskDialogTest, ResetButtonRestoresDefaults) {
    TaskDialog dialog("BuildDepthMaps", registry(), store());
    qobject_cast<QSpinBox*>(dialog.editor("downscale"))->setValue(2);
    auto* reset = dialog.findChild<QPushButton*>("task.reset");
    ASSERT_NE(reset, nullptr);
    reset->click();
    EXPECT_EQ(dialog.params().get_int("downscale"), 4);
}

TEST_F(TaskDialogTest, UnusableRememberedEntriesFallBackToTheDefault) {
    store().setValue(TaskDialog::settings_key("BuildDepthMaps", "downscale"), "not a number");
    store().setValue(TaskDialog::settings_key("BuildDem", "source_data"), "bogus");
    TaskDialog depth("BuildDepthMaps", registry(), store());
    EXPECT_EQ(depth.params().get_int("downscale"), 4);
    TaskDialog dem("BuildDem", registry(), store());
    EXPECT_EQ(dem.params().get_string("source_data"), "depth_maps");
}

TEST_F(TaskDialogTest, RequiredAndMalformedValuesBlockOkWithInlineErrors) {
    TaskDialog dialog("AddPhotos", registry(), store());
    dialog.accept();
    EXPECT_NE(dialog.result(), QDialog::Accepted);
    EXPECT_EQ(dialog.invalid_params(), (std::vector<std::string>{"filenames"}));
    EXPECT_FALSE(dialog.error("filenames").isEmpty());
    auto* label = dialog.findChild<QLabel*>("error.filenames");
    ASSERT_NE(label, nullptr);
    EXPECT_FALSE(label->isHidden());
    EXPECT_EQ(label->text(), dialog.error("filenames"));

    auto* edit = qobject_cast<QLineEdit*>(dialog.editor("filenames"));
    edit->setText("this is not a list");
    dialog.accept();
    EXPECT_NE(dialog.result(), QDialog::Accepted);
    EXPECT_FALSE(dialog.error("filenames").isEmpty());

    edit->setText("[\"a.jpg\", \"b.jpg\"]");
    dialog.accept();
    EXPECT_EQ(dialog.result(), QDialog::Accepted);
    EXPECT_TRUE(dialog.error("filenames").isEmpty());
    EXPECT_TRUE(label->isHidden());
    EXPECT_EQ(dialog.params().get_string_list("filenames"),
              (std::vector<std::string>{"a.jpg", "b.jpg"}));
}

TEST_F(TaskDialogTest, ARequiredPathBlocksOk) {
    TaskDialog dialog("ExportModel", registry(), store());
    dialog.accept();
    EXPECT_NE(dialog.result(), QDialog::Accepted);
    EXPECT_EQ(dialog.invalid_params(), (std::vector<std::string>{"path"}));
    dialog.set_value("path", std::string("out.ply"));
    dialog.accept();
    EXPECT_EQ(dialog.result(), QDialog::Accepted);
    EXPECT_EQ(dialog.params().get_string("path"), "out.ply");
}

TEST_F(TaskDialogTest, NamesAreReadable) {
    EXPECT_EQ(TaskDialog::display_name("keypoint_limit"), "Keypoint limit");
    EXPECT_EQ(TaskDialog::task_title("BuildDepthMaps"), "Build Depth Maps");
    EXPECT_EQ(TaskDialog::task_title("BuildUV"), "Build UV");
    EXPECT_EQ(TaskDialog::choices_from_doc("Only \"a\" here"), QStringList());
    EXPECT_EQ(TaskDialog::choices_from_doc("'ply' or 'obj'"), (QStringList{"ply", "obj"}));
    EXPECT_THROW(TaskDialog("NoSuchTask", registry(), store()), UnsupportedTaskError);
}
