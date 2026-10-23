#pragma once

// TaskDialog: a parameter dialog generated from an engine task's param schema.
//
// Nothing here is task-specific. The dialog reads the TaskEntry's declared parameters
// (name, type, default, doc) and builds one editor per parameter:
//
//   Bool        -> QCheckBox
//   Int         -> QSpinBox
//   Float       -> QDoubleSpinBox
//   String      -> QComboBox when the doc text lists the accepted values in quotes and the
//                  default is one of them; QLineEdit otherwise. A parameter named "path" gets a
//                  browse button.
//   StringList  -> QLineEdit holding the JSON array text the engine itself uses
//                  (["a", "b"]), parsed and checked on OK. "filenames" gets a browse button.
//
// The schema carries no ranges, enum tables, display names or groups, so those are heuristics:
// integers accept [min(default, 0), INT_MAX], floats [-1e9, 1e9]; the first kGeneralCount
// parameters form the "General" group and any further ones a collapsed "Advanced" group; names
// are made readable by replacing '_' with ' '. The doc string is the tooltip.
//
// Values come from the schema default, overlaid with what the user last accepted for this task
// (stored in QSettings under "taskdialog/<Task>/<param>"). An editor the user did not touch
// returns the exact schema value, so accepting an untouched dialog yields the defaults
// bit-for-bit. A parameter without a default (or whose doc says "Required") must be filled in.

#include <map>
#include <string>
#include <vector>

#include <QDialog>
#include <QString>

#include "common/tasks/registry.hpp"

class QDialogButtonBox;
class QFormLayout;
class QGroupBox;
class QLabel;
class QSettings;

namespace alphashape::gui {

class TaskDialog : public QDialog {
    Q_OBJECT

public:
    enum class EditorKind { CheckBox, SpinBox, DoubleSpinBox, ComboBox, LineEdit, PathEdit, ListEdit };

    /// Number of parameters shown in "General"; the rest go to "Advanced".
    static constexpr std::size_t kGeneralCount = 8;

    /// Throws alphashape::UnsupportedTaskError for an unknown task.
    TaskDialog(const std::string& task_name, const TaskRegistry& registry, QSettings& settings,
               QWidget* parent = nullptr);

    const std::string& task_name() const { return task_name_; }

    /// The values as the user set them, in the form JobManager::submit takes. Every declared
    /// parameter is set explicitly. Call after validate() returned true.
    Params params() const;

    /// Check every editor; shows the problems inline and returns true when there are none.
    bool validate();
    /// Inline error text of a parameter ("" when it is fine).
    QString error(const std::string& param) const;
    /// Parameters that currently show an error.
    std::vector<std::string> invalid_params() const;

    /// Put every editor back to the schema default and clear the errors.
    void reset_to_defaults();
    /// Load a value into a parameter's editor, as if the schema default had been that value.
    void set_value(const std::string& param, const ParamValue& value);

    /// The editor widget of a parameter, or nullptr. For path and list editors this is the
    /// line edit.
    QWidget* editor(const std::string& param) const;
    EditorKind kind(const std::string& param) const;
    /// True when the parameter has no default and must be provided.
    bool required(const std::string& param) const;
    /// True when the parameter sits in the collapsed "Advanced" group.
    bool advanced(const std::string& param) const;

    /// "keypoint_limit" -> "Keypoint limit".
    static QString display_name(const std::string& param);
    /// "BuildDepthMaps" -> "Build Depth Maps".
    static QString task_title(const std::string& task_name);
    /// The accepted values a string parameter's doc lists in quotes (needs at least two);
    /// empty when there is no such list.
    static QStringList choices_from_doc(const std::string& doc);
    /// The QSettings key under which a parameter's last accepted value is kept.
    static QString settings_key(const std::string& task_name, const std::string& param);

    void accept() override;

private:
    struct Field {
        ParamSpec spec;
        EditorKind kind = EditorKind::LineEdit;
        QWidget* editor = nullptr;
        QLabel* error_label = nullptr;
        bool dirty = false;      ///< the user changed the editor since the last load
        bool required = false;
        bool advanced = false;
        ParamValue loaded;       ///< what the editor was last loaded with
        QString error;
    };

    Field* find(const std::string& param);
    const Field* find(const std::string& param) const;
    void build();
    void add_field(const ParamSpec& spec, bool in_advanced, QFormLayout* form);
    void load(Field& field, const ParamValue& value);
    ParamValue read(const Field& field) const;
    ParamValue schema_default(const Field& field) const;
    void load_remembered();
    void remember() const;
    void set_error(Field& field, const QString& text);

    std::string task_name_;
    QSettings& settings_;
    std::vector<Field> fields_;
    QGroupBox* advanced_group_ = nullptr;
    QDialogButtonBox* buttons_ = nullptr;
};

}  // namespace alphashape::gui
