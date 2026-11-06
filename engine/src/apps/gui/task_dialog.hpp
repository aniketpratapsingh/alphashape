#pragma once

// TaskDialog: a parameter dialog generated from an engine task's param schema.
//
// Nothing here is task-specific. The dialog reads the TaskEntry's declared parameters
// (name, type, default, doc, and the optional value constraints) and builds one editor per
// parameter:
//
//   Bool        -> QCheckBox
//   Int         -> QSpinBox, bounded by the declared minimum/maximum (a power-of-two
//                  parameter snaps typed values to the nearest one and steps between them)
//   Float       -> QDoubleSpinBox, bounded by the declared minimum/maximum
//   String      -> QComboBox for a declared choice list (or a doc that quotes the values and
//                  whose default is one of them); QLineEdit otherwise. A parameter named
//                  "path" gets a browse button.
//   StringList  -> QLineEdit holding the JSON array text the engine itself uses
//                  (["a", "b"]), parsed and checked on OK. "filenames" gets a browse button.
//
// What the schema does not carry stays a heuristic: display names replace '_' with ' ', and
// the first kGeneralCount parameters form the "General" group while any further ones go to a
// collapsed "Advanced" group. The doc string is the tooltip.
//
// Values come from the schema default, overlaid with what the user last accepted for this task
// (stored in QSettings under "taskdialog/<Task>/<param>"). An editor the user did not touch
// returns the exact schema value, so accepting an untouched dialog yields the defaults
// bit-for-bit. A parameter without a default (or whose doc says "Required") must be filled in,
// and validate() also rejects a value that violates a declared constraint — the same range or
// choice set the editor offers, re-checked because a remembered setting bypasses the editor.

#include <cstdint>
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

namespace alphashape {
class Chunk;
}

namespace alphashape::gui {

class ThroughputHistory;

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
    /// Filter the Browse button offers for a path parameter ("PLY (*.ply);;All files (*)");
    /// empty keeps Qt's unfiltered dialog. Ignored for parameters without a browse button.
    void set_path_filter(const std::string& param, const QString& filter);
    /// The filter set for a parameter ("" when none).
    QString path_filter(const std::string& param) const;

    /// The editor widget of a parameter, or nullptr. For path and list editors this is the
    /// line edit.
    QWidget* editor(const std::string& param) const;
    EditorKind kind(const std::string& param) const;
    /// True when the parameter has no default and must be provided.
    bool required(const std::string& param) const;
    /// True when the parameter sits in the collapsed "Advanced" group.
    bool advanced(const std::string& param) const;

    /// Supply the chunk and measured history the estimate row reads. Both are
    /// borrowed and must outlive the dialog; null hides the row. Call after the
    /// constructor; the row also refreshes whenever a parameter changes.
    void set_estimate_context(const Chunk* chunk, const ThroughputHistory* history);
    /// Override the RAM budget the warning compares against (0 = the platform
    /// budget). Tests use this to pin the 75 % threshold.
    void set_memory_budget(std::uint64_t bytes);
    /// The estimate label and the RAM warning label (both always exist; hidden
    /// when there is no context or no warning).
    QLabel* estimate_label() const { return estimate_label_; }
    QLabel* ram_warning_label() const { return ram_warning_label_; }

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
        QString path_filter;     ///< Browse filter ("All files (*)" when empty)
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
    QString constraint_problem(const Field& field) const;
    void refresh_estimate();

    std::string task_name_;
    QSettings& settings_;
    std::vector<Field> fields_;
    QGroupBox* advanced_group_ = nullptr;
    QDialogButtonBox* buttons_ = nullptr;
    QLabel* estimate_label_ = nullptr;
    QLabel* ram_warning_label_ = nullptr;
    const Chunk* estimate_chunk_ = nullptr;
    const ThroughputHistory* estimate_history_ = nullptr;
    const TaskRegistry* registry_ = nullptr;
    std::uint64_t memory_budget_ = 0;
};

}  // namespace alphashape::gui
