#pragma once

// WorkflowController: the Workflow menu. One action per main pipeline task, in pipeline order.
//
// Triggering an action asks for the task's parameters (a TaskDialog built from the engine
// schema, or a file picker for Add Photos) and then submits the task to the JobManager for the
// active chunk. An action is enabled only when a chunk is active, no job is running, and the
// chunk has what the task needs (aligned cameras for depth maps, a model for texturing, ...).
// The disabled action's tooltip says what is missing.
//
// The prerequisite checks here only mirror the engine's own early checks so the menu can say up
// front what is missing; the engine stays the authority (a failed run is reported either way).
//
// Every user interaction goes through a replaceable hook so the tests can drive the whole flow.

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <QObject>
#include <QString>
#include <QStringList>

#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"

class QAction;
class QMenu;
class QWidget;

namespace alphashape::gui {

class GuiApp;

/// One entry of the menu.
struct WorkflowStep {
    std::string task;      ///< registry name
    QString text;          ///< menu text
    QString object_name;   ///< action objectName
    bool separator_before = false;
};

/// Values the controller presets before showing a task's dialog. The parameter that selects
/// which existing data a task consumes (BuildModel's source_data, BuildOrthomosaic's
/// surface_data, BuildDem's source_data) otherwise keeps its schema default, which may name
/// data the chunk does not have, so the run would fail. Empty for tasks with no such choice.
using Presets = std::vector<std::pair<std::string, ParamValue>>;

class WorkflowController : public QObject {
    Q_OBJECT

public:
    /// Fills `menu` with the actions. `dialog_parent` owns the dialogs the actions open.
    WorkflowController(GuiApp& app, QMenu* menu, QWidget* dialog_parent);

    /// The menu entries, in pipeline order. Built per call so the texts are translated when the
    /// window is built rather than once at first use.
    static std::vector<WorkflowStep> steps();

    /// Asked for a task's parameters; nullopt = the user cancelled. `presets` holds the values
    /// the controller wants the dialog to start on (see Presets above), so the default handler
    /// applies them to the generated TaskDialog before showing it. Default: a TaskDialog.
    using ParamsHandler =
        std::function<std::optional<Params>(const std::string& task, const Presets& presets)>;
    void set_params_handler(ParamsHandler handler);
    /// Asked for the photo files to add (empty list = cancelled). Default: a file dialog.
    using PhotosHandler = std::function<QStringList()>;
    void set_photos_handler(PhotosHandler handler);
    /// Receives the finished request. Default: JobManager::submit. May throw alphashape::Error.
    using SubmitHandler =
        std::function<void(const std::string& task, Params params, std::int32_t chunk_id)>;
    void set_submit_handler(SubmitHandler handler);
    /// Told about a request that could not be submitted. Default: a critical message box.
    using ErrorHandler = std::function<void(const QString& title, const QString& message)>;
    void set_error_handler(ErrorHandler handler);

    /// The action of a task, or nullptr.
    QAction* action(const std::string& task) const;

    /// Why `task` cannot run on the active chunk right now ("" = it can). A task this menu does
    /// not offer is never runnable, so its reason is the "not available" one.
    QString unavailable_reason(const std::string& task) const;

    /// Ask for parameters and submit `task`, as triggering its action does. False when the task
    /// is unavailable, the user cancelled or the submission failed.
    bool run(const std::string& task);

    /// Recompute every action's enabled state and tooltip.
    void update_enabled();

private:
    GuiApp& app_;
    QWidget* dialog_parent_;
    std::vector<std::pair<std::string, QAction*>> actions_;
    ParamsHandler params_handler_;
    PhotosHandler photos_handler_;
    SubmitHandler submit_handler_;
    ErrorHandler error_handler_;
};

}  // namespace alphashape::gui
