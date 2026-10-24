#pragma once

// ExportController: the File > Export submenu. One action per result the chunk can hold -
// model, point cloud, DEM, orthomosaic - each running the engine's export task through the
// schema-generated TaskDialog and the JobManager.
//
// Enabled state follows the same rule as the Workflow menu: an action is offered only when a
// chunk is active, no job runs, and the chunk actually holds that asset. Every update is ONE
// locked read of the document, and none at all while a job runs (the worker owns the document
// then).
//
// The dialog starts on a sensible destination (the project folder, or Documents for a project
// that has no file yet) and offers the task's own formats as file filters, so accepting an
// export writes a predictably named file. The dialog and the submission are replaceable hooks,
// so the tests can drive the whole flow.

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <QObject>
#include <QString>
#include <QStringList>

#include "apps/gui/workflow.hpp"  // Presets
#include "common/tasks/params.hpp"

class QAction;
class QMenu;
class QWidget;

namespace alphashape::gui {

class GuiApp;

/// One entry of the Export submenu.
struct ExportStep {
    std::string task;         ///< registry name of the export task
    QString text;             ///< menu text
    QString object_name;      ///< action objectName
    QString name_part;        ///< file-name stem, e.g. "_model"
    QString suffix;           ///< default extension the suggested file uses
    QStringList formats;      ///< formats the task can write, "" for a fixed-format task
    QString asset;            ///< what the chunk must hold ("model", "point cloud", ...)
};

class ExportController : public QObject {
    Q_OBJECT

public:
    /// Fills `menu` with the actions. `dialog_parent` owns the dialogs the actions open.
    ExportController(GuiApp& app, QMenu* menu, QWidget* dialog_parent);

    /// The menu entries, in catalog order. Built per call so the texts are translated when
    /// the window is built.
    static std::vector<ExportStep> steps();

    /// The step for `task`, or nullopt when the menu does not offer it.
    static std::optional<ExportStep> find_step(const std::string& task);

    /// The Browse filter the destination dialog offers for `task`: one pattern per format the
    /// task writes, then "All files" ("" for a task this menu does not offer).
    static QString path_filter_for(const std::string& task);

    /// Asked for a task's parameters; nullopt = the user cancelled. `presets` holds the
    /// destination path (and the format its extension names), so the default handler opens the
    /// generated dialog on a file name that needs no typing.
    using ParamsHandler =
        std::function<std::optional<Params>(const std::string& task, const Presets& presets)>;
    void set_params_handler(ParamsHandler handler);

    /// Receives the finished request and returns the queued job id (nullopt when a hook
    /// swallowed it). Default: JobManager::submit for `chunk_id`. May throw alphashape::Error.
    using SubmitHandler = std::function<std::optional<quint64>(const std::string& task,
                                                               Params params,
                                                               std::int32_t chunk_id)>;
    void set_submit_handler(SubmitHandler handler);

    /// Told about a request that could not be submitted. Default: a critical message box.
    using ErrorHandler = std::function<void(const QString& title, const QString& message)>;
    void set_error_handler(ErrorHandler handler);

    QAction* action(const std::string& task) const;

    /// Why `task` cannot run now ("" = it can). A task this menu does not offer is never
    /// runnable, so its reason is the "not available" one.
    QString unavailable_reason(const std::string& task) const;

    /// Ask for parameters and submit `task`, as triggering its action does. False when the task
    /// is unavailable, the user cancelled, or the submission failed.
    bool run(const std::string& task);

    /// Recompute every action's enabled state and tooltip.
    void update_enabled();

    /// The directory a project without a path suggests: the user's Documents folder.
    static QString default_directory();

    /// "<dir>/<title><name_part>.<suffix>"; the directory is `document_path`'s own folder, or
    /// default_directory() when the project has no path yet.
    static QString suggest_path(const QString& document_path, const QString& title,
                                const ExportStep& step);

Q_SIGNALS:
    /// The file an accepted export dialog asked for, once its job ended Succeeded.
    void export_written(const QString& task, const QString& path);

private:
    void on_job_finished(quint64 id);

    GuiApp& app_;
    QWidget* dialog_parent_;
    std::vector<std::pair<std::string, QAction*>> actions_;
    ParamsHandler params_handler_;
    SubmitHandler submit_handler_;
    ErrorHandler error_handler_;
    /// Jobs this controller queued: id -> (task, destination the dialog accepted).
    std::vector<std::pair<quint64, std::pair<std::string, QString>>> pending_;
};

}  // namespace alphashape::gui
