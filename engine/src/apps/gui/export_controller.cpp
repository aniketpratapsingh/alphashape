#include "apps/gui/export_controller.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <QAction>
#include <QDialog>
#include <QDir>
#include <QFileInfo>
#include <QMenu>
#include <QMessageBox>
#include <QStandardPaths>

#include "apps/gui/app.hpp"
#include "apps/gui/task_dialog.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"

namespace alphashape::gui {

namespace {

/// The active chunk of a document, read-only (Document::active_chunk() would be fine, but the
/// Workflow menu walks the list the same way and the two surveys read identically).
const Chunk* active_chunk_of(const Document& document) {
    const std::int32_t id = document.active_chunk_id();
    if (id < 0) {
        return nullptr;
    }
    for (const std::shared_ptr<Chunk>& chunk : document.workspace().chunks()) {
        if (chunk != nullptr && chunk->id() == id) {
            return chunk.get();
        }
    }
    return nullptr;
}

/// What the chunk holds. Only the CONST accessors are used: the non-const ones create an empty
/// asset, which a menu update must never do. "Holds" means "carries something", matching the
/// export task's own "No model" / "No point cloud" checks.
struct ExportSurvey {
    bool has_model = false;
    bool has_point_cloud = false;
    bool has_elevation = false;
    bool has_orthomosaic = false;
};

ExportSurvey survey(const Chunk& chunk) {
    ExportSurvey found;
    found.has_model = !chunk.model().empty();
    found.has_point_cloud = !chunk.point_cloud().points.empty();
    found.has_elevation = !chunk.elevation().empty();
    found.has_orthomosaic = !chunk.orthomosaic().empty();
    return found;
}

bool has_asset(const ExportSurvey& chunk, const ExportStep& step) {
    if (step.task == "ExportModel") {
        return chunk.has_model;
    }
    if (step.task == "ExportPointCloud") {
        return chunk.has_point_cloud;
    }
    if (step.task == "ExportDem") {
        return chunk.has_elevation;
    }
    if (step.task == "ExportOrthomosaic") {
        return chunk.has_orthomosaic;
    }
    return false;
}

}  // namespace

std::vector<ExportStep> ExportController::steps() {
    // Built per call so the texts are translated in the language active when the window is
    // created instead of at first use. `formats` mirrors what each task's own parameter check
    // accepts, so the destination filter and the dialog's format list cannot drift apart.
    return {
        {"ExportModel", QObject::tr("Export &Model..."), QStringLiteral("action.export.model"),
         QStringLiteral("_model"), QStringLiteral("ply"),
         {QStringLiteral("ply"), QStringLiteral("obj")}, QObject::tr("model")},
        {"ExportPointCloud", QObject::tr("Export &Point Cloud..."),
         QStringLiteral("action.export.point_cloud"), QStringLiteral("_point_cloud"),
         QStringLiteral("ply"), {QStringLiteral("ply")}, QObject::tr("point cloud")},
        {"ExportDem", QObject::tr("Export D&EM..."), QStringLiteral("action.export.dem"),
         QStringLiteral("_dem"), QStringLiteral("tif"), {}, QObject::tr("elevation model")},
        {"ExportOrthomosaic", QObject::tr("Export &Orthomosaic..."),
         QStringLiteral("action.export.orthomosaic"), QStringLiteral("_orthomosaic"),
         QStringLiteral("tif"), {}, QObject::tr("orthomosaic")},
    };
}

std::optional<ExportStep> ExportController::find_step(const std::string& task) {
    for (const ExportStep& step : steps()) {
        if (step.task == task) {
            return step;
        }
    }
    return std::nullopt;
}

QString ExportController::path_filter_for(const std::string& task) {
    const std::optional<ExportStep> step = find_step(task);
    if (!step) {
        return QString();
    }
    QStringList patterns;
    for (const QString& format : step->formats) {
        patterns << QStringLiteral("*.%1").arg(format);
    }
    if (patterns.isEmpty()) {
        // A task with no format parameter writes one thing: name its extension.
        patterns << QStringLiteral("*.%1").arg(step->suffix);
    }
    return QObject::tr("%1 (%2);;All files (*)").arg(step->asset, patterns.join(QLatin1Char(' ')));
}

ExportController::ExportController(GuiApp& app, QMenu* menu, QWidget* dialog_parent)
    : QObject(dialog_parent), app_(app), dialog_parent_(dialog_parent) {
    params_handler_ = [this](const std::string& task,
                             const Presets& presets) -> std::optional<Params> {
        TaskDialog dialog(task, builtin_registry(), app_.settings(), dialog_parent_);
        for (const auto& [param, value] : presets) {
            dialog.set_value(param, value);
        }
        // A filter naming the formats the task declares, so choosing a destination does not
        // require typing an extension the dialog already knows.
        dialog.set_path_filter("path", path_filter_for(task));
        if (dialog.exec() != QDialog::Accepted) {
            return std::nullopt;
        }
        return dialog.params();
    };
    submit_handler_ = [this](const std::string& task, Params params,
                             std::int32_t chunk_id) -> std::optional<quint64> {
        return app_.jobs().submit(task, std::move(params), chunk_id);
    };
    error_handler_ = [this](const QString& title, const QString& message) {
        QMessageBox::critical(dialog_parent_, title, message);
    };

    for (const ExportStep& step : steps()) {
        if (builtin_registry().find(step.task) == nullptr) {
            continue;  // a task this build does not ship
        }
        auto* act = new QAction(step.text, this);
        act->setObjectName(step.object_name);
        menu->addAction(act);
        connect(act, &QAction::triggered, this, [this, task = step.task]() { run(task); });
        actions_.emplace_back(step.task, act);
    }

    JobManager& jobs = app_.jobs();
    connect(&jobs, &JobManager::busy_changed, this, [this](bool) { update_enabled(); });
    connect(&jobs, &JobManager::job_finished, this,
            [this](quint64 id, JobManager::State state, const QString&) {
                update_enabled();
                if (state == JobManager::State::Succeeded) {
                    on_job_finished(id);
                } else {
                    pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                                  [id](const auto& entry) {
                                                      return entry.first == id;
                                                  }),
                                   pending_.end());
                }
            });
    connect(&app_, &GuiApp::document_replaced, this, [this]() { update_enabled(); });
    connect(&app_, &GuiApp::undo_changed, this, [this]() { update_enabled(); });
    connect(menu, &QMenu::aboutToShow, this, [this]() { update_enabled(); });
    update_enabled();
}

void ExportController::set_params_handler(ParamsHandler handler) {
    params_handler_ = std::move(handler);
}
void ExportController::set_submit_handler(SubmitHandler handler) {
    submit_handler_ = std::move(handler);
}
void ExportController::set_error_handler(ErrorHandler handler) {
    error_handler_ = std::move(handler);
}

QAction* ExportController::action(const std::string& task) const {
    for (const auto& [name, act] : actions_) {
        if (name == task) {
            return act;
        }
    }
    return nullptr;
}

QString ExportController::unavailable_reason(const std::string& task) const {
    const std::optional<ExportStep> step = find_step(task);
    if (action(task) == nullptr || !step) {
        return tr("This export is not available.");
    }
    if (app_.jobs().busy()) {
        return tr("A job is running.");
    }
    const TaskEntry* entry = builtin_registry().find(task);
    if (entry == nullptr || !entry->enabled) {
        return tr("This export is not available.");
    }
    // Read under the shared document lock; never nested inside another read.
    const ExportStep wanted = *step;
    return app_.read_document([&wanted](const Document& document) -> QString {
        const Chunk* chunk = active_chunk_of(document);
        if (chunk == nullptr) {
            return tr("There is no active chunk.");
        }
        return has_asset(survey(*chunk), wanted)
                   ? QString()
                   : tr("This project has no %1 to export.").arg(wanted.asset);
    });
}

void ExportController::update_enabled() {
    // While a job runs the answer does not depend on the document: every action is disabled
    // with the same reason. Reading it anyway would block the GUI thread, because the worker
    // holds the document lock for the whole task and busy_changed(true) is emitted
    // synchronously from submit() on this thread.
    if (app_.jobs().busy()) {
        for (const auto& [task, act] : actions_) {
            (void)task;
            act->setEnabled(false);
            act->setToolTip(tr("A job is running."));
            act->setStatusTip(tr("A job is running."));
        }
        return;
    }

    // One locked read and one walk over the chunk for the whole menu.
    const std::pair<bool, ExportSurvey> state =
        app_.read_document([](const Document& document) -> std::pair<bool, ExportSurvey> {
            const Chunk* chunk = active_chunk_of(document);
            return {chunk != nullptr, chunk != nullptr ? survey(*chunk) : ExportSurvey{}};
        });
    const bool has_chunk = state.first;
    const ExportSurvey& chunk = state.second;

    for (const auto& [task, act] : actions_) {
        const std::optional<ExportStep> step = find_step(task);
        const TaskEntry* entry = builtin_registry().find(task);
        QString reason;
        if (!step || entry == nullptr || !entry->enabled) {
            reason = tr("This export is not available.");
        } else if (!has_chunk) {
            reason = tr("There is no active chunk.");
        } else if (!has_asset(chunk, *step)) {
            reason = tr("This project has no %1 to export.").arg(step->asset);
        }
        act->setEnabled(reason.isEmpty());
        // QAction::toolTip() falls back to the action text when no tip is set, so an empty
        // reason must be stored as an empty string to keep that behaviour.
        act->setToolTip(reason);
        act->setStatusTip(reason);
    }
}

bool ExportController::run(const std::string& task) {
    if (!unavailable_reason(task).isEmpty()) {
        return false;
    }
    const std::optional<ExportStep> step = find_step(task);
    if (!step) {
        return false;
    }

    // Read the target chunk and the suggested destination before any dialog runs: a nested
    // event loop (the dialog) could in principle swap the document, and the export must target
    // the chunk and the project name the user was looking at.
    struct Target {
        std::int32_t chunk_id = -1;
        QString suggested;
    };
    const std::optional<Target> target =
        app_.read_document([&step](const Document& document) -> std::optional<Target> {
            const Chunk* chunk = active_chunk_of(document);
            if (chunk == nullptr) {
                return std::nullopt;
            }
            Target found;
            found.chunk_id = chunk->id();
            found.suggested = suggest_path(QString::fromStdString(document.path()),
                                          document_title_of(document), *step);
            return found;
        });
    if (!target) {
        return false;
    }

    // Everything that can throw (the dialog, the schema lookup, the submission) reports through
    // the error handler: a triggered menu action never unwinds into Qt.
    try {
        Presets presets;
        presets.emplace_back("path", ParamValue(target->suggested.toStdString()));
        if (const TaskEntry* entry = builtin_registry().find(task)) {
            for (const ParamSpec& spec : entry->schema.specs()) {
                if (spec.name == "format" && spec.type == ParamType::String) {
                    presets.emplace_back("format", ParamValue(step->suffix.toStdString()));
                    break;
                }
            }
        }
        std::optional<Params> params;
        if (params_handler_) {
            params = params_handler_(task, presets);
        }
        if (!params) {
            return false;
        }
        // A destination typed without an extension gets the task's own: "out" is never what the
        // user meant for a PLY export, and the reveal action names the real file.
        QString path = target->suggested;
        if (params->declares("path")) {
            path = QString::fromStdString(params->get_string("path"));
            if (!path.isEmpty() && QFileInfo(path).suffix().isEmpty()) {
                path += QLatin1Char('.');
                path += step->suffix;
                params->set("path", ParamValue(path.toStdString()));
            }
        }
        if (!submit_handler_) {
            return false;
        }
        const std::optional<quint64> id =
            submit_handler_(task, std::move(*params), target->chunk_id);
        if (!id) {
            return false;
        }
        pending_.emplace_back(*id, std::make_pair(task, path));
        return true;
    } catch (const std::exception& error) {
        error_handler_(tr("Cannot start export"), QString::fromUtf8(error.what()));
        return false;
    }
}

void ExportController::on_job_finished(quint64 id) {
    const auto found = std::find_if(pending_.begin(), pending_.end(),
                                    [id](const auto& entry) { return entry.first == id; });
    if (found == pending_.end()) {
        return;
    }
    const std::pair<std::string, QString> done = found->second;
    pending_.erase(found);
    Q_EMIT export_written(QString::fromStdString(done.first), done.second);
}

QString ExportController::default_directory() {
    const QString documents =
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    return documents.isEmpty() ? QDir::homePath() : documents;
}

QString ExportController::suggest_path(const QString& document_path, const QString& title,
                                       const ExportStep& step) {
    QString directory = default_directory();
    if (!document_path.isEmpty()) {
        const QFileInfo info(document_path);
        if (info.absoluteDir().exists()) {
            directory = info.absolutePath();
        }
    }
    const QString stem = title.isEmpty() ? QStringLiteral("project") : title;
    return QDir(directory).filePath(stem + step.name_part + QLatin1Char('.') + step.suffix);
}

}  // namespace alphashape::gui
