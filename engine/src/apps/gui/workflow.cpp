#include "apps/gui/workflow.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <utility>

#include <QAction>
#include <QDialog>
#include <QFileDialog>
#include <QMenu>
#include <QMessageBox>

#include "apps/gui/app.hpp"
#include "apps/gui/task_dialog.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"

namespace alphashape::gui {

namespace {

/// The active chunk of a document, read-only. Document::active_chunk() is non-const, so a
/// read under the shared document lock (GuiApp::read_document hands out a const Document)
/// has to find the chunk itself.
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

std::size_t aligned_cameras(const Chunk& chunk) {
    return static_cast<std::size_t>(
        std::count_if(chunk.cameras().begin(), chunk.cameras().end(),
                      [](const Camera& camera) { return camera.enabled; }));
}

bool has_sparse_points(const Chunk& chunk) {
    return std::any_of(chunk.tie_points().tracks.begin(), chunk.tie_points().tracks.end(),
                       [](const Track& track) { return track.point.has_value(); });
}

/// What `task` needs from the chunk, as the engine checks it when it runs; "" when satisfied.
/// Each check mirrors the engine's own early check (see the task's apply()), only so the menu
/// can say what is missing before the user picks parameters. The engine stays the authority: a
/// request that slips through is reported by the job ending Failed.
///
/// The data checks read the CONST accessors, which report "nothing" for an empty chunk instead
/// of creating an asset the way the non-const ones do. They ask the same question the engine
/// asks ("is there any depth map / a mesh with faces / an elevation raster"), so an asset that
/// exists but holds nothing does not count as a satisfied prerequisite.
QString missing_input(const std::string& task, const Chunk& chunk) {
    const QString none;
    if (task == "AddPhotos") {
        return none;  // any active chunk takes photos
    }
    if (task == "MatchPhotos") {
        return chunk.cameras().empty() ? QObject::tr("Add photos first.") : none;
    }
    if (task == "AlignCameras") {
        const TiePoints& tie = chunk.tie_points();
        return tie.matches.empty() || tie.tracks.empty() ? QObject::tr("Match photos first.")
                                                         : none;
    }
    if (task == "OptimizeCameras") {
        // The engine needs at least two aligned cameras with sparse points.
        return aligned_cameras(chunk) < 2 || !has_sparse_points(chunk)
                   ? QObject::tr("Align cameras first.")
                   : none;
    }
    if (task == "BuildDepthMaps") {
        return aligned_cameras(chunk) < 2 || !has_sparse_points(chunk)
                   ? QObject::tr("Align cameras first.")
                   : none;
    }
    if (task == "BuildPointCloud") {
        return chunk.depth_maps().empty() ? QObject::tr("Build depth maps first.") : none;
    }
    if (task == "BuildModel") {
        // Either source is fine; the dialog picks which one the run uses.
        return chunk.depth_maps().empty() && chunk.point_cloud().points.empty()
                   ? QObject::tr("Build depth maps or a point cloud first.")
                   : none;
    }
    if (task == "BuildUV") {
        return chunk.model().empty() ? QObject::tr("Build a model first.") : none;
    }
    if (task == "BuildTexture") {
        if (chunk.model().empty()) {
            return QObject::tr("Build a model first.");
        }
        return chunk.model().uv_faces.empty() ? QObject::tr("Build UV coordinates first.") : none;
    }
    if (task == "BuildDem") {
        return chunk.depth_maps().empty() && chunk.point_cloud().points.empty() &&
                       chunk.model().empty()
                   ? QObject::tr("Build depth maps, a point cloud or a model first.")
                   : none;
    }
    if (task == "BuildOrthomosaic") {
        // Either surface works: "elevation" (the default) needs the DEM, "model" a mesh.
        // The DEM is the one the automatic parameters assume, so it is what is asked for.
        return chunk.elevation().empty() && chunk.model().empty()
                   ? QObject::tr("Build an elevation model or a model first.")
                   : none;
    }
    return none;
}

}  // namespace

std::vector<WorkflowStep> WorkflowController::steps() {
    // Built per call rather than as a function-local static, so the texts are translated in the
    // language active when the window is created instead of at first use.
    return {
        {"AddPhotos", QObject::tr("Add &Photos..."), QStringLiteral("action.workflow.add_photos"),
         false},
        {"MatchPhotos", QObject::tr("&Match Photos..."),
         QStringLiteral("action.workflow.match_photos"), false},
        {"AlignCameras", QObject::tr("&Align Cameras..."),
         QStringLiteral("action.workflow.align_cameras"), false},
        {"OptimizeCameras", QObject::tr("&Optimize Cameras..."),
         QStringLiteral("action.workflow.optimize_cameras"), false},
        {"BuildDepthMaps", QObject::tr("Build &Depth Maps..."),
         QStringLiteral("action.workflow.build_depth_maps"), true},
        {"BuildPointCloud", QObject::tr("Build Point &Cloud..."),
         QStringLiteral("action.workflow.build_point_cloud"), false},
        {"BuildModel", QObject::tr("Build &Model..."),
         QStringLiteral("action.workflow.build_model"), false},
        {"BuildUV", QObject::tr("Build &UV..."), QStringLiteral("action.workflow.build_uv"),
         false},
        {"BuildTexture", QObject::tr("Build &Texture..."),
         QStringLiteral("action.workflow.build_texture"), false},
        {"BuildDem", QObject::tr("Build D&EM..."), QStringLiteral("action.workflow.build_dem"),
         true},
        {"BuildOrthomosaic", QObject::tr("Build &Orthomosaic..."),
         QStringLiteral("action.workflow.build_orthomosaic"), false},
    };
}

WorkflowController::WorkflowController(GuiApp& app, QMenu* menu, QWidget* dialog_parent)
    : QObject(dialog_parent), app_(app), dialog_parent_(dialog_parent) {
    params_handler_ = [this](const std::string& task) -> std::optional<Params> {
        TaskDialog dialog(task, builtin_registry(), app_.settings(), dialog_parent_);
        if (dialog.exec() != QDialog::Accepted) {
            return std::nullopt;
        }
        return dialog.params();
    };
    photos_handler_ = [this]() {
        return QFileDialog::getOpenFileNames(
            dialog_parent_, tr("Add Photos"), QString(),
            tr("Images (*.jpg *.jpeg *.png *.tif *.tiff);;All files (*)"));
    };
    submit_handler_ = [this](const std::string& task, Params params, std::int32_t chunk_id) {
        app_.jobs().submit(task, std::move(params), chunk_id);
    };
    error_handler_ = [this](const QString& title, const QString& message) {
        QMessageBox::critical(dialog_parent_, title, message);
    };

    for (const WorkflowStep& step : steps()) {
        if (builtin_registry().find(step.task) == nullptr) {
            continue;  // a task this build does not ship
        }
        if (step.separator_before) {
            menu->addSeparator();
        }
        auto* act = new QAction(step.text, this);
        act->setObjectName(step.object_name);
        menu->addAction(act);
        connect(act, &QAction::triggered, this, [this, task = step.task]() { run(task); });
        actions_.emplace_back(step.task, act);
    }

    // What can change an action's availability: a job starting or ending (it owns the document
    // and nothing else may touch it meanwhile), the document being replaced, or the chunk
    // contents changing through an undo/redo of a chunk command.
    JobManager& jobs = app_.jobs();
    connect(&jobs, &JobManager::busy_changed, this, [this](bool) { update_enabled(); });
    connect(&jobs, &JobManager::job_finished, this, [this]() { update_enabled(); });
    connect(&app_, &GuiApp::document_replaced, this, [this]() { update_enabled(); });
    connect(&app_, &GuiApp::undo_changed, this, [this]() { update_enabled(); });
    // The active chunk can also change without a signal of ours, so look again whenever the
    // menu opens (and once now, for the initial state).
    connect(menu, &QMenu::aboutToShow, this, [this]() { update_enabled(); });
    update_enabled();
}

void WorkflowController::set_params_handler(ParamsHandler handler) {
    params_handler_ = std::move(handler);
}
void WorkflowController::set_photos_handler(PhotosHandler handler) {
    photos_handler_ = std::move(handler);
}
void WorkflowController::set_submit_handler(SubmitHandler handler) {
    submit_handler_ = std::move(handler);
}
void WorkflowController::set_error_handler(ErrorHandler handler) {
    error_handler_ = std::move(handler);
}

QAction* WorkflowController::action(const std::string& task) const {
    for (const auto& [name, act] : actions_) {
        if (name == task) {
            return act;
        }
    }
    return nullptr;
}

QString WorkflowController::unavailable_reason(const std::string& task) const {
    // Only the steps this menu offers can be run from here, whatever the registry holds. This
    // is also what stops run() from opening a dialog for an arbitrary task name: it gates on
    // this function.
    if (action(task) == nullptr) {
        return tr("This task is not available.");
    }
    if (app_.jobs().busy()) {
        return tr("A job is running.");
    }
    const TaskEntry* entry = builtin_registry().find(task);
    if (entry == nullptr || !entry->enabled) {
        return tr("This task is not available.");
    }
    // Read under the shared document lock: the menu can be updated from any signal, and a
    // long-running job holds the write lock. A read here never nests inside another one.
    // (tr() is a static of this class, so the lambda needs no `this` capture.)
    return app_.read_document([&task](const Document& document) -> QString {
        const Chunk* chunk = active_chunk_of(document);
        if (chunk == nullptr) {
            return tr("There is no active chunk.");
        }
        return missing_input(task, *chunk);
    });
}

void WorkflowController::update_enabled() {
    for (const auto& [task, act] : actions_) {
        const QString reason = unavailable_reason(task);
        act->setEnabled(reason.isEmpty());
        act->setToolTip(reason);
        act->setStatusTip(reason);
    }
}

bool WorkflowController::run(const std::string& task) {
    if (!unavailable_reason(task).isEmpty()) {
        return false;
    }
    // Read the id before the dialogs run: a nested event loop (the dialog) could in principle
    // swap the document, and the task must target the chunk the user was looking at. A missing
    // chunk is impossible here (unavailable_reason just said there is one), but the id is read
    // the same guarded way as everything else rather than trusting that.
    const std::optional<std::int32_t> chunk_id =
        app_.read_document([](const Document& document) -> std::optional<std::int32_t> {
            const Chunk* chunk = active_chunk_of(document);
            if (chunk == nullptr) {
                return std::nullopt;
            }
            return chunk->id();
        });
    if (!chunk_id) {
        return false;
    }

    // Everything that can throw (the dialog construction, the schema lookup, the submission)
    // reports through the error handler: a triggered menu action never unwinds into Qt.
    try {
        std::optional<Params> params;
        if (task == "AddPhotos") {
            // Filenames are a StringList, which the generated dialog can only show as JSON
            // text; a file picker is the useful editor for the one parameter that matters
            // here. The schema's other parameters keep their defaults.
            const QStringList files = photos_handler_ ? photos_handler_() : QStringList();
            if (files.isEmpty()) {
                return false;
            }
            std::vector<std::string> names;
            names.reserve(static_cast<std::size_t>(files.size()));
            for (const QString& file : files) {
                names.push_back(file.toStdString());
            }
            Params values = make_params(builtin_registry().require(task));
            values.set("filenames", ParamValue(std::move(names)));
            params = std::move(values);
        } else if (params_handler_) {
            params = params_handler_(task);
        }
        if (!params) {
            return false;
        }
        if (submit_handler_) {
            submit_handler_(task, std::move(*params), *chunk_id);
        }
    } catch (const std::exception& error) {
        error_handler_(tr("Cannot start task"), QString::fromUtf8(error.what()));
        return false;
    }
    return true;
}

}  // namespace alphashape::gui
