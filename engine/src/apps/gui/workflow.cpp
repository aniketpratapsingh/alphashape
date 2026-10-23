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

/// One pass over the chunk, answering every question the menu asks about it. Computing this
/// once per update (rather than once per action) keeps a menu refresh to a single short read
/// even when the chunk holds many cameras or tracks: the counts are accumulated in one walk,
/// and the walk stops as soon as a later question cannot change.
struct ChunkSurvey {
    bool has_cameras = false;
    std::size_t aligned_cameras = 0;
    bool has_matches = false;
    bool has_tracks = false;
    bool has_sparse_points = false;
    bool has_depth_maps = false;
    bool has_point_cloud = false;
    bool has_model = false;
    bool has_model_uvs = false;
    bool has_elevation = false;
};

ChunkSurvey survey(const Chunk& chunk) {
    ChunkSurvey found;
    found.has_cameras = !chunk.cameras().empty();

    // One pass over the cameras. Alignment is the only per-camera question, and every later
    // question is answered by a container's emptiness, so the walk ends once the count can no
    // longer matter: everything that needs aligned cameras needs at least two.
    for (const Camera& camera : chunk.cameras()) {
        if (found.aligned_cameras >= 2) {
            break;
        }
        if (camera.enabled) {
            ++found.aligned_cameras;
        }
    }

    // Sparse points: one track with a point is enough, so stop at the first. The two empty()
    // checks come first because they are O(1) and settle the common "nothing tracked yet" case.
    const TiePoints& tie = chunk.tie_points();
    found.has_matches = !tie.matches.empty();
    found.has_tracks = !tie.tracks.empty();
    for (const Track& track : tie.tracks) {
        if (track.point.has_value()) {
            found.has_sparse_points = true;
            break;
        }
    }

    // The CONST accessors: the non-const ones create an empty asset, which a menu update must
    // never do. "Exists" means "holds something" for the products whose payload is the point.
    found.has_depth_maps = !chunk.depth_maps().empty();
    found.has_point_cloud = !chunk.point_cloud().points.empty();
    const Mesh& mesh = chunk.model();
    found.has_model = !mesh.empty();
    found.has_model_uvs = !mesh.uv_faces.empty();
    found.has_elevation = !chunk.elevation().empty();
    return found;
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
QString missing_input(const std::string& task, const ChunkSurvey& chunk) {
    const QString none;
    if (task == "AddPhotos") {
        return none;  // any active chunk takes photos
    }
    if (task == "MatchPhotos") {
        return chunk.has_cameras ? none : QObject::tr("Add photos first.");
    }
    if (task == "AlignCameras") {
        return chunk.has_matches && chunk.has_tracks ? none : QObject::tr("Match photos first.");
    }
    if (task == "OptimizeCameras" || task == "BuildDepthMaps") {
        // The engine needs at least two aligned cameras with sparse points.
        return chunk.aligned_cameras >= 2 && chunk.has_sparse_points
                   ? none
                   : QObject::tr("Align cameras first.");
    }
    if (task == "BuildPointCloud") {
        return chunk.has_depth_maps ? none : QObject::tr("Build depth maps first.");
    }
    if (task == "BuildModel") {
        // Either source is fine; the dialog is preset to one the chunk actually has.
        return chunk.has_depth_maps || chunk.has_point_cloud
                   ? none
                   : QObject::tr("Build depth maps or a point cloud first.");
    }
    if (task == "BuildUV") {
        return chunk.has_model ? none : QObject::tr("Build a model first.");
    }
    if (task == "BuildTexture") {
        if (!chunk.has_model) {
            return QObject::tr("Build a model first.");
        }
        return chunk.has_model_uvs ? none : QObject::tr("Build UV coordinates first.");
    }
    if (task == "BuildDem") {
        return chunk.has_depth_maps || chunk.has_point_cloud || chunk.has_model
                   ? none
                   : QObject::tr("Build depth maps, a point cloud or a model first.");
    }
    if (task == "BuildOrthomosaic") {
        // Either surface works. "elevation" (the schema default) needs the DEM; "model"
        // rasterises a DSM from the mesh and runs with no DEM at all (the task's own grid
        // handling covers that case). The dialog is preset to whichever one exists, so the
        // default cannot name a surface the chunk lacks.
        return chunk.has_elevation || chunk.has_model
                   ? none
                   : QObject::tr("Build an elevation model or a model first.");
    }
    return none;
}

/// The schema parameter through which a task picks which existing data it consumes, and the
/// values this chunk can satisfy, best first (the schema default heads the list when the chunk
/// has it). A generated dialog otherwise opens on the schema default (source_data="depth_maps",
/// surface_data="elevation") and the run fails on a chunk that only holds the other source, so
/// run() presets the parameter from this before showing the dialog.
struct SourceChoice {
    const char* param = nullptr;        ///< the schema parameter name
    std::vector<std::string> present;   ///< the values that exist, schema default first
};

std::optional<SourceChoice> source_choice(const std::string& task, const ChunkSurvey& chunk) {
    SourceChoice choice;
    if (task == "BuildModel") {
        choice.param = "source_data";
        if (chunk.has_depth_maps) {
            choice.present.emplace_back("depth_maps");
        }
        if (chunk.has_point_cloud) {
            choice.present.emplace_back("point_cloud");
        }
        return choice;
    }
    if (task == "BuildDem") {
        choice.param = "source_data";
        if (chunk.has_depth_maps) {
            choice.present.emplace_back("depth_maps");
        }
        if (chunk.has_point_cloud) {
            choice.present.emplace_back("point_cloud");
        }
        if (chunk.has_model) {
            choice.present.emplace_back("model");
        }
        return choice;
    }
    if (task == "BuildOrthomosaic") {
        choice.param = "surface_data";
        if (chunk.has_elevation) {
            choice.present.emplace_back("elevation");
        }
        if (chunk.has_model) {
            choice.present.emplace_back("model");
        }
        return choice;
    }
    return std::nullopt;
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
    params_handler_ = [this](const std::string& task,
                             const Presets& presets) -> std::optional<Params> {
        TaskDialog dialog(task, builtin_registry(), app_.settings(), dialog_parent_);
        // The presets override whatever the dialog would otherwise start on (the schema
        // default, or the value the user last accepted): they name data this chunk has, so
        // they are the right starting point for this run.
        for (const auto& [param, value] : presets) {
            dialog.set_value(param, value);
        }
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
        return missing_input(task, survey(*chunk));
    });
}

void WorkflowController::update_enabled() {
    // One locked read and one walk over the chunk for the whole menu, rather than one read per
    // action: undo_changed() fires on every edit and the menu is re-evaluated on every open, so
    // eleven locked scans of the camera and track lists would add up. Only the survey (plain
    // flags and counts) leaves the lock.
    const bool busy = app_.jobs().busy();
    const std::pair<bool, ChunkSurvey> state =
        app_.read_document([](const Document& document) -> std::pair<bool, ChunkSurvey> {
            const Chunk* chunk = active_chunk_of(document);
            return {chunk != nullptr, chunk != nullptr ? survey(*chunk) : ChunkSurvey{}};
        });
    const bool has_chunk = state.first;
    const ChunkSurvey& chunk = state.second;

    for (const auto& [task, act] : actions_) {
        QString reason;
        const TaskEntry* entry = builtin_registry().find(task);
        if (busy) {
            reason = tr("A job is running.");
        } else if (entry == nullptr || !entry->enabled) {
            reason = tr("This task is not available.");
        } else if (!has_chunk) {
            reason = tr("There is no active chunk.");
        } else {
            reason = missing_input(task, chunk);
        }
        act->setEnabled(reason.isEmpty());
        // QAction::toolTip() falls back to the action text when no tip is set, so an empty
        // reason must be stored as an empty string (the default) to keep that behaviour.
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
    // the same guarded way as everything else rather than trusting that. The same read picks
    // the source presets, so the dialog opens on data the chunk really holds instead of on the
    // schema default, which the run would then reject.
    struct Target {
        std::int32_t chunk_id = -1;
        Presets presets;
    };
    const std::optional<Target> target =
        app_.read_document([&task](const Document& document) -> std::optional<Target> {
            const Chunk* chunk = active_chunk_of(document);
            if (chunk == nullptr) {
                return std::nullopt;
            }
            Target found;
            found.chunk_id = chunk->id();
            if (const std::optional<SourceChoice> choice = source_choice(task, survey(*chunk))) {
                // The schema default heads `present` when the chunk has it, so taking the
                // first entry is "prefer the schema default when it is available".
                found.presets.emplace_back(choice->param,
                                           ParamValue(choice->present.front()));
            }
            return found;
        });
    if (!target) {
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
            params = params_handler_(task, target->presets);
        }
        if (!params) {
            return false;
        }
        if (submit_handler_) {
            submit_handler_(task, std::move(*params), target->chunk_id);
        }
    } catch (const std::exception& error) {
        error_handler_(tr("Cannot start task"), QString::fromUtf8(error.what()));
        return false;
    }
    return true;
}

}  // namespace alphashape::gui
