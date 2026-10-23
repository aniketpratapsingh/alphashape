// Workflow.*: the Workflow menu (task actions in pipeline order, enabled state, and the flow
// from a triggered action through the parameter hooks to JobManager::submit).

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <QAction>
#include <QMenu>
#include <QStringList>

#include "apps/gui/workflow.hpp"
#include "base/utils/error.hpp"
#include "common/core/chunk.hpp"
#include "common/tasks/registry.hpp"
#include "harness.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;
using alphashape::gui::testing::pump_until;

namespace {

/// Blocks until released, so a test can observe the "job running" state.
class HoldTask : public AbstractTask {
public:
    HoldTask(std::shared_ptr<std::atomic<bool>> release, std::shared_ptr<std::atomic<bool>> started)
        : release_(std::move(release)), started_(std::move(started)) {}
    std::string name() const override { return "HoldTask"; }
    void declare_params(Params&) const override {}
    TargetType target_type() const override { return TargetType::Chunk; }
    void apply(Document&, Params&, const Progress& progress) override {
        started_->store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!release_->load() && std::chrono::steady_clock::now() < deadline) {
            if (progress.is_cancelled()) {
                throw CancelledError("cancelled");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    std::shared_ptr<std::atomic<bool>> release_;
    std::shared_ptr<std::atomic<bool>> started_;
};

struct Submitted {
    std::string task;
    Params params;
    std::int32_t chunk_id = -2;
};

class WorkflowTest : public GuiTest {
protected:
    void SetUp() override {
        GuiTest::SetUp();
        Chunk& added = app().document().workspace().add_chunk("Chunk 1");
        chunk_id_ = added.id();
        app().document().set_active_chunk_id(chunk_id_);
        // A failing job would otherwise open the window's modal error box, which never closes
        // offscreen. Collect the messages instead.
        window().set_error_handler([this](const QString& title, const QString& message) {
            errors_.emplace_back(title, message);
        });
        window().workflow().set_submit_handler(
            [this](const std::string& task, Params params, std::int32_t target) {
                submitted_.push_back(Submitted{task, std::move(params), target});
            });
        window().workflow().update_enabled();
        ready_ = true;
    }

    /// Leave the job manager idle and pointed at the built-in registry, whatever the test did.
    /// Without this a failed assertion before the test's own cleanup would leave the manager
    /// holding a fixture-owned registry (and, for the blocking task, a worker waiting out its
    /// deadline) while the fixture is destroyed.
    void TearDown() override {
        if (!ready_) {
            GuiTest::TearDown();  // SetUp never finished: there is nothing of ours to unwind
            return;
        }
        hold_release_->store(true);
        if (!pump_until([this]() { return !app().jobs().busy(); }, 60000)) {
            app().jobs().cancel_all_and_wait();
        }
        app().jobs().set_registry(&builtin_registry());
        GuiTest::TearDown();
    }

    Chunk& chunk() { return *app().document().active_chunk(); }
    WorkflowController& workflow() { return window().workflow(); }
    bool enabled(const char* name) { return action(name)->isEnabled(); }
    void refresh() { workflow().update_enabled(); }

    std::int32_t chunk_id_ = -1;
    std::vector<Submitted> submitted_;
    std::vector<std::pair<QString, QString>> errors_;
    /// Released by the "job running" test and by TearDown, so a blocked worker always ends.
    std::shared_ptr<std::atomic<bool>> hold_release_ = std::make_shared<std::atomic<bool>>(false);
    /// A test registry for the "job running" test. A member, not a local: the job manager keeps
    /// the pointer until TearDown restores the built-in one.
    TaskRegistry hold_registry_;
    /// Set once SetUp has built everything TearDown unwinds, so a SetUp failure (no GuiApp) is
    /// not mistaken for a job left running.
    bool ready_ = false;
};

Camera make_camera(Chunk& chunk, bool aligned) {
    Camera camera;
    camera.id = chunk.allocate_id();
    camera.enabled = aligned;
    return camera;
}

/// Add a depth-map set that actually holds a map, as BuildDepthMaps would leave behind.
void give_depth_maps(Chunk& chunk) {
    DepthMap map;
    map.camera_id = chunk.cameras().empty() ? -1 : chunk.cameras().front().id;
    map.width = 2;
    map.height = 2;
    map.downscale = 1;
    map.depths.assign(4, 1.0F);
    chunk.add_depth_maps().data->push_back(std::move(map));
}

/// Add a model with geometry, as BuildModel would leave behind.
void give_model(Chunk& chunk) {
    Mesh& mesh = *chunk.add_model().data;
    mesh.vertices = {Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
                     Eigen::Vector3d(0.0, 1.0, 0.0)};
    mesh.faces = {Eigen::Vector3i(0, 1, 2)};
}

/// Give the active model UVs, as BuildUV would leave behind.
void give_uvs(Chunk& chunk) {
    Mesh& mesh = chunk.model();
    mesh.uvs = {Eigen::Vector2f(0.0F, 0.0F), Eigen::Vector2f(1.0F, 0.0F),
                Eigen::Vector2f(0.0F, 1.0F)};
    mesh.uv_faces = {Eigen::Vector3i(0, 1, 2)};
}

/// Add an elevation with a raster, as BuildDem would leave behind.
void give_elevation(Chunk& chunk) {
    Elevation& dem = *chunk.add_elevation().data;
    dem.raster.grid.width = 2;
    dem.raster.grid.height = 2;
    dem.raster.grid.pixel_size_x = 1.0;
    dem.raster.grid.pixel_size_y = 1.0;
    dem.raster.data.assign(4, 0.0F);
}

}  // namespace

TEST_F(WorkflowTest, ActionsExistInPipelineOrder) {
    QMenu* workflow_menu = menu("menu.workflow");
    ASSERT_NE(workflow_menu, nullptr);
    QStringList names;
    for (QAction* entry : workflow_menu->actions()) {
        if (!entry->isSeparator()) {
            names << entry->objectName();
        }
    }
    const QStringList expected = {
        "action.workflow.add_photos",       "action.workflow.match_photos",
        "action.workflow.align_cameras",    "action.workflow.optimize_cameras",
        "action.workflow.build_depth_maps", "action.workflow.build_point_cloud",
        "action.workflow.build_model",      "action.workflow.build_uv",
        "action.workflow.build_texture",    "action.workflow.build_dem",
        "action.workflow.build_orthomosaic"};
    EXPECT_EQ(names, expected);
    // Every step names a task the registry really has.
    for (const WorkflowStep& step : WorkflowController::steps()) {
        EXPECT_NE(builtin_registry().find(step.task), nullptr) << step.task;
        EXPECT_NE(workflow().action(step.task), nullptr) << step.task;
    }
    // The two group separators split the input/aligned block from the dense block and that
    // from the surface block.
    int separators = 0;
    for (QAction* entry : workflow_menu->actions()) {
        if (entry->isSeparator()) {
            ++separators;
        }
    }
    EXPECT_EQ(separators, 2);
}

TEST_F(WorkflowTest, NothingIsEnabledWithoutAnActiveChunk) {
    app().document().set_active_chunk_id(-1);
    refresh();
    for (const WorkflowStep& step : WorkflowController::steps()) {
        EXPECT_FALSE(workflow().action(step.task)->isEnabled()) << step.task;
        EXPECT_FALSE(workflow().action(step.task)->toolTip().isEmpty()) << step.task;
    }
    EXPECT_FALSE(workflow().run("AddPhotos"));
    EXPECT_TRUE(submitted_.empty());
    EXPECT_TRUE(errors_.empty());
}

TEST_F(WorkflowTest, PreconditionsFollowTheChunkState) {
    // An empty chunk can only take photos.
    for (const WorkflowStep& step : WorkflowController::steps()) {
        EXPECT_EQ(workflow().action(step.task)->isEnabled(), step.task == "AddPhotos") << step.task;
    }

    chunk().cameras().push_back(make_camera(chunk(), false));
    refresh();
    EXPECT_TRUE(enabled("action.workflow.match_photos"));
    EXPECT_FALSE(enabled("action.workflow.align_cameras"));
    EXPECT_FALSE(enabled("action.workflow.build_depth_maps"));

    // Two aligned cameras without sparse points are still not enough for depth maps.
    chunk().cameras().push_back(make_camera(chunk(), true));
    chunk().cameras().push_back(make_camera(chunk(), true));
    refresh();
    EXPECT_FALSE(enabled("action.workflow.build_depth_maps"));
    EXPECT_FALSE(enabled("action.workflow.optimize_cameras"));
    Track track;
    track.point = Eigen::Vector3d(0.0, 0.0, 1.0);
    chunk().tie_points().tracks.push_back(track);
    refresh();
    EXPECT_TRUE(enabled("action.workflow.build_depth_maps"));
    EXPECT_TRUE(enabled("action.workflow.optimize_cameras"));

    // No derived products yet: the dense and surface stages stay blocked.
    EXPECT_FALSE(enabled("action.workflow.build_point_cloud"));
    EXPECT_FALSE(enabled("action.workflow.build_model"));
    EXPECT_FALSE(enabled("action.workflow.build_dem"));

    // An empty asset does not count: the stage must really have produced data.
    chunk().add_depth_maps();
    refresh();
    EXPECT_FALSE(enabled("action.workflow.build_point_cloud"))
        << "an empty depth-map set is not a depth map";
    give_depth_maps(chunk());
    refresh();
    EXPECT_TRUE(enabled("action.workflow.build_point_cloud"));
    EXPECT_TRUE(enabled("action.workflow.build_model"));
    EXPECT_TRUE(enabled("action.workflow.build_dem"));

    EXPECT_FALSE(enabled("action.workflow.build_uv"));
    EXPECT_FALSE(enabled("action.workflow.build_texture"));
    give_model(chunk());
    refresh();
    EXPECT_TRUE(enabled("action.workflow.build_uv"));
    EXPECT_FALSE(enabled("action.workflow.build_texture"))
        << "texturing needs the UV coordinates BuildUV produces";
    give_uvs(chunk());
    refresh();
    EXPECT_TRUE(enabled("action.workflow.build_texture"));

    // The ortho projects on either the DEM ("elevation", the default) or the mesh ("model").
    // The model is already there, so it is available; the DEM alone would be too.
    EXPECT_TRUE(enabled("action.workflow.build_orthomosaic"))
        << "surface_data=\"model\" needs only the mesh";
}

TEST_F(WorkflowTest, TheOrthomosaicNeedsASurfaceToProjectOn) {
    // A chunk with neither a DEM nor a mesh: the ortho has nothing to project on.
    give_depth_maps(chunk());
    refresh();
    EXPECT_FALSE(enabled("action.workflow.build_orthomosaic"));
    EXPECT_TRUE(
        workflow().action("BuildOrthomosaic")->toolTip().contains(QStringLiteral("elevation")));

    give_elevation(chunk());
    refresh();
    EXPECT_TRUE(enabled("action.workflow.build_orthomosaic"));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(WorkflowTest, ATaskTheMenuDoesNotOfferCannotRun) {
    // Not in the menu, so no action; and run() refuses rather than opening a dialog for it
    // (exporting is a separate feature, not a Workflow step).
    EXPECT_EQ(workflow().action("ExportModel"), nullptr);
    EXPECT_EQ(workflow().unavailable_reason("ExportModel"), QStringLiteral("This task is not available."));
    EXPECT_FALSE(workflow().run("ExportModel"));
    // An invented name is refused the same way, before the registry is even consulted.
    EXPECT_FALSE(workflow().run("NoSuchTask"));
    EXPECT_TRUE(submitted_.empty());
    EXPECT_TRUE(errors_.empty());
}

TEST_F(WorkflowTest, DisabledActionsSayWhatIsMissing) {
    EXPECT_TRUE(workflow().action("AddPhotos")->toolTip().isEmpty());
    const QString blocking = workflow().action("BuildModel")->toolTip();
    EXPECT_FALSE(blocking.isEmpty());
    EXPECT_TRUE(blocking.contains(QStringLiteral("depth maps")));
    EXPECT_EQ(workflow().action("BuildModel")->statusTip(), blocking);
    EXPECT_EQ(workflow().unavailable_reason("BuildModel"), blocking);
    EXPECT_TRUE(workflow().unavailable_reason("AddPhotos").isEmpty());
}

TEST_F(WorkflowTest, ActionsAreDisabledWhileAJobRuns) {
    auto started = std::make_shared<std::atomic<bool>>(false);
    hold_registry_.register_task(std::make_shared<HoldTask>(hold_release_, started));
    app().jobs().set_registry(&hold_registry_);

    EXPECT_TRUE(enabled("action.workflow.add_photos"));
    app().jobs().submit("HoldTask", Params{}, chunk_id_);
    ASSERT_TRUE(pump_until([&]() { return started->load(); }, 60000));
    for (const WorkflowStep& step : WorkflowController::steps()) {
        EXPECT_FALSE(workflow().action(step.task)->isEnabled()) << step.task;
        EXPECT_EQ(workflow().action(step.task)->toolTip(), QStringLiteral("A job is running."));
    }
    EXPECT_FALSE(workflow().run("AddPhotos"));
    EXPECT_EQ(workflow().unavailable_reason("AddPhotos"), QStringLiteral("A job is running."));

    hold_release_->store(true);
    ASSERT_TRUE(pump_until([&]() { return !app().jobs().busy(); }, 60000));
    app().jobs().set_registry(&builtin_registry());
    EXPECT_TRUE(enabled("action.workflow.add_photos"));
    EXPECT_TRUE(submitted_.empty());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(WorkflowTest, TriggeringAnActionSubmitsTheDialogParamsForTheActiveChunk) {
    give_depth_maps(chunk());
    refresh();
    std::vector<std::string> asked;
    workflow().set_params_handler([&](const std::string& task) -> std::optional<Params> {
        asked.push_back(task);
        Params params = make_params(builtin_registry().require(task));
        params.set("resolution", 0.25);
        params.set("source_data", std::string("depth_maps"));
        return params;
    });

    ASSERT_TRUE(trigger("action.workflow.build_dem"));
    ASSERT_EQ(asked, (std::vector<std::string>{"BuildDem"}));
    ASSERT_EQ(submitted_.size(), 1U);
    EXPECT_EQ(submitted_[0].task, "BuildDem");
    EXPECT_EQ(submitted_[0].chunk_id, chunk_id_);
    EXPECT_DOUBLE_EQ(submitted_[0].params.get_float("resolution"), 0.25);
    EXPECT_EQ(submitted_[0].params.get_string("source_data"), "depth_maps");
    EXPECT_TRUE(errors_.empty());
}

TEST_F(WorkflowTest, ACancelledDialogSubmitsNothing) {
    workflow().set_params_handler(
        [](const std::string&) -> std::optional<Params> { return std::nullopt; });
    chunk().cameras().push_back(make_camera(chunk(), false));
    refresh();
    ASSERT_TRUE(trigger("action.workflow.match_photos"));
    EXPECT_TRUE(submitted_.empty());
    EXPECT_TRUE(errors_.empty());
}

TEST_F(WorkflowTest, AddPhotosUsesTheFilePickerAndSubmitsTheFileList) {
    int dialog_calls = 0;
    workflow().set_params_handler([&](const std::string&) -> std::optional<Params> {
        ++dialog_calls;
        return std::nullopt;
    });
    workflow().set_photos_handler([]() { return QStringList{"/data/a.jpg", "/data/b.jpg"}; });
    ASSERT_TRUE(trigger("action.workflow.add_photos"));
    EXPECT_EQ(dialog_calls, 0) << "AddPhotos asks for files, not for a JSON list";
    ASSERT_EQ(submitted_.size(), 1U);
    EXPECT_EQ(submitted_[0].task, "AddPhotos");
    EXPECT_EQ(submitted_[0].chunk_id, chunk_id_);
    EXPECT_EQ(submitted_[0].params.get_string_list("filenames"),
              (std::vector<std::string>{"/data/a.jpg", "/data/b.jpg"}));
    EXPECT_TRUE(submitted_[0].params.get_bool("load_reference"));

    // Cancelling the picker submits nothing.
    workflow().set_photos_handler([]() { return QStringList(); });
    ASSERT_TRUE(trigger("action.workflow.add_photos"));
    EXPECT_EQ(submitted_.size(), 1U);
    EXPECT_TRUE(errors_.empty());
}

TEST_F(WorkflowTest, ASubmitFailureIsReportedNotThrown) {
    workflow().set_photos_handler([]() { return QStringList{"x.jpg"}; });
    workflow().set_submit_handler(
        [](const std::string&, Params, std::int32_t) { throw Error("shutting down"); });
    QString title, message;
    workflow().set_error_handler([&](const QString& t, const QString& m) {
        title = t;
        message = m;
    });
    ASSERT_TRUE(trigger("action.workflow.add_photos"));
    EXPECT_FALSE(title.isEmpty());
    EXPECT_EQ(message, "shutting down");

    // A failed submission leaves the menu usable.
    EXPECT_TRUE(enabled("action.workflow.add_photos"));
}

TEST_F(WorkflowTest, TheDefaultSubmitReachesTheJobManager) {
    // No submit spy: a file that does not exist makes the real AddPhotos task fail, which proves
    // the request went through the job manager for the active chunk.
    workflow().set_submit_handler([this](const std::string& task, Params params,
                                         std::int32_t target) {
        app().jobs().submit(task, std::move(params), target);
    });
    workflow().set_photos_handler([]() { return QStringList{"/definitely/not/here.jpg"}; });
    // Shared, not a reference capture: the connection lives in the window, which outlives the
    // test body, so a late emit during teardown must not touch a destroyed local.
    auto final_state = std::make_shared<std::optional<JobManager::State>>();
    QObject::connect(&app().jobs(), &JobManager::job_finished, &app(),
                     [final_state](quint64, JobManager::State state, const QString&) {
                         *final_state = state;
                     });
    ASSERT_TRUE(trigger("action.workflow.add_photos"));
    ASSERT_TRUE(pump_until([&]() { return final_state->has_value(); }, 60000));
    EXPECT_EQ(*final_state, JobManager::State::Failed);
    ASSERT_EQ(errors_.size(), 1U);
    EXPECT_TRUE(errors_[0].second.contains(QStringLiteral("not/here.jpg")));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}
