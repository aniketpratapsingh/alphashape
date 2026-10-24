// GuiPipelineEndToEnd.*: the Phase 1 exit check.
//
// One offscreen GUI window, one real project, the whole pipeline driven the way a user drives
// it: Add photos, then the Workflow menu top to bottom, each action going through its parameter
// handler into the JobManager and running the real engine task on the worker thread. Nothing is
// stubbed except the two things a test cannot click: the file picker (a fixed list of paths) and
// the task dialogs (small parameters, so the run fits a CI budget).
//
// The photographs are the committed 12-image aerial block (engine/data/aerial_r*c*.jpg), the same
// images tools/sample/make_sample.py renders; they are copied into this test's scratch directory
// first, so the project references them the way a downloaded sample would.
//
// The assertions are the Phase 1 exit criteria: every stage's asset really exists, the 3-D view
// has cameras, points and a mesh, and the project still holds all of it after a save and reopen.
//
// The test gets a binary and a timeout of its own (see CMakeLists.txt) and is listed in the
// heavy lane of .github/ci/test-lanes.json: it runs the dense stage end to end, which is more
// than a light shard should carry.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QString>
#include <QStringList>

#include "apps/gui/job_manager.hpp"
#include "apps/gui/main_window.hpp"
#include "apps/gui/viewer/scene.hpp"
#include "apps/gui/viewer/viewport_widget.hpp"
#include "apps/gui/workflow.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/tasks/params.hpp"
#include "common/tasks/registry.hpp"
#include "harness.hpp"
#include "image/image.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;
using alphashape::gui::testing::pump;
using alphashape::gui::testing::pump_until;

namespace {

/// How long one stage may take before the test calls it hung. Generous: the whole pipeline runs
/// in about a minute on one CI core, and a slow shared runner must not turn that into a failure.
/// The ctest entry carries a larger overall TIMEOUT, so a hang is reported by this test with the
/// stage name rather than by ctest killing the process with no message.
constexpr int kStepTimeoutMs = 180000;

/// The committed synthetic aerial block; the same scene the sample dataset is rendered from.
QStringList aerial_images() {
    QStringList files;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            files << QStringLiteral(ALPHASHAPE_GUI_DATA_DIR "/aerial_r%1c%2.jpg")
                         .arg(row)
                         .arg(col);
        }
    }
    return files;
}

/// The small parameter set every stage runs with, so the pipeline fits the CI budget. The sizes
/// match what the existing end-to-end tests use: depth downscale 4, a 96^3 fusion volume over
/// the 120 m site, a 256 px atlas.
void apply_ci_params(const std::string& task, Params& params) {
    if (task == "MatchPhotos") {
        params.set("keypoint_limit_per_mpx", ParamValue{std::int64_t{10000}});
    } else if (task == "BuildDepthMaps") {
        params.set("downscale", ParamValue{std::int64_t{4}});
        params.set("filter_mode", ParamValue{std::string("mild")});
        params.set("max_neighbors", ParamValue{std::int64_t{8}});
    } else if (task == "BuildModel") {
        // The schema default volume, 128 voxels on the longest bounding-box axis, is cheap for
        // the TV-L1 fusion (well under a second) and keeps the roof a real surface rather than a
        // single voxel layer. "low" decimates to a twentieth of the extracted faces.
        params.set("volume_resolution", ParamValue{std::int64_t{128}});
        params.set("face_count", ParamValue{std::string("low")});
    } else if (task == "BuildUV" || task == "BuildTexture") {
        params.set("texture_size", ParamValue{std::int64_t{256}});
    }
}

class GuiPipelineEndToEnd : public GuiTest {
protected:
    void SetUp() override {
        GuiTest::SetUp();
        Chunk& added = app().document().workspace().add_chunk("Sample Site");
        chunk_id_ = added.id();
        app().document().set_active_chunk_id(chunk_id_);

        // A failing job would open a modal box that never closes offscreen.
        window().set_error_handler([this](const QString& title, const QString& message) {
            errors_.emplace_back(title, message);
        });

        // Stage the images where a downloaded sample would keep them, next to the project.
        const QString image_dir = scratch(QStringLiteral("sample/images"));
        ASSERT_TRUE(QDir().mkpath(image_dir)) << image_dir.toStdString();
        for (const QString& source : aerial_images()) {
            const QString target = QDir(image_dir).filePath(QFileInfo(source).fileName());
            ASSERT_TRUE(QFile::copy(source, target)) << source.toStdString();
            images_ << target;
        }
        ASSERT_EQ(images_.size(), 12);

        // The Workflow menu's two user-interaction hooks. Everything else -- the availability
        // check, the source presets, the submission and the worker -- is the production path.
        window().workflow().set_photos_handler([this]() { return images_; });
        window().workflow().set_params_handler(
            [](const std::string& task, const Presets& presets) -> std::optional<Params> {
                // Start where the real dialog would: the controller presets the parameter that
                // selects an existing source to data this chunk really holds. Then shrink the
                // settings to CI size.
                Params params = make_params(builtin_registry().require(task));
                for (const auto& [name, value] : presets) {
                    params.set(name, value);
                }
                apply_ci_params(task, params);
                return params;
            });
        window().workflow().set_submit_handler(
            [this](const std::string& task, Params params, std::int32_t target) {
                submitted_ = app().jobs().submit(task, std::move(params), target);
            });
        QObject::connect(&app().jobs(), &JobManager::job_finished, &app(),
                         [this](quint64 id, JobManager::State state, const QString& message) {
                             finished_.emplace_back(id, state, message);
                         });
        window().workflow().update_enabled();
        ready_ = true;
    }

    void TearDown() override {
        if (!ready_) {
            GuiTest::TearDown();  // SetUp never finished: nothing of ours to unwind
            return;
        }
        if (app().jobs().busy()) {
            app().jobs().cancel_all_and_wait();
        }
        GuiTest::TearDown();
    }

    /// Trigger a Workflow action, wait for the job it queued and require it to succeed. The
    /// action's own availability is part of the assertion: it is enabled only when the previous
    /// stage really produced what this one consumes.
    bool run_step(const QString& action_name, const char* task) {
        if (app().jobs().busy()) {
            ADD_FAILURE() << task << ": the previous job is still running";
            return false;
        }
        submitted_.reset();
        if (!trigger(action_name)) {
            return false;  // trigger() already recorded the failure
        }
        if (!submitted_) {
            ADD_FAILURE() << task << " submitted no job";
            return false;
        }
        const JobManager::JobId id = *submitted_;
        const auto has_finished = [this, id]() {
            for (const auto& entry : finished_) {
                if (std::get<0>(entry) == id) {
                    return true;
                }
            }
            return false;
        };
        if (!pump_until(has_finished, kStepTimeoutMs)) {
            const std::optional<JobManager::Job> job = app().jobs().job(id);
            ADD_FAILURE() << task << " did not finish within " << (kStepTimeoutMs / 1000)
                          << " s (stage "
                          << (job ? job->stage.toStdString() : std::string("?")) << ", "
                          << (job ? job->percent : 0.0) << "% done)";
            app().jobs().cancel_all_and_wait();
            return false;
        }
        // The document belongs to the worker until the manager goes idle; every read below is
        // after this point.
        if (!pump_until([this]() { return !app().jobs().busy(); }, kStepTimeoutMs)) {
            ADD_FAILURE() << task << ": the job manager stayed busy after the job finished";
            return false;
        }
        const std::optional<JobManager::Job> job = app().jobs().job(id);
        if (!job.has_value()) {
            ADD_FAILURE() << task << ": the manager forgot the finished job";
            return false;
        }
        std::printf("[gui e2e] %-16s %6lld ms\n", task,
                    static_cast<long long>(job->elapsed_ms));
        if (job->state != JobManager::State::Succeeded) {
            ADD_FAILURE() << task << " failed: " << job->message.toStdString();
            return false;
        }
        return true;
    }

    const Chunk& chunk() { return *app().document().active_chunk(); }

    std::int32_t chunk_id_ = -1;
    QStringList images_;
    std::optional<JobManager::JobId> submitted_;
    std::vector<std::tuple<JobManager::JobId, JobManager::State, QString>> finished_;
    std::vector<std::pair<QString, QString>> errors_;
    bool ready_ = false;
};

TEST_F(GuiPipelineEndToEnd, BuildsSavesAndReopensTheSampleProject) {
    show_window();
    ViewportWidget* viewport = window().viewport();
    ASSERT_NE(viewport, nullptr);
    EXPECT_TRUE(viewport->scene_is_empty()) << "a fresh chunk draws nothing";

    // -- the pipeline, one Workflow action at a time -------------------------
    ASSERT_TRUE(run_step("action.workflow.add_photos", "AddPhotos"));
    ASSERT_EQ(chunk().cameras().size(), 12U);
    EXPECT_FALSE(chunk().frames().empty());

    ASSERT_TRUE(run_step("action.workflow.match_photos", "MatchPhotos"));
    EXPECT_FALSE(chunk().tie_points().matches.empty());
    EXPECT_FALSE(chunk().tie_points().tracks.empty());

    ASSERT_TRUE(run_step("action.workflow.align_cameras", "AlignCameras"));
    std::size_t aligned = 0;
    for (const Camera& camera : chunk().cameras()) {
        if (camera.enabled) {
            ++aligned;
        }
    }
    std::printf("[gui e2e] aligned %zu of %zu cameras\n", aligned, chunk().cameras().size());
    EXPECT_GE(aligned, 9U) << "the block must align essentially in full";

    ASSERT_TRUE(run_step("action.workflow.build_depth_maps", "BuildDepthMaps"));
    ASSERT_FALSE(chunk().depth_maps().empty());
    // A camera needs at least two neighbours, so a well-aligned block should give a map per
    // camera; two thirds is the floor that still proves the stage really ran on the block.
    EXPECT_GE(chunk().depth_maps().size(), 6U) << "too few cameras produced a depth map";
    for (const DepthMap& map : chunk().depth_maps()) {
        EXPECT_GT(map.width, 0);
        EXPECT_EQ(map.downscale, 4);
        EXPECT_EQ(map.depths.size(),
                  static_cast<std::size_t>(map.width) * static_cast<std::size_t>(map.height));
    }

    ASSERT_TRUE(run_step("action.workflow.build_point_cloud", "BuildPointCloud"));
    const PointCloud& cloud = chunk().point_cloud();
    ASSERT_GT(cloud.points.size(), 1000U) << "fusion produced no usable cloud";
    EXPECT_EQ(cloud.normals.size(), cloud.points.size());
    EXPECT_EQ(cloud.colours.size(), cloud.points.size() * 3U);

    ASSERT_TRUE(run_step("action.workflow.build_model", "BuildModel"));
    const Mesh& mesh = chunk().model();
    ASSERT_FALSE(mesh.empty()) << "meshing produced no faces";
    ASSERT_FALSE(mesh.vertices.empty());
    EXPECT_EQ(mesh.colours.size(), mesh.vertices.size() * 3U);

    ASSERT_TRUE(run_step("action.workflow.build_uv", "BuildUV"));
    EXPECT_FALSE(mesh.uv_faces.empty()) << "BuildUV wrote no texture coordinates";
    EXPECT_EQ(mesh.uv_faces.size(), mesh.faces.size());

    ASSERT_TRUE(run_step("action.workflow.build_texture", "BuildTexture"));
    ASSERT_FALSE(chunk().model_textures().empty()) << "BuildTexture baked no page";
    ASSERT_FALSE(chunk().model_textures().front().empty());
    EXPECT_EQ(chunk().model_textures().front().width(), 256);

    EXPECT_TRUE(errors_.empty())
        << (errors_.empty() ? std::string() : errors_.front().second.toStdString());

    // -- the 3-D view shows the reconstruction ------------------------------
    // The viewport rebuilds on job_finished; the last one picks up the textured mesh, the camera
    // frustums and the dense cloud, so every layer of the scene is present.
    const Scene& scene = viewport->scene();
    EXPECT_EQ(scene.total_cameras, 12U);
    // The scene counts a placed camera (a non-identity pose), which is what it can draw a
    // frustum for; `aligned` counts the cameras the aligner flagged. On a successfully aligned
    // block the two agree, so requiring both keeps the assertion honest without depending on
    // the exact bookkeeping one of them does.
    EXPECT_EQ(scene.aligned_cameras, aligned);
    EXPECT_GE(scene.aligned_cameras, 9U);
    EXPECT_FALSE(scene.tie_points.empty()) << "no sparse points in the view";
    EXPECT_FALSE(scene.dense_points.empty()) << "no dense points in the view";
    EXPECT_TRUE(scene.has_mesh()) << "the view has no mesh";
    EXPECT_TRUE(scene.has_texture()) << "the view did not pick up the baked page";
    EXPECT_FALSE(scene.lines.empty()) << "no camera frustums, grid or axes";
    EXPECT_TRUE(scene.has_bounds);

    // -- save, reopen, and check every product survived ---------------------
    // Through the window's own File menu actions; the path handlers are the only stand-ins.
    const QString project = scratch(QStringLiteral("sample/sample-site.psx"));
    window().set_save_path_handler([&project](const QString&) { return project; });
    window().set_open_path_handler([&project]() { return project; });
    window().set_unsaved_handler([](const QString&) {
        ADD_FAILURE() << "no unsaved-changes prompt expected on a clean document";
        return UnsavedChoice::Cancel;
    });
    ASSERT_TRUE(trigger("action.file.save_as")) << "could not save the project";
    ASSERT_TRUE(QFileInfo::exists(project));
    // The project is the .psx file plus a sibling folder named after the stem
    // ("sample-site.files", NOT "sample-site.psx.files"; see ProjectIO::save).
    const QString sidecar =
        QFileInfo(project).absolutePath() + QStringLiteral("/") +
        QFileInfo(project).completeBaseName() + QStringLiteral(".files");
    EXPECT_TRUE(QFileInfo(sidecar).isDir()) << "the project's sidecar folder is missing: "
                                           << sidecar.toStdString();
    EXPECT_FALSE(app().dirty());

    const std::size_t saved_cameras = chunk().cameras().size();
    const std::size_t saved_faces = chunk().model().faces.size();
    const std::size_t saved_points = chunk().point_cloud().points.size();
    const std::size_t saved_maps = chunk().depth_maps().size();

    // Reopening replaces the document, so everything below reads the reloaded one.
    ASSERT_TRUE(trigger("action.file.open")) << "could not reopen the project";
    pump(50);
    ASSERT_NE(app().document().active_chunk(), nullptr);
    EXPECT_EQ(chunk().cameras().size(), saved_cameras);
    EXPECT_EQ(chunk().model().faces.size(), saved_faces);
    EXPECT_FALSE(chunk().model().uv_faces.empty()) << "UV coordinates did not survive the save";
    EXPECT_EQ(chunk().model().uv_faces.size(), saved_faces);
    EXPECT_EQ(chunk().point_cloud().points.size(), saved_points);
    EXPECT_EQ(chunk().depth_maps().size(), saved_maps);
    ASSERT_FALSE(chunk().model_textures().empty()) << "the texture page did not survive";
    EXPECT_EQ(chunk().model_textures().front().width(), 256);

    // A real texture, not a uniformly black page: the bake must have sampled the photographs.
    std::size_t coloured = 0;
    const Image& page = chunk().model_textures().front();
    for (int y = 0; y < page.height(); ++y) {
        for (int x = 0; x < page.width(); ++x) {
            if (page.at(x, y, 0) > 8 || page.at(x, y, 1) > 8 || page.at(x, y, 2) > 8) {
                ++coloured;
            }
        }
    }
    const std::size_t texels =
        static_cast<std::size_t>(page.width()) * static_cast<std::size_t>(page.height());
    std::printf("[gui e2e] baked page %dx%d, %zu of %zu texels coloured\n", page.width(),
                page.height(), coloured, texels);
    EXPECT_GT(coloured, texels / 4U) << "most of the atlas should carry photograph colour";

    EXPECT_TRUE(errors_.empty())
        << (errors_.empty() ? std::string() : errors_.front().second.toStdString());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

}  // namespace
