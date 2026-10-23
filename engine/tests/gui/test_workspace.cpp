// WorkspaceTree.*: the Workspace tree model and pane (WP-14.6) and the Properties pane (WP-14.9),
// driven through the offscreen harness against engine Documents built in the tests.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <QAbstractItemModelTester>
#include <QFont>
#include <QMenu>
#include <QTableWidget>
#include <QTreeView>

#include "apps/gui/properties_pane.hpp"
#include "apps/gui/workspace_model.hpp"
#include "apps/gui/workspace_pane.hpp"
#include "base/utils/error.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/tasks/registry.hpp"
#include "harness.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;
using alphashape::gui::testing::pump_until;

namespace {

class WorkspaceTree : public GuiTest {
protected:
    WorkspacePane& pane() { return *widget<WorkspacePane>("pane.workspace"); }
    WorkspaceModel& model() { return pane().model(); }
    PropertiesPane& props() { return *widget<PropertiesPane>("pane.properties"); }
    WorkspaceSelection& selection() {
        return *window().findChild<WorkspaceSelection*>("selection.workspace");
    }

    /// A chunk with 2 sensors, 4 cameras (2 aligned, 1 disabled, 2 in a group), a point cloud
    /// of 5 points, a model of 3 faces, 1 marker and 1 scale bar. Returns its id.
    std::int32_t build_chunk(const char* label) {
        Document& document = app().document();
        Chunk& chunk = document.workspace().add_chunk(label);
        Sensor a;
        a.label = "DJI FC6310";
        a.width = 5472;
        a.height = 3648;
        a.focal_length_mm = 8.8;
        a.pixel_size_um = 2.41;
        a.calibration = Calibration::initial(CameraType::Frame, 5472, 3648, 3650.0);
        Sensor b = a;
        b.label = "Second";
        chunk.sensors() = {a, b};
        Transform moved = Transform::Identity();
        moved(0, 3) = 2.0;
        for (int i = 0; i < 4; ++i) {
            Camera camera;
            camera.id = i;
            camera.sensor_id = i % 2;
            camera.label = "IMG_" + std::to_string(i);
            camera.enabled = (i != 3);
            camera.pose = (i < 2) ? moved : Transform::Identity();
            chunk.cameras().push_back(camera);
            auto frame = std::make_shared<Frame>("/photos/IMG_" + std::to_string(i) + ".jpg", 5472, 3648);
            frame->set_id(i);
            frame->set_camera_id(i);
            chunk.frames().push_back(frame);
        }
        CameraGroup group;
        group.id = chunk.allocate_id();
        group.label = "Flight 1";
        group.members = {2, 3};
        chunk.camera_groups().push_back(group);
        PointCloudAsset& cloud = chunk.add_point_cloud("Dense cloud");
        cloud.data->points.assign(5, Eigen::Vector3d::Zero());
        ModelAsset& mesh = chunk.add_model("Mesh");
        mesh.data->vertices.assign(3, Eigen::Vector3d::Zero());
        mesh.data->faces.assign(3, Eigen::Vector3i(0, 1, 2));
        Marker marker;
        marker.id = chunk.allocate_id();
        marker.label = "GCP 1";
        chunk.markers().push_back(marker);
        Scalebar bar;
        bar.id = chunk.allocate_id();
        bar.label = "Bar 1";
        bar.reference_length_m = 1.5;
        chunk.scalebars().push_back(bar);
        document.set_active_chunk_id(chunk.id());
        return chunk.id();
    }

    QModelIndex find(const NodeKind kind, std::int32_t chunk, std::int32_t item = -1) {
        const QModelIndex index = model().index_for(NodeRef{kind, chunk, item});
        EXPECT_TRUE(index.isValid()) << "node kind " << static_cast<int>(kind);
        return index;
    }
    QString text(const QModelIndex& index) { return index.data(Qt::DisplayRole).toString(); }
    QString text_of(NodeKind kind, std::int32_t chunk, std::int32_t item = -1) {
        return text(find(kind, chunk, item));
    }
    Chunk& chunk_ref(std::int32_t id) { return *app().document().workspace().find_chunk(id); }
    void tree_double_click(std::int32_t chunk_id) {
        pane().tree()->doubleClicked(find(NodeKind::Chunk, chunk_id));
    }
    std::vector<bool> enabled_flags(std::int32_t id) {
        std::vector<bool> flags;
        for (const Camera& camera : chunk_ref(id).cameras()) {
            flags.push_back(camera.enabled);
        }
        return flags;
    }
    int chunk_count() { return model().rowCount(find(NodeKind::Document, -1)); }
};

}  // namespace

TEST_F(WorkspaceTree, EmptyDocumentShowsOnlyTheDocumentNode) {
    QAbstractItemModelTester tester(&model(), QAbstractItemModelTester::FailureReportingMode::Warning);
    ASSERT_EQ(model().rowCount(), 1);
    EXPECT_EQ(chunk_count(), 0);
    EXPECT_TRUE(selection().items().isEmpty());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(WorkspaceTree, ModelStructureAndCounts) {
    QAbstractItemModelTester tester(&model(), QAbstractItemModelTester::FailureReportingMode::Warning);
    const std::int32_t id = build_chunk("Block A");
    model().refresh();

    ASSERT_EQ(chunk_count(), 1);
    const QModelIndex chunk = find(NodeKind::Chunk, id);
    EXPECT_EQ(text(chunk), "Block A");

    // Groups present in the data model, in a fixed order.
    const QStringList expected = {
        "Cameras (2/4)", "Sensors (2)", "Tie points (none)", "Depth maps (none)",
        "Point cloud (5 points)", "Model (3 faces)", "Tiled model (none)", "DEM (none)",
        "Orthomosaic (none)", "Markers (1)", "Scale bars (1)", "Shapes (none)"};
    QStringList actual;
    for (int row = 0; row < model().rowCount(chunk); ++row) {
        actual << text(model().index(row, 0, chunk));
    }
    EXPECT_EQ(actual, expected);

    // Empty groups are greyed, populated ones are not.
    EXPECT_TRUE(find(NodeKind::TiePoints, id).data(Qt::ForegroundRole).isValid());
    EXPECT_FALSE(find(NodeKind::Cameras, id).data(Qt::ForegroundRole).isValid());

    // Cameras: group "Flight 1" holds cameras 2 and 3, 0 and 1 sit directly under Cameras.
    const QModelIndex cameras = find(NodeKind::Cameras, id);
    EXPECT_EQ(model().rowCount(cameras), 3);
    const QModelIndex group = find(NodeKind::CameraGroup, id, chunk_ref(id).camera_groups().front().id);
    EXPECT_EQ(text(group), "Flight 1 (2)");
    EXPECT_EQ(model().parent(find(NodeKind::Camera, id, 2)), group);
    EXPECT_EQ(model().parent(find(NodeKind::Camera, id, 0)), cameras);
    EXPECT_EQ(text_of(NodeKind::Camera, id, 1), "IMG_1");
    // Disabled camera is greyed.
    EXPECT_TRUE(find(NodeKind::Camera, id, 3).data(Qt::ForegroundRole).isValid());
    EXPECT_FALSE(find(NodeKind::Camera, id, 0).data(Qt::ForegroundRole).isValid());

    EXPECT_EQ(model().rowCount(find(NodeKind::Sensors, id)), 2);
    EXPECT_EQ(text_of(NodeKind::Sensor, id, 1), "Second");
    EXPECT_EQ(text_of(NodeKind::PointCloud, id, chunk_ref(id).point_clouds().front().id),
              "Dense cloud (5 points)");
    EXPECT_EQ(text_of(NodeKind::Marker, id, chunk_ref(id).markers().front().id), "GCP 1");
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(WorkspaceTree, ActiveChunkIsBoldAndDoubleClickActivates) {
    QAbstractItemModelTester tester(&model(), QAbstractItemModelTester::FailureReportingMode::Warning);
    const std::int32_t first = build_chunk("One");
    const std::int32_t second = build_chunk("Two");
    app().document().set_active_chunk_id(first);
    model().refresh();

    auto bold = [&](std::int32_t id) {
        return find(NodeKind::Chunk, id).data(Qt::FontRole).value<QFont>().bold();
    };
    EXPECT_TRUE(bold(first));
    EXPECT_FALSE(bold(second));

    tree_double_click(second);
    EXPECT_EQ(app().document().active_chunk_id(), second);
    EXPECT_FALSE(bold(first));
    EXPECT_TRUE(bold(second));
    EXPECT_TRUE(app().dirty());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(WorkspaceTree, AddAndRemoveChunkUpdateTheTreeThroughUndoAndRedo) {
    QAbstractItemModelTester tester(&model(), QAbstractItemModelTester::FailureReportingMode::Warning);
    const std::int32_t first = build_chunk("One");
    model().refresh();
    EXPECT_EQ(chunk_count(), 1);

    // Add
    ASSERT_TRUE(pane().add_chunk());
    EXPECT_EQ(chunk_count(), 2);
    EXPECT_TRUE(app().dirty());
    const QString added = text(model().index(1, 0, find(NodeKind::Document, -1)));
    EXPECT_EQ(added, "Chunk 2");
    app().undo();
    EXPECT_EQ(chunk_count(), 1);
    app().redo();
    EXPECT_EQ(chunk_count(), 2);
    EXPECT_EQ(text(model().index(1, 0, find(NodeKind::Document, -1))), added);

    // Remove (the new chunk), then undo/redo.
    const std::int32_t second = app().document().workspace().chunks().back()->id();
    ASSERT_TRUE(pane().remove_chunk(second));
    EXPECT_EQ(chunk_count(), 1);
    app().undo();
    EXPECT_EQ(chunk_count(), 2);
    app().redo();
    EXPECT_EQ(chunk_count(), 1);
    app().undo();

    // The first chunk's subtree is intact after all of that.
    EXPECT_EQ(text_of(NodeKind::Cameras, first), "Cameras (2/4)");
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(WorkspaceTree, RemoveIsDisabledForTheLastChunkAndFailuresAreReported) {
    const std::int32_t only = build_chunk("Only");
    model().refresh();
    QString reported;
    pane().set_error_handler([&](const QString&, const QString& message) { reported = message; });

    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Chunk, only, -1}));
    auto menu = pane().build_context_menu();
    QAction* remove = menu->findChild<QAction*>("action.workspace.remove_chunk");
    ASSERT_NE(remove, nullptr);
    EXPECT_FALSE(remove->isEnabled());
    EXPECT_NE(menu->findChild<QAction*>("action.workspace.add_chunk"), nullptr);

    // Forcing it anyway reports the engine's refusal and leaves the tree alone.
    EXPECT_FALSE(pane().remove_chunk(only));
    EXPECT_FALSE(reported.isEmpty());
    EXPECT_EQ(chunk_count(), 1);
    EXPECT_FALSE(pane().remove_chunk(9999));
}

TEST_F(WorkspaceTree, EnableAndDisableCamerasThroughContextMenuUndoRedo) {
    QAbstractItemModelTester tester(&model(), QAbstractItemModelTester::FailureReportingMode::Warning);
    const std::int32_t id = build_chunk("One");
    model().refresh();
    ASSERT_EQ(enabled_flags(id), (std::vector<bool>{true, true, true, false}));

    // Select cameras 0 and 1: only "Disable" applies.
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Camera, id, 0}));
    ASSERT_TRUE(pane().add_to_selection(NodeRef{NodeKind::Camera, id, 1}));
    ASSERT_EQ(selection().items().size(), 2);
    {
        auto menu = pane().build_context_menu();
        EXPECT_FALSE(menu->findChild<QAction*>("action.workspace.enable_cameras")->isEnabled());
        QAction* disable = menu->findChild<QAction*>("action.workspace.disable_cameras");
        ASSERT_TRUE(disable->isEnabled());
        disable->trigger();
    }
    EXPECT_EQ(enabled_flags(id), (std::vector<bool>{false, false, true, false}));
    EXPECT_TRUE(find(NodeKind::Camera, id, 0).data(Qt::ForegroundRole).isValid());
    app().undo();
    EXPECT_EQ(enabled_flags(id), (std::vector<bool>{true, true, true, false}));
    EXPECT_FALSE(find(NodeKind::Camera, id, 0).data(Qt::ForegroundRole).isValid());
    app().redo();
    EXPECT_EQ(enabled_flags(id), (std::vector<bool>{false, false, true, false}));
    app().undo();

    // The camera group selects its members: camera 3 is disabled, so "Enable" applies.
    ASSERT_TRUE(pane().select(
        NodeRef{NodeKind::CameraGroup, id, chunk_ref(id).camera_groups().front().id}));
    {
        auto menu = pane().build_context_menu();
        QAction* enable = menu->findChild<QAction*>("action.workspace.enable_cameras");
        ASSERT_TRUE(enable->isEnabled());
        enable->trigger();
    }
    EXPECT_EQ(enabled_flags(id), (std::vector<bool>{true, true, true, true}));
    app().undo();
    EXPECT_EQ(enabled_flags(id), (std::vector<bool>{true, true, true, false}));
    app().redo();
    EXPECT_EQ(enabled_flags(id), (std::vector<bool>{true, true, true, true}));

    // Cameras from different chunks, or a non-camera, offer no camera actions.
    const std::int32_t other = build_chunk("Two");
    model().refresh();
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Camera, id, 0}));
    ASSERT_TRUE(pane().add_to_selection(NodeRef{NodeKind::Camera, other, 0}));
    EXPECT_EQ(pane().build_context_menu()->findChild<QAction*>("action.workspace.enable_cameras"),
              nullptr);
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(WorkspaceTree, TreeStateSurvivesInPlaceUpdates) {
    const std::int32_t id = build_chunk("One");
    model().refresh();
    const QModelIndex cameras = find(NodeKind::Cameras, id);
    pane().tree()->expand(cameras);
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Camera, id, 0}));

    ASSERT_TRUE(pane().add_chunk());  // a sibling appears; this chunk's subtree is untouched
    EXPECT_TRUE(pane().tree()->isExpanded(find(NodeKind::Cameras, id)));
    ASSERT_EQ(selection().items().size(), 1);
    EXPECT_EQ(selection().current(), (NodeRef{NodeKind::Camera, id, 0}));
}

TEST_F(WorkspaceTree, ReplacingTheDocumentResetsTreeAndSelection) {
    QAbstractItemModelTester tester(&model(), QAbstractItemModelTester::FailureReportingMode::Warning);
    const std::int32_t id = build_chunk("One");
    model().refresh();
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Chunk, id, -1}));
    ASSERT_FALSE(selection().items().isEmpty());

    app().new_document("");
    EXPECT_EQ(chunk_count(), 0);
    EXPECT_TRUE(selection().items().isEmpty());
    EXPECT_TRUE(props().rows().isEmpty());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(WorkspaceTree, SelectionDrivesTheProperties) {
    const std::int32_t id = build_chunk("Block A");
    app().document().workspace().find_chunk(id)->crs().epsg = 32644;
    model().refresh();
    EXPECT_TRUE(props().title().isEmpty());  // nothing selected

    // Chunk
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Chunk, id, -1}));
    EXPECT_EQ(props().title(), "Block A");
    EXPECT_EQ(props().value("Name"), "Block A");
    EXPECT_EQ(props().value("Cameras"), "4 (2 aligned)");
    EXPECT_EQ(props().value("Sensors"), "2");
    EXPECT_EQ(props().value("Coordinate system"), "EPSG:32644");
    EXPECT_FALSE(props().value("Region centre").isEmpty());
    EXPECT_FALSE(props().value("Region size").isEmpty());

    // Camera
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Camera, id, 1}));
    EXPECT_EQ(props().title(), "IMG_1");
    EXPECT_EQ(props().value("Label"), "IMG_1");
    EXPECT_EQ(props().value("Sensor"), "Second");
    EXPECT_EQ(props().value("Enabled"), "yes");
    EXPECT_EQ(props().value("Aligned"), "yes");
    EXPECT_EQ(props().value("Image path"), "/photos/IMG_1.jpg");
    EXPECT_EQ(props().value("Size"), "5472 x 3648");
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Camera, id, 3}));
    EXPECT_EQ(props().value("Enabled"), "no");
    EXPECT_EQ(props().value("Aligned"), "no");

    // Sensor
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Sensor, id, 0}));
    EXPECT_EQ(props().value("Type"), "Frame");
    EXPECT_EQ(props().value("Resolution"), "5472 x 3648");
    EXPECT_EQ(props().value("Focal length (mm)"), "8.8");
    EXPECT_EQ(props().value("Pixel size (um)"), "2.41");
    EXPECT_EQ(props().value("Calibration f (px)"), "3650");
    EXPECT_FALSE(props().value("Calibration k1").isEmpty());

    // Assets
    const Chunk& chunk = chunk_ref(id);
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::PointCloud, id, chunk.point_clouds().front().id}));
    EXPECT_EQ(props().value("Points"), "5");
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Model, id, chunk.models().front().id}));
    EXPECT_EQ(props().value("Faces"), "3");
    EXPECT_EQ(props().value("Vertices"), "3");
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Marker, id, chunk.markers().front().id}));
    EXPECT_EQ(props().title(), "GCP 1");
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::ScaleBar, id, chunk.scalebars().front().id}));
    EXPECT_EQ(props().value("Reference length (m)"), "1.5");

    // Several items: only a count.
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Camera, id, 0}));
    ASSERT_TRUE(pane().add_to_selection(NodeRef{NodeKind::Camera, id, 1}));
    EXPECT_EQ(props().title(), "2 items selected");

    // The sheet is read-only and follows undo: renaming is not offered, but disabling a camera
    // changes what the sheet says.
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Camera, id, 0}));
    EXPECT_EQ(props().value("Enabled"), "yes");
    ASSERT_TRUE(pane().set_cameras_enabled(id, {0}, false));
    EXPECT_EQ(props().value("Enabled"), "no");
    app().undo();
    EXPECT_EQ(props().value("Enabled"), "yes");

    // A selected item that disappears empties the sheet.
    const std::int32_t doomed = build_chunk("Doomed");
    model().refresh();
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Chunk, doomed, -1}));
    EXPECT_EQ(props().title(), "Doomed");
    ASSERT_TRUE(pane().remove_chunk(doomed));
    EXPECT_TRUE(selection().items().isEmpty());
    EXPECT_TRUE(props().rows().isEmpty());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(WorkspaceTree, PropertiesTableIsReadOnly) {
    const std::int32_t id = build_chunk("One");
    model().refresh();
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Chunk, id, -1}));
    auto* table = widget<QTableWidget>("properties.table");
    ASSERT_NE(table, nullptr);
    EXPECT_EQ(table->columnCount(), 2);
    EXPECT_EQ(table->rowCount(), props().rows().size());
    EXPECT_EQ(table->editTriggers(), QAbstractItemView::NoEditTriggers);
}

// --- Jobs: the tree and the Properties pane obey the document concurrency rule ------------
//
// While a job is active the panes must not read the document and nothing may mutate it; when the
// job ends (whatever the outcome) the tree is rebuilt from scratch. A test-only task that waits at
// a gate makes this deterministic, the same pattern as test_jobs.cpp.

namespace {

struct WorkspaceGate {
    std::atomic<bool> started{false};
    std::atomic<bool> release{false};
};

/// Waits for the gate, then makes the changes the tests look for: camera 3 becomes enabled and a
/// chunk "Task chunk" appears. mode "fail" throws after changing (a failed task leaves its
/// changes behind), mode "cancel" waits for cancellation and throws CancelledError.
class WorkspaceGateTask : public AbstractTask {
public:
    explicit WorkspaceGateTask(std::shared_ptr<WorkspaceGate> gate) : gate_(std::move(gate)) {}

    std::string name() const override { return "WorkspaceGateTask"; }
    void declare_params(Params& params) const override {
        params.declare({"mode", ParamType::String, std::string{"ok"}, "ok, fail or cancel."});
    }
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override { return {Stage{"a", 0}}; }

    void apply(Document& document, Params& params, const Progress& progress) override {
        const std::string mode = params.get_string("mode");
        progress.update("a", 0.0);
        gate_->started = true;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!gate_->release.load()) {
            if (progress.is_cancelled()) {
                document.workspace().add_chunk("Task chunk");
                throw CancelledError("WorkspaceGateTask was cancelled");
            }
            if (std::chrono::steady_clock::now() > deadline) {
                throw Error("WorkspaceGateTask timed out waiting for the test");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        for (const auto& workspace : document.workspaces()) {
            for (const auto& chunk : workspace->chunks()) {
                for (Camera& camera : chunk->cameras()) {
                    camera.enabled = true;
                }
            }
        }
        document.workspace().add_chunk("Task chunk");
        if (mode == "fail") {
            throw Error("workspace gate task failed on purpose");
        }
    }

private:
    std::shared_ptr<WorkspaceGate> gate_;
};

class WorkspaceJobs : public WorkspaceTree {
protected:
    void SetUp() override {
        WorkspaceTree::SetUp();
        gate_ = std::make_shared<WorkspaceGate>();
        registry_.register_task(std::make_shared<WorkspaceGateTask>(gate_));
        app().jobs().set_registry(&registry_);
        window().set_error_handler([this](const QString&, const QString& message) {
            errors_.push_back(message);
        });
        pane().set_error_handler([this](const QString&, const QString& message) {
            errors_.push_back(message);
        });
        // Count every structural or data change the model announces.
        auto count = [this]() { ++model_events_; };
        connect_all(count);
    }

    void connect_all(const std::function<void()>& count) {
        QObject::connect(&model(), &QAbstractItemModel::modelReset, &model(), count);
        QObject::connect(&model(), &QAbstractItemModel::rowsInserted, &model(), count);
        QObject::connect(&model(), &QAbstractItemModel::rowsRemoved, &model(), count);
        QObject::connect(&model(), &QAbstractItemModel::dataChanged, &model(), count);
        QObject::connect(&model(), &QAbstractItemModel::layoutChanged, &model(), count);
    }

    void submit(const char* mode) {
        Params params = make_params(registry_.require("WorkspaceGateTask"));
        params.set("mode", ParamValue{std::string(mode)});
        app().jobs().submit("WorkspaceGateTask", std::move(params));
        ASSERT_TRUE(pump_until([&] { return gate_->started.load(); }, 60000));
        ASSERT_TRUE(app().jobs().busy());
    }
    void finish_job() {
        gate_->release = true;
        ASSERT_TRUE(pump_until([&] { return !app().jobs().busy(); }, 60000));
    }

    std::shared_ptr<WorkspaceGate> gate_;
    TaskRegistry registry_;
    std::vector<QString> errors_;
    int model_events_ = 0;
};

}  // namespace

TEST_F(WorkspaceJobs, TreeAndMenuAreFrozenWhileAJobRunsAndRefreshWhenItEnds) {
    const std::int32_t first = build_chunk("One");
    const std::int32_t second = build_chunk("Two");  // becomes the active chunk
    model().refresh();
    ASSERT_EQ(chunk_count(), 2);
    pane().tree()->expand(find(NodeKind::Cameras, first));
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Camera, first, 3}));
    ASSERT_EQ(props().value("Enabled"), "no");
    ASSERT_TRUE(find(NodeKind::Camera, first, 3).data(Qt::ForegroundRole).isValid());  // greyed
    {
        auto idle_menu = pane().build_context_menu();
        EXPECT_TRUE(idle_menu->findChild<QAction*>("action.workspace.add_chunk")->isEnabled());
        EXPECT_TRUE(idle_menu->findChild<QAction*>("action.workspace.enable_cameras")->isEnabled());
    }
    model_events_ = 0;

    ASSERT_NO_FATAL_FAILURE(submit("ok"));

    // Refresh triggers during the job only mark the panes stale; nothing is rebuilt. (Only
    // undo_changed: path_changed would make the window re-read the title under the shared lock
    // and wait for the gate.)
    Q_EMIT app().undo_changed();
    EXPECT_TRUE(model().stale());
    EXPECT_TRUE(props().stale());
    EXPECT_EQ(model_events_, 0);
    EXPECT_EQ(chunk_count(), 2);
    EXPECT_EQ(props().value("Enabled"), "no");

    // Context menu: every mutating entry is listed but disabled; only the camera ones apply here.
    {
        auto menu = pane().build_context_menu();
        for (const char* name : {"add_chunk", "enable_cameras", "disable_cameras"}) {
            QAction* entry = menu->findChild<QAction*>(QString("action.workspace.") + name);
            ASSERT_NE(entry, nullptr) << name;
            EXPECT_FALSE(entry->isEnabled()) << name;
        }
    }
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Chunk, first, -1}));
    {
        auto menu = pane().build_context_menu();
        for (const char* name : {"add_chunk", "set_active_chunk", "remove_chunk"}) {
            QAction* entry = menu->findChild<QAction*>(QString("action.workspace.") + name);
            ASSERT_NE(entry, nullptr) << name;
            EXPECT_FALSE(entry->isEnabled()) << name;
        }
    }
    // The operations themselves (shortcuts, programmatic calls, double-click) refuse too.
    EXPECT_FALSE(pane().add_chunk());
    EXPECT_FALSE(pane().remove_chunk(first));
    EXPECT_FALSE(pane().set_cameras_enabled(first, {3}, true));
    EXPECT_FALSE(pane().activate_chunk(first));
    EXPECT_EQ(errors_.size(), 4U);
    tree_double_click(first);
    EXPECT_EQ(errors_.size(), 4U);  // double-click is silent
    EXPECT_EQ(model_events_, 0);

    finish_job();

    // The task changed the document without a command: the tree was rebuilt from scratch.
    EXPECT_FALSE(model().stale());
    EXPECT_FALSE(props().stale());
    EXPECT_GE(model_events_, 1);
    EXPECT_EQ(chunk_count(), 3);
    EXPECT_FALSE(find(NodeKind::Camera, first, 3).data(Qt::ForegroundRole).isValid());
    EXPECT_EQ(app().document().active_chunk_id(), second);  // the double-click did nothing
    EXPECT_TRUE(app().dirty());
    EXPECT_FALSE(app().undo_stack().can_undo());
    // Expansion and selection came back by identity.
    EXPECT_TRUE(pane().tree()->isExpanded(find(NodeKind::Cameras, first)));
    ASSERT_EQ(selection().items().size(), 1);
    EXPECT_EQ(selection().current(), (NodeRef{NodeKind::Chunk, first, -1}));

    // The menu works again.
    auto menu = pane().build_context_menu();
    EXPECT_TRUE(menu->findChild<QAction*>("action.workspace.add_chunk")->isEnabled());
    EXPECT_TRUE(menu->findChild<QAction*>("action.workspace.set_active_chunk")->isEnabled());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(WorkspaceJobs, PropertiesUpdateAfterTheJob) {
    const std::int32_t id = build_chunk("One");
    model().refresh();
    ASSERT_TRUE(pane().select(NodeRef{NodeKind::Camera, id, 3}));
    ASSERT_EQ(props().value("Enabled"), "no");

    ASSERT_NO_FATAL_FAILURE(submit("ok"));
    EXPECT_EQ(props().value("Enabled"), "no");  // still the pre-job sheet; the document is locked
    finish_job();

    EXPECT_FALSE(props().stale());
    EXPECT_EQ(props().title(), "IMG_3");
    EXPECT_EQ(props().value("Enabled"), "yes");
}

TEST_F(WorkspaceJobs, FailedAndCancelledJobsAlsoRebuildTheTree) {
    const std::int32_t id = build_chunk("One");
    model().refresh();
    ASSERT_EQ(chunk_count(), 1);

    // A task that fails after changing the document leaves its changes; the tree shows them.
    ASSERT_NO_FATAL_FAILURE(submit("fail"));
    finish_job();
    EXPECT_EQ(chunk_count(), 2);
    EXPECT_TRUE(find(NodeKind::Camera, id, 3).isValid());
    EXPECT_FALSE(find(NodeKind::Camera, id, 3).data(Qt::ForegroundRole).isValid());

    // A cancelled one that had already added a chunk does too.
    gate_->started = false;
    gate_->release = false;
    ASSERT_NO_FATAL_FAILURE(submit("cancel"));
    app().jobs().cancel();
    ASSERT_TRUE(pump_until([&] { return !app().jobs().busy(); }, 60000));
    EXPECT_EQ(chunk_count(), 3);
    EXPECT_FALSE(model().stale());
}
