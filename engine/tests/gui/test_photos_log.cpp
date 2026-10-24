// PhotosPane.*, LogPane.* and Undo.*: the Photos dock (WP-14.10, Phase 1 trim), the Log dock
// (P1-LOG) and the undo/redo integration (WP-14.16, Phase 1 trim), driven offscreen through the
// harness.
//
// The Photos pane tests build engine documents with cameras and frames. For the thumbnail test
// each frame points at a real, small JPEG copied into the test's scratch directory (the
// committed baseline fixture), because the thumbnail pipeline decodes through the engine's
// image IO and must really run off the GUI thread.

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <QAbstractItemModelTester>
#include <QAction>
#include <QFile>
#include <QImage>
#include <QItemSelectionModel>
#include <QKeySequence>
#include <QLineEdit>
#include <QListView>
#include <QMenu>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QString>
#include <QToolButton>
#include <QVariant>

#include "apps/gui/app.hpp"
#include "apps/gui/log_pane.hpp"
#include "apps/gui/photos_model.hpp"
#include "apps/gui/photos_pane.hpp"
#include "apps/gui/workspace_model.hpp"
#include "apps/gui/workspace_pane.hpp"
#include "base/utils/log.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "harness.hpp"

using namespace alphashape;
using namespace alphashape::gui;
using alphashape::gui::testing::GuiTest;
using alphashape::gui::testing::pump;
using alphashape::gui::testing::pump_until;

namespace {

/// What the document builder made, so a test can address the cameras and check flags.
struct BuiltChunk {
    std::int32_t id = -1;
    std::vector<std::int32_t> camera_ids;
};

/// Everything the three suites share: a scratch directory with real JPEGs, a chunk builder and
/// the widget accessors.
class PhotosLogUndoBase : public GuiTest {
protected:
    PhotosPane& photos() { return *widget<PhotosPane>("pane.photos"); }
    PhotosModel& model() { return photos().model(); }
    QListView* view() { return photos().view(); }
    QLineEdit* filter() { return photos().filter_edit(); }
    LogPane& log_pane() { return *widget<LogPane>("pane.log"); }
    WorkspaceSelection& selection() {
        return *window().findChild<WorkspaceSelection*>("selection.workspace");
    }
    WorkspacePane& tree() { return *widget<WorkspacePane>("pane.workspace"); }
    Chunk& chunk_ref(std::int32_t id) { return *app().document().workspace().find_chunk(id); }

    /// Copies the committed baseline JPEG into the scratch directory under `name`.
    QString make_jpeg(const QString& name) {
        const QString target = scratch(name);
        if (QFile::copy(QStringLiteral(ALPHASHAPE_GUI_DATA_DIR) + QStringLiteral("/baseline.jpg"),
                        target)) {
            return target;
        }
        ADD_FAILURE() << "could not copy the baseline fixture to " << name.toStdString();
        return target;
    }

    /// A chunk with `cameras` cameras and one frame each. Even camera ids are aligned (a
    /// non-identity pose), so aligned and unaligned rows both exist. With `with_files` the frame
    /// points at a real JPEG; otherwise at a path that does not exist.
    BuiltChunk build_chunk(int cameras, bool with_files, const char* prefix = "IMG_") {
        Document& document = app().document();
        Chunk& chunk = document.workspace().add_chunk("Block A");
        for (int i = 0; i < cameras; ++i) {
            Camera camera;
            camera.id = i;
            camera.sensor_id = 0;
            camera.label = std::string(prefix) + std::to_string(i);
            camera.enabled = true;
            Transform pose = Transform::Identity();
            if (i % 2 == 0) {
                pose(0, 3) = 1.0 + static_cast<double>(i);
            }
            camera.pose = pose;
            chunk.cameras().push_back(camera);

            const QString path =
                with_files
                    ? make_jpeg(QStringLiteral("photo_%1.jpg")
                                    .arg(QString::fromStdString(camera.label)))
                    : scratch(QStringLiteral("missing_%1.jpg").arg(i));
            auto frame = std::make_shared<Frame>(path.toStdString(), 64, 48);
            frame->set_id(i);
            frame->set_camera_id(i);
            chunk.frames().push_back(frame);
        }
        document.set_active_chunk_id(chunk.id());
        BuiltChunk built;
        built.id = chunk.id();
        for (const Camera& camera : chunk.cameras()) {
            built.camera_ids.push_back(camera.id);
        }
        // The chunk was inserted behind the models' backs (the pane paths were not used), so
        // both the Photos list and the Workspace tree have to re-read the document.
        model().refresh();
        tree().model().refresh();
        return built;
    }

    std::vector<bool> enabled_flags(const BuiltChunk& built) {
        std::vector<bool> flags;
        for (const Camera& camera : chunk_ref(built.id).cameras()) {
            flags.push_back(camera.enabled);
        }
        return flags;
    }

    void select_rows(const std::vector<int>& rows) {
        QItemSelectionModel* selection_model = view()->selectionModel();
        selection_model->clear();
        for (const int row : rows) {
            selection_model->select(model().index(row, 0), QItemSelectionModel::Select);
        }
    }
};

}  // namespace

// --- Photos pane (WP-14.10) ----------------------------------------------------------------

using PhotosPaneTest = PhotosLogUndoBase;

TEST_F(PhotosPaneTest, ModelHasOneRowPerCameraWithBadges) {
    QAbstractItemModelTester tester(&model(),
                                    QAbstractItemModelTester::FailureReportingMode::Warning);
    const BuiltChunk built = build_chunk(4, /*with_files=*/false);
    ASSERT_EQ(model().rowCount(), 4);
    EXPECT_EQ(model().chunk_id(), built.id);
    EXPECT_EQ(model().camera_ids(), built.camera_ids);

    // Even cameras are aligned, so the odd ones carry the "not aligned" badge.
    EXPECT_EQ(model().index(0, 0).data(Qt::DisplayRole).toString(), QStringLiteral("IMG_0"));
    EXPECT_TRUE(model().index(0, 0).data(PhotosModel::AlignedRole).toBool());
    EXPECT_TRUE(model().index(0, 0).data(PhotosModel::BadgesRole).toStringList().isEmpty());
    EXPECT_TRUE(model().index(1, 0)
                    .data(PhotosModel::BadgesRole)
                    .toStringList()
                    .contains(QStringLiteral("not aligned")));
    EXPECT_TRUE(model().index(0, 0).data(PhotosModel::EnabledRole).toBool());

    // A camera disabled through the document greys out and carries the badge.
    ASSERT_TRUE(photos().set_cameras_enabled({2}, false));
    const QModelIndex disabled = model().index(2, 0);
    EXPECT_FALSE(disabled.data(PhotosModel::EnabledRole).toBool());
    EXPECT_TRUE(disabled.data(PhotosModel::BadgesRole)
                    .toStringList()
                    .contains(QStringLiteral("disabled")));
    EXPECT_TRUE(disabled.data(Qt::ForegroundRole).isValid());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(PhotosPaneTest, FilterIsASubstringMatch) {
    QAbstractItemModelTester tester(&model(),
                                    QAbstractItemModelTester::FailureReportingMode::Warning);
    build_chunk(6, /*with_files=*/false, "DSC_");
    filter()->setText(QStringLiteral("dsc_1"));  // case-insensitive, substring
    ASSERT_EQ(model().rowCount(), 1);
    EXPECT_EQ(model().camera_id_at(0), 1);
    EXPECT_EQ(model().row_for_camera(1), 0);
    EXPECT_EQ(model().row_for_camera(0), -1);

    filter()->setText(QStringLiteral("nothing matches"));
    EXPECT_EQ(model().rowCount(), 0);
    filter()->clear();
    EXPECT_EQ(model().rowCount(), 6);
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(PhotosPaneTest, SelectionMirrorsBothWays) {
    const BuiltChunk built = build_chunk(4, /*with_files=*/false);
    select_rows({1, 2});
    ASSERT_EQ(selection().items().size(), 2);
    EXPECT_EQ(selection().items()[0], (NodeRef{NodeKind::Camera, built.id, 1}));
    EXPECT_EQ(selection().items()[1], (NodeRef{NodeKind::Camera, built.id, 2}));
    EXPECT_EQ(selection().current(), (NodeRef{NodeKind::Camera, built.id, 1}));

    // Selecting a chunk in the tree clears the photo selection.
    ASSERT_TRUE(tree().select(NodeRef{NodeKind::Chunk, built.id, -1}));
    pump(10);
    EXPECT_TRUE(photos().selected_camera_ids().empty());

    // A camera selected in the tree lands in the list.
    ASSERT_TRUE(tree().select(NodeRef{NodeKind::Camera, built.id, 3}));
    pump(10);
    EXPECT_EQ(photos().selected_camera_ids(), (std::vector<std::int32_t>{3}));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(PhotosPaneTest, DelDisablesTheSelectionAsOneUndoEntry) {
    show_window();
    const BuiltChunk built = build_chunk(4, /*with_files=*/false);
    ASSERT_EQ(enabled_flags(built), (std::vector<bool>{true, true, true, true}));

    // Del is the shortcut of the same action the context menu offers, so the key and the menu
    // entry cannot diverge. A real key event needs an active native window, which the offscreen
    // platform does not provide, so the binding and the action's effect are checked separately.
    QAction* disable = window().findChild<QAction*>(QStringLiteral("action.photos.disable"));
    ASSERT_NE(disable, nullptr);
    EXPECT_EQ(disable->shortcut(), QKeySequence(QKeySequence::Delete));

    select_rows({0, 2, 3});
    ASSERT_TRUE(disable->isEnabled());
    disable->trigger();
    pump(10);

    EXPECT_EQ(enabled_flags(built), (std::vector<bool>{false, true, false, false}));
    EXPECT_EQ(app().undo_stack().size(), 1U);
    EXPECT_EQ(action("action.edit.undo")->text(), QStringLiteral("Undo Disable 3 photos"));

    // One undo restores all three in a single step.
    ASSERT_TRUE(trigger("action.edit.undo"));
    EXPECT_EQ(enabled_flags(built), (std::vector<bool>{true, true, true, true}));
    EXPECT_EQ(action("action.edit.redo")->text(), QStringLiteral("Redo Disable 3 photos"));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(PhotosPaneTest, ContextMenuEnablesAndDisablesThroughTheUndoStack) {
    const BuiltChunk built = build_chunk(4, /*with_files=*/false);
    ASSERT_TRUE(photos().set_cameras_enabled({0, 1}, false));
    // The refresh after the edit must not lose the selection: it is what the menu acts on.
    select_rows({0, 1});
    {
        auto menu = photos().build_context_menu();
        QAction* enable = menu->findChild<QAction*>("action.photos.enable");
        ASSERT_NE(enable, nullptr);
        EXPECT_TRUE(enable->isEnabled());
        EXPECT_FALSE(menu->findChild<QAction*>("action.photos.disable")->isEnabled());
        enable->trigger();
    }
    EXPECT_EQ(enabled_flags(built), (std::vector<bool>{true, true, true, true}));
    app().undo();
    EXPECT_EQ(enabled_flags(built), (std::vector<bool>{false, false, true, true}));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(PhotosPaneTest, ThumbnailsAreDecodedOffTheGuiThreadAndAppear) {
    QAbstractItemModelTester tester(&model(),
                                    QAbstractItemModelTester::FailureReportingMode::Warning);
    build_chunk(3, /*with_files=*/true);
    ASSERT_EQ(model().rowCount(), 3);

    const bool ready = pump_until(
        [this]() {
            for (int row = 0; row < model().rowCount(); ++row) {
                if (model().index(row, 0).data(Qt::DecorationRole).isNull()) {
                    return false;
                }
            }
            return true;
        },
        15000);
    ASSERT_TRUE(ready) << "thumbnails did not appear within 15 s";
    EXPECT_TRUE(model().thumbnails().decoded_off_gui_thread());

    // Scaled to fit the 160x120 box, aspect preserved and never upscaled past the box.
    const QPixmap thumbnail = model().index(0, 0).data(Qt::DecorationRole).value<QPixmap>();
    ASSERT_FALSE(thumbnail.isNull());
    EXPECT_LE(thumbnail.width(), ThumbnailCache::kMaxWidth);
    EXPECT_LE(thumbnail.height(), ThumbnailCache::kMaxHeight);
    EXPECT_GT(thumbnail.width(), 0);
    EXPECT_EQ(static_cast<PhotosModel::ThumbnailState>(
                  model().index(0, 0).data(PhotosModel::ThumbnailStateRole).toInt()),
              PhotosModel::ThumbnailState::Ready);
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(PhotosPaneTest, MissingFilesReportNoPreviewInsteadOfWaiting) {
    build_chunk(2, /*with_files=*/false);
    EXPECT_EQ(static_cast<PhotosModel::ThumbnailState>(
                  model().index(0, 0).data(PhotosModel::ThumbnailStateRole).toInt()),
              PhotosModel::ThumbnailState::Missing);
    EXPECT_TRUE(model().index(0, 0).data(Qt::DecorationRole).isNull());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(PhotosPaneTest, ReplacingTheDocumentRebuildsTheList) {
    QAbstractItemModelTester tester(&model(),
                                    QAbstractItemModelTester::FailureReportingMode::Warning);
    build_chunk(3, /*with_files=*/false);
    ASSERT_EQ(model().rowCount(), 3);
    app().new_document("");
    EXPECT_EQ(model().rowCount(), 0);
    EXPECT_EQ(model().chunk_id(), -1);
    EXPECT_TRUE(photos().selected_camera_ids().empty());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

// --- Log pane (P1-LOG) ---------------------------------------------------------------------

using LogPaneTest = PhotosLogUndoBase;

TEST_F(LogPaneTest, WorkerThreadLogLineReachesThePane) {
    // The sink GuiApp installs marshals worker-thread lines onto the GUI thread.
    std::thread worker([]() { alphashape::log(LogLevel::Info) << "worker says hello"; });
    worker.join();
    const bool arrived = pump_until(
        [this]() {
            return log_pane().view()->plain_text().contains(QStringLiteral("worker says hello"));
        },
        5000);
    ASSERT_TRUE(arrived) << "the worker-thread line never reached the Log pane";
    EXPECT_TRUE(log_pane().view()->plain_text().contains(QStringLiteral("worker says hello")));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(LogPaneTest, LevelFilterShowsTheChosenLevelAndAbove) {
    LogView* console = log_pane().view();
    console->clear_entries();
    console->append(LogEntry{LogLevel::Debug, QStringLiteral("debug line")});
    console->append(LogEntry{LogLevel::Info, QStringLiteral("info line")});
    console->append(LogEntry{LogLevel::Warn, QStringLiteral("warn line")});
    console->append(LogEntry{LogLevel::Error, QStringLiteral("error line")});
    EXPECT_EQ(console->line_count(), 4U);
    EXPECT_EQ(console->plain_text().split(QLatin1Char('\n')).size(), 4);

    log_pane().set_minimum_level(LogLevel::Warn);
    const QString warnings_only = console->plain_text();
    EXPECT_TRUE(warnings_only.contains(QStringLiteral("warn line")));
    EXPECT_TRUE(warnings_only.contains(QStringLiteral("error line")));
    EXPECT_FALSE(warnings_only.contains(QStringLiteral("info line")));
    EXPECT_FALSE(warnings_only.contains(QStringLiteral("debug line")));

    // Hidden lines are kept, not discarded: widening the filter brings them back.
    log_pane().set_minimum_level(LogLevel::Trace);
    EXPECT_EQ(console->plain_text().split(QLatin1Char('\n')).size(), 4);
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(LogPaneTest, ClearAndCopyActOnTheVisibleLog) {
    LogView* console = log_pane().view();
    console->clear_entries();
    console->append(LogEntry{LogLevel::Info, QStringLiteral("first line")});
    console->append(LogEntry{LogLevel::Error, QStringLiteral("second line")});
    EXPECT_EQ(console->plain_text(), QStringLiteral("first line\nsecond line"));

    // Copy hands exactly the visible lines to the clipboard writer. The writer is a seam: the
    // offscreen platform has no clipboard, so a real QClipboard write could not be read back.
    QString copied;
    log_pane().set_clipboard_writer([&copied](const QString& text) { copied = text; });
    log_pane().copy_button()->click();
    EXPECT_EQ(copied, QStringLiteral("first line\nsecond line"));
    EXPECT_EQ(log_pane().copy_all(), copied);

    // A level filter narrows what Copy yields.
    log_pane().set_minimum_level(LogLevel::Error);
    EXPECT_EQ(log_pane().copy_all(), QStringLiteral("second line"));
    log_pane().set_minimum_level(LogLevel::Trace);

    log_pane().clear_button()->click();
    EXPECT_EQ(console->line_count(), 0U);
    EXPECT_TRUE(console->plain_text().isEmpty());
    EXPECT_TRUE(console->toPlainText().isEmpty());
    EXPECT_TRUE(log_pane().copy_all().isEmpty());
    EXPECT_FALSE(log_pane().copy_button()->isEnabled());
    EXPECT_FALSE(log_pane().clear_button()->isEnabled());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(LogPaneTest, TheLogKeepsAtMostTheLineCap) {
    LogView* console = log_pane().view();
    console->clear_entries();
    const std::size_t extra = 5;
    for (std::size_t i = 0; i < LogView::kMaxLines + extra; ++i) {
        console->append(LogEntry{LogLevel::Info, QStringLiteral("line %1").arg(static_cast<qulonglong>(i))});
    }
    EXPECT_EQ(console->line_count(), LogView::kMaxLines);
    const std::deque<LogEntry>& entries = console->entries();
    ASSERT_FALSE(entries.empty());
    // The oldest lines were dropped, the newest kept.
    EXPECT_EQ(entries.front().text, QStringLiteral("line %1").arg(static_cast<qulonglong>(extra)));
    EXPECT_EQ(entries.back().text,
              QStringLiteral("line %1").arg(static_cast<qulonglong>(LogView::kMaxLines + extra - 1)));
    // The widget document is bounded the same way, so memory cannot outgrow the cap.
    EXPECT_LE(console->document()->blockCount(), static_cast<int>(LogView::kMaxLines) + 1);
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

// --- Undo / redo integration (WP-14.16) ----------------------------------------------------

/// The suite name is the feature: the tests read `Undo.GuiEditsRoundTrip` and friends.
class Undo : public PhotosLogUndoBase {};

TEST_F(Undo, GuiEditsRoundTrip) {
    show_window();
    const BuiltChunk built = build_chunk(4, /*with_files=*/false);
    const Chunk before = chunk_ref(built.id);  // deep copy of the state to come back to

    // Several mixed edits through the pane: a multi-selection, a single toggle, an undo/redo
    // pair in the middle, then toggles back.
    select_rows({0, 1, 2});
    ASSERT_TRUE(photos().disable_selected());
    ASSERT_TRUE(photos().set_cameras_enabled({3}, false));
    app().undo();
    app().redo();
    select_rows({1});
    ASSERT_TRUE(photos().enable_selected());
    ASSERT_TRUE(photos().set_cameras_enabled({0, 2}, true));

    // Undo everything: the document comes back to exactly the state it started in.
    const std::size_t edits = app().undo_stack().position();
    ASSERT_GT(edits, 0U);
    for (std::size_t i = 0; i < edits; ++i) {
        app().undo();
    }
    EXPECT_FALSE(app().undo_stack().can_undo());
    EXPECT_TRUE(chunk_ref(built.id) == before);
    EXPECT_FALSE(app().dirty());

    // Redoing everything reaches the edited state again.
    for (std::size_t i = 0; i < edits; ++i) {
        app().redo();
    }
    EXPECT_EQ(enabled_flags(built), (std::vector<bool>{true, true, true, false}));
    EXPECT_TRUE(app().dirty());
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(Undo, LabelsFollowCommandNames) {
    show_window();
    const BuiltChunk built = build_chunk(4, /*with_files=*/false);
    QAction* undo = action("action.edit.undo");
    QAction* redo = action("action.edit.redo");
    EXPECT_FALSE(undo->isEnabled());
    EXPECT_FALSE(redo->isEnabled());

    ASSERT_TRUE(photos().set_cameras_enabled({0}, false));
    EXPECT_TRUE(undo->isEnabled());
    EXPECT_EQ(undo->text(), QStringLiteral("Undo Disable photo"));

    select_rows({1, 2, 3});
    ASSERT_TRUE(photos().disable_selected());
    EXPECT_EQ(undo->text(), QStringLiteral("Undo Disable 3 photos"));

    select_rows({1});
    ASSERT_TRUE(photos().enable_selected());
    EXPECT_EQ(undo->text(), QStringLiteral("Undo Enable photo"));

    // After undoing, the Redo label names the command being redone.
    ASSERT_TRUE(trigger("action.edit.undo"));
    EXPECT_EQ(redo->text(), QStringLiteral("Redo Enable photo"));
    ASSERT_TRUE(trigger("action.edit.undo"));
    EXPECT_EQ(redo->text(), QStringLiteral("Redo Disable 3 photos"));
    EXPECT_EQ(enabled_flags(built), (std::vector<bool>{false, true, true, true}));

    // Replacing the document clears the history and the actions follow it.
    app().new_document("");
    EXPECT_FALSE(undo->isEnabled());
    EXPECT_FALSE(redo->isEnabled());
    EXPECT_EQ(undo->text(), QStringLiteral("Undo"));
    EXPECT_EQ(redo->text(), QStringLiteral("Redo"));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}

TEST_F(Undo, MacroIsOneEntry) {
    QAbstractItemModelTester tester(&model(),
                                    QAbstractItemModelTester::FailureReportingMode::Warning);
    const BuiltChunk built = build_chunk(4, /*with_files=*/false);
    EXPECT_EQ(app().undo_stack().size(), 0U);
    EXPECT_EQ(kMaxUndoDepth, 100U);  // the depth the GUI relies on

    select_rows({0, 1, 2});
    ASSERT_TRUE(photos().disable_selected());
    EXPECT_EQ(app().undo_stack().size(), 1U);  // three cameras changed, one history entry
    EXPECT_EQ(app().undo_stack().undo_name(), "Disable 3 photos");
    EXPECT_FALSE(app().undo_stack().macro_open());

    // Undo reverts the whole group; redo re-applies it.
    app().undo();
    EXPECT_EQ(enabled_flags(built), (std::vector<bool>{true, true, true, true}));
    EXPECT_EQ(app().undo_stack().size(), 1U);
    app().redo();
    EXPECT_EQ(enabled_flags(built), (std::vector<bool>{false, false, false, true}));
    EXPECT_TRUE(warnings().empty()) << warnings().text();
}
