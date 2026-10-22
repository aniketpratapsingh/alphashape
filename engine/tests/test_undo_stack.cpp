// Tests for the undo stack and the one concrete command it carries.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/commands/command.hpp"
#include "common/commands/command_json.hpp"
#include "common/core/document.hpp"

namespace alphashape {
namespace {

/// A command that mutates nothing useful and can be made to fail on revert.
class ThrowingRevertCommand : public AbstractCommand {
public:
    std::string name() const override { return "Throwing Revert"; }
    void apply(Document& /*document*/) override { ++applies; }
    void revert(Document& /*document*/) override { throw Error("revert failed"); }

    int applies = 0;
};

/// Counters shared with the test: the probe is owned and destroyed by the
/// stack when execute() throws, so the test cannot read through a raw pointer.
struct ProbeCounters {
    int applies = 0;
    int reverts = 0;
};

/// A command that renames the first chunk and can be told to report itself
/// unbounded, either immediately or only after it has captured its payload
/// (the asset-command shape).
///
/// "apply, revert, apply" must leave exactly the state of the first apply; the
/// captured pre-state is per-application, which is what a real command does.
/// An unbounded command declines to commit, leaving the document alone.
class ProbeCommand : public AbstractCommand {
public:
    ProbeCommand(std::string name, std::shared_ptr<ProbeCounters> counters)
        : name_(std::move(name)), counters_(std::move(counters)) {}

    std::string name() const override { return name_; }

    void apply(Document& document) override {
        ++counters_->applies;
        if (bounded && !unbounded_after_apply) {
            Chunk& chunk = *document.workspace().chunks().front();
            captured.push_back(chunk.label());
            chunk.set_label(name_);
            return;
        }
        if (unbounded_after_apply) {
            bounded = false;   // captured, but cannot keep or spill it
        }
    }

    void revert(Document& document) override {
        ++counters_->reverts;
        if (captured.empty()) {
            return;
        }
        document.workspace().chunks().front()->set_label(captured.back());
        captured.pop_back();
    }

    bool is_memory_bounded() const override { return bounded; }

    bool bounded = true;
    bool unbounded_after_apply = false;
    std::vector<std::string> captured;

private:
    std::string name_;
    std::shared_ptr<ProbeCounters> counters_;
};

/// Scratch directory under the system temp dir, removed on destruction.
class TempDir {
public:
    explicit TempDir(std::string name) {
        std::error_code ec;
        path_ = std::filesystem::temp_directory_path(ec) /
                (!ec ? name : std::string("alphashape-undo"));
        std::filesystem::remove_all(path_, ec);
        std::filesystem::create_directories(path_, ec);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

std::int32_t add_chunk(Document& document, std::string label) {
    Chunk& chunk = document.workspace().add_chunk(std::move(label));
    document.set_active_chunk_id(chunk.id());
    return chunk.id();
}

/// The label of the sole chunk in the document.
const std::string& chunk_label(const Document& document) {
    return document.workspace().chunks().front()->label();
}

TEST(UndoStack, ExecuteUndoRedoRestoreExactLabel) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    stack.execute(make_set_chunk_label_command(id, document, "renamed"));
    EXPECT_EQ(chunk_label(document), "renamed");
    EXPECT_EQ(stack.position(), 1U);
    EXPECT_TRUE(stack.can_undo());

    stack.undo();
    EXPECT_EQ(chunk_label(document), "original");
    EXPECT_EQ(stack.position(), 0U);
    EXPECT_TRUE(stack.can_redo());

    stack.redo();
    EXPECT_EQ(chunk_label(document), "renamed");
    EXPECT_EQ(stack.position(), 1U);
}

TEST(UndoStack, UndoLabelUsesCommandName) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);
    stack.execute(make_set_chunk_label_command(id, document, "renamed"));
    EXPECT_EQ(stack.undo_label(), "Undo Set Chunk Label");
    EXPECT_EQ(stack.undo_name(), "Set Chunk Label");
    stack.undo();
    EXPECT_EQ(stack.redo_label(), "Redo Set Chunk Label");
    EXPECT_TRUE(stack.undo_label().empty());
}

TEST(UndoStack, ExecuteAfterUndoDropsRedoBranch) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    stack.execute(make_set_chunk_label_command(id, document, "first"));
    stack.undo();
    ASSERT_TRUE(stack.can_redo());

    // Executing a new command discards the redo branch.
    stack.execute(make_set_chunk_label_command(id, document, "second"));
    EXPECT_FALSE(stack.can_redo());
    EXPECT_EQ(stack.size(), 1U);
    EXPECT_EQ(chunk_label(document), "second");
}

TEST(UndoStack, UndoWithNothingThrows) {
    Document document;
    UndoStack stack(&document);
    EXPECT_THROW(stack.undo(), Error);
}

TEST(UndoStack, UnboundStackThrowsOnExecute) {
    Document document;
    add_chunk(document, "original");
    UndoStack stack;  // not bound
    EXPECT_THROW(stack.execute(make_set_chunk_label_command(0, document, "x")), Error);
}

TEST(UndoStack, UndoRestoresLabelCapturedAtExecuteTimeNotConstructionTime) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    // Construct, then change the label by something else, then execute.
    auto command = make_set_chunk_label_command(id, document, "renamed");
    document.workspace().chunks().front()->set_label("interloper");
    stack.execute(std::move(command));
    EXPECT_EQ(chunk_label(document), "renamed");

    // Undo must restore "interloper" (the value at apply time), not "original".
    stack.undo();
    EXPECT_EQ(chunk_label(document), "interloper");
    EXPECT_EQ(stack.undo_label(), "");
}

TEST(UndoStack, ThrowingRevertLeavesPositionUnchanged) {
    Document document;
    UndoStack stack(&document);
    stack.execute(std::make_unique<ThrowingRevertCommand>());
    ASSERT_EQ(stack.position(), 1U);

    EXPECT_THROW(stack.undo(), Error);
    EXPECT_EQ(stack.position(), 1U);  // revert threw, so nothing was popped
}

TEST(UndoStack, NullCommandThrows) {
    Document document;
    UndoStack stack(&document);
    EXPECT_THROW(stack.execute(nullptr), Error);
}

// --- WP-11.5a: transactions, macros, depth cap, memory policy ---------------

TEST(Undo, TransactionGroupsAsOne) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    auto transaction = std::make_unique<CommandTransaction>("Rename Twice");
    transaction->add(make_set_chunk_label_command(id, document, "first"));
    transaction->add(make_set_chunk_label_command(id, document, "second"));
    stack.execute(std::move(transaction));

    EXPECT_EQ(chunk_label(document), "second");
    EXPECT_EQ(stack.size(), 1U);          // one entry, not two
    EXPECT_EQ(stack.undo_label(), "Undo Rename Twice");

    stack.undo();
    EXPECT_EQ(chunk_label(document), "original");

    stack.redo();
    EXPECT_EQ(chunk_label(document), "second");
}

TEST(Undo, TransactionRevertsInReverseOrder) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    auto transaction = std::make_unique<CommandTransaction>("Rename Thrice");
    transaction->add(make_set_chunk_label_command(id, document, "a"));
    transaction->add(make_set_chunk_label_command(id, document, "b"));
    transaction->add(make_set_chunk_label_command(id, document, "c"));
    stack.execute(std::move(transaction));
    ASSERT_EQ(chunk_label(document), "c");

    // Reverse order is what makes this land back on "original": each child
    // restores the label its own apply() saw.
    stack.undo();
    EXPECT_EQ(chunk_label(document), "original");
}

TEST(Undo, TransactionRollsBackOnPartialFailure) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    auto transaction = std::make_unique<CommandTransaction>("Half Bad");
    transaction->add(make_set_chunk_label_command(id, document, "applied"));
    transaction->add(make_set_chunk_label_command(9999, document, "missing"));
    EXPECT_THROW(stack.execute(std::move(transaction)), ValueError);

    // The first child ran and was undone; nothing was pushed.
    EXPECT_EQ(chunk_label(document), "original");
    EXPECT_EQ(stack.size(), 0U);
    EXPECT_FALSE(stack.can_undo());
}

TEST(Undo, MacroNested) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    stack.begin_macro("Outer");
    stack.execute(make_set_chunk_label_command(id, document, "one"));
    stack.begin_macro("Inner");
    stack.execute(make_set_chunk_label_command(id, document, "two"));
    stack.execute(make_set_chunk_label_command(id, document, "three"));
    EXPECT_EQ(stack.macro_depth(), 2U);
    EXPECT_EQ(stack.size(), 0U);   // nothing pushed while open
    stack.end_macro();
    stack.end_macro();

    EXPECT_EQ(stack.macro_depth(), 0U);
    EXPECT_EQ(stack.size(), 1U);              // the outer group is one entry
    EXPECT_EQ(stack.undo_label(), "Undo Outer");

    // One undo reverses all three edits, through the nested group.
    stack.undo();
    EXPECT_EQ(chunk_label(document), "original");
    stack.redo();
    EXPECT_EQ(chunk_label(document), "three");
}

TEST(Undo, MacroRefusesUndoWhileOpen) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);
    stack.execute(make_set_chunk_label_command(id, document, "before"));

    stack.begin_macro("Open");
    stack.execute(make_set_chunk_label_command(id, document, "inside"));
    EXPECT_THROW(stack.undo(), Error);
    stack.end_macro();

    EXPECT_TRUE(stack.can_undo());
    stack.undo();  // the macro entry
    stack.undo();  // the command from before
    EXPECT_EQ(chunk_label(document), "original");
}

TEST(Undo, MacroEmptyGroupPushesNothing) {
    Document document;
    UndoStack stack(&document);
    stack.begin_macro("Nothing");
    stack.end_macro();
    EXPECT_FALSE(stack.can_undo());
    EXPECT_EQ(stack.size(), 0U);
}

TEST(Undo, EndMacroWithoutBeginThrows) {
    Document document;
    UndoStack stack(&document);
    EXPECT_THROW(stack.end_macro(), ValueError);
}

TEST(Undo, DestructorClosesOpenMacroSoAppliedEditsAreUndoableNowhere) {
    // A macro left open by a caller error must not leave an applied edit off
    // the stack; the destructor closes it. There is nothing to assert on the
    // dead stack, so assert the edit is still applied and the stack survived
    // scope without throwing.
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    {
        UndoStack stack(&document);
        stack.begin_macro("Forgotten");
        stack.execute(make_set_chunk_label_command(id, document, "changed"));
    }
    EXPECT_EQ(chunk_label(document), "changed");
}

TEST(Undo, ClearForgetsAnOpenMacro) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);
    stack.begin_macro("Macro");
    stack.execute(make_set_chunk_label_command(id, document, "applied"));
    stack.clear();

    EXPECT_FALSE(stack.macro_open());
    EXPECT_EQ(stack.size(), 0U);
    EXPECT_EQ(chunk_label(document), "applied");   // the edit stays applied
    EXPECT_THROW(stack.end_macro(), ValueError);   // the group is gone
}

TEST(Undo, FailedApplyLeavesDocumentUntouched) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    // No chunk 4242: validate() throws before anything is staged or written.
    EXPECT_THROW(
        stack.execute(make_set_chunk_label_command(4242, document, "nope")),
        ValueError);

    EXPECT_EQ(chunk_label(document), "original");
    EXPECT_EQ(stack.size(), 0U);
    EXPECT_FALSE(stack.can_undo());
    EXPECT_TRUE(document.workspace().find_chunk(id) != nullptr);
}

TEST(Undo, DepthCapDropsOldest) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    const std::size_t total = kMaxUndoDepth + 5;
    for (std::size_t i = 0; i < total; ++i) {
        stack.execute(make_set_chunk_label_command(id, document, "label " + std::to_string(i)));
    }

    EXPECT_EQ(stack.size(), kMaxUndoDepth);
    EXPECT_EQ(stack.position(), kMaxUndoDepth);
    EXPECT_EQ(stack.dropped_count(), 5U);
    EXPECT_TRUE(stack.history_truncated());

    // The oldest four edits were dropped: undoing everything reaches the
    // state left by command 4, not "original".
    while (stack.can_undo()) {
        stack.undo();
    }
    EXPECT_EQ(chunk_label(document), "label 4");
    EXPECT_TRUE(stack.can_redo());
}

TEST(Undo, UndoLabelReportsTruncationAtOldestStep) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    for (std::size_t i = 0; i < kMaxUndoDepth + 1; ++i) {
        stack.execute(make_set_chunk_label_command(id, document, "l" + std::to_string(i)));
    }
    ASSERT_EQ(stack.dropped_count(), 1U);

    // Other steps read normally; only the oldest one advertises truncation.
    EXPECT_EQ(stack.undo_label(), "Undo Set Chunk Label");
    while (stack.position() > 1) {
        stack.undo();
    }
    EXPECT_EQ(stack.undo_label(), "Undo Set Chunk Label (older steps dropped)");
}

TEST(Undo, RedoBranchClearedOnNewCommand) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    stack.execute(make_set_chunk_label_command(id, document, "one"));
    stack.execute(make_set_chunk_label_command(id, document, "two"));
    stack.undo();
    ASSERT_TRUE(stack.can_redo());

    stack.execute(make_set_chunk_label_command(id, document, "three"));
    EXPECT_FALSE(stack.can_redo());
    EXPECT_EQ(stack.size(), 2U);
    EXPECT_EQ(stack.position(), 2U);
}

TEST(Undo, RefusesOversizedCommandBeforeApply) {
    Document document;
    add_chunk(document, "original");
    UndoStack stack(&document);

    auto counters = std::make_shared<ProbeCounters>();
    auto probe = std::make_unique<ProbeCommand>("Huge", counters);
    probe->bounded = false;   // honours the "declines to commit" contract

    try {
        stack.execute(std::move(probe));
        FAIL() << "expected the oversized command to be refused";
    } catch (const Error& error) {
        EXPECT_STREQ(error.what(), "command too large for the undo history");
    }
    EXPECT_EQ(counters->applies, 0);   // refused before apply
    EXPECT_EQ(chunk_label(document), "original");
    EXPECT_EQ(stack.size(), 0U);
}

TEST(Undo, RefusesCommandThatBecomesOversizedWhileApplying) {
    Document document;
    add_chunk(document, "original");
    UndoStack stack(&document);

    // Reports bounded, then declines to commit: the asset command shape.
    auto counters = std::make_shared<ProbeCounters>();
    auto probe = std::make_unique<ProbeCommand>("Late Huge", counters);
    probe->unbounded_after_apply = true;

    EXPECT_THROW(stack.execute(std::move(probe)), Error);
    EXPECT_EQ(counters->applies, 1);
    EXPECT_EQ(counters->reverts, 0);   // execute never reverts a declined apply
    EXPECT_EQ(chunk_label(document), "original");
    EXPECT_EQ(stack.size(), 0U);
}

TEST(Undo, ReapplyAfterRevertEqualsFirstApply) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);

    stack.execute(make_set_chunk_label_command(id, document, "renamed"));
    const std::string after_first = chunk_label(document);

    // Undo/redo repeatedly: each cycle must reproduce the same state, and the
    // pre-state captured by each apply must be the same as the first time.
    for (int cycle = 0; cycle < 3; ++cycle) {
        stack.undo();
        ASSERT_EQ(chunk_label(document), "original") << "cycle " << cycle;
        stack.redo();
        ASSERT_EQ(chunk_label(document), after_first) << "cycle " << cycle;
    }
    EXPECT_EQ(stack.position(), 1U);
}

TEST(Undo, SpillRoundTrip) {
    TempDir dir("alphashape-undo-spill");
    Document document;
    add_chunk(document, "original");

    // 70 MiB: above the 64 MiB cap, so capture must spill rather than hold.
    const int width = 5000;
    const int height = 5000;   // 25M pixels * 3 = 75 MB
    Image image(width, height, 3);
    for (std::size_t i = 0; i < image.size_bytes(); i += 4093) {
        image.data()[i] = static_cast<std::uint8_t>(i % 251);
    }
    ASSERT_GT(image.size_bytes(), kUndoAssetSpillBytes);

    SpillBuffer buffer = SpillBuffer::capture(image, dir.path());
    ASSERT_TRUE(buffer.valid());
    EXPECT_TRUE(buffer.spilled());
    EXPECT_EQ(buffer.bytes(), image.size_bytes());
    EXPECT_TRUE(std::filesystem::exists(buffer.spill_path()));

    const Image restored = buffer.restore();
    ASSERT_EQ(restored.width(), image.width());
    ASSERT_EQ(restored.height(), image.height());
    ASSERT_EQ(restored.channels(), image.channels());
    EXPECT_EQ(std::memcmp(restored.data(), image.data(), image.size_bytes()), 0);

    const std::string spill_path = buffer.spill_path();
    buffer.release();
    EXPECT_FALSE(std::filesystem::exists(spill_path));
}

TEST(Undo, SmallPayloadStaysInMemory) {
    TempDir dir("alphashape-undo-memory");
    Image image(64, 64, 3);
    SpillBuffer buffer = SpillBuffer::capture(image, dir.path());
    ASSERT_TRUE(buffer.valid());
    EXPECT_FALSE(buffer.spilled());
    EXPECT_EQ(buffer.restore().width(), 64);
}

TEST(Undo, OversizedPayloadWithoutCacheDirIsUnbounded) {
    Image image(5000, 5000, 3);   // 75 MB, above the cap
    SpillBuffer buffer = SpillBuffer::capture(image, std::string());
    EXPECT_FALSE(buffer.valid());
    EXPECT_FALSE(buffer.spilled());
}

TEST(Undo, SpillFileIsRemovedWhenBufferDies) {
    std::string spill_path;
    {
        TempDir dir("alphashape-undo-leak");
        Image big(6000, 6000, 3);     // 108 MB, above the cap
        ASSERT_GT(big.size_bytes(), kUndoAssetSpillBytes);
        SpillBuffer buffer = SpillBuffer::capture(std::move(big), dir.path());
        ASSERT_TRUE(buffer.spilled());
        spill_path = buffer.spill_path();
        ASSERT_TRUE(std::filesystem::exists(spill_path));
    }   // buffer destroyed with the TempDir still alive
    EXPECT_FALSE(std::filesystem::exists(spill_path));
}

TEST(Undo, DebugDumpShowsPosition) {
    Document document;
    const std::int32_t id = add_chunk(document, "original");
    UndoStack stack(&document);
    stack.execute(make_set_chunk_label_command(id, document, "one"));
    stack.execute(make_set_chunk_label_command(id, document, "two"));
    stack.undo();

    const std::string dump = dump_undo_stack(stack);
    EXPECT_NE(dump.find("undo stack (1/2, dropped 0)"), std::string::npos) << dump;
    EXPECT_NE(dump.find("  [0] Set Chunk Label"), std::string::npos) << dump;
    EXPECT_NE(dump.find("[1] * Set Chunk Label"), std::string::npos) << dump;
}

}  // namespace
}  // namespace alphashape
