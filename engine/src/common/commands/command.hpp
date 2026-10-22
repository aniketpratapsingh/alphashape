#pragma once

// The command layer: undoable document mutations.
//
// Distinct from the task layer:
//
//   Command  an undoable document mutation, driven by a user action.
//            Has apply() and revert(). Named so the UI can show "Undo <name>".
//   Task     long-running computation, possibly distributed, not itself undoable.
//
// They are bridged: a task that must be undoable emits a reversal command, and
// command handlers wrap top-level commands for the GUI. We keep the split, because it is
// what keeps the undo stack meaningful: only cheap, bounded document edits sit
// on it, never a half-hour rebuild.
//
// Memory policy (M11 §4.7): a reverted command that replaced an asset keeps the
// replaced payload. Payloads above kUndoAssetSpillBytes spill to the cache
// directory on apply and are reloaded on revert. A command that can neither
// keep nor spill its payload reports is_memory_bounded() == false, and
// UndoStack::execute refuses it before touching the document.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "common/core/document.hpp"
#include "image/image.hpp"

namespace alphashape {

/// Payload size above which a command spills the replaced asset to disk.
inline constexpr std::size_t kUndoAssetSpillBytes = 64u * 1024u * 1024u;

/// Hard cap on entries the undo stack keeps. Older commands are dropped.
inline constexpr std::size_t kMaxUndoDepth = 100;

/// Per-execution context handed to a command just before it runs. Asset
/// commands need `cache_dir` to decide whether a payload can spill.
struct UndoContext {
    /// Directory for spilled undo payloads (`<base>.files/cache`, §4.9).
    /// Empty when the document has no on-disk location yet.
    std::string cache_dir;
};

/// One undoable edit.
class AbstractCommand {
public:
    virtual ~AbstractCommand() = default;

    /// Human-readable name, used to build "Undo <name>" / "Redo <name>".
    virtual std::string name() const = 0;

    /// Perform the edit. Called once when executed, and again on redo.
    virtual void apply(Document& document) = 0;

    /// Undo the edit, restoring the document to its pre-apply state.
    virtual void revert(Document& document) = 0;

    /// Called by UndoStack before apply(), so a command can consult the
    /// cache directory. Default: ignore the context.
    virtual void on_prepare(const UndoContext& /*context*/) {}

    /// False when the command's captured payload is too large for both the
    /// in-memory budget and the cache directory. Consulted by
    /// UndoStack::execute before apply (a command that already knows) and
    /// again after apply (the usual case: the replaced asset is read from the
    /// document while applying).
    ///
    /// Contract for the post-apply case: a command that reports false there
    /// must have left the document in its pre-apply state — it declines to
    /// commit rather than committing and asking for a rollback. execute() then
    /// throws without touching the document, and the caller re-runs the work
    /// instead of expecting it to be undoable (§4.7, "pre-decided").
    virtual bool is_memory_bounded() const { return true; }

    /// Approximate bytes this command retains between apply and revert.
    virtual std::size_t retained_bytes() const { return 0; }
};

/// Base for commands that mutate through a staged snapshot.
///
/// apply() is validate -> stage -> commit, with the staged pre-state restored
/// if commit() throws; revert() re-applies the staged pre-state. Subclasses
/// therefore keep the "a failing apply leaves the document untouched"
/// invariant (WP-11.5a) without each one re-implementing the rollback.
class StagedMutation : public AbstractCommand {
public:
    void apply(Document& document) final {
        validate(document);
        stage(document);
        try {
            commit(document);
        } catch (...) {
            restore(document);
            throw;
        }
    }

    void revert(Document& document) final { restore(document); }

protected:
    /// Check the target exists and the edit is legal. Must throw ValueError
    /// and must not mutate.
    virtual void validate(Document& document) = 0;
    /// Capture the pre-state. Must not mutate.
    virtual void stage(Document& document) = 0;
    /// Perform the edit, using the captured state.
    virtual void commit(Document& document) = 0;
    /// Put the staged pre-state back.
    virtual void restore(Document& document) = 0;
};

/// A group of commands applied, undone and reverted as one entry.
///
/// This is both the explicit composite (a caller builds one and executes it)
/// and the representation of a macro (UndoStack::begin_macro/end_macro), where
/// the children were already applied one by one before the group was closed.
class CommandTransaction : public AbstractCommand {
public:
    explicit CommandTransaction(std::string name = "Transaction");

    /// Take ownership of `command`. Null is rejected.
    void add(std::unique_ptr<AbstractCommand> command);

    std::string name() const override { return name_; }
    void apply(Document& document) override;
    void revert(Document& document) override;
    void on_prepare(const UndoContext& context) override;
    bool is_memory_bounded() const override;
    std::size_t retained_bytes() const override;

    bool empty() const { return commands_.empty(); }
    std::size_t size() const { return commands_.size(); }

private:
    std::string name_;
    std::vector<std::unique_ptr<AbstractCommand>> commands_;
};

/// The in-memory-or-on-disk holder for one replaced asset payload.
///
/// The §4.7 policy in one reusable place: capture() keeps an Image in memory
/// when it is small enough (or no cache directory is known) and spills it to a
/// uniquely named file in the cache directory otherwise. restore() copies it
/// back. A payload that is too large and cannot spill yields valid() == false
/// so the owning command can report itself unbounded. The spill file is
/// removed on destruction.
class SpillBuffer {
public:
    SpillBuffer() = default;
    SpillBuffer(SpillBuffer&& other) noexcept;
    SpillBuffer& operator=(SpillBuffer&& other) noexcept;
    SpillBuffer(const SpillBuffer&) = delete;
    SpillBuffer& operator=(const SpillBuffer&) = delete;
    ~SpillBuffer();

    /// Capture `image`, spilling it if it exceeds kUndoAssetSpillBytes.
    static SpillBuffer capture(Image image, const std::string& cache_dir);

    /// False only when the payload is above the cap and no cache directory
    /// was supplied. A failed capture has no payload to restore.
    bool valid() const { return valid_; }
    bool spilled() const { return !spill_path_.empty(); }
    std::size_t bytes() const { return bytes_; }
    const std::string& spill_path() const { return spill_path_; }

    /// Copy the held payload back out. Throws Error if the spill file is gone.
    Image restore() const;

    /// Drop the payload and delete the spill file now.
    void release();

private:
    Image image_;
    std::string spill_path_;
    std::size_t bytes_ = 0;
    bool valid_ = true;
};

/// A bounded stack of commands with the usual execute/undo/redo semantics.
///
/// Convention: menu entries read "Undo %1" / "Redo %1" with the
/// command's name substituted, so the stack is expected to expose the names of
/// the commands it can undo and redo.
///
/// Length: entries beyond kMaxUndoDepth are dropped oldest-first;
/// `dropped_count()` reports how many, and undo_label() appends
/// " (older steps dropped)" at the oldest retained entry so the UI can say the
/// history does not reach the beginning of the session.
///
/// Macros: begin_macro() opens a group; commands executed while it is open are
/// applied immediately and collected, and end_macro() pushes them as a single
/// CommandTransaction entry. Groups nest. undo()/redo() are refused while a
/// macro is open. An unclosed macro is closed by the destructor, innermost
/// frame first, so no applied edit is ever left off the stack.
class UndoStack {
public:
    explicit UndoStack(Document* document = nullptr);
    ~UndoStack();

    void bind(Document* document) { document_ = document; }

    /// Directory handed to commands for spilling (see UndoContext).
    void set_cache_dir(std::string cache_dir) { cache_dir_ = std::move(cache_dir); }
    const std::string& cache_dir() const { return cache_dir_; }

    /// Execute a command and push it. Clears the redo branch. Throws Error
    /// without touching the document when the command is unbounded.
    void execute(std::unique_ptr<AbstractCommand> command);

    /// Open a named group. Commands executed until the matching end_macro()
    /// form one undo entry called `name`.
    void begin_macro(std::string name);
    /// Close the innermost group. Inside another group the group is nested;
    /// at the outermost level it is pushed as one entry. Throws Error when no
    /// group is open.
    void end_macro();

    bool can_undo() const { return position_ > 0; }
    bool can_redo() const { return position_ < commands_.size(); }
    bool macro_open() const { return !macro_frames_.empty(); }
    std::size_t macro_depth() const { return macro_frames_.size(); }

    /// Name of the command that would be undone, or empty.
    std::string undo_name() const;
    /// Name of the command that would be redone, or empty.
    std::string redo_name() const;

    /// The menu label, e.g. "Undo Detect Features". Empty if nothing to undo.
    std::string undo_label() const;
    std::string redo_label() const;

    /// Names of all retained entries, oldest first (debug and UI).
    std::vector<std::string> command_names() const;

    /// How many entries the depth cap has pushed off the bottom.
    std::size_t dropped_count() const { return dropped_count_; }
    bool history_truncated() const { return dropped_count_ > 0; }

    void undo();
    void redo();

    std::size_t size() const { return commands_.size(); }
    std::size_t position() const { return position_; }

    void clear();

private:
    /// Truncate the redo branch and push an already-applied command.
    void push_applied(std::unique_ptr<AbstractCommand> command);
    void close_all_macros() noexcept;

    Document* document_ = nullptr;
    std::string cache_dir_;
    std::vector<std::unique_ptr<AbstractCommand>> commands_;
    std::size_t position_ = 0;   ///< number of commands currently applied
    std::size_t dropped_count_ = 0;
    std::vector<std::unique_ptr<CommandTransaction>> macro_frames_;
};

/// Rename a chunk. The one concrete command in v0, so the layer is exercised.
std::unique_ptr<AbstractCommand> make_set_chunk_label_command(
    std::int32_t chunk_id, Document& document, std::string new_label);

}  // namespace alphashape
