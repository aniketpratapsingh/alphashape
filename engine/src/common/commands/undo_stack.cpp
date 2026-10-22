#include "common/commands/command.hpp"

#include <algorithm>
#include <utility>

namespace alphashape {

namespace {

/// Suffix appended to undo_label() at the oldest retained entry once the depth
/// cap has dropped history (§4.7 / WP-11.5a).
constexpr const char* kTruncatedSuffix = " (older steps dropped)";

}  // namespace

UndoStack::UndoStack(Document* document) : document_(document) {}

UndoStack::~UndoStack() { close_all_macros(); }

void UndoStack::close_all_macros() noexcept {
    // Innermost first: push each open group as an entry, then its parent.
    // Nothing here may throw: this runs in the destructor.
    while (!macro_frames_.empty()) {
        try {
            end_macro();
        } catch (...) {
            // end_macro() cannot fail once a frame exists except on memory
            // exhaustion; drop the frame so destruction always terminates.
            macro_frames_.pop_back();
        }
    }
}

void UndoStack::execute(std::unique_ptr<AbstractCommand> command) {
    if (!command) {
        throw ValueError("cannot execute a null command");
    }
    if (document_ == nullptr) {
        throw ValueError("undo stack is not bound to a document");
    }

    UndoContext context;
    context.cache_dir = cache_dir_;
    command->on_prepare(context);

    // Memory policy (§4.7): an unbounded command is refused *before* it runs,
    // so the document is untouched and the caller can re-run instead of undoing.
    if (!command->is_memory_bounded()) {
        throw Error("command too large for the undo history");
    }

    command->apply(*document_);

    // Second half of the policy: a command usually learns its payload size only
    // while applying (the asset it replaces is read from the document). By
    // contract such a command declines to commit, so the document is already
    // untouched here — refuse without reverting (§4.7, "pre-decided").
    if (!command->is_memory_bounded()) {
        throw Error("command too large for the undo history");
    }

    if (!macro_frames_.empty()) {
        // Inside a macro the edit is already applied; collect it. The group
        // owns it until end_macro() pushes the transaction.
        macro_frames_.back()->add(std::move(command));
        return;
    }
    push_applied(std::move(command));
}

void UndoStack::push_applied(std::unique_ptr<AbstractCommand> command) {
    // Executing discards any redo branch: those states are unreachable now.
    commands_.resize(position_);
    commands_.push_back(std::move(command));
    position_ = commands_.size();

    // Depth cap: drop oldest entries. Every dropped entry is below the
    // position, so position_ stays the count of applied commands.
    while (commands_.size() > kMaxUndoDepth) {
        commands_.erase(commands_.begin());
        --position_;
        ++dropped_count_;
    }
}

void UndoStack::begin_macro(std::string name) {
    if (name.empty()) {
        name = "Transaction";
    }
    macro_frames_.push_back(std::make_unique<CommandTransaction>(std::move(name)));
}

void UndoStack::end_macro() {
    if (macro_frames_.empty()) {
        throw ValueError("end_macro() without begin_macro()");
    }
    std::unique_ptr<CommandTransaction> group = std::move(macro_frames_.back());
    macro_frames_.pop_back();

    if (!macro_frames_.empty()) {
        // Nested: the parent takes ownership of the already-applied group.
        macro_frames_.back()->add(std::move(group));
        return;
    }
    // An empty group applied nothing; pushing it would give the user an undo
    // step that does nothing.
    if (group->empty()) {
        return;
    }
    push_applied(std::move(group));
}

std::string UndoStack::undo_name() const {
    if (!can_undo()) {
        return {};
    }
    return commands_[position_ - 1]->name();
}

std::string UndoStack::redo_name() const {
    if (!can_redo()) {
        return {};
    }
    return commands_[position_]->name();
}

std::string UndoStack::undo_label() const {
    const std::string name = undo_name();
    if (name.empty()) {
        return {};
    }
    // Truncation is visible exactly where the user would hit the wall: on the
    // oldest retained step.
    if (history_truncated() && position_ == 1) {
        return "Undo " + name + kTruncatedSuffix;
    }
    return "Undo " + name;
}

std::string UndoStack::redo_label() const {
    const std::string name = redo_name();
    return name.empty() ? std::string{} : ("Redo " + name);
}

std::vector<std::string> UndoStack::command_names() const {
    std::vector<std::string> names;
    names.reserve(commands_.size());
    for (const auto& command : commands_) {
        names.push_back(command->name());
    }
    return names;
}

void UndoStack::undo() {
    if (macro_open()) {
        throw Error("cannot undo while a macro is open");
    }
    if (!can_undo()) {
        throw ValueError("nothing to undo");
    }
    if (document_ == nullptr) {
        throw ValueError("undo stack is not bound to a document");
    }
    // Revert first, decrement only on success: a throwing revert must leave
    // the stack unchanged (the mirror of redo(), which increments only after
    // apply() returns).
    commands_[position_ - 1]->revert(*document_);
    --position_;
}

void UndoStack::redo() {
    if (macro_open()) {
        throw Error("cannot redo while a macro is open");
    }
    if (!can_redo()) {
        throw ValueError("nothing to redo");
    }
    if (document_ == nullptr) {
        throw ValueError("undo stack is not bound to a document");
    }
    UndoContext context;
    context.cache_dir = cache_dir_;
    commands_[position_]->on_prepare(context);
    commands_[position_]->apply(*document_);
    ++position_;
}

void UndoStack::clear() {
    // Forget everything, including any open macro: the edits it collected stay
    // applied but are no longer undoable, which is what "clear history" means.
    // A later end_macro() then throws, as it would with no group open.
    commands_.clear();
    position_ = 0;
    dropped_count_ = 0;
    macro_frames_.clear();
}

}  // namespace alphashape
