#include "common/commands/command.hpp"

// One concrete command, so the layer is exercised rather than theoretical:
// setting a chunk's label. Small, obviously reversible, and a real edit the GUI
// would make. The algorithm milestones add commands that touch geometry; those
// come through model_commands.hpp (WP-11.5b/11.5c).

#include <utility>

namespace alphashape {

namespace {

/// Staged form: validate finds the chunk, stage records the old label, commit
/// writes the new one, restore puts the old one back. Applying to a missing
/// chunk throws ValueError before anything is captured, so the document is
/// untouched (WP-11.5a invariant).
class SetChunkLabelCommand : public StagedMutation {
public:
    SetChunkLabelCommand(std::int32_t chunk_id, std::string new_label)
        : chunk_id_(chunk_id), new_label_(std::move(new_label)) {}

    std::string name() const override { return "Set Chunk Label"; }

protected:
    void validate(Document& document) override {
        if (find(document) == nullptr) {
            throw ValueError("no chunk with id " + std::to_string(chunk_id_));
        }
    }

    // Capture the label at apply time, not in the constructor: a command may be
    // constructed and then executed later (or on a different document), so
    // only the value at apply time is the correct thing to restore.
    void stage(Document& document) override { previous_label_ = find(document)->label(); }

    void commit(Document& document) override {
        Chunk* chunk = find(document);
        if (chunk == nullptr) {
            // validate() saw it; a document mutated in between is a bug, and
            // StagedMutation::apply rolls back to the staged state.
            throw ValueError("no chunk with id " + std::to_string(chunk_id_));
        }
        chunk->set_label(new_label_);
    }

    void restore(Document& document) override {
        Chunk* chunk = find(document);
        if (chunk == nullptr) {
            throw ValueError("no chunk with id " + std::to_string(chunk_id_));
        }
        chunk->set_label(previous_label_);
    }

private:
    Chunk* find(Document& document) const {
        for (auto& workspace : document.workspaces()) {
            if (Chunk* chunk = workspace->find_chunk(chunk_id_)) {
                return chunk;
            }
        }
        return nullptr;
    }

    std::int32_t chunk_id_;
    std::string new_label_;
    std::string previous_label_;
};

}  // namespace

/// Factory declared in the header: the header carries the abstraction, and
/// concrete commands appear as they are needed.
std::unique_ptr<AbstractCommand> make_set_chunk_label_command(
    std::int32_t chunk_id, Document& /*document*/, std::string new_label) {
    return std::make_unique<SetChunkLabelCommand>(chunk_id, std::move(new_label));
}

}  // namespace alphashape
