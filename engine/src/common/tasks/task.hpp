#pragma once

// The task layer.
//
// Everything that computes lives behind AbstractTask, registered under a string
// name in one map. Shape:
//
//   * registration:  AbstractTask::registerTask<TaskX>() per task, one flat
//                    block of calls at startup
//   * container:     std::map<std::string, shared_ptr<AbstractTask>>
//   * lookup:        miss raises "Unsupported task: <name>"
//   * parameters:    a per-task declared schema, keyed "<Task>/<param>"
//   * targets:       a task either targets a Chunk or a Workspace
//   * stages:        a task declares an ordered list of stages; each stage is
//                    split into work items for parallel execution
//
// Commands vs tasks: a Command is an undoable document mutation driven by the
// GUI; a Task is long-running computation that may be distributed. A task that
// needs to be undoable emits a reversal command — see commands/command.hpp.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common/core/document.hpp"
#include "common/tasks/params.hpp"

namespace alphashape {

/// Progress reporting: a callback with a status string and a fraction, plus a cancellation flag
/// (the GIL is released per task and retaken per progress tick).
struct Progress {
    /// Called by tasks; implementations may be a no-op.
    std::function<void(const std::string& stage, double fraction)> report;

    /// Tasks poll this; returning true requests a clean stop.
    std::function<bool()> cancelled;

    void update(const std::string& stage, double fraction) const {
        if (report) {
            report(stage, fraction);
        }
    }

    bool is_cancelled() const { return cancelled && cancelled(); }
};

/// What a task operates on: a task declares either a Chunk target or a (workspace-level) frame target.
enum class TargetType { Chunk, Frames };

/// One stage of a task. A task is an ordered list of these; each is split into
/// work items so they can run in parallel and be reported separately.
///
/// The stage/item splitter design is ours (e.g. MatchPhotos = initialize,
/// detect, match, prematch, select, finalize, cleanup).
struct Stage {
    std::string name;
    /// Hint for the splitter: how many items of the stage's natural unit go
    /// into one work item. 0 means "do not split".
    std::int64_t workitem_size = 0;
};

/// The common interface for every task.
///
/// Deliberately close to the observed vtable: a destructor, an accessor that
/// yields the concrete task object (used to prove a registered entry casts to
/// the expected type), and the body that does the work.
class AbstractTask {
public:
    virtual ~AbstractTask() = default;

    /// The name this task is registered under (e.g. "DetectFeatures").
    virtual std::string name() const = 0;

    /// Declare the parameter schema. Called once at registration.
    virtual void declare_params(Params& params) const = 0;

    virtual TargetType target_type() const = 0;

    /// Ordered stages, for progress and work-item splitting.
    virtual std::vector<Stage> stages() const { return {}; }

    /// Does the work. Throws on failure.
    virtual void apply(Document& document, Params& params, const Progress& progress) = 0;
};

}  // namespace alphashape
