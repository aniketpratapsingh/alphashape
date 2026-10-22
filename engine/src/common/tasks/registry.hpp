#pragma once

// TaskRegistry: the string-keyed map every front end dispatches through.
//
// Design:
//   * one std::map<std::string, shared_ptr<AbstractTask>>
//   * registration is a flat sequence of registerTask<TaskX>() calls
//   * a lookup miss raises "Unsupported task: <name>"
//   * each entry also carries an enabled flag; disabled entries exist in the map but refuse to run

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "common/tasks/task.hpp"

namespace alphashape {

/// One registry entry: the task instance, its schema, and its enabled flag.
struct TaskEntry {
    std::shared_ptr<AbstractTask> task;
    Params schema;          ///< declared parameters (defaults, no values bound)
    bool enabled = true;
};

class TaskRegistry {
public:
    /// Register a task type under its name(). The schema is populated by
    /// calling task->declare_params().
    ///
    /// `enabled`: a disabled task is present in the
    /// registry (and so appears in listings and in the schema) but throws when
    /// run. It gates tasks that ship but are not exposed yet.
    void register_task(std::shared_ptr<AbstractTask> task, bool enabled = true);

    /// Look up by name. Throws UnsupportedTaskError on a miss.
    const TaskEntry& require(const std::string& name) const;

    /// Look up by name, returning nullptr on a miss.
    const TaskEntry* find(const std::string& name) const;

    std::vector<std::string> names() const;

    std::size_t size() const { return entries_.size(); }

    /// The process-wide registry, populated by register_builtin_tasks().
    static TaskRegistry& instance();

private:
    std::map<std::string, TaskEntry> entries_;
};

/// A fresh parameter set for one invocation: the entry's schema, no values.
Params make_params(const TaskEntry& entry);

/// The one execution path every front end uses (CLI, Python, later the GUI
/// and network worker). Looks the task up in `registry` (so the caller chooses
/// which registry to run against), checks the enabled flag and the target,
/// runs the task, then records the effective parameters into the active
/// chunk's meta under "<Task>/<param>" keys.
///
/// Processing parameters live in chunk metadata under those qualified keys.
/// That is also what makes the drift test possible: the keys in a saved project
/// are generated from the same schema the front ends bind against.
void run_task(const TaskRegistry& registry, Document& document, const std::string& task_name,
              Params& params, const Progress& progress = {});

/// Write every declared parameter's effective value into `meta`. The policy is
/// to record the effective value of every declared parameter, explicit or
/// default, so a project states exactly what was run even if a later version
/// changes a default.
void record_task_params(Meta& meta, const std::string& task_name, const Params& params);

/// Register every task the engine ships. Called once at startup; mirrors the
/// observed flat registration block. Calling it twice is a no-op.
void register_builtin_tasks(TaskRegistry& registry);

/// Convenience: the process-wide registry after registering built-ins.
TaskRegistry& builtin_registry();

}  // namespace alphashape
