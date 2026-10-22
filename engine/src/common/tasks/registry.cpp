#include "common/tasks/registry.hpp"

namespace alphashape {

void TaskRegistry::register_task(std::shared_ptr<AbstractTask> task, bool enabled) {
    if (!task) {
        throw ValueError("cannot register a null task");
    }
    const std::string name = task->name();
    if (entries_.find(name) != entries_.end()) {
        throw ValueError("task '" + name + "' is already registered");
    }

    TaskEntry entry;
    entry.task = std::move(task);
    entry.enabled = enabled;
    entry.task->declare_params(entry.schema);

    entries_.emplace(name, std::move(entry));
}

const TaskEntry& TaskRegistry::require(const std::string& name) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw UnsupportedTaskError(name);
    }
    return it->second;
}

const TaskEntry* TaskRegistry::find(const std::string& name) const {
    const auto it = entries_.find(name);
    return it == entries_.end() ? nullptr : &it->second;
}

std::vector<std::string> TaskRegistry::names() const {
    std::vector<std::string> result;
    result.reserve(entries_.size());
    for (const auto& [name, entry] : entries_) {
        (void)entry;
        result.push_back(name);  // map order: deterministic
    }
    return result;
}

TaskRegistry& TaskRegistry::instance() {
    static TaskRegistry registry;
    return registry;
}

Params make_params(const TaskEntry& entry) {
    // The schema holds declarations only, so a copy is a clean value set.
    return entry.schema;
}

void record_task_params(Meta& meta, const std::string& task_name, const Params& params) {
    for (const ParamSpec& spec : params.specs()) {
        const bool has_value =
            params.is_set(spec.name) ||
            !std::holds_alternative<std::monostate>(spec.default_value);
        if (has_value) {
            meta.set(spec.qualified(task_name), format_param_value(params.get(spec.name)));
        }
    }
}

void run_task(const TaskRegistry& registry, Document& document, const std::string& task_name,
              Params& params, const Progress& progress) {
    const TaskEntry& entry = registry.require(task_name);
    if (!entry.enabled) {
        // Present in the registry, refuses to run: the observed disabled state.
        throw Error("Task is disabled: " + task_name);
    }

    Chunk* chunk = document.active_chunk();
    if (entry.task->target_type() == TargetType::Chunk && chunk == nullptr) {
        throw Error("Task " + task_name + " needs an active chunk");
    }

    entry.task->apply(document, params, progress);

    if (chunk != nullptr) {
        record_task_params(chunk->meta(), task_name, params);
    }
}

}  // namespace alphashape
