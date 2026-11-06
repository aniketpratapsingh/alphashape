#pragma once

// RunScript (M11 §4.9, WP-11.10): run a user Python script against the open
// project.
//
// The command is `python3 <path> --project <absolute project path> <args...>`
// with no shell involved: the argv vector goes straight to execve/CreateProcess.
// The child gets ALPHASHAPE_PROJECT and a PYTHONPATH that starts with the
// built alphashape module directory (ALPHASHAPE_PYTHON_MODULE_DIR, overridable
// through the CLI's --python-path). stdout and stderr are captured, and the
// output is stored on the active chunk as `RunScript/<n>/output`, truncated to
// 8 KB on a UTF-8 boundary. A non-zero exit, a missing python3 or a timeout is
// an Error.

#include "common/tasks/task.hpp"

namespace alphashape {

class RunScriptTask : public AbstractTask {
public:
    std::string name() const override { return "RunScript"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
