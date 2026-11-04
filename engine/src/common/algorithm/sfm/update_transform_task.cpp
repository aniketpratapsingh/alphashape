#include "common/algorithm/sfm/update_transform_task.hpp"

#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/sfm/georeference.hpp"

namespace alphashape {
namespace {

/// Full-precision text for a metadata value.
std::string meta_number(double value) {
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

}  // namespace

void UpdateTransformTask::declare_params(Params&) const {
    // No parameters: the fit is fully determined by the chunk's references.
}

std::vector<Stage> UpdateTransformTask::stages() const {
    return {Stage{"update", 0}};
}

void UpdateTransformTask::apply(Document& document, Params& params, const Progress& progress) {
    (void)params;
    progress.update("update", 0.0);
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("UpdateTransform needs an active chunk");
    }

    // Stage on a copy: a refused fit leaves the chunk exactly as it was, and the
    // commit below is the one mutation the undo stack sees.
    Chunk staged = *chunk;
    const sfm::GeoreferenceReport report = sfm::update_transform(staged);
    progress.update("update", 1.0);

    if (!report.applied) {
        if (Log::enabled(LogLevel::Info)) {
            log(LogLevel::Info) << "UpdateTransform: not applied (" << report.reason << ")";
        }
        return;
    }

    staged.meta().set("update_transform.applied", "1");
    staged.meta().set("update_transform.control_points",
                      std::to_string(report.control_points_used));
    staged.meta().set("update_transform.cameras", std::to_string(report.cameras_used));
    staged.meta().set("update_transform.check_points",
                      std::to_string(report.check_points_used));
    staged.meta().set("update_transform.rms_control_m", meta_number(report.rms_control_m));
    staged.meta().set("update_transform.rms_check_m", meta_number(report.rms_check_m));
    staged.meta().set("update_transform.max_check_m", meta_number(report.max_check_m));
    staged.meta().set("update_transform.scale", meta_number(report.scale));

    // -- commit atomically --------------------------------------------------
    *chunk = std::move(staged);

    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info)
            << "UpdateTransform: " << report.control_points_used << " control points, "
            << report.cameras_used << " cameras, " << report.check_points_used
            << " check points; control RMS " << report.rms_control_m << " m, check RMS "
            << report.rms_check_m << " m (max " << report.max_check_m << " m), scale "
            << report.scale;
    }
}

}  // namespace alphashape
