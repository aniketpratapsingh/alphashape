#include "common/algorithm/sfm/update_transform_task.hpp"

#include <cstddef>
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
        // The fit touched no geometry on the staged copy, so committing now
        // records the refusal in meta without changing the chunk's geometry.
        staged.meta().set("update_transform.applied", "0");
        staged.meta().set("update_transform.reason", report.reason);
        *chunk = std::move(staged);
        if (Log::enabled(LogLevel::Info)) {
            log(LogLevel::Info) << "UpdateTransform: not applied (" << report.reason << ")";
        }
        return;
    }

    staged.meta().set("update_transform.applied", "1");
    staged.meta().set("update_transform.reason", "");
    staged.meta().set("update_transform.control_points",
                      std::to_string(report.control_points_used));
    staged.meta().set("update_transform.cameras", std::to_string(report.cameras_used));
    staged.meta().set("update_transform.check_points",
                      std::to_string(report.check_points_used));
    staged.meta().set("update_transform.rms_control_m", meta_number(report.rms_control_m));
    staged.meta().set("update_transform.rms_check_m", meta_number(report.rms_check_m));
    staged.meta().set("update_transform.max_check_m", meta_number(report.max_check_m));
    staged.meta().set("update_transform.scale", meta_number(report.scale));
    // Total 1/sigma^2 weight carried by each correspondence class in the
    // similarity fit: the datum must be dominated by the control markers.
    staged.meta().set("update_transform.camera_weight",
                      meta_number(report.camera_weight));
    staged.meta().set("update_transform.control_weight",
                      meta_number(report.control_weight));

    // Rejected control markers, comma-separated, and one key per check point so
    // a front end can show the error next to the marker id.
    std::string rejected;
    for (std::size_t i = 0; i < report.rejected_markers.size(); ++i) {
        if (i != 0) {
            rejected += ',';
        }
        rejected += std::to_string(report.rejected_markers[i]);
    }
    staged.meta().set("update_transform.rejected_markers", rejected);
    for (const auto& [marker_id, error] : report.check_errors) {
        staged.meta().set("update_transform.check_error_" + std::to_string(marker_id) + "_m",
                          meta_number(error));
    }

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
