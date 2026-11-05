#pragma once

/// @file calibrate_fiducials_task.hpp
/// @brief CalibrateFiducials: derive a sensor's fiducial coordinates from one
/// scan (WP-13.18a).
///
/// A reference scan with every listed fiducial measured fixes the film layout.
/// The layout centre is the centre of the measured marks' bounding box and
/// film_mm = ((u - cu) * resolution, -(v - cv) * resolution), with y flipping
/// so film coordinates are y up. The sensor's focal length and principal point
/// are untouched. Every camera with measured marks then has its film_transform
/// re-fitted against the new coordinates.

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "common/core/document.hpp"
#include "common/tasks/task.hpp"
#include "model/camera/film.hpp"

namespace alphashape {

struct CalibrateFiducialsOptions {
    /// Millimetres per scan pixel; must be positive.
    double resolution_mm_per_px = 0.0;
    /// Camera id of the reference scan, or -1 for the first camera of each
    /// sensor with every listed fiducial measured.
    std::int32_t camera = -1;
};

/// Derive film coordinates from one scan's measured pixels. `current` supplies
/// the labels and their order; `pixels` is parallel and must be complete.
/// Throws ValueError for a non-positive resolution, mismatched counts or a
/// non-finite measurement.
std::vector<Fiducial> derive_film_layout(const std::vector<Fiducial>& current,
                                         const std::vector<Eigen::Vector2d>& pixels,
                                         double resolution_mm_per_px);

class CalibrateFiducialsTask : public AbstractTask {
public:
    std::string name() const override { return "CalibrateFiducials"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
