#pragma once

// structure_builder.hpp -- incremental structure-from-motion.
//
// Clean-room implementation from the published literature only:
//   * N. Snavely, S. Seitz, R. Szeliski, "Photo Tourism: Exploring Photo
//     Collections in 3D" (SIGGRAPH 2006) -- the incremental pipeline: choose an
//     initial pair, add cameras one at a time by resection, triangulate new
//     tracks as cameras join, and refine everything with bundle adjustment.
//   * J. L. Schonberger & J.-M. Frahm, "Structure-from-Motion Revisited"
//     (CVPR 2016) -- the modern formulation of the same steps: rank the initial
//     pairs, prefer well-parallaxed two-view geometries, register the next-best
//     camera, filter by reprojection error and parallax between rounds.
//
// None of the constants here are taken from any reference implementation; they
// are our own defaults and are documented in structure_builder.cpp.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "common/algorithm/sfm/absolute_pose.hpp"
#include "common/core/types.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::sfm {

/// One input camera plus its keypoints. `pixels` is in the Calibration pixel
/// convention (top-left pixel centre = (0.5, 0.5)); the index into `pixels` is
/// the keypoint index used by PairMatches / Track observations.
struct BuilderCamera {
    std::int32_t id = -1;
    Calibration calibration;
    std::vector<Eigen::Vector2d> pixels;
    /// Already-aligned camera to keep when continuing an alignment.
    std::optional<AbsolutePose> initial_pose;
    /// Sensor grouping key. Cameras that share a non-negative `sensor_id` share
    /// one calibration in bundle adjustment, so self-calibration refines one
    /// lens per physical sensor rather than one per photo. A negative value
    /// (the default) makes the camera its own sensor, which -- with the
    /// minimum-camera gate below -- leaves its intrinsics fixed, the historical
    /// behaviour.
    std::int32_t sensor_id = -1;
};

/// Per-sensor metadata the builder needs to honour a sensor's held coefficients.
struct BuilderSensor {
    std::int32_t id = -1;
    /// Coefficients held fixed during self-calibration (Sensor::fixed_params),
    /// named as in kCalibrationParamNames. A coefficient named here is never
    /// freed, whatever the refinement schedule would otherwise allow.
    std::vector<std::string> fixed_params;
};

struct BuilderInput {
    std::vector<BuilderCamera> cameras;
    /// Optional per-sensor metadata, matched to BuilderCamera::sensor_id. A
    /// sensor id with no entry here simply has no held coefficients.
    std::vector<BuilderSensor> sensors;
    std::vector<PairMatches> matches;  ///< verified pair matches (camera ids)
    std::vector<Track> tracks;         ///< tracks over those keypoints;
                                       ///< .point = initial point if continuing
};

/// Every tunable in this builder (all ours; see structure_builder.cpp).
struct BuilderOptions {
    std::size_t min_pair_inliers = 30;     ///< initial-pair candidates
    double min_init_parallax_deg = 3.0;    ///< initial pair: median triangulation angle
    std::size_t init_pair_candidates = 30; ///< evaluate the top-N pairs by match count
    std::size_t min_resection_points = 12;
    double resection_threshold_px = 4.0;
    double triangulation_min_parallax_deg = 1.5;
    double max_reprojection_px = 4.0;  ///< filter threshold after BA
    double ba_loss_scale_px = 1.0;
    /// ours: run global BA after every N registrations while registered < 50;
    /// then every max(1, registered/10).
    std::size_t global_ba_every = 1;
    std::uint64_t seed = 0x5eed;

    /// Self-calibration during the global bundle adjustments. When true (the
    /// default) the builder refines each sensor's focal length, principal point
    /// and low-order radial/tangential distortion once enough of that sensor's
    /// cameras are registered; see the schedule in structure_builder.cpp. When
    /// false every intrinsic stays at its input value, the pre-self-calibration
    /// behaviour.
    bool fit_intrinsics = true;
    /// Minimum number of registered cameras of one sensor before its intrinsics
    /// are freed at all. Below this the sensor's calibration is held fixed, so a
    /// handful of views cannot drag the lens off its EXIF seed. Ten is the low
    /// end of the range mainstream incremental SfM uses, and it is early enough
    /// on a real block that a wrong lens is corrected before the structure can
    /// bend around it.
    std::size_t min_cameras_for_intrinsics = 10;
    /// Registered-camera count at which the full free set (f, cx, cy, k1, k2,
    /// k3, p1, p2) is released. Between min_cameras_for_intrinsics and this the
    /// builder frees only f and k1, the two best-conditioned terms.
    std::size_t full_intrinsics_cameras = 15;
    /// Divergence guards. A refined calibration is rejected (reverted to the
    /// previous value for that sensor) when the focal length moves by more than
    /// this fraction of its initial value, when |k1| exceeds the bound, when any
    /// coefficient is non-finite, or when the global BA RMS got worse.
    double max_focal_change_fraction = 0.30;
    double max_abs_k1 = 2.0;
};

struct BuilderResult {
    std::vector<std::optional<AbsolutePose>> poses;      ///< per input camera index
    std::vector<std::optional<Eigen::Vector3d>> points;  ///< per input track index
    std::size_t registered = 0;
    std::int32_t initial_camera_a = -1, initial_camera_b = -1;
    double final_rms_px = 0.0;
    /// The refined calibration of every sensor that was self-calibrated, keyed
    /// by BuilderCamera::sensor_id. A sensor absent here kept its input
    /// calibration (never freed, or reverted by a divergence guard). The caller
    /// writes these back to the sensor's ADJUSTED calibration.
    std::vector<std::pair<std::int32_t, Calibration>> refined_calibrations;
};

/// Incremental SfM.
///
/// Throws ValueError for malformed input (unknown camera id, keypoint index out
/// of range). `cancelled` (may be empty) is polled between registrations; if it
/// returns true, throws CancelledError("cancelled").
BuilderResult build_incremental(const BuilderInput& input, const BuilderOptions& options,
                                const std::function<bool()>& cancelled = {});

}  // namespace alphashape::sfm
