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
    /// GNSS reference position in a metric ENU frame (metres), when the camera
    /// carries an enabled reference. Used only when
    /// BuilderOptions::reference_priors is on; the builder fits the SfM -> ENU
    /// similarity itself, so the caller does not need to know the SfM frame.
    std::optional<Eigen::Vector3d> reference_enu;
    /// Per-axis 1-sigma accuracy of `reference_enu`, metres. A non-positive or
    /// non-finite axis is unconstrained, matching BACentrePrior.
    Eigen::Vector3d reference_sigma = Eigen::Vector3d::Zero();
    /// GNSS antenna lever arm in the CAMERA frame, metres. Zero means the
    /// reference measures the projection centre. Scaled into the SfM frame by
    /// the fitted similarity before it reaches BACentrePrior.
    Eigen::Vector3d reference_lever_arm = Eigen::Vector3d::Zero();
};

struct BuilderInput {
    std::vector<BuilderCamera> cameras;
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

    /// GNSS-aided bundle adjustment (Triggs et al. 2000, "Bundle Adjustment --
    /// A Modern Synthesis", sec. on priors; Cucci, Rehak & Skaloud 2017,
    /// INS/GNSS-aided BA). When true, every global BA after the first three
    /// non-collinear referenced cameras are registered adds a per-camera centre
    /// prior in the metric ENU frame, whitened by that camera's own reference
    /// accuracy and robustified with the same Huber loss OptimizeCameras uses.
    /// The SfM -> ENU similarity is re-estimated after each global BA, so the
    /// SfM frame itself never changes mid-build. Off by default: a build with
    /// this false is bit-identical to the pre-existing behaviour.
    bool reference_priors = false;
    /// Ours: the RANSAC threshold of the SfM -> ENU similarity fit, in metres.
    /// The georeference stage uses max(0.5 m, 5 x median horizontal accuracy);
    /// the builder uses the same rule so the two agree on what an inlier is.
    double reference_prior_gate_m = 0.5;
};

struct BuilderResult {
    std::vector<std::optional<AbsolutePose>> poses;      ///< per input camera index
    std::vector<std::optional<Eigen::Vector3d>> points;  ///< per input track index
    std::size_t registered = 0;
    std::int32_t initial_camera_a = -1, initial_camera_b = -1;
    double final_rms_px = 0.0;
};

/// Incremental SfM.
///
/// Throws ValueError for malformed input (unknown camera id, keypoint index out
/// of range). `cancelled` (may be empty) is polled between registrations; if it
/// returns true, throws CancelledError("cancelled").
BuilderResult build_incremental(const BuilderInput& input, const BuilderOptions& options,
                                const std::function<bool()>& cancelled = {});

}  // namespace alphashape::sfm
