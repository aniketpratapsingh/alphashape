#pragma once

// Cold-start job estimates.
//
// The estimator's fallback before a project has measured throughput: a small
// per-task linear model whose constants come from our own measured CPU
// profile (Apple M1, 8 threads, RelWithDebInfo). The numbers are deliberately
// revisable: every constant below names the measured row it came from and the
// arithmetic that turned it into a coefficient. Re-measuring on a different
// machine means editing this file, not the estimator.
//
// The shared shape is
//
//     seconds = a + b * cameras * megapixels + c * megapixels * neighbours
//
// with `megapixels` the working resolution per camera (full sensor megapixels
// divided by downscale squared) and `neighbours` the task's effective
// neighbourhood. Two stages do not fit that shape and use the extra volume /
// product fields in WorkItem instead; both are documented beside their
// constants.
//
// Peak memory follows the spec's rules (SP 4.8): 96 MB per working megapixel
// of depth maps, eight bytes per voxel for meshing, and twice the product
// bytes for DEM/ortho. Disk is the product bytes times 1.15.
//
// No Qt types here: this header is also usable from the headless estimator
// tests and from a future job.db replacement.

#include <cstdint>
#include <string>

namespace alphashape::gui::estimates {

/// Tasks with a calibrated cold model. `Generic` is the safe fallback for a
/// task the estimator has never heard of.
enum class ColdTask {
    Features,
    MatchPhotos,
    DepthMaps,
    PointCloud,
    Model,
    Texture,
    Dem,
    Ortho,
    Generic,
};

/// Map a registry task name to its cold-model family. Unknown names map to
/// Generic; this never throws.
ColdTask cold_task(const std::string& task_name);

/// The linear coefficients described above. A zero coefficient simply drops
/// that term for the task.
struct ColdCoefficients {
    double a = 0.0;  ///< seconds, fixed startup cost
    double b = 0.0;  ///< seconds per (camera * megapixel)
    double c = 0.0;  ///< seconds per (megapixel * neighbour)
};

/// Neighbour count the depth-map coefficient was measured at. The estimator
/// scales the per-camera term by neighbours / kReferenceNeighbours.
inline constexpr double kReferenceNeighbours = 7.0;

/// Working-set estimate for one run. All counts are absolute (already summed
/// over cameras where that is what the task does).
struct WorkItem {
    double cameras = 0.0;        ///< enabled cameras taking part
    double megapixels = 0.0;     ///< working-resolution megapixels per camera
    double neighbours = 0.0;     ///< effective neighbours per camera
    double volume_voxels = 0.0;  ///< meshing volume (resolution^3)
    double product_pixels = 0.0; ///< DEM / ortho output cells
    double texels = 0.0;         ///< texture atlas texels
};

/// Bytes per output cell used for the product-size rules. DEM products are
/// float elevation (4 B); ortho products are RGB (3 B). Texture pages are RGB
/// too. Documented here so the estimator and the dialog agree.
inline constexpr std::uint64_t kDemBytesPerCell = 4;
inline constexpr std::uint64_t kRgbBytesPerCell = 3;

/// Peak read/write working set of the depth-map stage.
inline constexpr std::uint64_t kDepthMegapixelRamBytes = 96ULL * 1000ULL * 1000ULL;

/// Meshing keeps eight bytes per volume voxel.
inline constexpr std::uint64_t kMeshBytesPerVoxel = 8;

/// The cold estimate for `task` at `work`: seconds plus peak RAM and product
/// bytes. Always returns finite, non-negative values.
struct ColdEstimate {
    double seconds = 0.0;
    std::uint64_t peak_ram_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t disk_bytes = 0;
};

ColdEstimate cold_estimate(ColdTask task, const WorkItem& work);

/// The documented coefficients. Kept as a function rather than an aggregate
/// table so each branch can carry its derivation comment.
ColdCoefficients cold_coefficients(ColdTask task);

}  // namespace alphashape::gui::estimates
