#include "apps/gui/jobs/estimate_model.hpp"

#include <algorithm>
#include <cmath>

namespace alphashape::gui::estimates {
namespace {

double non_negative(double value) {
    return std::isfinite(value) && value > 0.0 ? value : 0.0;
}

double product_megapixels(const WorkItem& work) {
    return work.product_pixels / 1.0e6;
}

constexpr double kModelReferenceVoxels = 128.0 * 128.0 * 128.0;

// 2^64 as a double. The largest double below it still converts to a valid
// uint64; anything at or above saturates instead of being undefined.
constexpr double kUint64Limit = 18446744073709551616.0;

}  // namespace

std::uint64_t saturating_bytes(double value) {
    if (!std::isfinite(value) || value <= 0.0) {
        return 0;
    }
    if (value >= kUint64Limit) {
        return UINT64_MAX;
    }
    return static_cast<std::uint64_t>(value);
}

ColdTask cold_task(const std::string& task_name) {
    if (task_name == "DetectFeatures") {
        return ColdTask::Features;
    }
    if (task_name == "MatchPhotos") {
        return ColdTask::MatchPhotos;
    }
    if (task_name == "BuildDepthMaps") {
        return ColdTask::DepthMaps;
    }
    if (task_name == "BuildPointCloud") {
        return ColdTask::PointCloud;
    }
    if (task_name == "BuildModel") {
        return ColdTask::Model;
    }
    if (task_name == "BuildTexture" || task_name == "BuildUV") {
        return ColdTask::Texture;
    }
    if (task_name == "BuildDem") {
        return ColdTask::Dem;
    }
    if (task_name == "BuildOrthomosaic") {
        return ColdTask::Ortho;
    }
    return ColdTask::Generic;
}

ColdCoefficients cold_coefficients(ColdTask task) {
    ColdCoefficients coefficients;
    switch (task) {
        case ColdTask::Features:
            // Measured CPU profile, M1, 8 threads. The DoG + MLDB row is
            // "0.35 s (4 tests, 480x360)": that 0.35 s is the TOTAL over four
            // tests, so treat it as four single-image runs of 0.1728 Mpx:
            // 0.35 / 4 = 0.0875 s per image, 0.0875 / 0.1728 = 0.51 s per
            // camera-megapixel. a = 0.4 s covers thread start and the keypoint
            // tables. Three measured runs replace this constant via history.
            coefficients = {0.4, kFeatureDetectionSecondsPerCameraMpx, 0.0};
            break;
        case ColdTask::MatchPhotos:
            // The Hamming matcher row only: ~2.6 s over ten tiny 8-image
            // tests, i.e. 0.26 s per test over 8 x 0.1728 = 1.3824 Mpx, so
            // 0.26 / 1.3824 = 0.19 s per camera-megapixel. Detection is a
            // separate term added only for cameras with no keypoints yet (see
            // cold_estimate). a = 0.8 s for pair bookkeeping and the pair sort.
            coefficients = {0.8, 0.19, 0.0};
            break;
        case ColdTask::DepthMaps:
            // PatchMatch row: 1.04 s per map at 0.0432 Mpx with 3-8
            // neighbours (reference 7). 1.04 / 0.0432 = 24.07 s per
            // camera-megapixel at the reference neighbourhood. The profile's
            // filter row keeps ~2-5 % of depth maps, midpoint 3.5 %, so
            // 24.07 x 1.035 = 24.91 -> 24.9. The neighbour dependence is
            // applied as b * (neighbours / 7), documented in cold_estimate.
            coefficients = {0.05, 24.9, 0.0};
            break;
        case ColdTask::PointCloud:
            // Fusion row: under 1 s for 12 maps at 0.0432 Mpx (0.518 Mpx
            // total) is 1.9 s per total megapixel. a = 0.2 s for the voxel
            // hash setup.
            coefficients = {0.2, 1.9, 0.0};
            break;
        case ColdTask::Model:
            // At the 128^3 reference volume: Poisson row 5.38 s; TV-L1 row
            // 0.8-1.3 s at 96^3 scaled by (128/96)^3 = 2.37 gives 1.9-3.1 s
            // (2.5 used); extract+decimate row 0.4-0.9 s at 96^3 scaled gives
            // 0.95-2.13 s (1.5 used). 5.38 + 2.5 + 1.5 = 9.38, rounded 9.4.
            // The coefficient is scaled by voxels / 128^3 at estimate time.
            coefficients = {9.4, 0.02, 0.0};
            break;
        case ColdTask::Texture:
            // Bake row 550 ms for a 2048^2 page = 4.19 texels-Mpx, i.e.
            // 0.131 s per texel-megapixel. a = 1.0 s covers atlas setup.
            // b = 0.02 s per camera-megapixel for input image access.
            coefficients = {1.0, 0.02, 0.0};
            break;
        case ColdTask::Dem:
            // BuildDem row 1.2-3.5 s for 171k cells; 2.35 / 171000 =
            // 13.7 s per product megacell (applied to product_pixels).
            // b = 0.01 s per camera-megapixel for depth-map reads.
            coefficients = {0.5, 0.01, 13.7};
            break;
        case ColdTask::Ortho:
            // Ortho row 8-30 s for 1025x779 cells x 12 cameras; 19 s /
            // (0.798 M cell x 12 cameras) = 1.98 s per camera-product-
            // megacell. a = 0.5 s for the tile table.
            coefficients = {0.5, 1.98, 0.0};
            break;
        case ColdTask::Generic:
            // Deliberately coarse: a fixed second plus two seconds per
            // camera-megapixel. Only used for a task the table does not know.
            coefficients = {1.0, 2.0, 0.0};
            break;
    }
    return coefficients;
}

ColdEstimate cold_estimate(ColdTask task, const WorkItem& raw) {
    WorkItem work = raw;
    work.cameras = non_negative(raw.cameras);
    work.megapixels = non_negative(raw.megapixels);
    work.neighbours = non_negative(raw.neighbours);
    work.volume_voxels = non_negative(raw.volume_voxels);
    work.product_pixels = non_negative(raw.product_pixels);
    work.texels = non_negative(raw.texels);
    work.detect_cameras = non_negative(raw.detect_cameras);

    const ColdCoefficients co = cold_coefficients(task);
    const double input_mpx = work.cameras * work.megapixels;
    const double detect_mpx = work.detect_cameras * work.megapixels;

    ColdEstimate out;
    switch (task) {
        case ColdTask::DepthMaps: {
            const double neighbours =
                work.neighbours > 0.0 ? work.neighbours : kReferenceNeighbours;
            out.seconds =
                co.a + co.b * input_mpx * (neighbours / kReferenceNeighbours);
            // 96 MB per working megapixel: depth, normals, confidence and the
            // PatchMatch scratch buffers held for the active neighbourhood.
            out.peak_ram_bytes = saturating_bytes(
                static_cast<double>(kDepthMegapixelRamBytes) * input_mpx);
            // 16 B per working pixel: 4 B depth + 12 B normal.
            out.output_bytes = saturating_bytes(input_mpx * 1.0e6 * 16.0);
            break;
        }
        case ColdTask::Model: {
            const double scale =
                work.volume_voxels > 0.0 ? work.volume_voxels / kModelReferenceVoxels : 1.0;
            out.seconds = co.a * scale + co.b * input_mpx;
            out.peak_ram_bytes = saturating_bytes(
                work.volume_voxels * static_cast<double>(kMeshBytesPerVoxel));
            // Rough mesh product: half a point per input pixel, 12 B per vertex.
            out.output_bytes = saturating_bytes(input_mpx * 1.0e6 * 6.0);
            break;
        }
        case ColdTask::Dem:
            out.seconds = co.a + co.b * input_mpx + co.c * product_megapixels(work);
            out.output_bytes = saturating_bytes(work.product_pixels * static_cast<double>(kDemBytesPerCell));
            out.peak_ram_bytes = 2ULL * out.output_bytes;
            break;
        case ColdTask::Ortho:
            out.seconds = co.a + co.b * work.cameras * product_megapixels(work);
            out.output_bytes = saturating_bytes(work.product_pixels * static_cast<double>(kRgbBytesPerCell));
            out.peak_ram_bytes = 2ULL * out.output_bytes;
            break;
        case ColdTask::Texture:
            // 0.131 s per texel-megapixel from the bake row.
            out.seconds = co.a + 0.131 * (work.texels / 1.0e6) + co.b * input_mpx;
            out.output_bytes = saturating_bytes(work.texels * static_cast<double>(kRgbBytesPerCell));
            out.peak_ram_bytes = out.output_bytes + saturating_bytes(input_mpx * 1.0e6 * 16.0);
            break;
        case ColdTask::PointCloud:
            out.seconds = co.a + co.b * input_mpx;
            out.output_bytes = saturating_bytes(input_mpx * 1.0e6 * 6.0);
            out.peak_ram_bytes = saturating_bytes(input_mpx * 1.0e6 * 48.0);
            break;
        case ColdTask::Features:
            out.seconds = co.a + co.b * input_mpx;
            out.output_bytes = saturating_bytes(input_mpx * 144000.0);  // ~2000 keypoints/Mpx at 72 B
            out.peak_ram_bytes = saturating_bytes(input_mpx * 1.0e6 * 24.0);
            break;
        case ColdTask::MatchPhotos:
            // Matching covers every camera; detection only the ones that have
            // no keypoint set yet (work.detect_cameras, 0 when all are done).
            out.seconds = co.a + co.b * input_mpx +
                          kFeatureDetectionSecondsPerCameraMpx * detect_mpx;
            out.output_bytes = saturating_bytes(input_mpx * 288000.0);
            out.peak_ram_bytes = saturating_bytes(input_mpx * 1.0e6 * 24.0);
            break;
        case ColdTask::Generic:
            out.seconds = co.a + co.b * input_mpx;
            out.output_bytes = saturating_bytes(input_mpx * 288000.0);
            out.peak_ram_bytes = saturating_bytes(input_mpx * 1.0e6 * 24.0);
            break;
    }
    if (!(out.seconds >= 0.05)) {
        out.seconds = 0.05;
    }
    out.disk_bytes = saturating_bytes(static_cast<double>(out.output_bytes) * 1.15);
    return out;
}

}  // namespace alphashape::gui::estimates
