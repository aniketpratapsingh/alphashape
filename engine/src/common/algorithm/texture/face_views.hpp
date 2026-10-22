#pragma once

// Face-to-view selection: for every face of the mesh, decide which input
// photographs actually see it, and pick the best one.
//
// The texturing stage needs, per face, both the single best camera to sample
// from (for the mosaic mode) and the full set of cameras that see it without
// being occluded (for the average mode and for colour correction across seams).
//
// This header declares the interface; select_face_views() implements it.

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "image/image.hpp"
#include "model/camera/calibration.hpp"
#include "model/mesh/mesh.hpp"

namespace alphashape::texturing {

/// One input photograph and the camera geometry it was taken with, at FULL
/// resolution. The texturing stage scales nothing itself; a caller that wants
/// smaller occlusion buffers sets FaceViewOptions::zbuffer_downscale.
struct TexView {
    /// Camera id this view came from; -1 marks an unused slot.
    std::int32_t camera_id = -1;
    /// The photograph, full resolution. Not owned: it must outlive the call.
    const Image* image = nullptr;
    /// The camera's full-resolution calibration.
    Calibration calibration;
    /// World-to-camera rotation.
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    /// World-to-camera translation.
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

/// Parameters for the view-selection pass.
struct FaceViewOptions {
    /// Occlusion buffers are rendered at 1/n the image resolution. A larger
    /// value is faster but a more conservative (blurrier) occlusion test.
    int zbuffer_downscale = 2;
    /// Reject faces seen at a grazing angle: a face is a candidate in this
    /// view only when the view direction meets its normal with a cosine of at
    /// least this value.
    double min_view_cos = 0.15;
    /// Weight of the label-smoothing pass relative to the data term. Zero
    /// disables smoothing.
    double smoothness = 1.0;
    /// Number of label-smoothing sweeps to run.
    int smoothing_passes = 6;
};

/// Per-face view assignment.
struct FaceViews {
    /// Per face: index into the views vector of its best camera, or -1 if no
    /// camera sees it well enough.
    std::vector<int> best;
    /// Per face: every view index that sees it unoccluded and within the
    /// grazing-angle limit, ordered best-first by data term (largest projected
    /// area first, ties to the lower view index); `best[f]` is its first entry,
    /// so this is a preference order, not ascending view order.
    std::vector<std::vector<int>> visible;
};

/// Choose each face's best and visible views. The occlusion test samples the
/// centroid and the three vertex-to-centroid midpoints against the low-res
/// z-buffer and keeps a face when a majority of those samples agree on the
/// surface depth (see the .cpp for the exact tolerance). Deterministic: the
/// result does not depend on the OpenMP thread count.
FaceViews select_face_views(const Mesh& mesh, const std::vector<TexView>& views,
                            const FaceViewOptions& options);

}  // namespace alphashape::texturing
