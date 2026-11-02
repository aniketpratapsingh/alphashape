#pragma once

/// @file detect_markers_task.hpp
/// @brief DetectMarkers: automatic circular, cross and AprilTag detection
/// (WP-13.8).
///
/// The task detects printed targets in each selected camera, resolves them to
/// markers and adds the image measurements as ONE undoable CommandTransaction.
///
/// ## Tolerances
///
/// `tolerance` (0..100, default 50) maps to the decoders' error-correction
/// budgets:
///
///   * **AprilTag**: `clamp(round(tolerance / 50 * default_max_hamming(family)),
///     0, (family.min_hamming - 1) / 2)`.
///   * **Circular**: `1` when tolerance >= 75, otherwise `0`. The circular
///     tables support only an exact match or one-sector correction.
///
/// ## Target styles
///
/// `Circular12/14/16/20` are coded concentric ring targets; `Circular` is the
/// non-coded ring or disc; `Cross` is the checker/bowtie or bar cross of
/// cross_decoder.hpp; `AprilTag...` are the published families. The script
/// aliases `CircularTarget12bit`, `CircularTarget14bit`, `CircularTarget16bit`,
/// `CircularTarget20bit`, `CircularTarget` and `CrossTarget` are accepted.
///
/// Coded detections resolve a marker by the label `target <code>`. A non-coded
/// detection is attached to the nearest projected position of an existing
/// marker within `maximum_residual` px, or becomes a new `point N` marker when
/// `create_markers` is true. A marker never receives two projections in one
/// camera; when detections compete, the lower Hamming distance wins and then
/// the larger radius.
///
/// `inverted` inverts the luminance before detection, so every style accepts a
/// light target on a dark background. `filter_mask` skips a candidate whose
/// full-resolution centre is excluded by the camera's mask. `marker_downscale`
/// 0 selects the smallest integer d with long_side / d <= 4096; detection runs
/// on the reduced image and the centre is refined again at full resolution.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

#include "common/algorithm/markers/tag_families.hpp"
#include "common/commands/command.hpp"
#include "common/tasks/task.hpp"

namespace alphashape {

int circular_max_hamming(double tolerance);
int apriltag_max_hamming(double tolerance, AprilTagFamily family);

struct DetectMarkersStats {
    std::size_t detected = 0;
    std::size_t created = 0;
    std::size_t attached = 0;
    std::size_t dropped = 0;
};
DetectMarkersStats detect_markers_last_stats();

/// Host/test hook. When set, the task hands its single undoable command here
/// instead of applying it, so the host can execute it on an UndoStack and make
/// the whole run one undo step. An empty function restores direct application.
using DetectMarkersCommandSink =
    std::function<void(Document&, std::unique_ptr<AbstractCommand>)>;
void set_detect_markers_command_sink(DetectMarkersCommandSink sink);

class DetectMarkersTask : public AbstractTask {
public:
    std::string name() const override { return "DetectMarkers"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
