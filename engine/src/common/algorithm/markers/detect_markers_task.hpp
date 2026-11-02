#pragma once

/// @file detect_markers_task.hpp
/// @brief DetectMarkers: automatic circular, cross and AprilTag detection
/// (WP-13.8).
///
/// The task detects printed targets in each selected camera, resolves them to
/// markers and builds ONE CommandTransaction("Detect Markers"). Like
/// ImportMarkers, the task applies that transaction directly to the document;
/// it is a single atomic mutation, and a host that wants an undo entry can call
/// plan_detect_markers() and execute the returned command on its UndoStack.
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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

/// Everything the task reads from `Params`, resolved and converted. Hosts can
/// build this directly to stage a detection without a task run.
struct DetectMarkersOptions {
    std::string target_type = "Circular12";
    double tolerance = 50.0;
    bool inverted = false;
    bool noparity = false;
    double minimum_size = 0.0;
    double minimum_dist = 5.0;
    double maximum_residual = 5.0;
    bool create_markers = true;
    std::vector<std::string> cameras;
    std::vector<std::string> frames;
    bool filter_mask = true;
    int marker_downscale = 0;
};

/// The staged result of one detection pass. `command` is a single
/// CommandTransaction("Detect Markers") holding every added marker and
/// projection, or null when the pass produced no assignment. The caller owns
/// it: the task applies it directly, exactly like ImportMarkers, while a host
/// or test may instead pass it to an UndoStack to make the run one undo step.
struct DetectMarkersPlan {
    std::unique_ptr<AbstractCommand> command;
    DetectMarkersStats stats;
    int max_hamming = 0;
};

/// Detect, resolve and stage a command for `chunk_id`. Pure with respect to the
/// document: it reads frames and the chunk but never applies `command`.
///
/// Memory rule: an image above 8 MP makes the camera loop run one camera at a
/// time (the outer OpenMP `if` is false), so only one full-resolution buffer
/// per large camera exists; each camera's internal image passes may still use
/// all threads. A smaller image keeps the normal parallel camera loop.
DetectMarkersPlan plan_detect_markers(const Document& document,
                                      std::int32_t chunk_id,
                                      const DetectMarkersOptions& options);

class DetectMarkersTask : public AbstractTask {
public:
    std::string name() const override { return "DetectMarkers"; }
    void declare_params(Params& params) const override;
    TargetType target_type() const override { return TargetType::Chunk; }
    std::vector<Stage> stages() const override;
    void apply(Document& document, Params& params, const Progress& progress) override;
};

}  // namespace alphashape
