#include "common/algorithm/markers/detect_markers_task.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "base/utils/omp.hpp"
#include "common/algorithm/markers/apriltag_decoder.hpp"
#include "common/algorithm/markers/circular_decoder.hpp"
#include "common/algorithm/markers/cross_decoder.hpp"
#include "common/algorithm/markers/marker_detector.hpp"
#include "common/algorithm/markers/project_markers.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/core/marker.hpp"
#include "common/core/mask.hpp"
#include "common/core/types.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape {
namespace {

enum class MarkerKind { CircularCoded, CircularNonCoded, Cross, AprilTag };

struct TargetSpec {
    MarkerKind kind = MarkerKind::CircularCoded;
    int bits = 12;
    AprilTagFamily family = AprilTagFamily::Tag36h11;
};

struct ParsedTarget {
    TargetSpec spec;
    bool circle21_unsupported = false;
    bool valid = false;
};
ParsedTarget parse_target(const std::string& name) {
    ParsedTarget out;
    out.valid = true;
    if (name == "Circular12" || name == "CircularTarget12bit") {
        out.spec.kind = MarkerKind::CircularCoded;
        out.spec.bits = 12;
        return out;
    }
    if (name == "Circular14" || name == "CircularTarget14bit") {
        out.spec.kind = MarkerKind::CircularCoded;
        out.spec.bits = 14;
        return out;
    }
    if (name == "Circular16" || name == "CircularTarget16bit") {
        out.spec.kind = MarkerKind::CircularCoded;
        out.spec.bits = 16;
        return out;
    }
    if (name == "Circular20" || name == "CircularTarget20bit") {
        out.spec.kind = MarkerKind::CircularCoded;
        out.spec.bits = 20;
        return out;
    }
    if (name == "Circular" || name == "CircularTarget") {
        out.spec.kind = MarkerKind::CircularNonCoded;
        return out;
    }
    if (name == "Cross" || name == "CrossTarget") {
        out.spec.kind = MarkerKind::Cross;
        return out;
    }
    if (name == "AprilTag16h5") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Tag16h5;
        return out;
    }
    if (name == "AprilTag25h9") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Tag25h9;
        return out;
    }
    if (name == "AprilTag36h10") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Tag36h10;
        return out;
    }
    if (name == "AprilTag36h11") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Tag36h11;
        return out;
    }
    if (name == "AprilTagStandard41h12") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Standard41h12;
        return out;
    }
    if (name == "AprilTagStandard52h13") {
        out.spec.kind = MarkerKind::AprilTag;
        out.spec.family = AprilTagFamily::Standard52h13;
        return out;
    }
    out.valid = false;
    out.circle21_unsupported = name == "AprilTagCircle21h7";
    return out;
}
const std::vector<std::string>& target_choices() {
    static const std::vector<std::string> choices = {
        "Circular12",          "Circular14",          "Circular16",
        "Circular20",          "Circular",            "Cross",
        "AprilTag16h5",        "AprilTag25h9",        "AprilTag36h10",
        "AprilTag36h11",       "AprilTagStandard41h12",
        "AprilTagStandard52h13", "AprilTagCircle21h7"};
    return choices;
}

std::string valid_target_names() {
    std::string text;
    for (const std::string& choice : target_choices()) {
        if (!text.empty()) {
            text += ", ";
        }
        text += choice;
    }
    return text;
}

FloatImage invert_luminance(const FloatImage& gray) {
    FloatImage out(gray.width(), gray.height(), 0.0F);
    for (int y = 0; y < gray.height(); ++y) {
        for (int x = 0; x < gray.width(); ++x) {
            out.at(x, y) = 1.0F - gray.at(x, y);
        }
    }
    return out;
}

FloatImage box_downsample(const FloatImage& source, int factor) {
    if (factor <= 1) {
        return source;
    }
    const int width = source.width() / factor;
    const int height = source.height() / factor;
    FloatImage out(width, height, 0.0F);
    const double total = static_cast<double>(factor * factor);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int dy = 0; dy < factor; ++dy) {
                for (int dx = 0; dx < factor; ++dx) {
                    sum += source.at(x * factor + dx, y * factor + dy);
                }
            }
            out.at(x, y) = static_cast<float>(sum / total);
        }
    }
    return out;
}
struct Detection {
    int code = -1;
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    double radius = 0.0;
    int hamming = 0;
    bool coded = false;
    std::array<Eigen::Vector2d, 4> corners{};
    bool has_corners = false;
};

struct DetectorParams {
    int max_hamming = 0;
    bool noparity = false;
    double minimum_size = 0.0;
    double minimum_dist = 5.0;
};

BlobParams blob_params(int width, int height, double minimum_size) {
    BlobParams params;
    params.dark_on_light = true;
    params.min_radius =
        std::max(3, static_cast<int>(std::ceil(std::max(0.0, minimum_size))));
    const int cap = std::max(params.min_radius,
                             std::min(128, std::max(3, std::min(width, height) / 3)));
    params.max_radius = cap;
    params.threshold = 0.02F;
    params.min_axis_ratio = 0.3;
    return params;
}
std::vector<Detection> detect_circular(const FloatImage& image,
                                        const TargetSpec& spec,
                                        const DetectorParams& params) {
    std::vector<Detection> out;
    const BlobParams blob =
        blob_params(image.width(), image.height(), params.minimum_size);
    const std::vector<MarkerCandidate> candidates = detect_blobs(image, blob);
    if (spec.kind == MarkerKind::CircularNonCoded) {
        for (const MarkerCandidate& candidate : candidates) {
            if (candidate.radius_px < params.minimum_size) {
                continue;
            }
            Detection detection;
            detection.center = candidate.center;
            detection.radius = candidate.radius_px;
            out.push_back(detection);
        }
        return out;
    }
    const GradientField gradient = compute_gradient_field(image);
    for (const MarkerCandidate& candidate : candidates) {
        const auto decoded = decode_circular(image, gradient, candidate, spec.bits,
                                             params.max_hamming);
        if (!decoded || decoded->radius_px < params.minimum_size) {
            continue;
        }
        Detection detection;
        detection.code = params.noparity ? (decoded->code / 2) * 2 : decoded->code;
        detection.center = decoded->center;
        detection.radius = decoded->radius_px;
        detection.hamming = decoded->hamming;
        detection.coded = true;
        out.push_back(detection);
    }
    return out;
}

std::vector<Detection> detect_tags(const FloatImage& image, const TargetSpec& spec,
                                    const DetectorParams& params) {
    std::vector<Detection> out;
    QuadParams quad;
    quad.dark_on_light = true;
    for (const CodedTarget& target :
         decode_tags(image, spec.family, params.max_hamming, quad)) {
        if (target.radius_px < params.minimum_size) {
            continue;
        }
        Detection detection;
        detection.code = target.code;
        detection.center = target.center;
        detection.radius = target.radius_px;
        detection.hamming = target.hamming;
        detection.coded = true;
        detection.corners = target.corners;
        detection.has_corners = target.has_corners;
        out.push_back(detection);
    }
    return out;
}
std::vector<Detection> detect_cross_targets(const FloatImage& image,
                                             const DetectorParams& params) {
    CrossParams cross;
    cross.inverted = false;
    cross.minimum_size = params.minimum_size;
    cross.minimum_dist = params.minimum_dist;
    std::vector<Detection> out;
    for (const CodedTarget& target : detect_crosses(image, cross)) {
        if (target.radius_px < params.minimum_size) {
            continue;
        }
        Detection detection;
        detection.center = target.center;
        detection.radius = target.radius_px;
        out.push_back(detection);
    }
    return out;
}

std::vector<Detection> detect_in_image(const FloatImage& image,
                                        const TargetSpec& spec,
                                        const DetectorParams& params) {
    if (spec.kind == MarkerKind::AprilTag) {
        return detect_tags(image, spec, params);
    }
    if (spec.kind == MarkerKind::Cross) {
        return detect_cross_targets(image, params);
    }
    return detect_circular(image, spec, params);
}
std::vector<Detection> suppress_duplicates(std::vector<Detection> detections,
                                            double minimum_dist) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  if (a.coded != b.coded) {
                      return a.coded < b.coded;
                  }
                  if (a.code != b.code) {
                      return a.code < b.code;
                  }
                  if (a.hamming != b.hamming) {
                      return a.hamming < b.hamming;
                  }
                  return a.radius > b.radius;
              });
    std::vector<Detection> out;
    for (const Detection& detection : detections) {
        bool suppressed = false;
        for (const Detection& kept : out) {
            if (kept.coded != detection.coded || kept.code != detection.code) {
                continue;
            }
            if ((kept.center - detection.center).norm() < minimum_dist) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            out.push_back(detection);
        }
    }
    std::sort(out.begin(), out.end(), [](const Detection& a, const Detection& b) {
        if (a.center.y() != b.center.y()) {
            return a.center.y() < b.center.y();
        }
        return a.center.x() < b.center.x();
    });
    return out;
}
void refine_full_resolution(std::vector<Detection>& detections, double scale,
                            const TargetSpec& spec, const FloatImage& full,
                            const GradientField& gradient, int max_hamming) {
    if (!(scale > 1.0)) {
        return;
    }
    for (Detection& detection : detections) {
        detection.center *= scale;
        detection.radius *= scale;
        if (spec.kind == MarkerKind::Cross) {
            if (auto refined =
                    refine_cross_center(full, detection.center, detection.radius)) {
                detection.center = *refined;
            }
        } else if (spec.kind == MarkerKind::AprilTag) {
            if (!detection.has_corners) {
                continue;
            }
            MarkerCandidate quad;
            quad.center = detection.center;
            quad.radius_px = detection.radius;
            for (std::size_t i = 0; i < 4; ++i) {
                quad.corners.push_back(detection.corners[i] * scale);
            }
            const auto decoded =
                decode_tag(full, gradient, quad, tag_family(spec.family), max_hamming);
            if (decoded) {
                detection.center = decoded->center;
                detection.radius = decoded->radius_px;
                detection.hamming = decoded->hamming;
                detection.corners = decoded->corners;
                detection.has_corners = decoded->has_corners;
            }
        } else {
            MarkerCandidate seed;
            seed.center = detection.center;
            seed.radius_px = detection.radius;
            seed.axis_ratio = 1.0;
            if (auto refined = refine_ellipse_center(gradient, seed)) {
                detection.center = *refined;
            }
        }
    }
}
std::filesystem::path resolve_frame_path(const Document& document,
                                         const std::string& path) {
    const std::filesystem::path recorded(path);
    if (recorded.is_absolute() || document.path().empty()) {
        return recorded;
    }
    return std::filesystem::path(document.path()).parent_path() / recorded;
}

const Frame* frame_of_camera(const Chunk& chunk, std::int32_t camera_id) {
    for (const auto& frame : chunk.frames()) {
        if (frame && frame->camera_id() == camera_id) {
            return frame.get();
        }
    }
    return nullptr;
}

const Mask* mask_of_camera(const Chunk& chunk, std::int32_t camera_id) {
    for (const Mask& mask : chunk.masks()) {
        if (mask.camera_id == camera_id) {
            return &mask;
        }
    }
    return nullptr;
}

bool mask_allows(const Mask& mask, const Eigen::Vector2d& centre) {
    const int x = static_cast<int>(std::floor(centre.x()));
    const int y = static_cast<int>(std::floor(centre.y()));
    if (x < 0 || y < 0 || x >= mask.width || y >= mask.height) {
        return true;
    }
    const std::uint8_t value =
        mask.data[static_cast<std::size_t>(y * mask.width + x)];
    return mask.invert ? value == 0 : value != 0;
}

bool contains(const std::vector<std::string>& list, const std::string& value) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

bool frame_matches(const Frame& frame,
                   const std::vector<std::string>& filters) {
    if (filters.empty()) {
        return true;
    }
    const std::filesystem::path path(frame.path());
    const std::string filename = path.filename().string();
    const std::string stem = path.stem().string();
    for (const std::string& filter : filters) {
        if (filter == frame.path() || filter == filename || filter == stem) {
            return true;
        }
    }
    return false;
}
struct Assignment {
    std::string label;
    Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
    double radius = 0.0;
    int hamming = 0;
    bool existing = false;
    std::int32_t camera_id = -1;
};

struct CameraPlan {
    std::vector<Assignment> assignments;
    std::size_t created = 0;
    std::size_t dropped = 0;
};

struct LabelState {
    std::set<std::string> used;
    int next_point = 1;

    std::string next_point_label() {
        while (true) {
            const std::string label = "point " + std::to_string(next_point);
            ++next_point;
            if (used.insert(label).second) {
                return label;
            }
        }
    }
};

using Predictions = std::vector<std::pair<std::string, Eigen::Vector2d>>;
using ProjectedCameras = std::map<std::string, std::set<std::int32_t>>;
CameraPlan plan_camera(const std::vector<Detection>& detections,
                       std::int32_t camera_id, const Predictions& predictions,
                       const std::map<std::string, std::int32_t>& ids,
                       const ProjectedCameras& projected, double maximum_residual,
                       bool create_markers, LabelState& labels) {
    CameraPlan plan;
    std::map<std::string, std::size_t> best;
    const auto consider = [&](const Assignment& assignment) {
        const auto found = best.find(assignment.label);
        if (found == best.end()) {
            best[assignment.label] = plan.assignments.size();
            plan.assignments.push_back(assignment);
            return;
        }
        Assignment& current = plan.assignments[found->second];
        if (assignment.hamming < current.hamming ||
            (assignment.hamming == current.hamming &&
             assignment.radius > current.radius)) {
            current = assignment;
        }
        ++plan.dropped;  // one of the two competing detections is discarded
    };
    for (const Detection& detection : detections) {
        if (detection.coded) {
            const std::string label = "target " + std::to_string(detection.code);
            const bool exists = ids.count(label) != 0;
            const auto projections = projected.find(label);
            if (exists && projections != projected.end() &&
                projections->second.count(camera_id) != 0) {
                ++plan.dropped;
                continue;  // the camera already has a measurement of this marker
            }
            if (!exists) {
                if (!create_markers) {
                    ++plan.dropped;
                    continue;
                }
                if (labels.used.insert(label).second) {
                    ++plan.created;
                }
            }
            consider(Assignment{label, detection.center, detection.radius,
                                detection.hamming, exists, camera_id});
            continue;
        }
        int nearest = -1;
        double nearest_distance = maximum_residual;
        for (std::size_t i = 0; i < predictions.size(); ++i) {
            const double distance = (predictions[i].second - detection.center).norm();
            if (distance <= nearest_distance) {
                nearest_distance = distance;
                nearest = static_cast<int>(i);
            }
        }
        if (nearest >= 0) {
            const std::string& label = predictions[static_cast<std::size_t>(nearest)].first;
            const auto projections = projected.find(label);
            if (projections != projected.end() &&
                projections->second.count(camera_id) != 0) {
                ++plan.dropped;
                continue;
            }
            consider(Assignment{label, detection.center, detection.radius, 0, true,
                                camera_id});
            continue;
        }
        if (!create_markers) {
            ++plan.dropped;
            continue;
        }
        const std::string label = labels.next_point_label();
        ++plan.created;
        consider(Assignment{label, detection.center, detection.radius, 0, false,
                            camera_id});
    }
    return plan;
}
}  // namespace

int circular_max_hamming(double tolerance) {
    return tolerance >= 75.0 ? 1 : 0;
}

int apriltag_max_hamming(double tolerance, AprilTagFamily family) {
    const int budget = default_max_hamming(family);
    const int cap = (tag_family(family).min_hamming - 1) / 2;
    const double scaled = tolerance / 50.0 * static_cast<double>(budget);
    const int value = static_cast<int>(std::lround(scaled));
    return std::clamp(value, 0, cap);
}

namespace {
DetectMarkersStats g_last_stats;
DetectMarkersCommandSink g_command_sink;
}  // namespace

DetectMarkersStats detect_markers_last_stats() {
    return g_last_stats;
}

void set_detect_markers_command_sink(DetectMarkersCommandSink sink) {
    g_command_sink = std::move(sink);
}

namespace {

struct CameraJob {
    const Camera* camera = nullptr;
    const Frame* frame = nullptr;
    const Calibration* calibration = nullptr;
    int scale = 1;
};

struct CameraResult {
    std::vector<Detection> detections;
    std::string error;
};

int auto_scale(int width, int height) {
    const int long_side = std::max(width, height);
    return std::max(1, (long_side + 4095) / 4096);
}

CameraResult process_camera(const Document& document, const CameraJob& job,
                            const TargetSpec& spec, const DetectorParams& params,
                            bool inverted, bool filter_mask, const Mask* mask) {
    CameraResult result;
    try {
        const FloatImage luminance =
            read_image(resolve_frame_path(document, job.frame->path()).string())
                .to_luminance();
        FloatImage inverted_image;
        const FloatImage* full = &luminance;
        if (inverted) {
            inverted_image = invert_luminance(luminance);
            full = &inverted_image;
        }
        FloatImage reduced;
        const FloatImage* detect_image = full;
        int scale = job.scale > 0 ? job.scale : auto_scale(full->width(), full->height());
        if (scale > 1 && (full->width() / scale < 8 || full->height() / scale < 8)) {
            scale = 1;
        }
        if (scale > 1) {
            reduced = box_downsample(*full, scale);
            detect_image = &reduced;
        }
        std::vector<Detection> detections =
            suppress_duplicates(detect_in_image(*detect_image, spec, params),
                                params.minimum_dist);
        if (scale > 1) {
            const GradientField gradient = compute_gradient_field(*full);
            refine_full_resolution(detections, scale, spec, *full, gradient,
                                   params.max_hamming);
        }
        if (filter_mask && mask != nullptr) {
            detections.erase(
                std::remove_if(detections.begin(), detections.end(),
                               [mask](const Detection& detection) {
                                   return !mask_allows(*mask, detection.center);
                               }),
                detections.end());
        }
        result.detections = std::move(detections);
    } catch (const std::exception& error) {
        result.error = error.what();
    }
    return result;
}

}  // namespace

void DetectMarkersTask::declare_params(Params& params) const {
    ParamSpec target{"target_type", ParamType::String, std::string{"Circular12"},
                     "Target style. Circular12/14/16/20 are coded ring targets; "
                     "Circular is a non-coded ring or disc; Cross is a checker "
                     "(bowtie) or bar cross; AprilTag names a published family. "
                     "Script aliases CircularTarget12bit/14bit/16bit/20bit, "
                     "CircularTarget and CrossTarget are also accepted."};
    target.choices = target_choices();
    params.declare(target);
    ParamSpec tolerance{"tolerance", ParamType::Float, 50.0,
                        "Error-correction strength 0..100. AprilTag: maps to "
                        "clamp(round(tolerance/50 * family default), 0, (h-1)/2). "
                        "Circular: 1 when tolerance >= 75, else 0."};
    tolerance.minimum = 0.0;
    tolerance.maximum = 100.0;
    params.declare(tolerance);
    params.declare({"inverted", ParamType::Bool, false,
                    "Detect light targets on a dark background (the luminance is "
                    "inverted before detection)."});
    params.declare({"noparity", ParamType::Bool, false,
                    "Circular codes ignore the centre-dot parity bit and report "
                    "code = ring index * 2."});
    ParamSpec minimum_size{"minimum_size", ParamType::Float, 0.0,
                           "Minimum target radius in pixels (0 = decoder default)."};
    minimum_size.minimum = 0.0;
    params.declare(minimum_size);
    ParamSpec minimum_dist{"minimum_dist", ParamType::Float, 5.0,
                           "Non-maximum-suppression distance between detections in "
                           "one image, in pixels."};
    minimum_dist.minimum = 0.0;
    params.declare(minimum_dist);
    ParamSpec maximum_residual{"maximum_residual", ParamType::Float, 5.0,
                               "Attachment radius for non-coded detections to the "
                               "predicted projection of an existing marker, in pixels."};
    maximum_residual.minimum = 0.0;
    params.declare(maximum_residual);
    params.declare({"create_markers", ParamType::Bool, true,
                    "Create a marker for a detection that matches no existing one. "
                    "When false such detections are dropped and counted."});
    params.declare({"cameras", ParamType::StringList, std::vector<std::string>{},
                    "Camera labels to process. Empty means every enabled camera; an "
                    "explicit list also selects unaligned cameras."});
    params.declare({"frames", ParamType::StringList, std::vector<std::string>{},
                    "Frame paths, file names or stems to process. Empty means all."});
    params.declare({"filter_mask", ParamType::Bool, true,
                    "Skip detections whose centre is excluded by the camera mask."});
    ParamSpec downscale{"marker_downscale", ParamType::Int, std::int64_t{0},
                        "Detection downscale factor. 0 selects the smallest integer d "
                        "with long_side / d <= 4096; the centre is refined at full "
                        "resolution either way."};
    downscale.minimum = std::int64_t{0};
    downscale.maximum = std::int64_t{1024};
    params.declare(downscale);
}

std::vector<Stage> DetectMarkersTask::stages() const {
    return {Stage{"detect", 0}, Stage{"apply", 0}};
}

void DetectMarkersTask::apply(Document& document, Params& params,
                              const Progress& progress) {
    const std::string target_text = params.get_string("target_type");
    const ParsedTarget parsed = parse_target(target_text);
    if (parsed.circle21_unsupported) {
        throw ValueError(
            "DetectMarkers: target_type \"AprilTagCircle21h7\" is not supported "
            "yet: its published tags are circular, so no quad candidate exists "
            "(see LIMITATIONS.md)");
    }
    if (!parsed.valid) {
        throw ValueError("DetectMarkers: unknown target_type '" + target_text +
                         "'; valid values are: " + valid_target_names());
    }
    const double tolerance = params.get_float("tolerance");
    if (!(tolerance >= 0.0 && tolerance <= 100.0)) {
        throw ValueError("DetectMarkers: tolerance must be in [0, 100]");
    }
    const double minimum_size = params.get_float("minimum_size");
    if (!(minimum_size >= 0.0)) {
        throw ValueError("DetectMarkers: minimum_size must be non-negative");
    }
    const double minimum_dist = params.get_float("minimum_dist");
    if (!(minimum_dist >= 0.0)) {
        throw ValueError("DetectMarkers: minimum_dist must be non-negative");
    }
    const double maximum_residual = params.get_float("maximum_residual");
    if (!(maximum_residual >= 0.0)) {
        throw ValueError("DetectMarkers: maximum_residual must be non-negative");
    }
    const std::int64_t downscale_value = params.get_int("marker_downscale");
    if (downscale_value < 0 || downscale_value > 1024) {
        throw ValueError("DetectMarkers: marker_downscale must be in [0, 1024]");
    }
    const bool inverted = params.get_bool("inverted");
    const bool noparity = params.get_bool("noparity");
    const bool create_markers = params.get_bool("create_markers");
    const bool filter_mask = params.get_bool("filter_mask");

    DetectorParams detector;
    detector.noparity = noparity;
    detector.minimum_size = minimum_size;
    detector.minimum_dist = minimum_dist;
    if (parsed.spec.kind == MarkerKind::AprilTag) {
        detector.max_hamming = apriltag_max_hamming(tolerance, parsed.spec.family);
    } else if (parsed.spec.kind == MarkerKind::CircularCoded) {
        detector.max_hamming = circular_max_hamming(tolerance);
    }

    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("DetectMarkers needs an active chunk");
    }
    const std::vector<std::string> camera_filter =
        params.get_string_list("cameras");
    const std::vector<std::string> frame_filter =
        params.get_string_list("frames");
    std::vector<std::string> problems;
    for (const std::string& label : camera_filter) {
        bool found = false;
        for (const Camera& camera : chunk->cameras()) {
            if (camera.label == label) {
                found = true;
                break;
            }
        }
        if (!found) {
            problems.push_back("no camera with label '" + label + "'");
        }
    }
    std::vector<CameraJob> jobs;
    for (const Camera& camera : chunk->cameras()) {
        const bool selected = camera_filter.empty()
                                  ? camera.enabled
                                  : contains(camera_filter, camera.label);
        if (!selected) {
            continue;
        }
        const Frame* frame = frame_of_camera(*chunk, camera.id);
        if (frame == nullptr || !frame_matches(*frame, frame_filter)) {
            if (!camera_filter.empty() && frame == nullptr) {
                problems.push_back("camera '" + camera.label +
                                   "' has no linked frame");
            }
            continue;
        }
        if (camera.sensor_id < 0 ||
            camera.sensor_id >= static_cast<std::int32_t>(chunk->sensors().size())) {
            if (!camera_filter.empty()) {
                problems.push_back("camera '" + camera.label + "' has no sensor");
            }
            continue;
        }
        CameraJob job;
        job.camera = &camera;
        job.frame = frame;
        job.calibration =
            &chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)].calibration;
        job.scale = static_cast<int>(downscale_value);
        jobs.push_back(job);
    }
    if (!problems.empty()) {
        throw ValueError("DetectMarkers: " + problems.front());
    }
    if (!frame_filter.empty() && jobs.empty()) {
        throw ValueError("DetectMarkers: frames filter matched no camera");
    }

    progress.update("detect", 0.0);
    const auto started = std::chrono::steady_clock::now();
    std::vector<CameraResult> results(jobs.size());
    const int count = static_cast<int>(jobs.size());
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < count; ++i) {
        const Camera& camera = *jobs[static_cast<std::size_t>(i)].camera;
        const Mask* mask = filter_mask ? mask_of_camera(*chunk, camera.id) : nullptr;
        results[static_cast<std::size_t>(i)] = process_camera(
            document, jobs[static_cast<std::size_t>(i)], parsed.spec, detector,
            inverted, filter_mask, mask);
    }
    for (const CameraResult& result : results) {
        if (!result.error.empty()) {
            throw Error("DetectMarkers: " + result.error);
        }
    }
    progress.update("detect", 1.0);

    progress.update("apply", 0.0);
    std::map<std::string, std::int32_t> ids;
    ProjectedCameras projected;
    LabelState labels;
    for (const Marker& marker : chunk->markers()) {
        ids[marker.label] = marker.id;
        labels.used.insert(marker.label);
        for (const MarkerProjection& projection : marker.projections) {
            projected[marker.label].insert(projection.camera_id);
        }
    }
    DetectMarkersStats stats;
    std::vector<Assignment> commands;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        const CameraJob& job = jobs[i];
        const std::vector<Detection>& detections = results[i].detections;
        stats.detected += detections.size();
        Predictions predictions;
        for (const Marker& marker : chunk->markers()) {
            if (!marker.position.has_value()) {
                continue;
            }
            const auto pixel =
                project_marker(*marker.position, *job.camera, *job.calibration);
            if (pixel) {
                predictions.emplace_back(marker.label, *pixel);
            }
        }
        const CameraPlan plan =
            plan_camera(detections, job.camera->id, predictions, ids, projected,
                        maximum_residual, create_markers, labels);
        stats.created += plan.created;
        stats.dropped += plan.dropped;
        for (const Assignment& assignment : plan.assignments) {
            if (assignment.existing) {
                ++stats.attached;
            }
            commands.push_back(assignment);
        }
    }
    if (!commands.empty()) {
        auto transaction = std::make_unique<CommandTransaction>("Detect Markers");
        for (const Assignment& assignment : commands) {
            transaction->add(make_add_marker_with_projection_command(
                chunk->id(), assignment.label, assignment.camera_id,
                assignment.pixel, false));
        }
        if (g_command_sink) {
            g_command_sink(document, std::move(transaction));
        } else {
            transaction->apply(document);
        }
    }
    progress.update("apply", 1.0);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    g_last_stats = stats;
    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "DetectMarkers: target " << target_text
                            << " max_hamming " << detector.max_hamming;
        log(LogLevel::Info) << "DetectMarkers: detected " << stats.detected
                            << ", created " << stats.created << ", attached "
                            << stats.attached << ", dropped " << stats.dropped
                            << " in " << seconds << " s";
    }
}

}  // namespace alphashape
