#include "common/algorithm/calibration/detect_fiducials_task.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <system_error>
#include <optional>
#include <string>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/calibration/fiducials.hpp"
#include "common/core/chunk.hpp"
#include "common/core/frame.hpp"
#include "common/core/types.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"

namespace alphashape {
namespace {

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

FiducialDetector parse_detector_name(const std::string& name) {
    if (name == "circle") return FiducialDetector::Circle;
    if (name == "corner") return FiducialDetector::Corner;
    if (name == "frame_corner") return FiducialDetector::FrameCorner;
    throw ValueError("DetectFiducials: unknown detector '" + name + "'");
}

/// One camera's outcome. A failed camera carries only a reason.
struct CameraOutcome {
    std::int32_t id = -1;
    std::string label;
    bool ok = false;
    std::string status;
    std::vector<std::optional<Eigen::Vector2d>> pixels;
    /// Raw marks when the sensor has no layout yet (pixels only).
    std::vector<FiducialMark> marks;
    std::optional<Eigen::Matrix3d> transform;
    double rms_px = 0.0;
    bool ambiguous = false;
};

CameraOutcome run_camera(const Document& document, const Chunk& chunk,
                         const Camera& camera, const Sensor& sensor,
                         const DetectFiducialsOptions& options) {
    CameraOutcome out;
    out.id = camera.id;
    out.label = camera.label;
    const Frame* frame = frame_of_camera(chunk, camera.id);
    if (frame == nullptr) {
        out.status = "no frame";
        return out;
    }
    FloatImage gray;
    try {
        gray = read_image(resolve_frame_path(document, frame->path()).string())
                   .to_luminance();
    } catch (const std::exception& error) {
        out.status = std::string("decode failed: ") + error.what();
        return out;
    }
    FiducialDetectOptions detect;
    detect.detector = options.detector.empty()
                          ? sensor.film.detector
                          : parse_detector_name(options.detector);
    detect.search_corners = sensor.film.search_corners;
    detect.search_sides = sensor.film.search_sides;
    detect.min_radius_px = options.min_radius_px;
    detect.max_radius_px = options.max_radius_px;
    detect.max_residual_px = options.max_residual_px;
    detect.max_anisotropy = options.max_anisotropy;
    detect.scan_orientation = options.scan_orientation;
    const FiducialDetection result = detect_fiducials(gray, sensor.film, detect);
    if (!result.failure.empty()) {
        out.status = result.failure;
        return out;
    }
    out.ok = true;
    out.status = result.transform.has_value() ? "ok" : "ok_pixels_only";
    out.transform = result.transform;
    out.rms_px = result.rms_px;
    out.ambiguous = result.ambiguous;
    // `Camera::fiducial_pixels` is parallel to the sensor's layout, so without
    // a layout it must stay empty; the raw marks go into metadata instead.
    if (!result.by_fiducial.empty()) {
        out.pixels = result.by_fiducial;
    } else {
        out.marks = result.marks;
    }
    return out;
}

}  // namespace

void DetectFiducialsTask::declare_params(Params& params) const {
    ParamSpec detector{"detector", ParamType::String, std::string(),
                       "Mark detector; empty uses the sensor's setting."};
    detector.choices = {"", "circle", "corner", "frame_corner"};
    params.declare(detector);
    ParamSpec min_radius{"min_radius", ParamType::Float, 3.0,
                         "Smallest circle radius searched, pixels."};
    min_radius.minimum = 0.0;
    params.declare(min_radius);
    ParamSpec max_radius{"max_radius", ParamType::Float, 40.0,
                         "Largest circle radius searched, pixels."};
    max_radius.minimum = 0.0;
    params.declare(max_radius);
    ParamSpec residual{"max_residual_px", ParamType::Float, 3.0,
                       "Largest acceptable ordering RMS, pixels."};
    residual.minimum = 0.0;
    params.declare(residual);
    ParamSpec orientation{
        "scan_orientation", ParamType::String, std::string("0"),
        "Assumed film-to-scan orientation: 0, 90, 180 or 270, with an 'm' "
        "suffix when the scan is mirrored. Used only when the layout is "
        "dihedrally symmetric."};
    orientation.choices = {"0", "90", "180", "270",
                           "0m", "90m", "180m", "270m"};
    params.declare(orientation);
    ParamSpec anisotropy{
        "max_anisotropy", ParamType::Float, 0.02,
        "Largest allowed |sigma1/sigma2 - 1| of the film-to-scan map."};
    anisotropy.minimum = 0.0;
    params.declare(anisotropy);
    params.declare({"cameras", ParamType::StringList, std::vector<std::string>{},
                    "Camera ids to process; empty means every film camera."});
}

std::vector<Stage> DetectFiducialsTask::stages() const {
    return {Stage{"detect", 0}, Stage{"apply", 0}};
}

void DetectFiducialsTask::apply(Document& document, Params& params,
                                const Progress& progress) {
    DetectFiducialsOptions options;
    options.detector = params.get_string("detector");
    options.min_radius_px = params.get_float("min_radius");
    options.max_radius_px = params.get_float("max_radius");
    options.max_residual_px = params.get_float("max_residual_px");
    options.max_anisotropy = params.get_float("max_anisotropy");
    options.scan_orientation = params.get_string("scan_orientation");
    for (const std::string& text : params.get_string_list("cameras")) {
        std::int32_t id = 0;
        const char* begin = text.data();
        const char* end = begin + text.size();
        const std::from_chars_result parsed = std::from_chars(begin, end, id);
        if (parsed.ec != std::errc() || parsed.ptr != end || id < 0) {
            throw ValueError("DetectFiducials: invalid camera id '" + text + "'");
        }
        options.cameras.push_back(id);
    }
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("DetectFiducials needs an active chunk");
    }
    Chunk staged = *chunk;
    int candidates = 0;
    int succeeded = 0;
    const double total = static_cast<double>(
        std::max<std::size_t>(std::size_t{1}, chunk->cameras().size()));
    for (std::size_t i = 0; i < chunk->cameras().size(); ++i) {
        const Camera& camera = chunk->cameras()[i];
        if (!options.cameras.empty() &&
            std::find(options.cameras.begin(), options.cameras.end(), camera.id) ==
                options.cameras.end()) {
            continue;
        }
        if (camera.sensor_id < 0 ||
            static_cast<std::size_t>(camera.sensor_id) >= chunk->sensors().size()) {
            continue;
        }
        const Sensor& sensor =
            chunk->sensors()[static_cast<std::size_t>(camera.sensor_id)];
        if (!sensor.film.enabled) {
            continue;
        }
        ++candidates;
        const CameraOutcome outcome =
            run_camera(document, *chunk, camera, sensor, options);
        const std::string prefix = "fiducials/" + std::to_string(camera.id) + "/";
        staged.meta().set(prefix + "status", outcome.status);
        staged.meta().set(prefix + "rms_px", std::to_string(outcome.rms_px));
        staged.meta().set(prefix + "ambiguous", outcome.ambiguous ? "1" : "0");
        if (!outcome.marks.empty()) {
            // No layout: report the raw marks as "u,v;u,v;..." so they survive
            // a save/load without violating the parallel-to-layout contract.
            std::string joined;
            for (const FiducialMark& mark : outcome.marks) {
                if (!joined.empty()) {
                    joined += ';';
                }
                joined += std::to_string(mark.pixel.x()) + "," +
                          std::to_string(mark.pixel.y());
            }
            staged.meta().set(prefix + "marks", joined);
        }
        if (outcome.ok) {
            Camera& target = staged.cameras()[i];
            target.fiducial_pixels = outcome.pixels;
            target.film_transform = outcome.transform;
            ++succeeded;
        } else {
            log(LogLevel::Info) << "DetectFiducials: camera '" << camera.label
                                << "' failed: " << outcome.status;
        }
        progress.update("detect", (static_cast<double>(i) + 1.0) / total);
    }
    if (candidates == 0) {
        // A project without film sensors has nothing to detect. Recording the
        // run parameters and leaving the chunk untouched is the same no-op the
        // task-schema parity harness expects from every task.
        progress.update("apply", 1.0);
        return;
    }
    if (succeeded == 0) {
        throw Error("DetectFiducials: every camera failed to detect fiducials");
    }
    progress.update("apply", 0.0);
    *chunk = std::move(staged);
    progress.update("apply", 1.0);
    log(LogLevel::Info) << "DetectFiducials: " << succeeded << " of " << candidates
                        << " cameras detected";
}

}  // namespace alphashape
