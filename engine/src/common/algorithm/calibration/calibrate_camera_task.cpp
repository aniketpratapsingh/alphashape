#include "common/algorithm/calibration/calibrate_camera_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "common/algorithm/calibration/board_detector.hpp"
#include "common/core/chunk.hpp"
#include "common/core/frame.hpp"
#include "common/core/types.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"

namespace alphashape {
namespace {

constexpr int kMinimumViews = 3;
constexpr int kMinimumCorners = 6;

std::filesystem::path resolve_frame_path(const Document& document,
                                         const std::string& path) {
    const std::filesystem::path recorded(path);
    if (recorded.is_absolute() || document.path().empty()) {
        return recorded;
    }
    return std::filesystem::path(document.path()).parent_path() / recorded;
}

const Chunk* find_chunk_by_id(const Document& document, std::int32_t chunk_id) {
    for (const auto& workspace : document.workspaces()) {
        if (!workspace) {
            continue;
        }
        for (const auto& chunk : workspace->chunks()) {
            if (chunk && chunk->id() == chunk_id) {
                return chunk.get();
            }
        }
    }
    return nullptr;
}

const Frame* frame_of_camera(const Chunk& chunk, std::int32_t camera_id) {
    for (const auto& frame : chunk.frames()) {
        if (frame && frame->camera_id() == camera_id) {
            return frame.get();
        }
    }
    return nullptr;
}

/// Full-precision text for a metadata value.
std::string meta_number(double value) {
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

std::string joined(const std::vector<std::string>& values) {
    std::string text;
    for (const std::string& value : values) {
        if (!text.empty()) {
            text += ",";
        }
        text += value;
    }
    return text;
}

/// Decode and detect the boards of every camera on one sensor that has a
/// frame. Alignment is not required: AddPhotos leaves `Camera::enabled` false
/// until SfM runs, and calibration from a chessboard needs no pose. Returns an
/// empty string on success, or a reason when no image/board was usable. Throws
/// for unreadable images and for cameras on the sensor that disagree on the
/// image size.
std::string collect_sensor_views(const Document& document, const Chunk& chunk,
                                 const Sensor& sensor, std::size_t sensor_index,
                                 const BoardSpec& spec, int& width, int& height,
                                 std::vector<CalibrationView>& views) {
    bool sized = false;
    for (const Camera& camera : chunk.cameras()) {
        if (camera.sensor_id != static_cast<std::int32_t>(sensor_index)) {
            continue;
        }
        const Frame* frame = frame_of_camera(chunk, camera.id);
        if (frame == nullptr) {
            continue;
        }
        FloatImage gray;
        try {
            gray = read_image(resolve_frame_path(document, frame->path()).string())
                       .to_luminance();
        } catch (const std::exception& error) {
            throw Error("CalibrateCamera: " + std::string(error.what()));
        }
        if (!sized) {
            width = gray.width();
            height = gray.height();
            sized = true;
        } else if (gray.width() != width || gray.height() != height) {
            throw ValueError(
                "CalibrateCamera: cameras on sensor '" + sensor.label +
                "' have differing image sizes");
        }
        const std::optional<BoardView> board = detect_board(gray, spec);
        if (!board) {
            continue;
        }
        CalibrationView view;
        view.camera_id = camera.id;
        view.board_points = board->board_points();
        view.pixels.reserve(board->corners.size());
        for (const BoardCorner& corner : board->corners) {
            view.pixels.push_back(corner.pixel);
        }
        if (view.board_points.size() >= static_cast<std::size_t>(kMinimumCorners)) {
            views.push_back(std::move(view));
        }
    }
    if (!sized) {
        return "no camera with a decoded frame";
    }
    return {};
}

}  // namespace

CalibrateCameraPlan plan_calibrate_camera(const Document& document,
                                          std::int32_t chunk_id,
                                          const CalibrateCameraOptions& options) {
    if (options.board_cols < 2 || options.board_rows < 2) {
        throw ValueError(
            "CalibrateCamera: board_cols and board_rows must be at least 2");
    }
    if (!std::isfinite(options.square_size) || !(options.square_size > 0.0)) {
        throw ValueError("CalibrateCamera: square_size must be finite and positive");
    }
    const Chunk* chunk = find_chunk_by_id(document, chunk_id);
    if (chunk == nullptr) {
        throw Error("CalibrateCamera needs an active chunk");
    }
    const BoardSpec spec{options.board_cols, options.board_rows,
                         options.square_size};
    CalibrateCameraPlan plan;
    for (std::size_t s = 0; s < chunk->sensors().size(); ++s) {
        const Sensor& sensor = chunk->sensors()[s];
        SensorCalibration entry;
        entry.sensor_id = static_cast<std::int32_t>(s);
        entry.label = sensor.label;
        std::vector<CalibrationView> views;
        int width = 0;
        int height = 0;
        // A frame that cannot be read, or a sensor whose frames disagree on
        // size, disqualifies this sensor only; the remaining sensors still get
        // their chance to calibrate.
        try {
            entry.skipped = collect_sensor_views(document, *chunk, sensor, s,
                                                 spec, width, height, views);
        } catch (const std::exception& error) {
            entry.skipped = error.what();
        }
        if (entry.skipped.empty() &&
            views.size() < static_cast<std::size_t>(kMinimumViews)) {
            entry.skipped = "need at least 3 views with a detected board";
        }
        if (entry.skipped.empty()) {
            try {
                entry.result =
                    calibrate_camera(width, height, views, sensor.calibration,
                                     options.fit, CalibrateOptions{});
                entry.solved = true;
                ++plan.solved;
            } catch (const std::exception& error) {
                entry.skipped = error.what();
            }
        }
        plan.sensors.push_back(std::move(entry));
    }
    return plan;
}

void CalibrateCameraTask::declare_params(Params& params) const {
    ParamSpec cols{"board_cols", ParamType::Int, std::int64_t{9},
                   "Inner corners across the chessboard."};
    cols.minimum = std::int64_t{2};
    params.declare(cols);
    ParamSpec rows{"board_rows", ParamType::Int, std::int64_t{6},
                   "Inner corners down the chessboard."};
    rows.minimum = std::int64_t{2};
    params.declare(rows);
    ParamSpec square{"square_size", ParamType::Float, 0.025,
                     "Chessboard square size in metres."};
    square.minimum = 0.0;
    params.declare(square);
    // Same fit defaults as OptimizeCameras: the principal point, focal length
    // and low-order radial/tangential terms are free; the affine pair, the
    // highest radial term and the higher-order tangential pair are held.
    params.declare({"fit_f", ParamType::Bool, true, "Adjust the focal length."});
    params.declare({"fit_cx", ParamType::Bool, true, "Adjust the principal point X offset."});
    params.declare({"fit_cy", ParamType::Bool, true, "Adjust the principal point Y offset."});
    params.declare({"fit_b1", ParamType::Bool, false, "Adjust the aspect (affinity) term b1."});
    params.declare({"fit_b2", ParamType::Bool, false, "Adjust the skew term b2."});
    params.declare({"fit_k1", ParamType::Bool, true, "Adjust the radial distortion k1."});
    params.declare({"fit_k2", ParamType::Bool, true, "Adjust the radial distortion k2."});
    params.declare({"fit_k3", ParamType::Bool, true, "Adjust the radial distortion k3."});
    params.declare({"fit_k4", ParamType::Bool, false, "Adjust the radial distortion k4."});
    params.declare({"fit_p1", ParamType::Bool, true, "Adjust the tangential distortion p1."});
    params.declare({"fit_p2", ParamType::Bool, true, "Adjust the tangential distortion p2."});
    params.declare(
        {"fit_p3", ParamType::Bool, false, "Adjust the higher-order tangential distortion p3."});
    params.declare(
        {"fit_p4", ParamType::Bool, false, "Adjust the higher-order tangential distortion p4."});
    params.declare({"apply", ParamType::Bool, true,
                    "Write the solved calibration to the sensor. When false only "
                    "the report metadata is written."});
}

std::vector<Stage> CalibrateCameraTask::stages() const {
    return {Stage{"prepare", 0}, Stage{"calibrate", 0}, Stage{"apply", 0}};
}

void CalibrateCameraTask::apply(Document& document, Params& params,
                                const Progress& progress) {
    CalibrateCameraOptions options;
    options.board_cols = static_cast<int>(params.get_int("board_cols"));
    options.board_rows = static_cast<int>(params.get_int("board_rows"));
    options.square_size = params.get_float("square_size");
    for (int k = 0; k < kCalibrationParamCount; ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        options.fit[kk] = params.get_bool(
            "fit_" + std::string(kCalibrationParamNames[kk]));
    }
    options.apply = params.get_bool("apply");
    progress.update("prepare", 0.0);
    progress.update("calibrate", 0.0);
    const CalibrateCameraPlan plan =
        plan_calibrate_camera(document, document.active_chunk_id(), options);
    progress.update("calibrate", 1.0);
    if (plan.solved == 0) {
        if (Log::enabled(LogLevel::Info)) {
            for (const SensorCalibration& entry : plan.sensors) {
                log(LogLevel::Info)
                    << "CalibrateCamera: sensor '" << entry.label
                    << "' skipped: " << entry.skipped;
            }
        }
        throw Error("calibrate_camera: need at least 3 views with a detected board");
    }
    progress.update("apply", 0.0);
    Chunk* chunk = document.active_chunk();
    if (chunk == nullptr) {
        throw Error("CalibrateCamera needs an active chunk");
    }
    Chunk staged = *chunk;
    for (const SensorCalibration& entry : plan.sensors) {
        const std::string prefix = "calibrate_camera." + entry.label + ".";
        if (!entry.solved) {
            staged.meta().set(prefix + "skipped", entry.skipped);
            continue;
        }
        const CalibrationResult& result = entry.result;
        staged.meta().set(prefix + "rms_px", meta_number(result.rms_px));
        staged.meta().set(prefix + "max_px", meta_number(result.max_px));
        staged.meta().set(prefix + "views_used",
                          std::to_string(result.views_used));
        staged.meta().set(prefix + "pruned", joined(result.pruned));
        for (int k = 0; k < kCalibrationParamCount; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            if (options.fit[kk]) {
                staged.meta().set(
                    prefix + "sigma." + kCalibrationParamNames[kk],
                    meta_number(result.sigma[kk]));
            }
        }
        if (options.apply && entry.sensor_id >= 0 &&
            entry.sensor_id < static_cast<std::int32_t>(staged.sensors().size())) {
            staged.sensors()[static_cast<std::size_t>(entry.sensor_id)]
                .calibration = result.calibration;
        }
    }
    *chunk = std::move(staged);
    progress.update("apply", 1.0);
    if (Log::enabled(LogLevel::Info)) {
        log(LogLevel::Info) << "CalibrateCamera: " << plan.solved << " of "
                            << plan.sensors.size() << " sensors calibrated";
    }
}

}  // namespace alphashape
