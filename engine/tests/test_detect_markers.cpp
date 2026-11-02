// WP-13.8: cross targets, non-coded targets and the DetectMarkers task.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "base/utils/log.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/feature/dog_detector.hpp"
#include "common/algorithm/markers/circular_decoder.hpp"
#include "common/algorithm/markers/cross_decoder.hpp"
#include "common/algorithm/markers/detect_markers_task.hpp"
#include "common/algorithm/markers/marker_detector.hpp"
#include "common/algorithm/markers/project_markers.hpp"
#include "common/commands/command.hpp"
#include "common/commands/model_commands.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/core/mask.hpp"
#include "common/core/types.hpp"
#include "common/tasks/registry.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "make_apriltag.hpp"
#include "make_circular_target.hpp"
#include "make_marker_sheet.hpp"
#include "perf_gate.hpp"

namespace alphashape {
namespace {

struct CrossTruth {
    double cx = 0.0;
    double cy = 0.0;
    double half = 20.0;
    double angle = 0.0;
};

/// A checker (bowtie) cross: two black and two white quadrants. Coordinates
/// use the calibration convention (the centre of the top-left pixel is
/// (0.5, 0.5), as in make_circular_target.hpp).
FloatImage render_checker_crosses(int width, int height,
                                  const std::vector<CrossTruth>& targets,
                                  float background = 0.5F, float dark = 0.05F,
                                  float light = 0.95F, int supersample = 4) {
    FloatImage image(width, height, background);
    const int samples = std::max(1, supersample);
    const double total = static_cast<double>(samples * samples);
    const double step = 1.0 / static_cast<double>(samples);
    for (const CrossTruth& target : targets) {
        const double c = std::cos(target.angle);
        const double s = std::sin(target.angle);
        const double extent = target.half * std::sqrt(2.0) + 1.0;
        const int x0 = std::max(0, static_cast<int>(std::floor(target.cx - extent)));
        const int y0 = std::max(0, static_cast<int>(std::floor(target.cy - extent)));
        const int x1 = std::min(width - 1, static_cast<int>(std::ceil(target.cx + extent)));
        const int y1 = std::min(height - 1, static_cast<int>(std::ceil(target.cy + extent)));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                double sum = 0.0;
                for (int sy = 0; sy < samples; ++sy) {
                    for (int sx = 0; sx < samples; ++sx) {
                        const double px = static_cast<double>(x) +
                                          (static_cast<double>(sx) + 0.5) * step;
                        const double py = static_cast<double>(y) +
                                          (static_cast<double>(sy) + 0.5) * step;
                        const double dx = px - target.cx;
                        const double dy = py - target.cy;
                        const double u = dx * c + dy * s;
                        const double v = -dx * s + dy * c;
                        if (std::abs(u) > target.half || std::abs(v) > target.half) {
                            sum += static_cast<double>(background);
                        } else {
                            sum += static_cast<double>(
                                (u > 0.0) == (v > 0.0) ? light : dark);
                        }
                    }
                }
                image.at(x, y) = static_cast<float>(sum / total);
            }
        }
    }
    return image;
}
void write_gray_png(const std::string& path, const FloatImage& image) {
    Image out(image.width(), image.height(), 1);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const double value = static_cast<double>(image.at(x, y)) * 255.0;
            out.at(x, y, 0) = static_cast<std::uint8_t>(
                std::lround(std::clamp(value, 0.0, 255.0)));
        }
    }
    write_png(path, out);
}

Camera& add_camera(Chunk& chunk, const std::string& label, int width, int height,
                   const Transform& pose, double focal = 500.0) {
    Sensor sensor;
    sensor.label = "sensor";
    sensor.type = CameraType::Frame;
    sensor.width = width;
    sensor.height = height;
    sensor.calibration =
        Calibration::initial(CameraType::Frame, width, height, focal);
    chunk.sensors().push_back(sensor);
    Camera camera;
    camera.id = chunk.allocate_id();
    camera.sensor_id = static_cast<std::int32_t>(chunk.sensors().size()) - 1;
    camera.label = label;
    camera.enabled = true;
    camera.pose = pose;
    chunk.cameras().push_back(camera);
    return chunk.cameras().back();
}

void add_frame(Chunk& chunk, std::int32_t camera_id, const std::string& path,
               int width, int height) {
    auto frame = std::make_shared<Frame>(path, width, height);
    frame->set_id(chunk.allocate_id());
    frame->set_camera_id(camera_id);
    chunk.frames().push_back(frame);
}

Params params_for(const std::string& task) {
    return make_params(builtin_registry().require(task));
}

void run_named(Document& document, const std::string& task, Params& params) {
    run_task(builtin_registry(), document, task, params);
}

Marker* find_marker(Chunk& chunk, const std::string& label) {
    for (Marker& marker : chunk.markers()) {
        if (marker.label == label) {
            return &marker;
        }
    }
    return nullptr;
}
/// Capture every log line for the lifetime of the object (the sink is
/// process-wide; gtest runs one test at a time).
class ScopedLogCapture {
public:
    ScopedLogCapture() {
        previous_level_ = Log::level();
        if (!Log::enabled(LogLevel::Trace)) {
            Log::set_level(LogLevel::Trace);
        }
        Log::set_sink([this](LogLevel level, const std::string& message) {
            lines_.push_back({level, message});
        });
    }
    ~ScopedLogCapture() {
        Log::set_sink({});
        Log::set_level(previous_level_);
    }
    ScopedLogCapture(const ScopedLogCapture&) = delete;
    ScopedLogCapture& operator=(const ScopedLogCapture&) = delete;

    std::string text(LogLevel level) const {
        std::string out;
        for (const auto& [lvl, message] : lines_) {
            if (lvl == level) {
                out += message;
                out += '\n';
            }
        }
        return out;
    }

private:
    std::vector<std::pair<LogLevel, std::string>> lines_;
    LogLevel previous_level_ = LogLevel::Info;
};

/// A document with `camera_count` enabled cameras sharing one sensor and one
/// pose, each with its own frame path. The caller writes the image files.
struct DetectScene {
    Document document;
    Chunk* chunk = nullptr;
    std::vector<std::string> paths;

    DetectScene() = default;
    DetectScene(DetectScene&&) = default;
    DetectScene& operator=(DetectScene&&) = default;
    DetectScene(const DetectScene&) = delete;
    DetectScene& operator=(const DetectScene&) = delete;
    ~DetectScene() {
        for (const std::string& path : paths) {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    }
};

DetectScene make_scene(int width, int height, int camera_count,
                       const Transform& pose = Transform::Identity()) {
    DetectScene scene;
    Chunk& chunk = scene.document.workspace().add_chunk("Chunk 1");
    scene.document.set_active_chunk_id(chunk.id());
    scene.chunk = &chunk;
    for (int i = 0; i < camera_count; ++i) {
        const std::string path =
            platform::unique_temp_path("alphashape_detect_", ".png");
        Camera& camera = add_camera(chunk, "cam" + std::to_string(i), width,
                                    height, pose);
        add_frame(chunk, camera.id, path, width, height);
        scene.paths.push_back(path);
    }
    return scene;
}
std::uint32_t read_circular_word(int bits, std::size_t index) {
    const std::string path = std::string(ALPHASHAPE_TEST_FIXTURE_DIR) +
                             "/markers/circular_codes_N" + std::to_string(bits) +
                             ".txt";
    std::ifstream input(path);
    EXPECT_TRUE(input.good()) << "missing fixture " << path;
    std::vector<std::uint32_t> words;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::uint32_t word = 0;
        for (const char character : line) {
            if (character == '0' || character == '1') {
                word = (word << 1) | static_cast<std::uint32_t>(character == '1');
            }
        }
        words.push_back(word);
    }
    return index < words.size() ? words[index] : 0U;
}

TEST(Markers, DetectCrossSheet) {
    const int width = 1500;
    const int height = 1200;
    std::mt19937 engine(1234U);
    std::vector<CrossTruth> truths;
    for (int i = 0; i < 20; ++i) {
        CrossTruth truth;
        truth.cx = (static_cast<double>(i % 5) + 0.5) * 300.0 +
                   rng::uniform_real(engine, -6.0, 6.0);
        truth.cy = (static_cast<double>(i / 5) + 0.5) * 300.0 +
                   rng::uniform_real(engine, -6.0, 6.0);
        truth.half = 38.0;
        truth.angle = rng::uniform_real(engine, 0.0, std::numbers::pi_v<double>);
        truths.push_back(truth);
    }
    FloatImage image = render_checker_crosses(width, height, truths);
    image = gaussian_blur(image, 1.5F);
    test::add_noise(image, 0.01, 99U);
    const std::vector<CodedTarget> found = detect_crosses(image);
    ASSERT_EQ(found.size(), truths.size());
    double squared = 0.0;
    for (const CrossTruth& truth : truths) {
        const Eigen::Vector2d expected(truth.cx, truth.cy);
        double best = std::numeric_limits<double>::infinity();
        for (const CodedTarget& target : found) {
            best = std::min(best, (target.center - expected).norm());
        }
        EXPECT_LE(best, 3.0) << "cross at " << truth.cx << ", " << truth.cy;
        squared += best * best;
    }
    const double rms =
        std::sqrt(squared / static_cast<double>(truths.size()));
    std::printf("[DetectCrossSheet] %zu crosses, centre RMS %.4f px\n",
                found.size(), rms);
    EXPECT_LE(rms, 0.2);
}
void paste(FloatImage& sheet, const FloatImage& patch, int x0, int y0) {
    for (int y = 0; y < patch.height(); ++y) {
        for (int x = 0; x < patch.width(); ++x) {
            const int sx = x0 + x;
            const int sy = y0 + y;
            if (sx >= 0 && sy >= 0 && sx < sheet.width() && sy < sheet.height()) {
                sheet.at(sx, sy) = patch.at(x, y);
            }
        }
    }
}

/// Run DetectMarkers on one rendered image and return the marker labels.
std::vector<std::string> detect_labels(const FloatImage& image,
                                       const std::string& target_type,
                                       bool inverted = false,
                                       bool create_markers = true,
                                       double minimum_size = 0.0) {
    DetectScene scene = make_scene(image.width(), image.height(), 1);
    write_gray_png(scene.paths[0], image);
    Params params = params_for("DetectMarkers");
    params.set("target_type", target_type);
    params.set("inverted", inverted);
    params.set("create_markers", create_markers);
    params.set("minimum_size", minimum_size);
    run_named(scene.document, "DetectMarkers", params);
    std::vector<std::string> labels;
    for (const Marker& marker : scene.chunk->markers()) {
        labels.push_back(marker.label);
    }
    return labels;
}
TEST(Markers, DetectMixedSheet) {
    const int width = 900;
    const int height = 400;
    FloatImage sheet(width, height, 0.5F);
    // A non-coded circular target (a filled dark disc) on its own white patch.
    // Coded rings are exercised by DetectsOnAllCameras and the timing test;
    // the plain disc makes this mixed-type test independent of ring decoding.
    const int disc_size = 100;
    test::EllipseTruth disc;
    disc.cx = 0.5 * static_cast<double>(disc_size);
    disc.cy = 0.5 * static_cast<double>(disc_size);
    disc.a = 45.0;
    disc.b = 45.0;
    paste(sheet,
          test::render_ellipses(disc_size, disc_size, {disc}, 0.95F, 0.05F),
          150 - disc_size / 2, 200 - disc_size / 2);
    // A checker cross on the mid-grey sheet.
    const int cross_size = 110;
    CrossTruth cross;
    cross.cx = 0.5 * static_cast<double>(cross_size);
    cross.cy = 0.5 * static_cast<double>(cross_size);
    cross.half = 35.0;
    cross.angle = 0.3;
    paste(sheet,
          render_checker_crosses(cross_size, cross_size, {cross}),
          450 - cross_size / 2, 200 - cross_size / 2);
    // An AprilTag 36h11 with its own white quiet zone.
    const int tag_size = test::april_tag_canvas_size(80.0);
    test::AprilTagTruth tag;
    tag.cx = 0.5 * static_cast<double>(tag_size);
    tag.cy = 0.5 * static_cast<double>(tag_size);
    tag.tag_px = 80.0;
    tag.id = 0;
    tag.family = AprilTagFamily::Tag36h11;
    const test::AprilTagRender tag_render =
        test::render_april_tag(tag_size, tag_size, tag, 0.95, 0.05);
    paste(sheet, tag_render.image, 750 - tag_size / 2, 200 - tag_size / 2);

    const FloatImage image = gaussian_blur(sheet, 0.8F);
    // minimum_size keeps the other targets' small fiducial cells out of each
    // decoder's candidate set; the disc is larger so its own run can use a
    // tighter floor that also excludes the tag's border blob.
    const std::vector<std::string> circular =
        detect_labels(image, "Circular", false, true, 40.0);
    ASSERT_EQ(circular.size(), 1U) << "circular run found "
                                   << circular.size() << " targets";
    EXPECT_EQ(circular[0], "point 1");
    const std::vector<std::string> crosses =
        detect_labels(image, "Cross", false, true, 20.0);
    ASSERT_EQ(crosses.size(), 1U) << "cross run found " << crosses.size();
    EXPECT_EQ(crosses[0], "point 1");
    const std::vector<std::string> tags =
        detect_labels(image, "AprilTag36h11", false, true, 20.0);
    ASSERT_EQ(tags.size(), 1U) << "tag run found " << tags.size();
    EXPECT_EQ(tags[0], "target 0");
}
TEST(Markers, NonCodedAttachesToNearestExistingMarker) {
    const int width = 640;
    const int height = 480;
    DetectScene scene = make_scene(width, height, 2);
    Transform second = Transform::Identity();
    second(0, 3) = -1.0;  // camera at (1, 0, 0) looking along +Z
    scene.chunk->cameras()[1].pose = second;
    const std::int32_t second_camera = scene.chunk->cameras()[1].id;
    make_add_marker_command(scene.chunk->id(), std::string("M"),
                            Eigen::Vector3d(0.0, 0.0, 10.0))
        ->apply(scene.document);
    make_add_marker_with_projection_command(scene.chunk->id(), std::string("M"),
                                            scene.chunk->cameras()[0].id,
                                            Eigen::Vector2d(320.0, 240.0), false)
        ->apply(scene.document);

    // The known point projects to (270, 240) in the translated camera.
    test::EllipseTruth disc;
    disc.cx = 270.0;
    disc.cy = 240.0;
    disc.a = 20.0;
    disc.b = 20.0;
    write_gray_png(scene.paths[0], FloatImage(width, height, 0.95F));
    write_gray_png(scene.paths[1],
                   test::render_ellipses(width, height, {disc}, 0.95F, 0.05F));

    Params params = params_for("DetectMarkers");
    params.set("target_type", std::string("Circular"));
    params.set("cameras",
               std::vector<std::string>{scene.chunk->cameras()[1].label});
    run_named(scene.document, "DetectMarkers", params);

    ASSERT_EQ(scene.chunk->markers().size(), 1U);
    Marker* marker = find_marker(*scene.chunk, "M");
    ASSERT_NE(marker, nullptr);
    ASSERT_EQ(marker->projections.size(), 2U);
    bool found = false;
    for (const MarkerProjection& projection : marker->projections) {
        if (projection.camera_id == second_camera) {
            found = true;
            EXPECT_NEAR(projection.pixel.x(), 270.0, 1.0);
            EXPECT_NEAR(projection.pixel.y(), 240.0, 1.0);
        }
    }
    EXPECT_TRUE(found);
    EXPECT_EQ(detect_markers_last_stats().attached, 1U);
}
class ScopedSink {
public:
    explicit ScopedSink(DetectMarkersCommandSink sink) {
        set_detect_markers_command_sink(std::move(sink));
    }
    ~ScopedSink() { set_detect_markers_command_sink({}); }
    ScopedSink(const ScopedSink&) = delete;
    ScopedSink& operator=(const ScopedSink&) = delete;
};

FloatImage render_one_disc(int width, int height, double cx, double cy) {
    test::EllipseTruth disc;
    disc.cx = cx;
    disc.cy = cy;
    disc.a = 22.0;
    disc.b = 22.0;
    return test::render_ellipses(width, height, {disc}, 0.95F, 0.05F);
}
TEST(Markers, CreateMarkersFalseDoesNotAdd) {
    const int width = 400;
    const int height = 300;
    DetectScene scene = make_scene(width, height, 1);
    write_gray_png(scene.paths[0], render_one_disc(width, height, 200.0, 150.0));
    Params params = params_for("DetectMarkers");
    params.set("target_type", std::string("Circular"));
    params.set("create_markers", false);
    std::string log_text;
    {
        ScopedLogCapture capture;
        run_named(scene.document, "DetectMarkers", params);
        log_text = capture.text(LogLevel::Info);
    }
    EXPECT_NE(log_text.find("detected 1"), std::string::npos) << log_text;
    EXPECT_NE(log_text.find("dropped 1"), std::string::npos) << log_text;

    EXPECT_TRUE(scene.chunk->markers().empty());
    const DetectMarkersStats stats = detect_markers_last_stats();
    EXPECT_EQ(stats.detected, 1U);
    EXPECT_EQ(stats.created, 0U);
    EXPECT_EQ(stats.attached, 0U);
    EXPECT_EQ(stats.dropped, 1U);
}

TEST(Markers, DetectsOnAllCameras) {
    const int width = 640;
    const int height = 480;
    test::CircularTruth truth = test::make_circular_truth(
        320.0, 240.0, 30.0, 12, read_circular_word(12, 0), 1);
    const FloatImage image = test::render_circular_target(width, height, truth);
    DetectScene scene = make_scene(width, height, 3);
    for (const std::string& path : scene.paths) {
        write_gray_png(path, image);
    }
    Params params = params_for("DetectMarkers");
    params.set("target_type", std::string("Circular12"));
    run_named(scene.document, "DetectMarkers", params);

    ASSERT_EQ(scene.chunk->markers().size(), 1U);
    const Marker& marker = scene.chunk->markers()[0];
    EXPECT_EQ(marker.label.rfind("target ", 0), 0U);
    EXPECT_EQ(marker.projections.size(), 3U);
    EXPECT_EQ(detect_markers_last_stats().created, 1U);
}
TEST(Markers, DetectIsUndoable) {
    const int width = 640;
    const int height = 480;
    test::CircularTruth truth = test::make_circular_truth(
        320.0, 240.0, 30.0, 12, read_circular_word(12, 0), 1);
    DetectScene scene = make_scene(width, height, 1);
    write_gray_png(scene.paths[0],
                   test::render_circular_target(width, height, truth));
    UndoStack stack(&scene.document);
    [[maybe_unused]] ScopedSink sink(
        [&stack](Document& document, std::unique_ptr<AbstractCommand> command) {
            (void)document;
            stack.execute(std::move(command));
        });
    Params params = params_for("DetectMarkers");
    params.set("target_type", std::string("Circular12"));
    run_named(scene.document, "DetectMarkers", params);

    ASSERT_EQ(scene.chunk->markers().size(), 1U);
    ASSERT_EQ(stack.size(), 1U);
    EXPECT_EQ(stack.command_names()[0], "Detect Markers");
    stack.undo();
    EXPECT_TRUE(scene.chunk->markers().empty());
    stack.redo();
    ASSERT_EQ(scene.chunk->markers().size(), 1U);
    EXPECT_EQ(scene.chunk->markers()[0].projections.size(), 1U);
}

TEST(Markers, FilterMaskSkipsMaskedTargets) {
    const int width = 400;
    const int height = 300;
    test::EllipseTruth left;
    left.cx = 100.0;
    left.cy = 150.0;
    left.a = 22.0;
    left.b = 22.0;
    test::EllipseTruth right;
    right.cx = 300.0;
    right.cy = 150.0;
    right.a = 22.0;
    right.b = 22.0;
    const FloatImage image =
        test::render_ellipses(width, height, {left, right}, 0.95F, 0.05F);
    DetectScene scene = make_scene(width, height, 1);
    write_gray_png(scene.paths[0], image);
    Mask mask;
    mask.camera_id = scene.chunk->cameras()[0].id;
    mask.width = width;
    mask.height = height;
    mask.invert = false;
    mask.data.assign(static_cast<std::size_t>(width * height),
                     std::uint8_t{255});
    for (int y = 120; y < 180; ++y) {
        for (int x = 270; x < 330; ++x) {
            mask.data[static_cast<std::size_t>(y * width + x)] =
                std::uint8_t{0};
        }
    }
    scene.chunk->masks().push_back(mask);

    Params params = params_for("DetectMarkers");
    params.set("target_type", std::string("Circular"));
    params.set("create_markers", false);
    run_named(scene.document, "DetectMarkers", params);
    EXPECT_EQ(detect_markers_last_stats().detected, 1U);
    params.set("filter_mask", false);
    run_named(scene.document, "DetectMarkers", params);
    EXPECT_EQ(detect_markers_last_stats().detected, 2U);
}
TEST(Markers, ToleranceMapping) {
    EXPECT_EQ(circular_max_hamming(0.0), 0);
    EXPECT_EQ(circular_max_hamming(74.9), 0);
    EXPECT_EQ(circular_max_hamming(75.0), 1);
    EXPECT_EQ(circular_max_hamming(100.0), 1);

    EXPECT_EQ(apriltag_max_hamming(0.0, AprilTagFamily::Tag36h11), 0);
    EXPECT_EQ(apriltag_max_hamming(25.0, AprilTagFamily::Tag36h11), 1);
    EXPECT_EQ(apriltag_max_hamming(50.0, AprilTagFamily::Tag36h11), 2);
    EXPECT_EQ(apriltag_max_hamming(100.0, AprilTagFamily::Tag36h11), 4);
    // 16h5's own default budget is 0, so tolerance cannot widen it.
    EXPECT_EQ(apriltag_max_hamming(100.0, AprilTagFamily::Tag16h5), 0);
    EXPECT_EQ(apriltag_max_hamming(50.0, AprilTagFamily::Standard41h12), 3);
    EXPECT_EQ(apriltag_max_hamming(100.0, AprilTagFamily::Standard52h13), 6);
}

TEST(Markers, UnknownTargetTypeRejected) {
    DetectScene scene = make_scene(200, 150, 1);
    Params params = params_for("DetectMarkers");
    params.set("target_type", std::string("NoSuchTarget"));
    try {
        run_named(scene.document, "DetectMarkers", params);
        FAIL() << "an unknown target_type must be rejected";
    } catch (const ValueError& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("target_type"), std::string::npos) << message;
        EXPECT_NE(message.find("Circular12"), std::string::npos) << message;
    }
    params.set("target_type", std::string("AprilTagCircle21h7"));
    try {
        run_named(scene.document, "DetectMarkers", params);
        FAIL() << "AprilTagCircle21h7 must be rejected";
    } catch (const ValueError& error) {
        EXPECT_NE(std::string(error.what()).find("not supported yet"),
                  std::string::npos);
    }
}
/// A bar cross: two orthogonal dark bars of finite width on a light sheet.
FloatImage render_bar_cross(int width, int height, double cx, double cy,
                            double half_length, double half_width, double angle,
                            float background = 0.95F, float dark = 0.05F,
                            int supersample = 4) {
    FloatImage image(width, height, background);
    const int samples = std::max(1, supersample);
    const double total = static_cast<double>(samples * samples);
    const double step = 1.0 / static_cast<double>(samples);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double extent = half_length * std::sqrt(2.0) + 2.0;
    const int x0 = std::max(0, static_cast<int>(std::floor(cx - extent)));
    const int y0 = std::max(0, static_cast<int>(std::floor(cy - extent)));
    const int x1 = std::min(width - 1, static_cast<int>(std::ceil(cx + extent)));
    const int y1 = std::min(height - 1, static_cast<int>(std::ceil(cy + extent)));
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            double sum = 0.0;
            for (int sy = 0; sy < samples; ++sy) {
                for (int sx = 0; sx < samples; ++sx) {
                    const double px = static_cast<double>(x) +
                                      (static_cast<double>(sx) + 0.5) * step;
                    const double py = static_cast<double>(y) +
                                      (static_cast<double>(sy) + 0.5) * step;
                    const double dx = px - cx;
                    const double dy = py - cy;
                    const double u = dx * c + dy * s;
                    const double v = -dx * s + dy * c;
                    const bool inside =
                        (std::abs(u) <= half_length && std::abs(v) <= half_width) ||
                        (std::abs(u) <= half_width && std::abs(v) <= half_length);
                    sum += static_cast<double>(inside ? dark : background);
                }
            }
            image.at(x, y) = static_cast<float>(sum / total);
        }
    }
    return image;
}

TEST(Markers, DetectBarCross) {
    const FloatImage image =
        render_bar_cross(400, 360, 200.3, 180.6, 42.0, 6.0, 0.4);
    const std::vector<CodedTarget> found = detect_crosses(image);
    ASSERT_FALSE(found.empty());
    const Eigen::Vector2d expected(200.3, 180.6);
    double best = std::numeric_limits<double>::infinity();
    for (const CodedTarget& target : found) {
        best = std::min(best, (target.center - expected).norm());
    }
    std::printf("[DetectBarCross] %zu targets, nearest error %.4f px\n",
                found.size(), best);
    EXPECT_LE(best, 1.0);
}
TEST(Markers, DetectCircularTiming) {
    const int width = 5472;
    const int height = 3648;
    std::vector<test::CircularTruth> targets;
    for (int i = 0; i < 6; ++i) {
        const double cx = (static_cast<double>(i % 3) + 0.5) *
                          (static_cast<double>(width) / 3.0);
        const double cy = (static_cast<double>(i / 3) + 0.5) *
                          (static_cast<double>(height) / 2.0);
        targets.push_back(test::make_circular_truth(
            cx, cy, 60.0, 12, read_circular_word(12, static_cast<std::size_t>(i)),
            1));
    }
    const FloatImage image = test::render_circular_targets(
        width, height, targets, 0.95F, 0.05F, 4);
    DetectScene scene = make_scene(width, height, 1);
    write_gray_png(scene.paths[0], image);
    Params params = params_for("DetectMarkers");
    params.set("target_type", std::string("Circular12"));
    const auto started = std::chrono::steady_clock::now();
    run_named(scene.document, "DetectMarkers", params);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    std::printf(
        "[DetectMarkers timing] %.3f s for %dx%d Circular12 (auto downscale)\n",
        seconds, width, height);
    EXPECT_EQ(scene.chunk->markers().size(), targets.size());
    if (test::perf_asserts_enabled()) {
        EXPECT_LT(seconds, 5.0);
    }
}
}  // namespace
}  // namespace alphashape
