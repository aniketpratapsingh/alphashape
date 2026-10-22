// Tests for the DoG detector, the M-LDB descriptor, and DetectFeatures as a
// task: synthetic fixtures written to a temp directory, no external data.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "base/utils/error.hpp"
#include "common/algorithm/feature/detect_features_task.hpp"
#include "common/algorithm/feature/dog_detector.hpp"
#include "common/algorithm/feature/mldb_descriptor.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/tasks/registry.hpp"
#include "image/image.hpp"
#include "image/io/pnm.hpp"

namespace alphashape {
namespace {

namespace fs = std::filesystem;

constexpr int kSize = 256;

/// A deterministic scene: a smooth gradient with several filled discs and
/// squares of different sizes and intensities. Distinct, well-separated blobs
/// give the detector strong, repeatable keypoints.
Image make_scene() {
    Image image(kSize, kSize, 1);
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            const int value = 40 + (x * 80) / (kSize - 1) + (y * 40) / (kSize - 1);
            image.at(x, y) = static_cast<std::uint8_t>(std::clamp(value, 0, 255));
        }
    }
    const auto disc = [&](int cx, int cy, int r, int value) {
        for (int y = cy - r; y <= cy + r; ++y) {
            for (int x = cx - r; x <= cx + r; ++x) {
                if (x < 0 || y < 0 || x >= kSize || y >= kSize) continue;
                const int dx = x - cx;
                const int dy = y - cy;
                if (dx * dx + dy * dy <= r * r) {
                    image.at(x, y) = static_cast<std::uint8_t>(value);
                }
            }
        }
    };
    const auto square = [&](int cx, int cy, int half, int value) {
        for (int y = cy - half; y <= cy + half; ++y) {
            for (int x = cx - half; x <= cx + half; ++x) {
                if (x < 0 || y < 0 || x >= kSize || y >= kSize) continue;
                image.at(x, y) = static_cast<std::uint8_t>(value);
            }
        }
    };
    disc(64, 64, 10, 235);
    disc(180, 80, 14, 25);
    disc(90, 170, 20, 240);
    disc(200, 205, 8, 30);
    square(150, 140, 16, 20);
    square(40, 210, 12, 220);
    return image;
}

/// Rotate 90 degrees clockwise: a pixel at (x, y) moves to (H-1-y, x).
Image rotate90_cw(const Image& in) {
    const int w = in.width();
    const int h = in.height();
    Image out(h, w, in.channels());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < in.channels(); ++c) {
                out.at(h - 1 - y, x, c) = in.at(x, y, c);
            }
        }
    }
    return out;
}

bool same_detection(const Detection& a, const Detection& b) {
    return a.keypoint == b.keypoint && a.response == b.response && a.octave == b.octave &&
           a.layer == b.layer && a.octave_x == b.octave_x && a.octave_y == b.octave_y &&
           a.octave_sigma == b.octave_sigma;
}

/// Detections sorted strongest-first, the order the task and rotation test use.
std::vector<Detection> detect_sorted(const FloatImage& luminance) {
    const DogDetector detector;
    const ScaleSpace space = detector.build_scale_space(luminance);
    std::vector<Detection> detections = detector.detect(space);
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.response > b.response;
    });
    return detections;
}

std::vector<Descriptor> describe(const FloatImage& luminance, const std::vector<Detection>& dets) {
    const DogDetector detector;
    const ScaleSpace space = detector.build_scale_space(luminance);
    return compute_mldb(space, dets);
}

// A temp dir holding written PGM fixtures for the task-level tests.
class DetectorTaskFixture : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("alphashape_detector_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        std::error_code ec;
        fs::remove_all(dir_, ec);
        fs::create_directories(dir_, ec);
        ASSERT_FALSE(ec);

        const Image scene = make_scene();
        write_pnm(scene, (dir_ / "a.pgm").string());
        write_pnm(scene, (dir_ / "b.pgm").string());
    }
    void TearDown() override {
        std::error_code ec;
        fs::remove_all(dir_, ec);
    }

    std::string path(const std::string& name) const { return (dir_ / name).string(); }

    fs::path dir_;
};

TEST(Detector, FindsKeypointsInsideBoundsWithOneDescriptorEach) {
    const FloatImage luminance = make_scene().to_luminance();
    const std::vector<Detection> detections = detect_sorted(luminance);
    ASSERT_GT(detections.size(), 20U);

    const std::vector<Descriptor> descriptors = describe(luminance, detections);
    EXPECT_EQ(descriptors.size(), detections.size());
    for (const Detection& d : detections) {
        EXPECT_GE(d.keypoint.x, 0.0F);
        EXPECT_GE(d.keypoint.y, 0.0F);
        EXPECT_LT(d.keypoint.x, static_cast<float>(kSize));
        EXPECT_LT(d.keypoint.y, static_cast<float>(kSize));
    }
    EXPECT_EQ(Descriptor::kBytes, 63U);
    EXPECT_EQ(hamming_distance(descriptors.front(), descriptors.front()), 0);
}

/// Settle the pixel convention empirically. Render an isolated bright Gaussian
/// blob whose centre is exactly the centre of pixel (i, j) in the
/// FloatImage::sample convention ("(0,0) is the centre of the top-left pixel",
/// image.hpp): sample(x, y) with (x, y) = (i, j) must hit the peak. The
/// detector emits keypoints in that same convention (dog_detector.cpp assigns
/// keypoint.x = octave_x, and the descriptor samples at octave_x through
/// FloatImage::sample), so the strongest keypoint must land within 0.15 px of
/// (i, j) — NOT (i+0.5, j+0.5). keypoint_to_pixel() adds the documented +0.5 to
/// reach Calibration's convention where the top-left pixel centre is (0.5,0.5).
TEST(Detector, KeypointConventionMatchesSampleCoordinates) {
    constexpr int kW = 200;
    constexpr int kH = 160;
    constexpr int kI = 100;
    constexpr int kJ = 80;
    constexpr float kSigma = 3.0F;

    FloatImage image(kW, kH, 0.1F);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const float dx = static_cast<float>(x) - static_cast<float>(kI);
            const float dy = static_cast<float>(y) - static_cast<float>(kJ);
            const float value =
                0.9F * std::exp(-(dx * dx + dy * dy) / (2.0F * kSigma * kSigma));
            image.at(x, y) = 0.1F + value;
        }
    }

    const std::vector<Detection> detections = detect_sorted(image);
    ASSERT_FALSE(detections.empty()) << "the blob produced no keypoints";
    const Keypoint& strongest = detections.front().keypoint;
    EXPECT_NEAR(strongest.x, static_cast<float>(kI), 0.15F)
        << "detector x is not in the sample (integer-pixel-centre) convention";
    EXPECT_NEAR(strongest.y, static_cast<float>(kJ), 0.15F)
        << "detector y is not in the sample (integer-pixel-centre) convention";

    // keypoint_to_pixel shifts to Calibration's convention by exactly +0.5.
    const Eigen::Vector2d pixel = keypoint_to_pixel(strongest);
    EXPECT_NEAR(pixel.x(), static_cast<double>(strongest.x) + 0.5, 1e-12);
    EXPECT_NEAR(pixel.y(), static_cast<double>(strongest.y) + 0.5, 1e-12);
}

TEST(Detector, DetectionIsDeterministicAcrossRuns) {
    const FloatImage luminance = make_scene().to_luminance();
    const std::vector<Detection> first = detect_sorted(luminance);
    const std::vector<Detection> second = detect_sorted(luminance);
    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_TRUE(same_detection(first[i], second[i])) << "at index " << i;
    }
}

TEST(Detector, DescriptorIsInvariantUnderNinetyDegreeRotation) {
    const Image scene = make_scene();
    const Image rotated = rotate90_cw(scene);
    const FloatImage original_luma = scene.to_luminance();
    const FloatImage rotated_luma = rotated.to_luminance();

    std::vector<Detection> original = detect_sorted(original_luma);
    std::vector<Detection> rotated_dets = detect_sorted(rotated_luma);
    ASSERT_GT(original.size(), 20U);
    ASSERT_GT(rotated_dets.size(), 20U);

    const std::vector<Descriptor> original_desc = describe(original_luma, original);
    const std::vector<Descriptor> rotated_desc = describe(rotated_luma, rotated_dets);

    // Top-20 strongest keypoints of the original image.
    const std::size_t checked = std::min<std::size_t>(20, original.size());
    const int max_hamming = static_cast<int>(504 * 0.30);  // < 30% of 504 bits
    int matches = 0;
    for (std::size_t i = 0; i < checked; ++i) {
        // (x, y) in the H x W original -> (H-1-y, x) in the rotated image.
        const float rx = static_cast<float>(kSize - 1) -
                         original[i].keypoint.y * 1.0F;  // height == width == kSize
        const float ry = original[i].keypoint.x * 1.0F;
        int best = 1 << 30;
        std::size_t best_index = 0;
        for (std::size_t j = 0; j < rotated_desc.size(); ++j) {
            const int distance = hamming_distance(original_desc[i], rotated_desc[j]);
            if (distance < best) {
                best = distance;
                best_index = j;
            }
        }
        const float dx = std::abs(rotated_dets[best_index].keypoint.x - rx);
        const float dy = std::abs(rotated_dets[best_index].keypoint.y - ry);
        const bool position_ok = dx <= 2.0F && dy <= 2.0F;
        if (position_ok && best < max_hamming) {
            ++matches;
        }
    }
    // At least half of the strongest keypoints must find their rotated twin.
    EXPECT_GE(matches, static_cast<int>(checked / 2))
        << matches << " of " << checked << " matched after 90-degree rotation";
}

TEST_F(DetectorTaskFixture, RunTaskCapsKeypointsAndRecordsMeta) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("chunk");
    document.set_active_chunk_id(chunk.id());
    chunk.cameras().push_back(Camera{});   // id 0
    chunk.cameras().back().id = 0;
    chunk.cameras().push_back(Camera{});   // id 1
    chunk.cameras().back().id = 1;

    auto frame_a = std::make_shared<Frame>("a.pgm", kSize, kSize);
    frame_a->set_id(0);
    frame_a->set_camera_id(0);
    chunk.frames().push_back(frame_a);
    auto frame_b = std::make_shared<Frame>("b.pgm", kSize, kSize);
    frame_b->set_id(1);
    frame_b->set_camera_id(1);
    chunk.frames().push_back(frame_b);

    document.set_path(path("proj.psx"));  // frames resolve relative to this dir

    Params params = make_params(builtin_registry().require("DetectFeatures"));
    params.set("keypoint_limit", std::int64_t{50});
    run_task(builtin_registry(), document, "DetectFeatures", params);

    ASSERT_EQ(chunk.tie_points().keypoint_sets.size(), 2U);
    for (const KeypointSet& set : chunk.tie_points().keypoint_sets) {
        EXPECT_LE(set.keypoints.size(), 50U);
        EXPECT_GT(set.keypoints.size(), 0U);
        EXPECT_TRUE(set.consistent());
    }
    const std::string* limit = chunk.meta().find("DetectFeatures/keypoint_limit");
    ASSERT_NE(limit, nullptr);
    EXPECT_EQ(*limit, "50");
}

TEST_F(DetectorTaskFixture, UnlinkedFrameThrowsAndLeavesTiePointsUnchanged) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("chunk");
    document.set_active_chunk_id(chunk.id());
    chunk.cameras().push_back(Camera{});
    chunk.cameras().back().id = 0;

    auto frame = std::make_shared<Frame>("a.pgm", kSize, kSize);
    frame->set_id(0);
    frame->set_camera_id(-1);  // unlinked
    chunk.frames().push_back(frame);

    // Pre-existing tie points must be untouched by the failed run.
    KeypointSet sentinel;
    sentinel.camera_id = 99;
    chunk.tie_points().keypoint_sets.push_back(sentinel);

    document.set_path(path("proj.psx"));
    Params params = make_params(builtin_registry().require("DetectFeatures"));
    EXPECT_THROW(run_task(builtin_registry(), document, "DetectFeatures", params), Error);

    ASSERT_EQ(chunk.tie_points().keypoint_sets.size(), 1U);
    EXPECT_EQ(chunk.tie_points().keypoint_sets.front().camera_id, 99);
    EXPECT_TRUE(chunk.tie_points().keypoint_sets.front().keypoints.empty());
}

TEST_F(DetectorTaskFixture, RelativeFramePathResolvesAgainstDocumentDirectory) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("chunk");
    document.set_active_chunk_id(chunk.id());
    chunk.cameras().push_back(Camera{});
    chunk.cameras().back().id = 0;

    // The image lives in a "frames" subdirectory of the project directory, and
    // the frame records a path relative to the project.
    std::error_code ec;
    fs::create_directories(dir_ / "frames", ec);
    ASSERT_FALSE(ec);
    write_pnm(make_scene(), (dir_ / "frames" / "rel.pgm").string());

    auto frame = std::make_shared<Frame>("frames/rel.pgm", kSize, kSize);
    frame->set_id(0);
    frame->set_camera_id(0);
    chunk.frames().push_back(frame);

    document.set_path(path("proj.psx"));
    Params params = make_params(builtin_registry().require("DetectFeatures"));
    run_task(builtin_registry(), document, "DetectFeatures", params);

    ASSERT_EQ(chunk.tie_points().keypoint_sets.size(), 1U);
    EXPECT_GT(chunk.tie_points().keypoint_sets.front().keypoints.size(), 0U);
}

TEST_F(DetectorTaskFixture, UnsupportedDescriptorTypeThrowsValueError) {
    Document document;
    Chunk& chunk = document.workspace().add_chunk("chunk");
    document.set_active_chunk_id(chunk.id());
    chunk.cameras().push_back(Camera{});
    chunk.cameras().back().id = 0;

    auto frame = std::make_shared<Frame>("a.pgm", kSize, kSize);
    frame->set_id(0);
    frame->set_camera_id(0);
    chunk.frames().push_back(frame);

    document.set_path(path("proj.psx"));
    Params params = make_params(builtin_registry().require("DetectFeatures"));
    params.set("descriptor_type", std::string("sift"));
    EXPECT_THROW(run_task(builtin_registry(), document, "DetectFeatures", params), ValueError);
}

}  // namespace
}  // namespace alphashape
