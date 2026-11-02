// Tests for WP-13.7 AprilTag decoding: the seven published family tables, the
// stated minimum Hamming distances, decoding of rendered tags through
// detect_quads -> decode_tag, centre accuracy, foreign-shape rejection and the
// distance boundary. Synthetic tags come from make_apriltag.hpp.

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/feature/dog_detector.hpp"
#include "common/algorithm/markers/apriltag_decoder.hpp"
#include "common/algorithm/markers/marker_detector.hpp"
#include "common/algorithm/markers/tag_families.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "make_apriltag.hpp"
#include "make_marker_sheet.hpp"
#include "perf_gate.hpp"

namespace alphashape {
namespace {

std::array<AprilTagFamily, kAprilTagFamilyCount> all_families() {
    return {AprilTagFamily::Tag16h5, AprilTagFamily::Tag25h9,
            AprilTagFamily::Tag36h10, AprilTagFamily::Tag36h11,
            AprilTagFamily::Circle21h7, AprilTagFamily::Standard41h12,
            AprilTagFamily::Standard52h13};
}

/// The decoder's working families. Circle21h7 is table-only: the published
/// tags are circular, so detect_quads cannot produce a candidate for them.
std::array<AprilTagFamily, 6> decode_families() {
    return {AprilTagFamily::Tag16h5, AprilTagFamily::Tag25h9,
            AprilTagFamily::Tag36h10, AprilTagFamily::Tag36h11,
            AprilTagFamily::Standard41h12, AprilTagFamily::Standard52h13};
}

struct FixtureFamily {
    std::string name;
    std::vector<std::uint64_t> codes;
    int hamming = 0;
    int bits = 0;
    int width_at_border = 0;
    int total_width = 0;
    bool reversed_border = false;
    bool loaded = false;
};

bool parse_metadata(const std::string& line, FixtureFamily& fixture) {
    std::string token;
    const auto assign = [&](const std::string& field) {
        const std::size_t equals = field.find('=');
        if (equals == std::string::npos) {
            return;
        }
        const std::string key = field.substr(0, equals);
        const std::string value = field.substr(equals + 1);
        if (key == "family") {
            fixture.name = value;
        } else if (key == "h") {
            fixture.hamming = std::stoi(value);
        } else if (key == "nbits") {
            fixture.bits = std::stoi(value);
        } else if (key == "width_at_border") {
            fixture.width_at_border = std::stoi(value);
        } else if (key == "total_width") {
            fixture.total_width = std::stoi(value);
        } else if (key == "reversed_border") {
            fixture.reversed_border = value == "true";
        }
    };
    std::size_t start = 0;
    while (start < line.size()) {
        const std::size_t space = line.find(' ', start);
        token = line.substr(start, space == std::string::npos ? std::string::npos
                                                              : space - start);
        assign(token);
        if (space == std::string::npos) {
            break;
        }
        start = space + 1;
    }
    return true;
}

std::optional<FixtureFamily> read_fixture(const std::string& name) {
    const std::string path =
        std::string(ALPHASHAPE_TEST_FIXTURE_DIR) + "/markers/apriltag_" + name + ".txt";
    std::ifstream input(path);
    if (!input.good()) {
        return std::nullopt;
    }
    FixtureFamily fixture;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.front() == '#') {
            if (line.rfind("# family=", 0) == 0) {
                parse_metadata(line.substr(2), fixture);
            }
            continue;
        }
        fixture.codes.push_back(std::stoull(line, nullptr, 16));
    }
    fixture.loaded = true;
    return fixture;
}

std::uint64_t reverse_layout_word(std::uint64_t word, int bits) {
    std::uint64_t out = 0;
    for (int i = 0; i < bits; ++i) {
        if (((word >> static_cast<unsigned>(i)) & 1u) != 0u) {
            out |= std::uint64_t{1} << static_cast<unsigned>(bits - 1 - i);
        }
    }
    return out;
}

std::array<std::uint64_t, 4> layout_rotations(std::uint64_t layout_word,
                                              const TagFamily& family) {
    std::array<std::uint64_t, 4> rotations{};
    rotations[0] = layout_word;
    for (std::size_t i = 1; i < rotations.size(); ++i) {
        rotations[i] = rotate_tag_word(rotations[i - 1], family);
    }
    return rotations;
}

test::AprilTagTruth make_truth(AprilTagFamily family, int id, double tag_px,
                         double rotation, double perspective, double offset_x,
                         double offset_y) {
    const int size = test::april_tag_canvas_size(tag_px);
    test::AprilTagTruth truth;
    truth.family = family;
    truth.id = id;
    truth.tag_px = tag_px;
    truth.rotation = rotation;
    truth.perspective_u = perspective;
    truth.perspective_v = 0.6 * perspective;
    truth.cx = 0.5 * static_cast<double>(size) + offset_x;
    truth.cy = 0.5 * static_cast<double>(size) + offset_y;
    return truth;
}

QuadParams quad_params_for(double tag_px) {
    QuadParams params;
    params.min_side = 6;
    params.max_side = static_cast<int>(1.6 * tag_px) + 8;
    params.threshold = 0.02F;
    return params;
}

FloatImage render_degraded(const test::AprilTagTruth& truth, double blur, double noise,
                           std::uint32_t seed, int supersample) {
    const int size = test::april_tag_canvas_size(truth.tag_px);
    test::AprilTagRender render =
        test::render_april_tag(size, size, truth, 0.95, 0.05, supersample);
    FloatImage image = std::move(render.image);
    if (blur > 0.0) {
        image = gaussian_blur(image, static_cast<float>(blur));
    }
    if (noise > 0.0) {
        test::add_noise(image, noise, seed);
    }
    test::quantise_to_8bit(image);
    return image;
}

std::vector<CodedTarget> detect_and_decode(const FloatImage& image,
                                           AprilTagFamily family, int max_hamming,
                                           const QuadParams& params) {
    std::vector<CodedTarget> decoded;
    const TagFamily& table = tag_family(family);
    std::vector<MarkerCandidate> candidates;
    if (table.reversed_border) {
        // The white reference frame is the published quad. On the inverted
        // image it is a solid dark ring, so the same detector finds the exact
        // width_at_border square even when perimeter payload cells notch the
        // dark silhouette.
        FloatImage inverted = image;
        for (float& value : inverted.pixels()) {
            value = 1.0F - value;
        }
        const std::vector<MarkerCandidate> frame_candidates =
            detect_quads(inverted, params);
        candidates.insert(candidates.end(), frame_candidates.begin(),
                          frame_candidates.end());
    }
    const std::vector<MarkerCandidate> dark_candidates =
        detect_quads(image, params);
    candidates.insert(candidates.end(), dark_candidates.begin(),
                      dark_candidates.end());
    for (const MarkerCandidate& candidate : candidates) {
        std::optional<CodedTarget> target =
            decode_tag(image, candidate, table, max_hamming);
        if (target) {
            decoded.push_back(*target);
        }
    }
    return decoded;
}

}  // namespace
}  // namespace alphashape

namespace alphashape {
namespace {

TEST(Markers, AprilTagFamilyTablesMatchPublished) {
    const std::array<AprilTagFamily, kAprilTagFamilyCount> families = all_families();
    for (AprilTagFamily family : families) {
        const TagFamily& table = tag_family(family);
        const std::optional<FixtureFamily> fixture = read_fixture(table.name);
        ASSERT_TRUE(fixture.has_value()) << "missing fixture for " << table.name;
        EXPECT_EQ(fixture->name, table.name);
        EXPECT_EQ(fixture->bits, table.bits) << table.name;
        EXPECT_EQ(fixture->hamming, table.min_hamming) << table.name;
        EXPECT_EQ(fixture->width_at_border, table.width_at_border) << table.name;
        EXPECT_EQ(fixture->total_width, table.total_width) << table.name;
        EXPECT_EQ(fixture->reversed_border, table.reversed_border) << table.name;
        EXPECT_EQ(fixture->codes, table.codes) << table.name;
        EXPECT_EQ(static_cast<int>(table.bit_xy.size()), table.bits) << table.name;
        EXPECT_EQ(parse_tag_family(table.name), family) << table.name;
        EXPECT_EQ(parse_tag_family("tag" + table.name), family) << table.name;
        EXPECT_EQ(parse_tag_family("TAG_" + table.name), family) << table.name;
        std::printf("[apriltag table] %-14s codes=%d h=%d wab=%d tw=%d reversed=%d\n",
                    table.name.c_str(), static_cast<int>(table.codes.size()),
                    table.min_hamming, table.width_at_border, table.total_width,
                    table.reversed_border ? 1 : 0);
    }
    EXPECT_FALSE(parse_tag_family("").has_value());
    EXPECT_FALSE(parse_tag_family("tag99h99").has_value());

    // Chosen defaults: the largest radius whose random-word acceptance is
    // below 1e-3 (see apriltag_decoder.hpp).
    EXPECT_EQ(default_max_hamming(AprilTagFamily::Tag16h5), 0);
    EXPECT_EQ(default_max_hamming(AprilTagFamily::Tag25h9), 1);
    EXPECT_EQ(default_max_hamming(AprilTagFamily::Tag36h10), 2);
    EXPECT_EQ(default_max_hamming(AprilTagFamily::Tag36h11), 3);
    EXPECT_EQ(default_max_hamming(AprilTagFamily::Circle21h7), 0);
    EXPECT_EQ(default_max_hamming(AprilTagFamily::Standard41h12), 4);
    EXPECT_EQ(default_max_hamming(AprilTagFamily::Standard52h13), 5);
}

TEST(Markers, AprilTagFamilyMinHamming) {
    const std::array<AprilTagFamily, kAprilTagFamilyCount> families = all_families();
    for (AprilTagFamily family : families) {
        const TagFamily& table = tag_family(family);
        ASSERT_FALSE(table.codes.empty()) << table.name;
        std::vector<std::uint64_t> layout(table.codes.size());
        for (std::size_t i = 0; i < table.codes.size(); ++i) {
            layout[i] = reverse_layout_word(table.codes[i], table.bits);
        }
        const bool sampled = table.codes.size() > 3000;
        std::vector<std::size_t> rows;
        if (sampled) {
            constexpr std::size_t kSample = 200;
            rows.reserve(kSample);
            for (std::size_t i = 0; i < kSample; ++i) {
                rows.push_back(i * table.codes.size() / kSample);
            }
        } else {
            rows.resize(table.codes.size());
            for (std::size_t i = 0; i < rows.size(); ++i) {
                rows[i] = i;
            }
        }
        int minimum = table.bits;
        for (std::size_t i : rows) {
            const std::array<std::uint64_t, 4> rotations =
                layout_rotations(layout[i], table);
            for (std::size_t j = 0; j < layout.size(); ++j) {
                const std::size_t first = (i == j) ? 1 : 0;
                for (std::size_t k = first; k < rotations.size(); ++k) {
                    const int distance = static_cast<int>(
                        std::popcount(rotations[k] ^ layout[j]));
                    minimum = std::min(minimum, distance);
                }
            }
        }
        std::printf("[apriltag min hamming] %-14s measured=%d published=%d%s\n",
                    table.name.c_str(), minimum, table.min_hamming,
                    sampled ? " (200 rows sampled)" : "");
        EXPECT_GE(minimum, table.min_hamming) << table.name;
    }
}

TEST(Markers, AprilTagDecodeRenderedSet) {
    const std::array<AprilTagFamily, 6> families = decode_families();
    const std::array<double, 3> rotations = {0.0, 0.42, 1.06};
    const std::array<double, 3> scales = {40.0, 80.0, 160.0};
    std::uint32_t seed = 2000U;
    int total = 0;
    int correct = 0;
    int wrong = 0;
    for (AprilTagFamily family : families) {
        const TagFamily& table = tag_family(family);
        const int count =
            std::min(100, static_cast<int>(table.codes.size()));
        int family_total = 0;
        int family_correct = 0;
        int family_wrong = 0;
        std::array<std::array<int, 3>, 3> combinations{};
        std::vector<int> failed_ids;
        for (std::size_t scale_index = 0; scale_index < scales.size();
             ++scale_index) {
            for (std::size_t rotation_index = 0; rotation_index < rotations.size();
                 ++rotation_index) {
                for (int id = 0; id < count; ++id) {
                    const double perspective =
                        0.012 + 0.004 * static_cast<double>(scale_index);
                    const double offset_x =
                        0.23 + 0.17 * static_cast<double>(rotation_index) +
                        0.09 * static_cast<double>(scale_index);
                    const double offset_y =
                        0.31 + 0.11 * static_cast<double>(scale_index) +
                        0.07 * static_cast<double>(rotation_index);
                    const test::AprilTagTruth truth =
                        make_truth(family, id, scales[scale_index],
                                   rotations[rotation_index], perspective,
                                   offset_x, offset_y);
                    // The 40 px tags have 4-5 px cells, so one sample per pixel
                    // leaves the edges too aliased for the subpixel fit.
                    const int supersample =
                        scales[scale_index] <= 40.0 ? 2 : 1;
                    const FloatImage image =
                        render_degraded(truth, 0.8, 0.02, seed++, supersample);
                    const QuadParams params = quad_params_for(scales[scale_index]);
                    const std::vector<CodedTarget> decoded = detect_and_decode(
                        image, family, default_max_hamming(family), params);
                    ++total;
                    ++family_total;
                    bool this_correct = false;
                    bool this_wrong = false;
                    for (const CodedTarget& target : decoded) {
                        if (target.code == id) {
                            this_correct = true;
                        } else {
                            this_wrong = true;
                        }
                    }
                    if (this_wrong) {
                        ++wrong;
                        ++family_wrong;
                    } else if (this_correct) {
                        ++correct;
                        ++family_correct;
                        ++combinations[scale_index][rotation_index];
                    } else if (failed_ids.size() < 12) {
                        failed_ids.push_back(id);
                    }
                }
            }
        }
        std::printf("[apriltag rendered] %-14s total=%d correct=%d wrong=%d "
                    "by-scale-rotation=%d,%d,%d/%d,%d,%d/%d,%d,%d\n",
                    table.name.c_str(), family_total, family_correct,
                    family_wrong, combinations[0][0], combinations[0][1],
                    combinations[0][2], combinations[1][0], combinations[1][1],
                    combinations[1][2], combinations[2][0], combinations[2][1],
                    combinations[2][2]);
        if (!failed_ids.empty()) {
            std::printf("[apriltag rendered] %-14s failed ids:", table.name.c_str());
            for (int id : failed_ids) {
                std::printf(" %d", id);
            }
            std::printf("\n");
        }
    }
    std::printf("[apriltag rendered] all total=%d correct=%d wrong=%d\n", total,
                correct, wrong);
    EXPECT_EQ(wrong, 0);
    EXPECT_EQ(correct, total);
}

TEST(Markers, AprilTagReversedSmallDiagnostic) {
    const std::array<AprilTagFamily, 2> families = {AprilTagFamily::Standard41h12,
                                                    AprilTagFamily::Standard52h13};
    for (AprilTagFamily family : families) {
        const TagFamily& table = tag_family(family);
        for (double tag_px : {40.0, 80.0}) {
            for (double rotation : {0.0, 0.42}) {
                const test::AprilTagTruth truth =
                    make_truth(family, 0, tag_px, rotation, 0.012, 0.25, 0.2);
                const int supersample = tag_px <= 40.0 ? 2 : 1;
                const FloatImage image =
                    render_degraded(truth, 0.8, 0.02, 9500U, supersample);
                const QuadParams params = quad_params_for(tag_px);
                FloatImage inverted = image;
                for (float& value : inverted.pixels()) value = 1.0F - value;
                const std::vector<MarkerCandidate> frame =
                    detect_quads(inverted, params);
                const std::vector<MarkerCandidate> dark =
                    detect_quads(image, params);
                std::printf("[diag] %s %.0f rot %.2f frame=%d dark=%d\n",
                            table.name.c_str(), tag_px, rotation,
                            static_cast<int>(frame.size()),
                            static_cast<int>(dark.size()));
                const auto report = [&](const char* what,
                                        const std::vector<MarkerCandidate>& list) {
                    for (const MarkerCandidate& candidate : list) {
                        const std::optional<CodedTarget> target =
                            decode_tag(image, candidate, table,
                                       default_max_hamming(family));
                        const std::optional<CodedTarget> loose = decode_tag(
                            image, candidate, table, table.min_hamming);
                        std::printf(
                            "[diag]   %s centre %.1f,%.1f r %.1f corners %d -> "
                            "%s id %d loose id %d d %d\n",
                            what, candidate.center.x(), candidate.center.y(),
                            candidate.radius_px,
                            static_cast<int>(candidate.corners.size()),
                            target ? "yes" : "no", target ? target->code : -1,
                            loose ? loose->code : -1, loose ? loose->hamming : -1);
                        std::printf("[diag]     corners:");
                        for (const Eigen::Vector2d& corner : candidate.corners) {
                            std::printf(" (%.1f,%.1f)", corner.x(), corner.y());
                        }
                        std::printf("\n");
                    }
                };
                report("frame", frame);
                report("dark", dark);
                for (int silhouette = 0; silhouette < 2; ++silhouette) {
                    MarkerCandidate ideal;
                    const std::array<Eigen::Vector2d, 4> corners =
                        test::april_tag_corners(truth, silhouette != 0);
                    ideal.corners.assign(corners.begin(), corners.end());
                    ideal.center = Eigen::Vector2d(truth.cx, truth.cy);
                    ideal.radius_px = 0.25 * ((corners[2] - corners[0]).norm() +
                                              (corners[3] - corners[1]).norm());
                    const std::optional<CodedTarget> target = decode_tag(
                        image, ideal, table, default_max_hamming(family));
                    std::printf("[diag]   ideal %s -> %s id %d corners:",
                                silhouette != 0 ? "silhouette" : "frame",
                                target ? "yes" : "no",
                                target ? target->code : -1);
                    for (const Eigen::Vector2d& corner : corners) {
                        std::printf(" (%.1f,%.1f)", corner.x(), corner.y());
                    }
                    std::printf("\n");
                }
            }
        }
    }
}

TEST(Markers, AprilTagCentreError) {
    const AprilTagFamily family = AprilTagFamily::Tag36h11;
    const int id = 7;
    const double tag_px = 120.0;
    std::uint32_t seed = 6000U;
    int decoded_count = 0;
    int missed = 0;
    int wrong = 0;
    double squared = 0.0;
    for (int iy = 0; iy < 5; ++iy) {
        for (int ix = 0; ix < 5; ++ix) {
            const double offset_x = -0.40 + 0.20 * static_cast<double>(ix) +
                                    0.013 * static_cast<double>(iy);
            const double offset_y = -0.35 + 0.17 * static_cast<double>(iy) -
                                    0.009 * static_cast<double>(ix);
            const test::AprilTagTruth truth = make_truth(
                family, id, tag_px, 0.37, 0.010, offset_x, offset_y);
            const FloatImage image = render_degraded(truth, 0.8, 0.01, seed++, 2);
            const std::vector<CodedTarget> decoded = detect_and_decode(
                image, family, default_max_hamming(family),
                quad_params_for(tag_px));
            if (decoded.empty()) {
                ++missed;
                continue;
            }
            const CodedTarget* match = nullptr;
            for (const CodedTarget& target : decoded) {
                if (target.code == id) {
                    match = &target;
                    break;
                }
            }
            if (match == nullptr) {
                ++wrong;
                continue;
            }
            const double dx = match->center.x() - truth.cx;
            const double dy = match->center.y() - truth.cy;
            squared += dx * dx + dy * dy;
            ++decoded_count;
        }
    }
    const double rms = decoded_count > 0
                           ? std::sqrt(squared / static_cast<double>(decoded_count))
                           : 1e9;
    std::printf("[apriltag centre] decoded=%d missed=%d wrong=%d rms=%.4f px\n",
                decoded_count, missed, wrong, rms);
    EXPECT_EQ(missed, 0);
    EXPECT_EQ(wrong, 0);
    EXPECT_EQ(decoded_count, 25);
    EXPECT_LE(rms, 0.2) << "measured " << rms << " px";
}

TEST(Markers, AprilTagCornerOrder) {
    const std::array<AprilTagFamily, 6> families = decode_families();
    const std::array<double, 5> rotations = {0.0, 0.6, 1.4, 2.2, 3.0};
    std::uint32_t seed = 9100U;
    int checked = 0;
    int misses = 0;
    for (AprilTagFamily family : families) {
        const TagFamily& table = tag_family(family);
        for (double rotation : rotations) {
            const test::AprilTagTruth truth = make_truth(family, 3, 120.0, rotation,
                                                   0.008, 0.27, 0.19);
            const FloatImage image =
                render_degraded(truth, 0.8, 0.01, seed++, 2);
            const std::vector<CodedTarget> decoded = detect_and_decode(
                image, family, default_max_hamming(family),
                quad_params_for(120.0));
            const CodedTarget* match = nullptr;
            for (const CodedTarget& target : decoded) {
                if (target.code == 3) {
                    match = &target;
                    break;
                }
            }
            if (match == nullptr) {
                std::printf("[apriltag corners] miss %s rotation %.2f\n",
                            table.name.c_str(), rotation);
                ++misses;
                continue;
            }
            const std::array<Eigen::Vector2d, 4> border =
                test::april_tag_corners(truth, false);
            const std::array<Eigen::Vector2d, 4> silhouette =
                test::april_tag_corners(truth, true);
            double border_error = 0.0;
            double silhouette_error = 0.0;
            for (std::size_t i = 0; i < border.size(); ++i) {
                border_error = std::max(
                    border_error, (match->corners[i] - border[i]).norm());
                silhouette_error = std::max(
                    silhouette_error, (match->corners[i] - silhouette[i]).norm());
            }
            EXPECT_LE(std::min(border_error, silhouette_error), 1.5)
                << table.name << " rotation " << rotation << " border "
                << border_error << " silhouette " << silhouette_error;
            ++checked;
        }
    }
    std::printf("[apriltag corners] checked=%d misses=%d\n", checked, misses);
    EXPECT_EQ(misses, 0);
    EXPECT_EQ(checked, 30);
}

TEST(Markers, AprilTagRejectsUnknownQuad) {
    const std::array<AprilTagFamily, 6> families = decode_families();
    const auto accepted_for = [&](const FloatImage& image,
                                  const QuadParams& params) {
        int accepted = 0;
        for (AprilTagFamily family : families) {
            accepted += static_cast<int>(detect_and_decode(
                image, family, default_max_hamming(family), params).size());
        }
        return accepted;
    };

    const int size = 140;
    QuadParams params;
    params.min_side = 6;
    params.max_side = 120;
    params.threshold = 0.02F;

    FloatImage solid(size, size, 0.95F);
    FloatImage board(size, size, 0.95F);
    constexpr int kCell = 14;
    constexpr int kOrigin = 14;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (x >= 30 && x < 110 && y >= 30 && y < 110) {
                solid.at(x, y) = 0.05F;
            }
            const int cx = (x - kOrigin) / kCell;
            const int cy = (y - kOrigin) / kCell;
            const bool inside = x >= kOrigin && y >= kOrigin && cx < 8 && cy < 8 &&
                                x - kOrigin >= 0 && y - kOrigin >= 0;
            if (inside && ((cx + cy) % 2) == 0) {
                board.at(x, y) = 0.05F;
            }
        }
    }
    solid = gaussian_blur(solid, 1.0F);
    test::add_noise(solid, 0.01, 7101U);
    test::quantise_to_8bit(solid);
    board = gaussian_blur(board, 1.0F);
    test::add_noise(board, 0.01, 7102U);
    test::quantise_to_8bit(board);
    const int solid_accepted = accepted_for(solid, params);
    const int board_accepted = accepted_for(board, params);

    std::mt19937 engine(7200U);
    int random_rendered = 0;
    int random_accepted = 0;
    std::uint32_t noise_seed = 8000U;
    for (AprilTagFamily family :
         {AprilTagFamily::Tag36h11, AprilTagFamily::Standard41h12}) {
        const TagFamily& table = tag_family(family);
        const std::uint64_t mask =
            (std::uint64_t{1} << static_cast<unsigned>(table.bits)) - 1u;
        int rendered = 0;
        while (rendered < 15) {
            const std::uint32_t high = static_cast<std::uint32_t>(engine());
            const std::uint32_t low = static_cast<std::uint32_t>(engine());
            const std::uint64_t word =
                ((static_cast<std::uint64_t>(high) << 32) |
                 static_cast<std::uint64_t>(low)) &
                mask;
            const std::uint64_t layout =
                reverse_layout_word(word, table.bits);
            if (match_tag_word(layout, table, default_max_hamming(family))) {
                continue;
            }
            test::AprilTagTruth truth =
                make_truth(family, 0, 80.0,
                           0.29 * static_cast<double>(rendered), 0.010, 0.2, 0.3);
            truth.use_word = true;
            truth.word = word;
            const FloatImage image =
                render_degraded(truth, 0.8, 0.02, noise_seed++, 1);
            random_accepted += accepted_for(image, quad_params_for(80.0));
            ++rendered;
            ++random_rendered;
        }
    }
    std::printf(
        "[apriltag unknown] solid=%d checkerboard=%d random=%d/%d accepted\n",
        solid_accepted, board_accepted, random_accepted, random_rendered);
    EXPECT_EQ(solid_accepted, 0);
    EXPECT_EQ(board_accepted, 0);
    EXPECT_EQ(random_accepted, 0);
}

TEST(Markers, AprilTagDistanceBoundary) {
    const AprilTagFamily family = AprilTagFamily::Tag36h11;
    const TagFamily& table = tag_family(family);
    const int allowed = default_max_hamming(family);
    ASSERT_GE(allowed, 1);
    const std::uint64_t code = table.codes.front();
    const std::uint64_t layout = reverse_layout_word(code, table.bits);

    const std::optional<TagMatch> exact = match_tag_word(layout, table, allowed);
    ASSERT_TRUE(exact.has_value());
    EXPECT_EQ(exact->id, 0);
    EXPECT_EQ(exact->distance, 0);

    const std::uint64_t one = layout ^ 1u;
    const std::optional<TagMatch> corrected = match_tag_word(one, table, 1);
    ASSERT_TRUE(corrected.has_value());
    EXPECT_EQ(corrected->id, 0);
    EXPECT_EQ(corrected->distance, 1);
    EXPECT_FALSE(match_tag_word(one, table, 0).has_value());

    std::uint64_t far = layout;
    for (int i = 0; i <= allowed; ++i) {
        far ^= std::uint64_t{1} << static_cast<unsigned>(i);
    }
    EXPECT_FALSE(match_tag_word(far, table, allowed).has_value());
    const std::optional<TagMatch> beyond = match_tag_word(far, table, allowed + 1);
    ASSERT_TRUE(beyond.has_value());
    EXPECT_EQ(beyond->id, 0);
    EXPECT_EQ(beyond->distance, allowed + 1);
    // Values above the family's correction budget are clamped to it.
    EXPECT_TRUE(match_tag_word(far, table, 99).has_value());

    test::AprilTagTruth corrupted =
        make_truth(family, 0, 120.0, 0.21, 0.010, 0.25, 0.15);
    corrupted.use_word = true;
    corrupted.word = code ^ 1u;
    const FloatImage image = render_degraded(corrupted, 0.8, 0.01, 8100U, 2);
    const QuadParams params = quad_params_for(120.0);
    EXPECT_TRUE(
        detect_and_decode(image, family, 0, params).empty())
        << "a one-bit error must not decode when correction is disabled";
    const std::vector<CodedTarget> loose =
        detect_and_decode(image, family, 1, params);
    ASSERT_FALSE(loose.empty());
    EXPECT_EQ(loose.front().code, 0);
    EXPECT_EQ(loose.front().hamming, 1);

    test::AprilTagTruth distant =
        make_truth(family, 0, 120.0, 0.21, 0.010, 0.25, 0.15);
    distant.use_word = true;
    distant.word = code;
    for (int i = 0; i <= allowed; ++i) {
        distant.word ^= std::uint64_t{1} << static_cast<unsigned>(i);
    }
    const FloatImage distant_image =
        render_degraded(distant, 0.8, 0.01, 8101U, 2);
    EXPECT_TRUE(
        detect_and_decode(distant_image, family, allowed, params).empty());
    const std::vector<CodedTarget> accepted =
        detect_and_decode(distant_image, family, allowed + 1, params);
    ASSERT_FALSE(accepted.empty());
    EXPECT_EQ(accepted.front().code, 0);
    EXPECT_EQ(accepted.front().hamming, allowed + 1);
}

TEST(Markers, AprilTagDecode36h11RealPhotos) {
    const std::filesystem::path directory =
        std::string(ALPHASHAPE_TEST_FIXTURE_DIR) + "/markers/apriltag_real";
    if (!std::filesystem::is_directory(directory)) {
        GTEST_SKIP() << "no " << directory.string() << " fixture directory";
    }
    int images = 0;
    int decoded = 0;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char character) {
                           return static_cast<char>(std::tolower(character));
                       });
        if (extension != ".png" && extension != ".jpg" && extension != ".jpeg") {
            continue;
        }
        ++images;
        try {
            const Image color = read_image(entry.path().string());
            const FloatImage gray = color.to_luminance();
            const std::vector<CodedTarget> found =
                detect_and_decode(gray, AprilTagFamily::Tag36h11,
                                  default_max_hamming(AprilTagFamily::Tag36h11),
                                  QuadParams{});
            decoded += static_cast<int>(found.size());
        } catch (const Error&) {
            continue;
        }
    }
    std::printf("[apriltag real] images=%d decodes=%d\n", images, decoded);
    if (images == 0) {
        GTEST_SKIP() << "fixture directory is empty";
    }
    EXPECT_GT(decoded, 0) << "no 36h11 tag decoded from the fixture photos";
}

}  // namespace
}  // namespace alphashape
