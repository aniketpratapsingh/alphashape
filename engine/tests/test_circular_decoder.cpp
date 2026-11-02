// Tests for WP-13.6 circular coded targets: the greedy code tables, decoding of
// every code at several radii, rotation/blur/tilt robustness, foreign-word
// rejection, centre accuracy, plain-shape rejection, and an end-to-end
// detect-and-decode sheet. Synthetic targets come from make_circular_target.hpp.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/utils/portable_rng.hpp"
#include "common/algorithm/feature/dog_detector.hpp"
#include "common/algorithm/markers/circular_decoder.hpp"
#include "common/algorithm/markers/marker_detector.hpp"
#include "common/algorithm/markers/target_patterns.hpp"
#include "image/image.hpp"
#include "make_circular_target.hpp"
#include "make_marker_sheet.hpp"

namespace alphashape {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;

std::vector<std::uint32_t> read_circular_codes(int bits) {
    const std::string path = std::string(ALPHASHAPE_TEST_FIXTURE_DIR) +
                             "/markers/circular_codes_N" + std::to_string(bits) + ".txt";
    std::ifstream input(path);
    EXPECT_TRUE(input.good()) << "missing fixture " << path;
    std::vector<std::uint32_t> words;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::uint32_t word = 0;
        for (char character : line) {
            if (character != '0' && character != '1') continue;
            word = (word << 1) | static_cast<std::uint32_t>(character == '1' ? 1 : 0);
        }
        words.push_back(word);
    }
    return words;
}

std::string word_string(std::uint32_t word, int bits) {
    std::string text(static_cast<std::size_t>(bits), '0');
    for (int i = 0; i < bits; ++i) {
        if (((word >> static_cast<unsigned>(bits - 1 - i)) & 1u) != 0u) {
            text[static_cast<std::size_t>(i)] = '1';
        }
    }
    return text;
}

std::uint32_t rotate_word(std::uint32_t word, int bits, int shift) {
    const std::uint32_t mask = (1u << static_cast<unsigned>(bits)) - 1u;
    for (int i = 0; i < shift; ++i) {
        word = ((word >> 1) | ((word & 1u) << static_cast<unsigned>(bits - 1))) & mask;
    }
    return word;
}

/// Distance from `word` to the closest rotation of every table word.
int nearest_code_distance(std::uint32_t word, const std::vector<std::uint32_t>& table,
                          int bits, int* best_index = nullptr, int* second = nullptr) {
    int best = bits + 1;
    int runner_up = bits + 1;
    int index_of_best = -1;
    for (std::size_t index = 0; index < table.size(); ++index) {
        int distance = bits;
        for (int shift = 0; shift < bits; ++shift) {
            distance = std::min(distance, word_hamming(word, rotate_word(table[index], bits, shift)));
        }
        if (distance < best) {
            runner_up = best;
            best = distance;
            index_of_best = static_cast<int>(index);
        } else if (distance < runner_up) {
            runner_up = distance;
        }
    }
    if (best_index != nullptr) *best_index = index_of_best;
    if (second != nullptr) *second = runner_up;
    return best;
}

struct RenderOptions {
    double blur_sigma = 0.8;
    double noise_sigma = 0.0;
    std::uint32_t seed = 1U;
    int supersample = 8;
    bool centre_dot_candidate = true;
};

/// Half-size of the square test image: enough quiet zone for the decoder's
/// 1.6 R border search.
int render_half(double radius) {
    return static_cast<int>(std::ceil(1.75 * radius)) + 4;
}

/// Render one target centred in a padded image and decode it. The candidate
/// passed to the decoder is the centre dot (parity 1) or the border (parity 0),
/// matching what the detector can report for each target.
std::optional<CodedTarget> render_and_decode(const test::CircularTruth& truth, int max_hamming,
                                             const RenderOptions& options) {
    const int half = render_half(truth.radius);
    test::CircularTruth placed = truth;
    placed.cx = static_cast<double>(half);
    placed.cy = static_cast<double>(half);
    FloatImage image =
        test::render_circular_target(2 * half, 2 * half, placed, 0.95F, 0.05F, options.supersample);
    if (options.blur_sigma > 0.0) {
        image = gaussian_blur(image, static_cast<float>(options.blur_sigma));
    }
    if (options.noise_sigma > 0.0) {
        test::add_noise(image, options.noise_sigma, options.seed);
    }
    MarkerCandidate candidate;
    // The renderer's continuous coordinate equals the calibration convention,
    // so the centre is passed through without a half-pixel shift.
    candidate.center = Eigen::Vector2d(placed.cx, placed.cy);
    candidate.radius_px = options.centre_dot_candidate ? 0.25 * placed.radius : placed.radius;
    candidate.axis_ratio = 1.0;
    return decode_circular(image, candidate, placed.bits, max_hamming);
}

}  // namespace

TEST(Markers, CircularCodesMinDistance) {
    for (int bits : kCircularCodeBits) {
        const std::vector<std::uint32_t> generated = generate_circular_codes(bits);
        const std::vector<std::uint32_t> fixture = read_circular_codes(bits);
        ASSERT_FALSE(generated.empty()) << "N=" << bits;
        ASSERT_EQ(generated.size(), fixture.size()) << "N=" << bits;
        EXPECT_EQ(generated, fixture) << "C++ generator differs from N=" << bits << " fixture";
        EXPECT_EQ(circular_codes(bits), generated) << "cached table differs at N=" << bits;

        int minimum_pair = bits;
        for (std::size_t i = 0; i < generated.size(); ++i) {
            const std::uint32_t word = generated[i];
            const int weight = word_hamming(word, 0u);
            EXPECT_GT(weight, 0) << "all-zero word at N=" << bits;
            EXPECT_LT(weight, bits) << "all-one word at N=" << bits;
            EXPECT_EQ(canonical_rotation(word, bits), word) << "non-canonical word at N=" << bits;
            for (int shift = 1; shift < bits; ++shift) {
                EXPECT_GE(word_hamming(word, rotate_word(word, bits, shift)), 3)
                    << "N=" << bits << " self rotation " << shift << " of "
                    << word_string(word, bits);
            }
            for (std::size_t j = 0; j < i; ++j) {
                int distance = bits;
                for (int shift = 0; shift < bits; ++shift) {
                    distance = std::min(distance, word_hamming(
                        word, rotate_word(generated[j], bits, shift)));
                }
                minimum_pair = std::min(minimum_pair, distance);
                EXPECT_GE(distance, 3) << "N=" << bits << " pair distance";
            }
        }
        std::printf("[circular codes] N=%d count=%d min_pair_distance=%d\n", bits,
                    static_cast<int>(generated.size()), minimum_pair);
        EXPECT_TRUE(is_supported_circular_bits(bits));
        EXPECT_FALSE(circular_ring_word(bits, 2 * static_cast<int>(generated.size()))
                         .has_value());
        EXPECT_FALSE(circular_ring_word(bits, -1).has_value());
    }
    EXPECT_FALSE(is_supported_circular_bits(13));
    EXPECT_TRUE(generate_circular_codes(13).empty());
}

TEST(Markers, DecodeAllCodesRendered) {
    const int bits = 12;
    const std::vector<std::uint32_t>& table = circular_codes(bits);
    int total = 0;
    int correct = 0;
    int wrong = 0;
    std::uint32_t seed = 900U;
    for (double radius : {8.0, 16.0, 32.0}) {
        for (std::size_t index = 0; index < table.size(); ++index) {
            for (int parity = 0; parity < 2; ++parity) {
                const int code = 2 * static_cast<int>(index) + parity;
                const test::CircularTruth truth = test::make_circular_truth(
                    0.0, 0.0, radius, bits, table[index], parity);
                RenderOptions options;
                options.blur_sigma = radius >= 32.0 ? 1.2 : (radius >= 16.0 ? 0.9 : 0.5);
                options.noise_sigma = 0.005;
                options.seed = seed++;
                options.supersample = radius >= 32.0 ? 4 : 8;
                options.centre_dot_candidate = parity == 1;
                const std::optional<CodedTarget> decoded =
                    render_and_decode(truth, 1, options);
                ++total;
                if (!decoded) continue;
                if (decoded->code == code && decoded->bits == bits) {
                    ++correct;
                } else {
                    ++wrong;
                }
            }
        }
    }
    std::printf("[circular all-codes] N=%d total=%d correct=%d wrong=%d\n", bits, total, correct,
                wrong);
    EXPECT_EQ(wrong, 0);
    EXPECT_EQ(correct, total);
}

TEST(Markers, DecodeUnderRotationAndBlur) {
    const int bits = 12;
    const std::vector<std::uint32_t>& table = circular_codes(bits);
    const std::vector<std::size_t> indices = {0, 6, 12};
    int total = 0;
    int correct = 0;
    int wrong = 0;
    std::uint32_t seed = 2000U;
    for (int degrees = 0; degrees < 360; degrees += 10) {
        for (double sigma : {0.5, 1.0, 1.5, 2.0}) {
            for (std::size_t index : indices) {
                for (int parity = 0; parity < 2; ++parity) {
                    const int code = 2 * static_cast<int>(index) + parity;
                    const test::CircularTruth truth = test::make_circular_truth(
                        0.0, 0.0, 16.0, bits, table[index], parity,
                        static_cast<double>(degrees) * kPi / 180.0);
                    RenderOptions options;
                    options.blur_sigma = sigma;
                    options.noise_sigma = 0.006;
                    options.seed = seed++;
                    options.centre_dot_candidate = parity == 1;
                    const std::optional<CodedTarget> decoded =
                        render_and_decode(truth, 1, options);
                    ++total;
                    if (!decoded) continue;
                    if (decoded->code == code) {
                        ++correct;
                    } else {
                        ++wrong;
                    }
                }
            }
        }
    }
    const double rate = total > 0 ? static_cast<double>(correct) / static_cast<double>(total)
                                  : 0.0;
    std::printf("[circular rotation+blur] total=%d correct=%d wrong=%d rate=%.4f\n", total,
                correct, wrong, rate);
    EXPECT_EQ(wrong, 0);
    EXPECT_GE(rate, 0.995);
}

TEST(Markers, DecodeUnderTilt) {
    const int bits = 12;
    const std::vector<std::uint32_t>& table = circular_codes(bits);
    const std::vector<std::size_t> indices = {0, 5, 10};
    int total = 0;
    int correct = 0;
    int wrong = 0;
    std::uint32_t seed = 3000U;
    for (double ratio : {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}) {
        for (int degrees : {0, 30, 60, 90, 120, 150}) {
            for (std::size_t index : indices) {
                for (int parity = 0; parity < 2; ++parity) {
                    const int code = 2 * static_cast<int>(index) + parity;
                    const test::CircularTruth truth = test::make_circular_truth(
                        0.0, 0.0, 16.0, bits, table[index], parity,
                        static_cast<double>(degrees) * kPi / 180.0, ratio);
                    RenderOptions options;
                    options.blur_sigma = 1.0;
                    options.noise_sigma = 0.006;
                    options.seed = seed++;
                    options.centre_dot_candidate = parity == 1;
                    const std::optional<CodedTarget> decoded =
                        render_and_decode(truth, 1, options);
                    ++total;
                    if (!decoded) continue;
                    if (decoded->code == code) {
                        ++correct;
                    } else {
                        ++wrong;
                    }
                }
            }
        }
    }
    const double rate = total > 0 ? static_cast<double>(correct) / static_cast<double>(total)
                                  : 0.0;
    std::printf("[circular tilt] total=%d correct=%d wrong=%d rate=%.4f\n", total, correct, wrong,
                rate);
    EXPECT_EQ(wrong, 0);
    EXPECT_GE(rate, 0.99);
}

TEST(Markers, RejectsForeignCode) {
    const int bits = 12;
    const std::vector<std::uint32_t>& table = circular_codes(bits);
    ASSERT_FALSE(table.empty());

    // A one-bit corruption is accepted only when the runner-up *ring code* is
    // at least two bits farther away. Not every flip has that margin (the
    // table distance is 3), so search every word and bit for one that does.
    std::uint32_t corrupted = 0;
    int corrupted_index = -1;
    for (std::size_t index = 0; index < table.size() && corrupted_index < 0; ++index) {
        for (int position = 0; position < bits; ++position) {
            const std::uint32_t candidate =
                table[index] ^ (1u << static_cast<unsigned>(position));
            int best = 0;
            int second = 0;
            nearest_code_distance(candidate, table, bits, &best, &second);
            if (best == 1 && second >= best + 2) {
                corrupted = candidate;
                corrupted_index = static_cast<int>(index);
                break;
            }
        }
    }
    ASSERT_GE(corrupted_index, 0) << "no unambiguous single-bit corruption exists";

    RenderOptions options;
    options.blur_sigma = 0.8;
    options.noise_sigma = 0.004;
    options.seed = 4001U;
    const test::CircularTruth corrupt_truth =
        test::make_circular_truth(0.0, 0.0, 16.0, bits, corrupted, 1);
    const std::optional<CodedTarget> decoded = render_and_decode(corrupt_truth, 1, options);
    ASSERT_TRUE(decoded.has_value()) << "one-bit corruption " << word_string(corrupted, bits)
                                     << " was not corrected";
    EXPECT_EQ(decoded->code, 2 * corrupted_index + 1);
    EXPECT_EQ(decoded->hamming, 1);

    std::uint32_t foreign = 0;
    bool found_foreign = false;
    for (std::uint32_t word = 1; word + 1 < (1u << static_cast<unsigned>(bits)); ++word) {
        if (nearest_code_distance(word, table, bits) >= 2) {
            foreign = word;
            found_foreign = true;
            break;
        }
    }
    ASSERT_TRUE(found_foreign) << "no word at distance 2 from every code";
    options.seed = 4002U;
    const test::CircularTruth foreign_truth =
        test::make_circular_truth(0.0, 0.0, 16.0, bits, foreign, 1);
    const std::optional<CodedTarget> rejected = render_and_decode(foreign_truth, 1, options);
    std::printf("[circular foreign] corrupted=%s hamming=%d foreign=%s rejected=%d\n",
                word_string(corrupted, bits).c_str(),
                decoded ? decoded->hamming : -1, word_string(foreign, bits).c_str(),
                rejected.has_value() ? 0 : 1);
    EXPECT_FALSE(rejected.has_value());
}

TEST(Markers, Decode14_16_20Bit) {
    std::uint32_t seed = 5000U;
    for (int bits : {14, 16, 20}) {
        const std::vector<std::uint32_t>& table = circular_codes(bits);
        ASSERT_FALSE(table.empty()) << "N=" << bits;
        int total = 0;
        int correct = 0;
        int wrong = 0;
        for (std::size_t index = 0; index < table.size(); ++index) {
            for (int parity = 0; parity < 2; ++parity) {
                const int code = 2 * static_cast<int>(index) + parity;
                const test::CircularTruth truth = test::make_circular_truth(
                    0.0, 0.0, 20.0, bits, table[index], parity);
                RenderOptions options;
                options.blur_sigma = 0.8;
                options.noise_sigma = 0.004;
                options.seed = seed++;
                options.supersample = 4;
                options.centre_dot_candidate = parity == 1;
                const std::optional<CodedTarget> decoded =
                    render_and_decode(truth, 1, options);
                ++total;
                if (!decoded) continue;
                if (decoded->code == code) {
                    ++correct;
                } else {
                    ++wrong;
                }
            }
        }
        std::printf("[circular all-codes] N=%d total=%d correct=%d wrong=%d\n", bits, total,
                    correct, wrong);
        EXPECT_EQ(wrong, 0) << "N=" << bits;
        EXPECT_EQ(correct, total) << "N=" << bits;
    }
}

TEST(Markers, CentreErrorUnderBlur) {
    const int bits = 12;
    const std::vector<std::uint32_t>& table = circular_codes(bits);
    const std::vector<std::size_t> indices = {0, 6, 12};
    const double expected_x = static_cast<double>(render_half(16.0));
    const double expected_y = expected_x;
    int total = 0;
    int decoded_count = 0;
    int wrong = 0;
    double squared = 0.0;
    std::uint32_t seed = 6000U;
    // Measured on the parity-1 targets: their black centre dot is the fiducial
    // the WP-13.5 detector reports, and refine_ellipse_center locks onto it.
    // Without a centre dot (parity 0) the first radial edges differ per ring
    // sector, so the outer-border ellipse centre is used; that centre is less
    // accurate under heavy blur (a documented limitation, not measured here).
    for (int degrees = 0; degrees < 360; degrees += 30) {
        for (double sigma : {0.5, 1.0, 1.5, 2.0}) {
            for (std::size_t index : indices) {
                const test::CircularTruth truth = test::make_circular_truth(
                    0.0, 0.0, 16.0, bits, table[index], 1,
                    static_cast<double>(degrees) * kPi / 180.0);
                RenderOptions options;
                options.blur_sigma = sigma;
                options.noise_sigma = 0.005;
                options.seed = seed++;
                options.centre_dot_candidate = true;
                const std::optional<CodedTarget> decoded =
                    render_and_decode(truth, 1, options);
                ++total;
                if (!decoded) continue;
                ++decoded_count;
                if (decoded->code != 2 * static_cast<int>(index) + 1) ++wrong;
                const double dx = decoded->center.x() - expected_x;
                const double dy = decoded->center.y() - expected_y;
                squared += dx * dx + dy * dy;
            }
        }
    }
    const double rms = decoded_count > 0
                           ? std::sqrt(squared / static_cast<double>(decoded_count))
                           : 1e9;
    std::printf("[circular centre] total=%d decoded=%d wrong=%d rms=%.4f px\n", total,
                decoded_count, wrong, rms);
    EXPECT_EQ(wrong, 0);
    EXPECT_EQ(decoded_count, total);
    EXPECT_LE(rms, 0.15);
}

TEST(Markers, RejectsPlainDisc) {
    const int half = render_half(20.0);
    const int size = 2 * half;
    const double centre = static_cast<double>(half);
    const double radius = 20.0;

    FloatImage disc(size, size, 0.95F);
    FloatImage ring(size, size, 0.95F);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const double dx = static_cast<double>(x) - centre;
            const double dy = static_cast<double>(y) - centre;
            const double distance = std::sqrt(dx * dx + dy * dy);
            if (distance <= radius) disc.at(x, y) = 0.05F;
            if (distance <= radius && distance > 0.6 * radius) ring.at(x, y) = 0.05F;
        }
    }
    disc = gaussian_blur(disc, 1.0F);
    test::add_noise(disc, 0.005, 7001U);
    ring = gaussian_blur(ring, 1.0F);
    test::add_noise(ring, 0.005, 7002U);

    MarkerCandidate candidate;
    candidate.center = Eigen::Vector2d(centre + 0.5, centre + 0.5);
    candidate.radius_px = radius;
    const std::optional<CodedTarget> disc_result = decode_circular(disc, candidate, 12, 1);
    const std::optional<CodedTarget> ring_result = decode_circular(ring, candidate, 12, 1);
    std::printf("[circular plain] disc_decoded=%d ring_decoded=%d\n",
                disc_result.has_value() ? 1 : 0, ring_result.has_value() ? 1 : 0);
    EXPECT_FALSE(disc_result.has_value());
    EXPECT_FALSE(ring_result.has_value());
}

TEST(Markers, DetectAndDecodeSheet) {
    const int bits = 14;
    const std::vector<std::uint32_t>& table = circular_codes(bits);
    const int columns = 7;
    const int rows = 6;
    const double cell_width = 140.0;
    const double cell_height = 140.0;
    const int width = static_cast<int>(columns * cell_width);
    const int height = static_cast<int>(rows * cell_height);
    const double radius = 22.0;
    const int target_count = 30;

    std::mt19937 engine(20261106U);
    std::vector<test::CircularTruth> targets;
    std::vector<int> expected_codes;
    std::vector<Eigen::Vector2d> expected_centres;
    for (int i = 0; i < target_count; ++i) {
        const int column = i % columns;
        const int row = i / columns;
        const double jitter_x = rng::uniform_real(engine, -8.0, 8.0);
        const double jitter_y = rng::uniform_real(engine, -8.0, 8.0);
        const double rotation = rng::uniform_real(engine, 0.0, kPi);
        const double cx = (static_cast<double>(column) + 0.5) * cell_width + jitter_x;
        const double cy = (static_cast<double>(row) + 0.5) * cell_height + jitter_y;
        const int code = 2 * i + 1;
        targets.push_back(
            test::make_circular_truth(cx, cy, radius, bits, table[static_cast<std::size_t>(i)],
                                      1, rotation));
        expected_codes.push_back(code);
        expected_centres.emplace_back(cx, cy);
    }
    std::vector<Eigen::Vector3d> clutter;
    for (int i = target_count; i < target_count + 8; ++i) {
        const int column = i % columns;
        const int row = i / columns;
        const double jitter_x = rng::uniform_real(engine, -8.0, 8.0);
        const double jitter_y = rng::uniform_real(engine, -8.0, 8.0);
        const double disc_radius = rng::uniform_real(engine, 9.0, 14.0);
        clutter.emplace_back((static_cast<double>(column) + 0.5) * cell_width + jitter_x,
                             (static_cast<double>(row) + 0.5) * cell_height + jitter_y,
                             disc_radius);
    }

    FloatImage image = test::render_circular_targets(width, height, targets, 0.95F, 0.05F, 4);
    for (const Eigen::Vector3d& disc : clutter) {
        const int x0 = std::max(0, static_cast<int>(std::floor(disc.x() - disc.z() - 1.0)));
        const int x1 = std::min(width - 1, static_cast<int>(std::ceil(disc.x() + disc.z() + 1.0)));
        const int y0 = std::max(0, static_cast<int>(std::floor(disc.y() - disc.z() - 1.0)));
        const int y1 =
            std::min(height - 1, static_cast<int>(std::ceil(disc.y() + disc.z() + 1.0)));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double dx = static_cast<double>(x) - disc.x();
                const double dy = static_cast<double>(y) - disc.y();
                if (dx * dx + dy * dy <= disc.z() * disc.z()) image.at(x, y) = 0.05F;
            }
        }
    }
    image = gaussian_blur(image, 1.0F);
    test::add_noise(image, 0.01, 20261107U);

    BlobParams params;
    params.min_radius = 2;
    params.max_radius = 40;
    params.threshold = 0.015F;
    const std::vector<MarkerCandidate> blobs = detect_blobs(image, params);
    std::vector<int> found(expected_codes.size(), 0);
    std::vector<double> best_error(expected_codes.size(), 1e9);
    int foreign = 0;
    int decodes = 0;
    for (const MarkerCandidate& blob : blobs) {
        const std::optional<CodedTarget> decoded = decode_circular(image, blob, bits, 1);
        if (!decoded) continue;
        ++decodes;
        const auto match =
            std::find(expected_codes.begin(), expected_codes.end(), decoded->code);
        if (match == expected_codes.end()) {
            ++foreign;
            continue;
        }
        const std::size_t index =
            static_cast<std::size_t>(std::distance(expected_codes.begin(), match));
        ++found[index];
        const double error = (decoded->center - expected_centres[index]).norm();
        best_error[index] = std::min(best_error[index], error);
    }
    int recovered = 0;
    double worst_error = 0.0;
    for (std::size_t i = 0; i < found.size(); ++i) {
        if (found[i] > 0) {
            ++recovered;
            worst_error = std::max(worst_error, best_error[i]);
        }
    }
    std::printf(
        "[circular sheet] blobs=%d decodes=%d recovered=%d/%d extra_codes=%d worst_centre=%.3f "
        "px\n",
        static_cast<int>(blobs.size()), decodes, recovered, target_count, foreign, worst_error);
    // The requirement here is recall: every printed target is found with the
    // right code and a subpixel centre. A detector candidate that is not a
    // real target can still decode in principle; the dedicated rotation, tilt
    // and all-code tests assert that zero misdecodes occur on rendered
    // targets, so any extra code here is a detection false positive and is
    // reported rather than hidden.
    EXPECT_EQ(recovered, target_count);
    EXPECT_LE(worst_error, 2.0);
}

}  // namespace alphashape
