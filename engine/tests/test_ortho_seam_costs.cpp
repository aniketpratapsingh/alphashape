// test_ortho_seam_costs.cpp -- opt-in pairwise seam costs (Q4a).
//
// Gradient matching (Agarwala et al. 2004) and the small-region seam merge are
// both off by default. These cases pin down that the default is unchanged, that
// a constant brightness offset adds no gradient cost while a shifted stripe
// does, and that the merge pass removes small islands without holes or pinned
// edits. Synthetic tiles only: the costs are properties of the rectified
// colours, so the smallest grid that shows the effect is used.

#include "aerial_fixture.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/ortho/ghosting.hpp"
#include "common/algorithm/ortho/orthorectify.hpp"
#include "common/algorithm/ortho/seamlines.hpp"
#include "common/tasks/registry.hpp"

namespace alphashape {
namespace {

using ortho::assign_seams;
using ortho::energy;
using ortho::filter_label_speckles;
using ortho::OrthoTile;
using ortho::SeamOptions;

constexpr std::uint8_t kBackground = 180;

/// A full-coverage tile over `w` x `h` with a flat background and one score.
OrthoTile make_tile(int w, int h, float score) {
    OrthoTile tile;
    tile.x0 = 0;
    tile.y0 = 0;
    tile.width = w;
    tile.height = h;
    tile.rgb.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3, kBackground);
    tile.valid.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 1);
    tile.score.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), score);
    return tile;
}

/// A full-coverage tile with independent per-cell noise, so two tiles disagree.
OrthoTile make_noise_tile(int w, int h, float score, std::mt19937& rng) {
    OrthoTile tile = make_tile(w, h, score);
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            const int noise = static_cast<int>(rng() % 41U);  // one draw per statement
            const std::uint8_t value = static_cast<std::uint8_t>(60 + noise);
            const std::size_t cell = tile.index(col, row);
            for (int c = 0; c < 3; ++c) {
                tile.rgb[cell * 3 + static_cast<std::size_t>(c)] = value;
            }
        }
    }
    return tile;
}

/// Paint one full-height column of a tile, all three channels.
void set_column(OrthoTile& tile, int col, std::uint8_t value) {
    for (int row = 0; row < tile.height; ++row) {
        const std::size_t cell = tile.index(col, row);
        for (int c = 0; c < 3; ++c) {
            tile.rgb[cell * 3 + static_cast<std::size_t>(c)] = value;
        }
    }
}

std::size_t cell_at(int width, int col, int row) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(col);
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Gradient term: invariant to a constant brightness offset
// ---------------------------------------------------------------------------

TEST(SeamCosts, GradientTermZeroForConstantOffset) {
    constexpr int kWidth = 32;
    constexpr int kHeight = 24;
    std::mt19937 rng(4321U);
    std::vector<OrthoTile> tiles;
    tiles.push_back(make_noise_tile(kWidth, kHeight, 0.7F, rng));
    const OrthoTile offset_copy = tiles[0];
    tiles.push_back(offset_copy);
    for (std::uint8_t& value : tiles[1].rgb) {
        value = static_cast<std::uint8_t>(value + 25);  // constant offset everywhere
    }

    const std::size_t cells = cell_at(kWidth, 0, kHeight);
    std::vector<std::int32_t> all_a(cells, 0);
    std::vector<std::int32_t> split(cells, 0);
    std::vector<std::int32_t> interleaved(cells, 0);
    for (int row = 0; row < kHeight; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const std::size_t cell = cell_at(kWidth, col, row);
            split[cell] = col < kWidth / 2 ? 0 : 1;
            interleaved[cell] = (col / 3) % 2 == 0 ? 0 : 1;
        }
    }

    SeamOptions plain;
    SeamOptions gradient;
    gradient.gradient_weight = 1.0;
    for (const std::vector<std::int32_t>* labels : {&all_a, &split, &interleaved}) {
        EXPECT_DOUBLE_EQ(energy(kWidth, kHeight, tiles, *labels, plain),
                         energy(kWidth, kHeight, tiles, *labels, gradient));
    }
}

// ---------------------------------------------------------------------------
// 2. Gradient term: penalises a shifted stripe, and moves the seam off it
// ---------------------------------------------------------------------------

TEST(SeamCosts, GradientTermPenalisesStructuralMismatch) {
    constexpr int kWidth = 48;
    constexpr int kHeight = 32;
    std::vector<OrthoTile> tiles;
    tiles.push_back(make_tile(kWidth, kHeight, 1.0F));
    tiles.push_back(make_tile(kWidth, kHeight, 1.0F));
    // Tile B's stripe sits 3 columns left of tile A's: the parallax case.
    const std::uint8_t pattern[6] = {40, 200, 40, 200, 40, 200};
    for (int i = 0; i < 6; ++i) {
        set_column(tiles[0], 20 + i, pattern[i]);
        set_column(tiles[1], 17 + i, pattern[i]);
    }
    // A is the better camera on the left, B on the right, so a seam exists and
    // its natural place crosses the shifted stripes.
    for (int col = 0; col < kWidth; ++col) {
        const float score_a = col <= 22 ? 1.0F : 0.0F;
        const float score_b = col >= 23 ? 1.0F : 0.0F;
        for (int row = 0; row < kHeight; ++row) {
            const std::size_t cell = cell_at(kWidth, col, row);
            tiles[0].score[cell] = score_a;
            tiles[1].score[cell] = score_b;
        }
    }

    const std::size_t cells = cell_at(kWidth, 0, kHeight);
    std::vector<std::int32_t> crossing(cells, 0);
    for (int row = 0; row < kHeight; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            crossing[cell_at(kWidth, col, row)] = col <= 22 ? 0 : 1;
        }
    }
    SeamOptions plain;
    SeamOptions gradient;
    gradient.gradient_weight = 1.0;
    EXPECT_GT(energy(kWidth, kHeight, tiles, crossing, gradient),
              energy(kWidth, kHeight, tiles, crossing, plain));

    SeamOptions strong;
    strong.gradient_weight = 2.0;
    const std::vector<std::int32_t> labels = assign_seams(kWidth, kHeight, tiles, strong);
    // The shifted stripes occupy columns 17..25 (A: 20..25, B: 17..22), and the
    // structural mismatch reaches one column past each side. Assert no row
    // changes label in the zone [16, 26]: the seam must sit outside it.
    for (int row = 0; row < kHeight; ++row) {
        const std::int32_t first = labels[cell_at(kWidth, 16, row)];
        ASSERT_GE(first, 0) << "row " << row << " must be covered";
        for (int col = 16; col <= 26; ++col) {
            EXPECT_EQ(labels[cell_at(kWidth, col, row)], first)
                << "row " << row << " col " << col << " changed inside the stripe zone";
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Defaults are byte-identical
// ---------------------------------------------------------------------------

TEST(SeamCosts, DefaultIsByteIdentical) {
    constexpr int kWidth = 64;
    constexpr int kHeight = 64;
    std::mt19937 rng(20261031U);
    std::vector<OrthoTile> tiles;
    tiles.push_back(make_noise_tile(kWidth, kHeight, 0.55F, rng));
    tiles.push_back(make_noise_tile(kWidth, kHeight, 0.60F, rng));
    tiles.push_back(make_noise_tile(kWidth, kHeight, 0.65F, rng));

    const SeamOptions defaults;
    SeamOptions explicit_zero;
    explicit_zero.gradient_weight = 0.0;
    const std::vector<std::int32_t> a = assign_seams(kWidth, kHeight, tiles, defaults);
    const std::vector<std::int32_t> b = assign_seams(kWidth, kHeight, tiles, explicit_zero);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(a, b);

    // The BuildOrthomosaic task's two new parameters default to exactly the
    // explicit-off values, checked without running the full aerial fixture
    // (which needs depth maps and a DEM and is not the point of this case).
    Params params = make_params(builtin_registry().require("BuildOrthomosaic"));
    EXPECT_DOUBLE_EQ(params.get_float("seam_gradient_weight"), 0.0);
    EXPECT_EQ(params.get_int("seam_min_region"), std::int64_t{0});
}

// ---------------------------------------------------------------------------
// 4. Small-region merge: islands vanish, no holes, pinned cells untouched
// ---------------------------------------------------------------------------

TEST(SeamCosts, MinRegionRemovesIslands) {
    constexpr int kWidth = 32;
    constexpr int kHeight = 24;
    std::mt19937 rng(77U);
    std::vector<OrthoTile> tiles;
    tiles.push_back(make_noise_tile(kWidth, kHeight, 0.6F, rng));
    tiles.push_back(make_noise_tile(kWidth, kHeight, 0.7F, rng));

    const std::size_t cells = cell_at(kWidth, 0, kHeight);
    std::vector<std::int32_t> labels(cells, 0);
    const std::pair<int, int> island_one[] = {{5, 5}, {6, 5}, {7, 5}};
    const std::pair<int, int> island_two[] = {{20, 10}, {20, 11}, {21, 11}};
    for (const auto& p : island_one) {
        labels[cell_at(kWidth, p.first, p.second)] = 1;
    }
    for (const auto& p : island_two) {
        labels[cell_at(kWidth, p.first, p.second)] = 1;
    }
    // A pinned two-cell island must survive untouched, and a 12-cell region is
    // exactly at/above the threshold and must also stay.
    for (int col = 28; col <= 29; ++col) {
        labels[cell_at(kWidth, col, 2)] = 1;
    }
    for (int row = 15; row <= 17; ++row) {
        for (int col = 10; col <= 13; ++col) {
            labels[cell_at(kWidth, col, row)] = 1;
        }
    }
    std::vector<std::uint8_t> pinned(cells, 0);
    pinned[cell_at(kWidth, 28, 2)] = 1;
    pinned[cell_at(kWidth, 29, 2)] = 1;

    const int removed =
        filter_label_speckles(kWidth, kHeight, tiles, labels, &pinned, 10, nullptr);
    EXPECT_EQ(removed, 2) << "the two unpinned three-cell islands";
    for (const auto& p : island_one) {
        EXPECT_EQ(labels[cell_at(kWidth, p.first, p.second)], 0);
    }
    for (const auto& p : island_two) {
        EXPECT_EQ(labels[cell_at(kWidth, p.first, p.second)], 0);
    }
    EXPECT_EQ(labels[cell_at(kWidth, 28, 2)], 1);
    EXPECT_EQ(labels[cell_at(kWidth, 29, 2)], 1);
    for (int row = 15; row <= 17; ++row) {
        for (int col = 10; col <= 13; ++col) {
            EXPECT_EQ(labels[cell_at(kWidth, col, row)], 1);
        }
    }
    for (std::size_t cell = 0; cell < cells; ++cell) {
        EXPECT_GE(labels[cell], 0) << "cell " << cell << " must not become a hole";
    }
}

// ---------------------------------------------------------------------------
// 5. Negative task parameters are refused at the boundary
// ---------------------------------------------------------------------------

TEST(SeamCosts, NegativeParamsRejected) {
    Document document = aerial_fixture::make_truth_chunk_document();

    Params bad_gradient = make_params(builtin_registry().require("BuildOrthomosaic"));
    bad_gradient.set("seam_gradient_weight", ParamValue{-1.0});
    EXPECT_THROW(run_task(builtin_registry(), document, "BuildOrthomosaic", bad_gradient),
                 ValueError);

    Params bad_region = make_params(builtin_registry().require("BuildOrthomosaic"));
    bad_region.set("seam_min_region", ParamValue{std::int64_t{-1}});
    EXPECT_THROW(run_task(builtin_registry(), document, "BuildOrthomosaic", bad_region),
                 ValueError);
}

}  // namespace alphashape
