// test_ortho_ghosting.cpp -- multi-view consensus ghosting filter (Q4).
//
// Synthetic tiles only: the filter is a property of overlapping rectified
// colours, so the cases here build the smallest block that exercises it and
// assert on the masks and the resulting seams.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common/algorithm/ortho/blend.hpp"
#include "common/algorithm/ortho/ghosting.hpp"
#include "common/algorithm/ortho/orthorectify.hpp"
#include "common/algorithm/ortho/seamlines.hpp"

namespace alphashape {
namespace {

using ortho::assign_seams;
using ortho::detect_ghosts;
using ortho::filter_label_speckles;
using ortho::GhostOptions;
using ortho::OrthoTile;
using ortho::SeamOptions;

/// A tile over `w` x `h`, all cells valid, carrying the same faint deterministic
/// texture in every camera plus a small independent noise draw.
OrthoTile make_tile(int x0, int y0, int w, int h, std::mt19937& rng, float score = 0.5F) {
    OrthoTile tile;
    tile.x0 = x0;
    tile.y0 = y0;
    tile.width = w;
    tile.height = h;
    tile.rgb.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3, 0);
    tile.valid.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 1);
    tile.score.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), score);
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            const int base = 80 + (col * 13 + row * 7) % 40;
            const int noise = static_cast<int>(rng() % 5U);  // one draw per statement
            const std::uint8_t value = static_cast<std::uint8_t>(base + noise - 2);
            const std::size_t cell = tile.index(col, row);
            for (int c = 0; c < 3; ++c) {
                tile.rgb[cell * 3 + static_cast<std::size_t>(c)] = value;
            }
        }
    }
    return tile;
}

/// Paint a saturated rectangle, the "car".
void paint(OrthoTile& tile, int col0, int row0, int w, int h, std::uint8_t r,
           std::uint8_t g, std::uint8_t b) {
    for (int row = row0; row < row0 + h; ++row) {
        for (int col = col0; col < col0 + w; ++col) {
            if (!tile.contains(col, row)) {
                continue;
            }
            const std::size_t cell = tile.index(col, row);
            tile.rgb[cell * 3 + 0] = r;
            tile.rgb[cell * 3 + 1] = g;
            tile.rgb[cell * 3 + 2] = b;
        }
    }
}

std::size_t marked(const std::vector<std::uint8_t>& mask) {
    return static_cast<std::size_t>(
        std::count_if(mask.begin(), mask.end(), [](std::uint8_t v) { return v != 0; }));
}

TEST(OrthoGhosting, SingleMovingObjectIsMarkedOnlyInItsCamera) {
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    std::mt19937 rng(1234U);
    std::vector<OrthoTile> tiles;
    for (int i = 0; i < 5; ++i) {
        tiles.push_back(make_tile(0, 0, kWidth, kHeight, rng));
    }
    // The ghost camera is the argmax on the car and nowhere else.
    const std::size_t ghost = 2;
    paint(tiles[ghost], 20, 18, 12, 6, 255, 0, 0);
    for (int row = 18; row < 24; ++row) {
        for (int col = 20; col < 32; ++col) {
            tiles[ghost].score[tiles[ghost].index(col, row)] = 1.0F;
        }
    }

    const std::vector<std::vector<std::uint8_t>> masks =
        detect_ghosts(kWidth, kHeight, tiles, GhostOptions{});
    ASSERT_EQ(masks.size(), tiles.size());
    std::printf("[OrthoGhosting.Single] ghost=%zu others=%zu %zu %zu %zu\n",
                marked(masks[ghost]), marked(masks[0]), marked(masks[1]),
                marked(masks[3]), marked(masks[4]));
    EXPECT_GE(marked(masks[ghost]), 60U) << "the 12x6 car should be marked in its camera";
    for (std::size_t t = 0; t < masks.size(); ++t) {
        if (t != ghost) {
            EXPECT_LE(marked(masks[t]), 5U) << "only the ghost camera carries the object";
        }
    }

    SeamOptions options;
    options.avoid = &masks;
    const std::vector<std::int32_t> labels =
        assign_seams(kWidth, kHeight, tiles, options);
    std::size_t ghost_car_cells = 0;
    for (int row = 18; row < 24; ++row) {
        for (int col = 20; col < 32; ++col) {
            const std::size_t cell =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(kWidth) + static_cast<std::size_t>(col);
            if (labels[cell] == static_cast<std::int32_t>(ghost)) {
                ++ghost_car_cells;
            }
        }
    }
    EXPECT_EQ(ghost_car_cells, 0U)
        << "the seam labeller must not award the car to the ghost camera";
}

TEST(OrthoGhosting, MajorityObjectIsNotAGhost) {
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    std::mt19937 rng(77U);
    std::vector<OrthoTile> tiles;
    for (int i = 0; i < 5; ++i) {
        tiles.push_back(make_tile(0, 0, kWidth, kHeight, rng));
    }
    // A parked car in three of five views: the car IS the consensus.
    for (std::size_t t = 0; t < 3; ++t) {
        paint(tiles[t], 20, 18, 12, 6, 255, 0, 0);
    }
    const std::vector<std::vector<std::uint8_t>> masks =
        detect_ghosts(kWidth, kHeight, tiles, GhostOptions{});
    for (std::size_t t = 0; t < 3; ++t) {
        EXPECT_EQ(marked(masks[t]), 0U)
            << "a stationary object seen by the majority is not a ghost";
    }
}

TEST(OrthoGhosting, MajorityCarSurvivesIntoTheMosaic) {
    // A car painted in three of five views is the consensus, so the two views
    // WITHOUT the car are the deviant minority where the car stands. The seam
    // labeller must keep the car anyway, and the blended mosaic still shows it.
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    std::mt19937 rng(515U);
    std::vector<OrthoTile> tiles;
    for (int i = 0; i < 5; ++i) {
        tiles.push_back(make_tile(0, 0, kWidth, kHeight, rng));
    }
    for (std::size_t t = 0; t < 3; ++t) {
        paint(tiles[t], 20, 18, 12, 6, 255, 0, 0);
    }
    // The car-less views outscore the car views inside the footprint, so a
    // plain argmax would hide the car; the filter has to move it back.
    for (std::size_t t = 3; t < 5; ++t) {
        for (int row = 18; row < 24; ++row) {
            for (int col = 20; col < 32; ++col) {
                tiles[t].score[tiles[t].index(col, row)] = 0.9F;
            }
        }
    }
    for (std::size_t t = 0; t < 3; ++t) {
        for (int row = 18; row < 24; ++row) {
            for (int col = 20; col < 32; ++col) {
                tiles[t].score[tiles[t].index(col, row)] = 0.4F;
            }
        }
    }
    const std::vector<std::vector<std::uint8_t>> masks =
        detect_ghosts(kWidth, kHeight, tiles, GhostOptions{});
    for (std::size_t t = 0; t < 3; ++t) {
        EXPECT_EQ(static_cast<int>(masks[t][tiles[t].index(26, 21)]), 0)
            << "the car views agree with the consensus";
    }
    for (std::size_t t = 3; t < 5; ++t) {
        EXPECT_GT(static_cast<int>(masks[t][tiles[t].index(26, 21)]), 0)
            << "the two car-less cameras are the deviant minority at the car";
    }

    SeamOptions options;
    options.avoid = &masks;
    const std::vector<std::int32_t> labels = assign_seams(kWidth, kHeight, tiles, options);
    const std::size_t centre =
        static_cast<std::size_t>(21) * static_cast<std::size_t>(kWidth) + 26U;
    EXPECT_LT(labels[centre], 3) << "the car must go to a camera that saw it";

    const ortho::OrthoMosaic mosaic =
        ortho::blend(kWidth, kHeight, tiles, labels, ortho::Blending::Mosaic, 2, &masks);
    const std::uint8_t* rgb = mosaic.rgb.data() + centre * 3U;
    EXPECT_GT(static_cast<int>(rgb[0]), 150) << "the mosaic must still show the red car";
    EXPECT_LT(static_cast<int>(rgb[2]), 80);
}

TEST(OrthoGhosting, TwoViewsHaveNoConsensus) {
    constexpr int kWidth = 40;
    constexpr int kHeight = 40;
    std::mt19937 rng(99U);
    std::vector<OrthoTile> tiles;
    for (int i = 0; i < 5; ++i) {
        tiles.push_back(make_tile(0, 0, kWidth, kHeight, rng));
    }
    // Only two cameras see the left half.
    for (std::size_t t = 2; t < tiles.size(); ++t) {
        for (int row = 0; row < kHeight; ++row) {
            for (int col = 0; col < kWidth / 2; ++col) {
                tiles[t].valid[tiles[t].index(col, row)] = 0;
            }
        }
    }
    paint(tiles[0], 4, 12, 12, 6, 255, 0, 0);
    const std::vector<std::vector<std::uint8_t>> masks =
        detect_ghosts(kWidth, kHeight, tiles, GhostOptions{});
    std::size_t marked_in_region = 0;
    for (std::size_t t = 0; t < masks.size(); ++t) {
        for (int row = 0; row < kHeight; ++row) {
            for (int col = 0; col < kWidth / 2; ++col) {
                if (masks[t][tiles[t].index(col, row)] != 0) {
                    ++marked_in_region;
                }
            }
        }
    }
    EXPECT_EQ(marked_in_region, 0U)
        << "fewer than min_cameras views means no consensus and no marking";
}

TEST(OrthoGhosting, LoneDeviantTileKeepsItsCells) {
    constexpr int kWidth = 16;
    constexpr int kHeight = 16;
    std::mt19937 rng(5U);
    std::vector<OrthoTile> tiles;
    tiles.push_back(make_tile(0, 0, kWidth, kHeight, rng, 0.1F));
    tiles.push_back(make_tile(0, 0, kWidth, kHeight, rng, 1.0F));
    // The high-score tile does not cover the central 8x8: the low-score one is
    // the only valid camera there, even though its mask says it is deviant.
    for (int row = 4; row < 12; ++row) {
        for (int col = 4; col < 12; ++col) {
            tiles[1].valid[tiles[1].index(col, row)] = 0;
        }
    }
    std::vector<std::vector<std::uint8_t>> masks(2);
    masks[0].assign(static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight), 1);
    masks[1].assign(static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight), 0);

    SeamOptions options;
    options.avoid = &masks;
    const std::vector<std::int32_t> labels = assign_seams(kWidth, kHeight, tiles, options);
    EXPECT_EQ(labels[static_cast<std::size_t>(8) * static_cast<std::size_t>(kWidth) + 8], 0)
        << "a deviant camera that is the only valid view must stay labelled (no hole)";
}

TEST(OrthoGhosting, LargeTransientIsCaught) {
    constexpr int kWidth = 128;
    constexpr int kHeight = 128;
    std::mt19937 rng(4242U);
    std::vector<OrthoTile> tiles;
    for (int i = 0; i < 5; ++i) {
        tiles.push_back(make_tile(0, 0, kWidth, kHeight, rng));
    }
    // A 60x60 shadow is a large, low-frequency transient.
    for (int row = 30; row < 90; ++row) {
        for (int col = 30; col < 90; ++col) {
            const std::size_t cell = tiles[3].index(col, row);
            for (int c = 0; c < 3; ++c) {
                const int value = static_cast<int>(tiles[3].rgb[cell * 3 +
                                                              static_cast<std::size_t>(c)]);
                tiles[3].rgb[cell * 3 + static_cast<std::size_t>(c)] =
                    static_cast<std::uint8_t>(std::max(0, value - 70));
            }
        }
    }
    const std::vector<std::vector<std::uint8_t>> masks =
        detect_ghosts(kWidth, kHeight, tiles, GhostOptions{});
    std::size_t inside = 0;
    for (int row = 30; row < 90; ++row) {
        for (int col = 30; col < 90; ++col) {
            inside += masks[3][tiles[3].index(col, row)] != 0 ? 1U : 0U;
        }
    }
    std::printf("[OrthoGhosting.Large] shadow marked %zu of 3600\n", inside);
    EXPECT_GE(inside, 3000U)
        << "the coarse levels must catch a large transient across its interior";
    for (std::size_t t = 0; t < masks.size(); ++t) {
        if (t != 3) {
            EXPECT_LE(marked(masks[t]), 10U);
        }
    }
}

TEST(OrthoGhosting, SpeckleIsRelabelledAndPinnedCellsAreKept) {
    constexpr int kWidth = 20;
    constexpr int kHeight = 20;
    std::mt19937 rng(31U);
    std::vector<OrthoTile> tiles;
    tiles.push_back(make_tile(0, 0, kWidth, kHeight, rng));
    tiles.push_back(make_tile(0, 0, kWidth, kHeight, rng));
    const auto island = [&](std::vector<std::int32_t>& labels) {
        for (int row = 8; row < 11; ++row) {
            for (int col = 8; col < 11; ++col) {
                labels[static_cast<std::size_t>(row) * static_cast<std::size_t>(kWidth) +
                       static_cast<std::size_t>(col)] = 1;
            }
        }
    };
    std::vector<std::int32_t> labels(static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight), 0);
    island(labels);
    EXPECT_EQ(filter_label_speckles(kWidth, kHeight, tiles, labels, nullptr, 16), 1);
    EXPECT_EQ(labels[static_cast<std::size_t>(9) * static_cast<std::size_t>(kWidth) + 9], 0)
        << "a 3x3 island must take the surrounding label";

    island(labels);
    std::vector<std::uint8_t> pinned(static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight), 0);
    pinned[static_cast<std::size_t>(9) * static_cast<std::size_t>(kWidth) + 9] = 1;
    EXPECT_EQ(filter_label_speckles(kWidth, kHeight, tiles, labels, &pinned, 16), 0);
    EXPECT_EQ(labels[static_cast<std::size_t>(9) * static_cast<std::size_t>(kWidth) + 9], 1)
        << "a pinned cell is never changed";

    // Two identical runs give byte-identical masks.
    const std::vector<std::vector<std::uint8_t>> first =
        detect_ghosts(kWidth, kHeight, tiles, GhostOptions{});
    const std::vector<std::vector<std::uint8_t>> second =
        detect_ghosts(kWidth, kHeight, tiles, GhostOptions{});
    ASSERT_EQ(first.size(), second.size());
    for (std::size_t t = 0; t < first.size(); ++t) {
        EXPECT_EQ(first[t], second[t]);
    }
}

TEST(OrthoGhosting, NonSubmodularPairNeverIncreasesEnergy) {
    // p=(0,0) is labelled by the white tile 1 and q=(1,0) by the black tile 2,
    // while the expansion target tile 0 covers q but NOT p. Then
    //   E00 = 2 * 3 seam units (both ends disagree),
    //   E(alpha,lq) + E(lp,alpha) sees only q (p contributes 0),
    // so e01 + e10 - e00 < 0: the old crossing edge was negative and add_edge
    // silently dropped it, breaking the min-cut decomposition. The truncated
    // surrogate plus true-energy acceptance means no sweep may raise energy.
    constexpr int kWidth = 8;
    constexpr int kHeight = 1;
    std::mt19937 rng(2024U);
    std::vector<OrthoTile> tiles;
    for (int i = 0; i < 3; ++i) {
        tiles.push_back(make_tile(0, 0, kWidth, kHeight, rng));
    }
    const auto paint_row = [](OrthoTile& tile, int from, int to, std::uint8_t value) {
        for (int col = from; col < to; ++col) {
            const std::size_t cell = tile.index(col, 0);
            for (int c = 0; c < 3; ++c) {
                tile.rgb[cell * 3 + static_cast<std::size_t>(c)] = value;
            }
        }
    };
    paint_row(tiles[0], 0, kWidth, 128);  // alpha: mid grey, covers cols 1..7
    paint_row(tiles[1], 0, kWidth, 255);  // white, covers cols 0..2
    paint_row(tiles[2], 0, kWidth, 0);    // black, covers cols 0..2
    tiles[0].valid[tiles[0].index(0, 0)] = 0;
    for (int col = 3; col < kWidth; ++col) {
        tiles[1].valid[tiles[1].index(col, 0)] = 0;
        tiles[2].valid[tiles[2].index(col, 0)] = 0;
    }
    for (int col = 0; col < kWidth; ++col) {
        tiles[0].score[tiles[0].index(col, 0)] = 0.2F;
        tiles[1].score[tiles[1].index(col, 0)] = 0.1F;
        tiles[2].score[tiles[2].index(col, 0)] = 0.1F;
    }
    tiles[1].score[tiles[1].index(0, 0)] = 0.9F;
    tiles[2].score[tiles[2].index(0, 0)] = 0.9F;
    tiles[2].score[tiles[2].index(1, 0)] = 0.9F;
    tiles[2].score[tiles[2].index(2, 0)] = 0.9F;
    // One mark far from the pair only opens the expansion window.
    std::vector<std::vector<std::uint8_t>> masks(3);
    for (auto& mask : masks) {
        mask.assign(static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight), 0);
    }
    masks[0][tiles[0].index(7, 0)] = 1;

    SeamOptions base_options;
    base_options.iterations = 0;
    base_options.avoid = &masks;
    const std::vector<std::int32_t> base = assign_seams(kWidth, kHeight, tiles, base_options);
    ASSERT_EQ(base[0], 1) << "the pair must start as (tile 1, tile 2)";
    ASSERT_EQ(base[1], 2);

    SeamOptions options;
    options.coarse_levels = 0;  // isolate the full-resolution expansion
    options.iterations = 1;
    options.avoid = &masks;
    const std::vector<std::int32_t> labels = assign_seams(kWidth, kHeight, tiles, options);
    EXPECT_LE(ortho::energy(kWidth, kHeight, tiles, labels, options),
              ortho::energy(kWidth, kHeight, tiles, base, options) + 1e-9)
        << "an expansion move must never raise the true energy";
}

TEST(OrthoGhosting, ThreadCountDoesNotChangeTheResult) {
#ifdef _OPENMP
    constexpr int kWidth = 64;
    constexpr int kHeight = 48;
    std::mt19937 rng(808U);
    std::vector<OrthoTile> tiles;
    for (int i = 0; i < 5; ++i) {
        tiles.push_back(make_tile(0, 0, kWidth, kHeight, rng));
    }
    paint(tiles[1], 8, 8, 12, 6, 255, 0, 0);
    const int previous = omp_get_max_threads();
    omp_set_num_threads(1);
    const std::vector<std::vector<std::uint8_t>> serial =
        detect_ghosts(kWidth, kHeight, tiles, GhostOptions{});
    omp_set_num_threads(4);
    const std::vector<std::vector<std::uint8_t>> parallel =
        detect_ghosts(kWidth, kHeight, tiles, GhostOptions{});
    omp_set_num_threads(previous);
    ASSERT_EQ(serial.size(), parallel.size());
    for (std::size_t t = 0; t < serial.size(); ++t) {
        EXPECT_EQ(serial[t], parallel[t]) << "tile " << t;
    }
#else
    GTEST_SKIP() << "built without OpenMP";
#endif
}

}  // namespace
}  // namespace alphashape

