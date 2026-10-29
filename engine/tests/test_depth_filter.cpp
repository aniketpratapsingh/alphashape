// Depth filter: cross-view geometric-consistency voting and speckle removal.
//
// The scenes are analytic: every view looks straight down at the world plane
// z = 0 from a known height, so each depth-map pixel's value is the exact
// distance along its unit bearing to the plane (h / b_z) — no solver, no noise.
//
// Suites are prefixed DepthFilter so `ctest -R DepthFilter` selects them.

#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "base/utils/portable_rng.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

#include "base/utils/error.hpp"
#include "common/algorithm/stereo/depth_filter.hpp"
#include "common/algorithm/stereo/depth_repair.hpp"
#include "common/core/chunk.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::stereo {
namespace {

/// A view over the ground plane z = 0. `map` owns the depth storage that the
/// DepthView handed to filter_depth_map() points at, so it must outlive it.
struct View {
    Calibration calib;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    DepthMap map;

    DepthView as_view() const { return DepthView{&map, calib, R, t}; }
};

/// World->camera rotation for an optical axis pointing straight down (-Z world).
Eigen::Matrix3d straight_down_R() {
    Eigen::Matrix3d R;
    R << 0.0, 1.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 0.0, -1.0;
    return R;  // det = +1
}

std::size_t pix_index(const Calibration& c, int i, int j) {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(c.width) +
           static_cast<std::size_t>(i);
}

/// Fill `map` with the exact ray-plane distance to z = 0 for every pixel centre.
/// Pixels whose ray does not meet the plane keep the -1 sentinel.
void fill_plane(DepthMap& map, const Calibration& c, const Eigen::Matrix3d& R,
                const Eigen::Vector3d& t) {
    map.width = c.width;
    map.height = c.height;
    map.downscale = 1;
    map.depths.assign(static_cast<std::size_t>(c.width) * static_cast<std::size_t>(c.height),
                      -1.0f);
    const Eigen::Vector3d n(0.0, 0.0, 1.0);
    const Eigen::Matrix3d Rt = R.transpose();
    const double num = n.dot(Rt * t);  // (c + n . R^T t), with plane offset c = 0
    for (int j = 0; j < c.height; ++j) {
        for (int i = 0; i < c.width; ++i) {
            const Eigen::Vector2d px(static_cast<double>(i) + 0.5,
                                     static_cast<double>(j) + 0.5);
            const Eigen::Vector3d b = c.unproject(px);
            const double den = n.dot(Rt * b);
            if (std::abs(den) <= 1e-9) {
                continue;
            }
            const double d = num / den;
            if (d > 0.0) {
                map.depths[pix_index(c, i, j)] = static_cast<float>(d);
            }
        }
    }
}

View make_view(int w, int h, double f, double cam_x, double cam_y, double height) {
    View v;
    v.calib = Calibration::initial(CameraType::Frame, w, h, f);
    v.R = straight_down_R();
    const Eigen::Vector3d C(cam_x, cam_y, height);
    v.t = -(v.R * C);
    fill_plane(v.map, v.calib, v.R, v.t);
    return v;
}

std::size_t count_valid(const DepthMap& m) {
    std::size_t n = 0;
    for (const float d : m.depths) {
        if (d > 0.0f) {
            ++n;
        }
    }
    return n;
}

/// World point a reference pixel asserts, in chunk-local coordinates.
Eigen::Vector3d world_point(const DepthView& view, int i, int j) {
    const Eigen::Vector2d px(static_cast<double>(i) + 0.5, static_cast<double>(j) + 0.5);
    const Eigen::Vector3d bearing = view.calibration.unproject(px);
    const double depth = static_cast<double>(view.map->depths[pix_index(view.calibration, i, j)]);
    const Eigen::Vector3d x_cam = bearing * depth;
    return view.R.transpose() * (x_cam - view.t);
}

/// True when neighbour `nb` has a depth at the pixel the reference point lands on.
bool neighbour_sees(const DepthView& nb, const Eigen::Vector3d& x_world) {
    const Eigen::Vector3d x_n = nb.R * x_world + nb.t;
    if (x_n.z() <= 1e-9) {
        return false;
    }
    const Eigen::Vector2d uv = nb.calibration.project(x_n);
    const int ci = static_cast<int>(std::floor(uv.x()));
    const int cj = static_cast<int>(std::floor(uv.y()));
    if (ci < 0 || cj < 0 || ci >= nb.map->width || cj >= nb.map->height) {
        return false;
    }
    return nb.map->depths[pix_index(nb.calibration, ci, cj)] > 0.0f;
}

constexpr int kW = 80;
constexpr int kH = 60;
constexpr double kF = 80.0;
constexpr double kHeight = 1.4;

std::vector<View> scene_3_neighbours() {
    std::vector<View> views;
    views.push_back(make_view(kW, kH, kF, 0.0, 0.0, kHeight));       // [0] reference
    views.push_back(make_view(kW, kH, kF, -0.12, 0.0, kHeight));     // [1]
    views.push_back(make_view(kW, kH, kF, 0.12, 0.0, kHeight));      // [2]
    views.push_back(make_view(kW, kH, kF, 0.0, 0.12, kHeight));      // [3]
    return views;
}

// -- parse_filter_mode / to_string ------------------------------------------

TEST(DepthFilter, ParseModeRoundTrip) {
    const FilterMode modes[] = {FilterMode::Disabled, FilterMode::Mild, FilterMode::Moderate,
                                FilterMode::Aggressive};
    const char* names[] = {"disabled", "mild", "moderate", "aggressive"};
    for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(parse_filter_mode(names[k]), modes[k]);
        EXPECT_EQ(to_string(modes[k]), std::string(names[k]));
    }
}

TEST(DepthFilter, DefaultOptionsMatchTheTaskDefault) {
    // BuildDepthMaps declares filter_mode="moderate" (the benchmark's winner:
    // 21% less hole area than "mild" at no time cost), so a default-constructed
    // FilterOptions must mean the same thing. Set them in one place or this
    // trips.
    EXPECT_EQ(FilterOptions{}.mode, FilterMode::Moderate);
    EXPECT_EQ(to_string(FilterOptions{}.mode), "moderate");
}

TEST(DepthFilter, ParseModeCaseInsensitive) {
    EXPECT_EQ(parse_filter_mode("MILD"), FilterMode::Mild);
    EXPECT_EQ(parse_filter_mode("Aggressive"), FilterMode::Aggressive);
}

TEST(DepthFilter, ParseModeUnknownThrows) {
    try {
        (void)parse_filter_mode("extreme");
        FAIL() << "expected ValueError";
    } catch (const ValueError& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("disabled"), std::string::npos);
        EXPECT_NE(msg.find("mild"), std::string::npos);
        EXPECT_NE(msg.find("moderate"), std::string::npos);
        EXPECT_NE(msg.find("aggressive"), std::string::npos);
    }
}

// -- input validation -------------------------------------------------------

TEST(DepthFilter, NullReferenceThrows) {
    const std::vector<View> views = scene_3_neighbours();
    DepthView bad;
    bad.calibration = views[0].calib;
    EXPECT_THROW(filter_depth_map(bad, {}, FilterOptions{}), ValueError);
}

TEST(DepthFilter, SizeMismatchThrows) {
    std::vector<View> views = scene_3_neighbours();
    views[0].map.depths.pop_back();  // depths no longer match width * height
    DepthView ref = views[0].as_view();
    EXPECT_THROW(filter_depth_map(ref, {}, FilterOptions{}), ValueError);
}

TEST(DepthFilter, CalibrationSizeMismatchThrows) {
    std::vector<View> views = scene_3_neighbours();
    views[0].calib.width += 1;  // calibration disagrees with the map geometry
    DepthView ref = views[0].as_view();
    EXPECT_THROW(filter_depth_map(ref, {}, FilterOptions{}), ValueError);
}

// -- geometric consistency --------------------------------------------------

TEST(DepthFilter, ConsistentMapsKeepAndCountSupport) {
    std::vector<View> views = scene_3_neighbours();
    const DepthView ref = views[0].as_view();
    std::vector<DepthView> neighbours{views[1].as_view(), views[2].as_view(), views[3].as_view()};

    FilterStats stats;
    const DepthMap out =
        filter_depth_map(ref, neighbours, FilterOptions{FilterMode::Mild}, &stats);

    // Reference oracle: how many neighbours actually have depth where each point lands.
    std::size_t visible_pixels = 0;
    std::size_t kept_visible = 0;
    std::size_t confidence_ok = 0;
    for (int j = 0; j < kH; ++j) {
        for (int i = 0; i < kW; ++i) {
            if (views[0].map.depths[pix_index(views[0].calib, i, j)] <= 0.0f) {
                continue;
            }
            const Eigen::Vector3d x_w = world_point(ref, i, j);
            int seen = 0;
            for (const DepthView& nb : neighbours) {
                if (neighbour_sees(nb, x_w)) {
                    ++seen;
                }
            }
            if (seen < 1) {
                continue;
            }
            ++visible_pixels;
            const std::size_t idx = pix_index(views[0].calib, i, j);
            if (out.depths[idx] > 0.0f) {
                ++kept_visible;
                if (out.confidence[idx] == static_cast<float>(seen)) {
                    ++confidence_ok;
                }
            }
        }
    }

    ASSERT_GT(visible_pixels, 0u);
    const double kept_fraction =
        static_cast<double>(kept_visible) / static_cast<double>(visible_pixels);
    EXPECT_GE(kept_fraction, 0.95);
    // Every kept pixel's confidence equals the number of neighbours that see it.
    EXPECT_EQ(confidence_ok, kept_visible);
    EXPECT_EQ(stats.input, count_valid(views[0].map));
}

TEST(DepthFilter, OutliersRejectedCleanPixelsKept) {
    std::vector<View> views = scene_3_neighbours();

    // Inject a 30% depth error into ~5% of the reference's valid pixels.
    std::mt19937 rng(1234567u);
    // Portable draw: std::uniform_real_distribution is implementation-defined
    // across libc++ and libstdc++ (TESTING.md, "Numeric consistency").
    const auto uni = [](std::mt19937& g) { return alphashape::rng::uniform_real(g, 0.0, 1.0); };
    const std::size_t pixels = static_cast<std::size_t>(kW) * static_cast<std::size_t>(kH);
    std::vector<std::uint8_t> injected(pixels, 0);
    std::size_t n_injected = 0;
    for (std::size_t idx = 0; idx < pixels; ++idx) {
        if (views[0].map.depths[idx] > 0.0f && uni(rng) < 0.05) {
            views[0].map.depths[idx] *= 1.3f;
            injected[idx] = 1;
            ++n_injected;
        }
    }
    ASSERT_GT(n_injected, 0u);

    const DepthView ref = views[0].as_view();
    std::vector<DepthView> neighbours{views[1].as_view(), views[2].as_view(), views[3].as_view()};
    const DepthMap out = filter_depth_map(ref, neighbours, FilterOptions{FilterMode::Mild});

    std::size_t injected_kept = 0;
    std::size_t clean_visible = 0;
    std::size_t clean_lost = 0;
    for (int j = 0; j < kH; ++j) {
        for (int i = 0; i < kW; ++i) {
            const std::size_t idx = pix_index(views[0].calib, i, j);
            if (injected[idx] != 0 && out.depths[idx] > 0.0f) {
                ++injected_kept;
            }
        }
    }
    EXPECT_EQ(injected_kept, 0u) << "all injected outliers must be rejected";

    // Recompute clean visibility using the oracle from a fresh consistent scene.
    std::vector<View> clean = scene_3_neighbours();
    const DepthView clean_ref = clean[0].as_view();
    std::vector<DepthView> clean_nb{clean[1].as_view(), clean[2].as_view(), clean[3].as_view()};
    for (int j = 0; j < kH; ++j) {
        for (int i = 0; i < kW; ++i) {
            const std::size_t idx = pix_index(clean[0].calib, i, j);
            if (injected[idx] != 0) {
                continue;  // not a clean pixel
            }
            if (clean[0].map.depths[idx] <= 0.0f) {
                continue;
            }
            const Eigen::Vector3d x_w = world_point(clean_ref, i, j);
            bool seen = false;
            for (const DepthView& nb : clean_nb) {
                if (neighbour_sees(nb, x_w)) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                continue;
            }
            ++clean_visible;
            if (out.depths[idx] <= 0.0f) {
                ++clean_lost;
            }
        }
    }
    ASSERT_GT(clean_visible, 0u);
    const double lost_fraction =
        static_cast<double>(clean_lost) / static_cast<double>(clean_visible);
    EXPECT_LT(lost_fraction, 0.02);
}

TEST(DepthFilter, ModeOrderingOnDisagreeingNeighbours) {
    std::vector<View> views = scene_3_neighbours();
    // Corrupt one neighbour: its depths are 50% too large, so it never supports.
    for (float& d : views[3].map.depths) {
        if (d > 0.0f) {
            d *= 1.5f;
        }
    }
    const DepthView ref = views[0].as_view();
    std::vector<DepthView> neighbours{views[1].as_view(), views[2].as_view(), views[3].as_view()};

    std::size_t kept[4] = {0, 0, 0, 0};
    const FilterMode modes[4] = {FilterMode::Disabled, FilterMode::Mild, FilterMode::Moderate,
                                 FilterMode::Aggressive};
    for (int k = 0; k < 4; ++k) {
        const DepthMap out = filter_depth_map(ref, neighbours, FilterOptions{modes[k]});
        kept[k] = count_valid(out);
    }
    EXPECT_GE(kept[0], kept[1]);
    EXPECT_GE(kept[1], kept[2]);
    EXPECT_GE(kept[2], kept[3]);
    EXPECT_GT(kept[0], kept[2]) << "moderate must reject more than disabled";
}

TEST(DepthFilter, SpeckleRemovesSmallIslandKeepsLargeRegion) {
    // A larger map so the speckle threshold (pixels / divisor) exceeds the 2x2 island.
    const int W = 320;
    const int H = 240;
    std::vector<View> views;
    views.push_back(make_view(W, H, 320.0, 0.0, 0.0, kHeight));
    views.push_back(make_view(W, H, 320.0, -0.12, 0.0, kHeight));
    views.push_back(make_view(W, H, 320.0, 0.12, 0.0, kHeight));
    views.push_back(make_view(W, H, 320.0, 0.0, 0.12, kHeight));

    // Keep only a large block and an isolated 2x2 island in the reference; the
    // neighbours keep the full plane so both pass the support test.
    const auto keep = [](int i, int j) {
        const bool big = (i >= 200 && i < 230 && j >= 150 && j < 180);   // 30x30 = 900 px
        const bool small = (i >= 40 && i < 42 && j >= 40 && j < 42);      // 2x2 = 4 px
        return big || small;
    };
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            if (!keep(i, j)) {
                views[0].map.depths[pix_index(views[0].calib, i, j)] = -1.0f;
            }
        }
    }

    const DepthView ref = views[0].as_view();
    std::vector<DepthView> neighbours{views[1].as_view(), views[2].as_view(), views[3].as_view()};
    FilterStats stats;
    const DepthMap out =
        filter_depth_map(ref, neighbours, FilterOptions{FilterMode::Mild}, &stats);

    // The 2x2 island is gone.
    EXPECT_FLOAT_EQ(out.depths[pix_index(views[0].calib, 40, 40)], -1.0f);
    EXPECT_FLOAT_EQ(out.depths[pix_index(views[0].calib, 41, 41)], -1.0f);
    EXPECT_FLOAT_EQ(out.confidence[pix_index(views[0].calib, 40, 40)], 0.0f);
    // The centre of the large block survives.
    EXPECT_GT(out.depths[pix_index(views[0].calib, 215, 165)], 0.0f);
    EXPECT_GE(stats.rejected_speckle, 4u);
}

TEST(DepthFilter, DisabledKeepsDepthsAndFillsConfidence) {
    std::vector<View> views = scene_3_neighbours();
    const std::size_t pixels = static_cast<std::size_t>(kW) * static_cast<std::size_t>(kH);
    // Give every view normals so the pass-through can be checked too.
    for (View& v : views) {
        v.map.normals.assign(3 * pixels, 0.0f);
        for (int j = 0; j < kH; ++j) {
            for (int i = 0; i < kW; ++i) {
                const std::size_t idx = pix_index(v.calib, i, j);
                const Eigen::Vector3d b =
                    v.calib.unproject({static_cast<double>(i) + 0.5, static_cast<double>(j) + 0.5});
                for (int k = 0; k < 3; ++k) {
                    v.map.normals[3 * idx + static_cast<std::size_t>(k)] =
                        static_cast<float>(b[k]);
                }
            }
        }
    }
    const DepthView ref = views[0].as_view();
    std::vector<DepthView> neighbours{views[1].as_view(), views[2].as_view(), views[3].as_view()};

    const DepthMap out = filter_depth_map(ref, neighbours, FilterOptions{FilterMode::Disabled});

    EXPECT_EQ(out.depths, views[0].map.depths);
    EXPECT_EQ(out.normals, views[0].map.normals);
    ASSERT_EQ(out.confidence.size(), pixels);
    // Confidence is still populated: at least some pixels see multiple neighbours.
    float max_conf = 0.0f;
    std::size_t non_zero = 0;
    for (const float c : out.confidence) {
        max_conf = std::max(max_conf, c);
        if (c > 0.0f) {
            ++non_zero;
        }
    }
    EXPECT_GT(non_zero, 0u);
    EXPECT_GT(max_conf, 0.0f);
}

TEST(DepthFilter, RejectedPixelZeroesNormalsAndConfidence) {
    std::vector<View> views = scene_3_neighbours();
    const std::size_t pixels = static_cast<std::size_t>(kW) * static_cast<std::size_t>(kH);
    for (View& v : views) {
        v.map.normals.assign(3 * pixels, 1.0f);
    }
    // One pixel with a wildly wrong depth (inconsistent with every neighbour).
    const std::size_t bad = pix_index(views[0].calib, 40, 30);
    views[0].map.depths[bad] *= 1.5f;

    const DepthView ref = views[0].as_view();
    std::vector<DepthView> neighbours{views[1].as_view(), views[2].as_view(), views[3].as_view()};
    const DepthMap out = filter_depth_map(ref, neighbours, FilterOptions{FilterMode::Moderate});

    ASSERT_LE(out.depths[bad], 0.0f);
    EXPECT_FLOAT_EQ(out.confidence[bad], 0.0f);
    EXPECT_FLOAT_EQ(out.normals[3 * bad + 0], 0.0f);
    EXPECT_FLOAT_EQ(out.normals[3 * bad + 1], 0.0f);
    EXPECT_FLOAT_EQ(out.normals[3 * bad + 2], 0.0f);
}

TEST(DepthFilter, ThreadCountDeterminism) {
    const std::vector<View> views = scene_3_neighbours();
    const DepthView ref = views[0].as_view();
    const std::vector<DepthView> neighbours{views[1].as_view(), views[2].as_view(),
                                            views[3].as_view()};
    const FilterOptions opts{FilterMode::Moderate};

    DepthMap single;
    DepthMap multi;
#if defined(ALPHASHAPE_HAVE_OPENMP)
    omp_set_num_threads(1);
    single = filter_depth_map(ref, neighbours, opts);
    omp_set_num_threads(4);
    multi = filter_depth_map(ref, neighbours, opts);
    omp_set_num_threads(omp_get_max_threads() > 0 ? omp_get_max_threads() : 1);
#else
    single = filter_depth_map(ref, neighbours, opts);
    multi = single;
#endif

    EXPECT_EQ(single.depths, multi.depths);
    EXPECT_EQ(single.confidence, multi.confidence);
    EXPECT_EQ(single.normals, multi.normals);
}

// ---------------------------------------------------------------------------
// Post-filter hole repair (depth_repair.hpp)
// ---------------------------------------------------------------------------

void punch(DepthMap& map, int i0, int j0, int i1, int j1) {
    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            map.depths[static_cast<std::size_t>(j) * static_cast<std::size_t>(map.width) +
                       static_cast<std::size_t>(i)] = -1.0f;
        }
    }
}

double index_depth(const DepthMap& map, int i, int j) {
    return static_cast<double>(map.depths[static_cast<std::size_t>(j) *
                                          static_cast<std::size_t>(map.width) +
                                          static_cast<std::size_t>(i)]);
}

TEST(DepthRepair, PlaneHoleIsFilled) {
    View view = make_view(kW, kH, kF, 0.0, 0.0, kHeight);
    const DepthMap before = view.map;
    punch(view.map, 30, 20, 40, 30);  // 11x11 square (area 121 <= default 2000)
    ASSERT_LE(index_depth(view.map, 35, 25), 0.0);

    RepairStats stats;
    const DepthMap out = repair_depth_map(view.map, view.calib, RepairOptions{}, &stats);
    EXPECT_EQ(stats.filled_components, 1u);
    EXPECT_EQ(stats.filled_pixels, 121u);
    for (int j = 20; j <= 30; ++j) {
        for (int i = 30; i <= 40; ++i) {
            const double got = index_depth(out, i, j);
            const double want = index_depth(before, i, j);
            ASSERT_GT(got, 0.0) << "pixel " << i << "," << j;
            EXPECT_LE(std::abs(got - want) / want, 1e-6)
                << "pixel " << i << "," << j << " got " << got << " want " << want;
        }
    }
}

TEST(DepthRepair, StepEdgeHoleIsNotFilled) {
    // Two PARALLEL planes at different depths (a step): scaling the ray depth
    // moves the surface to another plane parallel to the first. A hole
    // straddling the step has its ring split between them, so no single plane
    // explains it and the fill must be refused rather than bridge the gap.
    View view = make_view(kW, kH, kF, 0.0, 0.0, kHeight);
    const int edge_i = 35;
    for (int j = 0; j < kH; ++j) {
        for (int i = 0; i < kW; ++i) {
            const double d = index_depth(view.map, i, j);
            if (d > 0.0) {
                const double scale = i < edge_i ? 1.0 : 2.0;  // the far half
                view.map.depths[static_cast<std::size_t>(j) * kW + i] =
                    static_cast<float>(d * scale);
            }
        }
    }
    // A hole centred on the step, 21x15.
    punch(view.map, 25, 22, 45, 36);
    const std::size_t before_valid = count_valid(view.map);

    RepairStats stats;
    const DepthMap out = repair_depth_map(view.map, view.calib, RepairOptions{}, &stats);
    EXPECT_EQ(stats.filled_components, 0u) << "a step edge must not be bridged";
    EXPECT_EQ(stats.filled_pixels, 0u);
    EXPECT_EQ(count_valid(out), before_valid);
}

TEST(DepthRepair, OversizedHoleIsNotFilled) {
    View view = make_view(kW, kH, kF, 0.0, 0.0, kHeight);
    punch(view.map, 20, 15, 60, 45);  // 41*31 = 1271 px
    RepairOptions options;
    options.max_area = 500;
    RepairStats stats;
    const DepthMap out = repair_depth_map(view.map, view.calib, options, &stats);
    EXPECT_EQ(stats.filled_pixels, 0u);
    EXPECT_EQ(stats.rejected_area, 1u);
    EXPECT_EQ(index_depth(out, 40, 30), -1.0);
}

TEST(DepthRepair, BorderTouchingHoleIsNotFilled) {
    // A hole cut off by the map edge has too little of its perimeter on measured
    // pixels; a plane fitted to the rest would be extrapolated over ground
    // nothing measured. This one is 7 wide x 11 tall against the left border:
    // 11 of its 36 perimeter edges are image border, so only 25/36 = 69% is
    // measured, under the 75% threshold.
    View view = make_view(kW, kH, kF, 0.0, 0.0, kHeight);
    punch(view.map, 0, 10, 6, 20);
    RepairStats stats;
    const DepthMap out = repair_depth_map(view.map, view.calib, RepairOptions{}, &stats);
    EXPECT_EQ(stats.filled_pixels, 0u);
    EXPECT_EQ(index_depth(out, 3, 15), -1.0);

    // A WIDE, SHALLOW notch leaves most of the ring measured (21 wide x 5 tall:
    // 5 of 52 edges on the border, 90% measured), so it is a legitimate repair:
    // the surface is enclosed on three sides and the plane explains the rest.
    View wide = make_view(kW, kH, kF, 0.0, 0.0, kHeight);
    punch(wide.map, 0, 13, 20, 17);
    RepairStats wide_stats;
    const DepthMap wide_out = repair_depth_map(wide.map, wide.calib, RepairOptions{}, &wide_stats);
    EXPECT_EQ(wide_stats.filled_pixels, 105u) << "a 21x5 notch should be filled";
    EXPECT_GT(index_depth(wide_out, 10, 15), 0.0);
}

TEST(DepthRepair, IsDeterministicAndMarksConfidence) {
    View view = make_view(kW, kH, kF, 0.0, 0.0, kHeight);
    const std::size_t pixels = static_cast<std::size_t>(kW) * static_cast<std::size_t>(kH);
    view.map.normals.assign(3 * pixels, 0.0f);
    view.map.confidence.assign(pixels, 3.0f);
    punch(view.map, 30, 20, 40, 30);

    const DepthMap a = repair_depth_map(view.map, view.calib, RepairOptions{});
    const DepthMap b = repair_depth_map(view.map, view.calib, RepairOptions{});
    EXPECT_EQ(a.depths, b.depths);
    EXPECT_EQ(a.normals, b.normals);
    EXPECT_EQ(a.confidence, b.confidence);

    // A repaired pixel is marked with the reduced confidence and a plane
    // normal; a measured pixel keeps its own values.
    const std::size_t filled = static_cast<std::size_t>(25) * kW + 35;
    EXPECT_FLOAT_EQ(a.confidence[filled], kRepairedConfidence);
    EXPECT_GT(a.depths[filled], 0.0f);
    const std::size_t measured = static_cast<std::size_t>(5) * kW + 5;
    EXPECT_FLOAT_EQ(a.confidence[measured], 3.0f);
    const double norm = std::sqrt(static_cast<double>(a.normals[3 * filled + 0]) * a.normals[3 * filled + 0] +
                                  static_cast<double>(a.normals[3 * filled + 1]) * a.normals[3 * filled + 1] +
                                  static_cast<double>(a.normals[3 * filled + 2]) * a.normals[3 * filled + 2]);
    EXPECT_NEAR(norm, 1.0, 1e-6);
}

TEST(DepthRepair, ZeroMaxAreaFillsNothing) {
    // The degenerate configuration of an enabled pass: never modifies the input
    // and never fills. (The TASK simply does not call the pass when repair_holes
    // is false; that path is covered by the fingerprint test in the dense suite.)
    View view = make_view(kW, kH, kF, 0.0, 0.0, kHeight);
    punch(view.map, 30, 20, 40, 30);
    const std::vector<float> original = view.map.depths;
    RepairOptions options;
    options.max_area = 0;
    RepairStats stats;
    const DepthMap out = repair_depth_map(view.map, view.calib, options, &stats);
    EXPECT_EQ(out.depths, original);
    EXPECT_EQ(stats.filled_pixels, 0u);
    EXPECT_EQ(view.map.depths, original) << "the input map must not be modified in place";
}

}  // namespace
}  // namespace alphashape::stereo
