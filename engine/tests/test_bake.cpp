// test_bake.cpp -- texture baking (milestone 8 texturing).
//
// The scene is analytic: a subdivided box painted with a smooth procedural
// colour field, and six synthetic pinhole cameras on the cube-corner directions
// so every one of the six sides is seen by three cameras. The "photographs" are
// rendered by ray-casting the field, so the baked page can be compared against
// the field's true value at each texel's surface point.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/texture/bake.hpp"
#include "common/algorithm/texture/face_views.hpp"
#include "image/image.hpp"
#include "model/camera/calibration.hpp"
#include "model/mesh/atlas_pack.hpp"
#include "model/mesh/mesh.hpp"
#include "model/mesh/uv_charts.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape {
namespace texturing {
namespace {

constexpr int kViewW = 200;
constexpr int kViewH = 150;
constexpr double kFocal = 180.0;
constexpr double kCamDist = 4.0;
constexpr int kBoxSubdiv = 10;

/// Smooth analytic colour field C(X), per channel in about [40, 216].
double field(const Eigen::Vector3d& X, int ch) {
    const double c = static_cast<double>(ch);
    double s = std::sin(1.7 * X.x() + 0.9 * c) * std::cos(1.3 * X.y() - 0.4 * c);
    s += 0.7 * std::sin(1.9 * X.z() + 1.3 + 0.6 * c);
    s += 0.5 * std::cos(1.1 * (X.x() + X.y() + X.z()) + 2.1 * c);
    return 127.5 + 40.0 * s;
}

std::uint8_t field_u8(const Eigen::Vector3d& X, int ch) {
    const double v = std::clamp(field(X, ch), 0.0, 255.0);
    return static_cast<std::uint8_t>(std::lround(v));
}

/// A box centred at `c`, half-extent `h`, each side split into an `n` x `n` grid
/// of quads. Vertices on the box's edges and corners are SHARED between the two
/// (or three) sides that meet there, so the colour-correction pass sees the box
/// edges as genuine shared mesh edges. Outward-CCW winding.
Mesh make_box(const Eigen::Vector3d& c, double h, int n) {
    Mesh m;
    std::map<std::array<std::int64_t, 3>, int> vid;
    const auto key = [](const Eigen::Vector3d& p) {
        const auto q = [](double v) { return static_cast<std::int64_t>(std::llround(v * 1e6)); };
        return std::array<std::int64_t, 3>{q(p.x()), q(p.y()), q(p.z())};
    };
    const auto vertex = [&](const Eigen::Vector3d& p) {
        const auto k = key(p);
        const auto it = vid.find(k);
        if (it != vid.end()) {
            return it->second;
        }
        const int idx = static_cast<int>(m.vertices.size());
        m.vertices.push_back(p);
        vid.emplace(k, idx);
        return idx;
    };
    const struct Side {
        Eigen::Vector3d nrm;
        Eigen::Vector3d u;
        Eigen::Vector3d v;
    } sides[6] = {
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},   {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
        {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}},   {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},   {{0, 0, -1}, {0, 1, 0}, {1, 0, 0}},
    };
    for (const auto& s : sides) {
        const Eigen::Vector3d base = c + s.nrm * h - s.u * h - s.v * h;
        const Eigen::Vector3d du = s.u * (2.0 * h / n);
        const Eigen::Vector3d dv = s.v * (2.0 * h / n);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const Eigen::Vector3d p00 = base + du * i + dv * j;
                const int a = vertex(p00);
                const int b = vertex(p00 + du);
                const int cc = vertex(p00 + du + dv);
                const int d = vertex(p00 + dv);
                m.faces.push_back(Eigen::Vector3i(a, b, cc));
                m.faces.push_back(Eigen::Vector3i(a, cc, d));
            }
        }
    }
    return m;
}

/// Look-at camera as in the face-view tests: +X right, +Y down, +Z forward.
TexView look_at(const Eigen::Vector3d& eye, const Eigen::Vector3d& target, int width, int height,
                double f) {
    const Eigen::Vector3d fwd = (target - eye).normalized();
    Eigen::Vector3d up_hint(0.0, 1.0, 0.0);
    if (std::abs(fwd.y()) > 0.9) {
        up_hint = Eigen::Vector3d(0.0, 0.0, 1.0);
    }
    const Eigen::Vector3d right = up_hint.cross(fwd).normalized();
    const Eigen::Vector3d down = fwd.cross(right).normalized();
    Eigen::Matrix3d R;
    R.row(0) = right.transpose();
    R.row(1) = down.transpose();
    R.row(2) = fwd.transpose();
    TexView v;
    v.camera_id = 0;
    v.calibration = Calibration::initial(CameraType::Frame, width, height, f);
    v.R = R;
    v.t = -R * eye;
    return v;
}

/// Moller-Trumbore nearest hit against the whole mesh; writes the world point.
bool ray_hit(const Mesh& m, const Eigen::Vector3d& O, const Eigen::Vector3d& D, Eigen::Vector3d& X) {
    double best = std::numeric_limits<double>::infinity();
    bool found = false;
    for (const Eigen::Vector3i& tri : m.faces) {
        const Eigen::Vector3d& a = m.vertices[static_cast<std::size_t>(tri[0])];
        const Eigen::Vector3d& b = m.vertices[static_cast<std::size_t>(tri[1])];
        const Eigen::Vector3d& cc = m.vertices[static_cast<std::size_t>(tri[2])];
        const Eigen::Vector3d e1 = b - a;
        const Eigen::Vector3d e2 = cc - a;
        const Eigen::Vector3d p = D.cross(e2);
        const double det = e1.dot(p);
        if (std::abs(det) < 1e-12) {
            continue;
        }
        const double inv = 1.0 / det;
        const Eigen::Vector3d t = O - a;
        const double u = t.dot(p) * inv;
        if (u < 0.0 || u > 1.0) {
            continue;
        }
        const Eigen::Vector3d q = t.cross(e1);
        const double vv = D.dot(q) * inv;
        if (vv < 0.0 || u + vv > 1.0) {
            continue;
        }
        const double tt = e2.dot(q) * inv;
        if (tt > 1e-9 && tt < best) {
            best = tt;
            X = a + e1 * u + e2 * vv;
            found = true;
        }
    }
    return found;
}

}  // namespace

/// The shared analytic scene: a subdivided box, six rendered views (three per
/// side, on the cube-corner directions), a chart/atlas layout and the resulting
/// face-view assignment.
struct Scene {
    Mesh mesh;
    std::vector<Image> images;  // must outlive `views`
    std::vector<TexView> views;
    FaceViews fv;
    int texture_size = 512;
    int padding = 4;
    double texels_per_unit = 0.0;
};

Scene build_scene(int texture_size = 512, int padding = 4) {
    Scene s;
    s.texture_size = texture_size;
    s.padding = padding;
    s.mesh = make_box(Eigen::Vector3d(0, 0, 0), 1.0, kBoxSubdiv);

    const Eigen::Vector3d dirs[6] = {
        {1, 1, 1}, {1, -1, 1}, {1, 1, -1}, {1, -1, -1}, {-1, 1, 1}, {-1, -1, -1},
    };
    s.images.reserve(6);
    for (std::size_t i = 0; i < 6; ++i) {
        s.images.emplace_back(kViewW, kViewH, 3);
    }
    for (const Eigen::Vector3d& d : dirs) {
        s.views.push_back(look_at(d.normalized() * kCamDist, Eigen::Vector3d::Zero(), kViewW, kViewH,
                                  kFocal));
    }
    for (std::size_t i = 0; i < s.views.size(); ++i) {
        s.views[i].image = &s.images[i];
    }

    // Render each view by ray-casting the analytic field.
    for (std::size_t vi = 0; vi < s.views.size(); ++vi) {
        const TexView& v = s.views[vi];
        Image& img = s.images[vi];
        const Eigen::Vector3d C = -v.R.transpose() * v.t;
        const Eigen::Matrix3d Rt = v.R.transpose();
        for (int y = 0; y < kViewH; ++y) {
            for (int x = 0; x < kViewW; ++x) {
                const double px = (static_cast<double>(x) + 0.5 - kViewW * 0.5) / kFocal;
                const double py = (static_cast<double>(y) + 0.5 - kViewH * 0.5) / kFocal;
                const Eigen::Vector3d dir_cam(px, py, 1.0);
                const Eigen::Vector3d dir_world = (Rt * dir_cam).normalized();
                Eigen::Vector3d X;
                if (ray_hit(s.mesh, C, dir_world, X)) {
                    img.at(x, y, 0) = field_u8(X, 0);
                    img.at(x, y, 1) = field_u8(X, 1);
                    img.at(x, y, 2) = field_u8(X, 2);
                } else {
                    img.at(x, y, 0) = 0;
                    img.at(x, y, 1) = 0;
                    img.at(x, y, 2) = 0;
                }
            }
        }
    }

    const std::vector<Chart> charts = build_charts(s.mesh, ChartOptions{});
    PackOptions pack;
    pack.texture_size = texture_size;
    pack.padding_px = padding;
    const PackResult pr = pack_atlas(s.mesh, charts, pack);
    s.texels_per_unit = pr.texels_per_unit;

    FaceViewOptions fo;
    fo.zbuffer_downscale = 1;
    fo.min_view_cos = 0.15;
    fo.smoothing_passes = 0;
    s.fv = select_face_views(s.mesh, s.views, fo);
    return s;
}

/// Surface point of the covering face at texel (i,j), reconstructed with the
/// SAME formula the baker uses. Returns false when the texel lies in no chart.
bool texel_point(const Mesh& mesh, int size, int i, int j, Eigen::Vector3d& out) {
    const double s = static_cast<double>(size);
    const double px = (static_cast<double>(i) + 0.5) / s;
    const double py = (static_cast<double>(j) + 0.5) / s;
    int best = -1;
    double bb[3] = {0.0, 0.0, 0.0};
    for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
        const Eigen::Vector3i& tri = mesh.uv_faces[f];
        const Eigen::Vector2d u0(mesh.uvs[static_cast<std::size_t>(tri[0])].x(),
                                 mesh.uvs[static_cast<std::size_t>(tri[0])].y());
        const Eigen::Vector2d u1(mesh.uvs[static_cast<std::size_t>(tri[1])].x(),
                                 mesh.uvs[static_cast<std::size_t>(tri[1])].y());
        const Eigen::Vector2d u2(mesh.uvs[static_cast<std::size_t>(tri[2])].x(),
                                 mesh.uvs[static_cast<std::size_t>(tri[2])].y());
        const double area =
            (u1.x() - u0.x()) * (u2.y() - u0.y()) - (u2.x() - u0.x()) * (u1.y() - u0.y());
        if (std::abs(area) < 1e-20) {
            continue;
        }
        const double w0 = ((u1.x() - px) * (u2.y() - py) - (u2.x() - px) * (u1.y() - py)) / area;
        const double w1 = ((u2.x() - px) * (u0.y() - py) - (u0.x() - px) * (u2.y() - py)) / area;
        const double w2 = 1.0 - w0 - w1;
        if (w0 < -1e-6 || w1 < -1e-6 || w2 < -1e-6) {
            continue;
        }
        best = static_cast<int>(f);
        bb[0] = w0;
        bb[1] = w1;
        bb[2] = w2;
        break;
    }
    if (best < 0) {
        return false;
    }
    const Eigen::Vector3i& tri = mesh.faces[static_cast<std::size_t>(best)];
    out = bb[0] * mesh.vertices[static_cast<std::size_t>(tri[0])] +
          bb[1] * mesh.vertices[static_cast<std::size_t>(tri[1])] +
          bb[2] * mesh.vertices[static_cast<std::size_t>(tri[2])];
    return true;
}

// ---------------------------------------------------------------------------
// parse / to_string round trip and rejection.
// ---------------------------------------------------------------------------
TEST(Bake, ParseAndToString) {
    EXPECT_EQ(parse_blending_mode("mosaic"), BlendingMode::Mosaic);
    EXPECT_EQ(parse_blending_mode("average"), BlendingMode::Average);
    EXPECT_EQ(parse_blending_mode("disabled"), BlendingMode::Disabled);
    EXPECT_EQ(to_string(BlendingMode::Mosaic), "mosaic");
    EXPECT_EQ(to_string(BlendingMode::Average), "average");
    EXPECT_EQ(to_string(BlendingMode::Disabled), "disabled");
    for (BlendingMode mode : {BlendingMode::Mosaic, BlendingMode::Average, BlendingMode::Disabled}) {
        EXPECT_EQ(parse_blending_mode(to_string(mode)), mode);
    }
    EXPECT_THROW(parse_blending_mode(""), ValueError);
    EXPECT_THROW(parse_blending_mode("Mosaic"), ValueError);  // case-sensitive
    EXPECT_THROW(parse_blending_mode("blend"), ValueError);
}

// ---------------------------------------------------------------------------
// Mosaic accuracy on a consistent scene. The views are noise-free renderings of
// one smooth field, so with colour correction off the baked page must reproduce
// the field at every in-chart texel, and almost every in-chart texel must be
// coloured even with fill_holes off.
// ---------------------------------------------------------------------------
TEST(Bake, MosaicIsAccurateCoverageAndDeterministic) {
    Scene scene = build_scene(512, 4);
    BakeOptions opts;
    opts.blending = BlendingMode::Mosaic;
    opts.color_correction = false;
    opts.fill_holes = false;
    opts.dilation_px = 0;

    const Image page = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, opts);
    ASSERT_EQ(page.width(), 512);
    ASSERT_EQ(page.height(), 512);
    ASSERT_EQ(page.channels(), 3);

    const int S = scene.texture_size;
    double sum[3] = {0.0, 0.0, 0.0};
    double maxerr = 0.0;
    std::size_t n = 0;
    std::size_t in_chart = 0;
    std::size_t coloured = 0;
    std::vector<double> errors;
    for (int j = 0; j < S; ++j) {
        for (int i = 0; i < S; ++i) {
            Eigen::Vector3d X;
            if (!texel_point(scene.mesh, S, i, j, X)) {
                continue;
            }
            ++in_chart;
            const bool lit = page.at(i, j, 0) != 0 || page.at(i, j, 1) != 0 || page.at(i, j, 2) != 0;
            if (lit) {
                ++coloured;
            }
            ++n;
            for (int c = 0; c < 3; ++c) {
                const double e = std::abs(static_cast<double>(page.at(i, j, c)) - field(X, c));
                sum[c] += e;
                maxerr = std::max(maxerr, e);
                errors.push_back(e);
            }
        }
    }
    ASSERT_GT(n, 1000u);
    const double mean = (sum[0] + sum[1] + sum[2]) / (3.0 * static_cast<double>(n));
    std::sort(errors.begin(), errors.end());
    const double p99 =
        errors[static_cast<std::size_t>(static_cast<double>(errors.size() - 1) * 0.99)];
    std::printf("Bake mosaic: mean=%.3f p99=%.2f max=%.2f over %zu texels\n", mean, p99, maxerr, n);
    EXPECT_LT(mean, 4.0);
    EXPECT_LT(p99, 20.0);

    const double cov = static_cast<double>(coloured) / static_cast<double>(in_chart);
    std::printf("Bake mosaic coverage: %.4f (%zu/%zu)\n", cov, coloured, in_chart);
    EXPECT_GT(cov, 0.995);
}

// ---------------------------------------------------------------------------
// Disabled. Documented as "each face's best view, no colour correction, no
// blending" -- i.e. byte-identical to Mosaic with color_correction off. It must
// still paint the page (an earlier revision skipped every texel and produced an
// all-black texture).
// ---------------------------------------------------------------------------
TEST(Bake, DisabledIsMosaicWithoutCorrection) {
    Scene scene = build_scene(512, 4);
    BakeOptions mosaic;
    mosaic.blending = BlendingMode::Mosaic;
    mosaic.color_correction = false;
    mosaic.fill_holes = false;
    mosaic.dilation_px = 0;
    BakeOptions disabled = mosaic;
    disabled.blending = BlendingMode::Disabled;

    const Image a = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, mosaic);
    const Image b = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, disabled);
    ASSERT_EQ(a.size_bytes(), b.size_bytes());
    const bool same = std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;

    const int S = scene.texture_size;
    std::size_t in_chart = 0;
    std::size_t coloured = 0;
    for (int j = 0; j < S; ++j) {
        for (int i = 0; i < S; ++i) {
            Eigen::Vector3d X;
            if (!texel_point(scene.mesh, S, i, j, X)) {
                continue;
            }
            ++in_chart;
            if (b.at(i, j, 0) != 0 || b.at(i, j, 1) != 0 || b.at(i, j, 2) != 0) {
                ++coloured;
            }
        }
    }
    const double cov = static_cast<double>(coloured) / static_cast<double>(in_chart);
    std::printf("Bake disabled: coverage=%.4f (%zu/%zu), byte-identical to mosaic=%d\n", cov, coloured,
                in_chart, static_cast<int>(same));
    EXPECT_TRUE(same) << "Disabled must reproduce Mosaic with colour correction off, byte for byte";
    EXPECT_GT(cov, 0.995);
}

// ---------------------------------------------------------------------------
// Average mode weights several views; still accurate on a consistent scene.
// ---------------------------------------------------------------------------
TEST(Bake, AverageModeIsAccurate) {
    Scene scene = build_scene(512, 4);
    BakeOptions opts;
    opts.blending = BlendingMode::Average;
    opts.color_correction = false;
    opts.fill_holes = false;
    opts.dilation_px = 0;
    const Image page = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, opts);

    const int S = scene.texture_size;
    double sum = 0.0;
    std::size_t n = 0;
    for (int j = 0; j < S; ++j) {
        for (int i = 0; i < S; ++i) {
            Eigen::Vector3d X;
            if (!texel_point(scene.mesh, S, i, j, X)) {
                continue;
            }
            ++n;
            for (int c = 0; c < 3; ++c) {
                sum += std::abs(static_cast<double>(page.at(i, j, c)) - field(X, c));
            }
        }
    }
    ASSERT_GT(n, 1000u);
    const double mean = sum / (3.0 * static_cast<double>(n));
    std::printf("Bake average: mean=%.3f over %zu texels\n", mean, n);
    EXPECT_LT(mean, 5.0);
}

// ---------------------------------------------------------------------------
// Colour correction. One view is rendered with every pixel scaled by 0.7. With
// correction off that view's faces are uniformly dark, so a texel's baked/truth
// ratio deviates from the page's median ratio depending on which view it used;
// correction pulls every view's ratio toward the global one.
// ---------------------------------------------------------------------------
TEST(Bake, ColorCorrectionLevelsViews) {
    Scene scene = build_scene(512, 4);
    // Darken one whole view (index 0) by 0.7
    for (int y = 0; y < scene.images[0].height(); ++y) {
        for (int x = 0; x < scene.images[0].width(); ++x) {
            for (int c = 0; c < 3; ++c) {
                const double v = static_cast<double>(scene.images[0].at(x, y, c)) * 0.7;
                scene.images[0].at(x, y, c) = static_cast<std::uint8_t>(std::lround(v));
            }
        }
    }
    const int S = scene.texture_size;
    const auto seam_metric = [&](bool correction) -> double {
        BakeOptions opts;
        opts.blending = BlendingMode::Mosaic;
        opts.color_correction = correction;
        opts.fill_holes = false;
        opts.dilation_px = 0;
        const Image page = bake_texture(scene.mesh, scene.views, scene.fv, S, opts);
        // Mean absolute deviation of baked/truth ratio from the page median.
        std::vector<double> ratios;
        ratios.reserve(static_cast<std::size_t>(S) * static_cast<std::size_t>(S) / 4);
        for (int j = 0; j < S; ++j) {
            for (int i = 0; i < S; ++i) {
                Eigen::Vector3d X;
                if (!texel_point(scene.mesh, S, i, j, X)) {
                    continue;
                }
                const double truth = field(X, 1);
                if (truth < 20.0) {
                    continue;
                }
                ratios.push_back(static_cast<double>(page.at(i, j, 1)) / truth);
            }
        }
        if (ratios.empty()) {
            return 1e9;
        }
        std::sort(ratios.begin(), ratios.end());
        const double med = ratios[ratios.size() / 2];
        double dev = 0.0;
        for (double r : ratios) {
            dev += std::abs(r - med);
        }
        return dev / static_cast<double>(ratios.size());
    };
    const double uncorrected = seam_metric(false);
    const double corrected = seam_metric(true);
    std::printf("Bake colour correction: seam dev uncorrected=%.4f corrected=%.4f\n", uncorrected,
                corrected);
    EXPECT_LT(corrected, uncorrected);
}

// ---------------------------------------------------------------------------
// Holes. Forcing one face's best view to -1 leaves its in-chart texels
// uncoloured with fill_holes off; with it on they are coloured from neighbours
// (in-chart black texels go to zero) while no golden texel is disturbed.
// ---------------------------------------------------------------------------
TEST(Bake, FillHolesRecoversUnseenFaces) {
    Scene scene = build_scene(256, 4);
    const int victim = static_cast<int>(scene.fv.best.size() / 2);
    scene.fv.best[static_cast<std::size_t>(victim)] = -1;
    scene.fv.visible[static_cast<std::size_t>(victim)].clear();

    BakeOptions off;
    off.blending = BlendingMode::Mosaic;
    off.color_correction = false;
    off.fill_holes = false;
    off.dilation_px = 0;
    BakeOptions on = off;
    on.fill_holes = true;

    const Image a = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, off);
    const Image b = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, on);

    const int S = scene.texture_size;
    const auto black = [](const Image& img, int i, int j) {
        return img.at(i, j, 0) == 0 && img.at(i, j, 1) == 0 && img.at(i, j, 2) == 0;
    };
    std::size_t chart_total = 0;
    std::size_t hole_off = 0;
    std::size_t hole_on = 0;
    std::size_t colored_off = 0;
    std::size_t colored_on = 0;
    for (int j = 0; j < S; ++j) {
        for (int i = 0; i < S; ++i) {
            Eigen::Vector3d X;
            if (!texel_point(scene.mesh, S, i, j, X)) {
                continue;
            }
            ++chart_total;
            if (!black(a, i, j)) {
                ++colored_off;
            }
            if (!black(b, i, j)) {
                ++colored_on;
            }
            if (black(a, i, j)) {
                ++hole_off;
            }
            if (black(b, i, j)) {
                ++hole_on;
            }
        }
    }
    std::printf("Bake holes: in-chart=%zu, black(fill off)=%zu, black(fill on)=%zu\n", chart_total,
                hole_off, hole_on);
    EXPECT_GT(hole_off, 0u);       // the forced -1 face left real holes
    ASSERT_GT(hole_off, 0u);
    EXPECT_EQ(hole_on, 0u);        // every in-chart hole was filled
    EXPECT_GT(colored_on, colored_off);
}

// ---------------------------------------------------------------------------
// Dilation: a non-zero gutter colours the ring of texels just outside the
// charts, while page corners (far from any chart) stay black.
// ---------------------------------------------------------------------------
TEST(Bake, DilationGrowsChartsOutward) {
    Scene scene = build_scene(512, 8);
    BakeOptions opts;
    opts.blending = BlendingMode::Mosaic;
    opts.color_correction = false;
    opts.fill_holes = true;
    opts.dilation_px = 4;
    const Image page = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, opts);

    const int S = scene.texture_size;
    const auto black = [](const Image& img, int i, int j) {
        return img.at(i, j, 0) == 0 && img.at(i, j, 1) == 0 && img.at(i, j, 2) == 0;
    };

    // Page corners are far from any chart.
    int corner_black = 0;
    for (int c = 0; c < 4; ++c) {
        const int x = (c & 1) ? S - 1 : 0;
        const int y = (c & 2) ? S - 1 : 0;
        if (black(page, x, y)) {
            ++corner_black;
        }
    }
    EXPECT_EQ(corner_black, 4);

    // Compare against a zero-gutter bake: every texel coloured only by the
    // dilation pass is out-of-chart in `bare` and coloured in `page`.
    BakeOptions nodil = opts;
    nodil.dilation_px = 0;
    const Image bare = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, nodil);
    std::size_t filled_by_dilation = 0;
    std::size_t still_black_outside = 0;
    for (int j = 0; j < S; ++j) {
        for (int i = 0; i < S; ++i) {
            Eigen::Vector3d X;
            const bool in_chart = texel_point(scene.mesh, S, i, j, X);
            if (in_chart) {
                continue;  // only look outside charts
            }
            if (black(bare, i, j) && !black(page, i, j)) {
                ++filled_by_dilation;
            }
            if (black(page, i, j)) {
                ++still_black_outside;
            }
        }
    }
    std::printf("Bake dilation: texels filled=%zu, still black outside=%zu\n", filled_by_dilation,
                still_black_outside);
    EXPECT_GT(filled_by_dilation, 0u);
    EXPECT_GT(still_black_outside, 0u);  // far-out texels stayed black
}

// ---------------------------------------------------------------------------
// Determinism: identical bytes across OpenMP thread counts, and across repeats.
// ---------------------------------------------------------------------------
TEST(Bake, DeterministicAcrossThreadCounts) {
    Scene scene = build_scene(256, 4);
    BakeOptions opts;
    opts.blending = BlendingMode::Average;
    opts.color_correction = true;
    opts.fill_holes = true;
    opts.dilation_px = 4;

    const Image a = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, opts);
#if defined(ALPHASHAPE_HAVE_OPENMP)
    omp_set_num_threads(1);
    const Image one = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, opts);
    omp_set_num_threads(4);
    const Image four = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, opts);
    omp_set_num_threads(omp_get_max_threads());
    EXPECT_EQ(one.size_bytes(), four.size_bytes());
    EXPECT_TRUE(std::equal(one.data(), one.data() + one.size_bytes(), four.data()));
#else
    GTEST_SKIP() << "built without OpenMP";
#endif
    const Image again = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, opts);
    EXPECT_TRUE(std::equal(a.data(), a.data() + a.size_bytes(), again.data()));
}

// ---------------------------------------------------------------------------
// Invalid inputs.
// ---------------------------------------------------------------------------
TEST(Bake, InvalidInputsThrow) {
    Scene scene = build_scene(256, 4);
    const BakeOptions ok;

    // uv_faces size mismatch.
    {
        Mesh bad = scene.mesh;
        bad.uv_faces.pop_back();
        EXPECT_THROW(bake_texture(bad, scene.views, scene.fv, 256, ok), ValueError);
    }
    // uv index out of range.
    {
        Mesh bad = scene.mesh;
        bad.uv_faces[0][0] = static_cast<int>(bad.uvs.size()) + 5;
        EXPECT_THROW(bake_texture(bad, scene.views, scene.fv, 256, ok), ValueError);
    }
    // texture_size too small.
    EXPECT_THROW(bake_texture(scene.mesh, scene.views, scene.fv, 8, ok), ValueError);
    // face_views size mismatch.
    {
        FaceViews fv = scene.fv;
        fv.best.pop_back();
        EXPECT_THROW(bake_texture(scene.mesh, scene.views, fv, 256, ok), ValueError);
    }
    // null view image.
    {
        std::vector<TexView> views = scene.views;
        views[0].image = nullptr;
        EXPECT_THROW(bake_texture(scene.mesh, views, scene.fv, 256, ok), ValueError);
    }
    // image size != calibration size.
    {
        std::vector<TexView> views = scene.views;
        views[0].calibration.width += 4;
        EXPECT_THROW(bake_texture(scene.mesh, views, scene.fv, 256, ok), ValueError);
    }
    // negative dilation.
    {
        BakeOptions bad = ok;
        bad.dilation_px = -1;
        EXPECT_THROW(bake_texture(scene.mesh, scene.views, scene.fv, 256, bad), ValueError);
    }
    // visible referencing an unknown view.
    {
        FaceViews fv = scene.fv;
        fv.visible[0].push_back(999);
        EXPECT_THROW(bake_texture(scene.mesh, scene.views, fv, 256, ok), ValueError);
    }
}

// ---------------------------------------------------------------------------
// Perf: a 2048 page on the box scene. Prints the runtime.
// ---------------------------------------------------------------------------
TEST(Bake, PerfTwoThousandFortyEightPage) {
    Scene scene = build_scene(2048, 4);
    BakeOptions opts;
    opts.blending = BlendingMode::Average;
    opts.color_correction = true;
    opts.fill_holes = true;
    opts.dilation_px = 4;
    const auto t0 = std::chrono::steady_clock::now();
    const Image page = bake_texture(scene.mesh, scene.views, scene.fv, scene.texture_size, opts);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Bake perf: 2048 page, %zu faces, %zu views -> %.0f ms\n", scene.mesh.faces.size(),
                scene.views.size(), ms);
    EXPECT_EQ(page.width(), 2048);
    EXPECT_LT(ms, 120000.0);
}

}  // namespace texturing
}  // namespace alphashape
