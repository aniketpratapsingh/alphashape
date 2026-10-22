// test_face_views.cpp -- face-to-view selection (milestone 8 texturing).
//
// The meshes are analytic (a box, a rippled plane, a UV sphere) and the views
// are synthetic look-at cameras, so the expected assignment follows from the
// geometry alone. Images are blank: selection only reads their size.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "base/utils/error.hpp"
#include "common/algorithm/texture/face_views.hpp"
#include "image/image.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape {
namespace texturing {
namespace {

/// A look-at camera at `eye` aimed at `target`. The camera frame is +X right,
/// +Y down, +Z forward, so row 2 is the forward axis. The up hint falls back to
/// world +Z when the view direction is (anti)parallel to world +Y, which would
/// otherwise make the right vector degenerate.
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

/// Append a quad given its four corners in outward-CCW order; returns the index
/// of its first face (it adds two).
int add_quad(Mesh& m, const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
             const Eigen::Vector3d& d) {
    const int base = static_cast<int>(m.vertices.size());
    m.vertices.push_back(a);
    m.vertices.push_back(b);
    m.vertices.push_back(c);
    m.vertices.push_back(d);
    const int first = static_cast<int>(m.faces.size());
    m.faces.push_back(Eigen::Vector3i(base, base + 1, base + 2));
    m.faces.push_back(Eigen::Vector3i(base, base + 2, base + 3));
    return first;
}

/// An axis-aligned box centred at `c` with half-extent `half`. Face order is
/// +X, -X, +Y, -Y, +Z, -Z, two triangles each, so side `s` owns faces
/// [2s, 2s + 1].
Mesh make_box(const Eigen::Vector3d& c, double half) {
    Mesh m;
    const double h = half;
    // +X
    add_quad(m, {c.x() + h, c.y() - h, c.z() - h}, {c.x() + h, c.y() + h, c.z() - h},
             {c.x() + h, c.y() + h, c.z() + h}, {c.x() + h, c.y() - h, c.z() + h});
    // -X
    add_quad(m, {c.x() - h, c.y() - h, c.z() + h}, {c.x() - h, c.y() + h, c.z() + h},
             {c.x() - h, c.y() + h, c.z() - h}, {c.x() - h, c.y() - h, c.z() - h});
    // +Y
    add_quad(m, {c.x() - h, c.y() + h, c.z() + h}, {c.x() + h, c.y() + h, c.z() + h},
             {c.x() + h, c.y() + h, c.z() - h}, {c.x() - h, c.y() + h, c.z() - h});
    // -Y
    add_quad(m, {c.x() - h, c.y() - h, c.z() - h}, {c.x() + h, c.y() - h, c.z() - h},
             {c.x() + h, c.y() - h, c.z() + h}, {c.x() - h, c.y() - h, c.z() + h});
    // +Z
    add_quad(m, {c.x() - h, c.y() - h, c.z() + h}, {c.x() + h, c.y() - h, c.z() + h},
             {c.x() + h, c.y() + h, c.z() + h}, {c.x() - h, c.y() + h, c.z() + h});
    // -Z
    add_quad(m, {c.x() + h, c.y() - h, c.z() - h}, {c.x() - h, c.y() - h, c.z() - h},
             {c.x() - h, c.y() + h, c.z() - h}, {c.x() + h, c.y() + h, c.z() - h});
    return m;
}

bool any_visible_from(const FaceViews& fv, int face, int view) {
    const std::vector<int>& vis = fv.visible[static_cast<std::size_t>(face)];
    return std::find(vis.begin(), vis.end(), view) != vis.end();
}

/// A box centred at `c` with half-extent `half`, each side subdivided into an
/// `n x n` grid of quads (so `2 n^2` triangles per side, `12 n^2` in total),
/// every triangle wound outward-CCW. Face side s (0..5 = +X,-X,+Y,-Y,+Z,-Z)
/// owns faces [2 n^2 s, 2 n^2 (s + 1)).
Mesh make_subdivided_box(const Eigen::Vector3d& c, double half, int n) {
    Mesh m;
    const double h = half;
    // Axis order and outward orientation per side: (origin corner, u axis, v
    // axis) chosen so u x v points out of the box.
    struct Side {
        Eigen::Vector3d o;
        Eigen::Vector3d u;
        Eigen::Vector3d v;
    };
    const std::array<Side, 6> sides = {
        Side{{c.x() + h, c.y() - h, c.z() - h}, {0, 1, 0}, {0, 0, 1}},  // +X
        Side{{c.x() - h, c.y() - h, c.z() + h}, {0, 1, 0}, {0, 0, -1}},  // -X
        Side{{c.x() - h, c.y() + h, c.z() + h}, {1, 0, 0}, {0, 0, -1}},  // +Y
        Side{{c.x() - h, c.y() - h, c.z() - h}, {1, 0, 0}, {0, 0, 1}},   // -Y
        Side{{c.x() - h, c.y() - h, c.z() + h}, {1, 0, 0}, {0, 1, 0}},   // +Z
        Side{{c.x() + h, c.y() - h, c.z() - h}, {-1, 0, 0}, {0, 1, 0}},  // -Z
    };
    for (const Side& s : sides) {
        const int base = static_cast<int>(m.vertices.size());
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                const double ui = 2.0 * h * static_cast<double>(i) / n;
                const double vj = 2.0 * h * static_cast<double>(j) / n;
                m.vertices.push_back(s.o + ui * s.u + vj * s.v);
            }
        }
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const int a = base + j * (n + 1) + i;
                const int b = a + 1;
                const int d = a + (n + 1);
                const int e = d + 1;
                m.faces.push_back(Eigen::Vector3i(a, b, e));
                m.faces.push_back(Eigen::Vector3i(a, e, d));
            }
        }
    }
    return m;
}

/// Faces of a rippled plane z = amp*sin(a x)cos(b y), a triangle grid of n x n
/// quads split along the diagonal.
Mesh make_ripple(int n, double half, double amp) {
    Mesh m;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            const double x = -half + 2.0 * half * static_cast<double>(i) / n;
            const double y = -half + 2.0 * half * static_cast<double>(j) / n;
            const double z = amp * std::sin(11.0 * x) * std::cos(9.0 * y);
            m.vertices.push_back(Eigen::Vector3d(x, y, z));
        }
    }
    const auto vid = [n](int i, int j) { return j * (n + 1) + i; };
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int a = vid(i, j);
            const int b = vid(i + 1, j);
            const int c = vid(i + 1, j + 1);
            const int d = vid(i, j + 1);
            m.faces.push_back(Eigen::Vector3i(a, b, c));
            m.faces.push_back(Eigen::Vector3i(a, c, d));
        }
    }
    return m;
}

/// Number of shared edges whose two faces carry different labels.
int count_label_seams(const Mesh& m, const std::vector<int>& labels) {
    std::map<std::pair<int, int>, int> owner;
    int seams = 0;
    for (std::size_t f = 0; f < m.faces.size(); ++f) {
        const Eigen::Vector3i& t = m.faces[f];
        for (int k = 0; k < 3; ++k) {
            const int u = t[k];
            const int w = t[(k + 1) % 3];
            const std::pair<int, int> key(std::min(u, w), std::max(u, w));
            const auto it = owner.find(key);
            if (it == owner.end()) {
                owner.emplace(key, static_cast<int>(f));
            } else if (labels[static_cast<std::size_t>(it->second)] !=
                       labels[f]) {
                ++seams;
            }
        }
    }
    return seams;
}

/// A UV sphere, roughly `target_faces` triangles.
Mesh make_sphere(int stacks, int slices, double radius) {
    Mesh m;
    for (int j = 0; j <= stacks; ++j) {
        const double phi = M_PI * static_cast<double>(j) / stacks;  // 0..pi from +Y
        for (int i = 0; i < slices; ++i) {
            const double theta = 2.0 * M_PI * static_cast<double>(i) / slices;
            m.vertices.push_back(Eigen::Vector3d(radius * std::sin(phi) * std::cos(theta),
                                                 radius * std::cos(phi),
                                                 radius * std::sin(phi) * std::sin(theta)));
        }
    }
    const auto vid = [slices](int j, int i) { return j * slices + (i % slices); };
    for (int j = 0; j < stacks; ++j) {
        for (int i = 0; i < slices; ++i) {
            const int a = vid(j, i);
            const int b = vid(j, i + 1);
            const int c = vid(j + 1, i + 1);
            const int d = vid(j + 1, i);
            m.faces.push_back(Eigen::Vector3i(a, b, c));
            m.faces.push_back(Eigen::Vector3i(a, c, d));
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// A box seen by four cameras, one per +-X / +-Y side. Each side face's best
// view is the camera that faces it; the top and bottom faces are grazing or
// unseen at the default min_view_cos.
// ---------------------------------------------------------------------------
TEST(FaceViews, BoxSidesChooseTheFacingCamera) {
    const int W = 256;
    const int H = 256;
    Image blank(W, H, 3);
    const double f = 256.0;
    const double d = 4.0;
    const Eigen::Vector3d c(0.0, 0.0, 0.0);
    const Mesh box = make_box(c, 1.0);

    std::vector<TexView> views;
    std::vector<Eigen::Vector3d> eyes = {Eigen::Vector3d(d, 0, 0), Eigen::Vector3d(-d, 0, 0),
                                         Eigen::Vector3d(0, d, 0), Eigen::Vector3d(0, -d, 0)};
    for (const Eigen::Vector3d& eye : eyes) {
        TexView v = look_at(eye, c, W, H, f);
        v.image = &blank;
        views.push_back(v);
    }

    FaceViewOptions options;
    options.zbuffer_downscale = 2;
    options.min_view_cos = 0.15;
    options.smoothing_passes = 0;  // raw assignment
    const FaceViews fv = select_face_views(box, views, options);
    ASSERT_EQ(fv.best.size(), 12u);

    // Side s owns faces 2s, 2s+1 and faces the camera at eyes[s].
    for (int s = 0; s < 4; ++s) {
        for (int k = 0; k < 2; ++k) {
            const int face = 2 * s + k;
            EXPECT_EQ(fv.best[static_cast<std::size_t>(face)], s)
                << "side " << s << " face " << face;
            EXPECT_TRUE(any_visible_from(fv, face, s));
            for (int other = 0; other < 4; ++other) {
                if (other != s) {
                    EXPECT_FALSE(any_visible_from(fv, face, other));
                }
            }
        }
    }
    // +Y/-Y faces are 4,5 and 6,7 (they face the cameras at eyes[2], eyes[3]);
    // the top/bottom +Z/-Z faces are 8..11 and are grazing from these cameras.
    for (int face = 8; face < 12; ++face) {
        EXPECT_EQ(fv.best[static_cast<std::size_t>(face)], -1) << "top/bottom face " << face;
        EXPECT_TRUE(fv.visible[static_cast<std::size_t>(face)].empty());
    }
}

// ---------------------------------------------------------------------------
// Occlusion. Compared against a control: a small box alone is seen head-on by
// camera A, but once a big box is interposed between A and it, A loses the
// small box while camera B behind it keeps seeing it.
// ---------------------------------------------------------------------------
TEST(FaceViews, OccludedFacesExcludeTheBlockingView) {
    const int W = 320;
    const int H = 320;
    Image blank(W, H, 3);
    const Mesh small = make_box(Eigen::Vector3d(0, 0, 7.0), 0.5);

    TexView a = look_at(Eigen::Vector3d(0, 0, -14), Eigen::Vector3d(0, 0, 7.0), W, H, 400.0);
    TexView b = look_at(Eigen::Vector3d(0, 0, 14), Eigen::Vector3d(0, 0, 7.0), W, H, 400.0);
    a.image = &blank;
    b.image = &blank;
    const std::vector<TexView> views = {a, b};

    // Face 10/11 is the small box's -Z face (normal -Z), so camera A at -Z is
    // the one facing it and camera B at +Z is behind it.
    FaceViewOptions options;
    options.zbuffer_downscale = 1;
    options.min_view_cos = 0.15;
    options.smoothing_passes = 0;

    // Control: with nothing in the way, A sees the small box's -Z face (10, 11
    // in the standalone mesh).
    const FaceViews solo = select_face_views(small, views, options);
    for (int face : {10, 11}) {
        EXPECT_TRUE(any_visible_from(solo, face, 0)) << "control: face " << face << " not seen by A";
    }

    // Interpose the big box between A and the small box: the scene's faces are
    // the big box's 0..11 followed by the small box's 12..23, so the small box's
    // -Z face is 22, 23 and its +Z face is 20, 21.
    Mesh scene = make_box(Eigen::Vector3d(0, 0, 0), 2.0);
    const int base = static_cast<int>(scene.vertices.size());
    for (const Eigen::Vector3d& v : small.vertices) {
        scene.vertices.push_back(v);
    }
    for (const Eigen::Vector3i& fc : small.faces) {
        scene.faces.push_back(Eigen::Vector3i(fc[0] + base, fc[1] + base, fc[2] + base));
    }
    const FaceViews fv = select_face_views(scene, views, options);

    // The small box's -Z face is now hidden from A by the big box, and B never
    // faced it, so it resolves to no camera at all.
    for (int face : {22, 23}) {
        EXPECT_FALSE(any_visible_from(fv, face, 0))
            << "face " << face << " should be hidden from A by the big box";
        EXPECT_EQ(fv.best[static_cast<std::size_t>(face)], -1) << "face " << face;
    }
    // The small box's +Z face (20, 21) faces B, which is unobstructed.
    for (int face : {20, 21}) {
        EXPECT_TRUE(any_visible_from(fv, face, 1)) << "face " << face << " hidden from B";
    }
}

// ---------------------------------------------------------------------------
// Convex box, no false occlusion. A convex closed surface occludes nothing, so
// every face that faces a camera (cos >= min_view_cos) must resolve to one. The
// old centroid-only test against the half-resolution buffer rejected a fifth of
// the faces at the default zbuffer_downscale = 2, because a small face is
// sub-pixel there and its centroid pixel often holds a neighbour.
// ---------------------------------------------------------------------------
TEST(FaceViews, ConvexBoxHasNoFalseOcclusion) {
    const int W = 200;
    const int H = 150;
    const double f = 180.0;
    Image blank(W, H, 3);

    const int n = 10;  // 12 * 10 * 10 = 1200 faces
    const double half = 1.0;
    const Mesh box = make_subdivided_box(Eigen::Vector3d(0, 0, 0), half, n);
    ASSERT_EQ(box.faces.size(), 1200u);

    const double d = 4.0;
    const std::array<Eigen::Vector3d, 6> eyes = {
        Eigen::Vector3d(d, 0, 0),  Eigen::Vector3d(-d, 0, 0), Eigen::Vector3d(0, d, 0),
        Eigen::Vector3d(0, -d, 0), Eigen::Vector3d(0, 0, d),  Eigen::Vector3d(0, 0, -d)};
    std::vector<TexView> views;
    for (const Eigen::Vector3d& eye : eyes) {
        TexView v = look_at(eye, Eigen::Vector3d(0, 0, 0), W, H, f);
        v.image = &blank;
        views.push_back(v);
    }

    // Per face, the largest viewing cosine over the six cameras: faces at or
    // above min_view_cos are front-facing somewhere and must never be unseen.
    const double min_cos = FaceViewOptions{}.min_view_cos;
    std::vector<bool> must_see(box.faces.size(), false);
    for (std::size_t fi = 0; fi < box.faces.size(); ++fi) {
        const Eigen::Vector3i& t = box.faces[fi];
        const Eigen::Vector3d& a = box.vertices[static_cast<std::size_t>(t[0])];
        const Eigen::Vector3d& b = box.vertices[static_cast<std::size_t>(t[1])];
        const Eigen::Vector3d& cc = box.vertices[static_cast<std::size_t>(t[2])];
        const Eigen::Vector3d centroid = (a + b + cc) / 3.0;
        const Eigen::Vector3d nrm = (b - a).cross(cc - a).normalized();
        double best_cos = -2.0;
        for (const Eigen::Vector3d& eye : eyes) {
            const Eigen::Vector3d dir = (eye - centroid).normalized();
            best_cos = std::max(best_cos, nrm.dot(dir));
        }
        must_see[fi] = best_cos >= min_cos;
    }
    std::size_t n_must = 0;
    for (bool v : must_see) {
        n_must += v ? 1u : 0u;
    }

    for (int downscale : {1, 2, 4}) {
        FaceViewOptions options;  // defaults: downscale 2, cos 0.15, 6 passes
        options.zbuffer_downscale = downscale;
        const FaceViews fv = select_face_views(box, views, options);
        std::size_t unseen = 0;
        for (std::size_t fi = 0; fi < box.faces.size(); ++fi) {
            if (fv.best[fi] < 0) {
                ++unseen;
            }
        }
        std::size_t missing = 0;  // front-facing but unseen: a false negative
        for (std::size_t fi = 0; fi < box.faces.size(); ++fi) {
            if (must_see[fi] && fv.best[fi] < 0) {
                ++missing;
            }
        }
        std::printf("FaceViews convex box: downscale=%d unseen=%zu (%zu front-facing missing)\n",
                    downscale, unseen, missing);
    }

    // Default options (downscale 2): no false negative at all.
    FaceViewOptions def;
    const FaceViews fv2 = select_face_views(box, views, def);
    std::size_t missing2 = 0;
    for (std::size_t fi = 0; fi < box.faces.size(); ++fi) {
        if (must_see[fi] && fv2.best[fi] < 0) {
            ++missing2;
        }
    }
    EXPECT_EQ(missing2, 0u) << "false negatives at the default downscale (front-facing = " << n_must
                            << ")";

    // Downscale 4: still at most 1% of the front-facing faces lost.
    FaceViewOptions o4;
    o4.zbuffer_downscale = 4;
    const FaceViews fv4 = select_face_views(box, views, o4);
    std::size_t missing4 = 0;
    for (std::size_t fi = 0; fi < box.faces.size(); ++fi) {
        if (must_see[fi] && fv4.best[fi] < 0) {
            ++missing4;
        }
    }
    EXPECT_LE(missing4 * 100u, n_must) << "downscale 4 lost " << missing4 << " of " << n_must;
}

// ---------------------------------------------------------------------------
// Occlusion at depth. A box only 30% of the camera distance behind the blocker
// is still fully hidden from A: the multi-sample test must detect real
// occlusion, not just the adjacency of the depth tolerance.
// ---------------------------------------------------------------------------
TEST(FaceViews, OcclusionDetectedBehindBlocker) {
    const int W = 320;
    const int H = 320;
    Image blank(W, H, 3);

    // Blocker centred at the origin (half 2), small box at z = 3.9 (half 0.5):
    // the small box's -Z face sits ~0.3 of A's distance (13 units) past the
    // blocker's +Z face at z = 2, so A is blocked by a solid depth margin.
    Mesh scene = make_box(Eigen::Vector3d(0, 0, 0), 2.0);
    const Mesh small = make_box(Eigen::Vector3d(0, 0, 3.9), 0.5);
    const int base = static_cast<int>(scene.vertices.size());
    for (const Eigen::Vector3d& v : small.vertices) {
        scene.vertices.push_back(v);
    }
    for (const Eigen::Vector3i& fc : small.faces) {
        scene.faces.push_back(Eigen::Vector3i(fc[0] + base, fc[1] + base, fc[2] + base));
    }

    TexView a = look_at(Eigen::Vector3d(0, 0, -13), Eigen::Vector3d(0, 0, 2.0), W, H, 400.0);
    a.image = &blank;
    const std::vector<TexView> views = {a};

    FaceViewOptions options;
    options.zbuffer_downscale = 2;
    options.min_view_cos = 0.15;
    options.smoothing_passes = 0;
    const FaceViews fv = select_face_views(scene, views, options);

    // Small box -Z face is faces 22, 23 (small box occupies 12..23): it faces A
    // but is hidden behind the blocker, so no camera sees it.
    for (int face : {22, 23}) {
        EXPECT_FALSE(any_visible_from(fv, face, 0))
            << "face " << face << " should be hidden from A by the blocker";
        EXPECT_EQ(fv.best[static_cast<std::size_t>(face)], -1) << "face " << face;
    }
    // Blocker's own -Z face (10, 11) faces A and is unoccluded.
    for (int face : {10, 11}) {
        EXPECT_TRUE(any_visible_from(fv, face, 0)) << "blocker face " << face << " hidden";
    }
}

// ---------------------------------------------------------------------------
// Back faces are never candidates: a face whose normal points away from the
// camera fails the viewing-cosine test even with the most permissive limit.
// ---------------------------------------------------------------------------
TEST(FaceViews, BackFacesAreNeverVisible) {
    const int W = 256;
    const int H = 256;
    Image blank(W, H, 3);
    const Mesh box = make_box(Eigen::Vector3d(0, 0, 0), 1.0);
    TexView v = look_at(Eigen::Vector3d(0, 0, -6), Eigen::Vector3d(0, 0, 0), W, H, 300.0);
    v.image = &blank;
    const std::vector<TexView> views = {v};

    FaceViewOptions options;
    options.zbuffer_downscale = 1;
    options.min_view_cos = -0.999;  // accept nearly everything seen from the front
    options.smoothing_passes = 0;
    const FaceViews fv = select_face_views(box, views, options);

    // From -Z only the -Z face (faces 10, 11) is front-facing.
    for (int face = 0; face < 12; ++face) {
        const bool expected = (face == 10 || face == 11);
        EXPECT_EQ(!fv.visible[static_cast<std::size_t>(face)].empty(), expected)
            << "face " << face;
    }
}

// ---------------------------------------------------------------------------
// Smoothing. A rippled plane seen by two nearly-equally-good cameras gives a
// speckled raw assignment; ICM smoothing must not increase the number of label
// seams across shared edges.
// ---------------------------------------------------------------------------
TEST(FaceViews, SmoothingRemovesLabelSeams) {
    const int W = 400;
    const int H = 400;
    Image blank(W, H, 3);
    const Mesh plane = make_ripple(12, 1.0, 0.08);

    TexView a = look_at(Eigen::Vector3d(-0.2, 0, 3), Eigen::Vector3d(0, 0, 0), W, H, 600.0);
    TexView b = look_at(Eigen::Vector3d(0.2, 0, 3), Eigen::Vector3d(0, 0, 0), W, H, 600.0);
    a.image = &blank;
    b.image = &blank;
    const std::vector<TexView> views = {a, b};

    FaceViewOptions options;
    options.zbuffer_downscale = 1;
    options.min_view_cos = -1.0;
    options.smoothness = 1.0;

    options.smoothing_passes = 0;
    const FaceViews raw = select_face_views(plane, views, options);
    const int seams_raw = count_label_seams(plane, raw.best);

    options.smoothing_passes = 6;
    const FaceViews smooth = select_face_views(plane, views, options);
    const int seams_smooth = count_label_seams(plane, smooth.best);

    std::printf("FaceViews smoothing: seams raw=%d smooth=%d (faces=%zu)\n", seams_raw,
                seams_smooth, plane.faces.size());
    EXPECT_GT(seams_raw, 0);  // the raw assignment really is speckled
    EXPECT_LE(seams_smooth, seams_raw);
    EXPECT_LT(seams_smooth, seams_raw);
}

// ---------------------------------------------------------------------------
// The result must not depend on the OpenMP thread count.
// ---------------------------------------------------------------------------
TEST(FaceViews, DeterministicAcrossThreadCounts) {
#if defined(ALPHASHAPE_HAVE_OPENMP)
    const int W = 320;
    const int H = 240;
    Image blank(W, H, 3);
    Mesh m = make_box(Eigen::Vector3d(0, 0, 0), 1.0);
    const Mesh small = make_box(Eigen::Vector3d(1.5, 0, 3.0), 0.4);
    const int base = static_cast<int>(m.vertices.size());
    for (const Eigen::Vector3d& v : small.vertices) {
        m.vertices.push_back(v);
    }
    for (const Eigen::Vector3i& fc : small.faces) {
        m.faces.push_back(Eigen::Vector3i(fc[0] + base, fc[1] + base, fc[2] + base));
    }
    std::vector<TexView> views;
    for (const Eigen::Vector3d& eye : {Eigen::Vector3d(3, 1, 3), Eigen::Vector3d(-3, -1, 3),
                                       Eigen::Vector3d(0, 3, -3)}) {
        TexView v = look_at(eye, Eigen::Vector3d(0, 0, 0), W, H, 300.0);
        v.image = &blank;
        views.push_back(v);
    }
    FaceViewOptions options;
    options.zbuffer_downscale = 2;
    options.min_view_cos = -0.5;
    options.smoothing_passes = 8;

    omp_set_num_threads(1);
    const FaceViews one = select_face_views(m, views, options);
    omp_set_num_threads(4);
    const FaceViews four = select_face_views(m, views, options);
    omp_set_num_threads(omp_get_max_threads());

    EXPECT_EQ(one.best, four.best);
    EXPECT_EQ(one.visible, four.visible);
    EXPECT_EQ(one.best, select_face_views(m, views, options).best);
#else
    GTEST_SKIP() << "built without OpenMP";
#endif
}

// ---------------------------------------------------------------------------
// Validation: null image, image/calibration size mismatch, bad options.
// ---------------------------------------------------------------------------
TEST(FaceViews, InvalidInputsThrow) {
    const int W = 64;
    const int H = 64;
    Image blank(W, H, 3);
    const Mesh box = make_box(Eigen::Vector3d(0, 0, 0), 1.0);
    TexView good = look_at(Eigen::Vector3d(0, 0, -5), Eigen::Vector3d(0, 0, 0), W, H, 100.0);
    good.image = &blank;
    const FaceViewOptions defaults;

    // Null image.
    TexView no_image = good;
    no_image.image = nullptr;
    EXPECT_THROW(select_face_views(box, {no_image}, defaults), ValueError);

    // Image size != calibration size.
    Image wrong(W + 8, H, 3);
    TexView mismatched = good;
    mismatched.image = &wrong;
    EXPECT_THROW(select_face_views(box, {mismatched}, defaults), ValueError);

    // Options.
    FaceViewOptions bad;
    bad.min_view_cos = 1.0;
    EXPECT_THROW(select_face_views(box, {good}, bad), ValueError);
    bad = FaceViewOptions{};
    bad.min_view_cos = -1.5;
    EXPECT_THROW(select_face_views(box, {good}, bad), ValueError);
    bad = FaceViewOptions{};
    bad.smoothing_passes = -1;
    EXPECT_THROW(select_face_views(box, {good}, bad), ValueError);
    bad = FaceViewOptions{};
    bad.smoothness = -0.5;
    EXPECT_THROW(select_face_views(box, {good}, bad), ValueError);
    bad = FaceViewOptions{};
    bad.zbuffer_downscale = 0;
    EXPECT_THROW(select_face_views(box, {good}, bad), ValueError);
    // Not a power of two: scaled_calibration rejects it.
    bad = FaceViewOptions{};
    bad.zbuffer_downscale = 3;
    EXPECT_THROW(select_face_views(box, {good}, bad), ValueError);
}

// ---------------------------------------------------------------------------
// A 50k-face sphere with 8 views: a smoke/perf test. Prints the runtime.
// ---------------------------------------------------------------------------
TEST(FaceViews, SphereFiftyThousandFacesEightsViews) {
    const int W = 640;
    const int H = 480;
    Image blank(W, H, 3);

    // stacks x slices quads x 2 triangles ~= 50k: 160 * 160 * 2 = 51200.
    Mesh sphere = make_sphere(160, 160, 1.0);
    ASSERT_GT(sphere.faces.size(), 40000u);

    std::vector<TexView> views;
    for (int k = 0; k < 8; ++k) {
        const double a = 2.0 * M_PI * static_cast<double>(k) / 8.0;
        const Eigen::Vector3d eye(4.0 * std::cos(a), 1.5, 4.0 * std::sin(a));
        TexView v = look_at(eye, Eigen::Vector3d(0, 0, 0), W, H, 500.0);
        v.image = &blank;
        views.push_back(v);
    }

    FaceViewOptions options;
    options.zbuffer_downscale = 2;
    options.min_view_cos = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    const FaceViews fv = select_face_views(sphere, views, options);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("FaceViews perf: %zu faces x %zu views at %dx%d -> %.0f ms (downscale %d)\n",
                sphere.faces.size(), views.size(), W, H, ms, options.zbuffer_downscale);

    // A per-face occlusion buffer is a discrete approximation: faces lying
    // exactly on a silhouette can project to a sliver that falls between the
    // buffer's sample pixels, so a closed convex sphere loses about a fifth of
    // its faces to back-facing or silhouette cases rather than exactly half.
    // The test only requires that the great majority resolve to a camera.
    std::size_t covered = 0;
    for (int b : fv.best) {
        if (b >= 0) {
            ++covered;
        }
    }
    EXPECT_GT(covered, sphere.faces.size() * 3 / 4);
    EXPECT_LT(ms, 30000.0);
}

}  // namespace
}  // namespace texturing
}  // namespace alphashape