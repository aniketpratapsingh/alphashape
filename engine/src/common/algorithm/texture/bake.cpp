// Texture baking implementation. See bake.hpp for the interface contract.
//
// Pipeline (Waechter et al. 2014 family, simplified to a single texture page
// with global multiplicative per-view gains):
//
//   1. validate  - mesh/uv/face-view/view-list consistency, else ValueError
//   2. texel map - which face covers each texel, plus its uv barycentrics
//   3. gains     - per-view per-channel multiplicative colour gains, derived
//                  from mesh edges joining two faces assigned to different
//                  best views (computed BEFORE the colour pass)
//   4. colour    - sample the chosen view(s) per texel (mosaic / average)
//   5. holes     - fill in-chart texels no view covered, from their neighbours
//   6. dilation  - grow chart colour outward into the page gutter
//
// Per-texel colouring is the parallel loop (independent per texel); anything
// touching shared state (the texel map, the gain solve, the Jacobi fills) stays
// serial, so the output is a pure function of the inputs and bit-identical
// across thread counts.

#include "common/algorithm/texture/bake.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "base/utils/error.hpp"
#include "model/camera/calibration.hpp"

namespace alphashape::texturing {
namespace {

// Barycentric acceptance tolerance for the texel-inside-triangle test, matched
// to the rasteriser's fill rule so the baker's chart footprint lines up with
// the one the packer padded.
constexpr double kBaryEps = 1e-6;
// A sample at or behind the camera plane is unusable.
constexpr double kDepthEps = 1e-9;
// Channels at or below this in BOTH views are skipped in the gain solve, so the
// log of sensor noise in a near-black texel cannot drive the fit.
constexpr double kDarkGate = 8.0;
constexpr double kGainMin = 0.5;
constexpr double kGainMax = 2.0;
/// Huber threshold for the gain equations, in log units (~10% disagreement).
constexpr double kGainHuber = 0.10;
/// Reweighting sweeps for the gain fit. Fixed, so the result does not depend on
/// a convergence test.
constexpr int kGainIterations = 8;
// Below this camera-to-surface distance the inverse-square average weight blows
// up; such a degenerate sample is dropped instead.
constexpr double kMinViewDistance = 1e-6;

/// Which face covers a texel, and the uv barycentrics of the texel centre in
/// that face. face == -1 means the texel lies in no chart.
struct TexelRef {
    std::int32_t face = -1;
    float b[3] = {0.0F, 0.0F, 0.0F};
};

/// A TexView with its derived geometry hoisted out of the inner loops.
struct PreparedView {
    const Image* image = nullptr;
    Calibration calibration;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Vector3d C = Eigen::Vector3d::Zero();
    double params[alphashape::kCalibrationParamCount] = {};
};

/// Bilinear sample of an 8-bit image. (sx, sy) are pixel coordinates whose
/// origin is the CENTRE of the top-left pixel (0,0); i.e. a projected pixel px
/// is sampled at px - 0.5. One-channel images replicate their value to RGB.
/// Returns false only for an unusable image (empty or an unexpected channel
/// count); the coordinate itself is clamped to the image.
bool sample_image(const Image& img, double sx, double sy, double out[3]) {
    if (img.empty() || img.width() <= 0 || img.height() <= 0) {
        return false;
    }
    const int nc = img.channels();
    if (nc != 1 && nc != 3) {
        return false;
    }
    const int w = img.width();
    const int h = img.height();
    const double maxx = static_cast<double>(w - 1);
    const double maxy = static_cast<double>(h - 1);
    const double cx = std::clamp(sx, 0.0, maxx);
    const double cy = std::clamp(sy, 0.0, maxy);
    const int x0 = std::min(static_cast<int>(std::floor(cx)), w - 1);
    const int y0 = std::min(static_cast<int>(std::floor(cy)), h - 1);
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const double fx = cx - static_cast<double>(x0);
    const double fy = cy - static_cast<double>(y0);
    for (int c = 0; c < 3; ++c) {
        const int sc = (nc == 1) ? 0 : c;
        const double v00 = static_cast<double>(img.at(x0, y0, sc));
        const double v10 = static_cast<double>(img.at(x1, y0, sc));
        const double v01 = static_cast<double>(img.at(x0, y1, sc));
        const double v11 = static_cast<double>(img.at(x1, y1, sc));
        const double top = v00 + (v10 - v00) * fx;
        const double bot = v01 + (v11 - v01) * fx;
        out[c] = top + (bot - top) * fy;
    }
    return true;
}

/// Project a world-space surface point into `view` and sample its photograph.
/// Returns false when the point is behind the camera, projects off the sensor,
/// or the image itself is unusable — i.e. when there is simply no sample.
bool sample_prepared(const PreparedView& view, const Eigen::Vector3d& X, double out[3]) {
    const Eigen::Vector3d p_cam = view.R * X + view.t;
    if (p_cam.z() <= kDepthEps) {
        return false;
    }
    double pixel[2] = {0.0, 0.0};
    if (!try_project_point<double>(view.calibration.type, view.calibration.width,
                                   view.calibration.height, view.params, p_cam.data(),
                                   pixel)) {
        return false;
    }
    const double maxx = static_cast<double>(view.calibration.width) - 1.0;
    const double maxy = static_cast<double>(view.calibration.height) - 1.0;
    if (pixel[0] < 0.0 || pixel[0] > maxx || pixel[1] < 0.0 || pixel[1] > maxy) {
        return false;
    }
    if (view.image == nullptr) {
        return false;
    }
    return sample_image(*view.image, pixel[0] - 0.5, pixel[1] - 0.5, out);
}

/// Outward unit normal of face `f` (faces are CCW seen from outside).
Eigen::Vector3d face_normal(const Mesh& mesh, int f) {
    const Eigen::Vector3i& tri = mesh.faces[static_cast<std::size_t>(f)];
    const Eigen::Vector3d& a = mesh.vertices[static_cast<std::size_t>(tri[0])];
    const Eigen::Vector3d& b = mesh.vertices[static_cast<std::size_t>(tri[1])];
    const Eigen::Vector3d& c = mesh.vertices[static_cast<std::size_t>(tri[2])];
    Eigen::Vector3d n = (b - a).cross(c - a);
    const double len = n.norm();
    if (len > 1e-15) {
        n /= len;
    }
    return n;
}

std::vector<PreparedView> prepare_views(const std::vector<TexView>& views) {
    std::vector<PreparedView> out;
    out.reserve(views.size());
    for (const TexView& v : views) {
        PreparedView p;
        p.image = v.image;
        p.calibration = v.calibration;
        p.R = v.R;
        p.t = v.t;
        p.C = -v.R.transpose() * v.t;  // camera centre: C = -R^T t
        p.params[0] = v.calibration.f;
        p.params[1] = v.calibration.cx;
        p.params[2] = v.calibration.cy;
        p.params[3] = v.calibration.b1;
        p.params[4] = v.calibration.b2;
        p.params[5] = v.calibration.k1;
        p.params[6] = v.calibration.k2;
        p.params[7] = v.calibration.k3;
        p.params[8] = v.calibration.k4;
        p.params[9] = v.calibration.p1;
        p.params[10] = v.calibration.p2;
        p.params[11] = v.calibration.p3;
        p.params[12] = v.calibration.p4;
        out.push_back(p);
    }
    return out;
}

}  // namespace

BlendingMode parse_blending_mode(const std::string& text) {
    if (text == "mosaic") {
        return BlendingMode::Mosaic;
    }
    if (text == "average") {
        return BlendingMode::Average;
    }
    if (text == "disabled") {
        return BlendingMode::Disabled;
    }
    throw ValueError("parse_blending_mode: unknown blending mode '" + text + "'");
}

std::string to_string(BlendingMode mode) {
    switch (mode) {
        case BlendingMode::Mosaic:
            return "mosaic";
        case BlendingMode::Average:
            return "average";
        case BlendingMode::Disabled:
            return "disabled";
    }
    throw ValueError("to_string: unknown blending mode");
}

namespace {

void validate_inputs(const Mesh& mesh, const std::vector<TexView>& views,
                     const FaceViews& face_views, int texture_size,
                     const BakeOptions& options) {
    const std::size_t nf = mesh.faces.size();
    if (mesh.uv_faces.size() != nf) {
        throw ValueError("bake_texture: uv_faces size does not match faces");
    }
    for (std::size_t f = 0; f < nf; ++f) {
        const Eigen::Vector3i& uv = mesh.uv_faces[f];
        for (int k = 0; k < 3; ++k) {
            const int idx = uv[k];
            if (idx < 0 || static_cast<std::size_t>(idx) >= mesh.uvs.size()) {
                throw ValueError("bake_texture: uv index out of range");
            }
        }
    }
    if (texture_size < 16) {
        throw ValueError("bake_texture: texture_size must be at least 16");
    }
    if (face_views.best.size() != nf || face_views.visible.size() != nf) {
        throw ValueError("bake_texture: face_views size does not match faces");
    }
    for (std::size_t f = 0; f < nf; ++f) {
        for (int vis : face_views.visible[f]) {
            if (vis < 0 || static_cast<std::size_t>(vis) >= views.size()) {
                throw ValueError("bake_texture: face_views.visible references an unknown view");
            }
        }
        const int b = face_views.best[f];
        if (b < -1 || (b >= 0 && static_cast<std::size_t>(b) >= views.size())) {
            throw ValueError("bake_texture: face_views.best references an unknown view");
        }
    }
    for (const TexView& v : views) {
        if (v.image == nullptr) {
            throw ValueError("bake_texture: a view has a null image");
        }
        if (v.image->width() != v.calibration.width || v.image->height() != v.calibration.height) {
            throw ValueError("bake_texture: a view image size does not match its calibration");
        }
    }
    if (options.dilation_px < 0) {
        throw ValueError("bake_texture: dilation_px must not be negative");
    }
}

/// Stage 2: rasterise each face's uv triangle into the texel grid. Serial and
/// in face-id order, so the map is deterministic and the lowest-id face wins a
/// shared texel. Texel (i,j) centre lives at ((i+0.5)/S, (j+0.5)/S) with v down.
std::vector<TexelRef> build_texel_map(const Mesh& mesh, int size) {
    std::vector<TexelRef> map(static_cast<std::size_t>(size) * static_cast<std::size_t>(size));
    const double s = static_cast<double>(size);
    for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
        const Eigen::Vector3i& tri = mesh.uv_faces[f];
        const Eigen::Vector2d u0(mesh.uvs[static_cast<std::size_t>(tri[0])].x(),
                                 mesh.uvs[static_cast<std::size_t>(tri[0])].y());
        const Eigen::Vector2d u1(mesh.uvs[static_cast<std::size_t>(tri[1])].x(),
                                 mesh.uvs[static_cast<std::size_t>(tri[1])].y());
        const Eigen::Vector2d u2(mesh.uvs[static_cast<std::size_t>(tri[2])].x(),
                                 mesh.uvs[static_cast<std::size_t>(tri[2])].y());
        const double area = (u1.x() - u0.x()) * (u2.y() - u0.y()) -
                            (u2.x() - u0.x()) * (u1.y() - u0.y());
        if (std::abs(area) < 1e-20) {
            continue;
        }
        const double minu = std::min({u0.x(), u1.x(), u2.x()});
        const double maxu = std::max({u0.x(), u1.x(), u2.x()});
        const double minv = std::min({u0.y(), u1.y(), u2.y()});
        const double maxv = std::max({u0.y(), u1.y(), u2.y()});
        int i0 = static_cast<int>(std::floor(minu * s - 0.5));
        int i1 = static_cast<int>(std::ceil(maxu * s - 0.5));
        int j0 = static_cast<int>(std::floor(minv * s - 0.5));
        int j1 = static_cast<int>(std::ceil(maxv * s - 0.5));
        i0 = std::max(i0, 0);
        j0 = std::max(j0, 0);
        i1 = std::min(i1, size - 1);
        j1 = std::min(j1, size - 1);
        for (int j = j0; j <= j1; ++j) {
            for (int i = i0; i <= i1; ++i) {
                const std::size_t ti = static_cast<std::size_t>(j) * static_cast<std::size_t>(size) +
                                       static_cast<std::size_t>(i);
                if (map[ti].face != -1) {
                    continue;  // an earlier (lower-id) face already claimed this texel
                }
                const double px = (static_cast<double>(i) + 0.5) / s;
                const double py = (static_cast<double>(j) + 0.5) / s;
                const double w0 = ((u1.x() - px) * (u2.y() - py) - (u2.x() - px) * (u1.y() - py)) / area;
                const double w1 = ((u2.x() - px) * (u0.y() - py) - (u0.x() - px) * (u2.y() - py)) / area;
                const double w2 = 1.0 - w0 - w1;
                if (w0 < -kBaryEps || w1 < -kBaryEps || w2 < -kBaryEps) {
                    continue;
                }
                map[ti].face = static_cast<std::int32_t>(f);
                map[ti].b[0] = static_cast<float>(w0);
                map[ti].b[1] = static_cast<float>(w1);
                map[ti].b[2] = static_cast<float>(w2);
            }
        }
    }
    return map;
}

/// Stage 4 (runs before stage 3): per-view, per-channel multiplicative gains.
///
/// For every mesh edge shared by two faces whose best views differ (a != b,
/// both real), the two views are sampled at the edge midpoint and at the 1/4
/// and 3/4 points. Each channel where both samples clear kDarkGate contributes
/// an equation  gamma_a - gamma_b = log(c_b) - log(c_a)  in the log-gains.
/// We solve min sum(w eq^2) + mu sum(gamma^2) with a small Tikhonov weight mu
/// that keeps the gains near 1 (so a mesh seen by only one view is a no-op).
/// Dense normal equations (views are at most a few hundred) solved per channel
/// with a fixed Eigen LDLT; gains are clamped to [0.5, 2].
///
/// The weights make the fit robust. Most equations say "these two views agree
/// to a percent", but a minority are dominated by something other than
/// exposure -- a sample that landed on a silhouette, a grazing view, or the
/// parallax of the fused surface against the photograph. With unit weights
/// those pull the gains ~3% off 1.0 on a scene that has NO exposure differences
/// at all (the seq fixture is rendered from one albedo field), and the mosaic
/// page then disagrees with the photographs measurably more than the
/// un-corrected page does. Iteratively reweighted least squares with a Huber
/// weight keeps the large-residual equations from steering the fit: on the seq
/// fixture the recovered gains land within a fraction of a percent of 1.0,
/// while a genuinely darkened view (test_bake ColorCorrectionLevelsViews) still
/// moves. kGainHuber is in log units: 0.10 is ~10% disagreement.
std::vector<std::array<double, 3>> solve_gains(const Mesh& mesh,
                                               const std::vector<PreparedView>& views,
                                               const FaceViews& face_views) {
    const std::size_t nv = views.size();
    std::vector<std::array<double, 3>> gains(nv, {1.0, 1.0, 1.0});
    if (nv == 0) {
        return gains;
    }

    // Accumulate the continuity equations per channel, keyed by (a,b).
    struct Eq {
        int a;
        int b;
        double dc[3];
    };
    std::vector<Eq> eqs;
    std::map<std::pair<int, int>, std::array<int, 2>> edge_owner;  // edge -> two faces
    const std::size_t nf = mesh.faces.size();
    for (std::size_t f = 0; f < nf; ++f) {
        const Eigen::Vector3i& tri = mesh.faces[f];
        const int v[3] = {tri[0], tri[1], tri[2]};
        for (int k = 0; k < 3; ++k) {
            const int e0 = v[k];
            const int e1 = v[(k + 1) % 3];
            const std::pair<int, int> key = std::minmax(e0, e1);
            auto it = edge_owner.find(key);
            if (it == edge_owner.end()) {
                edge_owner.emplace(key, std::array<int, 2>{static_cast<int>(f), -1});
            } else if (it->second[1] == -1 && it->second[0] != static_cast<int>(f)) {
                it->second[1] = static_cast<int>(f);
            }
        }
    }

    for (const auto& entry : edge_owner) {
        const int f0 = entry.second[0];
        const int f1 = entry.second[1];
        if (f1 < 0) {
            continue;
        }
        const int a = face_views.best[static_cast<std::size_t>(f0)];
        const int b = face_views.best[static_cast<std::size_t>(f1)];
        if (a < 0 || b < 0 || a == b) {
            continue;
        }
        const Eigen::Vector3d& va = mesh.vertices[static_cast<std::size_t>(entry.first.first)];
        const Eigen::Vector3d& vb = mesh.vertices[static_cast<std::size_t>(entry.first.second)];
        for (double t : {0.25, 0.5, 0.75}) {
            const Eigen::Vector3d X = va + (vb - va) * t;
            double ca[3] = {0.0, 0.0, 0.0};
            double cb[3] = {0.0, 0.0, 0.0};
            if (!sample_prepared(views[static_cast<std::size_t>(a)], X, ca) ||
                !sample_prepared(views[static_cast<std::size_t>(b)], X, cb)) {
                continue;
            }
            Eq eq;
            eq.a = a;
            eq.b = b;
            for (int c = 0; c < 3; ++c) {
                eq.dc[c] = (ca[c] > kDarkGate && cb[c] > kDarkGate)
                               ? std::log(cb[c]) - std::log(ca[c])
                               : 0.0;
            }
            eqs.push_back(eq);
        }
    }

    if (eqs.empty()) {
        return gains;
    }
    const double mu = 0.1 * (static_cast<double>(eqs.size()) / static_cast<double>(nv));
    for (int c = 0; c < 3; ++c) {
        // Iteratively reweighted least squares: solve with unit weights, then
        // reweight each equation by the Huber function of its residual and
        // solve again. Every sweep rebuilds the same normal equations from the
        // same equations in the same order, so the result is deterministic.
        std::vector<double> weight(eqs.size(), 1.0);
        Eigen::VectorXd gamma = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nv));
        for (int sweep = 0; sweep < kGainIterations; ++sweep) {
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nv),
                                                      static_cast<Eigen::Index>(nv));
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nv));
            for (std::size_t e = 0; e < eqs.size(); ++e) {
                const Eq& eq = eqs[e];
                const auto ia = static_cast<Eigen::Index>(eq.a);
                const auto ib = static_cast<Eigen::Index>(eq.b);
                const double w = weight[e];
                A(ia, ia) += w;
                A(ib, ib) += w;
                A(ia, ib) -= w;
                A(ib, ia) -= w;
                rhs(ia) += w * eq.dc[c];
                rhs(ib) -= w * eq.dc[c];
            }
            for (std::size_t v = 0; v < nv; ++v) {
                A(static_cast<Eigen::Index>(v), static_cast<Eigen::Index>(v)) += mu;
            }
            gamma = A.ldlt().solve(rhs);
            if (sweep + 1 == kGainIterations) {
                break;  // the last solve is the answer; no weights needed
            }
            for (std::size_t e = 0; e < eqs.size(); ++e) {
                const Eq& eq = eqs[e];
                const double residual = (gamma[static_cast<Eigen::Index>(eq.a)] -
                                         gamma[static_cast<Eigen::Index>(eq.b)]) -
                                        eq.dc[c];
                const double magnitude = std::abs(residual);
                weight[e] = magnitude <= kGainHuber ? 1.0 : kGainHuber / magnitude;
            }
        }
        for (std::size_t v = 0; v < nv; ++v) {
            const double g = std::exp(gamma[static_cast<Eigen::Index>(v)]);
            gains[v][static_cast<std::size_t>(c)] = std::clamp(g, kGainMin, kGainMax);
        }
    }
    return gains;
}

/// Stage 3: colour every in-chart texel. Independent per texel, so the loop is
/// the parallel one. Writes `colors` (linear 0..255 floats) and `colored` (a
/// texel is coloured once at least one usable sample was combined).
void colour_texels(const Mesh& mesh, const std::vector<PreparedView>& views,
                   const FaceViews& face_views, const std::vector<TexelRef>& map,
                   const std::vector<std::array<double, 3>>& gains, int size,
                   BlendingMode mode, bool color_correction,
                   std::vector<std::array<float, 3>>& colors, std::vector<std::uint8_t>& colored) {
    for (int j = 0; j < size; ++j) {
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < size; ++i) {
            const std::size_t ti = static_cast<std::size_t>(j) * static_cast<std::size_t>(size) +
                                   static_cast<std::size_t>(i);
            const TexelRef& ref = map[ti];
            if (ref.face < 0) {
                continue;
            }
            const int f = ref.face;
            const Eigen::Vector3i& tri = mesh.faces[static_cast<std::size_t>(f)];
            Eigen::Vector3d X = static_cast<double>(ref.b[0]) *
                                    mesh.vertices[static_cast<std::size_t>(tri[0])] +
                                static_cast<double>(ref.b[1]) *
                                    mesh.vertices[static_cast<std::size_t>(tri[1])] +
                                static_cast<double>(ref.b[2]) *
                                    mesh.vertices[static_cast<std::size_t>(tri[2])];

            // Mosaic and Disabled both take each face's single best view with no
            // cross-face blending; Disabled additionally forces the gain step
            // off, making it Mosaic with color_correction = false (see bake.hpp).
            // Only Average combines several views on a texel.
            if (mode != BlendingMode::Average) {
                const int v = face_views.best[static_cast<std::size_t>(f)];
                if (v < 0) {
                    continue;  // no best view: leave the texel as a hole
                }
                double samp[3] = {0.0, 0.0, 0.0};
                if (!sample_prepared(views[static_cast<std::size_t>(v)], X, samp)) {
                    continue;
                }
                const bool correct = color_correction && mode != BlendingMode::Disabled;
                for (int c = 0; c < 3; ++c) {
                    const double g = correct ? gains[static_cast<std::size_t>(v)][static_cast<std::size_t>(c)]
                                             : 1.0;
                    colors[ti][static_cast<std::size_t>(c)] = static_cast<float>(samp[c] * g);
                }
                colored[ti] = 1;
            } else {  // Average
                const std::vector<int>& vis = face_views.visible[static_cast<std::size_t>(f)];
                const Eigen::Vector3d n = face_normal(mesh, f);
                double acc[3] = {0.0, 0.0, 0.0};
                double wsum = 0.0;
                for (int v : vis) {
                    const PreparedView& pv = views[static_cast<std::size_t>(v)];
                    double samp[3] = {0.0, 0.0, 0.0};
                    if (!sample_prepared(pv, X, samp)) {
                        continue;
                    }
                    const Eigen::Vector3d d = pv.C - X;
                    const double dist2 = d.squaredNorm();
                    if (dist2 < kMinViewDistance * kMinViewDistance) {
                        continue;
                    }
                    const Eigen::Vector3d dir = d / std::sqrt(dist2);
                    const double cosang = n.dot(dir);
                    if (cosang <= 0.0) {
                        continue;
                    }
                    const double w = cosang / dist2;
                    wsum += w;
                    for (int c = 0; c < 3; ++c) {
                        const double g =
                            color_correction ? gains[static_cast<std::size_t>(v)][static_cast<std::size_t>(c)] : 1.0;
                        acc[c] += w * samp[c] * g;
                    }
                }
                if (wsum > 1e-12) {
                    const double inv = 1.0 / wsum;
                    for (int c = 0; c < 3; ++c) {
                        colors[ti][static_cast<std::size_t>(c)] = static_cast<float>(acc[c] * inv);
                    }
                    colored[ti] = 1;
                }
            }
        }
    }
}

/// In the hole pass the victims are the uncoloured *in-chart* texels; in the
/// dilation pass they are the uncoloured *out-of-chart* ones. This is the test
/// used to decide whether a texel is a victim of the pass in progress.
bool is_victim(bool in_chart, bool victims_are_chart) {
    return in_chart == victims_are_chart;
}

/// One Jacobi pass: for every texel selected by `victims_are_chart`, average the
/// already-coloured 4-neighbours and stage the result. Returns the number of
/// texels that received a value. Deterministic (row-major, reads only the
/// pre-pass state, writes after the whole scan).
std::size_t jacobi_pass(int size, const std::vector<std::uint8_t>& colored,
                        const std::vector<std::array<float, 3>>& colors,
                        const std::vector<TexelRef>& map, bool victims_are_chart,
                        std::vector<std::array<float, 3>>& colors_out,
                        std::vector<std::uint8_t>& colored_out) {
    std::size_t changed = 0;
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            const std::size_t ti = static_cast<std::size_t>(j) * static_cast<std::size_t>(size) +
                                   static_cast<std::size_t>(i);
            if (colored[ti]) {
                continue;
            }
            if (!is_victim(map[ti].face >= 0, victims_are_chart)) {
                continue;
            }
            double acc[3] = {0.0, 0.0, 0.0};
            int cnt = 0;
            const int di[4] = {-1, 1, 0, 0};
            const int dj[4] = {0, 0, -1, 1};
            for (int k = 0; k < 4; ++k) {
                const int ni = i + di[k];
                const int nj = j + dj[k];
                if (ni < 0 || ni >= size || nj < 0 || nj >= size) {
                    continue;
                }
                const std::size_t nidx =
                    static_cast<std::size_t>(nj) * static_cast<std::size_t>(size) +
                    static_cast<std::size_t>(ni);
                if (!colored[nidx]) {
                    continue;
                }
                for (int c = 0; c < 3; ++c) {
                    acc[c] += static_cast<double>(colors[nidx][static_cast<std::size_t>(c)]);
                }
                cnt += 1;
            }
            if (cnt == 0) {
                continue;
            }
            const double inv = 1.0 / static_cast<double>(cnt);
            for (int c = 0; c < 3; ++c) {
                colors_out[ti][static_cast<std::size_t>(c)] = static_cast<float>(acc[c] * inv);
            }
            colored_out[ti] = 1;
            changed += 1;
        }
    }
    return changed;
}

}  // namespace

Image bake_texture(const Mesh& mesh, const std::vector<TexView>& views,
                   const FaceViews& face_views, int texture_size, const BakeOptions& options) {
    validate_inputs(mesh, views, face_views, texture_size, options);

    const std::vector<PreparedView> prepared = prepare_views(views);
    const std::vector<TexelRef> map = build_texel_map(mesh, texture_size);

    // Gains come first: the colour pass consumes them.
    std::vector<std::array<double, 3>> gains(prepared.size(), {1.0, 1.0, 1.0});
    if (options.color_correction && options.blending != BlendingMode::Disabled && !prepared.empty()) {
        gains = solve_gains(mesh, prepared, face_views);
    }

    const std::size_t n = static_cast<std::size_t>(texture_size) *
                          static_cast<std::size_t>(texture_size);
    std::vector<std::array<float, 3>> colors(n, {0.0F, 0.0F, 0.0F});
    std::vector<std::uint8_t> colored(n, 0);

    colour_texels(mesh, prepared, face_views, map, gains, texture_size, options.blending,
                  options.color_correction, colors, colored);

    // Stage 5: fill in-chart holes from already-coloured neighbours (any chart),
    // Jacobi-style until a pass changes nothing. `colored` gates the write, so a
    // texel joined in mid-fill cannot cascade within the same pass.
    if (options.fill_holes) {
        std::vector<std::array<float, 3>> next_colors = colors;
        std::vector<std::uint8_t> next_colored = colored;
        for (int pass = 0; pass < texture_size * texture_size; ++pass) {
            const std::size_t changed = jacobi_pass(texture_size, colored, colors, map, true,
                                                    next_colors, next_colored);
            colors = next_colors;
            colored = next_colored;
            if (changed == 0) {
                break;
            }
        }
    }

    // Stage 6: gutter dilation. Grow outward from chart colour for exactly
    // dilation_px passes (a non-zero count terminates by construction); every
    // texel still empty afterwards stays black.
    if (options.dilation_px > 0) {
        std::vector<std::array<float, 3>> next_colors = colors;
        std::vector<std::uint8_t> next_colored = colored;
        for (int pass = 0; pass < options.dilation_px; ++pass) {
            jacobi_pass(texture_size, colored, colors, map, false, next_colors, next_colored);
            colors = next_colors;
            colored = next_colored;
        }
    }

    Image out(texture_size, texture_size, 3);
    for (std::size_t ti = 0; ti < n; ++ti) {
        const auto x = static_cast<int>(ti % static_cast<std::size_t>(texture_size));
        const auto y = static_cast<int>(ti / static_cast<std::size_t>(texture_size));
        std::uint8_t rgb[3];
        for (int c = 0; c < 3; ++c) {
            const double v = std::clamp(static_cast<double>(colors[ti][static_cast<std::size_t>(c)]),
                                        0.0, 255.0);
            rgb[c] = static_cast<std::uint8_t>(std::lround(v));
        }
        out.at(x, y, 0) = rgb[0];
        out.at(x, y, 1) = rgb[1];
        out.at(x, y, 2) = rgb[2];
    }
    return out;
}

}  // namespace alphashape::texturing
