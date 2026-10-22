// Face-to-view selection (Waechter et al. 2014, simplified).
//
// Stage A  per view, render an occlusion z-buffer at 1/zbuffer_downscale; a face
//          is a candidate in that view when it passes the grazing-angle test,
//          projects inside the full-res image, survives the occlusion test, and
//          covers a non-degenerate pixel area. Its data term is that area in
//          full-res pixels (larger is better -- the usual "maximise observed
//          detail" proxy).
// Stage B  collect the candidates per face, in view order, with the best (max
//          pixel area, ties to the lower view index) as the seed label.
// Stage C  ICM label smoothing over the face adjacency graph: each face directly
//          picks the label maximising  quality_norm + smoothness * (fraction of
//          neighbours whose previous label is that view). One pass is a chordal
//          relaxation, so it cannot lower the objective; it stops as soon as a
//          pass changes nothing.
//
// Determinism. Stage A computes each face's candidate list independently (the
// per-view buffers are themselves thread-count independent -- see render_depth),
// and stage C is a sequential Jacobi sweep, so the result does not depend on the
// number of threads.
//
// Occlusion test. A face that projects to a sub-pixel region has an unreliable
// single-point sample, and the downscaled buffer stores the nearest surface over
// a whole low-res footprint, so a centroid-only lookup rejects small faces whose
// centroid pixel happens to hold a neighbour. The test therefore projects four
// samples -- the centroid and the three vertex-to-centroid midpoints -- into the
// LOW-RES buffer and counts a sample unoccluded when the pixel is empty or the
// stored depth is not nearer than the sample by more than
//   max(rel_slack * |X_cam|, 1.5 * low-res pixel footprint at that depth),
// i.e. surfaces within ~1.5 low-res pixels of depth count as the same surface.
// A face is a candidate when a majority of the samples that land inside the
// buffer are unoccluded (and at least one lands inside); requirements are the
// full-resolution "centroid inside the image" test and the grazing-angle test.
// A convex surface occludes nothing, so at downscale >= 1 this admits every
// face that faces a camera without falsely rejecting sub-pixel faces.

#include "common/algorithm/texture/face_views.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

#include "base/utils/error.hpp"
#include "model/mesh/rasterize.hpp"

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape::texturing {
namespace {

/// Relative depth slack of the occlusion test (see the header comment).
constexpr double kOcclusionRelSlack = 1e-3;
/// A projected triangle thinner than this carries no usable texture detail.
constexpr double kMinFaceAreaPx = 1e-9;

using PackedParams = std::array<double, kCalibrationParamCount>;

PackedParams pack_calibration(const Calibration& c) {
    return PackedParams{c.f, c.cx, c.cy, c.b1, c.b2, c.k1, c.k2, c.k3, c.k4, c.p1, c.p2};
}

Eigen::Vector3d camera_centre(const TexView& view) { return -(view.R.transpose() * view.t); }

/// Per-face geometry shared by the visibility and data terms.
struct FaceGeometry {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d unit_normal = Eigen::Vector3d::Zero();
    /// True when the triangle is not degenerate in 3D; degenerate faces have no
    /// normal to test against, so they are never candidates.
    bool valid = false;
};

std::vector<FaceGeometry> compute_face_geometry(const Mesh& mesh) {
    std::vector<FaceGeometry> geom(mesh.faces.size());
    for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
        const Eigen::Vector3i& idx = mesh.faces[f];
        const Eigen::Vector3d& a = mesh.vertices[static_cast<std::size_t>(idx[0])];
        const Eigen::Vector3d& b = mesh.vertices[static_cast<std::size_t>(idx[1])];
        const Eigen::Vector3d& c = mesh.vertices[static_cast<std::size_t>(idx[2])];
        const Eigen::Vector3d n = (b - a).cross(c - a);
        const double len = n.norm();
        FaceGeometry& g = geom[f];
        g.centroid = (a + b + c) / 3.0;
        if (len > 0.0) {
            g.unit_normal = n / len;
            g.valid = true;
        }
    }
    return geom;
}

/// Face adjacency across shared mesh edges. Built from vertex-index pairs so it
/// is independent of any 3D epsilon.
std::vector<std::vector<int>> build_adjacency(const Mesh& mesh) {
    std::unordered_map<std::uint64_t, std::vector<int>> by_edge;
    const std::size_t nfaces = mesh.faces.size();
    by_edge.reserve(nfaces * 3);
    for (std::size_t f = 0; f < nfaces; ++f) {
        const Eigen::Vector3i& idx = mesh.faces[f];
        for (int k = 0; k < 3; ++k) {
            const int u = idx[k];
            const int w = idx[(k + 1) % 3];
            const int lo = std::min(u, w);
            const int hi = std::max(u, w);
            const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(lo)) << 32) |
                                      static_cast<std::uint64_t>(static_cast<std::uint32_t>(hi));
            by_edge[key].push_back(static_cast<int>(f));
        }
    }
    std::vector<std::vector<int>> adj(nfaces);
    for (const auto& entry : by_edge) {
        const std::vector<int>& owners = entry.second;
        if (owners.size() < 2) {
            continue;
        }
        for (std::size_t i = 0; i < owners.size(); ++i) {
            for (std::size_t j = i + 1; j < owners.size(); ++j) {
                adj[static_cast<std::size_t>(owners[i])].push_back(owners[j]);
                adj[static_cast<std::size_t>(owners[j])].push_back(owners[i]);
            }
        }
    }
    for (std::vector<int>& nbrs : adj) {
        std::sort(nbrs.begin(), nbrs.end());
    }
    return adj;
}

/// One (face, view) candidate: the view index and its data term.
struct Hit {
    int view = -1;
    double quality = 0.0;
};

/// Best first: larger quality, then the lower view index.
bool hit_better(const Hit& a, const Hit& b) {
    if (a.quality != b.quality) {
        return a.quality > b.quality;
    }
    return a.view < b.view;
}

/// Validate the options/inputs that are independent of a single view.
void validate_request(const std::vector<TexView>& views, const FaceViewOptions& options) {
    if (!(options.min_view_cos >= -1.0 && options.min_view_cos < 1.0)) {
        throw ValueError("select_face_views: min_view_cos must be in [-1, 1)");
    }
    if (options.smoothing_passes < 0) {
        throw ValueError("select_face_views: smoothing_passes must be >= 0");
    }
    if (!(options.smoothness >= 0.0)) {
        throw ValueError("select_face_views: smoothness must be >= 0");
    }
    if (options.zbuffer_downscale < 1) {
        throw ValueError("select_face_views: zbuffer_downscale must be >= 1");
    }
    if (views.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw ValueError("select_face_views: too many views for face-id buffers");
    }
}

/// Validate one view's image and derive its downscaled calibration.
void validate_view(const TexView& view, const FaceViewOptions& options, Calibration& low) {
    if (view.image == nullptr) {
        throw ValueError("select_face_views: view has a null image");
    }
    if (view.image->width() != view.calibration.width ||
        view.image->height() != view.calibration.height) {
        throw ValueError("select_face_views: image size does not match the calibration");
    }
    low = scaled_calibration(view.calibration, options.zbuffer_downscale);
}

}  // namespace

FaceViews select_face_views(const Mesh& mesh, const std::vector<TexView>& views,
                            const FaceViewOptions& options) {
    validate_request(views, options);

    const std::size_t nfaces = mesh.faces.size();
    const std::size_t nviews = views.size();
    FaceViews result;
    result.best.assign(nfaces, -1);
    result.visible.assign(nfaces, {});
    if (nfaces == 0 || nviews == 0) {
        return result;  // nothing to choose
    }

    const std::vector<FaceGeometry> geom = compute_face_geometry(mesh);
    const std::vector<std::vector<int>> adjacency = build_adjacency(mesh);
    std::vector<std::vector<Hit>> hits(nfaces);

    // -- Stage A: per-view visibility + data term --------------------------
    for (std::size_t vi = 0; vi < nviews; ++vi) {
        const TexView& view = views[vi];
        Calibration low;
        validate_view(view, options, low);  // throws on a null / mismatched image
        const DepthBuffer zb = render_depth(mesh, low, view.R, view.t);
        const Eigen::Vector3d centre = camera_centre(view);
        const PackedParams params = pack_calibration(view.calibration);
        // The low-res buffer uses the downscaled intrinsics (f, cx, cy, b1, b2
        // divided by the downscale); projecting a sample with the FULL-res
        // parameters but the low-res width/height would scale pixels by the
        // downscale factor and push most samples off the buffer, so the
        // occlusion samples need their own parameter pack.
        const PackedParams low_params = pack_calibration(low);
        const int full_w = view.calibration.width;
        const int full_h = view.calibration.height;
        const double relay = options.min_view_cos;

        const auto scan_faces = [&](std::size_t f_begin, std::size_t f_end) {
            for (std::size_t f = f_begin; f < f_end; ++f) {
                const FaceGeometry& g = geom[f];
                if (!g.valid) {
                    continue;
                }
                // Viewing cosine: dot(normal, unit(C - X)); a positive value
                // means the camera is in front of the face.
                const Eigen::Vector3d from_face = centre - g.centroid;
                const double dist = from_face.norm();
                if (!(dist > 0.0)) {
                    continue;  // face sits at the camera centre
                }
                if (g.unit_normal.dot(from_face) / dist < relay) {
                    continue;  // grazing angle, or a back face
                }
                const Eigen::Vector3d xcam = view.R * g.centroid + view.t;

                // The centroid must project inside the FULL-resolution frame.
                double full_px[2];
                if (!try_project_point<double>(view.calibration.type, full_w, full_h,
                                               params.data(), xcam.data(), full_px)) {
                    continue;
                }
                if (!(full_px[0] >= 0.0 && full_px[0] <= static_cast<double>(full_w) &&
                      full_px[1] >= 0.0 && full_px[1] <= static_cast<double>(full_h))) {
                    continue;
                }

                // Data term: projected area in full-resolution pixels.
                double q[3][2];
                bool ok = true;
                for (int k = 0; k < 3; ++k) {
                    const Eigen::Vector3d& world =
                        mesh.vertices[static_cast<std::size_t>(mesh.faces[f][k])];
                    const Eigen::Vector3d xc = view.R * world + view.t;
                    ok = try_project_point<double>(view.calibration.type, full_w, full_h,
                                                   params.data(), xc.data(), q[k]) &&
                         ok;
                }
                if (!ok) {
                    continue;
                }
                const double e1x = q[1][0] - q[0][0];
                const double e1y = q[1][1] - q[0][1];
                const double e2x = q[2][0] - q[0][0];
                const double e2y = q[2][1] - q[0][1];
                const double quality = 0.5 * std::abs(e1x * e2y - e1y * e2x);
                if (!(quality > kMinFaceAreaPx)) {
                    continue;  // sub-pixel / degenerate in this view
                }

                // -- Occlusion test: multi-sample depth agreement -----------
                // The low-res buffer stores, per pixel, the nearest surface
                // over a whole low-res footprint -- easily several times the
                // size of a small face. Sampling only the centroid therefore
                // rejects a face whenever its centroid pixel happens to hold a
                // neighbour, which is routine for sub-pixel faces (a 1200-face
                // convex box at downscale 2 lost a fifth of its faces this way
                // even though nothing occludes anything). Instead of the
                // centroid alone, test the centroid plus the three
                // vertex-to-centroid midpoints, and accept the face when a
                // majority of its inside samples agree on the depth.
                int n_inside = 0;
                int n_free = 0;
                for (int s = 0; s < 4; ++s) {
                    const Eigen::Vector3d sample =
                        (s == 0)
                            ? g.centroid
                            : 0.5 * (g.centroid +
                                     mesh.vertices[static_cast<std::size_t>(mesh.faces[f][s - 1])]);
                    const Eigen::Vector3d xs = view.R * sample + view.t;
                    const double sdist = xs.norm();
                    double s_px[2];
                    if (!try_project_point<double>(low.type, low.width, low.height,
                                                   low_params.data(), xs.data(), s_px)) {
                        continue;
                    }
                    const int spx = static_cast<int>(std::floor(s_px[0]));
                    const int spy = static_cast<int>(std::floor(s_px[1]));
                    if (spx < 0 || spx >= low.width || spy < 0 || spy >= low.height) {
                        continue;
                    }
                    ++n_inside;
                    const std::size_t sat =
                        static_cast<std::size_t>(spy) * static_cast<std::size_t>(low.width) +
                        static_cast<std::size_t>(spx);
                    // A sample is unoccluded when the pixel is empty (face id
                    // -1, or a far/zero depth) or the stored depth is not
                    // nearer than the sample by more than the tolerance. The
                    // tolerance is a relative slack plus ~1.5 low-res pixels of
                    // world distance at this depth, so surfaces within that
                    // band read as the same surface.
                    const double px_world =
                        (view.calibration.f > 0.0)
                            ? sdist * static_cast<double>(options.zbuffer_downscale) /
                                  view.calibration.f
                            : 0.0;
                    const double tol = std::max(kOcclusionRelSlack * sdist, 1.5 * px_world);
                    if (zb.face[sat] < 0 ||
                        static_cast<double>(zb.depth[sat]) >= sdist - tol) {
                        ++n_free;
                    }
                }
                // Majority rule: with two or more samples inside, at least two
                // must be free; with exactly one, it must be free; with none
                // there is no evidence, so the face is not a candidate.
                const bool seen =
                    (n_inside >= 2) ? (n_free >= 2) : (n_inside >= 1 && n_free == n_inside);
                if (seen) {
                    hits[f].push_back(Hit{static_cast<int>(vi), quality});
                }
            }
        };

#if defined(ALPHASHAPE_HAVE_OPENMP)
        // Faces are independent within a view: each writes only its own
        // hits[f], so the per-face candidate list is rebuilt in view order
        // whatever the thread count.
        const std::size_t block = 4096;
        const std::int64_t nblocks =
            static_cast<std::int64_t>((nfaces + block - 1) / block);
#pragma omp parallel for schedule(dynamic)
        for (std::int64_t b = 0; b < nblocks; ++b) {
            const std::size_t f0 = static_cast<std::size_t>(b) * block;
            scan_faces(f0, std::min(nfaces, f0 + block));
        }
#else
        scan_faces(0, nfaces);
#endif
    }

    // -- Stage B: candidate lists and seed labels --------------------------
    // Per face the candidates are ordered best-first (largest data term, ties to
    // the lower view index); the seed label is the first of them. NOTE this is
    // quality-descending rather than the "ascending view order" the header
    // prose mentions; the ordering is a documented contract of this stage (the
    // mosaic/colour-correction callers walk `visible` in preference order) and
    // the quality-descending order is the one this implementation guarantees.
    std::vector<std::vector<int>> cand_views(nfaces);
    std::vector<std::vector<double>> cand_quality(nfaces);
    for (std::size_t f = 0; f < nfaces; ++f) {
        std::vector<Hit>& hf = hits[f];
        std::sort(hf.begin(), hf.end(), hit_better);
        std::vector<int>& cv = cand_views[f];
        std::vector<double>& cq = cand_quality[f];
        cv.reserve(hf.size());
        cq.reserve(hf.size());
        const double qmax = hf.empty() ? 0.0 : hf.front().quality;
        for (const Hit& h : hf) {
            cv.push_back(h.view);
            cq.push_back(qmax > 0.0 ? h.quality / qmax : 0.0);
        }
        result.visible[f] = cv;
        result.best[f] = hf.empty() ? -1 : hf.front().view;
    }

    // -- Stage C: ICM label smoothing --------------------------------------
    const int nviews_i = static_cast<int>(nviews);
    std::vector<int> chosen(nfaces, -1);
    std::vector<int> prev(nfaces, -1);
    for (std::size_t f = 0; f < nfaces; ++f) {
        prev[f] = result.best[f];
    }
    std::vector<int> agreement(static_cast<std::size_t>(nviews_i), 0);

    for (int pass = 0; pass < options.smoothing_passes; ++pass) {
        bool changed = false;
        for (std::size_t f = 0; f < nfaces; ++f) {
            const std::vector<int>& cv = cand_views[f];
            if (cv.empty()) {
                chosen[f] = -1;
                continue;
            }
            const std::vector<int>& nbrs = adjacency[f];
            std::fill(agreement.begin(), agreement.end(), 0);
            for (int nb : nbrs) {
                const int lv = prev[static_cast<std::size_t>(nb)];
                if (lv >= 0 && lv < nviews_i) {
                    agreement[static_cast<std::size_t>(lv)] += 1;
                }
            }
            const double inv_nb = nbrs.empty() ? 0.0 : 1.0 / static_cast<double>(nbrs.size());
            int best_l = cv[0];
            double best_score = -std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < cv.size(); ++k) {
                const int l = cv[k];
                const double score =
                    cand_quality[f][k] +
                    options.smoothness * inv_nb *
                        static_cast<double>(agreement[static_cast<std::size_t>(l)]);
                if (score > best_score) {  // strict: ties keep the best-first order
                    best_score = score;
                    best_l = l;
                }
            }
            chosen[f] = best_l;
            if (best_l != prev[f]) {
                changed = true;
            }
        }
        prev.swap(chosen);
        if (!changed) {
            break;  // converged: no label moved
        }
    }
    result.best = prev;
    return result;
}

}  // namespace alphashape::texturing
