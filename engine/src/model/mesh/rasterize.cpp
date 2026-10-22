// Z-buffer rasterisation of the mesh into a camera (texturing milestone).
//
// Design notes
// ------------
// * Triangles are projected with the FULL calibration (distortion included) and
//   scan-converted in image space. The 3D point is reconstructed with
//   perspective-correct screen barycentrics (barycentric / z_cam, renormalised),
//   so the stored value is the exact distance from the camera centre to the
//   surface, which equals the distance along the pixel's unit bearing -- matching
//   DepthBuffer's documented semantics and the dense-stereo DepthMap.
// * A vertex at or behind the near plane (z <= 1e-6) drops the whole triangle:
//   there is no near-plane clipping. Photogrammetry meshes sit in front of their
//   cameras, and a caller that needs clipped geometry can insert one.
// * Fill rule: the standard top-left rule applied to the edge functions after
//   normalising their sign, so a shared edge is covered exactly once regardless
//   of winding. Both windings are accepted (no backface culling here; view
//   selection does that by the face normal).
// * Depth test is `candidate < stored`, so ties keep the LOWER face id.
// * Determinism: faces are projected in id order once, and every image band
//   consumes that list in id order, so N threads reproduce the serial image bit
//   for bit. Bands partition the rows, so the peak memory is one buffer.

#include "model/mesh/rasterize.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#if defined(ALPHASHAPE_HAVE_OPENMP)
#include <omp.h>
#endif

namespace alphashape {
namespace {

/// Vertices at or behind this camera-frame Z are rejected (no clipping).
constexpr double kNearZ = 1e-6;

using PackedParams = std::array<double, kCalibrationParamCount>;

PackedParams pack_calibration(const Calibration& c) {
    return PackedParams{c.f, c.cx, c.cy, c.b1, c.b2, c.k1, c.k2, c.k3, c.k4, c.p1, c.p2};
}

/// A triangle that survived projection, with everything the scan converter
/// needs precomputed: pixel coordinates, camera-frame vertices, the reciprocal
/// of each camera-frame Z, the edge-function gradients and the top-left flags.
struct ProjectedFace {
    std::int32_t face_id = -1;
    Eigen::Vector2d p[3];                 ///< pixel coordinates (u, v)
    Eigen::Vector3d xc[3];                ///< camera-frame vertices
    double inv_z[3] = {0.0, 0.0, 0.0};    ///< 1 / xc[k].z()
    double area2 = 0.0;                   ///< positive doubled screen area
    /// Edge function w_i(P) = grad[i][0] * P.x + grad[i][1] * P.y + bias[i],
    /// positive strictly inside; i is the vertex opposite the edge.
    double grad[3][2] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    double bias[3] = {0.0, 0.0, 0.0};
    bool top_left[3] = {false, false, false};
};

/// Project one face. Returns false when it is partly behind the near plane,
/// when the camera model refuses a vertex, or when it is degenerate in image
/// space.
bool prepare_face(const Mesh& mesh, std::int32_t face_id, const Calibration& cal,
                  const PackedParams& params, const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                  ProjectedFace& out) {
    const Eigen::Vector3i& idx = mesh.faces[static_cast<std::size_t>(face_id)];
    for (int k = 0; k < 3; ++k) {
        const Eigen::Vector3d& world = mesh.vertices[static_cast<std::size_t>(idx[k])];
        const Eigen::Vector3d xc = R * world + t;
        if (!(xc.z() > kNearZ)) {
            return false;  // at or behind the near plane: no clipping, drop it
        }
        double pixel[2] = {0.0, 0.0};
        if (!try_project_point<double>(cal.type, cal.width, cal.height, params.data(), xc.data(),
                                       pixel)) {
            return false;
        }
        out.xc[k] = xc;
        out.p[k] = Eigen::Vector2d(pixel[0], pixel[1]);
        out.inv_z[k] = 1.0 / xc.z();
    }

    const double cross = (out.p[1].x() - out.p[0].x()) * (out.p[2].y() - out.p[0].y()) -
                         (out.p[1].y() - out.p[0].y()) * (out.p[2].x() - out.p[0].x());
    if (cross == 0.0) {
        return false;  // degenerate in image space
    }
    const double s = cross > 0.0 ? 1.0 : -1.0;
    out.area2 = s * cross;

    for (int i = 0; i < 3; ++i) {
        const Eigen::Vector2d& a = out.p[(i + 1) % 3];
        const Eigen::Vector2d& b = out.p[(i + 2) % 3];
        // w_i(P) = s * E(a, b, P), so it is positive strictly inside.
        const double dx = b.x() - a.x();
        const double dy = b.y() - a.y();
        out.grad[i][0] = -s * dy;
        out.grad[i][1] = s * dx;
        out.bias[i] = s * (dy * a.x() - dx * a.y());
        // Top-left rule, evaluated on the sign-normalised directed edge: the
        // interior lies on the w_i > 0 side, so a horizontal edge with dx > 0
        // has the interior below it (a "top" edge) and any downward-going edge
        // (dy < 0) is a "left" edge.
        const double edx = s * dx;
        const double edy = s * dy;
        out.top_left[i] = (edy == 0.0 && edx > 0.0) || edy < 0.0;
    }
    out.face_id = face_id;
    return true;
}

/// Scan-convert one prepared triangle into rows [row0, row1) of a buffer whose
/// row `row0` is its local row 0. Samples sit at pixel centres.
void rasterize_face(const ProjectedFace& pf, int width, int row0, int row1,
                    std::vector<float>& depth, std::vector<std::int32_t>& face) {
    double minx = pf.p[0].x();
    double maxx = minx;
    double miny = pf.p[0].y();
    double maxy = miny;
    for (int k = 1; k < 3; ++k) {
        minx = std::min(minx, pf.p[k].x());
        maxx = std::max(maxx, pf.p[k].x());
        miny = std::min(miny, pf.p[k].y());
        maxy = std::max(maxy, pf.p[k].y());
    }
    const int ix0 = std::max(0, static_cast<int>(std::ceil(minx - 0.5)));
    const int ix1 = std::min(width - 1, static_cast<int>(std::floor(maxx - 0.5)));
    const int iy0 = std::max(row0, static_cast<int>(std::ceil(miny - 0.5)));
    const int iy1 = std::min(row1 - 1, static_cast<int>(std::floor(maxy - 0.5)));
    if (ix0 > ix1 || iy0 > iy1) {
        return;
    }

    // Edge functions are tested after division by the (positive) doubled area,
    // so the coverage test happens in dimensionless barycentric space. The
    // tolerance absorbs floating-point rounding of the edge equation: without
    // it a shared edge, whose exact value is zero but whose computed value is
    // ~1e-15, would be rejected by both of the faces that share it and leave a
    // one-pixel crack. Since the two faces traverse the shared edge in opposite
    // directions, exactly one of them classifies it as a top-left edge, so a
    // near-edge pixel is still covered exactly once.
    constexpr double kEdgeEps = 1e-9;
    const double area2 = pf.area2;
    for (int iy = iy0; iy <= iy1; ++iy) {
        const double py = static_cast<double>(iy) + 0.5;
        const std::size_t row =
            static_cast<std::size_t>(iy - row0) * static_cast<std::size_t>(width);
        for (int ix = ix0; ix <= ix1; ++ix) {
            const double px = static_cast<double>(ix) + 0.5;
            double bary[3];
            bool covered = true;
            for (int i = 0; i < 3; ++i) {
                const double b =
                    (pf.grad[i][0] * px + pf.grad[i][1] * py + pf.bias[i]) / area2;
                bary[i] = b;
                if (!(b > kEdgeEps || (b >= -kEdgeEps && pf.top_left[i]))) {
                    covered = false;
                    break;
                }
            }
            if (!covered) {
                continue;
            }
            const double l0 = bary[0];
            const double l1 = bary[1];
            const double l2 = bary[2];
            // Perspective-correct: screen barycentrics scaled by 1/z, renormalised.
            const double inv_z = l0 * pf.inv_z[0] + l1 * pf.inv_z[1] + l2 * pf.inv_z[2];
            const double c0 = l0 * pf.inv_z[0] / inv_z;
            const double c1 = l1 * pf.inv_z[1] / inv_z;
            const double c2 = l2 * pf.inv_z[2] / inv_z;
            const Eigen::Vector3d point = c0 * pf.xc[0] + c1 * pf.xc[1] + c2 * pf.xc[2];
            const float d = static_cast<float>(point.norm());
            const std::size_t at = row + static_cast<std::size_t>(ix);
            if (d < depth[at]) {
                depth[at] = d;
                face[at] = pf.face_id;
            }
        }
    }
}

}  // namespace

DepthBuffer render_depth(const Mesh& mesh, const Calibration& calibration, const Eigen::Matrix3d& R,
                         const Eigen::Vector3d& t) {
    DepthBuffer out;
    out.width = calibration.width;
    out.height = calibration.height;
    if (calibration.width <= 0 || calibration.height <= 0) {
        return out;
    }
    const std::size_t pixels =
        static_cast<std::size_t>(calibration.width) * static_cast<std::size_t>(calibration.height);
    out.depth.assign(pixels, std::numeric_limits<float>::infinity());
    out.face.assign(pixels, -1);

    const PackedParams params = pack_calibration(calibration);

    // Pass 1: project every face once, in id order, into a compact list. Faces
    // that cannot be rasterised are simply absent, so their ids never reach the
    // buffers.
    std::vector<ProjectedFace> faces;
    faces.reserve(mesh.faces.size());
    for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
        ProjectedFace pf;
        if (prepare_face(mesh, static_cast<std::int32_t>(f), calibration, params, R, t, pf)) {
            faces.push_back(pf);
        }
    }

    // Pass 2: rasterise. Row bands tile the image; each band walks `faces` in
    // id order, so the result is independent of the number of threads.
    const int rows = calibration.height;
    const int width = calibration.width;
    const auto scan = [&](int row0, int row1, std::vector<float>& depth,
                          std::vector<std::int32_t>& face) {
        for (const ProjectedFace& pf : faces) {
            rasterize_face(pf, width, row0, row1, depth, face);
        }
    };
#if defined(ALPHASHAPE_HAVE_OPENMP)
#pragma omp parallel
    {
        const int threads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        if (threads <= 1) {
            scan(0, rows, out.depth, out.face);
        } else {
            const int per = (rows + threads - 1) / threads;
            const int row0 = tid * per;
            if (row0 < rows) {
                const int row1 = std::min(rows, row0 + per);
                const std::size_t base = static_cast<std::size_t>(row0) * static_cast<std::size_t>(width);
                const std::size_t count =
                    static_cast<std::size_t>(row1 - row0) * static_cast<std::size_t>(width);
                std::vector<float> band_depth(out.depth.begin() + static_cast<std::ptrdiff_t>(base),
                                              out.depth.begin() + static_cast<std::ptrdiff_t>(base + count));
                std::vector<std::int32_t> band_face(
                    out.face.begin() + static_cast<std::ptrdiff_t>(base),
                    out.face.begin() + static_cast<std::ptrdiff_t>(base + count));
                scan(row0, row1, band_depth, band_face);
                std::copy(band_depth.begin(), band_depth.end(),
                          out.depth.begin() + static_cast<std::ptrdiff_t>(base));
                std::copy(band_face.begin(), band_face.end(),
                          out.face.begin() + static_cast<std::ptrdiff_t>(base));
            }
        }
    }
#else
    scan(0, rows, out.depth, out.face);
#endif
    return out;
}

}  // namespace alphashape
