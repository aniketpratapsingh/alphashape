#include "model/mesh/uv_charts.hpp"

// UV chart segmentation + flattening for the texturing milestone.
//
// The pass is self-contained on purpose: it builds its own face-normal, area and
// adjacency tables instead of going through mesh_ops, so no dependency is added
// from `model` onto `core`.
//
//   1. validate the options and the face indices.
//   2. per-face unit normal + area. A zero-area face is "degenerate": it never
//      joins a chart (it has no normal to test) and is emitted as its own chart.
//   3. adjacency across edges shared by exactly two faces. A non-manifold edge
//      (1 or 3+ incident faces) does *not* connect, so growing never walks
//      through a seam.
//   4. seed-and-grow. Seeds are visited in decreasing-area order, ties by lower
//      face id. Growth is a deterministic min-face-id BFS. A face joins when its
//      normal is inside the cone of the chart's running area-weighted mean
//      normal AND inside the cone of the *seed* normal; the second, fixed,
//      reference is what stops the cone from drifting around a curved surface.
//   5. flatten: orthographic projection onto the plane through the chart
//      centroid with normal = mean normal. A face that would project with
//      negative orientation (possible only near the cone limit) is peeled off
//      and left to seed its own chart.
//   6. overlap check: triangles of one chart must not overlap in UV only if the
//      chart folds over itself. On overlap the chart is re-grown from the same
//      seed with half the cone angle (a strict subset, so this terminates); if
//      the cone bottoms out the chart degrades to one chart per face, which can
//      never self-overlap.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/utils/error.hpp"

namespace alphashape {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
/// Cone floor for the overlap re-grow; below this we fall back to single faces.
constexpr double kMinConeRadians = 1e-3;
/// Recursion guard for the overlap re-grow (halving from 90 deg reaches the
/// floor in far fewer steps than this, so it is a safety net only).
constexpr int kMaxSplitDepth = 64;
/// Slack so a face exactly on the cone boundary survives rounding.
constexpr double kConeEpsilon = 1e-12;

/// Per-face geometry cached for the whole pass.
struct FaceInfo {
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double area = 0.0;
    bool degenerate = false;
};

/// A projected chart: mesh face ids plus 3 UVs per face, in face order.
struct Flat {
    std::vector<int> faces;
    std::vector<Eigen::Vector2d> uv;
};

double degrees_to_radians(double degrees) { return degrees * (kPi / 180.0); }

/// Packed key for an undirected edge; the lower vertex id sits in the high half.
std::uint64_t edge_key(int a, int b) {
    const std::uint32_t lo = static_cast<std::uint32_t>(std::min(a, b));
    const std::uint32_t hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
}

/// Left-perpendicular of a 2-D vector.
Eigen::Vector2d perpendicular(const Eigen::Vector2d& v) {
    return Eigen::Vector2d(-v.y(), v.x());
}

/// z of the 2-D cross product; its sign is the triangle's orientation.
double cross_2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

void project_triangle(const Eigen::Vector2d* tri, const Eigen::Vector2d& axis, double& lo,
                      double& hi) {
    lo = tri[0].dot(axis);
    hi = lo;
    for (int k = 1; k < 3; ++k) {
        const double d = tri[k].dot(axis);
        lo = std::min(lo, d);
        hi = std::max(hi, d);
    }
}

/// Separating-axis test in 2-D. True only when the triangles share a strictly
/// positive area, so triangles that merely touch along an edge or a corner (the
/// norm for neighbouring faces of one chart) count as disjoint.
bool triangles_overlap(const Eigen::Vector2d* a, const Eigen::Vector2d* b, double eps) {
    Eigen::Vector2d axes[6];
    for (int k = 0; k < 3; ++k) {
        axes[k] = perpendicular(a[(k + 1) % 3] - a[k]);
        axes[k + 3] = perpendicular(b[(k + 1) % 3] - b[k]);
    }
    for (const Eigen::Vector2d& raw : axes) {
        const double len = raw.norm();
        if (!(len > 0.0)) {
            continue;  // a collapsed UV edge contributes no axis
        }
        const Eigen::Vector2d axis = raw / len;
        double a_lo = 0.0;
        double a_hi = 0.0;
        double b_lo = 0.0;
        double b_hi = 0.0;
        project_triangle(a, axis, a_lo, a_hi);
        project_triangle(b, axis, b_lo, b_hi);
        if (std::min(a_hi, b_hi) - std::max(a_lo, b_lo) <= eps) {
            return false;
        }
    }
    return true;
}

/// Two faces of one chart that share a vertex are neighbours, not a fold.
bool share_vertex(const Eigen::Vector3i& a, const Eigen::Vector3i& b) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (a[i] == b[j]) {
                return true;
            }
        }
    }
    return false;
}

int cell_index(double value, int count) {
    int index = static_cast<int>(std::floor(value));
    index = std::max(0, std::min(index, count - 1));
    return index;
}

/// Right-handed orthonormal basis (e1, e2, n): e1 is built from the axis least
/// parallel to n, e2 = n x e1, so e1 x e2 == n. Projecting a face whose normal
/// has a positive dot with n therefore preserves its winding.
void chart_basis(const Eigen::Vector3d& n, Eigen::Vector3d& e1, Eigen::Vector3d& e2) {
    const double ax = std::abs(n.x());
    const double ay = std::abs(n.y());
    const double az = std::abs(n.z());
    Eigen::Vector3d helper = Eigen::Vector3d::UnitX();
    if (ax <= ay && ax <= az) {
        helper = Eigen::Vector3d::UnitX();
    } else if (ay <= az) {
        helper = Eigen::Vector3d::UnitY();
    } else {
        helper = Eigen::Vector3d::UnitZ();
    }
    e1 = helper.cross(n).normalized();
    e2 = n.cross(e1);
}

/// Signed area of a projected triangle; its sign carries the orientation.
double tri_signed_area(const Eigen::Vector2d* t) {
    return 0.5 * cross_2d(t[1] - t[0], t[2] - t[0]);
}

/// Everything the pass needs: cached face geometry, adjacency and the growth
/// bookkeeping. One instance per build_charts() call.
class Builder {
public:
    Builder(const Mesh& mesh, const ChartOptions& options, ChartStats& stats)
        : mesh_(mesh), options_(options), stats_(stats),
          cone_radians_(degrees_to_radians(options.max_normal_angle_deg)) {}

    std::vector<Chart> run();

private:
    // -- setup ---------------------------------------------------------------
    void compute_face_info();
    void build_adjacency();

    // -- region growing ------------------------------------------------------
    std::vector<int> grow(int seed, double cone);
    void push_neighbours(int face, std::priority_queue<int, std::vector<int>, std::greater<int>>* q);

    // -- flattening ----------------------------------------------------------
    Flat flatten(const std::vector<int>& faces);
    bool has_overlap(const Flat& flat);

    // -- chart emission ------------------------------------------------------
    void emit_from_seed(int seed, double cone, int depth, std::vector<Chart>* out);
    void emit_face_chart(int face, std::vector<Chart>* out);
    void push_chart(const Flat& flat, std::vector<Chart>* out);

    const Mesh& mesh_;
    ChartOptions options_;
    /// Filled in as the pass runs; a local owned by build_charts(), never null.
    ChartStats& stats_;
    double cone_radians_ = 0.0;
    int face_count_ = 0;
    std::vector<FaceInfo> info_;
    std::vector<std::vector<int>> adjacency_;
    /// 1 once a face has been committed to a chart.
    std::vector<char> assigned_;
    /// Per-grow visitation marks, keyed by stamp_token_ so no per-call reset.
    std::vector<int> stamp_;
    int stamp_token_ = 0;
};

std::vector<Chart> Builder::run() {
    face_count_ = static_cast<int>(mesh_.faces.size());
    compute_face_info();
    build_adjacency();
    assigned_.assign(static_cast<std::size_t>(face_count_), 0);
    stamp_.assign(static_cast<std::size_t>(face_count_), 0);

    // Seeds by decreasing area, ties by lower face id: large flat regions win
    // their neighbours before the small faces scattered around them do.
    std::vector<int> order(static_cast<std::size_t>(face_count_));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [this](int a, int b) {
        if (info_[static_cast<std::size_t>(a)].area != info_[static_cast<std::size_t>(b)].area) {
            return info_[static_cast<std::size_t>(a)].area > info_[static_cast<std::size_t>(b)].area;
        }
        return a < b;
    });

    std::vector<Chart> charts;
    for (int seed : order) {
        const std::size_t s = static_cast<std::size_t>(seed);
        if (assigned_[s]) {
            continue;
        }
        if (info_[s].degenerate) {
            emit_face_chart(seed, &charts);  // no normal to grow on: its own chart
            continue;
        }
        emit_from_seed(seed, cone_radians_, 0, &charts);
    }
    return charts;
}

void Builder::emit_from_seed(int seed, double cone, int depth, std::vector<Chart>* out) {
    const std::vector<int> faces = grow(seed, cone);
    const Flat flat = faces.empty() ? Flat{} : flatten(faces);
    if (flat.faces.empty()) {
        emit_face_chart(seed, out);
        return;
    }
    if (!has_overlap(flat)) {
        push_chart(flat, out);
        return;
    }
    // The chart folds over itself in UV. Re-grow it tighter; a halved cone is a
    // strict subset, so this converges, and below the floor it degrades to one
    // chart per face, which can never self-overlap.
    ++stats_.overlap_splits;
    const double smaller = cone * 0.5;
    if (depth < kMaxSplitDepth && flat.faces.size() > 1 && smaller >= kMinConeRadians) {
        emit_from_seed(seed, smaller, depth + 1, out);
    } else {
        emit_face_chart(seed, out);
    }
}

void Builder::emit_face_chart(int face, std::vector<Chart>* out) {
    const std::size_t f = static_cast<std::size_t>(face);
    if (assigned_[f]) {
        return;
    }
    assigned_[f] = 1;

    const Eigen::Vector3d axis =
        info_[f].degenerate ? Eigen::Vector3d::UnitZ() : info_[f].normal;
    Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
    chart_basis(axis, e1, e2);

    const Eigen::Vector3i& tri = mesh_.faces[f];
    const Eigen::Vector3d origin = (mesh_.vertices[static_cast<std::size_t>(tri[0])] +
                                    mesh_.vertices[static_cast<std::size_t>(tri[1])] +
                                    mesh_.vertices[static_cast<std::size_t>(tri[2])]) /
                                   3.0;

    Chart chart;
    chart.faces.push_back(face);
    chart.corner_uv.reserve(3);
    for (int k = 0; k < 3; ++k) {
        const Eigen::Vector3d d = mesh_.vertices[static_cast<std::size_t>(tri[k])] - origin;
        chart.corner_uv.push_back(Eigen::Vector2d(d.dot(e1), d.dot(e2)));
    }
    out->push_back(std::move(chart));
}

void Builder::push_chart(const Flat& flat, std::vector<Chart>* out) {
    Chart chart;
    chart.faces = flat.faces;
    chart.corner_uv = flat.uv;
    for (int f : flat.faces) {
        assigned_[static_cast<std::size_t>(f)] = 1;
    }
    out->push_back(std::move(chart));
}

void Builder::compute_face_info() {
    info_.assign(static_cast<std::size_t>(face_count_), FaceInfo{});
    for (int f = 0; f < face_count_; ++f) {
        const Eigen::Vector3i& tri = mesh_.faces[static_cast<std::size_t>(f)];
        const Eigen::Vector3d& p0 = mesh_.vertices[static_cast<std::size_t>(tri[0])];
        const Eigen::Vector3d& p1 = mesh_.vertices[static_cast<std::size_t>(tri[1])];
        const Eigen::Vector3d& p2 = mesh_.vertices[static_cast<std::size_t>(tri[2])];
        const Eigen::Vector3d cross = (p1 - p0).cross(p2 - p0);
        FaceInfo fi;
        fi.area = 0.5 * cross.norm();
        // Zero area means the corner points are collinear (or coincide): there is
        // no plane to define a normal or a chart on.
        fi.degenerate = !(fi.area > 0.0);
        if (fi.degenerate) {
            fi.normal = Eigen::Vector3d::UnitZ();
            fi.area = 0.0;
        } else {
            fi.normal = cross.normalized();
        }
        info_[static_cast<std::size_t>(f)] = fi;
    }
}

void Builder::build_adjacency() {
    adjacency_.assign(static_cast<std::size_t>(face_count_), {});
    // Count incident faces per edge, then link the pairs that have exactly two.
    std::unordered_map<std::uint64_t, std::vector<int>> edge_faces;
    edge_faces.reserve(static_cast<std::size_t>(face_count_) * 3);
    for (int f = 0; f < face_count_; ++f) {
        const Eigen::Vector3i& tri = mesh_.faces[static_cast<std::size_t>(f)];
        for (int k = 0; k < 3; ++k) {
            edge_faces[edge_key(tri[k], tri[(k + 1) % 3])].push_back(f);
        }
    }
    for (const auto& entry : edge_faces) {
        const std::vector<int>& incident = entry.second;
        if (incident.size() != 2) {
            continue;  // boundary (1) or non-manifold (3+): no connection
        }
        adjacency_[static_cast<std::size_t>(incident[0])].push_back(incident[1]);
        adjacency_[static_cast<std::size_t>(incident[1])].push_back(incident[0]);
    }
    // Deterministic neighbour order for the pop-order tie break.
    for (std::vector<int>& row : adjacency_) {
        std::sort(row.begin(), row.end());
    }
}

void Builder::push_neighbours(int face, std::priority_queue<int, std::vector<int>, std::greater<int>>* q) {
    for (int neighbour : adjacency_[static_cast<std::size_t>(face)]) {
        if (!assigned_[static_cast<std::size_t>(neighbour)] &&
            stamp_[static_cast<std::size_t>(neighbour)] != stamp_token_) {
            stamp_[static_cast<std::size_t>(neighbour)] = stamp_token_;
            q->push(neighbour);
        }
    }
}

std::vector<int> Builder::grow(int seed, double cone) {
    ++stamp_token_;
    const std::size_t seed_index = static_cast<std::size_t>(seed);
    const Eigen::Vector3d seed_normal = info_[seed_index].normal;
    const double cos_limit = std::cos(cone);

    std::vector<int> faces;
    faces.reserve(64);
    faces.push_back(seed);

    Eigen::Vector3d weighted = info_[seed_index].normal * info_[seed_index].area;
    double total_area = info_[seed_index].area;
    // Seed-normal test against itself is trivially true, so the seed only needs
    // to be marked; committed faces are marked in push_chart/emit_face_chart.
    stamp_[seed_index] = stamp_token_;

    std::priority_queue<int, std::vector<int>, std::greater<int>> frontier;
    push_neighbours(seed, &frontier);

    const std::size_t cap = options_.max_chart_faces;
    while (!frontier.empty() && faces.size() < cap) {
        const int face = frontier.top();
        frontier.pop();
        const std::size_t f = static_cast<std::size_t>(face);
        if (assigned_[f] || stamp_[f] != stamp_token_) {
            continue;
        }
        stamp_[f] = stamp_token_;
        const FaceInfo& fi = info_[f];
        if (fi.degenerate) {
            continue;  // no normal: stays for its own single-face chart
        }
        const Eigen::Vector3d mean = total_area > 0.0 ? (weighted / total_area).normalized()
                                                      : seed_normal;
        // Two tests: closeness to the running mean keeps the chart tight, and
        // closeness to the fixed seed normal keeps the mean from drifting.
        if (fi.normal.dot(mean) < cos_limit - kConeEpsilon) {
            continue;
        }
        if (fi.normal.dot(seed_normal) < cos_limit - kConeEpsilon) {
            continue;
        }
        faces.push_back(face);
        weighted += fi.normal * fi.area;
        total_area += fi.area;
        push_neighbours(face, &frontier);
    }
    // Faces stay in deterministic discovery order (min-id priority queue), which
    // also keeps faces[0] == seed so the seed normal is recoverable from the
    // chart and the growth cone is testable.
    return faces;
}

Flat Builder::flatten(const std::vector<int>& faces) {
    Flat flat;
    flat.uv.reserve(faces.size() * 3);

    Eigen::Vector3d weighted = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (int f : faces) {
        const FaceInfo& fi = info_[static_cast<std::size_t>(f)];
        weighted += fi.normal * fi.area;
        const Eigen::Vector3i& tri = mesh_.faces[static_cast<std::size_t>(f)];
        for (int k = 0; k < 3; ++k) {
            centroid += mesh_.vertices[static_cast<std::size_t>(tri[k])];
        }
    }
    centroid /= static_cast<double>(faces.size() * 3);
    // Guard the zero vector before normalising: a chart of one degenerate face
    // has no direction at all.
    Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
    if (weighted.norm() > 0.0) {
        normal = weighted.normalized();
    }
    Eigen::Vector3d e1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d e2 = Eigen::Vector3d::Zero();
    chart_basis(normal, e1, e2);

    // Project first, then drop any face that came out mirrored. Because every
    // kept face has a positive dot with the chart mean normal, its projection is
    // typically CCW; the rare exception is a face near the cone limit whose
    // normal has an out-of-plane component that flips the sign.
    std::vector<int> kept;
    std::vector<Eigen::Vector2d> kept_uv;
    kept.reserve(faces.size());
    for (int f : faces) {
        const Eigen::Vector3i& tri = mesh_.faces[static_cast<std::size_t>(f)];
        Eigen::Vector2d uv[3];
        for (int k = 0; k < 3; ++k) {
            const Eigen::Vector3d d = mesh_.vertices[static_cast<std::size_t>(tri[k])] - centroid;
            uv[k] = Eigen::Vector2d(d.dot(e1), d.dot(e2));
        }
        if (tri_signed_area(uv) <= 0.0) {
            continue;
        }
        kept.push_back(f);
        kept_uv.push_back(uv[0]);
        kept_uv.push_back(uv[1]);
        kept_uv.push_back(uv[2]);
    }
    flat.faces = std::move(kept);
    flat.uv = std::move(kept_uv);
    return flat;
}

bool Builder::has_overlap(const Flat& flat) {
    const std::size_t n = flat.faces.size();
    if (n < 2) {
        return false;
    }
    double lo_x = flat.uv[0].x();
    double hi_x = lo_x;
    double lo_y = flat.uv[0].y();
    double hi_y = lo_y;
    for (const Eigen::Vector2d& p : flat.uv) {
        lo_x = std::min(lo_x, p.x());
        hi_x = std::max(hi_x, p.x());
        lo_y = std::min(lo_y, p.y());
        hi_y = std::max(hi_y, p.y());
    }
    const double width = hi_x - lo_x;
    const double height = hi_y - lo_y;
    const double span = std::max(width, height);
    if (!(span > 0.0)) {
        return false;  // everything collapsed to one point: nothing to cross
    }
    // ~two triangles per cell on average keeps the bucket scan near-linear.
    int cells = static_cast<int>(std::sqrt(static_cast<double>(n) / 2.0));
    cells = std::max(1, std::min(cells, 512));
    const double cell_w = std::max(width / static_cast<double>(cells), 1e-300);
    const double cell_h = std::max(height / static_cast<double>(cells), 1e-300);
    // Scale a touching tolerance to the chart, so float noise in a shared edge
    // never reads as a fold.
    const double eps = 1e-9 * span;

    std::unordered_map<std::uint64_t, std::vector<int>> buckets;
    buckets.reserve(n * 2);
    for (std::size_t i = 0; i < n; ++i) {
        const Eigen::Vector2d* tri = &flat.uv[i * 3];
        double tlo_x = tri[0].x();
        double thi_x = tlo_x;
        double tlo_y = tri[0].y();
        double thi_y = tlo_y;
        for (int k = 1; k < 3; ++k) {
            tlo_x = std::min(tlo_x, tri[k].x());
            thi_x = std::max(thi_x, tri[k].x());
            tlo_y = std::min(tlo_y, tri[k].y());
            thi_y = std::max(thi_y, tri[k].y());
        }
        const int cx0 = cell_index((tlo_x - lo_x) / cell_w, cells);
        const int cx1 = cell_index((thi_x - lo_x) / cell_w, cells);
        const int cy0 = cell_index((tlo_y - lo_y) / cell_h, cells);
        const int cy1 = cell_index((thi_y - lo_y) / cell_h, cells);
        for (int cy = cy0; cy <= cy1; ++cy) {
            for (int cx = cx0; cx <= cx1; ++cx) {
                const std::uint64_t key = (static_cast<std::uint64_t>(cy) << 20) |
                                          static_cast<std::uint64_t>(cx);
                buckets[key].push_back(static_cast<int>(i));
            }
        }
    }

    for (const auto& entry : buckets) {
        const std::vector<int>& cell = entry.second;
        for (std::size_t a = 0; a + 1 < cell.size(); ++a) {
            for (std::size_t b = a + 1; b < cell.size(); ++b) {
                const int ia = cell[a];
                const int ib = cell[b];
                if (share_vertex(mesh_.faces[static_cast<std::size_t>(flat.faces[static_cast<std::size_t>(ia)])],
                                 mesh_.faces[static_cast<std::size_t>(flat.faces[static_cast<std::size_t>(ib)])])) {
                    continue;  // share a mesh vertex: neighbours, not a fold
                }
                if (triangles_overlap(&flat.uv[static_cast<std::size_t>(ia) * 3],
                                      &flat.uv[static_cast<std::size_t>(ib) * 3], eps)) {
                    return true;
                }
            }
        }
    }
    return false;
}

}  // namespace

std::vector<Chart> build_charts(const Mesh& mesh, const ChartOptions& options,
                                ChartStats* stats) {
    // Validate up front, before any geometry work.
    if (!(options.max_normal_angle_deg > 0.0) || !(options.max_normal_angle_deg < 90.0)) {
        throw ValueError("build_charts: max_normal_angle_deg must be in (0, 90)");
    }
    if (options.max_chart_faces == 0) {
        throw ValueError("build_charts: max_chart_faces must be >= 1");
    }
    const int vertex_count = static_cast<int>(mesh.vertices.size());
    const int face_count = static_cast<int>(mesh.faces.size());
    for (int f = 0; f < face_count; ++f) {
        const Eigen::Vector3i& tri = mesh.faces[static_cast<std::size_t>(f)];
        for (int k = 0; k < 3; ++k) {
            const int index = tri[k];
            if (index < 0 || index >= vertex_count) {
                throw ValueError("build_charts: face " + std::to_string(f) + " references vertex " +
                                 std::to_string(index) + " outside [0, " +
                                 std::to_string(vertex_count) + ")");
            }
        }
    }

    // The tally is accumulated into a local and published at the end, so a
    // throwing run (validation aside) never leaves a half-updated counter
    // visible and the pass holds no cross-call state.
    ChartStats local;
    Builder builder(mesh, options, local);
    std::vector<Chart> charts = builder.run();
    if (stats != nullptr) {
        *stats = local;
    }
    return charts;
}

}  // namespace alphashape
