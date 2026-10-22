// Quadric error-metric decimation (Garland & Heckbert, SIGGRAPH '97).
//
// Every vertex accumulates the squared-distance quadrics of its incident face
// planes, weighted by face area. Edges are then collapsed in increasing order
// of the combined quadric's error. Two gates keep the result usable:
//
//   * the link condition — a collapse may not weld two vertices that already
//     share a neighbour through a face that does not contain the edge; that is
//     exactly the configuration that would create a non-manifold edge;
//   * a normal-flip test — a collapse may not turn an incident face inside out.
//
// When `preserve_boundary` is set, every boundary edge contributes a heavily
// weighted virtual plane (containing the edge, perpendicular to its face)
// which pins the boundary loop in place and collapses it only collinearly.

#include "model/mesh/decimate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "base/utils/error.hpp"

namespace alphashape {
namespace {

using Index = std::size_t;

/// Weight of a boundary-edge constraint plane, in units of "edge length
/// squared". Face planes carry their area as weight, so the boundary always
/// outvotes the surface. (Ours.)
constexpr double kBoundaryWeight = 1000.0;

/// A collapse is refused when it rotates an incident face normal by more than
/// acos(kMinNormalDot) (~78 degrees); a folded or reflected face lands at a
/// negative dot, so this also rejects inversions. (Ours.)
constexpr double kMinNormalDot = 0.2;

/// Relative determinant (against the cube of the largest diagonal entry) below
/// which the 3x3 quadric system is treated as singular. (Ours.)
constexpr double kConditionEpsilon = 1e-9;

/// A collapse that leaves an incident face with less than this fraction of its
/// former twice-area produces a sliver and is refused. (Ours.)
constexpr double kMinAreaRatio = 1e-12;

/// Upper triangle of a symmetric 4x4, in the order
/// (xx, xy, xz, xw, yy, yz, yw, zz, zw, ww).
struct Quadric {
    std::array<double, 10> a{};

    /// Accumulate w * q q^T for the plane {p : n . p + d == 0}.
    void add_plane(const Eigen::Vector3d& n, double d, double w) {
        const std::array<double, 4> q{n.x(), n.y(), n.z(), d};
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = i; j < 4; ++j) {
                a[i * 4 + j - i * (i + 1) / 2] += w * q[i] * q[j];
            }
        }
    }

    Quadric& operator+=(const Quadric& other) {
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] += other.a[i];
        }
        return *this;
    }

    /// Right-hand side of the 3x3 stationarity system A p = -b.
    Eigen::Vector3d rhs() const { return Eigen::Vector3d(-a[3], -a[6], -a[8]); }

    double evaluate(const Eigen::Vector3d& p) const {
        const double x = p.x();
        const double y = p.y();
        const double z = p.z();
        return a[0] * x * x + a[4] * y * y + a[7] * z * z + a[9] +
               2.0 * (a[1] * x * y + a[2] * x * z + a[5] * y * z) +
               2.0 * (a[3] * x + a[6] * y + a[8] * z);
    }
};

/// Unique key for the undirected edge {u, v}.
std::uint64_t edge_key(Index u, Index v) {
    const Index lo = std::min(u, v);
    const Index hi = std::max(u, v);
    return (static_cast<std::uint64_t>(lo) << 32U) | static_cast<std::uint64_t>(hi);
}

/// Twice the area vector of the triangle (p0, p1, p2), oriented along
/// (p1 - p0) x (p2 - p0).
Eigen::Vector3d face_normal(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                            const Eigen::Vector3d& p2) {
    return (p1 - p0).cross(p2 - p0);
}

constexpr Index kNoIndex = std::numeric_limits<Index>::max();

bool face_contains(const Eigen::Vector3i& tri, Index x) {
    return static_cast<Index>(tri[0]) == x || static_cast<Index>(tri[1]) == x ||
           static_cast<Index>(tri[2]) == x;
}

/// The one of the three vertices that is neither `a` nor `b`, or kNoIndex when
/// the face does not carry both.
Index third_vertex(const Eigen::Vector3i& tri, Index a, Index b) {
    for (int k = 0; k < 3; ++k) {
        const Index x = static_cast<Index>(tri[k]);
        if (x != a && x != b) {
            return x;
        }
    }
    return kNoIndex;
}

/// Minimises the quadric: the stationarity point A p = -b of the 3x3 block.
/// Returns false when the system is singular (a flat or collinear vertex
/// neighbourhood carries no information along the surface) or ill-conditioned,
/// leaving the caller to fall back on the endpoint positions.
///
/// Solved by the adjugate rather than a factorisation: this runs about a
/// million times on a large input, and the closed form is both allocation-free
/// and bit-deterministic, which the heap's tie-breaking relies on.
bool solve_minimum(const Quadric& q, Eigen::Vector3d& out) {
    const double a00 = q.a[0];
    const double a01 = q.a[1];
    const double a02 = q.a[2];
    const double a11 = q.a[4];
    const double a12 = q.a[5];
    const double a22 = q.a[7];
    const double scale = std::max(a00, std::max(a11, a22));
    if (!(scale > 0.0)) {
        return false;
    }
    const double c00 = a11 * a22 - a12 * a12;
    const double c01 = a12 * a02 - a01 * a22;
    const double c02 = a01 * a12 - a11 * a02;
    const double c11 = a00 * a22 - a02 * a02;
    const double c12 = a02 * a01 - a00 * a12;
    const double c22 = a00 * a11 - a01 * a01;
    const double det = a00 * c00 + a01 * c01 + a02 * c02;
    if (!(std::abs(det) > kConditionEpsilon * scale * scale * scale)) {
        return false;
    }
    const double bx = -q.a[3];
    const double by = -q.a[6];
    const double bz = -q.a[8];
    const double inv = 1.0 / det;
    out = Eigen::Vector3d((c00 * bx + c01 * by + c02 * bz) * inv,
                          (c01 * bx + c11 * by + c12 * bz) * inv,
                          (c02 * bx + c12 * by + c22 * bz) * inv);
    return out.allFinite();
}

/// Best of the two endpoints and their midpoint under the quadric — the
/// fallback when `solve_minimum` cannot find a stationarity point.
Eigen::Vector3d best_endpoint(const Quadric& q, const Eigen::Vector3d& a,
                              const Eigen::Vector3d& b) {
    Eigen::Vector3d best = a;
    double cost = q.evaluate(a);
    const double cb = q.evaluate(b);
    if (cb < cost) {
        cost = cb;
        best = b;
    }
    const Eigen::Vector3d mid = 0.5 * (a + b);
    const double cm = q.evaluate(mid);
    if (cm < cost) {
        best = mid;
    }
    return best;
}

/// A queued collapse. `u` is always the smaller index; the survivor is `u`.
struct Candidate {
    double cost = 0.0;
    Index u = 0;
    Index v = 0;
    std::uint32_t stamp_u = 0;
    std::uint32_t stamp_v = 0;
};

/// Min-heap ordering: cheapest first, ties broken by (u, v) so that two runs
/// over the same input visit edges in exactly the same order.
struct CandidateGreater {
    bool operator()(const Candidate& lhs, const Candidate& rhs) const {
        if (lhs.cost != rhs.cost) {
            return lhs.cost > rhs.cost;
        }
        if (lhs.u != rhs.u) {
            return lhs.u > rhs.u;
        }
        return lhs.v > rhs.v;
    }
};

/// Mutable state for one decimation run: the mesh plus the per-vertex quadrics,
/// the per-vertex incident-face lists, the validity flags and the heap.
struct Workspace {
    Mesh* mesh = nullptr;
    bool preserve_boundary = true;
    double max_error = 0.0;

    /// Working copy of the positions — a collapse moves a vertex here without
    /// touching the original array until the final compaction.
    std::vector<Eigen::Vector3d> positions;
    std::vector<Quadric> quadrics;
    /// Live faces per vertex. Entries for dead faces are filtered lazily; a
    /// vertex list only ever grows when it absorbs a collapsed neighbour's, and
    /// is compacted right after.
    std::vector<std::vector<Index>> incident;
    /// Bumped whenever a vertex's position or quadric changes; a heap entry
    /// whose stored stamps disagree is stale.
    std::vector<std::uint32_t> stamps;
    std::vector<char> alive;
    std::vector<char> face_alive;
    std::vector<char> boundary;

    /// Scratch for the link-condition test. `mark[x] == token` means "x belongs
    /// to the neighbourhood currently being scanned". The test runs on nearly
    /// every queued edge, which on a large input is over a million times, so it
    /// must neither allocate nor sort.
    std::vector<std::uint32_t> mark;
    std::uint32_t token = 0;
    /// Reusable scratch for the neighbour list rebuilt after each collapse.
    std::vector<Index> neighbours;

    std::size_t live_faces = 0;
    std::size_t target = 0;

    std::priority_queue<Candidate, std::vector<Candidate>, CandidateGreater> queue;

    /// Opens a fresh marking pass. Stale entries always carry an older token, so
    /// clearing is only needed on the (astronomically unlikely) counter wrap.
    std::uint32_t next_token() {
        ++token;
        if (token == 0) {
            std::fill(mark.begin(), mark.end(), 0U);
            token = 1;
        }
        return token;
    }
    /// Appends every alive vertex sharing a live face with `x` (excluding `x`)
    /// to `out`, sorted and duplicate-free.
    void collect_neighbours(Index x, std::vector<Index>& out);
    /// True when some live edge at `x` is used by a single live face.
    bool vertex_is_boundary(Index x) const;
    /// Heap entry for the undirected edge {a, b} under the current quadrics.
    Candidate make_candidate(Index a, Index b) const;
    /// Attempts the collapse of edge {u, v} (u < v) onto `u`. Returns true when
    /// applied.
    bool collapse(Index u, Index v);
};

void Workspace::collect_neighbours(Index x, std::vector<Index>& out) {
    out.clear();
    for (const Index f : incident[x]) {
        if (face_alive[f] == 0) {
            continue;
        }
        const Eigen::Vector3i& tri = mesh->faces[f];
        for (int k = 0; k < 3; ++k) {
            const Index y = static_cast<Index>(tri[k]);
            if (y != x) {
                out.push_back(y);
            }
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

bool Workspace::vertex_is_boundary(Index x) const {
    std::unordered_map<std::uint64_t, int> edges;
    for (const Index f : incident[x]) {
        if (face_alive[f] == 0) {
            continue;
        }
        const Eigen::Vector3i& tri = mesh->faces[f];
        for (int k = 0; k < 3; ++k) {
            const Index a = static_cast<Index>(tri[k]);
            const Index b = static_cast<Index>(tri[(k + 1) % 3]);
            ++edges[edge_key(a, b)];
        }
    }
    for (const auto& entry : edges) {
        if (entry.second == 1) {
            return true;
        }
    }
    return false;
}

Candidate Workspace::make_candidate(Index a, Index b) const {
    const Index u = std::min(a, b);
    const Index v = std::max(a, b);
    Quadric q = quadrics[u];
    q += quadrics[v];
    Eigen::Vector3d pos;
    if (!solve_minimum(q, pos)) {
        pos = best_endpoint(q, positions[u], positions[v]);
    }
    Candidate c;
    c.cost = q.evaluate(pos);
    if (!(c.cost >= 0.0)) {
        c.cost = 0.0;
    }
    c.u = u;
    c.v = v;
    c.stamp_u = stamps[u];
    c.stamp_v = stamps[v];
    return c;
}

bool Workspace::collapse(Index u, Index v) {
    // --- Find the faces shared by u and v ----------------------------------
    // Manifold mesh: an interior edge has two, a boundary edge one. Anything
    // else is not a collapsible edge.
    Index shared[2];
    int shared_count = 0;
    for (const Index f : incident[u]) {
        if (face_alive[f] == 0 || !face_contains(mesh->faces[f], v)) {
            continue;
        }
        if (shared_count == 2) {
            return false;
        }
        shared[shared_count] = f;
        ++shared_count;
    }
    if (shared_count == 0) {
        return false;
    }
    const bool is_boundary_edge = shared_count == 1;

    // --- Gate (a): the link condition ---------------------------------------
    // Welding u and v joins the two stars along the shared edge. Every neighbour
    // the two have in common must be the opposite vertex of a face they share:
    // any other shared neighbour is reached through a face that does not carry
    // the edge, and would end up joined to the merged vertex by two distinct
    // edges — a non-manifold edge.
    //
    // Instead of building and intersecting the two neighbour sets, mark u's
    // neighbours with a fresh token and subtract the two shared faces' opposite
    // vertices: `seen` must then stay zero as v's faces are walked, and every
    // mark met must belong to one of those two vertices.
    const std::uint32_t pass = next_token();
    for (const Index f : incident[u]) {
        if (face_alive[f] == 0) {
            continue;
        }
        const Eigen::Vector3i& tri = mesh->faces[f];
        for (int k = 0; k < 3; ++k) {
            const Index y = static_cast<Index>(tri[k]);
            if (y != u && y != v) {
                mark[y] = pass;
            }
        }
    }
    const Index link0 = third_vertex(mesh->faces[shared[0]], u, v);
    mark[link0] = 0;
    Index link1 = kNoIndex;
    if (shared_count == 2) {
        link1 = third_vertex(mesh->faces[shared[1]], u, v);
        mark[link1] = 0;
    }
    for (const Index f : incident[v]) {
        if (face_alive[f] == 0) {
            continue;
        }
        const Eigen::Vector3i& tri = mesh->faces[f];
        for (int k = 0; k < 3; ++k) {
            const Index y = static_cast<Index>(tri[k]);
            if (y != u && y != v && mark[y] == pass) {
                return false;
            }
        }
    }

    // --- Gate (b): boundary vertices may only merge along a boundary edge ----
    const bool was_boundary = boundary[u] != 0 || boundary[v] != 0;
    if (preserve_boundary && !is_boundary_edge) {
        if (boundary[u] != 0 && boundary[v] != 0) {
            return false;
        }
    }

    // --- Choose the target position ----------------------------------------
    Quadric q = quadrics[u];
    q += quadrics[v];
    Eigen::Vector3d pos;
    if (!solve_minimum(q, pos)) {
        pos = best_endpoint(q, positions[u], positions[v]);
    }

    // --- Gate (c): no incident face may fold -------------------------------
    for (std::size_t side = 0; side < 2; ++side) {
        const Index x = (side == 0) ? u : v;
        for (const Index f : incident[x]) {
            if (face_alive[f] == 0) {
                continue;
            }
            const Eigen::Vector3i& tri = mesh->faces[f];
            if (face_contains(tri, u) && face_contains(tri, v)) {
                continue;  // this face disappears in the collapse
            }
            Eigen::Vector3d p[3];
            for (int k = 0; k < 3; ++k) {
                const Index y = static_cast<Index>(tri[k]);
                p[k] = (y == u || y == v) ? pos : positions[y];
            }
            const Eigen::Vector3d before = face_normal(
                positions[static_cast<Index>(tri[0])],
                positions[static_cast<Index>(tri[1])],
                positions[static_cast<Index>(tri[2])]);
            const double before_sq = before.squaredNorm();
            const Eigen::Vector3d after = face_normal(p[0], p[1], p[2]);
            const double after_sq = after.squaredNorm();
            if (!(after_sq > kMinAreaRatio * before_sq)) {
                return false;  // collapsed to a sliver
            }
            if (before.dot(after) < kMinNormalDot * std::sqrt(before_sq * after_sq)) {
                return false;  // normal rotated too far, or flipped
            }
        }
    }

    // --- Apply --------------------------------------------------------------
    positions[u] = pos;
    quadrics[u] = q;
    ++stamps[u];

    // Move v's faces onto u, dropping the (at most two) that held both.
    for (const Index f : incident[v]) {
        if (face_alive[f] == 0) {
            continue;
        }
        Eigen::Vector3i& tri = mesh->faces[f];
        if (face_contains(tri, u)) {
            face_alive[f] = 0;
            --live_faces;
            continue;
        }
        for (int k = 0; k < 3; ++k) {
            if (static_cast<Index>(tri[k]) == v) {
                tri[k] = static_cast<int>(u);
            }
        }
        incident[u].push_back(f);
    }

    // Average the surviving vertex's attributes over the two it replaces.
    if (!mesh->colours.empty()) {
        for (Index c = 0; c < 3; ++c) {
            const Index iu = u * 3 + c;
            const Index iv = v * 3 + c;
            const unsigned sum =
                static_cast<unsigned>(mesh->colours[iu]) + static_cast<unsigned>(mesh->colours[iv]);
            mesh->colours[iu] = static_cast<std::uint8_t>((sum + 1U) / 2U);
        }
    }
    if (!mesh->confidence.empty()) {
        mesh->confidence[u] = 0.5F * (mesh->confidence[u] + mesh->confidence[v]);
    }

    alive[v] = 0;
    incident[v].clear();
    ++stamps[v];

    // Compact u's face list and refresh its boundary flag.
    std::vector<Index> keep;
    keep.reserve(incident[u].size());
    for (const Index f : incident[u]) {
        if (face_alive[f] != 0) {
            keep.push_back(f);
        }
    }
    incident[u].swap(keep);
    if (preserve_boundary && was_boundary) {
        boundary[u] = vertex_is_boundary(u) ? 1 : 0;
    }

    // Re-queue every edge at u: u's own quadric and position changed, so its
    // previous entries are stale. A neighbour's own quadric is untouched, so its
    // entries for *other* edges stay valid and must not be invalidated.
    collect_neighbours(u, neighbours);
    for (const Index y : neighbours) {
        queue.push(make_candidate(u, y));
    }
    return true;
}

}  // namespace

std::size_t decimate_quadric(Mesh& mesh, const DecimateOptions& options) {
    const std::size_t vertex_count = mesh.vertices.size();
    const std::size_t face_count = mesh.faces.size();

    // --- Validation ---------------------------------------------------------
    if (!mesh.colours.empty() && mesh.colours.size() != 3 * vertex_count) {
        throw ValueError("decimate_quadric: colours must be empty or hold 3 bytes per vertex");
    }
    if (!mesh.confidence.empty() && mesh.confidence.size() != vertex_count) {
        throw ValueError("decimate_quadric: confidence must be empty or hold one value per vertex");
    }
    // Decimation collapses vertices and rewrites face indices; it has no rule
    // for merging the per-corner texture coordinates of the faces it removes,
    // and the uv pool is indexed by uv_faces rather than by vertex. A textured
    // mesh (or one carrying a partial pool) must therefore be decimated first,
    // then mapped.
    if (!mesh.uv_faces.empty() || !mesh.uvs.empty()) {
        throw ValueError("decimate_quadric: decimate a mesh before building UVs");
    }
    for (const Eigen::Vector3i& tri : mesh.faces) {
        for (int k = 0; k < 3; ++k) {
            const int index = tri[k];
            if (index < 0 || static_cast<std::size_t>(index) >= vertex_count) {
                throw ValueError("decimate_quadric: face index out of range");
            }
        }
    }

    // Nothing asked for, or nothing removable.
    if (vertex_count == 0 || face_count == 0) {
        return face_count;
    }
    if (options.target_faces == 0) {
        // "Decimate as far as max_error allows": an unbounded error plus a zero
        // target means every admissible collapse, which for a closed mesh would
        // run to a degenerate point. Treat 0 as "no reduction requested" unless
        // max_error is finite, matching the documented contract.
        if (!std::isfinite(options.max_error)) {
            return face_count;
        }
    } else if (options.target_faces >= face_count) {
        return face_count;
    }

    // --- Workspace ----------------------------------------------------------
    Workspace ws;
    ws.mesh = &mesh;
    ws.preserve_boundary = options.preserve_boundary;
    ws.max_error = options.max_error;
    ws.target = options.target_faces;
    ws.live_faces = face_count;

    ws.positions.assign(mesh.vertices.begin(), mesh.vertices.end());
    ws.quadrics.assign(vertex_count, Quadric{});
    ws.incident.assign(vertex_count, {});
    ws.stamps.assign(vertex_count, 0);
    ws.alive.assign(vertex_count, 1);
    ws.face_alive.assign(face_count, 1);
    ws.boundary.assign(vertex_count, 0);
    ws.mark.assign(vertex_count, 0U);

    // Per-vertex incident faces, plus the undirected edge -> live-face count map
    // used to spot boundary edges.
    std::unordered_map<std::uint64_t, int> edge_faces;
    edge_faces.reserve(face_count * 3);
    for (std::size_t f = 0; f < face_count; ++f) {
        const Eigen::Vector3i& tri = mesh.faces[f];
        for (int k = 0; k < 3; ++k) {
            // Each vertex of the face gets the face exactly once; each edge of
            // the face bumps the edge count exactly once.
            ws.incident[static_cast<Index>(tri[k])].push_back(f);
            const Index a = static_cast<Index>(tri[k]);
            const Index b = static_cast<Index>(tri[(k + 1) % 3]);
            ++edge_faces[edge_key(a, b)];
        }
    }

    // --- Vertex quadrics ----------------------------------------------------
    for (std::size_t f = 0; f < face_count; ++f) {
        const Eigen::Vector3i& tri = mesh.faces[f];
        const Eigen::Vector3d& p0 = ws.positions[static_cast<Index>(tri[0])];
        const Eigen::Vector3d& p1 = ws.positions[static_cast<Index>(tri[1])];
        const Eigen::Vector3d& p2 = ws.positions[static_cast<Index>(tri[2])];
        const Eigen::Vector3d raw = face_normal(p0, p1, p2);
        const double twice_area = raw.norm();
        if (!(twice_area > 0.0)) {
            continue;  // degenerate face contributes no plane
        }
        const Eigen::Vector3d n = raw / twice_area;
        const double area = 0.5 * twice_area;
        const double d = -n.dot(p0);
        for (int k = 0; k < 3; ++k) {
            ws.quadrics[static_cast<Index>(tri[k])].add_plane(n, d, area);
        }
    }

    // Boundary edges: a single live face. Their constraint planes pin the loop.
    for (const auto& entry : edge_faces) {
        if (entry.second != 1) {
            continue;
        }
        const Index u = static_cast<Index>(entry.first >> 32U);
        const Index v = static_cast<Index>(entry.first & 0xffffffffULL);
        ws.boundary[u] = 1;
        ws.boundary[v] = 1;
        if (!ws.preserve_boundary) {
            continue;
        }
        // Find the unique face of the edge, then the plane through the edge
        // perpendicular to that face.
        for (const Index f : ws.incident[u]) {
            if (face_contains(mesh.faces[f], v)) {
                const Eigen::Vector3i& tri = mesh.faces[f];
                const Eigen::Vector3d& p0 = ws.positions[static_cast<Index>(tri[0])];
                const Eigen::Vector3d& p1 = ws.positions[static_cast<Index>(tri[1])];
                const Eigen::Vector3d& p2 = ws.positions[static_cast<Index>(tri[2])];
                const Eigen::Vector3d raw = face_normal(p0, p1, p2);
                const double twice_area = raw.norm();
                if (!(twice_area > 0.0)) {
                    break;
                }
                const Eigen::Vector3d fn = raw / twice_area;
                const Eigen::Vector3d edge = ws.positions[v] - ws.positions[u];
                const Eigen::Vector3d cn = fn.cross(edge);
                const double cn_norm = cn.norm();
                if (!(cn_norm > 0.0)) {
                    break;
                }
                const Eigen::Vector3d n = cn / cn_norm;
                const double d = -n.dot(ws.positions[u]);
                const double weight = kBoundaryWeight * edge.squaredNorm();
                ws.quadrics[u].add_plane(n, d, weight);
                ws.quadrics[v].add_plane(n, d, weight);
                break;
            }
        }
    }

    // --- Priority queue -----------------------------------------------------
    // Seed every distinct undirected edge once.
    for (const auto& entry : edge_faces) {
        const Index u = static_cast<Index>(entry.first >> 32U);
        const Index v = static_cast<Index>(entry.first & 0xffffffffULL);
        ws.queue.push(ws.make_candidate(u, v));
    }

    // --- Collapse loop ------------------------------------------------------
    while (ws.live_faces > ws.target && !ws.queue.empty()) {
        const Candidate c = ws.queue.top();
        ws.queue.pop();
        if (ws.alive[c.u] == 0 || ws.alive[c.v] == 0) {
            continue;
        }
        if (ws.stamps[c.u] != c.stamp_u || ws.stamps[c.v] != c.stamp_v) {
            continue;  // one endpoint changed since this entry was queued
        }
        if (c.cost > options.max_error) {
            break;  // the heap is ordered, so nothing cheaper remains
        }
        ws.collapse(c.u, c.v);
    }

    // --- Compaction ---------------------------------------------------------
    std::vector<int> remap(vertex_count, -1);
    std::vector<Eigen::Vector3d> new_vertices;
    new_vertices.reserve(mesh.vertices.size());
    std::vector<std::uint8_t> new_colours;
    std::vector<float> new_confidence;
    const bool has_colours = !mesh.colours.empty();
    const bool has_confidence = !mesh.confidence.empty();
    if (has_colours) {
        new_colours.reserve(mesh.colours.size());
    }
    if (has_confidence) {
        new_confidence.reserve(mesh.confidence.size());
    }

    for (std::size_t f = 0; f < face_count; ++f) {
        if (ws.face_alive[f] == 0) {
            continue;
        }
        Eigen::Vector3i& tri = mesh.faces[f];
        for (int k = 0; k < 3; ++k) {
            const Index old_index = static_cast<Index>(tri[k]);
            if (remap[old_index] < 0) {
                remap[old_index] = static_cast<int>(new_vertices.size());
                new_vertices.push_back(ws.positions[old_index]);
                if (has_colours) {
                    const Index base = old_index * 3;
                    new_colours.push_back(mesh.colours[base]);
                    new_colours.push_back(mesh.colours[base + 1]);
                    new_colours.push_back(mesh.colours[base + 2]);
                }
                if (has_confidence) {
                    new_confidence.push_back(mesh.confidence[old_index]);
                }
            }
            tri[k] = remap[old_index];
        }
    }

    std::vector<Eigen::Vector3i> new_faces;
    new_faces.reserve(ws.live_faces);
    for (std::size_t f = 0; f < face_count; ++f) {
        if (ws.face_alive[f] != 0) {
            new_faces.push_back(mesh.faces[f]);
        }
    }

    mesh.vertices.swap(new_vertices);
    mesh.faces.swap(new_faces);
    if (has_colours) {
        mesh.colours.swap(new_colours);
    }
    if (has_confidence) {
        mesh.confidence.swap(new_confidence);
    }
    return mesh.faces.size();
}

}  // namespace alphashape
