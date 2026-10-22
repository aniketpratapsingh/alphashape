#include "model/mesh/marching_tetrahedra.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/utils/error.hpp"
#include "model/mesh/mesh_ops.hpp"

namespace alphashape {
namespace {

// Cube corners, numbered so bit 0 -> +x, bit 1 -> +y, bit 2 -> +z: corner 0 is
// the low (0,0,0) corner and corner 7 the high (1,1,1) one.
constexpr int kDx[8] = {0, 1, 0, 1, 0, 1, 0, 1};
constexpr int kDy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
constexpr int kDz[8] = {0, 0, 0, 0, 1, 1, 1, 1};

// The six tetrahedra of a cube, all sharing the main diagonal 0-7:
//
//   {0,1,3,7} {0,3,2,7} {0,2,6,7} {0,6,4,7} {0,4,5,7} {0,5,1,7}
//
// The same split is used in every cube. Adjacent cubes therefore cut their
// shared face along the same diagonal, and the surface crosses cube borders
// without cracks. (Each interior face of the split is shared by two of the six
// tetrahedra, but with opposite winding, so a plane segment crossing one never
// turns into a duplicated interior wall.)
constexpr int kTets[6][4] = {
    {0, 1, 3, 7},
    {0, 3, 2, 7},
    {0, 2, 6, 7},
    {0, 6, 4, 7},
    {0, 4, 5, 7},
    {0, 5, 1, 7},
};

/// Bit-exact hash / equality for vertex positions, used to merge the (rare)
/// coincident vertices a corner sitting exactly on the iso value produces.
struct PositionHash {
    std::size_t operator()(const Eigen::Vector3d& p) const {
        std::uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < 3; ++i) {
            const double v = p[i] == 0.0 ? 0.0 : p[i];  // fold -0.0 onto 0.0
            std::uint64_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            h = (h ^ bits) * 1099511628211ULL;
        }
        return static_cast<std::size_t>(h);
    }
};

struct PositionEqual {
    bool operator()(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }
};

/// Iso-surface extraction over one grid. Holds the growing mesh and the two
/// vertex caches, so the whole extraction shares one lookup table.
class IsoExtractor {
public:
    IsoExtractor(const VoxelGrid<float>& field, float iso) : field_(&field), iso_(iso) {}

    Mesh run(const std::vector<std::uint8_t>* mask) {
        const GridSpec& spec = field_->spec;
        if (spec.nx >= 2 && spec.ny >= 2 && spec.nz >= 2) {
            // Deterministic order: z slowest, then y, then x fastest.
            for (int z = 0; z + 1 < spec.nz; ++z)
                for (int y = 0; y + 1 < spec.ny; ++y)
                    for (int x = 0; x + 1 < spec.nx; ++x) cube(spec, x, y, z, mask);
        }
        remove_unreferenced_vertices(mesh_);
        return std::move(mesh_);
    }

private:
    /// Load one cube's eight corner samples. Returns without extracting when a
    /// masked corner trims the cube, or when no tetrahedron can cross.
    void cube(const GridSpec& spec, int x, int y, int z, const std::vector<std::uint8_t>* mask) {
        int n_inside = 0;
        for (int c = 0; c < 8; ++c) {
            const int cx = x + kDx[c];
            const int cy = y + kDy[c];
            const int cz = z + kDz[c];
            const std::size_t idx = spec.index(cx, cy, cz);
            if (mask != nullptr && (*mask)[idx] == 0) return;
            gidx_[c] = idx;
            pos_[c] = spec.centre(cx, cy, cz);
            val_[c] = field_->data[idx];
            inside_[c] = val_[c] < iso_ ? 1 : 0;
            n_inside += inside_[c];
        }
        if (n_inside == 0 || n_inside == 8) return;
        for (int t = 0; t < 6; ++t) tetrahedron(t);
    }

    /// Polygonise one tetrahedron: one corner on either side -> a triangle; two
    /// on each side -> a quad, emitted as two triangles.
    void tetrahedron(int t) {
        const int* c = kTets[t];
        int n_inside = 0;
        for (int i = 0; i < 4; ++i) n_inside += inside_[c[i]];
        if (n_inside == 0 || n_inside == 4) return;

        Eigen::Vector3d sum_in = Eigen::Vector3d::Zero();
        Eigen::Vector3d sum_out = Eigen::Vector3d::Zero();
        int lone_in = -1;
        int lone_out = -1;
        for (int i = 0; i < 4; ++i) {
            if (inside_[c[i]]) {
                sum_in += pos_[c[i]];
                lone_in = c[i];
            } else {
                sum_out += pos_[c[i]];
                lone_out = c[i];
            }
        }
        const double n_in = static_cast<double>(n_inside);
        // Direction of increasing field, estimated from the corner centroids on
        // each side. Used only to orient the emitted triangles.
        const Eigen::Vector3d up = sum_out / (4.0 - n_in) - sum_in / n_in;

        if (n_inside == 1) {
            emit(vertex_on_edge(lone_in, successor(c, lone_in, 1)),
                 vertex_on_edge(lone_in, successor(c, lone_in, 2)),
                 vertex_on_edge(lone_in, successor(c, lone_in, 3)), up);
        } else if (n_inside == 3) {
            emit(vertex_on_edge(lone_out, successor(c, lone_out, 1)),
                 vertex_on_edge(lone_out, successor(c, lone_out, 2)),
                 vertex_on_edge(lone_out, successor(c, lone_out, 3)), up);
        } else {
            // Two inside (i0, i1) and two outside (o0, o1): every crossing edge
            // joins an inside to an outside corner, and the quad's winding is
            // i0-o0, i0-o1, i1-o1, i1-o0.
            int i0 = -1;
            int i1 = -1;
            int o0 = -1;
            int o1 = -1;
            for (int k = 0; k < 4; ++k) {
                if (inside_[c[k]]) {
                    if (i0 < 0) i0 = c[k]; else i1 = c[k];
                } else {
                    if (o0 < 0) o0 = c[k]; else o1 = c[k];
                }
            }
            const int p0 = vertex_on_edge(i0, o0);
            const int p1 = vertex_on_edge(i0, o1);
            const int p2 = vertex_on_edge(i1, o1);
            const int p3 = vertex_on_edge(i1, o0);
            emit(p0, p1, p2, up);
            emit(p0, p2, p3, up);
        }
    }

    /// k-th corner of the tetrahedron after `corner` (k = 1..3), wrapping around.
    static int successor(const int* c, int corner, int k) {
        int at = 0;
        for (int i = 0; i < 4; ++i)
            if (c[i] == corner) at = i;
        return c[(at + k) % 4];
    }

    /// Append a triangle oriented so its normal points toward increasing field.
    /// Coincident or collinear vertices make a zero-area triangle; those are
    /// dropped rather than emitted.
    void emit(int v0, int v1, int v2, const Eigen::Vector3d& up) {
        if (v0 == v1 || v1 == v2 || v0 == v2) return;
        const Eigen::Vector3d& a = mesh_.vertices[static_cast<std::size_t>(v0)];
        const Eigen::Vector3d& b = mesh_.vertices[static_cast<std::size_t>(v1)];
        const Eigen::Vector3d& d = mesh_.vertices[static_cast<std::size_t>(v2)];
        const Eigen::Vector3d normal = (b - a).cross(d - a);
        if (normal.squaredNorm() == 0.0) return;
        if (normal.dot(up) < 0.0) std::swap(v1, v2);
        mesh_.faces.emplace_back(v0, v1, v2);
    }

    /// The vertex where the segment between cube corners `a` and `b` crosses the
    /// iso value: linear interpolation of position by field value. Keyed by the
    /// pair of grid corner indices, so all cubes meeting at this grid edge get
    /// the same vertex.
    int vertex_on_edge(int a, int b) {
        const std::size_t ga = gidx_[a];
        const std::size_t gb = gidx_[b];
        const std::size_t lo = ga < gb ? ga : gb;
        const std::size_t hi = ga < gb ? gb : ga;
        const std::uint64_t key =
            (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
        const auto found = edge_vertex_.find(key);
        if (found != edge_vertex_.end()) return found->second;

        const int inside = inside_[a] ? a : b;
        const int outside = inside_[a] ? b : a;
        const double vi = static_cast<double>(val_[inside]);
        const double vo = static_cast<double>(val_[outside]);
        double t = 0.5;
        if (vi != vo) t = (static_cast<double>(iso_) - vi) / (vo - vi);
        if (!(t >= 0.0 && t <= 1.0)) t = 0.5;  // NaN-safe clamp
        const Eigen::Vector3d p = pos_[inside] + t * (pos_[outside] - pos_[inside]);

        const auto existing = position_vertex_.find(p);
        const int id = existing != position_vertex_.end()
                           ? existing->second
                           : static_cast<int>(mesh_.vertices.size());
        if (existing == position_vertex_.end()) {
            mesh_.vertices.push_back(p);
            position_vertex_.emplace(p, id);
        }
        edge_vertex_.emplace(key, id);
        return id;
    }

    const VoxelGrid<float>* field_;
    float iso_;
    Mesh mesh_;
    /// Grid edge (min,max corner index) -> vertex id.
    std::unordered_map<std::uint64_t, int> edge_vertex_;
    /// Exact vertex position -> vertex id, so coincident crossings merge.
    std::unordered_map<Eigen::Vector3d, int, PositionHash, PositionEqual> position_vertex_;
    // Per-cube scratch.
    std::size_t gidx_[8]{};
    Eigen::Vector3d pos_[8];
    float val_[8]{};
    int inside_[8]{};
};

}  // namespace

Mesh extract_isosurface(const VoxelGrid<float>& field, float iso,
                        const std::vector<std::uint8_t>* mask) {
    if (field.data.size() != field.spec.size()) {
        throw ValueError("extract_isosurface: field data size does not match the grid spec");
    }
    if (mask != nullptr && mask->size() != field.spec.size()) {
        throw ValueError("extract_isosurface: mask size does not match the grid spec");
    }
    return IsoExtractor(field, iso).run(mask);
}

}  // namespace alphashape
