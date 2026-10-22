#include "model/mesh/mesh_ops.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/utils/error.hpp"

namespace alphashape {
namespace {

/// Undirected edge of a triangle, keyed by the ordered pair of vertex indices
/// (low, high) packed into one integer.
std::uint64_t edge_key(int a, int b) {
    const std::uint64_t lo = static_cast<std::uint64_t>(a < b ? a : b);
    const std::uint64_t hi = static_cast<std::uint64_t>(a < b ? b : a);
    return (lo << 32) | hi;
}

/// Disjoint-set forest over faces, used to find connected components under
/// shared-edge adjacency.
class UnionFind {
public:
    explicit UnionFind(std::size_t n) : parent_(n) {
        for (std::size_t i = 0; i < n; ++i) parent_[i] = i;
    }

    std::size_t find(std::size_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];  // path halving
            x = parent_[x];
        }
        return x;
    }

    void unite(std::size_t a, std::size_t b) {
        a = find(a);
        b = find(b);
        if (a != b) parent_[a] = b;
    }

private:
    std::vector<std::size_t> parent_;
};

/// Index the mesh's edges, counting face uses and linking faces that share one.
/// `first_face` maps each edge to the first face that used it, which is what
/// makes the "unite a face with the previous owner" pass equivalent to uniting
/// all faces incident to an edge.
void index_edges(const Mesh& mesh, std::unordered_map<std::uint64_t, std::size_t>& edge_count,
                 std::unordered_map<std::uint64_t, std::size_t>& first_face, UnionFind& uf) {
    for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
        const Eigen::Vector3i& face = mesh.faces[f];
        for (int e = 0; e < 3; ++e) {
            const std::uint64_t key = edge_key(face[e], face[(e + 1) % 3]);
            // Only merge faces at a genuinely shared edge: the two faces that
            // make up a coincident back-to-back pair would otherwise be counted
            // as one component.
            const auto inserted = first_face.emplace(key, f);
            if (inserted.second) {
                ++edge_count[key];  // this is the edge's first user
            } else {
                ++edge_count[key];
                uf.unite(f, inserted.first->second);
            }
        }
    }
}

}  // namespace

std::vector<Eigen::Vector3f> vertex_normals(const Mesh& mesh) {
    std::vector<Eigen::Vector3f> normals(mesh.vertices.size(), Eigen::Vector3f::Zero());
    for (const Eigen::Vector3i& face : mesh.faces) {
        if (face[0] < 0 || face[1] < 0 || face[2] < 0) continue;
        const std::size_t i0 = static_cast<std::size_t>(face[0]);
        const std::size_t i1 = static_cast<std::size_t>(face[1]);
        const std::size_t i2 = static_cast<std::size_t>(face[2]);
        if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() ||
            i2 >= mesh.vertices.size()) {
            continue;
        }
        // Unnormalised cross product: its length is twice the triangle area, so
        // summing these weights each face by area.
        const Eigen::Vector3d e0 = mesh.vertices[i1] - mesh.vertices[i0];
        const Eigen::Vector3d e1 = mesh.vertices[i2] - mesh.vertices[i0];
        const Eigen::Vector3d area_normal = e0.cross(e1);
        const Eigen::Vector3f contribution = area_normal.cast<float>();
        normals[i0] += contribution;
        normals[i1] += contribution;
        normals[i2] += contribution;
    }
    for (Eigen::Vector3f& n : normals) {
        const float len = n.norm();
        if (len > 0.0f) n /= len;
    }
    return normals;
}

MeshTopology analyse_topology(const Mesh& mesh) {
    MeshTopology topo;
    if (mesh.faces.empty()) return topo;

    std::unordered_map<std::uint64_t, std::size_t> edge_count;
    std::unordered_map<std::uint64_t, std::size_t> first_face;
    UnionFind uf(mesh.faces.size());
    index_edges(mesh, edge_count, first_face, uf);

    for (const auto& entry : edge_count) {
        if (entry.second == 1) ++topo.boundary_edges;
        else if (entry.second > 2) ++topo.nonmanifold_edges;
    }

    std::vector<int> root_seen(mesh.faces.size(), 0);
    for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
        const std::size_t root = uf.find(f);
        if (root_seen[root] == 0) {
            root_seen[root] = 1;
            ++topo.components;
        }
    }

    std::vector<int> referenced(mesh.vertices.size(), 0);
    std::size_t referenced_count = 0;
    for (const Eigen::Vector3i& face : mesh.faces) {
        for (int i = 0; i < 3; ++i) {
            const int v = face[i];
            if (v < 0 || static_cast<std::size_t>(v) >= mesh.vertices.size()) continue;
            const std::size_t uv = static_cast<std::size_t>(v);
            if (referenced[uv] == 0) {
                referenced[uv] = 1;
                ++referenced_count;
            }
        }
    }

    const long v = static_cast<long>(referenced_count);
    const long e = static_cast<long>(edge_count.size());
    const long f = static_cast<long>(mesh.faces.size());
    topo.euler = v - e + f;
    return topo;
}

std::size_t remove_small_components(Mesh& mesh, double min_fraction) {
    if (!(min_fraction >= 0.0 && min_fraction <= 1.0)) {
        throw ValueError("remove_small_components: min_fraction must be in [0, 1]");
    }
    if (mesh.faces.empty()) return 0;

    // Component id per face, via the same shared-edge union-find.
    UnionFind uf(mesh.faces.size());
    std::unordered_map<std::uint64_t, std::size_t> edge_count;
    std::unordered_map<std::uint64_t, std::size_t> first_face;
    index_edges(mesh, edge_count, first_face, uf);

    std::unordered_map<std::size_t, std::size_t> component_faces;
    for (std::size_t f = 0; f < mesh.faces.size(); ++f) ++component_faces[uf.find(f)];

    std::size_t largest = 0;
    for (const auto& entry : component_faces) largest = std::max(largest, entry.second);
    const double threshold = min_fraction * static_cast<double>(largest);

    std::vector<Eigen::Vector3i> kept;
    kept.reserve(mesh.faces.size());
    // uv_faces, when present, carries one entry per face, so dropping a face
    // must drop its coordinate triple too — otherwise the two arrays desync.
    const bool has_uv_faces = mesh.uv_faces.size() == mesh.faces.size();
    std::vector<Eigen::Vector3i> kept_uv_faces;
    if (has_uv_faces) kept_uv_faces.reserve(mesh.uv_faces.size());
    for (std::size_t f = 0; f < mesh.faces.size(); ++f) {
        const std::size_t count = component_faces[uf.find(f)];
        if (static_cast<double>(count) < threshold) continue;
        kept.push_back(mesh.faces[f]);
        if (has_uv_faces) kept_uv_faces.push_back(mesh.uv_faces[f]);
    }
    const std::size_t removed = mesh.faces.size() - kept.size();
    mesh.faces = std::move(kept);
    if (has_uv_faces) {
        mesh.uv_faces = std::move(kept_uv_faces);
    } else if (!mesh.uv_faces.empty()) {
        // A malformed uv_faces (wrong length) can no longer be tracked: drop it
        // rather than leave it indexing faces that no longer exist.
        mesh.uv_faces.clear();
    }
    remove_unreferenced_vertices(mesh);
    return removed;
}

void remove_unreferenced_vertices(Mesh& mesh) {
    const std::size_t n = mesh.vertices.size();
    if (n == 0) return;

    // Mark referenced vertices, then walk them in index order so the survivors
    // keep their original relative order (a compact mesh is an exact no-op).
    std::vector<int> referenced(n, 0);
    for (const Eigen::Vector3i& face : mesh.faces) {
        for (int i = 0; i < 3; ++i) {
            const int v = face[i];
            if (v >= 0 && static_cast<std::size_t>(v) < n) {
                referenced[static_cast<std::size_t>(v)] = 1;
            }
        }
    }

    std::vector<std::size_t> remap(n, static_cast<std::size_t>(-1));
    std::vector<Eigen::Vector3d> vertices;
    vertices.reserve(n);
    const bool has_colours = mesh.colours.size() == 3 * n;
    const bool has_confidence = mesh.confidence.size() == n;
    std::vector<std::uint8_t> colours;
    std::vector<float> confidence;
    if (has_colours) colours.reserve(mesh.colours.size());
    if (has_confidence) confidence.reserve(n);

    for (std::size_t v = 0; v < n; ++v) {
        if (referenced[v] == 0) continue;
        remap[v] = vertices.size();
        vertices.push_back(mesh.vertices[v]);
        if (has_colours) {
            colours.push_back(mesh.colours[3 * v + 0]);
            colours.push_back(mesh.colours[3 * v + 1]);
            colours.push_back(mesh.colours[3 * v + 2]);
        }
        if (has_confidence) confidence.push_back(mesh.confidence[v]);
    }

    for (Eigen::Vector3i& face : mesh.faces) {
        for (int i = 0; i < 3; ++i) {
            const int v = face[i];
            if (v < 0 || static_cast<std::size_t>(v) >= n) continue;
            face[i] = static_cast<int>(remap[static_cast<std::size_t>(v)]);
        }
    }

    mesh.vertices = std::move(vertices);
    // A non-empty attribute array with the wrong stride cannot be kept in sync
    // by remapping, so it is dropped rather than left indexing stale vertices.
    if (has_colours) mesh.colours = std::move(colours);
    else if (!mesh.colours.empty()) mesh.colours.clear();
    if (has_confidence) mesh.confidence = std::move(confidence);
    else if (!mesh.confidence.empty()) mesh.confidence.clear();
    // Textures uvs/uv_faces are indexed independently of `vertices`, so vertex
    // compaction must leave both alone: no uv is per-vertex any more.
}

}  // namespace alphashape
