#pragma once

// Incremental 2-D Delaunay triangulation (Bowyer 1981; Watson 1981).
//
// Every orientation and in-circle decision is made with the exact predicates in
// predicates.hpp, so the result is a valid Delaunay triangulation for any finite
// input, including grids and cocircular sets. A triangle is replaced when the new
// site lies *strictly* inside its circumcircle; with that rule the replaced region
// is always star-shaped from the new site, so ties need no perturbation and the
// result is still Delaunay (one of the valid triangulations when several exist).
//
// The outside of the convex hull is covered by "ghost" triangles that share the
// vertex kGhost, so point location and insertion outside the hull need no
// bounding triangle. Sites are inserted along a Hilbert curve for locality, which
// makes the build deterministic for a given input.
//
// Exactly duplicated sites are merged: the triangulation uses the lowest index of
// each group, and site_alias() maps every input index to that representative.
// Fewer than three distinct sites, or all sites on one line, produce no triangles
// and degenerate() == true.

#include <array>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace alphashape {

class Delaunay2D {
public:
    /// Vertex id of the point at infinity shared by all ghost triangles.
    static constexpr int kGhost = -1;

    /// Vertices counter-clockwise. n[i] is the triangle across the edge opposite
    /// v[i], i.e. the edge (v[i+1], v[i+2]). A ghost triangle stores kGhost in v[2].
    struct Triangle {
        std::array<int, 3> v{};
        std::array<int, 3> n{};
    };

    explicit Delaunay2D(std::vector<Eigen::Vector2d> sites);

    bool degenerate() const { return degenerate_; }
    const std::vector<Eigen::Vector2d>& sites() const { return pts_; }

    /// Representative index of input site i (i itself unless it duplicates an
    /// earlier-indexed site at exactly the same coordinates).
    int site_alias(int i) const { return alias_[static_cast<std::size_t>(i)]; }

    /// Finite triangles, counter-clockwise, as input site indices.
    std::vector<std::array<int, 3>> triangles() const;

    /// Index of a finite triangle whose closure contains p, or -1 when p is
    /// outside the convex hull (or the triangulation is degenerate). `hint` is any
    /// triangle index to start the walk from (-1 = default).
    int locate(const Eigen::Vector2d& p, int hint = -1) const;

    const Triangle& triangle(int t) const { return tris_[static_cast<std::size_t>(t)]; }
    bool is_ghost(int t) const { return triangle(t).v[2] == kGhost; }
    bool is_alive(int t) const { return alive_[static_cast<std::size_t>(t)] != 0; }
    /// Size of the triangle store (alive and recycled slots).
    std::size_t triangle_slots() const { return tris_.size(); }

    /// Delaunay neighbours of representative site v, counter-clockwise.
    std::vector<int> vertex_neighbours(int v) const;
    /// Convex hull vertices, counter-clockwise (collinear hull sites included).
    std::vector<int> convex_hull() const;
    bool is_on_hull(int v) const;

private:
    int new_triangle(int a, int b, int c);
    int walk(const Eigen::Vector2d& p, int hint) const;
    bool in_conflict(int t, const Eigen::Vector2d& p) const;
    int insert(int site, int hint);

    std::vector<Eigen::Vector2d> pts_;
    std::vector<int> alias_;
    std::vector<int> vtri_;  // site -> one alive triangle containing it
    std::vector<Triangle> tris_;
    std::vector<char> alive_;
    std::vector<int> free_;
    std::vector<unsigned> mark_;
    unsigned stamp_ = 0;
    bool degenerate_ = false;
};

}  // namespace alphashape
