#include "base/algorithm/delaunay2d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "base/algorithm/predicates.hpp"

namespace alphashape {

namespace {

using V = Eigen::Vector2d;

// Distance along a Hilbert curve of (x, y) on an n x n grid, n a power of two.
std::uint64_t hilbert_index(std::uint32_t x, std::uint32_t y, std::uint32_t n) {
    std::uint64_t d = 0;
    for (std::uint32_t s = n / 2; s > 0; s /= 2) {
        const std::uint32_t rx = (x & s) != 0 ? 1U : 0U;
        const std::uint32_t ry = (y & s) != 0 ? 1U : 0U;
        d += static_cast<std::uint64_t>(s) * s * ((3U * rx) ^ ry);
        if (ry == 0) {
            if (rx == 1) {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            std::swap(x, y);
        }
    }
    return d;
}

// p is known to be exactly collinear with a and b; is it strictly between them?
bool strictly_between(const V& a, const V& b, const V& p) {
    if (a.x() != b.x()) {
        return p.x() > std::min(a.x(), b.x()) && p.x() < std::max(a.x(), b.x());
    }
    return p.y() > std::min(a.y(), b.y()) && p.y() < std::max(a.y(), b.y());
}

int index_of(const Delaunay2D::Triangle& t, int v) {
    for (int i = 0; i < 3; ++i) {
        if (t.v[static_cast<std::size_t>(i)] == v) {
            return i;
        }
    }
    return -1;
}

std::size_t u(int i) { return static_cast<std::size_t>(i); }

}  // namespace

Delaunay2D::Delaunay2D(std::vector<V> sites) : pts_(std::move(sites)) {
    const int n = static_cast<int>(pts_.size());
    for (const V& p : pts_) {
        if (!std::isfinite(p.x()) || !std::isfinite(p.y())) {
            throw std::invalid_argument("Delaunay2D: non-finite site");
        }
    }
    alias_.resize(u(n));
    vtri_.assign(u(n), -1);

    // Merge exact duplicates; the lowest index of a group represents it.
    std::vector<int> order(u(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const V& pa = pts_[u(a)];
        const V& pb = pts_[u(b)];
        if (pa.x() != pb.x()) return pa.x() < pb.x();
        if (pa.y() != pb.y()) return pa.y() < pb.y();
        return a < b;
    });
    std::vector<int> unique;
    for (std::size_t k = 0; k < order.size(); ++k) {
        const int i = order[k];
        if (k > 0 && pts_[u(i)] == pts_[u(alias_[u(order[k - 1])])]) {
            alias_[u(i)] = alias_[u(order[k - 1])];
        } else {
            alias_[u(i)] = i;
            unique.push_back(i);
        }
    }
    if (unique.size() < 3) {
        degenerate_ = true;
        return;
    }

    // Hilbert order over the bounding box for walk locality.
    V lo = pts_[u(unique[0])];
    V hi = lo;
    for (int i : unique) {
        lo = lo.cwiseMin(pts_[u(i)]);
        hi = hi.cwiseMax(pts_[u(i)]);
    }
    constexpr std::uint32_t kGrid = 1U << 16;
    const V span = (hi - lo).cwiseMax(V::Constant(1e-300));
    std::vector<std::pair<std::uint64_t, int>> keyed;
    keyed.reserve(unique.size());
    for (int i : unique) {
        const V q = (pts_[u(i)] - lo).cwiseQuotient(span) * (kGrid - 1);
        keyed.emplace_back(hilbert_index(static_cast<std::uint32_t>(q.x()),
                                         static_cast<std::uint32_t>(q.y()), kGrid),
                           i);
    }
    std::sort(keyed.begin(), keyed.end());
    for (std::size_t k = 0; k < keyed.size(); ++k) {
        unique[k] = keyed[k].second;
    }

    // First triangle: the first two sites and the first site off their line.
    std::size_t third = 2;
    while (third < unique.size() &&
           orient2d(pts_[u(unique[0])], pts_[u(unique[1])], pts_[u(unique[third])]) == 0) {
        ++third;
    }
    if (third == unique.size()) {
        degenerate_ = true;
        return;
    }
    std::rotate(unique.begin() + 2, unique.begin() + static_cast<std::ptrdiff_t>(third),
                unique.begin() + static_cast<std::ptrdiff_t>(third) + 1);
    int a = unique[0];
    int b = unique[1];
    int c = unique[2];
    if (orient2d(pts_[u(a)], pts_[u(b)], pts_[u(c)]) < 0) {
        std::swap(b, c);
    }
    const int t0 = new_triangle(a, b, c);
    const int ga = new_triangle(c, b, kGhost);
    const int gb = new_triangle(a, c, kGhost);
    const int gc = new_triangle(b, a, kGhost);
    tris_[u(t0)].n = {ga, gb, gc};
    tris_[u(ga)].n = {gc, gb, t0};
    tris_[u(gb)].n = {ga, gc, t0};
    tris_[u(gc)].n = {gb, ga, t0};

    int hint = t0;
    for (std::size_t k = 3; k < unique.size(); ++k) {
        hint = insert(unique[k], hint);
    }
}

int Delaunay2D::new_triangle(int a, int b, int c) {
    // Keep the ghost vertex, if any, in slot 2 (a rotation keeps the orientation).
    if (a == kGhost) {
        std::swap(a, b);
        std::swap(b, c);
    } else if (b == kGhost) {
        std::swap(a, c);
        std::swap(b, c);
    }
    int t;
    if (!free_.empty()) {
        t = free_.back();
        free_.pop_back();
        alive_[u(t)] = 1;
    } else {
        t = static_cast<int>(tris_.size());
        tris_.emplace_back();
        alive_.push_back(1);
        mark_.push_back(0);
    }
    tris_[u(t)].v = {a, b, c};
    tris_[u(t)].n = {-1, -1, -1};
    for (int v : {a, b, c}) {
        if (v != kGhost) {
            vtri_[u(v)] = t;
        }
    }
    return t;
}

int Delaunay2D::walk(const V& p, int hint) const {
    int t = hint;
    if (t < 0 || u(t) >= tris_.size() || alive_[u(t)] == 0) {
        t = 0;
        while (alive_[u(t)] == 0) {
            ++t;
        }
    }
    if (tris_[u(t)].v[2] == kGhost) {
        t = tris_[u(t)].n[2];
    }
    // Visibility walk: cross any edge that has p strictly on its far side. In a
    // Delaunay triangulation this terminates for every start edge order; the
    // order is rotated per step only to avoid long zig-zags.
    const std::size_t limit = 4 * tris_.size() + 16;
    for (std::size_t step = 0; step < limit; ++step) {
        const Triangle& T = tris_[u(t)];
        if (T.v[2] == kGhost) {
            return t;  // crossed a hull edge: p is strictly outside it
        }
        int next = -1;
        for (int k = 0; k < 3; ++k) {
            const int i = static_cast<int>((step + static_cast<std::size_t>(k)) % 3);
            const V& a = pts_[u(T.v[u((i + 1) % 3)])];
            const V& b = pts_[u(T.v[u((i + 2) % 3)])];
            if (orient2d(a, b, p) < 0) {
                next = T.n[u(i)];
                break;
            }
        }
        if (next < 0) {
            return t;  // p is in the closed triangle
        }
        t = next;
    }
    throw std::logic_error("Delaunay2D: point location did not terminate");
}

bool Delaunay2D::in_conflict(int t, const V& p) const {
    const Triangle& T = tris_[u(t)];
    if (T.v[2] == kGhost) {
        // Ghost (z, y): conflict when p is strictly outside the hull edge, or on
        // its line strictly inside the segment (then the edge gets split).
        const V& z = pts_[u(T.v[0])];
        const V& y = pts_[u(T.v[1])];
        const int o = orient2d(z, y, p);
        return o > 0 || (o == 0 && strictly_between(z, y, p));
    }
    return incircle(pts_[u(T.v[0])], pts_[u(T.v[1])], pts_[u(T.v[2])], p) > 0;
}

int Delaunay2D::insert(int site, int hint) {
    const V& p = pts_[u(site)];
    const int start = walk(p, hint);

    if (++stamp_ == 0) {
        std::fill(mark_.begin(), mark_.end(), 0U);
        stamp_ = 1;
    }
    struct Edge {
        int a, b, outer, outer_slot;
    };
    std::vector<int> cavity{start};
    std::vector<Edge> boundary;
    mark_[u(start)] = stamp_;
    for (std::size_t k = 0; k < cavity.size(); ++k) {
        const Triangle T = tris_[u(cavity[k])];
        for (int i = 0; i < 3; ++i) {
            const int nb = T.n[u(i)];
            if (mark_[u(nb)] == stamp_) {
                continue;
            }
            if (in_conflict(nb, p)) {
                mark_[u(nb)] = stamp_;
                cavity.push_back(nb);
                continue;
            }
            const int a = T.v[u((i + 1) % 3)];
            const int b = T.v[u((i + 2) % 3)];
            const Triangle& N = tris_[u(nb)];
            int slot = -1;
            for (int j = 0; j < 3; ++j) {
                if (N.v[u((j + 1) % 3)] == b && N.v[u((j + 2) % 3)] == a) {
                    slot = j;
                }
            }
            if (slot < 0) {
                throw std::logic_error("Delaunay2D: inconsistent adjacency");
            }
            boundary.push_back({a, b, nb, slot});
        }
    }
    for (int t : cavity) {
        alive_[u(t)] = 0;
        free_.push_back(t);
    }

    std::vector<int> created;
    created.reserve(boundary.size());
    for (const Edge& e : boundary) {
        const int nt = new_triangle(e.a, e.b, site);
        tris_[u(nt)].n[u(index_of(tris_[u(nt)], site))] = e.outer;
        tris_[u(e.outer)].n[u(e.outer_slot)] = nt;
        created.push_back(nt);
    }
    // The new triangles form a fan around the site; link consecutive ones.
    for (std::size_t x = 0; x < created.size(); ++x) {
        Triangle& A = tris_[u(created[x])];
        for (int s = 0; s < 3; ++s) {
            if (A.n[u(s)] != -1) {
                continue;
            }
            const int a0 = A.v[u((s + 1) % 3)];
            const int a1 = A.v[u((s + 2) % 3)];
            for (std::size_t y = 0; y < created.size() && A.n[u(s)] == -1; ++y) {
                if (y == x) {
                    continue;
                }
                Triangle& B = tris_[u(created[y])];
                for (int r = 0; r < 3; ++r) {
                    if (B.v[u((r + 1) % 3)] == a1 && B.v[u((r + 2) % 3)] == a0) {
                        A.n[u(s)] = created[y];
                        B.n[u(r)] = created[x];
                        break;
                    }
                }
            }
        }
    }
    for (int t : created) {
        if (tris_[u(t)].v[2] != kGhost) {
            return t;
        }
    }
    return created.front();
}

std::vector<std::array<int, 3>> Delaunay2D::triangles() const {
    std::vector<std::array<int, 3>> out;
    for (std::size_t t = 0; t < tris_.size(); ++t) {
        if (alive_[t] != 0 && tris_[t].v[2] != kGhost) {
            out.push_back(tris_[t].v);
        }
    }
    return out;
}

int Delaunay2D::locate(const V& p, int hint) const {
    if (degenerate_) {
        return -1;
    }
    const int t = walk(p, hint);
    return tris_[u(t)].v[2] == kGhost ? -1 : t;
}

std::vector<int> Delaunay2D::vertex_neighbours(int v) const {
    std::vector<int> out;
    if (degenerate_ || v < 0 || u(v) >= vtri_.size() || vtri_[u(v)] < 0) {
        return out;
    }
    const int first = vtri_[u(v)];
    int t = first;
    do {
        const Triangle& T = tris_[u(t)];
        const int i = index_of(T, v);
        const int a = T.v[u((i + 1) % 3)];
        if (a != kGhost) {
            out.push_back(a);
        }
        t = T.n[u((i + 1) % 3)];  // across the edge (v, next counter-clockwise)
    } while (t != first);
    return out;
}

bool Delaunay2D::is_on_hull(int v) const {
    if (degenerate_ || v < 0 || u(v) >= vtri_.size() || vtri_[u(v)] < 0) {
        return false;
    }
    const int first = vtri_[u(v)];
    int t = first;
    do {
        const Triangle& T = tris_[u(t)];
        if (T.v[2] == kGhost) {
            return true;
        }
        t = T.n[u((index_of(T, v) + 1) % 3)];
    } while (t != first);
    return false;
}

std::vector<int> Delaunay2D::convex_hull() const {
    std::vector<int> out;
    if (degenerate_) {
        return out;
    }
    int start = -1;
    for (std::size_t t = 0; t < tris_.size() && start < 0; ++t) {
        if (alive_[t] != 0 && tris_[t].v[2] == kGhost) {
            start = static_cast<int>(t);
        }
    }
    int g = start;
    do {
        out.push_back(tris_[u(g)].v[1]);
        g = tris_[u(g)].n[1];
    } while (g != start);
    return out;
}

}  // namespace alphashape
