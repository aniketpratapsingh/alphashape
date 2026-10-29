#include "common/algorithm/elevation/sibson.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "base/algorithm/predicates.hpp"

namespace alphashape::elevation {

namespace {

using V = Eigen::Vector2d;

std::size_t u(int i) { return static_cast<std::size_t>(i); }

// Circumcentre of the triangle (a, b, c), all given relative to the query so
// the coordinates stay small and the rounding error stays relative to the
// local feature size.
V circumcentre(const V& a, const V& b, const V& c) {
    const double d = 2.0 * (a.x() * (b.y() - c.y()) + b.x() * (c.y() - a.y()) +
                            c.x() * (a.y() - b.y()));
    const double a2 = a.squaredNorm();
    const double b2 = b.squaredNorm();
    const double c2 = c.squaredNorm();
    return {(a2 * (b.y() - c.y()) + b2 * (c.y() - a.y()) + c2 * (a.y() - b.y())) / d,
            (a2 * (c.x() - b.x()) + b2 * (a.x() - c.x()) + c2 * (b.x() - a.x())) / d};
}

double polygon_area(const std::vector<V>& poly) {
    double twice = 0.0;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const V& p = poly[i];
        const V& r = poly[(i + 1) % poly.size()];
        twice += p.x() * r.y() - r.x() * p.y();
    }
    return 0.5 * std::abs(twice);
}

int index_of(const Delaunay2D::Triangle& t, int v) {
    for (int i = 0; i < 3; ++i) {
        if (t.v[u(i)] == v) {
            return i;
        }
    }
    return -1;
}

struct BoundaryEdge {
    int a = -1;      // edge a -> b, counter-clockwise around the cavity
    int b = -1;
    int owner = -1;  // cavity triangle that has the edge
};

// Linear weights of q in triangle t (used on the hull boundary).
void barycentric(const Delaunay2D& dt, int t, const V& q, std::vector<SibsonWeight>& out) {
    const auto& T = dt.triangle(t);
    const auto& P = dt.sites();
    const V& a = P[u(T.v[0])];
    const V& b = P[u(T.v[1])];
    const V& c = P[u(T.v[2])];
    const auto cross = [](const V& x, const V& y, const V& z) {
        return (y.x() - x.x()) * (z.y() - x.y()) - (y.y() - x.y()) * (z.x() - x.x());
    };
    const double total = cross(a, b, c);
    const std::array<double, 3> w = {cross(q, b, c) / total, cross(a, q, c) / total,
                                     cross(a, b, q) / total};
    for (std::size_t i = 0; i < 3; ++i) {
        if (w[i] > 0.0) {
            out.push_back({T.v[i], w[i]});
        }
    }
}

}  // namespace

bool sibson_coordinates(const Delaunay2D& dt, const V& q, std::vector<SibsonWeight>& out,
                        int& hint) {
    out.clear();
    const int t0 = dt.locate(q, hint);
    if (t0 < 0) {
        return false;
    }
    hint = t0;
    const auto& P = dt.sites();
    {
        const auto& T = dt.triangle(t0);
        for (int v : T.v) {
            if (P[u(v)] == q) {
                out.push_back({v, 1.0});
                return true;
            }
        }
    }

    // The triangles q would replace: connected, star-shaped from q.
    std::vector<int> cavity{t0};
    std::vector<int> seen{t0};
    std::vector<BoundaryEdge> boundary;
    for (std::size_t k = 0; k < cavity.size(); ++k) {
        const auto& T = dt.triangle(cavity[k]);
        for (int i = 0; i < 3; ++i) {
            const int nb = T.n[u(i)];
            const int a = T.v[u((i + 1) % 3)];
            const int b = T.v[u((i + 2) % 3)];
            if (std::find(seen.begin(), seen.end(), nb) != seen.end()) {
                if (std::find(cavity.begin(), cavity.end(), nb) == cavity.end()) {
                    boundary.push_back({a, b, cavity[k]});
                }
                continue;
            }
            seen.push_back(nb);
            if (dt.is_ghost(nb)) {
                if (orient2d(P[u(a)], P[u(b)], q) == 0) {
                    // q lies on the hull edge: its cell is unbounded, and the
                    // Sibson coordinates reduce to the linear ones along the edge.
                    out.clear();
                    barycentric(dt, t0, q, out);
                    return true;
                }
                boundary.push_back({a, b, cavity[k]});
                continue;
            }
            const auto& N = dt.triangle(nb);
            if (incircle(P[u(N.v[0])], P[u(N.v[1])], P[u(N.v[2])], q) > 0) {
                cavity.push_back(nb);
            } else {
                boundary.push_back({a, b, cavity[k]});
            }
        }
    }

    // Order the boundary into a counter-clockwise cycle of neighbours.
    const std::size_t m = boundary.size();
    std::vector<BoundaryEdge> ring;
    ring.reserve(m);
    ring.push_back(boundary.front());
    while (ring.size() < m) {
        const int tail = ring.back().b;
        const auto next = std::find_if(boundary.begin(), boundary.end(),
                                       [&](const BoundaryEdge& e) { return e.a == tail; });
        ring.push_back(*next);
    }

    // New Voronoi vertices of q: circumcentres of (q, a, b) per boundary edge.
    std::vector<V> fresh(m);
    for (std::size_t i = 0; i < m; ++i) {
        fresh[i] = circumcentre(V::Zero(), P[u(ring[i].a)] - q, P[u(ring[i].b)] - q);
    }

    double total = 0.0;
    std::vector<V> poly;
    for (std::size_t i = 0; i < m; ++i) {
        // Neighbour s = ring[i].b, between edges ring[i] (in) and ring[i+1] (out).
        const BoundaryEdge& in = ring[i];
        const BoundaryEdge& outgoing = ring[(i + 1) % m];
        const int s = in.b;
        poly.clear();
        poly.push_back(fresh[i]);
        int t = in.owner;
        for (std::size_t guard = 0; guard <= cavity.size(); ++guard) {
            const auto& T = dt.triangle(t);
            poly.push_back(circumcentre(P[u(T.v[0])] - q, P[u(T.v[1])] - q, P[u(T.v[2])] - q));
            const int j = index_of(T, s);
            if (T.v[u((j + 1) % 3)] == outgoing.b) {
                break;  // this triangle owns the outgoing edge (s, next)
            }
            t = T.n[u((j + 2) % 3)];  // rotate around s across the edge (s, v[j+1])
        }
        poly.push_back(fresh[(i + 1) % m]);
        const double area = polygon_area(poly);
        out.push_back({s, area});
        total += area;
    }
    for (SibsonWeight& w : out) {
        w.weight /= total;
    }
    return true;
}

std::size_t exact_sibson_fill(int width, int height, const std::vector<float>& values,
                              const std::vector<std::uint8_t>& valid,
                              const std::vector<std::uint8_t>& fill,
                              std::vector<float>& result, int max_sites) {
    const std::size_t W = static_cast<std::size_t>(width);
    const std::size_t H = static_cast<std::size_t>(height);
    const std::size_t total = W * H;
    std::vector<std::uint8_t> seen(total, 0);
    std::vector<std::size_t> comp;
    std::vector<std::size_t> sites;
    std::size_t exact = 0;
    for (std::size_t start = 0; start < total; ++start) {
        if (fill[start] == 0 || seen[start] != 0) {
            continue;
        }
        comp.assign(1, start);
        sites.clear();
        seen[start] = 1;
        for (std::size_t k = 0; k < comp.size(); ++k) {
            const std::size_t x = comp[k] % W;
            const std::size_t y = comp[k] / W;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx == 0 && dy == 0) || (dx < 0 && x == 0) || (dy < 0 && y == 0) ||
                        (dx > 0 && x + 1 == W) || (dy > 0 && y + 1 == H)) {
                        continue;
                    }
                    const std::size_t j = (y + static_cast<std::size_t>(dy + 1) - 1) * W +
                                          (x + static_cast<std::size_t>(dx + 1) - 1);
                    if (valid[j] != 0) {
                        sites.push_back(j);
                    } else if (dx == 0 || dy == 0) {
                        if (fill[j] != 0 && seen[j] == 0) {
                            seen[j] = 1;
                            comp.push_back(j);
                        }
                    }
                }
            }
        }
        std::sort(sites.begin(), sites.end());
        sites.erase(std::unique(sites.begin(), sites.end()), sites.end());
        if (sites.size() < 3 || sites.size() > static_cast<std::size_t>(max_sites)) {
            continue;
        }
        std::vector<V> pts;
        pts.reserve(sites.size());
        for (std::size_t j : sites) {
            pts.emplace_back(static_cast<double>(j % W), static_cast<double>(j / W));
        }
        const Delaunay2D dt(std::move(pts));
        if (dt.degenerate()) {
            continue;
        }
        const auto ncomp = static_cast<std::int64_t>(comp.size());
        std::size_t done = 0;
#ifdef _OPENMP
#pragma omp parallel reduction(+ : done)
#endif
        {
            std::vector<SibsonWeight> w;
            int hint = -1;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (std::int64_t k = 0; k < ncomp; ++k) {
                const std::size_t i = comp[static_cast<std::size_t>(k)];
                const V q(static_cast<double>(i % W), static_cast<double>(i / W));
                if (!sibson_coordinates(dt, q, w, hint)) {
                    continue;
                }
                double v = 0.0;
                for (const SibsonWeight& s : w) {
                    v += s.weight * static_cast<double>(values[sites[u(s.site)]]);
                }
                result[i] = static_cast<float>(v);
                ++done;
            }
        }
        exact += done;
    }
    return exact;
}

}  // namespace alphashape::elevation
