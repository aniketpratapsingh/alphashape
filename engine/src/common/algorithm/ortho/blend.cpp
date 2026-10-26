// Orthomosaic blending: see blend.hpp for the contract.

#include "common/algorithm/ortho/blend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape::ortho {
namespace {

/// Read tile `tile`'s validity at output cell (col, row). A cropped tile is
/// exactly "not valid" outside its box, which is the zero the full-grid layout
/// used to hold there.
bool tile_valid(const OrthoTile& tile, int col, int row) {
    return tile.contains(col, row) && tile.valid[tile.index(col, row)] != 0;
}

/// The tile's colour at output cell (col, row), 0 outside its crop.
std::uint8_t tile_rgb(const OrthoTile& tile, int col, int row, int channel) {
    return tile.contains(col, row)
               ? tile.rgb[tile.index(col, row) * 3 + static_cast<std::size_t>(channel)]
               : std::uint8_t{0};
}

/// The tile's score at output cell (col, row), 0 outside its crop.
float tile_score(const OrthoTile& tile, int col, int row) {
    return tile.contains(col, row) ? tile.score[tile.index(col, row)] : 0.0F;
}

/// A single-channel float plane on the output grid (or a pyramid level of it).
struct Plane {
    int w = 0;
    int h = 0;
    std::vector<float> v;

    float& at(int x, int y) { return v[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)]; }
    float at(int x, int y) const { return v[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)]; }
};

Plane make_plane(int w, int h, float fill = 0.0F) {
    Plane p;
    p.w = w;
    p.h = h;
    p.v.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), fill);
    return p;
}

/// 5-tap Gaussian [1,4,6,4,1]/16, separable, edges clamp.
Plane blur5(const Plane& in) {
    const float k[5] = {1.0F / 16.0F, 4.0F / 16.0F, 6.0F / 16.0F, 4.0F / 16.0F,
                        1.0F / 16.0F};
    Plane tmp = make_plane(in.w, in.h);
    for (int y = 0; y < in.h; ++y) {
        for (int x = 0; x < in.w; ++x) {
            float acc = 0.0F;
            for (int m = -2; m <= 2; ++m) {
                acc += k[m + 2] * in.at(std::clamp(x + m, 0, in.w - 1), y);
            }
            tmp.at(x, y) = acc;
        }
    }
    Plane out = make_plane(in.w, in.h);
    for (int y = 0; y < in.h; ++y) {
        for (int x = 0; x < in.w; ++x) {
            float acc = 0.0F;
            for (int m = -2; m <= 2; ++m) {
                acc += k[m + 2] * tmp.at(x, std::clamp(y + m, 0, in.h - 1));
            }
            out.at(x, y) = acc;
        }
    }
    return out;
}

/// Blur then decimate by two: size ceil(w/2) x ceil(h/2).
Plane reduce(const Plane& in) {
    const Plane blurred = blur5(in);
    Plane out = make_plane((in.w + 1) / 2, (in.h + 1) / 2);
    for (int y = 0; y < out.h; ++y) {
        for (int x = 0; x < out.w; ++x) {
            out.at(x, y) = blurred.at(std::min(2 * x, in.w - 1), std::min(2 * y, in.h - 1));
        }
    }
    return out;
}

/// Bilinear resize to (dw, dh), edges clamp. Used as the pyramid expand: the
/// Laplacian is defined against this same operator, so collapse is exact.
///
/// The mapping is the "insertion" one (output i -> input i * in.w/dw), which is
/// the exact counterpoint to reduce() taking input sample 2*x for output x: expand
/// puts output x back at input x*2, so the pyramid levels stay pixel-aligned. A
/// half-pixel-aligned map here would leave the coarse levels shifted and the
/// Laplacian would carry real gradients instead of fine detail.
Plane resize(const Plane& in, int dw, int dh) {
    Plane out = make_plane(dw, dh);
    const double sx = static_cast<double>(in.w) / static_cast<double>(dw);
    const double sy = static_cast<double>(in.h) / static_cast<double>(dh);
    for (int y = 0; y < dh; ++y) {
        const double fy = static_cast<double>(y) * sy;
        const int y0 = static_cast<int>(std::floor(fy));
        const double ay = fy - static_cast<double>(y0);
        for (int x = 0; x < dw; ++x) {
            const double fx = static_cast<double>(x) * sx;
            const int x0 = static_cast<int>(std::floor(fx));
            const double ax = fx - static_cast<double>(x0);
            const int cx0 = std::clamp(x0, 0, in.w - 1);
            const int cx1 = std::clamp(x0 + 1, 0, in.w - 1);
            const int cy0 = std::clamp(y0, 0, in.h - 1);
            const int cy1 = std::clamp(y0 + 1, 0, in.h - 1);
            const double top = (1.0 - ax) * static_cast<double>(in.at(cx0, cy0)) +
                               ax * static_cast<double>(in.at(cx1, cy0));
            const double bot = (1.0 - ax) * static_cast<double>(in.at(cx0, cy1)) +
                               ax * static_cast<double>(in.at(cx1, cy1));
            out.at(x, y) = static_cast<float>((1.0 - ay) * top + ay * bot);
        }
    }
    return out;
}

/// Number of pyramid levels: floor(log2(min(W,H)/8)), capped at 6.
int level_count(int width, int height) {
    const int m = std::min(width, height);
    int levels = 0;
    while ((m >> (levels + 1)) >= 8 && levels < 6) {
        ++levels;
    }
    // Also stop when a level would be a single pixel.
    int w = width, h = height;
    int cap = 0;
    while (w > 1 && h > 1) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        ++cap;
    }
    return std::min(levels, cap);
}

std::vector<Plane> gaussian_pyramid(Plane base, int levels) {
    std::vector<Plane> pyramid;
    pyramid.push_back(std::move(base));
    for (int k = 0; k < levels; ++k) {
        pyramid.push_back(reduce(pyramid.back()));
    }
    return pyramid;
}

/// Mask (weight) pyramid. Level 0 is softened with one Gaussian pass: a plain
/// binary level-0 mask hard-switches the finest Laplacian detail at the seam,
/// which is exactly the step the spline exists to remove. The coarser levels
/// come from the usual blur-and-decimate.
std::vector<Plane> weight_pyramid(Plane base, int levels) {
    std::vector<Plane> pyramid;
    pyramid.push_back(blur5(base));
    for (int k = 0; k < levels; ++k) {
        pyramid.push_back(reduce(pyramid.back()));
    }
    return pyramid;
}

/// Laplacian pyramid: L_k = G_k - expand(G_{k+1}), top level is G_top.
std::vector<Plane> laplacian_pyramid(const std::vector<Plane>& gauss) {
    std::vector<Plane> laplacian;
    for (std::size_t k = 0; k + 1 < gauss.size(); ++k) {
        const Plane up = resize(gauss[k + 1], gauss[k].w, gauss[k].h);
        Plane l = make_plane(gauss[k].w, gauss[k].h);
        for (std::size_t i = 0; i < l.v.size(); ++i) {
            l.v[i] = gauss[k].v[i] - up.v[i];
        }
        laplacian.push_back(std::move(l));
    }
    laplacian.push_back(gauss.back());
    return laplacian;
}

/// Grow valid colour outward into the invalid cells of `p`, one ring per pass,
/// averaging whatever valid 8-neighbours exist. The Gaussian pyramid of a plane
/// that is zero outside the tile's footprint would drag the coarse levels toward
/// zero near every validity edge, leaving a dark halo along the seam. Extending
/// the real colour outward keeps the multiscale decomposition free of that
/// artificial step. Deterministic: each pass only reads the previous pass's
/// state, and cells are visited in scan order.
void fill_invalid(Plane& p, const std::vector<std::uint8_t>& valid) {
    const int w = p.w;
    const int h = p.h;
    std::vector<std::uint8_t> known(valid.begin(), valid.end());
    std::vector<std::uint8_t> next = known;
    std::size_t remaining = 0;
    for (std::size_t i = 0; i < known.size(); ++i) {
        if (known[i] == 0) {
            ++remaining;
        }
    }
    while (remaining > 0) {
        bool progressed = false;
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                const std::size_t cell =
                    static_cast<std::size_t>(row) * static_cast<std::size_t>(w) +
                    static_cast<std::size_t>(col);
                if (known[cell] != 0) {
                    continue;
                }
                double sum = 0.0;
                int count = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const int nx = col + dx;
                        const int ny = row + dy;
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
                            continue;
                        }
                        const std::size_t nc =
                            static_cast<std::size_t>(ny) * static_cast<std::size_t>(w) +
                            static_cast<std::size_t>(nx);
                        if (known[nc] != 0) {
                            sum += static_cast<double>(p.v[nc]);
                            ++count;
                        }
                    }
                }
                if (count > 0) {
                    p.v[cell] = static_cast<float>(sum / static_cast<double>(count));
                    next[cell] = 1;
                    --remaining;
                    progressed = true;
                }
            }
        }
        if (!progressed) {
            break;  // nothing valid to extend from
        }
        known = next;
    }
}

}  // namespace

OrthoMosaic blend(int width, int height, const std::vector<OrthoTile>& tiles,
                  const std::vector<std::int32_t>& labels, Blending mode,
                  int pyramid_levels) {
    OrthoMosaic out;
    out.width = width;
    out.height = height;
    const std::size_t cells =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    out.rgb.assign(cells * 3, 0);
    out.valid.assign(cells, 0);
    out.labels = labels;
    out.labels.resize(cells, -1);
    if (width <= 0 || height <= 0 || tiles.empty()) {
        return out;
    }

    if (mode == Blending::Disabled) {
        for (std::size_t cell = 0; cell < cells; ++cell) {
            const std::int32_t label = out.labels[cell];
            if (label < 0 || label >= static_cast<std::int32_t>(tiles.size())) {
                continue;
            }
            const OrthoTile& tile = tiles[static_cast<std::size_t>(label)];
            const int col = static_cast<int>(cell % static_cast<std::size_t>(width));
            const int row = static_cast<int>(cell / static_cast<std::size_t>(width));
            if (!tile_valid(tile, col, row)) {
                continue;
            }
            for (int c = 0; c < 3; ++c) {
                out.rgb[cell * 3 + static_cast<std::size_t>(c)] = tile_rgb(tile, col, row, c);
            }
            out.valid[cell] = 1;
        }
        return out;
    }

    if (mode == Blending::Average) {
        for (std::size_t cell = 0; cell < cells; ++cell) {
            const int col = static_cast<int>(cell % static_cast<std::size_t>(width));
            const int row = static_cast<int>(cell / static_cast<std::size_t>(width));
            double weight_sum = 0.0;
            double acc[3] = {0.0, 0.0, 0.0};
            for (const OrthoTile& tile : tiles) {
                if (!tile_valid(tile, col, row)) {
                    continue;
                }
                const double w = std::max(static_cast<double>(tile_score(tile, col, row)), 1e-6);
                weight_sum += w;
                for (int c = 0; c < 3; ++c) {
                    acc[c] += w * static_cast<double>(tile_rgb(tile, col, row, c));
                }
            }
            if (weight_sum <= 0.0) {
                continue;
            }
            for (int c = 0; c < 3; ++c) {
                const double value = acc[c] / weight_sum;
                out.rgb[cell * 3 + static_cast<std::size_t>(c)] =
                    static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0, 255.0)));
            }
            out.valid[cell] = 1;
        }
        return out;
    }

    // -- Mosaic: Burt & Adelson multiresolution spline ----------------------
    // Only tiles that label some cell participate.
    std::vector<char> used(tiles.size(), 0);
    for (std::size_t cell = 0; cell < cells; ++cell) {
        const std::int32_t label = out.labels[cell];
        if (label >= 0 && label < static_cast<std::int32_t>(tiles.size())) {
            const int col = static_cast<int>(cell % static_cast<std::size_t>(width));
            const int row = static_cast<int>(cell / static_cast<std::size_t>(width));
            if (tile_valid(tiles[static_cast<std::size_t>(label)], col, row)) {
                used[static_cast<std::size_t>(label)] = 1;
            }
        }
    }

    int levels = level_count(width, height);
    if (pyramid_levels >= 0) {
        // A pinned depth must still be representable on this grid: a level
        // larger than the grid itself would decimate to 1x1 and the collapse
        // would then be a resize of a single sample, not the operator the
        // whole grid applied.
        levels = std::min(pyramid_levels, level_count(width, height));
    }

    // Per-channel fused Laplacian pyramid. The masks partition the covered cells
    // (each covered cell has exactly one label), and the Gaussian operator keeps
    // that partition: sum_t G_k(mask_t) = G_k(1) = 1 at every level. That is the
    // reason the classic Burt & Adelson spline needs no per-level normalisation;
    // dividing by a summed denominator here would make the collapse nonlinear and
    // reintroduce the steps it is meant to remove.
    std::vector<std::vector<Plane>> num(3);
    for (int c = 0; c < 3; ++c) {
        num[static_cast<std::size_t>(c)].reserve(static_cast<std::size_t>(levels) + 1);
        for (int k = 0; k <= levels; ++k) {
            const int lw = std::max(1, (width + (1 << k) - 1) >> k);
            const int lh = std::max(1, (height + (1 << k) - 1) >> k);
            num[static_cast<std::size_t>(c)].push_back(make_plane(lw, lh));
        }
    }

    // Reused across tiles, so the blend never holds one validity copy per camera.
    std::vector<std::uint8_t> scratch_mask(cells, 0);

    for (std::size_t t = 0; t < tiles.size(); ++t) {
        if (used[t] == 0) {
            continue;
        }
        const OrthoTile& tile = tiles[t];

        // Per-channel plane (the tile's colour over its valid area, zero
        // elsewhere) and the binary label mask. Both are full-grid planes: the
        // pyramid's blur reads across the tile's validity edge, so the plane
        // cannot be cropped to the tile box without changing the result.
        std::vector<Plane> gauss_ch(3);
        for (int c = 0; c < 3; ++c) {
            gauss_ch[static_cast<std::size_t>(c)] = make_plane(width, height);
        }
        Plane mask = make_plane(width, height);
        // One reusable validity mask for fill_invalid(): rewritten over the
        // tile's crop each iteration and cleared afterwards, so the blend holds
        // no per-tile full-grid copy at all.
        std::vector<std::uint8_t>& tile_mask = scratch_mask;
        for (int row = tile.y0; row < tile.y0 + tile.height; ++row) {
            for (int col = tile.x0; col < tile.x0 + tile.width; ++col) {
                const std::size_t buf = tile.index(col, row);
                if (tile.valid[buf] == 0) {
                    continue;
                }
                const std::size_t cell = static_cast<std::size_t>(row) *
                                             static_cast<std::size_t>(width) +
                                         static_cast<std::size_t>(col);
                for (int c = 0; c < 3; ++c) {
                    gauss_ch[static_cast<std::size_t>(c)].v[cell] = static_cast<float>(
                        tile.rgb[buf * 3 + static_cast<std::size_t>(c)]);
                }
                tile_mask[cell] = 1;
                if (out.labels[cell] == static_cast<std::int32_t>(t)) {
                    mask.v[cell] = 1.0F;
                }
            }
        }

        const std::vector<Plane> mask_pyr = weight_pyramid(std::move(mask), levels);
        for (int c = 0; c < 3; ++c) {
            // Extend the tile's colour into its invalid region before the
            // pyramid, so the coarse levels see no artificial zero step at the
            // validity edge (see fill_invalid).
            Plane filled = gauss_ch[static_cast<std::size_t>(c)];
            fill_invalid(filled, tile_mask);
            const std::vector<Plane> g_ch = gaussian_pyramid(std::move(filled), levels);
            const std::vector<Plane> lap = laplacian_pyramid(g_ch);
            for (int k = 0; k <= levels; ++k) {
                Plane& acc = num[static_cast<std::size_t>(c)][static_cast<std::size_t>(k)];
                const Plane& l = lap[static_cast<std::size_t>(k)];
                const Plane& w = mask_pyr[static_cast<std::size_t>(k)];
                for (std::size_t i = 0; i < acc.v.size(); ++i) {
                    acc.v[i] += l.v[i] * w.v[i];
                }
            }
        }
        // Clear the reused mask over the same crop it was written across: a
        // stale 1 from this tile would look like real colour to the next tile's
        // fill_invalid().
        for (int row = tile.y0; row < tile.y0 + tile.height; ++row) {
            const std::ptrdiff_t start =
                static_cast<std::ptrdiff_t>(static_cast<std::size_t>(row) *
                                            static_cast<std::size_t>(width) +
                                            static_cast<std::size_t>(tile.x0));
            std::fill_n(tile_mask.begin() + start, static_cast<std::ptrdiff_t>(tile.width),
                        std::uint8_t{0});
        }
    }

    // Collapse each channel pyramid: top level plus the resized coarser sum.
    std::vector<Plane> result(3);
    for (int c = 0; c < 3; ++c) {
        const std::vector<Plane>& n = num[static_cast<std::size_t>(c)];
        Plane cur = n[static_cast<std::size_t>(levels)];
        for (int k = levels - 1; k >= 0; --k) {
            const Plane& nk = n[static_cast<std::size_t>(k)];
            const Plane up = resize(cur, nk.w, nk.h);
            Plane sum = make_plane(nk.w, nk.h);
            for (std::size_t i = 0; i < sum.v.size(); ++i) {
                sum.v[i] = nk.v[i] + up.v[i];
            }
            cur = std::move(sum);
        }
        result[static_cast<std::size_t>(c)] = std::move(cur);
    }

    // Copy the base level out and mark validity where any tile is valid.
    for (std::size_t cell = 0; cell < cells; ++cell) {
        const int col = static_cast<int>(cell % static_cast<std::size_t>(width));
        const int row = static_cast<int>(cell / static_cast<std::size_t>(width));
        bool valid = false;
        for (const OrthoTile& tile : tiles) {
            if (tile_valid(tile, col, row)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            continue;
        }
        for (int c = 0; c < 3; ++c) {
            const double value = static_cast<double>(result[static_cast<std::size_t>(c)].v[cell]);
            out.rgb[cell * 3 + static_cast<std::size_t>(c)] =
                static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0, 255.0)));
        }
        out.valid[cell] = 1;
    }
    return out;
}

int pyramid_level_count(int width, int height) { return level_count(width, height); }

int pyramid_support_cells(int levels) {
    // Level k of the pyramid is a 5-tap Gaussian (radius 2) applied to a plane
    // that is 2^k times decimated relative to the output, so one blur at level k
    // moves information 2 * 2^k cells at output resolution. The Laplacian and the
    // collapse mirror the same stencil, hence the factor of two over the sum of
    // the per-level reaches.
    if (levels <= 0) {
        return 0;
    }
    const int capped = std::min(levels, 16);  // 2^16 cells is far past any grid
    return 2 * ((1 << capped) - 1);
}

Blending parse_blending(const std::string& text) {
    if (text == "mosaic") {
        return Blending::Mosaic;
    }
    if (text == "average") {
        return Blending::Average;
    }
    if (text == "disabled") {
        return Blending::Disabled;
    }
    throw ValueError("unknown blending mode \"" + text +
                     "\" (expected \"mosaic\", \"average\" or \"disabled\")");
}

std::string to_string(Blending mode) {
    switch (mode) {
        case Blending::Mosaic:
            return "mosaic";
        case Blending::Average:
            return "average";
        case Blending::Disabled:
            return "disabled";
    }
    return "mosaic";
}

}  // namespace alphashape::ortho
