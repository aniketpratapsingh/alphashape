#!/usr/bin/env python3
"""Reference-free and reference-based quality metrics for the alphashape bench.

The module splits into three layers:

  * **Core maths** — numpy only, no file I/O, no rasterio. Every numerically
    load-bearing function lives here and is unit-tested in
    ``tools/bench/tests/test_metrics.py`` against synthetic arrays whose answer is
    known by construction. A metric that drifts is caught without downloading a
    dataset or building the engine.
  * **Readers** — GeoTIFF / PLY / LAS / OBJ, imported lazily, turning a file into
    the plain values the core maths consumes.
  * **Orchestration** — ``evaluate()`` wires one run's products and a dataset's
    references into a metrics dict; ``run_bench.py`` calls it.

Grid convention, stated once because getting it wrong shifts every comparison by
half a pixel: ``origin_x``/``origin_y`` are the world coordinates of the OUTER
corner of pixel (0, 0), the GeoTIFF convention. Pixel (0, 0)'s centre is
``(origin_x + 0.5 * pixel_size_x, origin_y - 0.5 * pixel_size_y)``, and rows
increase southward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# The decimated resolution the mask statistics run at. A 20000 x 20000 mosaic
# carries 4e8 cells; connected-component analysis in pure Python at that size is
# minutes of work and tells the reader nothing the same analysis at 1200 px does
# not. Areas are converted back to square metres, so the reported numbers are
# resolution-independent.
MASK_MAX_SIDE = 1200

# Ceiling on the zero-padded inverse transform used to refine a phase-correlation
# peak. 64 M complex cells is about 1 GB, which is more than CI runners want to
# spend on one refinement; the loop in phase_correlation_shift halves the factor
# until the padded grid fits under this.
_MAX_PADDED = 1 << 26


# ---------------------------------------------------------------------------
# Core value types
# ---------------------------------------------------------------------------


@dataclass
class Raster:
    """A raster on a north-up grid.

    ``data`` is ``(height, width)`` for one band or ``(height, width, bands)`` for
    several. ``nodata`` is the sentinel, or ``None`` when none is declared.
    """

    data: np.ndarray
    origin_x: float
    origin_y: float
    pixel_size_x: float
    pixel_size_y: float
    nodata: float | None = None
    epsg: int | None = None

    def __post_init__(self) -> None:
        if self.data.ndim not in (2, 3):
            raise ValueError(f"Raster.data must be 2-D or 3-D, got {self.data.ndim}-D")
        if self.pixel_size_x <= 0 or self.pixel_size_y <= 0:
            raise ValueError("Raster pixel sizes must be positive")

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def bands(self) -> int:
        return 1 if self.data.ndim == 2 else int(self.data.shape[2])

    def band(self, index: int = 0) -> np.ndarray:
        """One band as a 2-D array (a view, not a copy)."""
        if self.data.ndim == 2:
            if index != 0:
                raise IndexError(f"single-band raster has no band {index}")
            return self.data
        return self.data[:, :, index]

    def validity(self) -> np.ndarray:
        """True where the pixel carries a measurement.

        Invalid means: equal to the nodata sentinel (with a tolerance that
        survives float32 round-tripping), NaN, or non-finite. Multi-band rasters
        are judged on their first band, which both writers here follow.
        """
        first = self.band(0)
        finite = np.isfinite(first)
        if self.nodata is None:
            return finite
        if not math.isfinite(self.nodata):
            return finite & ~np.isnan(first)
        scale = max(1.0, abs(self.nodata))
        return finite & (np.abs(first.astype(np.float64) - float(self.nodata)) > 1e-5 * scale)

    def extent(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) of the OUTER raster bounds."""
        return (
            self.origin_x,
            self.origin_y - self.height * self.pixel_size_y,
            self.origin_x + self.width * self.pixel_size_x,
            self.origin_y,
        )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample_to_grid(
    source: Raster,
    *,
    origin_x: float,
    origin_y: float,
    pixel_size: float,
    width: int,
    height: int,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``source`` onto a north-up grid, returning (values, valid).

    Nearest-neighbour (``order=0``) for categorical rasters, bilinear
    (``order=1``) for continuous ones. Nodata cells never contribute: a bilinear
    tap with some invalid corners is renormalised over the valid ones, and an
    output cell with no valid tap is marked invalid. That is the rule the
    engine's own orthorectifier uses, so a hole in the source does not bleed a
    fabricated value into a comparison.

    ``values`` is always float64 with a trailing band axis, even for one band.
    """
    if width <= 0 or height <= 0:
        raise ValueError("resample_to_grid needs a positive output size")

    cols = (origin_x + (np.arange(width) + 0.5) * pixel_size - source.origin_x) / source.pixel_size_x
    rows = (source.origin_y - (origin_y - (np.arange(height) + 0.5) * pixel_size)) / source.pixel_size_y

    flat = np.asarray(source.data, dtype=np.float64)
    if flat.ndim == 2:
        flat = flat[:, :, None]
    src_valid = source.validity()
    values = np.full((height, width, flat.shape[2]), np.nan, dtype=np.float64)
    valid = np.zeros((height, width), dtype=bool)

    if order == 0:
        c0 = np.floor(cols).astype(np.int64)
        r0 = np.floor(rows).astype(np.int64)
        c_in = (c0 >= 0) & (c0 < source.width)
        r_in = (r0 >= 0) & (r0 < source.height)
        cc = np.clip(c0, 0, source.width - 1)
        rr = np.clip(r0, 0, source.height - 1)
        sample_valid = src_valid[np.ix_(rr, cc)] & r_in[:, None] & c_in[None, :]
        values[:] = flat[np.ix_(rr, cc)]
        values[~sample_valid] = np.nan
        return values, sample_valid

    cf = cols - 0.5
    rf = rows - 0.5
    c0 = np.floor(cf).astype(np.int64)
    r0 = np.floor(rf).astype(np.int64)
    fx = cf - c0
    fy = rf - r0

    acc = np.zeros((height, width, flat.shape[2]), dtype=np.float64)
    wsum = np.zeros((height, width), dtype=np.float64)

    for dr in (0, 1):
        for dc in (0, 1):
            rows_idx = r0 + dr
            cols_idx = c0 + dc
            wy = fy if dr else 1.0 - fy
            wx = fx if dc else 1.0 - fx
            row_ok = (rows_idx >= 0) & (rows_idx < source.height)
            col_ok = (cols_idx >= 0) & (cols_idx < source.width)
            rr = np.clip(rows_idx, 0, source.height - 1)
            cc = np.clip(cols_idx, 0, source.width - 1)
            tap_valid = src_valid[np.ix_(rr, cc)] & row_ok[:, None] & col_ok[None, :]
            weight = np.outer(wy, wx) * tap_valid
            # Mask the value, not just the weight: 0 * NaN is NaN, so a NaN
            # tap with zero weight would still poison the accumulator.
            acc += np.where(tap_valid[:, :, None], flat[np.ix_(rr, cc)], 0.0) * weight[:, :, None]
            wsum += weight

    good = wsum > 1e-9
    values[good] = acc[good] / wsum[good][:, None]
    return values, good


def common_grid(
    a: Raster, b: Raster, *, max_cells: int = 4_000_000
) -> tuple[float, float, float, int, int]:
    """The finest common north-up grid covering the intersection of a and b.

    Returns ``(origin_x, origin_y, pixel_size, width, height)``. The pixel size
    is the finer of the two inputs — never sub-sampled, so the comparison never
    invents resolution — coarsened by an integer factor only when the
    intersection would exceed ``max_cells``. An integer factor keeps every output
    cell aligned with the source grids, so the resampler adds no phase shift of
    its own.
    """
    ax0, ay0, ax1, ay1 = a.extent()
    bx0, by0, bx1, by1 = b.extent()
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    y0, y1 = max(ay0, by0), min(ay1, by1)
    if not (x1 > x0 and y1 > y0):
        raise ValueError("rasters do not overlap")

    base = min(a.pixel_size_x, a.pixel_size_y, b.pixel_size_x, b.pixel_size_y)
    x0 = math.ceil(x0 / base - 1e-9) * base
    y1 = math.floor(y1 / base + 1e-9) * base

    stride = 1
    width = int(math.floor((x1 - x0) / base + 1e-9))
    height = int(math.floor((y1 - y0) / base + 1e-9))
    while width * height > max_cells and stride < 64:
        stride += 1
        width = int(math.floor((x1 - x0) / (base * stride)))
        height = int(math.floor((y1 - y0) / (base * stride)))
    if width <= 0 or height <= 0:
        raise ValueError("rasters overlap by less than one pixel")
    return x0, y1, base * stride, width, height


# ---------------------------------------------------------------------------
# Masks: erosion, convex hull, holes
# ---------------------------------------------------------------------------


def erosion(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Binary erosion with a square element, edge-clamped. radius 0 is a no-op."""
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            ys = slice(max(0, dy), h + min(0, dy))
            yd = slice(max(0, -dy), h + min(0, -dy))
            xs = slice(max(0, dx), w + min(0, dx))
            xd = slice(max(0, -dx), w + min(0, -dx))
            shifted[yd, xd] = mask[ys, xs]
            out &= shifted
    return out


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """Fill the convex hull of the True cells of ``mask``.

    A hole inside the hull is a genuine gap in the reconstruction; a cell outside
    it is simply outside the surveyed area. Telling those apart is what makes
    "holes inside the convex hull" a meaningful number, so this computes a real
    convex hull (Andrew's monotone chain) rather than a bounding box, then fills
    it with a per-row crossing-number scan.
    """
    rows, cols = np.nonzero(mask)
    out = np.zeros(mask.shape, dtype=bool)
    if rows.size == 0:
        return out
    if rows.size <= 2:
        out[rows, cols] = True
        return out

    # Work in continuous pixel coordinates: hull vertices are CELL CENTRES, so
    # cell (col, row) becomes the point (col + 0.5, row + 0.5). Keeping the
    # polygon and the cell edges in the same space is what makes the overlap test
    # at the end land on the right columns; mixing centres and edges loses (or
    # gains) exactly one row and one column at every extreme.
    points = np.stack(
        [cols.astype(np.float64) + 0.5, rows.astype(np.float64) + 0.5], axis=1
    )
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]

    def cross(o: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
        return float((p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if hull.shape[0] < 3:
        out[rows, cols] = True
        return out

    x0 = hull[:, 0]
    y0 = hull[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    def span_at(height: float) -> tuple[int, int] | None:
        """The column interval the polygon covers at ``height``, or None."""
        straddles = ((y0 <= height) & (y1 > height)) | ((y1 <= height) & (y0 > height))
        if not straddles.any():
            return None
        ys0 = y0[straddles]
        ys1 = y1[straddles]
        with np.errstate(divide="ignore", invalid="ignore"):
            crossings = x0[straddles] + (height - ys0) * (
                (x1[straddles] - x0[straddles]) / (ys1 - ys0)
            )
        crossings = crossings[np.isfinite(crossings)]
        if crossings.size < 2:
            return None
        crossings.sort()
        # A convex polygon has exactly two crossings; take the outermost so a
        # degenerate vertex pair cannot split the span. The span is in continuous
        # coordinates, and a cell whose body touches it counts, so the integer
        # bounds are its floor/ceil.
        return int(math.floor(crossings[0])), int(math.ceil(crossings[-1]) + 0.5)

    # A cell is inside when any part of it lies inside the polygon. For a convex
    # hull that is the union of the spans at the cell's top and bottom edges, i.e.
    # the interval between the smallest left and the largest right. Testing only
    # the centre line drops the hull's extreme rows — a rectangle's top and
    # bottom rows carry horizontal hull edges, and a centre-line test straddles
    # nothing there.
    for y in range(max(0, int(math.floor(y0.min())) - 1), min(mask.shape[0], int(math.ceil(y0.max())) + 1)):
        spans = [s for s in (span_at(float(y)), span_at(float(y + 1))) if s is not None]
        if not spans:
            continue
        left = max(0, min(s[0] for s in spans) - 1)
        right = min(mask.shape[1], max(s[1] for s in spans) + 1)
        if right > left:
            out[y, left:right] = True
    return out


def decimate_mask(mask: np.ndarray, max_side: int = MASK_MAX_SIDE) -> tuple[np.ndarray, int]:
    """Stride-decimate a mask so its long side is at most ``max_side``.

    Returns ``(decimated, stride)``. Decimation uses ``np.maximum`` on blocks of
    ``stride``: a decimated cell is valid if ANY source cell in its block was,
    which keeps a thin but real covered strip from disappearing and so cannot
    invent a hole. It does mean the reported covered fraction is an
    over-estimate by at most one block's worth of cells, which at 1200 px on a
    20000 px mosaic is under 0.2% — stated in the report rather than corrected
    away by a slower full-resolution pass.
    """
    h, w = mask.shape
    stride = max(1, int(math.ceil(max(h, w) / max_side)))
    if stride == 1:
        return mask.copy(), 1
    hh = (h // stride) * stride
    ww = (w // stride) * stride
    if hh == 0 or ww == 0:
        return mask.copy(), 1
    block = mask[:hh, :ww].reshape(hh // stride, stride, ww // stride, stride)
    return block.any(axis=(1, 3)), stride


def hole_stats(
    valid: np.ndarray, *, interior: np.ndarray | None = None, min_px: int = 4
) -> dict:
    """Connected-component statistics over the invalid cells.

    A *hole* is a 4-connected run of invalid cells that neither touches the
    raster border (that is the edge of the survey, not a gap) nor is smaller than
    ``min_px`` cells (a single decimated cell is resampling noise, not a hole a
    user would ever see). ``interior`` restricts the search to the convex hull of
    the covered area, which is the definition worth reporting: outside the hull
    is "not surveyed", not "missing".

    Returns hole count, total area, largest hole, and the hole fraction of the
    considered area — all in decimated cells; the caller converts to metres.
    """
    invalid = ~valid
    if interior is not None:
        invalid &= interior
    h, w = invalid.shape
    if h == 0 or w == 0 or not invalid.any():
        return {"count": 0, "area_px": 0, "largest_px": 0, "fraction": 0.0}

    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 0
    touches_border: list[bool] = [False]
    areas: list[int] = [0]

    ys, xs = np.nonzero(invalid)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if labels[start_y, start_x]:
            continue
        next_label += 1
        labels[start_y, start_x] = next_label
        stack = [(start_y, start_x)]
        area = 0
        border = False
        while stack:
            y, x = stack.pop()
            area += 1
            if y == 0 or x == 0 or y == h - 1 or x == w - 1:
                border = True
            if y > 0 and invalid[y - 1, x] and not labels[y - 1, x]:
                labels[y - 1, x] = next_label
                stack.append((y - 1, x))
            if y + 1 < h and invalid[y + 1, x] and not labels[y + 1, x]:
                labels[y + 1, x] = next_label
                stack.append((y + 1, x))
            if x > 0 and invalid[y, x - 1] and not labels[y, x - 1]:
                labels[y, x - 1] = next_label
                stack.append((y, x - 1))
            if x + 1 < w and invalid[y, x + 1] and not labels[y, x + 1]:
                labels[y, x + 1] = next_label
                stack.append((y, x + 1))
        touches_border.append(border)
        areas.append(area)

    considered = int(interior.sum()) if interior is not None else int(valid.size)
    count = 0
    hole_area = 0
    largest = 0
    for label in range(1, len(areas)):
        area = areas[label]
        if touches_border[label] or area < min_px:
            continue
        count += 1
        hole_area += area
        largest = max(largest, area)
    return {
        "count": count,
        "area_px": hole_area,
        "largest_px": largest,
        "fraction": (hole_area / considered) if considered else 0.0,
    }


# ---------------------------------------------------------------------------
# Effective resolution
# ---------------------------------------------------------------------------


def gradient_energy(image: np.ndarray, *, valid: np.ndarray | None = None) -> float:
    """Mean squared Sobel gradient magnitude of an image normalised to [0, 1].

    This is the raw quantity behind the effective-resolution estimate: it rises
    with real texture and falls when a raster is blurred or upsampled, so a drop
    between runs means the mosaic is getting softer.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.shape[0] < 3 or img.shape[1] < 3:
        return 0.0
    if valid is not None:
        img = np.where(valid, img, np.nan)
    gx = np.full_like(img, np.nan)
    gy = np.full_like(img, np.nan)
    gx[1:-1, 1:-1] = (
        img[:-2, 2:] + 2 * img[1:-1, 2:] + img[2:, 2:]
        - img[:-2, :-2] - 2 * img[1:-1, :-2] - img[2:, :-2]
    ) / 8.0
    gy[1:-1, 1:-1] = (
        img[2:, :-2] + 2 * img[2:, 1:-1] + img[2:, 2:]
        - img[:-2, :-2] - 2 * img[:-2, 1:-1] - img[:-2, 2:]
    ) / 8.0
    magnitude = np.sqrt(gx * gx + gy * gy)
    finite = np.isfinite(magnitude)
    if not finite.any():
        return 0.0
    return float(np.nanmean(magnitude[finite] ** 2))


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur with edge padding (numpy only)."""
    if radius <= 0:
        return img
    out = np.asarray(img, dtype=np.float64)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for shift in range(2 * radius + 1):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(shift, shift + out.shape[axis])
            acc += padded[tuple(sl)]
        out = acc / (2 * radius + 1)
    return out


def mtf50_effective_gsd(
    image: np.ndarray,
    nominal_gsd_m: float,
    *,
    valid: np.ndarray | None = None,
    max_blur_radius: int = 12,
) -> dict:
    """Estimate effective ground sample distance from loss of sharpness.

    A matched-blur search, not a slanted-edge MTF: blur the image with growing
    box kernels and find the radius at which gradient energy halves, i.e. the
    sharpness a genuinely half-resolution image would have. That is the
    "MTF50-equivalent" resolution ratio the bench reports.

    It is cheap, monotone, and needs no detectable edge — a mosaic rarely has a
    clean slanted edge anywhere, and the same scene compared run over run is all
    this metric has to support. ``ratio`` is effective/nominal (1.0 = the mosaic
    carries its nominal resolution, 2.0 = it behaves twice as coarse).

    ``textured`` is False when the scene has no usable texture (no energy, or
    energy rising under more blur); a ratio is then reported as 0 rather than a
    meaningless number.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if valid is not None:
        img = np.where(valid, img, 0.0)

    base = gradient_energy(img)
    if base <= 0.0:
        return {"ratio": 0.0, "effective_gsd_m": 0.0, "textured": False, "attack_px": 0.0}

    target = base * 0.5
    previous = base
    for radius in range(1, max_blur_radius + 1):
        energy = gradient_energy(_box_blur(img, radius))
        if energy <= target:
            span = previous - energy
            frac = 0.0 if span <= 0 else (previous - target) / span
            attack = (radius - 1) + frac
            # A box of radius r behaves like a Gaussian of sigma r/sqrt(3), and a
            # half-energy point is about one such sigma of extra blur.
            attack *= math.sqrt(3.0)
            return {
                "ratio": attack,
                "effective_gsd_m": nominal_gsd_m * attack,
                "textured": True,
                "attack_px": attack,
            }
        if energy > previous + 1e-12:
            return {"ratio": 0.0, "effective_gsd_m": 0.0, "textured": False, "attack_px": 0.0}
        previous = energy

    return {
        "ratio": float(max_blur_radius),
        "effective_gsd_m": nominal_gsd_m * max_blur_radius,
        "textured": True,
        "attack_px": float(max_blur_radius),
    }


# ---------------------------------------------------------------------------
# Phase correlation
# ---------------------------------------------------------------------------


def _highpass(img: np.ndarray) -> np.ndarray:
    """Subtract a box-blurred copy: drops the DC term and slow illumination
    gradients that would otherwise dominate a phase correlation."""
    return img - _box_blur(img, max(1, min(img.shape) // 64))


def phase_correlation_shift(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_shift_fraction: float = 0.25,
    upsample: int = 8,
) -> dict:
    """Sub-pixel translation that best aligns ``a`` onto ``b``.

    Returns ``dx``/``dy`` in pixels of the input grid, meaning "``b`` is ``a``
    shifted by (dx, dy)". Shapes must match; the caller resamples onto a common
    grid first.

    The peak is found to whole pixels by phase correlation, then refined by
    zero-padding the cross-power spectrum and inverse-transforming again. That is
    the exact band-limited upsampling of the correlation (Guizar-Sicairos), not a
    parabola through three samples: a parabolic fit is biased by a wide peak,
    which is exactly the case for two mosaics of the same scene. The padding
    factor is capped so the padded transform stays under ``_MAX_PADDED`` cells,
    which is what keeps a 1200 px comparison affordable.

    ``max_shift_fraction`` rejects a peak outside the central window: the FFT is
    periodic, so an offset beyond half the extent aliases, and one that big means
    the rasters are not of the same scene.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"phase_correlation_shift needs equal shapes, got {x.shape} and {y.shape}")
    if x.ndim != 2:
        raise ValueError("phase_correlation_shift needs 2-D arrays")
    if min(x.shape) < 8:
        raise ValueError("phase_correlation_shift needs at least an 8x8 array")

    xs = _highpass(x - x.mean())
    ys = _highpass(y - y.mean())
    fx = np.fft.rfft2(xs)
    fy = np.fft.rfft2(ys)

    cross = fy * np.conj(fx)
    magnitude = np.abs(cross)
    magnitude[magnitude < 1e-12] = 1e-12
    spectrum = cross / magnitude
    correlation = np.fft.irfft2(spectrum, s=x.shape)

    h, w = correlation.shape
    py, px = (int(v) for v in np.unravel_index(int(np.argmax(correlation)), correlation.shape))
    dy = py if py <= h // 2 else py - h
    dx = px if px <= w // 2 else px - w
    peak = float(correlation.max())
    if abs(dy) > int(h * max_shift_fraction) or abs(dx) > int(w * max_shift_fraction):
        return {"dx": float(dx), "dy": float(dy), "peak": peak, "in_bounds": False}

    factor = max(1, int(upsample))
    while factor > 1 and h * factor * w * factor > _MAX_PADDED:
        factor //= 2
    if factor <= 1:
        return {"dx": float(dx), "dy": float(dy), "peak": peak, "in_bounds": True}

    # Zero-pad the (complex) spectrum to the upsampled grid, inverse transform,
    # and read the peak near the coarse one. Padding the unnormalised spectrum by
    # the factor keeps the transform unitary, but only the argmax matters here.
    big = np.zeros((h * factor, w // 2 + 1), dtype=np.complex128)
    big[:h, :] = spectrum
    upsampled = np.fft.irfft2(big, s=(h * factor, w * factor))

    # The coarse peak moves to (py*factor, px*factor); search a small window
    # around it for the fine peak.
    radius = factor
    centre_y = py * factor
    centre_x = px * factor
    ys_idx = (centre_y + np.arange(-radius, radius + 1)) % (h * factor)
    xs_idx = (centre_x + np.arange(-radius, radius + 1)) % (w * factor)
    window = upsampled[np.ix_(ys_idx, xs_idx)]
    wy, wx = np.unravel_index(int(np.argmax(window)), window.shape)
    fine_y = (centre_y + (wy - radius)) % (h * factor)
    fine_x = (centre_x + (wx - radius)) % (w * factor)
    fy_px = fine_y / factor
    fx_px = fine_x / factor
    if fy_px > h / 2:
        fy_px -= h
    if fx_px > w / 2:
        fx_px -= w

    # A refinement that jumps more than half a coarse cell is the padding's
    # periodic image of the true peak, not a refinement; keep the coarse value.
    if abs(fy_px - dy) > 0.5 or abs(fx_px - dx) > 0.5:
        return {"dx": float(dx), "dy": float(dy), "peak": peak, "in_bounds": True}
    return {"dx": float(fx_px), "dy": float(fy_px), "peak": peak, "in_bounds": True}


# ---------------------------------------------------------------------------
# Point clouds
# ---------------------------------------------------------------------------


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Keep one point per voxel, in index order so the result is deterministic.

    Voxel-normalising before a nearest-neighbour score is what ETH3D's evaluator
    does: without it a denser cloud scores higher purely by having more samples in
    the same volume.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    if voxel <= 0:
        raise ValueError("voxel must be positive")
    keys = np.floor(pts / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(first)]


def nn_distances(query: np.ndarray, target: np.ndarray, *, chunk: int = 1 << 18) -> np.ndarray:
    """Distance from each query point to its nearest target point.

    Brute force in chunks, using |q-t|^2 = |q|^2 + |t|^2 - 2 q.t. The |q|^2 term
    is constant within a row and does not move the argmin, so it is added back
    only for the winning column rather than for the whole distance matrix — which
    is what keeps the intermediate at one row per query instead of one row per
    (query, target) pair.

    A kd-tree is asymptotically better, but scipy is not a dependency of the unit
    tests, and after voxel-normalisation the clouds are tens of thousands of
    points, where this is fast and completely predictable.
    """
    q = np.ascontiguousarray(query, dtype=np.float64)
    t = np.ascontiguousarray(target, dtype=np.float64)
    if q.size == 0:
        return np.zeros(0, dtype=np.float64)
    if t.size == 0:
        return np.full(q.shape[0], np.inf)
    t_norm = np.einsum("ij,ij->i", t, t)
    out = np.empty(q.shape[0], dtype=np.float64)
    for start in range(0, q.shape[0], chunk):
        block = q[start:start + chunk]
        d2 = t_norm[None, :] - 2.0 * (block @ t.T)
        nearest = d2.argmin(axis=1)
        best = d2[np.arange(block.shape[0]), nearest] + np.einsum("ij,ij->i", block, block)
        out[start:start + chunk] = np.sqrt(np.maximum(best, 0.0))
    return out


def accuracy_completeness(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    tolerances: Sequence[float],
    *,
    voxel: float = 0.01,
    register: bool = True,
) -> dict:
    """ETH3D-style accuracy and completeness at each tolerance.

    Accuracy is the fraction of voxel-normalised reconstructed points within
    ``tolerance`` of a reference point; completeness is the converse. Each
    direction also gets its median and 90th percentile distance, because a
    fraction alone hides how far the misses are.

    ``register`` fits a similarity from the reconstruction to the reference first.
    It defaults on because the two clouds are in unrelated frames — the
    reconstruction's own, and the scanner's — and scoring distance across that
    difference would measure the frames, not the geometry. The fit is reported, so
    a run that only scores well because it collapsed onto the reference is visible.
    """
    rec_raw = np.asarray(reconstructed, dtype=np.float64)
    ref_raw = np.asarray(reference, dtype=np.float64)
    rec = voxel_downsample(rec_raw, voxel)
    ref = voxel_downsample(ref_raw, voxel)
    result: dict = {"reconstructed": int(rec.shape[0]), "reference": int(ref.shape[0])}
    if rec.size == 0 or ref.size == 0:
        result["accuracy"] = {f"{tol:g}": 0.0 for tol in tolerances}
        result["completeness"] = {f"{tol:g}": 0.0 for tol in tolerances}
        result["f_score_5cm"] = 0.0
        return result

    registered = ref_raw
    if register:
        rec_raw, alignment = register_clouds(rec_raw, ref_raw, voxel=voxel)
        result["alignment"] = alignment
        rec = voxel_downsample(rec_raw, voxel)

    to_ref = nn_distances(rec, registered)
    to_rec = nn_distances(registered, rec)
    accuracy = {f"{tol:g}": float(np.mean(to_ref <= tol)) for tol in tolerances}
    completeness = {f"{tol:g}": float(np.mean(to_rec <= tol)) for tol in tolerances}
    result.update(
        accuracy=accuracy,
        completeness=completeness,
        accuracy_median_m=float(np.median(to_ref)),
        accuracy_p90_m=float(np.percentile(to_ref, 90)),
        completeness_median_m=float(np.median(to_rec)),
        completeness_p90_m=float(np.percentile(to_rec, 90)),
        f_score_5cm=_f_score(accuracy, completeness, 0.05),
    )
    return result


def umeyama_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form similarity (scale, rotation, translation) mapping source -> target.

    Umeyama's least-squares solution. ``target ~= scale * R @ source + t``.

    Needed because a reconstruction lives in its own frame — the engine's chunk
    coordinates, or the ENU frame it georeferenced itself into — while a laser
    reference lives in the scanner's. Scoring distance without first removing that
    arbitrary similarity would measure the frame difference, not the geometry. Every
    MVS benchmark does the same registration step before scoring; this is it.

    Throws ValueError for fewer than three correspondences (a similarity has seven
    degrees of freedom and three points give nine equations) or a degenerate
    (collinear) configuration, where the rotation is not determined.
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"umeyama_similarity needs matching (n, 3) arrays, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError("umeyama_similarity needs at least three correspondences")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    covariance = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(covariance)

    # Reflection guard: with det(U) * det(V) < 0 the raw SVD gives a reflection,
    # which is not a rotation. Flipping the last singular direction is Umeyama's
    # correction and is what makes the result a proper rotation.
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2] = -1.0
    rotation = u @ np.diag(sign) @ vt

    source_variance = float(np.mean(np.sum(src_c * src_c, axis=1)))
    if source_variance <= 1e-15:
        raise ValueError("umeyama_similarity: the source points are all the same point")
    scale = float(np.sum(d * sign) / source_variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"umeyama_similarity: degenerate scale {scale!r}")
    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def apply_similarity(scale: float, rotation: np.ndarray, translation: np.ndarray,
                     points: np.ndarray) -> np.ndarray:
    """``scale * R @ p + t`` for every row of ``points``."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return scale * (pts @ rotation.T) + translation


def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centroid, axes, spread) of a cloud, axes ordered by decreasing spread.

    The eigenvectors of the covariance, with each axis orientated so that the
    projection of the cloud onto it is positive-skewed. That sign convention is
    what makes two runs of the same scene produce the same frame; without it an
    eigenvector's sign is arbitrary and a PCA alignment would flip the scene.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    covariance = centred.T @ centred / max(points.shape[0], 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    for axis in range(3):
        projection = centred @ vectors[:, axis]
        if np.sum(projection ** 3) < 0.0:
            vectors[:, axis] = -vectors[:, axis]
    return centroid, vectors, np.sqrt(np.maximum(values, 0.0))


def pca_align(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """A similarity that puts ``source`` in ``target``'s principal-axis frame.

    The coarse initialisation ICP needs. Two clouds of the same scene have almost
    the same principal axes, so rotating one onto the other is a good starting
    guess even when their frames are unrelated — where nearest-neighbour ICP from
    an identity rotation is not, because at a large offset the nearest neighbour
    of most points is simply the wrong point.

    The eigenvector signs are fixed by :func:`_pca_frame`, so the rotation is a
    proper rotation (det = +1) by construction rather than by a reflection guard.
    Scale comes from the ratio of the clouds' RMS spreads, which is the right
    statistic for a similarity whose scale is unknown.
    """
    src_centre, src_axes, src_spread = _pca_frame(source)
    dst_centre, dst_axes, dst_spread = _pca_frame(target)
    # source -> its own frame -> target's frame
    rotation = dst_axes @ src_axes.T
    if np.linalg.det(rotation) < 0.0:
        # A mirrored axis set: flip the least-significant axis of the SOURCE frame
        # so the result stays a rotation rather than becoming a reflection.
        flipped = src_axes.copy()
        flipped[:, 2] = -flipped[:, 2]
        rotation = dst_axes @ flipped.T

    src_rms = float(np.sqrt(np.mean(src_spread ** 2)))
    dst_rms = float(np.sqrt(np.mean(dst_spread ** 2)))
    scale = (dst_rms / src_rms) if src_rms > 1e-12 else 1.0
    translation = dst_centre - scale * (rotation @ src_centre)
    return scale, rotation, translation


def register_clouds(
    source: np.ndarray,
    target: np.ndarray,
    *,
    voxel: float,
    iterations: int = 20,
    trim_start: float = 1.0,
    trim_end: float = 0.7,
) -> tuple[np.ndarray, dict]:
    """Iterative similarity ICP of ``source`` onto ``target``.

    Two stages, because one does not work on two clouds in unrelated frames:

    1. **Coarse** — :func:`pca_align` puts the source in the target's principal-axis
       frame. This is what makes the first nearest-neighbour pairing meaningful.
    2. **Fine** — alternate nearest-neighbour correspondence with a closed-form
       similarity fit, keeping the closest ``trim`` fraction of pairs. The trim
       starts at every pair and tightens towards ``trim_end``: early rounds need
       all the data to settle the coarse error, later rounds want to exclude the
       pairs that come from genuinely different geometry (a face only one cloud
       saw), which would otherwise hold the fit off.

    A similarity rather than a rigid transform, because an SfM reconstruction has no
    metric scale until it is georeferenced and the reference scenes here carry no
    GPS, so the scale ratio between the two clouds is unknown and has to be fitted.

    Returns ``(aligned_source, report)``. The report carries the fitted scale, the
    RMS of the surviving pairs and whether the fit stopped moving; a run that only
    scores well because it collapsed onto the reference is visible in there.
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.size == 0 or dst.size == 0:
        return src.reshape(-1, 3), {"converged": False, "reason": "empty cloud"}

    moving = voxel_downsample(src, voxel)
    ref = voxel_downsample(dst, voxel)
    if moving.shape[0] < 10 or ref.shape[0] < 10:
        return src, {"converged": False, "reason": "too few points after voxel normalisation"}

    scale, rotation, translation = pca_align(moving, ref)
    report: dict = {"coarse_scale": scale, "iterations": 0, "converged": False}
    previous_rms = math.inf

    for step in range(1, iterations + 1):
        aligned = apply_similarity(scale, rotation, translation, moving)
        distances = nn_distances(aligned, ref)
        indices = _nn_indices(aligned, ref)

        # Tighten the trim linearly over the run.
        fraction = trim_start + (trim_end - trim_start) * (step - 1) / max(iterations - 1, 1)
        keep = max(3, int(round(fraction * moving.shape[0])))
        order = np.argsort(distances)[:keep]

        try:
            scale, rotation, translation = umeyama_similarity(
                moving[order], ref[indices[order]]
            )
        except ValueError as exc:
            report["reason"] = str(exc)
            break

        aligned = apply_similarity(scale, rotation, translation, moving)
        rms = float(np.sqrt(np.mean(nn_distances(aligned, ref)[order] ** 2)))
        report.update(iterations=step, rms_m=rms, scale=scale, kept=int(order.size))

        if abs(previous_rms - rms) <= 1e-5 * max(rms, 1.0):
            report["converged"] = True
            break
        previous_rms = rms

    aligned_all = apply_similarity(scale, rotation, translation, src)
    report["scale"] = scale
    report.setdefault("rms_m", None)
    return aligned_all, report


def _nn_indices(query: np.ndarray, target: np.ndarray, *, chunk: int = 1 << 18) -> np.ndarray:
    """Index of the nearest target point for each query point (see nn_distances)."""
    q = np.ascontiguousarray(query, dtype=np.float64)
    t = np.ascontiguousarray(target, dtype=np.float64)
    t_norm = np.einsum("ij,ij->i", t, t)
    out = np.empty(q.shape[0], dtype=np.int64)
    for start in range(0, q.shape[0], chunk):
        block = q[start:start + chunk]
        d2 = t_norm[None, :] - 2.0 * (block @ t.T)
        out[start:start + chunk] = d2.argmin(axis=1)
    return out


def _f_score(accuracy: dict, completeness: dict, tolerance: float) -> float:
    key = f"{tolerance:g}"
    if key not in accuracy or key not in completeness:
        return 0.0
    a, c = accuracy[key], completeness[key]
    return 0.0 if a + c <= 0 else 2.0 * a * c / (a + c)


# ---------------------------------------------------------------------------
# DEM comparison
# ---------------------------------------------------------------------------


def dem_difference(
    ours: np.ndarray, reference: np.ndarray, *, valid: np.ndarray | None = None
) -> dict:
    """ΔZ statistics over cells where both DEMs carry a height.

    Reports median (systematic offset), NMAD (1.4826 x MAD: robust spread, so a
    handful of building edges cannot drag it around the way a standard deviation
    can) and the 68/95 percentiles of |ΔZ|. Heights are in the rasters' own
    vertical units, metres for both datasets here.
    """
    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"dem_difference needs equal shapes, got {a.shape} and {b.shape}")
    both = np.isfinite(a) & np.isfinite(b)
    if valid is not None:
        both &= valid
    n = int(both.sum())
    if n == 0:
        return {"cells": 0, "median_m": None, "nmad_m": None, "p68_m": None, "p95_m": None}
    delta = a[both] - b[both]
    median = float(np.median(delta))
    mad = float(np.median(np.abs(delta - median)))
    return {
        "cells": n,
        "median_m": median,
        "nmad_m": 1.4826 * mad,
        "p68_m": float(np.percentile(np.abs(delta), 68)),
        "p95_m": float(np.percentile(np.abs(delta), 95)),
        "mean_m": float(np.mean(delta)),
        "rmse_m": float(np.sqrt(np.mean(delta * delta))),
    }


# ---------------------------------------------------------------------------
# Camera model (mirrors engine/src/model/camera/calibration.hpp)
# ---------------------------------------------------------------------------


def project_pixels(
    points: np.ndarray,
    pose: np.ndarray,
    calibration: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into pixels with the engine's camera model.

    ``pose`` is the 4x4 world->camera transform exactly as the project stores it,
    and ``calibration`` carries the Brown-Conrady parameters under the same names
    the .psx file uses (``f cx cy b1 b2 k1 k2 k3 k4 p1 p2`` plus ``type width
    height``). Returns ``(pixels, ok)``; ``ok`` is False for a point behind a
    frame camera, which has no projection, and those rows are NaN.

    This must exist in Python because the reprojection metric has to run on a CI
    machine with no Python bindings built — and because re-projecting from the
    saved project's own calibration is the cross-check that catches a writer which
    mangles a coefficient. It is a transcription of ``try_project_point``; the C++
    header remains the only implementation in the engine.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.size == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    m = np.asarray(pose, dtype=np.float64)
    cam = p @ m[:3, :3].T + m[:3, 3]
    x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]

    if calibration.get("type", "Frame") == "Frame":
        ok = z > 0.0
        safe_z = np.where(ok, z, 1.0)
        xs = x / safe_z
        ys = y / safe_z
    else:
        r3d = np.hypot(x, y)
        theta = np.arctan2(r3d, z)
        safe_r = np.where(r3d > 0, r3d, 1.0)
        scale = np.where(r3d > 0, theta / safe_r, 0.0)
        ok = ~((r3d == 0) & (z == 0))
        xs = x * scale
        ys = y * scale

    k1 = float(calibration.get("k1", 0.0))
    k2 = float(calibration.get("k2", 0.0))
    k3 = float(calibration.get("k3", 0.0))
    k4 = float(calibration.get("k4", 0.0))
    p1 = float(calibration.get("p1", 0.0))
    p2 = float(calibration.get("p2", 0.0))

    r2 = xs * xs + ys * ys
    r4 = r2 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2 + k4 * r4 * r4
    xd = xs * radial + p1 * (r2 + 2.0 * xs * xs) + 2.0 * p2 * xs * ys
    yd = ys * radial + p2 * (r2 + 2.0 * ys * ys) + 2.0 * p1 * xs * ys

    width = float(calibration.get("width", 0))
    height = float(calibration.get("height", 0))
    f = float(calibration.get("f", 0.0))
    b1 = float(calibration.get("b1", 0.0))
    b2 = float(calibration.get("b2", 0.0))
    u = width * 0.5 + float(calibration.get("cx", 0.0)) + xd * (f + b1) + yd * b2
    v = height * 0.5 + float(calibration.get("cy", 0.0)) + yd * f

    pixels = np.stack([u, v], axis=1)
    pixels[~ok] = np.nan
    return pixels, ok


def reprojection_stats(observed: np.ndarray, projected: np.ndarray) -> dict:
    """RMS/median/p90/max pixel error between observed and projected keypoints.

    Pairs with a non-finite value are dropped (a projection that went behind the
    camera). Fields are ``None`` rather than NaN when nothing is left, so a JSON
    consumer sees an explicit gap instead of a number that propagates.
    """
    o = np.asarray(observed, dtype=np.float64)
    p = np.asarray(projected, dtype=np.float64)
    if o.shape != p.shape:
        raise ValueError(f"reprojection_stats needs equal shapes, got {o.shape} and {p.shape}")
    good = np.isfinite(p).all(axis=1) & np.isfinite(o).all(axis=1)
    n = int(good.sum())
    if n == 0:
        return {"pairs": 0, "rms_px": None, "median_px": None, "p90_px": None, "max_px": None}
    error = np.linalg.norm(p[good] - o[good], axis=1)
    return {
        "pairs": n,
        "rms_px": float(np.sqrt(np.mean(error * error))),
        "median_px": float(np.median(error)),
        "p90_px": float(np.percentile(error, 90)),
        "max_px": float(np.max(error)),
    }


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_geotiff(path: str) -> Raster:
    """Read a GeoTIFF's bands into a ``Raster``.

    rasterio is imported here rather than at module scope so the unit tests import
    nothing beyond numpy. Bands stack into (height, width, bands); one band stays
    2-D.
    """
    import rasterio  # noqa: PLC0415 - deliberately lazy

    with rasterio.open(path) as src:
        data = src.read()
        nodata = src.nodata
        transform = src.transform
        epsg = src.crs.to_epsg() if src.crs else None
    payload = data[0] if data.shape[0] == 1 else np.moveaxis(data, 0, 2)
    return Raster(
        data=payload,
        origin_x=float(transform.c),
        origin_y=float(transform.f),
        pixel_size_x=float(transform.a),
        pixel_size_y=float(-transform.e),
        nodata=None if nodata is None else float(nodata),
        epsg=epsg,
    )


def read_geotiff_grid(path: str) -> dict:
    """Grid metadata only, without reading pixels.

    Lets the bench decide whether a reference is comparable at all (same CRS,
    overlapping extent) before spending time resampling it.
    """
    import rasterio  # noqa: PLC0415

    with rasterio.open(path) as src:
        return {
            "width": int(src.width),
            "height": int(src.height),
            "origin_x": float(src.transform.c),
            "origin_y": float(src.transform.f),
            "pixel_size_x": float(src.transform.a),
            "pixel_size_y": float(-src.transform.e),
            "bands": int(src.count),
            "dtype": str(src.dtypes[0]),
            "epsg": src.crs.to_epsg() if src.crs else None,
            "nodata": None if src.nodata is None else float(src.nodata),
        }


def read_ply_points(path: str, *, max_points: int | None = None) -> np.ndarray:
    """Vertex positions from a binary_little_endian or ASCII PLY.

    Only what the engine writes and a reference cloud needs: ``x y z`` as the
    first three vertex properties, extras skipped by record stride, faces ignored.
    ``max_points`` takes an evenly spaced stride subset, so a 30-million-point
    cloud does not have to fit in memory and the subset is deterministic.
    """
    with open(path, "rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: PLY header ended without end_header")
            text = line.decode("ascii", "replace").strip()
            header.append(text)
            if text == "end_header":
                break
        if not header or header[0] != "ply":
            raise ValueError(f"{path}: not a PLY file")

        fmt = None
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for text in header[1:]:
            parts = text.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    properties.append(("list", parts[2]))
                else:
                    properties.append((parts[2], parts[1]))

        if fmt not in ("binary_little_endian", "ascii"):
            raise ValueError(f"{path}: unsupported PLY format {fmt!r}")
        if vertex_count == 0:
            return np.zeros((0, 3))
        names = [name for name, _ in properties]
        if names[:3] != ["x", "y", "z"]:
            raise ValueError(f"{path}: PLY vertices must start with x y z, got {names[:3]}")

        stride = 1
        if max_points is not None and max_points > 0 and vertex_count > max_points:
            stride = int(math.ceil(vertex_count / max_points))
        wanted = np.arange(0, vertex_count, stride, dtype=np.int64)
        out = np.empty((wanted.size, 3), dtype=np.float64)

        if fmt == "ascii":
            for index in range(vertex_count):
                line = handle.readline()
                if not line:
                    raise ValueError(f"{path}: truncated ASCII PLY at vertex {index}")
                if index % stride:
                    continue
                fields = line.split()
                row = index // stride
                out[row] = (float(fields[0]), float(fields[1]), float(fields[2]))
            return out

        import struct  # noqa: PLC0415

        types = {
            "char": ("b", 1), "int8": ("b", 1), "uchar": ("B", 1), "uint8": ("B", 1),
            "short": ("h", 2), "int16": ("h", 2), "ushort": ("H", 2), "uint16": ("H", 2),
            "int": ("i", 4), "int32": ("i", 4), "uint": ("I", 4), "uint32": ("I", 4),
            "float": ("f", 4), "float32": ("f", 4), "double": ("d", 8), "float64": ("d", 8),
        }
        offsets: list[tuple[int, str]] = []
        offset = 0
        for name, kind in properties:
            if kind == "list":
                raise ValueError(f"{path}: a vertex property list is not supported")
            if kind not in types:
                raise ValueError(f"{path}: unknown PLY property type {kind!r}")
            code, size = types[kind]
            offsets.append((offset, code))
            offset += size
        record = offset
        unpack_x = struct.Struct("<" + offsets[0][1]).unpack_from
        unpack_y = struct.Struct("<" + offsets[1][1]).unpack_from
        unpack_z = struct.Struct("<" + offsets[2][1]).unpack_from

        body_start = handle.tell()
        for row, index in enumerate(wanted.tolist()):
            handle.seek(body_start + index * record)
            raw = handle.read(record)
            if len(raw) < record:
                raise ValueError(f"{path}: truncated binary PLY at vertex {index}")
            out[row] = (
                unpack_x(raw, offsets[0][0])[0],
                unpack_y(raw, offsets[1][0])[0],
                unpack_z(raw, offsets[2][0])[0],
            )
        return out


def read_las_points(path: str, *, max_points: int | None = None) -> np.ndarray:
    """x/y/z from a LAS or LAZ file, via laspy.

    laspy is the only practical way to read the LAZ the reference publishers ship
    (Laszip is not vendored here). Declared as an explicit dependency of the
    workflow rather than a silent optional, so a missing install fails loudly
    instead of scoring completeness against an empty cloud.
    """
    try:
        import laspy  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - only without laspy
        raise ImportError(
            f"reading {path} needs laspy; install it with `pip install laspy[lazrs]`"
        ) from exc

    with laspy.open(path) as reader:
        total = int(reader.header.point_count)
        stride = 1
        if max_points is not None and max_points > 0 and total > max_points:
            stride = int(math.ceil(total / max_points))
        chunks = []
        for points in reader.chunk_iterator(2_000_000):
            xyz = np.stack(
                [np.asarray(points.x), np.asarray(points.y), np.asarray(points.z)], axis=1
            )
            if stride > 1:
                xyz = xyz[::stride]
            chunks.append(xyz)
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3))


def read_obj_triangles(path: str, *, max_faces: int | None = None) -> np.ndarray:
    """``(n, 3, 3)`` triangle corners from a Wavefront OBJ.

    Faces with more than three corners are fan-triangulated, the standard OBJ
    interpretation and what makes the visibility test work on a quad mesh.
    """
    vertices: list[tuple[float, float, float]] = []
    corners: list[tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                indices = []
                for token in parts:
                    if not token:
                        continue
                    indices.append(int(token.split("/")[0]) - 1)
                if len(indices) < 3:
                    continue
                for i in range(1, len(indices) - 1):
                    corners.append((indices[0], indices[i], indices[i + 1]))
            if max_faces is not None and len(corners) >= max_faces:
                break
    if not corners:
        return np.zeros((0, 3, 3))
    v = np.asarray(vertices, dtype=np.float64)
    idx = np.asarray(corners, dtype=np.int64)
    return v[idx]


# ---------------------------------------------------------------------------
# Reference-free product metrics
# ---------------------------------------------------------------------------


def ortho_metrics(ortho: Raster, *, nominal_gsd_m: float, texture_max_side: int = 1024) -> dict:
    """Coverage, holes inside the hull, and effective resolution of a mosaic.

    The alpha channel (band 3, when present) is the engine's "this cell came from
    an image" flag, so validity is read from it rather than guessed from colour: a
    black pixel is a legitimate measurement of a black roof.
    """
    if ortho.bands >= 4:
        validity = ortho.band(3) > 0
    elif ortho.nodata is not None:
        validity = ortho.validity()
    else:
        validity = np.ones((ortho.height, ortho.width), dtype=bool)

    pixel_area = ortho.pixel_size_x * ortho.pixel_size_y
    small, stride = decimate_mask(validity, MASK_MAX_SIDE)
    hull = convex_hull_mask(small)
    interior = erosion(hull, 1)
    holes = hole_stats(small, interior=interior)
    hole_pixel_area = pixel_area * stride * stride

    texture = _texture_crop(ortho, validity, texture_max_side)
    resolution = mtf50_effective_gsd(texture["image"], nominal_gsd_m, valid=texture["valid"])

    return {
        "width": ortho.width,
        "height": ortho.height,
        "pixel_size_m": ortho.pixel_size_x,
        "mask_stride": stride,
        "covered_fraction": float(validity.sum()) / validity.size if validity.size else 0.0,
        "hull_fraction": float(hull.sum()) / hull.size if hull.size else 0.0,
        "holes_inside_hull": holes["count"],
        "hole_area_m2": holes["area_px"] * hole_pixel_area,
        "hole_fraction": holes["fraction"],
        "largest_hole_m2": holes["largest_px"] * hole_pixel_area,
        "gradient_energy": float(gradient_energy(texture["image"], valid=texture["valid"])),
        "effective_gsd_m": resolution["effective_gsd_m"],
        "effective_resolution_ratio": resolution["ratio"],
        "textured": resolution["textured"],
        "texture_samples": texture["samples"],
        "epsg": ortho.epsg,
    }


def _texture_crop(ortho: Raster, validity: np.ndarray, max_side: int) -> dict:
    """RGB float crop of the covered area, at most ``max_side`` on a side."""
    rows, cols = np.nonzero(validity)
    if rows.size == 0:
        return {"image": np.zeros((1, 1)), "valid": np.ones((1, 1), dtype=bool), "samples": 1}
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    stride = max(1, int(math.ceil(max(r1 - r0, c1 - c0) / max_side)))
    rgb = np.stack(
        [ortho.band(ch)[r0:r1:stride, c0:c1:stride] for ch in range(min(3, ortho.bands))],
        axis=2,
    ).astype(np.float64)
    # Normalise to [0, 1] by the actual peak: uint8 is the norm, but a 16-bit
    # ortho would otherwise swamp the gradient-energy scale.
    peak = float(rgb.max()) if rgb.size else 0.0
    if peak > 1.5:
        rgb /= 255.0 if peak <= 255.0 else peak
    return {
        "image": rgb,
        "valid": validity[r0:r1:stride, c0:c1:stride],
        "samples": int(rgb.shape[0] * rgb.shape[1]),
    }


def dem_metrics(dem: Raster, *, reference: Raster | None = None) -> dict:
    """Nodata fraction, height range, and — with a reference — ΔZ."""
    validity = dem.validity()
    heights = dem.band(0)
    good = heights[validity]
    out: dict = {
        "width": dem.width,
        "height": dem.height,
        "pixel_size_m": dem.pixel_size_x,
        "nodata_fraction": 1.0 - float(validity.mean()) if validity.size else 1.0,
        "epsg": dem.epsg,
        "height_min_m": float(good.min()) if good.size else None,
        "height_max_m": float(good.max()) if good.size else None,
    }
    if reference is not None:
        out["difference"] = compare_dem_to_reference(dem, reference)
    return out


def compare_dem_to_reference(ours: Raster, reference: Raster) -> dict:
    """Resample both DEMs to a common grid and report ΔZ."""
    ox, oy, pixel, width, height = common_grid(ours, reference)
    a, a_valid = resample_to_grid(
        ours, origin_x=ox, origin_y=oy, pixel_size=pixel, width=width, height=height
    )
    b, b_valid = resample_to_grid(
        reference, origin_x=ox, origin_y=oy, pixel_size=pixel, width=width, height=height
    )
    both = a_valid & b_valid
    stats = dem_difference(a[:, :, 0], b[:, :, 0], valid=both)
    stats["compared_cells"] = int(both.sum())
    stats["grid_pixel_m"] = pixel
    stats["grid_size"] = [int(width), int(height)]
    stats["overlap_fraction"] = (
        float(both.sum()) / float(a_valid.sum()) if a_valid.any() else 0.0
    )
    return stats


def compare_ortho_to_reference(ours: Raster, reference: Raster) -> dict:
    """Horizontal offset and overlap against a reference orthomosaic.

    The reference is another pipeline's mosaic of the same photographs, so it is
    registered to the same world frame up to that pipeline's own error. Phase
    correlation on luminance gives the residual plan-view offset in metres; a
    nonzero value means our georeferencing disagrees with the publisher's, not
    that one is wrong, which is why the report labels it "vs reference".
    """
    ox, oy, pixel, width, height = common_grid(ours, reference)
    a, a_valid = resample_to_grid(
        ours, origin_x=ox, origin_y=oy, pixel_size=pixel, width=width, height=height
    )
    b, b_valid = resample_to_grid(
        reference, origin_x=ox, origin_y=oy, pixel_size=pixel, width=width, height=height
    )
    both = a_valid & b_valid
    # A correlation needs enough structure to lock onto; 256 cells is a 16x16
    # patch, below which a peak is noise.
    if int(both.sum()) < 256:
        return {"comparable": False, "reason": "too few overlapping pixels"}

    lum_a = a[:, :, : min(3, a.shape[2])].mean(axis=2)
    lum_b = b[:, :, : min(3, b.shape[2])].mean(axis=2)
    # Fill holes with each image's own mean before correlating: a nodata region
    # that differs between the two would otherwise dominate as a giant step edge.
    lum_a = np.where(both, lum_a, float(lum_a[both].mean()))
    lum_b = np.where(both, lum_b, float(lum_b[both].mean()))
    # Zero mean / unit norm so brightness and contrast differences between the
    # two pipelines do not bias the phase.
    lum_a = (lum_a - lum_a.mean()) / (lum_a.std() or 1.0)
    lum_b = (lum_b - lum_b.mean()) / (lum_b.std() or 1.0)

    shift = phase_correlation_shift(lum_a, lum_b)
    return {
        "comparable": True,
        "dx_m": shift["dx"] * pixel,
        "dy_m": shift["dy"] * pixel,
        "offset_m": math.hypot(shift["dx"], shift["dy"]) * pixel,
        "dx_px": shift["dx"],
        "dy_px": shift["dy"],
        "grid_pixel_m": pixel,
        "grid_size": [int(width), int(height)],
        "overlap_fraction": float(both.sum()) / float(a_valid.sum()) if a_valid.any() else 0.0,
        "in_bounds": shift["in_bounds"],
        "peak": shift["peak"],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    """Everything the evaluator needs to know about one dataset."""

    name: str
    images: list[str]
    ortho_reference: str | None = None
    dem_reference: str | None = None
    laser_reference: str | None = None
    nominal_gsd_m: float | None = None
    licence: dict = field(default_factory=dict)


def evaluate(
    spec: DatasetSpec,
    *,
    ortho_path: str | None,
    dem_path: str | None,
    cloud_path: str | None,
    camera_stats: dict | None = None,
    tolerances: Iterable[float] = (0.02, 0.05, 0.10),
) -> dict:
    """Score one run's products against a dataset's references.

    Every section is optional. A section whose inputs are missing reports
    ``{"skipped": "<why>"}`` rather than disappearing, so a reader can tell
    "scored and good" from "not scored".
    """
    result: dict = {
        "dataset": spec.name,
        "licence": spec.licence,
        "reference_free": {},
        "reference_based": {},
    }

    ortho = None
    if ortho_path:
        try:
            ortho = read_geotiff(ortho_path)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            result["reference_free"]["ortho"] = {"skipped": f"orthomosaic unreadable: {exc}"}
    if ortho is not None:
        gsd = spec.nominal_gsd_m or ortho.pixel_size_x
        result["reference_free"]["ortho"] = ortho_metrics(ortho, nominal_gsd_m=gsd)
    elif "ortho" not in result["reference_free"]:
        result["reference_free"]["ortho"] = {"skipped": "no orthomosaic produced"}

    dem = None
    if dem_path:
        try:
            dem = read_geotiff(dem_path)
        except Exception as exc:  # noqa: BLE001
            result["reference_free"]["dem"] = {"skipped": f"DEM unreadable: {exc}"}
    if dem is not None:
        result["reference_free"]["dem"] = dem_metrics(dem)
    elif "dem" not in result["reference_free"]:
        result["reference_free"]["dem"] = {"skipped": "no DEM produced"}

    result["reference_free"]["cameras"] = (
        camera_stats if camera_stats is not None else {"skipped": "project not inspected"}
    )

    # -- reference: orthomosaic ---------------------------------------------
    if spec.ortho_reference is None:
        result["reference_based"]["ortho"] = {"skipped": "dataset has no ortho reference"}
    elif ortho is None:
        result["reference_based"]["ortho"] = {"skipped": "no orthomosaic produced"}
    else:
        try:
            result["reference_based"]["ortho"] = compare_ortho_to_reference(
                ortho, read_geotiff(spec.ortho_reference)
            )
        except Exception as exc:  # noqa: BLE001
            result["reference_based"]["ortho"] = {"skipped": f"reference unusable: {exc}"}

    # -- reference: DSM -----------------------------------------------------
    if spec.dem_reference is None:
        result["reference_based"]["dem"] = {"skipped": "dataset has no DSM reference"}
    elif dem is None:
        result["reference_based"]["dem"] = {"skipped": "no DEM produced"}
    else:
        try:
            result["reference_based"]["dem"] = compare_dem_to_reference(
                dem, read_geotiff(spec.dem_reference)
            )
        except Exception as exc:  # noqa: BLE001
            result["reference_based"]["dem"] = {"skipped": f"reference unusable: {exc}"}

    # -- reference: laser point cloud ---------------------------------------
    if spec.laser_reference is None:
        result["reference_based"]["cloud"] = {"skipped": "dataset has no laser reference"}
    elif not cloud_path:
        result["reference_based"]["cloud"] = {"skipped": "no point cloud produced"}
    else:
        try:
            ours = read_ply_points(cloud_path, max_points=400_000)
            reference = read_las_points(spec.laser_reference, max_points=400_000)
            stats = accuracy_completeness(ours, reference, list(tolerances))
            stats["tolerances_m"] = list(tolerances)
            result["reference_based"]["cloud"] = stats
        except Exception as exc:  # noqa: BLE001
            result["reference_based"]["cloud"] = {"skipped": f"reference unusable: {exc}"}

    return result
