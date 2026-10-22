#!/usr/bin/env python3
"""Aerial (nadir) test fixture for milestone 9 (DEMs + orthomosaics).

World is a local ENU frame in metres, z up. The scene is an analytic height
field h(x, y) (a gentle slope plus a Gaussian hill), plus an axis-aligned
building box with vertical walls and a flat roof, plus a ramp. A 4x3 nadir
camera grid at 60 m altitude images the area; the fixture writes the images
(with exact GPS EXIF), the ground-truth poses, a DEM (top-surface height on a
0.25 m grid) and a true orthomosaic (the top-surface texture on that grid).

Deterministic: every random draw is seeded, so repeated runs are byte-identical.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from PIL import Image

# Reuse the milestone-1..8 scene helpers (geodetic/ENU, look_at, noise, EXIF).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import make_scene_fixtures as scene  # noqa: E402

# -- scene constants ---------------------------------------------------------

ORIGIN_LON = 9.2
ORIGIN_LAT = 48.7
ORIGIN_H = 300.0

AREA_X0, AREA_X1 = -60.0, 60.0
AREA_Y0, AREA_Y1 = -45.0, 45.0

BUILDING_X0, BUILDING_X1 = -20.0, -4.0
BUILDING_Y0, BUILDING_Y1 = 5.0, 17.0
BUILDING_BASE = 2.0 + 0.04 * -12.0
BUILDING_TOP = BUILDING_BASE + 8.0

RAMP_X0, RAMP_X1 = 0.0, 20.0
RAMP_Y0, RAMP_Y1 = 15.0, 30.0
RAMP_SLOPE = 0.25

GRID_CELL = 0.25
GRID_WIDTH = int(round((AREA_X1 - AREA_X0) / GRID_CELL))   # 480
GRID_HEIGHT = int(round((AREA_Y1 - AREA_Y0) / GRID_CELL))  # 360

IMAGE_WIDTH, IMAGE_HEIGHT = 480, 360
FOCAL = 500.0
SUPERSAMPLE = 2

ALTITUDE = 60.0
COLS, ROWS = 4, 3
SPACING_X, SPACING_Y = 22.0, 24.0


# -- rendering constants -----------------------------------------------------

Z_CEIL = 9.0          # the heightfield ray-march starts where z crosses this
Z_FLOOR = -1.0        # ...and stops here (below the lowest terrain)
MARCH_STEP = 0.25     # metres along the ray, as required by the spec
BISECT_ITERS = 14     # 0.25 / 2^14 = 1.5e-5 m along the ray (< 1e-4 m goal)
BUILDING_Z0 = -10.0   # slab bottom (buried; walls are rendered from here up)
BUILDING_Z1 = BUILDING_TOP

TILT_DEG = 2.0        # per-camera deterministic pseudo-random tilt about x/y
TILT_SEED = 20240615

TEXTURE_SEEDS = (4101, 5209, 6311)   # RGB channels of the top-surface texture
WALL_SEED = 7703

# Per-channel feature counts. Sized so the whole texture precompute stays about
# a second, while the features live at ~0.3 m (a few pixels at the 0.12 m nadir
# GSD) and are locally distinctive enough for the matcher.
TEX_BLOBS = 26
TEX_DISCS = 68
TEX_SEGMENTS = 104
TEX_BASE_FREQ = 0.09
TEX_OCTAVE_DECAY = 0.62


def truth_height(x, y):
    """Ground surface height h(x, y), metres.

    KEEP IN SYNC with alphashape::aerial_fixture::truth_height in
    tests/aerial_fixture.hpp -- the DEM test compares the two to 1e-4 m.
    """
    h = 2.0 + 0.04 * x
    h += 3.0 * np.exp(-(((x - 25.0) ** 2 + (y + 15.0) ** 2) / (2.0 * 12.0 ** 2)))
    on_ramp = (x >= RAMP_X0) & (x <= RAMP_X1) & (y >= RAMP_Y0) & (y <= RAMP_Y1)
    return h + np.where(on_ramp, RAMP_SLOPE * (x - RAMP_X0), 0.0)


def surface_height(x, y):
    """Top surface height, metres: the max of the terrain and the building roof."""
    h = truth_height(x, y)
    in_box = (x >= BUILDING_X0) & (x <= BUILDING_X1) & (y >= BUILDING_Y0) & (y <= BUILDING_Y1)
    return np.where(in_box, np.maximum(h, BUILDING_TOP), h)


# -- procedural RGB texture --------------------------------------------------
#
# The texture is a pure function of the surface coordinates, but evaluating the
# ~120 segment/blob/disc features per ray for 12 supersampled images is far too
# slow in numpy. Instead each channel is rasterised once onto a 0.1 m grid --
# finer than the 0.12 m nadir ground sample distance, so a bilinear read from it
# is at least as sharp as the image -- and every consumer (rendering, the
# orthomosaic) samples that grid. Because the grid is fixed in world space, the
# same world point still gets the same colour in every view.

TEX_GRID_STEP = 0.1

# Feature domains: (u0, u1, v0, v1) of the (x, y) ground area and the (x, z)
# wall area, so a wall's features do not lie off its own extent.
GROUND_DOMAIN = (AREA_X0, AREA_X1, AREA_Y0, AREA_Y1)
WALL_DOMAIN = (BUILDING_X0, BUILDING_X1, 0.0, BUILDING_TOP + 0.5)


class _Field:
    """One channel: warped value noise plus blobs, discs and segments."""

    def __init__(self, seed, domain):
        rng = np.random.default_rng(seed)
        u0, u1, v0, v1 = domain
        self.seed = seed
        self.blobs = scene._random_blobs(rng, TEX_BLOBS, u0, u1, v0, v1, 0.5, 4.0)
        self.discs = scene._random_discs(rng, TEX_DISCS, u0, u1, v0, v1, 0.4, 3.0)
        self.segments = scene._random_segments(rng, TEX_SEGMENTS, u0, u1, v0, v1, 1.0, 8.0)

    def raw(self, x, y):
        wx, wy = scene._domain_warp(x, y, self.seed, strength=3.0)
        value = 0.55 * scene._value_noise_multi(wx, wy, self.seed + 1)
        value = value + 0.30 * scene._blob_field(x, y, self.blobs)
        value = value + 0.35 * scene._disc_field(x, y, self.discs)
        value = value + 0.30 * scene._segment_field(x, y, self.segments)
        return value


class GridTexture:
    """A single-channel field rasterised onto a regular (u, v) grid, bilinear read."""

    def __init__(self, seed, domain, step=TEX_GRID_STEP, scale=1.0, bias=0.0):
        u0, u1, v0, v1 = domain
        field = _Field(seed, domain)
        self.u0, self.v0, self.step = u0, v0, step
        self.nu = int(round((u1 - u0) / step)) + 1
        self.nv = int(round((v1 - v0) / step)) + 1
        us = u0 + step * np.arange(self.nu)
        vs = v0 + step * np.arange(self.nv)
        gu, gv = np.meshgrid(us, vs)

        raw = field.raw(gu.ravel(), gv.ravel())
        # A fixed affine stretch, measured before any scaling, so the same world
        # point yields the same value in every view.
        probe = field.raw(*np.meshgrid(np.linspace(u0, u1, 64), np.linspace(v0, v1, 64)))
        lo = float(np.percentile(probe, 2.0))
        hi = float(np.percentile(probe, 98.0))
        normalised = np.zeros_like(raw) if hi <= lo else np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
        self.data = (bias + scale * normalised).reshape(self.nv, self.nu).astype(np.float32)

    def sample(self, u, v):
        """Bilinear read, clamped at the grid edges."""
        fu = np.clip((np.asarray(u, dtype=np.float64) - self.u0) / self.step, 0.0, self.nu - 1.0001)
        fv = np.clip((np.asarray(v, dtype=np.float64) - self.v0) / self.step, 0.0, self.nv - 1.0001)
        iu = fu.astype(np.int64)
        iv = fv.astype(np.int64)
        du = (fu - iu).astype(np.float32)
        dv = (fv - iv).astype(np.float32)
        d = self.data
        return (d[iv, iu] * ((1.0 - du) * (1.0 - dv))
                + d[iv, iu + 1] * (du * (1.0 - dv))
                + d[iv + 1, iu] * ((1.0 - du) * dv)
                + d[iv + 1, iu + 1] * (du * dv))


class RgbGrid:
    """Three independent GridTextures, one per channel, read together."""

    def __init__(self, seeds, domain):
        self.channels = [GridTexture(seed, domain) for seed in seeds]

    def sample(self, x, y):
        return np.stack([ch.sample(x, y) for ch in self.channels], axis=-1)

    def grid(self):
        """The whole texture as an (nv, nu, 3) uint8 array (for the orthomosaic)."""
        return np.stack([ch.data for ch in self.channels], axis=-1)


# -- cameras -----------------------------------------------------------------

def camera_poses():
    """12 nadir poses: a 4x3 grid at 60 m, each tilted by a deterministic 2 deg.

    Rows are the down/forward/right triad, so X_cam = R X_world + t. The pure
    nadir triad is right = +x, down = -y, forward = -z; a tiny rotation about the
    camera x and y axes then breaks the degenerate all-parallel geometry.
    """
    rng = np.random.default_rng(TILT_SEED)
    x0 = -0.5 * (COLS - 1) * SPACING_X
    y0 = -0.5 * (ROWS - 1) * SPACING_Y
    nadir = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

    poses = []
    for row in range(ROWS):
        for col in range(COLS):
            centre = np.array([x0 + col * SPACING_X, y0 + row * SPACING_Y, ALTITUDE])
            tx = np.deg2rad(rng.uniform(-TILT_DEG, TILT_DEG))
            ty = np.deg2rad(rng.uniform(-TILT_DEG, TILT_DEG))
            rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(tx), -np.sin(tx)], [0.0, np.sin(tx), np.cos(tx)]])
            ry = np.array([[np.cos(ty), 0.0, np.sin(ty)], [0.0, 1.0, 0.0], [-np.sin(ty), 0.0, np.cos(ty)]])
            R = (ry @ rx) @ nadir
            t = -R @ centre
            poses.append((centre, R, t))
    return poses


# -- ray casting -------------------------------------------------------------

def _intersect_building(origin, direction):
    """Nearest hit on the axis-aligned building box, via a slab test.

    Returns t, and the hit point's (x, y, z). Whether a hit is the roof or a
    wall is decided from z by the caller.
    """
    lo = np.array([BUILDING_X0, BUILDING_Y0, BUILDING_Z0])
    hi = np.array([BUILDING_X1, BUILDING_Y1, BUILDING_Z1])
    t_near = np.full(direction.shape[:-1] + (), -np.inf)
    t_far = np.full(direction.shape[:-1] + (), np.inf)
    for axis in range(3):
        d = direction[..., axis]
        o = origin[axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t0 = (lo[axis] - o) * inv
            t1 = (hi[axis] - o) * inv
        tmin = np.minimum(t0, t1)
        tmax = np.maximum(t0, t1)
        # A zero direction component: inside the slab is fine (inf), outside is
        # a guaranteed miss (-inf/inf -> the test below rejects it).
        parallel = np.abs(d) < 1e-12
        tmin = np.where(parallel, np.where((o >= lo[axis]) & (o <= hi[axis]), -np.inf, np.inf), tmin)
        tmax = np.where(parallel, np.where((o >= lo[axis]) & (o <= hi[axis]), np.inf, -np.inf), tmax)
        t_near = np.maximum(t_near, tmin)
        t_far = np.minimum(t_far, tmax)
    hit = (t_far >= t_near) & (t_far > 1e-6)
    t = np.where(t_near > 1e-6, t_near, t_far)
    return t, hit


def _intersect_heightfield(origin, direction):
    """Ray-march the height field, then bisect each crossing to BISECT_ITERS.

    The camera sits above Z_CEIL. The ray marches in steps of MARCH_STEP until z
    drops below the local surface (a crossing) or the ray passes Z_FLOOR (a miss).
    A crossing is refined by bisection on the signed (z - surface) gap.
    """
    shape = direction.shape[:-1]
    t = np.full(shape, -1.0)
    hx = np.full(shape, np.nan)
    hy = np.full(shape, np.nan)
    hz = np.full(shape, np.nan)

    dz = direction[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        d_floor = (Z_FLOOR - origin[2]) / dz
        d_ceiling = (Z_CEIL - origin[2]) / dz
    active = (d_floor > 0.0) & (dz < 0.0)

    # Start the march where the ray first drops below Z_CEIL (above the highest
    # terrain), not at the camera, so only the ~40 useful steps are taken.
    tt = np.where(active, d_ceiling, 0.0)
    px = origin[0] + tt * direction[..., 0]
    py = origin[1] + tt * direction[..., 1]
    pz = origin[2] + tt * dz
    prev_gap = pz - truth_height(px, py)

    while np.any(active):
        tt = tt + MARCH_STEP
        px = origin[0] + tt * direction[..., 0]
        py = origin[1] + tt * direction[..., 1]
        pz = origin[2] + tt * dz
        gap = pz - truth_height(px, py)

        crossed = active & (gap <= 0.0) & (prev_gap > 0.0)
        if np.any(crossed):
            t_lo = tt[crossed] - MARCH_STEP
            t_hi = tt[crossed]
            d = direction[crossed]
            for _ in range(BISECT_ITERS):
                t_mid = 0.5 * (t_lo + t_hi)
                pm = origin + t_mid[:, None] * d
                below = (pm[:, 2] - truth_height(pm[:, 0], pm[:, 1])) <= 0.0
                t_hi = np.where(below, t_mid, t_hi)
                t_lo = np.where(below, t_lo, t_mid)
            p_final = origin + t_hi[:, None] * d
            t[crossed] = t_hi
            hx[crossed] = p_final[:, 0]
            hy[crossed] = p_final[:, 1]
            hz[crossed] = p_final[:, 2]

        active = (active & ~crossed) & (tt + MARCH_STEP <= d_floor)
        prev_gap = np.where(active, gap, prev_gap)
    return t, hx, hy, hz


TEX_TOP = None    # set in main(); lazily built so import has no side effects
TEX_WALL = None


def build_textures():
    """Rasterise the ground and wall textures (about 1 s)."""
    global TEX_TOP, TEX_WALL
    TEX_TOP = RgbGrid(TEXTURE_SEEDS, GROUND_DOMAIN)
    TEX_WALL = GridTexture(WALL_SEED, WALL_DOMAIN, scale=0.5, bias=0.45)


WALL_TINT = np.array([0.85, 0.75, 0.70])


def cast_pixels(centre, R, t, sample_w, sample_h):
    """Ray-cast `sample_w` x `sample_h` sub-samples; return RGB [0,1] and depth."""
    s = sample_w // IMAGE_WIDTH
    us = (np.arange(sample_w) + 0.5) / s
    vs = (np.arange(sample_h) + 0.5) / s
    uu, vv = np.meshgrid(us, vs)

    dx = (uu - IMAGE_WIDTH / 2.0) / FOCAL
    dy = (vv - IMAGE_HEIGHT / 2.0) / FOCAL
    direction = dx[..., None] * R[0] + dy[..., None] * R[1] + R[2]

    t_b, b_hit = _intersect_building(centre, direction)
    t_h, hx, hy, hz = _intersect_heightfield(centre, direction)

    # Ties (the building's base line, where wall meets terrain) resolve toward
    # the heightfield, matching the scene generator's tie rule.
    hit_h = np.isfinite(hz) & (~b_hit | (t_h < t_b))
    hit_b = b_hit & ~hit_h

    rgb = np.zeros(direction.shape[:-1] + (3,), dtype=np.float64)
    if np.any(hit_h):
        rgb[hit_h] = TEX_TOP.sample(hx[hit_h], hy[hit_h])
    if np.any(hit_b):
        bx = centre[0] + t_b * direction[..., 0]
        by = centre[1] + t_b * direction[..., 1]
        bz = centre[2] + t_b * direction[..., 2]
        on_roof = hit_b & (bz >= BUILDING_Z1 - 1e-6)
        on_wall = hit_b & ~on_roof
        if np.any(on_wall):
            shade = TEX_WALL.sample(bx[on_wall], bz[on_wall])[:, None] * WALL_TINT
            rgb[on_wall] = np.clip(shade, 0.0, 1.0)
        if np.any(on_roof):
            rgb[on_roof] = TEX_TOP.sample(bx[on_roof], by[on_roof])

    hit = hit_h | hit_b
    depth = np.where(hit_h, t_h, np.where(hit_b, t_b, np.nan))
    return rgb, hit, depth


def render_image(centre, R, t):
    """Render one 480x360 RGB image with 2x2 supersampling, as uint8."""
    sw = IMAGE_WIDTH * SUPERSAMPLE
    sh = IMAGE_HEIGHT * SUPERSAMPLE
    rgb, _hit, _depth = cast_pixels(centre, R, t, sw, sh)
    stacked = rgb.reshape(IMAGE_HEIGHT, SUPERSAMPLE, IMAGE_WIDTH, SUPERSAMPLE, 3)
    averaged = stacked.mean(axis=(1, 3))
    return np.clip(np.rint(averaged * 255.0), 0.0, 255.0).astype(np.uint8)


def _project(point, R, t):
    p = R @ point + t
    u = IMAGE_WIDTH / 2.0 + FOCAL * p[0] / p[2]
    v = IMAGE_HEIGHT / 2.0 + FOCAL * p[1] / p[2]
    return u, v


def _scalar_hit(centre, direction):
    """Nearest surface hit along one ray: the point and its t, or None.

    Mirrors the vectorised cast, for the self-check.
    """
    best_t = None
    best_p = None

    # Building slab.
    lo = (BUILDING_X0, BUILDING_Y0, BUILDING_Z0)
    hi = (BUILDING_X1, BUILDING_Y1, BUILDING_Z1)
    t_near, t_far = -np.inf, np.inf
    for axis in range(3):
        d = direction[axis]
        if abs(d) < 1e-12:
            if not (lo[axis] <= centre[axis] <= hi[axis]):
                t_near, t_far = np.inf, -np.inf
                break
        else:
            t0 = (lo[axis] - centre[axis]) / d
            t1 = (hi[axis] - centre[axis]) / d
            t_near = max(t_near, min(t0, t1))
            t_far = min(t_far, max(t0, t1))
    if t_far >= t_near:
        t = t_near if t_near > 1e-6 else t_far
        if t > 1e-6:
            best_t = t
            best_p = centre + t * direction

    # Height field.
    if direction[2] < 0.0:
        t = 0.0
        prev = None
        while True:
            p = centre + t * direction
            if p[2] <= Z_FLOOR:
                break
            gap = p[2] - float(truth_height(p[0], p[1]))
            if prev is not None and gap <= 0.0 < prev:
                t_lo = t - MARCH_STEP
                t_hi = t
                for _ in range(BISECT_ITERS):
                    t_mid = 0.5 * (t_lo + t_hi)
                    pm = centre + t_mid * direction
                    if pm[2] - float(truth_height(pm[0], pm[1])) <= 0.0:
                        t_hi = t_mid
                    else:
                        t_lo = t_mid
                if best_t is None or t_hi < best_t:
                    best_t = t_hi
                    best_p = centre + t_hi * direction
                break
            prev = gap
            t += MARCH_STEP
    return best_p


def self_check(poses):
    """Reprojection guard: a ray through an interior pixel, cast to its hit,
    must project back to that same pixel (and the hit must sit on the truth
    surface). This catches any pix<->ray or world->cam convention slip.
    """
    us = np.linspace(40.0, IMAGE_WIDTH - 40.0, 6)
    vs = np.linspace(40.0, IMAGE_HEIGHT - 40.0, 5)
    checked = 0
    worst_u = 0.0
    worst_v = 0.0
    for centre, R, t in poses:
        for u in us:
            for v in vs:
                dx = (u - IMAGE_WIDTH / 2.0) / FOCAL
                dy = (v - IMAGE_HEIGHT / 2.0) / FOCAL
                direction = dx * R[0] + dy * R[1] + R[2]
                point = _scalar_hit(centre, direction)
                if point is None:
                    continue
                # The hit must lie on *some* rendered surface: the terrain, the
                # building roof, or a vertical building wall. A wall hit is
                # inside the box's (x, y) footprint but below the roof, so it is
                # neither on the terrain nor at roof height -- check all three.
                on_terrain = abs(point[2] - float(truth_height(point[0], point[1]))) <= 1e-3
                in_box = (BUILDING_X0 - 1e-3 <= point[0] <= BUILDING_X1 + 1e-3
                          and BUILDING_Y0 - 1e-3 <= point[1] <= BUILDING_Y1 + 1e-3)
                on_roof = in_box and abs(point[2] - BUILDING_TOP) <= 1e-3
                on_wall = in_box and (BUILDING_Z0 - 1e-3 <= point[2] <= BUILDING_TOP + 1e-3)
                if not (on_terrain or on_roof or on_wall):
                    raise SystemExit("self-check: hit off the surface at pixel (%.1f, %.1f)" % (u, v))
                uu, vv = _project(point, R, t)
                worst_u = max(worst_u, abs(uu - u))
                worst_v = max(worst_v, abs(vv - v))
                checked += 1
    return checked, worst_u, worst_v


def _write_products(here, poses, origin, origin_ecef, rotation):
    """Write the truth JSON, the DEM grid and the orthomosaic."""
    import struct

    # DEM: top-surface height on cell centres, row 0 = north, col 0 = west.
    cells_x = AREA_X0 + (np.arange(GRID_WIDTH) + 0.5) * GRID_CELL
    cells_y = AREA_Y1 - (np.arange(GRID_HEIGHT) + 0.5) * GRID_CELL
    cxx, cyy = np.meshgrid(cells_x, cells_y)
    dem = surface_height(cxx.ravel(), cyy.ravel()).reshape(GRID_HEIGHT, GRID_WIDTH)
    with open(os.path.join(here, "aerial_truth_dem.f32"), "wb") as handle:
        handle.write(struct.pack("<2i", GRID_WIDTH, GRID_HEIGHT))
        handle.write(np.asarray(dem, dtype="<f4").tobytes())

    # Orthomosaic: the top-surface texture on the same grid (uint8 RGB PNG).
    ortho = np.rint(TEX_TOP.sample(cxx.ravel(), cyy.ravel()) * 255.0).astype(np.uint8)
    ortho = ortho.reshape(GRID_HEIGHT, GRID_WIDTH, 3)
    Image.fromarray(ortho, mode="RGB").save(os.path.join(here, "aerial_truth_ortho.png"), optimize=True)

    images = []
    for index, (centre, R, t) in enumerate(poses):
        ecef = scene.enu_to_ecef(centre, origin_ecef, rotation)
        lat, lon, height = scene.ecef_to_geodetic(*ecef)
        row, col = divmod(index, COLS)
        images.append({
            "file": "aerial_r%dc%d.jpg" % (row, col),
            "R": R.tolist(),                      # X_cam = R * X_world + t
            "t": t.tolist(),
            "centre": centre.tolist(),
            "f": FOCAL,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "gps": {"lon": lon, "lat": lat, "height": height},
        })

    truth = {
        "origin": {"lon": ORIGIN_LON, "lat": ORIGIN_LAT, "h": ORIGIN_H, "epsg": 4326},
        "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "f": FOCAL,
        "grid": {"origin_x": AREA_X0, "origin_y": AREA_Y1, "cell": GRID_CELL,
                 "width": GRID_WIDTH, "height": GRID_HEIGHT, "row0": "north", "col0": "west"},
        "building": {"x0": BUILDING_X0, "x1": BUILDING_X1, "y0": BUILDING_Y0, "y1": BUILDING_Y1,
                     "base": BUILDING_BASE, "top": BUILDING_TOP},
        "images": images,
        "notes": ("world = local ENU metres about `origin` (east, north, up); "
                  "X_cam = R * X_world + t; the chunk's local frame is ENU and its "
                  "transform is enu_to_ecef(origin). DEM = top surface (max of "
                  "terrain and the building roof), float32 little-endian, row 0 = "
                  "north (y = +45), col 0 = west (x = -60), cell centres."),
    }
    with open(os.path.join(here, "aerial_truth.json"), "w") as out:
        json.dump(truth, out, indent=2)
        out.write("\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    build_textures()

    poses = camera_poses()
    origin = {"lat": ORIGIN_LAT, "lon": ORIGIN_LON, "h": ORIGIN_H}
    origin_ecef = scene.geodetic_to_ecef(ORIGIN_LAT, ORIGIN_LON, ORIGIN_H)
    rotation = scene.enu_rotation(ORIGIN_LAT, ORIGIN_LON)

    # Every camera's GPS reference must round-trip exactly through the geodesy.
    for centre, _R, _t in poses:
        ecef = scene.enu_to_ecef(centre, origin_ecef, rotation)
        back = scene.ecef_to_enu(ecef, origin_ecef, rotation)
        if not np.allclose(back, centre, atol=1e-6):
            raise SystemExit("geodetic round-trip failed for camera centre")

    for index, (centre, R, t) in enumerate(poses):
        row, col = divmod(index, COLS)
        name = "aerial_r%dc%d.jpg" % (row, col)
        rgb = render_image(centre, R, t)
        ecef = scene.enu_to_ecef(centre, origin_ecef, rotation)
        lat, lon, height = scene.ecef_to_geodetic(*ecef)
        exif = scene.geo_exif(lat, lon, height)
        Image.fromarray(rgb, mode="RGB").save(os.path.join(here, name), quality=92,
                                              exif=exif.tobytes(), optimize=True)
        size = os.path.getsize(os.path.join(here, name))
        nonblack = float(np.mean(np.any(rgb > 0, axis=2)))
        print("wrote %s (%.1f%% non-black, %d bytes)" % (name, 100.0 * nonblack, size))

    checked, worst_u, worst_v = self_check(poses)
    print("self-check: %d rays, worst reprojection %.2e, %.2e px" % (checked, worst_u, worst_v))
    if worst_u > 1e-3 or worst_v > 1e-3:
        raise SystemExit("self-check failed: projection/ray cast disagree")

    _write_products(here, poses, origin, origin_ecef, rotation)
    print("wrote aerial_truth.json, aerial_truth_dem.f32, aerial_truth_ortho.png")


if __name__ == "__main__":
    main()
