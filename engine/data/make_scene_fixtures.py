#!/usr/bin/env python3
"""Render the deterministic ground-truth scene fixtures used by matching tests.

Why this exists
---------------
Milestone 3 recovers relative camera pose from matches. A test can only assert
that the recovered pose is *correct* if it knows the true one, so this script
renders a few overlapping views of a scene whose geometry it also writes down.

Why a non-planar scene
----------------------
A scene made of a single plane is degenerate for relative pose: a homography
explains it, and the translation is only recoverable up to an ambiguity. So the
scene here is a ground plane plus a vertical wall that meets it at an edge. The
edge (and the two differently-textured surfaces around it) makes the geometry
fully observable.

Why procedural texture
----------------------
Every visible surface must yield plenty of DoG / M-LDB features so a matcher has
something to work with, and those features must be *locally distinctive*: a
regular pattern (a checkerboard) makes every corner look alike, so the ratio test
throws away the true matches and the survivors can be off by a cell. Each surface
therefore gets domain-warped multi-octave value noise at high contrast, plus
randomly placed blobs, discs and short line segments of random size, intensity
and orientation. The ground and the wall use different seeds, so their textures
are uncorrelated (a matcher cannot "succeed" by lining up two copies of the same
pattern).

Determinism
-----------
Every pixel is a pure function of the world point it sees: the value noise is a
hashed lattice and the random blobs, discs and segments are drawn once from a
seeded generator, so rerunning this script reproduces the same images bit for
bit, apart from JPEG rounding.

Camera convention
-----------------
Matches src/model/camera/calibration.hpp: the centre of the top-left pixel is
(0.5, 0.5); camera axes are +X right, +Y down, +Z forward; u = W/2 + f*X/Z and
v = H/2 + f*Y/Z, with no distortion. f = 500 px.

Camera layout
-------------
Three cameras on a lateral baseline of 4 units at ~16 units from the scene (a
25% baseline relative to the wall distance), aimed at a distant point so the
rotation between views is only a few degrees — a well-conditioned but genuinely
non-degenerate rig. The world->camera transforms are stored in scene_truth.json.

EXIF
----
Written so the AddPhotos focal-plane route computes exactly f = 500 px:
    f_px = focal_mm * 1000 / pixel_size_um,  pixel_size_um = 1000 / resolution
with FocalLength = 5 mm, FocalPlaneXResolution = 100 px/mm (unit 4) -> 10 um.
PixelXDimension equals the real width, so the resize correction is a no-op.

Output
------
    data/scene_0.jpg, scene_1.jpg, scene_2.jpg   (480x360 8-bit grey, q92)
    data/scene_truth.json                        (world->camera R,t; relative pose)
"""

import json
import os

import numpy as np

WIDTH = 480
HEIGHT = 360
FOCAL_PX = 500.0
SUPERSAMPLE = 2  # 2x2 supersampling per pixel

# EXIF values that make AddPhotos recover exactly f = 500 px.
FOCAL_MM = 5.0
FOCAL_PLANE_RESOLUTION_PER_MM = 100.0  # -> pixel_size_um = 1000 / 100 = 10 um
FOCAL_PLANE_UNIT_MM = 4  # 4 == millimetre

# ---------------------------------------------------------------------------
# Scene: a ground plane (Z = 0) and a vertical wall (Y = 0) meeting at an edge.
# World frame: +X right, +Y into the scene, +Z up (right-handed).
#
# The cameras sit at Y = -16 looking toward +Y, so the wall at Y = 0 is 16 units
# in front of them and the ground lies between the cameras and the wall (the
# ground extends from Y = 0 back under/past the cameras). The edge where the two
# surfaces meet is the line Y = 0, Z = 0.
# ---------------------------------------------------------------------------
GROUND_Y_NEAR = -40.0  # ground extends from here...
GROUND_Y_FAR = 0.0     # ...up to the wall base
WALL_Z = 6.0           # wall rises from Z = 0 to here
WALL_X = 10.0          # wall spans X in [-WALL_X, WALL_X]
GROUND_X = 10.0        # ground spans X in [-GROUND_X, GROUND_X]

# Three cameras on a 4-unit lateral baseline at 16 units from the wall, aimed at
# a distant point so the convergence (rotation) between views stays a few
# degrees while the total baseline is 25% of the depth to the scene. (The
# combination was tuned by sweeping baseline and aim: adjacent views share ~86%
# of their visible surface and the two outer views ~73%, while the rotation
# between the outer views is 5 degrees. Both numbers are printed on each run.)
LOOK_AT = np.array([0.0, 30.0, 1.0])
CAMERA_POSITIONS = [
    np.array([-2.0, -16.0, 2.00]),
    np.array([0.0, -16.0, 2.05]),
    np.array([2.0, -16.0, 2.00]),
]
WORLD_UP = np.array([0.0, 0.0, 1.0])

IMAGE_NAMES = ["scene_0.jpg", "scene_1.jpg", "scene_2.jpg"]

# ---------------------------------------------------------------------------
# The SfM sequence: eight views of the same scene from cameras on a horizontal
# arc around the wall/ground corner (the edge at Y = 0, Z = 0). The arc spans
# about 40 degrees in total (5-6 degrees per step) at a similar distance, and
# every camera looks at a point near the corner, with a little height variation
# so the path is not degenerate. Consecutive views share > 70% of their visible
# surface and views 0 and 7 share > 30%; both numbers are measured and printed.
# ---------------------------------------------------------------------------
SEQ_IMAGE_NAMES = ["seq_%d.jpg" % i for i in range(8)]
SEQ_RADIUS = 16.0                        # arc radius (distance to the corner)
SEQ_ARC_CENTRE = np.array([0.0, 0.0])    # (X, Y) the arc is swept about: the corner
SEQ_ARC_DEGREES = 40.0                   # total angular span
SEQ_STEPS = 8
SEQ_LOOK_AT = np.array([0.0, 0.0, 1.0])  # every camera aims at the corner


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------
def look_at(cam_pos, target, world_up):
    """World->camera R (rows: right, down, forward) and t with X_cam = R*X + t.

    Camera +Z is forward (toward the target), +X is right, +Y is down. The world
    is right-handed, so right = forward x up and down = forward x right.
    """
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    t = -R @ cam_pos
    return R, t


def sequence_poses():
    """Eight poses on a horizontal arc around the wall/ground corner.

    The cameras are placed on a circle of radius SEQ_RADIUS centred on
    SEQ_ARC_CENTRE in the ground plane, swept through SEQ_ARC_DEGREES so
    consecutive views differ by a few degrees. Each is dropped toward the ground
    by a different amount (a small, non-monotonic height offset) so the path is
    not planar-degenerate, and every camera looks at the corner.
    """
    poses = []
    half = SEQ_ARC_DEGREES / 2.0
    for i in range(SEQ_STEPS):
        # Sweep from -half to +half degrees about the arc centre.
        angle = np.deg2rad(-half + SEQ_ARC_DEGREES * i / (SEQ_STEPS - 1))
        x = SEQ_ARC_CENTRE[0] + SEQ_RADIUS * np.sin(angle)
        y = SEQ_ARC_CENTRE[1] - SEQ_RADIUS * np.cos(angle)
        # A little height variation (+/- 0.5), non-monotonic so no three
        # consecutive cameras lie in one plane.
        z = 2.0 + 0.5 * np.sin(2.2 * i)
        cam_pos = np.array([x, y, z])
        R, t = look_at(cam_pos, SEQ_LOOK_AT, WORLD_UP)
        poses.append((cam_pos, R, t))
    return poses


def project(point, R, t):
    """Project a world point to pixels under the pinhole model."""
    p_cam = R @ point + t
    u = WIDTH / 2.0 + FOCAL_PX * p_cam[0] / p_cam[2]
    v = HEIGHT / 2.0 + FOCAL_PX * p_cam[1] / p_cam[2]
    return u, v


def ray_direction(u, v, R):
    """World-frame ray direction through pixel (u, v): (dx, dy, 1) camera frame."""
    dx = (u - WIDTH / 2.0) / FOCAL_PX
    dy = (v - HEIGHT / 2.0) / FOCAL_PX
    # R rows are right/down/forward, so d_world = dx*right + dy*down + 1*forward.
    return dx * R[0] + dy * R[1] + 1.0 * R[2]


# ---------------------------------------------------------------------------
# Procedural texture: domain-warped multi-octave value noise plus random
# blobs, discs and line segments. Non-repetitive and locally distinctive.
# ---------------------------------------------------------------------------
def _hash_lattice(ix, iy, seed):
    """Deterministic hash of integer lattice coords -> [0, 1). Vectorised."""
    ix = ix.astype(np.uint64)
    iy = iy.astype(np.uint64)
    h = ix * np.uint64(374761393) + iy * np.uint64(668265263)
    h += np.uint64(seed) * np.uint64(2654435761)
    h ^= h >> np.uint64(13)
    h *= np.uint64(1274126177)
    h ^= h >> np.uint64(16)
    return (h & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0


def _value_noise(x, y, freq, seed):
    """Bilinear-smoothed lattice noise in [0, 1) at the given frequency."""
    xs = x * freq
    ys = y * freq
    ix = np.floor(xs).astype(np.int64)
    iy = np.floor(ys).astype(np.int64)
    fx = xs - ix
    fy = ys - iy
    ux = fx * fx * (3.0 - 2.0 * fx)  # smoothstep
    uy = fy * fy * (3.0 - 2.0 * fy)
    v00 = _hash_lattice(ix, iy, seed)
    v10 = _hash_lattice(ix + 1, iy, seed)
    v01 = _hash_lattice(ix, iy + 1, seed)
    v11 = _hash_lattice(ix + 1, iy + 1, seed)
    top = v00 * (1.0 - ux) + v10 * ux
    bot = v01 * (1.0 - ux) + v11 * ux
    return top * (1.0 - uy) + bot * uy


# Octaves chosen so the finest features stay a few pixels wide at the fixture
# resolution (nearer frequencies alias and inflate the JPEG). The decay is gentle
# so the fine octaves keep enough amplitude to survive the contrast
# normalisation, and so the detector finds extrema at several scales.
_OCTAVE_BASE_FREQ = 0.7
_OCTAVE_COUNT = 6
_OCTAVE_DECAY = 0.68


def _domain_warp(x, y, seed, strength=0.08):
    """Displace (x, y) by a low-frequency value-noise field.

    Pure value noise has its extrema on a regular lattice, which is itself a mild
    repetition; warping the domain breaks that so no two features of the same
    scale look identical. The warp frequency is well below the octave base, so it
    bends large features without destroying the fine ones.
    """
    wx = _value_noise(x, y, 0.35, seed + 1013) - 0.5
    wy = _value_noise(x, y, 0.35, seed + 2027) - 0.5
    return x + strength * wx, y + strength * wy


def _value_noise_multi(x, y, seed):
    """Sum of `_OCTAVE_COUNT` value-noise octaves, normalised to about [0, 1]."""
    value = np.zeros_like(x)
    amp = 0.5
    freq = _OCTAVE_BASE_FREQ
    total = 0.0
    for octave in range(_OCTAVE_COUNT):
        value += amp * _value_noise(x, y, freq, seed + octave * 7)
        total += amp
        amp *= _OCTAVE_DECAY
        freq *= 2.0
    return value / total


def _random_blobs(rng, count, xlo, xhi, ylo, yhi, sigma_lo, sigma_hi):
    """`count` Gaussian blobs: random centre, sigma and signed amplitude."""
    blobs = []
    for _ in range(count):
        cx = rng.uniform(xlo, xhi)
        cy = rng.uniform(ylo, yhi)
        sigma = rng.uniform(sigma_lo, sigma_hi)
        amp = rng.uniform(0.35, 1.0) * (1.0 if rng.random() < 0.5 else -1.0)
        blobs.append((cx, cy, sigma, amp))
    return blobs


def _blob_field(x, y, blobs):
    out = np.zeros_like(x)
    for cx, cy, sigma, amp in blobs:
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        out += amp * np.exp(-r2 / (2.0 * sigma * sigma))
    return out


def _disc_field(x, y, discs):
    """Hard-edged discs: strong localisable step edges for the detector."""
    out = np.zeros_like(x)
    for cx, cy, radius, amp in discs:
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        out += amp * (r2 <= radius * radius)
    return out


def _segment_field(x, y, segments):
    """Short, random-oriented line segments on a near-zero background.

    Each segment is a thin anti-aliased capsule, so it produces distinctive
    corner and blob responses along a random direction rather than a shared one.
    """
    out = np.zeros_like(x)
    for x0, y0, x1, y1, amp, width in segments:
        ex, ey = x1 - x0, y1 - y0
        length2 = ex * ex + ey * ey
        if length2 <= 0.0:
            continue
        t = ((x - x0) * ex + (y - y0) * ey) / length2
        t = np.clip(t, 0.0, 1.0)
        px, py = x0 + t * ex, y0 + t * ey
        dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
        out += amp * np.exp(-(dist * dist) / (2.0 * width * width))
    return out


def _random_segments(rng, count, xlo, xhi, ylo, yhi, len_lo, len_hi):
    segments = []
    for _ in range(count):
        x0 = rng.uniform(xlo, xhi)
        y0 = rng.uniform(ylo, yhi)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        length = rng.uniform(len_lo, len_hi)
        x1 = x0 + length * np.cos(angle)
        y1 = y0 + length * np.sin(angle)
        amp = rng.uniform(0.3, 0.9) * (1.0 if rng.random() < 0.5 else -1.0)
        width = rng.uniform(0.02, 0.08)
        segments.append((x0, y0, x1, y1, amp, width))
    return segments


def _random_discs(rng, count, xlo, xhi, ylo, yhi, r_lo, r_hi):
    discs = []
    for _ in range(count):
        cx = rng.uniform(xlo, xhi)
        cy = rng.uniform(ylo, yhi)
        radius = rng.uniform(r_lo, r_hi)
        amp = rng.uniform(0.2, 0.6) * (1.0 if rng.random() < 0.5 else -1.0)
        discs.append((cx, cy, radius, amp))
    return discs


class Surface:
    """The random scene elements of one surface, drawn once from a seed.

    The random blobs/discs/segments live in the surface's own (x, y) frame; the
    visible ground runs from Y = -16 (under the cameras) to Y = 0 (the wall base)
    and the visible wall from Z = 0 to roughly Z = 3, so the sampling ranges below
    span a little beyond what any camera sees.
    """

    def __init__(self, seed, xlo, xhi, ylo, yhi):
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.blobs = _random_blobs(rng, 140, xlo, xhi, ylo, yhi, 0.04, 0.5)
        self.discs = _random_discs(rng, 120, xlo, xhi, ylo, yhi, 0.02, 0.25)
        self.segments = _random_segments(rng, 260, xlo, xhi, ylo, yhi, 0.04, 0.4)

        # Contrast normalisation must be a *fixed* affine map so that the same
        # world point gets the same grey in every view. Measure its [p2, p98]
        # range once here, on a fixed lattice spanning the surface, and reuse
        # those bounds when sampling any set of points. (Percentiles rather than
        # min/max, so one outlier cannot flatten the midtones.)
        gx, gy = np.meshgrid(np.linspace(xlo, xhi, 256), np.linspace(ylo, yhi, 256))
        reference = self._raw(gx.ravel(), gy.ravel())
        self.lo = float(np.percentile(reference, 2.0))
        self.hi = float(np.percentile(reference, 98.0))

    def _raw(self, x, y):
        """The un-normalised surface value, a pure function of (x, y)."""
        wx, wy = _domain_warp(x, y, self.seed)
        value = 0.55 * _value_noise_multi(wx, wy, self.seed + 1)
        value = value + 0.20 * _blob_field(x, y, self.blobs)
        value = value + 0.15 * _disc_field(x, y, self.discs)
        value = value + 0.10 * _segment_field(x, y, self.segments)
        return value

    def sample(self, x, y):
        """Surface value in [0, 1], stretched to a per-surface fixed [0, 255]."""
        if self.hi <= self.lo:
            return np.zeros_like(x)
        return np.clip((self._raw(x, y) - self.lo) / (self.hi - self.lo), 0.0, 1.0)


# Different seeds for the ground and the wall so the two surfaces are
# uncorrelated. The sampled ranges cover the visible extent of each surface.
GROUND_SURFACE = Surface(seed=101, xlo=-14.0, xhi=14.0, ylo=-30.0, yhi=0.0)
WALL_SURFACE = Surface(seed=907, xlo=-12.0, xhi=12.0, ylo=0.0, yhi=6.0)


# ---------------------------------------------------------------------------
# Ray casting
# ---------------------------------------------------------------------------
def _on_ground(px, py):
    return (np.abs(px) <= GROUND_X) & (py >= GROUND_Y_NEAR) & (py <= GROUND_Y_FAR)


def _intersect_ground(origin, direction):
    """Nearest positive hit on the finite ground plane (Z = 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (0.0 - origin[2]) / direction[..., 2]
    px = origin[0] + t * direction[..., 0]
    py = origin[1] + t * direction[..., 1]
    valid = (t > 1e-6) & _on_ground(px, py)
    return t, px, py, valid


def _intersect_wall(origin, direction):
    """Nearest positive hit on the finite vertical wall (Y = 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (0.0 - origin[1]) / direction[..., 1]
    px = origin[0] + t * direction[..., 0]
    pz = origin[2] + t * direction[..., 2]
    valid = (t > 1e-6) & (np.abs(px) <= WALL_X) & (pz >= 0.0) & (pz <= WALL_Z)
    return t, px, pz, valid


def cast_pixels(cam_pos, R, t, sample_w, sample_h):
    """Ray-cast `sample_w` x `sample_h` sub-samples; return colour and depth.

    Sub-sample (i, j) sits at continuous pixel coordinates ((i + 0.5) / s,
    (j + 0.5) / s) with s = sample_w / WIDTH, i.e. at the centre of each
    sub-pixel, so averaging s x s blocks reproduces uniform supersampling.
    """
    s = sample_w // WIDTH
    us = (np.arange(sample_w) + 0.5) / s  # pixel x per sub-sample column
    vs = (np.arange(sample_h) + 0.5) / s
    uu, vv = np.meshgrid(us, vs)

    dx = (uu - WIDTH / 2.0) / FOCAL_PX
    dy = (vv - HEIGHT / 2.0) / FOCAL_PX
    right, down, forward = R[0], R[1], R[2]
    direction = dx[..., None] * right + dy[..., None] * down + forward

    t_g, gx, gy, ground_valid = _intersect_ground(cam_pos, direction)
    t_w, wx, wz, wall_valid = _intersect_wall(cam_pos, direction)

    # Nearest hit wins; ties (the shared edge line) are unknowable, prefer ground.
    hit_wall = wall_valid & (~ground_valid | (t_w < t_g))
    hit_ground = ground_valid & ~hit_wall
    hit = hit_wall | hit_ground

    image = np.zeros((sample_h, sample_w), dtype=np.float64)
    if np.any(hit_ground):
        image[hit_ground] = GROUND_SURFACE.sample(gx[hit_ground], gy[hit_ground])
    if np.any(hit_wall):
        image[hit_wall] = WALL_SURFACE.sample(wx[hit_wall], wz[hit_wall])

    depth = np.where(hit, np.where(hit_ground, t_g, t_w), np.nan)
    return image, hit, depth


def render_image(cam_pos, R, t):
    """Render one image with 2x2 supersampling, as an 8-bit grey array."""
    sw = WIDTH * SUPERSAMPLE
    sh = HEIGHT * SUPERSAMPLE
    image, _hit, _depth = cast_pixels(cam_pos, R, t, sw, sh)
    # Average each 2x2 block of sub-samples down to one pixel.
    image = image.reshape(HEIGHT, SUPERSAMPLE, WIDTH, SUPERSAMPLE).mean(axis=(1, 3))
    return np.clip(np.rint(image * 255.0), 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Self-check: the stored truth must project a hit exactly back to its pixel,
# and a known surface point projected to a pixel must ray-cast back to it.
# ---------------------------------------------------------------------------
def _nearest_hit_1d(cam_pos, direction):
    """Scalar version of the ray cast, for the self-check."""
    best_t = None
    best_p = None
    if abs(direction[2]) > 1e-12:
        t = -cam_pos[2] / direction[2]
        if t > 1e-6:
            p = cam_pos + t * direction
            if abs(p[0]) <= GROUND_X and GROUND_Y_NEAR <= p[1] <= GROUND_Y_FAR:
                best_t, best_p = t, p
    if abs(direction[1]) > 1e-12:
        t = -cam_pos[1] / direction[1]
        if t > 1e-6 and (best_t is None or t < best_t):
            p = cam_pos + t * direction
            if abs(p[0]) <= WALL_X and 0.0 <= p[2] <= WALL_Z:
                best_t, best_p = t, p
    return best_p


def self_check(cam_pos, R, t):
    """Verify the projection and the ray-cast agree, guarding convention slips."""
    # -- pixels -> hit -> pixel ---------------------------------------------
    for (u, v) in [(0.5, 0.5), (WIDTH - 0.5, 0.5), (0.5, HEIGHT - 0.5),
                   (WIDTH - 0.5, HEIGHT - 0.5), (240.5, 180.5), (100.5, 260.5),
                   (380.5, 90.5), (240.5, 350.5)]:
        point = _nearest_hit_1d(cam_pos, ray_direction(u, v, R))
        if point is None:
            continue  # a pixel that sees only background
        u2, v2 = project(point, R, t)
        assert abs(u2 - u) < 1e-6 and abs(v2 - v) < 1e-6, (
            "pixel (%r, %r) round-tripped to (%r, %r)" % (u, v, u2, v2))

    # -- known 3D points -> pixel -> hit ------------------------------------
    for point in [np.array([0.0, 0.0, 0.0]),      # the corner
                  np.array([0.0, -1.0, 0.0]),     # on the ground
                  np.array([0.0, 0.0, 1.0]),      # on the wall
                  np.array([1.0, 0.0, 0.5]),
                  np.array([-1.0, -0.5, 0.0]),
                  np.array([2.0, -3.0, 0.0])]:
        u, v = project(point, R, t)
        if not (0.0 <= u < WIDTH and 0.0 <= v < HEIGHT):
            continue  # outside this view
        hit = _nearest_hit_1d(cam_pos, ray_direction(u, v, R))
        assert hit is not None, "known point projected to a background pixel"
        assert np.linalg.norm(hit - point) < 1e-6, (
            "known point %s hit %s" % (point, hit))


def coverage(grey):
    return float(np.count_nonzero(grey)) / grey.size


def visible_points(cam_pos, R, t, step=2):
    """World points visible in a view, sampled every `step` pixels (one per pixel
    is not needed to estimate coverage; every 2nd pixel keeps this quick)."""
    xs = np.arange(0, WIDTH, step) + 0.5
    ys = np.arange(0, HEIGHT, step) + 0.5
    uu, vv = np.meshgrid(xs, ys)
    dx = (uu - WIDTH / 2.0) / FOCAL_PX
    dy = (vv - HEIGHT / 2.0) / FOCAL_PX
    right, down, forward = R[0], R[1], R[2]
    direction = dx[..., None] * right + dy[..., None] * down + forward

    t_g, gx, gy, ground_valid = _intersect_ground(cam_pos, direction)
    t_w, wx, wz, wall_valid = _intersect_wall(cam_pos, direction)
    hit_wall = wall_valid & (~ground_valid | (t_w < t_g))
    hit_ground = ground_valid & ~hit_wall
    hit = hit_wall | hit_ground

    tg = t_g[..., None]
    tw = t_w[..., None]
    pts = np.where(hit_ground[..., None], cam_pos + tg * direction,
                   cam_pos + tw * direction)
    return pts[hit], hit.sum()


def pair_overlap(poses, i, j, step=2):
    """Fraction of the smaller view's visible surface shared with the other.

    A world point visible in view i is "shared" when it also projects inside
    view j and the ray-cast from j hits (approximately) the same point, i.e. it
    is genuinely visible there and not occluded.
    """
    pos_i, Ri, ti = poses[i]
    pos_j, Rj, tj = poses[j]
    pts_i, count_i = visible_points(pos_i, Ri, ti, step)
    _, count_j = visible_points(pos_j, Rj, tj, step)

    p_cam = pts_i @ Rj.T + tj
    with np.errstate(divide="ignore", invalid="ignore"):
        u = WIDTH / 2.0 + FOCAL_PX * p_cam[:, 0] / p_cam[:, 2]
        v = HEIGHT / 2.0 + FOCAL_PX * p_cam[:, 1] / p_cam[:, 2]
    in_frame = (p_cam[:, 2] > 0.0) & (u >= 0.0) & (u < WIDTH) & (v >= 0.0) & (v < HEIGHT)

    shared = 0
    for k in np.nonzero(in_frame)[0]:
        direction = ray_direction(u[k], v[k], Rj)
        hit = _nearest_hit_1d(pos_j, direction)
        if hit is not None and np.linalg.norm(hit - pts_i[k]) < 1e-3:
            shared += 1
    return shared / float(min(count_i, count_j))


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------
def scene_exif():
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

    exif = Image.Exif()
    exif[0x010F] = "AlphaShape"        # Make
    exif[0x0110] = "SceneRig-1"        # Model
    exif[0x0112] = 1                   # Orientation: normal
    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0x9003] = "2024:06:15 09:30:00"
    exif_ifd[0x920A] = IFDRational(int(FOCAL_MM), 1)                       # FocalLength (mm)
    exif_ifd[0xA20E] = IFDRational(int(FOCAL_PLANE_RESOLUTION_PER_MM), 1)  # px per unit (X)
    exif_ifd[0xA20F] = IFDRational(int(FOCAL_PLANE_RESOLUTION_PER_MM), 1)  # px per unit (Y)
    exif_ifd[0xA210] = FOCAL_PLANE_UNIT_MM  # resolution unit: mm == 4
    exif_ifd[0xA002] = WIDTH                # PixelXDimension
    exif_ifd[0xA003] = HEIGHT               # PixelYDimension
    exif_ifd[0xA431] = "SCENE-0001"         # BodySerialNumber
    exif_ifd[0xA434] = "AlphaShape 5mm f/2"
    # Deliberately no GPS IFD: these cameras have no measured positions.
    return exif


def write_image(path, grey, exif):
    from PIL import Image

    Image.fromarray(grey, mode="L").save(path, quality=92, exif=exif.tobytes(),
                                         optimize=True)


# ---------------------------------------------------------------------------
# WGS84 geodesy. Written out by hand here (no pyproj) so this script is an
# independent cross-check of the C++ geodesy: if the two disagree, the fixtures
# and the engine cannot both be right.
#
# The scene world frame is treated as local ENU in metres: +X = east,
# +Y = north, +Z = up, with its origin at the geodetic point below.
# ---------------------------------------------------------------------------
WGS84_A = 6378137.0                      # semi-major axis (metres)
WGS84_F = 1.0 / 298.257223563            # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)     # first eccentricity squared

GEO_ORIGIN_LAT = 45.0     # degrees
GEO_ORIGIN_LON = 7.0      # degrees
GEO_ORIGIN_HEIGHT = 300.0  # ellipsoidal height, metres

GEO_SIGMA_EAST = 0.02     # 1-sigma horizontal, metres
GEO_SIGMA_NORTH = 0.02
GEO_SIGMA_UP = 0.04       # 1-sigma vertical, metres
GEO_NOISE_SEED = 20240615

GEO_IMAGE_NAMES = ["geo_seq_%d.jpg" % i for i in range(8)]
CAL_IMAGE_NAMES = ["cal_seq_%d.jpg" % i for i in range(8)]


def geodetic_to_ecef(lat_deg, lon_deg, height):
    """WGS84 (lat, lon, h) -> ECEF (x, y, z), metres."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + height) * cos_lat * np.cos(lon)
    y = (n + height) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + height) * sin_lat
    return np.array([x, y, z])


def ecef_to_geodetic(x, y, z):
    """ECEF (x, y, z) -> WGS84 (lat_deg, lon_deg, height), metres.

    Fixed-point iteration on the latitude (its convergence is geometric with a
    factor of about e^2 = 0.0067, so a dozen rounds reach well under a
    nanometre), then the height from the prime-vertical radius.
    """
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(30):
        sin_lat = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        height = p / np.cos(lat) - n
        lat_next = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + height)))
        if abs(lat_next - lat) < 1e-15:
            lat = lat_next
            break
        lat = lat_next
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    height = p / np.cos(lat) - n
    return np.rad2deg(lat), np.rad2deg(lon), float(height)


def enu_rotation(lat_deg, lon_deg):
    """The ECEF->ENU rotation matrix R (rows east, north, up) at a point."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    return np.array([
        [-sin_lon, cos_lon, 0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ])


def enu_to_ecef(enu, origin_ecef, rotation):
    """Local ENU (east, north, up) -> ECEF about the given origin."""
    return origin_ecef + rotation.T @ np.asarray(enu, dtype=float)


def ecef_to_enu(ecef, origin_ecef, rotation):
    """ECEF -> local ENU about the given origin."""
    return rotation @ (np.asarray(ecef, dtype=float) - origin_ecef)


def _dms_rationals(value_deg):
    """Decimal degrees as (deg, min, sec) IFDRationals, seconds to 1e-6 arcsec.

    The decomposition is done in integer micro-arcseconds so rounding can never
    leave a value like 60.000000 seconds (which would be an invalid DMS triple)
    or push a coordinate past 1 mm on the ground.
    """
    from PIL.TiffImagePlugin import IFDRational

    total = int(round(value_deg * 3600.0 * 1e6))  # micro-arcseconds
    deg = total // (3600 * 1000000)
    total -= deg * 3600 * 1000000
    minute = total // (60 * 1000000)
    total -= minute * 60 * 1000000
    sec_uas = total
    return (IFDRational(deg, 1), IFDRational(minute, 1), IFDRational(sec_uas, 1000000))


def geo_exif(lat_deg, lon_deg, height_m):
    """The scene EXIF plus a GPS IFD carrying the given position."""
    from PIL.TiffImagePlugin import IFDRational

    exif = scene_exif()
    gps = exif.get_ifd(0x8825)
    gps[0x0001] = "N"                       # GPSLatitudeRef
    gps[0x0002] = _dms_rationals(lat_deg)   # GPSLatitude
    gps[0x0003] = "E"                       # GPSLongitudeRef
    gps[0x0004] = _dms_rationals(lon_deg)   # GPSLongitude
    gps[0x0005] = 0                         # GPSAltitudeRef: above sea level
    # GPSAltitude, metres * 1000 over 1000, so the written value keeps mm.
    gps[0x0006] = IFDRational(int(round(height_m * 1000.0)), 1000)
    return exif


def write_geo_sequence(here):
    """Eight GPS-tagged views of the scene, on the seq_* camera path.

    The scene world frame is local ENU about (GEO_ORIGIN_LAT, GEO_ORIGIN_LON,
    GEO_ORIGIN_HEIGHT). Each camera's true centre (metres, ENU) is converted to
    geodetic, perturbed by deterministic Gaussian noise applied in ENU first,
    and written as EXIF GPS. The pixels are rendered with the same pinhole model
    as seq_*, so they match seq_* pixel for pixel; the file differs only in its
    GPS IFD (hence the re-render rather than a copy).
    """
    import numpy as np

    poses = sequence_poses()
    origin_ecef = geodetic_to_ecef(GEO_ORIGIN_LAT, GEO_ORIGIN_LON, GEO_ORIGIN_HEIGHT)
    rotation = enu_rotation(GEO_ORIGIN_LAT, GEO_ORIGIN_LON)

    # Self-check the geodetic round-trip before writing anything.
    for cam_pos, _R, _t in poses:
        ecef = enu_to_ecef(cam_pos, origin_ecef, rotation)
        back = ecef_to_enu(ecef, origin_ecef, rotation)
        assert np.linalg.norm(back - cam_pos) < 1e-9, "ENU<->ECEF round-trip drifted"

    rng = np.random.default_rng(GEO_NOISE_SEED)

    truth = {
        "width": WIDTH,
        "height": HEIGHT,
        "f": FOCAL_PX,
        "enu_origin": {
            "lat": GEO_ORIGIN_LAT,
            "lon": GEO_ORIGIN_LON,
            "height": GEO_ORIGIN_HEIGHT,
        },
        "images": [],
        "centres": [],
    }

    for name, (cam_pos, R, t) in zip(GEO_IMAGE_NAMES, poses):
        # True centre, ENU metres.
        true_enu = cam_pos
        # Noise-free geodetic position of the true centre.
        lat0, lon0, h0 = ecef_to_geodetic(*enu_to_ecef(true_enu, origin_ecef, rotation))
        # Deterministic Gaussian noise, applied in ENU then converted.
        noisy_enu = true_enu + np.array([
            rng.normal(0.0, GEO_SIGMA_EAST),
            rng.normal(0.0, GEO_SIGMA_NORTH),
            rng.normal(0.0, GEO_SIGMA_UP),
        ])
        lat, lon, h = ecef_to_geodetic(*enu_to_ecef(noisy_enu, origin_ecef, rotation))

        grey = render_image(cam_pos, R, t)
        write_image(os.path.join(here, name), grey, geo_exif(lat, lon, h))
        size = os.path.getsize(os.path.join(here, name))
        assert size < 60 * 1024, "%s is %d bytes (>= 60 KB)" % (name, size)
        print("wrote %s (%dx%d, %.1f%% non-black, %d bytes, GPS %.7f/%.7f/%.3f)"
              % (name, WIDTH, HEIGHT, 100.0 * coverage(grey), size, lat, lon, h))

        truth["images"].append({
            "file": name,
            "R": R.tolist(),          # X_cam = R * X_world + t
            "t": t.tolist(),
            "f": FOCAL_PX,
            "W": WIDTH,
            "H": HEIGHT,
            "gps_written": [lat, lon, h],       # what was written to EXIF
            "gps_true": [lat0, lon0, h0],       # noise-free geodetic position
            "enu_true": true_enu.tolist(),      # true centre, ENU metres
        })
        truth["centres"].append(cam_pos.tolist())

    with open(os.path.join(here, "geo_seq_truth.json"), "w") as out:
        json.dump(truth, out, indent=2)
        out.write("\n")
    print("wrote geo_seq_truth.json (%d images)" % len(truth["images"]))


# ---------------------------------------------------------------------------
# Lens-distortion sequence. The true intrinsics are the Brown model in
# src/model/camera/calibration.hpp, but the EXIF is written 5% low (f = 475)
# so AddPhotos starts the sensor at a focal length that is deliberately wrong.
# ---------------------------------------------------------------------------
CAL_TRUE_INTRINSICS = {
    "f": 500.0,
    "cx": 3.0,
    "cy": -2.0,
    "k1": -0.08,
    "k2": 0.02,
    "k3": 0.0,
    "k4": 0.0,
    "p1": 2e-4,
    "p2": -1e-4,
    "b1": 0.0,
    "b2": 0.0,
}
CAL_EXIF_FOCAL_PX = 475.0
CAL_EXIF_FOCAL_MM = 4.75  # 4.75 mm * (100 px/mm) = 475 px


def brown_apply(xs, ys, intr):
    """Forward Brown distortion on the normalised plane (vectorised)."""
    k1, k2 = intr["k1"], intr["k2"]
    p1, p2 = intr["p1"], intr["p2"]
    r2 = xs * xs + ys * ys
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = xs * radial + p1 * (r2 + 2.0 * xs * xs) + 2.0 * p2 * xs * ys
    yd = ys * radial + p2 * (r2 + 2.0 * ys * ys) + 2.0 * p1 * xs * ys
    return xd, yd


def brown_undistort(xd, yd, intr):
    """Invert the Brown distortion with Newton's method (vectorised).

    Returns (xs, ys) with the residual on the distorted plane below 1e-12 for
    every element (asserted by the caller)."""
    k1, k2 = intr["k1"], intr["k2"]
    p1, p2 = intr["p1"], intr["p2"]
    x = np.array(xd, dtype=float, copy=True)
    y = np.array(yd, dtype=float, copy=True)
    for _ in range(60):
        r2 = x * x + y * y
        s = 1.0 + k1 * r2 + k2 * r2 * r2
        q = k1 + 2.0 * k2 * r2
        fx_model = x * s + p1 * (r2 + 2.0 * x * x) + 2.0 * p2 * x * y
        fy_model = y * s + p2 * (r2 + 2.0 * y * y) + 2.0 * p1 * x * y
        fx = fx_model - xd
        fy = fy_model - yd
        if max(np.max(np.abs(fx)), np.max(np.abs(fy))) < 1e-14:
            break
        j00 = s + 2.0 * x * x * q + 6.0 * p1 * x + 2.0 * p2 * y
        j01 = 2.0 * x * y * q + 2.0 * p1 * y + 2.0 * p2 * x
        j10 = 2.0 * x * y * q + 2.0 * p2 * x + 2.0 * p1 * y
        j11 = s + 2.0 * y * y * q + 6.0 * p2 * y + 2.0 * p1 * x
        det = j00 * j11 - j01 * j10
        x = x - (j11 * fx - j01 * fy) / det
        y = y - (j00 * fy - j10 * fx) / det
    return x, y


def cal_directions(uu, vv, R, intr):
    """World-frame ray directions through pixels (uu, vv) under the Brown model."""
    f, cx, cy = intr["f"], intr["cx"], intr["cy"]
    b1, b2 = intr["b1"], intr["b2"]
    # Undo the affine map (v is solved for y' first, then u for x').
    yd = (vv - HEIGHT / 2.0 - cy) / f
    xd = (uu - WIDTH / 2.0 - cx - yd * b2) / (f + b1)
    xs, ys = brown_undistort(xd, yd, intr)
    right, down, forward = R[0], R[1], R[2]
    return xs[..., None] * right + ys[..., None] * down + forward


def cast_directions(cam_pos, direction):
    """Ray-cast a precomputed direction grid; returns colour (and hit mask)."""
    t_g, gx, gy, ground_valid = _intersect_ground(cam_pos, direction)
    t_w, wx, wz, wall_valid = _intersect_wall(cam_pos, direction)
    hit_wall = wall_valid & (~ground_valid | (t_w < t_g))
    hit_ground = ground_valid & ~hit_wall
    hit = hit_wall | hit_ground
    image = np.zeros(direction.shape[:-1], dtype=np.float64)
    if np.any(hit_ground):
        image[hit_ground] = GROUND_SURFACE.sample(gx[hit_ground], gy[hit_ground])
    if np.any(hit_wall):
        image[hit_wall] = WALL_SURFACE.sample(wx[hit_wall], wz[hit_wall])
    return image, hit


def render_image_cal(cam_pos, R, t, intr):
    """Render one distorted image with 2x2 supersampling, as 8-bit grey."""
    s = SUPERSAMPLE
    sw, sh = WIDTH * s, HEIGHT * s
    us = (np.arange(sw) + 0.5) / s
    vs = (np.arange(sh) + 0.5) / s
    uu, vv = np.meshgrid(us, vs)
    direction = cal_directions(uu, vv, R, intr)
    image, _hit = cast_directions(cam_pos, direction)
    image = image.reshape(HEIGHT, s, WIDTH, s).mean(axis=(1, 3))
    return np.clip(np.rint(image * 255.0), 0.0, 255.0).astype(np.uint8)


def cal_project(point, R, t, intr):
    """Project a world point forward through the Brown model to a pixel."""
    p_cam = R @ point + t
    xs, ys = p_cam[0] / p_cam[2], p_cam[1] / p_cam[2]
    xd, yd = brown_apply(xs, ys, intr)
    u = WIDTH / 2.0 + intr["cx"] + xd * (intr["f"] + intr["b1"]) + yd * intr["b2"]
    v = HEIGHT / 2.0 + intr["cy"] + yd * intr["f"]
    return float(u), float(v)


def cal_pixel_ray(u, v, R, intr):
    """World-frame ray direction through pixel (u, v) under the Brown model."""
    xs, ys = brown_undistort(
        np.array([(u - WIDTH / 2.0 - intr["cx"]) / (intr["f"] + intr["b1"])]),
        np.array([(v - HEIGHT / 2.0 - intr["cy"]) / intr["f"]]),
        intr)
    return xs[0] * R[0] + ys[0] * R[1] + 1.0 * R[2]


def cal_self_check(cam_pos, R, t, intr):
    """A known point projected forward must land on the pixel whose ray hits it.

    This is the whole contract of the distortion model, checked the way the test
    will use it: project -> ray-cast -> project again. The forward and inverse
    maps must agree to well under the 0.05 px tolerance."""
    worst = 0.0
    for point in [np.array([0.0, 0.0, 0.0]),
                  np.array([0.0, -1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0]),
                  np.array([1.0, 0.0, 0.5]),
                  np.array([-1.0, -0.5, 0.0]),
                  np.array([2.0, -3.0, 0.0]),
                  np.array([0.5, 0.0, 2.0]),
                  np.array([-2.0, 0.0, 0.8])]:
        u, v = cal_project(point, R, t, intr)
        if not (0.0 <= u < WIDTH and 0.0 <= v < HEIGHT):
            continue  # outside this view
        hit = _nearest_hit_1d(cam_pos, cal_pixel_ray(u, v, R, intr))
        if hit is None:
            continue
        u2, v2 = cal_project(hit, R, t, intr)
        residual = float(np.hypot(u2 - u, v2 - v))
        worst = max(worst, residual)
        assert residual < 0.05, "distortion round-trip %.4f px on %s" % (residual, point)

    # Independently, the inverse map itself must leave a sub-1e-12 residual on
    # the distorted plane for a dense grid of pixels.
    xs = np.linspace(-0.5, 0.5, 41)
    uu, vv = np.meshgrid(xs * WIDTH / FOCAL_PX, xs * HEIGHT / FOCAL_PX)
    yd = (vv - HEIGHT / 2.0 - intr["cy"]) / intr["f"]
    xd = (uu - WIDTH / 2.0 - intr["cx"] - yd * intr["b2"]) / (intr["f"] + intr["b1"])
    xr, yr = brown_undistort(xd, yd, intr)
    xr_f, yr_f = brown_apply(xr, yr, intr)
    assert np.max(np.abs(xr_f - xd)) < 1e-12, "undistort residual too large"
    assert np.max(np.abs(yr_f - yd)) < 1e-12, "undistort residual too large"
    return worst


def cal_exif():
    """The seq EXIF with a 5%-low focal length and its own sensor identity."""
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

    exif = Image.Exif()
    exif[0x010F] = "AlphaShape"           # Make
    exif[0x0110] = "CalRig-1"             # Model
    exif[0x0112] = 1                      # Orientation: normal
    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0x9003] = "2024:06:15 09:30:00"
    # 4.75 mm with a 100 px/mm focal-plane resolution -> f = 475 px.
    exif_ifd[0x920A] = IFDRational(int(round(CAL_EXIF_FOCAL_MM * 100)), 100)
    exif_ifd[0xA20E] = IFDRational(100, 1)
    exif_ifd[0xA20F] = IFDRational(100, 1)
    exif_ifd[0xA210] = 4                  # resolution unit: mm
    exif_ifd[0xA002] = WIDTH
    exif_ifd[0xA003] = HEIGHT
    exif_ifd[0xA431] = "CAL-0001"
    exif_ifd[0xA434] = "AlphaShape 5mm f/2"
    return exif


def write_cal_sequence(here):
    """Eight views on the seq_* path, rendered with lens distortion.

    The pixels carry the true Brown distortion (f = 500, cx = 3, cy = -2,
    k1 = -0.08, k2 = 0.02, p1 = 2e-4, p2 = -1e-4), while the EXIF reports a focal
    length that makes AddPhotos compute f = 475 px — 5% low, so a bundle
    adjustment started from this file has real work to do.
    """
    import numpy as np

    intr = CAL_TRUE_INTRINSICS
    poses = sequence_poses()

    worst = 0.0
    for cam_pos, R, t in poses:
        worst = max(worst, cal_self_check(cam_pos, R, t, intr))
    print("cal self-check: worst forward/inverse round-trip %.4f px" % worst)

    exif = cal_exif()
    for name, (cam_pos, R, t) in zip(CAL_IMAGE_NAMES, poses):
        grey = render_image_cal(cam_pos, R, t, intr)
        path = os.path.join(here, name)
        write_image(path, grey, exif)
        size = os.path.getsize(path)
        assert size < 60 * 1024, "%s is %d bytes (>= 60 KB)" % (name, size)
        print("wrote %s (%dx%d, %.1f%% non-black, %d bytes)"
              % (name, WIDTH, HEIGHT, 100.0 * coverage(grey), size))

    cal_truth = {
        "width": WIDTH,
        "height": HEIGHT,
        "true_intrinsics": dict(intr),
        "exif_f": CAL_EXIF_FOCAL_PX,  # what AddPhotos recovers from the EXIF
        "images": [],
        "centres": [],
    }
    for name, (cam_pos, R, t) in zip(CAL_IMAGE_NAMES, poses):
        cal_truth["images"].append({
            "file": name,
            "R": R.tolist(),          # X_cam = R * X_world + t
            "t": t.tolist(),
            "f": FOCAL_PX,
            "W": WIDTH,
            "H": HEIGHT,
        })
        cal_truth["centres"].append(cam_pos.tolist())

    with open(os.path.join(here, "cal_seq_truth.json"), "w") as out:
        json.dump(cal_truth, out, indent=2)
        out.write("\n")
    print("wrote cal_seq_truth.json (%d images)" % len(cal_truth["images"]))


# ---------------------------------------------------------------------------
# Calibration-grid sequence. Same scene, renderer, texture, true intrinsics and
# distortion pipeline as cal_seq, but the camera network is a proper
# photogrammetric calibration net rather than a shallow arc:
#
#   * 12 cameras converging on the wall/ground corner, spread over ~120 deg of
#     azimuth and 30-40 deg of elevation, so the calibration observables (f, the
#     principal point, the distortion terms) are seen across the whole sensor and
#     no two of them are collinear with pose.
#   * The camera roll varies: a third of the views are rotated +90 deg about the
#     optical axis, a third -90 deg, a third left at 0. Roll is the rotation that
#     rotates the film plane under the lens, so a residual principal-point or
#     distortion error changes sign with it. A rolled camera writes the same
#     sensor, so the images stay 480x360.
#   * Tilted down (~30-40 deg) on a close radius, so almost every pixel sees the
#     textured wall or ground -- the corners included, because distortion is
#     observed mostly at large radii. The textured-pixel fraction is measured and
#     printed per image.
#
# The EXIF is identical to cal_seq (it implies f = 475 px, 5% below the true
# 500), so a bundle adjustment started from this file still has real work to do.
# ---------------------------------------------------------------------------
CALGRID_IMAGE_NAMES = ["calgrid_%d.jpg" % i for i in range(12)]
CALGRID_STEPS = 12
CALGRID_TARGET = np.array([0.0, 0.0, 0.0])  # every camera aims at the corner
# Per-camera azimuth (deg, centred on the corner, spanning 116 deg end to end),
# elevation (deg above the ground plane) and distance from the corner.
#
# Why the layout is irregular. A self-calibration only separates the radial
# distortion from the scene when a given 3D point is seen at *different image
# radii* in different views: if every camera sits at one distance, a point lands
# near the same radius everywhere and a k1 error is absorbed by the structure
# (this is exactly the flat k1/k2/k3 valley that makes cal_seq's self-calibration
# unobservable). Sweeping the distance across the network forces the near cameras
# to see a point large and the far cameras to see it small, which pins the radial
# scale. The elevation varies widely and non-monotonically for the same reason on
# the vertical axis. (The specific values below were found by sweeping the
# network design against the task's own bundle adjustment: several nearby layouts
# still leave a free k3 trading ~2-4% of k1 with pose, so the conditioning is
# genuinely sensitive to the exact geometry, not automatic.)
_CALGRID_AZ_DEG = [-58.0, -47.5, -36.9, -26.4, -15.8, -5.3,
                   5.3, 15.8, 26.4, 36.9, 47.5, 58.0]
_CALGRID_ELEV_DEG = [35.3, 30.6, 19.2, 27.8, 37.1, 16.7,
                     45.7, 45.0, 17.5, 43.3, 22.3, 39.7]
_CALGRID_RADIUS_BY_INDEX = [5.9, 12.3, 15.5, 15.8, 12.0, 6.0,
                            8.7, 7.3, 15.5, 7.6, 10.7, 8.9]
# Roll in degrees about the optical axis: a third +90, a third -90, a third 0.
# Roll preserves the pixel radius, so it does not by itself break the radial
# degeneracy above (the distance sweep does that); it decorrelates the
# *principal point* and the tangential terms from the camera pose -- a residual
# cx shifts a scene point sideways, which a +90-deg view maps onto a vertical
# move, so the two cannot be traded.
_CALGRID_ROLLS = [90.0, -90.0, 0.0, 90.0, -90.0, 0.0,
                  90.0, -90.0, 0.0, 90.0, -90.0, 0.0]

# Every consecutive pair must share at least this fraction of the smaller view's
# visible surface, so the matching graph is connected along the sweep.
CALGRID_MIN_PAIR_OVERLAP = 0.15


def _rxz(angle_deg):
    """Rotation about the camera optical axis (+Z of the camera frame)."""
    t = np.deg2rad(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def calgrid_poses():
    """12 poses: a rolled, elevation/distance-varying arc around the corner.

    Camera i sits at (azimuth, elevation, distance) on a sphere around the
    wall/ground corner, aims at the corner, then rolls about its optical axis by
    the per-view roll. The distance sweep and the 120 deg azimuth spread are what
    make the distortion observables separable from the pose; the roll makes the
    principal point separable.
    """
    poses = []
    for i in range(CALGRID_STEPS):
        az = np.deg2rad(_CALGRID_AZ_DEG[i])
        elev = np.deg2rad(_CALGRID_ELEV_DEG[i])
        radius = _CALGRID_RADIUS_BY_INDEX[i]
        cam_pos = CALGRID_TARGET + radius * np.array([
            np.cos(elev) * np.sin(az),
            -np.cos(elev) * np.cos(az),
            np.sin(elev),
        ])
        R_base, _t = look_at(cam_pos, CALGRID_TARGET, WORLD_UP)
        R = _rxz(_CALGRID_ROLLS[i]) @ R_base
        t = -R @ cam_pos
        poses.append((cam_pos, R, t))
    return poses


def calgrid_textured_fraction(pos, R, t, intr):
    """Fraction of pixels whose central ray hits the textured scene.

    The measure of "no sky": a pixel counts when its ray intersects the finite
    wall or ground (the renderer's own hit test), not when its grey is nonzero --
    a deep shadow in the texture would otherwise read as background.
    """
    us = np.arange(WIDTH) + 0.5
    vs = np.arange(HEIGHT) + 0.5
    uu, vv = np.meshgrid(us, vs)
    direction = cal_directions(uu, vv, R, intr)
    _image, hit = cast_directions(pos, direction)
    return float(hit.mean())


def calgrid_self_check(poses, intr):
    """The in-script projection self-check over every calgrid view."""
    worst = 0.0
    for cam_pos, R, t in poses:
        worst = max(worst, cal_self_check(cam_pos, R, t, intr))
    return worst


def write_calibration_grid(here):
    """Render the 12-view calibration net and calgrid_truth.json."""
    intr = CAL_TRUE_INTRINSICS
    poses = calgrid_poses()

    worst = calgrid_self_check(poses, intr)
    print("calgrid self-check: worst forward/inverse round-trip %.4f px" % worst)

    exif = cal_exif()
    textured = []
    for name, (cam_pos, R, t) in zip(CALGRID_IMAGE_NAMES, poses):
        grey = render_image_cal(cam_pos, R, t, intr)
        path = os.path.join(here, name)
        write_image(path, grey, exif)
        size = os.path.getsize(path)
        assert size < 60 * 1024, "%s is %d bytes (>= 60 KB)" % (name, size)
        frac = calgrid_textured_fraction(cam_pos, R, t, intr)
        textured.append(frac)
        print("wrote %s (%dx%d, textured %.1f%%, %d bytes)"
              % (name, WIDTH, HEIGHT, 100.0 * frac, size))
    print("calgrid textured-pixel fraction: min %.1f%% mean %.1f%%"
          % (100.0 * min(textured), 100.0 * sum(textured) / len(textured)))

    # Consecutive-pair overlap (the existing pinhole measure; enough to confirm
    # every neighbouring pair is matchable). The full loop is closed too.
    for i in range(len(poses) - 1):
        frac = pair_overlap(poses, i, i + 1)
        print("calgrid pair (%d, %d): %.1f%% shared visible surface" % (i, i + 1, 100.0 * frac))
        assert frac > CALGRID_MIN_PAIR_OVERLAP, (
            "calgrid pair (%d, %d) overlaps only %.1f%%" % (i, i + 1, 100.0 * frac))

    truth = {
        "width": WIDTH,
        "height": HEIGHT,
        "true_intrinsics": dict(intr),
        "exif_f": CAL_EXIF_FOCAL_PX,  # what AddPhotos recovers from the EXIF
        "images": [],
        "centres": [],
    }
    for name, (cam_pos, R, t) in zip(CALGRID_IMAGE_NAMES, poses):
        truth["images"].append({
            "file": name,
            "R": R.tolist(),          # X_cam = R X_world + t
            "t": t.tolist(),
            "f": FOCAL_PX,
            "W": WIDTH,
            "H": HEIGHT,
        })
        truth["centres"].append(cam_pos.tolist())

    with open(os.path.join(here, "calgrid_truth.json"), "w") as out:
        json.dump(truth, out, indent=2)
        out.write("\n")
    print("wrote calgrid_truth.json (%d images)" % len(truth["images"]))


# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))

    poses = []
    for cam_pos in CAMERA_POSITIONS:
        R, t = look_at(cam_pos, LOOK_AT, WORLD_UP)
        poses.append((cam_pos, R, t))

    # Self-check every view before writing anything.
    for cam_pos, R, t in poses:
        self_check(cam_pos, R, t)

    exif = scene_exif()
    for name, (cam_pos, R, t) in zip(IMAGE_NAMES, poses):
        grey = render_image(cam_pos, R, t)
        write_image(os.path.join(here, name), grey, exif)
        size = os.path.getsize(os.path.join(here, name))
        print("wrote %s (%dx%d, %.1f%% non-black, %d bytes)"
              % (name, WIDTH, HEIGHT, 100.0 * coverage(grey), size))

    truth = {"width": WIDTH, "height": HEIGHT, "f": FOCAL_PX, "images": [], "pairs": []}
    for name, (cam_pos, R, t) in zip(IMAGE_NAMES, poses):
        truth["images"].append({
            "file": name,
            "R": R.tolist(),          # X_cam = R * X_world + t
            "t": t.tolist(),
            "f": FOCAL_PX,
            "W": WIDTH,
            "H": HEIGHT,
        })

    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            _, Ri, ti = poses[i]
            _, Rj, tj = poses[j]
            # X_i = Ri X + ti  ->  X = Ri^T (X_i - ti)
            # X_j = Rj X + tj  =  Rj Ri^T X_i + (tj - Rj Ri^T ti)
            R_ij = Rj @ Ri.T
            t_ij = tj - R_ij @ ti
            t_ij = t_ij / np.linalg.norm(t_ij)  # unit-length direction
            truth["pairs"].append({
                "i": i,
                "j": j,
                "R": R_ij.tolist(),  # X_j = R_ij * X_i + t_ij
                "t": t_ij.tolist(),  # normalised to unit length
            })

    # Report the surface overlap of every pair, so the "> 60%" claim is measured,
    # not asserted. (A shared visible surface is what makes the pair matchable;
    # the numbers below are printed for the reviewer and checked by a test.)
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            frac = pair_overlap(poses, i, j)
            print("pair (%d, %d): %.1f%% shared visible surface" % (i, j, 100.0 * frac))

    with open(os.path.join(here, "scene_truth.json"), "w") as out:
        json.dump(truth, out, indent=2)
        out.write("\n")
    print("wrote scene_truth.json (%d images, %d pairs)"
          % (len(truth["images"]), len(truth["pairs"])))

    write_sequence(here, exif)
    write_geo_sequence(here)
    write_cal_sequence(here)
    write_calibration_grid(here)


def write_sequence(here, exif):
    """Render the eight-view SfM arc and its seq_truth.json.

    Reuses the same renderer, texture and EXIF as the three-view scene, so the
    only thing that changes is the camera path. The same pixel->hit->pixel
    self-check runs on every view before anything is written.
    """
    poses = sequence_poses()

    for cam_pos, R, t in poses:
        self_check(cam_pos, R, t)

    for name, (cam_pos, R, t) in zip(SEQ_IMAGE_NAMES, poses):
        grey = render_image(cam_pos, R, t)
        path = os.path.join(here, name)
        write_image(path, grey, exif)
        size = os.path.getsize(path)
        assert size < 60 * 1024, "%s is %d bytes (>= 60 KB)" % (name, size)
        print("wrote %s (%dx%d, %.1f%% non-black, %d bytes)"
              % (name, WIDTH, HEIGHT, 100.0 * coverage(grey), size))

    seq_truth = {
        "width": WIDTH,
        "height": HEIGHT,
        "f": FOCAL_PX,
        "images": [],
        "centres": [],
    }
    for name, (cam_pos, R, t) in zip(SEQ_IMAGE_NAMES, poses):
        seq_truth["images"].append({
            "file": name,
            "R": R.tolist(),          # X_cam = R * X_world + t
            "t": t.tolist(),
            "f": FOCAL_PX,
            "W": WIDTH,
            "H": HEIGHT,
        })
        seq_truth["centres"].append(cam_pos.tolist())

    for i in range(len(poses) - 1):
        frac = pair_overlap(poses, i, i + 1)
        print("seq pair (%d, %d): %.1f%% shared visible surface" % (i, i + 1, 100.0 * frac))
    endpoints = pair_overlap(poses, 0, len(poses) - 1)
    print("seq pair (0, %d): %.1f%% shared visible surface" % (len(poses) - 1, 100.0 * endpoints))

    with open(os.path.join(here, "seq_truth.json"), "w") as out:
        json.dump(seq_truth, out, indent=2)
        out.write("\n")
    print("wrote seq_truth.json (%d images)" % len(seq_truth["images"]))


if __name__ == "__main__":
    import sys

    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
    except ImportError as exc:
        sys.exit("make_scene_fixtures.py needs numpy and Pillow: %s" % exc)
    main()
