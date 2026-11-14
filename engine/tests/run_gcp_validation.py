#!/usr/bin/env python3
"""GCP / check-point validation scenarios (WP-13.20).

Two modes:

  * ``--synthetic`` aligns the 12-image aerial fixture (MatchPhotos +
    AlignCameras), imports a realistic consumer-GNSS reference for every camera
    (truth + seeded horizontal sigma 2 m / vertical sigma 4 m error and a
    common block bias), places 12 markers from the fixture's truth projections
    with seeded +-0.25 px noise, imports control references (truth + 1 cm
    noise) and holds 3 labels out as check points, triangulates the markers and
    optimises with OptimizeCameras, refits the datum with UpdateTransform, then
    reports the check-point RMS measured against the TRUE positions. The GCPs,
    not the camera references, must therefore carry the georeference.
  * ``--real <dir> --gcp <csv>`` runs match -> align -> detect markers ->
    import the CSV -> UpdateTransform -> OptimizeCameras and reports the
    check-point RMS. ``--crs`` must name a geographic (lon/lat) CRS; projected
    coordinates are refused because this script converts references as lon/lat.

``--ablations`` runs five synthetic scenarios in one process so a pipeline
defect can be told from noise-limited accuracy: (1) no projection noise, no
reference noise, with the fixture's exact camera position and attitude pinned
(the check-point 3D total must be <= 2 mm: the pipeline is otherwise exact),
(2) 1 cm reference noise only, (3) 0.25 px projection noise only, (4) noisy
consumer-GNSS camera references with every other input exact (a correctly
weighted adjustment is then governed by the control points, so its check-point
error must stay at millimetre level), and (5) both noises. Scenarios 2-5 keep
the realistic consumer-GNSS camera references and take their attitude from the
alignment, so their differences come from the two noise knobs.

Three gates are applied. Gate A is the milestone's own scenario: scenario 2
(noisy camera GNSS, 1 cm control references, exact projections) must stay at
or below 2 cm 3D total. Gate B is estimator efficiency: for scenarios 3 and 5
the script propagates the injected image noise through the TRUE geometry (a
first-order Gauss-Newton normal matrix over the camera poses, every marker and
a sampled tie-point cloud), takes the marginal covariance of each check point,
and requires the measured total to stay within 1.5x the predicted total. Gate
C is pipeline exactness: scenarios 1 and 4 must stay at or below 2 mm. Every
report carries the engine's solve diagnostics (per-category sigmas, iterations,
termination, cost) under ``diagnostics`` and, for Gate B, the prediction under
``predicted``.

The script prints one JSON object per scenario on stdout and exits non-zero
when a gate fails. The horizontal and vertical RMS are reported separately; the real
mode's target is horizontal. The total is the RMS of the per-check-point 3D
distances, i.e. the quadrature of the two axes. A check label with no marker or
no computed position is an error (exit 2), so a missing check point cannot
silently shrink the sample. No data is downloaded and nothing is written
outside a temporary directory.
"""

import argparse
import calendar
import datetime
import faulthandler
import json
import math
import os
import random
import re
import sys
import tempfile

import alphashape

# A native crash in the engine otherwise leaves only an exit code; an enabled
# faulthandler prints the Python call stack at the fault, which is enough to
# name the engine call that crashed.
faulthandler.enable()

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# WGS84 constants, identical to the engine's geodesy module, so the reference
# ENU coordinates this script reconstructs match the ones the fit uses.
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
ARCSEC_PER_RAD = 180.0 / math.pi * 3600.0


def geodetic_to_ecef(lon_deg, lat_deg, height_m):
    lon = lon_deg * DEG2RAD
    lat = lat_deg * DEG2RAD
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    p = (n + height_m) * cos_lat
    return (p * math.cos(lon), p * math.sin(lon),
            (n * (1.0 - WGS84_E2) + height_m) * sin_lat)


def ecef_to_geodetic(x, y, z):
    p = math.hypot(x, y)
    lon = math.atan2(y, x)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    n = WGS84_A
    for _ in range(100):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        lat_next = math.atan2(z + n * WGS84_E2 * sin_lat, p)
        if abs(lat_next - lat) < 1e-15:
            lat = lat_next
            break
        lat = lat_next
    height = p / math.cos(lat) - n
    return lat * RAD2DEG, lon * RAD2DEG, height


def enu_to_ecef_matrix(origin):
    """The 4x4 local-ENU -> ECEF transform, as the engine builds it."""
    lon = origin[0] * DEG2RAD
    lat = origin[1] * DEG2RAD
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    # Columns are the east, north and up unit vectors in ECEF (the engine's
    # enu_rotation), written here row by row.
    rotation = (
        (-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon),
        (cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon),
        (0.0, cos_lat, sin_lat),
    )
    ecef = geodetic_to_ecef(origin[0], origin[1], origin[2])
    return rotation, ecef


def enu_to_geodetic(enu, origin):
    rotation, ecef = enu_to_ecef_matrix(origin)
    x = (rotation[0][0] * enu[0] + rotation[0][1] * enu[1] +
         rotation[0][2] * enu[2] + ecef[0])
    y = (rotation[1][0] * enu[0] + rotation[1][1] * enu[1] +
         rotation[1][2] * enu[2] + ecef[1])
    z = (rotation[2][0] * enu[0] + rotation[2][1] * enu[1] +
         rotation[2][2] * enu[2] + ecef[2])
    lat, lon, height = ecef_to_geodetic(x, y, z)
    return (lon, lat, height)


def ecef_to_enu(ecef, origin):
    """Inverse of `enu_to_geodetic`'s rotation: ECEF -> local ENU."""
    rotation, origin_ecef = enu_to_ecef_matrix(origin)
    delta = [ecef[i] - origin_ecef[i] for i in range(3)]
    # `rotation` maps ENU -> ECEF (columns are east, north, up), so the inverse
    # is its transpose: enu[c] = sum_r rotation[r][c] * delta[r].
    return tuple(
        sum(rotation[r][c] * delta[r] for r in range(3)) for c in range(3))


def geodetic_to_enu(lon, lat, height, origin):
    return ecef_to_enu(geodetic_to_ecef(lon, lat, height), origin)


def ecef_to_local(ecef, transform):
    """Map an ECEF point into the chunk's local frame via chunk.transform.

    chunk.transform is a similarity local -> ECEF (E = scale * R * L + shift),
    so its inverse is L = R^T (E - shift) / scale. The rotation block is
    scaled in a similarity, hence the cube root of its determinant.
    """
    rotation = [[transform[r][c] for c in range(3)] for r in range(3)]
    shift = [transform[row][3] for row in range(3)]
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] -
                          rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] -
                            rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] -
                            rotation[1][1] * rotation[2][0]))
    scale = abs(determinant) ** (1.0 / 3.0)
    delta = [ecef[i] - shift[i] for i in range(3)]
    return tuple(
        sum(rotation[c][r] * delta[c] for c in range(3)) / scale
        for r in range(3))


# The synthetic block: the ground points sit at z = 0, the elevated ones on the
# building roof (truth height 9.52 m), all inside the 4x3 nadir footprint.
ORIGIN = (9.2, 48.7, 300.0)
GROUND = [(-45.0, -30.0), (-20.0, -35.0), (10.0, -30.0),
          (35.0, -20.0), (30.0, 25.0), (-5.0, 35.0)]
ELEVATED = [(-19.0, 6.0), (-14.0, 6.5), (-9.0, 7.0),
            (-18.0, 15.0), (-12.0, 14.0), (-6.0, 16.0)]
# Hold out the three best-observed roof targets. The ground targets at the
# fixture's edge are seen by only 1-3 cameras, so they cannot support a
# centimetre check; they still participate as control points.
CHECK_LABELS = {"E2", "E3", "E5"}
ROOF_HEIGHT = 9.52

# Consumer-GNSS error model for the camera references: per-camera seeded
# Gaussian horizontal sigma 2 m and vertical sigma 4 m on top of a common block
# bias. With these references the camera positions no longer carry the datum;
# the survey-grade GCPs must.
GNSS_HORIZONTAL_SIGMA_M = 2.0
GNSS_VERTICAL_SIGMA_M = 4.0
GNSS_BLOCK_BIAS_ENU_M = (1.5, -1.0, 3.0)
GNSS_SEED = 20261107
# The chunk-level fallback sigma for a camera reference (meters). The per-camera
# rows carry their own (2, 2, 4) accuracy, which the fit uses; this setting
# documents the order of magnitude and is what an unweighted reference gets.
CAMERA_LOCATION_ACCURACY_M = 3.0
# Scenarios that keep the fixture's exact EXIF GPS declare how exact it is. At
# the 10 m consumer default the near-planar nadir network is free to trade
# camera depth against scene depth (a centimetre-scale warp); the exact
# reference deserves the sub-millimetre accuracy the fixture actually writes.
EXACT_CAMERA_ACCURACY_M = 0.001
# The fixture's attitude is exact too; this is the 1-sigma it is declared at in
# the exact-camera scenarios (about 1 mm at the 60 m flying height).
EXACT_ROTATION_ACCURACY_DEG = 0.001

PROJECTION_SEED = 20261107
PIPELINE_TOLERANCE_M = 0.002
# Uniform noise of +-a has sigma a/sqrt(3). The marker-projection accuracy is
# declared from the noise this script actually injects (the fit's default
# 0.5 px would under-weight exact or 0.25 px-peak projections against the 1 px
# tie points). The floor keeps the exact scenarios positive.
PROJECTION_ACCURACY_FLOOR_PX = 0.001

# Gate B (estimator efficiency): the measured check-point total must stay within
# this factor of the first-order prediction through the true geometry. The
# factor is not tuned per run.
GATE_B_FACTOR = 1.5
# A scenario with no reference noise keeps its control points essentially
# exact. A zero sigma would make the normal matrix singular, so the prediction
# uses this floor; it is far below every other sigma in the system.
CONTROL_REFERENCE_FLOOR_M = 1e-4
# The tie-point cloud the prediction models: sampled on the truth surface, no
# 3-D prior, each observation weighted at the chunk's tie-point accuracy.
TIEPOINT_COUNT = 300
TIEPOINT_MIN_VIEWS = 3
TIEPOINT_SEED = 20261107
# Central-difference steps for the numeric Jacobians of the pinhole projection.
POSE_STEP_M = 1e-6
POSE_STEP_RAD = 1e-6
POINT_STEP_M = 1e-6

# Solve diagnostics the engine records on the chunk (chunk.meta), surfaced in
# every synthetic report so a mis-weighted adjustment can be told from a
# noise-limited one. The sigmas are metres / pixels as named.
DIAGNOSTIC_KEYS = (
    "optimize_cameras.sigma.camera_centre_m",
    "optimize_cameras.sigma.marker_control_m",
    "optimize_cameras.sigma.marker_projection_px",
    "optimize_cameras.sigma.tiepoint_px",
    "optimize_cameras.control_axes",
    "optimize_cameras.control_points_fully_constrained",
    "optimize_cameras.rotation_priors",
    "optimize_cameras.iterations",
    "optimize_cameras.termination",
    "optimize_cameras.initial_cost",
    "optimize_cameras.final_cost",
    "optimize_cameras.final_relative_cost_change",
    "optimize_cameras.prior_rms_sigma",
    "optimize_cameras.prior_axes",
    "optimize_cameras.free_parameters",
    "optimize_cameras.residuals",
    "optimize_cameras.gauge_fixed_poses",
    "optimize_cameras.gauge_scale_anchor_camera",
    "optimize_cameras.gauge_scale_anchor_axis",
    "optimize_cameras.control_rms_m",
    "optimize_cameras.control_rms_xyz_m",
    "optimize_cameras.checkpoint_rms_m",
    "optimize_cameras.checkpoint_rms_xyz_m",
)



def reference_noise(index, axis, amplitude_m):
    """Deterministic per-axis reference noise in [-amplitude, +amplitude]."""
    return (((index * 37 + axis * 11 + 5) % 101) / 100.0 - 0.5) * 2.0 * amplitude_m


def truth_local(enu, transform):
    """The true ENU point in the chunk-local frame after georeferencing."""
    lon, lat, height = enu_to_geodetic(enu, ORIGIN)
    return ecef_to_local(geodetic_to_ecef(lon, lat, height), transform)


def truth_points():
    points = []
    for index, (x, y) in enumerate(GROUND):
        points.append(("G{}".format(index + 1), (x, y, 0.0)))
    for index, (x, y) in enumerate(ELEVATED):
        points.append(("E{}".format(index + 1), (x, y, ROOF_HEIGHT)))
    return points


def project_truth(image, point):
    """Project a truth ENU point through the fixture's truth pose."""
    rotation = image["R"]
    translation = image["t"]
    camera = [sum(rotation[r][c] * point[c] for c in range(3)) + translation[r]
              for r in range(3)]
    if camera[2] <= 1e-6:
        return None
    u = image["width"] / 2.0 + image["f"] * camera[0] / camera[2]
    v = image["height"] / 2.0 + image["f"] * camera[1] / camera[2]
    if not (0.0 <= u < image["width"] and 0.0 <= v < image["height"]):
        return None
    return (u, v)


def write_reference_csv(path, rows):
    """Write label,x,y,z rows. In a geographic chunk x/y/z are lon/lat/alt."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("label,x,y,z\n")
        for label, x, y, z in rows:
            handle.write("{},{!r},{!r},{!r}\n".format(label, x, y, z))


def write_camera_reference_csv(path, rows):
    """label,lon,lat,alt,acc_x,acc_y,acc_z rows for ImportReference.

    No `crs` is passed on import: the rows are already lon/lat/ellipsoidal
    height, the chunk's camera-reference CRS (WGS 84 geographic).
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("label,lon,lat,alt,acc_x,acc_y,acc_z\n")
        for label, lon, lat, height, accuracy in rows:
            handle.write("{},{!r},{!r},{!r},{!r},{!r},{!r}\n".format(
                label, lon, lat, height,
                accuracy[0], accuracy[1], accuracy[2]))


def apply_exact_camera_accuracy(chunk, by_stem, directory):
    """Re-import the EXIF camera references with their true 1-sigma accuracy.

    AddPhotos stores the position but no accuracy, so the fit would fall back
    to the 10 m consumer default and leave the near-planar nadir network free
    to trade camera depth against scene depth. In the exact-camera scenarios
    the fixture GPS is genuinely sub-millimetre; declare that.

    The same rows carry the fixture's exact attitude. The ablation's purpose is
    to separate a defect in the GCP/bundle machinery from the SfM network's own
    tie-point noise, so these scenarios pin both the centre and the rotation;
    the realistic scenario leaves the rotation to the alignment.
    """
    path = os.path.join(directory, "camera_exact.csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("label,lon,lat,alt,acc_x,acc_y,acc_z,yaw,pitch,roll\n")
        for camera in chunk.cameras:
            reference = camera.reference
            image = by_stem.get(camera.label)
            if reference is None or image is None:
                continue
            yaw, pitch, roll = body_attitude_deg(image["R"])
            handle.write("{},{!r},{!r},{!r},{!r},{!r},{!r},{!r},{!r},{!r}\n".format(
                camera.label, reference[0], reference[1], reference[2],
                EXACT_CAMERA_ACCURACY_M, EXACT_CAMERA_ACCURACY_M,
                EXACT_CAMERA_ACCURACY_M, yaw, pitch, roll))
    chunk.importReference(path=path, create_markers=False)


# Fixed body-to-camera rotation for a nadir mount, as the engine's
# reference::body_to_camera_nadir() returns it: camera +X is body -y, +Y is
# body -x, +Z (boresight) is body -z.
BODY_TO_CAMERA_NADIR = ((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0))


def transpose(matrix):
    return [[matrix[r][c] for r in range(3)] for c in range(3)]


def multiply(left, right):
    return [[sum(left[r][k] * right[k][c] for k in range(3))
             for c in range(3)] for r in range(3)]


def axis_rotation(axis, degrees):
    angle = degrees * DEG2RAD
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rows = [[0.0] * 3 for _ in range(3)]
    rows[axis][axis] = 1.0
    first, second = (axis + 1) % 3, (axis + 2) % 3
    rows[first][first] = cos_a
    rows[first][second] = -sin_a
    rows[second][first] = sin_a
    rows[second][second] = cos_a
    return rows


def to_matrix_zyx(yaw, pitch, roll):
    """The engine's crs::to_matrix for the ZYX order (degrees)."""
    return multiply(multiply(axis_rotation(2, yaw), axis_rotation(1, pitch)),
                    axis_rotation(0, roll))


def from_matrix_zyx(matrix):
    """The engine's crs::from_matrix for the ZYX order (degrees)."""
    permutation = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    determinant = -1.0
    r = multiply(multiply(permutation, matrix),
                 transpose(permutation))
    cos_middle = math.hypot(r[0][0], r[0][1])
    if cos_middle > 1e-12:
        a = math.atan2(-r[1][2], r[2][2])
        b = math.atan2(r[0][2], cos_middle)
        c = math.atan2(-r[0][1], r[0][0])
    else:
        a = math.atan2(r[2][1], r[1][1])
        b = math.pi / 2.0 if r[0][2] >= 0.0 else -math.pi / 2.0
        c = 0.0
    return (determinant * a * RAD2DEG, determinant * b * RAD2DEG,
            determinant * c * RAD2DEG)


def body_attitude_deg(rotation):
    """ENU body attitude (yaw, pitch, roll) for a world->camera rotation.

    Mirrors reference::measured_rotation(): R_wc = body_to_camera * R_wb^T,
    and R_wb = Rz(90 deg) * to_matrix({x, -y, z}). The fixture's R is a
    world->camera rotation in the local ENU frame, so R_wb = R^T * body_to_camera
    and the stored triple follows from inverting that composition.
    """
    body_to_world = multiply(transpose(rotation), BODY_TO_CAMERA_NADIR)
    internal = multiply(axis_rotation(2, -90.0), body_to_world)
    yaw, pitch, roll = from_matrix_zyx(internal)
    return (yaw, -pitch, roll)


def apply_camera_gnss(chunk, by_stem, directory):
    """Overwrite every camera reference with seeded consumer-GNSS positions.

    The fixture EXIF GPS is exact; leaving it in place would let the camera
    references carry the datum and the GCPs would barely matter. Each camera
    gets a per-camera seeded Gauss offset (horizontal sigma 2 m, vertical sigma
    4 m) plus a common block bias, with the matching per-axis accuracy so the
    adjustment weights them as consumer GNSS.
    """
    rng = random.Random(GNSS_SEED)
    rows = []
    for camera in chunk.cameras:
        image = by_stem.get(camera.label)
        if image is None:
            continue
        gps = image["gps"]
        enu = geodetic_to_enu(gps["lon"], gps["lat"], gps["height"], ORIGIN)
        noisy = (
            enu[0] + GNSS_BLOCK_BIAS_ENU_M[0] + rng.gauss(0.0, GNSS_HORIZONTAL_SIGMA_M),
            enu[1] + GNSS_BLOCK_BIAS_ENU_M[1] + rng.gauss(0.0, GNSS_HORIZONTAL_SIGMA_M),
            enu[2] + GNSS_BLOCK_BIAS_ENU_M[2] + rng.gauss(0.0, GNSS_VERTICAL_SIGMA_M),
        )
        lon, lat, height = enu_to_geodetic(noisy, ORIGIN)
        rows.append((camera.label, lon, lat, height,
                     (GNSS_HORIZONTAL_SIGMA_M, GNSS_HORIZONTAL_SIGMA_M,
                      GNSS_VERTICAL_SIGMA_M)))
    path = os.path.join(directory, "camera_references.csv")
    write_camera_reference_csv(path, rows)
    chunk.importReference(path=path)


def check_point_errors(chunk, labels, targets):
    """Per-axis error of each held-out marker against `targets`.

    `targets` maps a label to its true chunk-local position. A label with no
    marker or no computed position raises ValueError: a missing check point has
    to fail the scenario instead of silently shrinking the sample.
    """
    markers = {marker.label: marker for marker in chunk.markers}
    missing = [label for label in labels
               if label not in markers or markers[label].position is None]
    if missing:
        raise ValueError(
            "missing or unpositioned check point(s): " + ", ".join(missing))
    errors = []
    for label in labels:
        position = markers[label].position
        target = targets[label]
        delta = [position[i] - target[i] for i in range(3)]
        errors.append({"label": label,
                       "horizontal": math.hypot(delta[0], delta[1]),
                       "vertical": delta[2],
                       "total": math.sqrt(sum(value * value
                                              for value in delta))})
    return errors


def reference_local_targets(chunk, labels):
    """Chunk-local positions of the labels' stored (lon/lat) references."""
    markers = {marker.label: marker for marker in chunk.markers}
    transform = chunk.transform
    targets = {}
    for label in labels:
        marker = markers.get(label)
        if marker is None or marker.reference is None:
            raise ValueError("missing check point: " + label)
        targets[label] = ecef_to_local(
            geodetic_to_ecef(*marker.reference), transform)
    return targets


def root_mean_square(values):
    if not values:
        raise ValueError("root mean square of an empty sample")
    return math.sqrt(sum(value * value for value in values) / len(values))


def rodrigues(omega):
    """Rotation matrix of a rotation vector (exact for any angle)."""
    theta = math.sqrt(omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2)
    if theta < 1e-12:
        return ((1.0, -omega[2], omega[1]),
                (omega[2], 1.0, -omega[0]),
                (-omega[1], omega[0], 1.0))
    kx, ky, kz = (value / theta for value in omega)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    one = 1.0 - cos_t
    return (
        (cos_t + kx * kx * one, kx * ky * one - kz * sin_t,
         kx * kz * one + ky * sin_t),
        (ky * kx * one + kz * sin_t, cos_t + ky * ky * one,
         ky * kz * one - kx * sin_t),
        (kz * kx * one - ky * sin_t, kz * ky * one + kx * sin_t,
         cos_t + kz * kz * one))


def project_world(rotation, centre, focal, width, height, point):
    """Project a world point through an (R, centre) pinhole camera."""
    delta = [point[i] - centre[i] for i in range(3)]
    camera = [sum(rotation[r][c] * delta[c] for c in range(3))
              for r in range(3)]
    if camera[2] <= 1e-6:
        return None
    return (width / 2.0 + focal * camera[0] / camera[2],
            height / 2.0 + focal * camera[1] / camera[2])


def load_truth():
    with open(os.path.join(DATA_DIR, "aerial_truth.json"),
              encoding="utf-8") as handle:
        return json.load(handle)


def surface_height(x, y, building):
    """The truth surface height: the roof inside its footprint, else ground."""
    if (building["x0"] <= x <= building["x1"] and
            building["y0"] <= y <= building["y1"]):
        return building["top"]
    return 0.0


def sample_tie_points(truth, count=TIEPOINT_COUNT):
    """Deterministic tie-point cloud on the truth surface, >= 3 views each."""
    images = truth["images"]
    building = truth["building"]
    rng = random.Random(TIEPOINT_SEED)
    points = []
    for _ in range(200000):
        if len(points) >= count:
            break
        x = rng.uniform(-58.0, 58.0)
        y = rng.uniform(-43.0, 43.0)
        point = (x, y, surface_height(x, y, building))
        views = sum(1 for image in images
                    if project_truth(image, point) is not None)
        if views >= TIEPOINT_MIN_VIEWS:
            points.append(point)
    if len(points) < count:
        raise ValueError("could not sample {} tie points".format(count))
    return points


def invert3(matrix):
    """Inverse of a 3x3 matrix by the adjugate; raises when singular."""
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    det = (a * (e * i - f * h) - b * (d * i - f * g) +
           c * (d * h - e * g))
    if abs(det) < 1e-18:
        raise ValueError("singular 3x3 block")
    return (
        ((e * i - f * h) / det, (c * h - b * i) / det,
         (b * f - c * e) / det),
        ((f * g - d * i) / det, (a * i - c * g) / det,
         (c * d - a * f) / det),
        ((d * h - e * g) / det, (b * g - a * h) / det,
         (a * e - b * d) / det))


def invert_matrix(matrix):
    """Gauss-Jordan inverse of a square matrix with partial pivoting."""
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("matrix is not square")
    work = [list(row) + [1.0 if i == j else 0.0 for j in range(size)]
            for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size),
                    key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) < 1e-15:
            raise ValueError("singular normal matrix")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        pivot_row = work[column]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            if factor == 0.0:
                continue
            current = work[row]
            for index in range(len(current)):
                current[index] -= factor * pivot_row[index]
    return [row[size:] for row in work]


def camera_projection_jacobian(image, point):
    """d(u, v)/d(rotation vector, centre), central differences at truth."""
    rotation, centre = image["R"], image["centre"]
    focal, width, height = image["f"], image["width"], image["height"]
    rows = []
    for axis in range(6):
        step = POSE_STEP_RAD if axis < 3 else POSE_STEP_M
        pixels = []
        for sign in (1.0, -1.0):
            omega = [0.0, 0.0, 0.0]
            shift = [0.0, 0.0, 0.0]
            if axis < 3:
                omega[axis] = sign * step
            else:
                shift[axis - 3] = sign * step
            perturbed = multiply(rodrigues(omega), rotation)
            moved = [centre[i] + shift[i] for i in range(3)]
            pixel = project_world(perturbed, moved, focal, width, height,
                                  point)
            if pixel is None:
                raise ValueError("projection left the image while differencing")
            pixels.append(pixel)
        rows.append(((pixels[0][0] - pixels[1][0]) / (2.0 * step),
                     (pixels[0][1] - pixels[1][1]) / (2.0 * step)))
    return rows


def point_projection_jacobian(image, point):
    """d(u, v)/d(point x, y, z), central differences at truth."""
    rotation, centre = image["R"], image["centre"]
    focal, width, height = image["f"], image["width"], image["height"]
    rows = []
    for axis in range(3):
        pixels = []
        for sign in (1.0, -1.0):
            moved = list(point)
            moved[axis] += sign * POINT_STEP_M
            pixel = project_world(rotation, centre, focal, width, height,
                                  moved)
            if pixel is None:
                raise ValueError("projection left the image while differencing")
            pixels.append(pixel)
        rows.append(((pixels[0][0] - pixels[1][0]) / (2.0 * POINT_STEP_M),
                     (pixels[0][1] - pixels[1][1]) / (2.0 * POINT_STEP_M)))
    return rows


def add_outer(normal, indices, row):
    """Add the outer product of ``row`` at ``indices`` into ``normal``."""
    for a, i in enumerate(indices):
        value = row[a]
        if value == 0.0:
            continue
        columns = normal[i]
        for b, j in enumerate(indices):
            columns[j] += value * row[b]


def add_projection(normal, indices, point_rows, camera_rows, sigma):
    """Add one weighted 2-D projection observation."""
    for component in (0, 1):
        row = [entry[component] / sigma for entry in point_rows]
        row += [entry[component] / sigma for entry in camera_rows]
        add_outer(normal, indices, row)


def predict_check_point_rms(truth, marker_projection_sigma_px,
                            control_sigma_m, camera_gnss,
                            tiepoint_sigma_px):
    """First-order check-point covariance through the true geometry (Gate B).

    This is the Gauss-Newton normal matrix J^T W J of a bundle adjustment at
    the fixture's truth parameters: six degrees of freedom per camera
    (rotation vector and centre; intrinsics fixed), three per marker and three
    per sampled tie point. Its observations are the truth projections weighted
    by the image sigma, the control markers' 3-D references weighted by the
    survey sigma, and the camera centres weighted by the GNSS sigma; tie points
    carry no prior. The block-diagonal tie-point blocks are eliminated by their
    Schur complement, so the reduced system is the camera/marker matrix the
    real adjustment effectively solves. Its inverse gives each check point's
    marginal 3x3 covariance in ENU; the predicted RMS is the root of the mean
    per-axis variance over the check points.
    """
    images = truth["images"]
    markers = truth_points()
    ties = sample_tie_points(truth)
    camera_parameters = 6 * len(images)
    marker_base = camera_parameters
    tie_base = marker_base + 3 * len(markers)
    size = tie_base + 3 * len(ties)
    normal = [[0.0] * size for _ in range(size)]
    marker_views = [
        [index for index, image in enumerate(images)
         if project_truth(image, point) is not None]
        for _, point in markers]
    tie_views = [
        [index for index, image in enumerate(images)
         if project_truth(image, point) is not None]
        for point in ties]
    tie_links = [[] for _ in ties]
    for j, (_, point) in enumerate(markers):
        base = marker_base + 3 * j
        for camera_index in marker_views[j]:
            image = images[camera_index]
            camera_base = 6 * camera_index
            indices = ([base, base + 1, base + 2] +
                       [camera_base + b for b in range(6)])
            add_projection(normal, indices,
                           point_projection_jacobian(image, point),
                           camera_projection_jacobian(image, point),
                           marker_projection_sigma_px)
    for k, point in enumerate(ties):
        base = tie_base + 3 * k
        for camera_index in tie_views[k]:
            image = images[camera_index]
            camera_base = 6 * camera_index
            indices = ([base, base + 1, base + 2] +
                       [camera_base + b for b in range(6)])
            add_projection(normal, indices,
                           point_projection_jacobian(image, point),
                           camera_projection_jacobian(image, point),
                           tiepoint_sigma_px)
            tie_links[k].extend(range(camera_base, camera_base + 6))
    for j, (label, _) in enumerate(markers):
        if label in CHECK_LABELS:
            continue
        base = marker_base + 3 * j
        for axis in range(3):
            normal[base + axis][base + axis] += 1.0 / (control_sigma_m ** 2)
    camera_sigma = ((GNSS_HORIZONTAL_SIGMA_M, GNSS_HORIZONTAL_SIGMA_M,
                     GNSS_VERTICAL_SIGMA_M) if camera_gnss
                    else (EXACT_CAMERA_ACCURACY_M,) * 3)
    for camera_index in range(len(images)):
        base = 6 * camera_index + 3
        for axis in range(3):
            normal[base + axis][base + axis] += 1.0 / (camera_sigma[axis] ** 2)
    reduced = [row[:tie_base] for row in normal[:tie_base]]
    for k in range(len(ties)):
        block = tie_base + 3 * k
        inverse = invert3([[normal[block + a][block + b] for b in range(3)]
                           for a in range(3)])
        links = sorted(set(tie_links[k]))
        transformed = []
        for index in links:
            row = [normal[index][block + b] for b in range(3)]
            transformed.append(
                [sum(inverse[a][b] * row[b] for b in range(3))
                 for a in range(3)])
        for a, row_index in enumerate(links):
            for b, column_index in enumerate(links):
                correction = sum(
                    transformed[a][c] * normal[column_index][block + c]
                    for c in range(3))
                reduced[row_index][column_index] -= correction
    covariance = invert_matrix(reduced)
    check_indices = [j for j, (label, _) in enumerate(markers)
                     if label in CHECK_LABELS]
    if not check_indices:
        raise ValueError("no check points to predict")
    horizontal, vertical, total = [], [], []
    per_point = {}
    for j in check_indices:
        base = marker_base + 3 * j
        trace = sum(covariance[base + axis][base + axis] for axis in range(3))
        h_sq = covariance[base][base] + covariance[base + 1][base + 1]
        v_sq = covariance[base + 2][base + 2]
        if min(h_sq, v_sq) < 0.0:
            raise ValueError("negative variance in the check-point covariance")
        horizontal.append(h_sq)
        vertical.append(v_sq)
        total.append(trace)
        per_point[markers[j][0]] = {
            "horizontal": math.sqrt(h_sq),
            "vertical": math.sqrt(v_sq),
            "total": math.sqrt(trace)}
    return {
        "tie_points": len(ties),
        "tie_point_sigma_px": tiepoint_sigma_px,
        "marker_projection_sigma_px": marker_projection_sigma_px,
        "control_reference_sigma_m": control_sigma_m,
        "camera_gnss": camera_gnss,
        "parameters": size,
        "reduced_parameters": tie_base,
        "horizontal": math.sqrt(sum(horizontal) / len(horizontal)),
        "vertical": math.sqrt(sum(vertical) / len(vertical)),
        "total": math.sqrt(sum(total) / len(total)),
        "check_points": per_point,
    }


def run_synthetic(args, projection_noise_px, reference_noise_m, camera_gnss):
    """Align the fixture, triangulate markers, georeference and optimise."""
    with open(os.path.join(DATA_DIR, "aerial_truth.json"),
              encoding="utf-8") as handle:
        truth = json.load(handle)
    images = [item["file"] for item in truth["images"]]
    by_stem = {os.path.splitext(item["file"])[0]: item
               for item in truth["images"]}
    doc = alphashape.Document()
    chunk = doc.add_chunk("Synthetic GCP block")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos",
                 filenames=[os.path.join(DATA_DIR, name) for name in images])
    # Declare WGS 84 geographic before any reference import, so both camera and
    # marker references are interpreted as (longitude, latitude, altitude).
    chunk.crs_epsg = 4326
    # Replace the fixture's exact EXIF GPS with realistic consumer-GNSS
    # references. The datum must then come from the GCPs; a block georeferenced
    # by exact camera centres would not test GCP georeferencing at all.
    if camera_gnss:
        chunk.camera_location_accuracy = CAMERA_LOCATION_ACCURACY_M
        with tempfile.TemporaryDirectory() as directory:
            apply_camera_gnss(chunk, by_stem, directory)
    else:
        # The fixture EXIF GPS is exact (sub-millimetre); say so, so the bundle
        # adjustment cannot trade camera depth against scene depth.
        chunk.camera_location_accuracy = EXACT_CAMERA_ACCURACY_M
        chunk.camera_rotation_accuracy = EXACT_ROTATION_ACCURACY_DEG
        with tempfile.TemporaryDirectory() as directory:
            apply_exact_camera_accuracy(chunk, by_stem, directory)
    # A real alignment gives the cameras poses from the (possibly noisy)
    # references above. With the exact references this is the fixture's truth
    # GPS, which is the zero-noise pipeline-exactness ablation.
    doc.run_task("MatchPhotos", keypoint_limit_per_mpx=10000)
    doc.run_task("AlignCameras")
    # The survey references carry the configured per-axis noise; say so, so the
    # control points are not treated as exact. Zero is not a valid accuracy
    # (the setter refuses it): the exact-reference scenarios keep the default,
    # which does not affect their datum because every reference is exact.
    if reference_noise_m > 0.0:
        chunk.marker_location_accuracy = reference_noise_m

    points = truth_points()
    for label, _ in points:
        chunk.addMarker(label=label)

    # Record the image-space accuracy of the marker projections this script
    # injects (uniform noise of +-a has sigma a/sqrt(3)). The bundle adjustment
    # whitens every observation by its own sigma, so marker projections get
    # this weight and tie points keep the (looser) tie-point accuracy.
    chunk.marker_projection_accuracy = max(
        PROJECTION_ACCURACY_FLOOR_PX, projection_noise_px / math.sqrt(3.0))

    # Marker positions are NOT seeded: they come from image measurements only.
    # Every projection is the fixture's truth projection plus seeded noise.
    rng = random.Random(PROJECTION_SEED)
    for camera in chunk.cameras:
        image = by_stem.get(camera.label)
        if image is None:
            continue
        for label, enu in points:
            pixel = project_truth(image, enu)
            if pixel is None:
                continue
            u = pixel[0] + rng.uniform(-projection_noise_px, projection_noise_px)
            v = pixel[1] + rng.uniform(-projection_noise_px, projection_noise_px)
            chunk.setMarkerProjection(label, camera.label, u, v)

    control_rows = []
    check_rows = []
    for index, (label, enu) in enumerate(points):
        if label in CHECK_LABELS:
            # Held out of the fit, so its (unused) reference is the truth.
            check_rows.append((label,) + enu_to_geodetic(enu, ORIGIN))
        else:
            noisy = tuple(
                enu[axis] + reference_noise(index, axis, reference_noise_m)
                for axis in range(3))
            control_rows.append((label,) + enu_to_geodetic(noisy, ORIGIN))

    with tempfile.TemporaryDirectory() as directory:
        control_path = os.path.join(directory, "control.csv")
        check_path = os.path.join(directory, "check.csv")
        write_reference_csv(control_path, control_rows)
        write_reference_csv(check_path, check_rows)
        chunk.importReference(path=control_path, marker_type="gcp")
        chunk.importReference(path=check_path, marker_type="check_point")

    # AlignCameras georeferenced the block from the camera references (exact
    # EXIF GPS or the seeded consumer GNSS). The bundle adjustment then
    # triangulates every marker from its projections (their only source of a
    # position) with the control references constraining the solve, and writes
    # the solved positions back. UpdateTransform then refits the datum, so the
    # survey GCPs, not the camera references, carry the georeference.
    transform = chunk.transform
    applied = any(abs(transform[row][col] -
                      (1.0 if row == col else 0.0)) > 1e-9
                  for row in range(4) for col in range(4))
    if not applied:
        raise ValueError(
            "AlignCameras did not georeference the block (no GPS references)")
    # The aerial fixture is a distortion-free pinhole render, and AlignCameras
    # already holds its calibration fixed. Letting the bundle adjustment
    # self-calibrate distortion on this near-planar scene would fit noise into a
    # radial term (a depth/distortion degeneracy) and bow the model, so this
    # validation holds every intrinsic at the alignment's value.
    chunk.optimizeCameras(use_gcps=True,
                          fit_f=False, fit_cx=False, fit_cy=False,
                          fit_k1=False, fit_k2=False, fit_k3=False, fit_k4=False,
                          fit_p1=False, fit_p2=False,
                          fit_p3=False, fit_p4=False)
    # The check-point error right after the bundle adjustment, in the datum the
    # adjustment itself worked in. Comparing it with the post-datum error tells
    # a bundle-adjustment defect from a datum-fit one.
    labels = [label for label, _ in points if label in CHECK_LABELS]
    pre_errors = check_point_errors(
        chunk, labels,
        {label: truth_local(enu, chunk.transform) for label, enu in points})
    pre_datum_total = root_mean_square([e["total"] for e in pre_errors])
    # Refit the datum now that the marker positions and control references
    # agree. This is the weighted similarity step: it moves the block without
    # changing its internal shape.
    chunk.updateTransform()
    if chunk.meta.get("update_transform.applied") != "1":
        raise ValueError("georeferencing failed: " +
                         str(chunk.meta.get("update_transform.reason",
                                            "unknown")))

    targets = {label: truth_local(enu, chunk.transform)
               for label, enu in points}
    errors = check_point_errors(chunk, labels, targets)
    if len(errors) != len(labels):
        raise ValueError("evaluated {} check points, expected {}".format(
            len(errors), len(labels)))
    horizontal = root_mean_square([e["horizontal"] for e in errors])
    vertical = root_mean_square([e["vertical"] for e in errors])
    total = root_mean_square([e["total"] for e in errors])
    altitudes = [item["gps"]["height"] - ORIGIN[2]
                 for item in truth["images"]]
    focal = [item["f"] for item in truth["images"]]
    gsd = (sum(altitudes) / len(altitudes)) / (sum(focal) / len(focal))
    return {
        "scenario": "synthetic-gcp-block",
        "images": len(images),
        "markers": len(chunk.markers),
        "ground_points": len(GROUND),
        "elevated_points": len(ELEVATED),
        "control_points": len(control_rows),
        "check_points": len(check_rows),
        "evaluated_check_points": len(errors),
        "projection_noise_px": projection_noise_px,
        "marker_projection_accuracy_px": chunk.marker_projection_accuracy,
        "marker_location_accuracy_m": chunk.marker_location_accuracy,
        "tiepoint_accuracy_px": chunk.tiepoint_accuracy,
        "reference_noise_m": reference_noise_m,
        "camera_gnss": camera_gnss,
        "camera_location_accuracy_m": (
            CAMERA_LOCATION_ACCURACY_M if camera_gnss
            else EXACT_CAMERA_ACCURACY_M),
        "camera_rotation_accuracy_deg": chunk.camera_rotation_accuracy,
        "camera_attitude_priors": not camera_gnss,
        "camera_gnss_model": None if not camera_gnss else {
            "horizontal_sigma_m": GNSS_HORIZONTAL_SIGMA_M,
            "vertical_sigma_m": GNSS_VERTICAL_SIGMA_M,
            "block_bias_enu_m": list(GNSS_BLOCK_BIAS_ENU_M),
        },
        "gsd_m_per_px": gsd,
        "camera_f": chunk.sensors[0].calibration.f,
        "diagnostic_marker_projections": {
            marker.label: len(marker.projections)
            for marker in chunk.markers},
        "georeferenced": applied,
        "engine_check_rms_m": float(
            chunk.meta.get("update_transform.rms_check_m", "nan")),
        "check_point_rms_before_datum_m": pre_datum_total,
        "diagnostics": {key: chunk.meta.get(key) for key in DIAGNOSTIC_KEYS},
        "update_transform": {
            "cameras_used": chunk.meta.get("update_transform.cameras"),
            "control_points_used": chunk.meta.get(
                "update_transform.control_points"),
            "check_points_used": chunk.meta.get("update_transform.check_points"),
            "rms_control_m": chunk.meta.get("update_transform.rms_control_m"),
            "rms_check_m": chunk.meta.get("update_transform.rms_check_m"),
            "scale": chunk.meta.get("update_transform.scale"),
            "camera_weight": chunk.meta.get("update_transform.camera_weight"),
            "control_weight": chunk.meta.get("update_transform.control_weight"),
        },
        "check_point_rms_m": {
            "horizontal": horizontal,
            "vertical": vertical,
            "total": total,
        },
        "check_point_errors_m": errors,
    }


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def collect_images(directory):
    return [os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if name.lower().endswith(IMAGE_SUFFIXES)]


# A drone RTK module writes two integer counters into the JPEG comment of each
# exposure: `timestamp` in milliseconds and `exppts` in microseconds, both on
# the module's own clock. They are the only sub-second capture times in the
# imagery (EXIF stores whole seconds on the camera's unsynchronised clock).
CAPTURE_TIMING_PATTERN = re.compile(rb"timestamp=(\d+)\s+exppts=(\d+)")


def parse_mrk(path):
    """Parse a per-exposure event-mark (``.mrk``) file.

    The documented layout is whitespace separated::

        index  gps_time_of_week  [gps_week]  <solution tokens>
        lat,Lat  lon,Lon  height,Ellh
        sigma_lat, sigma_lon, sigma_height  flag,Quality

    The first token numbers the exposure, the second is the GPS time of week in
    seconds and the bracketed third is the GPS week. The three ``,unit`` tokens
    carry the antenna position (decimal degrees and ellipsoidal metres) and the
    three following comma-separated numbers are its per-axis 1-sigma standard
    deviations in metres, ordered latitude, longitude, height. The last two
    tokens are the integer solution status (50 fixed, 34 float, 0/16
    single-point) and a quality letter. The tokens between the week and the
    latitude are solution details, not a position, so they are skipped: a
    firmware that adds or removes one does not shift the columns this parser
    reads. A malformed line is skipped, never fatal.
    """
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.split()
            if len(tokens) < 10 or not tokens[0].isdigit():
                continue
            try:
                index = int(tokens[0])
                time_s = float(tokens[1])
            except ValueError:
                continue
            week = None
            for token in tokens[:5]:
                match = re.fullmatch(r"\[(\d+)\]", token)
                if match:
                    week = int(match.group(1))
                    break
            latitude = longitude = height = None
            sigma_start = None
            for position, token in enumerate(tokens):
                try:
                    if token.endswith(",Lat") and latitude is None:
                        latitude = float(token[:-4])
                    elif token.endswith(",Lon") and longitude is None:
                        longitude = float(token[:-4])
                    elif token.endswith(",Ellh"):
                        height = float(token[:-5])
                        sigma_start = position + 1
                except ValueError:
                    latitude = longitude = height = None
                    break
            if (latitude is None or longitude is None or height is None or
                    sigma_start is None or sigma_start + 3 >= len(tokens)):
                continue
            try:
                sigma = [float(value.strip(","))
                         for value in tokens[sigma_start:sigma_start + 3]]
                # The status and quality letter share the final token ("50,Q").
                status = tokens[sigma_start + 3].split(",")
                flag = int(float(status[0]))
            except ValueError:
                continue
            quality = status[1].strip(",") if len(status) > 1 else ""
            if not all(math.isfinite(value) and value > 0.0 for value in sigma):
                continue
            records.append({
                "index": index,
                "time_s": time_s,
                "week": week,
                "lat": latitude,
                "lon": longitude,
                "height_m": height,
                "sigma_lat_m": sigma[0],
                "sigma_lon_m": sigma[1],
                "sigma_hgt_m": sigma[2],
                "fix_flag": flag,
                "quality": quality,
            })
    return records


def read_capture_timing(image_path, limit=262144):
    """The module's two per-exposure counters, or None when absent.

    Only the JPEG header area is scanned (the comment sits in the first few
    kilobytes); a full-size frame is never read. ``trigger_us`` is the shutter
    command, ``exposure_us`` the exposure event, both microseconds on the
    module clock.
    """
    try:
        with open(image_path, "rb") as handle:
            data = handle.read(limit)
    except OSError:
        return None
    match = CAPTURE_TIMING_PATTERN.search(data)
    if match is None:
        return None
    return {"trigger_us": int(match.group(1)) * 1000,
            "exposure_us": int(match.group(2))}


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def match_mrk_records(reference_rows, records):
    """Map camera labels to event-mark records by image number then position.

    ``reference_rows`` is ``[{"label", "reference"}]`` captured right after
    alignment (the reference triple is lon/lat/height). The event-mark index is
    the image's trailing number in this dataset (``MAX_0002`` -> record 2), but
    the match is accepted only when the record position equals the camera's
    stored reference; otherwise the nearest record within 0.1 m is used, and
    beyond that the camera is left unmatched rather than paired with a wrong
    exposure.
    """
    by_index = {record["index"]: record for record in records}
    matched = {}
    for row in reference_rows:
        reference = row["reference"]
        if reference is None or not all(
                math.isfinite(float(value)) for value in reference):
            continue
        lon, lat = float(reference[0]), float(reference[1])
        record = None
        number = re.search(r"(\d+)$", row["label"])
        if number is not None:
            candidate = by_index.get(int(number.group(1)))
            if (candidate is not None and
                    abs(candidate["lat"] - lat) < 1e-6 and
                    abs(candidate["lon"] - lon) < 1e-6):
                record = candidate
        if record is None and records:
            nearest = min(
                records,
                key=lambda item: (item["lat"] - lat) ** 2 +
                                 (item["lon"] - lon) ** 2)
            if (abs(nearest["lat"] - lat) < 1e-6 and
                    abs(nearest["lon"] - lon) < 1e-6):
                record = nearest
        if record is not None:
            matched[row["label"]] = record
    return matched


def analyse_timing(reference_rows, images, records):
    """Compare each exposure's event time with its capture time.

    The RTK log and the camera write their timestamps on different clocks, so
    the two series are aligned by the median difference first; what remains is
    the per-exposure disagreement between the event mark and the exposure.
    ``trigger_to_exposure`` is the delay between the shutter command and the
    exposure; it is NOT a position error because the event mark is stamped at
    the exposure, which the event-vs-exposure residual tests directly.

    Also reports the block flight speed from consecutive event marks (straight
    line between neighbouring exposures, so line turns lower individual
    samples and the median is the usable value) and the solution-status and
    sigma distributions, which are what a per-camera weight can be checked
    against.
    """
    paths = {os.path.splitext(os.path.basename(path))[0]: path
             for path in images}
    matched = match_mrk_records(reference_rows, records)
    rows = []
    offsets = []
    trigger_gaps = []
    for label, record in sorted(matched.items()):
        path = paths.get(label)
        if path is None:
            continue
        capture = read_capture_timing(path)
        if capture is None:
            continue
        exposure_s = capture["exposure_us"] / 1e6
        offsets.append(record["time_s"] - exposure_s)
        trigger_gaps.append(
            (capture["exposure_us"] - capture["trigger_us"]) / 1e6)
        rows.append({
            "label": label,
            "mrk_time_s": record["time_s"],
            "trigger_us": capture["trigger_us"],
            "exposure_us": capture["exposure_us"],
            "fix_flag": record["fix_flag"],
        })
    if not rows:
        return {"images": 0, "mrk_records": len(records),
                "matched_cameras": len(matched)}

    clock_offset = _median(offsets)
    residuals = [offset - clock_offset for offset in offsets]
    speeds = []
    for first, second in zip(records, records[1:]):
        delta = second["time_s"] - first["time_s"]
        if delta <= 0.05:
            continue
        mean_lat = math.radians(0.5 * (first["lat"] + second["lat"]))
        north = (second["lat"] - first["lat"]) * 111320.0
        east = (second["lon"] - first["lon"]) * 111320.0 * math.cos(mean_lat)
        speeds.append(math.hypot(north, east) / delta)
    status_counts = {}
    for record in records:
        key = str(record["fix_flag"])
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "images": len(rows),
        "mrk_records": len(records),
        "matched_cameras": len(matched),
        "clock_offset_s": clock_offset,
        "event_vs_exposure_rms_s": root_mean_square(residuals),
        "event_vs_exposure_max_abs_s": max(abs(value) for value in residuals),
        "trigger_to_exposure_median_s": _median(trigger_gaps),
        "trigger_to_exposure_min_s": min(trigger_gaps),
        "trigger_to_exposure_max_s": max(trigger_gaps),
        "block_speed_m_s": _median(speeds) if speeds else None,
        "block_speed_min_m_s": min(speeds) if speeds else None,
        "block_speed_max_m_s": max(speeds) if speeds else None,
        "fix_status_counts": status_counts,
        "sigma_lat_m": {
            "min": min(record["sigma_lat_m"] for record in records),
            "max": max(record["sigma_lat_m"] for record in records),
        },
        "sigma_lon_m": {
            "min": min(record["sigma_lon_m"] for record in records),
            "max": max(record["sigma_lon_m"] for record in records),
        },
        "sigma_hgt_m": {
            "min": min(record["sigma_hgt_m"] for record in records),
            "max": max(record["sigma_hgt_m"] for record in records),
        },
        "per_image": rows,
    }


def split_csv(path, check_labels):
    """Return (header, control_lines, check_lines) from a label,x,y,z CSV."""
    with open(path, encoding="utf-8") as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    header, rows = lines[0], lines[1:]
    control = [header] + [row for row in rows
                          if row.split(",")[0].strip() not in check_labels]
    checks = [header] + [row for row in rows
                         if row.split(",")[0].strip() in check_labels]
    return control, checks


def require_geographic_crs(code):
    """Refuse a projected --crs: references are converted as lon/lat here."""
    if not code:
        return
    if not hasattr(alphashape, "CoordinateSystem"):
        raise ValueError("--crs needs a build with PROJ")
    system = alphashape.CoordinateSystem(int(code))
    if "+proj=longlat" not in system.toProj4():
        raise ValueError(
            "--crs must name a geographic (lon/lat) CRS; projected "
            "coordinates are not supported")


# Observations of one surveyed point are grouped by identical (x, y, z), so a
# GCP is keyed on its coordinates rounded to this many decimals of a degree.
ODM_GCP_GROUP_DECIMALS = 9
# A CSV point is the same survey point as a gcp_list coordinate within this
# angular tolerance (~1 m at these latitudes) for the label match.
GCP_NAME_MATCH_TOLERANCE_DEG = 1e-5
# The published height reference. The two survey files carry the same z values,
# so comparing them cannot identify the datum; the publisher states the value,
# and an orthometric height would need a geoid model this script does not carry.
GCP_HEIGHT_CHOICES = ("ellipsoidal", "orthometric")

# How the gcp_list pixel columns are read. "file" trusts the values as written:
# this dataset's producer writes corner-origin coordinates, the engine's own
# convention, so the shift is zero. "format" applies the ODM/OpenSfM file
# format's documented integer-origin -> corner-origin conversion (+0.5 px; see
# parse_odm_gcp_list). The default is driven by the evidence, not tuned.
GCP_PIXEL_ORIGINS = ("file", "format")
GCP_PIXEL_ORIGIN_SHIFT_PX = {"file": 0.0, "format": 0.5}

# Camera RTK positions and the surveyed GCPs can be expressed in two different
# realizations of the same geodetic system. ETRS89 (what a European NTRIP
# network such as SAPOS/EPOSA broadcasts) is a static frame fixed at epoch
# 1989.0; WGS84(G2139)/ITRF2014 is the receiver's own global solution at the
# capture epoch. At this site in 2022.4 those differ by about 0.85 m
# horizontally. ``--frame-correction`` names the published transform to apply
# to the camera references before the adjustment. "none" leaves the references
# as imported (the historic behaviour). The two named directions are the
# hypothesis and its mirror image; "empirical" removes the translation measured
# on the same fold's control-only GCP adjustment, which needs no frame
# assumption and uses no check point.
FRAME_CORRECTION_CHOICES = (
    "none",
    "itrf2014-to-etrs89",
    "etrs89-to-itrf2014",
    "empirical",
)
# EPSG geographic 3D CRSs for the published transform. ITRF2014 (EPSG:7912) is
# the frame WGS84(G2139) is aligned to; ETRS89 (EPSG:4258) is the static
# European frame. PROJ resolves each pair to its time-dependent ETRF realization
# and applies the epoch passed as the coordinate's ``t``.
FRAME_CRS_EPSG = {
    "itrf2014-to-etrs89": (7912, 4258),
    "etrs89-to-itrf2014": (4258, 7912),
}
# GPS time is ahead of UTC by the leap seconds in force. 18 s covers every
# epoch since 2017-01-01, which is the era this dataset comes from.
GPS_UTC_LEAP_SECONDS = 18.0

# A GNSS reference records the antenna phase centre, not the camera's
# projection centre. The dataset carries no lever-arm field: the EXIF/XMP
# records the antenna position (RtkFlag 50, RtkStdLon/Lat/Hgt) and the gimbal
# angles, and the RINEX marker file records the same antenna positions, but
# nothing states where the phase centre sits relative to the lens. The fixed
# value below is the measured seed from the GCP-only adjustment, which put the
# camera centres about 0.23 m below the recorded RTK positions (m13-validation).
# In the camera frame +z is forward (down) for a nadir mount, so an antenna
# above the lens is a NEGATIVE z lever arm.
ANTENNA_FIXED_OFFSET_M = (0.0, 0.0, -0.23)
# The estimated variants use the mean camera-frame offset measured on the same
# fold's GCP-only adjustment. Only control GCPs take part, so the held-out
# check point never reaches the estimate.

# ``--camera-accuracy per-image`` takes each camera's own accuracy from the
# XMP the engine's AddPhotos (load_xmp_accuracy) stored, instead of re-importing
# one scalar for the whole block. The scalar form is still accepted and is the
# uniform baseline.
CAMERA_ACCURACY_PER_IMAGE = "per-image"

# A camera whose stored reference sits further than this many of its own
# per-axis sigmas from the adjusted centre is reported by the robust diagnostic.
# It is a report-only threshold: no variant's weights are changed by it.
OUTLIER_SIGMA_THRESHOLD = 3.0

# How many flagged cameras per fold carry the full residual record. Every
# flagged label is still listed, but a fold where a per-image weight turns a
# systematic datum offset into a ten-sigma residual on all 146 cameras would
# otherwise multiply the report (and the benchmark log) by its size.
OUTLIER_DETAIL_LIMIT = 20


def parse_camera_accuracy(text):
    """``per-image``, or a strictly positive uniform accuracy in metres."""
    if text == CAMERA_ACCURACY_PER_IMAGE:
        return text
    try:
        value = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "--camera-accuracy needs 'per-image' or a number of metres") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("--camera-accuracy must be positive and finite")
    return value


def camera_accuracy_summary(chunk):
    """Distribution of the per-camera reference accuracy the block carries.

    ``uniform_defaults`` counts cameras still on the 10 m consumer-default
    triple, which is what a block with no per-image metadata looks like. The
    per-axis min/mean/max show whether the import produced a real spread.
    """
    axes = {"east": [], "north": [], "up": []}
    defaults = 0
    for camera in chunk.cameras:
        if camera.reference is None:
            continue
        accuracy = camera.reference_accuracy
        if accuracy is None:
            continue
        values = tuple(float(axis) for axis in accuracy)
        axes["east"].append(values[0])
        axes["north"].append(values[1])
        axes["up"].append(values[2])
        if values == (10.0, 10.0, 10.0):
            defaults += 1
    count = len(axes["east"])
    summary = {"cameras": count, "uniform_defaults": defaults}
    if count:
        for name, values in axes.items():
            summary[name] = {"min_m": min(values), "mean_m": sum(values) / count,
                             "max_m": max(values)}
    return summary


def parse_odm_gcp_list(path, pixel_shift_px=0.0):
    """Parse an ODM gcp_list.txt into one dict per survey point.

    The header is a CRS string (``EPSG:4326``: x=longitude, y=latitude). Each
    following row is ``x y z px py image [label]`` — one image observation of
    one point. Rows with the same (x, y, z) are one GCP, collected in first-seen
    order so the returned list has one entry per distinct surveyed point.

    Pixel convention (verified 2026-11-11). The engine's pixel coordinates are
    corner-origin: ``Calibration::project`` returns ``width / 2 + f x/z``
    (``engine/src/model/camera/calibration.hpp``), so the centre of the
    top-left pixel is (0.5, 0.5) and the image centre is (width/2, height/2);
    ``keypoint_to_pixel`` adds 0.5 to the detector's corner indices for the
    same reason. The ODM/OpenSfM *file-format* documentation describes
    integer-origin pixel coordinates (top-left pixel centre (0,0),
    bottom-right (width-1, height-1); OpenSfM ``doc/source/geometry.rst``), and
    OpenSfM normalises a gcp_list value with ``(p + 0.5 - width/2) / size``
    (``opensfm/features.py``), so a value written to that convention maps to
    the engine's coordinate ``p + 0.5``. This dataset's file was produced by
    GCP Editor Pro, whose tagger records the click offset from the image's
    top-left edge in the continuous range [0, width] (``getPos`` in the 2022
    ``smartimage`` component, and the SVG viewBox in the current one), so its
    values are already corner-origin and match the engine. No half-pixel
    conversion is therefore applied to this file; ``--diagnostics`` keeps the
    ±0.5 px probes so a file written by an integer-origin producer (or a
    tool-convention change) remains measurable.
    """
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.read().splitlines() if line.strip()]
    if not lines:
        raise ValueError("gcp_list is empty")
    header = lines[0]
    gcps = []
    order = {}
    for line in lines[1:]:
        tokens = line.split()
        if len(tokens) < 6:
            continue
        x, y, z = float(tokens[0]), float(tokens[1]), float(tokens[2])
        # ``pixel_shift_px`` is the convention conversion selected by
        # --gcp-pixel-origin: 0 for a file already in the engine's corner-origin
        # frame, +0.5 for a file in the ODM/OpenSfM integer-origin frame.
        px = float(tokens[3]) + pixel_shift_px
        py = float(tokens[4]) + pixel_shift_px
        image = tokens[5]
        label = tokens[6] if len(tokens) > 6 else None
        key = (round(x, ODM_GCP_GROUP_DECIMALS),
               round(y, ODM_GCP_GROUP_DECIMALS),
               round(z, ODM_GCP_GROUP_DECIMALS))
        if key not in order:
            order[key] = len(gcps)
            gcps.append({"lon": x, "lat": y, "height": z, "label": label,
                         "observations": []})
        gcps[order[key]]["observations"].append(
            {"px": px, "py": py, "image": image})
    return header, gcps


def _csv_column(index, names):
    """The position of the first header name present, else None."""
    for name in names:
        if name in index:
            return index[name]
    return None


def parse_gcp_names_csv(path):
    """Parse the survey CSV into one dict per point.

    The publisher's header is ``Name;Easting;Northing;Elevation;Longitude;Latitude``
    but it carries a UTF-8 BOM, so the file is read as utf-8-sig and the header names
    are matched loosely so a rename or a comma/tab delimiter does not silently change
    the result. A header that names none of the expected columns fails with the header
    shown, not with a bare KeyError.
    """
    with open(path, encoding="utf-8-sig") as handle:
        lines = [line.strip() for line in handle.read().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("gcp names CSV has no data rows")
    header = lines[0]
    delimiter = ";" if ";" in header else ("," if "," in header else "\t")

    def split(line):
        return [cell.strip() for cell in line.split(delimiter)]

    columns = [cell.lower() for cell in split(header)]
    index = {name: position for position, name in enumerate(columns)}
    name_col = _csv_column(index, ("name", "label", "id", "point", "point_name"))
    lon_col = _csv_column(index, ("longitude", "lon", "long", "x"))
    lat_col = _csv_column(index, ("latitude", "lat", "y"))
    elev_col = _csv_column(index, ("elevation", "height", "alt", "altitude", "z"))
    if None in (name_col, lon_col, lat_col, elev_col):
        raise ValueError("gcp names CSV header not understood: " + header)
    rows = []
    for line in lines[1:]:
        fields = split(line)
        rows.append({
            "name": fields[name_col],
            "elevation": float(fields[elev_col]),
            "longitude": float(fields[lon_col]),
            "latitude": float(fields[lat_col]),
        })
    return columns, rows


def match_gcp_names(gcps, names):
    """Give each GCP its CSV survey name, falling back to G1..Gn when no point
    matches within ``GCP_NAME_MATCH_TOLERANCE_DEG``. Records the matched CSV
    elevation on the GCP for the z-datum determination."""
    for position, gcp in enumerate(gcps):
        best_name, best_distance, best_elevation = None, None, None
        for row in names:
            distance = math.hypot(gcp["lon"] - row["longitude"],
                                  gcp["lat"] - row["latitude"])
            if best_distance is None or distance < best_distance:
                best_distance, best_name, best_elevation = (
                    distance, row["name"], row["elevation"])
        if best_distance is not None and best_distance < GCP_NAME_MATCH_TOLERANCE_DEG:
            gcp["label"] = best_name
            gcp["csv_elevation"] = best_elevation
        else:
            gcp["label"] = "G{}".format(position + 1)
    return gcps


def z_consistency(gcps, gcp_height):
    """Record how the two survey files' z values compare, and what z means.

    gcp_list z and the CSV elevation are the same published numbers, so their
    offset only checks that the label match is right — it cannot identify the
    height datum. The datum is therefore taken from ``--gcp-height`` (the
    publisher labels the CSV column "Ellipsoidal height"), not inferred here.
    """
    offsets = [gcp["height"] - gcp["csv_elevation"]
               for gcp in gcps if "csv_elevation" in gcp]
    record = {"gcp_height": gcp_height, "matched_points": len(offsets)}
    if not offsets:
        record["mean_offset_m"] = None
        record["max_abs_offset_m"] = None
        record["reason"] = "no gcp matched a CSV elevation"
        return record
    record["mean_offset_m"] = sum(offsets) / len(offsets)
    record["max_abs_offset_m"] = max(abs(value) for value in offsets)
    return record


def run_real(args):
    """Match -> align -> detect -> import GCPs -> georeference -> bundle-adjust."""
    require_geographic_crs(args.crs)
    images = collect_images(args.real)
    if len(images) < 3:
        raise ValueError("need at least three images in " + args.real)
    doc = alphashape.Document()
    chunk = doc.add_chunk("Real block")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos", filenames=images)
    if args.crs:
        chunk.crs_epsg = int(args.crs)
    doc.run_task("MatchPhotos")
    doc.run_task("AlignCameras")
    chunk.detectMarkers(target_type=args.target)

    with open(args.gcp, encoding="utf-8") as handle:
        rows = [line for line in handle.read().splitlines() if line.strip()]
    labels = [row.split(",")[0].strip() for row in rows[1:]]
    if args.check_points:
        check_labels = {name.strip() for name in args.check_points.split(",")}
    else:
        # No designation supplied: hold out every fifth point, at least one.
        check_labels = set(labels[4::5]) or {labels[-1]}
    control_lines, check_lines = split_csv(args.gcp, check_labels)

    with tempfile.TemporaryDirectory() as directory:
        control_path = os.path.join(directory, "control.csv")
        check_path = os.path.join(directory, "check.csv")
        for path, lines in ((control_path, control_lines),
                            (check_path, check_lines)):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        chunk.importReference(path=control_path, marker_type="gcp")
        chunk.importReference(path=check_path, marker_type="check_point")

    # OptimizeCameras only consumes marker references in a georeferenced chunk.
    # A block without usable camera GPS needs this fit to seed the transform
    # from the control markers before the bundle adjustment runs.
    chunk.updateTransform()
    if chunk.meta.get("update_transform.applied") != "1":
        raise ValueError("georeferencing failed: " +
                         str(chunk.meta.get("update_transform.reason",
                                            "unknown")))
    chunk.optimizeCameras(use_gcps=True)

    labels = sorted(check_labels)
    targets = reference_local_targets(chunk, labels)
    errors = check_point_errors(chunk, labels, targets)
    if len(errors) != len(labels):
        raise ValueError("evaluated {} check points, expected {}".format(
            len(errors), len(labels)))
    return {
        "scenario": "real-block",
        "images": len(images),
        "markers": len(chunk.markers),
        "control_points": len(control_lines) - 1,
        "check_points": len(check_lines) - 1,
        "evaluated_check_points": len(errors),
        "optimizer": "OptimizeCameras",
        "check_point_rms_m": {
            "horizontal": root_mean_square(
                [e["horizontal"] for e in errors]),
            "vertical": root_mean_square([e["vertical"] for e in errors]),
            "total": root_mean_square([e["total"] for e in errors]),
        },
        "check_point_errors_m": errors,
    }


def project_path_for(args):
    """The on-disk project the protocol saves once and reloads per fold."""
    base = os.path.dirname(os.path.abspath(args.json_out or "gcp_report.json"))
    return os.path.join(base, "gcp_site.psx")


def skipped_observations(chunk, gcps):
    """Count gcp_list observations whose camera is missing or did not align.

    run_fold silently skips an observation whose image label resolves to no
    enabled camera; this records how many, so a block that registered a small
    fraction of its images cannot look well observed by accident.
    """
    cameras = {camera.label: camera for camera in chunk.cameras}
    total = 0
    skipped = 0
    for gcp in gcps:
        for observation in gcp["observations"]:
            total += 1
            stem = os.path.splitext(observation["image"])[0]
            camera = cameras.get(stem) or cameras.get(observation["image"])
            if camera is None or not camera.enabled:
                skipped += 1
    return {"observations": total, "in_unaligned_cameras": skipped}


def align_block(args, images, gcps):
    """MatchPhotos + AlignCameras once, save the project, return alignment facts.

    The same block is reloaded for every fold, so the expensive matching and
    alignment happen exactly once. keypoint_limit_per_mpx is the 16 GB-runner
    budget recorded in the manifest; the dense stage never runs.
    """
    doc = alphashape.Document()
    chunk = doc.add_chunk("Helenenschacht RTK block")
    doc.active_chunk = chunk
    # Per-image mode reads each photo's XMP RTK deviations (and fix status) at
    # import, so the alignment itself is weighted by them. The uniform modes
    # keep the historic behaviour (no XMP accuracy) so the comparison against
    # earlier runs is like for like.
    per_image = args.camera_accuracy == CAMERA_ACCURACY_PER_IMAGE
    doc.run_task("AddPhotos", filenames=images,
                 load_xmp_accuracy=per_image)
    chunk.crs_epsg = 4326
    doc.run_task("MatchPhotos", keypoint_limit_per_mpx=args.keypoint_limit)
    doc.run_task("AlignCameras")
    project = project_path_for(args)
    doc.save(project)
    total = len(chunk.cameras)
    aligned = sum(1 for camera in chunk.cameras if camera.enabled)
    # Kept out of the JSON report (run_real_odm pops it): the timing analysis
    # only needs labels and stored positions to pair exposures with event marks.
    reference_rows = [{"label": camera.label,
                       "reference": (tuple(float(v) for v in camera.reference)
                                     if camera.reference is not None else None)}
                      for camera in chunk.cameras]
    return {"project": project, "cameras_total": total,
            "cameras_aligned": aligned,
            "keypoint_limit_per_mpx": args.keypoint_limit,
            "camera_accuracy": args.camera_accuracy,
            "xmp_accuracy_imported":
                args.camera_accuracy == CAMERA_ACCURACY_PER_IMAGE,
            # The CRS each reference slot is interpreted in. Both are 4326:
            # the protocol sets the chunk and the importers take lon/lat/height.
            # EPSG:4326 carries no realization or coordinate epoch, which is
            # exactly why a camera-vs-GCP frame difference is invisible to the
            # engine and has to be tested and corrected here.
            "camera_crs_epsg": chunk.camera_crs_epsg,
            "marker_crs_epsg": chunk.marker_crs_epsg,
            "observations": skipped_observations(chunk, gcps),
            "_reference_rows": reference_rows}


def import_camera_references_for_block(chunk, accuracy_m, directory,
                                       along_shift_m=0.0,
                                       accuracy_scale=1.0,
                                       frame_shift_enu_m=None,
                                       frame_transform=None):
    """Re-import every camera reference with the variant's accuracy/position.

    ``accuracy_m`` is a scalar for the uniform baseline or
    CAMERA_ACCURACY_PER_IMAGE, in which case each row keeps the accuracy
    AddPhotos stored from that photo's XMP (a camera carrying the 10 m default
    stays there: no per-image accuracy exists). The binding's
    ``reference_accuracy`` is never None once a location exists, so a uniform
    run must write the scalar rather than read it back.

    ``accuracy_scale`` multiplies whichever accuracy was selected, which is how
    the GCP-weight probe loosens every camera's prior by the same factor while
    the marker weight stays at the survey's stated value.

    ``along_shift_m`` is an along-track lead to REMOVE: each position is moved
    this many metres opposite its local flight direction, estimated from the
    camera's neighbours in label order. That is how the timing diagnostic
    applies a speed x latency correction. The direction is computed from the
    UNshifted track, so the two do not feed back into each other.

    ``frame_transform`` converts each position into the GCP frame with the
    published time-dependent transformation (``(lon, lat, height)`` in,
    converted triple out). ``frame_shift_enu_m`` is the constant local
    displacement measured as *reference minus adjusted centre* on the same
    fold's control-only GCP adjustment; it is REMOVED from every position
    (``reference - shift``), which is the direction that moves the camera
    positions onto the GCP datum. That correction too is therefore leak-free.
    """
    cameras = []
    for camera in chunk.cameras:
        reference = camera.reference
        if reference is None or not all(
                math.isfinite(float(value)) for value in reference):
            continue
        cameras.append(camera)
    cameras.sort(key=lambda camera: camera.label)

    shifted = {}
    if along_shift_m:
        origin = tuple(float(v) for v in cameras[0].reference) if cameras else None
        track = {camera.label: geodetic_to_enu(
            float(camera.reference[0]), float(camera.reference[1]),
            float(camera.reference[2]), origin) for camera in cameras}
        for index, camera in enumerate(cameras):
            previous = cameras[max(0, index - 1)].label
            following = cameras[min(len(cameras) - 1, index + 1)].label
            dx = track[following][0] - track[previous][0]
            dy = track[following][1] - track[previous][1]
            norm = math.hypot(dx, dy)
            if norm < 1e-9:
                finish = tuple(float(v) for v in camera.reference)
            else:
                offset = (-along_shift_m * dx / norm,
                          -along_shift_m * dy / norm, 0.0)
                finish = enu_to_geodetic(
                    offset, tuple(float(v) for v in camera.reference))
            shifted[camera.label] = finish

    rows = []
    for camera in cameras:
        reference = shifted.get(
            camera.label, tuple(float(v) for v in camera.reference))
        lon, lat, height = (float(reference[0]), float(reference[1]),
                            float(reference[2]))
        if frame_transform is not None:
            lon, lat, height = frame_transform(lon, lat, height)
        if frame_shift_enu_m is not None:
            # reference - shift: the measured offset (reference minus the
            # GCP-frame adjusted centre) is removed, not added.
            lon, lat, height = enu_to_geodetic(
                (-float(frame_shift_enu_m[0]), -float(frame_shift_enu_m[1]),
                 -float(frame_shift_enu_m[2])), (lon, lat, height))
        reference = (lon, lat, height)
        if accuracy_m == CAMERA_ACCURACY_PER_IMAGE:
            accuracy = camera.reference_accuracy
            if accuracy is None:
                accuracy = (10.0, 10.0, 10.0)
            else:
                accuracy = tuple(float(v) for v in accuracy)
        else:
            accuracy = (accuracy_m, accuracy_m, accuracy_m)
        if accuracy_scale != 1.0:
            accuracy = tuple(float(value) * accuracy_scale for value in accuracy)
        rows.append((camera.label, reference[0], reference[1], reference[2],
                     accuracy))
    path = os.path.join(directory, "camera_references.csv")
    write_camera_reference_csv(path, rows)
    chunk.importReference(path=path)


def camera_consistency(chunk, antenna_offset_m=None):
    """How far each aligned camera's stored reference is from its adjusted centre.

    Per camera with a finite reference: the adjusted centre (chunk-local, mapped
    to ECEF by chunk.transform) expressed in the reference's local east/north/up
    frame. A constant up offset points at a height-datum mismatch between the
    camera altitudes and the GCP heights; a metre-level spread points at
    positions that are not RTK-fixed.

    When a GNSS antenna lever arm was applied to this fold, the comparison is
    made against the modelled antenna, ``A = C + R_pose^T * offset`` (the same
    camera-frame convention the engine's centre prior uses), so the reported
    offset is what the mount does not explain rather than the mount itself.

    The horizontal difference is also projected onto the flight direction,
    taken from each camera's neighbours in capture order (labels sorted). A
    constant along-track offset is the signature of a trigger/GNSS timing lag
    (and it does NOT change sign between flight lines, because the direction is
    per-camera); a constant world offset shows up in mean east/north instead.
    Measured only, never applied.
    """
    transform = chunk.transform
    block = [[float(transform[r][c]) for c in range(3)] for r in range(3)]
    chunk_scale = abs(matrix_determinant(block)) ** (1.0 / 3.0)
    chunk_rotation = None
    if antenna_offset_m is not None:
        if not math.isfinite(chunk_scale) or chunk_scale <= 0.0:
            raise ValueError("chunk transform has no invertible rotation")
        chunk_rotation = [[block[r][c] / chunk_scale for c in range(3)]
                          for r in range(3)]
    samples = []
    for camera in chunk.cameras:
        reference = camera.reference
        if not camera.enabled or reference is None or not all(
                math.isfinite(float(value)) for value in reference):
            continue
        local = camera.center
        ecef = [sum(transform[r][c] * local[c] for c in range(3)) + transform[r][3]
                for r in range(3)]
        if antenna_offset_m is not None:
            pose = camera.pose
            pose_rotation = orthonormal_rotation(
                [[pose[r][c] for c in range(3)] for r in range(3)])
            # R_pose^T maps the camera-frame lever to a local metric direction;
            # the chunk rotation carries it to ECEF (the chunk scale cancels).
            lever_local = [
                sum(pose_rotation[r][c] * antenna_offset_m[r] for r in range(3))
                for c in range(3)]
            lever_ecef = [
                sum(chunk_rotation[r][c] * lever_local[c] for c in range(3))
                for r in range(3)]
            ecef = [ecef[r] + lever_ecef[r] for r in range(3)]
        enu = ecef_to_enu(ecef, (reference[0], reference[1], reference[2]))
        samples.append({"label": camera.label, "reference": reference,
                        "east": enu[0], "north": enu[1], "up": enu[2]})
    if not samples:
        return {"cameras": 0}

    def mean(values):
        return sum(values) / len(values) if values else None

    ordered = sorted(samples, key=lambda item: item["label"])
    origin = ordered[0]["reference"]
    track = {}
    for item in ordered:
        reference = item["reference"]
        track[item["label"]] = geodetic_to_enu(
            reference[0], reference[1], reference[2], origin)

    along, cross = [], []
    for index, item in enumerate(ordered):
        previous = ordered[max(0, index - 1)]["label"]
        following = ordered[min(len(ordered) - 1, index + 1)]["label"]
        dx = track[following][0] - track[previous][0]
        dy = track[following][1] - track[previous][1]
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            continue
        hx, hy = dx / norm, dy / norm
        item["along"] = item["east"] * hx + item["north"] * hy
        item["cross"] = item["east"] * -hy + item["north"] * hx
        # Group by the dominant axis of travel so a world-frame bias that flips
        # sign with heading is visible, while a timing lag stays same-signed.
        item["bound"] = ("positive" if (hy if abs(hy) >= abs(hx) else hx) >= 0.0
                         else "negative")
        along.append(item["along"])
        cross.append(item["cross"])

    result = {"cameras": len(samples),
              "mean_east_m": mean([item["east"] for item in samples]),
              "mean_north_m": mean([item["north"] for item in samples]),
              "mean_up_m": mean([item["up"] for item in samples]),
              "rms_east_m": root_mean_square([item["east"] for item in samples]),
              "rms_north_m": root_mean_square([item["north"] for item in samples]),
              "rms_up_m": root_mean_square([item["up"] for item in samples])}
    if along:
        result["mean_along_track_m"] = mean(along)
        result["mean_cross_track_m"] = mean(cross)
        result["rms_along_track_m"] = root_mean_square(along)
        result["rms_cross_track_m"] = root_mean_square(cross)
        for bound in ("positive", "negative"):
            values = [item["along"] for item in ordered
                      if item.get("bound") == bound]
            result["mean_along_track_{}_m".format(bound)] = mean(values)
            result["cameras_{}".format(bound)] = len(values)
    return result


# Calibration coefficients in the engine's packed order
# (model/camera/calibration.hpp). The set_sensor_fixed_params binding and
# OptimizeCameras' fit_* flags use the same names.
CALIBRATION_PARAM_NAMES = (
    "f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4")

# The dome fits scale horizontal distance by this many metres before squaring,
# so a coefficient is "metres of height per (100 m)^2" instead of a number near
# 1e-6 that the normal equations cannot resolve cleanly.
DOME_RADIUS_SCALE_M = 100.0


def enu_of_local(chunk, point, origin):
    """Chunk-local point -> east/north/up metres about a geodetic `origin`."""
    transform = chunk.transform
    ecef = [sum(float(transform[r][c]) * point[c] for c in range(3)) +
            float(transform[r][3]) for r in range(3)]
    return ecef_to_enu(ecef, origin)


def block_origin(chunk):
    """A geodetic anchor for the fold's ENU frame.

    The first aligned camera's stored reference is used so the frame is the
    same one the camera residuals are expressed in; a block with no references
    falls back to its first adjusted centre.
    """
    cameras = sorted(chunk.cameras, key=lambda item: item.label)
    for camera in cameras:
        reference = camera.reference
        if camera.enabled and reference is not None and all(
                math.isfinite(float(value)) for value in reference):
            return (float(reference[0]), float(reference[1]),
                    float(reference[2]))
    for camera in cameras:
        if not camera.enabled:
            continue
        transform = chunk.transform
        local = camera.center
        ecef = [sum(float(transform[r][c]) * local[c] for c in range(3)) +
                float(transform[r][3]) for r in range(3)]
        lat, lon, height = ecef_to_geodetic(*ecef)
        return (lon, lat, height)
    raise ValueError("no aligned camera to anchor the block")


def convex_hull_area(points):
    """Area of the convex hull of (x, y) points in square metres.

    Andrew's monotone chain. Fewer than three distinct points has no area; the
    GCP hull of a five-point survey is a useful comparison with the block hull
    because a check point outside the control hull is extrapolation.
    """
    unique = sorted(set((float(x), float(y)) for x, y in points))
    if len(unique) < 3:
        return 0.0

    def cross(origin, a, b):
        return ((a[0] - origin[0]) * (b[1] - origin[1]) -
                (a[1] - origin[1]) * (b[0] - origin[0]))

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    area = 0.0
    for index, point in enumerate(hull):
        following = hull[(index + 1) % len(hull)]
        area += point[0] * following[1] - following[0] * point[1]
    return abs(area) * 0.5


def linear_least_squares(rows, values, names):
    """Ordinary least squares with an intercept, standard errors and t values.

    ``rows`` holds one predictor row per sample (the intercept is added here),
    ``names`` labels the predictors. A model with fewer samples than parameters
    is reported as ``underdetermined`` rather than silently interpolated: with
    five check points the six-term quadratic surface is exactly that case, and
    claiming a fit from it would overstate what the data support. A singular
    normal matrix (collinear predictors) is reported the same way.
    """
    if len(rows) != len(values):
        raise ValueError("dome fit: rows and values differ in length")
    samples = len(values)
    parameters = 1 + len(names)
    record = {"model": ["intercept"] + list(names), "samples": samples,
              "parameters": parameters}
    if samples < parameters:
        record["status"] = "underdetermined"
        record["note"] = ("fewer samples ({}) than parameters ({}); no unique "
                          "fit".format(samples, parameters))
        return record
    design = [[1.0] + [float(value) for value in row] for row in rows]
    normal = [[sum(design[i][a] * design[i][b] for i in range(samples))
               for b in range(parameters)] for a in range(parameters)]
    rhs = [sum(design[i][a] * values[i] for i in range(samples))
           for a in range(parameters)]
    try:
        inverse = invert_matrix(normal)
    except ValueError as error:
        record["status"] = "singular"
        record["note"] = str(error)
        return record
    coefficients = [sum(inverse[a][b] * rhs[b] for b in range(parameters))
                    for a in range(parameters)]
    residuals = [values[i] - sum(design[i][a] * coefficients[a]
                                 for a in range(parameters))
                 for i in range(samples)]
    rss = sum(value * value for value in residuals)
    mean = sum(values) / samples
    tss = sum((value - mean) ** 2 for value in values)
    degrees = samples - parameters
    record.update({
        "status": "ok" if degrees > 0 else "zero_degrees_of_freedom",
        "coefficients": coefficients,
        "residual_rms_m": math.sqrt(rss / samples),
        "r_squared": (1.0 - rss / tss) if tss > 0.0 else None,
        "degrees_of_freedom": degrees,
    })
    standard_errors = None
    t_values = None
    if degrees > 0:
        variance = rss / degrees
        standard_errors = [math.sqrt(max(variance * inverse[a][a], 0.0))
                           for a in range(parameters)]
        t_values = [coefficients[a] / standard_errors[a]
                    if standard_errors[a] > 0.0 else None
                    for a in range(parameters)]
    by_name = {}
    for index, name in enumerate(record["model"]):
        by_name[name] = {
            "coefficient": coefficients[index],
            "standard_error": (standard_errors[index]
                               if standard_errors is not None else None),
            "t": (t_values[index] if t_values is not None else None),
        }
    record["by_name"] = by_name
    record["standard_errors"] = standard_errors
    record["t_values"] = t_values
    return record


def block_geometry(chunk):
    """The fold's adjusted camera-centre geometry in ENU metres.

    The centroid is the mean of the aligned camera centres and every radius is
    measured from it, so the check-point and camera dome fits share one origin
    definition. The hull area is the horizontal extent of the flight plan, to
    compare with the much smaller survey-GCP hull.
    """
    origin = block_origin(chunk)
    cameras = []
    for camera in sorted(chunk.cameras, key=lambda item: item.label):
        if not camera.enabled:
            continue
        enu = enu_of_local(chunk, camera.center, origin)
        cameras.append({"label": camera.label, "east_m": enu[0],
                        "north_m": enu[1], "up_m": enu[2]})
    if not cameras:
        raise ValueError("no aligned cameras in the fold")
    centroid_east = sum(item["east_m"] for item in cameras) / len(cameras)
    centroid_north = sum(item["north_m"] for item in cameras) / len(cameras)
    for item in cameras:
        item["radius_m"] = math.hypot(item["east_m"] - centroid_east,
                                      item["north_m"] - centroid_north)
    return {
        "origin": origin,
        "cameras": cameras,
        "centroid_east_m": centroid_east,
        "centroid_north_m": centroid_north,
        "radius_m": max(item["radius_m"] for item in cameras),
        "hull_area_m2": convex_hull_area(
            [(item["east_m"], item["north_m"]) for item in cameras]),
    }


def camera_dome_fit(chunk, geometry):
    """Stored camera reference minus adjusted centre, vertical, versus r^2.

    Every aligned camera with a reference contributes one sample (146 in this
    block): far more than the five check points, so this is the strongest test
    for a smooth radial bend. The residual is the same signed quantity the
    protocol's camera-consistency report calls "camera vs GPS up": positive
    means the stored RTK position is above the adjusted centre.
    """
    origin = geometry["origin"]
    centroid_east = geometry["centroid_east_m"]
    centroid_north = geometry["centroid_north_m"]
    centres = {item["label"]: item for item in geometry["cameras"]}
    rows = []
    quadratic_rows = []
    values = []
    for camera in sorted(chunk.cameras, key=lambda item: item.label):
        if not camera.enabled or camera.reference is None:
            continue
        if not all(math.isfinite(float(value)) for value in camera.reference):
            continue
        centre = centres[camera.label]
        reference = geodetic_to_enu(float(camera.reference[0]),
                                    float(camera.reference[1]),
                                    float(camera.reference[2]), origin)
        up = reference[2] - centre["up_m"]
        east = (centre["east_m"] - centroid_east) / DOME_RADIUS_SCALE_M
        north = (centre["north_m"] - centroid_north) / DOME_RADIUS_SCALE_M
        rows.append([east * east + north * north])
        quadratic_rows.append([east, north, east * east, east * north,
                               north * north])
        values.append(up)
    if not values:
        return None
    return {
        "samples": len(values),
        "mean_up_residual_m": sum(values) / len(values),
        "radius_100m_squared": linear_least_squares(
            rows, values, ["radius_100m_squared"]),
        "quadratic": linear_least_squares(
            quadratic_rows, values,
            ["east_100m", "north_100m", "east_100m_squared",
             "east_100m_north_100m", "north_100m_squared"]),
    }


def gcp_horizontal(gcps, origin):
    """The survey points' horizontal ENU positions (held-out point included)."""
    points = []
    for gcp in gcps:
        enu = geodetic_to_enu(gcp["lon"], gcp["lat"], gcp["height"], origin)
        points.append({"label": gcp["label"], "east_m": enu[0],
                       "north_m": enu[1]})
    return points


def check_point_dome(chunk, gcps, check_label, error, geometry):
    """One check point's error and its position relative to the block centre."""
    origin = geometry["origin"]
    gcp = next(gcp for gcp in gcps if gcp["label"] == check_label)
    reference = geodetic_to_enu(gcp["lon"], gcp["lat"], gcp["height"], origin)
    markers = {marker.label: marker for marker in chunk.markers}
    position = markers[check_label].position
    if position is None:
        raise ValueError("check point has no computed position: " + check_label)
    adjusted = enu_of_local(chunk, position, origin)
    # The radius is taken from the surveyed reference, not the adjusted point,
    # so the geometry of the fit does not depend on the solution being tested.
    east = reference[0] - geometry["centroid_east_m"]
    north = reference[1] - geometry["centroid_north_m"]
    return {
        "label": check_label,
        "horizontal_m": error["horizontal"],
        "vertical_m": error["vertical"],
        "vertical_enu_m": adjusted[2] - reference[2],
        "east_from_centroid_m": east,
        "north_from_centroid_m": north,
        "radius_m": math.hypot(east, north),
        "radius_squared_m2": east * east + north * north,
        "reference_enu_m": list(reference),
        "adjusted_enu_m": list(adjusted),
    }


def fold_dome(chunk, gcps, check_label, error):
    """The per-fold dome diagnostic record (see the docstrings above)."""
    geometry = block_geometry(chunk)
    origin = geometry["origin"]
    check = check_point_dome(chunk, gcps, check_label, error, geometry)
    gcp_points = gcp_horizontal(gcps, origin)
    gcp_area = convex_hull_area(
        [(item["east_m"], item["north_m"]) for item in gcp_points])
    return {
        "centroid_east_m": geometry["centroid_east_m"],
        "centroid_north_m": geometry["centroid_north_m"],
        "block_radius_m": geometry["radius_m"],
        "block_hull_area_m2": geometry["hull_area_m2"],
        "gcp_hull_area_m2": gcp_area,
        "gcp_block_area_ratio": (gcp_area / geometry["hull_area_m2"]
                                 if geometry["hull_area_m2"] > 0.0 else None),
        "check_point": check,
        "camera": camera_dome_fit(chunk, geometry),
    }


def calibration_records(chunk):
    """Solved coefficients, standard errors and f-to-distortion correlations.

    The BA stores a 13x13 posterior covariance per sensor (all-zero rows/columns
    for coefficients the solve held fixed), so a correlation is only reported
    when both variances are positive. The standard errors are reported even
    when the correlation is not, because a fixed term's zero variance is itself
    information about the model that was solved.
    """
    records = []
    for sensor in chunk.sensors:
        calibration = sensor.calibration
        record = {"sensor": sensor.label}
        for name in CALIBRATION_PARAM_NAMES:
            record[name] = float(getattr(calibration, name))
        covariance = calibration.covariance_matrix
        params = calibration.covariance_params
        if covariance and params:
            index = {name: position for position, name in enumerate(params)}
            record["covariance_params"] = list(params)
            record["std_errors"] = {
                name: math.sqrt(max(covariance[index[name]][index[name]], 0.0))
                for name in CALIBRATION_PARAM_NAMES if name in index}
            correlations = {}
            for other in ("k1", "k2", "k3", "cx", "cy"):
                i = index.get("f")
                j = index.get(other)
                if i is None or j is None:
                    correlations[other] = None
                    continue
                variance = covariance[i][i] * covariance[j][j]
                correlations[other] = (covariance[i][j] / math.sqrt(variance)
                                       if variance > 0.0 else None)
            record["correlations_f"] = correlations
        records.append(record)
    return records


def summarise_dome(measured):
    """Aggregate the per-fold dome records into the checks a reader needs.

    The check-point fit uses the five leave-one-out points (one per fold); the
    camera fit has ~146 samples per fold, so its coefficient is quoted per fold
    and as a mean t statistic rather than pooled (folds share one alignment and
    are not independent replicates).
    """
    records = [fold["dome"] for fold in measured if "dome" in fold]
    if not records:
        return None
    checks = [record["check_point"] for record in records]
    # The fit uses the ENU vertical difference so it is independent of whether
    # the chunk-local z axis happens to be up; the per-fold reported error keeps
    # the protocol's existing local-frame definition.
    values = [item["vertical_enu_m"] for item in checks]
    radial_rows = [[(item["radius_m"] / DOME_RADIUS_SCALE_M) ** 2]
                   for item in checks]
    linear_rows = [[item["east_from_centroid_m"] / DOME_RADIUS_SCALE_M,
                    item["north_from_centroid_m"] / DOME_RADIUS_SCALE_M,
                    (item["radius_m"] / DOME_RADIUS_SCALE_M) ** 2]
                   for item in checks]
    quadratic_rows = [[
        item["east_from_centroid_m"] / DOME_RADIUS_SCALE_M,
        item["north_from_centroid_m"] / DOME_RADIUS_SCALE_M,
        (item["east_from_centroid_m"] / DOME_RADIUS_SCALE_M) ** 2,
        (item["east_from_centroid_m"] / DOME_RADIUS_SCALE_M) *
        (item["north_from_centroid_m"] / DOME_RADIUS_SCALE_M),
        (item["north_from_centroid_m"] / DOME_RADIUS_SCALE_M) ** 2]
        for item in checks]
    gcp_area = sum(record["gcp_hull_area_m2"] for record in records) / len(records)
    block_area = (sum(record["block_hull_area_m2"] for record in records) /
                  len(records))
    camera_records = [record["camera"] for record in records
                      if record.get("camera")]
    per_fold = []
    for fold, record in zip(
            [fold for fold in measured if "dome" in fold], records):
        camera = record.get("camera")
        if not camera:
            continue
        radial = camera["radius_100m_squared"]
        quadratic = camera["quadratic"]
        per_fold.append({
            "check": fold.get("check"),
            "samples": camera["samples"],
            "mean_up_residual_m": camera["mean_up_residual_m"],
            "radial": radial,
            "quadratic": quadratic,
        })
    camera_mean = None
    if per_fold:
        coefficients = []
        t_values = []
        for record in per_fold:
            item = record["radial"].get("by_name", {}).get(
                "radius_100m_squared", {})
            if item.get("coefficient") is not None:
                coefficients.append(item["coefficient"])
            if item.get("t") is not None:
                t_values.append(item["t"])
        camera_mean = {
            "folds": len(per_fold),
            "coefficient_mean": (sum(coefficients) / len(coefficients)
                                 if coefficients else None),
            "t_mean": (sum(t_values) / len(t_values) if t_values else None),
            "t_min": min(t_values) if t_values else None,
            "t_max": max(t_values) if t_values else None,
        }
    return {
        "check_points": checks,
        "check_fit_radial": linear_least_squares(
            radial_rows, values, ["radius_100m_squared"]),
        "check_fit_linear_radial": linear_least_squares(
            linear_rows, values,
            ["east_100m", "north_100m", "radius_100m_squared"]),
        "check_fit_quadratic": linear_least_squares(
            quadratic_rows, values,
            ["east_100m", "north_100m", "east_100m_squared",
             "east_100m_north_100m", "north_100m_squared"]),
        "gcp_hull_area_m2": gcp_area,
        "block_hull_area_m2": block_area,
        "gcp_block_area_ratio": (gcp_area / block_area) if block_area else None,
        "camera_folds": per_fold,
        "camera_fit_radial_mean": camera_mean,
    }


def summarise_calibration(measured):
    """Per-sensor coefficient and correlation spread across the folds."""
    by_sensor = {}
    for fold in measured:
        for record in fold.get("calibration") or []:
            by_sensor.setdefault(record["sensor"], []).append(record)
    summary = {}
    for sensor, records in by_sensor.items():
        entry = {"folds": len(records)}
        for name in CALIBRATION_PARAM_NAMES:
            values = [record[name] for record in records if name in record]
            if values:
                entry[name] = {"mean": sum(values) / len(values),
                               "min": min(values), "max": max(values)}
        correlations = {}
        for other in ("k1", "k2", "k3", "cx", "cy"):
            values = [record.get("correlations_f", {}).get(other)
                      for record in records
                      if record.get("correlations_f", {}).get(other) is not None]
            correlations[other] = (
                {"mean": sum(values) / len(values), "min": min(values),
                 "max": max(values)} if values else None)
        entry["correlations_f"] = correlations
        summary[sensor] = entry
    return summary


def camera_prior_outliers(chunk, antenna_offset_m=None,
                          threshold=OUTLIER_SIGMA_THRESHOLD):
    """Cameras whose prior residual exceeds ``threshold`` of their own sigma.

    The residual is the same stored-reference-minus-adjusted-centre vector the
    consistency report aggregates, whitened per axis by that camera's declared
    accuracy. A per-image RTK weight near 1.5 cm turns a 0.2 m vertical datum
    disagreement into a ten-sigma residual on every camera, while a uniform
    5 cm block weight would see four sigma: naming the cameras distinguishes
    the two. The GNSS fix status travels with each entry, so an outlier can be
    checked against the solution it came from. Report only; no weight changes.
    """
    transform = chunk.transform
    block = [[float(transform[r][c]) for c in range(3)] for r in range(3)]
    chunk_scale = abs(matrix_determinant(block)) ** (1.0 / 3.0)
    chunk_rotation = None
    if antenna_offset_m is not None:
        if not math.isfinite(chunk_scale) or chunk_scale <= 0.0:
            raise ValueError("chunk transform has no invertible rotation")
        chunk_rotation = [[block[r][c] / chunk_scale for c in range(3)]
                          for r in range(3)]
    samples = []
    for camera in chunk.cameras:
        reference = camera.reference
        if not camera.enabled or reference is None or not all(
                math.isfinite(float(value)) for value in reference):
            continue
        local = camera.center
        ecef = [sum(transform[r][c] * local[c] for c in range(3)) + transform[r][3]
                for r in range(3)]
        if antenna_offset_m is not None:
            pose = camera.pose
            pose_rotation = orthonormal_rotation(
                [[pose[r][c] for c in range(3)] for r in range(3)])
            lever_local = [
                sum(pose_rotation[r][c] * antenna_offset_m[r] for r in range(3))
                for c in range(3)]
            lever_ecef = [
                sum(chunk_rotation[r][c] * lever_local[c] for c in range(3))
                for r in range(3)]
            ecef = [ecef[r] + lever_ecef[r] for r in range(3)]
        enu = ecef_to_enu(ecef, (reference[0], reference[1], reference[2]))
        accuracy = camera.reference_accuracy
        samples.append({"label": camera.label, "reference": reference,
                        "east": enu[0], "north": enu[1], "up": enu[2],
                        "sigma": (tuple(float(v) for v in accuracy)
                                  if accuracy is not None else None),
                        "rtk_flag": camera.reference_rtk_flag})
    if not samples:
        return {"threshold": threshold, "cameras": 0, "flagged_count": 0,
                "flagged": [], "fix_status_counts": {}}

    # Project onto the local flight direction, exactly as camera_consistency
    # does, so a flagged camera's along-track residual is available too.
    ordered = sorted(samples, key=lambda item: item["label"])
    origin = ordered[0]["reference"]
    track = {item["label"]: geodetic_to_enu(
        item["reference"][0], item["reference"][1], item["reference"][2], origin)
        for item in ordered}
    for index, item in enumerate(ordered):
        previous = ordered[max(0, index - 1)]["label"]
        following = ordered[min(len(ordered) - 1, index + 1)]["label"]
        dx = track[following][0] - track[previous][0]
        dy = track[following][1] - track[previous][1]
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            continue
        hx, hy = dx / norm, dy / norm
        item["along"] = item["east"] * hx + item["north"] * hy
        item["cross"] = item["east"] * -hy + item["north"] * hx

    flagged = []
    for item in ordered:
        sigma = item["sigma"]
        if sigma is None or not all(math.isfinite(s) and s > 0.0 for s in sigma):
            continue
        values = (item["east"], item["north"], item["up"])
        z_scores = [value / sigma[axis] for axis, value in enumerate(values)]
        worst = max(abs(value) for value in z_scores)
        if worst <= threshold:
            continue
        flagged.append({
            "label": item["label"],
            "east_m": item["east"], "north_m": item["north"],
            "up_m": item["up"],
            "sigma_m": list(sigma),
            "z_east": z_scores[0], "z_north": z_scores[1], "z_up": z_scores[2],
            "max_abs_z": worst,
            "rtk_flag": item["rtk_flag"],
            "along_track_m": item.get("along"),
            "cross_track_m": item.get("cross"),
        })
    status_counts = {}
    for entry in flagged:
        key = "none" if entry["rtk_flag"] is None else str(entry["rtk_flag"])
        status_counts[key] = status_counts.get(key, 0) + 1
    flagged.sort(key=lambda entry: entry["max_abs_z"], reverse=True)
    return {"threshold": threshold, "cameras": len(samples),
            "flagged_count": len(flagged),
            "labels": [entry["label"] for entry in flagged],
            "flagged": flagged[:OUTLIER_DETAIL_LIMIT],
            "detail_truncated": len(flagged) > OUTLIER_DETAIL_LIMIT,
            "fix_status_counts": status_counts}


def summarise_outliers(records):
    """Collapse per-fold outlier reports into the variant's summary.

    ``unique_labels`` is the union over folds, ``repeat_labels`` those flagged
    in more than one fold (a systematic camera, not a one-fold artefact), and
    ``fix_status_counts`` sums the GNSS solution codes of every flagged camera
    appearance so a correlation with fix status is readable at a glance.
    """
    by_fold = {record.get("check"): list(record.get("labels", []))
               for record in records}
    appearances = {}
    statuses = {}
    for record in records:
        for label in record.get("labels", []):
            appearances[label] = appearances.get(label, 0) + 1
        for key, count in record.get("fix_status_counts", {}).items():
            statuses[key] = statuses.get(key, 0) + count
    repeat = sorted(label for label, count in appearances.items() if count > 1)
    return {
        "folds": len(records),
        "threshold": records[0]["threshold"],
        "camera_folds": sum(record["cameras"] for record in records),
        "flagged_camera_folds": sum(record["flagged_count"] for record in records),
        "unique_labels": sorted(appearances),
        "repeat_labels": repeat,
        "fix_status_counts": statuses,
        "all_fixed": set(statuses) <= {"34", "50"},
        "by_fold": by_fold,
    }


def annotate_outlier_timing(variants, timing):
    """Turn each variant's along-track outliers into an implied latency.

    ``along_track_m / block_speed`` is the time offset that would produce the
    measured along-track lead: near zero means the outliers are not a timing
    artefact, while a consistent, physically sensible value would be. Both the
    flagged cameras and the whole block are reported, so the two can be
    compared. The speed is the block median from the event-mark track.
    """
    speed = timing.get("block_speed_m_s")
    if not speed:
        return
    for data in variants.values():
        outliers = data.get("prior_outliers")
        if not outliers:
            continue
        flagged_latencies = []
        for fold in data.get("folds", []):
            record = fold.get("prior_outliers")
            if not record:
                continue
            for entry in record["flagged"]:
                if entry.get("along_track_m") is not None:
                    flagged_latencies.append(entry["along_track_m"] / speed)
        block_latencies = []
        for fold in data.get("folds", []):
            consistency = fold.get("camera_consistency") or {}
            along = consistency.get("mean_along_track_m")
            if along is not None:
                block_latencies.append(along / speed)
        data["outlier_timing"] = {
            "speed_m_s": speed,
            "flagged_implied_latency_median_s": _median(flagged_latencies),
            "flagged_implied_latency_max_abs_s": (
                max(abs(value) for value in flagged_latencies)
                if flagged_latencies else None),
            "flagged_with_along": len(flagged_latencies),
            "block_implied_latency_median_s": _median(block_latencies),
        }


def matrix_determinant(matrix):
    """Determinant of a 3x3 matrix given as rows."""
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return (a * (e * i - f * h) - b * (d * i - f * g) +
            c * (d * h - e * g))


def orthonormal_rotation(matrix):
    """The rotation part of a 3x3 similarity, with its scale divided out.

    A camera pose is rigid in an ordinary project but an AlignChunks
    similarity stores A = R / s (see the binding's ``center`` getter). Dividing
    by the cube root of |det| recovers R either way, so a direction vector is
    rotated without being scaled.
    """
    block = [[float(matrix[r][c]) for c in range(3)] for r in range(3)]
    scale = abs(matrix_determinant(block)) ** (1.0 / 3.0)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("camera pose has no invertible rotation")
    return [[block[r][c] / scale for c in range(3)] for r in range(3)]


def camera_frame_antenna_offsets(chunk):
    """The lever arm implied by each adjusted camera, in camera-frame metres.

    For every enabled camera with a finite reference, the stored reference is
    the GNSS antenna position A and the adjusted pose gives the projection
    centre C. With ``chunk.transform`` E = s R L + t mapping local to ECEF,

        offset_camera = R_pose * R^T * (A - C)

    because ``R^T (A - C) / s`` is the local-frame displacement and multiplying
    by s cancels (``R_pose`` maps local directions to the camera). The result
    is exactly what ``Antenna.offset_m`` expects: antenna relative to the
    projection centre, x right, y down, z forward. A nadir camera with the
    antenna above the lens therefore has a negative z.

    Only cameras that took part in the fit are included. Returns None when no
    camera carries a usable reference; otherwise the mean vector, the per-axis
    spread across cameras and the sample count.
    """
    transform = chunk.transform
    block = [[float(transform[r][c]) for c in range(3)] for r in range(3)]
    scale = abs(matrix_determinant(block)) ** (1.0 / 3.0)
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    rotation = [[block[r][c] / scale for c in range(3)] for r in range(3)]
    shift = [float(transform[r][3]) for r in range(3)]
    samples = []
    for camera in chunk.cameras:
        reference = camera.reference
        if not camera.enabled or reference is None:
            continue
        if not all(math.isfinite(float(value)) for value in reference):
            continue
        pose = camera.pose
        pose_rotation = orthonormal_rotation(
            [[pose[r][c] for c in range(3)] for r in range(3)])
        centre = camera.center
        centre_ecef = [
            sum(block[r][c] * centre[c] for c in range(3)) + shift[r]
            for r in range(3)]
        antenna_ecef = geodetic_to_ecef(*reference)
        delta = [antenna_ecef[i] - centre_ecef[i] for i in range(3)]
        # ECEF -> local is the transpose of the local -> ECEF rotation.
        local = [sum(rotation[r][c] * delta[r] for r in range(3))
                 for c in range(3)]
        samples.append([sum(pose_rotation[r][c] * local[c] for c in range(3))
                        for r in range(3)])
    if not samples:
        return None
    mean = [sum(sample[axis] for sample in samples) / len(samples)
            for axis in range(3)]
    spread = [
        math.sqrt(sum((sample[axis] - mean[axis]) ** 2 for sample in samples) /
                  len(samples))
        for axis in range(3)]
    return {"cameras": len(samples), "mean_m": mean, "spread_m": spread}


def gcp_frame_origin(gcps, check_index=None):
    """Mean surveyed position of the control GCPs, as an ENU frame origin.

    The frame origin only fixes the axes the translation is reported in; the
    translation vector itself does not depend on it. The held-out point is
    excluded so every per-fold diagnostic is computed from the fold's control
    points alone.
    """
    rows = [gcp for index, gcp in enumerate(gcps) if index != check_index]
    if not rows:
        rows = list(gcps)
    return (sum(gcp["lon"] for gcp in rows) / len(rows),
            sum(gcp["lat"] for gcp in rows) / len(rows),
            sum(gcp["height"] for gcp in rows) / len(rows))


def solve_normal_equations(rows, values, parameters):
    """Least-squares parameters from a design matrix and observations."""
    normal = [[sum(row[a] * row[b] for row in rows) for b in range(parameters)]
              for a in range(parameters)]
    rhs = [sum(row[a] * value for row, value in zip(rows, values))
           for a in range(parameters)]
    inverse = invert_matrix(normal)
    return [sum(inverse[a][b] * rhs[b] for b in range(parameters))
            for a in range(parameters)]


def camera_reference_frame_fit(chunk, origin):
    """Translation and Helmert fit between adjusted centres and stored references.

    Both coordinate sets are expressed in the local east/north/up frame of
    ``origin`` (the fold's surveyed control points), so the translation is
    directly readable as E/N/U in the frame the GCPs define. This is the test
    for a reference-frame mismatch: if the camera RTK positions are in a
    different realization (or epoch) of the geodetic system, the adjusted
    centres --- placed by the GCP datum --- sit a constant distance from them,
    and a translation of decimetres or more with a small residual after the fit
    is that offset.

    The translation-only fit removes a constant displacement and reports what
    remains (mount, network distortion). The 7-parameter Helmert additionally
    allows a rotation and a scale (`a = s R c + t`); with 146 cameras it is well
    determined, and the rotation/scale tell a frame rotation from a pure
    translation. Only cameras that took part in the adjustment are used. The
    fit consumes no check point: on a GCP-only fold it is computed from the
    control GCPs' adjustment alone.

    Rotation components are the small-angle vector about the frame's E/N/U
    axes (positive right-handed about each axis), in arcsec; scale is parts per
    million (a frame realization difference has scale ~0).
    """
    samples = []
    for camera in chunk.cameras:
        reference = camera.reference
        if not camera.enabled or reference is None:
            continue
        if not all(math.isfinite(float(value)) for value in reference):
            continue
        local = camera.center
        transform = chunk.transform
        centre_ecef = [
            sum(float(transform[r][c]) * local[c] for c in range(3))
            + float(transform[r][3]) for r in range(3)]
        centre_enu = ecef_to_enu(centre_ecef, origin)
        reference_enu = geodetic_to_enu(
            float(reference[0]), float(reference[1]), float(reference[2]),
            origin)
        samples.append({"label": camera.label, "reference": reference_enu,
                        "centre": centre_enu})
    if len(samples) < 7:
        return {"cameras": len(samples), "status": "underdetermined",
                "note": "fewer aligned cameras than the 7 Helmert parameters"}

    count = len(samples)
    differences = [[sample["reference"][axis] - sample["centre"][axis]
                    for axis in range(3)] for sample in samples]
    translation = [sum(item[axis] for item in differences) / count
                   for axis in range(3)]
    translation_residuals = [
        [item[axis] - translation[axis] for axis in range(3)]
        for item in differences]
    translation_rms_enu = [
        math.sqrt(sum(item[axis] ** 2 for item in translation_residuals)
                  / count) for axis in range(3)]
    translation_rms = math.sqrt(sum(
        sum(value * value for value in item)
        for item in translation_residuals) / count)

    centre_mean = [sum(sample["centre"][axis] for sample in samples) / count
                   for axis in range(3)]
    rows, values = [], []
    for index, sample in enumerate(samples):
        delta = differences[index]
        x, y, z = [sample["centre"][axis] - centre_mean[axis]
                   for axis in range(3)]
        # Unknowns: t_x, t_y, t_z, theta_x, theta_y, theta_z, scale (1 + mu).
        rows.append([1.0, 0.0, 0.0, 0.0, z, -y, x])
        values.append(delta[0])
        rows.append([0.0, 1.0, 0.0, -z, 0.0, x, y])
        values.append(delta[1])
        rows.append([0.0, 0.0, 1.0, y, -x, 0.0, z])
        values.append(delta[2])
    try:
        parameters = solve_normal_equations(rows, values, 7)
    except ValueError as error:
        return {"cameras": count, "status": "singular", "note": str(error)}
    helmert_residuals = []
    for index, sample in enumerate(samples):
        x, y, z = [sample["centre"][axis] - centre_mean[axis]
                   for axis in range(3)]
        theta = parameters[3:6]
        predicted = [
            parameters[0] + theta[1] * z - theta[2] * y + parameters[6] * x,
            parameters[1] - theta[0] * z + theta[2] * x + parameters[6] * y,
            parameters[2] + theta[0] * y - theta[1] * x + parameters[6] * z]
        helmert_residuals.append(
            [differences[index][axis] - predicted[axis] for axis in range(3)])
    helmert_rms_enu = [
        math.sqrt(sum(item[axis] ** 2 for item in helmert_residuals) / count)
        for axis in range(3)]
    helmert_rms = math.sqrt(sum(
        sum(value * value for value in item)
        for item in helmert_residuals) / count)
    rotation_arcsec = [value * ARCSEC_PER_RAD for value in parameters[3:6]]
    return {
        "cameras": count,
        "status": "ok",
        "frame_origin_lon_lat_height": list(origin),
        # Translation-only fit: the offset a frame mismatch produces.
        "translation_m": translation,
        "translation_horizontal_m": math.hypot(translation[0], translation[1]),
        "translation_residual_rms_enu_m": translation_rms_enu,
        "translation_residual_rms_m": translation_rms,
        # 7-parameter Helmert: translation at the camera centroid, rotation and
        # scale on top of it.
        "helmert_translation_m": parameters[0:3],
        "rotation_arcsec": rotation_arcsec,
        "rotation_total_arcsec": math.sqrt(
            sum(value * value for value in rotation_arcsec)),
        "scale_ppm": parameters[6] * 1e6,
        "helmert_residual_rms_enu_m": helmert_rms_enu,
        "helmert_residual_rms_m": helmert_rms,
    }


def decimal_year_from_gps(week, time_s):
    """Decimal year of a GPS week / time-of-week stamp, in UTC."""
    moment = (datetime.datetime(1980, 1, 6)
              + datetime.timedelta(weeks=int(week),
                                   seconds=float(time_s) - GPS_UTC_LEAP_SECONDS))
    start = datetime.datetime(moment.year, 1, 1)
    days = 366.0 if calendar.isleap(moment.year) else 365.0
    return moment.year + (moment - start).total_seconds() / 86400.0 / days


def capture_epoch_from_images(images):
    """Decimal year of the first image's EXIF capture time, or None.

    Pillow is in the benchmark environment; a manual run without it simply
    cannot derive an epoch and must pass ``--capture-epoch``.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    for path in images[:8]:
        try:
            with Image.open(path) as image:
                exif = image.getexif()
        except Exception:  # noqa: BLE001 - an unreadable image is not fatal
            continue
        # 36867 DateTimeOriginal, 36868 DateTimeDigitized, 306 DateTime.
        for tag in (36867, 36868, 306):
            value = exif.get(tag)
            if not value:
                continue
            try:
                moment = datetime.datetime.strptime(
                    str(value).strip(), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                continue
            start = datetime.datetime(moment.year, 1, 1)
            days = 366.0 if calendar.isleap(moment.year) else 365.0
            return moment.year + (moment - start).total_seconds() / 86400.0 / days
    return None


def make_published_frame_transform(mode, epoch):
    """Build the published time-dependent transform for ``mode``.

    Returns ``(transform, description)``. ``transform`` takes
    ``(lon, lat, height)`` in degrees and metres and returns the converted
    triple. The epoch is the coordinate epoch (decimal year) passed to PROJ as
    the point's time coordinate, so the plate motion between the static ETRS89
    realization and the ITRF realization at capture is applied, not a nominal
    constant. PROJ chooses the ETRF realization from its own database; the
    description it reports is returned for the report.
    """
    if mode not in FRAME_CRS_EPSG:
        raise ValueError("unknown frame correction " + str(mode))
    if epoch is None:
        raise ValueError(
            "frame correction {} needs a capture epoch".format(mode))
    try:
        from pyproj import CRS
        from pyproj.transformer import TransformerGroup
    except ImportError as error:  # pragma: no cover - environment error
        raise ValueError(
            "--frame-correction needs pyproj in the protocol environment") from error
    source, target = FRAME_CRS_EPSG[mode]
    # The group lists PROJ's candidate operations; the first is the global
    # time-dependent ETRF/ITRF realization pair (accuracy stated in its
    # description). A plain Transformer's ``description`` is not populated for
    # a CRS-to-CRS pair, so the group is also what gives the report a citable
    # operation string.
    group = TransformerGroup(
        CRS.from_epsg(source), CRS.from_epsg(target), always_xy=True)
    if not group.transformers:
        raise ValueError(
            "PROJ knows no {} operation between EPSG:{} and EPSG:{}".format(
                mode, source, target))
    operation = group.transformers[0]
    description = operation.description

    def transform(lon, lat, height):
        out = operation.transform(lon, lat, height, tt=float(epoch))
        return (float(out[0]), float(out[1]), float(out[2]))

    return transform, description


def published_frame_offsets(gcps, epoch):
    """The published transform's displacement at this site, in E/N/U.

    Evaluated on the surveyed points so the report states the expected size and
    direction of each hypothesis at the capture epoch and the block's own
    location. A frame offset should match this vector; a mount, a timing lag or
    a lens error has a different signature.
    """
    origin = gcp_frame_origin(gcps)
    result = {}
    for mode in FRAME_CRS_EPSG:
        try:
            transform, description = make_published_frame_transform(mode, epoch)
        except Exception as error:  # noqa: BLE001 - report, never fatal
            result[mode] = {"error": str(error)}
            continue
        shifts = []
        for gcp in gcps:
            converted = transform(gcp["lon"], gcp["lat"], gcp["height"])
            before = geodetic_to_enu(gcp["lon"], gcp["lat"], gcp["height"], origin)
            after = geodetic_to_enu(converted[0], converted[1], converted[2],
                                    origin)
            shifts.append([after[axis] - before[axis] for axis in range(3)])
        mean = [sum(item[axis] for item in shifts) / len(shifts)
                for axis in range(3)]
        result[mode] = {
            "description": description,
            "epoch": epoch,
            "mean_enu_m": mean,
            "horizontal_m": math.hypot(mean[0], mean[1]),
            "max_horizontal_m": max(math.hypot(item[0], item[1])
                                    for item in shifts),
        }
    return result


def parse_antenna_offset(text):
    """A ``x,y,z`` antenna lever-arm string in metres, or None when empty."""
    if not text:
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--antenna-offset needs three comma-separated metres")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as error:
        raise ValueError(
            "--antenna-offset needs three numbers in metres") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("--antenna-offset values must be finite")
    return values


def _meta_float(chunk, key):
    """A chunk meta value as a float, or None when absent or not a number."""
    try:
        value = float(chunk.meta.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def sensor_focals(chunk):
    """The solved focal length of each sensor, in pixels."""
    return [{"sensor": sensor.label, "f": sensor.calibration.f}
            for sensor in chunk.sensors]


def focal_changes(before, after):
    """Per-sensor focal before/after the adjustment and its relative change."""
    changes = []
    for old, new in zip(before, after):
        if old["sensor"] != new["sensor"]:
            raise ValueError("sensor list changed during OptimizeCameras")
        relative = ((new["f"] - old["f"]) / old["f"]) if old["f"] else None
        changes.append({"sensor": new["sensor"], "before_px": old["f"],
                        "after_px": new["f"], "relative_change": relative})
    return changes


def refine_marker_changes(chunk, before):
    """How many projections RefineMarkers moved or removed, and the mean shift."""
    after = {(marker.label, projection.camera_id): projection.pixel
             for marker in chunk.markers for projection in marker.projections}
    moved = 0
    removed = 0
    shifts = []
    for key, pixel in before.items():
        if key not in after:
            removed += 1
            continue
        shift = math.hypot(after[key][0] - pixel[0], after[key][1] - pixel[1])
        shifts.append(shift)
        if shift > 1e-6:
            moved += 1
    return {"moved": moved, "removed": removed,
            "kept": len(shifts),
            "mean_shift_px": (sum(shifts) / len(shifts)) if shifts else None}


def optimize_options(options):
    """The OptimizeCameras kwargs a diagnostic variant asks for."""
    kwargs = {"use_gcps": True}
    if options.get("adaptive_fitting"):
        # Adaptive mode reads the fit_* flags as permission, not instruction:
        # permit the full extended model and let the likelihood-ratio selection
        # keep only the terms whose gain pays for their degrees of freedom.
        kwargs["adaptive_fitting"] = True
        kwargs.update(fit_k4=True, fit_b1=True, fit_b2=True,
                      fit_p3=True, fit_p4=True)
    if options.get("extended_model"):
        # The default fit is f, cx, cy, k1, k2, p1, p2; free the higher-order
        # Brown/Fraser terms the defaults hold fixed.
        kwargs.update(fit_k4=True, fit_b1=True, fit_b2=True,
                      fit_p3=True, fit_p4=True)
    # A model-choice probe names the coefficients it wants to move (or hold).
    # OptimizeCameras intersects its fit_* flags with the sensor's fixed_params,
    # so a False here holds the term even when a fixed-parameter model is not
    # in play.
    for name, value in (options.get("fit") or {}).items():
        kwargs["fit_" + name] = bool(value)
    return kwargs


def run_fold(args, gcps, check_index, camera_refs, align_report, options=None):
    """One leave-one-out fold: reload, project, import, fit, measure the check.

    ``options`` selects a diagnostic variant: a projection pixel shift, a marker
    projection sigma, extra calibration terms, marker refinement, a loosened
    camera accuracy, or a GNSS antenna lever arm (``antenna_offset_m`` in the
    camera frame, metres). The check point's reference is imported as a
    check_point and never constrains the fit; a variant that estimates the
    lever arm from this fold therefore sees only control GCPs.
    """
    options = options or {}
    doc = alphashape.load_project(align_report["project"])
    chunk = doc.active_chunk

    # Control and check markers take the survey's stated accuracy: the import
    # rows carry no accuracy column, so the chunk setting is the one applied.
    chunk.marker_location_accuracy = args.marker_accuracy
    if "marker_projection_accuracy" in options:
        chunk.marker_projection_accuracy = options["marker_projection_accuracy"]
    shift = options.get("projection_shift_px", 0.0)
    cameras = {camera.label: camera for camera in chunk.cameras}
    for gcp in gcps:
        chunk.addMarker(label=gcp["label"])
    for gcp in gcps:
        for observation in gcp["observations"]:
            stem = os.path.splitext(observation["image"])[0]
            target = stem if stem in cameras else (
                observation["image"] if observation["image"] in cameras else None)
            if target is None:
                continue
            chunk.setMarkerProjection(gcp["label"], target,
                                      observation["px"] + shift,
                                      observation["py"] + shift)

    refine = None
    if options.get("refine_markers"):
        before = {(marker.label, projection.camera_id): projection.pixel
                  for marker in chunk.markers for projection in marker.projections}
        try:
            chunk.refineMarkers(markers=[gcp["label"] for gcp in gcps])
            refine = refine_marker_changes(chunk, before)
        except Exception as error:  # noqa: BLE001 - recorded, never fatal
            refine = {"error": str(error)}

    accuracy = options.get("camera_accuracy", args.camera_accuracy)
    accuracy_scale = options.get("camera_accuracy_scale", 1.0)
    along_shift = options.get("along_shift_m", 0.0)
    # Reference-frame correction. "published" converts every stored camera
    # position with the time-dependent PROJ operation at the capture epoch;
    # "empirical" removes the constant translation measured on this fold's
    # control-only GCP adjustment (run_variant supplies it). Both are applied
    # before the datum refit and the bundle adjustment, so the camera priors and
    # the GCP datum share one frame. Which references actually moved is visible
    # in camera_consistency below.
    frame_mode = options.get("frame_correction")
    frame_transform = None
    frame_description = None
    if frame_mode in FRAME_CRS_EPSG:
        frame_transform, frame_description = make_published_frame_transform(
            frame_mode, options.get("capture_epoch"))
    frame_shift = options.get("frame_shift_enu_m")
    # The "fit on ground control, then keep that calibration" workflow: this
    # fold's control-only GCP adjustment supplies the lens, and the camera-RTK
    # adjustment that follows may not move it. The calibration source uses only
    # the control markers (the held-out point is imported as a check point and
    # never constrains either fit), so the check never leaks into the lens.
    calibration_from_control = bool(options.get("fixed_calibration_from_control"))
    calibration_source = None
    with tempfile.TemporaryDirectory() as directory:
        control_rows = [(gcps[index]["label"], gcps[index]["lon"],
                         gcps[index]["lat"], gcps[index]["height"])
                        for index in range(len(gcps)) if index != check_index]
        check_rows = [(gcps[check_index]["label"], gcps[check_index]["lon"],
                       gcps[check_index]["lat"], gcps[check_index]["height"])]
        control_path = os.path.join(directory, "control.csv")
        check_path = os.path.join(directory, "check.csv")
        write_reference_csv(control_path, control_rows)
        write_reference_csv(check_path, check_rows)
        if calibration_from_control:
            # 1. Datum and lens from the control GCPs alone: cameras carry no
            #    reference in this pass, so the ground control is the only
            #    georeferencing input.
            for camera in chunk.cameras:
                camera.reference_enabled = False
            chunk.importReference(path=control_path, marker_type="gcp")
            chunk.importReference(path=check_path, marker_type="check_point")
            chunk.updateTransform()
            source_options = ({"adaptive_fitting": True}
                              if options.get("calibration_source_adaptive", True)
                              else {})
            chunk.optimizeCameras(**optimize_options(source_options))
            calibration_source = calibration_records(chunk)
            for sensor_index in range(len(chunk.sensors)):
                chunk.set_sensor_fixed_params(
                    sensor_index, list(CALIBRATION_PARAM_NAMES))
            # 2. Put the camera references back and adjust again; every
            #    intrinsic coefficient is held, so the lens cannot trade
            #    against the heights.
            import_camera_references_for_block(
                chunk, accuracy, directory, along_shift, accuracy_scale)
        else:
            if camera_refs:
                # Per-image mode without a shift needs no re-import: AddPhotos
                # with load_xmp_accuracy already stored each camera's own
                # accuracy (and the non-fixed rule applied there). Re-importing
                # the same rows would only round positions through the CSV.
                if (accuracy != CAMERA_ACCURACY_PER_IMAGE or along_shift
                        or accuracy_scale != 1.0
                        or frame_transform is not None or frame_shift is not None):
                    import_camera_references_for_block(
                        chunk, accuracy, directory, along_shift, accuracy_scale,
                        frame_shift_enu_m=frame_shift,
                        frame_transform=frame_transform)
            else:
                for camera in chunk.cameras:
                    camera.reference_enabled = False
            chunk.importReference(path=control_path, marker_type="gcp")
            chunk.importReference(path=check_path, marker_type="check_point")

    # The antenna mount must be in place before updateTransform: both the datum
    # refit and the bundle adjustment consume the lever arm, and the two must
    # agree. The binding takes an Eigen::Vector3d, which pybind11 converts from
    # a numpy array (not from a Python list).
    antenna = options.get("antenna_offset_m")
    if antenna is not None:
        for sensor_index in range(len(chunk.sensors)):
            chunk.set_sensor_antenna_offset(
                sensor_index, antenna[0], antenna[1], antenna[2])

    focal_before = sensor_focals(chunk)
    # Refit the datum after the control references are imported, exactly as for
    # every other variant; with a mount it places the projection centres, not
    # the antenna phase centres, on the camera GNSS references.
    chunk.updateTransform()
    chunk.optimizeCameras(**optimize_options(options))
    focal_after = sensor_focals(chunk)

    labels = [gcps[check_index]["label"]]
    targets = reference_local_targets(chunk, labels)
    errors = check_point_errors(chunk, labels, targets)[0]
    fold = {"check": labels[0],
            "control": [gcps[index]["label"]
                        for index in range(len(gcps)) if index != check_index],
            "horizontal_m": errors["horizontal"],
            "vertical_m": errors["vertical"],
            "total_m": errors["total"],
            "control_rms_m": _meta_float(chunk, "optimize_cameras.control_rms_m"),
            "control_rms_xyz_m": _meta_float(
                chunk, "optimize_cameras.control_rms_xyz_m"),
            "focal": focal_changes(focal_before, focal_after),
            "camera_consistency": camera_consistency(
                chunk, options.get("antenna_offset_m")),
            # What this fold's own adjusted cameras imply for the lever arm,
            # whether or not one was applied. On the GCP-only variant this is
            # the leak-free per-fold estimate the antenna variants reuse.
            "antenna_offset": camera_frame_antenna_offsets(chunk)}
    if camera_refs:
        fold["camera_accuracy"] = camera_accuracy_summary(chunk)
        fold["prior_outliers"] = camera_prior_outliers(
            chunk, options.get("antenna_offset_m"),
            options.get("outlier_threshold", OUTLIER_SIGMA_THRESHOLD))
    if antenna is not None:
        fold["antenna_applied_m"] = list(antenna)
    if frame_mode in FRAME_CRS_EPSG:
        fold["frame_applied"] = {"mode": frame_mode,
                                 "epoch": options.get("capture_epoch"),
                                 "description": frame_description}
    elif frame_shift is not None:
        fold["frame_applied"] = {"mode": "empirical",
                                 "shift_enu_m": list(frame_shift),
                                 "applied_as": "reference minus shift"}
    if along_shift:
        fold["timing"] = {"along_shift_m": along_shift,
                          "latency_s": options.get("timing_latency_s")}
    if refine is not None:
        fold["refine_markers"] = refine
    if calibration_source is not None:
        fold["calibration_source"] = calibration_source
    if args.diagnostics:
        # Report-only block-doming and calibration probes. They never change
        # the fit, and they are skipped in a plain run so its report stays the
        # one the regression gate has always compared.
        fold["calibration"] = calibration_records(chunk)
        fold["dome"] = fold_dome(chunk, gcps, labels[0], errors)
        if not camera_refs and (args.diagnostics
                                or args.frame_correction == "empirical"):
            # On the GCP-only fold the adjusted centres are placed by the
            # surveyed control points alone, so the fit against the stored RTK
            # positions is exactly the frame test. (On a camera-referenced fold
            # the cameras are pinned to those positions and the fit is
            # trivially zero.) Computed for the empirical correction too, even
            # without --diagnostics, because that variant needs the estimate.
            fold["frame_fit"] = camera_reference_frame_fit(
                chunk, gcp_frame_origin(gcps, check_index))
    return fold


def summarise_camera_consistency(measured):
    """Mean over folds of the per-fold camera-consistency records."""
    per_fold = [fold["camera_consistency"] for fold in measured
                if fold.get("camera_consistency", {}).get("cameras")]
    if not per_fold:
        return None
    keys = [key for key in per_fold[0]
            if key.endswith("_m") and per_fold[0][key] is not None]
    summary = {"folds": len(per_fold), "cameras": per_fold[0]["cameras"]}
    for key in keys:
        values = [item[key] for item in per_fold if item.get(key) is not None]
        if values:
            summary[key] = sum(values) / len(values)
    return summary


def summarise_focal(measured):
    """Per-sensor mean/min/max relative focal change across the folds."""
    by_sensor = {}
    for fold in measured:
        for change in fold.get("focal") or []:
            if change.get("relative_change") is not None:
                by_sensor.setdefault(change["sensor"], []).append(
                    change["relative_change"])
    if not by_sensor:
        return None
    return {sensor: {"folds": len(values),
                     "mean_relative_change": sum(values) / len(values),
                     "min_relative_change": min(values),
                     "max_relative_change": max(values)}
            for sensor, values in by_sensor.items()}


def summarise_antenna(measured):
    """Per-fold camera-frame lever arm implied by the adjusted cameras.

    Reported for every variant: on the camera-reference variants a lever arm
    was already applied, so this is the residual offset; on the GCP-only
    variants it is the unmodelled offset the antenna variants use. A stable
    value across folds means a rigid mount; a large fold-to-fold swing means
    the datum, not the mount, is setting it.
    """
    records = [fold for fold in measured if fold.get("antenna_offset")]
    if not records:
        return None
    per_fold = {}
    for fold in records:
        record = fold["antenna_offset"]
        per_fold[fold["check"]] = {
            "mean_m": record["mean_m"],
            "spread_m": record["spread_m"],
            "cameras": record["cameras"],
        }
    axes = [sum(record["antenna_offset"]["mean_m"][axis]
                for record in records) / len(records) for axis in range(3)]
    applied = [fold["antenna_applied_m"] for fold in measured
               if "antenna_applied_m" in fold]
    summary = {"folds": len(records), "mean_m": axes, "per_fold": per_fold}
    if applied:
        summary["applied_m"] = list(applied[0])
    return summary


def summarise_frame_fit(measured):
    """Per-fold frame fits and their stability across folds.

    The per-fold translation and rotation are kept as measured; the summary
    adds the mean and the min/max range of each axis. A frame offset is a
    consistent value across folds (the four control subsets differ by one
    point), so a tight range around a decimetre translation is the signature,
    while a fold-to-fold swing of the same size as the value means the datum,
    not a frame difference, is setting it.
    """
    records = [(fold["check"], fold["frame_fit"]) for fold in measured
               if fold.get("frame_fit")]
    if not records:
        return None
    summary = {"folds": len(records),
               "per_fold": {label: record for label, record in records}}
    for key, axes in (("translation_m", 3), ("rotation_arcsec", 3),
                      ("helmert_translation_m", 3)):
        values = [record[key] for _, record in records
                  if record.get("status") == "ok" and record.get(key)]
        if not values:
            continue
        summary[key + "_mean"] = [sum(item[axis] for item in values) / len(values)
                                  for axis in range(axes)]
        summary[key + "_min"] = [min(item[axis] for item in values)
                                 for axis in range(axes)]
        summary[key + "_max"] = [max(item[axis] for item in values)
                                 for axis in range(axes)]
    for key in ("translation_residual_rms_m", "helmert_residual_rms_m",
                "translation_horizontal_m", "rotation_total_arcsec",
                "scale_ppm"):
        values = [record[key] for _, record in records
                  if record.get("status") == "ok" and record.get(key) is not None]
        if values:
            summary[key + "_mean"] = sum(values) / len(values)
            summary[key + "_min"] = min(values)
            summary[key + "_max"] = max(values)
    return summary


def run_variant(args, gcps, align_report, camera_refs, options=None,
                catch_all=False, estimates=None, timing=None,
                frame_shifts=None):
    """Run the leave-one-out folds of one variant and summarise them.

    ``estimates`` maps a check label to the camera-frame lever arm measured on
    the SAME fold's GCP-only adjustment (``antenna_offset.mean_m``). An
    ``estimated_antenna`` option applies that fold's value; the held-out point
    never enters it because the GCP-only fit constrains only the other GCPs.

    ``timing`` maps a check label to the along-track lead (metres) of the same
    fold's GCP-only adjustment. An ``estimated_timing`` option removes that
    lead from every camera position (speed x latency, with the latency derived
    from the block speed and the control-only lead), and records the implied
    latency on the fold. Same leak-free construction as the antenna estimate.

    ``frame_shifts`` maps a check label to the E/N/U translation (metres) of
    the same fold's GCP-only frame fit, measured as reference minus adjusted
    centre. An ``estimated_frame_shift`` option removes it from every camera
    reference, which moves the camera RTK positions onto the GCP datum without
    assuming which realization they were recorded in. The held-out point is
    not in the GCP-only fit, so it is leak-free.
    """
    options = options or {}
    if options.get("estimated_antenna") and not estimates:
        raise ValueError(
            "estimated_antenna needs the GCP-only per-fold estimates")
    if options.get("estimated_timing") and not timing:
        raise ValueError(
            "estimated_timing needs the GCP-only per-fold along-track leads")
    if options.get("estimated_frame_shift") and not frame_shifts:
        raise ValueError(
            "estimated_frame_shift needs the GCP-only per-fold frame shifts")
    folds = []
    for check_index in range(len(gcps)):
        fold_options = options
        label = gcps[check_index]["label"]
        if options.get("estimated_antenna"):
            estimate = estimates.get(label)
            if estimate is None:
                raise ValueError(
                    "no GCP-only antenna estimate for check " + label)
            fold_options = dict(options)
            fold_options.pop("estimated_antenna", None)
            fold_options["antenna_offset_m"] = tuple(estimate)
        if options.get("estimated_timing"):
            lead = timing.get(label)
            if lead is None:
                raise ValueError(
                    "no GCP-only along-track lead for check " + label)
            fold_options = dict(fold_options)
            fold_options.pop("estimated_timing", None)
            speed = fold_options.pop("timing_speed_m_s", None)
            fold_options["along_shift_m"] = lead
            if speed:
                fold_options["timing_latency_s"] = lead / speed
        if options.get("estimated_frame_shift"):
            shift = frame_shifts.get(label)
            if shift is None:
                raise ValueError(
                    "no GCP-only frame shift for check " + label)
            fold_options = dict(fold_options)
            fold_options.pop("estimated_frame_shift", None)
            fold_options["frame_shift_enu_m"] = tuple(shift)
        try:
            folds.append(run_fold(args, gcps, check_index, camera_refs,
                                  align_report, fold_options))
        except Exception as error:  # noqa: BLE001 - diagnostics record, never fatal
            if not catch_all and not isinstance(error, ValueError):
                raise
            folds.append({"check": gcps[check_index]["label"],
                          "error": str(error)})
    measured = [fold for fold in folds if "error" not in fold]
    variant = {
        "folds": folds,
        "evaluated": len(measured),
        "rms_horizontal_m": (
            root_mean_square([fold["horizontal_m"] for fold in measured])
            if measured else None),
        "rms_vertical_m": (
            root_mean_square([fold["vertical_m"] for fold in measured])
            if measured else None),
        "rms_total_m": (
            root_mean_square([fold["total_m"] for fold in measured])
            if measured else None),
    }
    if options:
        variant["options"] = {key: value for key, value in options.items()}
    consistency = summarise_camera_consistency(measured)
    if consistency:
        variant["camera_consistency"] = consistency
    focal = summarise_focal(measured)
    if focal:
        variant["focal"] = focal
    controls = [fold["control_rms_m"] for fold in measured
                if fold.get("control_rms_m") is not None]
    if controls:
        variant["control_rms_m"] = sum(controls) / len(controls)
    refined = [fold["refine_markers"] for fold in measured
               if "refine_markers" in fold]
    if refined:
        variant["refine_markers"] = {
            "moved": sum(item.get("moved", 0) for item in refined),
            "removed": sum(item.get("removed", 0) for item in refined),
            "fold_errors": sum(1 for item in refined if "error" in item)}
    antenna = summarise_antenna(measured)
    if antenna:
        variant["antenna_estimates"] = antenna
    frame_fit = summarise_frame_fit(measured)
    if frame_fit:
        variant["frame_fit"] = frame_fit
    dome = summarise_dome(measured)
    if dome:
        variant["dome"] = dome
    calibration = summarise_calibration(measured)
    if calibration:
        variant["calibration"] = calibration
    # The control-only fit a fixed-calibration variant freezes is reported
    # beside the frozen result, so a reader can see the lens that was applied.
    sources = [fold["calibration_source"] for fold in measured
               if "calibration_source" in fold]
    if sources:
        source_summary = summarise_calibration(
            [{"calibration": source} for source in sources])
        if source_summary:
            variant["calibration_source"] = source_summary
    if measured and "camera_accuracy" in measured[0]:
        # The weights do not change between folds, so the first fold's
        # distribution describes the variant.
        variant["camera_accuracy"] = measured[0]["camera_accuracy"]
    # The fold record carries the diagnostics under its own check label; the
    # summary groups by it. (camera_prior_outliers itself has no fold identity.)
    outliers = [dict(fold["prior_outliers"], check=fold["check"])
                for fold in measured if "prior_outliers" in fold]
    if outliers:
        variant["prior_outliers"] = summarise_outliers(outliers)
    timing_folds = [fold["timing"] for fold in measured if "timing" in fold]
    if timing_folds:
        latencies = [item["latency_s"] for item in timing_folds
                     if item.get("latency_s") is not None]
        variant["timing"] = {
            "folds": len(timing_folds),
            "mean_along_shift_m": sum(item["along_shift_m"]
                                      for item in timing_folds) / len(timing_folds),
            "mean_latency_s": (sum(latencies) / len(latencies)
                               if latencies else None),
        }
    return variant


# Extra gcps_only leave-one-out variants that probe the leading explanations
# for the residual (focal/flying-height correlation, calibration model, marker
# conventions, weight balance). Each is report-only: none is ever gated.
DIAGNOSTIC_VARIANTS = (
    ("adaptive_fitting", False, {"adaptive_fitting": True}),
    ("extended_model", False, {"extended_model": True}),
    ("refined_markers", False, {"refine_markers": True}),
    ("px_shift_plus", False, {"projection_shift_px": 0.5}),
    ("px_shift_minus", False, {"projection_shift_px": -0.5}),
    ("marker_sigma_0p25", False, {"marker_projection_accuracy": 0.25}),
    ("marker_sigma_1p0", False, {"marker_projection_accuracy": 1.0}),
    ("camera_rtk_loose", True, {"camera_accuracy": 0.5}),
    # Combinations of the probes above that each moved the check error.
    ("adaptive_px_minus", False,
     {"adaptive_fitting": True, "projection_shift_px": -0.5}),
    ("camera_rtk_adaptive", True, {"adaptive_fitting": True}),
    ("camera_rtk_adaptive_px_minus", True,
     {"adaptive_fitting": True, "projection_shift_px": -0.5}),
    # Antenna lever-arm probes. The GCP-only adjustment puts the camera centres
    # about 0.23 m below the recorded RTK antenna positions, which is an
    # unmodelled mount. The fixed pair applies that measured seed; the
    # estimated pair applies the mean camera-frame offset of the SAME fold's
    # GCP-only adjustment, which uses only the control GCPs (the held-out
    # point never enters it). Each is tried with the default and the adaptive
    # lens model, because the seed value itself comes from the adaptive fit.
    ("antenna_fixed", True,
     {"antenna_offset_m": ANTENNA_FIXED_OFFSET_M}),
    ("antenna_fixed_adaptive", True,
     {"antenna_offset_m": ANTENNA_FIXED_OFFSET_M, "adaptive_fitting": True}),
    ("antenna_estimated", True, {"estimated_antenna": True}),
    ("antenna_estimated_adaptive", True,
     {"estimated_antenna": True, "adaptive_fitting": True}),
    # The best measured combination before the lever arm, with the estimated
    # mount and the adaptive model.
    ("antenna_estimated_adaptive_px_minus", True,
     {"estimated_antenna": True, "adaptive_fitting": True,
      "projection_shift_px": -0.5}),
    # Per-image accuracy probes. Only run when the block was imported in
    # per-image mode (otherwise AddPhotos stored no XMP accuracy and the option
    # would silently fall back to the 10 m default). Each is report-only.
    ("camera_rtk_per_image_adaptive", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE, "adaptive_fitting": True}),
    ("camera_rtk_per_image_antenna_estimated", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE, "estimated_antenna": True}),
    ("camera_rtk_per_image_antenna_estimated_adaptive", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE, "estimated_antenna": True,
      "adaptive_fitting": True}),
    # Block-doming probes (report-only, 2026-11-13). Each isolates one part of
    # the calibration-geometry coupling a nadir-only parallel-strip block is
    # prone to: radial-only distortion with the affine terms held, a held focal
    # length, held tangential terms, and the ground-controlled calibration that
    # is then frozen for the camera-RTK fit.
    ("camera_rtk_k1234", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE,
      # Radial terms free, affine terms (b1, b2) held: the classic radial
      # model. k4 is on so the radial curve can bend, not just scale.
      "fit": {"k1": True, "k2": True, "k3": True, "k4": True,
              "b1": False, "b2": False}}),
    ("camera_rtk_f_fixed", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE, "fit": {"f": False}}),
    ("camera_rtk_p_fixed", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE,
      "fit": {"p1": False, "p2": False, "p3": False, "p4": False}}),
    # The honest leak-free version of "pre-calibrated camera": per fold the
    # lens comes from the control-only GCP adaptive fit (the held-out point
    # never enters it), then every coefficient is held for the camera-RTK
    # adjustment.
    ("fixed_calibration_camera_rtk", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE,
      "fixed_calibration_from_control": True,
      "calibration_source_adaptive": True}),
    # Weight trend: markers unchanged at the survey accuracy, camera priors
    # loosened by 2x and 5x from the reported per-image value. A block datum
    # that only agrees with the GCPs when the cameras are outvoted would show
    # the check improving monotonically with these factors.
    ("camera_rtk_per_image_weight_x2", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE,
      "camera_accuracy_scale": 2.0}),
    ("camera_rtk_per_image_weight_x5", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE,
      "camera_accuracy_scale": 5.0}),
    # Reference-frame probes (2026-11-14, report-only). The camera RTK
    # positions and the surveyed GCPs can be in different realizations of the
    # same geodetic system: ETRS89 (a European NTRIP network, fixed at epoch
    # 1989.0) against WGS84(G2139)/ITRF2014 at the 2022.4 capture epoch, which
    # is about 0.85 m horizontally at this site. The "published" pair applies
    # PROJ's time-dependent ETRF/ITRF operation at the EXIF/mrk capture epoch
    # in each direction, so the hypothesis and its mirror image are both
    # measured. "estimated" removes the translation measured on the SAME fold's
    # control-only GCP adjustment instead: it makes no frame assumption and is
    # leak-free. The epoch is filled in at run time from the dataset's own
    # capture time.
    ("camera_rtk_frame_itrf_to_etrs", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE,
      "frame_correction": "itrf2014-to-etrs89", "capture_epoch": None}),
    ("camera_rtk_frame_itrf_to_etrs_adaptive", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE, "adaptive_fitting": True,
      "frame_correction": "itrf2014-to-etrs89", "capture_epoch": None}),
    ("camera_rtk_frame_etrs_to_itrf_adaptive", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE, "adaptive_fitting": True,
      "frame_correction": "etrs89-to-itrf2014", "capture_epoch": None}),
    ("camera_rtk_frame_estimated_adaptive", True,
     {"camera_accuracy": CAMERA_ACCURACY_PER_IMAGE, "adaptive_fitting": True,
      "estimated_frame_shift": True}),
)


def run_real_odm(args):
    """The real-block leave-one-out protocol over the ODM GCPs, two variants."""
    if args.gcp_height != "ellipsoidal":
        raise ValueError(
            "--gcp-height {} is not supported: converting an orthometric "
            "height needs a geoid model this script does not carry".format(
                args.gcp_height))
    pixel_shift = GCP_PIXEL_ORIGIN_SHIFT_PX[args.gcp_pixel_origin]
    header, gcps = parse_odm_gcp_list(args.gcp_list, pixel_shift)
    _, names = parse_gcp_names_csv(args.gcp_names)
    gcps = match_gcp_names(gcps, names)
    z_record = z_consistency(gcps, args.gcp_height)
    images = collect_images(args.real)
    if len(images) < 3:
        raise ValueError("need at least three images in " + args.real)

    def per_fold_estimates(variant_name):
        """The camera-frame lever arm measured on one variant's folds."""
        return {
            fold["check"]: fold["antenna_offset"]["mean_m"]
            for fold in variants[variant_name]["folds"]
            if "error" not in fold and fold.get("antenna_offset")}

    def per_fold_leads(variant_name):
        """The along-track lead measured on one variant's folds, metres."""
        return {
            fold["check"]: fold["camera_consistency"]["mean_along_track_m"]
            for fold in variants[variant_name]["folds"]
            if "error" not in fold and fold.get("camera_consistency")
            and fold["camera_consistency"].get("mean_along_track_m") is not None}

    def per_fold_frame_shifts(variant_name):
        """The frame translation measured on one variant's GCP-only folds."""
        record = variants.get(variant_name) or {}
        return {
            fold["check"]: fold["frame_fit"]["translation_m"]
            for fold in record.get("folds", [])
            if "error" not in fold and fold.get("frame_fit")
            and fold["frame_fit"].get("status") == "ok"}

    align_report = align_block(args, images, gcps)
    # Timing needs the aligned cameras' stored positions only; it is popped so
    # the 176-row helper list never reaches the JSON report.
    reference_rows = align_report.pop("_reference_rows", [])
    per_image_mode = args.camera_accuracy == CAMERA_ACCURACY_PER_IMAGE
    uniform_accuracy = parse_camera_accuracy(args.camera_accuracy_uniform)
    timing = None
    records = []
    if args.mrk:
        records = parse_mrk(args.mrk)
        if not records:
            raise ValueError("--mrk {} holds no usable exposure records".format(
                args.mrk))
        timing = analyse_timing(reference_rows, images, records)
        print("[gcp validation] timing: {} exposures matched, event vs exposure "
              "RMS {:.6f} s (max {:.6f} s), trigger -> exposure median "
              "{:.6f} s, block speed median {:.3f} m/s, fix statuses {}".format(
                  timing.get("matched_cameras", 0),
                  timing.get("event_vs_exposure_rms_s", float("nan")),
                  timing.get("event_vs_exposure_max_abs_s", float("nan")),
                  timing.get("trigger_to_exposure_median_s", float("nan")),
                  timing.get("block_speed_m_s") or float("nan"),
                  timing.get("fix_status_counts")))
    # Capture epoch for the published frame transform: the coordinate epoch is
    # what makes the ETRS89 <-> ITRF2014 transformation time-dependent. Take it
    # from the dataset's own record (an explicit flag, then the RTK module's GPS
    # event marks, then the EXIF capture time). Nothing is assumed when none is
    # available: the published variants then report the reason they could not
    # run rather than applying a nominal epoch.
    capture_epoch = args.capture_epoch
    capture_epoch_source = "flag" if capture_epoch is not None else None
    if capture_epoch is None and records:
        first = next((record for record in records
                      if record.get("week") is not None), None)
        if first is not None:
            capture_epoch = decimal_year_from_gps(first["week"], first["time_s"])
            capture_epoch_source = "mrk"
    if capture_epoch is None:
        capture_epoch = capture_epoch_from_images(images)
        capture_epoch_source = "exif" if capture_epoch is not None else None
    frame_summary = None
    if args.diagnostics:
        frame_summary = {"capture_epoch": capture_epoch,
                         "capture_epoch_source": capture_epoch_source}
        if capture_epoch is not None:
            frame_summary["published"] = published_frame_offsets(
                gcps, capture_epoch)
        print("[gcp validation] frame: capture epoch {} ({})".format(
            "{:.4f}".format(capture_epoch) if capture_epoch is not None
            else "unknown", capture_epoch_source or "not derivable"))
        for mode, record in (frame_summary.get("published") or {}).items():
            if "mean_enu_m" in record:
                print("[gcp validation] frame: {} is {:+.3f} / {:+.3f} / "
                      "{:+.3f} m E/N/U at this site ({:.3f} m horizontal)".format(
                          mode, record["mean_enu_m"][0],
                          record["mean_enu_m"][1], record["mean_enu_m"][2],
                          record["horizontal_m"]))
            else:
                print("[gcp validation] frame: {} unavailable: {}".format(
                    mode, record.get("error", "unknown error")))
    antenna = parse_antenna_offset(args.antenna_offset)
    variants = {}
    # GCPs only first: its per-fold adjustment is the leak-free source of the
    # estimated antenna lever arm the later variants reuse. The camera-RTK
    # variant takes an explicitly configured fixed arm when one is supplied.
    # The benchmark's adaptive_fitting dispatch parameter selects the same
    # adaptive calibration model on the two standard variants that the
    # diagnostics run as an extra probe. Keep the option on the variant's options
    # dict so the report shows which lens model produced each row.
    base_options = {"adaptive_fitting": True} if args.adaptive_fitting else None
    variants["gcps_only"] = run_variant(
        args, gcps, align_report, False, base_options)
    default_estimates = per_fold_estimates("gcps_only")
    default_frame_shifts = per_fold_frame_shifts("gcps_only")
    camera_options = dict(base_options or {})
    if antenna is not None:
        camera_options["antenna_offset_m"] = antenna
    if args.frame_correction in FRAME_CRS_EPSG:
        camera_options["frame_correction"] = args.frame_correction
        camera_options["capture_epoch"] = capture_epoch
    elif args.frame_correction == "empirical":
        camera_options["estimated_frame_shift"] = True
    variants["camera_rtk_gcps"] = run_variant(
        args, gcps, align_report, True, camera_options,
        frame_shifts=default_frame_shifts)
    # In per-image mode the same alignment is also refitted with one uniform
    # weight, so the per-image effect and the weight effect are separable in a
    # single run. Report-only.
    if per_image_mode:
        variants["camera_rtk_uniform"] = run_variant(
            args, gcps, align_report, True,
            {"camera_accuracy": uniform_accuracy}, catch_all=True)
    # The mount is estimated from the ADAPTIVE GCP-only adjustment, not the
    # default one: with the thin default lens model the GCP-only fit trades
    # focal length against distortion and puts the camera centres 3.3 m below
    # the antenna positions, which is a model artefact, not a mount. The
    # adaptive fit removes that trade and leaves the ~0.2 m mount.
    applied_estimates = default_estimates
    applied_frame_shifts = default_frame_shifts
    if args.diagnostics:
        selected = set(args.diagnostic_variants) if args.diagnostic_variants else None
        for variant, camera_refs, options in DIAGNOSTIC_VARIANTS:
            if selected is not None and variant not in selected:
                continue
            if (options.get("camera_accuracy") == CAMERA_ACCURACY_PER_IMAGE
                    and not per_image_mode):
                # AddPhotos stored no XMP accuracy in a uniform run, so this
                # probe would silently weight at the 10 m default.
                continue
            options = dict(options)
            if "frame_correction" in options and options.get(
                    "capture_epoch") is None:
                options["capture_epoch"] = capture_epoch
            if options.get("estimated_antenna") and variants.get(
                    "adaptive_fitting") is not None:
                adaptive = per_fold_estimates("adaptive_fitting")
                if adaptive:
                    applied_estimates = adaptive
            if options.get("estimated_frame_shift") and variants.get(
                    "adaptive_fitting") is not None:
                adaptive = per_fold_frame_shifts("adaptive_fitting")
                if adaptive:
                    applied_frame_shifts = adaptive
            variants[variant] = run_variant(
                args, gcps, align_report, camera_refs, options,
                catch_all=True, estimates=applied_estimates,
                frame_shifts=applied_frame_shifts)
        # The timing probe needs an event-mark log to know the speed; without
        # one there is nothing to estimate a latency from.
        if timing and timing.get("block_speed_m_s"):
            source = ("adaptive_fitting"
                      if variants.get("adaptive_fitting") else "gcps_only")
            leads = per_fold_leads(source)
            if leads:
                name = ("camera_rtk_per_image_timing_shift" if per_image_mode
                        else "camera_rtk_timing_shift")
                variants[name] = run_variant(
                    args, gcps, align_report, True,
                    {"camera_accuracy": args.camera_accuracy,
                     "estimated_timing": True,
                     "timing_speed_m_s": timing["block_speed_m_s"]},
                    catch_all=True, estimates=applied_estimates, timing=leads)
    if timing:
        annotate_outlier_timing(variants, timing)
    accuracy_summary = variants["camera_rtk_gcps"].get("camera_accuracy")
    if accuracy_summary:
        print("[gcp validation] camera accuracy ({}): {} cameras, {} still on "
              "the 10 m default; east {:.4f}-{:.4f} m, north {:.4f}-{:.4f} m, "
              "up {:.4f}-{:.4f} m".format(
                  args.camera_accuracy, accuracy_summary["cameras"],
                  accuracy_summary["uniform_defaults"],
                  accuracy_summary.get("east", {}).get("min_m", float("nan")),
                  accuracy_summary.get("east", {}).get("max_m", float("nan")),
                  accuracy_summary.get("north", {}).get("min_m", float("nan")),
                  accuracy_summary.get("north", {}).get("max_m", float("nan")),
                  accuracy_summary.get("up", {}).get("min_m", float("nan")),
                  accuracy_summary.get("up", {}).get("max_m", float("nan"))))
    for variant, data in variants.items():
        consistency = data.get("camera_consistency")
        if not consistency:
            continue
        print("[gcp validation] camera reference - adjusted centre "
              "({}, mean over folds): mean E/N/U "
              "{:.3f}/{:.3f}/{:.3f} m, RMS E/N/U {:.3f}/{:.3f}/{:.3f} m "
              "over {} cameras".format(
                  variant, consistency["mean_east_m"],
                  consistency["mean_north_m"], consistency["mean_up_m"],
                  consistency["rms_east_m"], consistency["rms_north_m"],
                  consistency["rms_up_m"], consistency["cameras"]))
    for variant, data in variants.items():
        outliers = data.get("prior_outliers")
        if not outliers:
            continue
        message = (
            "[gcp validation] prior outliers ({}, > {:.1f} sigma): "
            "{}/{} camera-folds, {} unique cameras, fix statuses {}".format(
                variant, outliers["threshold"],
                outliers["flagged_camera_folds"], outliers["camera_folds"],
                len(outliers["unique_labels"]), outliers["fix_status_counts"]))
        if outliers["repeat_labels"]:
            message += ", in every flagged fold: {}".format(
                ",".join(outliers["repeat_labels"][:8]))
        print(message)
        timing_record = data.get("outlier_timing")
        if timing_record:
            print("[gcp validation] outlier timing ({}, speed {:.3f} m/s): "
                  "flagged implied latency median {} s, block median {} s".format(
                      variant, timing_record["speed_m_s"],
                      timing_record["flagged_implied_latency_median_s"],
                      timing_record["block_implied_latency_median_s"]))
    # Dome and calibration diagnostics: one line per variant, so the benchmark
    # log carries the fit (not only the JSON artifact) and a reader can see the
    # sign, size and significance without opening the artifact.
    def number(value, digits=4):
        if value is None:
            return "n/a"
        try:
            return "{:.{}f}".format(float(value), digits)
        except (TypeError, ValueError):
            return "n/a"

    for variant, data in variants.items():
        dome = data.get("dome")
        if not dome:
            continue
        radial = dome.get("check_fit_radial") or {}
        slope = (radial.get("by_name") or {}).get("radius_100m_squared", {})
        camera = dome.get("camera_fit_radial_mean") or {}
        print("[gcp validation] dome ({}): check b {} m/(100 m)^2, t {} "
              "(n {}); camera b {} m/(100 m)^2, mean t {} over {} folds; "
              "GCP/block hull area {} / {} m^2".format(
                  variant, number(slope.get("coefficient")),
                  number(slope.get("t"), 2), radial.get("samples"),
                  number(camera.get("coefficient_mean")),
                  number(camera.get("t_mean"), 2), camera.get("folds") or 0,
                  number(dome.get("gcp_hull_area_m2"), 0),
                  number(dome.get("block_hull_area_m2"), 0)))
    for variant, data in variants.items():
        for sensor, entry in (data.get("calibration") or {}).items():
            correlations = entry.get("correlations_f") or {}
            f_k1 = (correlations.get("k1") or {}).get("mean")
            print("[gcp validation] calibration ({} / {}): f {} px, "
                  "k1 {}, k2 {}, corr(f,k1) {}".format(
                      variant, sensor,
                      number((entry.get("f") or {}).get("mean"), 2),
                      number((entry.get("k1") or {}).get("mean"), 8),
                      number((entry.get("k2") or {}).get("mean"), 8),
                      number(f_k1, 4)))
    # Print the per-fold lever-arm estimate the estimated variants apply: a
    # reader can see whether the mount is stable across folds.
    estimate_source = "adaptive" if args.diagnostics else "default"
    for label, offset in applied_estimates.items():
        print("[gcp validation] antenna estimate ({}, GCP-only {} adjusted "
              "cameras vs RTK, camera frame x/y/z): "
              "{:.3f} / {:.3f} / {:.3f} m".format(
                  label, estimate_source, offset[0], offset[1], offset[2]))
    # The frame test: the GCP-only adjusted centres against the stored camera
    # positions. A consistent translation across folds is a frame offset.
    for variant, data in variants.items():
        frame_fit = data.get("frame_fit")
        if not frame_fit:
            continue
        per_fold = frame_fit.get("per_fold") or {}
        for label, record in per_fold.items():
            if record.get("status") != "ok":
                print("[gcp validation] frame fit ({}, {}): {}".format(
                    variant, label, record.get("status")))
                continue
            print("[gcp validation] frame fit ({}, {}): translation "
                  "E/N/U {:.3f} / {:.3f} / {:.3f} m, rotation "
                  "{:.1f} / {:.1f} / {:.1f} arcsec, scale {:.2f} ppm, "
                  "translation residual {:.3f} m, Helmert residual {:.3f} m".format(
                      variant, label, record["translation_m"][0],
                      record["translation_m"][1], record["translation_m"][2],
                      record["rotation_arcsec"][0], record["rotation_arcsec"][1],
                      record["rotation_arcsec"][2], record["scale_ppm"],
                      record["translation_residual_rms_m"],
                      record["helmert_residual_rms_m"]))
        mean = frame_fit.get("translation_m_mean")
        if mean:
            print("[gcp validation] frame fit ({}): translation mean "
                  "E/N/U {:.3f} / {:.3f} / {:.3f} m over {} folds".format(
                      variant, mean[0], mean[1], mean[2],
                      frame_fit.get("folds", 0)))
    for label, shift in applied_frame_shifts.items():
        print("[gcp validation] frame shift estimate ({}, reference minus "
              "GCP-only adjusted centre, E/N/U; removed from every camera): "
              "{:.3f} / {:.3f} / {:.3f} m".format(
                  label, shift[0], shift[1], shift[2]))
    best_horizontal = variants["camera_rtk_gcps"]["rms_horizontal_m"]
    target = args.horizontal_target
    passed = (best_horizontal is not None and best_horizontal <= target)
    report = {
        "scenario": "real-block-gcp",
        "kind": "gcp",
        "images": len(images),
        "gcps": len(gcps),
        "observations": sum(len(gcp["observations"]) for gcp in gcps),
        "crs_header": header,
        "gcp_pixel_origin": args.gcp_pixel_origin,
        "gcp_pixel_shift_px": pixel_shift,
        "z_consistency": z_record,
        "align": align_report,
        "variants": variants,
        "diagnostics": bool(args.diagnostics),
        "camera_accuracy": args.camera_accuracy,
        "camera_accuracy_uniform_fallback_m": uniform_accuracy,
        "marker_accuracy_m": args.marker_accuracy,
        # Named to avoid the benchmark runner's own top-level "timing" block
        # (stage wall times), which it adds to every dataset's report.
        "gnss_timing": timing,
        "antenna_offset_m": list(antenna) if antenna is not None else None,
        "antenna_estimates_m": applied_estimates,
        "antenna_estimate_source": estimate_source,
        # What each coordinate set is declared/interpreted as, the capture
        # epoch, and the published transform's expected offset at this site.
        "frame": {
            "gcp_list_header": header,
            "chunk_crs_epsg": 4326,
            "camera_crs_epsg": align_report.get("camera_crs_epsg"),
            "marker_crs_epsg": align_report.get("marker_crs_epsg"),
            "capture_epoch": capture_epoch,
            "capture_epoch_source": capture_epoch_source,
            "correction": args.frame_correction,
            "published": (frame_summary or {}).get("published"),
            "shifts_m": applied_frame_shifts,
        },
        "target_horizontal_m": target,
        "passed": passed,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print("[gcp validation] scenario={} images={} gcps={} "
          "camera_rtk_gcps_h={:.6f} gcps_only_h={:.6f} target={:.6f} "
          "passed={}".format(
              report["scenario"], len(images), len(gcps),
              variants["camera_rtk_gcps"]["rms_horizontal_m"] or -1.0,
              variants["gcps_only"]["rms_horizontal_m"] or -1.0,
              target, passed))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="GCP / check-point validation scenarios.")
    parser.add_argument("--synthetic", action="store_true",
                        help="run the synthetic aerial GCP block")
    parser.add_argument("--real", metavar="DIR",
                        help="directory of a real aerial block")
    parser.add_argument("--gcp", metavar="CSV",
                        help="label,x,y,z control-point CSV for --real")
    parser.add_argument("--gcp-list", metavar="PATH",
                        help="ODM gcp_list.txt (header EPSG:4326, rows "
                             "'x y z px py image [label]') for --real")
    parser.add_argument("--gcp-names", metavar="CSV",
                        help="Name;Easting;Northing;Elevation;Longitude;Latitude "
                             "CSV whose points are matched to --gcp-list by "
                             "coordinates")
    parser.add_argument("--gcp-height", choices=GCP_HEIGHT_CHOICES,
                        default="ellipsoidal",
                        help="height reference of the gcp_list z values "
                             "(default ellipsoidal, as the publisher states); "
                             "orthometric is refused because no geoid model is "
                             "available here")
    parser.add_argument("--gcp-pixel-origin", choices=GCP_PIXEL_ORIGINS,
                        default="file",
                        help="pixel convention of the gcp_list columns: "
                             "'file' keeps the values as written (this "
                             "dataset's producer writes the engine's "
                             "corner-origin convention); 'format' applies the "
                             "ODM/OpenSfM integer-origin -> corner-origin "
                             "conversion (+0.5 px)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="after the two standard variants, run the extra "
                             "probes (adaptive/extended calibration, marker "
                             "refinement, pixel shift, marker sigma, loosened "
                             "camera weights, fixed and per-fold-estimated "
                             "antenna lever arms, model-choice and GCP-weight "
                             "probes, fixed-calibration workflow, published and "
                             "empirical reference-frame corrections) and report "
                             "the per-fold calibration, block-doming and "
                             "camera-vs-RTK frame fits; report-only")
    parser.add_argument("--camera-accuracy", default="0.05",
                        help="camera reference accuracy for the camera_rtk_gcps "
                             "variant: 'per-image' uses each photo's XMP RTK "
                             "standard deviations and fix status (AddPhotos "
                             "load_xmp_accuracy), a number of metres gives every "
                             "camera the same weight (the uniform baseline)")
    parser.add_argument("--camera-accuracy-uniform", default="0.05",
                        help="uniform accuracy in metres used by the "
                             "camera_rtk_uniform comparison variant when "
                             "--camera-accuracy is per-image")
    parser.add_argument("--mrk", metavar="PATH",
                        help="per-exposure event-mark (.mrk) log; enables the "
                             "timing comparison and the along-track timing probe")
    parser.add_argument("--diagnostic-variants", default="",
                        help="comma-separated allowlist of diagnostic variant "
                             "names to run instead of all of them (the base "
                             "variants and the timing probe always run); e.g. "
                             "'adaptive_fitting,camera_rtk_per_image_adaptive'")
    parser.add_argument("--adaptive-fitting", action="store_true",
                        help="run the two standard variants (gcps_only and "
                             "camera_rtk_gcps) with the adaptive calibration "
                             "model, the same configuration the benchmark's "
                             "adaptive_fitting dispatch parameter selects")
    parser.add_argument("--marker-accuracy", type=float, default=0.015,
                        help="control- and check-point reference accuracy in metres")
    parser.add_argument("--frame-correction", choices=FRAME_CORRECTION_CHOICES,
                        default="none",
                        help="transform the camera references into the GCP "
                             "frame before the adjustment (default none, the "
                             "historic behaviour). 'itrf2014-to-etrs89' applies "
                             "the published time-dependent transform for "
                             "cameras recorded in WGS84(G2139)/ITRF2014 and GCPs "
                             "in ETRS89; 'etrs89-to-itrf2014' is the mirror "
                             "image; 'empirical' removes the translation "
                             "measured on the same fold's control-only GCP "
                             "adjustment. The published modes use "
                             "--capture-epoch (or the mrk/EXIF capture time).")
    parser.add_argument("--capture-epoch", type=float, default=None,
                        help="coordinate epoch (decimal year) for the published "
                             "frame transform; default: derived from the --mrk "
                             "GPS event times, then the first image's EXIF")
    parser.add_argument("--antenna-offset", default="",
                        help="fixed GNSS antenna lever arm in metres in the "
                             "camera frame, 'x,y,z' (e.g. '0,0,-0.23' for an "
                             "antenna 0.23 m above a nadir lens); applied to "
                             "the camera_rtk_gcps variant and the fit's datum")
    parser.add_argument("--keypoint-limit", type=int, default=250,
                        help="MatchPhotos keypoint_limit_per_mpx for the block")
    parser.add_argument("--horizontal-target", type=float, default=0.05,
                        help="maximum acceptable horizontal check-point RMS over "
                             "the folds, metres (the real-block target)")
    parser.add_argument("--json-out", metavar="PATH",
                        help="write the JSON report to this file as well as stdout")
    parser.add_argument("--crs", default="",
                        help="EPSG code of the CSV coordinates")
    parser.add_argument("--check-points", default="",
                        help="comma-separated labels held out as check points")
    parser.add_argument("--target", default="Circular12",
                        help="marker target style for detection")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="maximum acceptable 3D total check-point RMS, "
                             "metres")
    parser.add_argument("--projection-noise-px", type=float, default=0.25,
                        help="seeded marker projection noise, pixels per axis")
    parser.add_argument("--reference-noise-m", type=float, default=0.01,
                        help="seeded per-axis control-reference noise, metres")
    parser.add_argument("--camera-gnss", choices=("on", "off"), default="on",
                        help="give every camera a consumer-GNSS reference "
                             "(on) or keep the fixture's exact GPS (off)")
    parser.add_argument("--ablations", action="store_true",
                        help="run the five synthetic noise/GNSS scenarios: "
                             "exact pipeline (<= 2 mm), reference noise only "
                             "(spec gate <= 2 cm), projection noise only and "
                             "both noises (<= 1.5x the first-order prediction), "
                             "and GNSS-only with exact inputs (<= 2 mm)")
    args = parser.parse_args(argv)
    if args.projection_noise_px < 0.0 or args.reference_noise_m < 0.0:
        parser.error("noise magnitudes must not be negative")
    if not 0.0 < args.tolerance < 1.0:
        parser.error("--tolerance must be between 0 and 1 metre")
    try:
        args.camera_accuracy = parse_camera_accuracy(args.camera_accuracy)
        args.camera_accuracy_uniform = parse_camera_accuracy(
            args.camera_accuracy_uniform)
    except ValueError as error:
        parser.error(str(error))
    if args.camera_accuracy_uniform == CAMERA_ACCURACY_PER_IMAGE:
        parser.error("--camera-accuracy-uniform must be a number of metres")
    if args.diagnostic_variants:
        names = [name.strip() for name in args.diagnostic_variants.split(",")
                 if name.strip()]
        known = {name for name, _, _ in DIAGNOSTIC_VARIANTS}
        unknown = [name for name in names if name not in known]
        if unknown:
            parser.error("unknown --diagnostic-variants: " + ", ".join(unknown))
        args.diagnostic_variants = names
    else:
        args.diagnostic_variants = []
    return args


def synthetic_passed(report, gate_m):
    """The synthetic gate: georeferenced, every check point evaluated, 3D total."""
    return (report["georeferenced"] and
            report["evaluated_check_points"] == report["check_points"] and
            report["check_point_rms_m"]["total"] <= gate_m)


def emit(report, gate_m, passed):
    """Print the bracket-tagged measurement line and the JSON report."""
    rms = report["check_point_rms_m"]
    report["gate_m"] = gate_m
    report["passed"] = passed
    predicted = report.get("predicted")
    if predicted:
        prediction_text = (
            "predicted_horizontal={:.6f} predicted_vertical={:.6f} "
            "predicted_total={:.6f} ratio={:.3f} ".format(
                predicted["horizontal"], predicted["vertical"],
                predicted["total"], report["predicted_ratio"]))
    else:
        prediction_text = ""
    # A bracket-tagged measurement line, the same shape the other
    # benchmark-style tests use, so CI surfaces the numbers even when the test
    # passes (ctest hides passing output).
    print("[gcp validation] scenario={} horizontal={:.6f} vertical={:.6f} "
          "total={:.6f} {}gate={} passed={}".format(
              report.get("ablation", report["scenario"]),
              rms["horizontal"], rms["vertical"], rms["total"],
              prediction_text,
              "none" if gate_m is None else "{:.6f}".format(gate_m), passed))
    print(json.dumps(report, indent=2, sort_keys=True))


def run_ablations(args):
    """Run the five synthetic scenarios and apply gates A, B and C."""
    truth = load_truth()
    scenarios = (
        # name, projection noise, reference noise, camera GNSS, gate kind, gate.
        # Scenario 1 is the pipeline-exactness control: every input the GCP
        # machinery consumes is exact. Scenarios 2-5 keep the realistic
        # consumer-GNSS camera references and vary only the two noise knobs, so
        # their differences are attributable to those noises. Scenario 4 keeps
        # the noisy GNSS but removes both other noises: with exact control
        # points the answer must be pinned by them, so it isolates the camera
        # prior / control interaction from the injected noise.
        ("exact", 0.0, 0.0, False, "absolute", PIPELINE_TOLERANCE_M),
        ("reference-noise", 0.0, args.reference_noise_m, True,
         "absolute", args.tolerance),
        ("projection-noise", args.projection_noise_px, 0.0, True,
         "predicted", GATE_B_FACTOR),
        ("gnss-only", 0.0, 0.0, True, "absolute", PIPELINE_TOLERANCE_M),
        ("combined", args.projection_noise_px, args.reference_noise_m,
         args.camera_gnss == "on", "predicted", GATE_B_FACTOR),
    )
    all_passed = True
    for name, projection, reference, camera_gnss, kind, value in scenarios:
        report = run_synthetic(args, projection, reference, camera_gnss)
        report["ablation"] = name
        report["gate_kind"] = kind
        if kind == "predicted":
            control_sigma = (reference / math.sqrt(3.0) if reference > 0.0
                             else CONTROL_REFERENCE_FLOOR_M)
            prediction = predict_check_point_rms(
                truth, report["marker_projection_accuracy_px"],
                control_sigma, camera_gnss, report["tiepoint_accuracy_px"])
            report["predicted"] = prediction
            measured = report["check_point_rms_m"]["total"]
            report["predicted_ratio"] = measured / prediction["total"]
            report["gate_factor"] = value
            gate = value * prediction["total"]
        else:
            gate = value
        passed = synthetic_passed(report, gate)
        emit(report, gate, passed)
        if not passed:
            all_passed = False
    return 0 if all_passed else 2


def main(argv):
    args = parse_args(argv)
    try:
        if args.ablations:
            if not args.synthetic:
                raise SystemExit("--ablations needs --synthetic")
            return run_ablations(args)
        if args.synthetic:
            report = run_synthetic(args, args.projection_noise_px,
                                   args.reference_noise_m,
                                   args.camera_gnss == "on")
            # The spec is the 3D total check-point RMS; the horizontal and
            # vertical RMS are reported alongside it.
            passed = synthetic_passed(report, args.tolerance)
            emit(report, args.tolerance, passed)
            return 0 if passed else 2
        if args.real and args.gcp_list and args.gcp_names:
            return run_real_odm(args)
        if args.real and args.gcp:
            report = run_real(args)
            passed = (report["evaluated_check_points"] ==
                      report["check_points"] and
                      report["check_point_rms_m"]["total"] <= args.tolerance)
            emit(report, args.tolerance, passed)
            return 0 if passed else 2
        raise SystemExit(
            "run_gcp_validation: choose --synthetic or --real DIR --gcp CSV")
    except ValueError as error:
        print("run_gcp_validation: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

