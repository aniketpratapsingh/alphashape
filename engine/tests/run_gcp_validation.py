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
import json
import math
import os
import random
import sys
import tempfile

import alphashape

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# WGS84 constants, identical to the engine's geodesy module, so the reference
# ENU coordinates this script reconstructs match the ones the fit uses.
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


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


def parse_odm_gcp_list(path):
    """Parse an ODM gcp_list.txt into one dict per survey point.

    The header is a CRS string (``EPSG:4326``: x=longitude, y=latitude). Each
    following row is ``x y z px py image [label]`` — one image observation of
    one point. Rows with the same (x, y, z) are one GCP, collected in first-seen
    order so the returned list has one entry per distinct surveyed point.
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
        px, py = float(tokens[3]), float(tokens[4])
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
    doc.run_task("AddPhotos", filenames=images)
    chunk.crs_epsg = 4326
    doc.run_task("MatchPhotos", keypoint_limit_per_mpx=args.keypoint_limit)
    doc.run_task("AlignCameras")
    project = project_path_for(args)
    doc.save(project)
    total = len(chunk.cameras)
    aligned = sum(1 for camera in chunk.cameras if camera.enabled)
    return {"project": project, "cameras_total": total,
            "cameras_aligned": aligned,
            "keypoint_limit_per_mpx": args.keypoint_limit,
            "observations": skipped_observations(chunk, gcps)}


def import_camera_references_for_block(chunk, accuracy_m, directory):
    """Re-import every camera reference at the block's stated ``accuracy_m``.

    The binding's ``reference_accuracy`` is never None once a location exists:
    AddPhotos stores the 10 m consumer default, so reading it back would weight
    the RTK positions as consumer GNSS and leave them without effect.
    """
    rows = []
    for camera in chunk.cameras:
        reference = camera.reference
        if reference is None or not all(
                math.isfinite(float(value)) for value in reference):
            continue
        rows.append((camera.label, reference[0], reference[1], reference[2],
                     (accuracy_m, accuracy_m, accuracy_m)))
    path = os.path.join(directory, "camera_references.csv")
    write_camera_reference_csv(path, rows)
    chunk.importReference(path=path)


def camera_consistency(chunk):
    """How far each aligned camera's stored reference is from its adjusted centre.

    Per camera with a finite reference: the adjusted centre (chunk-local, mapped
    to ECEF by chunk.transform) expressed in the reference's local east/north/up
    frame. A constant up offset points at a height-datum mismatch between the
    camera altitudes and the GCP heights; a metre-level spread points at
    positions that are not RTK-fixed.

    The horizontal difference is also projected onto the flight direction,
    taken from each camera's neighbours in capture order (labels sorted). A
    constant along-track offset is the signature of a trigger/GNSS timing lag
    (and it does NOT change sign between flight lines, because the direction is
    per-camera); a constant world offset shows up in mean east/north instead.
    Measured only, never applied.
    """
    transform = chunk.transform
    samples = []
    for camera in chunk.cameras:
        reference = camera.reference
        if not camera.enabled or reference is None or not all(
                math.isfinite(float(value)) for value in reference):
            continue
        local = camera.center
        ecef = [sum(transform[r][c] * local[c] for c in range(3)) + transform[r][3]
                for r in range(3)]
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
    return kwargs


def run_fold(args, gcps, check_index, camera_refs, align_report, options=None):
    """One leave-one-out fold: reload, project, import, fit, measure the check.

    ``options`` selects a diagnostic variant: a projection pixel shift, a marker
    projection sigma, extra calibration terms, marker refinement, or a
    loosened camera accuracy.
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

    with tempfile.TemporaryDirectory() as directory:
        if camera_refs:
            import_camera_references_for_block(
                chunk, options.get("camera_accuracy", args.camera_accuracy),
                directory)
        else:
            for camera in chunk.cameras:
                camera.reference_enabled = False

        control_rows = [(gcps[index]["label"], gcps[index]["lon"],
                         gcps[index]["lat"], gcps[index]["height"])
                        for index in range(len(gcps)) if index != check_index]
        check_rows = [(gcps[check_index]["label"], gcps[check_index]["lon"],
                       gcps[check_index]["lat"], gcps[check_index]["height"])]
        control_path = os.path.join(directory, "control.csv")
        check_path = os.path.join(directory, "check.csv")
        write_reference_csv(control_path, control_rows)
        write_reference_csv(check_path, check_rows)
        chunk.importReference(path=control_path, marker_type="gcp")
        chunk.importReference(path=check_path, marker_type="check_point")

    focal_before = sensor_focals(chunk)
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
            "camera_consistency": camera_consistency(chunk)}
    if refine is not None:
        fold["refine_markers"] = refine
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


def run_variant(args, gcps, align_report, camera_refs, options=None,
                catch_all=False):
    """Run the leave-one-out folds of one variant and summarise them."""
    options = options or {}
    folds = []
    for check_index in range(len(gcps)):
        try:
            folds.append(run_fold(args, gcps, check_index, camera_refs,
                                  align_report, options))
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
)


def run_real_odm(args):
    """The real-block leave-one-out protocol over the ODM GCPs, two variants."""
    if args.gcp_height != "ellipsoidal":
        raise ValueError(
            "--gcp-height {} is not supported: converting an orthometric "
            "height needs a geoid model this script does not carry".format(
                args.gcp_height))
    header, gcps = parse_odm_gcp_list(args.gcp_list)
    _, names = parse_gcp_names_csv(args.gcp_names)
    gcps = match_gcp_names(gcps, names)
    z_record = z_consistency(gcps, args.gcp_height)
    images = collect_images(args.real)
    if len(images) < 3:
        raise ValueError("need at least three images in " + args.real)

    align_report = align_block(args, images, gcps)
    variants = {}
    for variant, camera_refs in (("camera_rtk_gcps", True),
                                 ("gcps_only", False)):
        variants[variant] = run_variant(args, gcps, align_report, camera_refs)
    if args.diagnostics:
        for variant, camera_refs, options in DIAGNOSTIC_VARIANTS:
            variants[variant] = run_variant(
                args, gcps, align_report, camera_refs, options,
                catch_all=True)
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
        "z_consistency": z_record,
        "align": align_report,
        "variants": variants,
        "diagnostics": bool(args.diagnostics),
        "camera_accuracy_m": args.camera_accuracy,
        "marker_accuracy_m": args.marker_accuracy,
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
    parser.add_argument("--diagnostics", action="store_true",
                        help="after the two standard variants, run the extra "
                             "gcps_only probes (adaptive/extended calibration, "
                             "marker refinement, pixel shift, marker sigma, "
                             "loosened camera weights); report-only")
    parser.add_argument("--camera-accuracy", type=float, default=0.05,
                        help="camera reference accuracy in metres for the "
                             "camera_rtk_gcps variant (the RTK block's stated value)")
    parser.add_argument("--marker-accuracy", type=float, default=0.015,
                        help="control- and check-point reference accuracy in metres")
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

