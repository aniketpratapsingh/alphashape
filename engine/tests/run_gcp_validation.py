#!/usr/bin/env python3
"""GCP / check-point validation scenarios (WP-13.20).

Two modes:

  * ``--synthetic`` builds the aerial synthetic block, adds 12 control points
    (6 on the ground, 6 on the elevated building) with 1 cm reference noise,
    holds 3 of them out as check points, detects/refines marker projections and
    georeferences the block, then reports the check-point RMS.
  * ``--real <dir> --gcp <csv>`` runs match -> align -> detect markers ->
    import the CSV -> bundle-adjust and reports the check-point RMS.

The script prints ONE JSON object on stdout and exits 0 when the total
check-point RMS is within the tolerance (default 2 cm), 2 otherwise. No data is
downloaded and nothing is written outside a temporary directory.
"""

import argparse
import json
import math
import os
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


def ecef_to_local(ecef, transform):
    """Map an ECEF point into the chunk's local frame via chunk.transform."""
    rotation = [[transform[r][c] for c in range(3)] for r in range(3)]
    shift = [transform[row][3] for row in range(3)]
    delta = [ecef[i] - shift[i] for i in range(3)]
    return tuple(
        sum(rotation[r][c] * delta[c] for c in range(3)) for r in range(3))


# The synthetic block: the ground points sit at z = 0, the elevated ones on the
# building roof (truth height 9.52 m), all inside the 4x3 nadir footprint.
ORIGIN = (9.2, 48.7, 300.0)
GROUND = [(-45.0, -30.0), (-20.0, -35.0), (10.0, -30.0),
          (35.0, -20.0), (30.0, 25.0), (-5.0, 35.0)]
ELEVATED = [(-19.0, 6.0), (-14.0, 6.5), (-9.0, 7.0),
            (-18.0, 15.0), (-12.0, 14.0), (-6.0, 16.0)]
CHECK_LABELS = {"G6", "E2", "E5"}
ROOF_HEIGHT = 9.52

# The arbitrary local frame the block is reconstructed in: a known similarity
# the georeferencing step has to undo.
SIM_SCALE = 0.5
SIM_ANGLE = math.radians(25.0)
SIM_SHIFT = (12.0, -7.0, 3.0)


def noise(index, axis):
    """Deterministic per-axis reference noise in [-1 cm, +1 cm]."""
    return (((index * 37 + axis * 11 + 5) % 101) / 100.0 - 0.5) * 0.02


def to_local_frame(point):
    x, y, z = point
    cos_a, sin_a = math.cos(SIM_ANGLE), math.sin(SIM_ANGLE)
    return (SIM_SCALE * (cos_a * x - sin_a * y) + SIM_SHIFT[0],
            SIM_SCALE * (sin_a * x + cos_a * y) + SIM_SHIFT[1],
            SIM_SCALE * z + SIM_SHIFT[2])


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


def check_point_errors(chunk, labels):
    """Per-axis error of each held-out marker after georeferencing."""
    transform = chunk.transform
    markers = {marker.label: marker for marker in chunk.markers}
    errors = []
    for label in labels:
        marker = markers[label]
        if marker.position is None or marker.reference is None:
            continue
        local = ecef_to_local(geodetic_to_ecef(*marker.reference), transform)
        delta = [marker.position[i] - local[i] for i in range(3)]
        errors.append({"label": label,
                       "horizontal": math.hypot(delta[0], delta[1]),
                       "vertical": delta[2],
                       "total": math.sqrt(sum(value * value
                                              for value in delta))})
    return errors


def root_mean_square(values):
    return math.sqrt(sum(value * value for value in values) / len(values)) \
        if values else 0.0


def run_synthetic(args):
    """Build the synthetic GCP block and measure the held-out check points."""
    with open(os.path.join(DATA_DIR, "aerial_truth.json"),
              encoding="utf-8") as handle:
        truth = json.load(handle)
    images = [item["file"] for item in truth["images"]]
    doc = alphashape.Document()
    chunk = doc.add_chunk("Synthetic GCP block")
    doc.active_chunk = chunk
    doc.run_task("AddPhotos",
                 filenames=[os.path.join(DATA_DIR, name) for name in images])
    # Geographic CRS: marker references are (longitude, latitude, altitude).
    chunk.crs_epsg = 4326

    first_camera = chunk.cameras[0].label
    before = len(chunk.markers)
    chunk.detectMarkers(target_type="Circular", cameras=[first_camera])
    detected = len(chunk.markers) - before

    points = truth_points()
    for label, enu in points:
        chunk.addMarker(label=label, position=to_local_frame(enu))

    by_stem = {os.path.splitext(item["file"])[0]: item
               for item in truth["images"]}
    for camera in chunk.cameras:
        image = by_stem.get(camera.label)
        if image is None:
            continue
        for label, enu in points:
            pixel = project_truth(image, enu)
            if pixel is not None:
                chunk.setMarkerProjection(label, camera.label, pixel[0], pixel[1])

    control_rows = []
    check_rows = []
    for index, (label, enu) in enumerate(points):
        noisy = tuple(enu[axis] + noise(index, axis) for axis in range(3))
        lon, lat, height = enu_to_geodetic(noisy, ORIGIN)
        row = (label, lon, lat, height)
        if label in CHECK_LABELS:
            check_rows.append(row)
        else:
            control_rows.append(row)

    with tempfile.TemporaryDirectory() as directory:
        control_path = os.path.join(directory, "control.csv")
        check_path = os.path.join(directory, "check.csv")
        write_reference_csv(control_path, control_rows)
        write_reference_csv(check_path, check_rows)
        chunk.importReference(path=control_path, marker_type="gcp")
        chunk.importReference(path=check_path, marker_type="check_point")

    chunk.refineMarkers(cameras=[camera.label for camera in chunk.cameras])
    # The block is not aligned: every camera pose is still the identity, so its
    # EXIF reference would be an unusable correspondence. Disable the cameras
    # before the datum fit so the control markers alone drive it (an unaligned
    # camera's "centre" is the origin for all of them).
    for camera in chunk.cameras:
        camera.enabled = False
    chunk.updateTransform()

    labels = [label for label, _ in points if label in CHECK_LABELS]
    errors = check_point_errors(chunk, labels)
    horizontal = root_mean_square([e["horizontal"] for e in errors])
    vertical = root_mean_square([e["vertical"] for e in errors])
    total = root_mean_square([e["total"] for e in errors])
    applied = chunk.meta.get("update_transform.applied") == "1"
    return {
        "scenario": "synthetic-gcp-block",
        "images": len(images),
        "markers": len(chunk.markers),
        "ground_points": len(GROUND),
        "elevated_points": len(ELEVATED),
        "control_points": len(control_rows),
        "check_points": len(check_rows),
        "detected_markers": detected,
        "georeferenced": applied,
        "engine_check_rms_m": float(
            chunk.meta.get("update_transform.rms_check_m", "nan")),
        "check_point_rms_m": {
            "horizontal": horizontal,
            "vertical": vertical,
            "total": total,
        },
        "tolerance_m": args.tolerance,
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


def run_real(args):
    """Match -> align -> detect -> import GCPs -> bundle-adjust a real block."""
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

    chunk.optimizeCameras(use_gcps=True)
    errors = check_point_errors(chunk, sorted(check_labels))
    return {
        "scenario": "real-block",
        "images": len(images),
        "markers": len(chunk.markers),
        "control_points": len(control_lines) - 1,
        "check_points": len(check_lines) - 1,
        "optimizer": "OptimizeCameras",
        "check_point_rms_m": {
            "horizontal": root_mean_square(
                [e["horizontal"] for e in errors]),
            "vertical": root_mean_square([e["vertical"] for e in errors]),
            "total": root_mean_square([e["total"] for e in errors]),
        },
        "check_point_errors_m": errors,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="GCP / check-point validation scenarios.")
    parser.add_argument("--synthetic", action="store_true",
                        help="run the synthetic aerial GCP block")
    parser.add_argument("--real", metavar="DIR",
                        help="directory of a real aerial block")
    parser.add_argument("--gcp", metavar="CSV",
                        help="label,x,y,z control-point CSV for --real")
    parser.add_argument("--crs", default="",
                        help="EPSG code of the CSV coordinates")
    parser.add_argument("--check-points", default="",
                        help="comma-separated labels held out as check points")
    parser.add_argument("--target", default="Circular12",
                        help="marker target style for detection")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="maximum acceptable check-point RMS, metres")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    if args.synthetic:
        report = run_synthetic(args)
        passed = (report["georeferenced"] and
                  report["check_point_rms_m"]["total"] <= args.tolerance)
    elif args.real and args.gcp:
        report = run_real(args)
        passed = report["check_point_rms_m"]["total"] <= args.tolerance
    else:
        raise SystemExit(
            "run_gcp_validation: choose --synthetic or --real DIR --gcp CSV")
    report["passed"] = passed
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

