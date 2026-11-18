#!/usr/bin/env python3
"""Measure the EuroSDR RPAS benchmark's GCP image observations ourselves.

The EuroSDR RPAS benchmark (Zenodo 10059050, Wards Hill quarry) publishes the
surveyed coordinates of 8 GCPs and 51 check points but **not** the pixel
observations of their printed targets. The ground-control protocol needs both:
a 3-D coordinate and the pixel where the target appears in each image that sees
it. This script produces the second half.

It drives the engine through its Python module, exactly as
``engine/tests/run_gcp_validation.py`` does:

  1. import the P4 RTK flight's images with their DJI RTK XMP positions and
     accuracies, then ``MatchPhotos`` (reference preselection) and
     ``AlignCameras`` with **no** GCPs, so the block is georeferenced by the
     camera RTK positions alone;
  2. read the surveyed coordinates from ``GCPsandCPs_coords.xlsx`` (ETRS89
     geodetic lat/lon with ellipsoidal height on GRS80) and add every station as
     a marker in EPSG:4979;
  3. project each point into every aligned camera that sees it, then run
     ``RefineMarkers`` to snap each projection onto the circular target centre;
  4. keep only refinements that pass the detector's acceptance and have a sane
     residual, and write the accepted observations as an ODM ``gcp_list.txt``
     plus a JSON summary.

The output is our own derived data, not a redistribution of the publisher's
imagery: only the pixel coordinates of the targets are written, never an image.
The dataset is CC BY 4.0 (Mills et al. 2023, Newcastle University / EuroSDR);
the committed ``gcp_list`` carries that attribution.

The script is CI-only in practice (it needs the built ``alphashape`` module and
the multi-GB archive), but it is plain Python and unit-tested in
``tools/bench/tests`` for the parts that do not need the engine: the xlsx
reader, the projection geometry, the acceptance filter and the gcp_list writer.

Usage:

    python3 tools/bench/eurosdr_observations.py \
        --images build/bench/eurosdr/images \
        --xlsx build/bench/eurosdr/GCPsandCPs_coords.xlsx \
        --gcp-list build/bench/eurosdr/gcp_list.txt \
        --summary build/bench/eurosdr/observations.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

# The engine's Python module is built into a directory the workflow names; the
# import is deferred to main() so the pure helpers below can be imported and
# unit-tested on a machine with no engine build.
#
# 250 keypoints per megapixel is the density the aerial-nadir and
# terrestrial-laser datasets use, and it is what fits a 16 GB runner: the
# gcp-rtk block (176 images) runs at 500, and 430 images at 500 is 2.4x that
# block's keypoint load, which killed the runner out of memory on the first
# measurement attempt. 250 halves it, and the strided image subset below halves
# the image count again, so the block is smaller than gcp-rtk's.
DEFAULT_KEYPOINT_LIMIT_PER_MPX = 250
# The flight is 430 images at 20 MP. Matching and aligning all of them is more
# than a shared runner can hold, and the measurement does not need every frame:
# it needs each of the 59 targets seen by enough cameras. A stride of 2 keeps
# every other frame, which on a lawnmower flight leaves ~60% forward overlap
# within a strip (enough to match) and every strip present (so the whole site,
# and therefore every target, is still covered). A centroid cluster, which the
# other datasets use, would cover only the middle of the quarry and miss the
# edge targets, so a stride is the right selection here.
DEFAULT_MAX_IMAGES = 215
# The targets are 300 mm circles with a 100 mm black centre. At a 1-2 cm GSD the
# outer circle is 15-30 px across, so the search radius that must cover the
# RTK-only alignment error plus a margin is a few tens of pixels. The projection
# is the RTK-aligned guess; the refinement moves it by at most half the apparent
# target radius (or six full-resolution pixels, whichever is larger), so a
# projection that lands within a target radius of the true centre is refined and
# one that lands further is rejected rather than dragged across the image.
TARGET_DIAMETER_M = 0.30
TARGET_CENTRE_DIAMETER_M = 0.10
# A refinement is accepted when the projection moved by no more than this many
# target radii. The engine's own bound is half a radius (or six pixels); this is
# the outer sanity gate on the RESULT, so a projection that the refinement could
# not pull onto the target is dropped rather than written as a bad observation.
MAX_SHIFT_TARGET_RADII = 1.0
# A projection counts as refined only when the refinement MOVED it. RefineMarkers
# moves a projection only on corroborated evidence (at least two views agree and
# the NCC clears the threshold); a projection it leaves exactly where it was
# measured is "kept unrefined", which is what a point that is not actually
# visible in that camera looks like. The threshold is a fraction of a pixel: a
# genuine sub-pixel refinement moves the projection by more than this, and a
# failed one moves it by exactly zero.
MIN_REFINE_SHIFT_PX = 0.05
# A refined projection is accepted only when the marker's triangulated position
# reprojects within this many pixels of it (or half a target radius, whichever
# is larger). The engine triangulates the marker robustly from its refined
# projections, so a projection that refined onto a DIFFERENT target is an
# outlier and its residual is large; this drops it.
MIN_RESIDUAL_PX = 3.0
# A point needs at least this many accepted views to be a usable observation.
MIN_VIEWS_PER_POINT = 2
# The xlsx sheet and the columns the script reads. The header row names the
# groups; the second row names the per-column statistic ("lat_ave" etc.). The
# ETRS89 geodetic block is columns E/F/G (0-based 4/5/6): lat, lon, ellipsoidal
# height. The OSGB36/ODN block (7/8/9) is orthometric and is NOT read.
XLSX_SHEET = "Sheet1"
XLSX_FIRST_DATA_ROW = 3  # 1-based; rows 1-2 are the two header rows
XLSX_STATION_COLUMN = 0
XLSX_LAT_COLUMN = 4
XLSX_LON_COLUMN = 5
XLSX_HEIGHT_COLUMN = 6
# The 8 GCPs the paper names. Every other station is a check point. The xlsx is
# the authoritative label source; the Phase3 text file's labels are shifted by
# one and are never read here.
GCP_LABELS = ("GA1", "GA5", "GA8", "GC6", "GD4", "GF1", "GF4", "GF8")
# The unnamed row in the xlsx (a station with no label) is given this label so
# it can be a marker. It is a check point, never a GCP.
UNNAMED_STATION_LABEL = "UNNAMED"


# ---------------------------------------------------------------------------
# xlsx reader (openpyxl, imported lazily so the helpers below stay importable)
# ---------------------------------------------------------------------------


def read_stations(path):
    """Read the surveyed stations from the xlsx.

    Returns a list of dicts ``{"label", "lat", "lon", "height", "role"}`` in
    sheet order. ``role`` is ``"gcp"`` for the eight GCP labels and
    ``"check_point"`` for every other station. The coordinates are ETRS89
    geodetic latitude, longitude (degrees) and ellipsoidal height (metres) on
    the GRS80 ellipsoid, taken from the "ETRS89 Geodetic (GRS80 ellipsoid)"
    block. The OSGB36/ODN block in the same sheet is orthometric and is not
    read: the protocol needs ellipsoidal heights, and the xlsx supplies them.

    A row with no station label is the one unnamed check point the paper
    mentions; it is labelled ``UNNAMED`` so it can still be a marker. A row
    whose coordinates are not finite is a data error and raises, rather than
    silently dropping a station.
    """
    import openpyxl  # noqa: PLC0415 - benchmark-only dependency

    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        if XLSX_SHEET not in workbook.sheetnames:
            raise ValueError(
                f"{path}: no sheet named {XLSX_SHEET!r}; found {workbook.sheetnames!r}")
        sheet = workbook[XLSX_SHEET]
        stations = []
        for row_index, row in enumerate(
                sheet.iter_rows(min_row=XLSX_FIRST_DATA_ROW, values_only=True),
                start=XLSX_FIRST_DATA_ROW):
            if row is None:
                continue
            label = row[XLSX_STATION_COLUMN] if len(row) > XLSX_STATION_COLUMN else None
            lat = row[XLSX_LAT_COLUMN] if len(row) > XLSX_LAT_COLUMN else None
            lon = row[XLSX_LON_COLUMN] if len(row) > XLSX_LON_COLUMN else None
            height = row[XLSX_HEIGHT_COLUMN] if len(row) > XLSX_HEIGHT_COLUMN else None
            # A trailing blank row (openpyxl can emit one) has no coordinates.
            if lat is None and lon is None and height is None:
                continue
            if lat is None or lon is None or height is None:
                raise ValueError(
                    f"{path}: row {row_index} has a partial coordinate "
                    f"(lat={lat!r}, lon={lon!r}, height={height!r})")
            try:
                lat = float(lat)
                lon = float(lon)
                height = float(height)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}: row {row_index} has a non-numeric coordinate: {error}") from error
            if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(height)):
                raise ValueError(f"{path}: row {row_index} has a non-finite coordinate")
            name = str(label).strip() if label is not None else ""
            if not name:
                name = UNNAMED_STATION_LABEL
            stations.append({
                "label": name,
                "lat": lat,
                "lon": lon,
                "height": height,
                "role": "gcp" if name in GCP_LABELS else "check_point",
            })
        if not stations:
            raise ValueError(f"{path}: no station rows found")
        return stations
    finally:
        workbook.close()


# ---------------------------------------------------------------------------
# Projection geometry (pure; no engine, no numpy)
# ---------------------------------------------------------------------------


def geodetic_to_ecef(lon_deg, lat_deg, height_m):
    """WGS84 geodetic -> ECEF, the same closed form the engine uses.

    ETRS89 and WGS84 differ by well under a metre at this site and the RTK
    camera positions are recorded in the same frame, so the block is internally
    consistent; the engine's geodesy is WGS84 and this matches it exactly.
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    p = (n + height_m) * cos_lat
    return (p * math.cos(lon), p * math.sin(lon),
            (n * (1.0 - e2) + height_m) * sin_lat)


def ecef_to_local(ecef, transform):
    """Map an ECEF point into the chunk's local frame via ``chunk.transform``.

    ``chunk.transform`` is a similarity local -> ECEF (E = scale * R * L + shift),
    so its inverse is L = R^T (E - shift) / scale. The rotation block is scaled
    in a similarity, hence the cube root of its determinant. This is the same
    inverse ``run_gcp_validation.py`` uses, kept here so the projection step
    needs no engine call.
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
    if scale == 0.0:
        raise ValueError("chunk.transform has a singular rotation block")
    delta = [ecef[i] - shift[i] for i in range(3)]
    return tuple(
        sum(rotation[c][r] * delta[c] for c in range(3)) / scale
        for r in range(3))


def project_point(point_local, pose, calibration):
    """Project a chunk-local point through one camera.

    ``pose`` is the 4x4 world->camera transform (a list of four rows of four
    floats, as ``Camera.pose`` returns it); ``calibration`` is a dict with the
    Brown-Conrady parameters and the image size. Returns ``(u, v)`` in the
    engine's corner-origin pixel convention, or ``None`` when the point is not
    observable (behind a frame camera, or outside the image).

    This is a transcription of ``try_project_point`` for the frame model, which
    is the only model a DJI P4 RTK writes; a fisheye camera is refused rather
    than projected with the wrong model.
    """
    if str(calibration.get("type", "frame")).strip().lower() != "frame":
        raise ValueError(
            "project_point: only the frame model is supported; got "
            f"{calibration.get('type')!r}")
    x = pose[0][0] * point_local[0] + pose[0][1] * point_local[1] + \
        pose[0][2] * point_local[2] + pose[0][3]
    y = pose[1][0] * point_local[0] + pose[1][1] * point_local[1] + \
        pose[1][2] * point_local[2] + pose[1][3]
    z = pose[2][0] * point_local[0] + pose[2][1] * point_local[1] + \
        pose[2][2] * point_local[2] + pose[2][3]
    if not (z > 0.0):
        return None
    xs = x / z
    ys = y / z
    k1 = float(calibration.get("k1", 0.0))
    k2 = float(calibration.get("k2", 0.0))
    k3 = float(calibration.get("k3", 0.0))
    k4 = float(calibration.get("k4", 0.0))
    p1 = float(calibration.get("p1", 0.0))
    p2 = float(calibration.get("p2", 0.0))
    p3 = float(calibration.get("p3", 0.0))
    p4 = float(calibration.get("p4", 0.0))
    r2 = xs * xs + ys * ys
    r4 = r2 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2 + k4 * r4 * r4
    xd = xs * radial + p1 * (r2 + 2.0 * xs * xs) + 2.0 * p2 * xs * ys
    yd = ys * radial + p2 * (r2 + 2.0 * ys * ys) + 2.0 * p1 * xs * ys
    if p3 != 0.0 or p4 != 0.0:
        q1 = p1 + p3 * r2
        q2 = p2 + p4 * r2
        xd = xs * radial + q1 * (r2 + 2.0 * xs * xs) + 2.0 * q2 * xs * ys
        yd = ys * radial + q2 * (r2 + 2.0 * ys * ys) + 2.0 * q1 * xs * ys
    width = float(calibration.get("width", 0))
    height = float(calibration.get("height", 0))
    f = float(calibration.get("f", 0.0))
    b1 = float(calibration.get("b1", 0.0))
    b2 = float(calibration.get("b2", 0.0))
    u = width * 0.5 + float(calibration.get("cx", 0.0)) + xd * (f + b1) + yd * b2
    v = height * 0.5 + float(calibration.get("cy", 0.0)) + yd * f
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    if not (0.0 <= u <= width and 0.0 <= v <= height):
        return None
    return (u, v)


def apparent_target_radius_px(calibration, depth_m):
    """Apparent radius of the 300 mm target at ``depth_m``, in pixels.

    The physical radius is half the target diameter; the projection scales it by
    focal / depth. Used only to size the acceptance gate and to report the
    expected target size, never to move a projection.
    """
    focal = float(calibration.get("f", 0.0))
    if not (depth_m > 0.0) or not (focal > 0.0):
        return 0.0
    return 0.5 * TARGET_DIAMETER_M * focal / depth_m


# ---------------------------------------------------------------------------
# Acceptance and output
# ---------------------------------------------------------------------------


def accept_observation(shift_px, target_radius_px, residual_px=None):
    """Whether a refined projection is a usable observation.

    Three conditions, each of which a wrong projection fails:

    * the refinement MOVED the projection (``shift_px >= MIN_REFINE_SHIFT_PX``).
      RefineMarkers leaves a projection exactly where it was measured when no
      target-specific refiner converged or fewer than two views corroborated the
      correction, which is what a point that is not actually visible in that
      camera looks like. A projection that did not move is not an observation.
    * the move is no larger than ``MAX_SHIFT_TARGET_RADII`` target radii. The
      engine's own bound is half a radius (or six full-resolution pixels); this
      outer gate drops a projection the refinement could not pull onto the
      target. A target radius of zero (no scale) falls back to a fixed pixel
      bound so a scale-less view is not accepted unconditionally.
    * when a residual is supplied, the marker's triangulated position reprojects
      within ``max(MIN_RESIDUAL_PX, 0.5 * target_radius_px)`` of the refined
      pixel. A projection that refined onto a different target is an outlier of
      the triangulation and its residual is large.
    """
    if not math.isfinite(shift_px) or shift_px < MIN_REFINE_SHIFT_PX:
        return False
    if target_radius_px > 0.0:
        if shift_px > MAX_SHIFT_TARGET_RADII * target_radius_px:
            return False
    elif shift_px > 6.0:
        return False
    if residual_px is not None:
        if not math.isfinite(residual_px):
            return False
        limit = max(MIN_RESIDUAL_PX, 0.5 * target_radius_px)
        if residual_px > limit:
            return False
    return True


def write_gcp_list(path, stations, observations):
    """Write the accepted observations as an ODM ``gcp_list.txt``.

    The header is ``EPSG:4979`` (ETRS89 geodetic, ellipsoidal height). Each row
    is ``lon lat height px py image label``: one image observation of one
    station. ``observations`` maps a station label to a list of
    ``{"image", "px", "py"}`` dicts. A station with no accepted observation is
    omitted (it has no row to write). Rows are grouped by station in the order
    the stations were read, and within a station by image name, so the file is
    deterministic.
    """
    lines = ["EPSG:4979"]
    by_label = {station["label"]: station for station in stations}
    for station in stations:
        label = station["label"]
        rows = observations.get(label) or []
        for observation in sorted(rows, key=lambda item: item["image"]):
            lines.append(
                "{:.9f} {:.9f} {:.4f} {:.3f} {:.3f} {} {}".format(
                    station["lon"], station["lat"], station["height"],
                    observation["px"], observation["py"],
                    observation["image"], label))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(lines) - 1


def summarise(stations, observations, per_point):
    """Build the JSON summary: per point, views, score and shift from projection.

    ``per_point`` maps a station label to a list of per-view records, each with
    ``image``, ``shift_px``, ``target_radius_px`` and ``accepted``. The summary
    reports, per station, the number of views projected, the number accepted,
    the mean and maximum shift, and the role. A station with no views is listed
    with zero counts so a point that was never seen is visible rather than
    absent.
    """
    points = []
    for station in stations:
        label = station["label"]
        views = per_point.get(label) or []
        accepted = [view for view in views if view.get("accepted")]
        shifts = [view["shift_px"] for view in accepted
                  if math.isfinite(view.get("shift_px", float("nan")))]
        residuals = [view["residual_px"] for view in accepted
                     if view.get("residual_px") is not None
                     and math.isfinite(view["residual_px"])]
        points.append({
            "label": label,
            "role": station["role"],
            "views_projected": len(views),
            "views_accepted": len(accepted),
            "mean_shift_px": (sum(shifts) / len(shifts)) if shifts else None,
            "max_shift_px": max(shifts) if shifts else None,
            "mean_residual_px": (sum(residuals) / len(residuals)) if residuals else None,
            "max_residual_px": max(residuals) if residuals else None,
            "images": sorted(view["image"] for view in accepted),
        })
    total_views = sum(point["views_projected"] for point in points)
    total_accepted = sum(point["views_accepted"] for point in points)
    gcps = [point for point in points if point["role"] == "gcp"]
    checks = [point for point in points if point["role"] == "check_point"]
    return {
        "stations": len(points),
        "gcps": len(gcps),
        "check_points": len(checks),
        "views_projected": total_views,
        "views_accepted": total_accepted,
        "acceptance_rate": (total_accepted / total_views) if total_views else None,
        "gcps_with_min_views": sum(
            1 for point in gcps if point["views_accepted"] >= MIN_VIEWS_PER_POINT),
        "points": points,
    }


# ---------------------------------------------------------------------------
# Engine driver
# ---------------------------------------------------------------------------


def collect_images(directory):
    """Every image file under ``directory``, sorted by name."""
    suffixes = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    found = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(suffixes):
                found.append(os.path.join(root, name))
    return sorted(found)


def select_images(images, limit):
    """A strided subset of ``images`` that keeps the whole site covered.

    The flight is a lawnmower pattern, so taking every ``stride``-th frame keeps
    every strip (the whole site, and therefore every target) while leaving enough
    forward overlap within a strip to match. ``stride`` is chosen so the result
    is at most ``limit`` images: ``ceil(len / limit)``. A stride of 1 (no limit,
    or a set already small enough) returns every image.

    This is deliberately NOT the centroid cluster the other datasets use: that
    keeps the middle of the site and drops the edges, which is exactly where
    several of the 59 targets sit. A stride keeps the spatial extent.
    """
    if limit is None or limit <= 0 or len(images) <= limit:
        return list(images), 1
    stride = math.ceil(len(images) / limit)
    return images[::stride], stride


def calibration_dict(calibration):
    """A ``Calibration`` proxy as the plain dict :func:`project_point` reads."""
    return {
        "type": calibration.type,
        "width": calibration.width,
        "height": calibration.height,
        "f": calibration.f,
        "cx": calibration.cx,
        "cy": calibration.cy,
        "b1": calibration.b1,
        "b2": calibration.b2,
        "k1": calibration.k1,
        "k2": calibration.k2,
        "k3": calibration.k3,
        "k4": calibration.k4,
        "p1": calibration.p1,
        "p2": calibration.p2,
        "p3": calibration.p3,
        "p4": calibration.p4,
    }


def align_block(alphashape, images, keypoint_limit_per_mpx):
    """Import, match and align the block with no ground control.

    Returns ``(document, chunk)``. The chunk is georeferenced by the camera RTK
    positions alone, which is the point: the GCPs must not influence the
    alignment that is used to project them.
    """
    document = alphashape.Document()
    chunk = document.add_chunk("EuroSDR P4 RTK block")
    document.active_chunk = chunk
    # Per-image mode reads each photo's XMP RTK deviations and fix status at
    # import, so the alignment is weighted by each camera's own solution.
    document.run_task("AddPhotos", filenames=images, load_xmp_accuracy=True)
    chunk.crs_epsg = 4326
    document.run_task("MatchPhotos",
                      keypoint_limit_per_mpx=keypoint_limit_per_mpx,
                      reference_preselection=True)
    document.run_task("AlignCameras")
    return document, chunk


def add_stations(chunk, stations):
    """Add every station as a marker with its ETRS89 reference.

    The marker reference is ``(lon, lat, height)`` in the chunk's marker CRS,
    which is EPSG:4326 (WGS84 geographic) after ``crs_epsg = 4326``. ETRS89 and
    WGS84 differ by well under a metre here and the camera positions are in the
    same frame, so the reference is used as-is. The role (gcp / check_point) is
    carried in the summary and the gcp_list labels, not on the marker: the ODM
    gcp_list format has no role column, and the downstream protocol decides
    control from check by label.
    """
    for station in stations:
        marker = chunk.addMarker(label=station["label"])
        marker.reference = (station["lon"], station["lat"], station["height"])


def project_stations(chunk, stations):
    """Project every station into every aligned camera that sees it.

    Returns ``(projections, per_point, calibrations, poses)``. ``projections``
    maps a station label to a list of ``{"camera", "px", "py",
    "target_radius_px"}`` dicts, one per camera the point projects into.
    ``per_point`` is the same data keyed for the summary, with the shift and
    residual filled in after refinement. ``calibrations`` and ``poses`` map a
    camera label to its calibration dict and 4x4 pose, so the residual gate can
    reproject the triangulated marker without another engine call.
    """
    transform = chunk.transform
    # A camera names its sensor by id, which indexes Chunk.sensors. The P4 RTK
    # block has one sensor, but resolving by id is what the engine does and it
    # stays correct if the block ever carries more.
    sensors = list(chunk.sensors)
    # The chunk-local position of each station, from its geodetic reference.
    local = {}
    for station in stations:
        ecef = geodetic_to_ecef(station["lon"], station["lat"], station["height"])
        local[station["label"]] = ecef_to_local(ecef, transform)
    projections = {}
    per_point = {}
    calibrations = {}
    poses = {}
    for camera in chunk.cameras:
        if not camera.enabled:
            continue
        sensor_id = camera.sensor_id
        if not (0 <= sensor_id < len(sensors)):
            continue
        calibration = calibration_dict(sensors[sensor_id].calibration)
        pose = camera.pose
        centre = camera.center
        calibrations[camera.label] = calibration
        poses[camera.label] = pose
        for station in stations:
            point = local[station["label"]]
            pixel = project_point(point, pose, calibration)
            if pixel is None:
                continue
            depth = math.dist(point, centre)
            radius = apparent_target_radius_px(calibration, depth)
            projections.setdefault(station["label"], []).append({
                "camera": camera.label,
                "px": pixel[0],
                "py": pixel[1],
                "target_radius_px": radius,
            })
            per_point.setdefault(station["label"], []).append({
                "image": camera.label,
                "shift_px": 0.0,
                "residual_px": None,
                "target_radius_px": radius,
                "accepted": False,
            })
    return projections, per_point, calibrations, poses


def refine_and_collect(chunk, stations, projections, per_point, refine_options,
                       calibrations, poses):
    """Set the projections, refine them, and record the accepted observations.

    The projections are written onto the markers, ``RefineMarkers`` is run over
    every station, and each projection's movement is measured against the pixel
    it was projected to. A projection is accepted when the refinement moved it
    and the marker's triangulated position reprojects onto it; the accepted pixel
    is the refined one.

    ``calibrations`` and ``poses`` map a camera label to its calibration dict and
    its 4x4 pose, so the residual of each refined projection against the
    triangulated marker position can be computed here without another engine
    call. Returns ``(observations, per_point)`` where ``observations`` maps a
    station label to the accepted ``{"image", "px", "py"}`` rows.
    """
    before = {}
    for station in stations:
        label = station["label"]
        for projection in projections.get(label, []):
            chunk.setMarkerProjection(label, projection["camera"],
                                      projection["px"], projection["py"])
            before[(label, projection["camera"])] = (projection["px"], projection["py"])
    chunk.refineMarkers(markers=[station["label"] for station in stations],
                        **refine_options)
    after = {}
    for marker in chunk.markers:
        for projection in marker.projections:
            after[(marker.label, projection.camera.label)] = projection.pixel
    # The triangulated marker position, for the residual gate. A marker with no
    # position (fewer than two usable views) has no residual to check, so its
    # projections are accepted on the shift gate alone.
    positions = {marker.label: marker.position for marker in chunk.markers}
    observations = {}
    for station in stations:
        label = station["label"]
        position = positions.get(label)
        for record in per_point.get(label, []):
            key = (label, record["image"])
            refined = after.get(key)
            if refined is None:
                continue
            original = before.get(key)
            if original is None:
                continue
            shift = math.hypot(refined[0] - original[0], refined[1] - original[1])
            record["shift_px"] = shift
            residual = None
            if position is not None:
                calibration = calibrations.get(record["image"])
                pose = poses.get(record["image"])
                if calibration is not None and pose is not None:
                    reprojected = project_point(position, pose, calibration)
                    if reprojected is not None:
                        residual = math.hypot(reprojected[0] - refined[0],
                                              reprojected[1] - refined[1])
            record["residual_px"] = residual
            record["accepted"] = accept_observation(
                shift, record["target_radius_px"], residual)
            if record["accepted"]:
                observations.setdefault(label, []).append({
                    "image": record["image"],
                    "px": refined[0],
                    "py": refined[1],
                })
    return observations, per_point


def log(message):
    """Print a progress line and flush it.

    The measurement runs for hours in CI, and a runner that is killed (out of
    memory, or the job timeout) loses everything still in the stdout buffer. A
    flushed line per stage is what makes a killed run diagnosable: the last line
    says which stage it died in.
    """
    print(message, flush=True)


def run(args):
    """The whole measurement: align, project, refine, write."""
    import alphashape  # noqa: PLC0415 - needs the built module on PYTHONPATH

    available = collect_images(args.images)
    if len(available) < 3:
        raise SystemExit(f"need at least three images under {args.images}, found {len(available)}")
    images, stride = select_images(available, args.max_images)
    stations = read_stations(args.xlsx)
    log(f"[eurosdr] {len(images)} of {len(available)} images (stride {stride}), "
        f"{len(stations)} stations "
        f"({sum(1 for s in stations if s['role'] == 'gcp')} GCPs)")

    log("[eurosdr] importing, matching and aligning (no GCPs)")
    document, chunk = align_block(alphashape, images, args.keypoint_limit)
    aligned = sum(1 for camera in chunk.cameras if camera.enabled)
    log(f"[eurosdr] aligned {aligned} of {len(chunk.cameras)} cameras")
    if aligned < 3:
        raise SystemExit("fewer than three cameras aligned; cannot project the stations")

    add_stations(chunk, stations)
    projections, per_point, calibrations, poses = project_stations(chunk, stations)
    log(f"[eurosdr] projected {sum(len(v) for v in projections.values())} "
        f"observations over {len(projections)} stations")

    refine_options = {
        "patch_size": args.patch_size,
        "min_correlation": args.min_correlation,
        "search_steps": args.search_steps,
    }
    log("[eurosdr] refining the projections onto the targets")
    observations, per_point = refine_and_collect(
        chunk, stations, projections, per_point, refine_options,
        calibrations, poses)
    summary = summarise(stations, observations, per_point)
    summary["images"] = len(images)
    summary["images_available"] = len(available)
    summary["image_stride"] = stride
    summary["cameras_aligned"] = aligned
    summary["cameras_total"] = len(chunk.cameras)
    summary["keypoint_limit_per_mpx"] = args.keypoint_limit
    summary["refine_options"] = refine_options
    summary["target_diameter_m"] = TARGET_DIAMETER_M
    summary["max_shift_target_radii"] = MAX_SHIFT_TARGET_RADII

    rows = write_gcp_list(args.gcp_list, stations, observations)
    log(f"[eurosdr] wrote {rows} observations to {args.gcp_list}")
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    log(f"[eurosdr] acceptance {summary['views_accepted']}/"
        f"{summary['views_projected']} "
        f"({(summary['acceptance_rate'] or 0.0) * 100:.1f}%), "
        f"{summary['gcps_with_min_views']}/{summary['gcps']} GCPs with "
        f">= {MIN_VIEWS_PER_POINT} views")
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Measure the EuroSDR RPAS benchmark's GCP image observations.")
    parser.add_argument("--images", required=True,
                        help="directory of the P4 RTK flight's images")
    parser.add_argument("--xlsx", required=True,
                        help="GCPsandCPs_coords.xlsx from the Zenodo record")
    parser.add_argument("--gcp-list", required=True,
                        help="write the accepted observations here (ODM gcp_list.txt)")
    parser.add_argument("--summary", required=True,
                        help="write the JSON summary here")
    parser.add_argument("--keypoint-limit", type=int,
                        default=DEFAULT_KEYPOINT_LIMIT_PER_MPX,
                        help="MatchPhotos keypoint_limit_per_mpx (default 250)")
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES,
                        help="cap the images with a stride that keeps the whole "
                             "site covered (default 215; 0 = every image)")
    parser.add_argument("--patch-size", type=int, default=25,
                        help="RefineMarkers patch_size (default 25)")
    parser.add_argument("--min-correlation", type=float, default=0.5,
                        help="RefineMarkers min_correlation (default 0.5)")
    parser.add_argument("--search-steps", type=int, default=10,
                        help="RefineMarkers search_steps (default 10)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
