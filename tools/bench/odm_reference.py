#!/usr/bin/env python3
"""Leave-one-out ground-control reference with OpenDroneMap on the gcp-rtk block.

The engine's real-block GCP protocol (``engine/tests/run_gcp_validation.py``)
holds each of the five surveyed points out in turn, adjusts the block on the
other four and measures the held-out point. This script runs the same folds
through OpenDroneMap (ODM, which uses OpenSfM) so the engine's numbers can be
read against an independent tool on the same photographs and the same survey.
ODM is not ground truth; the survey is. The point is to learn whether the
block itself allows the 5 cm target, or whether another tool does better.

Subcommands, run in this order by ``.github/workflows/odm-reference.yml``:

``fetch``
    Download and verify the gcp-rtk archive (the bench's own downloader and
    manifest hash) and lay out an ODM project: ``<work>/project/images``.
``fold``
    Write the fold's ``gcp_list.txt`` (every point but the held-out one) and a
    ``holdout.json`` with the held-out point's survey position and its image
    observations.
``evaluate``
    Triangulate the held-out point from its observations through ODM's
    cameras, convert the survey position to ODM's topocentric frame and report
    the horizontal and vertical error. The control points are measured the same
    way, as a sanity check on the triangulation itself.
``summarize``
    Fold the per-fold results into per-mode RMS and a Markdown table.

Only numbers are written. No image, survey file or ODM product is ever
committed or published: the dataset is GPL-3.0 and is downloaded and measured
only, exactly as the quality bench does.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATASET = "gcp-rtk"

# WGS84, for the survey position -> ODM topocentric conversion.
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


# ---------------------------------------------------------------------------
# GCP list
# ---------------------------------------------------------------------------


def read_gcp_list(path: Path) -> tuple[str, list[dict]]:
    """The header line and the points of an ODM ``gcp_list.txt``.

    Points are grouped by their coordinates, which is how OpenSfM itself
    identifies a point; the optional label column names it. Each point keeps
    its original lines so a fold file is a verbatim subset of the input.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = lines[0].strip()
    points: dict[tuple[str, str, str], dict] = {}
    for line in lines[1:]:
        words = line.split()
        if len(words) < 6:
            continue
        key = (words[0], words[1], words[2])
        point = points.get(key)
        if point is None:
            point = {
                "x": float(words[0]), "y": float(words[1]), "z": float(words[2]),
                "label": words[6] if len(words) > 6 else f"GCP-{len(points) + 1}",
                "observations": [], "lines": [],
            }
            points[key] = point
        point["observations"].append(
            {"image": words[5], "px": float(words[3]), "py": float(words[4])})
        point["lines"].append(line)
    ordered = sorted(points.values(), key=lambda p: p["label"])
    return header, ordered


def write_fold(gcp_list: Path, fold: int, out_list: Path, out_holdout: Path) -> dict:
    header, points = read_gcp_list(gcp_list)
    if header != "EPSG:4326":
        raise SystemExit(f"{gcp_list}: expected an EPSG:4326 header, found {header!r}")
    if not 1 <= fold <= len(points):
        raise SystemExit(f"fold {fold} is outside 1..{len(points)}")
    held = points[fold - 1]
    control = [p for i, p in enumerate(points) if i != fold - 1]
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text(
        "\n".join([header] + [line for p in control for line in p["lines"]]) + "\n",
        encoding="utf-8")
    holdout = {
        "fold": fold,
        "check": _public_point(held),
        "control": [_public_point(p) for p in control],
    }
    out_holdout.write_text(json.dumps(holdout, indent=2), encoding="utf-8")
    return holdout


def _public_point(point: dict) -> dict:
    return {key: point[key] for key in ("label", "x", "y", "z", "observations")}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def ecef_from_lla(lat_deg: float, lon_deg: float, alt: float) -> tuple[float, float, float]:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return ((n + alt) * math.cos(lat) * math.cos(lon),
            (n + alt) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - WGS84_E2) + alt) * math.sin(lat))


def topocentric_from_lla(lat: float, lon: float, alt: float,
                         ref_lat: float, ref_lon: float, ref_alt: float) -> list[float]:
    """East/north/up of a point relative to the reference, as OpenSfM defines it."""
    x, y, z = ecef_from_lla(lat, lon, alt)
    x0, y0, z0 = ecef_from_lla(ref_lat, ref_lon, ref_alt)
    dx, dy, dz = x - x0, y - y0, z - z0
    phi, lam = math.radians(ref_lat), math.radians(ref_lon)
    east = -math.sin(lam) * dx + math.cos(lam) * dy
    north = (-math.sin(phi) * math.cos(lam) * dx - math.sin(phi) * math.sin(lam) * dy
             + math.cos(phi) * dz)
    up = (math.cos(phi) * math.cos(lam) * dx + math.cos(phi) * math.sin(lam) * dy
          + math.sin(phi) * dz)
    return [east, north, up]


def rotation_from_axis_angle(r):
    import numpy as np  # noqa: PLC0415

    r = np.asarray(r, dtype=float)
    theta = float(np.linalg.norm(r))
    if theta < 1e-15:
        return np.eye(3)
    k = r / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * kx + (1.0 - math.cos(theta)) * kx @ kx


def normalized_from_pixel(px: float, py: float, width: int, height: int) -> tuple[float, float]:
    """OpenSfM's normalized image coordinates (the +0.5 is its pixel-centre rule)."""
    size = float(max(width, height))
    return (px + 0.5 - width / 2.0) / size, (py + 0.5 - height / 2.0) / size


class Camera:
    """The OpenSfM camera models ODM can produce for a frame camera."""

    SUPPORTED = ("perspective", "brown", "radial", "simple_radial")

    def __init__(self, data: dict):
        kind = data.get("projection_type")
        if kind not in self.SUPPORTED:
            raise SystemExit(f"unsupported OpenSfM projection type {kind!r}")
        self.kind = kind
        self.width = int(data["width"])
        self.height = int(data["height"])
        if kind == "perspective":
            self.fx = self.fy = float(data["focal"])
            self.cx = self.cy = 0.0
        else:
            self.fx = float(data["focal_x"])
            self.fy = float(data["focal_y"])
            self.cx = float(data.get("c_x", 0.0))
            self.cy = float(data.get("c_y", 0.0))
        self.k1 = float(data.get("k1", 0.0))
        self.k2 = float(data.get("k2", 0.0)) if kind != "simple_radial" else 0.0
        self.k3 = float(data.get("k3", 0.0)) if kind == "brown" else 0.0
        self.p1 = float(data.get("p1", 0.0)) if kind == "brown" else 0.0
        self.p2 = float(data.get("p2", 0.0)) if kind == "brown" else 0.0

    def project(self, point_camera) -> tuple[float, float]:
        x = point_camera[0] / point_camera[2]
        y = point_camera[1] / point_camera[2]
        r2 = x * x + y * y
        radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
        xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        yd = y * radial + 2.0 * self.p2 * x * y + self.p1 * (r2 + 2.0 * y * y)
        return self.fx * xd + self.cx, self.fy * yd + self.cy

    def bearing(self, u: float, v: float):
        """Undistorted direction (x, y, 1) of a normalized observation."""
        import numpy as np  # noqa: PLC0415

        xd, yd = (u - self.cx) / self.fx, (v - self.cy) / self.fy
        x, y = xd, yd
        for _ in range(50):
            r2 = x * x + y * y
            radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
            dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
            dy = 2.0 * self.p2 * x * y + self.p1 * (r2 + 2.0 * y * y)
            x, y = (xd - dx) / radial, (yd - dy) / radial
        return np.array([x, y, 1.0])


def triangulate(observations: list[dict], reconstruction: dict) -> dict:
    """Least-squares point from every observation whose image ODM reconstructed."""
    import numpy as np  # noqa: PLC0415

    cameras = {key: Camera(value) for key, value in reconstruction["cameras"].items()}
    rays = []
    for obs in observations:
        shot = reconstruction["shots"].get(obs["image"])
        if shot is None:
            continue
        camera = cameras[shot["camera"]]
        rotation = rotation_from_axis_angle(shot["rotation"])
        translation = np.asarray(shot["translation"], dtype=float)
        u, v = normalized_from_pixel(obs["px"], obs["py"], camera.width, camera.height)
        rays.append((camera, rotation, translation, u, v))
    if len(rays) < 2:
        return {"status": "too_few_views", "views": len(rays)}

    # Linear start: the point closest to every viewing ray.
    a = np.zeros((3, 3))
    b = np.zeros(3)
    names = [obs["image"] for obs in observations if obs["image"] in reconstruction["shots"]]
    for camera, rotation, translation, u, v in rays:
        centre = -rotation.T @ translation
        direction = rotation.T @ camera.bearing(u, v)
        direction /= np.linalg.norm(direction)
        projector = np.eye(3) - np.outer(direction, direction)
        a += projector
        b += projector @ centre
    point = np.linalg.solve(a, b)

    # Gauss-Newton on the reprojection error, through the full camera model.
    def residuals(p):
        out = []
        for camera, rotation, translation, u, v in rays:
            pu, pv = camera.project(rotation @ p + translation)
            out.extend([pu - u, pv - v])
        return np.array(out)

    for _ in range(20):
        r = residuals(point)
        jac = np.zeros((len(r), 3))
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = 1e-4
            jac[:, axis] = (residuals(point + step) - residuals(point - step)) / 2e-4
        delta, *_ = np.linalg.lstsq(jac, -r, rcond=None)
        point = point + delta
        if np.linalg.norm(delta) < 1e-7:
            break
    size = max(rays[0][0].width, rays[0][0].height)
    final = residuals(point) * size
    rms_px = float(np.sqrt(np.mean(final ** 2)))
    per_view = {name: [float(final[2 * i]), float(final[2 * i + 1])]
                for i, name in enumerate(names)}
    return {"status": "ok", "views": len(rays), "point": point.tolist(),
            "reprojection_rms_px": rms_px, "residuals_px": per_view}


def measure(point: dict, reconstruction: dict, reference: dict) -> dict:
    result = triangulate(point["observations"], reconstruction)
    result["label"] = point["label"]
    if result["status"] != "ok":
        return result
    survey = topocentric_from_lla(point["y"], point["x"], point["z"],
                                  reference["latitude"], reference["longitude"],
                                  reference["altitude"])
    error = [result["point"][i] - survey[i] for i in range(3)]
    result["error_enu_m"] = error
    result["horizontal_m"] = math.hypot(error[0], error[1])
    result["vertical_m"] = error[2]
    return result


def load_reconstruction(opensfm: Path, name: str = "reconstruction.topocentric.json"
                        ) -> tuple[dict, dict, int]:
    """The largest topocentric reconstruction, its reference, and the part count.

    ODM moves the topocentric reconstruction aside and replaces
    ``reconstruction.json`` with projected coordinates once the block is
    georeferenced; the topocentric file is the one in OpenSfM's own frame. A
    reconstruction re-run with ``opensfm reconstruct`` after ODM is topocentric
    again under the plain name, which is what ``name`` selects.
    """
    path = opensfm / name
    if not path.exists():
        raise SystemExit(f"{path} is missing: ODM did not georeference the block")
    parts = json.loads(path.read_text(encoding="utf-8"))
    largest = max(parts, key=lambda part: len(part.get("shots", {})))
    reference = json.loads((opensfm / "reference_lla.json").read_text(encoding="utf-8"))
    return largest, reference, len(parts)


def evaluate(project: Path, holdout_path: Path, mode: str,
             reconstruction_name: str = "reconstruction.topocentric.json") -> dict:
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    reconstruction, reference, parts = load_reconstruction(project / "opensfm",
                                                           reconstruction_name)
    projection_types = sorted({c.get("projection_type") for c in reconstruction["cameras"].values()})
    return {
        "mode": mode,
        "fold": holdout["fold"],
        "check": measure(holdout["check"], reconstruction, reference),
        "control": [measure(p, reconstruction, reference) for p in holdout["control"]],
        "shots_reconstructed": len(reconstruction["shots"]),
        "reconstruction_parts": parts,
        "projection_types": projection_types,
        "cameras": reconstruction["cameras"],
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def rms(values: list[float]) -> float | None:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else None


def summarize(results: list[dict], target_m: float) -> tuple[dict, str]:
    modes: dict[str, dict] = {}
    for result in sorted(results, key=lambda r: (r["mode"], r["fold"])):
        entry = modes.setdefault(result["mode"], {"folds": [], "missing": []})
        check = result["check"]
        if check.get("status") != "ok":
            entry["missing"].append({"fold": result["fold"], "label": check.get("label"),
                                     "status": check.get("status")})
            continue
        control_h = [c["horizontal_m"] for c in result["control"] if c.get("status") == "ok"]
        entry["folds"].append({
            "fold": result["fold"], "label": check["label"],
            "horizontal_m": check["horizontal_m"], "vertical_m": check["vertical_m"],
            "views": check["views"], "reprojection_rms_px": check["reprojection_rms_px"],
            "control_horizontal_rms_m": rms(control_h),
            "shots_reconstructed": result["shots_reconstructed"],
        })
    for entry in modes.values():
        entry["horizontal_rms_m"] = rms([f["horizontal_m"] for f in entry["folds"]])
        entry["vertical_rms_m"] = rms([f["vertical_m"] for f in entry["folds"]])
        h = entry["horizontal_rms_m"]
        entry["meets_target"] = (h is not None and h <= target_m and not entry["missing"])

    lines = ["# OpenDroneMap leave-one-out reference (gcp-rtk)", "",
             f"Target: horizontal check RMS <= {target_m * 100:.1f} cm.", ""]
    for mode, entry in sorted(modes.items()):
        h, v = entry["horizontal_rms_m"], entry["vertical_rms_m"]
        lines.append(f"## {mode}")
        lines.append("")
        lines.append("| Fold | Check point | Horizontal (cm) | Vertical (cm) | Views | Reproj. (px) | Control h RMS (cm) | Shots |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for f in entry["folds"]:
            ctrl = f["control_horizontal_rms_m"]
            lines.append(
                f"| {f['fold']} | {f['label']} | {f['horizontal_m'] * 100:.2f} | "
                f"{f['vertical_m'] * 100:+.2f} | {f['views']} | {f['reprojection_rms_px']:.2f} | "
                f"{'' if ctrl is None else f'{ctrl * 100:.2f}'} | {f['shots_reconstructed']} |")
        for m in entry["missing"]:
            lines.append(f"| {m['fold']} | {m['label']} | — | — | {m['status']} | | | |")
        lines.append("")
        if h is not None:
            lines.append(f"**RMS: {h * 100:.2f} cm horizontal, {v * 100:.2f} cm vertical "
                         f"over {len(entry['folds'])} folds.**")
        lines.append("")
    return {"target_horizontal_m": target_m, "modes": modes}, "\n".join(lines)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch(work: Path, cache: Path) -> None:
    """Download and verify the archive, then lay out the ODM project directory.

    ``cache`` may be the quality bench's restored download cache, so the archive
    is verified (and only fetched when missing) by the bench's own downloader.
    """
    import run_bench  # noqa: PLC0415

    dataset = run_bench.load_manifest()["datasets"][DATASET]
    cache.mkdir(parents=True, exist_ok=True)
    for name, remote in run_bench.downloads_for(dataset):
        run_bench.download(remote, cache / Path(name).name)
    archive = cache / dataset["images_archive"]["download"]
    raw = work / "raw"
    if not (raw.exists() and any(raw.rglob("*"))):
        run_bench.extract_archive(archive, raw)
    # The runner's disk also has to hold the ODM image and its working files.
    archive.unlink()
    found = sorted(raw.glob(dataset["images_archive"]["extract_glob"]))
    if len(found) != dataset["expected_images"]:
        raise SystemExit(f"expected {dataset['expected_images']} images, found {len(found)}")
    images = work / "project" / "images"
    images.mkdir(parents=True, exist_ok=True)
    for path in found:
        shutil.move(str(path), images / path.name)
    for reference in dataset["reference_files"]:
        if reference["kind"] != "gcp_list":
            continue
        source = raw / reference["archive_member"]
        shutil.copy2(source, work / "gcp_list_full.txt")
    print(f"-- {len(found)} images in {images}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download, verify and lay out the ODM project")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--cache", type=Path, help="download cache (default: WORK/cache)")

    p = sub.add_parser("fold", help="write one fold's gcp_list.txt and holdout.json")
    p.add_argument("--gcp-list", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--out-list", type=Path, required=True)
    p.add_argument("--out-holdout", type=Path, required=True)

    p = sub.add_parser("evaluate", help="measure the held-out point in ODM's result")
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--holdout", type=Path, required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--reconstruction", default="reconstruction.topocentric.json",
                   help="file under opensfm/ holding the topocentric reconstruction")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("summarize", help="per-mode RMS over every fold result")
    p.add_argument("--target", type=float, default=0.05)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("results", type=Path, nargs="+")

    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch(args.work, args.cache or args.work / "cache")
    elif args.command == "fold":
        holdout = write_fold(args.gcp_list, args.fold, args.out_list, args.out_holdout)
        print(f"-- fold {args.fold}: check {holdout['check']['label']}, control "
              + ", ".join(p["label"] for p in holdout["control"]))
    elif args.command == "evaluate":
        result = evaluate(args.project, args.holdout, args.mode, args.reconstruction)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        check = result["check"]
        print(json.dumps({k: check.get(k) for k in
                          ("label", "status", "views", "horizontal_m", "vertical_m",
                           "reprojection_rms_px")}))
    elif args.command == "summarize":
        results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
        summary, markdown = summarize(results, args.target)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        args.out_md.write_text(markdown + "\n", encoding="utf-8")
        print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
