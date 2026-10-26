#!/usr/bin/env python3
"""Turn a run's metrics JSON into a Markdown report, a side-by-side PNG and a
regression verdict against the previous run.

Three outputs, in the order a reader wants them:

  * **Verdict** — one line: passed, or the named metrics that regressed and by how
    much against ``thresholds.json``. A run never fails on an absolute value;
    only a move away from the previous run fails it. Absolute numbers on real
    data move with the scene, the season and the dataset publisher's own
    processing, so gating on them would make the job red for reasons no commit
    caused.
  * **Markdown** — the tables the workflow writes to its job summary, plus a
    provenance block naming the dataset, its licence and its SHA-256. The licence
    line is there so a reader can see why a number exists and what may be done
    with the data behind it.
  * **PNG** — our orthomosaic beside a crop of the reference, when the dataset
    has one. A picture catches a mosaic that is globally right but locally
    striped, which no single number does.

Usage:

    python3 tools/bench/report.py --current build/bench/metrics.json \
        --previous build/bench/previous.json --out build/bench/report.md \
        --png build/bench/compare.png

Every argument except ``--current`` is optional. A missing ``--previous`` is the
first run: the report says so and the verdict is always "pass", which is what
keeps the very first weekly run green rather than failing on missing history.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_THRESHOLDS = HERE / "thresholds.json"

# How each metric's movement reads. Getting this wrong inverts the whole verdict,
# so it is data rather than a chain of ifs: "higher" means a larger value is
# better, "lower" means a smaller value is better, and a metric not listed here is
# reported but never gated (a timing, a count, a diagnostic).
DIRECTION = {
    "reference_free.ortho.covered_fraction": "higher",
    "reference_free.ortho.hole_fraction": "lower",
    "reference_free.ortho.holes_inside_hull": "lower",
    "reference_free.ortho.largest_hole_m2": "lower",
    "reference_free.ortho.effective_resolution_ratio": "lower",
    "reference_free.ortho.gradient_energy": "higher",
    "reference_free.dem.nodata_fraction": "lower",
    "reference_based.ortho.offset_m": "lower",
    "reference_based.ortho.overlap_fraction": "higher",
    # A signed bias is judged by its MAGNITUDE: either sign moving away from
    # zero is worse, so the comparison is on |value| rather than on the delta.
    "reference_based.dem.median_m": "magnitude",
    "reference_based.dem.nmad_m": "lower",
    "reference_based.dem.p68_m": "lower",
    "reference_based.dem.p95_m": "lower",
    "reference_based.dem.overlap_fraction": "higher",
    "reference_based.cloud.f_score_5cm": "higher",
    "reference_based.cloud.accuracy_median_m": "lower",
    "reference_based.cloud.completeness_median_m": "lower",
    "reference_free.cameras.cameras_aligned_fraction": "higher",
    "reference_free.cameras.reprojection.rms_px": "lower",
    "reference_free.cameras.reprojection.median_px": "lower",
}

# Metrics whose value is a fraction of a total and so is gated by a relative drop
# rather than a fixed move. A 1% coverage drop from 0.99 is noise; a 1% drop from
# 0.10 is most of the coverage gone.
FRACTION_PREFIXES = (
    "reference_free.ortho.covered_fraction",
    "reference_free.ortho.hole_fraction",
    "reference_free.dem.nodata_fraction",
    "reference_based.ortho.overlap_fraction",
    "reference_based.dem.overlap_fraction",
    "reference_based.cloud.f_score_5cm",
    "reference_free.cameras.cameras_aligned_fraction",
)


def flatten(metrics: dict, prefix: str = "") -> dict[str, float]:
    """Every numeric leaf, keyed by its dotted path.

    Nested tolerance dicts come out as ``...accuracy.0.05``, which is exactly the
    key a threshold wants. Anything non-numeric (a ``skipped`` reason, a string
    EPSG) is left out: those are reported in the tables by hand, not gated.
    """
    out: dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for key, value in metrics.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, path))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)) and value is not None:
            if math.isfinite(float(value)):
                out[path] = float(value)
    return out


def direction(path: str, thresholds: dict) -> str | None:
    """How to read a metric's movement, or None when it is not gated.

    An explicit entry in the thresholds file wins over the built-in table, so a
    dataset whose metric genuinely runs the other way can be configured without a
    code change.
    """
    override = thresholds.get("direction", {}).get(path)
    if override:
        return override
    return DIRECTION.get(path)


def is_fraction(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in FRACTION_PREFIXES)


def regressions(
    current: dict, previous: dict, thresholds: dict
) -> tuple[list[dict], list[dict]]:
    """Split the metric moves into (regressions, improvements).

    A metric is a regression when it moved the wrong way by more than its
    tolerance. The tolerance for a fraction is ``relative`` x the previous value;
    for anything else it is the metric's ``absolute`` entry, so a length in metres
    is gated in metres rather than in per-cent. A ``magnitude`` metric (a signed
    bias that should be zero) is judged on |value|, so either sign growing is a
    regression.

    A metric absent from the current run (a skipped reference, a dataset change)
    is never a regression: there is nothing to compare, and inventing a failure
    from a missing number would hide the real cause.
    """
    current_flat = flatten(current)
    previous_flat = flatten(previous)
    tolerance = thresholds.get("tolerances", {})
    default_absolute = float(tolerance.get("default_absolute", 0.0))
    default_relative = float(tolerance.get("default_relative", 0.0))

    bad: list[dict] = []
    good: list[dict] = []
    for path, value in current_flat.items():
        if path not in previous_flat:
            continue
        how = direction(path, thresholds)
        if how is None:
            continue
        before = previous_flat[path]
        delta = value - before

        entry = tolerance.get(path, {})
        if not isinstance(entry, dict):
            entry = {}
        if is_fraction(path):
            limit = float(entry.get("relative", default_relative)) * abs(before)
        else:
            limit = float(entry.get("absolute", default_absolute))
        # A zero threshold means "report it, never fail on it": a metric with no
        # configured tolerance is a number to watch, not a gate.
        if limit <= 0.0:
            continue

        if how == "higher":
            moved_wrong = delta < -limit
            moved_right = delta > limit
        elif how == "lower":
            moved_wrong = delta > limit
            moved_right = delta < -limit
        elif how == "magnitude":
            # A signed quantity that should sit at zero: what matters is whether
            # its size grew, and by how much beyond the tolerance.
            change = abs(value) - abs(before)
            moved_wrong = change > limit
            moved_right = change < -limit
        else:  # pragma: no cover - a typo in the config
            continue

        record = {
            "metric": path,
            "previous": before,
            "current": value,
            "delta": delta,
            "limit": limit,
            "direction": how,
        }
        if moved_wrong:
            bad.append(record)
        elif moved_right:
            good.append(record)
    return bad, good


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a metric for the tables, or ``-`` when it is not a number."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.{digits}g}"
    return str(value)


def _skipped(section: Any) -> str | None:
    if isinstance(section, dict) and "skipped" in section:
        return str(section["skipped"])
    return None


def markdown(
    current: dict,
    previous: dict | None,
    bad: list[dict],
    good: list[dict],
    *,
    thresholds: dict,
    png_name: str | None,
) -> str:
    """The report body. Pure formatting: every number comes from ``current``."""
    free = current.get("reference_free", {}) or {}
    based = current.get("reference_based", {}) or {}
    timing = current.get("timing", {}) or {}
    manifest = current.get("dataset_manifest", {}) or {}
    licence = current.get("licence", {}) or {}

    lines: list[str] = []
    lines.append(f"# Quality benchmark: {manifest.get('title') or current.get('dataset', '?')}")
    lines.append("")

    if previous is None:
        lines.append("**Verdict: pass** (first run for this dataset; no history to compare)")
    elif bad:
        lines.append(f"**Verdict: FAIL** — {len(bad)} metric(s) regressed")
    else:
        lines.append(f"**Verdict: pass** ({len(good)} improved, none regressed)")
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Dataset | {manifest.get('title') or current.get('dataset', '?')} |")
    if manifest.get("doi"):
        lines.append(f"| DOI | [{manifest['doi']}](https://doi.org/{manifest['doi']}) |")
    lines.append(f"| Flight | {manifest.get('flight') or '-'} |")
    lines.append(f"| Images | {manifest.get('images') or '-'} |")
    if manifest.get("nominal_gsd_m"):
        lines.append(f"| Nominal GSD | {_fmt(manifest['nominal_gsd_m'])} m |")
    lines.append(
        f"| Licence | {licence.get('spdx', '-')}"
        + (f" — [terms]({licence['url']})" if licence.get("url") else "")
        + " |"
    )
    if licence.get("redistribution"):
        lines.append(f"| Redistribution | {licence['redistribution']} |")
    lines.append(f"| Manifest hash | `{current.get('manifest_hash', '-')}` |")
    lines.append("")

    # -- products ----------------------------------------------------------
    lines.append("## Products")
    lines.append("")
    lines.append("| Stage | Time (s) |")
    lines.append("|---|---:|")
    for stage, seconds in (timing.get("stages_s") or {}).items():
        lines.append(f"| {stage} | {_fmt(seconds)} |")
    if timing.get("total_s") is not None:
        lines.append(f"| **total** | **{_fmt(timing['total_s'])}** |")
    lines.append("")
    if timing.get("peak_rss_mb") is not None:
        lines.append(f"Peak resident memory: **{_fmt(timing['peak_rss_mb'])} MB**")
        lines.append("")

    # -- reference-free ----------------------------------------------------
    lines.append("## Reference-free metrics")
    lines.append("")
    ortho = free.get("ortho")
    reason = _skipped(ortho)
    if reason:
        lines.append(f"- Orthomosaic: _{reason}_")
    else:
        lines.append("| Orthomosaic | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Grid | {ortho.get('width')} x {ortho.get('height')} px |")
        lines.append(f"| Pixel size | {_fmt(ortho.get('pixel_size_m'))} m |")
        lines.append(f"| Covered fraction | {_fmt(ortho.get('covered_fraction'))} |")
        lines.append(f"| Holes inside hull | {ortho.get('holes_inside_hull')} |")
        lines.append(f"| Hole area | {_fmt(ortho.get('hole_area_m2'))} m2 |")
        lines.append(f"| Hole fraction | {_fmt(ortho.get('hole_fraction'))} |")
        lines.append(f"| Largest hole | {_fmt(ortho.get('largest_hole_m2'))} m2 |")
        lines.append(f"| Effective GSD | {_fmt(ortho.get('effective_gsd_m'))} m |")
        lines.append(f"| Effective/nominal | {_fmt(ortho.get('effective_resolution_ratio'))} |")
        if ortho.get("textured") is False:
            lines.append("| Textured | _no usable texture; resolution not scored_ |")
        if ortho.get("epsg") is not None:
            lines.append(f"| EPSG | {ortho['epsg']} |")
    lines.append("")

    dem = free.get("dem")
    reason = _skipped(dem)
    if reason:
        lines.append(f"- DEM: _{reason}_")
    else:
        lines.append("| DEM | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Grid | {dem.get('width')} x {dem.get('height')} px |")
        lines.append(f"| Pixel size | {_fmt(dem.get('pixel_size_m'))} m |")
        lines.append(f"| Nodata fraction | {_fmt(dem.get('nodata_fraction'))} |")
        if dem.get("height_min_m") is not None:
            lines.append(
                f"| Height range | {_fmt(dem['height_min_m'])} .. {_fmt(dem.get('height_max_m'))} m |"
            )
    lines.append("")

    cameras = free.get("cameras")
    reason = _skipped(cameras)
    if reason:
        lines.append(f"- Cameras: _{reason}_")
    else:
        lines.append("| Cameras | Value |")
        lines.append("|---|---:|")
        lines.append(
            f"| Aligned | {cameras.get('cameras_aligned')} / {cameras.get('cameras_total')} "
            f"({_fmt(100.0 * cameras.get('cameras_aligned_fraction', 0.0), 3)}%) |"
        )
        reprojection = cameras.get("reprojection")
        if reprojection and "skipped" not in reprojection:
            lines.append(f"| Reprojection RMS | {_fmt(reprojection.get('rms_px'))} px |")
            lines.append(f"| Reprojection median | {_fmt(reprojection.get('median_px'))} px |")
            lines.append(f"| Observations | {reprojection.get('pairs')} |")
    lines.append("")

    # -- reference-based ---------------------------------------------------
    lines.append("## Reference-based metrics")
    lines.append("")
    if not any(not _skipped(based.get(key)) for key in ("ortho", "dem", "cloud")):
        lines.append("_This dataset publishes no reference product, so no reference metric ran._")
        lines.append("")
    else:
        ref_ortho = based.get("ortho")
        reason = _skipped(ref_ortho)
        if reason:
            lines.append(f"- Orthomosaic vs reference: _{reason}_")
        elif ref_ortho.get("comparable"):
            lines.append("| Orthomosaic vs reference | Value |")
            lines.append("|---|---:|")
            lines.append(f"| Offset | {_fmt(ref_ortho.get('offset_m'))} m |")
            lines.append(f"| dx / dy | {_fmt(ref_ortho.get('dx_m'))} / {_fmt(ref_ortho.get('dy_m'))} m |")
            lines.append(f"| Overlap | {_fmt(ref_ortho.get('overlap_fraction'))} |")
            lines.append(f"| Comparison grid | {_fmt(ref_ortho.get('grid_pixel_m'))} m |")
        else:
            lines.append(f"- Orthomosaic vs reference: _{ref_ortho.get('reason', 'not comparable')}_")
        lines.append("")

        ref_dem = based.get("dem")
        reason = _skipped(ref_dem)
        if reason:
            lines.append(f"- DSM vs reference: _{reason}_")
        else:
            lines.append("| DSM vs reference | Value |")
            lines.append("|---|---:|")
            lines.append(f"| ΔZ median | {_fmt(ref_dem.get('median_m'))} m |")
            lines.append(f"| ΔZ NMAD | {_fmt(ref_dem.get('nmad_m'))} m |")
            lines.append(f"| |ΔZ| p68 | {_fmt(ref_dem.get('p68_m'))} m |")
            lines.append(f"| |ΔZ| p95 | {_fmt(ref_dem.get('p95_m'))} m |")
            lines.append(f"| Cells compared | {ref_dem.get('cells')} |")
            lines.append(f"| Overlap | {_fmt(ref_dem.get('overlap_fraction'))} |")
        lines.append("")

        cloud = based.get("cloud")
        reason = _skipped(cloud)
        if reason:
            lines.append(f"- Point cloud vs laser: _{reason}_")
        else:
            lines.append("| Point cloud vs laser | 2 cm | 5 cm | 10 cm |")
            lines.append("|---|---:|---:|---:|")
            accuracy = cloud.get("accuracy", {})
            completeness = cloud.get("completeness", {})
            lines.append(
                "| Accuracy | "
                + " | ".join(_fmt(accuracy.get(k)) for k in ("0.02", "0.05", "0.1"))
                + " |"
            )
            lines.append(
                "| Completeness | "
                + " | ".join(_fmt(completeness.get(k)) for k in ("0.02", "0.05", "0.1"))
                + " |"
            )
            lines.append("")
            lines.append(
                f"Median distance: accuracy {_fmt(cloud.get('accuracy_median_m'))} m, "
                f"completeness {_fmt(cloud.get('completeness_median_m'))} m; "
                f"F-score @5 cm {_fmt(cloud.get('f_score_5cm'))}."
            )
            # The registration has to be visible: it is a fitted similarity, so a
            # scale that is nowhere near the expected one (or an RMS that never
            # settled) says the score is about the alignment rather than the
            # reconstruction.
            alignment = cloud.get("alignment")
            if isinstance(alignment, dict):
                lines.append("")
                lines.append(
                    f"Registered by similarity ICP: scale {_fmt(alignment.get('scale'))}, "
                    f"{alignment.get('iterations')} iterations, "
                    f"RMS {_fmt(alignment.get('rms_m'))} m, "
                    f"{'converged' if alignment.get('converged') else 'stopped at the iteration cap'}."
                )
        lines.append("")

    # -- regressions -------------------------------------------------------
    lines.append("## Comparison with the previous run")
    lines.append("")
    if previous is None:
        lines.append("No previous run was supplied.")
    else:
        previous_manifest = (previous.get("dataset_manifest") or {}).get("title")
        if previous_manifest and previous_manifest != manifest.get("title"):
            lines.append(
                f"> The previous run was for **{previous_manifest}**, not this dataset. "
                "Comparing across datasets is meaningless; the table is shown for the "
                "record only and the verdict ignores it."
            )
            lines.append("")
        if bad:
            lines.append("| Regressed metric | Previous | Current | Change | Limit |")
            lines.append("|---|---:|---:|---:|---:|")
            for record in sorted(bad, key=lambda r: -abs(r["delta"])):
                lines.append(
                    f"| `{record['metric']}` | {_fmt(record['previous'])} | "
                    f"{_fmt(record['current'])} | {_fmt(record['delta'])} | "
                    f"{_fmt(record['limit'])} |"
                )
            lines.append("")
        if good:
            lines.append(
                f"{len(good)} metric(s) moved the right way beyond their tolerance: "
                + ", ".join(f"`{r['metric']}`" for r in sorted(good, key=lambda r: r["metric"]))
                + "."
            )
            lines.append("")
        if not bad and not good:
            lines.append("No metric moved beyond its tolerance.")
            lines.append("")

    if png_name:
        lines.append("## Reference comparison")
        lines.append("")
        lines.append(f"![Our orthomosaic beside the reference crop]({png_name})")
        lines.append("")
        lines.append(
            "Left: this run's orthomosaic. Right: a crop of the dataset's reference, "
            "resampled onto the same grid. A seam or a tone step that no table shows "
            "is visible here."
        )
        lines.append("")

    thresholds_note = thresholds.get("comment")
    if thresholds_note:
        lines.append("---")
        lines.append("")
        lines.append(f"Regression policy: {thresholds_note}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Side-by-side PNG
# ---------------------------------------------------------------------------


def comparison_png(current: dict, metrics_json_path: Path, destination: Path) -> str | None:
    """Write our ortho beside a reference crop; return the file name or None.

    Needs the run's product paths, which are in the metrics JSON under ``timing``,
    and the reference path in the manifest. Both are re-read here so the report is
    a pure function of the files that already exist: nothing is recomputed and no
    network access happens after the pipeline.
    """
    try:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        import metrics  # noqa: PLC0415
        import run_bench  # noqa: PLC0415
    except ImportError:
        return None

    products = (current.get("timing") or {}).get("products") or {}
    ortho_path = products.get("ortho.tif")
    if not ortho_path or not Path(ortho_path).exists():
        return None

    try:
        manifest = run_bench.load_manifest()
        dataset = manifest["datasets"].get(current.get("dataset"))
        if dataset is None:
            return None
        reference_path = next(
            (ref for ref in dataset["reference_files"] if ref["kind"] == "orthomosaic"),
            None,
        )
        if reference_path is None:
            return None
        work = Path(metrics_json_path).resolve().parent
        reference = work / reference_path["path"]
        if not reference.exists():
            return None

        ours = metrics.read_geotiff(ortho_path)
        theirs = metrics.read_geotiff(str(reference))
        ox, oy, pixel, width, height = metrics.common_grid(ours, theirs, max_cells=1_000_000)
        a, a_valid = metrics.resample_to_grid(
            ours, origin_x=ox, origin_y=oy, pixel_size=pixel, width=width, height=height
        )
        b, b_valid = metrics.resample_to_grid(
            theirs, origin_x=ox, origin_y=oy, pixel_size=pixel, width=width, height=height
        )
        both = a_valid & b_valid
        if not both.any():
            return None

        # Crop to the middle half of the shared area: the border of a mosaic is
        # where the geometry is weakest on both sides, and a border crop would
        # make every run look worse than the interior it actually got right.
        rows, cols = np.nonzero(both)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        rm = r0 + (r1 - r0) // 4
        rx = r1 - (r1 - r0) // 4
        cm = c0 + (c1 - c0) // 4
        cx = c1 - (c1 - c0) // 4
        a = a[rm:rx, cm:cx, : min(3, a.shape[2])]
        b = b[rm:rx, cm:cx, : min(3, b.shape[2])]
        if a.size == 0 or b.size == 0:
            return None

        def to_uint8(array: np.ndarray) -> np.ndarray:
            array = np.nan_to_num(array, nan=0.0)
            peak = float(array.max()) if array.size else 0.0
            if peak > 1.5:
                array = array / (255.0 if peak <= 255.0 else peak)
            return np.clip(array * 255.0, 0, 255).astype(np.uint8)

        left = Image.fromarray(to_uint8(a))
        right = Image.fromarray(to_uint8(b))
        gap = 8
        canvas = Image.new("RGB", (left.width + gap + right.width, max(left.height, right.height)), 0)
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width + gap, 0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
        return destination.name
    except Exception as exc:  # noqa: BLE001 - a picture is a nice-to-have
        print(f"warning: comparison PNG not written: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", required=True, help="this run's metrics JSON")
    parser.add_argument("--previous", help="the previous run's metrics JSON")
    parser.add_argument("--out", help="write the Markdown report here")
    parser.add_argument("--png", help="write the side-by-side comparison here")
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 when a metric regressed (the workflow does this)",
    )
    args = parser.parse_args(argv)

    current_path = Path(args.current)
    with open(current_path, "r", encoding="utf-8") as handle:
        current = json.load(handle)

    previous = None
    if args.previous:
        previous_path = Path(args.previous)
        if previous_path.exists():
            with open(previous_path, "r", encoding="utf-8") as handle:
                previous = json.load(handle)

    with open(args.thresholds, "r", encoding="utf-8") as handle:
        thresholds = json.load(handle)

    bad: list[dict] = []
    good: list[dict] = []
    if previous is not None:
        # A different dataset is not a regression baseline; comparing across
        # datasets would compare a Comoros beach against a Swiss rock face.
        same_dataset = (previous.get("dataset") or "") == (current.get("dataset") or "")
        if same_dataset:
            bad, good = regressions(current, previous, thresholds)

    png_name = None
    if args.png:
        png_name = comparison_png(current, current_path, Path(args.png))

    report = markdown(
        current, previous, bad, good, thresholds=thresholds, png_name=png_name
    )

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")
        print(f"wrote {destination}")
    else:
        print(report)

    if bad:
        print(f"::error::{len(bad)} metric(s) regressed against the previous run", file=sys.stderr)
        for record in bad:
            print(
                f"  {record['metric']}: {record['previous']:.6g} -> {record['current']:.6g} "
                f"(limit {record['limit']:.6g})",
                file=sys.stderr,
            )
        if args.fail_on_regression:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
