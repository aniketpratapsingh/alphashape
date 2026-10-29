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
    # The DEM is gated on the OFFSET-REMOVED spread, not on the raw median: that
    # median is the difference between the two products' vertical datums (our DEM
    # is in WGS84 ellipsoidal heights, the publisher's is whatever its own datum
    # is), and no commit moves it. The raw figures are still reported, with each
    # side's datum named beside them.
    "reference_based.dem.offset_removed.nmad_m": "lower",
    "reference_based.dem.offset_removed.p68_m": "lower",
    "reference_based.dem.offset_removed.p95_m": "lower",
    "reference_based.dem.overlap_fraction": "higher",
    "reference_based.cloud.f_score_5cm": "higher",
    "reference_based.cloud.accuracy_median_m": "lower",
    "reference_based.cloud.completeness_median_m": "lower",
    "reference_free.cameras.cameras_aligned_fraction": "higher",
    # The independent re-projection agrees with the engine's own RMS once the
    # camera type is read the way the writer spells it, so it is gated. If the two
    # ever diverge again the cross-check row says so and this gate catches the move.
    "reference_free.cameras.reprojection.rms_px": "lower",
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


def experiment_params(metrics: dict) -> dict[str, str]:
    """The non-default params a run was dispatched with, if any.

    An empty dict means the run is at the dataset's own settings and is eligible
    to be a regression baseline. A non-empty one means somebody asked for a
    specific configuration to be measured — the census cost, say — and the run is
    evidence, not a reference point: it must never become the baseline the next
    week compares against, and it must never gate.

    Both dispatch inputs are folded into one mapping, prefixed by the task they
    belong to, because the question the comparison asks is "were these two runs
    the same configuration" and either input alone changes the answer. A run with
    only ``dem_params`` must not compare as if it were a control.
    """
    combined: dict[str, str] = {}
    for key, prefix in (("depth_params", "BuildDepthMaps"), ("dem_params", "BuildDem")):
        value = metrics.get(key)
        if isinstance(value, dict):
            for name, item in value.items():
                combined[f"{prefix}.{name}"] = str(item)
    return combined


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
        elif "relative" in entry:
            # A metric that is not a fraction of a total can still be gated
            # proportionally — a resolution ratio is dimensionless, so "15% worse
            # than last time" is the meaningful tolerance, not "0.15 coarser".
            limit = float(entry["relative"]) * abs(before)
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


def _wrap_list(values: list[str], per_chunk: int = 6) -> str:
    """A long list of file names as comma-separated groups, inside one table cell.

    A markdown table cell cannot contain a raw newline, so the groups are joined
    with an explicit ``<br>``.
    """
    groups = [
        ", ".join(values[start:start + per_chunk])
        for start in range(0, len(values), per_chunk)
    ]
    return "<br>".join(groups)


def _crs_rows(provenance: Any) -> list[str]:
    """Table rows saying whether a reference had to be reprojected, and from what.

    A reader comparing this week's number with last week's needs to know the
    comparison was made in the same frame; a silent CRS change would look like a
    metric movement.
    """
    if not isinstance(provenance, dict):
        return []
    rows = []
    if provenance.get("reprojected"):
        rows.append(
            f"| CRS | reference EPSG:{provenance.get('reference_epsg')} reprojected "
            f"to EPSG:{provenance.get('our_epsg')} |"
        )
    elif provenance.get("our_epsg") is not None:
        rows.append(f"| CRS | both EPSG:{provenance.get('our_epsg')} |")
    elif provenance.get("note"):
        rows.append(f"| CRS | _{provenance['note']}_ |")
    else:
        rows.append("| CRS | not declared by either product |")
    return rows


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
    experiment = experiment_params(current)

    lines: list[str] = []
    lines.append(f"# Quality benchmark: {manifest.get('title') or current.get('dataset', '?')}")
    lines.append("")

    if experiment:
        # An experimental dispatch is never a pass/fail verdict: it exists to
        # measure a configuration against the control run, not to defend a
        # baseline. Saying so up front is what keeps the numbers below from being
        # read as "the code regressed".
        given = ", ".join(f"`{key}={value}`" for key, value in sorted(experiment.items()))
        lines.append(f"**Verdict: experiment** ({given}) — reported, not gated, not baselined")
    elif previous is None:
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
    if manifest.get("images_available") and manifest.get("images_available") != manifest.get("images"):
        selection = manifest.get("image_selection") or {}
        lines.append(
            f"| Images | {manifest['images']} of {manifest['images_available']} "
            f"({selection.get('mode', 'subset')}) |"
        )
    else:
        lines.append(f"| Images | {manifest.get('images') or '-'} |")
    if manifest.get("nominal_gsd_m"):
        lines.append(f"| Nominal GSD | {_fmt(manifest['nominal_gsd_m'])} m |")
    selection = manifest.get("image_selection") or {}
    if selection and selection.get("mode") not in (None, "all"):
        # The actual selected files. A first..last range used to be printed, but
        # the subset is a nearest-to-centroid cluster and is not contiguous in file
        # order, so a range claimed images the run never used.
        chosen = selection.get("images") or []
        if chosen:
            lines.append(f"| Images used | {_wrap_list(chosen)} |")
        else:  # an older metrics JSON with only first/last
            lines.append(
                f"| Image block | {selection.get('first')} .. {selection.get('last')} |"
            )
    lines.append(
        f"| Licence | {licence.get('spdx', '-')}"
        + (f" — [terms]({licence['url']})" if licence.get("url") else "")
        + " |"
    )
    if licence.get("redistribution"):
        lines.append(f"| Redistribution | {licence['redistribution']} |")
    lines.append(f"| Manifest hash | `{current.get('manifest_hash', '-')}` |")
    if experiment:
        given = ", ".join(f"`{key}={value}`" for key, value in sorted(experiment.items()))
        lines.append(f"| Dispatch parameters | {given} |")
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

    # A run that used non-default parameters is not comparable with one that did
    # not, so it is called out above the numbers rather than left in the log.
    overrides = timing.get("param_overrides") or {}
    if overrides:
        rendered = "; ".join(
            f"`{task}` " + ", ".join(f"`{key}={value}`" for key, value in values.items())
            for task, values in overrides.items()
        )
        lines.append(f"**Not at defaults:** {rendered}. Results are comparable with other")
        lines.append("runs of this dataset, not with runs at the engine's defaults.")
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
        # The ratio is what is gated: effective resolution divided by the mosaic's
        # OWN cell size. The absolute metres follow from it and the cell size, so a
        # change in the resolution this run asked for does not move the ratio.
        unmeasured = (
            ortho.get("effective_measured") is False
            or ortho.get("textured") is False
            or ortho.get("effective_resolution_ratio") is None
        )
        if unmeasured:
            lines.append(
                "| Effective resolution | _unmeasured — "
                f"{ortho.get('effective_unmeasured_reason') or 'no usable texture'}_ |"
            )
        else:
            lines.append(
                f"| Effective resolution | {_fmt(ortho.get('effective_resolution_ratio'))} x cell size |"
            )
            if ortho.get("effective_resolution_m") is not None:
                lines.append(
                    f"| Effective resolution | {_fmt(ortho.get('effective_resolution_m'))} m "
                    f"(cell {_fmt(ortho.get('pixel_size_m'))} m) |"
                )
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
        reprojection = cameras.get("reprojection") or {}
        engine = cameras.get("engine_report") or {}

        # The engine's own bundle-adjustment RMS is the authoritative figure: it is
        # computed inside the fit, on the same observations, by the code that did
        # the work. It is the headline.
        if engine.get("rms_px") is not None:
            lines.append(f"| Reprojection RMS (engine BA) | {_fmt(engine['rms_px'])} px |")
        if engine.get("registered") is not None:
            lines.append(
                f"| Engine registered | {engine['registered']} / {engine['offered']} cameras |"
            )

        # This module re-projects independently from the saved project's own
        # poses and calibration. It is a cross-check on the project writer and on
        # this metric: after the camera-type comparison was made case-insensitive
        # (the writer emits "frame", not "Frame") it agrees with the engine's own
        # bundle-adjustment RMS, so it is reported and gated like any other
        # number. ``agrees`` is kept as a diagnostic: if the two ever diverge
        # again the report says so instead of silently headlining a wrong figure.
        if reprojection and "skipped" not in reprojection:
            mine = reprojection.get("rms_px")
            theirs = engine.get("rms_px")
            lines.append(f"| Reprojection RMS (independent) | {_fmt(mine)} px |")
            lines.append(f"| Observations | {reprojection.get('pairs')} |")
            if mine is not None and theirs is not None:
                # Agreement is "within 1 px, or within 50% of the engine's own
                # figure". The two are not the same computation — the engine's RMS
                # is taken inside the fit over the observations it kept, this one
                # re-projects every stored track from the saved poses — so a small
                # difference is expected, but an order of magnitude is a bug.
                agrees = abs(mine - theirs) <= max(1.0, 0.5 * abs(theirs))
                lines.append(
                    f"| Cross-check vs engine BA | "
                    f"{'agrees' if agrees else '**disagrees** — investigate the writer before trusting either'} |"
                )
                if not agrees:
                    spread = reprojection.get("worst_cameras") or []
                    if spread:
                        lines.append(
                            "| Worst camera medians | "
                            + ", ".join(
                                f"#{entry['camera']} {_fmt(entry['median_px'])} px"
                                for entry in spread
                            )
                            + " |"
                        )
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
            lines.extend(_crs_rows(ref_ortho.get("crs")))
        else:
            lines.append(f"- Orthomosaic vs reference: _{ref_ortho.get('reason', 'not comparable')}_")
        lines.append("")

        ref_dem = based.get("dem")
        reason = _skipped(ref_dem)
        if reason:
            lines.append(f"- DSM vs reference: _{reason}_")
        else:
            # Two families of number. The raw offset is a datum difference, not an
            # error the reconstruction can fix, so it is reported with both datums
            # named and the gate compares the offset-removed spread instead.
            lines.append("| DSM vs reference | Value |")
            lines.append("|---|---:|")
            lines.append(f"| ΔZ median (raw offset) | {_fmt(ref_dem.get('median_m'))} m |")
            lines.append(f"| |ΔZ| p68 (raw) | {_fmt(ref_dem.get('p68_m'))} m |")
            lines.append(f"| |ΔZ| p95 (raw) | {_fmt(ref_dem.get('p95_m'))} m |")
            removed = ref_dem.get("offset_removed") or {}
            lines.append(
                f"| ΔZ NMAD, offset removed | {_fmt(removed.get('nmad_m'))} m |"
            )
            lines.append(
                f"| |ΔZ| p68, offset removed | {_fmt(removed.get('p68_m'))} m |"
            )
            lines.append(
                f"| |ΔZ| p95, offset removed | {_fmt(removed.get('p95_m'))} m |"
            )
            lines.append(f"| Cells compared | {ref_dem.get('cells')} |")
            lines.append(f"| Overlap | {_fmt(ref_dem.get('overlap_fraction'))} |")
            lines.extend(_crs_rows(ref_dem.get("crs")))
            datums = ref_dem.get("vertical_datum") or {}
            lines.append(
                f"| Vertical datum (ours) | {datums.get('ours') or 'unknown — not recorded'} |"
            )
            lines.append(
                f"| Vertical datum (reference) | {datums.get('reference') or 'unknown — not recorded'} |"
            )
            lines.append(
                "| Gated on | offset-removed spread; the raw median is the datum "
                "difference |"
            )
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
        previous_experiment = experiment_params(previous)
        different_settings = previous_experiment != experiment
        if previous_manifest and previous_manifest != manifest.get("title"):
            lines.append(
                f"> The previous run was for **{previous_manifest}**, not this dataset. "
                "Comparing across datasets is meaningless; the table is shown for the "
                "record only and the verdict ignores it."
            )
            lines.append("")
        elif different_settings:
            # The comparison below is still useful — it is how an experiment is
            # read against the control — but it is not a regression verdict, and
            # the reader has to know that before they read a moving metric.
            before = (
                ", ".join(f"`{k}={v}`" for k, v in sorted(previous_experiment.items()))
                or "defaults"
            )
            now = ", ".join(f"`{k}={v}`" for k, v in sorted(experiment.items())) or "defaults"
            lines.append(
                f"> The previous run used {before}; this one used {now}. The two are "
                "different configurations, so no metric is gated here — the table is "
                "shown for comparison only."
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
            if different_settings:
                lines.append(
                    "No metric was gated: the two runs used different dispatch "
                    "parameters. Compare the tables above by hand."
                )
            else:
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

        # Both reads are cell-bounded: a survey orthomosaic is hundreds of
        # megapixels, and a thumbnail beside a crop does not need it at full size.
        # The reference is additionally windowed to our extent and reprojected onto
        # our CRS when the two differ — a side-by-side of two different projections
        # stretches one of them against a grid it does not belong to.
        ours = metrics.read_geotiff(ortho_path, max_cells=1_000_000)
        theirs, _ = metrics.aligned_reference(
            str(reference), ours, bounds=ours.extent(), max_cells=1_000_000
        )
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
    experiment = experiment_params(current)
    if previous is not None:
        # A different dataset is not a regression baseline; comparing across
        # datasets would compare a Comoros beach against a Swiss rock face.
        same_dataset = (previous.get("dataset") or "") == (current.get("dataset") or "")
        # Nor is a run that used different parameters: a baseline restored from a
        # configuration somebody was evaluating says nothing about this one.
        same_settings = experiment_params(previous) == experiment
        if same_dataset and same_settings:
            bad, good = regressions(current, previous, thresholds)
        elif same_dataset:
            print(
                "note: the previous run used different dispatch parameters; "
                "the comparison is report-only",
                file=sys.stderr,
            )

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
        # An experimental run measures a configuration; it does not defend one.
        # Its report still names every metric that moved, but it never fails the
        # job, because a red check on "census was worse this week" would block the
        # branch that is being evaluated rather than inform it.
        if args.fail_on_regression and not experiment:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
