#!/usr/bin/env python3
"""Unit tests for the report renderer and its regression verdict.

The regression logic is the part of the bench that decides whether a weekly run is
green, so it is tested against synthetic metrics dicts: a known move in a known
direction with a known tolerance has an unambiguous verdict.

Run with plain unittest (no pytest):

    python3 -m unittest discover -s tools/bench/tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import report  # noqa: E402


def metrics(**overrides):
    """A minimal metrics dict shaped like run_bench's output."""
    base = {
        "dataset": "synthetic",
        "manifest_hash": "deadbeefdeadbeef",
        "licence": {"spdx": "CC0-1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"},
        "dataset_manifest": {
            "title": "Synthetic scene",
            "flight": "nadir",
            "images": 12,
            "nominal_gsd_m": 0.02,
        },
        "reference_free": {
            "ortho": {
                "width": 100,
                "height": 100,
                "pixel_size_m": 0.02,
                "covered_fraction": 0.95,
                "holes_inside_hull": 2,
                "hole_area_m2": 1.0,
                "hole_fraction": 0.01,
                "largest_hole_m2": 0.5,
                "effective_resolution_m": 0.024,
                "effective_resolution_ratio": 1.2,
                "textured": True,
                "epsg": 32632,
            },
            "dem": {
                "width": 50,
                "height": 50,
                "pixel_size_m": 0.05,
                "nodata_fraction": 0.02,
                "height_min_m": 1.0,
                "height_max_m": 9.0,
            },
            "cameras": {
                "cameras_total": 12,
                "cameras_aligned": 12,
                "cameras_aligned_fraction": 1.0,
                "reprojection": {"pairs": 500, "rms_px": 0.6, "median_px": 0.5},
            },
        },
        "reference_based": {
            "ortho": {"comparable": True, "offset_m": 0.04, "dx_m": 0.03, "dy_m": 0.02,
                      "overlap_fraction": 0.9, "grid_pixel_m": 0.02},
            "dem": {"median_m": 0.02, "nmad_m": 0.05, "p68_m": 0.06, "p95_m": 0.11,
                    "cells": 1000, "overlap_fraction": 0.8,
                    "offset_removed": {"nmad_m": 0.05, "p68_m": 0.04, "p95_m": 0.09},
                    "vertical_datum": {"ours": "WGS 84 ellipsoidal",
                                       "reference": None}},
            "cloud": {"accuracy": {"0.02": 0.7, "0.05": 0.95, "0.1": 0.99},
                      "completeness": {"0.02": 0.6, "0.05": 0.9, "0.1": 0.98},
                      "f_score_5cm": 0.92, "accuracy_median_m": 0.012,
                      "completeness_median_m": 0.015},
        },
        "timing": {
            "stages_s": {"AddPhotos": 2.0, "MatchPhotos": 60.0},
            "total_s": 62.0,
            "peak_rss_mb": 800.0,
            "products": {},
        },
    }
    for path, value in overrides.items():
        section, _, key = path.partition(".")
        if key:
            base[section][key] = value
        else:
            base[section] = value
    return base


def thresholds():
    return {
        "tolerances": {
            "default_absolute": 0.0,
            "default_relative": 0.0,
            "reference_free.ortho.covered_fraction": {"relative": 0.02},
            "reference_free.ortho.holes_inside_hull": {"absolute": 3},
            "reference_based.ortho.offset_m": {"absolute": 0.05},
            "reference_based.dem.median_m": {"absolute": 0.03},
            "reference_based.cloud.f_score_5cm": {"relative": 0.03},
        }
    }


class FlattenTest(unittest.TestCase):
    def test_numeric_leaves_get_dotted_paths(self):
        flat = report.flatten({"a": {"b": 1.0, "c": {"d": 2.0}}})
        self.assertEqual(flat, {"a.b": 1.0, "a.c.d": 2.0})

    def test_tolerance_keys_come_out_as_written(self):
        flat = report.flatten({"cloud": {"accuracy": {"0.02": 0.7}}})
        self.assertEqual(flat, {"cloud.accuracy.0.02": 0.7})

    def test_non_numeric_values_are_dropped(self):
        flat = report.flatten({"skipped": "no data", "epsg": None, "flag": True, "n": 3})
        self.assertEqual(flat, {"n": 3.0})

    def test_nan_is_dropped(self):
        self.assertEqual(report.flatten({"x": float("nan")}), {})

    def test_a_non_dict_is_empty(self):
        self.assertEqual(report.flatten(None), {})


class DirectionTest(unittest.TestCase):
    def test_the_table_covers_the_gating_metrics(self):
        for path in ("reference_free.ortho.covered_fraction",
                     "reference_based.ortho.offset_m",
                     "reference_based.cloud.f_score_5cm"):
            self.assertIsNotNone(report.direction(path, {}), path)

    def test_an_ungated_metric_has_no_direction(self):
        self.assertIsNone(report.direction("timing.total_s", {}))

    def test_an_override_wins(self):
        self.assertEqual(
            report.direction("timing.total_s", {"direction": {"timing.total_s": "lower"}}),
            "lower",
        )


class RegressionTest(unittest.TestCase):
    def test_no_change_is_not_a_regression(self):
        bad, good = report.regressions(metrics(), metrics(), thresholds())
        self.assertEqual(bad, [])
        self.assertEqual(good, [])

    def test_a_small_drop_within_tolerance_is_ignored(self):
        previous = metrics()
        current = metrics()
        # 0.95 -> 0.94 is a 1.05% drop, inside the 2% relative tolerance.
        current["reference_free"]["ortho"]["covered_fraction"] = 0.94
        bad, _ = report.regressions(current, previous, thresholds())
        self.assertEqual(bad, [])

    def test_a_large_drop_in_coverage_is_a_regression(self):
        previous = metrics()
        current = metrics()
        current["reference_free"]["ortho"]["covered_fraction"] = 0.80
        bad, _ = report.regressions(current, previous, thresholds())
        self.assertEqual([r["metric"] for r in bad],
                         ["reference_free.ortho.covered_fraction"])

    def test_coverage_improving_is_an_improvement(self):
        previous = metrics()
        current = metrics()
        current["reference_free"]["ortho"]["covered_fraction"] = 0.99
        bad, good = report.regressions(current, previous, thresholds())
        self.assertEqual(bad, [])
        self.assertEqual([r["metric"] for r in good],
                         ["reference_free.ortho.covered_fraction"])

    def test_hole_count_growing_is_a_regression(self):
        previous = metrics()
        current = metrics()
        current["reference_free"]["ortho"]["holes_inside_hull"] = 9
        bad, _ = report.regressions(current, previous, thresholds())
        self.assertIn("reference_free.ortho.holes_inside_hull", [r["metric"] for r in bad])

    def test_hole_count_growing_within_tolerance_is_ignored(self):
        previous = metrics()
        current = metrics()
        current["reference_free"]["ortho"]["holes_inside_hull"] = 4
        bad, _ = report.regressions(current, previous, thresholds())
        self.assertEqual(bad, [])

    def test_a_larger_ortho_offset_is_a_regression(self):
        previous = metrics()
        current = metrics()
        current["reference_based"]["ortho"]["offset_m"] = 0.12
        bad, _ = report.regressions(current, previous, thresholds())
        self.assertIn("reference_based.ortho.offset_m", [r["metric"] for r in bad])

    def test_the_raw_dem_bias_is_reported_but_not_gated(self):
        # A change in the raw ΔZ median is a vertical datum difference between the
        # two products, not a regression: the pipeline cannot move it.
        previous = metrics()
        previous["reference_based"]["dem"]["median_m"] = 0.01
        current = metrics()
        current["reference_based"]["dem"]["median_m"] = -8.0
        bad, _ = report.regressions(current, previous, thresholds())
        self.assertNotIn("reference_based.dem.median_m", [r["metric"] for r in bad])

    def test_a_widening_offset_removed_spread_is_a_regression(self):
        policy = thresholds()
        policy["tolerances"]["reference_based.dem.offset_removed.p68_m"] = {"absolute": 0.08}
        previous = metrics()
        previous["reference_based"]["dem"]["offset_removed"] = {"p68_m": 0.05, "p95_m": 0.1}
        current = metrics()
        current["reference_based"]["dem"]["offset_removed"] = {"p68_m": 0.20, "p95_m": 0.1}
        bad, _ = report.regressions(current, previous, policy)
        self.assertIn("reference_based.dem.offset_removed.p68_m",
                      [r["metric"] for r in bad])

    def test_a_relative_tolerance_gates_a_dimensionless_ratio(self):
        # effective_resolution_ratio is dimensionless: the policy gates "15% worse
        # than last time", not "0.15 coarser".
        policy = thresholds()
        policy["tolerances"]["reference_free.ortho.effective_resolution_ratio"] = {"relative": 0.15}
        previous = metrics()
        previous["reference_free"]["ortho"]["effective_resolution_ratio"] = 1.0

        current = metrics()
        current["reference_free"]["ortho"]["effective_resolution_ratio"] = 1.14
        bad, _ = report.regressions(current, previous, policy)
        self.assertEqual(bad, [])

        current = metrics()
        current["reference_free"]["ortho"]["effective_resolution_ratio"] = 1.30
        bad, _ = report.regressions(current, previous, policy)
        self.assertIn("reference_free.ortho.effective_resolution_ratio",
                      [r["metric"] for r in bad])

    def test_an_unmeasured_effective_resolution_never_enters_the_gate(self):
        # A None ratio disappears in flatten(), so the metric cannot be compared;
        # this test is the contract that keeps a "could not measure" from reading
        # as a regression.
        policy = thresholds()
        policy["tolerances"]["reference_free.ortho.effective_resolution_ratio"] = {"relative": 0.15}
        previous = metrics()
        previous["reference_free"]["ortho"]["effective_resolution_ratio"] = 1.0
        current = metrics()
        current["reference_free"]["ortho"]["effective_resolution_ratio"] = None
        bad, _ = report.regressions(current, previous, policy)
        self.assertEqual(bad, [])

    def test_a_falling_f_score_is_a_regression(self):
        previous = metrics()
        current = metrics()
        current["reference_based"]["cloud"]["f_score_5cm"] = 0.70
        bad, _ = report.regressions(current, previous, thresholds())
        self.assertIn("reference_based.cloud.f_score_5cm", [r["metric"] for r in bad])

    def test_a_metric_with_no_tolerance_is_reported_never_gated(self):
        previous = metrics()
        current = metrics()
        # gradient_energy has a direction in the table but no tolerance entry, so
        # the default 0 keeps it out of the verdict.
        current["reference_free"]["ortho"]["gradient_energy"] = 0.0001
        bad, good = report.regressions(current, previous, thresholds())
        self.assertEqual(bad, [])
        self.assertEqual(good, [])

    def test_a_metric_missing_from_the_current_run_is_not_a_regression(self):
        previous = metrics()
        current = metrics()
        del current["reference_free"]["ortho"]["covered_fraction"]
        bad, _ = report.regressions(current, previous, thresholds())
        self.assertEqual(bad, [])

    def test_a_skipped_reference_does_not_invent_a_regression(self):
        previous = metrics()
        current = metrics()
        current["reference_based"]["cloud"] = {"skipped": "no point cloud produced"}
        bad, _ = report.regressions(current, previous, thresholds())
        self.assertEqual(bad, [])

    def test_the_record_carries_what_the_reader_needs(self):
        previous = metrics()
        current = metrics()
        current["reference_free"]["ortho"]["covered_fraction"] = 0.50
        bad, _ = report.regressions(current, previous, thresholds())
        record = bad[0]
        self.assertAlmostEqual(record["previous"], 0.95)
        self.assertAlmostEqual(record["current"], 0.50)
        self.assertAlmostEqual(record["delta"], -0.45)
        self.assertAlmostEqual(record["limit"], 0.019)
        self.assertEqual(record["direction"], "higher")


class MarkdownTest(unittest.TestCase):
    def test_the_dem_table_names_both_vertical_datums(self):
        text = report.markdown(metrics(), None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("offset removed", text)
        self.assertIn("Vertical datum (ours)", text)
        self.assertIn("WGS 84 ellipsoidal", text)
        self.assertIn("unknown", text)  # the reference's is not recorded

    def test_the_verdict_names_a_first_run(self):
        text = report.markdown(metrics(), None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("Verdict: pass", text)
        self.assertIn("first run", text)

    def test_the_verdict_fails_with_a_regression(self):
        text = report.markdown(
            metrics(), metrics(),
            [{"metric": "m", "previous": 1.0, "current": 0.5, "delta": -0.5, "limit": 0.1,
              "direction": "higher"}],
            [], thresholds=thresholds(), png_name=None,
        )
        self.assertIn("Verdict: FAIL", text)
        self.assertIn("`m`", text)

    def test_provenance_names_the_dataset_and_its_licence(self):
        text = report.markdown(metrics(), None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("Synthetic scene", text)
        self.assertIn("CC0-1.0", text)
        self.assertIn("deadbeefdeadbeef", text)

    def test_every_section_appears(self):
        text = report.markdown(metrics(), None, [], [], thresholds=thresholds(), png_name=None)
        for heading in ("## Provenance", "## Products", "## Reference-free metrics",
                        "## Reference-based metrics", "## Comparison with the previous run"):
            self.assertIn(heading, text)

    def test_a_skipped_product_is_named_not_omitted(self):
        current = metrics()
        current["reference_free"]["ortho"] = {"skipped": "no orthomosaic produced"}
        text = report.markdown(current, None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("no orthomosaic produced", text)

    def test_a_dataset_without_references_says_so(self):
        current = metrics()
        current["reference_based"] = {
            "ortho": {"skipped": "dataset has no ortho reference"},
            "dem": {"skipped": "dataset has no DSM reference"},
            "cloud": {"skipped": "dataset has no laser reference"},
        }
        text = report.markdown(current, None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("publishes no reference product", text)

    def test_a_cross_dataset_comparison_is_flagged(self):
        previous = metrics()
        previous["dataset_manifest"]["title"] = "A different scene"
        text = report.markdown(metrics(), previous, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("A different scene", text)
        self.assertIn("meaningless", text)

    def test_the_png_is_referenced_when_present(self):
        text = report.markdown(metrics(), None, [], [], thresholds=thresholds(),
                               png_name="compare.png")
        self.assertIn("![", text)
        self.assertIn("compare.png", text)

    def test_an_untextured_mosaic_is_called_out(self):
        current = metrics()
        current["reference_free"]["ortho"]["textured"] = False
        current["reference_free"]["ortho"]["effective_measured"] = False
        current["reference_free"]["ortho"]["effective_resolution_ratio"] = None
        current["reference_free"]["ortho"]["effective_resolution_m"] = None
        text = report.markdown(current, None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("no usable texture", text)
        self.assertIn("unmeasured", text)

    def test_the_effective_resolution_is_reported_against_the_cell(self):
        text = report.markdown(metrics(), None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("1.200 x cell size", text)
        self.assertIn("cell 0.02 m", text)
        self.assertNotIn("Effective/nominal", text)


class ReprojectionReportingTest(unittest.TestCase):
    def test_a_diverging_independent_figure_is_still_reported_but_flagged(self):
        # The independent figure is gated like any other number once the camera
        # type is read case-insensitively; a divergence is a diagnostic, not a
        # reason to drop the number. A 163 px figure against 0.51 px must still
        # appear in the table AND say it disagrees.
        current = metrics()
        current["reference_free"]["cameras"]["reprojection"] = {
            "pairs": 500, "rms_px": 163.0, "median_px": 89.0,
            "worst_cameras": [{"camera": 3, "pairs": 10, "median_px": 270.0}],
        }
        current["reference_free"]["cameras"]["engine_report"] = {"rms_px": 0.51,
                                                                "registered": 30, "offered": 30}
        text = report.markdown(current, None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("engine BA", text)
        self.assertIn("| Reprojection RMS (independent) |", text)
        self.assertIn("disagrees", text)
        self.assertIn("#3 270", text)

    def test_agreeing_figures_are_both_reported(self):
        current = metrics()
        current["reference_free"]["cameras"]["reprojection"] = {
            "pairs": 500, "rms_px": 0.55, "median_px": 0.5,
        }
        current["reference_free"]["cameras"]["engine_report"] = {"rms_px": 0.51,
                                                                "registered": 30, "offered": 30}
        text = report.markdown(current, None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("Reprojection RMS (independent)", text)
        self.assertIn("agrees", text)
        self.assertNotIn("disagrees", text)


class SubsetNoteTest(unittest.TestCase):
    def test_a_subset_run_says_how_many_of_how_many(self):
        current = metrics()
        current["dataset_manifest"].update(
            images=30, images_available=102,
            image_selection={
                "mode": "nearest-to-centroid cluster",
                "images": ["A.JPG", "C.JPG", "D.JPG"],
            },
        )
        text = report.markdown(current, None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("30 of 102", text)
        self.assertIn("nearest-to-centroid cluster", text)
        self.assertIn("A.JPG, C.JPG, D.JPG", text)

    def test_an_older_metrics_file_with_only_endpoints_still_renders(self):
        current = metrics()
        current["dataset_manifest"].update(
            images=30, images_available=102,
            image_selection={"mode": "gps-centred block", "first": "A.JPG", "last": "B.JPG"},
        )
        text = report.markdown(current, None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("A.JPG .. B.JPG", text)

    def test_a_full_run_does_not_mention_a_subset(self):
        text = report.markdown(metrics(), None, [], [], thresholds=thresholds(), png_name=None)
        self.assertNotIn("Image block", text)


class OverrideNoteTest(unittest.TestCase):
    def test_non_default_parameters_are_called_out(self):
        current = metrics()
        current["timing"]["param_overrides"] = {"BuildDepthMaps": {"downscale": 8}}
        text = report.markdown(current, None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("Not at defaults", text)
        self.assertIn("downscale=8", text)

    def test_a_default_run_says_nothing_about_overrides(self):
        text = report.markdown(metrics(), None, [], [], thresholds=thresholds(), png_name=None)
        self.assertNotIn("Not at defaults", text)


class ExperimentTest(unittest.TestCase):
    """A run dispatched with extra depth parameters is evidence, not a baseline."""

    def experiment(self, **params):
        current = metrics()
        current["depth_params"] = params or {"cost_type": "census"}
        return current

    def verdict(self, current, previous=None, *, gates=True):
        """Run report.main over two metrics dicts and return its exit status."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "thresholds.json").write_text(json.dumps(thresholds()), encoding="utf-8")
            current_path = root / "current.json"
            current_path.write_text(json.dumps(current), encoding="utf-8")
            argv = [
                "--current", str(current_path),
                "--thresholds", str(root / "thresholds.json"),
            ]
            if previous is not None:
                previous_path = root / "previous.json"
                previous_path.write_text(json.dumps(previous), encoding="utf-8")
                argv += ["--previous", str(previous_path)]
            if gates:
                argv.append("--fail-on-regression")
            return report.main(argv)

    def test_a_default_run_has_no_dispatch_parameters(self):
        self.assertEqual(report.experiment_params(metrics()), {})

    def test_a_missing_key_is_not_an_experiment(self):
        # Every metrics file written before this flag existed has no key at all.
        current = metrics()
        current.pop("depth_params", None)
        self.assertEqual(report.experiment_params(current), {})

    def test_dispatch_parameters_are_read_back_as_strings(self):
        current = metrics()
        current["depth_params"] = {"cost_type": "census", "max_cost": 0.4}
        self.assertEqual(
            report.experiment_params(current),
            {"BuildDepthMaps.cost_type": "census", "BuildDepthMaps.max_cost": "0.4"},
        )

    def test_a_dem_only_run_is_an_experiment(self):
        # The key alone must be enough: BuildDem parameters are not defaults either.
        current = metrics()
        current["dem_params"] = {"sibson_exact_max_sites": 20000}
        self.assertEqual(
            report.experiment_params(current),
            {"BuildDem.sibson_exact_max_sites": "20000"},
        )

    def test_both_inputs_are_folded_into_one_mapping(self):
        current = metrics()
        current["depth_params"] = {"cost_type": "census"}
        current["dem_params"] = {"sibson_exact_max_sites": "20000"}
        self.assertEqual(
            report.experiment_params(current),
            {
                "BuildDepthMaps.cost_type": "census",
                "BuildDem.sibson_exact_max_sites": "20000",
            },
        )

    def test_a_dem_experiment_never_gates(self):
        current = self.experiment()
        current.pop("depth_params", None)
        current["dem_params"] = {"sibson_exact_max_sites": "20000"}
        current["reference_free"]["ortho"]["covered_fraction"] = 0.50
        self.assertEqual(self.verdict(current, metrics()), 0)

    def test_a_dem_experiment_is_not_mistaken_for_the_control(self):
        # Same regression, control parameters in the previous run: the two are
        # different configurations, so the comparison must not gate.
        current = self.experiment()
        current.pop("depth_params", None)
        current["dem_params"] = {"sibson_exact_max_sites": "20000"}
        text = report.markdown(current, metrics(), [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("different configurations", text)

    def test_the_verdict_is_experiment_not_pass_or_fail(self):
        text = report.markdown(self.experiment(), None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("Verdict: experiment", text)
        self.assertIn("cost_type=census", text)
        self.assertNotIn("Verdict: pass", text)

    def test_an_experiment_never_fails_even_with_a_regression(self):
        current = self.experiment()
        # A coverage drop well past its tolerance would fail a control run.
        current["reference_free"]["ortho"]["covered_fraction"] = 0.50
        self.assertEqual(self.verdict(current, metrics()), 0)

    def test_a_control_run_with_the_same_regression_does_fail(self):
        # The mirror of the test above: the gate itself still works. Without this,
        # "experiments do not gate" could be satisfied by disabling the gate.
        current = metrics()
        current["reference_free"]["ortho"]["covered_fraction"] = 0.50
        self.assertEqual(self.verdict(current, metrics()), 1)

    def test_a_different_configuration_is_compared_but_not_gated(self):
        current = self.experiment()
        previous = metrics()  # the control, at defaults
        text = report.markdown(current, previous, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("different configurations", text)

    def test_an_experiment_never_gates_even_against_another_experiment(self):
        # "Must not fail the regression gate" is unconditional: the run is evidence
        # about a configuration, and a red check would block the branch being
        # evaluated. The report still lists what moved, which is the useful part.
        current = self.experiment()
        current["reference_free"]["ortho"]["covered_fraction"] = 0.50
        self.assertEqual(self.verdict(current, self.experiment()), 0)
        text = report.markdown(current, self.experiment(), [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("Verdict: experiment", text)

    def test_provenance_names_the_dispatch_parameters(self):
        text = report.markdown(self.experiment(), None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("Dispatch parameters", text)


class FormatTest(unittest.TestCase):
    def test_none_is_a_dash(self):
        self.assertEqual(report._fmt(None), "-")

    def test_a_small_float_uses_significant_digits(self):
        self.assertEqual(report._fmt(0.001234567), "0.001235")

    def test_a_large_float_is_grouped(self):
        self.assertIn(",", report._fmt(1234.5))

    def test_a_bool_is_yes_or_no(self):
        self.assertEqual(report._fmt(True), "yes")
        self.assertEqual(report._fmt(False), "no")


if __name__ == "__main__":
    unittest.main()
