#!/usr/bin/env python3
"""Unit tests for the report renderer and its regression verdict.

The regression logic is the part of the bench that decides whether a weekly run is
green, so it is tested against synthetic metrics dicts: a known move in a known
direction with a known tolerance has an unambiguous verdict.

Run with plain unittest (no pytest):

    python3 -m unittest discover -s tools/bench/tests -v
"""

import os
import sys
import unittest

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
                "effective_gsd_m": 0.024,
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
                    "cells": 1000, "overlap_fraction": 0.8},
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

    def test_a_signed_dem_bias_growing_either_way_is_a_regression(self):
        previous = metrics()
        previous["reference_based"]["dem"]["median_m"] = 0.01
        for value in (0.08, -0.08):
            current = metrics()
            current["reference_based"]["dem"]["median_m"] = value
            bad, _ = report.regressions(current, previous, thresholds())
            self.assertIn("reference_based.dem.median_m", [r["metric"] for r in bad], value)

    def test_a_signed_dem_bias_shrinking_is_not_a_regression(self):
        previous = metrics()
        previous["reference_based"]["dem"]["median_m"] = 0.06
        current = metrics()
        current["reference_based"]["dem"]["median_m"] = -0.01
        bad, _ = report.regressions(current, previous, thresholds())
        self.assertNotIn("reference_based.dem.median_m", [r["metric"] for r in bad])

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
        text = report.markdown(current, None, [], [], thresholds=thresholds(), png_name=None)
        self.assertIn("no usable texture", text)


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
