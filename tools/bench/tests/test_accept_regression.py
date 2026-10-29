#!/usr/bin/env python3
"""Unit tests for the accepted-regression escape hatch.

An accepted regression is the one narrow path by which a run that FAILED the gate
may still publish a baseline. It is a manual, auditable act, so each of its
conditions is tested separately: the reason must be valid, the branch must be the
default one, the run must be a control run, and the record it writes must be the
evidence a later reader needs. The empty-input case is tested too, because "no
change for every ordinary run" is the property the whole workflow depends on.

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

import accept_regression  # noqa: E402
import report  # noqa: E402
from test_report import metrics, thresholds  # noqa: E402


def regressing(**overrides):
    """Metrics that fail the gate hard against :func:`metrics` (the control)."""
    current = metrics()
    # A coverage drop from 0.95 to 0.50 is far past its 2% relative tolerance.
    current["reference_free"]["ortho"]["covered_fraction"] = 0.50
    current["reference_based"]["ortho"]["offset_m"] = 0.10
    for path, value in overrides.items():
        section, _, key = path.partition(".")
        current[section][key] = value
    return current


class Harness:
    """A throwaway work directory holding a current/previous metrics pair."""

    def __init__(self, current, previous=None, policy=None):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.root = root
        self.thresholds_path = root / "thresholds.json"
        self.thresholds_path.write_text(json.dumps(policy or thresholds()), encoding="utf-8")
        self.output_path = root / "github_output"
        self.output_path.write_text("", encoding="utf-8")
        self.current_path = root / "current.json"
        self.current_path.write_text(json.dumps(current), encoding="utf-8")
        self.previous_path = root / "previous.json"
        if previous is not None:
            self.previous_path.write_text(json.dumps(previous), encoding="utf-8")

    def run(self, reason="a deliberate trade-off", *, branch="main", default="main", argv=()):
        """Invoke the script as the workflow does and return (status, outputs, metrics)."""
        saved = os.environ.get("ACCEPT_REGRESSION")
        os.environ["ACCEPT_REGRESSION"] = reason
        try:
            args = [
                "--current", str(self.current_path),
                "--thresholds", str(self.thresholds_path),
                "--branch", branch,
                "--default-branch", default,
                "--actor", "aniket",
                "--run-id", "40833481199",
                "--github-output", str(self.output_path),
                *argv,
            ]
            if self.previous_path.exists():
                args += ["--previous", str(self.previous_path)]
            status = accept_regression.main(args)
        finally:
            if saved is None:
                os.environ.pop("ACCEPT_REGRESSION", None)
            else:
                os.environ["ACCEPT_REGRESSION"] = saved
        outputs = dict(
            line.split("=", 1)
            for line in self.output_path.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
        written = json.loads(self.current_path.read_text(encoding="utf-8"))
        return status, outputs, written

    def close(self):
        self._tmp.cleanup()


class AcceptTest(unittest.TestCase):
    def test_an_acceptance_records_and_licenses_publishing(self):
        harness = Harness(regressing(), metrics())
        try:
            status, outputs, written = harness.run()
        finally:
            harness.close()
        self.assertEqual(status, 0)
        self.assertEqual(outputs.get("accepted"), "true")
        self.assertEqual(outputs.get("publishing"), "true")
        record = written["accepted_regression"]
        self.assertEqual(record["reason"], "a deliberate trade-off")
        self.assertEqual(record["actor"], "aniket")
        self.assertEqual(record["run_id"], "40833481199")
        # The exact moves the reason covers, with the numbers they moved between.
        covered = {item["metric"] for item in record["metrics"]}
        self.assertIn("reference_free.ortho.covered_fraction", covered)
        self.assertIn("reference_based.ortho.offset_m", covered)
        moved = next(
            item for item in record["metrics"]
            if item["metric"] == "reference_based.ortho.offset_m"
        )
        self.assertAlmostEqual(moved["previous"], 0.04)
        self.assertAlmostEqual(moved["current"], 0.10)

    def test_the_recorded_moves_match_the_gate(self):
        # One predicate decides both what regressed and what was accepted, so the
        # acceptance can never list a metric the gate did not fail on.
        current, previous = regressing(), metrics()
        bad, _ = report.regressions(current, previous, thresholds())
        recorded = accept_regression.regression_records(current, previous, thresholds())
        self.assertEqual(
            [record["metric"] for record in recorded],
            [record["metric"] for record in bad],
        )

    def test_the_report_names_the_acceptance_and_keeps_the_fail(self):
        current = regressing()
        accept_regression.apply(
            current, reason="trade-off", actor="aniket",
            run_id="1", previous=metrics(), thresholds=thresholds(),
        )
        text = report.markdown(
            current, metrics(), *report.regressions(current, metrics(), thresholds()),
            thresholds=thresholds(), png_name=None,
        )
        self.assertIn("FAIL (accepted)", text)
        self.assertIn("## Accepted regression", text)
        self.assertIn("trade-off", text)
        # The regressed metrics are still shown with old -> new.
        self.assertIn("reference_based.ortho.offset_m", text)
        self.assertIn("reference_free.ortho.covered_fraction", text)

    def test_a_first_run_accepts_with_no_moves(self):
        # Nothing to regress against is not an error: the run still records why it
        # published, with an empty list of moves.
        harness = Harness(metrics(), None)
        try:
            status, outputs, written = harness.run()
        finally:
            harness.close()
        self.assertEqual(status, 0)
        self.assertEqual(outputs.get("publishing"), "true")
        self.assertEqual(written["accepted_regression"]["metrics"], [])

    def test_an_empty_previous_path_is_a_first_run(self):
        # The workflow always passes --previous, with an empty value on a first
        # run. That must be read as "no baseline", not as the current directory.
        harness = Harness(metrics(), None)
        try:
            status, outputs, written = harness.run(argv=("--previous", ""))
        finally:
            harness.close()
        self.assertEqual(status, 0)
        self.assertEqual(written["accepted_regression"]["previous_run_supplied"], False)


class RefusalTest(unittest.TestCase):
    def test_a_non_default_branch_is_refused(self):
        harness = Harness(regressing(), metrics())
        try:
            status, outputs, written = harness.run(branch="feat/depth-defaults")
        finally:
            harness.close()
        self.assertEqual(status, 2)
        self.assertNotIn("publishing", outputs)
        # A refused acceptance must not touch the metrics file.
        self.assertNotIn("accepted_regression", written)

    def test_the_branch_refusal_says_what_to_do(self):
        problems = accept_regression.refusals(
            branch="feat/depth-defaults", default_branch="main",
            current=regressing(),
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("default branch", problems[0])
        self.assertIn("feat/depth-defaults", problems[0])

    def test_an_experiment_is_refused(self):
        current = regressing()
        current["depth_params"] = {"cost_type": "census"}
        problems = accept_regression.refusals(
            branch="main", default_branch="main", current=current
        )
        self.assertTrue(problems)
        self.assertIn("depth_params", problems[0])

    def test_dem_params_are_refused_too(self):
        current = regressing()
        current["dem_params"] = {"alpha1": "0.2"}
        problems = accept_regression.refusals(
            branch="main", default_branch="main", current=current
        )
        self.assertTrue(problems)
        self.assertIn("dem_params", problems[0])

    def test_a_control_run_on_the_default_branch_is_allowed(self):
        self.assertEqual(
            accept_regression.refusals(
                branch="main", default_branch="main", current=regressing()
            ),
            [],
        )


class ReasonTest(unittest.TestCase):
    def test_a_valid_reason_is_trimmed(self):
        self.assertEqual(accept_regression.validate_reason("  why  "), "why")

    def test_empty_is_invalid(self):
        for value in (None, "", "   "):
            with self.assertRaises(accept_regression.Refused):
                accept_regression.validate_reason(value)

    def test_a_multiline_reason_is_invalid(self):
        with self.assertRaises(accept_regression.Refused):
            accept_regression.validate_reason("line one\nline two")

    def test_an_overlong_reason_is_invalid(self):
        self.assertEqual(
            len(accept_regression.validate_reason("x" * accept_regression.MAX_REASON)),
            accept_regression.MAX_REASON,
        )
        with self.assertRaises(accept_regression.Refused):
            accept_regression.validate_reason("x" * (accept_regression.MAX_REASON + 1))


class EmptyInputTest(unittest.TestCase):
    """With no reason, the script does nothing and changes nothing."""

    def test_an_empty_reason_leaves_the_metrics_untouched(self):
        harness = Harness(regressing(), metrics())
        before = json.loads(harness.current_path.read_text(encoding="utf-8"))
        try:
            status, outputs, written = harness.run(reason="")
        finally:
            harness.close()
        self.assertEqual(status, 2)
        self.assertEqual(outputs, {})
        self.assertEqual(before, written)
        self.assertNotIn("accepted_regression", written)

    def test_the_shell_condition_skips_the_step_entirely(self):
        # The workflow's `if:` is the first line of defence: with an empty input
        # the step never runs, so the script is not even invoked. This asserts the
        # contract the workflow relies on rather than the YAML text.
        workflow = (
            Path(__file__).resolve().parents[3]
            / ".github" / "workflows" / "quality-bench.yml"
        )
        if not workflow.exists():  # pragma: no cover - installed copies
            self.skipTest("workflow file is not in this checkout")
        text = workflow.read_text(encoding="utf-8")
        self.assertIn("github.event.inputs.accept_regression != ''", text)
        # And the publish gate only widens for an accepted run.
        self.assertIn("steps.accept.outputs.accepted == 'true'", text)


if __name__ == "__main__":
    unittest.main()
