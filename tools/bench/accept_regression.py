#!/usr/bin/env python3
"""Record a deliberate, accepted regression so its run can publish a baseline.

A weekly run fails when a metric moves the wrong way past its tolerance, and a
failed run never replaces the baseline. That is the right default: it keeps a
regression red until somebody fixes it. It has one dead end, though — a change
that is worse on one metric and clearly better on the rest. The gate must still
show the move, but the trade-off has to be adoptable, or every later week fails
against a baseline nobody can reach.

An acceptance is a manual, auditable act, so it requires ALL of:

  * the ``accept_regression`` dispatch input, a non-empty reason;
  * a run on the default branch (a feature branch must never move the shared
    baseline — dispatch the accepting run on main after merging);
  * a control run: ``depth_params`` and ``dem_params`` empty. An experiment
    measures a configuration; it is not the yardstick and is refused here too.

What it does, and does not, do:

  * the verdict is still computed and reported, naming every regressed metric
    old -> new with its limit. The run is still a FAIL; it is an accepted FAIL.
  * the reason, the dispatching actor, the run id and the exact regressions are
    recorded in the metrics JSON under ``accepted_regression``, so the published
    baseline carries the evidence for why it is where it is.
  * the workflow is told the run may publish, so the next week compares against
    these numbers instead of the ones the trade-off replaced.

Exit codes, read directly by the workflow:

    0   nothing requested, or the request was applied (see the outputs)
    2   the request was refused: bad reason, wrong branch, or not a control run

Usage:

    ACCEPT_REGRESSION="why" python3 tools/bench/accept_regression.py \
        --current build/bench/metrics-aerial-nadir.json \
        --previous build/bench/previous/metrics-aerial-nadir.json \
        --branch main --default-branch main --actor someone --run-id 123 \
        --github-output "$GITHUB_OUTPUT"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import report  # noqa: E402

MAX_REASON = 200
DEFAULT_REASON_ENV = "ACCEPT_REGRESSION"
DEFAULT_THRESHOLDS = HERE / "thresholds.json"


class Refused(Exception):
    """The dispatch asked for something this run must not do."""


def validate_reason(text: str | None) -> str:
    """The reason, trimmed, or a Refused saying exactly what is wrong with it.

    The value arrives from a dispatch input, so it is untrusted text that ends up
    in a report, a JSON record and the baseline. It is therefore held to what can
    be shown verbatim: printable characters only (no newlines or control codes
    that could forge a report line) and a bounded length.
    """
    reason = (text or "").strip()
    if not reason:
        raise Refused("accept_regression is empty")
    if not reason.isprintable():
        raise Refused("accept_regression must be printable text, on one line")
    if len(reason) > MAX_REASON:
        raise Refused(f"accept_regression is {len(reason)} characters; the limit is {MAX_REASON}")
    return reason


def measure_params(metrics: dict) -> dict[str, object]:
    """Any dispatch parameters that make a run an experiment rather than a control.

    Both dispatch inputs the workflow offers — ``depth_params`` for BuildDepthMaps
    and ``dem_params`` for BuildDem — mean the run measured a configuration rather
    than the defaults. Either one being present and non-empty withholds the
    baseline, so either one refuses an acceptance.
    """
    found: dict[str, object] = {}
    for key in ("depth_params", "dem_params"):
        value = metrics.get(key)
        if value:
            found[key] = value
    return found


def refusals(*, branch: str, default_branch: str, current: dict) -> list[str]:
    """Everything that disqualifies this run from accepting, in the order a
    reader would want to fix it. Empty means the acceptance is allowed."""
    problems: list[str] = []
    if branch != default_branch:
        problems.append(
            f"accept_regression is only honoured on the default branch "
            f"({default_branch}); this run is on {branch or '(unknown)'}. "
            "Merge the change first, then dispatch the accepting run on the default branch."
        )
    params = measure_params(current)
    if params:
        problems.append(
            "accept_regression cannot be combined with "
            + " or ".join(sorted(params))
            + ": an experiment measures a configuration and never publishes a baseline"
        )
    return problems


def regression_records(current: dict, previous: dict | None, thresholds: dict) -> list[dict]:
    """The regressions the acceptance covers, in the shape the report gates on.

    The same predicate the report uses decides what regressed, so an accepted run
    and a gated run can never disagree about which metrics the reason is about. A
    first run (no previous) has nothing to regress against and records an empty
    list: the acceptance is then a no-op that still leaves its audit trail.
    """
    if previous is None:
        return []
    bad, _ = report.regressions(current, previous, thresholds)
    keys = ("metric", "previous", "current", "delta", "limit", "direction")
    return [dict({key: record[key] for key in keys}) for record in bad]


def apply(
    current: dict,
    *,
    reason: str,
    actor: str,
    run_id: str,
    previous: dict | None,
    thresholds: dict,
) -> dict:
    """Write the ``accepted_regression`` record into ``current`` and return it.

    The record is what makes the published baseline auditable months later: who
    accepted what, against which run, and the exact numbers. It is written into
    the metrics file — not kept in the workflow log — because the baseline the
    next week downloads is that file and nothing else.
    """
    current["accepted_regression"] = {
        "reason": reason,
        "actor": actor,
        "run_id": run_id,
        "previous_run_supplied": previous is not None,
        "metrics": regression_records(current, previous, thresholds),
    }
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", required=True, help="this run's metrics JSON")
    parser.add_argument("--previous", help="the baseline it was compared with")
    parser.add_argument("--branch", default="", help="the branch this run is on")
    parser.add_argument("--default-branch", required=True)
    parser.add_argument("--actor", default="", help="the dispatching actor")
    parser.add_argument("--run-id", default="", help="this workflow run's id")
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT", ""),
        help="append accepted/publishing decisions here (default $GITHUB_OUTPUT)",
    )
    args = parser.parse_args(argv)

    with open(args.current, "r", encoding="utf-8") as handle:
        current = json.load(handle)
    previous = None
    # The workflow always passes --previous, but its value is empty on a first run.
    # ``Path("")`` is ``.``, which exists, so an unguarded check would try to read
    # the working directory as JSON.
    if args.previous and Path(args.previous).is_file():
        with open(args.previous, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
    with open(args.thresholds, "r", encoding="utf-8") as handle:
        thresholds = json.load(handle)

    outputs: dict[str, str] = {}

    try:
        reason = validate_reason(os.environ.get(DEFAULT_REASON_ENV))
    except Refused as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2

    problems = refusals(
        branch=args.branch, default_branch=args.default_branch, current=current
    )
    if problems:
        for problem in problems:
            print(f"::error::{problem}", file=sys.stderr)
        return 2

    apply(
        current,
        reason=reason,
        actor=args.actor,
        run_id=args.run_id,
        previous=previous,
        thresholds=thresholds,
    )
    with open(args.current, "w", encoding="utf-8") as handle:
        # Same layout run_bench.py writes, so the file stays byte-comparable in
        # review and the baseline keeps the shape every reader expects.
        json.dump(current, handle, indent=2, sort_keys=True)
        handle.write("\n")

    accepted = current["accepted_regression"]["metrics"]
    # ``publishing=true`` is the one output the workflow acts on: it says the
    # numbers in this file are allowed to become the next baseline even though the
    # report's verdict is FAIL. ``accepted_regressions`` is for the annotation.
    outputs["accepted"] = "true"
    outputs["publishing"] = "true"
    outputs["accepted_regressions"] = ",".join(record["metric"] for record in accepted)

    for name, value in outputs.items():
        print(f"{name}={value}")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            for name, value in outputs.items():
                handle.write(f"{name}={value}\n")
    print(
        f"accepted regression: {len(accepted)} metric(s), reason={reason!r}, actor={args.actor!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
