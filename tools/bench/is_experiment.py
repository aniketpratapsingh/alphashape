#!/usr/bin/env python3
"""Report whether a metrics JSON records a dispatched experiment.

The workflow has to answer one question about a run's metrics file: was this run
made with extra dispatch parameters (``depth_params`` for BuildDepthMaps,
``dem_params`` for BuildDem), or at the dataset's own settings? The answer
decides two things that must agree — whether the run may become the regression
baseline, and whether a baseline restored from it may be used. Both call this so
there is one definition, in one place.

Exit codes, because a shell `if` reads them directly:

    0   a control run (no dispatch parameters) — eligible to be a baseline
    1   an experiment (dispatch parameters present) — not a baseline
    2   the file is missing, unreadable or not a JSON object — cannot tell

2 is deliberately distinct from 1. A caller that treats every non-zero as "not a
baseline" is right; a caller that collapses the two would report an experiment
where the file was simply absent, and the notice in the workflow log would be
wrong about why.

    python3 tools/bench/is_experiment.py build/bench/metrics-x.json
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: is_experiment.py METRICS_JSON", file=sys.stderr)
        return 2
    try:
        with open(argv[0], "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, ValueError):
        return 2
    if not isinstance(metrics, dict):
        return 2
    if metrics.get("depth_params") or metrics.get("dem_params"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
