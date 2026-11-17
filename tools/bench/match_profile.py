#!/usr/bin/env python3
"""Peak memory of MatchPhotos and AlignCameras, per stage, on a real block.

The GUI's job dialog warns when a task will not fit in RAM, and the bench
manifests lower the keypoint density because the default once ran a 16 GB
runner out of memory. Both need the same number: how much memory each stage
of the matcher holds, and how that grows with the keypoint density and the
thread count. This script measures it.

It runs the CLI once per task on one project, samples the process's resident
set every ``--interval`` seconds, and attributes each sample to the progress
stage the CLI last printed ("[detect] 40%"). A stage that reached 100% hands
over to "after <stage>"; after the task's last stage that is the commit and
the project save.

    match_profile.py run --dataset gcp-rtk --work build/profile \\
        --cache build/bench/cache --kp 1000 --out profile.json
    match_profile.py summarize --out-md summary.md profile-*.json

Linux only: the resident set is read from /proc. Only numbers are written;
no image or project file leaves the runner.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_bench  # noqa: E402

DEFAULT_TASKS = ("MatchPhotos", "AlignCameras")


def parse_progress(line: str) -> str | None:
    """The stage name of a CLI progress line ``[stage] NN%``, else None."""
    stripped = line.strip()
    if not (stripped.startswith("[") and stripped.endswith("%")):
        return None
    close = stripped.find("]")
    if close <= 1:
        return None
    percent = stripped[close + 1:-1].strip()
    if not percent.isdigit():
        return None
    return stripped[1:close]


def rss_bytes(pid: int) -> int | None:
    """Current resident set of ``pid`` from /proc, or None once it has exited."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return None


class StageTracker:
    """The stage the CLI last reported, plus each stage's peak and duration."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = "start"
        self.order: list[str] = ["start"]
        self.peak: dict[str, int] = {}
        self.entered: dict[str, float] = {"start": time.monotonic()}
        self.seconds: dict[str, float] = {}
        self.finished = False

    def enter(self, stage: str) -> None:
        with self.lock:
            if stage == self.current:
                return
            now = time.monotonic()
            self.seconds[self.current] = (self.seconds.get(self.current, 0.0)
                                          + now - self.entered[self.current])
            self.current = stage
            self.entered[stage] = now
            if stage not in self.order:
                self.order.append(stage)

    def sample(self, value: int) -> None:
        with self.lock:
            self.peak[self.current] = max(self.peak.get(self.current, 0), value)

    def close(self) -> None:
        self.enter("end")


def run_task(cli: str, arguments: list[str], log: Path, interval: float) -> dict:
    """Run one CLI invocation, sampling its resident set by stage."""
    tracker = StageTracker()
    started = time.monotonic()
    with open(log, "a", encoding="utf-8") as sink:
        process = subprocess.Popen([cli, *arguments], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True, bufsize=1)

        def read() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                sink.write(line)
                stage = parse_progress(line)
                if stage is None:
                    continue
                # A finished stage hands over to "after <stage>": between stages
                # that is a moment, after the last one it is the commit and save.
                tracker.enter(f"after {stage}" if line.strip().endswith(" 100%") else stage)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        while process.poll() is None:
            value = rss_bytes(process.pid)
            if value is not None:
                tracker.sample(value)
            time.sleep(interval)
        reader.join()
    tracker.close()
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        name = {9: "SIGKILL (out of memory?)"}.get(-process.returncode, str(process.returncode))
        return {"status": f"failed: {name}", "seconds": elapsed,
                "stages": _stages(tracker), "peak_rss_mb": _peak_mb(tracker)}
    return {"status": "ok", "seconds": elapsed, "stages": _stages(tracker),
            "peak_rss_mb": _peak_mb(tracker)}


def _stages(tracker: StageTracker) -> list[dict]:
    rows = []
    for stage in tracker.order:
        if stage in ("start", "end"):
            continue
        rows.append({"stage": stage,
                     "peak_rss_mb": round(tracker.peak.get(stage, 0) / 1e6, 1),
                     "seconds": round(tracker.seconds.get(stage, 0.0), 1)})
    return rows


def _peak_mb(tracker: StageTracker) -> float:
    return round(max(tracker.peak.values(), default=0) / 1e6, 1)


def lay_out_images(dataset_name: str, work: Path, cache: Path) -> list[Path]:
    """Download (or reuse) the dataset's archive and return its images."""
    dataset = run_bench.load_manifest()["datasets"][dataset_name]
    cache.mkdir(parents=True, exist_ok=True)
    for name, remote in run_bench.downloads_for(dataset):
        run_bench.download(remote, cache / Path(name).name)
    archive = cache / dataset["images_archive"]["download"]
    raw = work / "raw"
    if not (raw.exists() and any(raw.rglob("*"))):
        run_bench.extract_archive(archive, raw)
    images = sorted(p for p in raw.glob(dataset["images_archive"]["extract_glob"]) if p.is_file())
    expected = dataset.get("expected_images")
    if expected is not None and len(images) != expected:
        raise SystemExit(f"{dataset_name}: expected {expected} images, found {len(images)}")
    return images


def image_megapixels(path: Path) -> float | None:
    """Pixel count of one image, from Pillow (a bench dependency)."""
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return None
    with Image.open(path) as image:
        return image.width * image.height / 1e6


def command_run(args: argparse.Namespace) -> None:
    cli = os.environ.get("ALPHASHAPE_CLI")
    if not cli:
        raise SystemExit("ALPHASHAPE_CLI is not set")
    work = Path(args.work)
    images = lay_out_images(args.dataset, work, Path(args.cache))
    project_dir = work / f"kp{args.kp}"
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)
    project = project_dir / "site.psx"
    log = project_dir / "cli.log"
    run_bench.run_cli(["--new", str(project), "--add-images", *[str(p) for p in images]], log=log)

    tasks = []
    for task in args.tasks.split(","):
        arguments = ["--project", str(project), "--task", task]
        if task == "MatchPhotos":
            arguments += ["--set", f"keypoint_limit_per_mpx={args.kp}"]
        print(f"{task}: kp/mpx {args.kp}, threads {os.environ.get('OMP_NUM_THREADS', 'all')}",
              flush=True)
        result = run_task(cli, arguments, log, args.interval)
        result["task"] = task
        tasks.append(result)
        print(json.dumps(result, indent=1), flush=True)
        if result["status"] != "ok":
            break

    report = {
        "dataset": args.dataset,
        "images": len(images),
        "megapixels": image_megapixels(images[0]),
        "keypoint_limit_per_mpx": args.kp,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cpu_count": os.cpu_count(),
        "tasks": tasks,
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(project_dir, ignore_errors=True)


def summarize(reports: list[dict]) -> str:
    lines = ["| kp/mpx | threads | task | stage | peak RSS (MB) | seconds |",
             "|---:|---:|---|---|---:|---:|"]
    for report in sorted(reports, key=lambda r: (r["keypoint_limit_per_mpx"],
                                                 str(r["omp_num_threads"]))):
        threads = report["omp_num_threads"] or f"all ({report['cpu_count']})"
        for task in report["tasks"]:
            for stage in task["stages"]:
                lines.append(f"| {report['keypoint_limit_per_mpx']} | {threads} | {task['task']} "
                             f"| {stage['stage']} | {stage['peak_rss_mb']:.0f} | {stage['seconds']:.0f} |")
            if task["status"] != "ok":
                lines.append(f"| {report['keypoint_limit_per_mpx']} | {threads} | {task['task']} "
                             f"| **{task['status']}** | | |")
    first = reports[0] if reports else {}
    head = (f"# MatchPhotos memory profile\n\n{first.get('dataset')}: {first.get('images')} images "
            f"at {first.get('megapixels') or 0:.1f} MP.\n\n")
    return head + "\n".join(lines) + "\n"


def command_summarize(args: argparse.Namespace) -> None:
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
    markdown = summarize(reports)
    Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--dataset", default="gcp-rtk")
    run.add_argument("--work", required=True)
    run.add_argument("--cache", required=True)
    run.add_argument("--kp", type=int, default=1000)
    run.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    run.add_argument("--interval", type=float, default=0.25)
    run.add_argument("--out", required=True)
    run.set_defaults(func=command_run)
    summary = sub.add_parser("summarize")
    summary.add_argument("--out-md", required=True)
    summary.add_argument("reports", nargs="+")
    summary.set_defaults(func=command_summarize)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
