#!/usr/bin/env python3
"""Download, run and measure the alphashape CLI on a public dataset.

One invocation drives a whole benchmark dataset:

  1. read ``datasets.json`` and lay out a cache directory,
  2. download each archive from its publisher and verify its SHA-256 (byte count
     first, so a truncated download fails fast instead of hashing a partial file),
  3. extract the images and the reference products,
  4. run the CLI pipeline task by task on the default parameters, timing each
     stage and sampling peak resident memory,
  5. hand the products to ``metrics.evaluate`` and write the metrics JSON.

Everything is addressable by path so the CI job can cache steps 2-3 across runs.
A run is reproducible from the manifest alone: no imagery is committed and no
step reaches a network resource that is not named in ``datasets.json``.

Usage:

    python3 tools/bench/run_bench.py --dataset aerial-nadir --work build/bench
    python3 tools/bench/run_bench.py --dataset aerial-nadir --list
    python3 tools/bench/run_bench.py --virtualenv build/bench/venv ...

``--list`` prints what the dataset would download and from where, without
touching the network. It is the check to run after editing the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "datasets.json"

# The bench is a set of scripts, not an installed package: ``metrics`` is a
# sibling module, and putting its directory on the path is what lets the file run
# both as ``python3 tools/bench/run_bench.py`` and under the test discovery.
#
# ``metrics`` itself is imported where it is used, not here: it needs numpy, and the
# manifest commands (--list, --list-datasets, --manifest-hash) are what a bare
# checkout and the workflow's plan job run first. Importing it at module scope made
# those fail on a machine without numpy, which is a needless prerequisite for
# printing a URL.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# The pipeline, in the order the README documents it. Defaults only: the point of
# a benchmark is to measure what a user gets without tuning, so a run that needs a
# parameter tweak to work is a finding, not something to work around here.
PIPELINE = [
    "MatchPhotos",
    "AlignCameras",
    "BuildDepthMaps",
    "BuildPointCloud",
    "BuildDem",
    "BuildOrthomosaic",
]

# Products exported at the end, by task and destination.
EXPORTS = [
    ("ExportDem", "dem.tif"),
    ("ExportOrthomosaic", "ortho.tif"),
    ("ExportPointCloud", "cloud.ply"),
]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class RemoteFile:
    """One download: a URL, the expected SHA-256 and the expected byte count."""

    url: str
    sha256: str
    bytes: int


def load_manifest(path: Path = MANIFEST) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("version") != 1:
        raise SystemExit(f"{path}: unsupported manifest version {manifest.get('version')!r}")
    return manifest


def manifest_hash(manifest: dict) -> str:
    """A stable digest of the manifest, for the workflow cache key.

    Cache invalidation has to follow the *content* of the manifest, not the file's
    mtime: a renamed dataset or a corrected hash must produce a different key so
    the cache is not reused against a dataset that no longer matches.
    """
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class ParamError(ValueError):
    """A dispatch parameter string that is not a list of well-formed ``key=value``."""


class DepthParamError(ParamError):
    """A ``depth_params`` string that is not a list of well-formed ``key=value``."""


class DemParamError(ParamError):
    """A ``dem_params`` string that is not a list of well-formed ``key=value``."""


# A task parameter name is what the engine declares: lowercase words and
# underscores (``cost_type``, ``max_cost``, ``max_neighbors``). A value is a number,
# a word, or a decimal (``census``, ``-1``, ``0.05``). Both are deliberately
# narrower than "anything without a space": the string arrives from a workflow
# input, so it is untrusted text reaching an argv list, and a value containing a
# semicolon or a quote has no legitimate use here.
PARAM_KEY = re.compile(r"^[a-z_]+$")
PARAM_VALUE = re.compile(r"^[A-Za-z0-9_.-]+$")

# Backwards-compatible names: these two patterns were documented as the depth
# parameter grammar and are the same grammar every dispatch input uses.
DEPTH_PARAM_KEY = PARAM_KEY
DEPTH_PARAM_VALUE = PARAM_VALUE


def parse_params(text: str | None, error: type[ParamError] = ParamError) -> dict[str, str]:
    """Parse a workflow's optional ``key=value ...`` string into a dict.

    ``"cost_type=census max_cost=0.4"`` becomes
    ``{"cost_type": "census", "max_cost": "0.4"}``. Everything is validated here,
    at the system boundary, because after this point the values are handed to the
    CLI as ``--set key=value`` and a malformed pair would surface as a stage
    failure hours into a run — or worse, as a silently ignored parameter.

    A missing separator, an empty key or value, a character outside the two
    patterns, and a repeated key are all errors rather than being skipped: a typo
    that is quietly dropped would produce a run that looks like an experiment and
    is actually the default, which is exactly the confusion these flags exist to
    avoid.

    ``error`` is the exception class the caller wants, so each dispatch input can
    name itself in the error it raises while sharing one grammar.
    """
    params: dict[str, str] = {}
    for token in (text or "").split():
        key, separator, value = token.partition("=")
        if not separator:
            raise error(f"{token!r} is not a key=value pair")
        if not PARAM_KEY.match(key):
            raise error(f"{key!r} is not a valid parameter name")
        if not PARAM_VALUE.match(value):
            raise error(f"{value!r} is not a valid value for {key}")
        if key in params:
            raise error(f"{key} is given more than once")
        params[key] = value
    return params


def parse_depth_params(text: str | None) -> dict[str, str]:
    """Parse the workflow's optional ``depth_params`` string (BuildDepthMaps)."""
    return parse_params(text, DepthParamError)


def parse_dem_params(text: str | None) -> dict[str, str]:
    """Parse the workflow's optional ``dem_params`` string (BuildDem)."""
    return parse_params(text, DemParamError)


def merge_task_params(dataset: dict, task: str, params: dict[str, str]) -> dict:
    """The dataset with ``params`` merged into its ``task`` overrides.

    The merge is over the dataset's own entries, not a replacement of them: an
    experiment such as ``cost_type=census`` keeps the measured ``downscale`` of the
    control run, which is what makes the two comparable at all. Nothing is mutated —
    the manifest is loaded once and shared by every run in a job, so an in-place
    update here would leak into whatever ran next.
    """
    if not params:
        return dataset
    overrides = {name: dict(values or {}) for name, values in (dataset.get("param_overrides") or {}).items()}
    merged = overrides.setdefault(task, {})
    merged.update(params)
    return {**dataset, "param_overrides": overrides}


def merge_depth_params(dataset: dict, depth_params: dict[str, str]) -> dict:
    """The dataset with ``depth_params`` merged into its BuildDepthMaps overrides."""
    return merge_task_params(dataset, "BuildDepthMaps", depth_params)


def merge_dem_params(dataset: dict, dem_params: dict[str, str]) -> dict:
    """The dataset with ``dem_params`` merged into its BuildDem overrides."""
    return merge_task_params(dataset, "BuildDem", dem_params)


def downloads_for(dataset: dict) -> list[tuple[str, RemoteFile]]:
    """Every (cache file name, remote file) pair a dataset needs, deduplicated.

    ``download`` is the name the file takes in the cache, and its extension is what
    decides how it is unpacked. It is deliberately independent of the extracted
    path: a reference unzips to ``reference/ortho.tif`` but arrives as
    ``processed.zip``, and naming the cache file after the extracted path would make
    the archive look like a GeoTIFF.

    Several reference members can come out of one archive; that archive appears
    once here and the caller extracts each member from it, rather than fetching it
    again per member.
    """
    out: list[tuple[str, RemoteFile]] = []
    seen: set[str] = set()

    images = dataset["images_archive"]
    out.append(
        (
            images["download"],
            RemoteFile(images["url"], images["sha256"], int(images["bytes"])),
        )
    )
    seen.add(images["url"])

    for reference in dataset["reference_files"]:
        url = reference.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(
            (
                reference["download"],
                RemoteFile(url, reference["sha256"], int(reference["bytes"])),
            )
        )
    return out


def describe(dataset: dict) -> str:
    """Human-readable summary for ``--list`` and the workflow's job summary."""
    lines = [
        f"{dataset['title']}",
        f"  publisher: {dataset['publisher']}",
        f"  licence:   {dataset['licence']['spdx']} ({dataset['licence']['url']})",
        f"  flight:    {dataset['flight']}",
    ]
    if dataset.get("doi"):
        lines.append(f"  doi:       {dataset['doi']}")
    total = 0
    for name, remote in downloads_for(dataset):
        total += remote.bytes
        lines.append(f"  download:  {name} <- {remote.url}")
        lines.append(f"             {remote.bytes:,} bytes, sha256 {remote.sha256}")
    lines.append(f"  total download: {total / 1e6:.1f} MB")
    references = [ref["kind"] for ref in dataset["reference_files"]]
    lines.append(f"  reference products: {', '.join(references) if references else 'none'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Download and verify
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Dataset mirrors are university and institute hosts, not a CDN: three separate CI
# runs died mid-download with ``ssl.SSLEOFError``/``URLError`` on a several-hundred
# MB archive. One dropped connection must not cost the whole job, so a transfer is
# retried a bounded number of times, with exponential backoff, and RESUMES from the
# bytes already on disk when the server honours a Range request. The attempt count
# is a module constant rather than a literal so a test can drive the loop directly.
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_BACKOFF_S = 5.0
DOWNLOAD_TIMEOUT_S = 120


def _is_transient_download_error(error: BaseException) -> bool:
    """True for a network failure that is worth retrying.

    The distinction is the point: a 403 or a 404 will say the same thing on every
    attempt, so retrying it only hides the reason behind a delay, while a dropped
    TLS stream, a reset connection or a read timeout is a mirror having a bad
    moment and is very likely to succeed immediately after.

    ``urllib`` wraps the low-level SSL error, so the ``URLError`` check covers the
    ``SSLEOFError`` that broke the last three runs; the raw types are listed as well
    because ``_fetch_once`` is called directly by the tests.
    """
    if isinstance(error, urllib.error.HTTPError):
        # 408 and 429 are "come back later" by definition; 5xx is a broken mirror.
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(
        error,
        (
            urllib.error.URLError,
            ssl.SSLError,
            http.client.HTTPException,
            ConnectionError,
            TimeoutError,
            socket.timeout,
        ),
    )


def _fetch_once(remote: RemoteFile, tmp: Path, *, timeout: float = DOWNLOAD_TIMEOUT_S) -> None:
    """Stream one response into ``tmp``, appending when a partial file exists.

    When ``tmp`` already holds bytes from an interrupted attempt the request asks
    for the remainder with a Range header. A mirror that ignores the header answers
    200 with the whole file, which must NOT be appended to the partial bytes — that
    would produce a corrupt file whose size still looks plausible — so the write
    mode falls back to truncation in that case. The caller's size and SHA-256 check
    is what ultimately decides; this function only avoids re-fetching what is
    already on disk.
    """
    have = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "alphashape-quality-bench/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(remote.url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp, "ab" if have else "wb") as out:
        # A 206 is the server honouring the Range request; anything else means the
        # body starts at byte 0 and the partial file has to be discarded.
        if have and getattr(response, "status", 0) != 206:
            out.seek(0)
            out.truncate()
        shutil.copyfileobj(response, out, length=1 << 20)


def download(
    remote: RemoteFile,
    destination: Path,
    *,
    attempts: int = DOWNLOAD_ATTEMPTS,
    backoff_s: float = DOWNLOAD_BACKOFF_S,
) -> None:
    """Fetch ``remote`` to ``destination``, verifying size and hash.

    A file already present with the right hash is left alone, which is what makes
    a cached work directory cheap. A file with the right size but the wrong hash
    is deleted rather than overwritten in place, so a subsequent run re-downloads
    instead of hashing a corrupt cache every time.

    A failed transfer is retried up to ``attempts`` times with exponential backoff
    and, where the server allows it, resumed rather than restarted. Every attempt
    is verified the same way, so a resume that silently produced a wrong file is
    caught exactly like a fresh download would be.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if destination.stat().st_size == remote.bytes and sha256_file(destination) == remote.sha256:
            print(f"  cached   {destination.name} ({remote.bytes:,} bytes)")
            return
        print(f"  stale    {destination.name}: removing")
        destination.unlink()

    print(f"  download {remote.url}")
    print(f"           -> {destination} ({remote.bytes:,} bytes)")
    tmp = destination.with_suffix(destination.suffix + ".part")
    started = time.monotonic()
    reason = "no attempt was made"
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            delay = backoff_s * (2 ** (attempt - 2))
            print(f"  retry    {destination.name}: {reason}; attempt {attempt}/{attempts} in {delay:.0f} s")
            time.sleep(delay)
        # A complete-but-unverified file cannot be resumed — a Range request would
        # ask for zero bytes — so it is dropped before the attempt. Only a partial
        # file is kept, and that is the one resumption is for.
        if tmp.exists() and tmp.stat().st_size >= remote.bytes:
            tmp.unlink()
        try:
            _fetch_once(remote, tmp)
        except Exception as error:  # noqa: BLE001 - classified immediately below
            if isinstance(error, urllib.error.HTTPError) and error.code == 416:
                # The resume offset is not what the server expects. Restart the
                # file from byte zero instead of treating it as a hard failure.
                tmp.unlink(missing_ok=True)
                reason = "range not satisfiable; restarting from the first byte"
                continue
            if not _is_transient_download_error(error):
                tmp.unlink(missing_ok=True)
                raise SystemExit(f"{remote.url}: {error}") from error
            reason = f"{type(error).__name__}: {error}"
            continue

        size = tmp.stat().st_size
        if size < remote.bytes:
            # The connection ended early without raising, which is how a truncated
            # chunked body looks. Keep the bytes: the next attempt resumes.
            reason = f"short body: {size:,} of {remote.bytes:,} bytes"
            continue
        if size > remote.bytes:
            reason = f"overlong body: {size:,} of {remote.bytes:,} bytes"
            tmp.unlink()
            continue
        actual = sha256_file(tmp)
        if actual != remote.sha256:
            reason = f"sha256 mismatch (got {actual})"
            tmp.unlink()
            continue

        tmp.rename(destination)
        elapsed = time.monotonic() - started
        rate = size / elapsed / 1e6 if elapsed > 0 else 0.0
        print(f"  verified {destination.name} in {elapsed:.1f} s ({rate:.1f} MB/s)")
        return

    tmp.unlink(missing_ok=True)
    plural = "attempt" if attempts == 1 else "attempts"
    raise SystemExit(f"{remote.url}: download failed after {attempts} {plural} ({reason})")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def check_member_name(name: str, *, archive: Path) -> str:
    """Reject an archive member that would land outside the extraction root.

    The archives are downloaded, not authored here, and an archive is untrusted
    input in exactly the sense that matters: a member named ``../../etc/cron.d/x``
    or ``/etc/ld.so.preload`` writes wherever the runner's user can write. Python
    resolves a bare name inside ``destination`` for us, but it does so for the
    paths it is handed — ``zipfile.extract`` strips a leading ``/`` and drops ``..``
    components silently, which means the file lands somewhere other than where the
    manifest says and the run continues on the wrong data. ``7z`` is worse: it is a
    separate process that honours absolute paths and will happily write outside
    ``-o``. Checking once, before anything is unpacked, makes both cases a named
    error instead of a surprise.

    Windows separators are rejected too: a member is a POSIX path inside the
    archive, and ``..\\..\\x`` is only dangerous on a platform this job does not
    run on — but an archive carrying it is malformed, and a run that silently
    accepted it would be trusting a file that lied about its own layout.
    """
    if not name or "\x00" in name:
        raise SystemExit(f"{archive.name}: archive member has an empty or NUL-containing name")
    if "\\" in name:
        raise SystemExit(
            f"{archive.name}: archive member {name!r} uses a Windows separator; "
            "archive paths must be POSIX"
        )
    if name.startswith("/"):
        raise SystemExit(f"{archive.name}: archive member {name!r} is an absolute path")
    # A drive-letter or UNC-style prefix ("C:foo", "//host/share") is absolute in
    # spirit even though it does not start with a bare "/".
    if re.match(r"^[A-Za-z]:", name) or name.startswith("//"):
        raise SystemExit(f"{archive.name}: archive member {name!r} is an absolute path")
    parts = Path(name).parts
    if any(part == ".." for part in parts):
        raise SystemExit(f"{archive.name}: archive member {name!r} escapes the extraction root")
    return name


def _seven_zip() -> str:
    seven = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
    if seven is None:
        raise SystemExit(
            "extracting a .7z needs 7-Zip on PATH "
            "(the workflow installs it with `apt-get install p7zip-full`)"
        )
    return seven


def seven_zip_members(archive: Path) -> list[str]:
    """The entry names inside a ``.7z``, without extracting anything.

    ``-slt -ba`` is the machine-readable listing: ``-slt`` prints one
    ``Key = value`` block per entry, ``-ba`` drops the banner and the summary, so
    every ``Path = `` line is a member and nothing else in the output is. Needed
    because 7-Zip enforces no path containment of its own, so the names have to be
    checked before the ``x`` that would act on them.
    """
    seven = _seven_zip()
    listing = subprocess.run(
        [seven, "l", "-slt", "-ba", str(archive)],
        check=True, capture_output=True, text=True,
    ).stdout
    return [line[len("Path = "):] for line in listing.splitlines() if line.startswith("Path = ")]


def extract_archive(archive: Path, destination: Path, *, members: list[str] | None = None) -> None:
    """Extract ``archive`` under ``destination``.

    ``members`` limits the extraction to the named entries, which is how a single
    reference archive is unpacked once even when several reference files come out
    of it. Zip and tar (including .tar.zst handled by the external `tar`) are
    supported; a `.7z` needs `7z`, which the workflow installs.

    Every entry is checked against :func:`check_member_name` BEFORE it is written,
    including the entries of a full extraction: an archive is downloaded from a
    third party, and a member that resolves outside ``destination`` is either a
    malformed archive or an attempt to write outside it. Both fail the run here,
    with the offending name, rather than landing a file somewhere else.
    """
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    name = archive.name.lower()

    def checked(entry: str) -> str:
        return check_member_name(entry, archive=archive)

    def check_resolved(target: Path, entry: str) -> None:
        # Belt and braces on top of the name check: what matters is where the file
        # ends up, and only the resolved path says that (symlinked parents, a
        # destination that is itself relative).
        resolved = target.resolve()
        if not resolved.is_relative_to(root):
            raise SystemExit(
                f"{archive.name}: archive member {entry!r} resolves outside "
                f"the extraction root ({resolved} is not under {root})"
            )

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            wanted = members if members is not None else bundle.namelist()
            for member in wanted:
                checked(member)
                check_resolved(destination / member, member)
            if members is None:
                bundle.extractall(destination)
            else:
                for member in members:
                    bundle.extract(member, destination)
        return

    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.zst", ".tar.xz", ".tar.bz2")):
        with tarfile.open(archive) as bundle:
            wanted = members if members is not None else [entry.name for entry in bundle.getmembers()]
            for member in wanted:
                checked(member)
                check_resolved(destination / member, member)
            if members is None:
                # `data` refuses absolute paths on its own, but the names are
                # already checked above and leaving it off keeps the behaviour
                # identical to the named-member path.
                bundle.extractall(destination)
            else:
                for member in members:
                    bundle.extract(member, destination)
        return

    if name.endswith(".7z"):
        seven = _seven_zip()
        present = seven_zip_members(archive)
        wanted = members if members is not None else present
        for member in wanted:
            checked(member)
            check_resolved(destination / member, member)
        if members is not None:
            listed = set(present)
            missing = [member for member in members if member not in listed]
            if missing:
                # 7-Zip exits 0 on an unmatched name and unpacks nothing, which
                # would leave a reference file silently absent and the scoring
                # step reading the wrong thing (or failing much later).
                raise SystemExit(
                    f"{archive.name}: archive has no member(s) {missing!r}; it holds "
                    f"{len(present)} entries, e.g. {present[:3]!r}"
                )
        command = [seven, "x", "-y", f"-o{destination}", str(archive)]
        if members:
            command.extend(members)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        return

    raise SystemExit(f"{archive}: unsupported archive type")


def collect_images(dataset: dict, work: Path) -> list[Path]:
    """The image files the manifest says the archive supplies, sorted by name."""
    pattern = dataset["images_archive"].get("extract_glob")
    images_root = work / "images"
    if pattern:
        found = sorted(images_root.glob(pattern))
    else:
        found = sorted(
            path
            for path in images_root.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        )
    expected = dataset.get("expected_images")
    if expected is not None and len(found) != expected:
        raise SystemExit(
            f"{dataset['title']}: expected {expected} images, found {len(found)} under "
            f"{images_root} with pattern {pattern!r}"
        )
    if not found:
        raise SystemExit(f"{dataset['title']}: no images matched {pattern!r}")
    return found


def select_subset(images: list[Path], limit: int | None) -> tuple[list[Path], dict]:
    """The ``limit`` images nearest the flight's centre, as a spatial cluster.

    Needed because the dense stage's cost is images x pixels, and this dataset is
    102 frames at 20 MP: at any downscale that still resolves the 2.4 cm GSD, the
    full set does not fit a weekly job on a shared CPU runner. A subset is the only
    way to keep the run inside its budget without falling back to a resolution so
    coarse that none of the metrics mean anything.

    The selection is the ``limit`` images with the smallest squared distance to the
    MEDIAN position of the whole set (longitude/latitude, or file order when the
    images carry no GPS). On a lawnmower flight that is a contiguous block around
    the middle of the site — a real spatial neighbourhood, so consecutive frames
    overlap the way the stereo stage needs. It is not a strided sample across the
    whole flight (those only just overlap and produce sparse depth maps for a
    reason that has nothing to do with the code), and it is not "the first N" in
    file order (which is the edge of the site, where a drone's first passes can be
    the least overlapping).

    Deterministic: the same images, in the same order, every run of the dataset.

    Returns ``(images, selection)`` where ``selection`` records what was done, for
    the report's provenance. ``selection["images"]`` is the actual list of selected
    file names, not a first..last range: the cluster is not contiguous in file
    order when the flight crosses the centroid more than once, and a range would
    claim images that were not used.
    """
    if limit is None or limit <= 0 or len(images) <= limit:
        return images, {
            "requested": limit,
            "selected": len(images),
            "mode": "all",
            "images": [path.name for path in images],
        }

    positions = _image_positions(images)
    if positions is not None:
        centre_lon = float(np_median([lon for lon, _ in positions]))
        centre_lat = float(np_median([lat for _, lat in positions]))
        order = sorted(
            range(len(images)),
            key=lambda index: (positions[index][0] - centre_lon) ** 2
            + (positions[index][1] - centre_lat) ** 2,
        )
        chosen = sorted(order[:limit])
    else:
        start = max(0, (len(images) - limit) // 2)
        chosen = list(range(start, start + limit))

    selected = [images[index] for index in chosen]
    return selected, {
        "requested": limit,
        "selected": len(selected),
        "mode": "nearest-to-centroid cluster" if positions is not None
                else "middle file-order block",
        "images": [path.name for path in selected],
    }


def np_median(values: list[float]) -> float:
    """Median of a list of floats, without pulling numpy in for one number."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _image_positions(images: list[Path]) -> list[tuple[float, float]] | None:
    """(longitude, latitude) per image from EXIF, or None when any is missing.

    Reads only the EXIF block, and only the first and last few hundred bytes of
    the file where Pillow can, so selecting a subset does not decode 102 full-size
    JPEGs. A dataset without GPS gets None and the caller falls back to file order.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:  # pragma: no cover - Pillow is in the workflow
        return None

    positions: list[tuple[float, float]] = []
    for path in images:
        try:
            with Image.open(path) as image:
                exif = image.getexif()
                gps = exif.get_ifd(34853)
        except Exception:  # noqa: BLE001 - an unreadable EXIF means "no position"
            return None
        lat = gps.get(2)
        lat_ref = gps.get(1)
        lon = gps.get(4)
        lon_ref = gps.get(3)
        if not lat or not lon:
            return None
        lat_value = _dms(lat) * (-1.0 if lat_ref in ("S", b"S") else 1.0)
        lon_value = _dms(lon) * (-1.0 if lon_ref in ("W", b"W") else 1.0)
        if lat_value == 0.0 and lon_value == 0.0:
            return None
        positions.append((lon_value, lat_value))
    return positions


def _dms(parts) -> float:
    """Degrees from an EXIF (degrees, minutes, seconds) triple."""
    degrees, minutes, seconds = (float(value) for value in parts[:3])
    return degrees + minutes / 60.0 + seconds / 3600.0


def reference_paths(dataset: dict, work: Path) -> dict:
    """Where each reference product landed, keyed by ``kind``.

    A kind that appears more than once (ETH3D ships two laser scans for the same
    scene) becomes a list, so the caller can score against either or both.
    """
    out: dict = {}
    for reference in dataset["reference_files"]:
        destination = str(work / reference["path"])
        kind = reference["kind"]
        if kind in out:
            existing = out[kind]
            out[kind] = existing + [destination] if isinstance(existing, list) else [existing, destination]
        else:
            out[kind] = destination
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_cli(args: list[str], *, log: Path | None = None, timeout: float | None = None) -> float:
    """Run one CLI invocation, returning its wall time in seconds.

    A non-zero exit is fatal: the benchmark is about a pipeline that works on real
    photographs, so a stage that errors is a finding to fix, not a number to record
    as zero. stderr goes to ``log`` when one is given, which is what makes a CI
    failure diagnosable after the fact.

    ``timeout`` is a hard per-stage ceiling. It exists so a stage that is far slower
    than expected fails with the stage's own name instead of waiting out the whole
    job's timeout and reporting nothing: the first version of this benchmark spent
    five hours in BuildDepthMaps and produced no evidence at all, which is worse
    than a clear failure.
    """
    cli = os.environ.get("ALPHASHAPE_CLI")
    if not cli:
        raise SystemExit("ALPHASHAPE_CLI is not set; see tools/bench/README.md")
    command = [cli, *args]
    print(f"  $ {' '.join(command)}")
    started = time.monotonic()
    # Both streams go to the log. The engine writes its Info diagnostics — the
    # bundle-adjustment RMS, the georeferencing outcome — to STDOUT, and its
    # progress to stderr, so discarding stdout would throw away the one number that
    # says whether the alignment is any good.
    try:
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "ab") as errlog:
                result = subprocess.run(
                    command, stdout=errlog, stderr=errlog, timeout=timeout
                )
        else:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(
            f"alphashape-cli exceeded its {timeout:.0f} s budget: {' '.join(args)}"
            + (f" (see {log})" if log else "")
        ) from None
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise SystemExit(_failure_message(result.returncode, args, log))
    return elapsed


def _failure_message(returncode: int, args: list[str], log: Path | None) -> str:
    """What to print when a stage fails.

    The reason matters more than the fact. A negative return code is a signal, and
    a signal of 9 on a CI runner means the kernel killed the process — almost always
    for memory — which no amount of reading a truncated progress log would reveal.
    So the signal case is named explicitly, and the last few non-progress lines of
    stderr are included because the CLI's error message is the one line that says
    what actually went wrong.
    """
    if returncode < 0:
        signal_number = -returncode
        name = {9: "SIGKILL (out of memory?)", 15: "SIGTERM", 11: "SIGSEGV"}.get(
            signal_number, f"signal {signal_number}"
        )
        head = f"alphashape-cli was killed by {name}"
    else:
        head = f"alphashape-cli exited {returncode}"
    message = f"{head}: {' '.join(args)}"
    if log is not None and log.exists():
        lines = [line for line in _tail(log, 40) if not _is_progress(line)]
        if lines:
            message += "\n  last output:\n    " + "\n    ".join(lines[-6:])
        message += f"\n  full log: {log}"
    return message


def _is_progress(line: str) -> bool:
    """True for the CLI's ``[stage] NN%`` progress lines."""
    stripped = line.strip()
    return stripped.startswith("[") and stripped.endswith("%") and "]" in stripped


def run_pipeline(
    work: Path,
    images: list[Path],
    overrides: dict | None = None,
    stage_budget_s: float | None = None,
    stage_budgets: dict | None = None,
) -> dict:
    """Create a project from ``images`` and run the pipeline, timing each stage.

    Every stage runs at the engine's DEFAULTS. ``overrides`` exists only for a
    dataset that cannot fit the job budget at full resolution, and whatever it
    contains is recorded in the returned dict so the report can say the run was not
    at defaults. That honesty is the point: a tuned run and a default run are not
    comparable, and the metric history would silently mix them otherwise.

    Peak RSS is the child's own reported peak via ``getrusage(RUSAGE_CHILDREN)``,
    since the CLI is a short-lived process; it is ``None`` where the platform does
    not report it rather than a guess.
    """
    project = work / "site.psx"
    log = work / "cli.log"
    stages: dict[str, float] = {}
    overrides = overrides or {}
    started = time.monotonic()
    # The log is truncated per RUN, not per stage: it is one run's evidence, and a
    # stale log from a previous attempt would bury this one's error.
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_bytes(b"")

    stages["AddPhotos"] = _timed(run_cli, ["--new", str(project), "--add-images",
                                           *[str(p) for p in images]], log=log)
    for task in PIPELINE:
        arguments = ["--project", str(project), "--task", task]
        for key, value in (overrides.get(task) or {}).items():
            arguments += ["--set", f"{key}={cli_value(value)}"]
        budget = None
        if stage_budget_s:
            budget = float(stage_budgets.get(task, stage_budget_s)) if stage_budgets else stage_budget_s
        stages[task] = _timed(run_cli, arguments, log=log, timeout=budget)
    total = time.monotonic() - started

    products: dict[str, str] = {}
    for task, filename in EXPORTS:
        destination = work / "products" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        stages[task] = _timed(
            run_cli,
            ["--project", str(project), "--task", task, "--set", f"path={destination}"],
            log=log,
        )
        if not destination.exists():
            raise SystemExit(f"{task} reported success but {destination} is missing")
        products[filename] = str(destination)

    return {
        "stages_s": stages,
        "total_s": total,
        "project": str(project),
        "products": products,
        "peak_rss_mb": peak_rss_mb(),
        "log_bytes": log.stat().st_size if log.exists() else 0,
        "log_tail": _tail(log, 20),
        # Non-empty means this run was NOT at defaults; the report prints it.
        "param_overrides": overrides,
    }


def peak_rss_mb() -> float | None:
    """Peak RSS of this process tree, in MB, or None when the platform hides it."""
    try:
        import resource  # noqa: PLC0415 - POSIX only

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        # macOS reports bytes, Linux kilobytes; both are covered by the platform
        # check rather than a magic constant.
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024
    except (ImportError, AttributeError):  # pragma: no cover - non-POSIX
        return None


def cli_value(value) -> str:
    """Render a manifest override the way the CLI's --set parser reads it.

    The parser accepts exactly ``true``/``false``/``1``/``0`` for a bool — not
    Python's ``True``/``False``, which is what an f-string on a JSON-loaded value
    produces. Getting that wrong is a one-line bug that costs a full pipeline run
    before it surfaces, so the conversion lives here with a test rather than
    inline in the argument loop.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _timed(function, *args, **kwargs):
    """Call ``function`` and return its own elapsed seconds (it reports them)."""
    return function(*args, **kwargs)


def engine_alignment_report(log: Path) -> dict:
    """The engine's own alignment numbers, parsed from its Info output.

    ``AlignCameras`` logs "registered N/M cameras, RMS X px" and whether it
    georeferenced. Those are the engine's OWN figures, computed inside the bundle
    adjustment on the same observations, so quoting them beside this module's
    independently re-projected RMS is what turns a suspicious number into a
    diagnosis: agreement means the metric is measuring what it claims, and a large
    disagreement means one of the two is reading the project wrongly.
    """
    out: dict = {}
    if not log.exists():
        return out
    import re  # noqa: PLC0415

    text = log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"AlignCameras: registered (\d+)/(\d+) cameras, RMS ([0-9.eE+-]+) px", text)
    if match:
        out["registered"] = int(match.group(1))
        out["offered"] = int(match.group(2))
        out["rms_px"] = float(match.group(3))
    if "AlignCameras: georeferenced from" in text:
        out["georeferenced"] = True
    elif "AlignCameras: not georeferenced" in text:
        out["georeferenced"] = False
    return out


def _tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return [line.rstrip("\n") for line in content[-lines:]]


# ---------------------------------------------------------------------------
# Project inspection (camera statistics)
# ---------------------------------------------------------------------------


def inspect_project(project: Path) -> dict:
    """Camera-count and alignment statistics from the saved project.

    Reads the .psx manifest, opens ``<stem>.files/project.zip`` and parses
    ``doc.xml`` for the sensors and cameras. Kept here rather than in ``metrics``
    because it is project-format knowledge, not metric maths; the numbers it feeds
    (aligned cameras, reprojection error) are computed by ``metrics``.

    A project that cannot be read returns ``{"error": ...}`` instead of raising:
    the run already failed loudly if a stage broke, and an unreadable project
    after a successful pipeline is a finding worth reporting, not a crash.
    """
    try:
        base = project.with_suffix("")
        bundle = base.parent / f"{base.name}.files" / "project.zip"
        if not bundle.exists():
            return {"error": f"{bundle} not found"}
        with zipfile.ZipFile(bundle) as archive:
            xml = archive.read("doc.xml").decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"error": f"cannot read project: {exc}"}

    import xml.etree.ElementTree as ET  # noqa: PLC0415

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        return {"error": f"doc.xml is not well-formed: {exc}"}

    chunk = root.find("./chunks/chunk")
    if chunk is None:
        return {"error": "doc.xml has no chunk"}
    cameras = chunk.findall("./cameras/camera")
    aligned = [c for c in cameras if c.get("enabled") == "true"]
    sensors = chunk.findall("./sensors/sensor")
    return {
        "cameras_total": len(cameras),
        "cameras_aligned": len(aligned),
        "cameras_aligned_fraction": (len(aligned) / len(cameras)) if cameras else 0.0,
        "sensors": len(sensors),
        "has_transform": chunk.find("./transform") is not None,
        "epsg": (chunk.find("./crs").get("epsg") if chunk.find("./crs") is not None else None),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    dataset_name: str,
    work: Path,
    *,
    keep: bool = False,
    depth_params: dict[str, str] | None = None,
    dem_params: dict[str, str] | None = None,
) -> dict:
    """Run one dataset and score it.

    ``depth_params`` are extra ``BuildDepthMaps`` keyword arguments from a manual
    dispatch, already validated by :func:`parse_depth_params`; ``dem_params`` are
    the same for ``BuildDem``. Each is merged over the dataset's own manifest
    overrides for its task, so an experiment such as ``cost_type=census`` keeps the
    dataset's measured ``downscale`` and is therefore comparable with the control
    run of the same dataset. The manifest's own dicts are never mutated: they are
    copied, because the manifest is loaded once and a mutation would leak into the
    next run.
    """
    manifest = load_manifest()
    if dataset_name not in manifest["datasets"]:
        raise SystemExit(
            f"unknown dataset {dataset_name!r}; known: {', '.join(sorted(manifest['datasets']))}"
        )
    dataset = manifest["datasets"][dataset_name]
    dataset = merge_depth_params(dataset, depth_params or {})
    dataset = merge_dem_params(dataset, dem_params or {})
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    print(f"== {dataset['title']} ({dataset_name})")
    print(f"   licence {dataset['licence']['spdx']}")

    print("-- fetch")
    for path, remote in downloads_for(dataset):
        download(remote, cache / Path(path).name)

    print("-- extract")
    archive = cache / dataset["images_archive"]["download"]
    image_root = work / "images"
    if not any(image_root.rglob("*")):
        extract_archive(archive, image_root)
    else:
        print(f"  cached   {image_root}")

    # References: one extraction per download, then the named members. Keyed by the
    # cache file name so a single archive that supplies several members (two laser
    # scans, both GeoTIFFs from one ODM run) is unpacked once.
    by_download: dict[str, list[str]] = {}
    for reference in dataset["reference_files"]:
        source = reference.get("from_archive") or reference["download"]
        by_download.setdefault(source, [])
        if reference.get("archive_member"):
            by_download[source].append(reference["archive_member"])

    for name, members in by_download.items():
        destination = work / "reference_raw" / Path(name).stem
        if destination.exists() and any(destination.rglob("*")):
            print(f"  cached   {destination}")
            continue
        extract_archive(cache / name, destination, members=members or None)

    # Promote the reference members to their manifest paths.
    reference_root = work / "reference"
    for reference in dataset["reference_files"]:
        target = work / reference["path"]
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        archive_name = reference.get("from_archive") or reference["download"]
        source_root = work / "reference_raw" / Path(archive_name).stem
        member = reference.get("archive_member")
        if member is None:
            found = next(iter(source_root.rglob("*")))
        else:
            found = source_root / member
        if not found.exists():
            raise SystemExit(f"{dataset_name}: reference {member!r} missing after extraction")
        shutil.copy2(found, target)
    references = reference_paths(dataset, work)

    print("-- pipeline")
    available = collect_images(dataset, work)
    images, selection = select_subset(available, dataset.get("max_images"))
    print(f"   {len(images)} of {len(available)} images ({selection['mode']})")
    if selection["mode"] != "all":
        # The actual list, not a first..last range: the cluster is chosen by
        # position, so its members are not always contiguous in file order and a
        # range would name images that were not used.
        print("   selected: " + ", ".join(selection["images"]))
    overrides = dataset.get("param_overrides") or {}
    if overrides:
        print(f"   parameter overrides: {overrides}")
    timing = run_pipeline(
        work, images, overrides,
        dataset.get("stage_budget_s"), dataset.get("stage_budgets"),
    )

    print("-- inspect")
    camera_stats = inspect_project(Path(timing["project"]))
    engine_report = engine_alignment_report(work / "cli.log")
    if engine_report:
        camera_stats["engine_report"] = engine_report
    if "error" not in camera_stats:
        camera_stats["reprojection"] = _reprojection(work, Path(timing["project"]))
    else:
        print(f"   {camera_stats['error']}")

    print("-- metrics")
    import metrics  # noqa: PLC0415 - see the note at the top on why this is late

    spec = metrics.DatasetSpec(
        name=dataset_name,
        images=[str(p) for p in images],
        ortho_reference=references.get("orthomosaic"),
        dem_reference=references.get("dsm"),
        laser_reference=references.get("laser_point_cloud"),
        nominal_gsd_m=dataset.get("nominal_gsd_m"),
        licence=dataset["licence"],
    )
    products = timing["products"]
    result = metrics.evaluate(
        spec,
        ortho_path=products.get("ortho.tif"),
        dem_path=products.get("dem.tif"),
        cloud_path=products.get("cloud.ply"),
        camera_stats=camera_stats,
    )
    result["timing"] = timing
    result["manifest_hash"] = manifest_hash(manifest)
    # A default run's JSON is unchanged byte for byte: each key is present only when
    # the run was dispatched with extra parameters. The report keys "is this an
    # experiment" off the keys' presence, so their absence means "control run" for
    # every metrics file that predates these flags as well.
    if depth_params:
        result["depth_params"] = dict(depth_params)
    if dem_params:
        result["dem_params"] = dict(dem_params)
    result["dataset_manifest"] = {
        "title": dataset["title"],
        "doi": dataset.get("doi"),
        "flight": dataset["flight"],
        "nominal_gsd_m": dataset.get("nominal_gsd_m"),
        "images": len(images),
        "images_available": len(available),
        "image_selection": selection,
    }
    return result


def _reprojection(work: Path, project: Path) -> dict:
    """Tie-point reprojection RMS from the saved project.

    This is the one reference-free accuracy number available without ground truth:
    re-project every triangulated tie point through every observing camera's pose
    and calibration, then compare with the keypoint the detector actually found.
    It measures the alignment's internal consistency, not its absolute accuracy —
    a self-consistent but wrongly scaled model scores well here, which is why the
    report never presents it alone.

    The binaries are read straight out of the project archive so this works on a
    machine with no Python bindings built.
    """
    import numpy as np  # noqa: PLC0415

    import metrics  # noqa: PLC0415 - late for the reason in the module note

    base = project.with_suffix("")
    bundle = base.parent / f"{base.name}.files" / "project.zip"
    try:
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
            tracks_entry = next((n for n in names if n.endswith("/tie_points/tracks.bin")), None)
            if tracks_entry is None:
                return {"skipped": "project has no tracks"}
            prefix = tracks_entry.rsplit("/tie_points/", 1)[0]
            tracks = _decode_tracks(archive.read(tracks_entry))

            keypoints: dict[int, np.ndarray] = {}
            for name in names:
                if name.startswith(prefix + "/tie_points/camera_") and name.endswith(".kpt"):
                    camera_id = int(name.rsplit("_", 1)[1].split(".")[0])
                    keypoints[camera_id] = _decode_keypoints(archive.read(name))

            cameras = _decode_cameras(archive.read("doc.xml"))
            sensors = _decode_sensors(archive.read("doc.xml"))
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return {"skipped": f"cannot read tie points: {exc}"}

    observed = []
    projected = []
    camera_of: list[int] = []
    for track in tracks:
        if track["point"] is None:
            continue
        point = np.asarray([track["point"]], dtype=np.float64)
        for camera_id, keypoint_index in track["observations"]:
            camera = cameras.get(camera_id)
            if camera is None:
                continue
            sensor = sensors.get(camera["sensor_id"])
            if sensor is None:
                continue
            pixels, ok = metrics.project_pixels(point, camera["pose"], sensor)
            if not ok[0]:
                continue
            kp = keypoints.get(camera_id)
            if kp is None or keypoint_index >= kp.shape[0]:
                continue
            # Keypoint coordinates are pixel-CORNER based (the detector's integer
            # index is the top-left corner of that pixel), while Calibration::project
            # puts the CENTRE of the top-left pixel at (0.5, 0.5). The engine bridges
            # the two with keypoint_to_pixel (+0.5, +0.5); measuring without it would
            # charge every observation a constant half-pixel bias.
            observed.append(kp[keypoint_index][:2] + 0.5)
            projected.append(pixels[0])
            camera_of.append(camera_id)

    if not observed:
        return {"skipped": "no projectable observations"}
    observed_array = np.asarray(observed, dtype=np.float64)
    projected_array = np.asarray(projected, dtype=np.float64)
    stats = metrics.reprojection_stats(observed_array, projected_array)
    stats["tracks"] = len(tracks)

    # A per-camera median, so a disagreement with the engine's own RMS can be
    # located rather than guessed at. The engine computes its figure inside the
    # bundle adjustment on the same observations, so if the two disagree by an
    # order of magnitude, this says whether it is one camera or all of them — a
    # single outlier camera and a systematic misreading look identical in the
    # aggregate number.
    finite = np.isfinite(projected_array).all(axis=1) & np.isfinite(observed_array).all(axis=1)
    if finite.any():
        errors = np.linalg.norm(projected_array[finite] - observed_array[finite], axis=1)
        labels = np.asarray(camera_of, dtype=np.int64)[finite]
        worst: list[dict] = []
        for camera_id in np.unique(labels):
            subset = errors[labels == camera_id]
            worst.append({
                "camera": int(camera_id),
                "pairs": int(subset.size),
                "median_px": float(np.median(subset)),
            })
        worst.sort(key=lambda entry: -entry["median_px"])
        stats["worst_cameras"] = worst[:3]
        stats["best_camera_median_px"] = float(min(entry["median_px"] for entry in worst))
    return stats


def _decode_tracks(data: bytes) -> list[dict]:
    """Decode the engine's ATR2/ATR1 tracks sidecar (see common/io/sidecars.hpp)."""
    if len(data) < 8 or data[:4] != b"ATR2":
        if len(data) < 8 or data[:4] != b"ATR1":
            raise ValueError("tracks.bin: bad magic")
    magic = data[:4]
    count = struct.unpack_from("<I", data, 4)[0]
    at = 8
    tracks = []
    for _ in range(count):
        observations = struct.unpack_from("<I", data, at)[0]
        at += 4
        observations_list = []
        for _ in range(observations):
            camera_id, keypoint_index = struct.unpack_from("<iI", data, at)
            at += 8
            observations_list.append((camera_id, keypoint_index))
        has_point = data[at]
        at += 1
        point = None
        if has_point:
            point = struct.unpack_from("<ddd", data, at)
            at += 24
        tracks.append({"observations": observations_list, "point": point})
    return tracks


def _decode_keypoints(data: bytes) -> "object":
    """Decode the engine's AKP1 keypoints sidecar into an (n, 4) float array."""
    import numpy as np  # noqa: PLC0415

    if len(data) < 8 or data[:4] != b"AKP1":
        raise ValueError("keypoints: bad magic")
    count = struct.unpack_from("<I", data, 4)[0]
    at = 8
    rows = np.empty((count, 4), dtype=np.float64)
    for index in range(count):
        rows[index] = struct.unpack_from("<ffff", data, at)
        at += 16
    return rows


def _decode_cameras(xml_bytes: bytes) -> dict:
    """Parse <cameras> out of doc.xml: world->camera pose per camera id."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    root = ET.fromstring(xml_bytes.decode("utf-8", "replace"))
    chunk = root.find("./chunks/chunk")
    out: dict = {}
    if chunk is None:
        return out
    for camera in chunk.findall("./cameras/camera"):
        transform = camera.get("transform")
        if transform is None:
            continue
        values = [float(v) for v in transform.split()]
        if len(values) != 16:
            continue
        out[int(camera.get("id"))] = {
            "sensor_id": int(camera.get("sensor_id", -1)),
            "pose": np.asarray(values, dtype=np.float64).reshape(4, 4),
            "enabled": camera.get("enabled") == "true",
        }
    return out


def _decode_sensors(xml_bytes: bytes) -> dict:
    """Parse <sensors> out of doc.xml: the Brown-Conrady calibration per sensor."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    root = ET.fromstring(xml_bytes.decode("utf-8", "replace"))
    chunk = root.find("./chunks/chunk")
    out: dict = {}
    if chunk is None:
        return out
    for sensor in chunk.findall("./sensors/sensor"):
        calibration = sensor.find("./calibration")
        if calibration is None:
            continue
        entry = {
            # The writer emits lower-case type names; keep them verbatim and let
            # metrics.project_pixels interpret them case-insensitively.
            "type": (calibration.get("type") or "frame").strip().lower(),
            "width": int(calibration.get("width", 0)),
            "height": int(calibration.get("height", 0)),
        }
        for field in ("f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2"):
            entry[field] = float(calibration.get(field, 0.0))
        out[int(sensor.get("id"))] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="dataset name from datasets.json")
    parser.add_argument("--work", default="build/bench", help="work directory (default: build/bench)")
    parser.add_argument("--json", help="write the metrics JSON here")
    parser.add_argument("--list", action="store_true", help="describe the dataset and exit")
    parser.add_argument("--list-datasets", action="store_true", help="list dataset names and exit")
    parser.add_argument("--manifest-hash", action="store_true", help="print the cache key and exit")
    parser.add_argument(
        "--depth-params",
        default="",
        help='extra BuildDepthMaps keyword args, e.g. "cost_type=census max_cost=0.4"; '
        "marks the run as an experiment (never a baseline, never gated)",
    )
    parser.add_argument(
        "--dem-params",
        default="",
        help='extra BuildDem keyword args, e.g. "sibson_exact_max_sites=20000"; '
        "marks the run as an experiment (never a baseline, never gated)",
    )
    args = parser.parse_args(argv)

    try:
        depth_params = parse_depth_params(args.depth_params)
    except DepthParamError as exc:
        parser.error(f"--depth-params: {exc}")

    try:
        dem_params = parse_dem_params(args.dem_params)
    except DemParamError as exc:
        parser.error(f"--dem-params: {exc}")

    manifest = load_manifest()

    if args.list_datasets:
        for name, dataset in sorted(manifest["datasets"].items()):
            print(f"{name}\t{dataset['licence']['spdx']}\t{dataset['title']}")
        return 0

    if args.manifest_hash:
        print(manifest_hash(manifest))
        return 0

    if not args.dataset:
        parser.error("--dataset is required")

    dataset = manifest["datasets"].get(args.dataset)
    if dataset is None:
        parser.error(f"unknown dataset {args.dataset!r}")

    if args.list:
        print(describe(dataset))
        return 0

    result = run(args.dataset, Path(args.work), depth_params=depth_params, dem_params=dem_params)

    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"-- wrote {destination}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
