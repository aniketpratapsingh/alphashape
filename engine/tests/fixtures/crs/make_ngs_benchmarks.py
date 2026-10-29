#!/usr/bin/env python3
"""Regenerate tests/fixtures/crs/ngs_geoid_benchmarks.csv.

Reference values come from the National Geodetic Survey's public GEOID web
service,

    https://geodesy.noaa.gov/api/geoid/ght?lat=<lat>&lon=<lon>&model=<id>

which returns the NGS-published geoid height for a model at a position. The
model used here is USGG2012 (id 11) — NGS's *gravimetric* geoid, i.e. the
un-smoothed counterpart of a global model such as EGM2008, which is what a
global grid can be compared with. GEOID18 (id 14) is not used: it is fitted to
NAVD88 GPS-on-benchmark data and carries a national datum bias (~0.44 m here)
that would swamp a global-model comparison.

The per-point values are checked in, so the tests never touch the network and
CI needs no NGS access. Re-running this script re-fetches them.

Position selection is seeded (SEED below), so the same 50 points come back every
time. They are spread over the CONUS lowland box where USGG2012 is defined.
US-government public-domain data (NGS/NOAA), no usage restriction.

Usage:
    python3 make_ngs_benchmarks.py [--check]

--check only verifies the recorded values still match what is checked in
(useful when the service updates a model), without rewriting the file.
"""
import argparse
import pathlib
import random
import re
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "ngs_geoid_benchmarks.csv"
SERVICE = "https://geodesy.noaa.gov/api/geoid/ght"
MODEL_ID = 11          # USGG2012
MODEL_NAME = "USGG2012"
SEED = 20261029
COUNT = 50
# CONUS lowland box: latitude 26..48.5 N, longitude -122..-72 W. The service
# returns 0 for a point outside the model's coverage, which the checker rejects.
LAT_RANGE = (26.0, 48.5)
LON_RANGE = (-122.0, -72.0)
RETRIEVED = "2026-10-29"


def fetch(lat, lon):
    url = f"{SERVICE}?lat={lat}&lon={lon}&model={MODEL_ID}"
    with urllib.request.urlopen(url, timeout=30) as response:
        body = response.read().decode("utf-8")
    match = re.search(r'"geoidHeight"\s*:\s*(-?[0-9.]+)', body)
    if not match:
        raise SystemExit(f"no geoidHeight in the response for {lat},{lon}:\n{body}")
    return float(match.group(1))


def points():
    rng = random.Random(SEED)
    out = []
    for index in range(COUNT):
        lat = round(rng.uniform(*LAT_RANGE), 4)
        lon = round(rng.uniform(*LON_RANGE), 4)
        out.append((f"P{index + 1:02d}", lat, lon))
    return out


def rows():
    out = []
    for pid, lat, lon in points():
        height = fetch(lat, lon)
        if height == 0.0:
            raise SystemExit(f"{pid} at {lat},{lon} is outside USGG2012 coverage")
        out.append((pid, lat, lon, height))
    return out


def render(data):
    header = (
        "# NGS geoid heights, US government public domain (NOAA/NGS).\n"
        f"# Source: {SERVICE}?lat=<lat>&lon=<lon>&model={MODEL_ID} ({MODEL_NAME}),\n"
        f"# retrieved {RETRIEVED}; point selection seeded with {SEED},\n"
        f"# {COUNT} positions drawn uniformly from lat {LAT_RANGE[0]}..{LAT_RANGE[1]},\n"
        f"# lon {LON_RANGE[0]}..{LON_RANGE[1]} (CONUS lowland).\n"
        "# Regenerate with make_ngs_benchmarks.py; the tests read this file offline.\n"
    )
    body = "id,lat,lon,published_geoid_height_m\n"
    for pid, lat, lon, height in data:
        body += f"{pid},{lat:.4f},{lon:.4f},{height:.3f}\n"
    return header + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    data = rows()
    text = render(data)
    if args.check:
        if OUT.read_text() != text:
            raise SystemExit(f"{OUT} differs from what the service returns now")
        print(f"{OUT} matches the service")
        return
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(data)} points, model {MODEL_NAME})")


if __name__ == "__main__":
    main()
