#!/usr/bin/env python3
"""Regenerate tests/fixtures/crs/egm96_subset.tif.

Source: ``us_nga_egm96_15.tif`` (NGA / EGM96, public domain). It is published
by NGA and mirrored by PROJ at https://cdn.proj.org/us_nga_egm96_15.tif; the
copy installed with ``proj-data`` in a PROJ data directory is identical.

The subset keeps every 4th node of the native 0.25 degree grid, i.e. a 1 degree
grid over lon -180..180, lat 90..-90 (181 x 361 float32 samples). Catmull-Rom
is interpolating, so it reproduces those nodes exactly, which is what the
``Crs.GeoidBicubicVsProj`` test pins.

Usage:
    pip install numpy tifffile imagecodecs
    python3 make_egm96_subset.py [--full /path/to/us_nga_egm96_15.tif]

Without --full the script downloads the grid from cdn.proj.org (the only
network access this task's tooling performs; the engine itself never
downloads). The checked-in file was produced this way on 2026-10-29.
"""
import argparse
import hashlib
import pathlib
import sys
import urllib.request

import numpy as np
import tifffile

URL = "https://cdn.proj.org/us_nga_egm96_15.tif"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "egm96_subset.tif"

# PROJ describes a vertical-offset grid through GDAL_METADATA; without it the
# grid is read as a plain sampling grid (which vgridshift would then use as
# arbitrary offsets). Same items as us_nga_egm96_15.tif.
GDAL_METADATA = (
    '<GDALMetadata>\n'
    '  <Item name="area_of_use">World</Item>\n'
    '  <Item name="target_crs_epsg_code">5773</Item>\n'
    '  <Item name="TYPE">VERTICAL_OFFSET_GEOGRAPHIC_TO_VERTICAL</Item>\n'
    '  <Item name="UNITTYPE" sample="0" role="unittype">metre</Item>\n'
    '  <Item name="DESCRIPTION" sample="0" role="description">geoid_undulation</Item>\n'
    '</GDALMetadata>\n'
)


def load_full(path):
    a = tifffile.imread(path).astype(np.float32)
    if a.shape != (721, 1440):
        sys.exit(f"unexpected grid shape {a.shape}, expected (721, 1440)")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=pathlib.Path)
    args = ap.parse_args()
    if args.full:
        full = load_full(args.full)
    else:
        cache = HERE / ".egm96_15.tif"
        if not cache.exists():
            print(f"downloading {URL} -> {cache}")
            urllib.request.urlretrieve(URL, cache)
        full = load_full(cache)
    # Native nodes sit at lon -180 + col*0.25 and lat 90 - row*0.25 (the
    # ModelTiepoint corner convention PROJ itself uses), so taking every 4th
    # node in both directions gives a 1-degree grid. The last column is the
    # +180 node, which wraps to the -180 node of the same row: appending it
    # keeps the grid covering the full circle without a wrap rule in the
    # reader.
    coarse = full[::4, ::4]
    sub = np.ascontiguousarray(np.concatenate([coarse, coarse[:, :1]], axis=1))
    assert sub.shape == (181, 361), sub.shape
    # ModelTiepoint at the north-west corner, ModelPixelScale 1 degree.
    tags = [
        (33550, "d", 3, (1.0, 1.0, 0.0)),          # ModelPixelScale
        (33922, "d", 6, (0.0, 0.0, 0.0, -180.0, 90.0, 0.0)),  # ModelTiepoint
        # PixelIsPoint: a sample is the undulation AT its node, which is how
        # PROJ reads the original NGA grid (its samples are 0.25-degree nodes,
        # not cell averages). GeoKey 4096 names the interpolation CRS.
        (34735, "H", 20, (1, 1, 1, 4,
                          1024, 0, 1, 2,       # GTModelTypeGeoKey = geographic
                          1025, 0, 1, 2,       # GTRasterTypeGeoKey = pixel-is-point
                          2048, 0, 1, 4326,    # GeographicTypeGeoKey = WGS 84
                          4096, 0, 1, 4979)),  # GeogGeodeticDatumGeoKey/CRS = WGS84-3D
        (42112, "s", 1, GDAL_METADATA),
    ]
    tifffile.imwrite(OUT, sub, photometric="minisblack", compression="deflate",
                     predictor=True, metadata=None, extratags=tags,
                     description="1-degree subset of us_nga_egm96_15.tif "
                                 "(EGM96 geoid undulation, metres). Public "
                                 "domain, NGA.")
    data = OUT.read_bytes()
    print(f"wrote {OUT} ({len(data)} bytes)")
    print(f"sha256 {hashlib.sha256(data).hexdigest()}")
    # A few values the C++ tests can be eyeballed against.
    for lon, lat in [(0.0, 0.0), (9.0, 50.0), (-77.0, 39.0)]:
        col = int(lon + 180)
        row = int(90 - lat)
        print(f"N({lat:8.4f}, {lon:9.4f}) = {sub[row, col]:.4f} m")


if __name__ == "__main__":
    main()
