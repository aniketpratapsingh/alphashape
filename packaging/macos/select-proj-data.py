#!/usr/bin/env python3
"""Stage a size-bounded, self-contained PROJ data directory for the app bundle.

The Homebrew `proj` keg ships the *complete* proj-data collection: about 760 MB of
GeoTIFF/GTX grids covering every national datum on earth. Copying all of it made the
DMG 520 MB while the Linux AppImage, which bundles Ubuntu's curated `proj-data`
package, is 52 MB. A distribution package is a subset by design, so this script
applies the same idea to the bundle:

  * proj.db, proj.ini, the licence/attribution text and small auxiliary JSON always
    ship -- without proj.db nothing works at all;
  * a short essentials list ships unconditionally (EGM96 and the small continental
    datum-shift grid) even when a regional grid would not fit the budget;
  * the remaining grids are taken smallest first until the byte budget is spent,
    skipping any single file over the per-file cap, so the bundle covers as many
    countries as possible for the bytes available.

Anything left out is still reachable: `ALPHASHAPE_PROJ_DATA` can point at a full
proj-data checkout, and `crs::resolve_data_directory` prefers that over the bundle.

Usage: select-proj-data.py <source-data-dir> <destination-dir>
       [--grid-budget-mib N] [--grid-max-mib N]
"""

import argparse
import os
import shutil
import sys

# Grids that ship whatever their size. EGM96 (15', worldwide) is the geoid model the engine's
# vertical-CRS support names, and us_noaa_conus is the small continental datum-shift grid the
# staging script documents. Everything else competes for the remaining budget. The older GTX
# spelling of EGM96 is included too because Debian/Ubuntu packages still use it, and it is
# skipped automatically when the source has only the GeoTIFF.
ESSENTIAL_GRIDS = (
    "us_nga_egm96_15.tif",
    "egm96_15.gtx",
    "us_noaa_conus.tif",
)

# proj.db, the config, the attribution READMEs and the licence CSV always ship.
ALWAYS_EXTENSIONS = (".db", ".ini", ".txt", ".csv")
# Small PROJJSON TIN grids are cheap and tiny; the multi-megabyte ones are not.
JSON_MAX_BYTES = 256 * 1024
GRID_EXTENSIONS = (".tif", ".gtx")


def files_in(source):
    for name in sorted(os.listdir(source)):
        path = os.path.join(source, name)
        if os.path.isfile(path):
            yield name, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--grid-budget-mib", type=int, default=16,
                        help="total bytes of grid files to ship (default: 16)")
    parser.add_argument("--grid-max-mib", type=int, default=16,
                        help="largest single grid file to ship (default: 16)")
    args = parser.parse_args()

    source = args.source
    destination = args.destination
    if not os.path.isfile(os.path.join(source, "proj.db")):
        sys.exit(f"error: no proj.db in {source}")
    os.makedirs(destination, exist_ok=True)

    budget = args.grid_budget_mib * 1024 * 1024
    per_file = args.grid_max_mib * 1024 * 1024

    essential = []
    grids = []
    auxiliary = []
    for name, path in files_in(source):
        extension = os.path.splitext(name)[1].lower()
        size = os.path.getsize(path)
        if extension in GRID_EXTENSIONS:
            if name in ESSENTIAL_GRIDS:
                essential.append((size, name, path))
            else:
                grids.append((size, name, path))
        elif extension in ALWAYS_EXTENSIONS:
            auxiliary.append((size, name, path))
        elif extension == ".json" and size <= JSON_MAX_BYTES:
            auxiliary.append((size, name, path))

    chosen = list(essential)
    chosen_bytes = sum(size for size, _name, _path in essential)
    dropped_large = 0
    dropped_budget = 0
    for size, name, path in sorted(grids):
        if size > per_file:
            dropped_large += 1
            continue
        if chosen_bytes + size > budget:
            dropped_budget += 1
            continue
        chosen.append((size, name, path))
        chosen_bytes += size
    # Sorting the fill by name keeps the shipped set reproducible and the directory
    # listing readable; the essentials keep their original (priority) position.
    chosen = essential + sorted(chosen[len(essential):], key=lambda item: item[1])

    copied = 0
    for _size, name, path in sorted(auxiliary) + chosen:
        target = os.path.join(destination, name)
        if os.path.exists(target) and os.path.getsize(target) == os.path.getsize(path):
            continue
        shutil.copyfile(path, target)
        copied += 1

    total = sum(size for size, _n, _p in auxiliary) + chosen_bytes
    print(f"PROJ data: {len(auxiliary)} auxiliary + {len(chosen)} grid files, "
          f"{total / 1048576:.1f} MB (grid budget {args.grid_budget_mib} MB, "
          f"per-file cap {args.grid_max_mib} MB)")
    print(f"  copied {copied}; left out {dropped_budget} grids over the budget and "
          f"{dropped_large} grids over the per-file cap")
    if not os.path.isfile(os.path.join(destination, "proj.db")):
        sys.exit("error: proj.db was not copied")
    missing = [name for name in ESSENTIAL_GRIDS
               if os.path.isfile(os.path.join(source, name))
               and not os.path.isfile(os.path.join(destination, name))]
    if missing:
        sys.exit(f"error: essential grids missing from the bundle: {', '.join(missing)}")


if __name__ == "__main__":
    main()
