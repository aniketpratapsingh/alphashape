#!/usr/bin/env python3
"""Build the alphashape sample dataset from our own synthetic scene generator.

The sample is a complete 12-image nadir aerial block of a small synthetic site: a
gently sloping terrain with a hill, a ramp and a flat-roofed building, imaged by a
4x3 camera grid at 60 m. Every image carries exact EXIF GPS, so the block
georeferences itself and the whole pipeline (align -> dense -> mesh -> texture ->
DEM -> orthomosaic) works without any real-world data.

Nothing here is derived from a third-party dataset. The scene, its textures and
the renderer are ours (engine/data/make_aerial_fixture.py), and the generated
data is released under CC0-1.0 -- see SOURCE.md next to this script.

Deterministic: the generator seeds every random draw, so two runs produce
byte-identical images. The output is:

    <out>/alphashape-sample/
        images/aerial_r0c0.jpg ... aerial_r2c3.jpg    the 12 input photographs
        manifest.json                                camera grid + truth poses
        README.md                                    how to run the sample
        SOURCE.md                                    licence and provenance

Usage:

    python3 tools/sample/make_sample.py --out build/sample
    zip -r build/alphashape-sample-0.1.0.zip -j ...   (the workflow does this)

Requires numpy and Pillow, the same two packages the fixture generators use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_DATA = os.path.join(_REPO, "engine", "data")
if _DATA not in sys.path:
    sys.path.insert(0, _DATA)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import make_aerial_fixture as aerial  # noqa: E402
import make_scene_fixtures as scene  # noqa: E402

SAMPLE_NAME = "alphashape-sample"


def build_images(out_dir: str) -> list[dict]:
    """Render the 12 photographs and return their manifest entries.

    Reuses the committed fixture generator verbatim: same scene, same textures,
    same camera grid, same EXIF writer. Only the destination differs, so the
    sample and the test fixtures cannot drift apart.
    """
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    aerial.build_textures()
    poses = aerial.camera_poses()

    origin_ecef = scene.geodetic_to_ecef(aerial.ORIGIN_LAT, aerial.ORIGIN_LON, aerial.ORIGIN_H)
    rotation = scene.enu_rotation(aerial.ORIGIN_LAT, aerial.ORIGIN_LON)

    entries: list[dict] = []
    for index, (centre, rot, trans) in enumerate(poses):
        row, col = divmod(index, aerial.COLS)
        name = "aerial_r%dc%d.jpg" % (row, col)
        rgb = aerial.render_image(centre, rot, trans)
        ecef = scene.enu_to_ecef(centre, origin_ecef, rotation)
        lat, lon, height = scene.ecef_to_geodetic(*ecef)
        exif = scene.geo_exif(lat, lon, height)
        path = os.path.join(images_dir, name)
        Image.fromarray(rgb, mode="RGB").save(path, quality=92, exif=exif.tobytes(),
                                              optimize=True)
        entries.append({
            "file": "images/" + name,
            "width": aerial.IMAGE_WIDTH,
            "height": aerial.IMAGE_HEIGHT,
            "focal_pixels": aerial.FOCAL,
            # X_cam = R * X_world + t, world = local ENU metres about `origin`.
            "R": rot.tolist(),
            "t": trans.tolist(),
            "centre": centre.tolist(),
            "gps": {"lat": lat, "lon": lon, "height": height},
        })
        print("wrote %s (%.1f%% non-black)" % (
            name, 100.0 * float(np.mean(np.any(rgb > 0, axis=2)))))
    return entries


def write_manifest(out_dir: str, entries: list[dict]) -> None:
    manifest = {
        "name": SAMPLE_NAME,
        "description": (
            "Synthetic 12-image nadir aerial block of a small site: sloping terrain with a "
            "hill, a ramp and a flat-roofed building."),
        "licence": "CC0-1.0",
        "generator": "engine/data/make_aerial_fixture.py",
        "seed": aerial.TILT_SEED,
        "origin": {"lon": aerial.ORIGIN_LON, "lat": aerial.ORIGIN_LAT,
                   "h": aerial.ORIGIN_H, "epsg": 4326},
        "camera": {"width": aerial.IMAGE_WIDTH, "height": aerial.IMAGE_HEIGHT,
                   "focal_pixels": aerial.FOCAL},
        "grid": {"cols": aerial.COLS, "rows": aerial.ROWS,
                 "spacing_x_m": aerial.SPACING_X, "spacing_y_m": aerial.SPACING_Y,
                 "altitude_m": aerial.ALTITUDE},
        "images": entries,
        "notes": (
            "World frame is local ENU metres about `origin` (east, north, up). The truth "
            "poses are X_cam = R * X_world + t. They are provided so the sample doubles as "
            "a ground-truth check; a normal run ignores them and aligns from the pixels."),
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def write_readme(out_dir: str) -> None:
    text = """# alphashape sample dataset

Twelve overlapping nadir photographs of a small synthetic site (sloping ground, a hill,
a ramp and a flat-roofed building), 480x360 px, with exact EXIF GPS. The images are
generated by alphashape's own scene renderer and released under CC0-1.0 (see SOURCE.md).

## Run it

```sh
# command line, one task at a time
alphashape-cli --new sample.psx --add-images images/*.jpg
alphashape-cli --project sample.psx --task MatchPhotos
alphashape-cli --project sample.psx --task AlignCameras
alphashape-cli --project sample.psx --task BuildDepthMaps
alphashape-cli --project sample.psx --task BuildPointCloud
alphashape-cli --project sample.psx --task BuildModel
alphashape-cli --project sample.psx --task BuildUV
alphashape-cli --project sample.psx --task BuildTexture
```

In the desktop app: **Add photos...**, then run the Workflow menu top to bottom. A
laptop processes this block in well under a minute.

## What to expect

The block is small (60 m across) and the camera grid is regular, so it is a smoke test
of the pipeline rather than a survey: the reconstruction is a coarse but complete mesh
with a baked texture. `manifest.json` carries the true camera poses and the site layout
if you want to measure the result.
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join("build", "sample"),
                        help="directory to write the sample into (default: build/sample)")
    args = parser.parse_args()

    out_dir = os.path.join(args.out, SAMPLE_NAME)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    entries = build_images(out_dir)
    write_manifest(out_dir, entries)
    write_readme(out_dir)
    shutil.copyfile(os.path.join(_HERE, "SOURCE.md"), os.path.join(out_dir, "SOURCE.md"))

    total = sum(os.path.getsize(os.path.join(out_dir, e["file"])) for e in entries)
    print("sample written to %s (%d images, %.1f KiB)" % (out_dir, len(entries),
                                                          total / 1024.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
