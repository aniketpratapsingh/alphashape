# Sample data: source and licence

## What this is

`tools/sample/make_sample.py` renders twelve synthetic aerial photographs with
alphashape's own scene generator, `engine/data/make_aerial_fixture.py`. The generator
builds a small analytic site -- sloping terrain with a Gaussian hill, a raised ramp and
an axis-aligned flat-roofed building -- and ray-casts it from a 4x3 grid of nadir
cameras at 60 m. Every random draw is seeded (seed `20240615`), so the images are
byte-identical on every run.

The generator is developed in this repository and is not derived from, and contains no
data from, any third-party photogrammetry dataset or product. The procedural texture is
value noise plus seeded blobs, discs and line segments; the geometry is an analytic
height field and a box.

## Licence

**The generated sample data is released under CC0 1.0 Universal
(public domain dedication).** See <https://creativecommons.org/publicdomain/zero/1.0/>.
You may copy, modify, distribute and use the images and the manifest for any purpose,
without attribution.

The generator source code that produces the data stays under the project licence,
GPL-3.0-or-later (see `LICENSE`); the CC0 dedication applies to the output.

## What is in the dataset

| File | Contents |
|---|---|
| `images/aerial_r<row>c<col>.jpg` | the twelve input photographs, 480x360 px, JPEG, with EXIF including GPS latitude/longitude/altitude |
| `manifest.json` | camera grid, true camera poses (`X_cam = R X_world + t`), site origin, seed and licence |
| `README.md` | how to run the sample with the CLI and the desktop app |

The GPS EXIF is exact rather than measured, which is what lets the block georeference
itself; on a real survey it carries the usual GPS error.

## Regenerating

```sh
python3 tools/sample/make_sample.py --out build/sample
```

Requires `numpy` and `Pillow`. The CI workflow `.github/workflows/sample.yml` runs this
on `workflow_dispatch` and on `v*` tags and publishes `alphashape-sample-<version>.zip`
as an artifact.
