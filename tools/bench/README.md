# Quality benchmark

A weekly, real-data quality check: download a public aerial or laser dataset, run the
CLI pipeline on it at default settings, and record what came out. Two runs of the same
commit on the same dataset should produce the same numbers; a run whose numbers moved the
wrong way against the last one is a regression.

The images and reference products are **not** in this repository and are not redistributed
from it. Every dataset is fetched from the publisher at run time and verified against
[`datasets.json`](datasets.json). [`DATASETS.md`](DATASETS.md) records what was surveyed,
what was chosen, each licence, and why.

## What it measures

Two families of metric. The reference-free ones always run; the reference-based ones run
only when the publisher ships a matching reference product.

**Reference-free** — is the product complete and sharp on its own terms?

| Metric | What it says |
|---|---|
| Ortho covered fraction | share of the mosaic's grid that came from an image |
| Holes inside the convex hull | gaps inside the surveyed area, with their count and area in m² |
| Effective resolution | how much resolution the mosaic actually carries, as a multiple of the mosaic's own cell size (a matched-blur MTF50-equivalent measured on a full-resolution window) |
| Gradient energy | raw sharpness, the quantity the effective resolution is derived from |
| DEM nodata fraction | share of DEM cells with no measurement |
| Cameras aligned | aligned / total cameras in the saved project |
| Reprojection RMS | the engine's own bundle-adjustment RMS, parsed from its `AlignCameras` output, plus an independent re-projection from the saved project (see below) |

**Reference-based** — does it agree with someone else's answer?

| Metric | Reference | What it says |
|---|---|---|
| Ortho offset | a reference orthomosaic | residual plan-view offset in metres, by phase correlation on luminance |
| DSM ΔZ | a reference DSM | the raw median offset (a vertical-datum difference, reported with both datums named) and, after removing it, the NMAD and the 68/95 percentiles of \|ΔZ\| — the spread the reconstruction itself contributes, which is what the gate compares |
| Cloud accuracy / completeness | a laser scan | ETH3D's definition at 2/5/10 cm, after voxel normalisation |

Reprojection RMS is an internal-consistency measure, not an accuracy one: a self-consistent
but wrongly scaled model scores well. The report never presents it alone for that reason,
and the reference-based rows are what actually say whether the model is right.

Every reference-based metric first reads both products' CRSs. A reference in a different CRS
is reprojected onto ours with rasterio and the report says so; a reference that states no CRS
against a product that does is a hard error, because there is no defensible way to compare
them by coordinate value. Neither comparison is ever made silently.

There is also an **independent** re-projection: `run_bench` re-projects every triangulated
tie point from the saved project's own poses and calibration, which is a cross-check on
both the project writer and the metric. It is reported beside the engine's RMS and gated,
and the report prints whether the two agree. They do agree: the writer emits the sensor
type as `frame`/`fisheye`, and the projection compares case-insensitively (an earlier
version compared against `Frame`, so every frame camera was projected with the fisheye
model and the figure read ~160 px). If the cross-check row ever says "disagrees", read
that as a bug in the project writer before trusting either number.

## Running it locally

Two prerequisites: the CLI, and Python 3 with the metrics dependencies.

```sh
# The CLI. Any build works; point ALPHASHAPE_CLI at it.
cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DALPHASHAPE_GPU=OFF
cmake --build build --target alphashape-cli
export ALPHASHAPE_CLI="$PWD/build/src/apps/cli/alphashape-cli"
```

```sh
# The metrics dependencies. rasterio reads GeoTIFF, laspy the reference LAZ clouds,
# Pillow writes the side-by-side PNG, numpy does the maths.
python3 -m pip install numpy rasterio Pillow 'laspy[lazrs]'
```

Then:

```sh
# What would be downloaded, from where, and under which licence. No network access.
python3 tools/bench/run_bench.py --dataset aerial-nadir --list

# A full run: fetch (cached), pipeline, metrics. Takes tens of minutes on real photographs.
python3 tools/bench/run_bench.py --dataset aerial-nadir \
    --work build/bench --json build/bench/metrics-aerial-nadir.json

# The report. Without --previous it is a first run and always passes.
python3 tools/bench/report.py \
    --current build/bench/metrics-aerial-nadir.json \
    --out build/bench/report.md --png build/bench/compare.png
```

`--work` is reused between runs: archives that already match their SHA-256 are not
downloaded again, and extracted trees are left in place. Point it at a directory with some
room — the aerial dataset is about 2.2 GB of downloads.

### The unit tests

```sh
python3 -m unittest discover -s tools/bench/tests -v
```

They need only numpy, run in well under a second, and are wired into the workflow's plan
job so a broken metric is caught before an engine is built or a dataset downloaded. Every
number they assert has an answer known by construction: a ramp whose interpolation is
exact, a mask with a counted number of holes, a shift applied by hand, a camera whose
projection is checked against a hand-computed pixel.

## The weekly workflow

[`.github/workflows/quality-bench.yml`](../../.github/workflows/quality-bench.yml) runs
every Monday at 04:17 UTC and on demand.

1. **Plan** — resolves the dataset list, computes the cache key from the manifest's
   content, and runs the unit tests.
2. **Build** — builds only the CLI (no GUI, no Python module, no GPU), uploads it.
3. **Bench**, one job per dataset:
   - restores the dataset cache keyed on the manifest hash,
   - downloads and verifies anything missing,
   - runs the pipeline, timing each stage and recording peak RSS,
   - compares against the last **successful** run's metrics for the same dataset,
   - uploads the report, the metrics JSON, the side-by-side PNG and the CLI log,
   - writes the report to the job summary,
   - publishes the new baseline — but only when the run passed.

Both datasets are bounded in two ways, because 20–24 MP frames at the default dense
settings do not fit a weekly job on a shared CPU runner — the first version of this
benchmark spent over five hours in BuildDepthMaps and was cancelled:

- **a 30-image cluster**, chosen as the 30 images nearest the flight's centre (the median GPS
  position, or file order where the frames carry no GPS) rather than a strided sample across
  the whole flight, which would only just overlap and produce sparse depth maps for the wrong
  reason. The cluster is deterministic and every selected file name is printed in the log and
  listed in the report;
- **`MatchPhotos keypoint_limit_per_mpx=250`**, a quarter of the default density: at
  20–24 MP per frame the default is 20 000+ keypoints per image, and the resulting matching
  ran the first working CI attempt out of memory at 97%;
- **`BuildDepthMaps downscale=8`**, which quarters the depth-map pixel count while leaving
  the grid finer than the imagery's own ground sample distance;
- **`BuildOrthomosaic resolution=0.10 refine_seamlines=false`** on the aerial set: a 2.4
  cm mosaic of a 0.048 km² site is 83 million cells, and the refined seam search is an
  alpha-expansion over 30 camera labels per cell — 20 billion label operations, which
  overran the stage budget at both 2.4 cm and 5 cm.

Both are recorded in every metrics JSON and printed at the top of the report, and every
stage carries a 40-minute budget so an overrunning stage fails with its own name rather
than waiting out the job. `DATASETS.md` gives the arithmetic.

A run fails only on a **regression against the previous run of the same dataset**. It
never fails on an absolute value: real-data results move with the scene, the season and
the publisher's own processing, so an absolute gate would go red for reasons no commit
caused. The tolerances are in [`thresholds.json`](thresholds.json): fraction-valued
metrics are gated by a relative move, lengths and heights by an absolute move in metres.
A metric with no tolerance is reported and never fails a run.

A failing run does **not** replace the baseline, so a regression keeps failing until it is
fixed rather than being silently adopted as the new normal.

The first run of a dataset has no baseline, so it always passes and simply establishes
one.

## The datasets

| Name | Flight | Images | Reference products | Licence |
|---|---|---|---|---|
| `aerial-nadir` | nadir drone | 102 | orthomosaic + DSM + dense cloud, from the publisher's own OpenDroneMap run | CC BY 4.0 |
| `terrestrial-laser` | oblique handheld DSLR | 42 | registered terrestrial laser scan (ETH3D `terrains`) | CC BY-NC-SA 4.0 |

`terrestrial-laser` scores accuracy and completeness against the laser scan. ETH3D also
publishes per-image occlusion masks, in a separate archive this benchmark does **not**
download; accuracy is therefore scored over the whole scan and is a lower bound, because a
correct reconstruction of a surface the scanner could not see counts as an error.
Completeness is unaffected. `DATASETS.md` records why the masks are not applied.

`terrestrial-laser` is ETH3D's `terrains` training scene. Its non-commercial licence
permits downloading it to evaluate; it does not permit redistribution, so the job downloads
it in CI and never commits or publishes it. If this project's use ever becomes commercial,
drop that dataset from the run — the workflow treats each dataset as its own job for
exactly that reason.

`DATASETS.md` has the full survey, including the datasets that were considered and
rejected.

## Adding a dataset

1. Find images plus a reference product from the same publisher, under a licence that
   permits downloading for evaluation.
2. Add an entry to `datasets.json`. Every file that is downloaded needs its publisher URL,
   its SHA-256 and its exact byte count; the byte count is checked first so a truncated
   download fails before it is hashed. Reference files that come out of an archive name the
   archive member in `archive_member`, and a file served by the same archive as another
   reference sets `from_archive` to that reference's `path` so the archive is fetched once.
3. Document the survey in `DATASETS.md` — the licence, the flight type, what the reference
   actually is, and why this dataset and not the alternatives.
4. Check the manifest without touching the network:
   `python3 tools/bench/run_bench.py --dataset <name> --list`.
5. Run it once locally to confirm the archive layout matches the manifest. The
   `expected_images` count is checked against what the archive actually extracts, so a
   wrong glob fails immediately rather than producing a score over a partial set.

A dataset with no reference product works: the reference-free metrics run on their own and
the report says in its provenance block that no reference metric ran, rather than omitting
the rows.

To add the dataset to the weekly run, add its name to the `dataset` input's `options` list
in the workflow. It is picked up automatically once the manifest has it, because the plan
job derives the matrix from the manifest.

## Layout

```
tools/bench/
  DATASETS.md          the survey: what was evaluated, chosen, licensed and rejected
  datasets.json        the manifest: URLs, SHA-256, byte counts, reference members
  run_bench.py         download, verify, extract, run the pipeline, call the metrics
  metrics.py           the metric maths (numpy only) plus lazily-imported readers
  report.py            Markdown, the side-by-side PNG, and the regression verdict
  thresholds.json      the regression policy
  tests/               unit tests for the metrics and the verdict
```
