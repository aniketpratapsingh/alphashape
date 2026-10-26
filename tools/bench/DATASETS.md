# Quality benchmark datasets

This directory holds the weekly real-data quality benchmark. It downloads images and
reference products from their original publishers at run time, verifies them against
[`datasets.json`](datasets.json), and never commits imagery. This page records what was
surveyed, what was chosen, and why.

Every dataset below is downloaded from its publisher. Nothing is mirrored here, and
nothing is redistributed. The manifest pins a URL, an expected SHA-256, an image count
and a byte budget; a mismatch fails the run rather than producing a silently wrong score.

## The shortlist

| Dataset | Images | Reference products | Licence | Role |
|---|---|---|---|---|
| `aerial-nadir` (Zenodo 8436963, Bandamaji, Comoros) | 102 nadir JPEG | Orthomosaic + DSM GeoTIFF + dense cloud (LAZ), all from the publisher's own ODM run | CC BY 4.0 | primary aerial |
| `terrestrial-laser` (ETH3D `terrains`, high-res multi-view training) | 42 DSLR JPEG | Laser-scan point cloud, registered into the image frame (the `scan_eval` archive) | CC BY-NC-SA 4.0 | ground-truth geometry |

The Comoros survey is the primary aerial set. Its publisher ran OpenDroneMap on the
images it publishes and released the orthomosaic, the DSM and the dense cloud beside
them, which is exactly the pairing this benchmark needs: the same photographs and a
reference product from a different pipeline. `aerial-nadir` therefore exercises the
reference-based metrics (ortho offset, DSM ΔZ).

ETH3D is the ground-truth set. Its reference is not a photogrammetric product but a
registered terrestrial laser scan, so it can score absolute accuracy and completeness
of a dense cloud at fixed tolerances rather than only agreement with another SfM run.
Its images are oblique/handheld DSLR photographs of a `terrains` scene (rocky ground,
outdoors), not a drone flight — that is the point: it is the one reference here that
does not come from a photogrammetry pipeline at all.

## Primary aerial: `aerial-nadir`

**Bandamaji, Grande Comore, Comoros — 26 June 2023.**

- Publisher: Zenodo record 8436963, DOI [10.5281/zenodo.8436963](https://doi.org/10.5281/zenodo.8436963).
  Part of a Mayotte/Comoros drone survey series released by the G2OI project.
- Licence: CC BY 4.0 (`metadata.license.id == "cc-by-4.0"` on the record). Redistribution
  is allowed with attribution; this benchmark only downloads, so the requirement does not
  bind it, but the attribution is recorded here regardless.
- Images: 102 JPEGs in `DCIM.zip`, a single DJI flight, ~2.4 cm ground sample distance.
  The publisher states the flight was nadir-oriented ("survol", ODM defaults). Good
  overlap: the publisher's own ODM run reconstructed 102 of 102 images.
- Files used:
  - `DCIM.zip` — the raw photographs (the only input the benchmark gives the engine).
  - `PROCESSED_DATA.zip` — contains `odm_orthophoto/odm_orthophoto.tif` (the reference
    orthomosaic), `odm_dem/dsm.tif` (the reference DSM) and
    `odm_georeferencing/odm_georeferenced_model.laz` (a reference dense cloud). Only the
    two GeoTIFFs are used as references; the LAZ is not read, because the engine's
    point-cloud export is compared against the laser dataset instead.
  - `000_photogrammetry_report.pdf` — the publisher's quality report, used to record the
    nominal GSD and the reconstruction rate in the report's provenance block.
- Publisher settings, quoted from the record (they are the reproducibility contract):
  `orthophoto-resolution=1`, `dem-resolution=2.0`, `dsm=true`, `auto-boundary=true`.
  OpenDroneMap reads those two numbers as centimetres, and the files confirm it: both
  the ortho and the DSM come out at **2.19 cm/px** (11754 x 20163 cells). The reference
  is therefore at the same scale as the imagery, not coarser than it. The benchmark
  resamples both products onto a common grid before comparing, and reports that grid's
  pixel size, so the comparison scale is never implied.
- **The run is a 30-image block, and `BuildDepthMaps downscale=8`.**
  102 frames at 5472 x 3648 is 20 MP each, so the full set at the default downscale of 4
  is 127 million depth-map pixels, each costing a multi-view PatchMatch search. The first
  version of this benchmark ran it exactly that way and spent over five hours in
  BuildDepthMaps before being cancelled — which is a finding in itself, and the reason
  both limits exist.

  Downscale 8 quarters the pixel count, and the resulting 1.7 cm depth grid is still
  finer than the 2.4 cm GSD the imagery can justify. 30 images is the contiguous block
  nearest the flight's centre by GPS: a real spatial strip with the overlap the stereo
  stage needs. A strided sample of all 102 would only just overlap and would produce
  sparse depth maps for a reason that has nothing to do with the code. The 30 are chosen
  deterministically and named in the report.

  Both are recorded in every metrics JSON and printed at the top of the report, so a
  bounded run is never mistaken for a default one. Each stage also carries a 40-minute
  budget: a stage that overruns fails with its own name rather than waiting out the job's
  timeout and reporting nothing.
- Why this one over the other candidates: it is the smallest member of the Comoros/Mayotte
  series whose publisher report shows a full reconstruction (102/102 images) on a compact
  site (0.048 km²) with a 2.4 cm GSD. Neighbouring records in the same series are 187,
  194, 327 or 396 images and several reconstructed under 60% of their shots, which makes
  them a poor fit for a weekly run under a 6 h timeout. See "Also surveyed" below.

## Ground truth: `terrestrial-laser`

**ETH3D high-res multi-view, training scene `terrains`.**

- Publisher: [eth3d.net](https://www.eth3d.net/datasets), Computer Vision and Geometry
  Group, ETH Zurich (Schöps et al., CVPR 2017).
- Licence: **CC BY-NC-SA 4.0**, stated verbatim on the benchmark's front page: "The data
  provided here is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
  4.0 International License." The site publishes no separate terms page (`/terms`,
  `/license`, `/legal` all 404).
- Consequence for this repository: a non-commercial licence permits downloading the data
  for evaluation and reporting numbers about it. It does **not** permit redistributing the
  data, and it does not permit using the data commercially. This benchmark therefore only
  ever downloads it inside a CI job, never commits it, and never ships it in a release.
  The engine itself does not read the data at build time, and no code in this repository is
  derived from it — the dataset is an input to a test, not a component. If the project's
  use ever becomes commercial, this dataset has to be dropped from the run, which is why
  the workflow treats it as an optional lane rather than a hard requirement.
- Files used (all under `https://www.eth3d.net/data/`):
  - `terrains_dslr_jpg.7z` — 42 JPEGs. The pre-undistorted variant is deliberately **not**
    used: the engine estimates distortion itself, and feeding it undistorted images would
    hide distortion-model errors.
  - `terrains_scan_clean.7z` — the cleaned laser scan (PLY) in the scan's own frame.
  - `terrains_dslr_scan_eval.7z` — the scan/pose alignment (`scan_alignment.mlp`) that
    places the laser scan in the image frame, plus the per-image masks. Without this the
    two clouds live in different coordinate systems and any distance is meaningless.
  - `terrains_dslr_occlusion.7z` — the occlusion masks. ETH3D's evaluator uses these to
    score accuracy only on points the laser could actually see.

    **This benchmark does not use them yet.** Applying them needs a ray test from each
    scanner position against the reference surface, which the evaluator does across the
    full scan; the work is straightforward but is not in this version. The consequence is
    known and stated rather than hidden: accuracy is scored over the whole cloud, so a
    correct reconstruction of a region the scanner was blind to counts as an error, and
    the reported accuracy is therefore a lower bound. Completeness is unaffected — every
    reference point genuinely should be reconstructed. The manifest does not download
    the occlusion archive, so a run does not pay for a file it does not read.
- Accuracy/completeness follow ETH3D's own definition as implemented by
  `ETH3D/multi-view-evaluation`: accuracy is the fraction of reconstructed points within
  a tolerance of the reference surface, completeness the fraction of reference points
  within tolerance of the reconstruction, both scored after voxel-normalisation at the
  requested tolerances. This benchmark reports them at 2, 5 and 10 cm.
- ETH3D also ships COLMAP poses (`images.txt`, `cameras.txt`) inside the ground-truth
  archive. The benchmark does **not** feed those to the engine: the pipeline must align the
  images itself, and using the published poses would measure only the dense stage.
- **Registration before scoring.** A reconstruction lives in its own frame, and an SfM
  model has no metric scale until it is georeferenced — these images carry no GPS at all
  (only the `GPSVersionID` tag). Comparing distances across that difference would measure
  the frames, not the geometry. The benchmark therefore fits a similarity from the
  reconstruction to the scan first: principal axes plus a yaw search for a coarse
  alignment, then trimmed nearest-neighbour ICP. The fitted scale, the iteration count and
  the residual RMS are in the report, so a score that is really about the alignment is
  visible as one.
- **The run is a 30-image block, and `BuildDepthMaps downscale=8`.** 42 frames at
  6048 x 4032 is 24 MP each, so the default downscale of 4 is 64 million depth-map pixels
  over 42 views. 30 of the 42 is still a 30-view dense reconstruction of the scene, and
  the accuracy tolerances (2/5/10 cm against the laser scan) are what decide whether the
  result is good enough — the downscale costs no information those tolerances can use.
  Each stage also carries a 40-minute budget and both limits are printed in the report.

## Also surveyed, not chosen

Recorded so a future run does not redo the work. None of these beat the two above on the
combination of licence, size, nadir flight and publisher-supplied reference.

| Candidate | Licence | Reference | Why not |
|---|---|---|---|
| Zenodo 10085148 (Mayotte, Musicalplage 20220911) | CC BY 4.0 | ortho + DSM + LAZ | 400 images, 324 reconstructed; ~2.5 GB of images alone |
| Zenodo 10085131 (Mayotte, Jollyroger 20220913) | CC BY 4.0 | ortho + DSM + LAZ | 327 images; publisher reconstructed 251 |
| Zenodo 10085138 (Mayotte, Ngouja) | CC BY 4.0 | ortho + DSM + LAZ | 396 images; heavy for a weekly job |
| Zenodo 10085135 / 10085143 (Mayotte, Aeroport / Jollyroger) | CC BY 4.0 | ortho + DSM + LAZ | 187 / 421 images; the publisher's own runs reconstructed 79/187 and 251/421, so the site is hard and a low score would be uninformative |
| Zenodo 14733182 (Storflaket mire, Sweden) | CC BY 4.0 | ortho + DSM (GeoTIFF) | 417 images, 4.9 GB; mixed nadir/oblique/orbit flight, so not a clean nadir case |
| Zenodo 15282970 (TUM Downtown campus) | CC BY 4.0 | ortho + DSM + **UAV laser scan** | Excellent ground truth, but 20 GB total and `Images.zip` alone is 5.9 GB — over the weekly budget |
| Zenodo 2654620 (Averbode, Belgium) | CC0 1.0 | ortho + DSM | Cleanest licence of all, but 29 GB across six flights and no single small flight with a matching pair of products |
| Zenodo 4014948 / 5596168 (Bear Trap, Greenland) | CC BY 4.0 | ortho + DEM | 491 / 1032 images and 4.7 GB; the smaller companion records have no images |
| Zenodo 3731024 (Kobbefjord island, Greenland) | CC BY 4.0 | ortho + DEM | 218 images, 1.1 GB — a good fallback, but 218 images against 102 |
| OpenDroneMap `odm_data_*` (aukerman, seneca, bellus, ...) | CC0 / BSD-2 / none | **none** | The ODM sample repos publish images only. DroneDB Hub shows *user* runs, not publisher references, so they cannot support reference-based metrics |
| `pierotofy/drone_dataset_*` (sand_key, brighton_beach) | BSD-2-Clause | none | Same problem: images only |
| ETH3D `bridge` / `exhibition_hall` (test split) | CC BY-NC-SA 4.0 | laser scans | Ground truth for the test split is not published; only training scenes can be scored |

## Adding a dataset

1. Find images plus a reference product from the same publisher under a licence that
   permits downloading for evaluation.
2. Add an entry to [`datasets.json`](datasets.json): `url` for every file, `sha256` for
   every file, `images` (glob within the extracted tree), `reference` (the files that
   count as reference, with their kind), and `licence` with a `spdx` field and a `url`.
3. Record the survey in this file under "Also surveyed" or as a new section.
4. Run `python3 tools/bench/run_bench.py --dataset <name> --list` locally to confirm the
   archive layout matches the manifest before opening a PR.

A dataset without a reference product still works: the reference-free metrics (coverage,
holes, effective resolution, camera RMS) run on their own. Only the reference-based
metrics are skipped, and the report says so in its provenance block rather than silently
omitting rows.
