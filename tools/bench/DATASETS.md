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
| `gcp-rtk` (OpenDroneMap `odm_data_helenenschacht`, RTK sample) | 176 nadir JPEG | 5 surveyed ground-control points (35 image observations) + a 5-point survey CSV | GPL-3.0 | ground-control validation |

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

  `MatchPhotos keypoint_limit_per_mpx=250` (from 1000) is the second measured limit: at
  20 MP per frame the default density is 20 000 keypoints and 1.26 MB of 63-byte
  descriptors per image, and the resulting exhaustive matching ran the next CI attempt
  out of memory at 97% of the match stage. 250 per megapixel is 5 000 per image, which is
  still several thousand well-distributed keypoints for a 30-image block with heavy
  overlap.

  `BuildOrthomosaic resolution=0.10 refine_seamlines=false` is the fourth. The site is
  0.048 km², so a 2.4 cm mosaic is 83 million cells and even 5 cm is 19 million: each
  cell carries a label, and the seam search is an alpha-expansion over the 30 camera
  labels, so the refined search is cells × labels × expansion passes × pyramid levels ×
  iterations. At 2.4 cm that is 20 billion label operations, which overran the stage
  budget twice — once at 2.4 cm and once at 5 cm — while matching, alignment, depth maps,
  the dense cloud and the DEM had all already succeeded.

  At 10 cm the mosaic is 4.8 million cells and the argmax labelling keeps the cost linear
  in cells; the trade is visible seams where a camera's score steps, which is exactly what
  the side-by-side comparison image is for. 10 cm is also an ordinary orthomosaic
  deliverable for a site this size. The effective-resolution metric is measured on a
  full-resolution window of the mosaic and reported against the mosaic's own cell size,
  never against this nominal: the cell and the imagery are different grids, and mixing
  them made the number depend on a resolution that describes neither. The nominal is
  provenance only, and "did the mosaic get softer" stays a valid run-over-run signal
  because the ratio moves with the mosaic, not with the grid it was asked for.

  All of this is recorded in every metrics JSON and printed at the top of the report, so a
  bounded run is never mistaken for a default one. Each stage also carries a budget
  (40 minutes, 60 for the orthomosaic): a stage that overruns fails with its own name
  rather than waiting out the job's timeout and reporting nothing.
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
  - `terrains_dslr_scan_eval.7z` — the two laser scans already placed in the image frame
    (`scan1.ply`, `scan2.ply`) and the transform that put them there
    (`scan_alignment.mlp`). Without the alignment the two clouds live in different
    coordinate systems and any distance is meaningless. There is one download for this
    dataset: the archive holds all three files, so it is fetched and extracted once.
  - `terrains_dslr_occlusion.7z` — the occlusion masks and the scanner's own surface and
    splats. ETH3D's evaluator uses these to score accuracy only on points the laser could
    actually see.

    **This benchmark does not download or apply them.** They live in a separate 43 MB
    archive whose SHA-256 and byte count are not pinned in `datasets.json`, and the
    manifest's rule is that a file whose bytes are not verified is not downloaded.
    Applying them would also need a ray test from each scanner position against the
    reference surface. The consequence is stated rather than hidden: accuracy is scored
    over the whole cloud, so a correct reconstruction of a region the scanner was blind to
    counts as an error, and the reported accuracy is therefore a lower bound. Completeness
    is unaffected — every reference point genuinely should be reconstructed. `README.md`
    says the same beside the dataset table, so the claim and the code agree.

    The absence is checked rather than assumed: `terrains_dslr_scan_eval.7z` holds five
    entries — two scans, the alignment and two directory records — and no masks. That was
    verified against the real archive, which is why no code looks for masks in it.
  - `terrains_scan_clean.7z` — the cleaned scan in the scan's own frame. **Not used:**
    `scan_eval.7z` already carries the scans registered into the image frame, and the
    benchmark never needs the scanner's own frame.
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
- **The run is a 30-image block, with `MatchPhotos keypoint_limit_per_mpx=250` and
  `BuildDepthMaps downscale=8`.** 42 frames at 6048 x 4032 is 24 MP each: the default
  downscale of 4 is 64 million depth-map pixels over 42 views, and the default keypoint
  density is 24 000 keypoints per image. The same two measured limits as the aerial set
  apply, for the same reason. 30 of the 42 is still a 30-view dense reconstruction, and
  the accuracy tolerances (2/5/10 cm against the laser scan) are what decide whether the
  result is good enough — neither limit costs information those tolerances can measure.
  Each stage also carries a 40-minute budget and every limit is printed in the report.

## Ground control: `gcp-rtk`

**Helenenschacht RTK sample dataset, photos by Daniel Dobosi, published by
OpenDroneMap (GPL-3.0).**

- Publisher: [github.com/OpenDroneMap/odm_data_helenenschacht](https://github.com/OpenDroneMap/odm_data_helenenschacht),
  pinned commit `eece33e9ece2ff358a7bdc02e1db6f67925e00c4`. Downloaded at run time as a
  codeload tarball, verified by SHA-256 and byte count in `datasets.json`.
- Licence: **GPL-3.0**. The benchmark downloads and measures only, and does **not**
  redistribute: no image, GCP file, or converted copy of the GCP file is committed to
  this repository, and no derived imagery (orthomosaic, screenshot) is ever published.
  Only numbers are published.
- Images: 176 nadir JPEGs (5472 x 3648, 1" sensor, ~2.5 GB) from an Autel Evo II Pro
  RTK drone, with EXIF GPS from the RTK receiver.
- Files used (all inside the pinned tarball, never committed):
  - `images/` — the photographs (the only input the engine gets).
  - `GCP/gcp_list.txt` — the ODM GCP list: header `EPSG:4326`, then
    `x y z px py image [label]` per line, 35 image observations of 5 GCPs. Parsed at
    run time; nothing is converted or committed.
  - `GCP/latlon-easting_northing.csv` — 5 points
    (`Name;Easting;Northing;Elevation;Longitude;Latitude`), surveyed with an Emlid
    Reach RS2 via NTRIP at a stated 1–2 cm accuracy. Matched to the GCP list by
    coordinates to recover the survey names.
  - `RTK_Data/101FTASK_Timestamp.mrk` — the RTK module's per-exposure event-mark log,
    used only by the timing diagnostics. Each line carries the antenna position, its
    per-axis deviations and the solution flag, plus the GPS event time; the protocol
    pairs it with the JPEG comment's trigger/exposure counters. Parsed at run time.
- The gcp-list coordinate `z` is compared with the CSV `Elevation` to decide whether it
  is ellipsoidal or orthometric; the decision and the mean offset are recorded in the
  report rather than assumed.
- The run is a **leave-one-out over the 5 GCPs**: the block is aligned once
  (`MatchPhotos keypoint_limit_per_mpx=250` + `AlignCameras`, no dense stage), saved,
  then per fold 4 points are control and 1 is a held-out check. Two variants are
  measured — camera RTK references at their stated accuracy plus the GCPs, and GCPs
  alone with the camera references disabled. The primary camera variant imports each
  photo's XMP RTK deviations and fix status (`--camera-accuracy per-image`), so a
  camera is weighted by its own solution rather than one block-wide scalar; the
  uniform 5 cm weight remains as a comparison variant. The target is <= 5 cm
  horizontal RMS over the folds, reported as PASS/MISS and not gated in this first
  version.
- Why this one: it is the only ODM sample repository with surveyed ground-control
  points and RTK camera positions in one place, which is exactly what a ground-control
  validation needs. `odm_data_*` sample repos publish images only (see "Also surveyed").

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
| Zenodo 10059050 (EuroSDR RPAS benchmark, Wards Hill quarry) | CC BY 4.0 | 51 check points + 8 GCPs, ETRS89 ellipsoidal coordinates | **Observations are not published, so we measure them ourselves** in a CI-only workflow. See the section below |

## Second ground control: EuroSDR RPAS benchmark (Zenodo 10059050)

**Wards Hill quarry, Morpeth, Northumberland, UK — August 2021.** Studied 2026-11-18 as
the second independent ground-control block, so the 5 cm horizontal target would not be
tuned to one survey. The publisher releases the surveyed coordinates of 8 GCPs and 51
check points but **not** the pixel observations of their targets. Rather than reject the
block, the benchmark measures the observations itself: a CI-only workflow aligns the
RTK flight with no ground control, projects each surveyed point into every camera that
sees it, and snaps the projection onto the printed target with the engine's own
multi-view marker refinement. The accepted observations are committed as our own derived
data (see below), so the ordinary `gcp-eurosdr` run needs no measurement step.

- Publisher: Zenodo record 10059050, DOI [10.5281/zenodo.10059050](https://doi.org/10.5281/zenodo.10059050)
  (concept DOI 10.5281/zenodo.8309678). Newcastle University Geospatial Engineering for
  EuroSDR. Described in Mills et al., *The EuroSDR RPAS benchmark: open dataset
  description and summary of key results*, ISPRS Archives XLVIII-1/W3-2023, 107–114,
  DOI [10.5194/isprs-archives-XLVIII-1-W3-2023-107-2023](https://doi.org/10.5194/isprs-archives-XLVIII-1-W3-2023-107-2023).
- Licence: **CC BY 4.0** (`metadata.license.id == "cc-by-4.0"`). Redistribution is
  allowed with attribution, so the licence is not an obstacle. The derived observation
  file committed under `tools/bench/data/eurosdr/` carries that attribution.
- Files (four, all on the record):
  - `GCPsandCPs_coords.xlsx` (30 KB) — 59 stations, each with ETRS89 Cartesian
    (X/Y/Z), ETRS89 geodetic on GRS80 (lat/lon/**ellipsoidal** height) and OSGB36/ODN
    (E/N/orthometric height). 51 check points (50 named `A2`–`F8`, `S1`–`S3`, plus one
    unnamed row) and 8 GCPs (`GA1`, `GA5`, `GA8`, `GC6`, `GD4`, `GF1`, `GF4`, `GF8`),
    matching the paper's "51 CP and eight GCP targets". Surveyed with Leica GS18
    receivers in static/kinematic mode against four GNSS reference stations; the paper
    quotes average standard deviations of 5.3 mm E, 6.1 mm N, 3.4 mm H.
  - `Phase1.7z` (33.2 GB) — no ground control: DJI P4 RTK (489 images, 3 flights),
    DJI P1 (1024 nadir), DJI L1 (254), Riegl MiniVUX, Routescene LidarPod.
  - `Phase2.7z` (9.9 GB) — RTK / local base station: P4 RTK (500 images, 3 flights),
    DJI L1 (252), Riegl MiniVUX, Routescene LidarPod, plus the local base-station RINEX.
    The measurement workflow extracts only the P4 RTK flight `100_0004` (430 JPEGs and
    its `100_0004_Timestamp.MRK`) from this archive.
  - `Phase3.7z` (401 bytes) — a single text file, `EuroSDR-Phase3-GCPs.txt`: the 8 GCPs
    as `id E N H` in OSGB36/ODN. Coordinates only.
- **The point coordinates come from the xlsx, in ETRS89 ellipsoidal terms.** The xlsx
  carries ETRS89 geodetic latitude, longitude and **ellipsoidal** height on the GRS80
  ellipsoid for every station, so the protocol's `--gcp-height ellipsoidal` applies
  directly and no geoid model is needed. (The OSGB36/ODN columns in the same sheet are
  orthometric, but they are not the source: the xlsx's ETRS89 geodetic columns are.)
  The engine's geodesy is WGS84, which differs from ETRS89 by well under a metre at
  this site and is the same frame the RTK camera positions are recorded in, so the
  block is internally consistent.
- **The observations are measured, not published.** The record ships no GCP list,
  marker file or observation table — the complete file lists of both large archives
  were read from their 7z headers over HTTP range requests (no multi-GB download) and
  contain imagery, PPK/RINEX/event-mark logs, lidar and dataset-information PDFs only.
  The targets are 300 mm circular Perspex disks with a 100 mm black centre (white for
  check points, yellow for GCPs), mounted on stakes 0.15–0.45 m above ground (Mills
  et al. 2023, §2.1.2). The P4 RTK flew at 55 m AGL with an 8.8 mm lens and 2.41 µm
  pixels, so the GSD is ~1.5 cm: the black centre is only ~3.3 px radius and the disk
  ~10 px. The measurement workflow (`.github/workflows/eurosdr-observations.yml`,
  driving `tools/bench/eurosdr_observations.py`) does exactly that:
  1. import the 430 P4 RTK images with their DJI RTK XMP positions, accuracies and
     gimbal attitudes, then `MatchPhotos` (`keypoint_limit_per_mpx=250`, reference
     preselection);
  2. place every camera from its **own** RTK position and gimbal attitude with
     `AlignFromReference`, which builds a unit-scale local ENU frame. The incremental
     `AlignCameras` is deliberately not used: it georeferences the block with one global
     similarity fitted to the camera centres, and on this 430-frame flight that fit keeps
     only 6 cameras inside its 0.5 m inlier threshold, so the similarity is extrapolated
     from a small cluster and is metres off across the rest of the block. A bundle
     adjustment started from that georeference diverges (its priors are ~19 m from the
     poses it linearises around). Placing each camera from its own RTK solution avoids
     the global fit entirely;
  3. rebuild the tie points from the stored matches with `TriangulateTiePoints` (the
     placement cleared them, as they belonged to the old frame), then refine the poses
     with `OptimizeCameras`, which uses each camera's RTK position as a prior. No
     station exists yet, so the surveyed control cannot influence the alignment;
  4. read the xlsx with openpyxl and add all 59 stations as markers in EPSG:4979
     (ETRS89 geodetic, ellipsoidal height);
  5. project each point into every aligned camera that sees it, then run `DetectMarkers`
     (non-coded `Circular`, full resolution, 2 px floor, both polarities) to FIND the
     printed target near each projection, and `RefineMarkers` to snap the detected centre
     onto the printed centre with multi-view consensus;
  6. keep only detections within the search radius of the projection and, where the
     refinement triangulated the marker, a small reprojection residual; write the
     accepted observations as an ODM `gcp_list.txt` plus a JSON summary.
  The accepted `gcp_list` is committed under `tools/bench/data/eurosdr/` as our own
  derived data (CC BY 4.0, Mills et al. 2023, Newcastle University / EuroSDR), so the
  weekly `gcp-eurosdr` run reads it from the repository and never re-measures. The
  committed file is the GCP-only list (the eight GCPs): the protocol does leave-one-out
  over every point in the list, so a 59-station list would be 59 folds, while the eight
  GCPs give eight folds — the same shape as `gcp-rtk`'s five.
- A label trap worth recording: the `Phase3` text file's GCP labels are shifted by one
  against the xlsx. `Phase3` `GA1` is the xlsx `GA5`, `GA5` is `GA8`, and so on, with
  `Phase3` `GF8` matching no xlsx row within 9 m. The coordinates are identical to the
  millimetre under the shifted labels, so a loader that joined the two files by label
  would silently pair every point with the wrong name. The xlsx is the authoritative
  file (it is the one the paper describes), and the measurement script reads labels
  from the xlsx only.
- The smallest usable photogrammetric flight (P4 RTK `100_0004`, 430 images) is larger
  than the current ground-control block (176 images). The `gcp-rtk` block (176 images,
  5 folds) takes ~3 h of protocol time on a shared runner, so 430 images with 8 folds
  would run past the 330-minute job ceiling. The manifest therefore caps the weekly
  `gcp-eurosdr` run with `max_images`, selecting the nearest-to-centroid cluster the
  pipeline datasets use (a contiguous block with the overlap the alignment needs) and
  recording the choice and the selected file names in the report.

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
