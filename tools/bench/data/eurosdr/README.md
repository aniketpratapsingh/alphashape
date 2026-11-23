# EuroSDR RPAS benchmark: measured GCP image observations

These files are **our own derived data**, not a redistribution of the publisher's
imagery. They contain only the pixel coordinates of the printed targets, never an
image.

## Source and attribution

The surveyed coordinates and the imagery come from the **EuroSDR RPAS benchmark**
(Zenodo record [10059050](https://doi.org/10.5281/zenodo.10059050), concept DOI
[10.5281/zenodo.8309678](https://doi.org/10.5281/zenodo.8309678)): the Wards Hill
quarry block, Morpeth, Northumberland, UK, flown in August 2021.

- Publisher: Newcastle University Geospatial Engineering for EuroSDR.
- Described in: Mills et al., *The EuroSDR RPAS benchmark: open dataset description
  and summary of key results*, ISPRS Archives XLVIII-1/W3-2023, 107-114,
  [doi:10.5194/isprs-archives-XLVIII-1-W3-2023-107-2023](https://doi.org/10.5194/isprs-archives-XLVIII-1-W3-2023-107-2023).
- Licence: **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>).

The publisher ships the surveyed coordinates of the 8 GCPs and 51 check points but
**not** the pixel observations of their targets. The files here are the observations
we measured ourselves, so the ground-control protocol has both halves of the data.

## Files

| File | Contents |
|---|---|
| `gcp_list_gcps.txt` | The 8 GCPs, 56 observations, ODM `gcp_list` format (header `EPSG:4979`). **This is the list the weekly `gcp-eurosdr` bench reads.** |
| `gcp_list.txt` | All 59 stations, 454 observations, same format. The complete derived data. |
| `gcp_names.csv` | The 59 surveyed stations as `Name;Easting;Northing;Elevation;Longitude;Latitude`, with the ETRS89 geodetic longitude/latitude and **ellipsoidal** height from `GCPsandCPs_coords.xlsx`. |

The GCP-only list is what the bench uses because the protocol does leave-one-out over
every point in the list: a 59-station list would be 59 folds, while the eight GCPs
give eight folds, the same shape as `gcp-rtk`'s five.

## How they were measured

`.github/workflows/eurosdr-observations.yml` drives
`tools/bench/eurosdr_observations.py`:

1. Import the 430 P4 RTK images with their DJI RTK XMP positions, accuracies and
   gimbal attitudes, then `MatchPhotos` (reference preselection).
2. Place every camera from its **own** RTK position and gimbal attitude with
   `AlignFromReference`, which builds a unit-scale local ENU frame. The incremental
   `AlignCameras` is deliberately not used: it georeferences the block with one
   global similarity fitted to the camera centres, and on this 430-frame flight that
   fit keeps only 6 cameras inside its 0.5 m inlier threshold, so the similarity is
   extrapolated from a small cluster and is metres off across the rest of the block.
3. Rebuild the tie points from the stored matches with `TriangulateTiePoints`, then
   refine the poses with `OptimizeCameras`, which uses each camera's RTK position as
   a prior. No station exists yet, so the surveyed control cannot influence the
   alignment used to project it.
4. Add all 59 stations as markers in EPSG:4979 (ETRS89 geodetic, ellipsoidal height).
   The one xlsx row with no station label is the missing `A1` grid cell: the sheet
   carries every cell of the 6x8 `A1`..`F8` array but `A1`, and the unlabelled row
   sits at that lattice cell, so the script recovers the label from the labels the
   sheet does carry rather than writing `UNNAMED`.
5. Project each point into every aligned camera that sees it, run `DetectMarkers`
   (non-coded `Circular`, full resolution, 2 px floor, both polarities) to find the
   printed target near each projection, and `RefineMarkers` to snap the detected
   centre onto the printed centre with multi-view consensus. The refinement runs on
   the `max_views` observers that best span the station (farthest-point sampling on
   the camera centres), so the triangulation is well conditioned; **every** detected
   view is then a candidate observation, not only the refined ones.
6. Triangulate each station by **consensus over all its detected views**, not from
   the engine's single triangulation of the refined subset: candidate positions from
   every pair with adequate parallax plus the all-ray least-squares fit are scored by
   how many detected views reproject within the residual limit, and the best is
   re-fit on its inliers. A single triangulation can collapse onto a degenerate pair
   (two consecutive frames) or a consistent wrong detection, and the residual gate
   then rejects the correct majority — the A3 regression, where 7 accepted views fell
   to 2. The consensus keeps the majority and records the inlier count per station.
7. Keep only detections within the search radius of the projection and, where the
   consensus triangulated the station, a small reprojection residual.

The measurement run that produced these files is GitHub Actions run
[44642611873](https://github.com/aniketpratapsingh/alphashape/actions/runs/44642611873)
(commit `a51110d`). Its summary: 430/430 cameras placed, bundle adjustment RMS
2.19 -> 0.62 px with a prior RMS of 0.14 sigma, 2228 targets detected over 58
stations, 454 observations accepted (9.9%), 8/8 GCPs with at least two views.

## Stations excluded from scoring

A station is scored only when its accepted observations give a well-conditioned
intersection. The rule is fixed before any result is seen, from photogrammetric
first principles:

- **fewer than 3 accepted images** — two views have no redundancy, so a blunder
  cannot be detected and the point is not an independent check;
- **a baseline/height ratio below 0.1** — the ratio is the largest camera-pair
  baseline over the mean range to the target. A ratio `r` is a parallax of
  `atan(r)`; the vertical precision scales as `sigma_xy / r` relative to the
  horizontal, so at `r = 0.1` (parallax ~5.7 deg) the height is already ~10x worse
  than the horizontal. Below that the point's error is set by its own geometry, not
  by the block.

A station that fails either test is excluded from scoring and listed with its
reason in the run output and the JSON summary (`excluded_stations`). The two
consecutive D3 frames are the canonical case: a few metres of baseline at ~55 m
range is a ratio of ~0.05, which is why D3's height was undetermined.

## Wrong-target safeguards

The within-station consensus (step 6) rejects a *small* wrong subset, but when a
wrong-target cluster is large enough to pull the consensus position between the
two clusters, both clusters reproject within the residual limit and nothing is
flagged. This is the **S2 regression**: 24 accepted views with a mean residual of
1.72 px against their own consensus, yet a 52 px maximum detection shift and a
170 px reprojection RMS in the benchmark adjustment, which alone pushed the
49-point check RMS from 0.055 m to 0.477 m. Two pre-declared safeguards catch it,
both derived from the block's **own** distribution in-run (robustly, median +
k·MAD over all stations), never from the offending station:

- **Detection-shift bound.** The detection shift is the distance from the
  hand-computed projection to the pixel `DetectMarkers` found. Across a correctly
  georeferenced block it is tight: on this flight the per-station maximum is
  median 6.9 px, MAD 0.8 px, every good station ≤ 9.6 px, while S2's wrong-target
  views reach 52 px. A station whose maximum detection shift exceeds
  `median + 5·MAD` (≈ 10.8 px here, floored at 10 px) is a wrong-target cluster
  and is excluded. 5 MAD is ~14× the robust spread: it admits every good station
  (~3.4 MAD) with margin and rejects S2 (~58 MAD) by a wide gap. Because the
  bound is computed from the stations themselves, a run where the whole block
  degrades raises the bound rather than mass-rejecting.
- **Consensus-vs-survey distance bound.** The survey is independent of the
  measurement, so a station whose consensus position lies far from its surveyed
  position is a wrong-target or blunder signal the within-station consensus
  cannot see. The block's own georeference error is the distribution of
  `|consensus − survey|` over the stations; a station beyond `median + 6·MAD`
  (floored at 0.25 m) is excluded. 6 MAD is deliberately loose — the
  georeference error is itself what is being measured, so the gate must not
  reject a station merely because the block is a few cm off; it targets the
  metre-scale error a wrong-target cluster produces (S2: 3.3 m horizontal,
  ~200× the block's ~1.5 cm median).

A **GCP** that fails either wrong-target gate fails the run loudly (a control
point that cannot be trusted invalidates the block); a **check point** that fails
is excluded from scoring with its reason, exactly like the intersection rule.
Both bounds are reported in the JSON summary (`shift_bound_px`,
`distance_bound_m`) and the run output.

## Regenerating

The files are committed so the weekly bench never re-measures. To regenerate them
(a new engine version, a changed refinement parameter), dispatch
`eurosdr-observations.yml` on the branch and commit the artifact's
`gcp_list_gcps.txt`, `gcp_list.txt` and `gcp_names.csv` here. The data files are
never hand-edited: the workflow regenerates them.
