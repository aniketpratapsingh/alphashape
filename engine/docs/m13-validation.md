# M13 validation — markers, GCP and calibration

The M13 close-out is validated by five reproducible synthetic scenarios and a
real-block procedure. The script is `engine/tests/run_gcp_validation.py`; it
prints one JSON object per scenario on stdout and exits non-zero when any gate
fails.

```
python3 engine/tests/run_gcp_validation.py --synthetic --ablations
```

## Synthetic GCP block

The scenario loads the 12-image nadir fixture (`engine/data/aerial_r*c*.jpg`)
whose truth poses, ground grid and building are recorded in
`engine/data/aerial_truth.json`. It then:

1. imports the images and declares the chunk WGS 84 geographic. The fixture
   EXIF carries exact GPS, but that is not the camera reference this scenario
   trusts: every camera gets a seeded consumer-GNSS reference of its own, a
   per-camera Gaussian offset (horizontal sigma 2 m, vertical sigma 4 m) on top
   of a common block bias of (1.5, -1.0, 3.0) m in ENU, with the matching
   per-axis accuracy (2, 2, 4) m imported through `importReference`. The
   chunk's fallback `camera_location_accuracy` is 3 m;
2. runs `MatchPhotos` and `AlignCameras`, so the block's initial georeference
   comes from those noisy camera references;
3. creates 12 markers: 6 on the ground (z = 0) and 6 on the building roof
   (z = 9.52 m), with NO seeded 3-D position;
4. adds each marker's projections by projecting its truth position through the
   fixture's truth camera pose, then offsets every projection by seeded uniform
   noise of 0.5 px peak to peak (±0.25 px per axis). The chunk's
   marker-projection accuracy is recorded from that noise (sigma = a/sqrt(3)).
   The bundle adjustment whitens every observation by its own declared sigma:
   marker projections by that value, tie-point observations by the chunk's
   tie-point accuracy;
5. sets the marker reference accuracy to the 1 cm the survey noise actually
   has, so the control points are not treated as exact;
6. imports 8 usable references as control points (truth + deterministic 1 cm
   per-axis noise) and 3 as check points (held out of the fit; the three
   best-observed roof targets, since the fixture's edge targets have only 1-3
   views);
7. runs `optimizeCameras` with `use_gcps=True` and the pinhole intrinsics held
   fixed (the fixture is a distortion-free render), so every marker position is
   triangulated from its image measurements only and the control references
   constrain the solve, then `updateTransform` (the weighted similarity fit)
   refits the datum from the control markers;
8. measures every check point against its TRUE position (not its reference)
   and reports the horizontal, vertical and total RMS.

The camera references are deliberately much worse than the survey points
(metre-level versus centimetre-level), so the GCPs carry the georeference; a
block georeferenced by exact camera centres would not exercise GCP
georeferencing at all.

### Ablation scenarios

`--ablations` runs the same block five times, so a pipeline defect can be told
from noise-limited accuracy. Scenario 1 is the pipeline-exactness control: the
camera-GNSS references are off and the fixture's exact GPS position and
attitude are pinned, so every input the marker/GCP machinery consumes is exact.
Scenarios 2-5 keep the realistic consumer-GNSS camera references (attitude from
the alignment) and vary only the two noise knobs; their differences are
therefore attributable to those noises. Scenario 4 keeps the noisy GNSS and
removes both other noises, so it isolates the camera-prior/control interaction.

| # | name | projection noise | reference noise | camera references | gate |
| --- | --- | --- | --- | --- | --- |
| 1 | `exact` | 0 px | 0 m | exact position + attitude | 3D total <= 2 mm |
| 2 | `reference-noise` | 0 px | 1 cm per axis | consumer GNSS (2 m / 4 m sigma) | Gate A: 3D total <= 2 cm |
| 3 | `projection-noise` | ±0.25 px per axis | 0 m | consumer GNSS (2 m / 4 m sigma) | Gate B: measured <= 1.5x predicted |
| 4 | `gnss-only` | 0 px | 0 m | consumer GNSS (2 m / 4 m sigma) | 3D total <= 2 mm |
| 5 | `combined` | ±0.25 px per axis | 1 cm per axis | consumer GNSS (2 m / 4 m sigma) | Gate B: measured <= 1.5x predicted |

The three gates exist for different reasons.

* **Gate C — pipeline exactness (scenarios 1 and 4).** With exact projections,
  exact references and exact camera GPS, a correct pipeline reconstructs the
  truth to well under a millimetre. Scenario 4 pins the camera-prior/control
  interaction: with exact control and exact projections a correctly weighted
  adjustment is governed by the control. A failure here is a defect to fix, not
  a bound to loosen.
* **Gate A — the milestone scenario (scenario 2).** The milestone asks for 12
  GCPs with 1 cm noise to hold the check-point RMS at or below 2 cm. Scenario 2
  is that scenario with exact marker projections, so the 2 cm value is the
  spec's own and is not changed.
* **Gate B — estimator efficiency (scenarios 3 and 5).** Once image noise is
  injected the absolute 2 cm number says nothing about whether the estimator is
  efficient: a bias-free pipeline can still exceed it on one seeded noise
  realisation. The gate therefore compares the measurement with the
  information limit computed from the same geometry, and requires the measured
  total to stay within 1.5x the prediction. The factor is fixed in the script
  and is not tuned per run.

#### First-order prediction (Gate B)

The prediction is computed in the script, not quoted from an external bound.
At the fixture's TRUE parameters it assembles the Gauss-Newton normal matrix
`J^T W J` of the bundle adjustment:

* six degrees of freedom per camera (rotation vector and centre; intrinsics
  fixed), three per marker, three per tie point;
* one 2-D projection observation per visible marker/tie point, weighted by the
  sigma actually declared to the engine (0.25/sqrt(3) px for the marker
  projections, the chunk's tie-point accuracy for the ties);
* one 3-D reference observation per control marker, weighted by
  1 cm/sqrt(3) (a 1e-4 m floor when the scenario has no reference noise);
* one 3-D camera-centre prior per camera, weighted by the consumer-GNSS sigma
  (2, 2, 4) m.

Tie points are included, not omitted: a deterministic sample of 300 points on
the truth surface (ground plane and roof), each visible in at least three
images. They have no 3-D prior, so their block-diagonal 3x3 blocks are
eliminated by a Schur complement and the reduced camera/marker system (108
parameters) is inverted. The diagonal 3x3 block of each check point gives its
marginal covariance in ENU. The predicted horizontal, vertical and total RMS
are the roots of the mean per-axis variance over the three check points, and
the Jacobians are central differences of the pinhole projection. Including the
tie points matters: omitting them would remove real information from the
predicted poses and make the gate looser, not tighter.

### Diagnostics

Every report also carries the engine's solve diagnostics under `diagnostics`:
the per-category sigmas actually used (camera centre, marker control, marker
projection, tie point), the iteration count and termination category, the
initial and final robust cost, the last relative cost change, the prior RMS in
sigma units, the gauge state, and the control/check residuals split per axis.
`update_transform` additionally reports how many cameras and control markers
the similarity fit used and the total 1/sigma^2 weight of each class.

### Measured in CI

Measured by `python_gcp_validation` on CI run 42361458754. The Linux x86-64 and
macOS arm64 lanes produced identical numbers. The test reported
`Passed 77.36 sec` on the Linux lane, which covers the five alignments and the
prediction. Numbers are metres.

| scenario | horizontal | vertical | total | predicted h / v / total | gate | result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 `exact` | 0.000135 | 0.000324 | 0.000351 | — | 0.002 | pass |
| 2 `reference-noise` | 0.001972 | 0.007280 | 0.007542 | — | 0.020 (Gate A) | pass |
| 3 `projection-noise` | 0.008216 | 0.018440 | 0.020187 | 0.013599 / 0.028190 / 0.031299 | 0.046949 = 1.5x (Gate B) | pass, ratio 0.645 |
| 4 `gnss-only` | 0.000561 | 0.000704 | 0.000900 | — | 0.002 | pass |
| 5 `combined` | 0.008028 | 0.025646 | 0.026873 | 0.014407 / 0.028721 / 0.032132 | 0.048197 = 1.5x (Gate B) | pass, ratio 0.836 |

Scenario 5's 26.9 mm is above 2 cm, which is why the two noisy-image scenarios
are governed by Gate B rather than by the spec's absolute number: the measured
error stays well inside 1.5x the information limit computed from the same
geometry. The ratios (0.645 and 0.836) are the measured total divided by the
predicted total.

The test prints the same five bracket-tagged measurement lines and the full JSON
objects; CI surfaces them even when the test passes.

### Defect found and fixed

Before the fix the five scenarios measured 0.49 mm, 5.10 cm, 5.10 cm, 5.33 cm
and 4.87 cm (CI run 42357021231). Scenarios 2 and 3 removed different noise
sources yet both sat at 5.1 cm, so the error tracked the noisy camera GNSS, not
the injected marker noise. The diagnostics pinned it: scenario 4 converged with
the control points satisfied to 0.08 mm, yet its check points re-triangulated
5.2 cm away, almost all vertical, while the reprojection RMS stayed at
0.127 px. The control points were pinned in 3-D, but the camera poses were
being set by the ~7990 tie-point observations, and every observation carried
one shared 1 px weight because `BAObservation` had no per-observation sigma.
With 30 marker observations against 7990 tie observations at equal weight, the
poses followed the tie-point cloud and the held-out points drifted by
centimetres in depth.

The fix follows the standard weighted least-squares formulation (Triggs et al.
2000; Foerstner & Wrobel 2016): each observation is whitened by its own 1-sigma
accuracy, so the problem is one weighted system with weights 1/sigma^2. Marker
projections use the chunk's marker-projection accuracy, tie-point observations
its tie-point accuracy, and the robust loss acts on the whitened residual.
Check-point triangulation uses the same weights. The camera-centre, marker
control and datum weights were already correct (2 m / 4 m per axis, 1 cm, and
the datum fit's 1/sigma^2 weights). Two C++ tests cover the fix:
`OptimizeMarkers.ExactControlPinsModelAgainstNoisyCameraPriors` and
`OptimizeMarkers.CheckPointFollowsProperlyWeightedPoses`;
`OptimizeMarkers.UseGcpFalseDropsMarkerReferences` covers `use_gcps`.

### Fixed in every scenario

12 images, 12 markers (6 ground, 6 roof), 8 usable control points (9 imported;
one edge target has a single projection), 3 check points (`E2`, `E3`, `E5`, all
on the roof), pinhole intrinsics, and the fixture's near-planar geometry. GSD
is 0.12 m/px (60 m above ground at f = 500 px).

## Real block

```
python3 engine/tests/run_gcp_validation.py \
    --real /path/to/aerial-block \
    --gcp /path/to/gcps.csv \
    --crs 4326 \
    --check-points GCP17,GCP23,GCP31
```

The procedure is: import every image, declare the geographic reference CRS, run
`MatchPhotos` and `AlignCameras`, detect markers, import the CSV as control
points and the designated labels as check points, run `updateTransform` (which
seeds the datum from the control markers when the block has no usable camera
GPS), run `optimizeCameras`, and report the check-point RMS against each
marker's stored reference with the same per-axis reconstruction. The gate is
the 3D total RMS against `--tolerance`; the horizontal RMS is reported
separately because the real-block target is horizontal.

`--crs` must name a geographic (lon/lat) CRS. The script converts references to
ECEF as lon/lat, so a projected CRS is refused with an error rather than
silently misinterpreting the coordinates. A check label with no marker or no
computed position also exits non-zero, so a missing check point cannot be
mistaken for a perfect one.

### Measured: Helenenschacht RTK block (GPL-3.0)

Measured by the `gcp-rtk` dataset of the quality benchmark on CI run
`42378055451`. Dataset: the OpenDroneMap `odm_data_helenenschacht` sample (pinned
commit `eece33e9ece2ff358a7bdc02e1db6f67925e00c4`, tarball SHA-256
`4e71dbfc08efcf1ff854c4b82a01f2ba5b6b42dd7530fd64c890ab2261643bab`, 2 546 560 515
bytes) — 176 nadir JPEGs from an Autel Evo II Pro RTK drone, 5 surveyed GCPs (35
image observations) and a 5-point survey CSV (Emlid Reach RS2 via NTRIP). Photos
by Daniel Dobosi, published by OpenDroneMap (GPL-3.0); the benchmark downloads and
measures only and does not redistribute.

Protocol: align once (`MatchPhotos keypoint_limit_per_mpx=250` + `AlignCameras`,
146/176 cameras registered, 0.740 px RMS), save, then leave-one-out over the 5
GCPs — 4 control + 1 held-out check per fold, `UpdateTransform` +
`OptimizeCameras(use_gcps=True)`, and the check point's horizontal / vertical / 3D
error measured against its surveyed reference. Two variants: camera RTK references
at `--camera-accuracy` (0.05 m) plus the GCPs at `--marker-accuracy` (0.015 m), and
GCPs only with the camera references disabled. Both variants report how far the
stored camera references sit from the adjusted camera centres (mean and RMS
east/north/up, plus the horizontal difference split into along-track and
cross-track from the flight direction), which separates a timing offset from
unfixed positions. Runtime 2702.7 s on a shared CPU runner. The GCP heights are
**ellipsoidal**: the publisher labels the CSV column "Ellipsoidal height", and
the gcp_list z carries the same values (the mean gcp_list − CSV offset is a
consistency check on the label match, not a datum test). The script records this
as `z_consistency` and refuses an orthometric height because it carries no geoid
model.

| check | camera_rtk_gcps h / v / 3D (m) | gcps_only h / v / 3D (m) |
| --- | --- | --- |
| 1 | 0.0753 / 0.1368 / 0.1562 | 0.0741 / 0.1359 / 0.1548 |
| 2 | 0.0385 / -0.0581 / 0.0696 | 0.0409 / -0.0399 / 0.0572 |
| 3 | 0.0926 / -0.0905 / 0.1295 | 0.0921 / -0.0851 / 0.1254 |
| 4 | 0.1043 / 0.2697 / 0.2891 | 0.1058 / 0.2823 / 0.3015 |
| 5 | 0.0638 / 0.1841 / 0.1948 | 0.0646 / 0.1845 / 0.1954 |
| RMS | **0.0783 / 0.1655 / 0.1831** | **0.0788 / 0.1680 / 0.1855** |

**Superseded.** In this first run the camera variant read each camera's stored
accuracy back from the binding, which is the 10 m consumer default once a location
exists, so the RTK positions carried no weight and the two variants agreed to
within a millimetre. The control markers also used the 5 mm chunk default instead
of the stated 0.015 m. Both are fixed; the table above is kept only as the record
of that run.

Rerun with the stated accuracies (CI run `42383670611`, same alignment: 146/176,
0.740 px):

| check | camera_rtk_gcps h / v / 3D (m) | gcps_only h / v / 3D (m) |
| --- | --- | --- |
| 1 | 0.1028 / -0.1322 / 0.1674 | 0.0711 / 0.0243 / 0.0751 |
| 2 | 0.0380 / -0.3137 / 0.3160 | 0.0348 / 0.0103 / 0.0363 |
| 3 | 0.1133 / -0.1296 / 0.1722 | 0.0885 / -0.1129 / 0.1434 |
| 4 | 0.0399 / 0.0919 / 0.1002 | 0.1127 / 0.1757 / 0.2087 |
| 5 | 0.0446 / 0.0142 / 0.0469 | 0.0670 / 0.0820 / 0.1059 |
| RMS | **0.0754 / 0.1681 / 0.1843** | **0.0791 / 0.1010 / 0.1283** |

The 5 cm horizontal target is still **missed** (reported, not gated). The GCPs-only
folds explain why the camera variant does not help: against the GCP-adjusted
camera centres, the stored camera positions sit on average 3.45 m above the
adjusted ones (mean E/N/U -0.61 / 0.20 / 3.45 m, RMS 0.72 / 0.50 / 3.49 m over 146
cameras). Both heights are ellipsoidal and the camera import is correct, so this is
not a datum mismatch.

#### Diagnostics (CI runs `42645790811`, `42661142021`)

Leave-one-out check RMS, horizontal / vertical / 3D in cm; "camera vs GPS up" is
the mean height of the stored camera positions minus the adjusted centres;
"focal" is the mean relative change of the solved focal length.

| variant | h / v / 3D (cm) | camera vs GPS up | focal |
| --- | --- | --- | --- |
| gcps_only (default fit) | 7.9 / 10.1 / 12.8 | +3.45 m | +1.2 % |
| gcps_only, `adaptive_fitting` | 6.5 / 6.4 / 9.1 | -0.23 m | -4.6 % |
| gcps_only, extended model | 6.9 / 6.6 / 9.5 | -0.27 m | -4.9 % |
| gcps_only, `adaptive_fitting`, -0.5 px | 6.2 / **5.9** / **8.6** | -0.24 m | -4.5 % |
| gcps_only, -0.5 px / +0.5 px | 7.3 / 8.9 / 11.5 — 8.6 / 11.7 / 14.5 | | |
| gcps_only, marker sigma 0.25 / 1.0 px | 8.7 / 15.2 / 17.5 — 7.8 / 13.8 / 15.9 | | |
| gcps_only, `refineMarkers` (fixed, 2026-11-15) | 7.9 / 10.2 / 12.9 | | |
| gcps_only, `refineMarkers` + `adaptive_fitting` | 6.5 / 6.3 / 9.1 | | |
| camera_rtk_gcps (default fit) | 7.5 / 16.8 / 18.4 | 0.00 m | -4.3 % |
| camera_rtk_gcps, camera sigma 0.5 m | 7.1 / 17.4 / 18.8 | | |
| camera_rtk_gcps, `adaptive_fitting` | 5.4 / 10.3 / 11.6 | 0.00 m | -4.2 % |
| camera_rtk_gcps, `adaptive_fitting`, -0.5 px | **5.1** / 9.9 / 11.1 | 0.00 m | -4.2 % |

Findings:

- **The default lens model is too thin for this camera.** With the extra terms the
  3.45 m camera offset disappears and the solved focal length agrees with the one
  the RTK cameras imply (about -4.5 %): focal length was trading against the
  distortion terms the default fit holds, not against flying height alone. The
  RTK camera positions agree with the adjusted centres to ~4.5 cm RMS once they
  are weighted, so they are good data.
- **Pixel convention.** -0.5 px helps in every combination and +0.5 px hurts; the
  evidence below shows the dataset file and the engine already share the same
  corner-origin convention, so this is not a convention mismatch. Both probes stay
  report-only (`--gcp-pixel-origin` selects the conversion for a file written to
  the documented OpenSfM format).
- **Vertical with camera RTK.** The per-fold vertical errors are systematic (-7,
  -9, -1, +11, +15 cm) and the GCP-only adjustment puts the camera centres ~0.23 m
  below the recorded positions — consistent with an unmodelled GNSS antenna offset
  above the lens. Tested below with fixed and per-fold-estimated lever arms; the
  BA prior pins the mount, but the datum fit does not consume it, and the check
  does not improve.
- **`refineMarkers` no longer touches these clicks.** The first run removed about
  half of the projections and made the result much worse (12.9 / 24.5 / 27.7 cm).
  The targets turn out to be printed 2x2 checkers whose quadrant crossing is on
  the surveyor's click, so the refinement now declines to move them; see the
  marker-refinement subsection below.
- Ruled out: trigger/GNSS timing (no along-track pattern once the cameras are
  weighted), marker-vs-tie weight (0.5 px is best), unaligned cameras (1 of 35
  observations).

#### Marker refinement on the real block (2026-11-15)

The five GCPs are printed 2x2 checkerboards: two opposite quadrants white, the
other two light grey, about 31 px across in the 20 MP frames, lying on grass and
gravel. The benchmark's before/after crops (`gcp_crops/` in the report
artifact) show the quadrant crossing within a pixel of the drawn gcp_list click
on all six sampled projections. **The check residual is therefore not click
error**, and moving an accurate click can only add noise.

The original probe made the check worse (12.9 / 24.5 / 27.7 cm) and removed
about half of the projections. Two mechanisms did it. The imported markers carry
no `target_type`, so the task ran the coded-circular light-interior seed; on a
checker the brightest blob is one quadrant, half a target from the printed
centre, and the ellipse refiner fitted that quadrant. And when no 2-D refiner
converged, `apply_winner` adopted the consensus candidate's reprojection as the
"refined" pixel, so a depth step moved accurate clicks by several pixels.

The fix (`engine/src/common/algorithm/markers/refine_markers_task.cpp`):

- A bright blob seeds a circle only for a declared circular target or a printed
  centre dot; the seed search at `fill_scales` (`refine_markers_task.cpp:1936`)
  no longer reads one checker quadrant as a target interior.
- A saddle correction is validated as a 2x2 checker with the same validator the
  scale search uses (`checker_at`, `refine_markers_task.cpp:1170`), and the
  pattern is identified once per marker so a foreshortened view can still use
  the saddle.
- The refiner starts on the validated detection found on the click
  (`refined_2d_pixel`, `refine_markers_task.cpp:1230`), because the consensus
  candidate can be several pixels away after its depth step.
- A projection with no converged refinement is kept exactly where it was
  measured — the candidate is never adopted (`refine_markers_task.cpp:1758`).
- A correction is applied only when at least two views corroborate it
  (`refine_markers_task.cpp:1783`); the only correction the intermediate code
  applied on this block was one view per fold moving one projection by 5.65 px,
  which made the check worse.
- No projection is ever deleted; every move is bounded by `maximum_shift`
  (6 px, or half the apparent radius when larger) against the measured pixel.

Final quality-bench run `43746926638` (`dataset=gcp-rtk`, diagnostics on,
head `4c6bed5`). Per fold: 35 projections kept, 0 removed, 34 evaluated, 34 kept
unrefined; of the 5 markers, 4 were confidently processed and left at their
clicks and 1 had no specific scale. Nothing moved, so the refined variants
reproduce the unrefined rows to within the run's noise:

| variant | h / v / 3D (cm) |
| --- | --- |
| gcps_only | 7.9 / 10.1 / 12.9 |
| `refineMarkers` | 7.9 / 10.2 / 12.9 |
| gcps_only, `adaptive_fitting` | 6.5 / 6.5 / 9.2 |
| `refineMarkers` + `adaptive_fitting` | 6.5 / 6.3 / 9.1 |
| camera_rtk_gcps | 7.9 / 16.3 / 18.1 |
| camera_rtk_gcps + `refineMarkers` | 8.0 / 16.3 / 18.1 |
| camera_rtk_gcps, `adaptive_fitting` | 5.3 / 12.2 / 13.3 |
| camera_rtk_gcps, `adaptive_fitting` + `refineMarkers` | 5.4 / 12.2 / 13.4 |

**On this dataset refinement cannot beat the unrefined clicks**, because the
clicks are already on the printed centre; the honest outcome is that it declines
to move them rather than adding noise. On rendered checkers (dark and bright,
normal incidence and 50-60 degree obliquity, and on fine ground grain) the same
code refines to under 0.1 px. The 5 cm horizontal target is still missed (best
5.3 cm with camera RTK and the adaptive model), so it remains reported and not
gated.

#### Pixel convention: evidence and decision (2026-11-11)

*Engine.* Pixel coordinates are corner-origin. `try_project_point` in
`engine/src/model/camera/calibration.hpp` returns
`width * 0.5 + cx + xd (f + b1) + yd b2`, so a point on the optical axis lands at
`(width/2, height/2)`; the same header says "top-left pixel centre at
(0.5, 0.5)". `keypoint_to_pixel` (`detect_features_task.hpp`) adds 0.5 to the
detector's corner indices, and `test_calibration_io.cpp` /
`make_chessboard.hpp` state the same convention.

*ODM/OpenSfM file format.* The published format is integer-origin:
`opensfm/doc/source/geometry.rst` defines pixel coordinates with the origin at
the centre of the top-left pixel and the bottom-right pixel at
`(width-1, height-1)`, and `opensfm/features.py:normalized_image_coordinates`
implements `(p + 0.5 - width/2) / max(width, height)`. A value written to that
convention therefore maps to the engine's coordinate `p + 0.5`.

*This dataset's file.* The GCPs were placed with GCP Editor Pro (the tool ODM
links to; the dataset's [community
thread](https://community.opendronemap.org/t/gcp-projected-coordinates/11762/7)
says so). Its 2022 tagger, current when the dataset was made, computes the
click as `(clientX - rect.left) / zoom / scale`
(`src/app/smartimage/smartimage.component.ts:getPos`), i.e. the offset from the
image's top-left edge over the continuous range `[0, width]`; the current SVG
implementation clamps to `[0, naturalWidth]`. Those are corner-origin values,
the engine's convention, so no shift is applied by default.

The protocol therefore reads the file as written (`--gcp-pixel-origin file`,
shift 0). `--gcp-pixel-origin format` applies the documented +0.5 conversion for
a file produced to the OpenSfM convention. The measured ±0.5 px probes remain in
the diagnostics so a producer or tool-convention change is visible.

#### GNSS antenna lever arm: hardware and per-fold estimate (2026-11-11)

*Hardware.* The block is an Autel EVO II Pro RTK (XT705 1-inch CMOS,
5472x3648, calibrated focal 4404.1667 px, nadir gimbal; dataset README). The
EXIF/XMP records the antenna position and RTK quality (`RtkFlag=50`,
`RtkStdLon/Lat/Hgt` about 0.014 / 0.014 / 0.030 m), the gimbal angles and the
absolute altitude; the RINEX marker file (`RTK_Data/101FTASK_Timestamp.mrk`)
records the same antenna positions per image. Neither states where the phase
centre sits relative to the camera's projection centre, so the dataset carries
no hardware lever-arm value. The recorded positions are not corrected to the
camera centre: the GCP-only adaptive adjustment leaves the camera centres about
0.2 m below them and about 0.2 m horizontally (mostly along the camera y axis),
which is the mount itself.

*Protocol.* `--antenna-offset x,y,z` sets a fixed camera-frame lever arm (the
antenna phase centre relative to the projection centre; x right, y down, z
forward, so an antenna above a nadir lens is negative z). It is set before
`UpdateTransform` and `OptimizeCameras`. `Chunk.set_sensor_antenna_offset`
takes the three scalars; the older `set_sensor_antenna` takes an Eigen vector,
whose pybind11 caster needs `pybind11/eigen.h` and a numpy array, and on the
benchmark's pybind11 2.11.1 with numpy 2.x that conversion faults (the first
scalar binding added here therefore avoids the caster). `OptimizeCameras`
consumes the mount through its centre prior; `UpdateTransform`
(`update_transform`) did not before PR #115 and now applies the same
correction as the `AlignCameras` datum path (`georeference_chunk`).
Diagnostic variants apply either
the measured seed
`(0, 0, -0.23) m` — the GCP-only adjustment puts the camera centres 0.23 m below
the antenna positions — or the **per-fold estimate**: the mean camera-frame
offset `R_pose R^T (A - C)` over the aligned cameras of that fold's GCP-only
adjustment, where `A` is the stored RTK position and `C` the adjusted centre.
Only the fold's control GCPs constrain that adjustment, so the held-out check
point never enters the estimate.

| check | estimated lever arm x / y / z (m) | camera spread x / y / z (m) |
| --- | --- | --- |
| 1 | -0.003 / -0.224 / -0.283 | 0.226 / 0.237 / 0.171 |
| 2 | 0.016 / -0.195 / -0.246 | 0.194 / 0.189 / 0.130 |
| 3 | -0.040 / -0.188 / -0.170 | 0.154 / 0.162 / 0.095 |
| 4 | -0.040 / -0.188 / -0.108 | 0.173 / 0.157 / 0.098 |
| 5 | 0.002 / -0.238 / -0.212 | 0.176 / 0.168 / 0.119 |
| mean | -0.013 / -0.207 / -0.204 | |

The estimate is a block mean, not a per-camera measurement: the spread over
the 146 aligned cameras is 0.15-0.24 m per axis, comparable to the value
itself, so the block datum and the residual lens model still move individual
cameras. The y and z components (about -0.21 m each, magnitude 0.29 m) are
stable across folds; z varies from -0.11 to -0.28 m. The measured seed in the
fixed variant is the vertical component alone, `(0, 0, -0.23) m`.

#### Measured: pixel-origin and antenna variants (CI run `43027876529`, commit `78816cf`)

Leave-one-out check RMS, horizontal / vertical / 3D, in cm. The alignment is
the same 146/176 cameras at 0.740 px. `--gcp-pixel-origin file` (the default)
reads the gcp_list values as written; `format` would apply the documented
+0.5 px conversion. The antenna variants apply the fixed seed
`(0, 0, -0.23) m` or the per-fold control-only estimate from the adaptive
GCP-only adjustment.

| variant | lens model | h | v | 3D |
| --- | --- | ---: | ---: | ---: |
| `gcps_only` (default) | default | 7.9 | 10.1 | 12.8 |
| `camera_rtk_gcps` (default) | default | 7.5 | 16.8 | 18.4 |
| `gcps_only`, -0.5 px | default | 7.3 | 8.9 | 11.5 |
| `gcps_only`, +0.5 px (format conversion) | default | 8.6 | 11.7 | 14.5 |
| `gcps_only`, adaptive | adaptive | 6.5 | 6.4 | 9.1 |
| `gcps_only`, adaptive, -0.5 px | adaptive | 6.2 | 6.0 | 8.6 |
| `antenna_fixed` | default | 7.6 | 16.9 | 18.6 |
| `antenna_estimated` | default | 7.9 | 20.2 | 21.7 |
| `camera_rtk_adaptive` | adaptive | 5.4 | 10.3 | 11.6 |
| `camera_rtk_adaptive_px_minus` | adaptive | 5.1 | 9.9 | 11.1 |
| `antenna_fixed_adaptive` | adaptive | 5.5 | 10.4 | 11.7 |
| `antenna_estimated_adaptive` | adaptive | 5.3 | 12.9 | 14.0 |
| `antenna_estimated_adaptive_px_minus` | adaptive | 5.1 | 11.9 | 12.9 |

The adaptive rows in this table are one run each. The replicated comparison of
the two standard variants is in "Replicated: adaptive fitting on and off"
above; it repeats `gcps_only` and `camera_rtk_gcps` only, so the antenna variants
here remain single-run.

Findings:

- **The antenna does not improve the check in this run.** The BA centre prior
  does consume it — after each antenna fold the residual offset the protocol
  measures is the applied arm itself (fixed seed: 0.006 / -0.001 / -0.225 m;
  estimated: about -0.02 / -0.18 / -0.20 m), so the mount is pinned. The
  vertical still does not improve, and with the default lens the estimated
  mount is worse (20.2 cm vertical). In this run the datum was still the
  cause: the task's `UpdateTransform` (`update_transform`) fitted one
  similarity without subtracting the lever arm; only the `AlignCameras` path
  (`georeference_chunk`) was antenna-aware, so the BA prior pulled the block
  against the datum. `update_transform` was made antenna-aware in PR #115;
  the re-measurement is the next subsection.
- **The 5 cm horizontal target is not met.** With the defaults it is 7.9 cm
  (`gcps_only`) and 7.5 cm (`camera_rtk_gcps`). The closest configurations are
  `camera_rtk_adaptive_px_minus` at 5.08 cm (vertical 9.85) and
  `antenna_estimated_adaptive_px_minus` at 5.05 cm (vertical 11.86), i.e.
  0.05-0.08 cm above the target. The target is not loosened.
- **The vertical.** Default `camera_rtk_gcps` is 16.8 cm and `gcps_only` 10.1;
  the best camera-RTK configuration is 9.85 cm (adaptive, -0.5 px) and the
  best overall is 5.95 cm (`gcps_only`, adaptive, -0.5 px). No antenna variant
  reduces it below the configuration without the antenna.
- **The ±0.5 px probe is not a convention correction.** The gcp_list file and
  the engine share the corner-origin convention, so no shift is applied. The
  probes show the nudge still changes the check by about 0.7 cm horizontally
  (-0.5 improves, +0.5 worsens); it is an image-space compensation for a
  residual in this block, not a half-pixel convention difference, and it is
  not made the default.

#### Replicated: adaptive fitting on and off (2026-11-13, runs `43450507699` / `43450529571`)

Every dataset was run three times through the quality benchmark at commit
`4645cc0`, once with `optimize_params=adaptive_fitting=true` (`43450507699`) and
once with `adaptive_fitting=false` (`43450529571`). The repetition support
(`--reps 3`) keeps each run in its own directory and records an alignment pose
hash and a final pose hash per run, so the numbers below are a distribution, not
a single sample.

**Determinism.** All three repetitions of every dataset reproduced exactly:
every numeric metric's range is zero and the pose hashes agree. The
per-dataset alignment hash is identical within a configuration and across the
two dispatches (`857ee1ac11308946` aerial-nadir, `ea26af008f7fbe47`
terrestrial-laser, `a4b0f0c4d7916625` gcp-rtk), so the alignment before
OptimizeCameras is bit-identical; only the final pose differs between adaptive
on and off, as the different calibration model implies. The GCP protocol never
re-saves its block after the folds, so it records an alignment hash but no final
hash. The earlier caveat that the alignment was not bit-identical between runs
is superseded by these hashes; separately, the adaptive-fit selection that could
previously cross its accept threshold differently under threaded summation is
now pinned to one thread (`engine/src/common/algorithm/sfm/adaptive_fit.cpp`).

Every value below is the median of the three runs, and every metric's range was
`[median–median]` (zero) — the per-run values are in that run's
`repetitions` block in the benchmark artifact. GCP leave-one-out check RMS is in
cm:

| variant | adaptive | h | v | 3D |
| --- | --- | ---: | ---: | ---: |
| `gcps_only` | **on** | 6.54 | 6.36 | 9.12 |
| `gcps_only` | off | 7.91 | 10.10 | 12.83 |
| `camera_rtk_gcps` | **on** | 5.40 | 10.26 | 11.60 |
| `camera_rtk_gcps` | off | 7.54 | 16.81 | 18.43 |

Aerial-nadir (Bandamaji) reference comparison:

| metric | adaptive on | adaptive off |
| --- | ---: | ---: |
| DEM ΔZ NMAD, offset removed (m) | 1.017 | **0.764** |
| DEM ΔZ p68, offset removed (m) | 1.005 | **0.777** |
| ortho offset (m) | 1.496 | **0.802** |
| ortho tile dispersion (MAD, m) | 0.490 | **0.044** |

Terrestrial-laser (ETH3D `terrains`) accuracy and completeness are the fraction
of reconstructed / reference points within the tolerance (higher is better):

| metric | adaptive on | adaptive off |
| --- | ---: | ---: |
| accuracy @ 2 cm | 0.0758 | **0.0925** |
| accuracy @ 5 cm | 0.2148 | **0.2516** |
| accuracy @ 10 cm | 0.4169 | **0.4899** |
| completeness @ 2 cm | 0.0475 | **0.0640** |
| completeness @ 5 cm | 0.1122 | **0.1513** |
| completeness @ 10 cm | 0.2183 | **0.2773** |
| F-score @ 5 cm | 0.1474 | **0.1890** |

**Recommendation: keep `adaptive_fitting` off by default, and enable it
conditionally when the chunk has enabled control points.** On the GCP block the
adaptive model is a clear win — 6.54 / 6.36 / 9.12 against 7.91 / 10.10 / 12.83 cm
for `gcps_only`, and 5.40 / 10.26 / 11.60 against 7.54 / 16.81 / 18.43 cm for
`camera_rtk_gcps` — and every improvement is many times the (zero) rep range. On
aerial-nadir it is a real loss on both gated metrics (NMAD +0.252 m, ortho
offset +0.693 m), and on terrestrial-laser it loses accuracy, completeness and
F-score at every tolerance. A global default-on is therefore not supported.

A camera-reference trigger is also not supported: aerial-nadir is georeferenced
(90 GNSS prior axes) and has no control points, so turning adaptive on for
"enabled camera references" would reproduce the regression above. The condition
that separates the three datasets is enabled **control points** — gcp-rtk has
five GCPs (four control points in each leave-one-out fold), aerial-nadir and
terrestrial-laser have none. The engine has an `adaptive_fitting_auto`
parameter that selects adaptive fitting when `control_markers > 0`
(`optimize_cameras_task.cpp`).

**Default since 0.3.0: `adaptive_fitting_auto = true`.** The rule, in full:

- It engages only on a block with at least one control point: a referenced,
  triangulated marker that is not a check point, with `use_gcp` on. Camera
  references alone never trigger it, and a block without control solves exactly
  as before, so aerial-nadir and terrestrial-laser are unaffected by
  construction.
- It does not engage when the caller holds every base coefficient (f, cx, cy,
  k1–k3, p1, p2): that call asked for a fixed lens.
- When it engages and the caller enabled none of k4, b1, b2, p3, p4, those terms
  become candidates for the selection, matching the measured configuration. A
  caller that enabled some of them keeps exactly its own set, and a term in the
  sensor's `fixed_params` stays fixed.
- An explicit `adaptive_fitting = true` behaves as before. To keep the plain
  model on a controlled block, set `adaptive_fitting_auto = false`.

The tests `AdaptiveFittingAutoIsTheDefault`, `AdaptiveFittingAutoOffKeepsThePlainModel`,
`AdaptiveFittingAutoIgnoresCheckPoints`, `AdaptiveFittingAutoLeavesAFixedLensFixed`,
`AdaptiveFittingAutoSelectsWithControlMarkers` and
`AdaptiveFittingAutoIgnoresCameraReferencesAlone` pin the rule. The validation
protocol names the lens model of every variant explicitly (a variant that does
not ask for the adaptive model passes `adaptive_fitting_auto=False`), so the
`gcps_only` and `camera_rtk_gcps` rows keep their meaning across versions; the
report-only `engine_defaults` variant shows what a caller who sets nothing gets.

**The 5 cm horizontal target is still missed.** The closest replicated
configuration is `camera_rtk_gcps` with adaptive fitting at 5.40 cm (the
report-only −0.5 px probe reached 5.05 cm in the single diagnostic run), against
7.54 cm with the default model and 7.91 cm for `gcps_only`. The target is not
loosened. For an RTK/GCP block, enabling `adaptive_fitting` (or
`adaptive_fitting_auto`) is the configuration to use, but it must not be
described as meeting the target; the defaults miss by 2.5–2.9 cm, and the
antenna lever arm does not close the gap.

#### Datum fix: UpdateTransform consumes the mount (2026-11-12, PR #115)

**The bug.** `update_transform` paired each camera centre directly with its
GNSS reference and never read `sensor.antenna`
(`engine/src/common/algorithm/sfm/georeference.cpp`, the camera sample in the
`update_transform` loop). Every other consumer did read it:
`georeference_chunk` (the AlignCameras datum path, same file),
`OptimizeCameras`' centre prior (`optimize_cameras_task.cpp:668`, evaluated as
`A = C + R^T lever` by `predicted_antenna_position`,
`bundle_adjustment.cpp:60`), `AlignFromReference`
(`align_from_reference_task.cpp:236`), and the reference table's camera
error/RMSE (`reference_table.cpp:44`). With a mount set, the datum therefore
placed projection centres on the antenna phase centres while the adjustment
placed them a lever arm away. Auditing those paths also found a second, smaller
inconsistency: `georeference_chunk` resolved the antenna through
`camera.sensor_id` directly, while the BA prior, AlignFromReference and the
reference table use the sensor's **effective master** (the mount is part of the
physical body). Both datum paths now share one helper,
`antenna_mount` (`georeference.cpp:66`), which resolves the master.

`update_transform` now shifts each camera fit target to
`reference - enu_from_source * R_c^T * lever`, re-estimating the fit's rotation
until it settles, exactly as `georeference_chunk` does
(`georeference.cpp:704-733`). Without a mount the code takes the unchanged
centre-to-reference path. Unit test `test_georeference_antenna.cpp` builds a
block whose references are exact antenna positions and checks the datum
recovers the true centres with the mount, misses by the lever-arm length
without it, and resolves a master-sensor mount.

**Sign and frame.** No mismatch was found. The antenna model defines
`offset_m` as the antenna relative to the projection centre in the camera
frame (x right, y down, z forward; `antenna.hpp:57`), so
`A = C + R^T * camera_offset()`. The BA prior divides the metre offset by the
chunk's metres-per-local-unit scale (`local_scale`, `optimize_cameras_task.cpp:356`)
and evaluates the same `A = C + R^T lever`; the datum's `A - R^T lever` target
is its inverse; and the protocol's per-fold estimate
`R_pose R^T (A - C)` (`run_gcp_validation.py:1452`) is the same camera-frame
quantity. The vertical sign is correct: an antenna above a nadir lens is
negative camera z, and applying it lowers the centres. The fixed seed
`(0, 0, -0.23) m` is only the vertical component; the measured mount also has
a large camera-y component.

Measured by the `gcp-rtk` quality-bench run `43073749931` (PR branch, same
protocol and diagnostics; alignment 146/176 cameras at 0.740 px). Leave-one-out
check RMS, horizontal / vertical / 3D, in cm:

| variant | lens model | h | v | 3D |
| --- | --- | ---: | ---: | ---: |
| `gcps_only` (default) | default | 7.91 | 10.10 | 12.83 |
| `gcps_only`, `adaptive_fitting` | adaptive | 6.54 | 6.36 | 9.12 |
| `camera_rtk_gcps` (default) | default | 7.54 | 16.81 | 18.43 |
| `camera_rtk_gcps`, `adaptive_fitting` | adaptive | 5.40 | 10.26 | 11.60 |
| `antenna_fixed` | default | 7.64 | 16.91 | 18.56 |
| `antenna_fixed`, `adaptive_fitting` | adaptive | 5.47 | 10.37 | 11.72 |
| `antenna_estimated` | default | 7.99 | 20.15 | 21.68 |
| `antenna_estimated`, `adaptive_fitting` | adaptive | 5.34 | 12.89 | 13.95 |
| `antenna_estimated`, adaptive, -0.5 px (report-only) | adaptive | 5.05 | 11.89 | 12.92 |

The adaptive rows here are one run per variant. The replicated comparison in
"Replicated: adaptive fitting on and off" repeats only `gcps_only` and
`camera_rtk_gcps`, so the antenna variants remain single-run.

The fix makes the two datum consumers agree. The protocol's camera-reference
residual against the **modelled antenna** (mean E/N/U over the 146 cameras)
drops to `0.003 / 0.002 / 0.004 m` with the fixed arm and
`0.003 / 0.004 / 0.005 m` with the estimated arm (RMS
`0.042 / 0.047 / 0.037` and `0.047 / 0.055 / 0.041 m`). Before the fix the same
comparison would have shown the full arm, because only the adjustment applied
it.

**The check does not improve, and this is the honest result.** Every variant's
check RMS moves by at most about a tenth of a centimetre relative to the run
with the old datum. The mount is real — the
datum and the prior now both apply it — but it is not what limits the check.
The GCP-only adjustment's per-camera antenna estimate has a 0.15–0.24 m spread
over 146 cameras (table above), so a block-mean arm applied as a 5 cm prior to
every camera over-constrains the network; the mean 0.29 m arm predicts a 0.2 m
shift that the data do not support per camera. Adding the estimate is
therefore slightly worse than not applying it (vertical 20.15 vs 16.81 cm with
the default lens; 12.89 vs 10.26 cm adaptive).

**Is ≤5 cm horizontal met? No.** With the defaults it is 7.91 cm (`gcps_only`)
and 7.54 cm (`camera_rtk_gcps`). The best non-diagnostic configuration is
`antenna_estimated` with `adaptive_fitting` at **5.34 cm** (vertical 12.89), and
the report-only −0.5 px probe reaches 5.05 cm. The target is not met and is not
loosened.

**Vertical.** The fix leaves it essentially unchanged: with the default lens
`camera_rtk_gcps` is 16.81 cm (was 16.81) and `gcps_only` 10.10 cm; with the
adaptive model they are 10.26 and 6.36 cm. Applying the fixed arm gives 16.91 cm
(adaptive 10.37), the estimated arm 20.15 cm (adaptive 12.89). The vertical
error is the camera-RTK versus GCP datum disagreement, not the mean lever arm:
the GCP-only adjustment already reaches 6.36 cm vertical without any camera
positions.

#### Per-photo GNSS accuracy, fix status and timing (2026-11-13, PR #119)

The tables above were measured with one uniform camera weight; this section
imports each photo's own accuracy and measures the difference.

**Metadata.** Every image's XMP packet (drone namespace
`http://www.dji.com/drone-dji/1.0/`) carries `RtkFlag=50` and
`RtkStdLon`/`RtkStdLat`/`RtkStdHgt`. Across all 176 exposures the reported
1-sigma deviations are 1.21-1.46 cm east (mean 1.29), 1.33-1.55 cm north (mean
1.39) and 2.45-3.01 cm up (mean 2.66); all 176 are fixed, none float or
single-point. The RTK module's per-exposure event-mark log
(`RTK_Data/101FTASK_Timestamp.mrk`, GPS week 2211) carries the same antenna
positions and deviations, ordered latitude, longitude, height, plus the GPS
time of week and the same `50` status. Its positions equal the XMP values
exactly. Each JPEG's comment carries two module counters: `timestamp` (shutter
command, ms) and `exppts` (exposure event, us).

**Import rule.** `AddPhotos(load_xmp_accuracy=True)` stores each photo's
reported east/north/up deviation as that camera's reference accuracy. A fixed
(50) or float (34) solution uses the deviations directly. Any other reported
status keeps the position but raises each axis to at least 0.5 m, so a
single-point fix cannot outvote centimetre survey control. A packet with
deviations but no status uses them as written; a packet with no deviations
keeps the 10 m consumer default. The status is persisted as the optional
`rtk_flag` attribute on the camera's `<reference>` (doc.xml v2) and exposed as
`camera.reference_rtk_flag`. The parser previously recognised only the
documentation namespace `http://ns.dji.com/...`; this dataset writes
`http://www.dji.com/...`, so a real packet parsed as empty. Both spellings are
accepted now.

**Protocol.** `--camera-accuracy per-image` enables the import at AddPhotos
time and aligns with it; `camera_rtk_uniform` refits the same alignment with one
uniform weight (0.05 m, `--camera-accuracy-uniform`) to separate the weight
effect from the alignment. The per-fold diagnostic flags every camera whose
prior residual exceeds 3 sigma on any axis, keeping all flagged labels and the
worst 20 full residual records, with the camera's GNSS status and along-track
residual attached. `--mrk` parses the event-mark log for the timing comparison.

**Timing.** Pairing each exposure's event mark with the image's `exppts`
counter and removing the constant clock offset leaves a 0.38 ms RMS
disagreement (max 1.45 ms) over all 176 exposures: the event mark is stamped at
the exposure, so the recorded position is not late. The shutter command
precedes the exposure by 0.183-0.188 s (median 0.186), but that is
trigger-to-exposure, not a position lag. Consecutive event marks give a
3.99 m/s median block speed (up to 4.08 m/s on straight runs, 0.15 m/s in
turns); a 0.186 s position lag would be 0.74 m along track. The protocol also
reports the control-only along-track latency implied by the folded camera
residuals, and the `camera_rtk_per_image_timing_shift` diagnostic removes that
lead from every camera position. Both are report-only; the direct
event/exposure comparison already rules out a real lag.

**Measured** by the `gcp-rtk` quality-bench runs `43468914296` (full diagnostic
set) and `43499714006` (the antenna subset on the same code; the base variants
reproduce the full run exactly). Alignment: 146/176 cameras, 0.740 px; all 176
cameras received their own XMP accuracy (`uniform_defaults` 0). The per-image
weights tighten `AlignCameras`' robust datum gate (0.5 m floor) to 39/146
inliers at 0.274 m, so the initial datum is camera-only from 39 cameras; the
fold's BA (438 prior axes) then moves the block onto the references. Leave-one-out
check RMS in cm:

| variant | lens | h | v | 3D |
| --- | --- | ---: | ---: | ---: |
| `gcps_only` | default | 7.91 | 10.15 | 12.87 |
| `gcps_only`, `adaptive_fitting` | adaptive | 6.54 | 6.46 | 9.19 |
| `camera_rtk_uniform` (0.05 m) | default | 7.54 | 16.81 | 18.43 |
| `camera_rtk_gcps` (per-image) | default | 7.94 | 16.26 | 18.10 |
| `camera_rtk_per_image_adaptive` | adaptive | 5.31 | 12.22 | 13.32 |
| `camera_rtk_per_image_antenna_estimated` | default | 8.17 | 19.90 | 21.51 |
| `camera_rtk_per_image_antenna_estimated_adaptive` | adaptive | 5.45 | 12.14 | 13.30 |
| `camera_rtk_per_image_timing_shift` (report-only) | default | 7.09 | 15.80 | 17.32 |
| per-image adaptive, −0.5 px (report-only, full run) | adaptive | 4.95 | 10.56 | 11.66 |

Findings:

- **The import does what it says.** With per-image weights the folded camera
  references sit 0.92 / 0.99 / 2.29 cm RMS (east/north/up) from the adjusted
  centres; the same alignment refitted at a uniform 0.05 m leaves 4.07 / 4.58
  / 3.64 cm. The block is genuinely fitted to the RTK positions.
- **The check does not improve.** The per-image base is 7.94 / 16.26 / 18.10 cm
  and its adaptive form 5.31 / 12.22 / 13.32; the uniform-weight fold on the
  same alignment is 7.54 / 16.81 / 18.43 and GCPs-only adaptive is
  6.54 / 6.46 / 9.19. The camera prior and the GCP datum disagree by about
  0.2 m vertically (the mount), so weighting the cameras more strongly cannot
  raise check-point accuracy.
- **3-sigma outliers.** In the per-image base fold only `MAX_0009` exceeds
  3 sigma, in all 5 folds (5 of 730 camera-folds); the estimated-antenna
  adaptive variant flags `MAX_0121` in 3 folds. Every camera in the block is
  RTK fixed (code 50), so there is no fix-status correlation to find. The
  flagged cameras' implied latency is −1.6 ms (base) and +16.7 ms (antenna
  adaptive) against a direct event/exposure disagreement of 0.38 ms; neither
  is a timing signature.
- **The antenna does not help.** The per-fold estimated arm is
  (-0.012, -0.204, -0.205) m, consistent with the earlier measurement. Applied
  with the adaptive lens it gives 5.45 / 12.14 / 13.30, no better than the same
  configuration without the mount (5.31 / 12.22 / 13.32).
- **The timing shift is not a timing fix.** Removing the control-only
  along-track lead (-0.148 m, -37 ms at 3.99 m/s) improves the check to
  7.09 / 15.80 / 17.32 from 7.94 / 16.26 / 18.10, but the direct
  event/exposure comparison rules out a shutter-to-fix lag; the lead is an
  along-track datum/mount difference, not time.
- **Target.** No standard configuration reaches 5 cm horizontal: the best is
  5.31 cm (per-image + adaptive). The report-only −0.5 px probe reaches
  4.95 cm. The target is not met and is not loosened. Vertical is 12.22 cm in
  the same configuration, against 6.46 cm for GCPs-only adaptive.

#### Block deformation from calibration–geometry coupling (2026-11-14, PR #120, run `43540658771`)

This section tests the classic nadir-only parallel-strip failure mode: radial
distortion and focal length/height correlate, an error in them bends the block
into a smooth dome or bowl, and the check-point vertical error grows with the
square of the distance from the block centre. The alignment and folds are the
same protocol as the run above (146/176 cameras at 0.740 px, per-image camera
accuracy, 0.015 m marker accuracy, 0.740 px alignment RMS). Every probe is
report-only; no engine default changed.

**Method.** Under `--diagnostics` each fold now also records

* the adjusted calibration (all 13 coefficients) and the BA posterior standard
  errors and f-to-coefficient correlations — `OptimizeCameras`'s
  `camera_covariance` parameter is already on by default, and the 13x13
  posterior covariance it produces is stored per sensor;
* the check point's ENU position and its horizontal radius from the centroid of
  the fold's aligned camera centres;
* an ordinary-least-squares fit `vertical = a + b (r/100 m)^2` and a full
  quadratic surface in east/north. The quadratic has six parameters and there
  are five check points, so it is reported as **underdetermined** and no
  coefficients are claimed for it; the four-parameter linear+radial fallback
  has one residual degree of freedom and its t values are not usable either.
  With five points the two-parameter radial fit has three degrees of freedom,
  and a two-sided 5% test needs |t| > 3.18;
* the same radial fit on `stored RTK position - adjusted camera centre` (up
  component) for all 146 aligned cameras, which has 146 samples instead of 5;
* the surveyed GCP convex-hull area against the hull of the camera centres.
  The five GCPs span 773.9 m² (all are within 23 m of their centroid); the
  camera-centre hull is 4560 m², a ratio of 0.170, and the block's maximum
  radius is 50.7 m. The check points sit at r = 8.3–28.2 m, so they sample the
  inner half of the block and a smooth block-wide bend has weak leverage on
  them. This is exactly why the 146-camera fit is the decisive test.

Sign convention: the camera residual is stored RTK minus adjusted centre, so a
**bowl** (adjusted block high at the edges relative to the centre) gives b < 0
in the camera fit and b > 0 in the check-point fit (adjusted minus reference);
a dome is the opposite.

The fixed-calibration workflow uses the new `Chunk.set_sensor_fixed_params`
binding. Per fold it first runs the control-only GCP **adaptive** adjustment
(camera references disabled, held-out point imported as a check point), reads
that lens, marks all 13 coefficients fixed, re-imports the camera references,
refits the datum and runs the camera-RTK adjustment with the lens frozen. The
held-out check point never enters either fit, so the workflow is leak-free.
Model-choice probes hold or free individual coefficients through the same
`fit_*` flags (`camera_rtk_k1234` frees k1–k4 with b1/b2 off,
`camera_rtk_f_fixed`, `camera_rtk_p_fixed`), and the weight probes re-import
each camera at its reported per-image accuracy ×2 and ×5 with the marker sigma
unchanged.

**Dome fits.** Check-point fit: 5 points, 2 parameters (df = 3); camera fit:
146 samples per fold; `b` is metres of height per (100 m)². A 0.5 m/(100 m)²
slope is about 13 cm across this block's 50.7 m radius.

| variant | check b | check t | check R² | camera b | camera t (per fold) | camera R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gcps_only` (default) | 2.82 | 1.80 | 0.52 | **−2.26** | **−3.53** [−3.95, −3.11] | 0.06–0.10 |
| `gcps_only`, adaptive | 0.94 | 0.66 | 0.13 | +0.13 | +0.88 [+0.25, +1.48] | 0.00–0.02 |
| `camera_rtk_gcps` (default) | 2.75 | 0.93 | 0.23 | −0.03 | −0.91 [−1.63, −0.20] | 0.00–0.02 |
| `camera_rtk_adaptive` | 2.95 | 1.28 | 0.35 | +0.02 | +0.70 [0.00, +1.42] | 0.00–0.01 |
| `camera_rtk_k1234` | 2.55 | 0.93 | 0.22 | −0.03 | −0.99 [−1.70, −0.15] | 0.00–0.02 |
| `camera_rtk_f_fixed` | 3.99 | 1.64 | 0.47 | **+0.31** | **+4.74** [+4.50, +5.07] | 0.12–0.15 |
| `camera_rtk_p_fixed` | 3.08 | 1.02 | 0.26 | −0.05 | −1.58 [−2.17, −0.92] | 0.01–0.03 |
| weight ×2 | 3.23 | 1.10 | 0.29 | −0.07 | −1.45 [−2.82, −0.56] | 0.00–0.05 |
| weight ×5 | 2.85 | 1.09 | 0.28 | −0.11 | −1.49 [−2.30, −1.04] | 0.01–0.04 |
| fixed calibration | **4.55** | **3.30** | 0.78 | +0.03 | +0.53 [+0.13, +0.79] | 0.00–0.00 |

The camera residual RMS (after its intercept) is what makes the amplitudes
comparable: 45–53 cm for `gcps_only` default, 10–16 cm for GCP-only adaptive,
1.8–1.9 cm for `camera_rtk_adaptive`, 2.2–2.3 cm for `camera_rtk_gcps`, and
3.7–4.7 cm for the fixed-calibration variant. The default GCP-only fit's
residual is not purely radial — a full quadratic in east/north explains 60–70%
of it — but its radial term is significant in all five folds.

**Calibration and correlations.** Mean over the five folds, with the per-fold
range in brackets where it matters. `f` in pixels; `cx/cy/b1/b2` in pixels;
`k1`–`k4` are the normalised radial terms.

| coefficient | `gcps_only` default | `gcps_only` adaptive | `camera_rtk_gcps` | `camera_rtk_adaptive` |
| --- | ---: | ---: | ---: | ---: |
| f | 4457.30 | 4203.62 [4199.44, 4210.38] | 4217.48 | 4219.45 [4216.12, 4222.84] |
| f posterior σ | ±5.0 | ±8.0 | ±1.1 | ±1.0 |
| cx | −19.59 | −15.57 | −20.73 | −13.97 |
| cy | **−77.15** | +22.00 | −10.06 | −2.70 |
| k1 | 0.0520 | 0.02656 | 0.0538 | 0.02854 |
| k2 | −0.2019 | 0 (not selected) | −0.1721 | 0 (not selected) |
| k3 | 0.2633 | −0.1813 | 0.2029 | −0.2134 |
| k4 | held | 0.2863 | held | 0.3376 |
| b1 | held | −12.07 | held | −10.85 |
| b2 | held | 0.97 [0, 2.45] | held | 1.31 [0, 1.92] |
| p1 / p2 | −0.00127 / 0.00182 | −0.00106 / 0.00058 | −0.00154 / −0.00005 | −0.00086 / 0.00051 |
| p3 / p4 | held | not selected (0 / 0) | held | not selected (0 / 0) |
| corr(f, k1) | **+0.551** | +0.168 | +0.037 | +0.178 |
| corr(f, k2) | −0.372 | — | +0.038 | — |
| corr(f, k3) | +0.517 | −0.600 | +0.025 | −0.028 |
| corr(f, cy) | −0.881 | **−0.957** | −0.558 | −0.564 |

The coupling in this block is **f ↔ cy and f ↔ k1 in the default model**, not
f ↔ k1 alone. In the default GCP-only fit f and k1 correlate at +0.55, cy
drifts to −77 px, f sits 1.2% above the EXIF value, and the camera centres end
up 3.45 m above the RTK positions with a significant radial bowl. The adaptive
model replaces that trade with a large b1 (−12 px), k4 (0.29) and k3 (−0.18),
drops the f–k1 correlation to 0.17, leaves f 4.5% below EXIF but determined to
±8 px, removes the mean camera offset (0.23 m, the mount), and removes the
radial bowl. In the camera-RTK fits the same extended solution has f determined
to ±1 px and no radial residual. The strong f–cy correlation (−0.96 GCP-only,
−0.56 camera-RTK) is the remaining weak direction; it is what makes the
control-only focal length uncertain, and it is why the frozen lens below does
not transfer.

**Results.** Leave-one-out check RMS, horizontal / vertical / 3D, in cm. All
variants are leak-free; the fixed-calibration lens comes from the fold's
control-only fit.

| variant | h | v | 3D |
| --- | ---: | ---: | ---: |
| `gcps_only` (default) | 7.91 | 10.15 | 12.87 |
| `gcps_only`, adaptive | 6.54 | **6.46** | **9.19** |
| `camera_rtk_gcps` (default) | 7.94 | 16.26 | 18.10 |
| `camera_rtk_adaptive` | **5.31** | 12.22 | 13.32 |
| `camera_rtk_k1234` | 7.93 | 15.58 | 17.48 |
| `camera_rtk_f_fixed` | 29.21 | 170.92 | 173.40 |
| `camera_rtk_p_fixed` | 8.40 | 16.85 | 18.83 |
| weight ×2 | 7.91 | 17.06 | 18.80 |
| weight ×5 | 6.91 | 15.70 | 17.15 |
| fixed calibration from control GCPs, adaptive lens | 8.62 | 34.43 | 35.49 |

Findings:

- **Doming is confirmed for the default GCP-only model, as a bowl, and it is
  removed by the extended lens model.** The 146-camera residual has a
  significant negative radial slope in every fold (−2.26 m/(100 m)², t −3.1 to
  −3.95, R² 6–10%). The five check points give a slope of the same sign
  (+2.82): it implies 0.22 m at the outer check radius against the camera fit's
  0.58 m at the 50.7 m block edge, but with five points and df = 3 it does not
  reach significance (t 1.80 < 3.18). Adding b1/k4/k3 drops the camera slope to +0.13 (t 0.25–1.48,
  not significant) and the check slope to +0.94 (t 0.66), and halves the
  vertical RMS from 10.15 to 6.46 cm. The camera centres in the default fit are
  also 3.45 m above the RTK positions; the adaptive fit leaves only the 0.23 m
  mount.
- **It is not the explanation for the camera-RTK vertical residual.** With
  camera priors in play the block is anchored: the camera residual is 1.8–2.3 cm
  RMS with no significant radial term (|t| ≤ 1.4 per fold, R² ≤ 2%), and the
  check radial fit is not significant (t 0.93–1.28, and only 5 points). The
  12.2 cm vertical RMS of `camera_rtk_adaptive` is not a function of radius:
  checks 1 and 4 sit at r = 24.0 and 22.1 m yet differ by 14 cm in height, and
  checks 2 and 4 at r = 17.5 and 22.1 m differ by 28 cm. The scatter is local
  (azimuthal) inside the ~30x40 m survey patch, not a smooth bend. Loosening
  the camera priors ×2/×5 makes the camera slope
  slightly more negative (−0.07/−0.11, mean t ≈ −1.5) and moves the check from
  7.94/16.26 to 6.91/15.70, so the weight balance is not the missing piece
  either.
- **Focal length must be free.** Holding f at its EXIF/initial value (4404.17
  px) is catastrophic — 29.21 / 170.92 / 173.40 cm — because the block's
  solution needs f ≈ 4217 px and a held focal length forces the height scale
  wrong even though the camera centres still fit the RTK positions. It is also
  the mirror image of the coupling: with f held the camera residual gains a
  significant positive radial slope (+0.31, t 4.50–5.07, R² 12–15%), showing a
  lens error does bend the camera network, but here a ~1.7 m uniform vertical
  bias dominates the radial part. Freeing the radial terms alone (`k1234`) is
  neutral (15.58 vertical), and holding the tangential terms is slightly worse
  (16.85).
- **The fixed-calibration workflow as specified does not work.** The
  control-only adaptive lens (f 4203.6, cy +22.0, b1 −12.1, k4 0.286) is
  overfitted to four control points in a survey patch whose five-point hull is
  only 774 m²: b1 absorbs a −12 px affine term, and freezing it and refitting
  to the camera positions gives
  8.62 / 34.43 / 35.49 cm. The camera centres still fit the RTK positions
  (1.2/1.3 cm east/north, 4.2 cm up RMS), but the triangulated ground surface is
  bent by 0.2–0.56 m and the check radial fit becomes significant (+4.55,
  t 3.30, R² 0.78). This is a
  clean demonstration of the coupling mechanism — a wrong lens deforms the
  surface while the cameras stay pinned — but the control-only calibration is
  not transferable to the camera-RTK datum.
- **The 5 cm horizontal target is not met by any leak-free setting.** The best
  is `camera_rtk_adaptive` at **5.31 cm** (12.22 vertical), followed by
  `gcps_only` adaptive at 6.54 cm (its vertical, 6.46 cm, is the best). Weight
  ×5 reaches 6.91 cm. The target is not loosened. The check-point quadratic
  surface is underdetermined at five points and the linear+radial fallback has
  one degree of freedom, so neither supports a stronger claim than the radial
  fit; the 146-camera fit is the one with power, and it says the standard
  camera-RTK block does not dome.

#### Reference frames: camera RTK vs the surveyed GCPs (2026-11-15, PR #121, run `43721644793`)

The camera RTK positions and the surveyed GCPs are independent absolute
sources. If they were expressed in different realizations of the same geodetic
system — the European NTRIP frame ETRS89 (fixed at epoch 1989.0) against the
receiver's WGS 84(G2139)/ITRF2014 solution at the 2022.4 capture epoch — they
would differ by about 0.85 m horizontally and a few millimetres vertically at
this site. That would explain a systematic fault when the camera references are
added while the GCP-only adaptive adjustment reaches 6.5 cm. This section
measures the frame directly. Every probe is report-only; no engine default
changed.

**What each set declares.** None of the dataset's files names a realization or
a coordinate epoch.

| set | declaration in the file | source |
| --- | --- | --- |
| camera RTK | EXIF `GPSMapDatum = WGS-84`; XMP has no datum, `RtkFlag=50`, `RtkStdLon/Lat/Hgt` 0.0137 / 0.0144 / 0.0297 m | `images/MAX_0002.JPG`, GPS IFD tag 18 and the drone XMP |
| GCP list | header `EPSG:4326`, then `lon lat height px py image label` | `GCP/gcp_list.txt` line 1 |
| survey CSV | `Name;Easting;Northing;Elevation;Longitude;Latitude`, no CRS column; the eastings/northings are UTM zone 33N | `GCP/latlon-easting_northing.csv` |
| drone GNSS | RINEX 3.04 rover file; `MARKER NAME/NUMBER/TYPE`, `OBSERVER`, `ANT # / TYPE` blank, `APPROX POSITION XYZ` 0 0 0 | `RTK_Data/101FTASK_Rinex.obs` header |
| event marks | per-exposure latitude, longitude, ellipsoidal height, per-axis deviations and status `50`, GPS week 2211 | `RTK_Data/101FTASK_Timestamp.mrk` |
| dataset README | GCP "Emlid Reach RS2, NTRIP, estimated 1-2cm accuracy"; RTK yes | `README.md` |

No file records the NTRIP mountpoint, the correction network, or a coordinate
epoch, so "WGS-84" in the EXIF is the receiver's label rather than a
realization. The site is 47.6435 N, 16.4759 E, which reverse-geocodes to
Ritzing, Bezirk Oberpullendorf, Burgenland, Austria; the dataset is Austrian,
not German. The projected CSV does not resolve the realization either:
projecting its own longitude/latitude in EPSG:32633 (WGS 84 / UTM 33N) or
EPSG:25833 (ETRS89 / UTM 33N) reproduces the CSV eastings and northings to
within 3.6 cm, because PROJ applies the null ETRS89-to-WGS 84 operation when no
coordinate epoch is supplied.

**How the engine reads them.** The protocol declares the chunk `EPSG:4326` and
both reference slots default to the chunk CRS (`engine/src/common/core/chunk.hpp`
lines 499-511). The CRS DTO carries WKT, an EPSG code and an optional vertical
datum only (`engine/src/common/core/types.hpp` line 72); it has no realization
or coordinate epoch, and EPSG:4326 has none by definition. Both sets are
therefore read as one realization-less geographic frame, so a realization
difference cannot appear inside the engine — only in the residual against the
independent set. That is why the test lives in the protocol.

**Capture epoch.** The first RTK event mark is GPS week 2211, time of week
296288.242869 s, i.e. 2022-05-25 10:17:50 UTC after the 18 s leap-second
offset: decimal year 2022.3957. The EXIF `DateTimeOriginal` is 2022:05:25
12:17:49 local and the RINEX covers 10:18:04.4-10:26:30.6 UTC, so the epoch is
consistent to the minute.

**Expected offset.** PROJ 9.5.1 (EPSG database) resolves the time-dependent
EPSG:7912 (ITRF2014, geographic 3D) to EPSG:4258 (ETRS89) operation through the
ETRF2020 chain, with a stated accuracy of 0.1 m. At this site and epoch it is
**-0.6981 / -0.4912 / +0.0001 m** east / north / up: 0.854 m horizontally
(azimuth 234.9°) and no vertical. The mirror operation is its exact inverse.
ITRF2020 to ETRS89 gives -0.6993 / -0.4879 / -0.0016 m (0.853 m); the ETRF2014
and ETRF2000 realizations give 0.871 and 0.893 m. The realization choice moves
the magnitude by 2 cm and the vertical not at all. The parameters are those of
Altamimi et al. (2016) for ITRF2014 and the EUREF definition of ETRS89
(coincident with the ITRS at epoch 1989.0).

**Leak-free per-fold test.** Each fold first runs the GCP-only adjustment with
the adaptive lens model: the four control GCPs are the only georeferencing
input, the held-out point is imported as a check marker, and the camera
references are disabled. The fit is then `stored RTK position - adjusted camera
centre` over the 146 aligned cameras. A 3-parameter translation and a
7-parameter Helmert (`a = s R c + t`) are reported; rotations are the
small-angle vector about the frame's E/N/U axes in arcsec, scale in ppm.

| fold | translation E / N / U (m) | horizontal (m) | rotation E / N / U (arcsec) | scale (ppm) | translation residual (m) | Helmert residual (m) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | +0.215 / +0.175 / +0.318 | 0.277 | -819.8 / +757.8 / +13.1 | +489.4 | 0.309 | 0.276 |
| 2 | +0.173 / +0.112 / +0.268 | 0.206 | -493.1 / +613.1 / +213.8 | -1630.9 | 0.273 | 0.248 |
| 3 | +0.052 / +0.113 / +0.200 | 0.124 | -618.9 / +87.2 / -128.4 | -241.7 | 0.264 | 0.254 |
| 4 | +0.145 / +0.019 / +0.138 | 0.146 | -192.9 / +552.3 / -250.2 | -1741.4 | 0.268 | 0.252 |
| 5 | +0.097 / +0.093 / +0.247 | 0.134 | -609.1 / +250.2 / +169.8 | -1765.5 | 0.310 | 0.294 |
| mean | **+0.137 / +0.102 / +0.234** | **0.178** | -546.8 / +452.1 / +3.6 | -978.0 | **0.285** | **0.265** |

The per-axis ranges across folds are 0.052-0.215 m east, 0.019-0.175 m north
and 0.138-0.318 m up; the horizontal mean is 0.178 m (0.124-0.277 m). The
translation-only residual is 0.285 m and the 7-parameter fit leaves 0.265 m,
only 7% smaller. Its rotation (total 636-1116 arcsec) and scale (-1766 to
+489 ppm) change sign across folds. With a 0.28 m camera scatter and a control
pattern spanning 774 m², those four extra parameters are not a rigid frame
rotation and scale: they are the datum absorbing the block, and the translation
is the only stable part. The default-lens GCP-only fold is reported too and is
not a frame estimate at all: it puts the centres 3.45 m below the references
(mean +0.611 / -0.203 / -3.448 m, residual 0.772 m, rotation 3734 arcsec,
scale 11984 ppm), which is the calibration-geometry trade identified above.
This is why the frame test, and the empirical correction below, use the
adaptive GCP-only fold. (The earlier run `43556154502` was dispatched with
`optimize_params=adaptive_fitting=true`, so its `gcps_only` frame fit was
already this adaptive one; the present run separates the two and reproduces the
same adaptive translation.)

**The translation is the antenna mount, not a frame offset.** The same folds
estimate the camera-frame antenna offset at -0.013 / -0.207 / -0.206 m (mean;
magnitude 0.292 m, per-fold magnitude 0.22-0.37 m). The measured translation has
per-fold magnitude 0.20-0.42 m (mean 0.290 m) and a +0.234 m up component,
which is the mount's +0.21 m up. The two magnitudes agree within 5 cm in every
fold and to 1% on the mean. A realization offset would be 0.854 m horizontal
with 0.000 m vertical: nearly three times the whole measured vector, with all
of it in the horizontal instead of the vertical.

**Remedy variants.** All are leak-free: the published transform is applied to
the stored camera references before the datum refit and the adjustment, and the
empirical translation is measured on the same fold's control-only GCP
adjustment (the held-out point never enters it). Leave-one-out check RMS,
horizontal / vertical / 3D, in cm:

| variant | lens | h | v | 3D |
| --- | --- | ---: | ---: | ---: |
| `gcps_only` (default) | default | 7.91 | 10.15 | 12.87 |
| `gcps_only`, adaptive | adaptive | 6.54 | 6.46 | 9.19 |
| `camera_rtk_gcps` (per-image) | default | 7.94 | 16.26 | 18.10 |
| `camera_rtk_adaptive` (best horizontal) | adaptive | **5.31** | 12.22 | 13.32 |
| published ITRF2014 → ETRS89 | default | 85.26 | 26.41 | 89.26 |
| published ITRF2014 → ETRS89, adaptive | adaptive | 81.44 | 13.36 | 82.53 |
| published ETRS89 → ITRF2014 (mirror), adaptive | adaptive | 76.29 | 13.30 | 77.44 |
| empirical per-fold translation, adaptive | adaptive | 16.01 | 15.27 | 22.12 |

The published transform is applied in the hypothesis direction (cameras in
WGS 84/ITRF2014, GCPs in ETRS89) and in the mirror direction. Both take the
check to about 0.8 m, the size of the applied displacement: the correct
direction would remove a real 0.85 m bias and leave the block at the
no-transform level, whereas here it adds it. That is the signature of
references that were already in the target frame. The empirical translation
also fails to help: it removes the mount from the references but the adjustment
already places the centres acceptably, and the check moves from 5.31 to
16.01 cm horizontal.

**Verdict: no frame mismatch.** The camera RTK positions and the surveyed GCPs
are in the same geodetic frame to within the test's resolution. The observed
camera-vs-GCP offset is the antenna mount (0.29 m, dominated by +0.23 m
vertical), not a realization difference: the published ETRS89 ↔ ITRF2014 offset
is 0.854 m horizontal and 0.000 m vertical, while the measured translation is
0.178 m horizontal and +0.234 m up. Applying the published transform in either
direction adds about 0.8 m to the check; a real mismatch would be removed by
the correct direction. The empirical correction is leak-free but also does not
improve the check. The 5 cm horizontal target is **not met**: the best leak-free
setting remains per-image camera RTK with adaptive fitting at **5.31 cm**
horizontal (12.22 vertical, 13.32 3D); GCP-only adaptive is 6.54 / 6.46 /
9.19 cm. No frame correction beats the uncorrected variant, and the target is
not loosened.

#### Independent reference: OpenDroneMap on the same block (2026-11-18, PR #126)

The engine's numbers say what this pipeline achieves; they do not say whether
the block allows 5 cm with any tool. `.github/workflows/odm-reference.yml` runs
the same five leave-one-out folds through OpenDroneMap 3.6.2 (OpenSfM) on the
same photographs and the same survey, and `tools/bench/odm_reference.py`
triangulates the held-out point through ODM's own camera model and poses. ODM
is an independent reference, not ground truth; the survey is. Only numbers are
uploaded — the GPL-3.0 dataset is downloaded and measured, never redistributed.

**Protocol.** Each fold writes a `gcp_list.txt` with the four control points and
runs ODM to the end of OpenSfM (`--end-with opensfm`). The held-out point's
image observations are triangulated from the reconstruction's shots and cameras
(OpenSfM normalised pixels, the Brown distortion model, Gauss–Newton on the
reprojection error), converted to ODM's topocentric frame through
`reference_lla.json`, and compared with the survey. The control points are
measured the same way. Three modes: `gcp_only` (ODM's default with a GCP file),
`gcp_gps` (`--force-gps`, the camera RTK positions adjusted with the GCPs), and
`gcp_strong` (the reconstruction re-run with the GCPs weighted as a survey:
`gcp_global_weight 1.0`, 2 cm horizontal and vertical sd, against ODM's default
1% weight and 10 cm vertical sd).

**The triangulation is correct.** It was checked against ODM's own
`stats/ground_control_points.json`, which `compute_statistics` writes from the
topocentric reconstruction. Our control points agree with ODM's own to
**0.14 cm horizontal and 1.9 cm vertical RMS** on every fold (max 0.20 / 2.8
cm). The four conventions that could have been wrong were each checked against
OpenSfM's source and are right: the pixel normalisation
`(px + 0.5 - w/2) / max(w, h)` (`opensfm/features.py`), the Brown distortion
(`DistoBrown`, `opensfm/src/geometry/camera_distortions_functions.h`), the
topocentric transform (`ecef_from_topocentric_transform`, `opensfm/geo.py`),
and the `gcp_list` pixel convention (0-based corner origin,
`_read_gcp_list_lines`, `opensfm/io.py`). ODM's `reference_lla.json` altitude is
hard-coded 0.0 and the script uses it, so both sides share the frame. The
survey heights are ellipsoidal on both sides (the dataset's `gcp_height` is
`ellipsoidal` and the `gcp_list` z matches the CSV exactly), so a geoid swap —
which would be about 46 m at this site, not 0.5 m — is not in play.

**Results** (CI run `44355652940`, ODM 3.6.2, 176/176 shots reconstructed in
every fold). Check-point error against the survey, cm:

| mode | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | h RMS | v RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gcp_only` h | 13.04 | 6.25 | 8.77 | 11.76 | 4.23 | **9.40** | |
| `gcp_only` v | +73.67 | -2.67 | -24.82 | +65.11 | +24.13 | | **46.63** |
| `gcp_gps` h | 12.66 | 6.23 | 8.88 | 11.76 | 3.97 | **9.30** | |
| `gcp_gps` v | +72.21 | -3.22 | -25.44 | +66.24 | +24.85 | | **46.64** |
| `gcp_strong` h | 13.15 | 8.33 | 7.85 | 13.22 | 2.88 | **9.87** | |
| `gcp_strong` v | +81.98 | -1.57 | -19.39 | +70.01 | +26.23 | | **50.37** |

Control points, measured the same way: **3.06 cm horizontal, 11.41 cm vertical
RMS** in `gcp_gps` and 3.06 / 11.48 cm in `gcp_only`. In `gcp_strong` the
control points are pinned by the weights, so their residual is 0.00 cm by
construction and says nothing about the triangulation; the held-out point is
the only measurement there.

**The 3–4 px reprojection is ODM's real residual, not a projection mismatch.**
ODM's own statistics report the same residual on the same observations: a mean
of about (-2.2, -3.6) px with a scatter of about 2 px, i.e. a systematic shift
that dominates the noise, present identically in `gcp_only` and in `gcp_strong`
where the control points are pinned. Our own triangulation's control residual
is 3.20 px RMS with mean (-1.66, -3.45) px and scatter (1.87, 1.50) px — the
same shift, measured through our projection. It is not a pixel-origin or
normalisation error, because our projection reproduces ODM's own reprojected
pixels to the same statistics and our triangulated points match ODM's to
0.14 cm horizontal. It is the block's lens model and geometry not fitting the
GCP observations better than a few pixels; the same residual appears in the
engine's own diagnostics on this block (the default lens model is too thin for
this camera, and the adaptive model reduces it).

**Reading it against the engine.** ODM's 9.3–9.8 cm horizontal and 46–51 cm
vertical are worse than the engine's best leak-free setting (5.31 cm horizontal,
12.22 vertical with per-image camera RTK and adaptive fitting; 6.54 / 6.46 with
GCPs only). The vertical is the striking part: ODM's control points already sit
11 cm off vertically and the held-out point 46 cm, so the block's vertical
geometry is the limit, not the triangulation. The engine reaches 6.5 cm vertical
on the same block with the adaptive lens model, which is the difference the
lens model makes. Caveats: ODM ran at `--feature-quality high` with its own
defaults otherwise, the engine's numbers are from a different run and a
different alignment, and neither tool is ground truth — the survey is. The
comparison says the 5 cm horizontal target is not reachable by ODM on this
block either, and that the engine's remaining gap is in the vertical, where the
block itself is weak.