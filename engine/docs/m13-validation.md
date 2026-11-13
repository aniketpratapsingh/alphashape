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
| gcps_only, `refineMarkers` | 12.9 / 24.5 / 27.7 | | |
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
- **`refineMarkers` removes about half of the hand-placed projections** on these
  targets and makes the result much worse — an engine issue, tracked separately.
- Ruled out: trigger/GNSS timing (no along-track pattern once the cameras are
  weighted), marker-vs-tie weight (0.5 px is best), unaligned cameras (1 of 35
  observations).

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
terrestrial-laser have none. The engine now has an
opt-in `adaptive_fitting_auto` parameter (default `false`, so existing behaviour
is unchanged) that selects adaptive fitting when `control_markers > 0`
(`optimize_cameras_task.cpp`); the two tests
`AdaptiveFittingAutoSelectsWithControlMarkers` and
`AdaptiveFittingAutoIgnoresCameraReferencesAlone` pin both halves of the rule.
Flipping that default is a one-line change for the coordinator.

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
