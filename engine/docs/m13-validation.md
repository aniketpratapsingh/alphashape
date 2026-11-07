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
GCPs only with the camera references disabled. The GCPs-only folds also report how
far the stored camera references sit from the adjusted camera centres (mean and RMS
east/north/up), which separates a height-datum mismatch from unfixed positions. Runtime 2702.7 s on a shared CPU runner. The gcp-list
z is orthometric (mean offset to the CSV elevation 0.0 m).

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
camera centres, the stored camera positions sit on average 3.45 m too high
(mean E/N/U -0.61 / 0.20 / 3.45 m, RMS 0.72 / 0.50 / 3.49 m over 146 cameras). The
near-constant vertical offset is a height-reference mismatch between the camera
altitudes and the surveyed GCP heights, and the ~0.4 m horizontal scatter is far
above RTK-fixed precision. Weighted at 5 cm, those positions pull against the GCPs:
the image residual rises from ~0.95 to ~1.18 px and the camera-variant vertical
error is worse than GCPs only. Until the camera altitude reference of this block is
resolved, the GCPs-only variant is the meaningful one: 7.9 cm horizontal and
10.1 cm vertical from four control points per fold.
