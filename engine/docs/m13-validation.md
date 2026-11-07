# M13 validation — markers, GCP and calibration

The M13 close-out is validated by five reproducible synthetic scenarios and by
a real-block procedure. The script is `engine/tests/run_gcp_validation.py`; it
prints one JSON object per scenario on stdout and exits non-zero when the
held-out check-point 3D total RMS exceeds the tolerance (2 cm by default). The
horizontal and vertical RMS are reported alongside.

```
python3 engine/tests/run_gcp_validation.py --synthetic --ablations
```

## Synthetic GCP block

The scenario loads the 12-image nadir fixture (`engine/data/aerial_r*c*.jpg`)
whose truth poses, ground grid and building are recorded in
`engine/data/aerial_truth.json`. It then:

1. imports the images and declares the chunk WGS 84 geographic. The fixture EXIF
   carries exact GPS, but that is not the camera reference this scenario
   trusts: every camera gets a seeded consumer-GNSS reference of its own, a
   per-camera Gaussian offset (horizontal sigma 2 m, vertical sigma 4 m) on
   top of a common block bias of (1.5, -1.0, 3.0) m in ENU, with the matching
   per-axis accuracy (2, 2, 4) m imported through `importReference`. The
   chunk's fallback `camera_location_accuracy` is 3 m;
2. runs `MatchPhotos` and `AlignCameras`, so the block's initial georeference
   comes from those noisy camera references;
3. creates 12 markers: 6 on the ground (z = 0) and 6 on the building roof
   (z = 9.52 m), with NO seeded 3-D position;
4. adds each marker's projections by projecting its truth position through the
   fixture's truth camera pose, then offsets every projection by seeded uniform
   noise of 0.5 px peak to peak (±0.25 px per axis). The chunk's
   marker-projection accuracy is recorded from that noise (σ = a/√3). The
   bundle adjustment currently gives marker and tie observations one shared
   reprojection weight, so the value documents the injection rather than
   changing the solve;
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
| 2 | `reference-noise` | 0 px | 1 cm per axis | consumer GNSS (2 m / 4 m sigma) | reported, not gated |
| 3 | `projection-noise` | ±0.25 px per axis | 0 m | consumer GNSS (2 m / 4 m sigma) | reported, not gated |
| 4 | `gnss-only` | 0 px | 0 m | consumer GNSS (2 m / 4 m sigma) | reported, not gated |
| 5 | `combined` | ±0.25 px per axis | 1 cm per axis | consumer GNSS (2 m / 4 m sigma) | 3D total <= 2 cm |

Scenario 1 is the pipeline-exactness check: with exact projections, exact
references and exact camera GPS, a correct pipeline reconstructs the truth to
well under a millimetre. A scenario-1 failure is a defect to fix, not a bound
to loosen. Scenario 4 pins the camera-prior/control interaction: with exact
control and exact projections a correctly weighted adjustment is governed by
the control, so its check-point error must be millimetre-level. Scenario 5 is
the validation gate.

Every report also carries the engine's solve diagnostics under `diagnostics`:
the per-category sigmas actually used (camera centre, marker control, marker
projection, tie point), the iteration count and termination category, the
initial and final robust cost, the last relative cost change, the prior RMS in
sigma units, the gauge state, and the control/check residuals split per axis.
`update_transform` additionally reports how many cameras and control markers
the similarity fit used and the total 1/sigma^2 weight of each class.

### Measured in CI

The test `python_gcp_validation` prints five JSON objects and five
bracket-tagged measurement lines; CI surfaces them even when the test passes.
Numbers are metres; the gates are 2 mm (scenario 1) and 2 cm (scenario 5).

Measured by `python_gcp_validation` on the PR's CI run (macOS arm64, run
42357021231, about 20 s for all five scenarios; Linux x86-64 in the same run
produced the same numbers):

| scenario | horizontal RMS | vertical RMS | 3D total RMS | gate |
| --- | --- | --- | --- | --- |
| 1 `exact` | 0.000135 m | 0.000324 m | 0.000351 m | 2 mm, pass |
| 2 `reference-noise` | 0.001972 m | 0.007280 m | 0.007542 m | not gated |
| 3 `projection-noise` | 0.008216 m | 0.018440 m | 0.020187 m | not gated |
| 4 `gnss-only` | 0.000561 m | 0.000704 m | 0.000900 m | not gated |
| 5 `combined` | 0.008028 m | 0.025646 m | 0.026873 m | 2 cm, **fail** |

### Defect found and fixed

Before the fix these five scenarios were 0.49 mm, 5.10 cm, 5.10 cm, 5.33 cm and
4.87 cm. Scenarios 2 and 3 removed different noise sources yet both sat at
5.1 cm, so the error tracked the noisy camera GNSS, not the injected marker
noise. The diagnostics pinned it: scenario 4 converged in 31 iterations with
the control points satisfied to 0.08 mm, yet its check points re-triangulated
5.2 cm away, almost all vertical, while the reprojection RMS stayed at
0.127 px. The control points were pinned in 3-D, but the camera poses were
being set by the ~7990 tie-point observations — and every observation carried
one shared 1 px weight, because `BAObservation` had no per-observation sigma
(the marker-projection accuracy the chunk declares was never consumed). With
30 marker observations against 7990 tie observations at equal weight, the
poses followed the tie-point cloud and the held-out points drifted by
centimetres in depth.

The fix follows the standard weighted least-squares formulation (Triggs et al.
2000; Foerstner & Wrobel 2016): each observation is whitened by its own 1-sigma
accuracy, so the problem is one weighted system with weights 1/sigma^2. Marker
projections use the chunk's marker-projection accuracy, tie-point observations
its tie-point accuracy, and the robust loss now acts on the whitened residual.
Check-point triangulation uses the same weights. The camera-centre, marker
control and datum weights were already correct (2 m / 4 m per axis, 1 cm, and
the datum fit's 1/sigma^2 weights: 80000 for the control markers against 1.69
for the twelve camera references).

Scenario 4 now recovers to 0.9 mm, which is the direct confirmation that
camera priors and control interact correctly. Scenarios 2 and 3 fall to 7.5 mm
and 20.2 mm.

### Why the gate is still exceeded

Scenario 3 is at its own information limit, not a pipeline limit. Propagating
the injected 0.25 px uniform noise (sigma 0.144 px) through the truth camera
geometry of the three check points gives a linearised triangulation sigma of
1.7-2.5 cm 3D per point (vertical 1.4-2.3 cm), an RMS of about 2.14 cm over the
three; the measured 2.02 cm is at that bound. Scenario 5 combines that with the
datum error from the 1 cm control noise (scenario 2: 0.75 cm), so its expected
RMS is about 2.3 cm, and the three seeded realisations happen to add rather
than cancel (E2 alone is 3.9 cm). The 2 cm gate therefore sits below the
check-point noise floor of this fixture and cannot be met without changing the
noise, the layout or the gate — none of which is done here. The failing gate is
reported as evidence, not hidden.

Fixed in every scenario: 12 images, 12 markers (6 ground, 6 roof), 8 usable
control points (9 imported; one edge target has a single projection), 3 check
points (`E2`, `E3`, `E5`, all on the roof), pinhole intrinsics, and the
fixture's near-planar geometry. GSD is 0.12 m/px (60 m above ground at
f = 500 px).

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

The real-block measurement is **pending a cloud run**: no real aerial block with
surveyed check points is checked into this repository, so this document does not
state a number for it. The synthetic figure above is the committed, reproducible
measurement.
