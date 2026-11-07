# M13 validation — markers, GCP and calibration

The M13 close-out is validated by four reproducible synthetic scenarios and by
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

`--ablations` runs the same block four times, so a pipeline defect can be told
from noise-limited accuracy. Scenario 1 is the pipeline-exactness control: the
camera-GNSS references are off and the fixture's exact GPS position and
attitude are pinned, so every input the marker/GCP machinery consumes is exact.
Scenarios 2-4 keep the realistic consumer-GNSS camera references (attitude from
the alignment) and vary only the two noise knobs; their differences are
therefore attributable to those noises.

| # | name | projection noise | reference noise | camera references | gate |
| --- | --- | --- | --- | --- | --- |
| 1 | `exact` | 0 px | 0 m | exact position + attitude | 3D total <= 2 mm |
| 2 | `reference-noise` | 0 px | 1 cm per axis | consumer GNSS (2 m / 4 m sigma) | reported, not gated |
| 3 | `projection-noise` | ±0.25 px per axis | 0 m | consumer GNSS (2 m / 4 m sigma) | reported, not gated |
| 4 | `combined` | ±0.25 px per axis | 1 cm per axis | consumer GNSS (2 m / 4 m sigma) | 3D total <= 2 cm |

Scenario 1 is the pipeline-exactness check: with exact projections, exact
references and exact camera GPS, a correct pipeline reconstructs the truth to
well under a millimetre. A scenario-1 failure is a defect to fix, not a bound
to loosen. Scenario 4 is the validation gate.

### Measured in CI

The test `python_gcp_validation` prints four JSON objects and four
bracket-tagged measurement lines; CI surfaces them even when the test passes.
Numbers are metres; the gates are 2 mm (scenario 1) and 2 cm (scenario 4).

Measured by `python_gcp_validation` on the PR's CI run (macOS arm64, run
42349669604, about 15 s for all four scenarios; the same numbers were produced
by run 42347626176 on Linux and macOS):

| scenario | horizontal RMS | vertical RMS | 3D total RMS | gate |
| --- | --- | --- | --- | --- |
| 1 `exact` | 0.000146 m | 0.000469 m | 0.000491 m | 2 mm, pass |
| 2 `reference-noise` | 0.012760 m | 0.049357 m | 0.050980 m | not gated |
| 3 `projection-noise` | 0.020654 m | 0.046611 m | 0.050982 m | not gated |
| 4 `combined` | 0.020175 m | 0.044273 m | 0.048653 m | 2 cm, **fail** |

Scenario 1 passes by a wide margin: with every input exact, the marker/GCP
machinery reproduces the truth to half a millimetre, so there is no pipeline
defect in the marker, reference or bundle-adjustment path.

Scenarios 2-4 all sit at about 5 cm, and removing either injected noise does
not reduce that: scenario 2 (no projection noise) and scenario 3 (no reference
noise) have the same total as scenario 4. The common limit is therefore the
consumer-GNSS camera references, not the marker noise: with 2 m / 4 m camera
positions and a near-planar 12-image nadir block, the vertical scale and the
camera depths are only weakly determined, and the bundle adjustment can trade a
few centimetres of camera depth against a sub-pixel reprojection residual even
with the exact control references of scenario 3. The scenario-4 gate is left
failing as specified; these four numbers are the evidence for the coordinator's
decision. Raising the fixture's parallax or adding vertical control would be a
change to the scenario, not a fix to the pipeline, so neither is done here.

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
