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
   noise of 0.5 px peak to peak (±0.25 px per axis);
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

`--ablations` runs the same block four times with the noise sources switched
independently, so a pipeline defect can be told from noise-limited accuracy.
Scenarios 1-3 disable the camera-GNSS references, keep the fixture's exact GPS
and pin its exact attitude, so only the stated image/reference noise source is
present; the SfM tie-point network's own noise is not part of those ablations.
Scenario 4 is the realistic pipeline: the attitude comes from the alignment and
the camera positions from the noisy GNSS.

| # | name | projection noise | reference noise | camera references | gate |
| --- | --- | --- | --- | --- | --- |
| 1 | `exact` | 0 px | 0 m | exact position + attitude | 3D total <= 2 mm |
| 2 | `reference-noise` | 0 px | 1 cm per axis | exact position + attitude | reported, not gated |
| 3 | `projection-noise` | ±0.25 px per axis | 0 m | exact position + attitude | reported, not gated |
| 4 | `combined` | ±0.25 px per axis | 1 cm per axis | consumer GNSS (2 m / 4 m sigma), attitude from the alignment | 3D total <= 2 cm |

Scenario 1 is the pipeline-exactness check: with exact projections, exact
references and exact camera GPS, a correct pipeline reconstructs the truth to
well under a millimetre. A scenario-1 failure is a defect to fix, not a bound
to loosen. Scenario 4 is the validation gate.

### Measured in CI

The test `python_gcp_validation` prints four JSON objects and four
bracket-tagged measurement lines; CI surfaces them even when the test passes.
Numbers are metres; the gates are 2 mm (scenario 1) and 2 cm (scenario 4).

Measured by `python_gcp_validation` on the PR's CI run (Linux x86-64,
run RUN_ID_PLACEHOLDER, about TIME_PLACEHOLDER):

| scenario | horizontal RMS | vertical RMS | 3D total RMS |
| --- | --- | --- | --- |
| 1 `exact` | H1_PLACEHOLDER | V1_PLACEHOLDER | T1_PLACEHOLDER |
| 2 `reference-noise` | H2_PLACEHOLDER | V2_PLACEHOLDER | T2_PLACEHOLDER |
| 3 `projection-noise` | H3_PLACEHOLDER | V3_PLACEHOLDER | T3_PLACEHOLDER |
| 4 `combined` | H4_PLACEHOLDER | V4_PLACEHOLDER | T4_PLACEHOLDER |

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
