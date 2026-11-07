# M13 validation — markers, GCP and calibration

The M13 close-out is validated by one reproducible synthetic scenario and by a
real-block procedure. The script is `engine/tests/run_gcp_validation.py`; it
prints one JSON object on stdout and exits non-zero when the held-out
check-point RMS exceeds the tolerance (2 cm by default).

```
python3 engine/tests/run_gcp_validation.py --synthetic
```

## Synthetic GCP block

The scenario loads the 12-image nadir fixture (`engine/data/aerial_r*c*.jpg`)
whose truth poses, ground grid and building are recorded in
`engine/data/aerial_truth.json`. It then:

1. imports the images and declares the chunk WGS 84 geographic;
2. runs `detectMarkers` on the first frame;
3. places 12 markers: 6 on the ground (z = 0) and 6 on the building roof
   (z = 9.52 m);
4. adds their projections by projecting the truth pose through each camera;
5. perturbs each reference by deterministic 1 cm per-axis noise, imports 9 as
   control points and 3 as check points;
6. runs `refineMarkers`, then `updateTransform` (the weighted similarity fit);
7. reports the horizontal, vertical and total check-point RMS.

The chunk starts in an arbitrary local frame (scale 0.5, 25 degree rotation,
a translation), so the similarity fit has to recover the datum rather than
confirm an identity.

### Measured in CI

The test `python_gcp_validation` prints one JSON object and a bracket-tagged
measurement line; CI surfaces the line even when the test passes. Numbers are
metres; the bound is 2 cm.

Measured by `python_gcp_validation` on the PR's CI run (Linux x86-64,
run 42314821252, test 1966, 1.35 s):

| quantity | value |
| --- | --- |
| images | 12 |
| markers after detection + placement | 12 |
| markers detected automatically | 0 (the fixture has no printed targets; the projections come from the truth poses) |
| control points | 9 |
| check points | 3 |
| georeferenced by `updateTransform` | yes |
| horizontal check-point RMS | 0.012331 m |
| vertical check-point RMS | 0.008786 m |
| total check-point RMS | 0.015141 m |
| engine `update_transform.rms_check_m` | 0.0151408 m |
| passed the 2 cm bound | yes |

## Real block

```
python3 engine/tests/run_gcp_validation.py \
    --real /path/to/aerial-block \
    --gcp /path/to/gcps.csv \
    --crs 4326 \
    --check-points GCP17,GCP23,GCP31
```

The procedure is: import every image, declare the reference CRS, run
`MatchPhotos` and `AlignCameras`, detect markers, import the CSV as control
points and the designated labels as check points, run `optimizeCameras`, and
report the check-point RMS with the same per-axis reconstruction.

The real-block measurement is **pending a cloud run**: no real aerial block with
surveyed check points is checked into this repository, so this document does not
state a number for it. The synthetic figure above is the committed, reproducible
measurement.
