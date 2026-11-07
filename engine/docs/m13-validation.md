# M13 validation — markers, GCP and calibration

The M13 close-out is validated by one reproducible synthetic scenario and by a
real-block procedure. The script is `engine/tests/run_gcp_validation.py`; it
prints one JSON object on stdout and exits non-zero when the held-out
check-point horizontal or vertical RMS exceeds the tolerance (2 cm by default).
The 3D total is the quadrature of the two and is reported alongside.

```
python3 engine/tests/run_gcp_validation.py --synthetic
```

## Synthetic GCP block

The scenario loads the 12-image nadir fixture (`engine/data/aerial_r*c*.jpg`)
whose truth poses, ground grid and building are recorded in
`engine/data/aerial_truth.json`. It then:

1. imports the images, runs `MatchPhotos` and `AlignCameras` (the fixture EXIF
   carries exact GPS), and declares the chunk WGS 84 geographic;
2. creates 12 markers: 6 on the ground (z = 0) and 6 on the building roof
   (z = 9.52 m), with NO seeded 3-D position;
3. adds each marker's projections by projecting its truth position through the
   fixture's truth camera pose, then offsets every projection by seeded uniform
   noise of 0.5 px peak to peak (±0.25 px per axis);
4. sets the marker reference accuracy to the 1 cm the survey noise actually
   has, so the control points are not treated as exact;
5. imports 8 usable references as control points (truth + deterministic 1 cm
   per-axis noise) and 3 as check points (held out of the fit; the three
   best-observed roof targets, since the fixture's edge targets have only 1-3
   views);
6. runs `optimizeCameras` with the control references constraining and the
   pinhole intrinsics held fixed (the fixture is a distortion-free render), so
   every marker position is triangulated from its image measurements only, then
   `updateTransform` (the weighted similarity fit) refits the datum;
7. measures every check point against its TRUE position (not its reference)
   and reports the horizontal, vertical and total RMS.

The chunk is reconstructed by a real alignment and georeferenced from the
fixture GPS, so the bundle adjustment and the similarity fit both have real
work to do, and the check-point errors are independent of the noisy control
references.

### Measured in CI

The test `python_gcp_validation` prints one JSON object and a bracket-tagged
measurement line; CI surfaces the line even when the test passes. Numbers are
metres; the bound is 2 cm.

Measured by `python_gcp_validation` on the PR's CI run (Linux x86-64,
run 42335619894, about 28 s including the alignment):

| quantity | value |
| --- | --- |
| images | 12 |
| markers | 12 (6 ground, 6 roof) |
| control points | 8 usable (9 imported; one edge target has a single projection) |
| check points | 3 (`E2`, `E3`, `E5`, all on the roof) |
| markers detected automatically | 0 (the fixture has no printed targets; the projections are the truth projections plus seeded noise) |
| projection noise | 0.5 px peak to peak |
| georeferenced | yes (`AlignCameras` from exact GPS, then `updateTransform`) |
| horizontal check-point RMS | 0.009053 m |
| vertical check-point RMS | 0.018104 m |
| total check-point RMS (quadrature) | 0.020241 m |
| engine `update_transform.rms_check_m` | 0.020241 m |
| worst camera centre vs truth | 0.0747 m |
| truth reprojection through the final poses | 0.193 px RMS, 0.323 px max |
| passed the 2 cm per-axis bound | yes |

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
marker's stored reference with the same per-axis reconstruction.

`--crs` must name a geographic (lon/lat) CRS. The script converts references to
ECEF as lon/lat, so a projected CRS is refused with an error rather than
silently misinterpreting the coordinates. A check label with no marker or no
computed position also exits non-zero, so a missing check point cannot be
mistaken for a perfect one.

The real-block measurement is **pending a cloud run**: no real aerial block with
surveyed check points is checked into this repository, so this document does not
state a number for it. The synthetic figure above is the committed, reproducible
measurement.
