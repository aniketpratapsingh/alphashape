# Markers, ground control and calibration

Markers are user-measurable points that tie the reconstruction to the ground or
to other chunks. This guide covers the Python surface that WP-13.20 closes out:
placing and detecting markers, importing reference data, weighting control and
check points, fitting the datum, scale bars, camera calibration and film-scan
fiducials.

All names below are methods or properties of `alphashape.Chunk`,
`alphashape.Marker`, `alphashape.Sensor` and `alphashape.Calibration`.

## Marker types

`Marker.type` reports the role recorded on the marker:

- `"regular"` — an ordinary point, the default for a placed marker;
- `"vertex"` — a corner of a printed target;
- `"fiducial"` — the centre of a film fiducial mark.

The reference role is separate and stored in the marker's metadata: a control
point (`"gcp"`) constrains the adjustment, a check point (`"check_point"`) is
held out and only measured afterwards. `Chunk.ground_control` groups the
referenced markers for you.

## Placing and editing markers

`addMarker` creates a marker and returns its proxy. A missing label generates
`"point N"`; the position is an optional chunk-local `(x, y, z)` tuple.

```python
marker = chunk.addMarker(label="GCP 1", position=(12.0, -4.0, 1.5))
marker.reference = (9.20, 48.70, 300.0)   # reference slot, in its own CRS
marker.pinned = True                      # informational, not a constraint
marker.valid = True                       # pass filtering
```

Projections are the image measurements of a marker. Place one with
`setMarkerProjection` and read it back as a `MarkerProjection`:

```python
chunk.setMarkerProjection("GCP 1", camera.label, 1234.5, 987.0)
projection = chunk.markers[0].projections[0]
print(projection.camera.label, projection.pixel, projection.enabled)
```

`removeMarkerProjection(label, camera, delete_marker=True)` removes a marker
with its last projection. `Marker.group` reads and moves the marker between
marker groups (`addMarkerGroup` creates one; assigning `None` detaches it).

## Detecting, refining and tracking

`detectMarkers` runs a target detector over the selected cameras. `target_type`
accepts the coded ring targets (`Circular12`, `Circular14`, `Circular16`,
`Circular20`), the non-coded `Circular`, `Cross`, and coded tag families.
`tolerance` is the error-correction strength (0–100) and `create_markers`
controls whether a detection that matches no existing marker becomes a new one.

```python
chunk.detectMarkers(target_type="Circular12", tolerance=50,
                    cameras=[camera.label], create_markers=True)
chunk.refineMarkers(cameras=[camera.label])     # multi-view consensus
chunk.trackMarkers(cameras=[camera.label], direction="both")
```

`refineMarkers` re-locates each projection from the multi-view consensus and
drops views that disagree; `trackMarkers` follows a projection through the
camera sequence with a pyramidal tracker and can re-detect the target locally
when the track is lost.

## Importing and exporting reference data

`importReference` reads CSV, whitespace text or XML. `columns` gives an
explicit positional mapping when the file has no header; `marker_type` sets the
role of every matched row (`"gcp"`, `"check_point"` or `"marker"`). Existing
markers are matched by label; `create_markers=True` creates the rest.

```python
chunk.importReference(path="gcps.csv", format="csv",
                      columns=["label", "x", "y", "z"],
                      marker_type="gcp", crs="EPSG:4326")
chunk.exportReference(path="reference.csv", columns=["label", "x", "y", "z"],
                      include_cameras=True, include_scalebars=False)
```

`exportReference` writes cameras, markers and scale bars; `columns` (an alias
of the task's `columns_order`) selects and orders the fields, and `precision`
sets the digits after the decimal point. `importMarkers`/`exportMarkers` are the
projection-only codec for moving marker picks between projects.

## Accuracies

Six per-chunk properties weight the adjustment. Each assignment is an undoable
edit:

| property | default | meaning |
| --- | --- | --- |
| `camera_location_accuracy` | 10 m | camera centre reference |
| `camera_rotation_accuracy` | 10 deg | measured attitude prior |
| `marker_location_accuracy` | 5 mm | marker reference position |
| `marker_projection_accuracy` | 0.5 px | marker image measurement |
| `scalebar_accuracy` | 1 mm | scale-bar length |
| `tiepoint_accuracy` | 1 px | tie-point measurement |

Smaller numbers mean more confidence and a stronger pull on the solution.

### Per-photo GNSS accuracy and fix status

RTK drone photos usually carry a per-photo position accuracy in their XMP
packet. Give it to `AddPhotos` and each camera is weighted by its own solution
instead of the 10 m consumer default:

```python
doc.run_task("AddPhotos", filenames=photos, load_xmp_accuracy=True)
camera = chunk.cameras[0]
print(camera.reference_accuracy)   # (east, north, up) metres
print(camera.reference_rtk_flag)   # 50 fixed, 34 float, None if not reported
```

The import rule, in order:

- a fixed (50) or float (34) solution uses the reported `RtkStdLon`,
  `RtkStdLat` and `RtkStdHgt` directly as the east/north/up accuracies;
- any other reported status (single-point GNSS, for example) keeps the
  position but its accuracy is raised to at least **0.5 m per axis**, so a
  metre-level fix cannot outvote centimetre survey control;
- a packet with standard deviations but no status uses them as written;
- a packet with no deviations leaves the accuracy at the chunk default (10 m).
  The fix status is still stored as provenance.

The same per-camera value can be set from a reference file: camera rows import
`acc_x,acc_y,acc_z`, so `importReference` can carry a manually corrected
accuracy. `camera.reference_accuracy` itself is read-only; a bare
`(lon, lat, alt)` reference from EXIF without XMP accuracy keeps the chunk
default.

## Control points and check points

A control point is used by the adjustment; a check point is measured after it,
so its residual measures how well the model predicts unseen ground. Mark at
least three non-collinear control points, and keep some points held out.

```python
chunk.importReference(path="control.csv", marker_type="gcp")
chunk.importReference(path="check.csv", marker_type="check_point")
ground = chunk.ground_control
print(len(ground["control_points"]), len(ground["check_points"]))
```

`updateTransform` fits one weighted similarity from the control markers (and
any referenced cameras) and applies it to the whole chunk. The held-out error
lands in `chunk.meta["update_transform.rms_check_m"]`; a per-marker error is
stored under `update_transform.check_error_<id>_m`.

## Optimizing the block

`optimizeCameras` runs the self-calibrating bundle adjustment. `fit_f`,
`fit_p3`, `fit_p4` and the other `fit_*` flags free or hold individual
coefficients; `adaptive_fitting=True` treats those flags as permission and
automatically frees the higher-order groups whose likelihood gain pays for
their degrees of freedom. `use_gcps` overrides whether marker references act
as control points for this call only: omit it to use the chunk's persistent
`use_gcps` setting (the same value the reference-settings dialog edits), and
the chunk's setting is unchanged when the call returns, including after a
failure. `tiepoint_covariance=True` stores the posterior covariance of every tie
point.

```python
chunk.optimizeCameras(fit_f=True, fit_p3=False, adaptive_fitting=True,
                      use_gcps=True, tiepoint_covariance=False)
```

## Scale bars

A scale bar is a known distance between two endpoints (markers or cameras). Its
`reference` is the measured length in metres.

```python
bar = chunk.addScalebar("A", "B", reference=2.500, accuracy=0.001)
chunk.scalebars[0].reference = 2.501       # undoable edit
```

## Camera calibration from a chessboard

`calibrateCamera` detects an inner-corner chessboard in the selected views and
solves one calibration per sensor. `board_cols`/`board_rows` count inner
corners; `square_size` is the square edge in metres. The fit flags match
`optimizeCameras`, and `report_path`/`report_format` write a calibration report.

```python
chunk.calibrateCamera(board_cols=9, board_rows=6, square_size=0.025,
                      report_path="calibration.html")
sensor = chunk.sensors[0]
print(sensor.calibration.f, sensor.calibration.k1,
      sensor.calibration.covariance_matrix, sensor.fixed_params)
```

`Calibration.p3`/`p4` are the higher-order tangential terms and
`Calibration.covariance_matrix` is the posterior covariance (a list of rows, or
`None` before a covariance-producing fit). `Calibration.save`/`Calibration.load`
round trip the model through the selected `CalibrationFormat`. A sensor that
shares a calibration with a master reports the master index in `Sensor.master`;
`Sensor.makeMaster()` detaches it through the undoable command.

## Film scans and fiducials

For scanned analogue film, `detectFiducials` finds the frame marks and fits
each scan's film-millimetre to pixel transform. `fiducial_type` selects the
detector (`"circle"`, `"corner"` or `"frame_corner"`).

```python
chunk.detectFiducials(fiducial_type="corner")
chunk.calibrateFiducials()          # film coordinates from one reference scan
```

## Distortion maps

`exportSTMap` writes a 32-bit float map that resamples an image into its
undistorted geometry. `sensor` selects the sensor and `width`/`height` override
the calibration size.

```python
chunk.exportSTMap(path="undistort.tif", sensor=0, width=0, height=0)
```


