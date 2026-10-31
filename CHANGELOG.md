# Changelog

All notable changes to alphashape are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation

- **WP-12.12: M12 coordinate-systems close-out.** New `engine/docs/crs-guide.md`,
  expanded CRS limits in `engine/LIMITATIONS.md` and CRS test notes in
  `engine/TESTING.md`, plus an opt-in `alphashape_crs_fixtures` CMake target that
  regenerates the CRS fixtures from their committed generator scripts.

### Changed

- **`BuildOrthomosaic`'s `color_correction` is now a choice, not a Bool, and
  defaults to `"affine"`.** The accepted values are `"none"`, `"gain"` and
  `"affine"`, applied per camera per channel before seams are chosen. Affine is
  the default because the seam benchmark shows a clear improvement (seam colour
  step p90 0.0241 → 0.0214, −11%); pass `color_correction="none"` to get the
  old uncorrected product. A model that cannot be solved (one camera, no
  overlap, too few samples) logs and falls back to identity, so a single-camera
  orthomosaic still works. An old project file or script that recorded `true`
  still works: the value is read as `"gain"`, which is exactly what the Bool
  selected, and `false` reads as `"none"`. The Python binder likewise accepts a
  Bool and converts it to its text form.

### Added

- **Multi-view consensus ghosting filter for BuildOrthomosaic (Q4).** New
  `ghosting_filter` parameter (Bool, default `false` while the quality
  benchmark decides the shipping default). Moving objects — cars, people,
  transient shadows — seen by only a minority of overlapping views are found
  before the seams are chosen: at three scales (1x, 4x, 16x) each cell is
  compared with the per-channel median of the cameras valid there, and a camera
  further than a robust (MAD) distance plus an absolute grey floor has its cells
  marked deviant. Per tile the mask is closed then opened and small components
  are dropped. The seam labeller pays an extra `avoid_cost` on a deviant cell
  (larger than one smoothness edge, and never charged when the camera is the
  only valid view, so no hole appears), average blending drops deviant cameras
  from the mean, and small label islands are relabelled afterwards. The
  formulation follows robust multi-view deghosting and moving-object removal in
  mosaics (Jacobs, Loscos & Ward 2008; Gallagher 2005; Davis 1998; Uyttendaele,
  Eden & Szeliski 2001). New `engine/src/common/algorithm/ortho/ghosting.*`;
  the masks add 1 byte per tile footprint cell to the block memory estimate and
  an INFO line reports marked cells and removed speckles.

- **Per-camera, per-channel affine colour correction for the orthomosaic.**
  `BuildOrthomosaic`'s `color_correction="affine"` fits `I' = a_c I + b_c` per
  camera per channel, jointly over all cameras, from observations of the SAME
  3-D surface point seen by several cameras: a coarse lattice of DEM cells is
  projected into every camera with the rectifier's own occlusion test and each
  point's colour is read bilinearly from that photograph. The residual is the
  disagreement of the two corrected colours and the fit uses an IRLS Cauchy loss
  (Cauchy 1824), so a moving object, a specular highlight or an occlusion — each a
  large residual on a minority of points — cannot drag the result; a weak prior
  pulls `a` toward 1 and `b` toward 0. The model is gauged to mean gain 1 and mean
  offset 0 and clamped only AFTER that, and it is applied to each camera's
  rectified pixels BEFORE the seam labeller runs, so seams are chosen on corrected
  colours and a block-wise mosaic applies the same whole-flight model in every
  block. The form follows joint radiometric alignment in the published literature
  (Goldman & Chen 2005; Xiong & Pulli 2010) on top of the gain model Brown & Lowe
  (2007) give. New `engine/src/common/algorithm/ortho/colour_model.*`; the
  visibility test it reuses is exposed as `ortho::project_surface_point()`.

- **Post-filter depth-map hole repair.** `BuildDepthMaps` gains `repair_holes`
  (bool, default `false`) and `repair_max_area` (int pixels, default `2000`).
  When enabled, after the cross-view consistency filter each depth map's
  invalid 4-connected components are examined: a component no larger than
  `repair_max_area`, with at least 75% of its perimeter on measured pixels, is
  refilled from a robust plane fitted to that boundary in camera space (RANSAC,
  then a least-squares refit on the inliers) when at least 70% of the boundary
  points lie within 1% of the boundary's mean depth of the same plane, that
  fraction being re-checked against the refitted plane. The ring is also checked
  for coherence: a step whose far side owns only a contiguous arc of the ring
  (more than 10%, or whose outliers sit one-sided on the plane) is rejected,
  because the fraction alone would let the near plane bridge it. Each hole pixel
  is filled by intersecting its viewing ray with the plane and receives the
  plane normal and a `kDepthRepairedFlag` provenance bit (`flags`, depth-map
  sidecar ADM3), so later stages can tell a repaired pixel from a measured one;
  the repaired confidence is `0`, because the filter can keep a measured pixel
  at `0` too, so a value could never carry that distinction. A repaired sample
  never outvotes a measured one: BuildDem's per-cell median ignores repaired
  samples where a cell has any measured sample (they still fill a cell nothing
  measured), and point-cloud fusion weights each repaired sample at a fraction
  of a measured sample's. The pass does nothing when `filter_mode` is
  "disabled": those holes are genuine no-data, not filter artefacts. The fill is
  bounded: a depth
  outside the range the hole's own ring measured (plus 1%), or an intersection
  at less than ~3 degrees to the plane, is left empty, so a near-edge-on plane
  cannot write a depth far outside the surface. A hole straddling a depth
  discontinuity, an oversized hole, or one cut off by the map border is left
  empty. The two parameters are folded into the depth-map fingerprint only when
  repair actually runs, so existing maps stay reusable. The pass runs on the
  filtered map in both the CPU and GPU depth paths.

- **Project format v4 and depth-map sidecar ADM3.** The depth-map sidecar gains
  an optional per-pixel provenance byte (`flags`: bit 0 = repaired by the pass
  above) and moves ADM2 -> ADM3. ADM1 and ADM2 are still read, and an ADM2 map
  decodes with every pixel measured, so no migration is needed. The doc.xml
  version is bumped to 4 only so an older build reports "newer than this build
  supports" instead of a header error.

- **GeoTIFF GeoKeys carry a full CRS.** The GeoTIFF writer now records the
  chunk's `crs::CoordinateSystem`, not just an EPSG number: an authority-coded
  CRS is written as its type key plus a name citation, while a custom CRS with
  no EPSG code is written as user-defined (32767) with its complete WKT2
  definition in `GeoAsciiParamsTag`/`PCSCitationGeoKey` (GDAL's convention).
  A vertical, geocentric or compound CRS (and any code above 65535, which the
  16-bit type key cannot hold) is never written as a horizontal code: it is
  user-defined 32767 with its full WKT2 citation. The reader prefers a real
  projected code, then a real geographic one, then citation WKT (PCS, then
  geographic, then `GTCitationGeoKey`, accepting GDAL's and ArcGIS's
  `ESRI PE String = ` prefix), then the local frame. The DEM and orthomosaic
  sidecars and the export tasks pass the asset's CRS through and restore it on
  load, so a custom-WKT grid round-trips through a GeoTIFF; a malformed stored
  WKT now fails the export/save with a message naming it instead of writing a
  silently local file. `GeoTiffOptions` keeps `epsg` as the fallback when no
  `CoordinateSystem` is supplied, and `GeoTiffInfo` gains the decoded CRS
  beside the existing `epsg`.

- **Python `CoordinateSystem` and `Chunk.crs` (WP-12.11).** The bindings expose
  the CRS engine as a value type: `CoordinateSystem(epsg)` / `CoordinateSystem()`
  for the local frame, `CoordinateSystem(wkt=..., proj=..., name=...)`,
  `CoordinateSystem.init(name)` over the built-in list, and the
  `name` / `authority` / `epsg` (None when absent) / `wkt` properties plus
  `toWKT()` and `toProj4()`. `__eq__` is PROJ equivalence rather than string
  identity, so two definitions of one CRS compare equal, and `repr` is
  `<CoordinateSystem 'WGS 84 / UTM zone 33N'>`. `listBuiltinCRS()` returns the
  curated 48-name list; `addGeoid(path)` / `listGeoids()` expose the process
  geoid registry (a custom grid is referenced where it is, never copied or
  downloaded). `Chunk.crs` reads and writes the engine CRS; assignment runs
  the same "Set Chunk CRS" command the GUI uses and then discards it (the GUI
  owns the document's undo history, and Metashape's Python API has no undo).
  It only relabels the chunk — no reference, pose, tie point or asset is
  transformed — and assigning an unset `CoordinateSystem()` clears the chunk
  to LOCAL. The legacy `crs_epsg` setter routes through that same command, and
  `__hash__` is disabled because `__eq__` is PROJ equivalence. There is no `datumTransform()`: the engine lists candidate
  operations per CRS pair (`available_transformations`) and pins one with
  `with_transformation`, but no single "preferred" operation exists to report.
  `BuildDem`'s `projection` (EPSG int) and `crs` (text) parameters are covered
  by the new `python_crs` test.

- **Raster reprojection and CRS-aware DEM/ortho export.** `ExportDem` and
  `ExportOrthomosaic` gain `projection` (Int EPSG code), `crs` (String: an
  `EPSG:<code>`, a PROJ string or WKT; setting both to disagreeing values is a
  `ValueError`) and `resolution` (Float in TARGET-CRS units, 0 keeps the ground
  sample distance). A target CRS reprojects the raster on the fly and the
  GeoTIFF's georeferencing and world file are written in that CRS. The new
  `model/raster/reproject.{hpp,cpp}` drives the mapping from the TARGET grid:
  the extent comes from a densified source boundary (21 points per edge, so a
  bowed projected edge is not clipped), the default cell is the source ground
  sample distance measured at the raster centre, and every target cell centre is
  inverse-transformed back to the source. The transform is evaluated exactly only
  on a coarse control lattice (16 px), bilinearly interpolated in between, and
  the spacing is halved until a midpoint check against exact transforms agrees to
  0.1 source pixels — else the exact per-pixel mapping is used. Float rasters
  (DEMs) resample bilinearly and nodata-aware: tags that are out of bounds, the
  `nodata` sentinel or non-finite carry no weight, the remaining weights
  renormalise, and a cell whose valid weight is under one half is written as
  nodata, so a hole stays a hole and never becomes a blended value. 8-bit rasters
  (orthomosaics) always resample nearest, because a blended colour was never
  photographed. An identical CRS is a bit-identical copy with no PROJ call; a
  same-CRS request at a different `resolution` keeps the source's north-west
  corner rather than snapping the origin to a multiple of the cell, so a real
  georeference is never shifted and no blank border row appears. A geographic
  target takes `resolution` in DEGREES, and a value that is plainly metres (a
  hundredth of a degree or more against a metre-scale source) is refused with
  the degree value to pass instead; a request that would leave a grid smaller
  than 2x2 cells is refused too. Float rasters keep every channel, and a lattice
  cell whose corner a refused transform poisons falls back to the exact
  per-pixel mapping instead of blanking all 16x16 pixels. Rows are produced in
  bands and the output is checked against the machine's memory budget before it
  is allocated. `BuildOrthomosaic` with a `crs` different from
  the DEM's now reprojects the DEM surface once and builds directly in that CRS,
  so the photographs are sampled in the requested projection exactly once
  (previously the mosaic always landed on the DEM's plane and the request was
  only honoured at export). With `surface_data="model"` the DSM is rasterised
  straight into the requested CRS, so no DEM reprojection is paid for; the
  elevation case checks the source DEM plus its reprojected copy against the
  memory budget before allocating. `BuildDem` already built directly in the
  requested CRS.

- **Per-camera gain compensation for the orthomosaic.** `BuildOrthomosaic` gains
  a `color_correction` parameter, now selected as `"gain"` (affine is the
  default; see Changed). When set, one brightness gain
  per camera per channel is solved over the whole flight (Brown & Lowe 2007,
  "Automatic Panoramic Image Stitching using Invariant Features", section 6):
  each gain minimises a sum over overlapping camera pairs of a data term that
  drives overlapping photographs to agree and a prior that keeps a gain near 1,
  giving a small symmetric positive-definite system solved by Cholesky per
  channel. The gains are estimated ONCE on a coarse whole-grid lattice before any
  block is processed, so a block-wise mosaic applies exactly the same multiplier
  as a whole-grid one and no block border disagrees. Gains are solved and applied
  on stored (sRGB-encoded) values — the pipeline never decodes, and the defect
  being removed is a step in the stored mosaic — so no colour-space conversion is
  introduced. New `engine/src/common/algorithm/ortho/gain.*`.

- **Orthomosaic colour-consistency metrics (bench).** `tools/bench/metrics.py`
  gains two metrics that measure the brightness steps between photographs in an
  orthomosaic, computed without any engine change. `seam_step` is a
  reference-free measure: the GeoTIFF export carries no seam labels, so it scans
  every interior row/column boundary for a coherent brightness step between two
  flat 3-cell windows and reports the median step (and its maximum) as a fraction
  of full scale. `colour_error` is reference-based: after a single global
  per-channel affine fit against the published reference orthomosaic — which
  absorbs a white-balance difference — it reports the median and p90 local
  residual, so a spatially varying gain error is what shows up. Both are reported
  in the metrics JSON and the report and are deliberately not yet gated. A third
  dispatch input, `ortho_params` beside `depth_params` and `dem_params`, lets a
  `BuildOrthomosaic` configuration be measured against a control run.

- **Reference data conversion and rotation orders.** `convert_reference`
  (`common/algorithm/reference/convert_reference.hpp`) rewrites a chunk's camera
  and/or marker reference positions into a target CRS and relabels the
  `camera_crs`/`marker_crs` slot, so an imported lon/lat/alt campaign can be
  expressed in a UTM zone (or the reverse) without touching poses, tie points or
  assets. Horizontal positions go through `CrsTransform`; heights go through
  `VerticalTransform` whenever either slot carries a vertical datum. The route
  follows the datum, not the store: an EPSG vertical CRS always uses PROJ's
  compound pipeline (exact `cs2cs` agreement, and a store grid never shadows
  it), while a datum with no EPSG code uses the store's grid and applies
  `H = h - N`, evaluating N in that grid's own horizontal reference (its
  `horizontal_epsg`, persisted in doc.xml; EPSG:4979 when unset). An absent
  height stays absent. Accuracies are measurement weights in metres and are
  carried unchanged. The conversion is
  all-or-nothing (a point outside the target's area of use fails the call and
  leaves the chunk untouched) and undoable as one `Convert Camera/Marker
  References` step.

  `model/crs/euler.hpp` adds the six Tait-Bryan rotation orders (`XYZ` ...
  `ZYX`, default `ZYX`), intrinsic and active with degrees at the API and the
  `yaw`/`pitch`/`roll` fields taken as the first/second/third axis of the order.
  The chunk meta key `reference/rotation_order` selects the order; an unknown
  value is reported rather than defaulted. Gimbal lock is defined and tested:
  the middle angle stays at ±90°, `roll` becomes 0 and the whole remaining
  rotation is carried by `yaw`. `convert_orientation` re-expresses a
  camera-to-world orientation in the target CRS's local axes, so a camera
  heading true north in geographic reads the meridian convergence
  (`γ = atan(tan Δλ · sin φ)`) as its grid yaw.

- **Local tangent frames and per-object CRS slots.** A `LocalFrame`
  (`model/crs/local_frame.hpp`) anchors chunk-local Cartesian coordinates in an
  ECEF point and an explicit local-to-ECEF rotation, then converts to and from
  any CRS in `crs::CoordinateSystem`: local -> ECEF, ECEF -> geodetic by an
  iterated Bowring inversion, geodetic -> target by PROJ (a geocentric target
  needs no conversion). The conversion is exact, not a small-angle
  approximation: it is done in doubles with the geodetic leg iterated, and a
  point within 500 km of the anchor round-trips inside a micrometre. PROJ's own
  `+proj=cart` inverse is deliberately not used there -- it loses millimetres at
  ECEF magnitudes. `transform()` gives the 4x4 affine form -- exact for a
  geocentric CRS, the anchor linearisation for a curved one -- and
  `transformationMatrix()` its inverse.

  A `Chunk` now carries three independent per-object CRS slots:
  `camera_crs` (camera references), `marker_crs` (marker references) and
  `world_crs` (chunk geometry as displayed or exported). Each follows
  `chunk.crs` until set explicitly, and setting one only relabels which CRS the
  numbers are read in -- no coordinate is ever transformed behind your back.
  The slots are written to doc.xml as separate `<camera_crs>`, `<marker_crs>`
  and `<world_crs>` elements, and only when set, so a project without them
  loads and re-saves byte-identically; the reader treats absent elements as
  "follow the chunk CRS". Georeferencing now sets `camera_crs` and `world_crs`
  to the WGS 84 coordinate system it fits to. Converting the data itself to a
  target CRS is a separate, explicit step (WP-12.7).

- **Geoid grids and vertical datums.** A project can carry geoid models —
  EGM96 and EGM2008 when `proj-data` has installed them, and any custom grid a
  user adds — and convert between ellipsoidal and orthometric heights. A
  geoid grid is evaluated through PROJ's `+proj=vgridshift` step, which is the
  same code `cs2cs EPSG:4979 EPSG:5773` runs, with a direct GeoTIFF read plus
  Catmull–Rom bicubic interpolation as the fallback; the two agree at the grid
  nodes, and the off-node difference is the grid's own curvature because PROJ
  interpolates bilinearly (see `engine/LIMITATIONS.md`). Outside a grid's
  coverage every entry point returns "no value" rather than extrapolating, and
  nothing is downloaded. `GeoidStore` writes a project's grids and their index
  to `<project>.files/geoids/`, and a project that names a geoid it does not
  carry fails to load, naming it, instead of silently treating its orthometric
  heights as ellipsoidal. Heights convert as `H = h − N`; when a vertical datum
  is EPSG-defined, PROJ builds the whole compound pipeline
  (`EPSG:4979` → `EPSG:32633+5773`), and a custom grid is applied after the
  horizontal transform. The vertical datum persists in doc.xml as a
  `<vertical>` child of `<crs>`, written only when one is set. Accuracy is
  checked against NGS's published USGG2012 geoid heights at 50 CONUS points
  (EGM2008 RMSE 0.026 m, EGM96 0.284 m), skipping when the full grid is not
  installed.
- **Exact natural-neighbour hole filling in Build DEM.** Holes bordered by at
  most `sibson_exact_max_sites` data cells (new `BuildDem` parameter; off by
  default until the quality benchmark confirms it, 20 000 is a sensible value)
  can now be filled with exact Sibson
  interpolation: the bordering cells are triangulated (incremental Delaunay
  with exact orientation and in-circle predicates), and each hole cell takes
  the area-weighted average of its natural neighbours. Planes are reproduced
  exactly, so the small step the grid approximation could leave at the edge of
  a hole is gone. Larger holes and the orthomosaic keep the grid approximation.

- **Datum transformations can be listed, chosen and pinned.** A coordinate
  system now reports every datum shift PROJ knows to a target CRS —
  `CoordinateSystem::available_transformations()` — with each operation's
  authority code, stated accuracy, ballpark flag, grids and pipeline text,
  ordered by increasing accuracy with unknown accuracy last. That order is
  accuracy-sorted, not PROJ's own per-point pick: no single entry is "the one
  PROJ would use", because PROJ chooses per point by area of use.
  `CoordinateSystem::with_transformation()` turns one of those entries into a
  CRS bound to the operation's hub (a WKT2 `BOUNDCRS`), so a transform built
  from it applies exactly that shift instead of re-deriving one: that is how a
  caller says "use the 7-parameter Helmert" or "use the NTv2 grid" and gets it.
  A missing grid is never downloaded and never silently downgraded to a
  ballpark: the operation stays in the list with the grid named under
  `missing_grids`, and applying it throws
  `transformation needs grid <name>; install proj-data`, which also replaces
  PROJ's opaque "File not found" on the plain `CrsTransform::create()` path.
  `+towgs84` in a CRS definition is honoured by PROJ and needs no special
  handling. The Helmert shifts are checked against the worked examples and
  formulas of IOGP Publication 373-7-2 to a millimetre; the grid cases skip,
  naming the grid, where it is not installed. The macOS and Windows packages
  still ship only a subset of the PROJ grids, so some national shifts need
  `proj-data` installed (see `engine/LIMITATIONS.md`).

- **`BuildDepthMaps` exposes `best_k` and `min_texture_stddev`.** Two
  `PatchMatchOptions` fields that were previously fixed are now declared task
  parameters. `best_k` (Int, default 0) sets how many per-neighbour costs the
  multi-view cost averages: 0 keeps the historical `ceil(n/2)`, 1 trusts the
  single best neighbour. `min_texture_stddev` (Float, default 1/255) is the
  intensity standard deviation below which a window is not matched and so keeps
  no depth; lowering it matches windows that were previously skipped. Both are
  honoured by the CPU and GPU search paths. Each is folded into a depth map's
  fingerprint only when it differs from its default, so existing maps stay
  reusable; out-of-range values are rejected before the chunk is touched.

### Changed

- **`BuildDem` `interpolation="enabled"` now fills the area the cameras
  cover.** It used to fill only holes fully enclosed by data, which on a real
  flight is almost none of them: the nodata outside the photos' cover surrounds
  the surveyed block, so nearly every genuine hole inside it is connected to
  the raster border through invalid cells. The mode now fills holes whose cells
  lie inside the union of the enabled cameras' ground footprints (each camera's
  image border projected onto the DEM's surface plane, convex hulled per camera
  and rasterised -- NOT the convex hull of all cameras, so a gap between
  adjacent flight strips stays nodata). Cells outside every footprint stay
  nodata. `"extrapolated"` (fill everything) and `"disabled"` (fill nothing)
  are unchanged. If no footprint can be computed the builder logs a warning and
  falls back to the old interior-only rule. A georeferenced aerial DEM built
  with `enabled` therefore has fewer nodata cells, including across gaps in the
  observation raster that used to survive only because they touched the outer
  rim. The new `max_fill_distance` parameter (Float, metres, default 0 =
  unlimited) caps how far a filled cell may sit from the nearest measured cell,
  for every fill method; filled cells sag with distance (see
  `engine/LIMITATIONS.md`), so `max_fill_distance=10` suits a measurement-grade
  DEM while the orthomosaic keeps the uncapped default.

- **`BuildDepthMaps` now defaults to `filter_mode="moderate"` and
  `max_neighbors=8`.** Both changes come from the aerial-nadir quality
  benchmark (CPU, 30 images, downscale 8) and cost nothing:
  - `moderate` filtering against `mild` left **21% less hole area**, with DSM
    NMAD **0.653 → 0.574 m** and p95 error **2.51 → 2.12 m**, at no measurable
    time cost.
  - `max_neighbors=8` against `16` made the depth stage **30-40% faster**, with
    coverage, hole area and DSM accuracy equal or slightly better — a wider
    neighbourhood is search time that buys nothing on typical blocks.

  A project whose maps were built with the old defaults is rebuilt on the next
  run with defaults: `max_neighbors` is part of the depth-map fingerprint and
  the filter mode is part of a map's identity, so the stored maps do not
  describe the requested settings. That is the intended behaviour, and the task
  now logs a per-camera reason ("built from different settings, alignment or
  neighbourhood", "filtered with a different mode", …) plus the run's neighbour
  and reuse counts. `filter_mode="mild"` and `max_neighbors=16` restore the
  previous behaviour exactly; every mode and the whole `[1, 256)` neighbour
  range remain legal.
- **A pair on aligned datums no longer needs `allow_ballpark`.** An unpinned
  transform is PROJ's own pipeline, and PROJ used to be asked for a pipeline
  that PROJ itself described as a ballpark for pairs such as WGS 84 ↔ ETRS89,
  NZGD2000, GDA2020, SIRGAS 2000, NAD83(2011), OSGB36 and NTF, so `create()`
  refused them even though the real EPSG operation exists (a null shift, a
  Helmert or a grid shift). The pipeline is now built bare, so those 38 pairs
  are accepted and their numbers come from the EPSG operation PROJ picks per
  point rather than a `+proj=noop` fallback; `accuracy_m()` reports the best
  stated accuracy among the pair's real operations (1 m for ETRS89 ↔ WGS 84,
  2 m for the NAD83(2011) Helmert, 3 m for GDA2020).
- **A target whose extent is wider than the real shift is not a ballpark.**
  The refusal decision now reads a partial-intersection candidate list, not
  PROJ's default strict one. Strict containment asks an operation's area to
  contain *both* CRSs' entire extents, so WGS 84 → EPSG:3035 (ETRS89 / LAEA
  Europe) listed nothing but the world ballpark: EPSG:3035 reaches 24.6°N while
  `ETRS89 to WGS 84 (1)` (EPSG:1149) stops at 33.26°N. PROJ's per-point
  pipeline applies the 1 m operation throughout its area regardless, so
  `create()` was refusing a transform the engine runs correctly (`BuildDem` in
  EPSG:3035 works again). Pairs newly accepted without `allow_ballpark`, each
  reporting the best stated accuracy among its real operations and transforming
  per point exactly as bare PROJ does: WGS 84 / ETRS89, geographic 2D and 3D, →
  EPSG:3035; WGS 84 → EPSG:28992 (Amersfoort / RD New); WGS 84 ↔ NAD 27
  (EPSG:4267); WGS 84 ↔ NAD83 (EPSG:4269); and ETRS89 → EPSG:27700 (British
  National Grid, the OSNet ETRS89-GBR realisation) and OSGB36 → EPSG:25832.
  Nothing previously accepted is now refused. A pair with no published
  transformation at all (WGS 84 ↔ OSGB70/EPSG:4278, a custom `+ellps=` with no
  datum name) still has only a ballpark candidate and is still refused without
  `allow_ballpark`. See `engine/LIMITATIONS.md`.
- **An optional grid is not reported missing.** PROJ's `+grids=@name` means
  "use it if installed, otherwise run without it". Such a grid is now listed
  without the leading `@` and never reported under `missing_grids`, so a
  transform whose CRS carries `+nadgrids=@name` builds as it did before WP-12.4
  instead of failing with `transformation needs grid @name; install proj-data`.
- **DEMs and orthomosaics are built through the CRS engine, in any CRS.**
  `SurfaceFrame` no longer carries a hand-written Transverse Mercator series
  and a three-CRS selector; it is now the exact chain chunk-local → ECEF (the
  chunk transform) → `crs::LocalFrame` → target CRS. DEM and orthomosaic grids
  can therefore be built in any CRS PROJ knows — a national grid such as
  ETRS89 / LAEA Europe (EPSG:3035), a compound CRS, or one described by WKT —
  not just UTM or geographic lon/lat. Auto-selection for an unspecified CRS is
  still the regular 6-degree UTM rule at the chunk origin, so an existing
  project's DEM lands in exactly the same zone as before, and the change is
  numerically neutral for UTM/lon-lat output (the retired series and PROJ agree
  well inside a millimetre). `BuildDem` and `BuildOrthomosaic` keep the
  `projection` (Int, EPSG) parameter and gain `crs` (String: `"EPSG:<code>"`, a
  PROJ string or WKT); setting both to disagreeing values is a `ValueError`. A
  DEM names its CRS as `Elevation::crs` (`Orthomosaic::crs` likewise) with an
  `epsg()` compatibility accessor; doc.xml writes the EPSG code as before and
  adds a `crs_wkt` attribute only when the CRS carries one, so an old project
  (an EPSG code and nothing else) loads to the same CRS and a plain-code
  project re-saves byte-identically. `BuildOrthomosaic` now always builds on the
  DEM's own plane: a request for a different CRS no longer fails with "reproject
  the DEM first" — it is accepted, and the ortho is written in the DEM's CRS (a
  cross-CRS conversion is an export-time reprojection). The `geo::Projection`
  and `TransverseMercator` implementation is removed.

## [0.2.0] - 2026-10-28

### Added

- **Depth maps can search with a census cost.** `BuildDepthMaps` gains
  `cost_type` (`"zncc"`, the original intensity cost, or `"census"`) and
  `max_cost`. The census cost replaces each pixel's intensity with the 16-bit
  wide census transform of Zabih & Woodfill (1994) using the four-arm
  descriptor of Mei et al. (2011) -- the sign of the intensity difference to
  four arms of four samples (left, right, top, bottom; 16 comparisons fill the
  word) -- read as a sample in [0, 1] (mask / 65535) and then correlated by the
  existing ZNCC. Reuse of the ZNCC core keeps its bilinear sub-pixel sampling
  and its variance floor while the descriptor makes the cost invariant to any
  monotone intensity change
  -- exposure, white balance, gamma -- which is where depth-map holes on real
  photo sets come from. The texture gate that skips flat windows is measured on
  the intensity window for both costs, so a low-contrast window is skipped by the
  same rule either way. Each view is transformed once per pyramid level and
  cached, so the search pays the transform once, not per candidate. The census
  cost is (1 - ZNCC) / 2 on its own [0, 1] interval; a `-1` `max_cost` resolves to
  the cost type's own default (0.5 for `zncc` on [0, 2], 0.6 for `census` on
  [0, 1], which runs slightly higher on a correct match because its descriptor is
  coarser). The depth-map fingerprint records both the choice and the effective
  threshold, so switching cost or `max_cost` rebuilds the maps instead of reusing
  them. The GPU PatchMatch kernels implement the `zncc` cost only and decline a
  census run to the CPU with a log line (the per-view descriptor buffer does not
  fit the kernels' binding budget); the CPU census cost is used either way.
- **Build Depth Maps searches coarse to fine.** The PatchMatch search now runs
  first on a reduced copy of each camera's block and starts every finer level
  from the level beneath, instead of spending the whole budget at the working
  resolution. The transfer needs no rescaling: halving a calibration also halves
  the focal length and re-expresses the principal point about the halved image
  centre, so a coarse pixel's bearing is the bearing of the fine coordinate
  between the two pixels it covers, and depth along a bearing and a
  camera-frame normal are the same numbers at both levels. Depths and normals
  are therefore carried bilinearly (a coarse hole is excluded from the
  interpolation rather than averaged in as a hole), the random search at a
  finer level starts from the measured local depth variation instead of the
  full configured range, and the iteration budget is split so the finest level
  takes 40 % and the coarse levels — which hold a quarter of the pixels each —
  take the rest. Two new `BuildDepthMaps` parameters control it:
  `pyramid_levels` (0 = automatic, 1 = a single level at the working
  resolution, N = exactly N) and `pyramid_min_size` (the shortest side an
  automatic pyramid stops shrinking at, 256 px by default). The depth-map
  fingerprint records the level count the run actually used, so a map built
  before this change is still reused by a single-level run and rebuilt by a
  pyramid run. The CPU and the GPU path both implement it from the same plan.
- **Build Orthomosaic processes the output in blocks, so a full-resolution ortho
  fits in memory.** A run whose whole-grid estimate fits the machine's budget
  still builds as one piece, byte for byte as before; a larger one is split into
  square blocks with a margin, each block rectifying only the cameras whose
  footprint reaches it and writing only its interior into the assembled output.
  Seams stay continuous across block borders: the margin an earlier block decided
  is pinned as a hard constraint, in raster order, out to the blend pyramid's
  reach (cells nearer the border stay free so the optimiser keeps its full
  window context instead of inheriting a window-edge artefact). Mosaic blending
  applies the whole grid's pyramid depth to every block, edge-padding a window
  too narrow to carry that depth, so an edge block does not blend with a
  shallower operator than its neighbours. Photographs are decoded through a
  byte-bounded LRU cache sized from what the block plan leaves over, with the
  next block's frames kept at the head of the LRU, and the decode count is
  logged at Info. Progress is reported per block and Cancel is honoured between
  blocks.
- **The orthomosaic block plan takes the largest block the machine can carry.**
  It walks 4096 → 512 cells and takes the first side whose peak (one window's
  stages plus the assembled output raster, the assembled-grid `fill_holes` and
  the photo cache) fits `platform::memory_budget_bytes`, which already reserves
  headroom (60% of physical RAM). A machine with the RAM does fewer, bigger
  blocks instead of being held to a fixed absolute target. The reported
  36-photo, 4592x3056, 1.9 cm case over a 100 m site (28 M cells, ~8 GB before)
  plans at 2048-cell blocks (3x3, ~4.5 GB peak) on an 8 GB machine, 1024-cell
  blocks (~2.5 GB) on a 2 GB budget, 4096-cell (2x2, ~12 GB) on 32 GB, and as
  one piece once the whole grid's ~14 GB estimate fits. The resolution is only
  coarsened when even a 512-cell block cannot fit. `block_size` overrides the
  block side (0 = automatic).
- **Build Depth Maps keeps the cameras it finished when cancelled.** A cancel
  (including one during filtering) stores the finished maps before it stops,
  and `reuse_depth` (now on by default) skips their search on the next run and
  filters them. A cancel commits only depth maps: point clouds, models, DEMs,
  orthomosaics and earlier depth-map sets are left untouched. Repeated cancels
  merge into one set marked "partial" instead of piling up a set per cancel, and
  the run that finally completes replaces it in place.
- **Depth-map reuse is keyed on the whole neighbourhood.** Each map records a
  fingerprint of the reference camera's pose and calibration, the downscale,
  `max_neighbors`, the neighbour cameras' ids, poses and calibrations, the depth
  range, the reference image file and the PatchMatch settings, so moving one
  camera rebuilds every map that had it as a neighbour, not only its own map,
  and changing `max_neighbors` rebuilds every map. A map also records the filter
  mode it has been through, so changing `filter_mode` filters rather than
  returns the weaker old result.
- **Build Depth Maps reports progress inside one camera.** The search reports
  after every refinement sweep on both the CPU and the GPU path, so the
  progress bar moves during a long image and Cancel is honoured within one
  sweep instead of after the current camera. Each finished camera logs its
  index, resolution, device (GPU/CPU) and wall-clock time.
- **The orthomosaic's default cell is the ground sample distance the photographs actually resolve.** `BuildOrthomosaic` measures each camera's footprint on the DEM (median of slant range / (focal × cos incidence) over the enabled cameras) and uses that for the output grid instead of a fixed fraction of the DEM cell, so an oblique flight correctly comes out at about twice the nadir footprint. The measurement assumes level ground at the DEM median height: the incidence term is the vertical, not the local surface normal, so it does not follow a slope. A Geographic (EPSG:4326) grid gets the result converted to degrees at the surface's latitude.
- **`BuildDem`'s automatic cell applies the same incidence correction**, so an oblique flight's DEM is no longer over-sampled by 1/cos(incidence).
- Orthomosaic memory is bounded before allocation: per-camera tiles are cropped to their footprints, photographs are decoded one at a time, and a run whose estimated peak exceeds 60 % of physical RAM (`ALPHASHAPE_MEMORY_BUDGET_MB` overrides for tests) is refused with a message naming a coarser resolution.
- **On a low-memory machine the orthomosaic's default resolution is coarsened automatically.** With `resolution` left at 0, a run whose estimate does not fit the memory budget is built at the finest cell that does fit (three significant figures), and a Warning reports the measured ground sample distance, the estimate, the budget and the chosen cell. An explicitly set `resolution` is never changed: it is still refused with an actionable message.

### Changed

- **Project format version 3.** The depth-map sidecar moved from ADM1 to ADM2,
  so the project version is bumped: an older 0.1.0 build refuses such a project
  with its clear "newer than this build supports" message instead of failing
  deep inside the sidecar decoder with "depth map file: bad header". Version 2
  projects still open: their ADM1 maps read with fingerprint 0 ("unknown", so
  never reused) and the "mild" filter a pre-ADM2 project could only have run.
- **BuildPointCloud and unfiltered depth maps.** A map with no confidence (one
  a cancelled Build Depth Maps checkpointed) now logs a warning and is fused
  rather than silently dropped when `min_confidence > 0`: the absence of a
  measurement is not a failing measurement.
- The occlusion ray march steps at half a **DEM** cell rather than half the larger of the ortho and DEM cells. Its stop is now metric and exact — the ray ends when it rises above the tallest DEM sample — instead of a fixed 4096-sample cap, which could drop a genuine occluder on a long slope viewed at a grazing angle. Asking for an ortho 4× finer than the DEM no longer multiplies the march cost per unit area. Occluders thinner than about one DEM cell can still be missed.
- The orthomosaic memory estimate is now per stage (rectify, seams — including the alpha-expansion push-relabel graph — blend and finalize), reporting the largest stage rather than a sum of unrelated ones, and the guard runs **before** any grid-sized allocation, including the plane→local node map that used to be built first.
- `BuildDem`'s robust extent clipping now applies from 200 samples instead of 1000, so a small block is protected too, and it logs an Info line naming the axis, the pre-clip and clipped extents in metres, and how many far samples were ignored.
- `BuildDem` and `BuildOrthomosaic` no longer clamp an automatic resolution to a quarter of the DEM cell. The measured GSD stands, with only an eighth-of-a-cell sanity floor.

### Security

- **Windows: no DLL planting on the Vulkan loader.** A bare `vulkan-1.dll` is now loaded with `LOAD_LIBRARY_SEARCH_SYSTEM32`, so it only ever comes from `System32`. Plain `LoadLibraryW` searched the application directory and the current working directory first, so a planted copy beside the executable (or in whatever directory the app was started from) would have been loaded instead of the driver's loader. `ALPHASHAPE_VULKAN_LIBRARY` obeys the same rule: name an alternative loader by absolute path.

### Fixed

- **Build Orthomosaic no longer overflows its grid dimensions.** A resolution fine enough to need more than 2^31−1 output cells (or a non-finite one) is refused in double precision with the resolution, the extent, the cell count and "choose a coarser resolution", instead of casting an unbounded double to `int` and then sizing buffers from a negative length.
- **Build Orthomosaic reports an exhausted allocation honestly.** A `std::length_error` (a buffer whose size exceeds the maximum the process can represent) is treated the same as `std::bad_alloc` and converted to the actionable memory message, instead of escaping as an internal error.
- **Build Model no longer freezes a 16 GB M1.** Long GPU iteration loops (TV-L1 fusion, screened Poisson, PatchMatch) are now split into submissions sized to about 50–100 ms, and a fence timeout or device loss falls back to the CPU for the rest of the session instead of leaving the display starved.
- **The GPU PatchMatch shaders no longer rely on undefined behaviour.** Two
  latent defects in the Vulkan kernels were fixed. A never-taken branch in
  `keep_bindings_alive()` indexed `views[]`, `view_ints[]` and `offsets[]` by the
  lane's pixel, but those buffers hold 24×, 4× and 2× the neighbour/window counts
  — out of range for almost every image, and an out-of-range storage-buffer
  access is undefined behaviour in Vulkan even in a branch the optimiser may
  speculate over. It now reads element 0 of each. Search state was also threaded
  through aggregate `inout` parameters, which glslang lowers to a Function-storage
  pointer with a copy-back that inlining is least likely to preserve: the RNG is
  now a stateless keyed stream (draw *n* = `splitmix64^(n+1)` of a key of seed,
  pixel, iteration and stream, with a literal draw index at every call site), and
  `Best` is passed and returned by value. `neighbour_cost()` also gained the
  CPU's quorum early exit, so a neighbour that cannot reach 75% of the window
  stops after a few samples instead of walking all 49.
  **Speed is unchanged: both search kernels stay unoptimised.** `pm_init` and
  `pm_refine` are still built without the SPIR-V optimiser, which is what they
  were before, because an optimised module of either changes results on NVIDIA
  hardware: on a Tesla T4 the optimised `pm_init` agrees with the CPU on 100.00%
  of depths but 0.92% of normals, and optimised `pm_refine` on 93.4% of the dense
  fixture, where the unoptimised modules are at 100%. The fix that would restore
  that agreement — marking the whole chain from a candidate plane to the
  projected pixel coordinate `precise`, because `sample_image()` floors that
  coordinate and a fused multiply-add moves the sample a texel — makes `glslc`
  need more memory than a CI runner has: it is OOM-killed at `-O`, at `-Os`, and
  at `-O0`, so this toolchain cannot compile it at all. Two tests now hold the
  line for a future attempt: `PatchMatchGpu.RefineSweepsMatchCpu` gates the
  search kernel itself (≥99% against the CPU on hardware, which fails the 93.4%
  regression; the existing `iterations = 0` test never reaches it), and
  `TvFusionGpu.BuildModelMeetsCpuBounds` asserts the fused TV-L1 mesh's
  boundary-edge count is *equal* to the CPU's again — a discontinuous function of
  that agreement — instead of tolerating a difference. See `LIMITATIONS.md` for
  the measurements and what has been ruled out.

### Limitations

- The orthomosaic is now rectified and blended in memory-bounded blocks, but the assembled output raster is still held in one piece, so a very large site on a low-memory machine can still be limited by it: when even a 512-cell block cannot fit, the automatic cell is coarsened (with a Warning) or an explicit cell is refused. See `engine/LIMITATIONS.md`.

## [0.1.0] - 2026-10-26

The first release. Photos in, 3-D models and maps out, from a desktop app, the command
line or Python.

### Added

**Desktop application (Qt 6)**

- One window with a 3-D view, a workspace tree, a properties pane, a photo list and a
  log. Light and dark themes, saveable dock layout.
- Start screen with New project, Open..., Add photos..., a recent-project list, and
  drag-and-drop of a project file, photos or a folder of photos.
- Workflow menu in pipeline order (Add Photos, Match Photos, Align Cameras, Optimize
  Cameras, Build Depth Maps, Build Point Cloud, Build Model, Build UV, Build Texture,
  Build DEM, Build Orthomosaic). Each entry opens a parameter dialog generated from the
  task's own schema and greys itself out with a reason when the project is missing what
  the task needs.
- Long-running tasks run on a worker thread with a progress bar, an estimated time and a
  Cancel button; the window stays responsive and the project is locked while a task runs.
- 3-D view of camera positions, sparse and dense point clouds, and the textured mesh,
  with orbit, pan, zoom, fit-to-scene, perspective/orthographic toggle and a ground grid.
- File > Export for the model (OBJ with its material and texture page, or PLY), the point
  cloud, the DEM and the orthomosaic (GeoTIFF).
- Undo and redo for project edits, and unsaved-changes protection on New, Open and Close.

**Pipeline**

- Structure from motion: difference-of-Gaussians keypoints with binary M-LDB
  descriptors, exhaustive, sequential, GPS-reference and generic preselection, match
  ratio test, RANSAC, incremental alignment and Ceres bundle adjustment.
- Dense stereo: PatchMatch multi-view matching, depth-map filtering, and voxel fusion
  into a dense point cloud with normals, colours and confidence.
- Meshing: TV-L1 volumetric fusion or screened Poisson, with quadric decimation.
- Texturing: chart-based UV unwrapping, single-page atlas packing, and a baked texture
  page. `blending_mode` is `mosaic` (the default: each face takes its single best view,
  with optional gain correction between views), `average` (weighted across the views that
  see the texel), or `disabled` (the same best view with gain correction off). `BuildUV`
  clears the model's texture pages and `BuildTexture` replaces page 0, so a texture is
  rebuilt rather than kept beside an older one.
- Rasters: DEM and orthomosaic on a north-up grid in a PROJ coordinate system, exported
  as GeoTIFF.
- Multi-chunk projects: duplicate, merge, append and align chunks. Models, point clouds,
  DEMs and orthomosaics are labelled assets, so building one again adds a new asset and
  keeps the previous one. Textures are the exception: there is one texture set per model,
  replaced in place by the runs above.
- GPU acceleration through Vulkan compute (PatchMatch and the Poisson solve) where a
  suitable device and driver are present, with an automatic CPU fallback.

**Front ends and formats**

- `alphashape-cli` for scripted runs: `--list-tasks`, `--list-params`, `--gpu-info`,
  `--proj-info`, `--new`, and `--project ... --task ... --set key=value`.
- A Python module (`import alphashape`) with the same tasks, parameters and error types,
  so a pipeline script ports between the two front ends unchanged.
- Projects are a `.psx` file plus a `.files/` folder with a versioned format; the reader
  refuses project files from a newer format version instead of guessing.
- A sample dataset of twelve synthetic aerial photographs with true camera poses, released
  under CC0, and a generator script that regenerates it.

**Packaging**

- macOS (Apple silicon) `.dmg`, Ubuntu 24.04 `.deb` and AppImage, and a Windows
  x86-64 `.exe` installer, with `SHA256SUMS` beside every release asset.
- The command line and the Python module ship inside the macOS app and the Windows
  installer; the AppImage runs the command line through `alphashape-*.AppImage cli`.

**Windows GPU acceleration**

- The Windows build now compiles the Vulkan GPU layer in, like the Linux and macOS
  builds. The Vulkan headers and the shader tools (`glslc`, `spirv-val`) come from
  vcpkg, and the configuration fails loudly rather than quietly dropping back to a
  CPU-only build.
- Nothing links a Vulkan loader on any platform: the backend opens
  `vulkan-1.dll` / `libvulkan.so.1` / `libvulkan.1.dylib` at run time. On a Windows
  machine without a Vulkan driver (a VM, a server, a driver that predates Vulkan)
  the application starts normally, reports why GPU acceleration is unavailable,
  and runs every task on the CPU. The installer deliberately does not bundle a
  loader; a private `vulkan-1.dll` would shadow the driver's own loader and could
  not see the driver's ICDs.
- The Windows test lanes run the GPU suites against Mesa's lavapipe software
  driver, the same arrangement Linux uses with llvmpipe. Correctness assertions
  run; the timing and speed-up assertions skip themselves on a software device,
  and the TV-L1 fusion port keeps declining software devices (its fp16 kernel
  cannot be compiled by the gallium LLVM backend).

### Known limitations

See [`engine/LIMITATIONS.md`](engine/LIMITATIONS.md). The short version for this release:
the installers are unsigned, the macOS build is Apple silicon only, Windows GPU
acceleration needs a Vulkan-capable graphics driver (the app falls back to the CPU
without one), and there is no georeferencing, marker editing or tiled-model output yet.

### Fixed

- The default orthomosaic resolution now follows the true ground sample distance instead of
  falling back to a 0.25 m cell. On a 36-photo oblique building flight (Sony NEX-5,
  16 mm, true GSD 1.9 cm) the automatic ortho was ~25 cm per pixel, 13x coarser than the
  imagery, because the measured footprint was dropped whenever the DEM's own cell was
  coarse (a decimated mesh's median edge, or a sparse point cloud's spacing) and the
  hard-coded 0.25 m literal took over. The rule now measures each camera's footprint on
  the surface at its image centre (slant range / focal / cos incidence), so an oblique
  flight is no longer under-estimated by up to 2x, and the cell falls back to the DEM's
  own cell rather than a metric literal.
- Automatic DEM and ortho cells are expressed in the output CRS's units. On an EPSG:4326
  (geographic) grid the depth-map ground sample distance and the TGV height scaling were
  applied as if the pixel size were metres, and the occlusion march mixed degree steps with
  metre heights; all three now convert through the frame's metres-per-degree at the model's
  latitude. An orthomosaic built in degrees also used to step the occlusion test by ~1e-6
  per iteration.
- BuildOrthomosaic's memory is now proportional to the imagery, not to cameras x grid. The
  ortho default following the ground sample distance made a 2 cm ortho of a 36-photo block
  28 million cells instead of ~0.2 million, and at that size the old layout needed ~12 GB:
  every camera's rectified tile was allocated on the whole output grid (8 bytes per cell)
  and all of them stayed resident through seamline optimisation and blending. Each tile is
  now cropped to that camera's own footprint box, photographs are decoded one at a time
  instead of all up front, the projection node map stores float offsets, and the hole-fill
  accumulator runs in bounded row bands. The result is byte-identical to before.
- BuildOrthomosaic refuses a run it cannot fit instead of dying with "bad allocation": it
  estimates the peak before allocating anything and raises a ValueError naming the memory
  needed, the machine's RAM and a coarser resolution to try. The budget is 60% of physical
  memory (override with `ALPHASHAPE_MEMORY_BUDGET_MB`); an allocation that still fails is
  reported the same way.
- The DEM's extent is robust against far samples. It used the raw min/max over every
  depth-map point, so a handful of far mismatches could turn a 100 m block into a
  kilometres-wide grid -- the actual cause of the "bad allocation" an 8 GB laptop hit at the
  old 0.25 m ortho default, before any ortho existed. An axis whose extent exceeds twice the
  0.5th-99.5th percentile core is now clipped to that core plus 3%; an ordinary,
  outlier-free sample set keeps exactly the grid it always had.

[0.2.0]: https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.2.0
[0.1.0]: https://github.com/aniketpratapsingh/alphashape/releases/tag/v0.1.0
