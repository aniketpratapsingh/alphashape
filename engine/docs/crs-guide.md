# Coordinate systems (M12)

This guide describes the coordinate-system layer M12 added: how it is layered,
how to add a CRS or a geoid, and the conventions every caller relies on. The
headers and implementation are normative; this page is the map. Operational
limits are in [`../LIMITATIONS.md`](../LIMITATIONS.md), and the tests and fixture
regeneration commands are in [`../TESTING.md`](../TESTING.md).

## Layout and layering

`alphashape_crs` is the engine library, built from `engine/src/model/crs/`:

| Concern | Files |
|---|---|
| CRS value type, vertical component, built-in list | `coordinate_system.*`, `builtin_crs.cpp` |
| Transform and datum-operation selection | `transform.*`, `datum.*` |
| PROJ discovery, data directory, thread contexts | `proj_context.*` |
| Geoid grids and the geoid store | `geoid.*` |
| Ellipsoidal vs orthometric heights | `vertical.*` |
| ECEF-anchored local tangent frames | `local_frame.*` |
| Tait-Bryan angles | `euler.*` |

PROJ is linked **PRIVATE**. No public header includes `<proj.h>`:
`proj_context.hpp` forward-declares the opaque `pj_ctx` and exposes
`ProjContext = pj_ctx*`, so consumers need neither PROJ's headers nor its target
(`src/model/crs/CMakeLists.txt`). `alphashape_crs` sits *below*
`alphashape_model` in the dependency order (model links crs PRIVATE), which is
why `local_frame.cpp` re-derives its anchor's geodetic position instead of
calling `model/geo/geodesy.hpp`, and why `geoid.cpp` decodes its single-band
GeoTIFF itself instead of using `model/raster`.

Consumers are `model/raster` (GeoTIFF CRS encoding, reprojection),
`common/algorithm` (reference conversion, DEM/ortho CRS migration), and the
Python bindings.

## `CoordinateSystem`, the value type

`crs::CoordinateSystem` holds only identifying strings: canonical WKT2:2019,
authority name and code, display name, and an optional `VerticalCrs`. It never
holds a `PJ*`, so copies are cheap and thread-safe; PROJ objects are created per
call on the thread context.

- The default constructor and `local()` are the unset engineering CRS `LOCAL`
  (chunk-space metres, no datum).
- `from_epsg(code)`, `from_wkt(text)`, `from_proj(text)` and `from_name(name)`
  are the factories. `from_wkt` accepts any dialect PROJ reads but stores the
  canonical WKT2:2019, so `to_wkt()` does not reproduce the input bytes.
  `from_proj` appends `+type=crs` when absent, because without it PROJ returns a
  coordinate operation rather than a CRS. `from_name` is case-sensitive, matches
  the built-in names plus `LOCAL`, and `from_epsg` is the path for every other
  code.
- Unparseable input throws `alphashape::ValueError` naming it; only `local()`
  cannot fail.
- `operator==` is string identity (vertical component, then the authority
  name/code pair, otherwise the WKT). `is_equivalent_to()` is the semantic
  comparison and asks PROJ; by default it ignores geographic axis order.

Python mirrors this in `engine/python/bindings/module.cpp`:
`CoordinateSystem()`, `CoordinateSystem(epsg=32633)`, and exactly one keyword of
`wkt=`, `proj=` or `name=`. `CoordinateSystem.init()` additionally accepts the
script-facing spellings `"EPSG:<code>"` and `"EPSG::<code>"`, a WKT definition
(any text containing `[`), or a built-in name. Read-only properties are `name`,
`authority`, `epsg` (`None` when the CRS has no EPSG code), `wkt`, plus
`toWKT()`, `toProj4()` and the static `listBuiltinCRS()`. A project uses
`chunk.crs = CoordinateSystem(...)`; assigning `CoordinateSystem()` clears it
back to `LOCAL`.

## Adding a CRS

There is no registration step for ordinary use. Any CRS in `proj.db` is
reachable by code with `CoordinateSystem::from_epsg`, by WKT with `from_wkt`, or
by PROJ string with `from_proj`; Python reaches the same three through
`CoordinateSystem(epsg=/wkt=/proj=)`.

The built-in list is a curated iteration list, not the limit of the engine. It
is defined by `kBuiltinCodes` in `engine/src/model/crs/builtin_crs.cpp`: exactly
48 EPSG codes, with names and kinds read from PROJ (`proj_get_name`,
`proj_get_type`) rather than hard-coded. To add an entry, add the code to
`kBuiltinCodes`, keep the duplicated `kCodes` table in
`engine/tests/fixtures/crs/make_skeletons.cpp` in step (that tool deliberately
compiles without the engine library), and regenerate
`builtin_wkt_skeletons.txt` with the `alphashape_crs_fixtures` target. A code
outside the list needs none of this: `from_epsg` is not limited to it.

## Vertical datums and geoids

A height is only meaningful with its datum. The engine distinguishes:

- **Ellipsoidal height `h`** above the reference ellipsoid, as GNSS and a
  geographic 3D CRS (EPSG:4979) report it. `VerticalCrs::Kind::Ellipsoidal`
  means "no vertical datum".
- **Orthometric height `H`** above the geoid (mean sea level), as a national
  datum such as NAVD88 or EGM96 height reports it.

The relation is `H = h - N`, where `N` is the geoid undulation at the point.
`orthometric_height(h, lon, lat, geoid)` computes it and `ellipsoidal_height()`
is its inverse. `VerticalCrs::offset_m` is a datum's constant bias, applied
around the geoid: on the source side `h = H + N + offset_src`, on the target
side `H = h - N - offset_dst`. The default offset 0 is exactly `H = h - N`.

The vertical datum is kept beside the horizontal WKT, not inside it:
`CoordinateSystem::with_vertical()` returns a copy carrying it, and `to_wkt()`
still returns the horizontal CRS's WKT. Two transform routes exist:

- `VerticalTransform::compound()` builds PROJ's own compound pipeline. Every
  non-ellipsoidal side must resolve to an EPSG vertical CRS through
  `lookup_vertical_crs()`; a custom grid has no code and is refused with a
  message pointing at `custom()`.
- `VerticalTransform::custom()` runs the horizontal transform first and then
  applies the geoid shift at the point's converted lon/lat, using the
  `geographic_crs` (normally the geoid's own frame, EPSG:4979 by default).
  It is required when either side names a project grid.

`CrsTransform::create()` itself accepts no geoid: `TransformOptions::src_vertical`
and `TransformOptions::dst_vertical` may only be ellipsoidal, so a set vertical
component is rejected rather than silently ignored. A point outside a geoid's
coverage makes the vertical conversion return `std::nullopt`; nothing is
extrapolated.

### Geoid grids and the store

A custom geoid grid must be a **single-band, 32-bit IEEE float GeoTIFF**,
north-up, in the NGA layout: the bounding box is of the grid *nodes*
(pixel-is-point), one band of undulations in metres. `GeoidGrid::open()` throws
`ProjectIOError` when the file is not readable that way or its tags do not
describe a north-up grid.

`GeoidGrid` evaluates the file two ways. `proj_undulation()` uses PROJ's
`+proj=vgridshift` on a private context whose search path is the grid's own
directory, which is the same code `cs2cs EPSG:4979 EPSG:5773` runs.
`grid_undulation()` reads the GeoTIFF directly with Catmull-Rom bicubic
interpolation. `undulation()` prefers PROJ and falls back to the direct read,
which is a correctness path: PROJ splits `+grids=` on whitespace, so a grid path
containing a space (including this repository's own checkout path) can only be
read by the fallback. Both evaluators agree at nodes; between nodes PROJ
interpolates bilinearly and the fallback bicubically.

`GeoidStore` is the project-facing collection. `from_proj_data()` seeds it with
entries for the grids PROJ's resolved data directory actually has, recognising
exactly `us_nga_egm96_15.tif` (id `EGM96`) and `us_nga_egm08_25.tif` (id
`EGM2008`); an absent file is simply not listed, and nothing is downloaded.
`add(id, name, grid_path, source, licence)` registers another grid; when the
store has a directory it copies the grid in, and `save(dir)` writes every grid
plus `index.xml` (normally under `<project>.files/geoids/`). `load(dir)` throws
naming a grid the index references but cannot find, so a project never silently
loses its vertical datum.

In Python the process-wide store is seeded by `GeoidStore::from_proj_data()` on
first use:

```python
import alphashape

alphashape.CoordinateSystem.addGeoid("/data/geoids/my_geoid.tif")
print(alphashape.CoordinateSystem.listGeoids())   # includes EGM96/EGM2008 when installed
```

`addGeoid(path)` canonicalises the path, uses the file stem as the id, opens the
file to validate it (a bad grid is a `ValueError` at registration, not later),
and registers it where it is; nothing is copied or downloaded. `listGeoids()`
returns the ids.

The repository bundles no user geoid grids. The only checked-in grid is the
test fixture `engine/tests/fixtures/crs/egm96_subset.tif`, a 1-degree subset used
by the tests. Packaged builds carry a curated PROJ data subset (see
`../LIMITATIONS.md`); EGM2008's 77 MB grid is not part of it. A user who needs a
grid supplies it with `addGeoid`, puts it in the project store, or installs the
platform's `proj-data`. The GUI has no vertical-CRS or geoid editor yet.

## Axis order at the engine boundary

Every coordinate that crosses the CRS API is in **visualisation order**:
`x = east/lon`, `y = north/lat`, `z = up` (metres, ellipsoidal height).
`CrsPoint{9.0, 50.0, 0.0}` therefore always means 9°E 50°N, never 9°N 50°E.

PROJ's authority axis order is normalised away, not exposed as an option: both
the automatic and the pinned pipeline are passed through
`proj_normalize_for_visualization()` (`transform.cpp`). EPSG:4326 is lat, lon in
the authority order, so a caller holding raw EPSG:4326 tuples must swap them
before calling `forward()`/`forward_batch()`. There is no "raw" mode. A batch
keeps per-point validity; a refused point comes back with NaN coordinates.

## How a transform is chosen

`CrsTransform::create(src, dst, options)` validates the inputs (a local or unset
CRS is rejected; the local engineering frame needs a `LocalFrame` instead), and
returns an identity transform without touching PROJ when the two CRSs compare
equal.

**Unpinned (the default).** The pipeline is the bare
`proj_create_crs_to_crs_from_pj(ctx, src, dst, nullptr, nullptr)` result,
normalised to visualisation order. PROJ keeps every candidate operation and
chooses **per point**, applying an area-limited national operation only inside
its area of use and falling back outside it. The engine does not select "the
first candidate": that would hand one national operation to the whole world.
`description()` reports the pipeline PROJ actually built.

`accuracy_m()` is a diagnosis, not the per-point choice: it is the smallest
positive accuracy among the non-ballpark candidates in a partial-intersection
listing (grid availability ignored), or -1 when PROJ states none. A per-point
pipeline may state no single accuracy.

**Ballpark refusal.** A ballpark is PROJ's null datum shift. If the candidate
listing contains no non-ballpark operation at all, `create()` throws unless
`TransformOptions::allow_ballpark` is true. An area-limited real operation is
accepted even where its area is smaller than the target's, because PROJ can
still apply it per point. A listed non-ballpark operation can also degrade to a
ballpark when its required grid is absent; the built pipeline is checked again
and refused under the same rule.

**Pinned.** `TransformOptions::preferred_operation` is an `AUTH:CODE` matched
case-insensitively against each candidate after PROJ's computed wrappers
(`INVERSE(...)`, `DERIVED_FROM(...)`) are stripped, so `EPSG:1693` finds a
listed `INVERSE(DERIVED_FROM(EPSG)):1693`. An unknown id is an error, not a
silent fallback. Pinning builds that operation itself (normalised), so it
applies everywhere instead of per point; the operation's own accuracy is
reported. `CoordinateSystem::with_transformation()` is the CRS-level equivalent,
returning a WKT2 `BOUNDCRS`; PROJ attaches a datum shift to a geodetic CRS, so a
projected source is refused and the caller binds the geographic CRS underneath.

A pinned operation, or a fixed pipeline, that requires a grid which is not
installed throws
`Error("transformation needs grid <name>; install proj-data")`. An **optional**
grid (`+grids=@name`) is never reported missing, and an unpinned per-point
pipeline may legitimately fall back to a grid-free operation.

## PROJ data directory resolution

`crs::resolve_data_directory()` applies this exact order. A candidate qualifies
only when it contains `proj.db`; otherwise the next step is tried.

1. **`ALPHASHAPE_PROJ_DATA`**, when set and valid.
2. **Beside the executable / module image**, tried in this order:
   - `<exe_dir>/share/proj`
   - `<exe_dir>/../share/proj`
   - on macOS, `<image_dir>/../Resources/share/proj` (app bundle)
   - `<image_dir>/../share/proj`
   - `<image_dir>/share/proj`

   `image_dir` is the directory of the loaded image, which is the Python
   extension module when Python imported it and the executable otherwise. That
   is how `Contents/MacOS/<exe>` and Windows' `<install>\python\alphashape.pyd`
   both reach `<install>\share\proj`.
3. **`<project_source>/../tools/proj-data`**, the staged development layout.
   `ALPHASHAPE_PROJECT_SOURCE_DIR` is defined by the `alphashape_proj` interface
   target; `tools/stage_proj_data.sh` populates the directory.
4. **PROJ's own search**, `proj_context_get_database_path()`.

`crs::proj_context_initialize()` runs once per process and is idempotent and
thread-safe. On the chosen directory it:

- exports `PROJ_DATA` and `ALPHASHAPE_PROJ_DATA`, so PROJ itself and child
  processes resolve grids identically even when the choice came from an
  exe-relative step;
- forces `PROJ_NETWORK=OFF` and disables network access and grid caching on
  every thread context, so a missing grid can be an error but is never a
  download;
- logs `PROJ data: <dir> (<step>)` at info level, and warns when no directory
  was found or when an `ALPHASHAPE_PROJ_DATA` without `proj.db` was ignored.

`crs::describe_environment()` (and the CLI's `--proj-info`) prints the version,
PROJ data version, database path and network state. `find_data_file(name)`
looks in the resolved directory first, then in PROJ's default search directory.
`proj_data_info()` reads `PROJ.VERSION` and `DATABASE.LAYOUT.VERSION.*` from the
`proj.db` PROJ will actually use.

A candidate is never modified: the resolved directory is only published as an
environment variable. `ALPHASHAPE_PROJ_DATA` is advisory, so a stale value is
skipped with a warning rather than breaking startup.

## Local tangent frames

`crs::LocalFrame` bridges the chunk's internal Cartesian coordinates (metres,
right-handed) and a geographic, geocentric or projected CRS. It stores an ECEF
anchor (`origin_ecef`, metres, EPSG:4978), a rotation whose **columns are the
local axes expressed in ECEF**, and the target `CoordinateSystem`.

- The default-constructed frame is the identity: local coordinates *are* the CRS
  coordinates, `to_crs()` is a copy, and no PROJ transform is built. That is the
  state of a chunk that has not been georeferenced.
- `enu_at(cs, origin_ecef)` builds the standard east-north-up tangent frame;
  `with_rotation()` takes an explicit local-to-ECEF rotation and uses it as
  given.
- `to_crs()` computes `origin_ecef + rotation * local`, converts ECEF to
  geodetic with an iterated Bowring inversion, then asks PROJ only for
  geodetic-to-projection. The ECEF-to-geodetic leg is deliberately not PROJ's
  closed-form `+inv +proj=cart`, whose inverse loses millimetres at these
  magnitudes. A geocentric target skips the geodetic step entirely.
  `to_local()` is the exact inverse.
- `transform()` is the 4x4 local-to-CRS affine: exact for a geocentric target,
  the anchor linearisation (built from `to_crs` at a 100 m offset) for a curved
  target, and identity for the identity frame. `transformationMatrix()` is its
  inverse.

Within 500 km of the anchor, a local point round-trips through `to_crs`/
`to_local` to better than a micrometre for any target CRS PROJ can express.
The factories throw `ValueError` for an unset or local CRS with a non-zero
anchor/rotation.

## Orientation and heading convention

Orientation angles use Tait-Bryan sequences, intrinsic and active:
`RotationOrder::ZYX` means "rotate about body z, then the new y, then the new
x", and the matrix is `R = Rz(yaw) * Ry(pitch) * Rx(roll)`. The fields are
positional: `yaw` is about the first axis of `order`, `pitch` the second, `roll`
the third. Default is `ZYX`. All six orders with three distinct axes are
implemented; the six repeated-axis proper-Euler sequences are not.

The world basis is east-north-up, the same as `CrsPoint`. A positive `ZYX` yaw
about +z turns +y (north) toward -x (west), so **yaw is counter-clockwise from
north** and a compass bearing (clockwise from true north, as in a flight log or
a DJI-style heading) maps to `yaw = -heading`. Heading 90° east is yaw -90°.
`to_matrix`/`from_matrix` take and return degrees; `EulerAngles::radians()`
converts.

A camera's body frame is **FLU: x forward, y left, z up**. Because ENU's x is
east while zero yaw points north, the zero-yaw body frame is x = north,
y = west, z = up. For `ZYX` the body-to-world rotation is

```
R_wb = Rz(90 deg) * Rz(yaw) * Ry(-pitch) * Rx(roll)
     = Rz(yaw + 90 deg) * Ry(-pitch) * Rx(roll)
```

so heading east (yaw -90°) makes body x = east, y = north, z = up.
`body_to_camera()` and `body_to_camera_nadir()` map that FLU frame to the engine
camera frame (x right, y down, z forward).

When a reference is converted between CRSs, `convert_orientation()` applies
`frame_rotation()` as `G * R`: the physical camera is unchanged and the world
axes are relabelled. For geographic-to-projected that is exactly the meridian
convergence about up; between two UTM zones it is convergence plus grid
rotation. Conjugating instead would wrongly put the convergence into roll. The
chunk meta key `reference/rotation_order` selects the order; absent means
`ZYX`.

## GeoTIFF CRS encoding

`model/raster/geotiff` writes GeoTIFF 1.1 directly against libtiff, without
GDAL. It emits ModelPixelScale (33550), ModelTiepoint (33922), GeoKeyDirectory
(34735), GeoAsciiParams (34737) and GDAL_NODATA (42113), with `PixelIsArea` and
the tiepoint at the outer top-left corner of pixel (0,0). A companion
`.tfw`/`.jgw` world file is written by `write_world_file()`.

The CRS comes from `GeoTiffOptions::crs` when engaged and set, otherwise from
the legacy integer `options.epsg`. The writer chooses the GeoKey model:

- **Projected** (kind `Projected`) with a horizontal code that fits 16 bits:
  `GTModelTypeGeoKey` = 1, `ProjectedCSTypeGeoKey` (3072) carries the EPSG code,
  and `PCSCitationGeoKey` (3073) carries the CRS name.
- **Geographic 2D/3D** (or a 4xxx `options.epsg` fallback) with a code that
  fits: model type 2, `GeographicTypeGeoKey` (2048) carries the code,
  `GeogCitationGeoKey` (2049) the name.
- **Everything else** — a custom CRS, code 0, a vertical, compound or geocentric
  CRS — stays user-defined: the type value is **32767** and the citation holds
  the full WKT2 definition. This is GDAL's user-defined convention.

GeoKey values are 16-bit, so an EPSG code is written inline only when
`0 < epsg <= 0xFFFF`. A code above 65535 cannot be represented and takes the
user-defined path with WKT2 instead. A compound CRS has no GeoTIFF model type,
and the engine stores its vertical datum beside the horizontal WKT
(`CoordinateSystem::to_wkt()` is the horizontal part), so a GeoTIFF records only
the horizontal CRS; the vertical datum stays in the project XML.

On read, the engine interprets keys 1024, 1026, 2048, 2049, 3072 and 3073. A
user-defined 32767 is treated as "no code", not as EPSG:32767. It prefers a real
projected code over a real geographic one, then takes the first citation in
priority order PCS, geographic, GTCitation. An ArcGIS `"ESRI PE String = ..."`
prefix is stripped and the remainder parsed as WKT. The reader's resolution
order is: EPSG type key, then WKT from the citation, then the local frame. Files
that decompose a projection into GDAL's individual GeoKeys (3075 and above,
such as `ProjCoordTransGeoKey`) are not interpreted.

`read_geotiff_info()` returns all of this without decoding pixels: grid, channel
count, bit depth, float flag, EPSG, CRS, tiling and nodata.

## Raster reprojection

`model/raster/reproject` maps a raster from one CRS to another by driving the
**target** grid, never the source: for every target pixel centre it finds the
source coordinate by an inverse CRS transform and samples the source there.
Driving from the source would leave target pixels empty wherever a projection
stretches.

- `suggest_grid()` builds the target grid from the transformed, densified source
  boundary (21 samples per edge, not just the four corners). Its origin is
  snapped outward to a whole multiple of the cell. `resolution = 0` derives the
  cell from the source GSD measured at the raster centre, so a geographic
  target gets degrees and a projected one metres.
- Because a CRS transform is expensive, the source coordinate is evaluated
  exactly on a coarse lattice starting at `kLatticeStep` = 16 target pixels and
  interpolated in between. The spacing is halved until the measured source error
  is at most `kMaxLatticeError` = 0.1 source pixel. `lattice_step = 1` forces
  the exact per-pixel mapping; any other positive value pins that spacing.
- Float rasters use nodata-aware bilinear resampling. Out-of-bounds taps,
  `nodata_in` (default -32767) and non-finite values carry no weight; the rest
  renormalise, and a cell whose total weight is below `kMinValidWeight` = 0.5
  becomes `nodata_out` (default -32767). Every source channel is sampled
  separately, so a multi-channel float raster keeps its channels.
- 8-bit rasters are **always** nearest: a blended colour was never
  photographed. Cells the source does not cover are written as zero, which for
  an RGBA orthomosaic is exactly "no observation" (alpha 0).
- An identical source and target CRS with an identical grid returns a
  bit-identical copy with no PROJ call. A same-CRS re-cell uses
  `resample_grid()`, which keeps the source's north-west corner rather than
  snapping it, so re-celling does not shift a real georeference by up to one
  cell.

Domain limits are deliberate: the target extent is one plane, so a source
straddling the antimeridian is not wrapped and refused points become nodata;
near a pole the inverse transform is singular, and cells whose centres PROJ
refuses are nodata. An unset or `LOCAL` CRS cannot be transformed between, and
`suggest_grid()` rejects one rather than inventing a target grid.

## Related documentation

- `engine/TESTING.md` lists the CRS suites, their PROJ requirement and how each
  fixture is regenerated. The opt-in CMake target `alphashape_crs_fixtures`
  (not built by `all`, not a ctest) runs the generators.
- `engine/LIMITATIONS.md` records the operational limits: offline grids,
  time-dependent operations, antimeridian/poles, decomposed GeoKeys, and the
  absent vertical-CRS UI.
- `engine/tools/stage_proj_data.sh` stages an installed PROJ data package into
  `tools/proj-data/` for the third search step.
