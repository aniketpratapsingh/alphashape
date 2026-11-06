# doc.xml format v2

`doc.xml` is the document description stored at the root of a project archive.
It is written by `common/io/doc_xml_v2.cpp` (the only writer; the reader for
v1 lives in `doc_xml_v1.cpp`). Bulk data never lives in the XML: every element
that carries a payload names a binary entry in the same archive.

This file is authoritative for the vocabulary the writer can emit. The
`ProjectV2.ElementVocabulary` and `ProjectV2.AttributeVocabulary` tests in
`tests/test_project_v2.cpp` pin it: a new element or attribute in the writer,
without a matching update here and in those expected sets, fails CI with
"doc.xml vocabulary changed".

## Container layout

The directory form is a text manifest plus a directory:

    myproject.psx              text manifest
    myproject.files/
        project.zip            doc.xml + every binary entry
        cache/                 scratch (undo spill, GeoTIFF codec files)
        lock                   present only while a writer holds the project

`myproject.psx` contains:

    alphashape-project <format-version>
    files=myproject.files/project.zip

A path ending in `.psz` is the single-file form: one zip holding exactly the
entries `project.zip` holds, with `doc.xml` at its root. No `.files` directory
is created for it. Loading either form parses the archive fully into memory.

Inside the archive, the document is `doc.xml`; each chunk's binary entries are
under `chunks/<chunk_id>/`:

    chunks/<id>/tie_points/camera_<camera>.kpt
    chunks/<id>/tie_points/matches.bin
    chunks/<id>/tie_points/tracks.bin
    chunks/<id>/tie_points/track_covariance.bin
    chunks/<id>/masks/<camera>.bin
    chunks/<id>/depth_maps/<asset>.bin
    chunks/<id>/point_clouds/<asset>.bin
    chunks/<id>/models/<asset>/mesh.bin
    chunks/<id>/models/<asset>/texture_<page>.png
    chunks/<id>/tiled_models/<asset>.bin
    chunks/<id>/elevations/<asset>.tif
    chunks/<id>/orthomosaics/<asset>.tif
    chunks/<id>/orthomosaics/<asset>.labels.bin
    chunks/<id>/trajectories/<asset>.bin
    chunks/<id>/shapes.shp
    chunks/<id>/frames.bin
    chunks/<id>/panorama_<asset>.png

Every entry except `doc.xml` starts with a four-byte magic that selects its
decoder (see `src/common/io/sidecars.hpp`).

## Value types used in the attribute column

`int`, `float` (shortest round-trippable decimal), `string`, `bool` (`1`/`0`,
or `true`/`false` where the attribute is naturally boolean), `v2`/`v3` (space
separated numbers), `m3` (9 row-major numbers), `m4` (16 row-major numbers),
`ids` (space separated ints), `cov9` (9 row-major numbers), `list` (space
separated strings).

## Root

| element | parent | attributes | description |
| --- | --- | --- | --- |
| `document` | — | `version`: int — format version (currently 4); `next_id`: int — next chunk id; `active_id`: int — active chunk id. Always written. | Archive root. |
| `chunks` | `document` | — | Holds every `<chunk>`. Always written. |
| `global_crs` | `document` | `epsg`: int; `wkt`: string. Always written, even when unset (0 / empty). | Document-wide CRS. |
| `meta` | `document`, `chunk`, asset slots, `marker` | — | Metadata container; omitted on an asset only when it has no properties. |
| `property` | `meta` | `name`: string; `value`: string. Always written. | One metadata entry; written in sorted key order. |

## Chunk skeleton

| element | parent | attributes | description |
| --- | --- | --- | --- |
| `chunk` | `chunks` | `id`: int; `label`: string; `next_id`: int — id allocator, only when non-zero. | One chunk; `id` and `label` always. Children are emitted in a fixed order. |
| `crs` | `chunk` | `epsg`: int; `wkt`: string. Always written. | Chunk CRS. |
| `camera_crs` | `chunk` | `epsg`: int; `wkt`: string. Only when the slot is set explicitly. | Camera-position CRS override; absent means "follow the chunk CRS". |
| `marker_crs` | `chunk` | `epsg`: int; `wkt`: string. Only when the slot is set explicitly. | Marker CRS override. |
| `world_crs` | `chunk` | `epsg`: int; `wkt`: string. Only when the slot is set explicitly. | World/export CRS override. |
| `vertical` | any CRS element (`global_crs`, `crs`, the three slots) | `kind`: string — `ellipsoidal`, `geoid`, or `gravity-related`; `name`: string; `grid`: string; `offset`: float; `horizontal_epsg`: int — only when non-zero. Only when a vertical datum is set. | Vertical datum of that CRS. |
| `region` | `chunk` | `centre`: v3; `size`: v3 — always written. `rotation`: m3 — only when non-identity. | Oriented bounding region in chunk-local axes. |
| `transform` | `chunk` | `matrix`: m4. Always written. | Chunk-local to world placement. |
| `reference` | `chunk` | `camera_location_accuracy`: float; `camera_rotation_accuracy`: float; `marker_location_accuracy`: float; `marker_projection_accuracy`: float; `scalebar_accuracy`: float; `tiepoint_accuracy`: float; `use_gcp`: bool. All always written; the element appears only when a value differs from the default. | Chunk-level observation weights. (The camera child `<reference>` is a different element vocabulary, documented below.) |

## Sensors and cameras

| element | parent | attributes | description |
| --- | --- | --- | --- |
| `sensors` | `chunk` | — | Always written; a sensor's index in this list is its id. |
| `sensor` | `sensors` | `id`: int — sequential index; `label`: string; `type`: string — `frame` or `fisheye`; `width`: int; `height`: int; `pixel_size`: float — micrometres; `focal_length`: float — millimetres; `make`: string; `model`: string; `serial`: string; `fixed_params`: list of coefficient names; `master`: int — only when this sensor is a slave (>= 0). | Physical imaging device. |
| `calibration` | `sensor` | `type`: string; `width`: int; `height`: int; `f`: float; `cx`: float; `cy`: float; `b1`: float; `b2`: float; `k1`: float; `k2`: float; `k3`: float; `k4`: float; `p1`: float; `p2`: float — all always written. `p3`: float; `p4`: float — only when non-zero. | Intrinsics and distortion in normalised pixels. |
| `covariance` | `calibration` | `params`: list of coefficient names. Only when a posterior covariance exists. | Element text is the row-major matrix. |
| `film` | `sensor` | `transform`: string — `affine`/`projective`; `detector`: string — `frame_corner`/`circle`/`corner`; `corners`: bool; `sides`: bool — always. `resolution`: float — mm/px, only when > 0; `size_x`: float; `size_y`: float — mm, only when both > 0; `normalised`: bool — only when true. Only when film mode is enabled. | Analogue film layout. |
| `fiducial` | `film` | `label`: string; `x`: float; `y`: float — film millimetres. Always written. | One film mark. |
| `antenna` | `sensor` | `offset`: v3 — metres; `boresight`: v3 — degrees, only when set. Only when an offset is non-zero or a boresight is recorded. | GNSS antenna mount. |
| `cameras` | `chunk` | — | Always written. |
| `camera` | `cameras` | `id`: int; `sensor_id`: int; `label`: string; `enabled`: bool; `orientation`: int; `transform`: m4 — all always. `selected`: bool — only when true; `location_covariance`: cov9; `rotation_covariance`: cov9 — only when non-zero. | One image. |
| `reference` | `camera` | `x`: float; `y`: float; `z`: float; `accuracy`: v3 — all always. `yaw`: float; `pitch`: float; `roll`: float — only together when orientation is recorded; `rotation_accuracy`: v3 — only when set; `enabled`: bool — always. Only when a location was set. | Measured GNSS position/attitude. |
| `film_transform` | `camera` | `values`: m3. Only when a per-scan transform was fitted. | Film millimetres to scan pixels. |
| `fiducial` | `camera` | `index`: int — position in the sensor's fiducial list; `u`: float; `v`: float — pixels. Always written for each recorded mark. | One measured fiducial pixel. |
| `camera_groups` | `chunk` | — | Only when at least one group exists. |
| `group` | `camera_groups`, `marker_groups`, `shape_groups` | `id`: int; `label`: string; `members`: ids — always. `type`: string — `none`/`folder`/`station`/`fixed`, camera groups only; `folder`: string — camera groups only, when non-empty. | Group membership. |
| `frames` | `chunk` | `path`: string — the `frames.bin` entry. Only when frames exist. | Source images. |
| `frame` | `frames` | `id`: int; `path`: string; `width`: int; `height`: int; `camera_id`: int. Always written. | One source image; the sidecar holds the same records. |

## Markers, scale bars, tie points and masks

| element | parent | attributes | description |
| --- | --- | --- | --- |
| `markers` | `chunk` | — | Only when markers or marker groups exist. |
| `marker` | `markers` | `id`: int; `label`: string; `type`: string — `regular`/`vertex`/`fiducial` — always. `group`: int — only when >= 0; `projections`: int — only when non-empty; `location`: v3; `reference_accuracy`: v3 — only together, when a reference exists; `location_estimate`: v3; `position_covariance`: cov9 — only together, when an estimate exists; `pinned`: bool; `valid`: bool; `internal`: bool — only when true. | Control/tie marker. |
| `projection` | `marker` | `camera_id`: int; `pixel`: v2 — always; `enabled`: bool — only when false. | One image measurement. |
| `marker_groups` | `markers` | — | Only when marker groups exist. |
| `scalebars` | `chunk` | — | Only when scale bars exist. |
| `scalebar` | `scalebars` | `id`: int; `label`: string; `from_kind`: string — `marker`/`camera`; `from_id`: int; `to_kind`: string; `to_id`: int; `reference_length`: float — metres; `reference_accuracy`: float — metres; `estimated_length`: float — metres. All always written. | A known distance. |
| `tie_points` | `chunk` | — | Only when keypoints, matches, or tracks exist. |
| `keypoint_set` | `tie_points` | `camera_id`: int; `count`: int; `path`: string — `.kpt` entry. Always written. | One camera's detected features. |
| `matches` | `tie_points` | `pairs`: int; `path`: string — `matches.bin`. Only when matches exist. | Verified camera pairs. |
| `tracks` | `tie_points` | `count`: int; `path`: string — `tracks.bin`; `covariance_path`: string — `track_covariance.bin`, only when at least one track has a covariance. Only when tracks exist. | Multi-view tracks. |
| `mask_sets` | `chunk` | — | Only when masks exist. |
| `masks` | `mask_sets` | — | Holds the mask elements. |
| `mask` | `masks` | `camera_id`: int; `width`: int; `height`: int; `path`: string — always. `invert`: bool — only when true. | One per-camera mask. |

## Assets and products

Every asset element (`depth_maps`, `dense_cloud`, `model`, `tiled_model`,
`elevation`, `orthomosaic`, `panorama`) carries the same slot identity:
`id`: int and `label`: string always; `group`: int only when >= 0; `enabled`:
bool only when false; a `<meta>` child only when it has properties; `active`:
bool only on the active asset of a kind that has an active id (every kind
except the panorama). The payload attributes below are written only when the
slot holds a payload.

| element | parent | attributes | description |
| --- | --- | --- | --- |
| `depth_map_sets` | `chunk` | — | Only when depth-map sets exist. |
| `depth_maps` | `depth_map_sets` | slot identity; `count`: int; `path`: string — `.bin` entry. | One depth-map set. |
| `dense_clouds` | `chunk` | — | Only when point clouds exist. |
| `dense_cloud` | `dense_clouds` | slot identity; `count`: int; `hasPointColors`: bool; `hasPointNormals`: bool; `hasPointConfidence`: bool; `path`: string. | One point cloud. |
| `models` | `chunk` | — | Only when models exist. |
| `model` | `models` | slot identity; `source_id`: int — only when >= 0; `faceCount`: int; `vertexCount`: int; `hasVertexColors`: bool; `hasVertexConfidence`: bool; `hasUV`: bool; `path`: string — `mesh.bin`. | One mesh. |
| `texture` | `model` | `pages`: int. Only when the model has texture pages. | Texture page list. |
| `page` | `texture` | `path`: string — `texture_<n>.png`. Always written per page. | One texture page. |
| `tiled_models` | `chunk` | — | Only when tiled models exist. |
| `tiled_model` | `tiled_models` | slot identity; `path`: string; `size`: int — payload bytes. | One opaque tiled-model payload. |
| `elevations` | `chunk` | — | Only when elevation models exist. |
| `elevation` | `elevations` | slot identity; `path`: string; `width`: int; `height`: int; `origin`: v2; `pixel_size`: v2; `epsg`: int; `nodataValue`: float; `source_data`: string; `crs_wkt`: string — only when the CRS carries a WKT. | One DEM; the GeoTIFF carries the samples. |
| `orthomosaics` | `chunk` | — | Only when orthomosaics exist. |
| `orthomosaic` | `orthomosaics` | slot identity; `path`: string; `labels_path`: string; `width`: int; `height`: int; `origin`: v2; `pixel_size`: v2; `epsg`: int; `blendingMode`: string — surface (`elevation`/`model`); `crs_wkt`: string — only when the CRS carries a WKT. | One orthophoto plus its per-cell camera labels. |
| `trajectories` | `chunk` | — | Only when trajectories exist. |
| `trajectory` | `trajectories` | `id`: int; `label`: string; `count`: int; `path`: string. All always written. | One timed GNSS trajectory. |
| `shapes` | `chunk` | `path`: string — `shapes.shp`; `count`: int — both only when features exist. | Vector features. |
| `shape` | `shapes` | `id`: int; `type`: string — `point`/`polyline`/`polygon`; `label`: string; `layer`: string; `closed`: bool — all always; `has_altitude`: bool — only when true. | One vector feature. |
| `shape_groups` | `shapes` | — | Only when shape groups exist. |
| `panorama` | `chunk` | slot identity; `path`: string; `width`: int; `height`: int — payload attributes only when an image is held. | The chunk panorama. |

## Determinism and compatibility

Attribute and child order is fixed by the writer, so saving an untouched,
loaded project reproduces identical bytes. A missing attribute means the
documented default: an absent `group` is -1, an absent `enabled` is true, an
absent `region@rotation` is identity, an absent `crs_wkt` means the `epsg`
code alone identifies the CRS, and an absent `master`/`source_id` is -1.

The reader is strict about elements inside `<chunk>`: an unknown element is a
`ProjectIOError`, not a skipped node. An unknown attribute is ignored with a
warning in `ReadReport`. Every element the writer emits is understood by the
reader (the vocabulary tests load the fixture back and require a warning-free
round-trip), so there is no write-only element.
