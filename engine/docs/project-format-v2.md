# doc.xml format v2

`doc.xml` is the document description stored at the root of a project archive.
It is written by `common/io/doc_xml_v2.cpp` (the only writer; the reader for
v1 lives in `doc_xml_v1.cpp`). Bulk data never lives in the XML: every element
that carries a payload names a binary entry in the same archive.

This file is authoritative for the vocabulary the writer can emit. The
`ProjectV2.ElementVocabulary` and `ProjectV2.AttributeVocabulary` tests in
`tests/test_project_v2.cpp` pin the writer, and `ProjectV2.FormatDocMatchesVocabulary`
parses the tables below and requires the same sets: a new element or attribute
in the writer, without a matching update here, fails CI with "doc.xml
vocabulary changed".

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

## Value types used in the type column

`int`, `float` (shortest round-trippable decimal), `string`, `bool` (`1`/`0`,
or `true`/`false` where the attribute is naturally boolean), `v2`/`v3` (space
separated numbers), `m3` (9 row-major numbers), `m4` (16 row-major numbers),
`ids` (space separated ints), `cov9` (9 row-major numbers), `list` (space
separated strings).

## Element table

Every element the writer can emit. `parent` names the container it appears in;
an element that appears in more than one context lists every parent.

| element | parent | description |
| --- | --- | --- |
| `document` | — | Archive root. |
| `chunks` | `document` | Holds every `<chunk>`. Always written. |
| `global_crs` | `document` | Document-wide CRS. Always written, even when unset (0 / empty). |
| `meta` | `document`, `chunk`, asset slots, `marker` | Metadata container; omitted on an asset only when it has no properties. |
| `property` | `meta` | One metadata entry; written in sorted key order. |
| `chunk` | `chunks` | One chunk; `id` and `label` always. Children are emitted in a fixed order. |
| `crs` | `chunk` | Chunk CRS. |
| `camera_crs` | `chunk` | Camera-position CRS override; absent means "follow the chunk CRS". |
| `marker_crs` | `chunk` | Marker CRS override. |
| `world_crs` | `chunk` | World/export CRS override. |
| `vertical` | any CRS element (`global_crs`, `crs`, the three slots) | Vertical datum of that CRS. |
| `sensors` | `chunk` | Always written; a sensor's index in this list is its id. |
| `sensor` | `sensors` | Physical imaging device. |
| `calibration` | `sensor` | Intrinsics and distortion in normalised pixels. |
| `covariance` | `calibration` | Element text is the row-major matrix. |
| `film` | `sensor` | Analogue film layout. |
| `fiducial` | `film`, `camera` | One film mark (`film`) or one measured fiducial pixel (`camera`). |
| `antenna` | `sensor` | GNSS antenna mount. |
| `cameras` | `chunk` | Always written. |
| `camera` | `cameras` | One image. |
| `reference` | `chunk`, `camera` | Chunk-level observation weights (`chunk`) or measured GNSS position/attitude (`camera`). |
| `film_transform` | `camera` | Film millimetres to scan pixels. |

| `camera_groups` | `chunk` | Only when at least one group exists. |
| `group` | `camera_groups`, `marker_groups`, `shape_groups` | Group membership. |
| `frames` | `chunk` | Source images; only when frames exist. |
| `frame` | `frames` | One source image; the sidecar holds the same records. |
| `markers` | `chunk` | Only when markers or marker groups exist. |
| `marker` | `markers` | Control/tie marker. |
| `projection` | `marker` | One image measurement. |
| `marker_groups` | `markers` | Only when marker groups exist. |
| `scalebars` | `chunk` | Only when scale bars exist. |
| `scalebar` | `scalebars` | A known distance. |
| `tie_points` | `chunk` | Only when keypoints, matches, or tracks exist. |
| `keypoint_set` | `tie_points` | One camera's detected features. |
| `matches` | `tie_points` | Verified camera pairs. |
| `tracks` | `tie_points` | Multi-view tracks. |
| `mask_sets` | `chunk` | Only when masks exist. |
| `masks` | `mask_sets` | Holds the mask elements. |
| `mask` | `masks` | One per-camera mask. |
| `depth_map_sets` | `chunk` | Only when depth-map sets exist. |
| `depth_maps` | `depth_map_sets` | One depth-map set. |
| `dense_clouds` | `chunk` | Only when point clouds exist. |
| `dense_cloud` | `dense_clouds` | One point cloud. |

| `models` | `chunk` | Only when models exist. |
| `model` | `models` | One mesh. |
| `texture` | `model` | Texture page list. |
| `page` | `texture` | One texture page. |
| `tiled_models` | `chunk` | Only when tiled models exist. |
| `tiled_model` | `tiled_models` | One opaque tiled-model payload. |
| `elevations` | `chunk` | Only when elevation models exist. |
| `elevation` | `elevations` | One DEM; the GeoTIFF carries the samples. |
| `orthomosaics` | `chunk` | Only when orthomosaics exist. |
| `orthomosaic` | `orthomosaics` | One orthophoto plus its per-cell camera labels. |
| `trajectories` | `chunk` | Only when trajectories exist. |
| `trajectory` | `trajectories` | One timed GNSS trajectory. |
| `shapes` | `chunk` | Vector features. |
| `shape` | `shapes` | One vector feature. |
| `shape_groups` | `shapes` | Only when shape groups exist. |
| `panorama` | `chunk` | The chunk panorama. |
| `region` | `chunk` | Oriented bounding region in chunk-local axes. |
| `transform` | `chunk` | Chunk-local to world placement. |

Every asset element (`depth_maps`, `dense_cloud`, `model`, `tiled_model`,
`elevation`, `orthomosaic`, `panorama`) carries the same slot identity:
`id`: int and `label`: string always; `group`: int only when >= 0; `enabled`:
bool only when false; a `<meta>` child only when it has properties; `active`:
bool only on the active asset of a kind that has an active id (every kind
except the panorama). The payload attributes below are written only when the
slot holds a payload.

## Attribute table

One row per attribute. `when written` describes the condition inside the
element; a payload attribute is only emitted when the slot holds a payload.

| element | attribute | type | when written | meaning |
| --- | --- | --- | --- | --- |
| `document` | `version` | int | always | format version (currently 4) |
| `document` | `next_id` | int | always | next chunk id |
| `document` | `active_id` | int | always | active chunk id |
| `global_crs` | `epsg` | int | always, even when unset | 0 means no EPSG code |
| `global_crs` | `wkt` | string | always, even when unset | empty when no WKT is carried |
| `vertical` | `kind` | string | only when a vertical datum is set | `ellipsoidal`, `geoid`, or `gravity-related` |
| `vertical` | `name` | string | only when a vertical datum is set | datum name |
| `vertical` | `grid` | string | only when a vertical datum is set | grid file name |
| `vertical` | `offset` | float | only when a vertical datum is set | geoid offset in metres |
| `vertical` | `horizontal_epsg` | int | only when set and non-zero | coupled horizontal EPSG code |
| `property` | `name` | string | always | metadata key |
| `property` | `value` | string | always | metadata value |
| `chunk` | `id` | int | always | chunk id |
| `chunk` | `label` | string | always | chunk label |
| `chunk` | `next_id` | int | only when non-zero | id allocator |

| `sensor` | `id` | int | always | sequential index |
| `sensor` | `label` | string | always | sensor label |
| `sensor` | `type` | string | always | `frame` or `fisheye` |
| `sensor` | `width` | int | always | sensor width in pixels |
| `sensor` | `height` | int | always | sensor height in pixels |
| `sensor` | `pixel_size` | float | always | pixel pitch, micrometres |
| `sensor` | `focal_length` | float | always | focal length, millimetres |
| `sensor` | `make` | string | always | camera make |
| `sensor` | `model` | string | always | camera model |
| `sensor` | `serial` | string | always | camera serial number |
| `sensor` | `fixed_params` | list | always | list of coefficient names held fixed |
| `sensor` | `master` | int | only when this sensor is a slave (>= 0) | master sensor id |
| `calibration` | `type` | string | always | calibration model |
| `calibration` | `width` | int | always | calibrated width in pixels |
| `calibration` | `height` | int | always | calibrated height in pixels |
| `calibration` | `f` | float | always | focal length in normalised pixels |
| `calibration` | `cx` | float | always | principal point x |
| `calibration` | `cy` | float | always | principal point y |
| `calibration` | `b1` | float | always | affinity coefficient b1 |
| `calibration` | `b2` | float | always | affinity coefficient b2 |
| `calibration` | `k1` | float | always | radial coefficient k1 |
| `calibration` | `k2` | float | always | radial coefficient k2 |
| `calibration` | `k3` | float | always | radial coefficient k3 |
| `calibration` | `k4` | float | always | radial coefficient k4 |
| `calibration` | `p1` | float | always | tangential coefficient p1 |
| `calibration` | `p2` | float | always | tangential coefficient p2 |
| `calibration` | `p3` | float | only when non-zero | tangential coefficient p3 |
| `calibration` | `p4` | float | only when non-zero | tangential coefficient p4 |
| `covariance` | `params` | list | only when a posterior covariance exists | list of coefficient names |
| `film` | `transform` | string | only when film mode is enabled | `affine`/`projective` |
| `film` | `detector` | string | only when film mode is enabled | `frame_corner`/`circle`/`corner` |
| `film` | `corners` | bool | only when film mode is enabled | search frame corners |
| `film` | `sides` | bool | only when film mode is enabled | search frame sides |
| `film` | `resolution` | float | only when film mode is enabled and > 0 | mm per pixel |
| `film` | `size_x` | float | only when film mode is enabled and both > 0 | film width, mm |
| `film` | `size_y` | float | only when film mode is enabled and both > 0 | film height, mm |
| `film` | `normalised` | bool | only when true | calibration already in the film frame |
| `fiducial` | `label` | string | always, for a film mark | film mark label |
| `fiducial` | `x` | float | always, for a film mark | film millimetres |
| `fiducial` | `y` | float | always, for a film mark | film millimetres |
| `fiducial` | `index` | int | always, for a measured fiducial | position in the sensor's fiducial list |
| `fiducial` | `u` | float | always, for a measured fiducial | pixels |
| `fiducial` | `v` | float | always, for a measured fiducial | pixels |
| `antenna` | `offset` | v3 | only when an offset is non-zero | metres |
| `antenna` | `boresight` | v3 | only when a boresight is recorded | degrees |

| `camera` | `id` | int | always | camera id |
| `camera` | `sensor_id` | int | always | owning sensor id |
| `camera` | `label` | string | always | camera label |
| `camera` | `enabled` | bool | always | camera participates in processing |
| `camera` | `orientation` | int | always | orientation index |
| `camera` | `transform` | m4 | always | camera-to-world pose |
| `camera` | `selected` | bool | only when true | camera is selected |
| `camera` | `location_covariance` | cov9 | only when non-zero | position covariance |
| `camera` | `rotation_covariance` | cov9 | only when non-zero | rotation covariance |
| `reference` | `x` | float | always, when the element is present | measured x |
| `reference` | `y` | float | always, when the element is present | measured y |
| `reference` | `z` | float | always, when the element is present | measured z |
| `reference` | `accuracy` | v3 | always, when the element is present | position accuracy |
| `reference` | `yaw` | float | only together when orientation is recorded | yaw, degrees |
| `reference` | `pitch` | float | only together when orientation is recorded | pitch, degrees |
| `reference` | `roll` | float | only together when orientation is recorded | roll, degrees |
| `reference` | `rotation_accuracy` | v3 | only when set | rotation accuracy, degrees |
| `reference` | `rtk_flag` | int | only when the source reported a fix status | GNSS solution status (50 fixed, 34 float) |
| `reference` | `enabled` | bool | always, when the element is present | reference is used |
| `reference` | `camera_location_accuracy` | float | always, on the chunk element | chunk-level camera weight |
| `reference` | `camera_rotation_accuracy` | float | always, on the chunk element | chunk-level rotation weight |
| `reference` | `marker_location_accuracy` | float | always, on the chunk element | chunk-level marker weight |
| `reference` | `marker_projection_accuracy` | float | always, on the chunk element | chunk-level projection weight |
| `reference` | `scalebar_accuracy` | float | always, on the chunk element | chunk-level scale-bar weight |
| `reference` | `tiepoint_accuracy` | float | always, on the chunk element | chunk-level tie-point weight |
| `reference` | `use_gcp` | bool | always, on the chunk element | use ground control points |
| `film_transform` | `values` | m3 | only when a per-scan transform was fitted | film millimetres to scan pixels |
| `group` | `id` | int | always | group id |
| `group` | `label` | string | always | group label |
| `group` | `type` | string | camera groups only | `none`/`folder`/`station`/`fixed` |
| `group` | `members` | ids | always | member ids |
| `group` | `folder` | string | camera groups only, when non-empty | folder path |
| `frames` | `path` | string | only when frames exist | the `frames.bin` entry |
| `frame` | `id` | int | always | frame id |
| `frame` | `path` | string | always | source image path |
| `frame` | `width` | int | always | source image width |
| `frame` | `height` | int | always | source image height |
| `frame` | `camera_id` | int | always | owning camera id |
| `elevation` | `elevations` | slot identity; `path`: string; `width`: int; `height`: int; `origin`: v2; `pixel_size`: v2; `epsg`: int; `nodataValue`: float; `source_data`: string; `crs_wkt`: string — only when the CRS carries a WKT. | One DEM; the GeoTIFF carries the samples. |
| `marker` | `id` | int | always | marker id |
| `marker` | `label` | string | always | marker label |
| `marker` | `type` | string | always | `regular`/`vertex`/`fiducial` |
| `marker` | `group` | int | only when >= 0 | owning marker group |
| `marker` | `projections` | int | only when non-empty | projection count |
| `marker` | `location` | v3 | only together, when a reference exists | reference position |
| `marker` | `reference_accuracy` | v3 | only together, when a reference exists | reference accuracy |
| `marker` | `location_estimate` | v3 | only together, when an estimate exists | estimated position |
| `marker` | `position_covariance` | cov9 | only together, when an estimate exists | position covariance |
| `marker` | `pinned` | bool | only when true | marker is pinned |
| `marker` | `valid` | bool | only when true | marker is valid |
| `marker` | `internal` | bool | only when true | internal tie marker |
| `projection` | `camera_id` | int | always | owning camera id |
| `projection` | `pixel` | v2 | always | image position |
| `projection` | `enabled` | bool | only when false | projection is disabled |
| `scalebar` | `id` | int | always | scale-bar id |
| `scalebar` | `label` | string | always | scale-bar label |
| `scalebar` | `from_kind` | string | always | `marker`/`camera` |
| `scalebar` | `from_id` | int | always | first endpoint id |
| `scalebar` | `to_kind` | string | always | `marker`/`camera` |
| `scalebar` | `to_id` | int | always | second endpoint id |
| `scalebar` | `reference_length` | float | always | metres |
| `scalebar` | `reference_accuracy` | float | always | metres |
| `scalebar` | `estimated_length` | float | always | metres |
| `keypoint_set` | `camera_id` | int | always | owning camera id |
| `keypoint_set` | `count` | int | always | keypoint count |
| `keypoint_set` | `path` | string | always | `.kpt` entry |
| `matches` | `pairs` | int | only when matches exist | verified pair count |
| `matches` | `path` | string | only when matches exist | `matches.bin` |
| `tracks` | `count` | int | only when tracks exist | track count |
| `tracks` | `path` | string | only when tracks exist | `tracks.bin` |
| `tracks` | `covariance_path` | string | only when at least one track has a covariance | `track_covariance.bin` |
| `mask` | `camera_id` | int | always | owning camera id |
| `mask` | `width` | int | always | mask width |
| `mask` | `height` | int | always | mask height |
| `mask` | `path` | string | always | mask entry |
| `mask` | `invert` | bool | only when true | mask polarity is inverted |

| `depth_maps` | `id` | int | always | asset id |
| `depth_maps` | `label` | string | always | asset label |
| `depth_maps` | `group` | int | only when >= 0 | owning asset group |
| `depth_maps` | `enabled` | bool | only when false | asset is disabled |
| `depth_maps` | `active` | bool | only on the active asset | asset is active |
| `depth_maps` | `count` | int | only when the slot holds a payload | depth-map count |
| `depth_maps` | `path` | string | only when the slot holds a payload | `.bin` entry |
| `dense_cloud` | `id` | int | always | asset id |
| `dense_cloud` | `label` | string | always | asset label |
| `dense_cloud` | `group` | int | only when >= 0 | owning asset group |
| `dense_cloud` | `enabled` | bool | only when false | asset is disabled |
| `dense_cloud` | `active` | bool | only on the active asset | asset is active |
| `dense_cloud` | `count` | int | only when the slot holds a payload | point count |
| `dense_cloud` | `hasPointColors` | bool | only when the slot holds a payload | payload carries colours |
| `dense_cloud` | `hasPointNormals` | bool | only when the slot holds a payload | payload carries normals |
| `dense_cloud` | `hasPointConfidence` | bool | only when the slot holds a payload | payload carries confidence |
| `dense_cloud` | `path` | string | only when the slot holds a payload | `.bin` entry |
| `model` | `id` | int | always | asset id |
| `model` | `label` | string | always | asset label |
| `model` | `group` | int | only when >= 0 | owning asset group |
| `model` | `enabled` | bool | only when false | asset is disabled |
| `model` | `active` | bool | only on the active asset | asset is active |
| `model` | `source_id` | int | only when >= 0 | id of the source product |
| `model` | `faceCount` | int | only when the slot holds a payload | face count |
| `model` | `vertexCount` | int | only when the slot holds a payload | vertex count |
| `model` | `hasVertexColors` | bool | only when the slot holds a payload | payload carries colours |
| `model` | `hasVertexConfidence` | bool | only when the slot holds a payload | payload carries confidence |
| `model` | `hasUV` | bool | only when the slot holds a payload | mesh has a UV layout |
| `model` | `path` | string | only when the slot holds a payload | `mesh.bin` |
| `texture` | `pages` | int | only when the model has texture pages | texture page count |
| `page` | `path` | string | always, per page | `texture_<n>.png` |
| `tiled_model` | `id` | int | always | asset id |
| `tiled_model` | `label` | string | always | asset label |
| `tiled_model` | `group` | int | only when >= 0 | owning asset group |
| `tiled_model` | `enabled` | bool | only when false | asset is disabled |
| `tiled_model` | `active` | bool | only on the active asset | asset is active |
| `tiled_model` | `path` | string | only when the slot holds a payload | payload entry |
| `tiled_model` | `size` | int | only when the slot holds a payload | payload bytes |

| `elevation` | `id` | int | always | asset id |
| `elevation` | `label` | string | always | asset label |
| `elevation` | `group` | int | only when >= 0 | owning asset group |
| `elevation` | `enabled` | bool | only when false | asset is disabled |
| `elevation` | `active` | bool | only on the active asset | asset is active |
| `elevation` | `path` | string | only when the slot holds a payload | GeoTIFF entry |
| `elevation` | `width` | int | only when the slot holds a payload | raster width |
| `elevation` | `height` | int | only when the slot holds a payload | raster height |
| `elevation` | `origin` | v2 | only when the slot holds a payload | north-west corner |
| `elevation` | `pixel_size` | v2 | only when the slot holds a payload | ground sample distance |
| `elevation` | `epsg` | int | only when the slot holds a payload | CRS EPSG code |
| `elevation` | `nodataValue` | float | only when the slot holds a payload | no-data sentinel |
| `elevation` | `source_data` | string | only when the slot holds a payload | source product |
| `elevation` | `crs_wkt` | string | only when the slot holds a payload and the CRS carries a WKT | CRS WKT |
| `orthomosaic` | `id` | int | always | asset id |
| `orthomosaic` | `label` | string | always | asset label |
| `orthomosaic` | `group` | int | only when >= 0 | owning asset group |
| `orthomosaic` | `enabled` | bool | only when false | asset is disabled |
| `orthomosaic` | `active` | bool | only on the active asset | asset is active |
| `orthomosaic` | `path` | string | only when the slot holds a payload | GeoTIFF entry |
| `orthomosaic` | `labels_path` | string | only when the slot holds a payload | per-cell camera labels |
| `orthomosaic` | `width` | int | only when the slot holds a payload | raster width |
| `orthomosaic` | `height` | int | only when the slot holds a payload | raster height |
| `orthomosaic` | `origin` | v2 | only when the slot holds a payload | north-west corner |
| `orthomosaic` | `pixel_size` | v2 | only when the slot holds a payload | ground sample distance |
| `orthomosaic` | `epsg` | int | only when the slot holds a payload | CRS EPSG code |
| `orthomosaic` | `blendingMode` | string | only when the slot holds a payload | surface (`elevation`/`model`) |
| `orthomosaic` | `crs_wkt` | string | only when the slot holds a payload and the CRS carries a WKT | CRS WKT |

| `trajectory` | `id` | int | always | trajectory id |
| `trajectory` | `label` | string | always | trajectory label |
| `trajectory` | `count` | int | always | timed-pose count |
| `trajectory` | `path` | string | always | payload entry |
| `shapes` | `path` | string | only when features exist | `shapes.shp` |
| `shapes` | `count` | int | only when features exist | feature count |
| `shape` | `id` | int | always | feature id |
| `shape` | `type` | string | always | `point`/`polyline`/`polygon` |
| `shape` | `label` | string | always | feature label |
| `shape` | `layer` | string | always | feature layer |
| `shape` | `closed` | bool | always | feature is closed |
| `shape` | `has_altitude` | bool | only when true | vertices carry altitude |
| `panorama` | `id` | int | always | asset id |
| `panorama` | `label` | string | always | asset label |
| `panorama` | `group` | int | only when >= 0 | owning asset group |
| `panorama` | `enabled` | bool | only when false | asset is disabled |
| `panorama` | `path` | string | only when an image is held | PNG entry |
| `panorama` | `width` | int | only when an image is held | image width |
| `panorama` | `height` | int | only when an image is held | image height |
| `region` | `centre` | v3 | always | region centre |
| `region` | `size` | v3 | always | region size |
| `region` | `rotation` | m3 | only when non-identity | region rotation |
| `transform` | `matrix` | m4 | always | chunk-local to world placement |
| `crs` | `epsg` | int | always | chunk CRS EPSG code |
| `crs` | `wkt` | string | always | chunk CRS WKT |
| `camera_crs` | `epsg` | int | only when the slot is set explicitly | camera CRS EPSG code |
| `camera_crs` | `wkt` | string | only when the slot is set explicitly | camera CRS WKT |
| `marker_crs` | `epsg` | int | only when the slot is set explicitly | marker CRS EPSG code |
| `marker_crs` | `wkt` | string | only when the slot is set explicitly | marker CRS WKT |
| `world_crs` | `epsg` | int | only when the slot is set explicitly | world CRS EPSG code |
| `world_crs` | `wkt` | string | only when the slot is set explicitly | world CRS WKT |

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
