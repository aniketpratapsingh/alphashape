# Icons

**Family chosen: Lucide** (ISC license, ~1,600 icons, 24 px grid, 2 px stroke, consistent round caps).

Why Lucide over Phosphor: Lucide's stroke geometry holds up at **14–16 px** in dense toolbars (Phosphor's
duotone/fill variants invite visual noise, and its regular weight is optically lighter at small sizes).
Lucide also ships `lucide-static` SVGs, so we can vendor exactly the set below without an icon font and
without a runtime dependency. Attribution: "Icons by Lucide (ISC)".

Rules: 16 px default in toolbars/panels, 14 px inside table rows and tree rows, 20 px only for the stage
rail and empty-state glyphs. Never recolor individual paths; set `currentColor`. Status icons are always
paired with a text label (color is not the only signal).

## Icon map (Lucide name → alphashape use)

### App / project
| Lucide | Use |
|---|---|
| `mountain-snow` | App mark, Start screen |
| `folder-plus` | New project |
| `folder-open` | Open project |
| `history` | Recent projects |
| `sparkles` | What's new |
| `settings-2` | Preferences |
| `panel-left` | Toggle Workspace |
| `panel-right` | Toggle Properties/Reference |
| `panel-bottom` | Toggle Photos/Queue |
| `maximize-2` / `minimize-2` | Full-screen viewport |
| `command` | Command palette |

### Data / Workspace
| Lucide | Use |
|---|---|
| `layers` | Chunk (project root) |
| `camera` | Cameras group |
| `image` | Single photo |
| `images` | Photos group |
| `scan-line` | Tie points |
| `cloud` | Point cloud (dense) |
| `grid-3x3` | Mesh / model |
| `mountain` | DEM |
| `map` | Orthomosaic |
| `crosshair` | Markers / GCPs |
| `ruler` | Scale bar |
| `shapes` | Shape layers |
| `box` | Bounding box / region |
| `eye` / `eye-off` | Visibility |
| `lock` / `lock-open` | Lock transform |
| `power` | Enable/disable for processing |

### Stages
| Lucide | Use |
|---|---|
| `folder-input` | Add photos |
| `scan-search` | Align |
| `crosshair` | Control (markers) |
| `cloud-cog` | Point cloud |
| `box` | Surface (mesh) |
| `mountain` | Surface (DEM) |
| `badge-check` | Deliver / quality report |
| `package` | Export |

### Viewport
| Lucide | Use |
|---|---|
| `orbit` | Orbit navigation mode |
| `hand` | Pan |
| `zoom-in` / `zoom-out` | Zoom |
| `box-select` | Rectangle select |
| `lasso-select` | Free-form select |
| `move-3d` | Move gizmo |
| `rotate-3d` | Rotate gizmo |
| `scaling` | Scale gizmo |
| `grid-3d` | Toggle grid |
| `axis-3d` | Toggle axes/gizmo corner |
| `perspective` | Perspective/ortho toggle |
| `maximize` | Fit all |
| `camera` | Show/hide cameras |
| `crosshair` | Show/hide markers |
| `sun` / `moon` | Theme toggle |

### Jobs / status
| Lucide | Use |
|---|---|
| `play` / `pause` | Start / pause job |
| `square` | Stop/cancel |
| `rotate-ccw` | Retry / reset |
| `list-ordered` | Queue |
| `terminal` | Console |
| `circle-check` | Done |
| `circle-alert` | Warning |
| `circle-x` | Failed |
| `loader` (animated) | Processing |
| `clock` | Queued/ETA |
| `timer` | Elapsed |
| `hard-drive` | Disk estimate |
| `memory-stick` | RAM estimate |
| `file-text` | Log |
| `folder-search` | Reveal output |

### Editing / masks
| Lucide | Use |
|---|---|
| `square-dashed` | Rectangle mask |
| `scissors` | Scissors mask |
| `paintbrush` | Paint mask |
| `wand-sparkles` | Magic wand mask |
| `plus` / `minus` | Add / subtract selection |
| `contrast` | Invert selection/mask |
| `eraser` | Clear mask |

### Misc
| Lucide | Use |
|---|---|
| `search` | Photos filter / search |
| `filter` | Filter rows |
| `chevron-down` / `chevron-right` | Disclosure / tree |
| `more-horizontal` | Row overflow menu |
| `grip-vertical` | Drag handle |
| `check` | Checkbox checked |
| `info` | Inline help |
| `triangle-alert` | Blocking warning |
| `keyboard` | Shortcuts |

Total mapped: 66 icons (some reuse intentionally, e.g. `crosshair` for markers in three contexts).

Vendoring: `tools/icons/lucide/` (future) via `lucide-static` at a pinned version, license file included.
