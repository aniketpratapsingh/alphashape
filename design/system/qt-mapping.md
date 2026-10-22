# Qt 6 Widgets mapping

Three layers, in order of authority:

1. **QPalette** — OS-integrated roles (native menus, native dialogs, disabled groups, QLineEdit internals
   that QSS does not fully cover). Set in `Theme::apply()`.
2. **QSS** — the visual layer for our widgets (`alphashape-dark.qss`, `alphashape-light.qss`).
3. **Custom paint** — only the Vulkan viewport overlays and the stage rail's progress ticks.

Tokens in `tokens.json` are compiled, not referenced at runtime: QSS has no custom properties, `calc()`,
flex/grid, transforms, transitions, or `box-shadow`. Anything that looks like a shadow in the mockups is
either a 1 px border + a slightly raised surface (real Qt drawing) or is desktop-compositor chrome
(dialog shadows come from the window manager). Do not port mockup CSS `box-shadow` into QSS; it will be
silently ignored.

## 1. QPalette roles

| QPalette role | Dark | Light |
|---|---|---|
| `Window` | `surface.app` #0D1114 | `surface.app` #EDF0F1 |
| `WindowText` | `text.primary` #E8EEF1 | #101619 |
| `Base` | `surface.panel` #12171B | `surface.raised` #FFFFFF |
| `AlternateBase` | `surface.raised` #171E23 | #F2F5F5 |
| `Text` | `text.primary` #E8EEF1 | #101619 |
| `PlaceholderText` | `text.disabled` #6F7E88 | #76838C |
| `Button` | `surface.raised` #171E23 | #FFFFFF |
| `ButtonText` | `text.primary` #E8EEF1 | #101619 |
| `BrightText` | `text.primary` #FFFFFF | #000000 |
| `Highlight` | `selection.bg` #1E3F3C | #CFEAE6 |
| `HighlightedText` | `text.primary` #E8EEF1 | #101619 |
| `Accent` (6.6+) | `accent.base` #35BDAF | #0E7C72 |
| `Link` | `text.link` #6FD3C8 | `accent.text` #0B6E66 |
| `LinkVisited` | #A78BFA | #6D4AC0 |
| `ToolTipBase` | `surface.overlay` #1D252B | #FFFFFF |
| `ToolTipText` | `text.primary` #E8EEF1 | #101619 |
| `Light` / `Midlight` | #33414B / #28323A | #FFFFFF / #F2F5F5 |
| `Mid` / `Dark` / `Shadow` | `border.default` / `border.subtle` / #000000 | #C7D0D4 / #DDE3E5 / #A3B0B6 |

Disabled group: same hues, `text.disabled`, buttons at 45 % opacity equivalent (`QPalette::Disabled`
`ButtonText` = #6F7E88 dark / #76838C light).

Inactive group: `Highlight` = `selection.bgInactive` so unfocused selection is visibly calmer.

## 2. QSS architecture

- One stylesheet per theme. No `@media`, no variables. Compiled from a small Python emitter
  (`tools/theme/compile_qss.py`, later) that reads `tokens.json` and substitutes `$token.path`.
- Selectors used: type (`QTreeView`), ID (`#stageRail`), class (`QPushButton[role="primary"]`), and
  pseudo-states (`:hover`, `:pressed`, `:checked`, `:selected`, `:disabled`, `:focus`, `:open`).
- `qproperty-*` used only for enum properties QSS can't express: `qproperty-iconSize` on toolbars.
- All sizes in `px`. `Qt::AA_EnableHighDpiScaling` is default in Qt 6; use device-independent px only.

### Gotchas that shaped the sheets

- `QTreeView::indicator` styling for checkboxes breaks native tri-state rendering; we use a custom
  `QStyledItemDelegate` for the enable/visibility toggle column instead and leave `::indicator` unstyled.
- `QDockWidget::title` accepts background/border/padding but not flexbox; the title's buttons are
  separate `QToolButton`s positioned by a custom title bar widget (`DockTitleBar`).
- `QTabBar::tab` supports `padding`, `margin`, `min-width`; it does not support `border-radius` on the
  first/last child only — use `:first` / `:last` (supported by Qt as sub-control pseudo-states).
- `QProgressBar::chunk` supports background and border-radius; indeterminate mode is animated by Qt only
  for the native style, so our "processing" state uses a custom `QProgressBar` subclass with a marquee
  timer (paint only), not QSS animation.
- `QScrollBar` must style `::handle`, `::add-line`, `::sub-line`, `::add-page`, `::sub-page`, or
  sub-controls fall back to native art and look inconsistent.
- `QMenu::item` supports `:selected` (not `:hover`) for keyboard+mouse highlight consistency.
- `QToolTip` needs `border` + `padding`; opacity is not available — use an opaque overlay color.

## 3. Starter stylesheets

Both files are complete for the requested widget list and are written only against supported QSS.
They live beside this file: `alphashape-dark.qss`, `alphashape-light.qss`.

### Coverage

`QMainWindow`, `QWidget#appBackground`, `QDockWidget` + custom `DockTitleBar`, `QTreeView`,
`QTableView`/`QHeaderView`, `QTabBar`/`QTabWidget`, `QToolBar`/`QToolButton`, `QPushButton`
(primary/secondary/destructive/ghost), `QLineEdit`, `QSpinBox`, `QDoubleSpinBox`, `QComboBox`,
`QProgressBar`, `QScrollBar`, `QMenu`, `QToolTip`, `QGroupBox`, `QStatusBar`, `QCheckBox`,
`QRadioButton`, `QSplitter::handle`, `QFrame[role=card]`.

### Accent usage rule

Primary buttons are `role="primary"` and use `accent.base`. Only **one** primary button per surface
(dialog footer, empty state). Destructive buttons are outline-error, not filled, so a mis-click near
"Run" is not catastrophic.

### Accent variants (round 3)

`color.accentVariants` holds three complete families. A variant is **not** only a hue swap: the status
colours move with it, because warning amber sits at hue 37 deg next to an ember accent at 17 deg.
Ember and copper therefore compile to a gold warning (~45-53 deg) and a rose error (~345 deg);
teal keeps amber/red. The light blocks are separate hexes, not darkened dark values: orange at
lightness 0.40 cannot carry white button text, so light ember/copper use `#A83A0E` / `#8F5C1F`
(fills), `#9A3409` / `#7E5219` (accent-as-text). When emitting QSS, replace
`accent.base|hover|pressed|text|subtle|onAccent`, `selection.bg`, and the `status.warning*` /
`status.error*` values; everything else is theme-stable.

The viewport is not styled by QSS: it is Vulkan. `color.viewport` gives the two canvas modes and the
camera/GCP palettes per accent family. The View menu item "Viewport background: Follow theme / Dark /
Light" persists per project (`ui.viewportBackground=follow|dark|light`, default `follow`).
