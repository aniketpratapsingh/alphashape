# Linux packages

| File | Purpose |
|------|---------|
| `org.alphashape.app.desktop`, `.svg`, `.metainfo.xml.in` | Desktop entry, icon, AppStream metadata |
| `copyright` | Debian copyright file (GPL-3.0-or-later; Qt under LGPL-3) |
| `bundle-qt.sh` | Copies a private Qt 6.8 into the `.deb` install tree |
| `build-appimage.sh`, `AppRun` | AppImage via linuxdeploy + linuxdeploy-plugin-qt + appimagetool |
| `smoke-deb.sh`, `smoke-appimage.sh` | Install/run checks executed in a clean `ubuntu:24.04` container |

Install rules and CPack settings live in `engine/cmake/alphashape_install.cmake`.

## .deb (Ubuntu 24.04, amd64)

Layout under `/usr`: `bin/alphashape-cli`, `bin/alphashape-gui`,
`lib/python3/dist-packages/alphashape*.so` (so the system `python3` imports it),
`lib/alphashape/qt/{lib,plugins}` (private Qt), desktop/icon/metainfo, `share/doc/alphashape`.

**Qt decision.** Ubuntu 24.04 ships Qt 6.4 and the GUI needs 6.6+ (`QPalette::Accent`). The `.deb`
therefore bundles the Qt 6.8.3 libraries and plugins privately in `/usr/lib/alphashape/qt` (found
through the GUI's `RUNPATH`; Qt is relocatable, so no `qt.conf` or environment variable is needed).
Nothing is placed in the system library path, so it cannot clash with distribution Qt packages. Qt
stays dynamically linked and replaceable (LGPL-3).

The shaders are embedded in the binaries; PROJ data comes from the `proj-data` package.
Vulkan: `libvulkan1` is required, `mesa-vulkan-drivers` recommended; with no GPU the engine falls
back to the CPU.

## AppImage

Built from the same install tree (without the Python module). Bundles Qt and every non-glibc
dependency, plus `share/proj`. Run `./alphashape-*.AppImage` for the GUI and
`./alphashape-*.AppImage cli --list-tasks` for the command line. Without FUSE, add
`--appimage-extract-and-run`. It is built on Ubuntu 24.04 and needs glibc/libstdc++ at least that new.
