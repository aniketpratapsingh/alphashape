# macOS app bundle and DMG

arm64 only, ad-hoc signed (no notarization).

| File | Purpose |
|------|---------|
| `Info.plist.in` | Bundle metadata; rendered by `engine/cmake/alphashape_macos_bundle.cmake` (version) and `build-app.sh` (minimum macOS) |
| `build-app.sh` | Assembles `alphashape.app` from a build tree, runs `macdeployqt`, `bundle_libs.py`, and signs |
| `bundle_libs.py` | Copies the remaining non-system dylibs into `Contents/Frameworks`, rewrites install names to `@rpath`, audits the result |
| `make-icns.sh` | Renders `../linux/org.alphashape.app.svg` into `alphashape.icns` (`rsvg-convert`, `iconutil`) |
| `build-dmg.sh` | UDZO disk image with the app and an `/Applications` link |
| `smoke-app.sh` | Mount, audit and run checks, executed on a fresh runner with Homebrew moved away |

## Layout

```
alphashape.app/Contents/
  MacOS/alphashape-gui, alphashape-cli
  Frameworks/                 Qt 6.8 (macdeployqt), Ceres, PROJ, glog, gflags, jpeg, png, tiff, libomp,
                              libvulkan.1.dylib (Vulkan loader), libMoltenVK.dylib
  PlugIns/                    Qt plugins
  Resources/python/           alphashape.cpython-3XX-darwin.so  (use with PYTHONPATH)
  Resources/share/proj/       proj.db and PROJ data
  Resources/vulkan/icd.d/     MoltenVK_icd.json (library_path is relative to the manifest)
  Resources/doc/              README, LICENSE, limitations
```

The GUI target is not a `MACOSX_BUNDLE` target; the bundle is assembled from the finished build
tree so the test suite keeps using the plain executable.

## Runtime lookups

* **Vulkan.** The engine loads the loader with `dlopen`. On macOS it first looks for
  `Frameworks/libvulkan.1.dylib` and `Resources/vulkan/icd.d/MoltenVK_icd.json` relative to the
  Mach-O image it lives in (the executable or the Python module), and exports `VK_DRIVER_FILES`
  for that manifest unless the caller already set `VK_DRIVER_FILES` / `VK_ICD_FILENAMES`.
  `alphashape-cli --gpu-info` lists the visible devices.
* **PROJ.** `ALPHASHAPE_PROJ_DATA` wins; otherwise `Resources/share/proj` is found from the
  executable (`../Resources/share/proj`) or from the Python module (`../share/proj`).
  `alphashape-cli --proj-info` shows the database in use.
* **Python.** `PYTHONPATH=/Applications/alphashape.app/Contents/Resources/python`, with the
  Python minor version in the module's file name.

## Opening an ad-hoc signed app

Right-click the app and choose Open, or `xattr -dr com.apple.quarantine /Applications/alphashape.app`.
