# macOS app bundle and DMG

arm64 only, ad-hoc signed (no notarization).

| File | Purpose |
|------|---------|
| `Info.plist.in` | Bundle metadata; rendered by `engine/cmake/alphashape_macos_bundle.cmake` (version) and `build-app.sh` (minimum macOS) |
| `build-app.sh` | Assembles `alphashape.app` from a build tree, runs `macdeployqt`, `bundle_libs.py`, and signs |
| `select-proj-data.py` | Stages the size-bounded PROJ data subset into `Resources/share/proj` |
| `bundle_libs.py` | Copies the remaining non-system dylibs into `Contents/Frameworks`, rewrites install names to `@rpath`, audits the result |
| `make-icns.sh` | Renders `../linux/org.alphashape.app.svg` into `alphashape.icns` (`rsvg-convert`, `iconutil`) |
| `build-dmg.sh` | ULFO (lzfse) disk image with the app and an `/Applications` link |
| `smoke-app.sh` | Mount, audit and run checks, executed on a fresh runner with Homebrew moved away |

## Size budget

The DMG must stay under 72 MB, which the CI packaging job enforces. Two things account for it:

* **PROJ data is a curated subset.** Homebrew's `proj` keg carries the complete world-wide
  `proj-data` collection (~760 MB of GeoTIFF/GTX grids). Copying every grid under a per-file size
  cap produced a 520 MB bundle, because nearly all of those grids are under 32 MB.
  `select-proj-data.py` ships `proj.db`, the licence/attribution text, every PROJJSON TIN grid
  (a second kind of grid, under 3 MB in total), a fixed essentials list (one EGM96 variant and,
  when the source has it, the `us_noaa_conus` datum-shift grid) and then as many raster grids as
  fit a 16 MB budget, smallest first: about **29 MB** in the bundle. That is a different rule
  from Ubuntu's `proj-data` package (about 8 MB, hand-picked for EPSG coverage), not a
  comparable set — the budget is what decides here, so coverage grows with the byte cap rather
  than with a curated country list. The script writes `Resources/share/proj/SELECTED.txt`, a
  sorted manifest of what it copied, which `smoke-app.sh` checks the bundle against.
  Large grids are deliberately absent: **EGM2008 is not bundled** (a single
  `us_nga_egm08_25.tif` is 77 MB, more than the whole DMG budget), and neither are the large
  national NTv2 grids. To use one, download the grid from `cdn.proj.org` into a directory that
  also has `proj.db` — e.g. `curl -O https://cdn.proj.org/us_nga_egm08_25.tif` in
  `/usr/local/share/proj` — and set `ALPHASHAPE_PROJ_DATA` to that directory; it outranks the
  bundle in the search order.
* **DMG compression is LZFSE.** `hdiutil`'s UDZO default is zlib level 1 (fastest). Measured on
  the code portion alone: UDZO 41.2 MB, UDZO at level 9 38.2 MB, ULFO (lzfse) 37.2 MB, UDBZ
  (bzip2) 35.9 MB but slow to mount. ULFO is the size/speed trade the script uses, with UDZO at
  level 9 as the fallback for a host without it.

`build-app.sh` prints the directory breakdown, the 20 largest files and the bundle total on every
packaging run, so a regression shows up in the job log.

## Layout

```
alphashape.app/Contents/
  MacOS/alphashape-gui, alphashape-cli
  Frameworks/                 Qt 6.8 (macdeployqt), Ceres, PROJ, glog, gflags, jpeg, png, tiff, libomp,
                              libvulkan.1.dylib (Vulkan loader), libMoltenVK.dylib
  PlugIns/                    Qt plugins (macdeployqt) plus platforms/libqoffscreen.dylib (and
                              libqminimal.dylib when the Qt kit has it) for headless runs
  Resources/python/           alphashape.cpython-3XX-darwin.so  (use with PYTHONPATH)
  Resources/share/proj/       proj.db and the curated PROJ data subset (see Size budget)
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
