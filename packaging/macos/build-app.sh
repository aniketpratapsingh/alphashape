#!/usr/bin/env bash
# Assembles alphashape.app from a finished CMake build tree (arm64, ad-hoc signed).
#
# Usage: build-app.sh <build-dir> <out-dir>
#   needs on PATH: macdeployqt (from the Qt 6.8 install), rsvg-convert, iconutil, codesign,
#   python3. Homebrew supplies the libraries being bundled (proj, MoltenVK, Vulkan loader, ...).
#
# Environment overrides: PROJ_DATA_SRC, MOLTENVK_LIB, MOLTENVK_ICD, VULKAN_LOADER_LIB,
# QT_PLUGINS_DIR. QT_ROOT_DIR and qmake are used to locate the Qt plugin directory when
# QT_PLUGINS_DIR is not given.
set -euo pipefail

build=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
app="$out/alphashape.app"
contents="$app/Contents"

brew_prefix() { brew --prefix "$1" 2>/dev/null || true; }

PROJ_DATA_SRC=${PROJ_DATA_SRC:-$(brew_prefix proj)/share/proj}
MOLTENVK_LIB=${MOLTENVK_LIB:-$(brew_prefix molten-vk)/lib/libMoltenVK.dylib}
MOLTENVK_ICD=${MOLTENVK_ICD:-$(brew_prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json}
VULKAN_LOADER_LIB=${VULKAN_LOADER_LIB:-$(brew_prefix vulkan-loader)/lib/libvulkan.1.dylib}

for f in "$build/src/apps/gui/alphashape-gui" "$build/src/apps/cli/alphashape-cli" \
         "$PROJ_DATA_SRC/proj.db" "$MOLTENVK_LIB" "$VULKAN_LOADER_LIB" "$build/macos/Info.plist"; do
    [ -e "$f" ] || { echo "error: missing $f" >&2; exit 1; }
done

rm -rf "$app"
mkdir -p "$contents/MacOS" "$contents/Frameworks" "$contents/Resources/python" \
         "$contents/Resources/share/proj" "$contents/Resources/vulkan/icd.d" \
         "$contents/Resources/doc"

echo "== binaries"
cp "$build/src/apps/gui/alphashape-gui" "$build/src/apps/cli/alphashape-cli" "$contents/MacOS/"
modules=$(find "$build/python" -maxdepth 2 -name 'alphashape*.so')
[ -n "$modules" ] || { echo "error: no Python module under $build/python" >&2; exit 1; }
for m in $modules; do cp "$m" "$contents/Resources/python/"; done

echo "== icon, documentation"
"$here/make-icns.sh" "$repo/packaging/linux/org.alphashape.app.svg" "$contents/Resources/alphashape.icns"
cp "$repo/README.md" "$repo/LICENSE" "$repo/engine/LIMITATIONS.md" "$contents/Resources/doc/"
cp "$here/python-usage.txt" "$contents/Resources/python/README.txt"

echo "== PROJ data"
# The engine finds Contents/Resources/share/proj from Contents/MacOS and from Resources/python.
# proj.db and the config always ship. Grid files ship when they are at most 32 MB, which keeps the
# worldwide geoid models (EGM96, EGM2008) and drops the very large national grids.
(cd "$PROJ_DATA_SRC" && find . -maxdepth 1 -type f \( -name '*.db' -o -name '*.ini' -o -name '*.json' \) \
    -exec cp {} "$contents/Resources/share/proj/" \;)
(cd "$PROJ_DATA_SRC" && find . -maxdepth 1 -type f \( -name '*.tif' -o -name '*.gtx' \) -size -32769k \
    -exec cp {} "$contents/Resources/share/proj/" \;)
echo "PROJ data shipped: $(ls "$contents/Resources/share/proj" | wc -l | tr -d ' ') files," \
     "$(du -sh "$contents/Resources/share/proj" | cut -f1)"

echo "== Vulkan loader and MoltenVK"
# The loader (a dlopen target, so not in any otool -L) and the MoltenVK driver live in
# Contents/Frameworks; the manifest points at the driver relative to itself. The engine looks
# for both next to the app at run time (src/gpu/vulkan/loader.cpp).
cp -L "$VULKAN_LOADER_LIB" "$contents/Frameworks/libvulkan.1.dylib"
cp -L "$MOLTENVK_LIB" "$contents/Frameworks/libMoltenVK.dylib"
chmod 755 "$contents/Frameworks/"*.dylib
install_name_tool -id @rpath/libvulkan.1.dylib "$contents/Frameworks/libvulkan.1.dylib"
install_name_tool -id @rpath/libMoltenVK.dylib "$contents/Frameworks/libMoltenVK.dylib"
python3 - "$MOLTENVK_ICD" "$contents/Resources/vulkan/icd.d/MoltenVK_icd.json" <<'PY'
import json, os, sys
src, dst = sys.argv[1:3]
icd = {"file_format_version": "1.0.0",
       "ICD": {"library_path": "", "api_version": "1.4.0", "is_portability_driver": True}}
if os.path.exists(src):
    icd = json.load(open(src))
icd["ICD"]["library_path"] = "../../../Frameworks/libMoltenVK.dylib"
json.dump(icd, open(dst, "w"), indent=2)
PY

echo "== Info.plist (provisional), Qt"
sed 's/__MIN_OS__/11.0/' "$build/macos/Info.plist" > "$contents/Info.plist"
macdeployqt "$app" -always-overwrite

echo "== extra Qt platform plugins"
# macdeployqt deploys only the platform plugin for the machine it runs on (cocoa), but the smoke
# test and any headless run need the offscreen plugin; minimal is the generic fallback. Both are
# copied here, before bundle_libs.py, so their install names and rpaths are normalised with the
# rest of the bundle and the signing pass below covers them.
if [ -z "${QT_PLUGINS_DIR:-}" ]; then
    candidates=""
    if [ -n "${QT_ROOT_DIR:-}" ]; then
        candidates="$QT_ROOT_DIR/plugins $QT_ROOT_DIR/../plugins"
    fi
    if command -v macdeployqt >/dev/null 2>&1; then
        candidates="$candidates $(cd "$(dirname "$(command -v macdeployqt)")/.." && pwd)/plugins"
    fi
    for candidate in $candidates; do
        if [ -d "$candidate/platforms" ]; then
            QT_PLUGINS_DIR="$candidate"
            break
        fi
    done
fi
[ -n "${QT_PLUGINS_DIR:-}" ] && [ -d "$QT_PLUGINS_DIR/platforms" ] || {
    echo "error: cannot locate the Qt plugin directory (set QT_PLUGINS_DIR)" >&2; exit 1; }
echo "  Qt plugin directory: $QT_PLUGINS_DIR"
mkdir -p "$contents/PlugIns/platforms"
# offscreen is what the headless smoke run needs; minimal is a useful generic fallback for a
# session without a window server. Ship offscreen unconditionally, minimal when the Qt kit has it.
for plugin in libqoffscreen.dylib libqminimal.dylib; do
    src="$QT_PLUGINS_DIR/platforms/$plugin"
    if [ ! -f "$src" ]; then
        if [ "$plugin" = "libqminimal.dylib" ]; then
            echo "  (no $plugin in this Qt install; skipping)"
            continue
        fi
        echo "error: missing Qt platform plugin $src" >&2
        exit 1
    fi
    cp -L "$src" "$contents/PlugIns/platforms/$plugin"
    chmod 755 "$contents/PlugIns/platforms/$plugin"
    echo "  added $plugin  <-  $src"
done
ls "$contents/PlugIns/platforms"
# Qt resolves a bundle's plugins to Contents/PlugIns without a qt.conf, and macdeployqt does not
# write one. If it ever appears, keep it honest: a Plugins entry must name PlugIns, since paths
# in Resources/qt.conf are relative to the bundle's Contents directory.
qt_conf="$contents/Resources/qt.conf"
if [ -f "$qt_conf" ]; then
    echo "qt.conf in the bundle:"; sed 's/^/  /' "$qt_conf"
    plugins=$(sed -n 's/^[[:space:]]*Plugins[[:space:]]*=[[:space:]]*\([^[:space:]]*\)[[:space:]]*$/\1/p' \
              "$qt_conf" | tail -1)
    if [ -n "$plugins" ] && [ "$plugins" != "PlugIns" ]; then
        echo "error: $qt_conf points Plugins at '$plugins', expected PlugIns" >&2
        exit 1
    fi
fi

echo "== remaining libraries"
python3 "$here/bundle_libs.py" "$app" --min-os-out "$out/min-os.txt"
min_os=$(cat "$out/min-os.txt")
sed "s/__MIN_OS__/$min_os/" "$build/macos/Info.plist" > "$contents/Info.plist"
plutil -lint "$contents/Info.plist"

echo "== ad-hoc signing"
# Inner code first, then the bundle (--deep also covers the Qt frameworks).
find "$app" -type f \( -name '*.dylib' -o -name '*.so' \) -exec codesign --force -s - {} \;
codesign --force -s - "$contents/MacOS/alphashape-cli"
codesign --force --deep -s - "$app"
codesign --verify --deep --strict --verbose=2 "$app"

echo "== size breakdown"
du -sk "$contents"/* "$contents/Frameworks/"* 2>/dev/null | sort -n | tail -15
du -sh "$app"
echo "built $app (minimum macOS $min_os)"
