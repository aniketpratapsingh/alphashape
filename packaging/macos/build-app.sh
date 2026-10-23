#!/usr/bin/env bash
# Assembles alphashape.app from a finished CMake build tree (arm64, ad-hoc signed).
#
# Usage: build-app.sh <build-dir> <out-dir>
#   needs on PATH: macdeployqt (from the Qt 6.8 install), rsvg-convert, iconutil, codesign,
#   python3. Homebrew supplies the libraries being bundled (proj, MoltenVK, Vulkan loader, ...).
#
# Environment overrides: PROJ_DATA_SRC, MOLTENVK_LIB, MOLTENVK_ICD, VULKAN_LOADER_LIB.
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
(cd "$PROJ_DATA_SRC" && find . -maxdepth 1 -type f \( -name '*.db' -o -name '*.ini' -o -name '*.tif' \
    -o -name '*.gtx' -o -name '*.json' \) -exec cp {} "$contents/Resources/share/proj/" \;)

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

du -sh "$app"
echo "built $app (minimum macOS $min_os)"
