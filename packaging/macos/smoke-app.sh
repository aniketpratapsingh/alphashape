#!/usr/bin/env bash
# Install-and-run check for the macOS DMG, executed on a fresh runner.
# Usage: smoke-app.sh <alphashape-*.dmg> <data-dir>
#   python3 on PATH must be the version the bundled module was built for.
set -euo pipefail

dmg=$1
data=$2
work=$(mktemp -d)
mnt="$work/mnt"
mkdir "$mnt"
hdiutil attach "$dmg" -mountpoint "$mnt" -nobrowse -readonly
trap 'hdiutil detach "$mnt" -force >/dev/null 2>&1 || true; rm -rf "$work"' EXIT

echo "== image contents"
ls -la "$mnt"
[ -L "$mnt/Applications" ] || { echo "error: no /Applications link in the DMG" >&2; exit 1; }

ditto "$mnt/alphashape.app" "$work/alphashape.app"
app="$work/alphashape.app"
cli="$app/Contents/MacOS/alphashape-cli"
gui="$app/Contents/MacOS/alphashape-gui"

echo "== bundle metadata"
plutil -p "$app/Contents/Info.plist"
[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$app/Contents/Info.plist")" = org.alphashape.app ]
[ -f "$app/Contents/Resources/alphashape.icns" ]
codesign --verify --deep --strict --verbose=2 "$app"
lipo -archs "$cli" | grep -qx arm64

echo "== no Homebrew or /usr/local references in any Mach-O"
count=0
bad=0
while IFS= read -r f; do
    if ! head -c4 "$f" | xxd -p | grep -qE '^(cffaedfe|feedfacf|cafebabe|bebafeca)$'; then continue; fi
    count=$((count + 1))
    if otool -L "$f" | tail -n +2 | grep -E '/opt/homebrew|/usr/local|/Users/|/Volumes/'; then
        echo "error: $f references a build-machine path" >&2
        bad=1
    fi
    # @executable_path means "next to the running process": the bundle's MacOS directory for the
    # two executables, but python3 for the module in Resources/python. Every reference must be
    # @rpath, which the bundle's contents can actually satisfy.
    if otool -L "$f" | tail -n +2 | grep -q '@executable_path/'; then
        otool -L "$f" | tail -n +2 | grep '@executable_path/' >&2
        echo "error: $f uses @executable_path, which is not the bundle for a loaded module" >&2
        bad=1
    fi
    if otool -l "$f" | grep -A2 LC_RPATH | grep -E 'path /'; then
        echo "error: $f has an absolute rpath" >&2
        bad=1
    fi
done < <(find "$app" -type f ! -name '*.png' ! -name '*.icns')
echo "checked $count Mach-O files"
[ "$count" -gt 20 ] || { echo "error: suspiciously few Mach-O files" >&2; exit 1; }
[ "$bad" = 0 ] || exit 1

echo "== CLI"
"$cli" --help
"$cli" --list-tasks | tee "$work/tasks.txt"
grep -q '^AddPhotos' "$work/tasks.txt"

echo "== PROJ data comes from the bundle"
"$cli" --proj-info | tee "$work/proj.txt"
grep -q "alphashape.app/Contents/Resources/share/proj/proj.db" "$work/proj.txt"
# The bundle ships a curated subset of proj-data, not Homebrew's whole collection. The selector
# writes SELECTED.txt with exactly the names it copied (which EGM96 spelling that platform's
# package uses, and whether conus was available at all, both vary), so check the bundle against
# the manifest rather than against one distribution's filenames. A regression that copies every
# grid is caught by the DMG size guard; this catches the opposite mistake of trimming the
# essentials away.
proj="$app/Contents/Resources/share/proj"
[ -f "$proj/proj.db" ] || { echo "error: no proj.db in the bundle" >&2; exit 1; }
[ -f "$proj/SELECTED.txt" ] || { echo "error: no SELECTED.txt manifest in the bundle" >&2; exit 1; }
grep -qx 'proj.db' "$proj/SELECTED.txt" || {
    echo "error: SELECTED.txt does not list proj.db" >&2; exit 1; }
missing=0
while IFS= read -r name; do
    [ -n "$name" ] || continue
    if [ ! -f "$proj/$name" ]; then
        echo "error: SELECTED.txt lists $name, which is not in the bundle" >&2
        missing=1
    fi
done <"$proj/SELECTED.txt"
[ "$missing" = 0 ] || exit 1
# EGM96 is the geoid the vertical-CRS support names; either spelling is accepted, but one must
# be there.
if [ ! -f "$proj/us_nga_egm96_15.tif" ] && [ ! -f "$proj/egm96_15.gtx" ]; then
    echo "error: no EGM96 grid (us_nga_egm96_15.tif or egm96_15.gtx) in the bundle" >&2
    exit 1
fi
proj_mb=$(( $(du -sk "$proj" | cut -f1) / 1024 ))
echo "bundled PROJ data: $(ls -1 "$proj" | wc -l | tr -d ' ') files, ${proj_mb} MB"
[ "$proj_mb" -le 64 ] || {
    echo "error: the PROJ data is ${proj_mb} MB; the bundle must ship a curated subset" >&2
    exit 1; }

echo "== project from images (JPEG and project I/O)"
"$cli" --new "$work/smoke.psx" --add-images "$data"/aerial_r0c0.jpg "$data"/aerial_r0c1.jpg
test -s "$work/smoke.psx"

echo "== GPU: Vulkan loader and MoltenVK from the bundle"
DYLD_PRINT_LIBRARIES=1 "$cli" --gpu-info >"$work/gpu.out" 2>"$work/gpu.err" || {
    cat "$work/gpu.out" "$work/gpu.err"; echo "error: no Vulkan device enumerated" >&2; exit 1; }
cat "$work/gpu.out"
grep -q '^gpu device 0:' "$work/gpu.out"
grep -q "alphashape.app/Contents/Frameworks/libvulkan.1.dylib" "$work/gpu.err"
grep -q "alphashape.app/Contents/Frameworks/libMoltenVK.dylib" "$work/gpu.err"

echo "== GUI (offscreen)"
# The offscreen platform plugin is not one macdeployqt picks (it deploys cocoa only), so
# build-app.sh copies it in; check it is present before handing the GUI a headless platform.
[ -f "$app/Contents/PlugIns/platforms/libqoffscreen.dylib" ] || {
    echo "error: the bundle is missing PlugIns/platforms/libqoffscreen.dylib" >&2
    ls "$app/Contents/PlugIns/platforms" >&2 || true
    exit 1; }
QT_QPA_PLATFORM=offscreen "$gui" --offscreen-smoke | tee "$work/gui.txt"
grep -q 'gui ready' "$work/gui.txt"

echo "== Python module"
PYTHONPATH="$app/Contents/Resources/python" python3 - <<'PY'
import sys
import alphashape
assert "alphashape.app/Contents/Resources/python" in alphashape.__file__, alphashape.__file__
tasks = alphashape.list_tasks()
assert "AddPhotos" in tasks, tasks
print("alphashape python module ok:", alphashape.__file__, len(tasks), "tasks")
PY

echo "macOS bundle smoke test passed"
