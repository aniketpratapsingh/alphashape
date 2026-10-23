#!/usr/bin/env bash
# Runs inside a clean ubuntu:24.04 container with only the libraries a desktop provides.
#   smoke-appimage.sh <file.AppImage> [fixture-dir]
set -euo pipefail
img=$(readlink -f "$1")
fixtures="${2:-}"
export DEBIAN_FRONTEND=noninteractive
apt_retry() { for i in 1 2 3 4; do timeout 300 apt-get -q -o Acquire::Retries=3 "$@" && return 0; echo "apt attempt $i failed"; done; return 1; }
apt_retry update
# What an AppImage expects of the host (linuxdeploy never bundles these).
apt_retry install -y --no-install-recommends file libgl1 libegl1 libopengl0 libglib2.0-0t64 \
    libfontconfig1 libfreetype6 libdbus-1-3 libxkbcommon0 libx11-6 libxcb1 libvulkan1 libstdc++6

cp "$img" /tmp/alphashape.AppImage   # the source mount is read-only
img=/tmp/alphashape.AppImage
chmod +x "$img"
export APPIMAGE_EXTRACT_AND_RUN=1
echo "== size =="; ls -lh "$img"

echo "== extract + library audit =="
work=$(mktemp -d); cd "$work"
"$img" --appimage-extract >/dev/null
fail=0
while IFS= read -r f; do
    file -b "$f" | command grep -q ELF || continue
    missing=$(ldd "$f" 2>&1 | command grep 'not found' || true)
    if [ -n "$missing" ]; then echo "UNRESOLVED in $f:"; echo "$missing"; fail=1; fi
done < <(find squashfs-root/usr -type f \( -name '*.so*' -o -path '*/bin/*' \))
[ "$fail" = 0 ] || { echo "ldd audit failed"; exit 1; }
echo "all binaries resolve"
du -sh squashfs-root

echo "== CLI (via the AppImage, no FUSE) =="
"$img" --appimage-extract-and-run cli --list-tasks | tee tasks.txt
test "$(wc -l < tasks.txt)" -gt 5
if [ -n "$fixtures" ] && [ -f "$fixtures/baseline.jpg" ]; then
    "$img" --appimage-extract-and-run cli --new "$work/smoke.psx" --add-images "$fixtures/baseline.jpg"
    test -s "$work/smoke.psx"
fi
echo "== PROJ data is found inside the image =="
test -f squashfs-root/usr/share/proj/proj.db

echo "== GUI (offscreen) =="
out=$(QT_QPA_PLATFORM=offscreen "$img" --appimage-extract-and-run --offscreen-smoke)
echo "$out"
echo "$out" | command grep -q "gui ready"
echo "appimage smoke OK"
