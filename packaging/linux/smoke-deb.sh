#!/usr/bin/env bash
# Runs inside a clean ubuntu:24.04 container: install the .deb with apt and exercise it.
#   smoke-deb.sh <package.deb> [fixture-dir]
set -euo pipefail
deb=$(readlink -f "$1")
fixtures="${2:-}"
export DEBIAN_FRONTEND=noninteractive

apt_retry() { for i in 1 2 3 4; do timeout 300 apt-get -q -o Acquire::Retries=3 "$@" && return 0; echo "apt attempt $i failed"; done; return 1; }
apt_retry update
# No Recommends: proves the package works with no GPU driver installed (CPU fallback).
apt_retry install -y --no-install-recommends "$deb"

echo "== package =="
dpkg-deb -f "$deb" Package Version Depends Recommends Installed-Size
dpkg -L alphashape | command grep -E '/(bin|dist-packages|applications|metainfo|icons)/' || true

echo "== library audit =="
fail=0
while IFS= read -r f; do
    [ -f "$f" ] || continue
    file -b "$f" 2>/dev/null | command grep -q ELF || continue
    missing=$(ldd "$f" 2>&1 | command grep -E 'not found|version .* not found' || true)
    if [ -n "$missing" ]; then echo "UNRESOLVED in $f:"; echo "$missing"; fail=1; fi
done < <(dpkg -L alphashape)
[ "$fail" = 0 ] || { echo "ldd audit failed"; exit 1; }
echo "all binaries resolve"

echo "== CLI =="
alphashape-cli --help
alphashape-cli --list-tasks | tee /tmp/tasks.txt
test "$(wc -l < /tmp/tasks.txt)" -gt 5
if [ -n "$fixtures" ] && [ -f "$fixtures/baseline.jpg" ]; then
    alphashape-cli --new /tmp/smoke.psx --add-images "$fixtures/baseline.jpg"
    test -s /tmp/smoke.psx
fi

echo "== Python =="
python3 -c "import alphashape, sys; print('alphashape from', alphashape.__file__, 'python', sys.version.split()[0])"

echo "== GUI (offscreen) =="
out=$(QT_QPA_PLATFORM=offscreen alphashape-gui --offscreen-smoke)
echo "$out"
echo "$out" | command grep -q "gui ready"
alphashape-gui --version

echo "== desktop integration =="
test -f /usr/share/applications/org.alphashape.app.desktop
test -f /usr/share/icons/hicolor/scalable/apps/org.alphashape.app.svg
test -f /usr/share/metainfo/org.alphashape.app.metainfo.xml
echo "deb smoke OK"
