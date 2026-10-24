#!/usr/bin/env bash
# Wraps alphashape.app and an /Applications link into a compressed disk image.
# Usage: build-dmg.sh <alphashape.app> <out.dmg>
#
# Format choice, measured on the arm64 bundle:
#   UDZO (default)      41.2 MB on the code alone -- hdiutil's default zlib level is 1
#   UDZO zlib-level=9   38.2 MB
#   ULFO (lzfse)        37.2 MB, and it mounts quickly
#   UDBZ (bzip2)        35.9 MB, but mounting is noticeably slower
# ULFO is the best size/speed trade for a download. -fs HFS+ keeps the image readable on
# every macOS that the bundle's minimum version allows.
set -euo pipefail
app=$1
dmg=$2
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
ditto "$app" "$stage/alphashape.app"
ln -s /Applications "$stage/Applications"
rm -f "$dmg"
if ! hdiutil create -volname alphashape -srcfolder "$stage" -fs HFS+ -format ULFO -ov "$dmg"; then
    echo "ULFO unavailable on this host; falling back to UDZO at zlib level 9" >&2
    hdiutil create -volname alphashape -srcfolder "$stage" -fs HFS+ -format UDZO \
        -imagekey zlib-level=9 -ov "$dmg"
fi
# A truncated or corrupt image is a download nobody can mount, and CI would only find out in the
# fresh-runner smoke job. Verify checksums against the image's partition map here instead.
hdiutil verify "$dmg"
ls -lh "$dmg"
