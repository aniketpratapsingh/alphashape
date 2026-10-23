#!/usr/bin/env bash
# Wraps alphashape.app and an /Applications link into a compressed disk image.
# Usage: build-dmg.sh <alphashape.app> <out.dmg>
set -euo pipefail
app=$1
dmg=$2
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
ditto "$app" "$stage/alphashape.app"
ln -s /Applications "$stage/Applications"
rm -f "$dmg"
hdiutil create -volname alphashape -srcfolder "$stage" -fs HFS+ -format UDZO -ov "$dmg"
ls -lh "$dmg"
