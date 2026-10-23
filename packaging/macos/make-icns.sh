#!/usr/bin/env bash
# Renders the SVG app icon into an .icns (needs rsvg-convert and iconutil).
# Usage: make-icns.sh <icon.svg> <out.icns>
set -euo pipefail
svg=$1
out=$2
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
iconset="$work/alphashape.iconset"
mkdir "$iconset"
for size in 16 32 128 256 512; do
    rsvg-convert -w "$size" -h "$size" "$svg" -o "$iconset/icon_${size}x${size}.png"
    rsvg-convert -w $((size * 2)) -h $((size * 2)) "$svg" -o "$iconset/icon_${size}x${size}@2x.png"
done
iconutil -c icns "$iconset" -o "$out"
