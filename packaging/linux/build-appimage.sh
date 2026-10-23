#!/usr/bin/env bash
# Assemble the AppImage from a build tree configured with
#   -DALPHASHAPE_BUNDLE_QT=OFF -DALPHASHAPE_BUNDLE_PROJ_DATA=ON
#
#   build-appimage.sh <build-dir> <output.AppImage>
#
# Needs QT_ROOT_DIR (the Qt 6.8 install, as set by install-qt-action). linuxdeploy bundles every
# non-glibc dependency; Qt comes from linuxdeploy-plugin-qt. FUSE is not required: the tools run
# with APPIMAGE_EXTRACT_AND_RUN.
set -euo pipefail

build=$(readlink -f "$1")
out=$(readlink -f "$2")
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${QT_ROOT_DIR:?QT_ROOT_DIR must point at the Qt 6.8 install}"

LINUXDEPLOY_URL=https://github.com/linuxdeploy/linuxdeploy/releases/download/1-alpha-20250213-2/linuxdeploy-x86_64.AppImage
QT_PLUGIN_URL=https://github.com/linuxdeploy/linuxdeploy-plugin-qt/releases/download/1-alpha-20250213-1/linuxdeploy-plugin-qt-x86_64.AppImage
APPIMAGETOOL_URL=https://github.com/AppImage/appimagetool/releases/download/1.9.0/appimagetool-x86_64.AppImage

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
for u in "$LINUXDEPLOY_URL" "$QT_PLUGIN_URL" "$APPIMAGETOOL_URL"; do
    curl -fsSL --retry 4 --max-time 120 -o "$work/$(basename "$u")" "$u"
done
chmod +x "$work"/*.AppImage

export APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64
appdir="$work/AppDir"
DESTDIR="$appdir" cmake --install "$build" --prefix /usr --component runtime
test -f "$appdir/usr/share/proj/proj.db"

export QMAKE="$QT_ROOT_DIR/bin/qmake"
export LD_LIBRARY_PATH="$QT_ROOT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export EXTRA_PLATFORM_PLUGINS="libqoffscreen.so"
export EXTRA_QT_PLUGINS="svg;iconengines;imageformats"
export PATH="$work:$PATH"
cp "$here/AppRun" "$work/AppRun"

"$work/linuxdeploy-x86_64.AppImage" --appdir "$appdir" \
    --plugin qt \
    --executable "$appdir/usr/bin/alphashape-gui" \
    --executable "$appdir/usr/bin/alphashape-cli" \
    --desktop-file "$appdir/usr/share/applications/org.alphashape.app.desktop" \
    --icon-file "$appdir/usr/share/icons/hicolor/scalable/apps/org.alphashape.app.svg" \
    --icon-filename org.alphashape.app \
    --custom-apprun "$work/AppRun"

du -sh "$appdir"
"$work/appimagetool-x86_64.AppImage" --no-appstream "$appdir" "$out"
chmod +x "$out"
ls -lh "$out"
size=$(stat -c %s "$out")
if [ "$size" -gt $((400 * 1024 * 1024)) ]; then
    echo "AppImage is $((size / 1048576)) MB, over the 400 MB budget" >&2
    exit 1
fi
