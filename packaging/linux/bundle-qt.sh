#!/usr/bin/env bash
# Copy a private Qt (libraries + the plugins the GUI needs) into an install tree.
#
#   bundle-qt.sh <install-root> <qt-root>
#
# <install-root> is the prefix as staged on disk (e.g. $DESTDIR/usr). Qt is relocatable: it finds
# its plugins relative to libQt6Core, so lib/alphashape/qt/{lib,plugins} needs no qt.conf. The
# alphashape-gui binary reaches the libraries through its install RUNPATH.
set -euo pipefail

root="$1"
qt="$2"
dest="$root/lib/alphashape/qt"
gui="$root/bin/alphashape-gui"

[ -x "$gui" ] || { echo "bundle-qt: $gui missing (install the runtime component first)" >&2; exit 1; }
mkdir -p "$dest/lib" "$dest/plugins"

copy_plugins() {
    local dir=$1; shift
    mkdir -p "$dest/plugins/$dir"
    local p
    for p in "$@"; do
        for f in "$qt/plugins/$dir"/$p; do
            if [ -e "$f" ]; then cp -L "$f" "$dest/plugins/$dir/"; fi
        done
    done
}
copy_plugins platforms 'libqxcb.so' 'libqoffscreen.so' 'libqminimal.so'
copy_plugins xcbglintegrations '*.so'
copy_plugins platforminputcontexts '*.so'
copy_plugins imageformats 'libqjpeg.so' 'libqgif.so' 'libqico.so' 'libqsvg.so'
copy_plugins iconengines 'libqsvgicon.so'
copy_plugins tls '*.so'
# Remove directories that ended up empty (e.g. a Qt without the svg plugins).
find "$dest/plugins" -type d -empty -delete

# Transitive closure of Qt-owned libraries: everything the GUI and the plugins resolve to under
# the Qt root. Libraries from the distribution stay system dependencies.
export LD_LIBRARY_PATH="$qt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
declare -A seen=()
queue=("$gui")
while IFS= read -r -d '' so; do queue+=("$so"); done < <(find "$dest/plugins" -name '*.so' -print0)

while [ ${#queue[@]} -gt 0 ]; do
    item="${queue[0]}"; queue=("${queue[@]:1}")
    while read -r _ _ path _; do
        case "$path" in
            "$qt"/*)
                real=$(readlink -f "$path")
                if [ -z "${seen[$real]:-}" ]; then
                    seen[$real]=1
                    # Keep the soname the consumers asked for (libQt6Core.so.6 -> real file).
                    cp -L "$path" "$dest/lib/$(basename "$path")"
                    queue+=("$real")
                fi ;;
        esac
    done < <(ldd "$item" | grep '=> /' || true)
done

# The soname links Qt libraries expect between themselves must all exist as files.
strip --strip-unneeded "$dest"/lib/*.so* "$dest"/plugins/*/*.so 2>/dev/null || true
echo "bundle-qt: $(ls "$dest/lib" | wc -l) libraries, $(find "$dest/plugins" -name '*.so' | wc -l) plugins, $(du -sh "$dest" | cut -f1)"
