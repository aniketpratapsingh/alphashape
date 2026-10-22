#!/usr/bin/env bash
# Stage a self-contained PROJ data directory (M12 §4.1).
#
# Copies proj.db, proj.ini, the EGM geoid grids and the NTv2 grids out of the
# installed PROJ data package into tools/proj-data/ as real files — not
# symlinks — so a packaged build can resolve <exe_dir>/share/proj with no
# system dependency. A packaged build copies this directory the same way.
#
# Usage:  tools/stage_proj_data.sh [source-data-dir] [destination]
#
# The default source is the data directory of the `proj` on PATH; the default
# destination is <repo>/tools/proj-data. Re-running is cheap: unchanged files
# are skipped by rsync-style size/mtime comparison, and a file that vanished
# from the source package is left in place (the destination is also where a
# project may have added custom grids by hand — never delete there).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
engine_root="$(cd "${here}/.." && pwd)"
repo_root="$(cd "${engine_root}/.." && pwd)"

source_dir="${1:-}"
if [ -z "${source_dir}" ]; then
    proj_bin="$(command -v proj || true)"
    if [ -z "${proj_bin}" ]; then
        echo "stage_proj_data: proj not on PATH; pass the data directory" >&2
        echo "  macOS:  /opt/homebrew/opt/proj/share/proj" >&2
        echo "  Ubuntu: /usr/share/proj" >&2
        exit 2
    fi
    prefix="$(cd "$(dirname "${proj_bin}")/.." && pwd)"
    for candidate in "${prefix}/share/proj" /usr/share/proj /opt/homebrew/share/proj; do
        if [ -f "${candidate}/proj.db" ]; then
            source_dir="${candidate}"
            break
        fi
    done
fi
if [ -z "${source_dir}" ] || [ ! -f "${source_dir}/proj.db" ]; then
    echo "stage_proj_data: no proj.db under '${source_dir:-<unset>}'" >&2
    exit 2
fi

destination="${2:-${repo_root}/tools/proj-data}"
mkdir -p "${destination}"

copied=0
skipped=0
copy() {
    local name="$1"
    if [ ! -e "${source_dir}/${name}" ]; then
        return 0
    fi
    if [ -e "${destination}/${name}" ] && \
       [ "${source_dir}/${name}" -nt "${destination}/${name}" ]; then
        :
    elif [ -e "${destination}/${name}" ]; then
        skipped=$((skipped + 1))
        return 0
    fi
    cp -p "${source_dir}/${name}" "${destination}/${name}"
    copied=$((copied + 1))
}

# The database first: without it nothing else matters.
for name in proj.db proj.ini; do
    copy "${name}"
done

# Geoids: EGM96 (15') and EGM2008 (2.5') are the two the engine supports out of
# the box. The naming differs by package: PROJ-data (and Homebrew) ship
# GeoTIFF grids (us_nga_egm96_15.tif), while the Debian/Ubuntu proj-data
# package still ships the older GTX EGM96 grid (egm96_15.gtx). Both are copied
# when present; a missing one is simply skipped, since PROJ finds whichever
# the package installed. The README files carry the NGA attribution to ship.
for name in us_nga_egm96_15.tif us_nga_egm08_25.tif egm96_15.gtx \
            us_nga_egm96_15.tif.README.txt us_nga_egm08_25.tif.README.txt; do
    copy "${name}"
done

# NTv2 / datum-shift grids, plus the PROJ READMEs that accompany them. These
# are the grids the M12 datum-transformation tests reference.
for name in ca_nrc_ntv2_0.tif us_noaa_conus.tif \
            ca_nrc_ntv2_0.tif.README.txt us_noaa_conus.tif.README.txt; do
    copy "${name}"
done

echo "stage_proj_data: ${source_dir} -> ${destination}"
echo "  copied ${copied}, kept ${skipped}, $(du -sh "${destination}" | cut -f1) total"
if [ ! -f "${destination}/proj.db" ]; then
    echo "stage_proj_data: proj.db missing from the destination" >&2
    exit 1
fi
