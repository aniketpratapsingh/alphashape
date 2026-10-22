#!/bin/bash
# guard.sh -- run a command under a wall-clock limit and a free-memory floor.
#
#   tools/guard.sh <seconds> <command> [args...]
#
# The command runs in its own process group. Every 2 s the guard checks
#   * elapsed time against <seconds>, and
#   * the percent of memory the system considers available, against
#     GUARD_MIN_FREE_PCT (default 13 = ~2 GB on a 16 GB machine).
#     macOS: kern.memorystatus_level (0-100). Linux: MemAvailable/MemTotal from
#     /proc/meminfo, which excludes reclaimable page cache the same way.
# On a breach the whole group gets SIGTERM, then SIGKILL 5 s later.
# Exit status: the command's own, 124 on timeout, 125 on low memory.

set -u
limit="$1"; shift
min_free="${GUARD_MIN_FREE_PCT:-13}"

# Prints the percentage of memory the kernel considers available (0-100).
# Falls back to 100 (i.e. never trips) when the value cannot be read, which
# matches the macOS behaviour of treating a missing sysctl as healthy.
free_pct() {
    case "$(uname -s)" in
        Linux)
            awk '
                /^MemTotal:/     { total = $2 }
                /^MemAvailable:/ { avail = $2 }
                END               { if (total > 0) print int(avail * 100 / total); else print 100 }
            ' /proc/meminfo 2>/dev/null || echo 100
            ;;
        *)
            sysctl -n kern.memorystatus_level 2>/dev/null || echo 100
            ;;
    esac
}

perl -e 'setpgrp(0,0); exec @ARGV or die "exec failed: $!"' "$@" &
pid=$!
start=$SECONDS
reason=""

while kill -0 "$pid" 2>/dev/null; do
    sleep 2
    kill -0 "$pid" 2>/dev/null || break
    if (( SECONDS - start > limit )); then
        reason="timeout after ${limit}s"; status=124; break
    fi
    free_now=$(free_pct)
    if (( free_now < min_free )); then
        reason="free memory ${free_now}% < ${min_free}%"; status=125; break
    fi
done

if [[ -n "$reason" ]]; then
    echo "[guard] killing '$*': $reason" >&2
    kill -TERM -- "-$pid" 2>/dev/null
    sleep 5
    kill -KILL -- "-$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    exit "$status"
fi
wait "$pid"
