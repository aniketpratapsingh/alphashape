#!/usr/bin/env bash
# Resilient `apt-get install` for CI.
#
# The GitHub-hosted runner's default mirror (azure.archive.ubuntu.com) is intermittently
# throttled to a crawl. A plain 4x `apt-get -o Acquire::Retries=3` retry loop then spends
# twelve minutes timing out against the same dead mirror before the job even starts compiling.
#
# This script replaces that loop in every workflow:
#
#   1. Cache: when actions/cache restored the .deb cache for this exact package set, install
#      with `--no-download` (no network at all). Anything missing falls through to step 2.
#   2. Rotation: retry with a per-attempt timeout, rewriting the Ubuntu sources to the next
#      mirror after every failure, starting from the mirror the image is already configured
#      with:
#          azure.archive.ubuntu.com -> archive.ubuntu.com -> mirrors.edge.kernel.org
#          -> mirror.math.princeton.edu
#      Both the deb822 form (/etc/apt/sources.list.d/*.sources, Ubuntu 24.04's
#      ubuntu.sources) and the classic /etc/apt/sources.list are rewritten.
#   3. `apt-get update` refreshes only the Ubuntu source files (third-party sources on the
#      image are never contacted), and every apt invocation is bounded by `timeout`.
#   4. The mirror that worked is printed, so a run's log says which one was healthy.
#
# Usage:
#   .github/ci/apt-install.sh --print-key cmake ninja-build   # cache key -> stdout/$GITHUB_OUTPUT
#   .github/ci/apt-install.sh cmake ninja-build               # update + install
#
# Environment:
#   APT_DEB_CACHE_DIR   user-writable cache, default ~/.cache/apt-debs. Holds the .debs, the
#                       apt lists they were resolved against, and the mirror that served them.
#   APT_CACHE_HIT       "true" when actions/cache restored APT_DEB_CACHE_DIR.
#   APT_ATTEMPT_TIMEOUT seconds per `apt-get update` attempt (default 150).
#   APT_INSTALL_TIMEOUT seconds per `apt-get install` (default 300).
#   APT_PASSES          passes over the mirror list (default 1 => 4 attempts).
#   APT_CACHE_SCOPE     ref name folded into the cache key; the action sets it to
#                       github.ref_name so one branch cannot seed another's cache entry.
#   APT_SOURCES_ROOT, APT_ARCHIVES_DIR, APT_LISTS_DIR
#                       overrides for /etc/apt, /var/cache/apt/archives and /var/lib/apt/lists;
#                       used by .github/ci/test-apt-install.sh to exercise this script unprivileged.
#
# Exit status: 0 when the packages are installed, 1 when every mirror failed.

set -uo pipefail

CACHE_DIR="${APT_DEB_CACHE_DIR:-$HOME/.cache/apt-debs}"
ATTEMPT_TIMEOUT="${APT_ATTEMPT_TIMEOUT:-150}"
INSTALL_TIMEOUT="${APT_INSTALL_TIMEOUT:-300}"
PASSES="${APT_PASSES:-1}"
SOURCES_ROOT="${APT_SOURCES_ROOT:-/etc/apt}"
ARCHIVES_DIR="${APT_ARCHIVES_DIR:-/var/cache/apt/archives}"
LISTS_DIR="${APT_LISTS_DIR:-/var/lib/apt/lists}"
# Empty => run privileged commands directly. The test harness sets this to "" with stubs.
ROOT_CMD="${APT_ROOT_CMD-sudo}"

# Mirrors in rotation order: host + path prefix, no trailing slash.
MIRRORS=(
  "azure.archive.ubuntu.com/ubuntu"
  "archive.ubuntu.com/ubuntu"
  "mirrors.edge.kernel.org/ubuntu"
  "mirror.math.princeton.edu/pub/ubuntu"
)

log() { printf 'apt: %s\n' "$*"; }
die() { printf 'apt: error: %s\n' "$*" >&2; exit 1; }

as_root() {
  if [ "$(id -u)" -eq 0 ] || [ -z "$ROOT_CMD" ]; then "$@"; else $ROOT_CMD "$@"; fi
}

# Shared by every apt-get call. Acquire::Retries=1 keeps a dead mirror from multiplying the
# wait by three before the rotation gets its turn; ForceIPv4 removes the IPv6-hang class of
# stalls seen on hosted runners; Keep-Downloaded-Packages fills the .deb cache.
APT_OPTS=(
  -q
  -o Acquire::ForceIPv4=true
  -o Acquire::http::Timeout=20
  -o Acquire::https::Timeout=20
  -o Acquire::Retries=1
  -o Dpkg::Use-Pty=0
  -o APT::Keep-Downloaded-Packages=true
)

PRINT_KEY=0
if [ "${1:-}" = "--print-key" ]; then
  PRINT_KEY=1
  shift
fi

PKGS=("$@")
[ "${#PKGS[@]}" -gt 0 ] || die "no packages given"

# --- cache key ------------------------------------------------------------------------------

# Image version, architecture, ref and the sorted package list identify the cache entry: a
# change to any of them must miss the cache rather than install stale debs.
#
# The ref is part of the key because the debs are installed as root from a cache entry, and a
# workflow that checks out an arbitrary `inputs.ref` must not be able to seed the entry that
# `main` reads (CodeQL's "cache poisoning via untrusted code"). PRs read main's entry through
# restore-keys, which is safe: main's cache is trusted, a branch's is not.
os_version() {
  # shellcheck disable=SC1091
  (. /etc/os-release 2>/dev/null && printf '%s' "${VERSION_ID:-unknown}") || printf 'unknown'
}
arch() { dpkg --print-architecture 2>/dev/null || uname -m; }

cache_key() {
  local hash scope
  hash=$(printf '%s\n' "${PKGS[@]}" | LC_ALL=C sort -u | sha256sum | cut -c1-16)
  scope="${APT_CACHE_SCOPE:-}"
  if [ -n "$scope" ]; then
    printf 'apt-v4-%s-%s-%s-%s' "$(os_version)" "$(arch)" "$scope" "$hash"
  else
    printf 'apt-v4-%s-%s-%s' "$(os_version)" "$(arch)" "$hash"
  fi
}

if [ "$PRINT_KEY" -eq 1 ]; then
  key=$(cache_key)
  printf '%s\n' "$key"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    {
      printf 'key=%s\n' "$key"
      # A prefix that only matches entries main wrote; used as restore-keys for PR runs.
      printf 'shared_prefix=apt-v4-%s-%s-main-\n' "$(os_version)" "$(arch)"
    } >>"$GITHUB_OUTPUT"
  fi
  exit 0
fi

# --- sources --------------------------------------------------------------------------------

# A source file is ours when it names an Ubuntu archive, one of the rotation mirrors, or the
# GitHub runner image's mirrorlist indirection. ports.ubuntu.com (arm) and third-party
# repositories (packages.microsoft.com, ...) never match.
UBUNTU_HOST_RE='([a-z0-9.-]*\.)?(archive\.ubuntu\.com|security\.ubuntu\.com|mirrors\.edge\.kernel\.org|mirror\.math\.princeton\.edu)'
MIRRORFILE_RE='mirror\+file:([^[:space:],]+)'

SOURCE_FILES=()
MIRROR_FILES=()
discover_sources() {
  local f m
  for f in "$SOURCES_ROOT/sources.list" "$SOURCES_ROOT"/sources.list.d/*.list \
    "$SOURCES_ROOT"/sources.list.d/*.sources; do
    [ -f "$f" ] || continue
    if grep -Eq "https?://${UBUNTU_HOST_RE}|${MIRRORFILE_RE}" "$f"; then
      SOURCE_FILES+=("$f")
      # GitHub's Ubuntu images ship `URIs: mirror+file:/etc/apt/apt-mirrors.txt` instead of a
      # host, so the host to rotate lives in that list, not in the sources file.
      while read -r m; do
        [ -n "$m" ] || continue
        case " ${MIRROR_FILES[*]-} " in
          *" $m "*) continue ;;
        esac
        MIRROR_FILES+=("$m")
      done < <(grep -oE "$MIRRORFILE_RE" "$f" | sed -E "s#^mirror\\+file:##")
    fi
  done
}

# Rewrite the URI, keeping suites/components/signing untouched. Handles deb822
# (`URIs: http://host/ubuntu/`, possibly comma-separated), classic `deb http://host/ubuntu`, and
# the runner image's `mirror+file:` indirection, which is replaced by the plain mirror URI.
# The trailing slash matches the convention Ubuntu ships in both forms.
# Written through a temp file rather than `sed -i`, whose spelling differs between GNU and BSD.
rewrite_sources() {
  local mirror="$1" url="http://$1/" f tmp rc
  for f in ${SOURCE_FILES[@]+"${SOURCE_FILES[@]}"}; do
    tmp=$(mktemp) || return 1
    # Comment lines are skipped: Ubuntu ships `# See http://help.ubuntu.com/...` in
    # ubuntu.sources, and rewriting that produces a useless "mirror" the next run would cache.
    if sed -E \
      -e "/^[[:space:]]*#/!s#mirror\\+file:[^[:space:],]+#${url}#g" \
      -e "/^[[:space:]]*#/!s#https?://${UBUNTU_HOST_RE}(/[^[:space:],]*)?#${url}#g" "$f" >"$tmp"; then
      as_root cp "$tmp" "$f"
      rc=$?
      rm -f "$tmp"
      [ "$rc" -eq 0 ] || return 1
    else
      rm -f "$tmp"
      return 1
    fi
  done
}

current_mirror() {
  local f m
  for f in ${SOURCE_FILES[@]+"${SOURCE_FILES[@]}"}; do
    m=$(grep -vE '^[[:space:]]*#' "$f" | grep -oE "https?://[^[:space:],]+" | head -n1)
    if [ -n "${m:-}" ]; then printf '%s' "$m"; return 0; fi
  done
  # No host in the sources: report the highest-priority entry of the mirrorlist.
  for f in ${MIRROR_FILES[@]+"${MIRROR_FILES[@]}"}; do
    m=$(grep -vE '^[[:space:]]*#' "$f" 2>/dev/null | grep -oE 'https?://[^[:space:]]+' | head -n1)
    if [ -n "${m:-}" ]; then printf '%s' "$m"; return 0; fi
  done
  printf 'unknown'
}

# Index into MIRRORS for the mirror the image is already configured with, else 1 (the
# canonical archive). Attempt 1 therefore never rewrites a working setup, and the dead azure
# default costs only one bounded attempt before the rotation moves on.
start_index() {
  local host i
  host=$(current_mirror)
  host=${host#*://}
  host=${host%%/*}
  for i in "${!MIRRORS[@]}"; do
    case "${MIRRORS[$i]}" in
      "$host"/*) printf '%s' "$i"; return 0 ;;
    esac
  done
  printf '1'
}

# True when the sources use the runner image's `mirror+file:` indirection. apt then reads the
# host list itself and tries each entry in priority order, so attempt 1 is left untouched and
# the rotation takes over only when that whole list has failed.
uses_mirrorlist() {
  local f
  for f in ${SOURCE_FILES[@]+"${SOURCE_FILES[@]}"}; do
    grep -qE "$MIRRORFILE_RE" "$f" && return 0
  done
  return 1
}

# Refresh only the Ubuntu lists: apt is pointed at a temp sourceparts directory holding copies
# of the matched files, so unrelated repositories on the image cannot stall the job.
refresh_ubuntu_lists() {
  local dir f rc
  if [ "${#SOURCE_FILES[@]}" -eq 0 ]; then
    log "no Ubuntu source file under $SOURCES_ROOT; running a full apt-get update"
    as_root timeout "$ATTEMPT_TIMEOUT" apt-get "${APT_OPTS[@]}" update
    return $?
  fi

  dir=$(mktemp -d) || return 1
  : >"$dir/placeholder.list"
  for f in ${SOURCE_FILES[@]+"${SOURCE_FILES[@]}"}; do
    if ! as_root cp "$f" "$dir/$(basename "$f")"; then
      rm -rf "$dir"
      return 1
    fi
  done

  as_root timeout "$ATTEMPT_TIMEOUT" apt-get "${APT_OPTS[@]}" \
    -o Dir::Etc::sourcelist="$dir/placeholder.list" \
    -o Dir::Etc::sourceparts="$dir" \
    update
  rc=$?
  rm -rf "$dir"
  return $rc
}

# --- cache ----------------------------------------------------------------------------------

seed_debs_from_cache() {
  local debs=() f
  for f in "$CACHE_DIR"/*.deb; do [ -f "$f" ] && debs+=("$f"); done
  [ "${#debs[@]}" -gt 0 ] || return 1
  as_root mkdir -p "$ARCHIVES_DIR"
  # Overwrite, never skip: a deb of a different version already on the image would otherwise be
  # kept, and the cached lists would then reference versions that are not there.
  as_root cp -f "${debs[@]}" "$ARCHIVES_DIR/" 2>/dev/null
  log "cache: seeded ${#debs[@]} .deb(s) into $ARCHIVES_DIR"
  return 0
}

restore_lists_from_cache() {
  [ -d "$CACHE_DIR/lists" ] || return 1
  local f n=0
  # The image ships its own lists, from a different mirror and possibly a different snapshot of
  # the archive. Overwriting them is the whole point: the cached .debs are only usable against
  # the exact lists that were cached with them. (`cp -n` here silently kept the image's lists,
  # and the offline install then failed with "Unable to fetch some archives".)
  as_root rm -f "$LISTS_DIR"/* >/dev/null 2>&1
  as_root mkdir -p "$LISTS_DIR" || return 1
  for f in "$CACHE_DIR"/lists/*; do
    [ -f "$f" ] || continue
    as_root cp -f "$f" "$LISTS_DIR/" 2>/dev/null && n=$((n + 1))
  done
  [ "$n" -gt 0 ] || return 1
  log "cache: restored $n apt list file(s)"
  return 0
}

# The .deb files and the lists are keyed by the URI that served them, so an offline install only
# works against the exact sources that produced them. Those are cached verbatim and put back
# before the offline attempt; anything else (a rotated mirror, the image's own mirrorlist) would
# make apt reject every cached file as a URI mismatch.
#
# Names carry the location they came from: "d__x" is sources.list.d/x, "r__x" is /etc/apt/x.
restore_sources_from_cache() {
  [ -d "$CACHE_DIR/sources" ] || return 1
  local f base n=0
  for f in "$CACHE_DIR"/sources/*; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    case "$base" in
      d__*)
        as_root mkdir -p "$SOURCES_ROOT/sources.list.d"
        as_root cp "$f" "$SOURCES_ROOT/sources.list.d/${base#d__}" 2>/dev/null || continue
        ;;
      r__*)
        as_root cp "$f" "$SOURCES_ROOT/${base#r__}" 2>/dev/null || continue
        ;;
      *) continue ;;
    esac
    n=$((n + 1))
  done
  [ "$n" -gt 0 ] || return 1
  log "cache: restored $n source file(s)"
  return 0
}

# Keep the debs apt downloaded, the lists they were resolved against, and the mirror that
# served them, so the next run with this key can install with no network at all.
harvest_cache() {
  local f n=0 base
  # The cache lives under $HOME and must stay owned by the job user: creating it with sudo
  # would leave a root-owned ~/.cache behind, and every later tool that writes there (ccache
  # is the first) would fail with "Permission denied".
  mkdir -p "$CACHE_DIR/lists" "$CACHE_DIR/sources" || return 1

  for f in "$ARCHIVES_DIR"/*.deb; do
    [ -f "$f" ] || continue
    as_root cp -f "$f" "$CACHE_DIR/" 2>/dev/null && n=$((n + 1))
  done
  log "cache: stored $n .deb(s)"

  n=0
  for f in "$LISTS_DIR"/*; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in lock) continue ;; esac
    as_root cp -f "$f" "$CACHE_DIR/lists/" 2>/dev/null && n=$((n + 1))
  done
  log "cache: stored $n apt list file(s)"

  n=0
  for f in ${SOURCE_FILES[@]+"${SOURCE_FILES[@]}"}; do
    base=$(basename "$f")
    if [ "$(dirname "$f")" = "$SOURCES_ROOT" ]; then
      base="r__$base"
    else
      base="d__$base"
    fi
    as_root cp "$f" "$CACHE_DIR/sources/$base" 2>/dev/null && n=$((n + 1))
  done
  log "cache: stored $n source file(s)"

  as_root chown -R "$(id -u):$(id -g)" "$CACHE_DIR" 2>/dev/null
}

# Install without touching the network. Non-zero means something the packages need is absent
# from the cache, which sends the caller down the rotation path.
try_offline_install() {
  log "cache hit: trying apt-get install --no-download"
  restore_sources_from_cache || log "cache: no cached sources; using the image's own"
  restore_lists_from_cache || log "cache: no cached apt lists; using the lists on the image"

  as_root timeout "$INSTALL_TIMEOUT" apt-get "${APT_OPTS[@]}" \
    install -y --no-install-recommends --no-download "${PKGS[@]}"
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    log "success from the cache: no network used"
  else
    log "cache could not satisfy the package set (rc=$rc); falling back to the mirrors"
  fi
  return $rc
}

# --- network path ---------------------------------------------------------------------------

network_install() {
  local total=$(( ${#MIRRORS[@]} * PASSES )) attempt=0 mirror rc start keep_mirrorlist=0
  start=$(start_index)
  uses_mirrorlist && keep_mirrorlist=1

  while [ "$attempt" -lt "$total" ]; do
    mirror="${MIRRORS[$(( (start + attempt) % ${#MIRRORS[@]} ))]}"
    attempt=$((attempt + 1))

    # Attempt 1 on the runner image keeps its own mirrorlist: apt walks every entry there
    # itself, which is a better first try than any single mirror. The rotation takes over from
    # attempt 2, once that whole list has failed.
    if [ "$attempt" -eq 1 ] && [ "$keep_mirrorlist" -eq 1 ]; then
      log "attempt $attempt/$total: runner mirrorlist (apt tries each entry in priority order)"
      if refresh_ubuntu_lists; then
        log "mirrors listed in the runner mirrorlist are responsive"
        keep_mirrorlist=0
      else
        log "attempt $attempt failed: every mirror in the runner list timed out"
        [ "$attempt" -lt "$total" ] && sleep 3
        continue
      fi
    else
      log "attempt $attempt/$total: mirror http://$mirror"
      if ! rewrite_sources "$mirror"; then
        log "attempt $attempt: could not rewrite sources for $mirror"
        continue
      fi
      if ! refresh_ubuntu_lists; then
        log "attempt $attempt failed: apt-get update timed out or errored on $mirror"
        [ "$attempt" -lt "$total" ] && sleep 3
        continue
      fi
    fi

    as_root timeout "$INSTALL_TIMEOUT" apt-get "${APT_OPTS[@]}" \
      install -y --no-install-recommends "${PKGS[@]}"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      log "success on attempt $attempt/$total (sources: $(current_mirror))"
      harvest_cache
      return 0
    fi
    log "attempt $attempt failed: apt-get install rc=$rc"
    [ "$attempt" -lt "$total" ] && sleep 3
  done

  log "all $total attempts failed; mirrors tried: ${MIRRORS[*]}"
  return 1
}

# --- main -----------------------------------------------------------------------------------

discover_sources
log "sources: ${SOURCE_FILES[*]:-none}"
log "current mirror: $(current_mirror)"

if [ "${APT_CACHE_HIT:-}" = "true" ] && seed_debs_from_cache && try_offline_install; then
  exit 0
fi

network_install || die "could not install: ${PKGS[*]}"
