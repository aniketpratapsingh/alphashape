#!/usr/bin/env bash
# Tests for .github/ci/apt-install.sh. Runs unprivileged: sudo, apt-get and dpkg are replaced
# by stubs on PATH, and the script's filesystem roots are redirected into a temp directory.
#
# What is proven here:
#   1. A dead mirror is rotated past: the failed mirror is logged, the next one is tried, and
#      the log names the mirror that succeeded.
#   2. Both source formats are rewritten (deb822 ubuntu.sources and classic sources.list),
#      https URIs are normalised, and third-party sources are left alone.
#   3. Rotation starts at the configured mirror, so a healthy azure setup is not rewritten.
#   4. The cache key covers image version, architecture and the sorted package list.
#   5. A cache hit installs with --no-download and no update; a cache that cannot satisfy the
#      set falls back to the network path.
#
# Usage: .github/ci/test-apt-install.sh        (exit 0 = all pass)

set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
script="$here/apt-install.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

pass=0
fail=0
ok() { pass=$((pass + 1)); printf 'ok   %s\n' "$*"; }
no() { fail=$((fail + 1)); printf 'FAIL %s\n' "$*"; }
check() { local d="$1"; shift; if "$@"; then ok "$d"; else no "$d"; fi; }

# --- stubs ----------------------------------------------------------------------------------

mkdir -p "$work/bin"

# sudo: drop the leading flags apt does not use here and exec the rest.
cat >"$work/bin/sudo" <<'EOF'
#!/usr/bin/env bash
exec "$@"
EOF

# apt-get: records its arguments. Driven by the files in $STUB_DIR:
#   update_fail_until   mirror substring that keeps failing on `update`
#   install_fail_until  mirror substring that keeps failing on `install`
#   all_fail            "yes" makes every update and install fail
#   offline_ok          "yes" when `install --no-download` should succeed
cat >"$work/bin/apt-get" <<'EOF'
#!/usr/bin/env bash
set -u
args=("$@")
echo "$*" >>"$STUB_DIR/calls"
mirror=""
for f in "$STUB_DIR"/sources/sources.list.d/*.sources "$STUB_DIR"/sources/sources.list; do
  [ -f "$f" ] || continue
  m=$(grep -oE 'https?://[^[:space:],/]+' "$f" | head -n1) && [ -n "$m" ] && { mirror=$m; break; }
done
mirror=${mirror#*://}

sub=""
for a in "${args[@]}"; do case "$a" in update|install) sub=$a ;; esac; done

if [ "$(cat "$STUB_DIR/all_fail" 2>/dev/null)" = yes ]; then
  echo "E: Failed to fetch http://$mirror (timeout)" >&2
  exit 100
fi

if [ "$sub" = update ]; then
  if [ -f "$STUB_DIR/update_fail_until" ] &&
     [ "$mirror" = "$(cat "$STUB_DIR/update_fail_until")" ]; then
    echo "E: Failed to fetch http://$mirror (timeout)" >&2
    exit 100
  fi
  exit 0
fi

# --no-download is only satisfiable from a populated cache.
for a in "${args[@]}"; do
  if [ "$a" = --no-download ]; then
    if [ "$(cat "$STUB_DIR/offline_ok" 2>/dev/null)" = yes ]; then exit 0; fi
    echo "E: --no-download was used but the package could not be found" >&2
    exit 100
  fi
done

if [ -f "$STUB_DIR/install_fail_until" ] &&
   [ "$mirror" = "$(cat "$STUB_DIR/install_fail_until")" ]; then
  echo "E: Failed to fetch http://$mirror (timeout)" >&2
  exit 100
fi

# Emulate a successful download for the .deb cache.
mkdir -p "$STUB_DIR/archives"
for a in "${args[@]}"; do
  case "$a" in -*|install|-y|--no-install-recommends) continue ;; esac
  : >"$STUB_DIR/archives/${a}_1.0_amd64.deb"
  : >"$STUB_DIR/lists/${a}_Packages"
done
exit 0
EOF

cat >"$work/bin/dpkg" <<'EOF'
#!/usr/bin/env bash
[ "${1:-}" = --print-architecture ] && { echo amd64; exit 0; }
exit 0
EOF

# timeout: the stubs are instant; drop the duration argument and exec.
cat >"$work/bin/timeout" <<'EOF'
#!/usr/bin/env bash
shift
exec "$@"
EOF

chmod +x "$work/bin"/*

# --- fixtures -------------------------------------------------------------------------------

# A fresh scratch tree with a deb822 sources file, a classic list and a third-party list.
new_stub() {
  STUB_DIR="$work/case-$1"
  rm -rf "$STUB_DIR"
  mkdir -p "$STUB_DIR/sources/sources.list.d" "$STUB_DIR/archives" "$STUB_DIR/lists" \
    "$STUB_DIR/cache" "$STUB_DIR/home"

  cat >"$STUB_DIR/sources/sources.list.d/ubuntu.sources" <<'EOF'
Types: deb
URIs: http://azure.archive.ubuntu.com/ubuntu/
Suites: noble noble-updates
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: http://security.ubuntu.com/ubuntu/
Suites: noble-security
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
  cat >"$STUB_DIR/sources/sources.list" <<'EOF'
deb https://azure.archive.ubuntu.com/ubuntu noble main
EOF
  cat >"$STUB_DIR/sources/sources.list.d/microsoft.list" <<'EOF'
deb [arch=amd64] https://packages.microsoft.com/repos/edge stable main
EOF
  : >"$STUB_DIR/calls"
}

# Args are the script's own (--print-key, packages); prefix a call with VAR=... to set extra
# environment (bash applies it to the function's environment).
run_script() {
  local stub="$STUB_DIR"
  env \
    HOME="$stub/home" \
    PATH="$work/bin:$PATH" \
    APT_SOURCES_ROOT="$stub/sources" \
    APT_ARCHIVES_DIR="$stub/archives" \
    APT_LISTS_DIR="$stub/lists" \
    APT_DEB_CACHE_DIR="$stub/cache" \
    APT_ROOT_CMD="" \
    STUB_DIR="$stub" \
    bash "$script" "$@"
}

mirrors_in_sources() {
  grep -ohE 'https?://[^[:space:],]+' "$STUB_DIR/sources/sources.list.d/ubuntu.sources" \
    "$STUB_DIR/sources/sources.list" | sort -u
}

# --- 1. dead mirror is rotated past ---------------------------------------------------------

new_stub rotate
echo "azure.archive.ubuntu.com" >"$STUB_DIR/update_fail_until"
out=$(run_script cmake ninja-build 2>&1)
rc=$?
check "rotation: exits 0 after a dead azure mirror" [ "$rc" -eq 0 ]
check "rotation: logs the failed attempt" grep -q "apt-get update timed out or errored on azure.archive.ubuntu.com" <<<"$out"
check "rotation: logs the mirror that succeeded" grep -q "success via http://archive.ubuntu.com/ubuntu" <<<"$out"
check "rotation: both source files now point at the working mirror" \
  [ "$(mirrors_in_sources)" = "http://archive.ubuntu.com/ubuntu/" ]
check "rotation: third-party source untouched" \
  grep -q 'packages.microsoft.com/repos/edge' "$STUB_DIR/sources/sources.list.d/microsoft.list"
check "rotation: archives harvested into the cache" \
  bash -c 'ls "$1"/cache/*.deb >/dev/null' _ "$STUB_DIR"

# --- 2. rotation starts at the configured (healthy) mirror ----------------------------------

new_stub healthy
out=$(run_script zlib1g-dev 2>&1)
check "healthy: first attempt uses the configured azure mirror" \
  grep -q "attempt 1/4: mirror http://azure.archive.ubuntu.com/ubuntu" <<<"$out"
check "healthy: azure succeeds without rotating" grep -q "success via http://azure.archive.ubuntu.com/ubuntu" <<<"$out"

# --- 3. install failure (not update) also rotates -------------------------------------------

new_stub install-fail
echo "azure.archive.ubuntu.com" >"$STUB_DIR/install_fail_until"
out=$(run_script libtiff6 2>&1)
check "install failure: rotates to the next mirror" grep -q "success via http://archive.ubuntu.com/ubuntu" <<<"$out"

# --- 4. every mirror dead => non-zero -------------------------------------------------------

new_stub all-dead
echo yes >"$STUB_DIR/all_fail"
out=$(APT_PASSES=1 run_script libtiff6 2>&1)
rc=$?
check "all dead: exit status is 1" [ "$rc" -eq 1 ]
check "all dead: reports the mirrors it tried" grep -q "all 4 attempts failed" <<<"$out"
check "all dead: names each mirror in the failure summary" \
  grep -q "mirror.math.princeton.edu/pub/ubuntu" <<<"$out"

# --- 5. cache key ---------------------------------------------------------------------------

k1=$(run_script --print-key cmake ninja-build 2>/dev/null)
k2=$(run_script --print-key ninja-build cmake 2>/dev/null)
k3=$(run_script --print-key cmake 2>/dev/null)
check "cache key: is stable across package order" [ "$k1" = "$k2" ]
check "cache key: changes with the package set" [ "$k1" != "$k3" ]
check "cache key: carries the image version" bash -c '[[ "$1" == apt-v2-*-amd64-* ]]' _ "$k1"

# --- 6. cache hit installs offline ----------------------------------------------------------

new_stub offline
echo yes >"$STUB_DIR/offline_ok"
mkdir -p "$STUB_DIR/cache/lists"
echo "mirrors.edge.kernel.org/ubuntu" >"$STUB_DIR/cache/mirror"
: >"$STUB_DIR/cache/libtiff6_1.0_amd64.deb"
: >"$STUB_DIR/cache/lists/libtiff6_Packages"
out=$(APT_CACHE_HIT=true run_script libtiff6 2>&1)
rc=$?
check "offline: exits 0 on a cache hit" [ "$rc" -eq 0 ]
check "offline: used --no-download" grep -q -- '--no-download' "$STUB_DIR/calls"
check "offline: never ran apt-get update" bash -c '! grep -q " update" "$1"' _ "$STUB_DIR/calls"
check "offline: reported the cached mirror" grep -q "success via cache (originally mirrors.edge.kernel.org/ubuntu)" <<<"$out"
check "offline: seeded the cached deb" [ -f "$STUB_DIR/archives/libtiff6_1.0_amd64.deb" ]

# --- 7. cache hit that cannot satisfy the set falls back to the network ----------------------

new_stub offline-miss
echo no >"$STUB_DIR/offline_ok"
mkdir -p "$STUB_DIR/cache/lists"
: >"$STUB_DIR/cache/libtiff6_1.0_amd64.deb"
out=$(APT_CACHE_HIT=true run_script libtiff6 2>&1)
rc=$?
check "offline miss: still exits 0 via the network" [ "$rc" -eq 0 ]
check "offline miss: reported the fallback" grep -q "falling back to the mirrors" <<<"$out"
check "offline miss: ran apt-get update" bash -c 'grep -q " update" "$1"' _ "$STUB_DIR/calls"

# --- 8. no source file at all ---------------------------------------------------------------

new_stub no-sources
rm -f "$STUB_DIR/sources/sources.list" "$STUB_DIR/sources/sources.list.d/ubuntu.sources"
out=$(run_script cmake 2>&1)
check "no sources: falls back to a full update" grep -q "running a full apt-get update" <<<"$out"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
