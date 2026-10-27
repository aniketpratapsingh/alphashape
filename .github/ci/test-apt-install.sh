#!/usr/bin/env bash
# Tests for .github/ci/apt-install.sh. Runs unprivileged: sudo, apt-get and dpkg are replaced
# by stubs on PATH, and the script's filesystem roots are redirected into a temp directory.
#
# What is proven here:
#   1. A dead mirror is rotated past: the failed mirror is logged, the next one is tried, and
#      the log names the mirror that succeeded.
#   2. Both source formats are rewritten (deb822 ubuntu.sources and classic sources.list), the
#      original scheme is preserved (https stays https), and third-party sources are left alone.
#   3. Rotation starts at the configured mirror, so a healthy azure setup is not rewritten.
#   4. The cache key covers image version, architecture, ISO week and the sorted package list;
#      the restore prefix deliberately omits the ISO week.
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

# sudo: record the call, then exec the rest. The recording is what proves the script never
# takes root for a path under $HOME (a root-owned ~/.cache breaks ccache with EACCES).
cat >"$work/bin/sudo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$STUB_DIR/sudo_calls"
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
  # `mirror+file:` means apt walks the mirrorlist itself; the stub models that as "every
  # entry is tried", so the first entry stands in for the whole list.
  # apt ignores comment lines, so the stub must too.
  m=$(grep -vE '^[[:space:]]*#' "$f" | grep -oE 'https?://[^[:space:],/]+' | head -n1)
  [ -n "$m" ] && { mirror=$m; break; }
  m=$(grep -vE '^[[:space:]]*#' "$f" | grep -oE 'mirror\+file:[^[:space:],]+' | head -n1)
  if [ -n "$m" ]; then
    m=$(grep -oE 'https?://[^[:space:]/]+' "${m#mirror+file:}" | head -n1)
    [ -n "$m" ] && { mirror=$m; break; }
  fi
done
mirror=${mirror#*://}

sub=""
for a in "${args[@]}"; do case "$a" in update|install) sub=$a ;; esac; done

if [ "$(cat "$STUB_DIR/all_fail" 2>/dev/null)" = yes ]; then
  echo "E: Failed to fetch http://$mirror (timeout)" >&2
  exit 100
fi

if [ "$sub" = update ]; then
  # Every entry of the runner mirrorlist is dead.
  if [ -f "$STUB_DIR/update_fail_mirrorlist" ] &&
     grep -q 'mirror+file:' "$STUB_DIR"/sources/sources.list.d/*.sources 2>/dev/null; then
    echo "E: Failed to fetch http://azure.archive.ubuntu.com/ubuntu (timeout)" >&2
    exit 100
  fi
  if [ -f "$STUB_DIR/update_fail_until" ] &&
     [ "$mirror" = "$(cat "$STUB_DIR/update_fail_until")" ]; then
    echo "E: Failed to fetch http://$mirror (timeout)" >&2
    exit 100
  fi
  exit 0
fi

# --no-download is only satisfiable from a populated cache whose lists are the ones apt sees:
# a cached list that the image's own copy shadowed is the bug this models.
for a in "${args[@]}"; do
  if [ "$a" = --no-download ]; then
    if [ -f "$STUB_DIR/cache/lists/libtiff6_Packages" ] &&
       [ -f "$STUB_DIR/lists/libtiff6_Packages" ] &&
       ! cmp -s "$STUB_DIR/cache/lists/libtiff6_Packages" "$STUB_DIR/lists/libtiff6_Packages"; then
      echo "E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?" >&2
      exit 100
    fi
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
check "rotation: logs the mirror that succeeded" grep -q "success on attempt 2/4 (sources: https://archive.ubuntu.com/ubuntu/)" <<<"$out"
# Both files rotate to the same mirror, each keeping the scheme it was configured with: the
# classic list was https and must stay https rather than being downgraded.
check "rotation: both source files now point at the working mirror" \
  [ "$(mirrors_in_sources | tr '\n' ' ')" = "http://archive.ubuntu.com/ubuntu/ https://archive.ubuntu.com/ubuntu/ " ]
check "rotation: third-party source untouched" \
  grep -q 'packages.microsoft.com/repos/edge' "$STUB_DIR/sources/sources.list.d/microsoft.list"
check "rotation: archives harvested into the cache" \
  bash -c 'ls "$1"/cache/*.deb >/dev/null' _ "$STUB_DIR"
# A root-owned ~/.cache (from `sudo mkdir ~/.cache/...`) makes ccache fail on every compile,
# so nothing the script runs as root may name a path under the cache directory.
check "rotation: never runs sudo on a path under the cache dir" \
  bash -c '! grep -q "$1" "$2" 2>/dev/null' _ "$STUB_DIR/cache" "$STUB_DIR/sudo_calls"
check "rotation: cache directory stays user-owned" [ -O "$STUB_DIR/cache" ]

# --- 2. rotation starts at the configured (healthy) mirror ----------------------------------

new_stub healthy
out=$(run_script zlib1g-dev 2>&1)
check "healthy: first attempt uses the configured azure mirror" \
  grep -q "attempt 1/4: mirror http://azure.archive.ubuntu.com/ubuntu" <<<"$out"
check "healthy: azure succeeds without rotating" grep -q "success on attempt 1/4 (sources: https://azure.archive.ubuntu.com/ubuntu/)" <<<"$out"

# --- 3. install failure (not update) also rotates -------------------------------------------

new_stub install-fail
echo "azure.archive.ubuntu.com" >"$STUB_DIR/install_fail_until"
out=$(run_script libtiff6 2>&1)
check "install failure: rotates to the next mirror" grep -q "success on attempt 2/4 (sources: https://archive.ubuntu.com/ubuntu/)" <<<"$out"

# --- 4. every mirror dead => non-zero -------------------------------------------------------

new_stub all-dead
echo yes >"$STUB_DIR/all_fail"
out=$(APT_PASSES=1 run_script libtiff6 2>&1)
rc=$?
check "all dead: exit status is 1" [ "$rc" -eq 1 ]
check "all dead: reports the mirrors it tried" grep -q "all 4 attempts failed" <<<"$out"
check "all dead: names each mirror in the failure summary" \
  grep -q "mirror.math.princeton.edu/pub/ubuntu" <<<"$out"

# --- 4b. the GitHub runner image layout (mirror+file:) --------------------------------------

new_stub runner-mirrorlist
cat >"$STUB_DIR/sources/sources.list.d/ubuntu.sources" <<'EOF2'
Types: deb
URIs: mirror+file:/etc/apt/apt-mirrors.txt
Suites: noble noble-updates
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF2
rm -f "$STUB_DIR/sources/sources.list"
printf 'http://azure.archive.ubuntu.com/ubuntu/\tpriority:1\n' >"$STUB_DIR/sources/apt-mirrors.txt"
printf 'https://archive.ubuntu.com/ubuntu/\tpriority:2\n' >>"$STUB_DIR/sources/apt-mirrors.txt"

out=$(run_script cmake 2>&1)
rc=$?
check "mirrorlist image: the source file is recognised" \
  grep -q "sources: $STUB_DIR/sources/sources.list.d/ubuntu.sources" <<<"$out"
check "mirrorlist image: attempt 1 keeps the runner's own list" \
  grep -q "attempt 1/4: runner mirrorlist" <<<"$out"
check "mirrorlist image: succeeds without rewriting a working image" [ "$rc" -eq 0 ]
check "mirrorlist image: source left as mirror+file:" \
  grep -q 'mirror+file:/etc/apt/apt-mirrors.txt' "$STUB_DIR/sources/sources.list.d/ubuntu.sources"

# The whole runner list dead: the rotation must take over from attempt 2.
new_stub runner-mirrorlist-dead
cp "$work/case-runner-mirrorlist/sources/sources.list.d/ubuntu.sources" \
  "$STUB_DIR/sources/sources.list.d/ubuntu.sources"
rm -f "$STUB_DIR/sources/sources.list"
echo yes >"$STUB_DIR/update_fail_mirrorlist"
out=$(run_script cmake 2>&1)
rc=$?
check "mirrorlist image, list dead: falls through to a rotated mirror" [ "$rc" -eq 0 ]
check "mirrorlist image, list dead: names the mirror that worked" \
  grep -qE "success on attempt [2-9]/4 \(sources: http://(archive.ubuntu.com|mirrors.edge.kernel.org|mirror.math.princeton.edu)" <<<"$out"
check "mirrorlist image, list dead: indirection is replaced" \
  bash -c '! grep -q "mirror+file:" "$1"' _ "$STUB_DIR/sources/sources.list.d/ubuntu.sources"

# --- 4c. comment lines are not mirrors ------------------------------------------------------

new_stub comments
{
  echo "# See http://help.ubuntu.com/community/UpgradeNotes for how to upgrade to"
  cat <<'EOF2'
Types: deb
URIs: http://azure.archive.ubuntu.com/ubuntu/
Suites: noble noble-updates
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF2
} >"$STUB_DIR/sources/sources.list.d/ubuntu.sources"
rm -f "$STUB_DIR/sources/sources.list"
echo "azure.archive.ubuntu.com" >"$STUB_DIR/update_fail_until"
out=$(run_script cmake 2>&1)
check "comments: a URL in a comment is not reported as the mirror" \
  bash -c '! grep -q "current mirror: http://help.ubuntu.com" <<<"$1"' _ "$out"
check "comments: the comment itself is preserved" \
  grep -q "help.ubuntu.com/community/UpgradeNotes" "$STUB_DIR/sources/sources.list.d/ubuntu.sources"
check "comments: the real mirror is rotated" \
  grep -q "URIs: http://archive.ubuntu.com/ubuntu/" "$STUB_DIR/sources/sources.list.d/ubuntu.sources"
# The fixture above has no mirror+file:, so rotation starts at azure and must move to the
# canonical archive; a commented URL must not have been taken for the configured mirror.
check "comments: rotation did not start at the commented URL" \
  bash -c 'grep -q "URIs: http://archive.ubuntu.com/ubuntu/" "$1"' _ "$STUB_DIR/sources/sources.list.d/ubuntu.sources"

# --- 4d. the original scheme is preserved ---------------------------------------------------

# A source configured over https must not be downgraded to http by the rotation. Downloads over
# plain http can be tampered with in transit, so the scheme the operator chose has to survive.
new_stub https-only
cat >"$STUB_DIR/sources/sources.list.d/ubuntu.sources" <<'EOF2'
Types: deb
URIs: https://azure.archive.ubuntu.com/ubuntu/
Suites: noble noble-updates
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF2
rm -f "$STUB_DIR/sources/sources.list"
echo "azure.archive.ubuntu.com" >"$STUB_DIR/update_fail_until"
out=$(run_script cmake 2>&1)
check "https: rotates to the next mirror" grep -q "success on attempt 2/4" <<<"$out"
check "https: stays https after the rotation" \
  grep -q "URIs: https://archive.ubuntu.com/ubuntu/" "$STUB_DIR/sources/sources.list.d/ubuntu.sources"
check "https: nothing was downgraded to http" \
  bash -c '! grep -q "URIs: http://" "$1"' _ "$STUB_DIR/sources/sources.list.d/ubuntu.sources"

# http sources stay http: the point is to keep what was configured, not to force https.
new_stub http-only
rm -f "$STUB_DIR/sources/sources.list"
echo "azure.archive.ubuntu.com" >"$STUB_DIR/update_fail_until"
out=$(run_script cmake 2>&1)
check "http: stays http after the rotation" \
  grep -q "URIs: http://archive.ubuntu.com/ubuntu/" "$STUB_DIR/sources/sources.list.d/ubuntu.sources"

# A third-party https line in the same file must not make the Ubuntu archive https: the scheme
# follows the Ubuntu URIs only.
new_stub third-party-https
cat >"$STUB_DIR/sources/sources.list" <<'EOF2'
deb http://azure.archive.ubuntu.com/ubuntu noble main
deb [arch=amd64] https://packages.microsoft.com/repos/edge stable main
EOF2
echo "azure.archive.ubuntu.com" >"$STUB_DIR/update_fail_until"
out=$(run_script cmake 2>&1)
check "http: a third-party https line in the same file does not force https" \
  grep -q "^deb http://archive.ubuntu.com/ubuntu/ noble main$" "$STUB_DIR/sources/sources.list"
check "http: the third-party line is untouched" \
  grep -q 'packages.microsoft.com/repos/edge' "$STUB_DIR/sources/sources.list"

# The runner image's mirror+file: indirection names no scheme of its own, so the rotation has to
# fall back to the mirrors' default of http.
new_stub mirrorlist-scheme
cat >"$STUB_DIR/sources/sources.list.d/ubuntu.sources" <<'EOF2'
Types: deb
URIs: mirror+file:/etc/apt/apt-mirrors.txt
Suites: noble noble-updates
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF2
rm -f "$STUB_DIR/sources/sources.list"
printf 'http://azure.archive.ubuntu.com/ubuntu/\tpriority:1\n' >"$STUB_DIR/sources/apt-mirrors.txt"
echo yes >"$STUB_DIR/update_fail_mirrorlist"
out=$(run_script cmake 2>&1)
check "mirror+file: the rotated replacement defaults to http" \
  bash -c 'grep -qE "URIs: http://(archive\.ubuntu\.com|mirrors\.edge\.kernel\.org|mirror\.math\.princeton\.edu)/ubuntu/" "$1"' \
  _ "$STUB_DIR/sources/sources.list.d/ubuntu.sources"

# --- 5. cache key ---------------------------------------------------------------------------


k1=$(run_script --print-key cmake ninja-build 2>/dev/null)
k2=$(run_script --print-key ninja-build cmake 2>/dev/null)
k3=$(run_script --print-key cmake 2>/dev/null)
check "cache key: is stable across package order" [ "$k1" = "$k2" ]
check "cache key: changes with the package set" [ "$k1" != "$k3" ]
check "cache key: carries the image version and ISO week" \
  bash -c '[[ "$1" =~ ^apt-v5-[^-]+-amd64-[0-9]{4}-[0-9]{2}- ]]' _ "$k1"

# The key is scoped to the ref: a run that checks out an arbitrary ref must not be able to seed
# the entry another branch reads (CodeQL "cache poisoning via untrusted code").
kb=$(APT_CACHE_SCOPE=main run_script --print-key cmake 2>/dev/null)
kf=$(APT_CACHE_SCOPE=feature run_script --print-key cmake 2>/dev/null)
check "cache key: differs per ref" [ "$kb" != "$kf" ]
check "cache key: main's own key is stable" \
  [ "$kb" = "$(APT_CACHE_SCOPE=main run_script --print-key cmake 2>/dev/null)" ]
out=$(APT_CACHE_SCOPE=feature run_script --print-key cmake 2>/dev/null)
check "cache key: no scope still yields a usable key" [ -n "$k1" ]
gout="$work/github_output"
: >"$gout"
out=$(GITHUB_OUTPUT="$gout" APT_CACHE_SCOPE=main run_script --print-key cmake 2>/dev/null)
check "cache key: publishes a main-only restore prefix" \
  bash -c 'grep -q "^shared_prefix=apt-v5-.*-amd64-main-$" "$1"' _ "$gout"
# The restore prefix must not carry the ISO week, or a new week would find nothing to restore
# and every run would pay for a cold download.
check "cache key: the restore prefix omits the ISO week" \
  bash -c 'grep -q "^shared_prefix=apt-v5-[^-]*-amd64-main-$" "$1"' _ "$gout"
# And the week is really in the exact key, so last week's entry is never an exact hit.
check "cache key: the exact key carries the ISO week" \
  bash -c 'grep -qE "^key=apt-v5-[^-]+-amd64-main-[0-9]{4}-[0-9]{2}-" "$1"' _ "$gout"
check "cache key: main's exact key matches main's restore prefix" \
  bash -c 'k=$(sed -n "s/^key=//p" "$1"); p=$(sed -n "s/^shared_prefix=//p" "$1");
           [ "${k#"$p"}" != "$k" ]' _ "$gout"

# --- 6. cache hit installs offline ----------------------------------------------------------

new_stub offline
echo yes >"$STUB_DIR/offline_ok"
echo yes >"$STUB_DIR/accept_cached_sources"
mkdir -p "$STUB_DIR/cache/lists" "$STUB_DIR/cache/sources"
: >"$STUB_DIR/cache/libtiff6_1.0_amd64.deb"
: >"$STUB_DIR/cache/lists/libtiff6_Packages"
# The cached source snapshot is the one that produced the lists; it is restored verbatim.
cat >"$STUB_DIR/cache/sources/d__ubuntu.sources" <<'EOF2'
Types: deb
URIs: http://mirrors.edge.kernel.org/ubuntu/
Suites: noble noble-updates
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF2
out=$(APT_CACHE_HIT=true run_script libtiff6 2>&1)
rc=$?
check "offline: exits 0 on a cache hit" [ "$rc" -eq 0 ]
check "offline: used --no-download" grep -q -- '--no-download' "$STUB_DIR/calls"
check "offline: never ran apt-get update" bash -c '! grep -q " update" "$1"' _ "$STUB_DIR/calls"
check "offline: reported the cache as the source of the packages" grep -q "success from the cache: no network used" <<<"$out"
check "offline: restored the matching source snapshot" \
  grep -q "mirrors.edge.kernel.org" "$STUB_DIR/sources/sources.list.d/ubuntu.sources"
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

# --- 7a. the image's own lists must not win over the cached ones ---------------------------

new_stub stale-image-lists
echo yes >"$STUB_DIR/offline_ok"
mkdir -p "$STUB_DIR/cache/lists" "$STUB_DIR/cache/sources"
: >"$STUB_DIR/cache/libtiff6_1.0_amd64.deb"
echo "cached-list" >"$STUB_DIR/cache/lists/libtiff6_Packages"
# The runner image already has apt lists from a different mirror/snapshot.
echo "image-list" >"$STUB_DIR/lists/libtiff6_Packages"
out=$(APT_CACHE_HIT=true run_script libtiff6 2>&1)
rc=$?
check "stale image lists: the cached list replaces the image's" \
  [ "$(cat "$STUB_DIR/lists/libtiff6_Packages")" = "cached-list" ]
# The stub refuses --no-download while the lists disagree, so a clean exit proves the restore
# actually took effect rather than the fallback network path having run.
check "stale image lists: the offline install succeeds on the first try" [ "$rc" -eq 0 ]
check "stale image lists: no mirror fallback was needed" \
  bash -c '! grep -q "falling back to the mirrors" <<<"$1"' _ "$out"
check "stale image lists: reported no network use" \
  grep -q "success from the cache: no network used" <<<"$out"

# --- 7b. round trip: a cold run's harvest serves the next run offline ----------------------

new_stub roundtrip
echo yes >"$STUB_DIR/offline_ok"
echo yes >"$STUB_DIR/accept_cached_sources"
: >"$STUB_DIR/sources/sources.list.d/ubuntu.sources.comment"
out1=$(run_script libtiff6 2>&1)
check "round trip: cold run installs over the network" grep -q "success on attempt" <<<"$out1"
check "round trip: cold run harvested the source snapshot" \
  [ -f "$STUB_DIR/cache/sources/d__ubuntu.sources" ]
check "round trip: cold run harvested the deb" [ -f "$STUB_DIR/cache/libtiff6_1.0_amd64.deb" ]
: >"$STUB_DIR/calls"
out2=$(APT_CACHE_HIT=true run_script libtiff6 2>&1)
rc=$?
check "round trip: warm run exits 0" [ "$rc" -eq 0 ]
check "round trip: warm run used --no-download" grep -q -- '--no-download' "$STUB_DIR/calls"
check "round trip: warm run ran exactly one apt-get call, and it was the offline install" \
  bash -c '[ "$(wc -l < "$1" | tr -d " ")" = 1 ] && grep -q -- "--no-download" "$1"' _ "$STUB_DIR/calls"
check "round trip: warm run reported no network use" grep -q "success from the cache: no network used" <<<"$out2"

# --- 8. no source file at all ---------------------------------------------------------------

new_stub no-sources
rm -f "$STUB_DIR/sources/sources.list" "$STUB_DIR/sources/sources.list.d/ubuntu.sources"
out=$(run_script cmake 2>&1)
check "no sources: falls back to a full update" grep -q "running a full apt-get update" <<<"$out"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
