#!/bin/sh
# Repository hygiene check (WP-11.0). Exit 0 when clean.
# 1. No tracked file under build trees, datasets, SDKs or project files.
# 2. Working tree clean (run after build + ctest: nothing generated in-tree).
set -u
cd "$(dirname "$0")/.." || exit 2
status=0
bad=$(git ls-files | command grep -E '(^|/)build[^/]*/|^datasets/|\.psx$|\.psz$|\.files/|\.spv$' || true)
if [ -n "$bad" ]; then
    echo "tracked paths that must be ignored:"; echo "$bad"; status=1
fi
dirty=$(git status --porcelain)
if [ -n "$dirty" ]; then
    echo "working tree not clean:"; echo "$dirty"; status=1
fi
[ $status -eq 0 ] && echo "repo hygiene ok"
exit $status
