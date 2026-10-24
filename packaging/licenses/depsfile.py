#!/usr/bin/env python3
"""Read packaging/licenses/deps.yaml.

A deliberately small parser: deps.yaml is a block sequence of mappings whose values are
single-line scalars, which is the whole of YAML this project needs for it. PyYAML is not
installed on the CI runners, and a release step that fails because a build image dropped a
Python package is not worth the twenty lines this replaces.

Fields:
    name        str            required
    copyright   str            required
    spdx        str            required, an SPDX licence expression
    url         str            required, where the component comes from
    version     str | None     only when every platform ships the same one
    platforms   list[str]      subset of linux, macos, windows; default all three
    note        str | None     one line of prose for NOTICE.txt

parse_deps() raises ValueError with the offending line number on anything it does not
understand: a silently dropped entry would mean a shipped component with no notice.
"""

from __future__ import annotations

import re

PLATFORMS = ("linux", "macos", "windows")
REQUIRED = ("name", "copyright", "spdx", "url")

_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _split_inline_list(value: str, lineno: int) -> list[str]:
    body = value.strip()
    if not (body.startswith("[") and body.endswith("]")):
        raise ValueError(f"line {lineno}: expected an inline list, got {value!r}")
    items = [item.strip() for item in body[1:-1].split(",")]
    items = [_unquote(item) for item in items if item]
    unknown = [item for item in items if item not in PLATFORMS]
    if unknown:
        raise ValueError(f"line {lineno}: unknown platform(s) {unknown}; use {list(PLATFORMS)}")
    return items


def parse_deps(path) -> list[dict]:
    """Return the entries of `path`, in file order."""
    entries: list[dict] = []
    current: dict | None = None
    seen: set[str] = set()

    with open(path, encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("- "):
                current = {}
                entries.append(current)
                match = _KEY_RE.match(stripped[2:].strip())
                if not match:
                    raise ValueError(f"line {lineno}: expected 'key: value' after '- '")
                key, _, value = match.group(1), None, match.group(2)
            elif line[:1] in (" ", "\t"):
                match = _KEY_RE.match(stripped)
                if not match:
                    raise ValueError(f"line {lineno}: expected 'key: value'")
                if current is None:
                    raise ValueError(f"line {lineno}: indented field outside an entry")
                key, value = match.group(1), match.group(2)
            else:
                raise ValueError(f"line {lineno}: expected '- name: ...' or an indented field")

            if key in current:
                raise ValueError(f"line {lineno}: duplicate key {key!r} in the same entry")
            if key == "platforms":
                current[key] = _split_inline_list(value, lineno)
            else:
                current[key] = _unquote(value)

    if not entries:
        raise ValueError(f"{path}: no entries")

    for index, entry in enumerate(entries, start=1):
        missing = [key for key in REQUIRED if not entry.get(key)]
        if missing:
            label = entry.get("name") or f"entry {index}"
            raise ValueError(f"{label}: missing required {', '.join(missing)}")
        if entry["name"] in seen:
            raise ValueError(f"duplicate component {entry['name']!r}")
        seen.add(entry["name"])
        entry.setdefault("platforms", list(PLATFORMS))
        entry.setdefault("version", None)
        entry.setdefault("note", None)
        if not entry["platforms"]:
            raise ValueError(f"{entry['name']}: empty platforms list")

    return entries


def platforms_for(entries, platform: str) -> list[dict]:
    """The entries a given OS ships or depends on, in file order."""
    return [entry for entry in entries if platform in entry["platforms"]]
