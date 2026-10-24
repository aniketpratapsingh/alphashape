#!/usr/bin/env python3
"""Generate NOTICE.txt: the aggregated third-party licence notice for a release.

    collect.py --version v0.2.0 --out dist/NOTICE.txt
    collect.py --version v0.2.0 --out -            # to stdout

The component list is packaging/licenses/deps.yaml; this file only formats it. Every
entry in deps.yaml appears in the output, so a release cannot ship a component that the
notice does not mention. The full licence texts are not inlined: each installer already
carries them (LICENSES/third-party/<port>.txt in the Windows installer, Resources/doc in
the macOS bundle, /usr/share/doc/alphashape/copyright in the .deb).

Not hand-edited. If the notice is wrong, fix deps.yaml and regenerate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from depsfile import PLATFORMS, parse_deps  # noqa: E402

RULE = "=" * 78
THIN = "-" * 78


def render(entries, version: str, repo_url: str) -> str:
    out: list[str] = []
    add = out.append

    add("THIRD-PARTY NOTICES")
    add(RULE)
    add("")
    add(f"alphashape {version}")
    add("")
    add("alphashape is free software: you can redistribute it and/or modify it under the")
    add("terms of the GNU General Public License as published by the Free Software")
    add("Foundation, either version 3 of the License, or (at your option) any later")
    add("version. The full text is in the LICENSE file beside this one, and at")
    add("https://www.gnu.org/licenses/gpl-3.0.html.")
    add("")
    add("It comes with ABSOLUTELY NO WARRANTY, to the extent permitted by law.")
    add("")
    add("alphashape is an independent project. It is not affiliated with, endorsed by or")
    add("derived from Agisoft or Metashape.")
    add("")
    add("This release ships, or depends on, the third-party components listed below.")
    add("The 'Platforms' line of each entry says where it applies: a component that is")
    add("not bundled may still be a run-time dependency of one of the packages. Each")
    add("component keeps its own licence and is shipped unmodified and, where it is")
    add("bundled, as a separate replaceable file.")
    add("")
    add(f"Component list: {repo_url}/blob/main/packaging/licenses/deps.yaml")
    add("")
    add(f"{len(entries)} components, {len({e['spdx'] for e in entries})} distinct licences.")
    add("")

    by_platform = []
    for platform in PLATFORMS:
        count = sum(1 for entry in entries if platform in entry["platforms"])
        by_platform.append(f"{platform} {count}")
    add("Per platform: " + ", ".join(by_platform) + ".")
    add("")

    for index, entry in enumerate(entries, start=1):
        add(RULE)
        title = entry["name"]
        if entry.get("version"):
            title = f"{title} {entry['version']}"
        add(f"{index}. {title}")
        add(THIN)
        add(f"Copyright: {entry['copyright']}")
        add(f"Licence:   {entry['spdx']}")
        add(f"Source:    {entry['url']}")
        add(f"Platforms: {', '.join(entry['platforms'])}")
        if entry.get("note"):
            for line in _wrap(entry["note"], 74):
                add(f"  {line}")
        add("")

    add(RULE)
    add("")
    add("The licence texts named above ship with each package: in LICENSES/ in the")
    add("Windows installer, in alphashape.app/Contents/Resources/doc in the macOS")
    add("bundle, and in /usr/share/doc/alphashape in the Debian and AppImage")
    add("packages. Where a component's own package does not carry its text, the")
    add("entry above names the project page it can be read at.")
    add("")
    add(f"To report an error in this notice, open an issue at {repo_url}/issues.")
    add("")
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    parser.add_argument("--version", required=True, help="version string stamped into the notice")
    parser.add_argument("--out", required=True, help="output path, or - for stdout")
    parser.add_argument("--deps", default=str(here / "deps.yaml"))
    parser.add_argument("--repo-url", default="https://github.com/aniketpratapsingh/alphashape")
    args = parser.parse_args(argv)

    entries = parse_deps(args.deps)
    text = render(entries, args.version, args.repo_url.rstrip("/"))

    if args.out == "-":
        sys.stdout.write(text)
        return 0

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"{path}: {len(entries)} components, {len(text.splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
