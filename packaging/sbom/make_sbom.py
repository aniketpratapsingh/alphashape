#!/usr/bin/env python3
"""Generate sbom.spdx.json: an SPDX 2.3 bill of materials for a release.

    make_sbom.py --version v0.2.0 --out dist/sbom.spdx.json
    make_sbom.py --version v0.2.0 --out - --now 2026-10-23T00:00:00Z

One package entry per component in packaging/licenses/deps.yaml, plus the application
itself, plus a DESCRIBES relationship from the document to the application and a
DEPENDS_ON from the application to each component. No network access, no external
service: what is published is exactly what a user can reproduce from the repository.

The document is byte-stable for a given version and --now value (keys sorted, two-space
indent, trailing newline), which is what makes it diffable between releases.

Dependencies are recorded as packages, not as files: filesAnalyzed is false, because the
bill is about which components a release contains, not about hashing every bundled file
(the installers' SHA256SUMS covers those).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "licenses"))

from depsfile import parse_deps  # noqa: E402

# SPDX identifiers are "[A-Za-z0-9.-]+"; a component name is not, so every generated id is
# a slug. The app's own id is fixed so downstream tooling can refer to it.
APP_ID = "SPDXRef-Package-alphashape"
DOC_ID = "SPDXRef-DOCUMENT"

# SPDX 2.3, section 7.15: an SPDX licence expression. This is not a full validator (that is
# thousands of identifiers); it rejects the shapes that are always wrong -- an empty value,
# a URL, prose with spaces -- so a typo in deps.yaml fails at release time rather than
# publishing a document no consumer can read.
_EXPRESSION_RE = re.compile(
    r"^[A-Za-z0-9.+\-]+( (AND|OR) [A-Za-z0-9.+\-]+| WITH [A-Za-z0-9.\-]+)*$"
)


def slug(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-")
    return out or "component"


def check_expression(entry: dict) -> str:
    expression = entry["spdx"].strip()
    if not _EXPRESSION_RE.match(expression):
        raise ValueError(
            f"{entry['name']}: {expression!r} is not an SPDX licence expression "
            "(expected e.g. 'MIT', 'Apache-2.0 WITH LLVM-exception', 'BSD-3-Clause AND MIT')"
        )
    return expression


def build_document(entries, version: str, now: str, repo_url: str, ref: str) -> dict:
    namespace = f"{repo_url}/spdx/{version}-{slug(ref)}"
    packages: list[dict] = []
    relationships: list[dict] = []

    app_name = "alphashape"
    packages.append(
        {
            "name": app_name,
            "SPDXID": APP_ID,
            "versionInfo": version,
            "downloadLocation": f"{repo_url}/releases/tag/{version}",
            "filesAnalyzed": False,
            "licenseConcluded": "GPL-3.0-or-later",
            "licenseDeclared": "GPL-3.0-or-later",
            "copyrightText": "Aniket Singh and the alphashape contributors",
            "homepage": repo_url,
            "summary": "Open-source photogrammetry engine",
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:github/aniketpratapsingh/alphashape@{version}",
                }
            ],
            "comment": f"Source tree {ref}",
        }
    )
    relationships.append(
        {
            "spdxElementId": DOC_ID,
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": APP_ID,
        }
    )

    used_ids = {APP_ID}
    for entry in entries:
        package_id = f"SPDXRef-Package-{slug(entry['name'])}"
        if package_id in used_ids:
            raise ValueError(f"{entry['name']}: generated SPDXID {package_id!r} collides; rename it")
        used_ids.add(package_id)

        package = {
            "name": entry["name"],
            "SPDXID": package_id,
            # Not all components have a single version across the three installers (PROJ and
            # Ceres come from each platform's package manager), so an absent version is
            # recorded as NOASSERTION rather than invented.
            "versionInfo": entry.get("version") or "NOASSERTION",
            "downloadLocation": entry["url"],
            "filesAnalyzed": False,
            "licenseConcluded": check_expression(entry),
            "licenseDeclared": check_expression(entry),
            "copyrightText": entry["copyright"],
            "comment": f"Platforms: {', '.join(entry['platforms'])}",
        }
        if entry.get("note"):
            package["description"] = entry["note"]
        packages.append(package)

        relationships.append(
            {
                "spdxElementId": APP_ID,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": package_id,
            }
        )

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": DOC_ID,
        "name": f"alphashape {version}",
        "documentNamespace": namespace,
        "creationInfo": {
            "created": now,
            "creators": [
                "Tool: alphashape-sbom",
                f"Organization: alphashape contributors ({repo_url})",
            ],
            "comment": f"Generated from packaging/licenses/deps.yaml at {ref}",
        },
        "packages": packages,
        "relationships": relationships,
    }


def normalise_timestamp(value: str) -> str:
    """SPDX wants UTC 'YYYY-MM-DDThh:mm:ssZ'; reject anything else rather than emit it."""
    value = value.strip()
    if not value.endswith("Z"):
        raise ValueError(f"--now must be a UTC timestamp ending in Z, got {value!r}")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ValueError(f"--now must look like 2026-10-23T12:00:00Z, got {value!r}") from None
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    parser.add_argument("--version", required=True, help="release version, e.g. v0.2.0")
    parser.add_argument("--out", required=True, help="output path, or - for stdout")
    parser.add_argument("--ref", default="", help="source ref the build came from (commit sha)")
    parser.add_argument("--now", default=None, help="UTC timestamp (default: the current time)")
    parser.add_argument("--deps", default=str(here.parent / "licenses" / "deps.yaml"))
    parser.add_argument("--repo-url", default="https://github.com/aniketpratapsingh/alphashape")
    args = parser.parse_args(argv)

    now = args.now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries = parse_deps(args.deps)
    document = build_document(
        entries,
        version=args.version,
        now=normalise_timestamp(now),
        repo_url=args.repo_url.rstrip("/"),
        ref=args.ref or args.version,
    )
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    if args.out == "-":
        sys.stdout.write(text)
        return 0

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"{path}: {len(document['packages'])} packages, "
          f"{len(document['relationships'])} relationships")
    return 0


if __name__ == "__main__":
    sys.exit(main())
