#!/usr/bin/env python3
"""Import the published AprilTag family tables (WP-13.7).

The seven families below are interoperability data: users print the standard
tags, so our tables must equal the published ones exactly. This script reads
the family sources from AprilRobotics/apriltag (BSD-2-Clause) and emits

  * engine/tests/fixtures/markers/apriltag_<family>.txt  (one hex code per line)
  * engine/src/common/algorithm/markers/tag_family_data.cpp (constexpr arrays)

Run from anywhere; paths are resolved from the repository root.

    python3 tools/markers/import_apriltag_families.py [--cache DIR]

The tables are NOT regenerated with our own lexicode: the published generator's
pseudo-random search order is not reproducible. The checked-in data is the
published data, and the C++ tests compare it against the fixtures.
"""

import argparse
import hashlib
import pathlib
import re
import sys
import urllib.request

FAMILIES = [
    ("16h5", "tag16h5"),
    ("25h9", "tag25h9"),
    ("36h10", "tag36h10"),
    ("36h11", "tag36h11"),
    ("Circle21h7", "tagCircle21h7"),
    ("Standard41h12", "tagStandard41h12"),
    ("Standard52h13", "tagStandard52h13"),
]

# The header of every published family source, BSD-2-Clause. Reproduced
# verbatim in the generated file and in every fixture.
BSD_LICENSE = """Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.

This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan."""

# Pinned upstream commit. Never replace this with a branch name: the checked-in
# tables must stay reproducible, and the SHA-256 recorded in every fixture makes
# the provenance check non-tautological.
COMMIT = "b7c0ebe9aa20f82ec7a828579004f9e706bfecd9"
COMMIT_URL = "https://github.com/AprilRobotics/apriltag/commit/{}".format(COMMIT)
RAW_URL = ("https://raw.githubusercontent.com/AprilRobotics/apriltag/{}/{{}}"
           .format(COMMIT))
BLOB_URL = "https://github.com/AprilRobotics/apriltag/blob/{}/{{}}".format(COMMIT)
ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "engine/src/common/algorithm/markers"
FIXTURE_DIR = ROOT / "engine/tests/fixtures/markers"
LICENSE_PATH = ROOT / "packaging/licenses/BSD-2-Clause-apriltag.txt"


def read_source(published, cache):
    """Return the published source text, from the cache directory or the web."""
    if cache is not None:
        path = pathlib.Path(cache) / (published + ".c")
        if path.is_file():
            return path.read_text(encoding="utf-8")
    request = urllib.request.Request(
        RAW_URL.format(published + ".c"),
        headers={"User-Agent": "alphashape-apriltag-importer"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _int_field(text, pattern, what):
    match = re.search(pattern, text)
    if match is None:
        sys.exit("missing {} in published source".format(what))
    return int(match.group(1))


def parse_family(text, published):
    """Extract codes and layout from one published family source."""
    nbits = _int_field(text, r"tf->nbits\s*=\s*(\d+)", "nbits")
    declared = re.search(r"codedata\[(\d+)\]\s*=\s*\{(.*?)\};", text, re.S)
    if declared is None:
        sys.exit("missing codedata array in {}".format(published))
    codes = [int(value, 16)
             for value in re.findall(r"0x([0-9a-fA-F]+)UL", declared.group(2))]
    if len(codes) != int(declared.group(1)):
        sys.exit("{}: codedata has {} entries, declared {}".format(
            published, len(codes), declared.group(1)))
    bit_x = [int(value)
             for value in re.findall(r"bit_x\[\d+\]\s*=\s*(-?\d+)", text)]
    bit_y = [int(value)
             for value in re.findall(r"bit_y\[\d+\]\s*=\s*(-?\d+)", text)]
    if len(bit_x) != nbits or len(bit_y) != nbits:
        sys.exit("{}: layout has {}/{} entries for {} bits".format(
            published, len(bit_x), len(bit_y), nbits))
    if any(code >> nbits for code in codes):
        sys.exit("{}: a code does not fit in {} bits".format(published, nbits))
    reversed_match = re.search(r"reversed_border\s*=\s*(\w+)", text)
    if reversed_match is None:
        sys.exit("missing reversed_border in " + published)
    return {
        "published": published,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_url": BLOB_URL.format(published + ".c"),
        "name": published[3:] if published.startswith("tag") else published,
        "hamming": _int_field(text, r"tf->h\s*=\s*(\d+)", "h"),
        "nbits": nbits,
        "width_at_border": _int_field(text, r"width_at_border\s*=\s*(\d+)",
                                      "width_at_border"),
        "total_width": _int_field(text, r"total_width\s*=\s*(\d+)", "total_width"),
        "reversed_border": reversed_match.group(1) == "true",
        "codes": codes,
        "bit_x": bit_x,
        "bit_y": bit_y,
    }


def license_comment(prefix):
    return "\n".join((prefix + line).rstrip() for line in BSD_LICENSE.splitlines())


def emit_fixture(family):
    """Return the text of one family's checked-in fixture."""
    header = [
        "# AprilTag family {} (published {}).".format(
            family["name"], family["published"]),
        "# Source: {}".format(family["source_url"]),
        "# Source commit: {}".format(COMMIT_URL),
        "# source-sha256: {}".format(family["sha256"]),
        "# family={} h={} nbits={} width_at_border={} total_width={} "
        "reversed_border={} codes={}".format(
            family["name"], family["hamming"], family["nbits"],
            family["width_at_border"], family["total_width"],
            "true" if family["reversed_border"] else "false",
            len(family["codes"])),
        "#",
        license_comment("# "),
        "#",
        "# One published code per line, hexadecimal, in published order.",
        "",
    ]
    lines = header + ["0x{:016x}".format(code) for code in family["codes"]]
    return "\n".join(lines) + "\n"


def emit_code_array(family):
    suffix = family["name"]
    lines = ["constexpr std::uint64_t kCodes{}[] = {{".format(suffix)]
    per_line = 4
    for start in range(0, len(family["codes"]), per_line):
        chunk = family["codes"][start:start + per_line]
        lines.append("    " + ", ".join("0x{:016x}ULL".format(c) for c in chunk) + ",")
    lines.append("};")
    return lines


def emit_layout_array(family):
    suffix = family["name"]
    lines = ["constexpr std::pair<int, int> kBits{}[] = {{".format(suffix)]
    per_line = 6
    pairs = list(zip(family["bit_x"], family["bit_y"]))
    for start in range(0, len(pairs), per_line):
        chunk = pairs[start:start + per_line]
        lines.append("    " + ", ".join("{{{}, {}}}".format(x, y)
                                        for x, y in chunk) + ",")
    lines.append("};")
    return lines


def emit_cpp(families):
    """Return the generated tag_family_data.cpp text."""
    lines = [
        "// Generated by tools/markers/import_apriltag_families.py -- do not edit.",
        "//",
        "// Published AprilTag family tables, from the AprilRobotics/apriltag",
        "// repository (BSD-2-Clause). The order of kFamilies matches the order of",
        "// enum class AprilTagFamily in tag_families.hpp.",
        "//",
        "// Source commit: {}".format(COMMIT_URL),
        "// Source files: " + ", ".join(
            "tag{}.c".format(published[len("tag"):])
            for _, published in FAMILIES) + ".",
        "//",
        license_comment("// "),
        "",
        '#include "common/algorithm/markers/tag_family_data.hpp"',
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <span>",
        "#include <utility>",
        "",
        "namespace alphashape {",
        "namespace {",
        "",
    ]
    for family in families:
        lines.extend(emit_code_array(family))
        lines.append("")
        lines.extend(emit_layout_array(family))
        lines.append("")
    lines.append("constexpr TagFamilyData kFamilies[] = {")
    for family in families:
        lines.append(
            '    TagFamilyData{{"{}", {}, {}, {}, {}, {}, kCodes{}, kBits{}}},'.format(
                family["name"], family["nbits"], family["hamming"],
                family["total_width"], family["width_at_border"],
                "true" if family["reversed_border"] else "false",
                family["name"], family["name"]))
    lines.extend([
        "};",
        "",
        "}  // namespace",
        "",
        "const TagFamilyData& tag_family_data(AprilTagFamily family) {",
        "    return kFamilies[static_cast<std::size_t>(family)];",
        "}",
        "",
        "}  // namespace alphashape",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", metavar="DIR",
                        help="directory holding tag*.c (falls back to the web)")
    args = parser.parse_args()
    families = []
    for name, published in FAMILIES:
        family = parse_family(read_source(published, args.cache), published)
        if family["name"] != name:
            sys.exit("published {} is named {}, expected {}".format(
                published, family["name"], name))
        families.append(family)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for family in families:
        path = FIXTURE_DIR / "apriltag_{}.txt".format(family["name"])
        path.write_text(emit_fixture(family), encoding="utf-8")
    generated = SOURCE_DIR / "tag_family_data.cpp"
    generated.write_text(emit_cpp(families), encoding="utf-8")
    LICENSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LICENSE_PATH.write_text(BSD_LICENSE + "\n", encoding="utf-8")
    for family in families:
        print("{:14s} codes={:6d} nbits={} h={}".format(
            family["name"], len(family["codes"]), family["nbits"],
            family["hamming"]))
    print("wrote", generated)


if __name__ == "__main__":
    main()
