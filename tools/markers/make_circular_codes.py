#!/usr/bin/env python3
"""Generate the checked-in circular-target code tables (WP-13.6).

This is the reference implementation of the greedy lexicode documented in
engine/src/common/algorithm/markers/target_patterns.hpp. Rings are `bits` wide;
sector 0 is the most significant bit and cyclic rotations of a ring word model
rotation of the physical target.

A word is accepted when its Hamming distance to every non-trivial rotation of
itself, and to every rotation of every already-accepted word, is at least 3.
Words are visited in increasing numeric order and only the canonical
(lexicographically smallest) rotation of each word is considered, so a ring and
all of its rotations are the same code.

The table size is whatever the greedy construction yields. The forbidden set is
maintained as one byte per word: accepting a word marks the radius-2 Hamming
ball around each of its rotations, which is exactly the set of later words that
would violate the distance-3 rule.

Usage:
    python3 tools/markers/make_circular_codes.py [OUTPUT_DIR]
"""

import os
import sys

BITS = (12, 14, 16, 20)


def rotate_right(word, bits):
    return ((word >> 1) | ((word & 1) << (bits - 1))) & ((1 << bits) - 1)


def is_canonical(word, bits):
    """True when no cyclic rotation of `word` is numerically smaller."""
    rotated = word
    for _ in range(bits - 1):
        rotated = rotate_right(rotated, bits)
        if rotated < word:
            return False
    return True


def min_self_distance(word, bits):
    distance = bits
    rotated = word
    for _ in range(bits - 1):
        rotated = rotate_right(rotated, bits)
        distance = min(distance, bin(word ^ rotated).count("1"))
    return distance


def mark_ball(forbidden, word, bits):
    forbidden[word] = 1
    for i in range(bits):
        forbidden[word ^ (1 << i)] = 1
    for i in range(bits):
        for j in range(i + 1, bits):
            forbidden[word ^ (1 << i) ^ (1 << j)] = 1


def generate(bits):
    """Return the sorted list of canonical ring words for `bits`."""
    size = 1 << bits
    forbidden = bytearray(size)
    accepted = []
    for word in range(1, size - 1):
        if forbidden[word]:
            continue
        if not is_canonical(word, bits):
            continue
        if min_self_distance(word, bits) < 3:
            continue
        accepted.append(word)
        rotated = word
        for _ in range(bits):
            mark_ball(forbidden, rotated, bits)
            rotated = rotate_right(rotated, bits)
    return accepted


def format_words(words, bits):
    lines = [
        "# Circular target ring codes for N=%d." % bits,
        "# count=%d" % len(words),
        "# construction=greedy lexicode over canonical cyclic rotations, "
        "minimum Hamming distance 3 between all rotations of all words.",
        "# One word per line, %d characters, MSB = sector 0 (12 o'clock, "
        "clockwise)." % bits,
    ]
    for word in words:
        lines.append(format(word, "0%db" % bits))
    return "\n".join(lines) + "\n"


def main(argv):
    output_dir = argv[1] if len(argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "engine", "tests", "fixtures", "markers")
    os.makedirs(output_dir, exist_ok=True)
    for bits in BITS:
        words = generate(bits)
        path = os.path.join(output_dir, "circular_codes_N%d.txt" % bits)
        with open(path, "w", encoding="ascii", newline="\n") as handle:
            handle.write(format_words(words, bits))
        print("N=%d count=%d -> %s" % (bits, len(words), path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
