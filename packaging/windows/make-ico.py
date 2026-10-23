#!/usr/bin/env python3
"""Render packaging/windows/alphashape.ico from the design-token icon geometry.

The icon is drawn here rather than converted from the SVG so that producing it
needs no ImageMagick, rsvg or Pillow: the packaging step runs on a plain
Windows runner and the file must come out the same everywhere, byte for byte
apart from nothing at all. It draws what packaging/linux/org.alphashape.app.svg
draws: a rounded accent square with the triangulated-surface mark in the dark
token colour.

Usage: make-ico.py [output.ico]
"""
from __future__ import annotations

import math
import struct
import sys

# design/system/tokens.json: accent and on-accent colours.
ACCENT = (0x35, 0xBD, 0xAF)
MARK = (0x04, 0x21, 0x1E)

# Sizes Windows asks for: the shell picks the closest 16 for the title bar and
# the largest for the large-icon view.
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

# The mark, in the SVG's 256x256 coordinate space.
TRIANGLE = ((128.0, 52.0), (204.0, 176.0), (52.0, 176.0))
SPINE = ((128.0, 52.0), (128.0, 176.0))
CROSSBAR = ((90.0, 114.0), (166.0, 114.0))
NODES = ((128.0, 52.0, 12.0), (204.0, 176.0, 12.0), (52.0, 176.0, 12.0), (128.0, 176.0, 9.0))
STROKE = 9.0
CORNER_RADIUS = 52.0
PADDING = 8.0


def _inside_rounded_square(x: float, y: float, size: float) -> bool:
    """Coverage test for the rounded square, in the same 256 space."""
    left, top = PADDING, PADDING
    right, bottom = size - PADDING, size - PADDING
    if not (left <= x <= right and top <= y <= bottom):
        return False
    r = CORNER_RADIUS
    # Only the four corner boxes need the arc test.
    cx = None
    cy = None
    if x < left + r and y < top + r:
        cx, cy = left + r, top + r
    elif x > right - r and y < top + r:
        cx, cy = right - r, top + r
    elif x < left + r and y > bottom - r:
        cx, cy = left + r, bottom - r
    elif x > right - r and y > bottom - r:
        cx, cy = right - r, bottom - r
    if cx is None:
        return True
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def _segment_distance(px: float, py: float, a: tuple[float, float],
                      b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _sample(x: float, y: float, size: float) -> tuple[int, int, int, int]:
    """RGBA at a point, in the 256 coordinate space."""
    if not _inside_rounded_square(x, y, size):
        return (0, 0, 0, 0)
    # The mark sits on top of the accent square: strokes are lines, nodes are
    # discs, and the triangle itself is an outline (the SVG fills it "none").
    on_stroke = (_segment_distance(x, y, TRIANGLE[0], TRIANGLE[1]) <= STROKE / 2
                 or _segment_distance(x, y, TRIANGLE[1], TRIANGLE[2]) <= STROKE / 2
                 or _segment_distance(x, y, TRIANGLE[2], TRIANGLE[0]) <= STROKE / 2
                 or _segment_distance(x, y, SPINE[0], SPINE[1]) <= STROKE / 2
                 or _segment_distance(x, y, CROSSBAR[0], CROSSBAR[1]) <= STROKE / 2)
    on_node = any(math.hypot(x - nx, y - ny) <= nr for nx, ny, nr in NODES)
    if on_stroke or on_node:
        return (*MARK, 0xFF)
    return (*ACCENT, 0xFF)


def _render(size: int) -> bytes:
    """One 32-bit BGRA, bottom-up bitmap body (the ICO/DIB convention)."""
    scale = size / 256.0
    rows = []
    for py in range(size):
        row = bytearray()
        for px in range(size):
            # 2x2 supersampling: the mark's strokes are thin at 16 px.
            acc_r = acc_g = acc_b = acc_a = 0
            for sy in (0.25, 0.75):
                for sx in (0.25, 0.75):
                    x = (px + sx) / scale
                    y = (py + sy) / scale
                    r, g, b, a = _sample(x, y, 256.0)
                    acc_r += r * a
                    acc_g += g * a
                    acc_b += b * a
                    acc_a += a
            if acc_a == 0:
                row += b"\x00\x00\x00\x00"
            else:
                row += bytes((acc_b // acc_a, acc_g // acc_a, acc_r // acc_a, acc_a // 4))
        rows.append(bytes(row))
    # DIB rows run bottom-up.
    return b"".join(reversed(rows))


def _bmp_entry(size: int) -> bytes:
    """A BITMAPINFOHEADER + pixels, the classic ICO payload (no PNG needed)."""
    header = struct.pack(
        "<IiiHHIIiiII",
        40,          # biSize
        size,        # biWidth
        size * 2,    # biHeight: image plus the AND mask
        1,           # biPlanes
        32,          # biBitCount
        0,           # biCompression: BI_RGB
        size * size * 4,  # biSizeImage
        0, 0,        # pixels per metre
        0, 0,        # colours used / important
    )
    # The AND mask is all zeroes: alpha already carries transparency, but the
    # format requires the mask to be present and row-aligned to 32 bits.
    mask_row = ((size + 31) // 32) * 4
    return header + _render(size) + b"\x00" * (mask_row * size)


def main(argv: list[str]) -> int:
    out = argv[1] if len(argv) > 1 else "alphashape.ico"
    payload = [_bmp_entry(size) for size in SIZES]
    offset = 6 + 16 * len(SIZES)
    directory = bytearray(struct.pack("<HHH", 0, 1, len(SIZES)))
    for size, data in zip(SIZES, payload):
        # 0 means 256 in the directory entry.
        directory += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32,
                                 len(data), offset)
        offset += len(data)
    with open(out, "wb") as handle:
        handle.write(bytes(directory) + b"".join(payload))
    print(f"wrote {out}: {len(SIZES)} sizes, {offset} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
