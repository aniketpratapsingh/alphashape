#!/usr/bin/env python3
"""Generate the deterministic PGM fixtures used by the CLI and parity tests.

Why a generator and not checked-in blobs: the fixtures must be exactly
reproducible and reviewable. Anyone can rerun this script and get the same
bytes, so a failing test can never be blamed on a stale binary fixture.

Content: a diagonal gradient (so the Difference-of-Gaussians detector always has
structure) plus a sum of sinusoids at several frequencies and orientations. The
sinusoids are deliberate — they give well-localised DoG extrema at many scales
and orientations, so the detector reliably finds well over 30 keypoints per
image. That matters for the parity test: with only a handful of keypoints,
"keypoint_limit=30 caps at 30" would pass without the cap ever truncating
anything. These fixtures make the cap the binding constraint.

Size: 200x150 pixels = 30,000 bytes of payload, comfortably under the 30 KB
budget, and large enough that the default per-megapixel cap is not the only
thing limiting the count.

Output is 8-bit binary PGM (P5), which is what read_pnm() accepts.
"""

import math
import os

WIDTH = 200
HEIGHT = 150


def field(x, y, phase):
    """Intensity in [0, 1]: gradient + multi-frequency sinusoids."""
    value = 0.50 * (0.25 * (x / (WIDTH - 1)) + 0.75 * (y / (HEIGHT - 1)))
    value += 0.20 * math.sin(x * 0.55 + phase) * math.cos(y * 0.45)
    value += 0.18 * math.sin((x + y) * 0.30)
    value += 0.12 * math.cos(x * 1.1 + y * 0.4)
    value += 0.10 * math.sin(x * 0.9) * (0.5 + 0.5 * math.cos(y * 0.7))
    return max(0.0, min(1.0, value))


def render(phase):
    pixels = bytearray(WIDTH * HEIGHT)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pixels[y * WIDTH + x] = int(round(field(x, y, phase) * 255.0))
    return pixels


def write_pgm(path, pixels):
    with open(path, "wb") as out:
        out.write(b"P5\n%d %d\n255\n" % (WIDTH, HEIGHT))
        out.write(bytes(pixels))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    # Different phases give two views of the same structure, so the fixtures are
    # related (as two photos of a scene are) yet not identical.
    write_pgm(os.path.join(here, "fixture_a.pgm"), render(0.0))
    write_pgm(os.path.join(here, "fixture_b.pgm"), render(0.7))
    print("wrote fixture_a.pgm and fixture_b.pgm (%dx%d)" % (WIDTH, HEIGHT))
    write_codec_fixtures(here)


# ---------------------------------------------------------------------------
# Codec fixtures (milestone 2, package A)
# ---------------------------------------------------------------------------
# These are the files the image codec and EXIF tests decode. They are written
# with Pillow, but the tests only ever read them through our own code: exiftool
# is used to *verify* the generator, never as a test dependency.
#
# Determinism: every pixel is a pure function of its coordinates, exactly like
# the PGM fixtures above, so rerunning this script reproduces the same bytes.
# Except for the JPEGs, which are lossy and therefore only compared to their
# source within a tolerance.

CODE_W = 64
CODE_H = 48

# A small 8-bit texture. Flat colour would let a channel-swap bug through, so
# every channel gets a different pattern. The multipliers are deliberately large
# so the pattern aliases heavily: a decoder that mixed up two channels, or that
# read a row at the wrong stride, cannot pass by accident.
def rgb_pattern(x, y):
    r = (x * 4 + y * 2) % 256
    g = (x * 2 + y * 7 + 40) % 256
    b = (x * 9 + y * 3 + 90) % 256
    return r, g, b


def grey_pattern(x, y):
    return (x * 5 + y * 3 + 20) % 256


# JPEG is lossy, so its fixtures use a smooth ramp instead: a near-Nyquist
# pattern (as above) costs up to 160 levels of error at quality 90, which would
# force a tolerance so wide it would hide real bugs. A smooth ramp keeps the
# worst per-channel error in single digits, so the test can be strict.
def rgb_jpeg_pattern(x, y):
    return (min(255, 30 + x * 3 + y), min(255, 50 + x + y * 3), min(255, 70 + x * 2 + y * 2))


def grey_jpeg_pattern(x, y):
    return min(255, 20 + x * 4 + y)


def write_codec_fixtures(here):
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

    def path(name):
        return os.path.join(here, name)

    # --- PNG ---------------------------------------------------------------
    grey = Image.new("L", (CODE_W, CODE_H))
    rgb = Image.new("RGB", (CODE_W, CODE_H))
    rgba = Image.new("RGBA", (CODE_W, CODE_H))
    for y in range(CODE_H):
        for x in range(CODE_W):
            grey.putpixel((x, y), grey_pattern(x, y))
            r, g, b = rgb_pattern(x, y)
            rgb.putpixel((x, y), (r, g, b))
            # Alpha varies so that a decoder which forgets to drop it produces
            # visibly wrong colour rather than passing by accident.
            rgba.putpixel((x, y), (r, g, b, (x * 4 + y * 4) % 256))
    grey.save(path("grey.png"))
    rgb.save(path("rgb.png"))
    rgba.save(path("rgba.png"))

    # Palette PNG: a 256-entry palette the test can reproduce. The pixel index
    # at (x, y) is (x + y) % 256 and the palette entry for index i is
    # (i, 255 - i, (i * 7) % 256), which the test re-derives exactly.
    pal_img = Image.new("P", (CODE_W, CODE_H))
    palette = []
    for i in range(256):
        palette += [i, 255 - i, (i * 7) % 256]
    pal_img.putpalette(palette)
    for y in range(CODE_H):
        for x in range(CODE_W):
            pal_img.putpixel((x, y), (x + y) % 256)
    pal_img.save(path("palette.png"))

    # 16-bit grey PNG with a known value per pixel, so the test can assert the
    # exact rounding rule (v * 255 + 32767) / 65535 rather than a tolerance.
    # value(i) = (i * 4067) % 65536 for i = y * CODE_W + x.
    grey16 = Image.new("I;16", (CODE_W, CODE_H))
    for y in range(CODE_H):
        for x in range(CODE_W):
            i = y * CODE_W + x
            grey16.putpixel((x, y), (i * 4067) % 65536)
    grey16.save(path("grey16.png"))

    # --- TIFF --------------------------------------------------------------
    rgb.save(path("rgb8.tiff"))
    # 16-bit grey TIFF, same value function as the PNG so the two agree.
    grey16.save(path("grey16.tiff"))
    # A TIFF carrying a real Exif IFD, so read_exif can be exercised on the TIFF
    # container as well as the JPEG one. Same 8-bit grey pixels as grey.png.
    tiff_exif = Image.Exif()
    tiff_exif[0x010F] = "AlphaOptics"
    tiff_exif[0x0110] = "AO-T100"
    tiff_exif[0x0112] = 1
    tiff_exif_ifd = tiff_exif.get_ifd(0x8769)
    tiff_exif_ifd[0x9003] = "2024:06:15 09:30:00"
    tiff_exif_ifd[0x920A] = IFDRational(24, 1)
    tiff_exif_ifd[0xA405] = 36
    tiff_exif_ifd[0xA20E] = IFDRational(2500, 1)
    tiff_exif_ifd[0xA20F] = IFDRational(2500, 1)
    tiff_exif_ifd[0xA210] = 3
    tiff_exif_ifd[0xA431] = "AOT100-0001"
    tiff_exif_ifd[0xA434] = "AlphaOptics 24mm f/1.8"
    grey.save(path("exif_grey.tiff"), exif=tiff_exif.tobytes())
    # The same metadata again in a PNG's eXIf chunk, so the PNG path the reader
    # supports is exercised too.
    grey.save(path("exif_grey.png"), exif=tiff_exif.tobytes())

    # A tiled 16-bit grey TIFF. Pillow's TIFF writer only emits strips, so this
    # one is produced by tiffcp, which ships with libtiff and is present here.
    # The reader's tile path is otherwise untested, and tiled is a common layout
    # for scanner and satellite imagery. The fixture is skipped, with a warning,
    # if tiffcp is not on PATH; the strip fixtures still cover everything else.
    write_tiled_tiff(here, path("grey16.tiff"), path("grey16_tiled.tiff"))

    # A palette TIFF. The pixel index at (x, y) is (x + y) % 256 and the palette
    # entry for index i is (i, 255 - i, (i * 7) % 256) — the same map as the
    # palette PNG, so the two tests check the same expected colours through
    # different containers. The palette itself is what palette.png carries.
    pal_img.save(path("palette.tiff"))

    # A big-endian RGB TIFF, produced with tiffcp -B. Pillow always writes
    # little-endian, and the reader has to honour the byte order for multi-byte
    # fields, so this is the only fixture that exercises the MM path.
    write_big_endian_tiff(here, path("rgb8.tiff"), path("rgb8_be.tiff"))

    # --- Baseline JPEG -----------------------------------------------------
    # Quality 90 keeps the fixture small while the decoded result stays close
    # to the source, so the tolerance in the test can be tight.
    baseline = Image.new("RGB", (CODE_W, CODE_H))
    for y in range(CODE_H):
        for x in range(CODE_W):
            baseline.putpixel((x, y), rgb_jpeg_pattern(x, y))
    baseline.save(path("baseline.jpg"), quality=90)

    # --- Camera JPEGs for the sensor-grouping tests ------------------------
    # 200x150, matching the PGM fixtures' scale. cam_a1/cam_a2 are the *same
    # camera* (Make, Model, FocalLength and FocalPlane resolution all equal), so
    # a later grouping stage should put them together; cam_b1 is a different
    # model with only FocalLengthIn35mmFilm and no focal-plane resolution.
    write_camera_jpeg(here, "cam_a1.jpg", make="AlphaOptics", model="AO-100",
                      focal=IFDRational(35, 1), focal35=35,
                      fp_res=(IFDRational(3000, 1), IFDRational(3000, 1), 3),
                      gps={"lat_ref": "N", "lat": (43, 30, 0),
                           "lon_ref": "E", "lon": (79, 20, 0), "alt": None},
                      serial="AO100-0001", orientation=1)
    write_camera_jpeg(here, "cam_a2.jpg", make="AlphaOptics", model="AO-100",
                      focal=IFDRational(35, 1), focal35=35,
                      fp_res=(IFDRational(3000, 1), IFDRational(3000, 1), 3),
                      # Southern / western, and 350 m below sea level.
                      gps={"lat_ref": "S", "lat": (33, 52, 0),
                           "lon_ref": "W", "lon": (151, 12, 0), "alt": 350},
                      serial="AO100-0002", orientation=1)
    write_camera_jpeg(here, "cam_b1.jpg", make="BetaCam", model="BC-7",
                      focal=IFDRational(50, 1), focal35=75,
                      fp_res=None, gps=None, serial="BC7-0001", orientation=1)

    print("wrote codec fixtures (PNG/TIFF/JPEG) and cam_a1/cam_a2/cam_b1")


def write_tiled_tiff(here, source, target):
    """Re-encode `source` as a tiled TIFF with tiffcp, 32x32 tiles."""
    import shutil
    import subprocess

    tiffcp = shutil.which("tiffcp")
    if tiffcp is None:
        print("tiffcp not found; skipping grey16_tiled.tiff")
        return
    result = subprocess.run(
        [tiffcp, "-t", "-w", "32", "-l", "32", source, target],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("tiffcp failed; skipping grey16_tiled.tiff: %s" % result.stderr.strip())
    else:
        print("wrote grey16_tiled.tiff (tiled, via tiffcp)")


def write_big_endian_tiff(here, source, target):
    """Re-encode `source` as a big-endian (MM) uncompressed TIFF with tiffcp."""
    import shutil
    import subprocess

    tiffcp = shutil.which("tiffcp")
    if tiffcp is None:
        print("tiffcp not found; skipping rgb8_be.tiff")
        return
    result = subprocess.run(
        [tiffcp, "-B", "-c", "none", source, target],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("tiffcp failed; skipping rgb8_be.tiff: %s" % result.stderr.strip())
    else:
        print("wrote rgb8_be.tiff (big-endian, via tiffcp)")


def write_camera_jpeg(here, name, make, model, focal, focal35, fp_res, gps, serial, orientation):
    """A 200x150 camera-like JPEG with a populated Exif IFD and optional GPS."""
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

    width, height = 200, 150
    image = Image.new("RGB", (width, height))
    # A textured scene so that any downstream keypoint work has structure; the
    # codec tests only compare decoded pixels, so exact content is unimportant.
    for y in range(height):
        for x in range(width):
            image.putpixel(
                (x, y),
                ((x * 3 + y) % 256, (x + y * 3 + 30) % 256, (x * 5 + y * 2 + 60) % 256),
            )

    exif = Image.Exif()
    exif[0x010F] = make
    exif[0x0110] = model
    exif[0x0112] = orientation

    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[0x9003] = "2024:06:15 09:30:00"
    exif_ifd[0x920A] = focal
    exif_ifd[0xA405] = focal35
    exif_ifd[0xA002] = width
    exif_ifd[0xA003] = height
    exif_ifd[0xA431] = serial
    exif_ifd[0xA434] = "AlphaOptics 35mm f/2"
    if fp_res is not None:
        exif_ifd[0xA20E] = fp_res[0]
        exif_ifd[0xA20F] = fp_res[1]
        exif_ifd[0xA210] = fp_res[2]

    if gps is not None:
        gps_ifd = exif.get_ifd(0x8825)
        gps_ifd[1] = gps["lat_ref"]
        gps_ifd[2] = tuple(IFDRational(v, 1) for v in gps["lat"])
        gps_ifd[3] = gps["lon_ref"]
        gps_ifd[4] = tuple(IFDRational(v, 1) for v in gps["lon"])
        if gps["alt"] is not None:
            gps_ifd[5] = 1  # altitude ref: below sea level
            gps_ifd[6] = IFDRational(gps["alt"], 1)

    image.save(os.path.join(here, name), quality=90, exif=exif.tobytes())


if __name__ == "__main__":
    main()
