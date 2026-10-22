"""Accent families, themes and viewport palettes for the alphashape mockups.

Everything colour-related that varies between an accent variant and a theme
lives here.  The builders only consume tokens; no screen HTML hard-codes a
brand colour.

One accent token family: base / hover / pressed / text / subtle / on-accent
(plus selection tints and derived fade/ring/chip-border values).  A theme's
status colours are part of the same variant because the warning amber collides
with an orange accent (see NOTES round 3).
"""
import re, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE / "assets"

# ---------------------------------------------------------------------------
# Accent variants
ACCENTS = {
    "teal": {
        "label": "Teal (current)",
        "dark": {
            "base": "#35BDAF", "hover": "#48CBBE", "pressed": "#2AA79B",
            "text": "#4FD1C3", "subtle": "#16302E", "on": "#04211E",
            "sel": "#1E3F3C", "sel_t2": "#B2C0C8",
            "fade": "rgba(53,189,175,.10)", "ring": "rgba(53,189,175,.35)",
            "chipbd": "#22443F",
            "ok": "#48B875", "okbg": "#12281B", "okbd": "#22402C",
            "warn": "#E3A33C", "warnbg": "#2A2214", "warnbd": "#3A3018",
            "err": "#F07068", "errbg": "#2E1917", "errbd": "#43231F",
            "info": "#6BA8E0", "proc": "#A78BFA",
            "sel_ok": "#7DD99A", "sel_warn": "#F0BE6A",
        },
        "light": {
            "base": "#0E7C72", "hover": "#0B6E66", "pressed": "#085B55",
            "text": "#0B6E66", "subtle": "#DDF1EE", "on": "#FFFFFF",
            "sel": "#CFEAE6", "sel_t2": "#4C5B63",
            "fade": "rgba(14,124,114,.08)", "ring": "rgba(14,124,114,.35)",
            "chipbd": "#A9D8D2",
            "ok": "#1E7A44", "okbg": "#E3F2E8", "okbd": "#B7DEC4",
            "warn": "#8A5A00", "warnbg": "#F8EEDC", "warnbd": "#E4CFA6",
            "err": "#B3261E", "errbg": "#F9E5E3", "errbd": "#EBC0BC",
            "info": "#1F5F96", "proc": "#6D4AC0",
            "sel_ok": "#14632F", "sel_warn": "#6B5200",
        },
        "vp": {"family": "warm"},
    },
    "ember": {
        "label": "Ember (burnt orange)",
        "dark": {
            "base": "#C2410C", "hover": "#CC4A15", "pressed": "#A93608",
            "text": "#F08A58", "subtle": "#331A0E", "on": "#FFFFFF",
            "sel": "#3A1D12", "sel_t2": "#B2C0C8",
            "fade": "rgba(194,65,12,.12)", "ring": "rgba(194,65,12,.40)",
            "chipbd": "#5E2A12",
            # gold warning: hue 45 deg instead of amber 37, so it separates
            # from the 17 deg ember accent; error shifted to rose (345 deg).
            "ok": "#48B875", "okbg": "#12281B", "okbd": "#22402C",
            "warn": "#EAC65A", "warnbg": "#2C2712", "warnbd": "#3D3714",
            "err": "#F4738B", "errbg": "#2E191F", "errbd": "#43232B",
            "info": "#6BA8E0", "proc": "#A78BFA",
            "sel_ok": "#7DD99A", "sel_warn": "#EAC65A",
        },
        "light": {
            "base": "#A83A0E", "hover": "#B64313", "pressed": "#8F2F08",
            "text": "#9A3409", "subtle": "#F8E6DC", "on": "#FFFFFF",
            "sel": "#F3D5C7", "sel_t2": "#4C5B63",
            "fade": "rgba(168,58,14,.08)", "ring": "rgba(168,58,14,.35)",
            "chipbd": "#E7B79F",
            "ok": "#1E7A44", "okbg": "#E3F2E8", "okbd": "#B7DEC4",
            "warn": "#6B5200", "warnbg": "#F7EFD4", "warnbd": "#E0CC93",
            "err": "#A81E45", "errbg": "#F9E5EC", "errbd": "#EFC2D0",
            "info": "#1F5F96", "proc": "#6D4AC0",
            "sel_ok": "#14632F", "sel_warn": "#6B5200",
        },
        "vp": {"family": "cool"},
    },
    "copper": {
        "label": "Copper (desaturated amber-copper)",
        "dark": {
            "base": "#C0873F", "hover": "#CB9550", "pressed": "#B57C33",
            "text": "#DFAE72", "subtle": "#2E2417", "on": "#1F1305",
            "sel": "#37291A", "sel_t2": "#B2C0C8",
            "fade": "rgba(192,135,63,.13)", "ring": "rgba(192,135,63,.40)",
            "chipbd": "#5A4526",
            "ok": "#48B875", "okbg": "#12281B", "okbd": "#22402C",
            # copper sits at hue 33 deg, so warning moves further toward
            # chartreuse (62 deg) to stay recognisably "not the accent".
            "warn": "#D8E45A", "warnbg": "#282812", "warnbd": "#3A3A16",
            "err": "#F4738B", "errbg": "#2E191F", "errbd": "#43232B",
            "info": "#6BA8E0", "proc": "#A78BFA",
            "sel_ok": "#7DD99A", "sel_warn": "#D8E45A",
        },
        "light": {
            "base": "#8F5C1F", "hover": "#9C6624", "pressed": "#7A4D18",
            "text": "#7E5219", "subtle": "#F5E9D6", "on": "#FFFFFF",
            "sel": "#EFDDC1", "sel_t2": "#4C5B63",
            "fade": "rgba(143,92,31,.09)", "ring": "rgba(143,92,31,.35)",
            "chipbd": "#DCC59A",
            "ok": "#1E7A44", "okbg": "#E3F2E8", "okbd": "#B7DEC4",
            "warn": "#545C00", "warnbg": "#F0F0D5", "warnbd": "#CFD094",
            "err": "#A81E45", "errbg": "#F9E5EC", "errbd": "#EFC2D0",
            "info": "#1F5F96", "proc": "#6D4AC0",
            "sel_ok": "#14632F", "sel_warn": "#545C00",
        },
        "vp": {"family": "cool"},
    },
}

# ---------------------------------------------------------------------------
# Chrome surfaces (independent of accent)
SURFACES = {
    "dark": {
        "app": "#0D1114", "panel": "#12171B", "raised": "#171E23",
        "overlay": "#1D252B", "sunken": "#0A0E11",
        "t1": "#E8EEF1", "t2": "#A7B4BC", "t3": "#828F98",
        "bd": "#28323A", "bd2": "#33414B", "bd3": "#4A5A65",
        "hover": "rgba(255,255,255,.05)", "track": "#0A0E11",
        "ph": "#0B0F12", "lights": "#3A464E", "scroll": "#33414B",
        "dim": "rgba(5,8,10,.62)",
    },
    "light": {
        "app": "#EDF0F1", "panel": "#F6F8F8", "raised": "#FFFFFF",
        "overlay": "#FFFFFF", "sunken": "#E4E8EA",
        "t1": "#101619", "t2": "#4C5B63", "t3": "#5C6A72",
        "bd": "#DDE3E5", "bd2": "#C7D0D4", "bd3": "#A3B0B6",
        "hover": "rgba(16,22,25,.06)", "track": "#D5DCDE",
        "ph": "#F6F8F8", "lights": "#C3CCD0", "scroll": "#C7D0D4",
        "dim": "rgba(16,22,25,.38)",
    },
}

# ---------------------------------------------------------------------------
# Viewport palettes.  The viewport is a self-contained instrument pane: its
# toolbar and overlay cards follow the canvas mode, not the app theme.
VP_MODES = {
    "dark": {
        "hi": "#243038", "base": "#1D252B", "lo": "#141B20",
        "grid": "#6B8391", "grid_op": "1", "bbox": "#5FA79C",
        "p1": "#9DB2BE", "p1_op": "0.75", "p2": "#A9C6D4", "p2_op": "0.9",
        "p3": "#DCEBF3", "p3_op": "0.9",
        "tie": "#A9C6D4", "tie_op": "0.55",
        "panel": "#121A1F", "raised": "#1D252B", "card": "#141C21", "card2": "#10171B",
        "t1": "#E8EEF1", "t2": "#A7B4BC", "t3": "#9AA7AE", "bd": "#33414B",
        "vig": "radial-gradient(95% 80% at 52% 42%,transparent 45%,rgba(6,10,13,.5) 100%)",
        "gz": "#F07068", "gx": "#48B875", "gy": "#6BA8E0",
        "gfill": "rgba(13,17,20,.5)", "gstroke": "#33414B",
        "ok": "#7DD99A", "warn": "#F0BE6A", "err": "#F47C94",
        "kbd": "#1D252B",
    },
    "light": {
        "hi": "#FAFBFC", "base": "#EDF1F2", "lo": "#E2E8EA",
        "grid": "#64717A", "grid_op": "1", "bbox": "#5E6B73",
        "p1": "#6F7C84", "p1_op": "1", "p2": "#46545B", "p2_op": "0.95",
        "p3": "#162026", "p3_op": "0.95",
        "tie": "#56646C", "tie_op": "0.85",
        "panel": "#FBFCFD", "raised": "#FFFFFF", "card": "#FFFFFF", "card2": "#FFFFFF",
        "t1": "#101619", "t2": "#4C5B63", "t3": "#5C6A72", "bd": "#C7D0D4",
        "vig": "radial-gradient(95% 80% at 52% 42%,transparent 55%,rgba(255,255,255,.45) 100%)",
        "gz": "#B3261E", "gx": "#1E7A44", "gy": "#1F5F96",
        "gfill": "rgba(255,255,255,.8)", "gstroke": "#C7D0D4",
        "ok": "#14632F", "warn": "#6B5200", "err": "#A81E45",
        "kbd": "#F1F4F5",
    },
}

# Camera / marker palettes by viewport family, per mode.
#   warm = teal accent (yellow cameras still work); cool = ember/copper
#   (cameras move to sky-cyan, GCPs become white core + orange ring).
#   `core` is the signal colour that must clear 3:1; `ring`/`ring_w` is the
#   separator outline (or the ring itself for the cool family).
VP_ENTITIES = {
    ("warm", "dark"): dict(cam="#F2C94C", cam_op="0.55",
                           core="#F2994A", ring="#2A1A0C", ring_w="0.6",
                           label="#F5C08C", signal="core"),
    ("warm", "light"): dict(cam="#7A5C00", cam_op="1",
                            core="#8A4A0A", ring="#5A3510", ring_w="0.6",
                            label="#5A3510", signal="core"),
    ("cool", "dark"): dict(cam="#5BC8E8", cam_op="0.7",
                           core="#FFFFFF", ring="#E8681F", ring_w="1.9",
                           label="#F5C08C", signal="ring"),
    ("cool", "light"): dict(cam="#0F5D78", cam_op="1",
                            core="#FFFFFF", ring="#8A4A0A", ring_w="1.9",
                            label="#5A3510", signal="ring"),
}


def theme_tokens(theme, accent, vp=None):
    """Complete CSS custom property set for one (theme, accent, viewport) render."""
    vp = vp or theme
    a = ACCENTS[accent][theme]
    s = SURFACES[theme]
    t = {
        "--app": s["app"], "--panel": s["panel"], "--raised": s["raised"],
        "--overlay": s["overlay"], "--sunken": s["sunken"],
        "--t1": s["t1"], "--t2": s["t2"], "--t3": s["t3"],
        "--bd": s["bd"], "--bd2": s["bd2"], "--bd3": s["bd3"],
        "--hover": s["hover"], "--track": s["track"], "--phbg": s["ph"],
        "--lights": s["lights"], "--scroll-thumb": s["scroll"], "--dim": s["dim"],
        "--accent": a["base"], "--accent-h": a["hover"], "--accent-p": a["pressed"],
        "--accent-t": a["text"], "--accent-s": a["subtle"], "--on-accent": a["on"],
        "--accent-chipbd": a["chipbd"], "--accent-fade": a["fade"], "--accent-ring": a["ring"],
        "--sel": a["sel"], "--sel-t2": a["sel_t2"], "--sel-ok": a["sel_ok"], "--sel-warn": a["sel_warn"],
        "--sel-bd": a["text"] if theme == "dark" else a["base"],
        "--ok": a["ok"], "--okbg": a["okbg"], "--okbd": a["okbd"],
        "--warn": a["warn"], "--warnbg": a["warnbg"], "--warnbd": a["warnbd"],
        "--err": a["err"], "--errbg": a["errbg"], "--errbd": a["errbd"],
        "--info": a["info"], "--proc": a["proc"],
        "--app-hi": "#1B242A" if theme == "dark" else "#FFFFFF",
        "--app-mid": "#12181C" if theme == "dark" else "#F2F5F5",
        "--scrim": "rgba(13,17,20,.72)" if theme == "dark" else "rgba(255,255,255,.85)",
        "--sgvig": ("radial-gradient(120% 90% at 50% 40%,transparent 55%,rgba(0,0,0,.45))"
                    if theme == "dark" else "radial-gradient(120% 90% at 50% 40%,transparent 60%,rgba(255,255,255,.5))"),
        "--thumbf": "brightness(.9) saturate(.9)" if theme == "dark" else "none",
        "--proc-fade": "rgba(167,139,250,.12)" if theme == "dark" else "rgba(109,74,192,.10)",
        "--proc-fade-sm": "rgba(167,139,250,.07)" if theme == "dark" else "rgba(109,74,192,.06)",
        "--proc-bd": "rgba(167,139,250,.40)" if theme == "dark" else "rgba(109,74,192,.35)",
        "--err-bd-soft": "rgba(240,112,104,.35)" if theme == "dark" else "rgba(179,38,30,.30)",
        "--err-bg-soft": "rgba(240,112,104,.08)" if theme == "dark" else "rgba(179,38,30,.07)",
    }
    t.update(vp_tokens(accent, vp))
    return t


def vp_tokens(accent, mode):
    """Viewport CSS vars. `mode` is the canvas mode (dark|light).  Viewport
    controls use the accent variant of *that* mode, so a light-chrome app with
    a dark canvas keeps a translucent teal/ember chip on a dark toolbar."""
    m = VP_MODES[mode]
    a = ACCENTS[accent][mode]
    ent = VP_ENTITIES[(ACCENTS[accent]["vp"]["family"], mode)]
    return {
        "--vp-hi": m["hi"], "--vp": m["base"], "--vp2": m["lo"],
        "--vvig": m["vig"],
        "--vgrid": m["grid"], "--vbbox": m["bbox"],
        "--vp-p1": m["p1"], "--vp-p2": m["p2"], "--vp-p3": m["p3"],
        "--vcam": ent["cam"], "--vcam-op": ent["cam_op"],
        "--vcore": ent["core"], "--vring": ent["ring"], "--vlabel": ent["label"],
        "--vsig": ent[ent["signal"]],
        "--vsel": a["text"] if mode == "dark" else a["base"],
        "--vsel-s": a["subtle"],
        "--vacc": a["base"], "--vacc-h": a["hover"], "--vacc-t": a["text"],
        "--vacc-s": a["subtle"], "--von-accent": a["on"],
        "--vt1": m["t1"], "--vt2": m["t2"], "--vt3": m["t3"], "--vbd": m["bd"],
        "--vpanel": m["panel"], "--vraised": m["raised"],
        "--vcard": m["card"], "--vcard2": m["card2"],
        "--gz": m["gz"], "--gx": m["gx"], "--gy": m["gy"],
        "--gfill": m["gfill"], "--gstroke": m["gstroke"],
        "--vok": m["ok"], "--vwarn": m["warn"], "--verr": m["err"],
        "--vkbd": m["kbd"],
    }


def css_vars(tokens):
    return ":root{" + "".join(f"{k}:{v};" for k, v in tokens.items()) + "}"


def light_css():
    """Non-token light adjustments (rules whose dark defaults are structural)."""
    return r"""
.railitem.disabled{opacity:.5}
.phname{color:var(--t2)}
.lightbar .lights i, .light .lights i{background:var(--lights)}
"""


# ---------------------------------------------------------------------------
# Sparse cloud asset, recoloured per (viewport mode, accent).  The source SVG
# is authored once; only entity colours are substituted.
_BASE_SVG = None


def _grid_lines():
    """Ground-plane grid over the same quad the point cloud sits on (1200x760)."""
    P0 = (104.1, 536.8); P1 = (764.9, 732.2); P2 = (1099.2, 514.9); P3 = (438.4, 319.6)
    lerp = lambda a, b, t: (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t)
    out = []
    n = 8
    for i in range(1, n):
        t = i / n
        a = lerp(P0, P3, t); b = lerp(P1, P2, t)
        out.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}"/>')
        a = lerp(P0, P1, t); b = lerp(P3, P2, t)
        out.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}"/>')
    return "".join(out)


def _scale_path(d, k):
    """Scale an absolute-coordinate path (M/L/Z only) about its centroid."""
    nums = [float(v) for v in re.findall(r'-?\d+\.?\d*', d)]
    n = len(nums) // 2
    cx = sum(nums[0::2]) / n; cy = sum(nums[1::2]) / n
    out, i = [], 0
    for tok in re.finditer(r'[MLml]|-?\d+\.?\d*|[Zz]', d):
        t = tok.group(0)
        if t in 'Zz':
            out.append('Z')
        elif t in 'MLml':
            out.append(t)
        else:
            v = float(t)
            out.append(f'{cx + (v - cx) * k:.1f}' if i % 2 == 0 else f'{cy + (v - cy) * k:.1f}')
            i += 1
    return ' '.join(out)


def cloud_svg(accent, mode):
    global _BASE_SVG
    if _BASE_SVG is None:
        _BASE_SVG = (ASSETS / "sparse_cloud.svg").read_text()
    m = VP_MODES[mode]
    ent = VP_ENTITIES[(ACCENTS[accent]["vp"]["family"], mode)]
    s = _BASE_SVG
    # The source's first group is the tie-line fan (long rays, one opacity).
    s = s.replace('<g stroke="#3A4A54" stroke-width="1" opacity="0.45">',
                  f'<g stroke="{m["tie"]}" stroke-width="1" opacity="{m["tie_op"]}">')
    s = s.replace('stroke="#3B736A"', f'stroke="{m["bbox"]}"')
    # Ground grid, inserted before the bounding box so points sit on top.
    grid = (f'<g class="grid" stroke="{m["grid"]}" stroke-width="0.8" '
            f'opacity="{m["grid_op"]}" stroke-linecap="round">{_grid_lines()}</g>')
    s = s.replace('<line x1="104.1" y1="536.8"', grid + '<line x1="104.1" y1="536.8"', 1)
    # point ramp
    s = s.replace('fill="#7F9AA8"', f'fill="{m["p1"]}"').replace('opacity="0.55"', f'opacity="{m["p1_op"]}"')
    s = s.replace('fill="#A9C6D4"', f'fill="{m["p2"]}"').replace('fill="#DCEBF3"', f'fill="{m["p3"]}"')
    s = s.replace('opacity="0.88"', f'opacity="{m["p2_op"]}"')
    # cameras
    s = s.replace('stroke-opacity="0.3"', f'stroke-opacity="{ent["cam_op"]}"')
    s = s.replace('stroke-opacity="0.5"', f'stroke-opacity="{ent["cam_op"]}"')
    s = s.replace('#F2C94C', ent["cam"])
    # GCP markers: solid diamond (warm) or white core + orange ring (cool).
    # The cool family scales the 8 px arrowhead 1.6x about its own centre so
    # the white core and ring read at canvas zoom.
    def gcp(mm):
        d = mm.group(2) if ent["signal"] == "ring" else mm.group(2)
        if ent["signal"] == "ring":
            d = _scale_path(d, 1.6)
        return (mm.group(1) + '"' + d + '"' + mm.group(3) + f'"{ent["core"]}"' +
                mm.group(4) + f'"{ent["ring"]}"' + mm.group(5) + f'"{ent["ring_w"]}"')
    s = re.sub(r'(<path d=)"([^"]+)"( fill=)"#F2994A"( stroke=)"#2A1A0C"( stroke-width=)"0.6"', gcp, s)
    s = s.replace('fill="#F5C08C"', f'fill="{ent["label"]}"')
    # selection: highlight the 6th camera frustum (aerial_r1c1.jpg) in the
    # accent colour, drawn last so it sits above the point cloud.
    polys = re.findall(r'<polygon points="([^"]+)"', s)
    if len(polys) >= 6:
        sel = ACCENTS[accent][mode]["text"] if mode == "dark" else ACCENTS[accent][mode]["base"]
        s = s.replace('</svg>',
                      f'<polygon points="{polys[5]}" fill="{sel}" fill-opacity="0.10" '
                      f'stroke="{sel}" stroke-width="1.8" stroke-linejoin="round"/></svg>')
    out = ASSETS / f"cloud-{mode}-{accent}.svg"
    out.write_text(s)
    return out.name


def palette(theme, accent, vp_mode):
    """JSON-serialisable palette for the DOM contrast checker."""
    a = ACCENTS[accent][theme]
    av = ACCENTS[accent][vp_mode]       # accent values for the canvas's own mode
    s = SURFACES[theme]
    m = VP_MODES[vp_mode]
    ent = VP_ENTITIES[(ACCENTS[accent]["vp"]["family"], vp_mode)]
    sel = av["text"] if vp_mode == "dark" else av["base"]
    return {
        "theme": theme, "accent": accent, "vp": vp_mode,
        "family": ACCENTS[accent]["vp"]["family"],
        "app": s["app"], "panel": s["panel"], "raised": s["raised"],
        "overlay": s["overlay"], "sunken": s["sunken"],
        "t1": s["t1"], "t2": s["t2"], "t3": s["t3"],
        "accent": a["base"], "accent_h": a["hover"], "accent_p": a["pressed"],
        "accent_text": a["text"], "accent_subtle": a["subtle"], "on_accent": a["on"],
        "sel": a["sel"],
        "ok": a["ok"], "okbg": a["okbg"],
        "warn": a["warn"], "warnbg": a["warnbg"],
        "err": a["err"], "errbg": a["errbg"],
        "info": a["info"], "proc": a["proc"],
        "vp_hi": m["hi"], "vp_mid": m["base"], "vp_lo": m["lo"],
        "vp_grid": m["grid"], "vp_grid_op": float(m["grid_op"]), "vp_bbox": m["bbox"],
        "vp_tie": m["tie"], "vp_tie_op": float(m["tie_op"]),
        "vp_p1": m["p1"], "vp_p1_op": float(m["p1_op"]),
        "vp_p2": m["p2"], "vp_p2_op": float(m["p2_op"]),
        "vp_p3": m["p3"], "vp_p3_op": float(m["p3_op"]),
        "vp_cam": ent["cam"], "vp_cam_op": float(ent["cam_op"]),
        "vp_core": ent["core"], "vp_ring": ent["ring"], "vp_label": ent["label"],
        "vp_gcp_signal": ent[ent["signal"]],
        "vp_sel_subtle": av["subtle"],
        "vp_kbd": m["kbd"],
        "vp_panel": m["panel"],
        "vp_sel": sel,
    }
