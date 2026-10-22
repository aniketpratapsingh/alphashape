import sys, json
sys.path.insert(0, "/Volumes/Extreme SSD/Projects/alphashape/design/mockups")
from _common import icon, page, LIGHT_CSS, OUT
from _tokens import theme_tokens, cloud_svg, palette, ACCENTS

# camera, reprojection error (px), projections, estimated ground error (m)
CAM_ROWS = [
    ("aerial_r0c0.jpg", 0.38, 1284, 0.041), ("aerial_r0c1.jpg", 0.41, 1301, 0.044), ("aerial_r0c2.jpg", 0.44, 1298, 0.047),
    ("aerial_r0c3.jpg", 0.40, 1277, 0.043), ("aerial_r1c0.jpg", 0.45, 1310, 0.048), ("aerial_r1c1.jpg", 0.41, 1284, 0.044),
    ("aerial_r1c2.jpg", 0.43, 1299, 0.046), ("aerial_r1c3.jpg", 0.47, 1265, 0.050), ("aerial_r2c0.jpg", 0.51, 1240, 0.057),
    ("aerial_r2c1.jpg", 0.49, 1252, 0.054), ("aerial_r2c2.jpg", 0.62, 1198, 0.068), ("aerial_r2c3.jpg", 0.46, 1271, 0.049),
]
MARK_ROWS = [("GCP-01", 0.014, 6), ("GCP-02", 0.019, 5), ("GCP-03", 0.031, 6), ("GCP-04", 0.022, 5)]


def cam_rows():
    out = []
    for f, px, n, m in CAM_ROWS:
        cls = "sel" if f == "aerial_r1c1.jpg" else ""
        color = "okt" if px < 0.5 else "warnt"
        out.append(
            f'<div class="rr {cls}"><span class="mono tiny nm">{f}</span>'
            f'<span class="num tiny mut">{m:.3f}</span>'
            f'<span class="num tiny {color}">{px:.2f}</span>'
            f'<span class="num tiny mut">{n}</span></div>'
        )
    return "".join(out)


def marker_rows():
    out = []
    for m, e, n in MARK_ROWS:
        color = "okt" if e < 0.03 else "warnt"
        out.append(
            f'<div class="rr"><span class="mono tiny nm">{m}</span>'
            f'<span class="num tiny {color}">{e:.3f}</span>'
            f'<span class="num tiny mut">—</span>'
            f'<span class="num tiny mut">{n}</span></div>'
        )
    return "".join(out)


def photos():
    out = []
    i = 0
    for r in range(3):
        for c in range(4):
            sel = "sel" if i == 5 else ""
            px = CAM_ROWS[i][1]
            bcls = "ok" if px < 0.5 else "warn"
            out.append(
                f'<div class="ph {sel}"><img src="assets/aerial_r{r}c{c}.jpg" alt="">'
                f'<span class="phbadge {bcls}">{px:.2f}</span>'
                f'<span class="phname mono">aerial_r{r}c{c}.jpg</span></div>'
            )
            i += 1
    return "".join(out)


def camera_tree(count=3):
    rows = []
    for i in range(count):
        rows.append(
            f'<div class="row l2"><span class="tw"></span>{icon("image",13,1.7,"ico dim")}'
            f'<span class="nm mono tiny">aerial_r0c{i}.jpg</span>'
            f'<span class="badge g">{0.38 + i * 0.03:.2f} px</span><span class="sp"></span>'
            f'<span class="ic">{icon("eye",12,1.7)}</span></div>'
        )
    return "".join(rows)


def vp_mode_control(mode, theme, menu_open=False):
    """Little viewport dropdown: Background  Follow theme / Dark / Light."""
    opts = [("Follow theme", mode == "theme"), ("Dark", mode == "dark"), ("Light", mode == "light")]
    cur = {"theme": "Follow theme", "dark": "Dark", "light": "Light"}[mode]
    items = "".join(
        f'<span class="vmenu-item{" on" if on else ""}">{icon("check",10) if on else ""}<span>{lbl}</span></span>'
        for lbl, on in opts
    )
    return f'''<span class="vbgctl" title="Viewport background: Follow theme / Dark / Light">
          {icon("contrast",14)}<span class="vbg-l">Background</span><span class="vbg-v">{cur}</span>{icon("chevron-down",11)}
          <span class="vmenu{' open' if menu_open else ''}"><span class="vmenu-h">Viewport background</span>{items}</span>
        </span>'''


def shell(theme="dark", accent="teal", vp="dark", theme_icon="sun", menu_open=False, vp_display=None):
    cloud = cloud_svg(accent, vp)
    return f'''
<div class="app">
  <div class="winbar">
    <div class="lights"><i></i><i></i><i></i></div>
    <div class="brand"><span style="color:var(--accent-t)">{icon("mountain",16,1.7)}</span>alphashape</div>
    <div class="mut" style="font-size:11px">Rheinfeld survey.alphashape</div>
    <div class="right">
      <span class="chip">{icon("cpu",11)}GPU · M1</span>
      <span class="chip">{icon("hard-drive",11)}74.2 GB free</span>
      <span class="iconbtn" title="Command palette">{icon("command",15)}</span>
      <span class="iconbtn" title="Theme">{icon(theme_icon,15)}</span>
      <span class="iconbtn" title="Preferences">{icon("settings",15)}</span>
    </div>
  </div>

  <div class="work">
    <div class="rail">
      <div class="railitem done"><span class="rn">{icon("folder-input",17,1.7)}</span><span class="rlabel">Photos</span></div>
      <div class="railitem done"><span class="rn">{icon("scan",17,1.7)}</span><span class="rlabel">Align</span></div>
      <div class="railitem done"><span class="rn">{icon("crosshair",17,1.7)}</span><span class="rlabel">Control</span></div>
      <div class="railitem active"><span class="rn">{icon("cloud",17,1.7)}</span><span class="rlabel">Points</span></div>
      <div class="railitem disabled"><span class="rn">{icon("box",17,1.7)}</span><span class="rlabel">Surface</span></div>
      <div class="railitem disabled"><span class="rn">{icon("badge",17,1.7)}</span><span class="rlabel">Check</span></div>
      <div class="railitem disabled"><span class="rn">{icon("package",17,1.7)}</span><span class="rlabel">Export</span></div>
      <div style="flex:1"></div>
      <div class="railitem" style="height:34px">{icon("panels",16)}</div>
    </div>

    <div class="dock ws">
      <div class="dockhead">
        <span class="ttl">Workspace</span>
        <div class="right">
          <span class="iconbtn">{icon("plus",14)}</span>
          <span class="iconbtn">{icon("filter",14)}</span>
          <span class="iconbtn">{icon("more",14)}</span>
        </div>
      </div>
      <div class="wssearch"><div class="field" style="margin:6px 8px">{icon("search",12)}<input placeholder="Search cameras, markers…"><span class="kbd">/</span></div></div>
      <div class="tree scroll">
        <div class="row root"><span class="tw">{icon("chevron-down",12)}</span>{icon("layers",14,1.7,"ico")}<span class="nm">Chunk 1</span><span class="badge">12 aligned</span><span class="sp"></span><span class="ic">{icon("eye",13,1.7)}</span></div>
        <div class="row l1"><span class="tw">{icon("chevron-down",12)}</span>{icon("camera",14,1.7,"ico")}<span class="nm">Cameras</span><span class="cnt">12</span><span class="sp"></span><span class="ic">{icon("eye",13,1.7)}</span></div>
        {camera_tree(3)}
        <div class="row l2 sel"><span class="tw"></span>{icon("image",13,1.7,"ico dim")}<span class="nm mono tiny">aerial_r1c1.jpg</span><span class="badge">selected</span><span class="sp"></span><span class="ic">{icon("eye",12,1.7)}</span></div>
        <div class="row l2"><span class="tw"></span>{icon("image",13,1.7,"ico dim")}<span class="nm mono tiny">aerial_r1c2.jpg</span><span class="badge g">0.43 px</span><span class="sp"></span><span class="ic">{icon("eye",12,1.7)}</span></div>
        <div class="row l2 dimrow"><span class="tw"></span>{icon("image",13,1.7,"ico dim")}<span class="nm mono tiny">+ 7 more</span></div>
        <div class="row l1"><span class="tw">{icon("chevron-right",12)}</span>{icon("crosshair",14,1.7,"ico")}<span class="nm">Markers</span><span class="cnt">4</span><span class="endbadge">RMSE 0.021 m</span></div>
        <div class="row l1"><span class="tw"></span>{icon("scan",14,1.7,"ico")}<span class="nm">Tie points</span><span class="cnt">4,945</span><span class="sp"></span><span class="ic">{icon("eye",13,1.7)}</span></div>
        <div class="row l1"><span class="tw"></span>{icon("layers",14,1.7,"ico")}<span class="nm">Depth maps</span><span class="cnt">—</span><span class="sp"></span><span class="ic">{icon("eyeoff",13,1.7)}</span></div>
        <div class="row l1 next"><span class="tw"></span>{icon("cloud",14,1.7,"ico")}<span class="nm">Point cloud</span><span class="cnt">—</span><span class="sp"></span><span class="go">Build {icon("arrow-right",11)}</span></div>
        <div class="row l1 dimrow"><span class="tw"></span>{icon("box",14,1.7,"ico")}<span class="nm">Model</span><span class="cnt">—</span></div>
        <div class="row l1 dimrow"><span class="tw"></span>{icon("mountain2",14,1.7,"ico")}<span class="nm">DEM</span><span class="cnt">—</span></div>
        <div class="row l1 dimrow"><span class="tw"></span>{icon("map",14,1.7,"ico")}<span class="nm">Orthomosaic</span><span class="cnt">—</span></div>
      </div>
      <div class="wsfoot">
        <div class="tiny mut" style="display:flex;justify-content:space-between"><span>12 cameras · 4 markers</span><span class="mono">4,945 pts</span></div>
        <div class="tiny" style="color:var(--t2);margin-top:4px;display:flex;align-items:center;gap:5px">{icon("info",11)}<span>Alignment done · median 0.42 px</span></div>
      </div>
    </div>

    <div class="center">
      <div class="viewport">
        <div class="vptop">
          <div class="modes">
            <span class="iconbtn on">{icon("orbit",15)}</span>
            <span class="iconbtn">{icon("hand",15)}</span>
            <span class="iconbtn">{icon("zoom",15)}</span>
          </div>
          <div class="vsep"></div>
          <div class="modes">
            <span class="vchip on">Sparse</span><span class="vchip">Dense</span><span class="vchip">Mesh</span>
          </div>
          <div class="vsep"></div>
          <span class="iconbtn on">{icon("camera",15)}</span>
          <span class="iconbtn on">{icon("crosshair",15)}</span>
          <span class="iconbtn">{icon("grid3d",15)}</span>
          <div class="vsep"></div>
          {vp_mode_control(vp_display or vp, theme, menu_open)}
          <div style="flex:1"></div>
          <span class="vchip ghost">Perspective</span>
          <span class="iconbtn">{icon("fit",15)}</span>
        </div>
        <div class="canvas">
          <img class="cloudimg" src="assets/{cloud}" alt="">
          <div class="cvignette"></div>
          <div class="vp-info">
            <div class="mono tiny" style="color:var(--vt2)">Tie points · 4,945</div>
            <div class="mono micro" style="margin-top:3px;color:var(--vt3)">Colour: height · 0–9.5 m</div>
            <div class="legend"><span style="background:var(--vp-p1)"></span><span style="background:var(--vp-p2)"></span><span style="background:var(--vp-p3)"></span></div>
          </div>
          <div class="scalebar"><div class="bar"></div><span class="mono micro">20 m</span></div>
          <svg class="gizmo" viewBox="0 0 90 90">
            <circle cx="45" cy="45" r="34" fill="var(--gfill)" stroke="var(--gstroke)"/>
            <line x1="45" y1="45" x2="45" y2="16" stroke="var(--gz)" stroke-width="1.8"/>
            <line x1="45" y1="45" x2="71" y2="56" stroke="var(--gx)" stroke-width="1.8"/>
            <line x1="45" y1="45" x2="19" y2="56" stroke="var(--gy)" stroke-width="1.8"/>
            <text x="41" y="12" fill="var(--gz)" font-family="JetBrains Mono" font-size="9">Z</text>
            <text x="74" y="62" fill="var(--gx)" font-family="JetBrains Mono" font-size="9">X</text>
            <text x="9" y="62" fill="var(--gy)" font-family="JetBrains Mono" font-size="9">Y</text>
          </svg>
          <div class="selcard">
            <div class="sc-h">{icon("image",13)}<span class="mono tiny" style="color:var(--vt2)">aerial_r1c1.jpg</span><span class="sp"></span><span class="chip">500 px</span></div>
            <div class="sc-grid">
              <span class="k">Centre</span><span class="mono tiny num">−11.000, 0.000, 60.000</span>
              <span class="k">Reproj.</span><span class="mono tiny num okt">0.41 px</span>
              <span class="k">Observed</span><span class="mono tiny num">1,284 points</span>
            </div>
            <div class="sc-actions"><span class="btn sm ghost">Open photo</span><span class="btn sm ghost">Disable</span></div>
          </div>
          <div class="vp-hint"><span class="kbd">LMB</span> orbit <span class="kbd">MMB</span> pan <span class="kbd">wheel</span> zoom <span class="kbd">F</span> fit</div>
        </div>
      </div>

      <div class="dock photos">
        <div class="tabs">
          <span class="tab on">{icon("images",13)}Photos <span class="cnt">12</span></span>
          <span class="tab">{icon("contrast",13)}Masks <span class="cnt">0</span></span>
          <span class="tab">{icon("timer",13)}Depth <span class="cnt">—</span></span>
          <div style="flex:1"></div>
          <span class="phfilter">{icon("search",12)}<input placeholder="Filter…"></span>
          <span class="iconbtn">{icon("grid",14)}</span>
          <span class="iconbtn">{icon("filter",14)}</span>
        </div>
        <div class="strip">{photos()}</div>
      </div>
    </div>

    <div class="rightdock">
      <div class="tabs">
        <span class="tab">Properties</span>
        <span class="tab on">Reference <span class="cnt">16</span></span>
      </div>
      <div class="reftoolbar">
        <span class="iconbtn on" title="Pick in viewport">{icon("crosshair",14)}</span>
        <span class="iconbtn" title="Re-optimise">{icon("refresh",14)}</span>
        <span class="iconbtn" title="Import…">{icon("upload",14)}</span>
        <span class="iconbtn" title="Export…">{icon("download",14)}</span>
        <div style="flex:1"></div>
        <span class="btn sm" title="Optimise camera and marker alignment">{icon("wand",12)}Optimize</span>
      </div>
      <div class="refmeta">
        <span class="chip acc">{icon("crosshair",11)}4 markers</span>
        <span class="chip ok">RMSE 0.021 m</span>
        <span class="chip">EPSG:25832</span>
      </div>
      <div class="reftable">
        <div class="rt-head"><span>Camera</span><span class="num">Error (m)</span><span class="num">Error (px)</span><span class="num">Projections</span></div>
        <div class="rt scroll">
          {cam_rows()}
          <div class="rr rhead" style="margin-top:6px"><span>Markers</span><span class="num">est.</span><span class="num">reproj.</span><span class="num">obs.</span></div>
          {marker_rows()}
        </div>
      </div>
      <div class="reffoot">
        <div class="tiny mut">Errors estimated vs. source coordinates · sort by |error|</div>
      </div>

      <div class="jobcard">
        <div class="jobhead"><span class="dot run"></span><span class="label" style="letter-spacing:.02em">Processing</span><span class="sp"></span><span class="mono micro mut">00:04:12</span></div>
        <div class="jobtitle">Build point cloud <span class="mut" style="font-weight:400">· Chunk 1</span></div>
        <div class="jobstage tiny mut">Depth maps · 7 of 12 cameras</div>
        <div class="progress proc" style="margin:7px 0 6px"><i style="width:58%"></i></div>
        <div class="jobmeta tiny mut"><span>ETA 00:02:40</span><span class="sp"></span><span class="mono">RAM 3.1 / 12 GB</span></div>
        <div class="joba">
          <span class="btn sm ghost">{icon("pause",12)}Pause</span>
          <span class="btn sm ghost">{icon("square",12)}Stop</span>
          <span class="sp"></span>
          <span class="tiny" style="color:var(--accent-t);display:inline-flex;align-items:center;gap:3px">Queue {icon("arrow-right",11)}</span>
        </div>
      </div>
    </div>
  </div>

  <div class="statusbar">
    <span class="sb">{icon("layers",12)}Chunk 1</span><span class="sbd"></span>
    <span class="sb mono">EPSG:25832</span><span class="sbd"></span>
    <span class="sb mono">GSD 0.29 m</span><span class="sbd"></span>
    <span class="sb mono">X 9.200000° · Y 48.700000°</span>
    <span class="sp"></span>
    <span class="sb jobpill"><span class="dot run"></span>1 job running<span class="mono mut">· ETA 00:02:40</span></span>
    <span class="sbd"></span><span class="sb mono">12 / 12 aligned</span>
  </div>
</div>'''


CSS = r"""
.work{display:grid;grid-template-columns:44px 268px minmax(0,1fr) 320px;min-height:0;gap:0}
.rail{background:var(--app);border-right:1px solid var(--bd);display:flex;flex-direction:column;align-items:center;padding:8px 0;gap:4px}
.railitem{width:40px;height:46px;border-radius:5px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;
  color:var(--t3);border:1px solid transparent;position:relative}
.railitem .rlabel{font-size:9px;letter-spacing:.02em}
.railitem:hover{background:var(--raised);color:var(--t2)}
.railitem.done{color:var(--t2)}
.railitem.done:after{content:"";position:absolute;left:6px;top:7px;width:5px;height:5px;border-radius:50%;background:var(--ok)}
.railitem.active{background:var(--accent-s);border-color:var(--accent);color:var(--accent-t)}
.railitem.active:after{content:"";position:absolute;left:6px;top:7px;width:5px;height:5px;border-radius:50%;background:var(--accent)}
.railitem.disabled{opacity:.42}
.ws{border-left:none;border-top:none;border-bottom:none;border-right:1px solid var(--bd)}
.ws .tree{flex:1;overflow:auto;font-size:12px}
.row{display:flex;align-items:center;gap:6px;height:22px;padding:0 8px 0 4px;border-left:2px solid transparent;white-space:nowrap}
.row:hover{background:var(--hover)}
.row.sel{background:var(--sel);border-left-color:var(--sel-bd)}
.row .tw{width:14px;display:flex;justify-content:center;color:var(--t3)}
.row .ico{color:var(--t2)}
.row .nm{overflow:hidden;text-overflow:ellipsis}
.row .sp{flex:1}
.row .cnt{color:var(--t3);font-size:10.5px;font-variant-numeric:tabular-nums}
.row .badge{margin-left:auto;font-size:9.5px;padding:0 5px;line-height:14px;border-radius:8px;border:1px solid var(--bd2);color:var(--t2);background:var(--raised)}
.row .badge.g{color:var(--ok);border-color:var(--okbd);background:var(--okbg)}
.row .endbadge{margin-left:auto;font-size:9.5px;color:var(--ok)}
.row .go{margin-left:auto;font-size:10px;color:var(--accent-t);display:flex;align-items:center;gap:3px}
.row .ic{display:flex;align-items:center;color:var(--t3)}
.row.root{height:26px;font-weight:600}
.row.l1{padding-left:18px}
.row.l2{padding-left:36px}
.row.dimrow{color:var(--t3)}
.row.next{background:linear-gradient(90deg,var(--accent-fade),transparent 70%)}
.wsfoot{border-top:1px solid var(--bd);padding:7px 9px}
.center{display:grid;grid-template-columns:minmax(0,1fr);grid-template-rows:minmax(0,1fr) 160px;min-width:0;min-height:0}
/* ---- viewport: toolbar + overlays are part of the instrument pane and
   follow --vp-* (the canvas mode), never the app theme. ---- */
.viewport{position:relative;min-width:0;overflow:hidden;background:radial-gradient(120% 100% at 45% 25%,var(--vp-hi) 0%,var(--vp) 45%,var(--vp2) 100%);
  background-color:var(--vp);border-bottom:1px solid var(--bd);display:flex;flex-direction:column}
.vptop{height:36px;flex:none;display:flex;align-items:center;gap:4px;padding:0 8px;background:var(--vpanel);
  border-bottom:1px solid var(--vbd);position:relative;z-index:3;min-width:0;color:var(--vt2)}
.vptop .iconbtn{color:var(--vt2)}
.vptop .iconbtn.on{background:var(--vacc-s);border-color:var(--vacc);color:var(--vacc-t)}
.vptop .vsep{background:var(--vbd)}
.vptop .vchip{color:var(--vt2)}
.vptop .vchip.on{color:var(--vt1);background:var(--vraised);border-color:var(--vbd)}
.vptop .vchip.ghost{color:var(--vt2);border-color:var(--vbd)}
.vbgctl{position:relative;display:inline-flex;align-items:center;gap:6px;height:22px;padding:0 7px;border-radius:3px;
  border:1px solid var(--vbd);color:var(--vt2);font-size:11px;white-space:nowrap;flex:none;background:var(--vraised)}
.vbgctl .vbg-l{color:var(--vt3);font-size:10.5px}
.vbgctl .vbg-v{color:var(--vt1)}
.vmenu{position:absolute;top:30px;left:0;width:190px;background:var(--vcard);border:1px solid var(--vbd);border-radius:4px;
  padding:4px;box-shadow:0 8px 24px rgba(0,0,0,.35);display:none;z-index:9}
.vbgctl:hover .vmenu, .vmenu.open{display:block}
.vmenu-h{font-size:9.5px;text-transform:uppercase;letter-spacing:.055em;color:var(--vt3);padding:4px 6px 5px}
.vmenu-item{display:flex;align-items:center;gap:6px;height:22px;padding:0 6px;border-radius:2px;color:var(--vt2);font-size:11.5px}
.vmenu-item svg{visibility:hidden}
.vmenu-item.on{color:var(--vt1);background:var(--vacc-s)}
.vmenu-item.on svg{visibility:visible;color:var(--vacc-t)}
.modes{display:flex;gap:2px;align-items:center}
.vsep{width:1px;height:18px;background:var(--vbd);margin:0 6px}
.vchip{padding:0 8px;height:22px;display:inline-flex;align-items:center;border-radius:3px;font-size:11px;color:var(--vt3);border:1px solid transparent}
.vchip.on{color:var(--vt1);background:var(--vraised);border-color:var(--vbd)}
.vchip.ghost{color:var(--vt2);border-color:var(--vbd)}
.canvas{position:relative;flex:1;min-height:0;overflow:hidden}
.cloudimg{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.cvignette{position:absolute;inset:0;background:var(--vvig)}
.vp-info{position:absolute;left:12px;top:12px;background:var(--vcard2);border:1px solid var(--vbd);border-radius:3px;padding:6px 8px}
.legend{display:flex;height:4px;width:92px;margin-top:5px;border-radius:2px;overflow:hidden}
.legend span{flex:1}
.scalebar{position:absolute;left:12px;bottom:12px;color:var(--vt2)}
.scalebar .bar{height:5px;border-left:1px solid var(--vt2);border-right:1px solid var(--vt2);border-bottom:1px solid var(--vt2);width:96px;margin-bottom:3px}
.scalebar .mono{color:var(--vt2)}
.gizmo{position:absolute;right:14px;bottom:12px;width:90px;height:90px}
.selcard{position:absolute;right:14px;top:12px;width:244px;background:var(--vcard2);
  border:1px solid var(--vbd);border-radius:4px;padding:8px;color:var(--vt1)}
.sc-h{display:flex;align-items:center;gap:6px;color:var(--vt2)}
.sc-grid{display:grid;grid-template-columns:60px 1fr;gap:3px 8px;margin-top:7px}
.sc-grid .k{color:var(--vt3);font-size:10.5px}
.sc-grid .mono{color:var(--vt1)}
.sc-actions{display:flex;gap:4px;margin-top:6px;border-top:1px solid var(--vbd);padding-top:5px}
.sc-actions .btn{color:var(--vt2);border-color:var(--vbd);background:transparent}
.sc-actions .btn:hover{background:var(--hover)}
.selcard .chip{background:var(--vraised);border-color:var(--vbd);color:var(--vt2)}
.vp-hint{margin-bottom:0}
.vp-hint .kbd, .kbdfixed{background:var(--vkbd);border-color:var(--vbd);color:var(--vt2)}
.dlgbody .kbd{background:var(--sunken)}
.vp-hint{position:absolute;left:50%;transform:translateX(-50%);bottom:10px;color:var(--vt2);font-size:10.5px;
  background:var(--vcard);border:1px solid var(--vbd);border-radius:3px;padding:3px 8px}
.photos .strip{display:flex;gap:6px;padding:8px;overflow-x:auto;flex:1;align-items:flex-start}
.ph{width:100px;flex:none;border:1px solid var(--bd2);border-radius:3px;overflow:hidden;position:relative;background:var(--phbg)}
.ph.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent);background:var(--accent-s)}
.ph img{width:100%;height:70px;object-fit:cover;display:block;filter:saturate(.95) contrast(1.02)}
.phbadge{position:absolute;right:3px;top:3px;height:15px;padding:0 4px;border-radius:7px;background:#0B0F12;
  font:500 9.5px "JetBrains Mono";display:flex;align-items:center;gap:2px;border:1px solid rgba(72,184,117,.45);color:#7DD99A}
.phbadge.warn{border-color:rgba(227,163,60,.5);color:#F0BE6A}
.phname{display:block;font-size:9px;line-height:13px;color:var(--t3);padding:2px 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ph.sel .phname{color:var(--accent-t);font-weight:500}
.phfilter{height:22px;border:1px solid var(--bd2);border-radius:3px;display:flex;align-items:center;gap:5px;padding:0 6px;color:var(--t3);margin:4px 6px 4px 0;width:150px;background:var(--sunken)}
.phfilter input{background:transparent;border:none;outline:none;color:var(--t1);font:11px Inter;width:100%}
.rightdock{display:grid;grid-template-rows:auto auto auto minmax(0,1fr) auto auto;min-height:0;min-width:0;
  border-left:1px solid var(--bd);background:var(--panel)}
.rightdock .tabs{background:var(--panel)}
.reftoolbar{display:flex;align-items:center;gap:2px;padding:5px 7px;border-bottom:1px solid var(--bd)}
.refmeta{display:flex;gap:5px;padding:5px 7px;border-bottom:1px solid var(--bd);flex-wrap:wrap}
.reftable{min-height:0;display:flex;flex-direction:column}
.rt-head{display:grid;grid-template-columns:minmax(0,1fr) 54px 56px 74px;gap:3px;padding:5px 8px;border-bottom:1px solid var(--bd);
  font-size:9.5px;color:var(--t3);text-transform:uppercase;letter-spacing:.03em}
.rt-head .num{text-align:right}
.rt{flex:1;overflow:auto}
.rr{display:grid;grid-template-columns:minmax(0,1fr) 54px 56px 74px;gap:3px;align-items:center;height:21px;padding:0 8px;white-space:nowrap}
.rr:hover{background:var(--hover)}
.rr.sel{background:var(--sel);box-shadow:inset 2px 0 0 var(--sel-bd)}
.rr.sel .mut{color:var(--sel-t2);font-weight:500}
.rr.sel .okt{color:var(--sel-ok);font-weight:500}
.rr.sel .warnt{color:var(--sel-warn);font-weight:500}
.rr .nm{overflow:hidden;text-overflow:ellipsis}
.rr .num{text-align:right}
.rr.rhead{height:19px;color:var(--t3);font-size:9px;text-transform:uppercase;letter-spacing:.04em}
.okt{color:var(--ok)} .warnt{color:var(--warn)}
.reffoot{padding:6px 8px;border-top:1px solid var(--bd)}
.jobcard{padding:8px 10px;background:var(--raised);border-top:1px solid var(--bd2)}
.jobhead{display:flex;align-items:center;gap:7px}
.jobtitle{margin-top:6px;font-size:12px;font-weight:600}
.jobstage{margin-top:2px}
.jobmeta{display:flex;align-items:center;gap:8px;color:var(--t2)}
.joba{display:flex;align-items:center;gap:2px;margin-top:6px}
.statusbar{display:flex;align-items:center;gap:0;height:26px;padding:0 10px;background:var(--panel);border-top:1px solid var(--bd);
  font-size:10.5px;color:var(--t2)}
.sb{display:inline-flex;align-items:center;gap:5px;padding:0 10px}
.sbd{width:1px;height:12px;background:var(--bd2)}
.sp{flex:1}
.jobpill{display:inline-flex;align-items:center;gap:6px;color:var(--t2)}
"""


def render(theme, accent, vp, name, menu_open=False):
    tokens = theme_tokens(theme, accent, vp)
    # "Follow theme" is the stored default; only an explicit override shows a
    # concrete mode (the mixed light-chrome/dark-canvas render is a preview of
    # the user picking Dark in this menu).
    body = shell(theme, accent, vp, "sun" if theme == "dark" else "moon", menu_open,
                 vp_display="theme" if vp == theme else vp)
    css = CSS + (LIGHT_CSS if theme == "light" else "")
    html = page(f"alphashape — main ({theme}, {accent}, {vp} viewport)", body, css,
                tokens=tokens, body_class="light" if theme == "light" else "")
    path = OUT / f"{name}.html"
    path.write_text(html)
    (OUT / f"{name}.palette.json").write_text(json.dumps(palette(theme, accent, vp), indent=1))
    return path


RENDERS = [
    ("dark", "teal", "dark", "r3-main-dark-teal", False),
    ("dark", "ember", "dark", "r3-main-dark-ember", False),
    ("dark", "copper", "dark", "r3-main-dark-copper", False),
    ("light", "teal", "light", "r3-main-light-teal", False),
    ("light", "ember", "light", "r3-main-light-ember", False),
    ("light", "copper", "light", "r3-main-light-copper", False),
    # light chrome + dark canvas, with the background dropdown open so the
    # three options (Follow theme / Dark / Light) are visible.
    ("light", "ember", "dark", "r3-main-light-ember-darkviewport", True),
]

if __name__ == "__main__":
    which = sys.argv[1:] or [r[3] for r in RENDERS]
    for theme, accent, vp, name, mo in RENDERS:
        if name in which or not sys.argv[1:]:
            p = render(theme, accent, vp, name, mo)
            print("written", p.name)
