import sys, json
sys.path.insert(0, "/Volumes/Extreme SSD/Projects/alphashape/design/mockups")
from _common import icon, page, LIGHT_CSS, OUT
from _tokens import theme_tokens, palette, cloud_svg


def dimmed_main(vp, cloud):
    return f'''
  <div class="dim">
    <div class="mini-rail">{icon("folder-input",16)}{icon("scan",16)}{icon("crosshair",16)}{icon("cloud",16)}</div>
    <div class="mini-ws">
      <div class="mini-h">Workspace</div>
      <div class="mini-r strong">Chunk 1</div>
      <div class="mini-r">Cameras · 12</div>
      <div class="mini-r">Markers · 4</div>
      <div class="mini-r">Tie points · 4,945</div>
      <div class="mini-r hi">Point cloud</div>
      <div class="mini-r">Model</div>
      <div class="mini-r">DEM</div>
    </div>
    <div class="mini-center">
      <div class="mini-vp"><img src="assets/{cloud}" alt=""></div>
      <div class="mini-strip"><i></i><i></i><i></i><i></i><i></i><i></i></div>
    </div>
    <div class="mini-right"><div class="mini-h">Reference</div><div class="mini-r">aerial_r0c0.jpg 0.38</div><div class="mini-r">aerial_r0c1.jpg 0.41</div><div class="mini-r">aerial_r0c2.jpg 0.44</div><div class="mini-r">GCP-01 0.014</div></div>
  </div>
  <div class="scrim"></div>'''


def dialog():
    return f'''
<div class="dlgwrap">
  <div class="dlg">
    <div class="dlghead">
      <div class="dh-l">
        <span class="dhi">{icon("cloud",16,1.8)}</span>
        <div>
          <div class="dh-t">Build point cloud</div>
          <div class="dh-s">Chunk 1 · 12 aligned cameras · 4 markers</div>
        </div>
      </div>
      <span class="iconbtn"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18M6 6l12 12"/></svg></span>
    </div>

    <div class="dlgbody">
      <div class="presets-h">
        <span class="label">Quality</span>
        <span class="tiny mut">Sets depth-map resolution and filtering. This dataset: 12 images, GSD 0.29 m.</span>
      </div>
      <div class="presets">
        <div class="preset">
          <div class="pr-top"><span class="pr-n">Preview</span><span class="pr-b">fast</span></div>
          <div class="pr-d">Downscaled depth maps. Good for checking coverage before a full run.</div>
          <div class="pr-stat mono">~0.3 M pts · 00:02:20</div>
        </div>
        <div class="preset on">
          <div class="pr-top"><span class="pr-n">Balanced</span><span class="pr-b rec">recommended</span></div>
          <div class="pr-d">Full-resolution depth maps with moderate filtering. The default for survey work.</div>
          <div class="pr-stat mono">~1.2 M pts · 00:06:50</div>
        </div>
        <div class="preset">
          <div class="pr-top"><span class="pr-n">Dense</span><span class="pr-b">slowest</span></div>
          <div class="pr-d">No downscaling, aggressive filtering. Use when thin structures matter.</div>
          <div class="pr-stat mono">~4.8 M pts · 00:24:30</div>
        </div>
      </div>

      <div class="plainset">
        <div class="ps-i">{icon("info",14)}</div>
        <div>
          <div class="ps-t">What this does</div>
          <div class="ps-d">Matches pixels across overlapping photos and triangulates depth for each camera,
            producing a coloured point cloud you can measure on, classify, and export.</div>
        </div>
      </div>

      <div class="advanced">
        <span class="adv-c">{icon("chevron-right",13)}</span>
        <span class="adv-t">Advanced</span>
        <span class="tiny mut">12 parameters · depth filtering, reuse depth maps, confidence threshold…</span>
      </div>

      <div class="cols">
        <div class="col">
          <div class="col-h"><span class="label">Estimate</span><span class="chip ok">confident</span></div>
          <div class="est">
            <div class="estr"><span class="k">{icon("clock",12)}Time</span><span class="v mono">00:06:50</span><span class="r">±12%</span></div>
            <div class="estr"><span class="k">{icon("memory",12)}Peak RAM</span><span class="v mono">3.1 GB</span><span class="r">of 12 GB</span></div>
            <div class="estr"><span class="k">{icon("hard-drive",12)}Disk</span><span class="v mono">1.42 GB</span><span class="r">on Projects</span></div>
            <div class="estr"><span class="k">{icon("cloud",12)}Points</span><span class="v mono">1.2 M</span><span class="r">~0.03 pts/px</span></div>
          </div>
          <div class="estnote tiny mut">Based on this block's 12 cameras and the last 3 runs on M1.
            <span style="color:var(--accent-t)">How estimates are computed</span></div>
        </div>
        <div class="col">
          <div class="col-h"><span class="label">Will apply</span></div>
          <div class="apply">
            <div class="ap"><span class="dot idle"></span><span>Masks · none <span class="mut" style="font-weight:400">(0 / 12 cameras)</span></span><span class="mono tiny mut">—</span></div>
            <div class="ap"><span class="dot ok"></span><span>Colour · from source images</span><span class="mono tiny mut">RGB</span></div>
            <div class="ap"><span class="dot ok"></span><span>Region · bounding box</span><span class="mono tiny mut">120 × 90 × 12 m</span></div>
            <div class="ap"><span class="dot idle"></span><span>Classification</span><span class="mono tiny mut">off</span></div>
          </div>
          <div class="warnbox">
            {icon("alert",13)}
            <span>Same settings on a 480-image block would need <b>~11 GB RAM</b>. Estimates scale with camera count.</span>
          </div>
        </div>
      </div>
    </div>

    <div class="dlgfoot">
      <label class="remember"><span class="toggle on">{icon("check",11)}</span>Remember for this project</label>
      <span class="sp"></span>
      <span class="btn ghost">Cancel</span>
      <span class="btn">Add to queue</span>
      <span class="btn primary">{icon("play",12)}Run now</span>
    </div>
  </div>
</div>'''


CSS = r"""
.app{position:relative}
.dim{position:absolute;inset:38px 0 0 0;display:grid;grid-template-columns:44px 240px 1fr 280px;opacity:.5;filter:blur(.6px)}
.mini-rail{background:var(--app);border-right:1px solid var(--bd);display:flex;flex-direction:column;align-items:center;gap:16px;padding-top:14px;color:var(--t3)}
.mini-ws,.mini-right{background:var(--panel);border-right:1px solid var(--bd);padding:0}
.mini-right{border-right:none;border-left:1px solid var(--bd)}
.mini-h{height:30px;display:flex;align-items:center;padding:0 10px;font-size:10px;letter-spacing:.055em;text-transform:uppercase;color:var(--t2);border-bottom:1px solid var(--bd)}
.mini-r{height:24px;display:flex;align-items:center;padding:0 14px;font-size:12px;color:var(--t2)}
.mini-r.strong{font-weight:600;color:var(--t1)} .mini-r.hi{background:var(--sel);border-left:2px solid var(--sel-bd)}
.mini-center{display:grid;grid-template-rows:1fr 160px;min-width:0;min-height:0}
.mini-vp{background-color:var(--vp);position:relative;overflow:hidden;min-height:0}
.mini-vp img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:.7}
.mini-strip{background:var(--panel);border-top:1px solid var(--bd);display:flex;gap:6px;padding:8px;overflow:hidden}
.mini-strip i{width:100px;height:132px;flex:none;background:var(--raised);border:1px solid var(--bd2);border-radius:3px}
.scrim{position:absolute;inset:38px 0 0 0;background:var(--dim)}
.dlgwrap{position:absolute;inset:38px 0 0 0;display:flex;align-items:flex-start;justify-content:center;padding-top:62px}
.dlg{width:880px;background:var(--overlay);border:1px solid var(--bd2);border-radius:6px;overflow:hidden;
  box-shadow:0 24px 64px rgba(0,0,0,.6)}
.dlghead{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--bd);background:var(--panel)}
.dh-l{display:flex;align-items:center;gap:10px}
.dhi{width:30px;height:30px;border-radius:4px;background:var(--accent-s);border:1px solid var(--accent);color:var(--accent-t);
  display:flex;align-items:center;justify-content:center}
.dh-t{font-size:15px;font-weight:600;letter-spacing:-.01em}
.dh-s{font-size:11px;color:var(--t3);margin-top:1px}
.dlghead .iconbtn{margin-left:auto}
.dlgbody{padding:16px 18px 14px}
.presets-h{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.presets{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.preset{border:1px solid var(--bd2);border-radius:4px;background:var(--panel);padding:10px 11px;position:relative;min-height:112px}
.preset.on{border-color:var(--accent);background:linear-gradient(180deg,var(--accent-fade),transparent);box-shadow:0 0 0 1px var(--accent-ring)}
.pr-top{display:flex;align-items:center;gap:8px}
.pr-n{font-size:13px;font-weight:600}
.pr-b{font-size:9.5px;padding:0 6px;line-height:15px;border-radius:8px;border:1px solid var(--bd2);color:var(--t3);text-transform:uppercase;letter-spacing:.04em}
.preset.on .pr-b.rec{color:var(--on-accent);background:var(--accent);border-color:var(--accent)}
.pr-d{font-size:11px;line-height:16px;color:var(--t2);margin-top:7px}
.pr-stat{position:absolute;left:11px;bottom:9px;font-size:10.5px;color:var(--t3)}
.preset.on .pr-stat{color:var(--accent-t)}
.plainset{display:flex;gap:10px;margin-top:12px;padding:10px 12px;border:1px solid var(--bd);border-radius:4px;background:var(--panel)}
.ps-i{color:var(--info);margin-top:1px}
.ps-t{font-size:11.5px;font-weight:600}
.ps-d{font-size:11.5px;line-height:17px;color:var(--t2);margin-top:2px}
.advanced{display:flex;align-items:center;gap:7px;margin-top:10px;padding:7px 2px;border-top:1px solid var(--bd);border-bottom:1px solid var(--bd)}
.adv-c{color:var(--t3);display:flex} .adv-t{font-size:12px;font-weight:500}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:14px}
.col-h{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.est{display:flex;flex-direction:column}
.estr{display:grid;grid-template-columns:1fr auto 74px;gap:8px;align-items:center;height:26px;border-bottom:1px solid var(--bd);font-size:11.5px}
.estr .k{display:flex;align-items:center;gap:7px;color:var(--t2)}
.estr .k svg{color:var(--t3)}
.estr .v{font-size:12px;color:var(--t1)}
.estr .r{text-align:right;font-size:10.5px;color:var(--t3)}
.estnote{margin-top:8px;line-height:15px;color:var(--t2)}
.apply{display:flex;flex-direction:column}
.ap{display:grid;grid-template-columns:10px 1fr auto;gap:8px;align-items:center;height:26px;border-bottom:1px solid var(--bd);font-size:11.5px;color:var(--t2)}
.ap:first-child{color:var(--t1)}
.warnbox{display:flex;gap:8px;margin-top:9px;padding:8px 10px;border:1px solid var(--warnbd);background:var(--warnbg);
  border-radius:4px;font-size:11px;line-height:16px;color:var(--t2)}
.warnbox span{color:var(--t1)}
.warnbox svg{color:var(--warn);margin-top:1px}
.warnbox b{color:var(--t1);font-weight:600}
.dlgfoot{display:flex;align-items:center;gap:8px;padding:12px 16px;border-top:1px solid var(--bd);background:var(--panel)}
.remember{display:flex;align-items:center;gap:7px;font-size:11.5px;color:var(--t2)}
.dlgfoot .sp{flex:1}
.light .dlg{background:#fff}
"""


def render(theme, accent, vp, name):
    tokens = theme_tokens(theme, accent, vp)
    body = f'''<div class="app">
  <div class="winbar">
    <div class="lights"><i></i><i></i><i></i></div>
    <div class="brand"><span style="color:var(--accent-t)">{icon("mountain",16,1.7)}</span>alphashape</div>
    <div class="mut" style="font-size:11px">Rheinfeld survey.alphashape</div>
    <div class="right"><span class="chip">{icon("cpu",11)}GPU · M1</span><span class="chip">74.2 GB free</span></div>
  </div>
  {dimmed_main(vp, cloud_svg(accent, vp))}
  {dialog()}
</div>'''
    css = CSS + (LIGHT_CSS if theme == "light" else "")
    html = page(f"alphashape — Build point cloud ({theme}, {accent})", body, css,
                tokens=tokens, body_class="light" if theme == "light" else "")
    (OUT / f"{name}.html").write_text(html)
    (OUT / f"{name}.palette.json").write_text(json.dumps(palette(theme, accent, vp), indent=1))
    return name


RENDERS = [
    ("dark", "ember", "dark", "r3-dialog-dark-ember"),
    ("light", "ember", "light", "r3-dialog-light-ember"),
]

if __name__ == "__main__":
    for theme, accent, vp, name in RENDERS:
        print("written", render(theme, accent, vp, name))
