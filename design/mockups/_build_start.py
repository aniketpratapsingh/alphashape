import sys, json
sys.path.insert(0, "/Volumes/Extreme SSD/Projects/alphashape/design/mockups")
from _common import page, OUT
from _tokens import theme_tokens, palette, cloud_svg

CSS = r"""
  :root{
    --border:var(--bd); --border2:var(--bd2); --border3:var(--bd3);
    --accent-text:var(--accent-t); --accent-subtle:var(--accent-s);
    --success:var(--ok); --warning:var(--warn); --error:var(--err); --processing:var(--proc);
    --cam:var(--vcam); --marker:var(--vring);
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--app);color:var(--t1);
    font-family:Inter,-apple-system,"Segoe UI",sans-serif;font-size:13px;line-height:19px;
    -webkit-font-smoothing:antialiased;font-feature-settings:"tnum" 1}
  .app{width:1600px;height:1000px;display:grid;grid-template-rows:38px 1fr 26px;overflow:hidden;position:relative}
  /* window chrome */
  .winbar{display:flex;align-items:center;gap:12px;padding:0 14px;background:var(--panel);
    border-bottom:1px solid var(--border);-webkit-app-region:drag}
  .lights{display:flex;gap:8px;margin-right:6px}
  .lights i{width:11px;height:11px;border-radius:50%;background:#3A464E;display:block}
  .brand{display:flex;align-items:center;gap:8px;font-weight:600;letter-spacing:.01em}
  .brand svg{width:16px;height:16px}
  .winbar .where{color:var(--t3);font-size:11px;margin-left:2px}
  .winbar .right{margin-left:auto;display:flex;align-items:center;gap:8px;color:var(--t3);font-size:11px}
  .chip{display:inline-flex;align-items:center;gap:6px;height:22px;padding:0 8px;border:1px solid var(--border2);
    border-radius:3px;color:var(--t2);font-size:11px}
  .chip svg{width:12px;height:12px}

  main{display:grid;grid-template-columns:340px 1fr;min-height:0}
  /* sidebar */
  aside{background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:18px 16px;gap:14px;min-height:0}
  .sidebar-title{display:flex;align-items:center;justify-content:space-between;color:var(--t2);font-size:11px;font-weight:600;letter-spacing:.055em;text-transform:uppercase}
  .btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;height:28px;padding:0 12px;border-radius:3px;
    border:1px solid var(--border2);background:var(--raised);color:var(--t1);font:500 12px Inter;cursor:default}
  .btn svg{width:14px;height:14px}
  .btn.primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent);font-weight:600}
  .btn.block{width:100%}
  .btn.ghost{background:transparent;border-color:transparent;color:var(--t2);justify-content:flex-start;padding:0 8px;height:26px}
  .recent{display:flex;flex-direction:column;gap:2px;overflow:auto}
  .proj{display:grid;grid-template-columns:34px 1fr auto;gap:10px;align-items:center;padding:8px;border-radius:3px;border:1px solid transparent}
  .proj:hover{background:var(--raised);border-color:var(--border)}
  .thumb{width:34px;height:34px;border-radius:2px;overflow:hidden;border:1px solid var(--border2);background:var(--phbg)}
  .thumb img{display:block;width:100%;height:100%;object-fit:cover;filter:var(--thumbf)}
  .proj .nm{font-size:12px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .proj .meta{font-size:11px;color:var(--t3);font-family:"JetBrains Mono",monospace}
  .proj .when{font-size:10px;color:var(--t3)}
  .spacer{flex:1}
  .help{border-top:1px solid var(--border);padding-top:12px;display:flex;flex-direction:column;gap:8px}
  .help .row{display:flex;align-items:center;gap:8px;color:var(--t2);font-size:12px}
  .help kbd{font-family:"JetBrains Mono";font-size:10px;padding:1px 5px;border:1px solid var(--border3);border-radius:2px;color:var(--t2);background:var(--sunken)}

  /* hero */
  section{position:relative;display:grid;grid-template-rows:1fr auto;min-height:0;background:
    radial-gradient(1100px 620px at 68% 26%, var(--app-hi) 0%, var(--app-mid) 52%, var(--app) 100%)}
  .hero{position:relative;padding:44px 56px 0;display:grid;grid-template-columns:minmax(420px,1fr) 460px;gap:40px;min-height:0}
  .eyebrow{display:inline-flex;align-items:center;gap:8px;color:var(--accent-text);font-size:11px;font-weight:600;
    letter-spacing:.055em;text-transform:uppercase}
  .eyebrow .dot{width:6px;height:6px;border-radius:50%;background:var(--accent)}
  h1{font-size:34px;line-height:40px;letter-spacing:-.02em;margin:14px 0 10px;font-weight:600}
  h1 span{color:var(--t2);font-weight:400}
  .lede{color:var(--t2);font-size:13px;line-height:20px;max-width:520px;margin:0 0 22px}
  .actions{display:flex;gap:10px;align-items:center}
  .actions .btn{height:32px;font-size:12px}
  .sample{display:flex;align-items:center;gap:10px;margin-top:26px;padding:12px 14px;width:520px;
    border:1px solid var(--border);background:linear-gradient(180deg,rgba(255,255,255,.02),transparent);border-radius:4px}
  .sample .sq{width:40px;height:40px;border-radius:3px;background:url('assets/aerial_truth_ortho.png') center/cover;border:1px solid var(--border2)}
  .sample .tx{font-size:12px}
  .sample .tx b{font-weight:600}
  .sample .tx div{color:var(--t3);font-size:11px;margin-top:2px}
  .sample .btn{margin-left:auto;height:26px;font-size:11px}

  /* viewport-ish visual */
  .scene{position:relative;border:1px solid var(--border);border-radius:6px;overflow:hidden;min-height:0;
    background:linear-gradient(180deg,var(--vp2),var(--vp));align-self:stretch}
  .scene .tag{position:absolute;left:12px;top:12px;display:flex;gap:6px;align-items:center;
    background:var(--scrim);border:1px solid var(--border);border-radius:3px;padding:4px 8px;
    font-family:"JetBrains Mono";font-size:10px;color:var(--t2)}
  .scene .tag b{color:var(--vcam);font-weight:500}
  .scene img.cloud{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:.95}
  .scene .vignette{position:absolute;inset:0;background:var(--sgvig)}
  .gizmo{position:absolute;right:16px;bottom:14px;width:74px;height:74px}
  .scalebar{position:absolute;left:14px;bottom:14px;font-family:"JetBrains Mono";font-size:10px;color:var(--t2)}
  .scalebar .bar{height:4px;border-left:1px solid var(--t2);border-right:1px solid var(--t2);border-bottom:1px solid var(--t2);width:90px;margin-bottom:4px}

  /* what's new */
  .news{border-top:1px solid var(--border);background:var(--panel);display:grid;grid-template-columns:1.25fr 1fr 1fr 1fr}
  .news .col{padding:18px 20px;border-right:1px solid var(--border);min-height:198px}
  .news .col:last-child{border-right:none}
  .news h3{margin:0 0 4px;font-size:12px;font-weight:600}
  .news .sub{color:var(--t3);font-size:11px;margin-bottom:12px}
  .news .ver{display:inline-flex;align-items:center;gap:6px;font-family:"JetBrains Mono";font-size:10px;color:var(--t2)}
  .news ul{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:9px}
  .news li{display:grid;grid-template-columns:16px 1fr;gap:8px;color:var(--t2);font-size:11.5px;line-height:16px}
  .news li svg{width:14px;height:14px;margin-top:1px;color:var(--accent-text)}
  .news li b{color:var(--t1);font-weight:500}
  .news .cards{display:flex;flex-direction:column;gap:8px}
  .news .mini{display:grid;grid-template-columns:90px 1fr;gap:10px;align-items:center;padding:7px;border:1px solid var(--border);
    border-radius:3px;background:var(--raised)}
  .news .mini img{width:90px;height:56px;object-fit:cover;border-radius:2px;border:1px solid var(--border2)}
  .news .mini .tt{font-size:11.5px;font-weight:500}
  .news .mini .dd{font-size:10.5px;color:var(--t3);margin-top:2px}
  .news .link{color:var(--accent-text);font-size:11px;display:inline-flex;align-items:center;gap:5px;margin-top:10px}
  .statline{display:flex;gap:18px;margin-top:10px}
  .statline div{font-size:10.5px;color:var(--t3)}
  .statline b{display:block;font-family:"JetBrains Mono";font-size:15px;color:var(--t1);font-weight:500}

  footer{display:flex;align-items:center;gap:14px;padding:0 14px;background:var(--panel);border-top:1px solid var(--border);
    color:var(--t3);font-size:11px}
  footer .right{margin-left:auto;display:flex;gap:14px;align-items:center}
  .dotsep{width:3px;height:3px;border-radius:50%;background:var(--border3);display:inline-block;margin:0 6px;vertical-align:middle}
"""

BODY = '''
<div class="app">
  <div class="winbar">
    <div class="lights"><i></i><i></i><i></i></div>
    <div class="brand">
      <svg viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m8 3 4 8 5-5 5 15H2L8 3z"/></svg>
      alphashape
    </div>
    <div class="where">Start</div>
    <div class="right">
      <span class="chip"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3v18"/><path d="M5 8h14"/><path d="M5 16h14"/></svg>GPU: Apple M1 · 16 GB</span>
      <span>v0.4.0-dev</span>
    </div>
  </div>

  <main>
    <aside>
      <div class="sidebar-title"><span>Projects</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#A7B4BC" stroke-width="1.8"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>
      </div>
      <div style="display:flex;flex-direction:column;gap:6px">
        <div class="btn primary block"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>New project</div>
        <div class="btn block" style="background:transparent;color:var(--t2)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>Open project…</div>
      </div>
      <div class="sidebar-title" style="margin-top:6px">Recent</div>
      <div class="recent">
        <div class="proj">
          <div class="thumb"><img src="assets/aerial_truth_ortho.png" alt=""></div>
          <div><div class="nm">Rheinfeld survey — 12 photos</div><div class="meta">12 img · 4 GCP · 0.29 m/px</div></div>
          <div class="when">12 min ago</div>
        </div>
        <div class="proj">
          <div class="thumb" style="background:linear-gradient(135deg,#2B3B44,#1A2429)"></div>
          <div><div class="nm">Chapel facade (object)</div><div class="meta">264 img · masked · 3.2 M faces</div></div>
          <div class="when">Yesterday</div>
        </div>
        <div class="proj">
          <div class="thumb" style="background:linear-gradient(135deg,#3B3226,#241D15)"></div>
          <div><div class="nm">Quarry stockpile — batch</div><div class="meta">3 chunks · queued</div></div>
          <div class="when">Mon</div>
        </div>
        <div class="proj">
          <div class="thumb" style="background:linear-gradient(135deg,#26313B,#171E24)"></div>
          <div><div class="nm">OSM test — sample data</div><div class="meta">12 img · tutorial</div></div>
          <div class="when">Oct 12</div>
        </div>
      </div>
      <div class="spacer"></div>
      <div class="help">
        <div class="row"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#A7B4BC" stroke-width="1.8"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M6 8h.01M10 8h.01M6 12h.01M10 12h.01M14 8h4M14 12h4M6 16h12"/></svg><span>Shortcuts <kbd>?</kbd></span></div>
        <div class="row"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#A7B4BC" stroke-width="1.8"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg><span>Documentation</span></div>
        <div class="row"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#A7B4BC" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/></svg><span>Diagnostics</span></div>
      </div>
    </aside>

    <section>
      <div class="hero">
        <div style="padding-top:26px">
          <div class="eyebrow"><span class="dot"></span>Photogrammetry, end to end</div>
          <h1>Turn a flight into<br><span>measured ground.</span></h1>
          <p class="lede">Align photos, control the block with markers, and deliver point clouds, meshes,
            DEMs and orthomosaics — in one project, with every long job visible.</p>
          <div class="actions">
            <div class="btn primary"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>New project</div>
            <div class="btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>Open…</div>
            <div class="btn ghost" style="border:1px solid transparent;background:transparent;color:var(--t2)">Drag a folder anywhere</div>
          </div>

          <div class="sample">
            <div class="sq"></div>
            <div class="tx"><b>Sample dataset — aerial_r 3×4</b><div>12 JPEG · 480×360 · f 500 px · pre-anchored to WGS 84</div></div>
            <div class="btn">Open sample</div>
          </div>

          <div class="statline">
            <div><b>12</b>photos in last project</div>
            <div><b>0.29 m</b>estimated GSD</div>
            <div><b>4</b>GCPs · RMSE 0.021 m</div>
          </div>
        </div>

        <div class="scene">
          <div class="tag"><b>Rheinfeld survey</b> · sparse cloud · 4,945 pts</div>
          <img class="cloud" src="assets/{cloud}" alt="sparse point cloud preview">
          <div class="vignette"></div>
          <div class="scalebar"><div class="bar"></div>20 m</div>
          <svg class="gizmo" viewBox="0 0 74 74">
            <circle cx="37" cy="37" r="30" fill="var(--gfill)" stroke="var(--gstroke)"/>
            <line x1="37" y1="37" x2="37" y2="13" stroke="var(--gz)" stroke-width="1.6"/>
            <line x1="37" y1="37" x2="60" y2="46" stroke="var(--gx)" stroke-width="1.6"/>
            <line x1="37" y1="37" x2="15" y2="48" stroke="var(--gy)" stroke-width="1.6"/>
            <text x="34" y="10" fill="var(--gz)" font-family="JetBrains Mono" font-size="8">Z</text>
            <text x="62" y="50" fill="var(--gx)" font-family="JetBrains Mono" font-size="8">X</text>
            <text x="6" y="52" fill="var(--gy)" font-family="JetBrains Mono" font-size="8">Y</text>
          </svg>
        </div>
      </div>

      <div class="news">
        <div class="col">
          <div class="ver"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#4FD1C3" stroke-width="2"><path d="m12 3 1.9 5.8H20l-4.9 3.6 1.9 5.8L12 14.6 7 18.2l1.9-5.8L4 8.8h6.1z"/></svg>What's new in 0.4</div>
          <h3 style="margin-top:8px">Non-blocking everything</h3>
          <div class="sub">Every stage now runs in the queue with estimates and retry.</div>
          <ul>
            <li><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6 9 17l-5-5"/></svg><span><b>Queue</b> — pause, resume, retry with the original parameters.</span></li>
            <li><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6 9 17l-5-5"/></svg><span><b>Estimates</b> — time, RAM and disk before you commit.</span></li>
            <li><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6 9 17l-5-5"/></svg><span><b>Resume</b> — interrupted jobs restart at the last stage.</span></li>
          </ul>
        </div>
        <div class="col">
          <h3>Guides</h3><div class="sub">Short, task-shaped</div>
          <div class="cards">
            <div class="mini"><img src="assets/aerial_r1c1.jpg" alt=""><div><div class="tt">First aerial project</div><div class="dd">12 photos → ortho · 6 min read</div></div></div>
            <div class="mini"><img src="assets/aerial_truth_ortho.png" alt=""><div><div class="tt">Ground control with markers</div><div class="dd">Projection, residuals, optimize</div></div></div>
          </div>
        </div>
        <div class="col">
          <h3>Recent exports</h3><div class="sub">From all projects</div>
          <div class="cards">
            <div class="mini" style="grid-template-columns:1fr"><div><div class="tt" style="font-family:'JetBrains Mono';font-size:11px">rheinfeld_ortho.tif</div><div class="dd">6,144×4,608 · EPSG:25832 · 184 MB</div></div></div>
            <div class="mini" style="grid-template-columns:1fr"><div><div class="tt" style="font-family:'JetBrains Mono';font-size:11px">rheinfeld_cloud.laz</div><div class="dd">4.2 M pts · classified · 71 MB</div></div></div>
          </div>
          <div class="link">Open export folder <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 17 17 7M7 7h10v10"/></svg></div>
        </div>
        <div class="col">
          <h3>System</h3><div class="sub">Ready to process</div>
          <div class="statline" style="flex-direction:column;gap:10px">
            <div><b style="color:var(--ok)">GPU ready</b>Apple M1 · Metal · 8 cores</div>
            <div><b>74.2 GB</b>free on Projects (SSD)</div>
            <div><b>0</b>jobs queued</div>
          </div>
          <div class="link" style="margin-top:14px">Preferences <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 17 17 7M7 7h10v10"/></svg></div>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <span>Ready</span><span class="dotsep"></span>
    <span>Log: <span style="font-family:'JetBrains Mono'">~/.alphashape/logs/2026-10-21.log</span></span>
    <div class="right"><span>No network calls</span><span class="dotsep"></span><span>ISC-licensed icons · OFL fonts</span></div>
  </footer>
</div>
'''


def render(theme, accent, vp, name):
    tokens = theme_tokens(theme, accent, vp)
    html = page(f"alphashape — Start ({theme}, {accent})", BODY.replace("{cloud}", cloud_svg(accent, vp)),
                CSS, tokens=tokens, body_class="light" if theme == "light" else "")
    (OUT / f"{name}.html").write_text(html)
    (OUT / f"{name}.palette.json").write_text(json.dumps(palette(theme, accent, vp), indent=1))
    print("written", name)


if __name__ == "__main__":
    render("dark", "ember", "dark", "r3-start-dark-ember")
