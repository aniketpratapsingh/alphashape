import sys, json
sys.path.insert(0, "/Volumes/Extreme SSD/Projects/alphashape/design/mockups")
from _common import icon, page, OUT, LIGHT_CSS
from _tokens import theme_tokens, palette


def job_row(state, title, sub, pct, meta, actions, cls=""):
    dot = {"run": "run", "queued": "idle", "done": "ok", "fail": "err", "paused": "warn"}[state]
    state_label = {"run": "Running", "queued": "Queued", "done": "Done", "fail": "Failed", "paused": "Paused"}[state]
    color = {"run": "proc", "queued": "", "done": "ok", "fail": "err", "paused": "warn"}[state]
    bar = f'<div class="progress {color}"><i style="width:{pct}%"></i></div>' if state in ("run", "done") else ""
    return f'''
    <div class="job {cls}">
      <div class="jstate"><span class="dot {dot}"></span><span class="js-t {state}">{state_label}</span></div>
      <div class="jmain">
        <div class="jt">{title}</div>
        <div class="js">{sub}</div>
      </div>
      <div class="jprog">{bar}<div class="jp-meta">{meta}</div></div>
      <div class="jact">{actions}</div>
    </div>'''


def build(theme="dark", accent="ember", vp="dark", name="r3-queue-dark-ember"):
    rows = ""
    rows += job_row(
        "run", "Build point cloud", "Chunk 1 · <span class='mut'>Depth maps · 7 / 12 cameras</span>", 58,
        "<span class='mono'>58%</span> · ETA 00:02:40 · RAM 3.1 / 12 GB",
        f'<span class="iconbtn" title="Pause">{icon("pause",14)}</span><span class="iconbtn" title="Stop">{icon("square",14)}</span><span class="iconbtn" title="Show log">{icon("terminal",14)}</span>',
        "runrow")
    rows += job_row(
        "queued", "Build DEM", "Chunk 1 · <span class='mut'>after point cloud</span>", 0,
        "<span class='mut'>Queued 00:04 ago</span> · est. 00:01:30 · 640 MB",
        f'<span class="iconbtn" title="Move up">{icon("chevron-up",14)}</span><span class="iconbtn" title="Remove">{icon("trash",14)}</span><span class="iconbtn" title="Edit…">{icon("more",14)}</span>')
    rows += job_row(
        "queued", "Build orthomosaic", "Chunk 1 · <span class='mut'>after DEM</span>", 0,
        "<span class='mut'>Queued 00:04 ago</span> · est. 00:04:10 · 3.8 GB",
        f'<span class="iconbtn" title="Move up">{icon("chevron-up",14)}</span><span class="iconbtn" title="Remove">{icon("trash",14)}</span><span class="iconbtn">{icon("more",14)}</span>')
    rows += job_row(
        "fail", "Align photos", "Chunk 2 · <span class='mut'>3 cameras could not be matched</span>", 34,
        "<span style='color:var(--err)'>Failed after 00:00:52</span> · overlap below 40%",
        f'<span class="iconbtn" title="Retry">{icon("refresh",14)}</span><span class="iconbtn" title="Retry with changes">{icon("settings",14)}</span><span class="iconbtn" title="Show log">{icon("terminal",14)}</span>',
        "failrow")
    rows += job_row(
        "done", "Align photos", "Chunk 1 · <span class='mut'>12 / 12 cameras · median 0.42 px</span>", 100,
        "<span class='mono'>00:00:41</span> · finished 14:02 · 1.4 GB peak",
        f'<span class="iconbtn" title="Show log">{icon("terminal",14)}</span><span class="iconbtn" title="Reveal output">{icon("folder-search",14)}</span><span class="iconbtn">{icon("more",14)}</span>',
        "donerow")
    rows += job_row(
        "done", "Import markers", "Chunk 1 · <span class='mut'>4 GCPs from gcp_rheinfeld.csv</span>", 100,
        "<span class='mono'>00:00:01</span> · finished 14:01",
        f'<span class="iconbtn" title="Show log">{icon("terminal",14)}</span><span class="iconbtn">{icon("more",14)}</span>', "donerow")

    body = f'''
<div class="app">
  <div class="winbar">
    <div class="lights"><i></i><i></i><i></i></div>
    <div class="brand"><span style="color:var(--accent-t)">{icon("mountain",16,1.7)}</span>alphashape</div>
    <div class="mut" style="font-size:11px">Rheinfeld survey.alphashape</div>
    <div class="right"><span class="chip">{icon("cpu",11)}GPU · M1</span><span class="chip">{icon("list",11)}4 active · 2 finished</span><span class="iconbtn">{icon("command",15)}</span></div>
  </div>

  <div class="work">
    <div class="rail">
      <div class="railitem done"><span class="rn">{icon("folder-input",17,1.7)}</span><span class="rlabel">Photos</span></div>
      <div class="railitem done"><span class="rn">{icon("scan",17,1.7)}</span><span class="rlabel">Align</span></div>
      <div class="railitem done"><span class="rn">{icon("crosshair",17,1.7)}</span><span class="rlabel">Control</span></div>
      <div class="railitem active"><span class="rn">{icon("cloud",17,1.7)}</span><span class="rlabel">Points</span></div>
      <div class="railitem disabled"><span class="rn">{icon("box",17,1.7)}</span><span class="rlabel">Surface</span></div>
      <div class="railitem disabled"><span class="rn">{icon("package",17,1.7)}</span><span class="rlabel">Export</span></div>
      <div style="flex:1"></div>
      <div class="railitem" style="height:34px">{icon("panels",16)}</div>
    </div>

    <div class="dock ws">
      <div class="dockhead"><span class="ttl">Workspace</span><div class="right"><span class="iconbtn">{icon("plus",14)}</span><span class="iconbtn">{icon("filter",14)}</span></div></div>
      <div class="tree">
        <div class="row root"><span class="tw">{icon("chevron-down",12)}</span>{icon("layers",14,1.7,"ico")}<span class="nm">Chunk 1</span><span class="badge">12 aligned</span><span class="sp"></span><span class="ic">{icon("eye",13,1.7)}</span></div>
        <div class="row l1"><span class="tw">{icon("chevron-right",12)}</span>{icon("camera",14,1.7,"ico")}<span class="nm">Cameras</span><span class="cnt">12</span><span class="sp"></span><span class="ic">{icon("eye",13,1.7)}</span></div>
        <div class="row l1"><span class="tw">{icon("chevron-right",12)}</span>{icon("crosshair",14,1.7,"ico")}<span class="nm">Markers</span><span class="cnt">4</span><span class="sp"></span><span class="ic">{icon("eye",13,1.7)}</span></div>
        <div class="row l1"><span class="tw"></span>{icon("scan",14,1.7,"ico")}<span class="nm">Tie points</span><span class="cnt">4,945</span><span class="sp"></span><span class="ic">{icon("eye",13,1.7)}</span></div>
        <div class="row l1 run"><span class="tw"></span>{icon("cloud",14,1.7,"ico")}<span class="nm">Point cloud</span><span class="mono tiny" style="color:var(--proc)">building</span><span class="sp"></span><span class="ic"></span></div>
        <div class="row l1 dimrow"><span class="tw"></span>{icon("box",14,1.7,"ico")}<span class="nm">Model</span></div>
        <div class="row root" style="margin-top:8px;height:22px;font-weight:500;color:var(--t2)"><span class="tw">{icon("chevron-right",12)}</span>{icon("layers",14,1.7,"ico")}<span class="nm">Chunk 2</span><span class="badge err">3 failed</span></div>
        <div class="row root" style="height:22px;font-weight:500;color:var(--t2)"><span class="tw">{icon("chevron-right",12)}</span>{icon("layers",14,1.7,"ico")}<span class="nm">Chunk 3</span><span class="badge">queued</span></div>
      </div>
      <div class="wsfoot"><div class="tiny mut">3 chunks · 2 processing</div></div>
    </div>

    <div class="center">
      <div class="qtop">
        <div class="qtitle">
          <span class="qt-i">{icon("list",16,1.9)}</span>
          <span>Processing queue</span>
          <span class="chip acc">1 running</span>
          <span class="chip">3 waiting</span>
          <span class="chip ok">2 done</span>
          <span class="chip err">1 failed</span>
        </div>
        <div class="qactions">
          <span class="btn sm">{icon("pause",12)}Pause all</span>
          <span class="btn sm ghost">{icon("trash",12)}Clear finished</span>
          <span class="vsep"></span>
          <span class="tiny mut">Concurrency</span>
          <span class="field" style="width:64px;height:24px">1 job <span style="margin-left:auto">{icon("chevron-down",11)}</span></span>
        </div>
      </div>

      <div class="jobs scroll">
        {rows}
      </div>

      <div class="qlog">
        <div class="qlog-h">
          <span class="tab on">{icon("terminal",13)}Log · Build point cloud</span>
          <span class="tab">{icon("file",13)}Log file</span>
          <span class="sp"></span>
          <span class="tiny mut">All levels {icon("chevron-down",10)}</span>
          <span class="iconbtn">{icon("search",13)}</span>
          <span class="iconbtn">{icon("download",13)}</span>
        </div>
        <div class="log scroll">
          <div class="ln"><span class="lt">14:06:02</span><span class="lv info">info</span><span>Build point cloud started · Chunk 1 · 12 cameras · Balanced preset</span></div>
          <div class="ln"><span class="lt">14:06:02</span><span class="lv info">info</span><span>Depth maps: reusing none · 12 to compute · 480×360 · f=500.00</span></div>
          <div class="ln"><span class="lt">14:06:11</span><span class="lv info">info</span><span>depth aerial_r0c0.jpg 100% (0.8 s)</span></div>
          <div class="ln"><span class="lt">14:06:12</span><span class="lv info">info</span><span>depth aerial_r0c1.jpg 100% (0.9 s)</span></div>
          <div class="ln"><span class="lt">14:06:14</span><span class="lv warn">warn</span><span>aerial_r2c2.jpg: overlap 38% below filter threshold 40% · depth will be sparse in 14% of pixels</span></div>
          <div class="ln"><span class="lt">14:06:15</span><span class="lv info">info</span><span>depth aerial_r1c0.jpg 100% (1.0 s)</span></div>
          <div class="ln cur"><span class="lt">14:06:16</span><span class="lv proc">run</span><span>depth aerial_r1c1.jpg 62% …<i class="cursor"></i></span></div>
          <div class="ln dim"><span class="lt">—</span><span class="lv">·</span><span class="mut">Triangulating points · filtering · colouring</span></div>
        </div>
      </div>
    </div>

    <div class="dock side">
      <div class="tabs"><span class="tab">Properties</span><span class="tab">Reference</span><span class="tab on">Queue <span class="cnt">6</span></span></div>
      <div class="sidebody">
        <div class="label" style="margin:10px 10px 6px">Selected job</div>
        <div class="kv"><span class="k">Stage</span><span>Point cloud</span></div>
        <div class="kv"><span class="k">Preset</span><span>Balanced</span></div>
        <div class="kv"><span class="k">Started</span><span class="mono">14:06:02</span></div>
        <div class="kv"><span class="k">Elapsed</span><span class="mono">00:04:12</span></div>
        <div class="kv"><span class="k">ETA</span><span class="mono">00:02:40</span></div>
        <div class="kv"><span class="k">Peak RAM</span><span class="mono">3.1 GB</span></div>
        <div class="kv"><span class="k">Disk</span><span class="mono">1.42 GB</span></div>
        <div class="divider" style="margin:10px"></div>
        <div class="label" style="margin:0 10px 6px">On failure</div>
        <div class="kv"><span class="k">Policy</span><span>Pause queue</span></div>
        <div class="kv"><span class="k">Retries</span><span>Manual</span></div>
        <div class="fmtnote tiny mut">
          Point clouds are written to <span class="mono">chunk1/point_cloud/</span> and appear in the Workspace as soon as the job finishes.
        </div>
      </div>
    </div>
  </div>

  <div class="statusbar">
    <span class="sb"><span class="dot run"></span>1 running</span><span class="sbd"></span>
    <span class="sb mono">Queue 3</span><span class="sbd"></span>
    <span class="sb mono">2 done</span><span class="sbd"></span>
    <span class="sb" style="color:var(--err)">1 failed</span>
    <span class="sp"></span>
    <span class="sb mono">CPU 312% · GPU 41% · RAM 3.1 / 16 GB</span>
  </div>
</div>'''

    CSS = r"""
.work{display:grid;grid-template-columns:44px 240px 1fr 300px;min-height:0}
.rail{background:var(--app);border-right:1px solid var(--bd);display:flex;flex-direction:column;align-items:center;padding:8px 0;gap:4px}
.railitem{width:40px;height:46px;border-radius:5px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;color:var(--t3);border:1px solid transparent;position:relative}
.railitem .rlabel{font-size:9px}
.railitem.done{color:var(--t2)} .railitem.done:after{content:"";position:absolute;left:6px;top:7px;width:5px;height:5px;border-radius:50%;background:var(--ok)}
.railitem.active{background:var(--accent-s);border-color:var(--accent);color:var(--accent-t)}
.railitem.active:after{content:"";position:absolute;left:6px;top:7px;width:5px;height:5px;border-radius:50%;background:var(--accent)}
.railitem.disabled{opacity:.42}
.ws{border:none;border-right:1px solid var(--bd)}
.tree{flex:1;overflow:auto;font-size:12px}
.row{display:flex;align-items:center;gap:6px;height:22px;padding:0 8px 0 4px;border-left:2px solid transparent;white-space:nowrap}
.row .tw{width:14px;display:flex;justify-content:center;color:var(--t3)}
.row .ico{color:var(--t2)} .row .nm{overflow:hidden;text-overflow:ellipsis}
.row .sp{flex:1} .row .cnt{color:var(--t3);font-size:10.5px} .row .ic{color:var(--t3)}
.row .badge{margin-left:auto;font-size:9.5px;padding:0 5px;line-height:14px;border-radius:8px;border:1px solid var(--bd2);color:var(--t2);background:var(--raised)}
.row .badge.err{color:var(--err);border-color:var(--err-bd-soft);background:var(--err-bg-soft)}
.row.run{background:linear-gradient(90deg,var(--proc-fade),transparent 80%);border-left-color:var(--proc)}
.row.l1{padding-left:18px}.row.dimrow{color:var(--t3)}
.wsfoot{border-top:1px solid var(--bd);padding:7px 9px}
.center{display:grid;grid-template-rows:46px 1fr 236px;min-width:0;min-height:0;border-right:1px solid var(--bd)}
.qtop{display:flex;align-items:center;gap:10px;padding:0 14px;border-bottom:1px solid var(--bd);background:var(--panel)}
.qtitle{display:flex;align-items:center;gap:8px;font-size:14px;font-weight:600}
.qt-i{color:var(--accent-t)}
.qactions{margin-left:auto;display:flex;align-items:center;gap:6px}
.vsep{width:1px;height:16px;background:var(--bd2);margin:0 4px}
.jobs{overflow:auto;padding:8px 10px;display:flex;flex-direction:column;gap:6px;background:var(--app)}
.job{display:grid;grid-template-columns:104px 1fr 210px 96px;gap:12px;align-items:center;
  background:var(--panel);border:1px solid var(--bd);border-radius:4px;padding:9px 11px;min-height:58px}
.job.runrow{border-color:var(--proc-bd);background:linear-gradient(90deg,var(--proc-fade-sm),var(--panel) 45%)}
.job.failrow{border-color:var(--err-bd-soft)}
.jstate{display:flex;align-items:center;gap:7px}
.js-t{font-size:10px;text-transform:uppercase;letter-spacing:.055em;font-weight:600;color:var(--t3)}
.js-t.run{color:var(--proc)} .js-t.done{color:var(--ok)} .js-t.fail{color:var(--err)} .js-t.paused{color:var(--warn)}
.jt{font-size:12.5px;font-weight:600}
.js{font-size:11px;color:var(--t2);margin-top:2px}
.jp-meta{font-size:10.5px;color:var(--t3);margin-top:5px;font-variant-numeric:tabular-nums}
.jact{display:flex;justify-content:flex-end;gap:2px}
.qlog{border-top:1px solid var(--bd);background:var(--panel);display:flex;flex-direction:column;min-height:0}
.qlog-h{display:flex;align-items:center;gap:4px;border-bottom:1px solid var(--bd);height:32px}
.qlog-h .sp{flex:1}
.log{flex:1;overflow:auto;padding:6px 0;font-family:"JetBrains Mono",monospace;font-size:10.5px;line-height:17px}
.ln{display:grid;grid-template-columns:64px 40px 1fr;gap:8px;padding:0 12px;white-space:nowrap}
.ln:hover{background:var(--hover)}
.lt{color:var(--t3)} .lv{font-size:9.5px;text-transform:uppercase}
.lv.info{color:var(--info)} .lv.warn{color:var(--warn)} .lv.proc{color:var(--proc)}
.ln.dim{opacity:.6}
.ln.cur{background:var(--proc-fade-sm)}
.cursor{display:inline-block;width:6px;height:11px;background:var(--proc);vertical-align:-1px;margin-left:3px}
.side{border:none;border-left:1px solid var(--bd)}
.sidebody{flex:1;overflow:auto}
.kv{display:grid;grid-template-columns:84px 1fr;gap:8px;padding:4px 10px;font-size:11.5px}
.kv .k{color:var(--t3)}
.fmtnote{padding:10px;line-height:16px}
.statusbar{display:flex;align-items:center;height:26px;padding:0 10px;background:var(--panel);border-top:1px solid var(--bd);font-size:10.5px;color:var(--t2)}
.sb{display:inline-flex;align-items:center;gap:5px;padding:0 10px}
.sbd{width:1px;height:12px;background:var(--bd2)} .sp{flex:1}
"""
    tokens = theme_tokens(theme, accent, vp)
    html = page(f"alphashape — processing queue ({theme}, {accent})", body, CSS + LIGHT_CSS,
                tokens=tokens, body_class="light" if theme == "light" else "")
    (OUT / f"{name}.html").write_text(html)
    (OUT / f"{name}.palette.json").write_text(json.dumps(palette(theme, accent, vp), indent=1))
    print("written", name)


build()
