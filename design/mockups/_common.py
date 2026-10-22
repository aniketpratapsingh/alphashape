import json, pathlib

OUT = pathlib.Path("/Volumes/Extreme SSD/Projects/alphashape/design/mockups")

ICONS = {
 "mountain": '<path d="m8 3 4 8 5-5 5 15H2L8 3z"/>',
 "plus": '<path d="M12 5v14M5 12h14"/>',
 "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
 "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/>',
 "images": '<path d="M18 22H4a2 2 0 0 1-2-2V6"/><path d="m22 13-1.3-1.3a2 2 0 0 0-2.8 0L11 18"/><circle cx="12" cy="8" r="2"/><rect x="6" y="2" width="16" height="16" rx="2"/>',
 "camera": '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3z"/><circle cx="12" cy="13" r="3"/>',
 "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/>',
 "cloud": '<path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>',
 "grid": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>',
 "mountain2": '<path d="m8 3 4 8 5-5 5 15H2L8 3z"/><path d="M4.14 15.08c2.62-1.57 5.24-1.43 7.86.42 2.74 1.94 5.49 2 8.23.19"/>',
 "map": '<path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3z"/><path d="M9 3v15M15 6v15"/>',
 "marker": '<path d="M12 22s8-6 8-12a8 8 0 1 0-16 0c0 6 8 12 8 12z"/><circle cx="12" cy="10" r="2.5"/>',
 "eye": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>',
 "eyeoff": '<path d="M9.9 4.24A9.1 9.1 0 0 1 12 4c6.5 0 10 7 10 7a13.2 13.2 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.5 13.5 0 0 0 2 11s3.5 7 10 7a9 9 0 0 0 5.39-1.61"/><path d="m2 2 20 20"/><path d="M14.12 14.12a3 3 0 1 1-4.24-4.24"/>',
 "chevron-right": '<path d="m9 18 6-6-6-6"/>',
 "chevron-down": '<path d="m6 9 6 6 6-6"/>',
 "chevron-up": '<path d="m18 15-6-6-6 6"/>',
 "search": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
 "filter": '<path d="M3 5h18l-7 8v6l-4 2v-8z"/>',
 "play": '<path d="m6 4 14 8-14 8z"/>',
 "pause": '<rect x="7" y="5" width="4" height="14" rx="1"/><rect x="13" y="5" width="4" height="14" rx="1"/>',
 "square": '<rect x="5" y="5" width="14" height="14" rx="2"/>',
 "rotate": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
 "terminal": '<path d="m4 17 6-5-6-5"/><path d="M12 19h8"/>',
 "list": '<path d="M8 6h13M8 12h13M8 18h13"/><path d="M3 6h.01M3 12h.01M3 18h.01"/>',
 "check": '<path d="M20 6 9 17l-5-5"/>',
 "checkcircle": '<circle cx="12" cy="12" r="9"/><path d="m8.5 12 2.5 2.5L16 9"/>',
 "alert": '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/>',
 "xcircle": '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6M15 9l-6 6"/>',
 "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
 "timer": '<path d="M10 2h4"/><path d="M12 14v-4"/><circle cx="12" cy="14" r="8"/>',
 "disk": '<path d="M4 6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z"/><path d="M4 10h16"/><path d="M8 16h.01M12 16h.01"/>',
 "memory": '<path d="M6 4v16M18 4v16"/><rect x="6" y="7" width="12" height="10" rx="1"/><path d="M9 7V4M15 7V4M9 20v-3M15 20v-3"/>',
 "orbit": '<circle cx="12" cy="12" r="3"/><path d="M3.5 9.5c-1.5 3-.5 6 2 7.5s6.5 1 9-1.5"/><path d="M20.5 14.5c1.5-3 .5-6-2-7.5s-6.5-1-9 1.5"/>',
 "hand": '<path d="M18 11V6a2 2 0 0 0-4 0v4"/><path d="M14 10V5a2 2 0 0 0-4 0v5"/><path d="M10 10V6a2 2 0 0 0-4 0v8a6 6 0 0 0 12 0v-3a2 2 0 0 0-4 0"/>',
 "boxselect": '<path d="M5 3a2 2 0 0 0-2 2"/><path d="M19 3a2 2 0 0 1 2 2"/><path d="M21 19a2 2 0 0 1-2 2"/><path d="M5 21a2 2 0 0 1-2-2"/><path d="M9 3h1M9 21h1M3 9v1M21 9v1"/>',
 "lasso": '<path d="M7 22a5 5 0 0 1-2-4"/><path d="M3.3 14A6.8 6.8 0 0 1 2 10c0-4.4 4.5-8 10-8s10 3.6 10 8-4.5 8-10 8a12 12 0 0 1-5-1"/><path d="M5 18a2 2 0 1 0 0-4 2 2 0 0 0 0 4z"/>',
 "move": '<path d="M12 2v20M2 12h20"/><path d="m8 6 4-4 4 4M8 18l4 4 4-4M6 8l-4 4 4 4M18 8l4 4-4 4"/>',
 "scale": '<path d="M21 3 9 15"/><path d="M12 3H3v18h18v-9"/><path d="M14 14h6M14 10h6M10 14v6M14 18h6"/>',
 "axis": '<path d="M3 3v18h18"/><path d="m19 9-4 4-4-4"/><path d="m3 15 4-4 4 4"/>',
 "grid3d": '<path d="M12 3 3 8v8l9 5 9-5V8z"/><path d="M3 8l9 5 9-5M12 13v8"/>',
 "fit": '<path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M16 21h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/>',
 "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
 "moon": '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
 "more": '<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
 "info": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>',
 "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
 "folder-search": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><circle cx="14.5" cy="13.5" r="2.5"/><path d="m16.3 15.3 1.7 1.7"/>',
 "folder-input": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 11v5M9.5 13.5 12 11l2.5 2.5"/>',
 "hard-drive": '<path d="M4 6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z"/><path d="M4 10h16"/><path d="M8 16h.01M12 16h.01"/>',
 "grip": '<circle cx="9" cy="6" r="1"/><circle cx="15" cy="6" r="1"/><circle cx="9" cy="12" r="1"/><circle cx="15" cy="12" r="1"/><circle cx="9" cy="18" r="1"/><circle cx="15" cy="18" r="1"/>',
 "command": '<path d="M18 3a3 3 0 0 0-3 3v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 0 0 0-6z"/>',
 "layers": '<path d="m12 2 9 5-9 5-9-5z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
 "ruler": '<path d="m3 15 6-12 12 6-6 12z"/><path d="m7.5 10.5 2 1M9.5 6.5l2 1M11.5 2.5l2 1M5.5 14.5l2 1"/>',
 "shapes": '<path d="m8 3 4 8H4z"/><rect x="13" y="13" width="8" height="8" rx="1"/><circle cx="17" cy="7" r="4"/>',
 "box": '<path d="M21 8v8a2 2 0 0 1-1 1.7l-7 4a2 2 0 0 1-2 0l-7-4A2 2 0 0 1 3 16V8a2 2 0 0 1 1-1.7l7-4a2 2 0 0 1 2 0l7 4A2 2 0 0 1 21 8z"/><path d="m3.3 7 8.7 5 8.7-5M12 22V12"/>',
 "package": '<path d="M16.5 9.4 7.5 4.2"/><path d="M21 16V8a2 2 0 0 0-1-1.7l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.7l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="m3.3 7 8.7 5 8.7-5M12 22V12"/>',
 "badge": '<path d="M12 2 15 5h4v4l3 3-3 3v4h-4l-3 3-3-3H5v-4l-3-3 3-3V5h4z"/><path d="m9 12 2 2 4-4"/>',
 "sparkles": '<path d="m12 3 1.9 5.8H20l-4.9 3.6 1.9 5.8L12 14.6 7 18.2l1.9-5.8L4 8.8h6.1z"/>',
 "history": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>',
 "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9"/>',
 "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
 "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.9.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
 "zoom": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3M8 11h6"/>',
 "panels": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/>',
 "crosshair": '<circle cx="12" cy="12" r="8"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>',
 "link": '<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/>',
 "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5M12 3v12"/>',
 "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5M12 15V3"/>',
 "external": '<path d="M7 17 17 7"/><path d="M7 7h10v10"/>',
 "wand": '<path d="m15 4 5 5"/><path d="M3 21 17 7l-5-5L-2 16z" transform="translate(5 0)"/><path d="M9 3h.01M4 9h.01M15 21h.01M20 15h.01"/>',
 "scissors": '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M20 4 8.1 15.9M14.5 14.5 20 20M8.1 8.1 12 12"/>',
 "brush": '<path d="m9.1 15.9 8.4-8.4a2.1 2.1 0 0 0-3-3l-8.4 8.4"/><path d="M6 18c-1.5 0-3 1.5-3 3 2 0 3-1.5 3-3z"/><path d="m14 14 4 4"/>',
 "contrast": '<circle cx="12" cy="12" r="9"/><path d="M12 3v18"/><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" stroke="none"/>',
 "eraser": '<path d="m7 21-4-4a2 2 0 0 1 0-3L14 3a2 2 0 0 1 3 0l5 5a2 2 0 0 1 0 3L11 21z"/><path d="M18 13 9 4"/>',
 "keyboard": '<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M6 9h.01M10 9h.01M14 9h.01M18 9h.01M6 13h.01M10 13h.01M14 13h.01M18 13h.01M8 16h8"/>',
 "book": '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>',
 "mouse": '<rect x="6" y="3" width="12" height="18" rx="6"/><path d="M12 7v4"/>',
 "lightbulb": '<path d="M9 18h6M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.3h6c0-1 .4-1.8 1-2.3A7 7 0 0 0 12 2z"/>',
 "arrow-right": '<path d="M5 12h14M13 6l6 6-6 6"/>',
 "arrow-up-right": '<path d="M7 17 17 7M7 7h10v10"/>',
 "refresh": '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
 "trash": '<path d="M4 7h16M10 11v6M14 11v6"/><path d="M6 7l1 14h10l1-14"/><path d="M9 7V4h6v3"/>',
 "star": '<path d="m12 3 2.6 6.6 7.1.5-5.5 4.6 1.8 6.9L12 17.7 6 21.6l1.8-6.9L2.3 10l7.1-.5z"/>',
 "clipboard": '<rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M9 12h6M9 16h4"/>',
 "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a15 15 0 0 1 0 18 15 15 0 0 1 0-18z"/>',
 "layers2": '<path d="m12 2 9 5-9 5-9-5z"/><path d="m3 12 9 5 9-5"/>',
}

def icon(name, size=16, sw=1.8, cls=""):
    body = ICONS.get(name, ICONS["info"])
    return (f'<svg class="{cls}" viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
            f'stroke="currentColor" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')

BASE_CSS = r"""
  /* Colour custom properties are emitted by page() from _tokens.theme_tokens();
     BASE_CSS only carries structure and the fallback for non-tokenised values. */
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--app);color:var(--t1);
    font-family:Inter,-apple-system,"Segoe UI",sans-serif;font-size:13px;line-height:19px;
    -webkit-font-smoothing:antialiased;font-feature-settings:"tnum" 1;letter-spacing:0}
  .app{width:1600px;height:1000px;display:grid;grid-template-rows:38px 1fr;overflow:hidden;background:var(--app)}
  .mono{font-family:"JetBrains Mono",monospace}
  .sec{color:var(--t2)} .mut{color:var(--t3)}
  .k{color:var(--t3)}
  svg{flex:none}
  .winbar{display:flex;align-items:center;gap:12px;padding:0 14px;background:var(--panel);
    border-bottom:1px solid var(--bd)}
  .lights{display:flex;gap:8px;margin-right:6px}
  .lights i{width:11px;height:11px;border-radius:50%;background:var(--lights,#3A464E);display:block}
  .brand{display:flex;align-items:center;gap:8px;font-weight:600}
  .winbar .right{margin-left:auto;display:flex;align-items:center;gap:10px;color:var(--t3);font-size:11px}
  .iconbtn{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:3px;
    border:1px solid transparent;color:var(--t2);background:transparent}
  .iconbtn:hover{background:var(--hover);color:var(--t1)}
  .iconbtn.on{background:var(--accent-s);border-color:var(--accent);color:var(--accent-t)}
  .btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;height:26px;padding:0 10px;border-radius:3px;
    border:1px solid var(--bd2);background:var(--raised);color:var(--t1);font:500 12px Inter;white-space:nowrap}
  .btn.primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent);font-weight:600}
  .btn.ghost{background:transparent;border-color:transparent;color:var(--t2)}
  .btn.sm{height:22px;font-size:11px;padding:0 8px}
  .btn.danger{background:transparent;border-color:var(--err);color:var(--err)}
  .chip{display:inline-flex;align-items:center;gap:5px;height:20px;padding:0 7px;border-radius:3px;
    background:var(--raised);border:1px solid var(--bd);color:var(--t2);font-size:10.5px}
  .chip.ok{color:var(--ok);border-color:var(--okbd);background:var(--okbg)}
  .chip.warn{color:var(--warn);border-color:var(--warnbd);background:var(--warnbg)}
  .chip.err{color:var(--err);border-color:var(--errbd);background:var(--errbg)}
  .chip.acc{color:var(--accent-t);border-color:var(--accent-chipbd);background:var(--accent-s)}
  .dock{background:var(--panel);border:1px solid var(--bd);display:flex;flex-direction:column;min-height:0}
  .dockhead{display:flex;align-items:center;gap:6px;height:30px;padding:0 8px;border-bottom:1px solid var(--bd);background:var(--panel)}
  .dockhead .ttl{font-size:10.5px;font-weight:600;letter-spacing:.055em;text-transform:uppercase;color:var(--t2)}
  .dockhead .right{margin-left:auto;display:flex;gap:2px}
  .tabs{display:flex;height:30px;border-bottom:1px solid var(--bd);background:var(--app);align-items:stretch}
  .tab{display:inline-flex;align-items:center;gap:6px;padding:0 12px;font-size:12px;color:var(--t2);
    border-bottom:2px solid transparent;white-space:nowrap}
  .tab.on{color:var(--t1);border-bottom-color:var(--accent);background:var(--panel)}
  .tab .cnt{font-size:10px;color:var(--t3);background:var(--raised);border:1px solid var(--bd);border-radius:8px;padding:0 5px;line-height:14px}
  .tab.on .cnt{color:var(--t2)}
  .field{height:24px;background:var(--sunken);border:1px solid var(--bd2);border-radius:3px;display:flex;align-items:center;
    gap:6px;padding:0 7px;color:var(--t3);font-size:11px}
  .field input{border:none;background:transparent;color:var(--t1);font:12px Inter;outline:none;width:100%}
  .toggle{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:2px;
    border:1px solid var(--bd3);background:var(--sunken);color:var(--on-accent)}
  .toggle.on{background:var(--accent);border-color:var(--accent)}
  .toggle.on svg{display:block} .toggle svg{display:none}
  .kbd{font-family:"JetBrains Mono";font-size:10px;padding:1px 4px;border:1px solid var(--bd3);border-radius:2px;color:var(--t2);background:var(--sunken)}
  .divider{height:1px;background:var(--bd)}
  .progress{height:4px;border-radius:2px;background:var(--track);overflow:hidden;position:relative}
  .progress > i{display:block;height:100%;background:var(--accent);border-radius:2px}
  .progress.proc > i{background:var(--proc)}
  .progress.err > i{background:var(--err)}
  .scroll::-webkit-scrollbar{width:10px;height:10px}
  .scroll::-webkit-scrollbar-thumb{background:var(--scroll-thumb);border-radius:4px;border:2px solid var(--panel)}
  .scroll::-webkit-scrollbar-track{background:transparent}
  .tiny{font-size:10.5px;line-height:15px}
  .micro{font-size:10px;line-height:14px}
  .label{font-size:10.5px;font-weight:600;letter-spacing:.055em;text-transform:uppercase;color:var(--t2)}
  .num{font-variant-numeric:tabular-nums}
  .dot{width:6px;height:6px;border-radius:50%;display:inline-block;flex:none}
  .dot.ok{background:var(--ok)} .dot.err{background:var(--err)} .dot.warn{background:var(--warn)}
  .dot.run{background:var(--proc);box-shadow:0 0 0 3px rgba(167,139,250,.15)}
  .dot.idle{background:var(--bd3)}
  .callout{background:var(--overlay);border:1px solid var(--bd2);border-radius:4px}
"""

def page(title, body, extra_css="", tokens=None, body_class=""):
    vars_css = css_vars(tokens or {})
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{vars_css}{BASE_CSS}{extra_css}</style></head>
<body class="{body_class}">{body}</body></html>"""

# Light-theme adjustments that base component rules still express with dark
# literal values (badge tints etc.).  Colours themselves come from tokens.
LIGHT_CSS = r"""
.row .badge.g{color:var(--ok);border-color:var(--okbd);background:var(--okbg)}
.row .badge.err{color:var(--err);border-color:var(--errbd);background:var(--errbg)}
.railitem.disabled{opacity:.5}
.ph.sel{background:var(--sel)}
.phname{color:var(--t2)}
.rr.sel .mut{color:var(--sel-t2)}
.rr.sel .okt{color:var(--sel-ok)}
.rr.sel .warnt{color:var(--sel-warn)}
.lights i{background:var(--lights)}
"""

# Back-compat alias for builders still importing LIGHT_VARS (round-2 names).
from _tokens import theme_tokens as _theme_tokens, css_vars  # noqa: E402
LIGHT_VARS = _theme_tokens("light", "teal", "dark")
