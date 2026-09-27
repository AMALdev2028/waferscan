"""Self-contained HTML batch report in the landing page's monochrome style
(Bebas Neue + JetBrains Mono, #000-#fff only). Images are inline PNG data
URIs, the Pareto drill-down is plain <details> - no JavaScript needed."""
from __future__ import annotations

import base64
import datetime as dt
import html
import io

import numpy as np
from PIL import Image

CSS = """
:root{color-scheme:dark;--g0:#000;--g1:#0d0d0d;--g2:#1a1a1a;--g3:#2a2a2a;--g5:#555;--g6:#666;--g7:#888;--g9:#ccc;--g12:#fff;
--disp:'Bebas Neue','Arial Black',sans-serif;--mono:'JetBrains Mono','Consolas',monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--g0);color:var(--g7);font-family:var(--mono);font-size:13px;padding:32px 16px;max-width:1100px;margin:0 auto}
.stripe{height:5px;background:repeating-linear-gradient(-45deg,var(--g12) 0,var(--g12) 8px,var(--g0) 8px,var(--g0) 16px);margin-bottom:24px}
h1{font-family:var(--disp);font-size:3rem;color:var(--g12);letter-spacing:.04em;line-height:.95}
h2{font-family:var(--disp);font-size:1.5rem;color:var(--g11,#f5f5f5);letter-spacing:.06em;margin:40px 0 12px}
.tag{font-size:.62rem;font-weight:700;letter-spacing:.3em;text-transform:uppercase;color:var(--g6);margin-bottom:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:2px;background:var(--g3);border:1px solid var(--g3)}
.cell{background:var(--g1);padding:18px}
.val{font-family:var(--disp);font-size:2.2rem;color:var(--g12);font-variant-numeric:tabular-nums}
.lab{font-size:.58rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;color:var(--g6)}
.sub{font-size:.62rem;color:var(--g5);margin-top:4px}
.wafers{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:2px;background:var(--g3)}
.wafer{background:var(--g0);padding:10px;font-size:.6rem}
.wafer img{width:100%;image-rendering:pixelated;display:block;margin-bottom:6px}
.wafer b{color:var(--g12)}
.review{border-left:3px solid var(--g12)}
table{width:100%;border-collapse:collapse;font-size:.65rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--g2)}
th{color:var(--g6);font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:.55rem}
td{color:var(--g9)}
.scroll{overflow-x:auto}
details{background:var(--g1);border:1px solid var(--g3);padding:10px 14px;margin-top:2px}
summary{cursor:pointer;color:var(--g12);font-weight:700}
details li{margin:6px 0 0 18px;color:var(--g7)}
.foot{margin-top:48px;padding-top:16px;border-top:2px solid var(--g12);font-size:.6rem;color:var(--g5);line-height:1.8}
"""


def _png(arr: np.ndarray, scale: int = 4) -> str:
    img = Image.fromarray(arr.astype(np.uint8)).resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def wafer_png(dm: np.ndarray, quant: dict) -> str:
    """off #000 · pass #1a1a1a · background fail #666 · pattern fail #fff"""
    g = np.where(dm > 0, 26, 0).astype(np.uint8)
    g[dm == 2] = 102
    for d in quant.get("failed_dies", []):
        if d["in_pattern"]:
            g[d["row"], d["col"]] = 255
    return _png(g, max(1, 160 // max(dm.shape)))


def _pareto_svg(pareto: list[dict]) -> str:
    items = pareto[:8]
    if not items:
        return ""
    w, h, bw = 640, 200, 640 / len(items)
    bars, pts = [], []
    for i, p in enumerate(items):
        bh = p["contribution"] * (h - 20)
        bars.append(f'<rect x="{i * bw + 6:.1f}" y="{h - bh:.1f}" width="{bw - 12:.1f}" height="{bh:.1f}" fill="#ccc"/>'
                    f'<text x="{i * bw + bw / 2:.1f}" y="{h - bh - 4:.1f}" fill="#fff" font-size="10" '
                    f'text-anchor="middle" font-family="JetBrains Mono">{100 * p["contribution"]:.0f}%</text>')
        pts.append(f"{i * bw + bw / 2:.1f},{h - p['cumulative'] * (h - 20):.1f}")
    line = f'<polyline points="{" ".join(pts)}" fill="none" stroke="#fff" stroke-width="1.5" stroke-dasharray="4,3"/>'
    labels = "".join(f'<text x="{i * bw + bw / 2:.1f}" y="{h + 14}" fill="#888" font-size="8" text-anchor="middle" '
                     f'font-family="JetBrains Mono">{html.escape(p["cause"][:18])}</text>' for i, p in enumerate(items))
    return (f'<div class="scroll"><svg viewBox="0 -10 {w} {h + 30}" width="100%" style="min-width:560px">'
            f'<line x1="0" y1="{h}" x2="{w}" y2="{h}" stroke="#2a2a2a"/>{"".join(bars)}{line}{labels}</svg></div>')


def render(batch_id: str, b: dict) -> str:
    s, rows = b["summary"], b["rows"]
    esc = html.escape
    cells = [("Wafers", s["wafers"], ""),
             ("Fail % (pooled)", f'{s["fail_pct_pooled"]:.2f}', f'95% Wilson {s["fail_pct_pooled_wilson95"]}'),
             ("Pattern-affected %", f'{s["pattern_affected_pct_pooled"]:.2f}', "dies inside classified patterns"),
             ("Human review", s["human_review"], "low confidence or failed morphology check")]
    metric_html = "".join(f'<div class="cell"><div class="val">{esc(str(v))}</div><div class="lab">{esc(k)}</div>'
                          f'<div class="sub">{esc(sub)}</div></div>' for k, v, sub in cells)
    wafers = "".join(
        f'<div class="wafer{" review" if r["action"] == "human_review" else ""}">'
        f'<img alt="wafer {esc(str(r["wafer_id"]))}" src="{thumb}">'
        f'<b>{esc(r["class"].upper())}</b> {100 * r["confidence"]:.1f}%<br>{esc(str(r["wafer_id"]))} · '
        f'{r["affected_pct"]:.1f}% affected<br>{esc(r["action"].replace("_", " "))}</div>'
        for r, thumb in zip(rows, b["thumbs"]))
    heat = np.asarray(s["heatmap"])
    heat_img = _png((255 * heat / max(heat.max(), 1e-9)).astype(np.uint8), 5)
    radial = "".join(f'<rect x="{i * 30 + 4}" y="{100 - 95 * v / max(max(s["radial_profile"]), 1e-9):.1f}" width="22" '
                     f'height="{95 * v / max(max(s["radial_profile"]), 1e-9):.1f}" fill="#ccc"/>'
                     for i, v in enumerate(s["radial_profile"]))
    table = "".join(f'<tr><td>{esc(str(r["wafer_id"]))}</td><td>{esc(str(r["lot_id"] or ""))}</td><td>{esc(r["class"])}</td>'
                    f'<td>{100 * r["confidence"]:.1f}%</td><td>{r["verification"]}</td><td>{esc(r["action"])}</td>'
                    f'<td>{r["fail_pct"]:.2f}</td><td>{r["affected_pct"]:.2f}</td><td>{r["area_mm2"]:.0f}</td></tr>'
                    for r in rows)
    rc = b.get("rootcause")
    rc_html = ""
    if rc:
        drill = "".join(f'<details><summary>{100 * p["contribution"]:.1f}% · {esc(p["label"])}</summary><ul>'
                        + "".join(f"<li>{esc(e)}</li>" for e in (p["evidence"] or ["prior only - no significant "
                                                                                     "process evidence"]))
                        + "".join(f"<li>{esc(c)}: P(cause|class) = {v:.2f}</li>" for c, v in p["classes"].items())
                        + "</ul></details>" for p in rc["pareto"])
        rc_html = (f'<h2>ROOT CAUSE PARETO</h2><p class="tag">{esc(rc["caveat"])}</p>{_pareto_svg(rc["pareto"])}'
                   f'{drill}')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>WaferScan Report {esc(batch_id)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=JetBrains+Mono:wght@400;700&display=swap">
<style>{CSS}</style></head><body><div class="stripe"></div>
<p class="tag">WaferScan · batch report · {esc(batch_id)} · {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC</p>
<h1>BATCH<br>ANALYSIS</h1>
<h2>SUMMARY</h2><div class="grid">{metric_html}</div>
<h2>CLASS DISTRIBUTION</h2><div class="grid">{"".join(f'<div class="cell"><div class="val">{n}</div><div class="lab">{esc(c)}</div></div>' for c, n in s["class_counts"].items())}</div>
<h2>WAFERS</h2><p class="tag">white = dies in the defect pattern · grey = isolated fails · bar = sent to human review</p>
<div class="wafers">{wafers}</div>
<h2>BATCH DEFECT DENSITY</h2><div class="grid"><div class="cell"><img alt="batch heatmap" src="{heat_img}" style="width:100%;max-width:320px;image-rendering:pixelated">
<div class="sub">mean failed-die density across the batch (white = high)</div></div>
<div class="cell"><svg viewBox="0 0 300 110" width="100%">{radial}<line x1="0" y1="100" x2="300" y2="100" stroke="#2a2a2a"/></svg>
<div class="sub">fail density by radius · centre → edge (10 rings)</div></div></div>
{rc_html}
<h2>WAFER TABLE</h2><div class="scroll"><table><tr><th>wafer</th><th>lot</th><th>class</th><th>conf</th><th>verify</th><th>action</th><th>fail %</th><th>affected %</th><th>mm²</th></tr>{table}</table></div>
<div class="foot">Fail-rate intervals are Wilson 95% (dies treated as independent - optimistic for clustered fails).
Areas assume the stated wafer diameter and an estimated die pitch unless die size was supplied.
Root-cause rankings are statistical associations weighted by engineering priors, not proof of causation.</div>
</body></html>"""
