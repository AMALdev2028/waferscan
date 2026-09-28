/* WaferScan front end. Talks to the FastAPI service at /api/v1; when the page is opened
   without the server it falls back to three recorded results so the design still works. */
'use strict';

const API = '/api/v1';
const MAX_UPLOAD = 20 * 1024 * 1024;               // same limit as the server
const OK_TYPES = /\.(png|jpe?g|bmp|tiff?)$/i;
const CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None'];
const ABBR = { 'Center': 'CEN', 'Donut': 'DON', 'Edge-Loc': 'E-L', 'Edge-Ring': 'E-R', 'Loc': 'LOC', 'Near-full': 'N-F', 'Random': 'RAN', 'Scratch': 'SCR', 'None': 'NON' };
const REDUCED = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
const MIN_SCAN_MS = REDUCED ? 0 : 900;             // the real answer takes ~11 ms; keep the sweep readable

const $ = s => document.querySelector(s);
const size = b => b < 1024 ? b + ' B' : b < 1048576 ? Math.round(b / 1024) + ' KB' : (b / 1048576).toFixed(1) + ' MB';
const fmt = (n, d = 1) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const esc = s => String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
const icon = (id, cls = 'icon-sm') => `<svg class="${cls}" aria-hidden="true"><use href="icons.svg#${id}"/></svg>`;
const sleep = ms => new Promise(r => setTimeout(r, ms));

const app = { live: false, card: null, samples: [], source: 'sample', cls: 'Scratch', res: null, truth: null,
              map: null, pattern: new Set(), clusters: null, busy: false };

/* ── recorded fallback (no server) ─────────────────────────────── */
const DEMO_MAPS = {
  "Center": [
    "000000000000000000002211111111100000000000000000000",
    "000000000000000001111112111111121100000000000000000",
    "000000000000001111111131112111111233300000000000000",
    "000000000000111212111313112112121113111000000000000",
    "000000000002111111111131111111111112111100000000000",
    "000000000011111113111112121111121111111110000000000",
    "000000001111221213311111131133131111111111100000000",
    "000000011211111113111111333311331112111111110000000",
    "000000122113311111312111311111111111111331111000000",
    "000000111133111112113111311111211211111312111000000",
    "000001111111111111111311212221111211121111121100000",
    "000021111111111111121131111111111111311123111110000",
    "000111111111112111111113211111211233111111311121000",
    "000112111131111211121111111121121111211111131111000",
    "001111113331111112211121112111112111111211133112100",
    "001211131111111111111111112121111121111111113111100",
    "001111133111212121111211111111111121211111111111100",
    "011111111211212121111211212211111111212111331111120",
    "011211121111112111111111121111211111111113311111110",
    "011111111112111111113131111111111111111113111111330",
    "211211111211211111333333111112111211112131121113131",
    "121111111112111113333333111111313111211133111212111",
    "211111111111121133333333311111133121111311111111121",
    "111133121111121333333333331111111111121131112111111",
    "111131111111111333333333331112112111111113131121111",
    "111131212112113333333333331113111311211113313311121",
    "111121111111211333333333331113333111213311131311111",
    "111111121311113333333333331231111121113111133111111",
    "111111131331111133333333311111221121213112331111111",
    "111111113131121133333333311111111212112111113331111",
    "112111212111111111333333112121111112111111231331111",
    "011121111121211211111131113131111111111111111111110",
    "011112111111111122111131111311111111213312111111110",
    "011211111111112111121333113111111112111311112221110",
    "001111112111111111111131121112121111111111111111100",
    "001111111112121211111331111131111111111111111111100",
    "001111111113111111111111112113311112111211112131200",
    "000112111131312111112111111111311211111121211133000",
    "000121121113311111111113311111111133311111111131000",
    "000011113311111111111111311212111311121111211310000",
    "000001133112112111122112111211112111111111111200000",
    "000000113111211121111211121111111111111111111000000",
    "000000121111111111111212113111112111311111111000000",
    "000000011111112111211112111311112123311111110000000",
    "000000001111111211111111123111111111111111100000000",
    "000000000012111111111112111111111211111120000000000",
    "000000000001111211111111111212111111212100000000000",
    "000000000000111111111112111111111111121000000000000",
    "000000000000001111111111112111211111100000000000000",
    "000000000000000001121111111121111200000000000000000",
    "000000000000000000002212211111100000000000000000000"],
  "Scratch": [
    "000000000000000000002111111211100000000000000000000",
    "000000000000000001111111121311113100000000000000000",
    "000000000000001112111112113131113311100000000000000",
    "000000000000111211211111211313131111112000000000000",
    "000000000001111111112121111113311111112100000000000",
    "000000000011111121111111111111111113211110000000000",
    "000000002111211111221111111111221131111111100000000",
    "000000012111211111111111111112111312111121110000000",
    "000000111112112211122111331121133111111112111000000",
    "000000212211111111111111131111313111112111111000000",
    "000001211111111111311111331111133311211111111100000",
    "000012111111111113331111111111111311112111111110000",
    "000112111111212111112111111111111331121112222111000",
    "000111111111112111111113111211111111112111111111000",
    "001111121211211111111113311111111111111111221121100",
    "001211111121111111111111111111111121121111111122100",
    "001111111112111112111121111111211111111111111111100",
    "012111211111111111121212112133111111212113111212110",
    "011133111131111112111111111131111111111331211111220",
    "011131111331111111213312111331212111111111112111210",
    "112111111311111112113311121311111111111112111111112",
    "111111331211121112111111211111111111112111211111131",
    "122123311111211111111211112121111111121111111111311",
    "111111111211111121111111111111111111111112113313112",
    "221111111111112112111112112111112112111111131131111",
    "111111121111121111122112112111111111111111131131111",
    "121111111111111121111111111111111331111111311112112",
    "112111131111111111112111111122113131111111211211111",
    "111121133212112111211211112111113111112111111111111",
    "111121111112121111211111111111131311113112211212111",
    "111112111311111112121111333311211312131311111121121",
    "011111213131121111112113111131113111113311212111110",
    "011311113311111111111111311121112112111111112111110",
    "013311111311111112211111312111211112111111111123310",
    "001112213313311111111123112111111111111212112111300",
    "002111111111311111111111111111121111111111111111100",
    "002111111111111331111121111121111111111111111312100",
    "000112211121113113311111112111112111111112111331000",
    "000111112113113111121111111111111112112112111311000",
    "000011111113131311111111111112111111221111111120000",
    "000001121131113131112111212111111111111121111100000",
    "000000111112111121111221111111131122111112111000000",
    "000000331111111111331111131111331111111111121000000",
    "000000033331131111311121133111113111111212110000000",
    "000000001133331111111111131112111332111111100000000",
    "000000000011133331111113333111121311112110000000000",
    "000000000001111133311111311311111311211200000000000",
    "000000000000121111333111112111111131111000000000000",
    "000000000000001212133331311121133311100000000000000",
    "000000000000000001131113311111113100000000000000000",
    "000000000000000000003211333211100000000000000000000"],
  "Edge-Ring": [
    "000000000000000000003333333333300000000000000000000",
    "000000000000000003333333313131331300000000000000000",
    "000000000000002133333111113113333333300000000000000",
    "000000000000113111111211231111111331331000000000000",
    "000000000003331111211111111111211133333300000000000",
    "000000000033113111111111111111311113113330000000000",
    "000000003333333111111112211313111111111133300000000",
    "000000011311311111111111112131111111111113310000000",
    "000000331111112111121111111111211111211331333000000",
    "000000333111111111111131111211311121111133133000000",
    "000003311211111121112133112113311111112111131300000",
    "000033311111111121111111311111113111111221113130000",
    "000311111111111112211111211111331212111111231133000",
    "000131212111111121111111111112111111112111111113000",
    "003331112111121111111111121111111211111111111113100",
    "003111112111112112121111111111121112121111111111300",
    "003311111111112111121111121121111111111112111111300",
    "033113111112111121211111111113111212113311113311330",
    "033133111111111111112111111113111112111311113111310",
    "031311111211111111212111111131111111113111133311330",
    "133111121111112331111121121331121112131112111111131",
    "111111121111211131121111131111211211133111111111333",
    "311111111211111112111111113331211112113111221111133",
    "331111111211111111111111113311111111211121111121113",
    "331121111122111212212112111312111131111111211111233",
    "331111111111121111111212111131111331111111131111113",
    "331311112111111111111111211111121311121113113333133",
    "131331112111111111211121111211131111131121331131313",
    "313131112112111121111111111211331111311111111133133",
    "311111111111311111211121211111131123131111211111133",
    "131111112211311121111211311121131111131211111111311",
    "033111111111331121111113311211331211113111113113330",
    "033111111111111111111111111211311121111111133111130",
    "033112111111121111111212111211132111112211111111130",
    "003313111121211221111111111111111111121111113311200",
    "003313111111112111111311121211211112111122113111100",
    "003311331211111113313311111111111121211111111131100",
    "000331131211112113331111111111111111112111211133000",
    "000333111111121211131122111111122112221111111133000",
    "000013312111111111211211111112111111111111121330000",
    "000003111112111111111112111111111132111111113300000",
    "000000313111121212121121111211133311111111133000000",
    "000000333111111111112111111112131111211113333000000",
    "000000033311111121111122121111112111111113310000000",
    "000000003333111111211211111221111111111333300000000",
    "000000000033311121111111111111111212113330000000000",
    "000000000003311111121333311111111113131300000000000",
    "000000000000333111111133111123311331333000000000000",
    "000000000000001333131133121111333333300000000000000",
    "000000000000000003331133311313333300000000000000000",
    "000000000000000000002133331333300000000000000000000"]
};
const DEMO = {
  'Center':    { conf: 0.9975, probs: { 'Center': 0.9975, 'Near-full': 0.0004, 'Scratch': 0.0003, 'Random': 0.0003 }, dies: 261, pct: 12.66, area: 8674 },
  'Scratch':   { conf: 0.9913, probs: { 'Scratch': 0.9913, 'Loc': 0.0035, 'Edge-Loc': 0.0016, 'None': 0.0013 }, dies: 189, pct: 9.17, area: 6281 },
  'Edge-Ring': { conf: 0.9976, probs: { 'Edge-Ring': 0.9976, 'Edge-Loc': 0.0014, 'Scratch': 0.0002, 'None': 0.0002 }, dies: 389, pct: 18.87, area: 12927 },
};
const DEMO_CM = [[701,0,0,0,0,0,0,0,0],[0,700,0,0,0,0,1,0,0],[1,0,685,13,1,0,0,1,0],[0,0,5,696,0,0,0,0,0],[0,0,1,0,696,0,0,4,0],
                 [0,0,0,0,0,690,11,0,0],[0,0,0,0,0,9,692,0,0],[0,0,1,0,2,0,0,698,0],[0,0,0,0,0,0,0,0,701]];
function demoResult(cls) {
  const rows = DEMO_MAPS[cls], d = DEMO[cls];
  const die_map = rows.map(s => [...s].map(ch => (ch === '0' ? 0 : ch === '1' ? 1 : 2)));
  const failed = [];
  rows.forEach((s, r) => [...s].forEach((ch, c) => { if (ch === '2' || ch === '3') failed.push({ row: r, col: c, in_pattern: ch === '3' }); }));
  const total = die_map.flat().filter(Boolean).length;
  return {
    prediction: { class: cls, confidence: d.conf, path: 'fast', probabilities: d.probs },
    decision: { action: 'auto_accept', note: null },
    verification: { status: 'PASS', reasons: [] },
    quantification: { units: { die_w_mm: 300 / 51, die_h_mm: 300 / 51, wafer_diameter_mm: 300 },
      pattern: { dies: d.dies, affected_pct: d.pct, area_mm2: d.area }, wafer: { dies_total: total, dies_failed: failed.length },
      failed_dies: failed, failed_dies_truncated: false },
    wafer: { rows: 51, cols: 51, die_map }, cam: null, timing_ms: { total: 11 }, demo: true,
  };
}

/* ── small helpers ─────────────────────────────────────────────── */
async function api(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let detail = r.statusText || 'request failed';
    try { const j = await r.json(); detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    const err = new Error(detail); err.status = r.status; throw err;
  }
  return r.json();
}
function log(text, kind = '') {
  const el = $('#log'), p = document.createElement('p');
  p.textContent = text; if (kind) p.className = kind;
  el.appendChild(p);
  while (el.children.length > 40) el.firstChild.remove();
  el.scrollTop = el.scrollHeight;
}
function setStatus(state, text) { $('#status').dataset.state = state; $('#status-text').textContent = text; }
let toastTimer = 0;
function toast(text) {
  const t = $('#toast'); t.innerHTML = icon('i-check') + esc(text); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 4000);
}
function severity(cls, pct) {
  if (cls === 'None') return 'NONE';
  return pct < 2 ? 'LOW' : pct <= 10 ? 'MEDIUM' : 'HIGH';    // proposed cut-offs, see the UI brief
}

/* ── die map ───────────────────────────────────────────────────── */
const map = DefectMap.create($('#map'), {
  onSelect: rc => showInspector(rc),
  onZoom: z => { $('#zoom-text').textContent = (Math.round(z * 10) / 10) + '×'; },
  onShape: name => { if (map && map.state.mode === 'idle') $('#map-meta').textContent = 'IDLE · ' + name.toUpperCase(); },
});

function clusterSizes() {            // connected groups of failing dies (4-neighbour), once per result
  if (app.clusters) return app.clusters;
  const m = app.map, id = m.map(row => row.map(() => -1)), sizes = [];
  for (let r0 = 0; r0 < m.length; r0++) for (let c0 = 0; c0 < m[0].length; c0++) {
    if (m[r0][c0] !== 2 || id[r0][c0] >= 0) continue;
    const stack = [[r0, c0]], n = sizes.length; let size = 0; id[r0][c0] = n;
    while (stack.length) {
      const [r, c] = stack.pop(); size++;
      for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const rr = r + dr, cc = c + dc;
        if (m[rr] && m[rr][cc] === 2 && id[rr][cc] < 0) { id[rr][cc] = n; stack.push([rr, cc]); }
      }
    }
    sizes.push(size);
  }
  return (app.clusters = { id, sizes });
}

function showInspector(rc) {
  const box = $('#inspector');
  if (!rc || !app.res) { box.hidden = true; $('#inspect-text').textContent = app.res ? 'CLICK A DIE' : 'INSPECT AFTER SCAN'; return; }
  const [r, c] = rc, m = app.map, v = m[r][c], res = app.res, u = res.quantification.units || {};
  const inPattern = app.pattern.has(r + ',' + c), cl = clusterSizes(), cid = cl.id[r][c];
  const w = u.die_w_mm || 300 / m[0].length, h = u.die_h_mm || w;
  const x = (c - (m[0].length - 1) / 2) * w, y = ((m.length - 1) / 2 - r) * h;
  const rows = [
    ['State', v === 2 ? 'Fail' : 'Pass'],
    ['In pattern', inPattern ? 'Yes · ' + res.prediction.class : 'No', inPattern],
    ['Cluster', cid >= 0 ? cl.sizes[cid] + (cl.sizes[cid] === 1 ? ' die' : ' dies') : '—'],
    ['Position', `${fmt(x)}, ${fmt(y)} mm`],
  ];
  if (res.cam && res.cam[r] && res.cam[r][c] != null) {
    const s = map.state, t = (res.cam[r][c] - s.camMin) / ((s.camMax - s.camMin) || 1);
    rows.push(['CNN attention', fmt(t, 2)]);
  }
  $('#ins-title').textContent = `Die ${r}, ${c}`;
  $('#ins-list').innerHTML = rows.map(([k, val, hot]) => `<dt>${k}</dt><dd${hot ? ' class="is-signal"' : ''}>${esc(val)}</dd>`).join('');
  $('#inspect-text').textContent = `DIE ${r},${c}`;
  box.hidden = false;
}

/* ── result panel states ───────────────────────────────────────── */
function renderIdle() {
  $('#result').innerHTML = `<span class="label">Result</span><h2 class="state-word">Ready</h2>
    <p class="note">Pick a sample pattern or upload a wafer image, then run the scan. The map on the left is cycling through the nine patterns the model knows.</p>`;
}
function renderScanning() {
  $('#result').innerHTML = `<span class="label">Result</span><h2><em>Scanning</em></h2>
    <div class="loading-dots" aria-hidden="true"><i></i><i></i><i></i></div><p class="sr-only">Scanning the wafer</p>`;
}

function renderResult(res, truth) {
  app.res = res; app.truth = truth || null; app.clusters = null;
  app.map = res.wafer.die_map;
  const keys = res.quantification.failed_dies.filter(d => d.in_pattern).map(d => [d.row, d.col]);
  app.pattern = new Set(keys.map(k => k.join(',')));
  map.result(app.map, keys, res.cam);

  const p = res.prediction, d = res.decision, v = res.verification, q = res.quantification.pattern, total = res.quantification.wafer.dies_total;
  const accept = d.action === 'auto_accept', sev = severity(p.class, q.affected_pct);
  const top = Object.entries(p.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 4);
  const shapeChip = v.status === 'PASS' ? `<span class="chip">${icon('i-check')}SHAPE CHECK PASS</span>`
    : v.status === 'FAIL' ? `<span class="chip chip-signal">${icon('i-alert')}SHAPE CHECK FAIL</span>`
    : `<span class="chip chip-dashed">${icon('i-alert')}SHAPE CHECK ${esc(v.status)}</span>`;
  const bar = pr => { const n = pr >= 0.005 ? Math.max(1, Math.round(pr * 20)) : 0;
    return Array.from({ length: 20 }, (_, i) => `<i${i < n ? ' class="on"' : ''}></i>`).join(''); };

  $('#result').innerHTML = `
    <div class="rise"><span class="label">Defect pattern</span><h2>${esc(p.class)}</h2></div>
    <div class="result-row">
      <span class="conf">${fmt(100 * p.confidence)}%</span>
      <span class="chip ${accept ? 'chip-solid' : 'chip-outline'}">${accept ? 'AUTO-ACCEPT' : 'HUMAN REVIEW'}</span>
      ${shapeChip}
      <span class="chip chip-dashed">SEVERITY · ${sev}</span>
    </div>
    ${accept ? '' : `<p class="note">Sent to review: ${esc(d.note || 'the model is not sure enough, or the shape check disagrees')}. Click the red dies to check them.</p>`}
    ${truth ? `<p class="note">Dataset label: <b>${esc(truth)}</b> · ${truth === p.class ? 'matches' : 'does not match'} the prediction.</p>` : ''}
    <div class="divided"><span class="label">Class probability</span>
      ${top.map(([c, pr]) => `<div class="prob"><span>${esc(c)}</span><span class="prob-dots" role="img" aria-label="${fmt(100 * pr, 2)} percent">${bar(pr)}</span><span class="mono">${fmt(100 * pr, pr >= 0.01 ? 1 : 2)}%</span></div>`).join('')}
    </div>
    <div class="divided"><div class="sizes">
      <div><b>${q.dies.toLocaleString()}</b><span>Pattern dies</span></div>
      <div><b>${fmt(q.affected_pct)}%</b><span>Of wafer</span></div>
      <div><b>${fmt(q.area_mm2, 0)}</b><span>mm² affected</span></div>
    </div></div>
    <div class="actions">
      <button class="btn btn-secondary" type="button" id="dl-json">${icon('i-download')}Download JSON</button>
      <button class="btn btn-ghost" type="button" id="copy-json">${icon('i-copy')}Copy</button>
    </div>`;
  $('#dl-json').addEventListener('click', downloadJson);
  $('#copy-json').addEventListener('click', copyJson);

  $('#map-meta').textContent = `${res.wafer.rows} × ${res.wafer.cols} · ${total.toLocaleString()} DIES`;
  $('#map').setAttribute('aria-label', `Wafer die map: ${p.class}, ${q.dies} of ${total} dies in the pattern, shown in red. Decision: ${accept ? 'auto-accept' : 'human review'}. Use arrow keys to inspect dies.`);
  $('#inspect-text').textContent = 'CLICK A DIE';

  log(`Inference · ${p.path} path · ${fmt(res.timing_ms.total)} ms total`);
  log(`Class ${p.class} · ${fmt(100 * p.confidence)}% · shape check ${v.status}${v.reasons && v.reasons.length ? ' · ' + v.reasons[0] : ''}`);
  log(`Pattern ${q.dies} dies · ${fmt(q.affected_pct)}% of wafer · ${fmt(q.area_mm2, 0)} mm²`);
  log(accept ? 'DECISION: AUTO-ACCEPT' : 'DECISION: HUMAN REVIEW', accept ? 'ok' : 'err');
}

function explain(err) {
  const msg = (err && err.message) || 'unknown error', s = err && err.status;
  if (!s) return ['—', 'Server not reachable', 'The page could not reach the WaferScan API.', 'Start the server (uvicorn waferscan.api.main:app) and open the page from it, or try a sample wafer.'];
  if (s === 413) return [s, 'File too large', 'Uploads are limited to 20 MB.', 'Crop to the wafer or save as a smaller PNG or JPG.'];
  if (s === 415) return [s, 'Unsupported file type', msg, 'Use PNG, JPG, BMP or TIFF.'];
  if (/die grid/i.test(msg)) return [s, 'No die grid found', 'The image decoded, but no repeating grid of dies was visible.', 'Upload a wafer map or a photo where the whole wafer and its die grid are in frame.'];
  if (/decod/i.test(msg)) return [s, 'Not a decodable image', 'The file reached the server but is not a real image: usually a renamed or damaged file, or a format like HEIC or PDF.', 'Open it in an image viewer, save it again as PNG or JPG, and upload it once more.'];
  return [s, 'Upload failed', msg, 'Check the file and try again, or try a sample wafer.'];
}
function renderError(err) {
  const [code, title, cause, fix] = explain(err);
  map.error(); app.res = null;
  $('#map-meta').textContent = 'NO INPUT';
  $('#result').innerHTML = `
    <div class="rise" role="alert"><span class="label" style="color:var(--signal)">Upload failed</span></div>
    <span class="error-code">${esc(code)}</span>
    <p class="error-title">${esc(title)}</p>
    <p class="note">${esc(cause)}</p>
    <p class="note">${esc(fix)}</p>
    <div class="actions"><button class="btn btn-secondary" type="button" id="try-sample">${icon('i-sample')}Try a sample wafer</button></div>`;
  $('#try-sample').addEventListener('click', () => { setSource('sample'); run(); });
  log(`Upload failed: ${code} ${err.message || ''}`.trim(), 'err');
}

/* ── actions ───────────────────────────────────────────────────── */
async function withScan(label, work) {
  if (app.busy) return;
  app.busy = true; $('#run').disabled = true;
  app.res = null; showInspector(null);
  renderScanning(); log(label);
  const t0 = performance.now();
  try {
    const out = await work();
    await sleep(Math.max(0, MIN_SCAN_MS - (performance.now() - t0)));
    renderResult(out.res, out.truth);
  } catch (err) {
    await sleep(Math.max(0, MIN_SCAN_MS / 2 - (performance.now() - t0)));
    renderError(err);
  } finally {
    app.busy = false; $('#run').disabled = false;
  }
}

function runSample() {
  const cls = app.cls;
  if (!app.live) {
    return withScan(`Loaded recorded ${cls} wafer (offline demo)`, async () => {
      const res = demoResult(cls); map.scanning(res.wafer.die_map); return { res, truth: cls };
    });
  }
  const pool = app.samples.filter(s => s.label === cls);
  const s = pool[Math.floor(Math.random() * pool.length)];
  return withScan(`Loading validation wafer #${s.id} · ${s.rows}×${s.cols} dies`, async () => {
    const w = await api('/samples/' + s.id);
    map.scanning(w.wafer_map);
    const res = await api('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ wafer_map: w.wafer_map }) });
    return { res, truth: w.label };
  });
}

function runUpload(file) {
  $('#drop-text').textContent = `${file.name} · ${size(file.size)}`;
  return withScan(`Uploading ${file.name} (${size(file.size)})`, async () => {
    map.scanning();
    if (!file.size) { const e = new Error('empty file'); e.status = 422; throw e; }
    if (file.size > MAX_UPLOAD) { const e = new Error('file larger than 20 MB'); e.status = 413; throw e; }
    if (!OK_TYPES.test(file.name) && !/^image\//.test(file.type)) { const e = new Error(`${file.type || 'unknown type'} is not supported`); e.status = 415; throw e; }
    if (!app.live) throw new Error('offline');
    const fd = new FormData(); fd.append('file', file); fd.append('kind', 'auto');
    const res = await api('/predict/image', { method: 'POST', body: fd });
    const conv = res.image_to_diemap || {};
    log(`Read as ${(res.input_kind || 'image').replace('_', ' ')} → ${res.wafer.rows}×${res.wafer.cols} die grid${conv.ms ? ` in ${fmt(conv.ms, 0)} ms` : ''}`);
    return { res, truth: null };
  });
}

function run() {
  if (app.source === 'sample') return runSample();
  const f = $('#file').files[0];
  if (f) return runUpload(f);
  $('#dropzone').classList.add('is-over'); setTimeout(() => $('#dropzone').classList.remove('is-over'), 600);
  $('#file').focus();
}

function downloadJson() {
  const r = app.res; if (!r) return;
  const blob = new Blob([JSON.stringify(r, null, 1)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = `waferscan_${r.prediction.class}_${Date.now()}.json`;
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  toast('Result saved as JSON');
}
async function copyJson() {
  try { await navigator.clipboard.writeText(JSON.stringify(app.res, null, 1)); toast('Result copied as JSON'); }
  catch (_) { toast('Copy is blocked here — use Download'); }
}

/* ── controls ──────────────────────────────────────────────────── */
function setSource(src) {
  app.source = src;
  $('#src-sample').setAttribute('aria-pressed', String(src === 'sample'));
  $('#src-upload').setAttribute('aria-pressed', String(src === 'upload'));
  $('#panel-sample').hidden = src !== 'sample';
  $('#panel-upload').hidden = src !== 'upload';
}
function buildChips(classes) {
  if (!classes.includes(app.cls)) app.cls = classes[0];
  $('#class-chips').innerHTML = classes.map(c => `<button type="button" data-cls="${esc(c)}" aria-pressed="${c === app.cls}">${esc(c)}</button>`).join('');
}
$('#class-chips').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  app.cls = b.dataset.cls;
  document.querySelectorAll('#class-chips button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
});
$('#src-sample').addEventListener('click', () => setSource('sample'));
$('#src-upload').addEventListener('click', () => setSource('upload'));
$('#run').addEventListener('click', run);
$('#file').addEventListener('change', e => { const f = e.target.files[0]; if (f) runUpload(f); });
const dz = $('#dropzone');
['dragenter', 'dragover'].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.add('is-over'); }));
['dragleave', 'drop'].forEach(t => dz.addEventListener(t, () => dz.classList.remove('is-over')));
dz.addEventListener('drop', e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) runUpload(f); });

$('#zoom-in').addEventListener('click', () => map.zoomIn());
$('#zoom-out').addEventListener('click', () => map.zoomOut());
$('#zoom-fit').addEventListener('click', () => map.fit());
$('#inspector-close').addEventListener('click', () => { map.select(null); $('#map').focus(); });
const layersBtn = $('#layers-btn'), layersBox = $('#layers');
const setLayersOpen = open => { layersBox.hidden = !open; layersBtn.setAttribute('aria-expanded', String(open)); };
layersBtn.addEventListener('click', () => setLayersOpen(layersBox.hidden));
layersBox.addEventListener('click', e => {
  const b = e.target.closest('.switch'); if (!b || b.disabled) return;
  const on = b.getAttribute('aria-checked') !== 'true';
  b.setAttribute('aria-checked', String(on)); map.setLayer(b.dataset.layer, on);
});
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !layersBox.hidden) { setLayersOpen(false); layersBtn.focus(); } });
document.addEventListener('pointerdown', e => { if (!layersBox.hidden && !layersBox.contains(e.target) && !layersBtn.contains(e.target)) setLayersOpen(false); });

/* theme: dark by default, remembered per browser (theme.js applies it before paint) */
(function () {
  const btn = $('#theme-toggle'), root = document.documentElement;
  const label = () => {
    const light = root.dataset.theme === 'light';
    $('#theme-icon').setAttribute('href', 'icons.svg#' + (light ? 'i-moon' : 'i-sun'));
    $('#theme-text').textContent = light ? 'DARK' : 'LIGHT';
    btn.setAttribute('aria-label', light ? 'Switch to dark theme' : 'Switch to light theme');
  };
  label();
  btn.addEventListener('click', () => {
    if (root.dataset.theme === 'light') delete root.dataset.theme; else root.dataset.theme = 'light';
    try { localStorage.setItem('waferscan-theme', root.dataset.theme || 'dark'); } catch (_) {}
    label();
  });
})();

/* nav highlight follows the section in view */
if (window.IntersectionObserver) {
  const links = [...document.querySelectorAll('.nav-links a')];
  const io = new IntersectionObserver(es => es.forEach(en => {
    if (en.isIntersecting) links.forEach(a => a.setAttribute('aria-current', String(a.getAttribute('href') === '#' + en.target.id)));
  }), { rootMargin: '-45% 0px -50% 0px' });
  ['scan', 'how', 'results', 'limits'].forEach(id => io.observe(document.getElementById(id)));
}

/* ── model card → numbers on the page ─────────────────────────── */
function renderCM(classes, cm) {
  const max = Math.max(...cm.flat());
  const head = `<thead><tr><th scope="col"><span class="sr-only">Actual</span></th>${classes.map(c => `<th scope="col" title="${esc(c)}">${ABBR[c] || c.slice(0, 3)}</th>`).join('')}</tr></thead>`;
  const body = cm.map((row, i) => `<tr><th scope="row" title="${esc(classes[i])}">${ABBR[classes[i]] || classes[i]}</th>${row.map((v, j) => {
    const size = v ? Math.round(6 + 26 * Math.sqrt(v / max)) : 0;
    const cls = i === j ? 'diag' : v ? 'err' : '';
    return `<td class="${cls}" title="${esc(classes[i])} → ${esc(classes[j])}: ${v}">${v ? `<span class="cm-dot" style="width:${size}px;height:${size}px"></span>` : ''}<span class="sr-only">${v}</span></td>`;
  }).join('')}</tr>`).join('');
  $('#cm').innerHTML = `<caption class="sr-only">Confusion matrix, actual class by row, predicted class by column</caption>${head}<tbody>${body}</tbody>`;
}

function applyCard(card) {
  const m = card.metrics.stacked, man = card.manifest, rt = card.routing_oof, e2e = man.end_to_end_ms || {};
  $('#stat-acc').textContent = fmt(100 * m.accuracy, 2) + '%';
  $('#stat-auto').textContent = fmt(100 * rt.coverage_auto_accepted) + '%';
  if (e2e.predict_auto) $('#stat-lat').textContent = Math.round(e2e.predict_auto.p50_ms) + ' ms';
  $('#m-acc').textContent = fmt(100 * m.accuracy, 2) + '%';
  if (m.ci95) $('#m-acc-sub').textContent = `95% CI ${fmt(100 * m.ci95.accuracy[0], 2)}–${fmt(100 * m.ci95.accuracy[1], 2)}% · n = ${m.n.toLocaleString()}`;
  $('#m-f1').textContent = m.macro.f1.toFixed(3);
  $('#m-f1-sub').textContent = `Balanced across ${card.classes.length} classes · MCC ${m.mcc.toFixed(3)}`;
  $('#m-ece').textContent = m.ece.toFixed(4);
  $('#m-auto').textContent = fmt(100 * rt.coverage_auto_accepted) + '%';
  $('#m-auto-sub').textContent = `at ${fmt(100 * rt.accuracy_auto_accepted, 2)}% accuracy · the rest go to review`;
  if (rt.fast_path) { $('#fast-thr').textContent = rt.fast_path.threshold.toFixed(3); $('#fast-cov').textContent = fmt(100 * rt.fast_path.coverage) + '%'; }
  renderCM(card.classes, m.confusion_matrix);
  if (e2e.predict_auto && e2e.predict_full) {
    const L = { a50: e2e.predict_auto.p50_ms, a95: e2e.predict_auto.p95_ms, f50: e2e.predict_full.p50_ms, f95: e2e.predict_full.p95_ms };
    const top = Math.max(...Object.values(L));
    Object.entries(L).forEach(([k, v]) => { $('#lat-' + k).style.width = (100 * v / top).toFixed(1) + '%'; $('#lat-' + k + '-t').textContent = Math.round(v) + ' ms'; });
  }
}

/* ── boot ──────────────────────────────────────────────────────── */
async function boot() {
  renderIdle(); setSource('sample');
  let card = null;
  try {
    const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), 2500);
    const ping = await fetch(API + '/stats', { signal: ctl.signal }); clearTimeout(t);
    if (ping.ok) { log('Server found · loading model…'); card = await api('/model/card'); }
  } catch (_) { /* no server: offline demo */ }
  if (card) {
    app.live = true; app.card = card;
    try { applyCard(card); } catch (err) { console.error(err); }
    try {
      app.samples = await api('/samples');
      const present = CLASSES.filter(c => app.samples.some(s => s.label === c));
      buildChips(present);
      setStatus('ready', 'MODEL READY');
      log(`Model ready · ${man(card)} · ${app.samples.length} validation wafers`, 'ok');
    } catch (err) { log(err.message, 'err'); setStatus('demo', 'NO SAMPLES'); }
  } else {
    buildChips(Object.keys(DEMO));
    renderCM(CLASSES, DEMO_CM);
    setStatus('demo', 'OFFLINE DEMO');
    log('Server not reachable · showing three recorded results. Upload needs the server.');
  }
}
const man = card => `${card.manifest.backbone} × ${card.manifest.members} + ${card.base_models.filter(b => b !== card.manifest.backbone).join(' + ')}`;

boot();
