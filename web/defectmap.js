/* WaferScan defect map: one <canvas>, every die drawn per frame, grouped by colour.
   States: idle (morphs through defect shapes), scanning (sweep), result (inspectable), error.
   window.DefectMap.create(canvas, {onSelect, onZoom, onShape}) -> controller */
(function () {
  'use strict';

  const hash = (r, c, k) => { const n = Math.sin(r * 127.1 + c * 311.7 + k * 74.7) * 43758.5453; return n - Math.floor(n); };
  const SHAPES = [
    ['Center', (x, y) => Math.hypot(x, y) < 0.32],
    ['Donut', (x, y) => { const d = Math.hypot(x, y); return d > 0.38 && d < 0.62; }],
    ['Edge-Ring', (x, y) => Math.hypot(x, y) > 0.84],
    ['Edge-Loc', (x, y) => { const a = Math.atan2(y, x); return Math.hypot(x, y) > 0.66 && a > 0.2 && a < 1.3; }],
    ['Loc', (x, y) => Math.hypot(x + 0.35, y + 0.3) < 0.24],
    ['Scratch', (x, y) => Math.abs(y - 0.9 * x + 0.1) < 0.06 && x > -0.7 && x < 0.6],
    ['Random', (x, y, r, c) => hash(r, c, 7) < 0.22],
    ['Near-full', (x, y) => Math.hypot(x, y) < 0.9],
  ];
  const TICK_MS = 420, TICKS_PER_SHAPE = 5, MIN_ZOOM = 1, MAX_ZOOM = 8;

  function disk(n) {             // a generic round wafer for the idle animation
    const c = (n - 1) / 2, m = [];
    for (let r = 0; r < n; r++) { m.push([]); for (let k = 0; k < n; k++) m[r].push(Math.hypot(r - c, k - c) <= n / 2 - 0.3 ? 1 : 0); }
    return m;
  }

  function create(canvas, opts = {}) {
    const ctx = canvas.getContext('2d');
    const reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    const S = {
      map: disk(51), pattern: new Set(), cam: null, camMin: 0, camMax: 1,
      mode: 'idle', zoom: 1, panX: 0, panY: 0, sel: null,
      layers: { pattern: true, fail: true, heat: false },
      tick: 0, sweep: -1, reveal: 1, size: 0,
    };
    let C = {}, onScreen = true, raf = 0;

    const readColors = () => {
      const cs = getComputedStyle(canvas), v = n => cs.getPropertyValue(n).trim();
      C = { bg: v('--surface'), line: v('--line'), pass: v('--die-pass'), fail: v('--die-fail'), ink: v('--ink'), signal: v('--signal') };
    };

    function geom() {
      const rows = S.map.length, cols = S.map[0].length, n = Math.max(rows, cols);
      const cell = (S.size * 0.9 / n) * S.zoom;
      return { rows, cols, cell,
        ox: S.size / 2 - cols * cell / 2 + S.panX,
        oy: S.size / 2 - rows * cell / 2 + S.panY };
    }

    function colourOf(v, r, c, g, shape) {
      if (!v) return null;
      if (S.mode === 'idle') {
        if (reduce) return C.pass;
        const h = hash(r, c, S.tick);
        const x = (c - g.cols / 2) / (g.cols / 2), y = (r - g.rows / 2) / (g.rows / 2);
        return shape[1](x, y, r, c) ? (h < 0.12 ? C.fail : C.ink) : (h < 0.05 ? C.fail : C.pass);
      }
      if (S.mode === 'error') return C.line;
      if (S.mode === 'scanning') return !reduce && hash(r, c, S.tick) < 0.08 ? C.ink : (v === 1 ? C.pass : C.fail);
      if (v === 1) return C.pass;
      const inPattern = S.pattern.has(r * 4096 + c);
      if (inPattern && S.layers.pattern && r <= S.reveal * g.rows) return C.signal;
      return S.layers.fail ? C.fail : C.pass;
    }

    function draw() {
      if (!S.size) return;
      const g = geom(), rad = g.cell * 0.38, shape = SHAPES[Math.floor(S.tick / TICKS_PER_SHAPE) % SHAPES.length];
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, S.size, S.size);
      // wafer outline
      ctx.beginPath();
      ctx.arc(g.ox + g.cols * g.cell / 2, g.oy + g.rows * g.cell / 2, Math.max(g.rows, g.cols) * g.cell / 2 + g.cell * 0.6, 0, Math.PI * 2);
      ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.stroke();
      // dies, one path per colour (a few fills instead of 2,601)
      const paths = new Map();
      ctx.globalAlpha = S.mode === 'scanning' ? 0.72 : 1;
      for (let r = 0; r < g.rows; r++) {
        const y = g.oy + (r + 0.5) * g.cell;
        if (y < -g.cell || y > S.size + g.cell) continue;
        for (let c = 0; c < g.cols; c++) {
          const x = g.ox + (c + 0.5) * g.cell;
          if (x < -g.cell || x > S.size + g.cell) continue;
          const col = colourOf(S.map[r][c], r, c, g, shape);
          if (!col) continue;
          let p = paths.get(col); if (!p) { p = new Path2D(); paths.set(col, p); }
          p.moveTo(x + rad, y); p.arc(x, y, rad, 0, Math.PI * 2);
        }
      }
      paths.forEach((p, col) => { ctx.fillStyle = col; ctx.fill(p); });
      ctx.globalAlpha = 1;
      // heatmap: ring the dies in the top 20% of the CNN's attention
      if (S.mode === 'result' && S.layers.heat && S.cam) {
        ctx.strokeStyle = C.ink; ctx.lineWidth = Math.max(1, g.cell * 0.09);
        const span = S.camMax - S.camMin || 1;
        for (let r = 0; r < g.rows; r++) for (let c = 0; c < g.cols; c++) {
          const a = S.cam[r] && S.cam[r][c];
          if (a == null || !S.map[r][c]) continue;
          const t = (a - S.camMin) / span;
          if (t < 0.8) continue;
          ctx.globalAlpha = Math.min(1, (t - 0.8) * 5);
          ctx.beginPath(); ctx.arc(g.ox + (c + 0.5) * g.cell, g.oy + (r + 0.5) * g.cell, g.cell * 0.5, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      // selected die: paper gap + ink ring
      if (S.sel && S.mode === 'result') {
        const x = g.ox + (S.sel[1] + 0.5) * g.cell, y = g.oy + (S.sel[0] + 0.5) * g.cell;
        ctx.lineWidth = 2;
        ctx.strokeStyle = C.bg; ctx.beginPath(); ctx.arc(x, y, rad + 2, 0, Math.PI * 2); ctx.stroke();
        ctx.strokeStyle = C.ink; ctx.beginPath(); ctx.arc(x, y, rad + 4, 0, Math.PI * 2); ctx.stroke();
      }
      // scan sweep
      if (S.mode === 'scanning' && S.sweep >= 0) {
        ctx.fillStyle = C.signal;
        ctx.fillRect(0, g.oy + S.sweep * g.rows * g.cell - 1, S.size, 2);
      }
    }

    function resize() {
      const w = canvas.clientWidth; if (!w) return;
      const dpr = window.devicePixelRatio || 1;
      S.size = w; canvas.width = Math.round(w * dpr); canvas.height = Math.round(w * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    // ── animation ──
    setInterval(() => {                       // idle morph + scan twinkle
      if (reduce || document.hidden || !onScreen) return;
      if (S.mode !== 'idle' && S.mode !== 'scanning') return;
      S.tick++;
      if (S.mode === 'idle' && S.tick % TICKS_PER_SHAPE === 0 && opts.onShape) opts.onShape(SHAPES[(S.tick / TICKS_PER_SHAPE) % SHAPES.length][0]);
      draw();
    }, TICK_MS);
    function animate(fn, ms, done) {          // rAF helper, t runs 0 -> 1
      cancelAnimationFrame(raf);
      if (reduce) { fn(1); draw(); if (done) done(); return; }
      const t0 = performance.now();
      const step = now => { const t = Math.min(1, (now - t0) / ms); fn(t); draw(); if (t < 1) raf = requestAnimationFrame(step); else if (done) done(); };
      raf = requestAnimationFrame(step);
    }
    function sweepLoop() {
      if (S.mode !== 'scanning') return;
      animate(t => { S.sweep = t; }, 1400, sweepLoop);
    }

    // ── zoom / pan / select ──
    function clampPan() {
      const g = geom(), lim = Math.max(0, (Math.max(g.rows, g.cols) * g.cell - S.size) / 2 + g.cell * 2);
      S.panX = Math.max(-lim, Math.min(lim, S.panX)); S.panY = Math.max(-lim, Math.min(lim, S.panY));
    }
    function zoomAt(z, px, py) {
      z = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
      const g = geom(), gx = (px - g.ox) / g.cell, gy = (py - g.oy) / g.cell;
      S.zoom = z;
      if (z === 1) { S.panX = 0; S.panY = 0; }
      else {
        const n = geom();   // with the new zoom and old pan
        S.panX += px - (n.ox + gx * n.cell); S.panY += py - (n.oy + gy * n.cell);
        clampPan();
      }
      draw(); if (opts.onZoom) opts.onZoom(S.zoom);
    }
    function anchor() {
      if (!S.sel) return [S.size / 2, S.size / 2];
      const g = geom(); return [g.ox + (S.sel[1] + 0.5) * g.cell, g.oy + (S.sel[0] + 0.5) * g.cell];
    }
    function keepInView() {
      if (!S.sel || S.zoom === 1) return;
      const g = geom(), x = g.ox + (S.sel[1] + 0.5) * g.cell, y = g.oy + (S.sel[0] + 0.5) * g.cell, m = g.cell * 2;
      if (x < m) S.panX += m - x; if (x > S.size - m) S.panX -= x - (S.size - m);
      if (y < m) S.panY += m - y; if (y > S.size - m) S.panY -= y - (S.size - m);
      clampPan();
    }
    function select(rc) {
      S.sel = rc; keepInView(); draw();
      if (opts.onSelect) opts.onSelect(rc);
    }
    function dieAt(px, py) {
      const g = geom(), c = Math.floor((px - g.ox) / g.cell), r = Math.floor((py - g.oy) / g.cell);
      return r >= 0 && c >= 0 && r < g.rows && c < g.cols && S.map[r][c] ? [r, c] : null;
    }

    // pointer: click selects, drag pans (when zoomed)
    let down = null;
    canvas.addEventListener('pointerdown', e => {
      down = { x: e.clientX, y: e.clientY, px: S.panX, py: S.panY, moved: false };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', e => {
      if (!down) {
        if (S.mode === 'result') { const b = canvas.getBoundingClientRect(); canvas.classList.toggle('can-pick', !!dieAt(e.clientX - b.left, e.clientY - b.top)); }
        return;
      }
      const dx = e.clientX - down.x, dy = e.clientY - down.y;
      if (Math.hypot(dx, dy) > 4) down.moved = true;
      if (down.moved && S.zoom > 1) { canvas.classList.add('is-panning'); S.panX = down.px + dx; S.panY = down.py + dy; clampPan(); draw(); }
    });
    canvas.addEventListener('pointerup', e => {
      canvas.classList.remove('is-panning');
      if (down && !down.moved && S.mode === 'result') {
        const b = canvas.getBoundingClientRect(), rc = dieAt(e.clientX - b.left, e.clientY - b.top);
        if (rc) select(rc);
      }
      down = null;
    });
    canvas.addEventListener('pointercancel', () => { down = null; canvas.classList.remove('is-panning'); });
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const b = canvas.getBoundingClientRect();
      zoomAt(S.zoom * (e.deltaY < 0 ? 1.25 : 0.8), e.clientX - b.left, e.clientY - b.top);
    }, { passive: false });
    canvas.addEventListener('dblclick', () => zoomAt(1, 0, 0));
    canvas.addEventListener('keydown', e => {
      const k = e.key;
      const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
      if (k === '+' || k === '=') zoomAt(S.zoom * 2, ...anchor());
      else if (k === '-' || k === '_') zoomAt(S.zoom / 2, ...anchor());
      else if (k === '0') zoomAt(1, 0, 0);
      else if (k === 'Escape' && S.sel) select(null);
      else if (moves[k] && S.mode === 'result') {
        const g = geom(), d = moves[k];
        if (!S.sel) {                        // first arrow press lands on the centre die
          const r0 = Math.floor(g.rows / 2), c0 = Math.floor(g.cols / 2);
          if (S.map[r0][c0]) select([r0, c0]);
          e.preventDefault(); return;
        }
        let p = S.sel.slice();
        for (let i = 0; i < Math.max(g.rows, g.cols); i++) {   // skip off-wafer gaps
          p = [p[0] + d[0], p[1] + d[1]];
          if (p[0] < 0 || p[1] < 0 || p[0] >= g.rows || p[1] >= g.cols) break;
          if (S.map[p[0]][p[1]]) { select(p); break; }
        }
      } else return;
      e.preventDefault();
    });

    // theme changes and size changes redraw
    new MutationObserver(() => { readColors(); draw(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas); else window.addEventListener('resize', resize);
    if (window.IntersectionObserver) new IntersectionObserver(es => { onScreen = es[0].isIntersecting; }).observe(canvas);
    readColors(); resize();

    function setMode(mode) { cancelAnimationFrame(raf); S.mode = mode; S.sweep = -1; S.sel = null; S.zoom = 1; S.panX = 0; S.panY = 0; if (opts.onSelect) opts.onSelect(null); if (opts.onZoom) opts.onZoom(1); }

    return {
      state: S,
      shapeName: () => SHAPES[Math.floor(S.tick / TICKS_PER_SHAPE) % SHAPES.length][0],
      idle() { setMode('idle'); S.map = disk(51); draw(); },
      scanning(map) { setMode('scanning'); if (map) S.map = map; draw(); sweepLoop(); },
      error() { setMode('error'); draw(); },
      result(map, patternKeys, cam) {
        setMode('result');
        S.map = map; S.pattern = new Set(patternKeys.map(([r, c]) => r * 4096 + c));
        S.cam = cam || null;
        if (S.cam) { const f = S.cam.flat(); S.camMin = Math.min(...f); S.camMax = Math.max(...f); }
        S.reveal = 0; animate(t => { S.reveal = t; }, 480);
      },
      select,
      zoomIn() { zoomAt(S.zoom * 2, ...anchor()); },
      zoomOut() { zoomAt(S.zoom / 2, ...anchor()); },
      fit() { zoomAt(1, 0, 0); },
      setLayer(name, on) { S.layers[name] = on; draw(); },
      redraw() { readColors(); draw(); },
    };
  }

  window.DefectMap = { create };
})();
