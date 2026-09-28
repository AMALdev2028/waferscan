/* Streamlit bridge. On Streamlit Community Cloud there is no FastAPI server, so when this page runs
   inside a Streamlit component iframe, every fetch('/api/v1/...') is sent to Python instead
   (streamlit_app.py answers it with the same FastAPI app). Outside Streamlit this file does nothing. */
(function () {
  'use strict';
  if (window.parent === window || !/[?&]streamlitUrl=/.test(location.search)) return;

  const post = (type, extra) => window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type }, extra), '*');
  const pending = new Map();          // id -> resolve
  const queue = [];                   // one request at a time: each answer needs a Streamlit rerun
  let busy = false, seq = 0;

  function pump() {
    if (busy || !queue.length) return;
    busy = true;
    const job = queue.shift();
    pending.set(job.req.id, job.resolve);
    post('streamlit:setComponentValue', { value: job.req, dataType: 'json' });
  }

  window.addEventListener('message', e => {
    const d = e.data;
    if (!d || d.type !== 'streamlit:render') return;
    const r = d.args && d.args.response;
    if (r && pending.has(r.id)) {
      const resolve = pending.get(r.id); pending.delete(r.id);
      busy = false;
      resolve(new Response(r.body, { status: r.status, headers: { 'Content-Type': 'application/json' } }));
      pump();
    }
  });

  const toB64 = buf => { let s = ''; const b = new Uint8Array(buf); for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000)); return btoa(s); };

  const realFetch = window.fetch.bind(window);
  window.fetch = async function (url, opts = {}) {
    const u = String(url);
    if (!u.startsWith('/api/v1')) return realFetch(url, opts);
    const req = { id: Date.now() + '-' + (++seq), method: (opts.method || 'GET').toUpperCase(), path: u };
    if (opts.body instanceof FormData) {
      req.form = {};
      for (const [k, v] of opts.body.entries()) {
        if (v instanceof Blob) req.file = { field: k, name: v.name || 'upload', type: v.type || 'application/octet-stream', b64: toB64(await v.arrayBuffer()) };
        else req.form[k] = v;
      }
    } else if (opts.body) req.json = String(opts.body);
    return new Promise((resolve, reject) => {
      queue.push({ req, resolve });
      if (opts.signal) opts.signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      pump();
    });
  };

  // Fill the browser window: the page scrolls inside the frame, so the sticky nav still works.
  const fit = () => { let h = 900; try { h = window.parent.innerHeight; } catch (_) {} post('streamlit:setFrameHeight', { height: h }); };
  post('streamlit:componentReady', { apiVersion: 1 });
  fit();
  try { window.parent.addEventListener('resize', fit); } catch (_) {}
})();
