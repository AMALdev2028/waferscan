/* WaferScan landing page logic.
 *
 * DEMO mode (page opened without the API, e.g. from disk): exactly the
 * original simulated behaviour.
 * LIVE mode (served by the WaferScan API): real WM-811K validation wafers, real
 * inference, real out-of-fold metrics. Only data sources change - every
 * colour, font, stroke and layout is the original design.
 */
const API = '/api/v1';
const LIVE = { on: false, card: null, res: null, wafer: null, pattern: new Set(), order: [], idx: -1, lat: [] };

async function api(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) throw new Error(r.status + ' ' + (await r.text()).slice(0, 200));
  return r.json();
}
const fmt = (n, d = 1) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

/* ── HERO CANVAS: animated particle grid ── */
(function(){
  const c=document.getElementById('hero-canvas');
  const ctx=c.getContext('2d');
  let W,H,pts,raf;
  function resize(){W=c.width=c.offsetWidth;H=c.height=c.offsetHeight;init()}
  function init(){
    pts=Array.from({length:60},()=>({
      x:Math.random()*W,y:Math.random()*H,
      vx:(Math.random()-.5)*.4,vy:(Math.random()-.5)*.4
    }));
  }
  function draw(){
    ctx.clearRect(0,0,W,H);
    pts.forEach(p=>{
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<0||p.x>W)p.vx*=-1;
      if(p.y<0||p.y>H)p.vy*=-1;
      ctx.beginPath();ctx.arc(p.x,p.y,1.2,0,Math.PI*2);
      ctx.fillStyle='#555';ctx.fill();
    });
    pts.forEach((a,i)=>{
      pts.slice(i+1).forEach(b=>{
        const d=Math.hypot(a.x-b.x,a.y-b.y);
        if(d<120){
          ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
          ctx.strokeStyle=`rgba(100,100,100,${(1-d/120)*.4})`;
          ctx.lineWidth=.5;ctx.stroke();
        }
      });
    });
    raf=requestAnimationFrame(draw);
  }
  new ResizeObserver(resize).observe(c);
  resize();draw();
})();

/* ── COUNTER ANIMATION ── */
function animateCounter(el){
  const target=parseFloat(el.dataset.target);
  const suffix=el.dataset.suffix||'';
  const isFloat=String(target).includes('.');
  const dur=1800;const start=performance.now();
  function tick(now){
    const t=Math.min((now-start)/dur,1);
    const ease=1-Math.pow(1-t,4);
    const val=target*ease;
    el.textContent=(isFloat?val.toFixed(1):Math.round(val)).toLocaleString()+suffix;
    if(t<1)requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
function startCounters(){
  const statsObs=new IntersectionObserver(entries=>{
    entries.forEach(e=>{if(e.isIntersecting){
      e.target.querySelectorAll('[data-target]').forEach(animateCounter);
      statsObs.unobserve(e.target);
    }});
  },{threshold:.4});
  statsObs.observe(document.querySelector('.hero-stats'));
}

/* ── SCROLL REVEAL ── */
const revealObs=new IntersectionObserver(entries=>{
  entries.forEach(e=>{if(e.isIntersecting)e.target.classList.add('in')});
},{threshold:.1});
document.querySelectorAll('.reveal').forEach(el=>revealObs.observe(el));

/* ── WAFER CANVAS ── */
const wc=document.getElementById('wafer-canvas');
const wctx=wc.getContext('2d');
const dpr=window.devicePixelRatio||1;
const SIZE=420;
wc.width=SIZE*dpr;wc.height=SIZE*dpr;
wc.style.width=SIZE+'px';wc.style.height=SIZE+'px';
wctx.scale(dpr,dpr);

const LAYERS={overlay:true,heat:false,grid:true};
let zoom=1,panX=0,panY=0,dragging=false,dragStart={x:0,y:0},panStart={x:0,y:0},dragDist=0;

// Demo defect die positions [col,row,class,conf] (demo mode only)
const defects=[
  [3,4,'scratch',.941],[4,4,'scratch',.888],[3,5,'scratch',.912],
  [7,3,'particle',.783],[7,4,'particle',.691],[8,3,'particle',.724],
  [11,9,'edge',.61],[12,9,'edge',.543],
  [5,8,'contam',.447]
];

// the original per-class fill patterns, built once
const PATTERNS=(()=>{
  const mk=draw=>{const p=document.createElement('canvas');p.width=6;p.height=6;const x=p.getContext('2d');
    x.fillStyle='#000';x.fillRect(0,0,6,6);draw(x);return wctx.createPattern(p,'repeat');};
  return {
    scratch:mk(x=>{x.strokeStyle='rgba(255,255,255,0.7)';x.lineWidth=1;x.beginPath();x.moveTo(0,6);x.lineTo(6,0);x.stroke();}),
    particle:mk(x=>{x.fillStyle='rgba(200,200,200,0.6)';x.beginPath();x.arc(3,3,1.2,0,Math.PI*2);x.fill();}),
    edge:mk(x=>{x.fillStyle='rgba(120,120,120,0.5)';x.beginPath();x.arc(3,3,.9,0,Math.PI*2);x.fill();}),
    contam:mk(()=>{})
  };
})();
const COLORS={scratch:'#fff',particle:'#ccc',edge:'#888',contam:'#555'};
const WIDTHS={scratch:1.8,particle:1.2,edge:.8,contam:.6};

function drawDie(x,y,s,cls){
  wctx.fillStyle=PATTERNS[cls];wctx.fillRect(x+1,y+1,s-2,s-2);
  wctx.strokeStyle=COLORS[cls];wctx.lineWidth=Math.min(WIDTHS[cls],s*.25);
  if(cls==='contam')wctx.setLineDash([3,3]);else wctx.setLineDash([]);
  wctx.strokeRect(x+1,y+1,s-2,s-2);
  wctx.setLineDash([]);
}

// live wafer geometry: the die grid spans the wafer diameter
function liveGeom(){
  const R=SIZE/2-12, rows=LIVE.wafer.length, cols=LIVE.wafer[0].length;
  const d=(2*R-16)/Math.max(rows,cols);
  return {d,rows,cols,gx:SIZE/2-cols*d/2,gy:SIZE/2-rows*d/2};
}

function drawWafer(){
  const cx=SIZE/2,cy=SIZE/2,R=SIZE/2-12;
  wctx.save();
  wctx.translate(cx+panX,cy+panY);wctx.scale(zoom,zoom);wctx.translate(-cx,-cy);

  // Background
  wctx.fillStyle='#000';wctx.fillRect(0,0,SIZE,SIZE);

  // Clip to circle
  wctx.save();
  wctx.beginPath();wctx.arc(cx,cy,R,0,Math.PI*2);wctx.clip();

  if(LIVE.wafer){drawLive();}
  else{
    // Heatmap layer
    if(LAYERS.heat){
      const grad=wctx.createRadialGradient(cx-30,cy-40,10,cx,cy,R);
      grad.addColorStop(0,'rgba(255,255,255,0.22)');
      grad.addColorStop(.3,'rgba(150,150,150,0.12)');
      grad.addColorStop(.7,'rgba(60,60,60,0.08)');
      grad.addColorStop(1,'rgba(0,0,0,0)');
      wctx.fillStyle=grad;wctx.fillRect(0,0,SIZE,SIZE);
      // contour lines
      [.85,.65,.45,.25].forEach((r,i)=>{
        wctx.beginPath();
        wctx.arc(cx-30*(1-r),cy-40*(1-r),R*r*.55,0,Math.PI*2);
        wctx.strokeStyle=`rgba(255,255,255,${.05+i*.02})`;
        wctx.lineWidth=.8;wctx.stroke();
      });
    }

    // Die grid
    const DIE=28,COLS=15,ROWS=15;
    const gx=cx-COLS*DIE/2,gy=cy-ROWS*DIE/2;
    if(LAYERS.grid){
      for(let r=0;r<ROWS;r++){for(let c=0;c<COLS;c++){
        const x=gx+c*DIE,y=gy+r*DIE;
        const dist=Math.hypot(x+DIE/2-cx,y+DIE/2-cy);
        if(dist>R-14)continue;
        wctx.fillStyle='#111';wctx.fillRect(x+1,y+1,DIE-2,DIE-2);
        wctx.strokeStyle='#1e1e1e';wctx.lineWidth=.5;
        wctx.strokeRect(x+1,y+1,DIE-2,DIE-2);
      }}
    }

    // Defect overlay
    if(LAYERS.overlay){
      defects.forEach(([c,r,cls])=>{
        const x=gx+c*DIE,y=gy+r*DIE;
        const dist=Math.hypot(x+DIE/2-cx,y+DIE/2-cy);
        if(dist>R-14)return;
        drawDie(x,y,DIE,cls);
      });
    }
  }

  // Wafer outline + notch
  wctx.restore();
  wctx.strokeStyle='#2a2a2a';wctx.lineWidth=1.5;
  wctx.beginPath();wctx.arc(cx,cy,R,0,Math.PI*2);wctx.stroke();
  wctx.fillStyle='#2a2a2a';
  wctx.beginPath();wctx.moveTo(cx-6,cy+R-2);wctx.lineTo(cx+6,cy+R-2);wctx.lineTo(cx,cy+R+6);wctx.fill();

  // Compass
  wctx.font=`bold 9px 'JetBrains Mono',monospace`;wctx.fillStyle='#444';wctx.textAlign='center';
  wctx.fillText('N',cx,cy-R+14);wctx.fillText('S',cx,cy+R-4);

  wctx.restore();
}

// real die map: pass dies = original grid style; failed dies inside the
// classified pattern = the strongest (scratch) style; isolated fails = the
// weakest (contam) style; heatmap = where the CNN looked (CAM), in grayscale
function drawLive(){
  const g=liveGeom(), cam=LIVE.res&&LIVE.res.cam;
  for(let r=0;r<g.rows;r++){for(let c=0;c<g.cols;c++){
    const v=LIVE.wafer[r][c]; if(!v)continue;
    const x=g.gx+c*g.d,y=g.gy+r*g.d;
    if(LAYERS.grid){
      wctx.fillStyle='#111';wctx.fillRect(x+.5,y+.5,g.d-1,g.d-1);
      wctx.strokeStyle='#1e1e1e';wctx.lineWidth=.5;wctx.strokeRect(x+.5,y+.5,g.d-1,g.d-1);
    }
    if(v===2&&LAYERS.overlay)drawDie(x,y,g.d,LIVE.pattern.has(r+','+c)?'scratch':'contam');
    if(LAYERS.heat&&cam&&cam[r][c]>0.05){
      wctx.fillStyle=`rgba(255,255,255,${(0.35*cam[r][c]).toFixed(3)})`;wctx.fillRect(x,y,g.d,g.d);
    }
  }}
}

function toggleLayer(l){
  LAYERS[l]=!LAYERS[l];
  const btn=document.getElementById('btn-'+l);
  if(btn){btn.classList.toggle('active',LAYERS[l]);}
  drawWafer();
}
function zoomStep(factor){zoom=Math.min(Math.max(zoom*factor,.5),8);drawWafer();}
function resetZoom(){zoom=1;panX=0;panY=0;drawWafer();}

// Click detection
wc.addEventListener('click',e=>{
  if(dragDist>4)return;
  const rect=wc.getBoundingClientRect();
  const mx=(e.clientX-rect.left)*(SIZE/rect.width),my=(e.clientY-rect.top)*(SIZE/rect.height);
  const cx=SIZE/2,cy=SIZE/2;
  const R=SIZE/2-12;
  // convert click to wafer coordinates
  const wx=(mx-cx-panX)/zoom+cx;
  const wy=(my-cy-panY)/zoom+cy;
  if(LIVE.wafer){
    const g=liveGeom();
    inspectDie(Math.floor((wy-g.gy)/g.d),Math.floor((wx-g.gx)/g.d));
    return;
  }
  const DIE=28,COLS=15,ROWS=15;
  const gx=cx-COLS*DIE/2,gy=cy-ROWS*DIE/2;
  const col=Math.floor((wx-gx)/DIE);
  const row=Math.floor((wy-gy)/DIE);
  const def=defects.find(([c,r])=>c===col&&r===row);
  if(def){
    const [,,cls,conf]=def;
    simulateInference(cls,conf,col,row);
  } else {
    const dist=Math.hypot(wx-cx,wy-cy);
    if(dist<R)simulateInference(null,null,col,row);
  }
});

// Drag/pan
wc.addEventListener('mousedown',e=>{dragging=true;dragDist=0;dragStart={x:e.clientX,y:e.clientY};panStart={x:panX,y:panY}});
window.addEventListener('mousemove',e=>{if(!dragging)return;panX=panStart.x+(e.clientX-dragStart.x);panY=panStart.y+(e.clientY-dragStart.y);dragDist=Math.hypot(e.clientX-dragStart.x,e.clientY-dragStart.y);drawWafer()});
window.addEventListener('mouseup',()=>dragging=false);
wc.addEventListener('wheel',e=>{e.preventDefault();zoomStep(e.deltaY<0?1.1:.9)},{passive:false});

function logLine(html){
  const log=document.getElementById('inf-log');
  const line=document.createElement('span');
  line.className='log-line';line.innerHTML=html;
  log.appendChild(line);log.scrollTop=log.scrollHeight;
}
function now(){return new Date().toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit',second:'2-digit'})}
const esc=s=>String(s).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const TS=()=>`<span class="ts">[${now()}]</span> `;

function simulateInference(cls,conf,col,row){
  const ts=now();
  logLine(`<span class="ts">[${ts}]</span> Inferring die (${col},${row})…`);
  setTimeout(()=>{
    if(cls){
      const pct=(conf*100).toFixed(1);
      logLine(`<span class="ts">[${ts}]</span> <span class="ok">CLASS: ${cls.toUpperCase()} · ${pct}%</span>`);
      // jitter the confidence slightly
      const el=document.getElementById('conf-'+cls);
      const jitter=(Math.random()-.5)*3;
      const newConf=Math.min(99.9,Math.max(40,conf*100+jitter));
      if(el)el.textContent=newConf.toFixed(1)+'%';
      // select that defect row
      document.querySelectorAll('.defect-item').forEach(d=>d.classList.remove('selected'));
      const item=document.querySelector(`[data-class="${cls}"]`);
      if(item)item.classList.add('selected');
    } else {
      logLine(`<span class="ts">[${ts}]</span> No defect detected. Die clean.`);
    }
  },Math.random()*400+200);
}

// LIVE: inspect one die of the real wafer
function inspectDie(row,col){
  const g=liveGeom();
  if(row<0||col<0||row>=g.rows||col>=g.cols||!LIVE.wafer[row][col]){logLine(TS()+'Off-wafer position.');return;}
  const q=LIVE.res.quantification, u=q.units;
  const on=[];LIVE.wafer.forEach((r,i)=>r.forEach((v,j)=>{if(v)on.push([i,j]);}));
  const rs=on.map(p=>p[0]),cs=on.map(p=>p[1]);
  const cy=(Math.min(...rs)+Math.max(...rs))/2,cx=(Math.min(...cs)+Math.max(...cs))/2;
  const x=((col-cx)*u.die_w_mm).toFixed(1),y=((cy-row)*u.die_h_mm).toFixed(1);
  const fail=LIVE.wafer[row][col]===2, inPat=LIVE.pattern.has(row+','+col);
  const att=LIVE.res.cam[row][col];
  const state=fail?(inPat?`<span class="ok">FAIL · IN ${esc(LIVE.res.prediction.class.toUpperCase())} PATTERN</span>`
                          :'<span class="err">FAIL · ISOLATED</span>'):'PASS';
  logLine(`${TS()}DIE (${col},${row}) · ${state} · x=${x} y=${y} mm · CNN attention ${att.toFixed(2)}`);
  document.querySelectorAll('.defect-item').forEach((d,i)=>d.classList.toggle('selected',i===0));
}

function selectDefect(el,cls){
  document.querySelectorAll('.defect-item').forEach(d=>d.classList.remove('selected'));
  el.classList.add('selected');
  if(LIVE.res){
    const p=LIVE.res.prediction.probabilities[cls];
    if(p!==undefined)logLine(`${TS()}${esc(cls.toUpperCase())}: p=${p.toFixed(4)}`
      +(LIVE.res.prediction.alarms.includes(cls)?' · above its Youden alarm threshold':''));
  }
}

function runNewScan(){
  if(LIVE.on){liveScan();return;}
  logLine(`<span class="ts">[${now()}]</span> Loading new scan…`);
  defects.forEach(d=>{
    d[3]=Math.min(.99,Math.max(.3,d[3]+(Math.random()-.5)*.15));
  });
  ['scratch','particle','edge','contam'].forEach(cls=>{
    const def=defects.find(d=>d[2]===cls);
    const el=document.getElementById('conf-'+cls);
    if(el&&def)el.textContent=(def[3]*100).toFixed(1)+'%';
  });
  logLine(`<span class="ts">[${now()}]</span> <span class="ok">Scan complete · 4 classes detected</span>`);
  drawWafer();
}

function exportResults(){
  if(LIVE.res){exportLive();return;}
  const rows=['class,confidence,die_count'];
  [['SCRATCH',document.getElementById('conf-scratch').textContent,'14'],
   ['PARTICLE',document.getElementById('conf-particle').textContent,'6'],
   ['EDGE_CRACK',document.getElementById('conf-edge').textContent,'2'],
   ['CONTAMINATION',document.getElementById('conf-contam').textContent,'1']
  ].forEach(r=>rows.push(r.join(',')));
  logLine(`<span class="ts">[${now()}]</span> Export ready (${rows.length-1} classes)`);
  alert('CSV export:\n\n'+rows.join('\n'));
}

/* ── CONFUSION MATRIX ── */
function renderCM(labels,data){
  const max=Math.max(...data.flat());
  const g=document.getElementById('cm-grid');
  g.innerHTML='';
  g.style.gridTemplateColumns=`56px repeat(${labels.length},1fr)`;
  g.style.gridTemplateRows=`32px repeat(${labels.length},1fr)`;
  // header row
  const blank=document.createElement('div');g.appendChild(blank);
  labels.forEach(l=>{const d=document.createElement('div');d.className='cm-cell cm-head';d.textContent=l;g.appendChild(d)});
  data.forEach((row,ri)=>{
    const ax=document.createElement('div');
    ax.className='cm-cell cm-axis';ax.textContent=labels[ri];g.appendChild(ax);
    row.forEach((v,ci)=>{
      const luma=Math.round((v/max)*220+10);
      const bg=`rgb(${luma},${luma},${luma})`;
      const fg=luma>110?'#000':'#fff';
      const d=document.createElement('div');
      d.className='cm-cell cm-data';
      d.style.background=bg;d.style.color=fg;
      if(ri===ci)d.style.fontWeight='700';
      d.textContent=v;
      d.title=`Actual:${labels[ri]} Predicted:${labels[ci]} n=${v}`;
      g.appendChild(d);
    });
  });
}

/* ── LATENCY BARS ── */
function renderLatency(vals){
  const container=document.getElementById('lat-bars');
  container.innerHTML='';
  const max=Math.max(...vals);
  vals.forEach((v,i)=>{
    const b=document.createElement('div');
    b.className='lat-bar';
    b.style.height=(v/max*100)+'%';
    b.style.flex='1';
    b.style.background=i===vals.length-1?'#fff':'#333';
    b.style.minHeight='3px';
    b.title=`${v<100?v.toFixed(1):Math.round(v)}ms`;
    b.addEventListener('mouseenter',()=>{b.style.background='#999'});
    b.addEventListener('mouseleave',()=>{b.style.background=i===vals.length-1?'#fff':'#333'});
    container.appendChild(b);
  });
}

/* ════════════════════ LIVE MODE ════════════════════ */
const ABBR={'Center':'CEN','Donut':'DON','Edge-Loc':'E-L','Edge-Ring':'E-R','Loc':'LOC','Near-full':'N-F',
            'Random':'RAN','Scratch':'SCR','None':'NON'};

function setText(sel,text,root=document){const el=root.querySelector(sel);if(el)el.textContent=text;}

function applyModelCard(card){
  const m=card.metrics.stacked, man=card.manifest, lat=man.cpu_latency_batch1, rt=card.routing_oof;
  const e2e=man.end_to_end_ms&&man.end_to_end_ms.predict_auto;   // whole request, what a user waits for
  const n=m.n;
  // hero stats (animated counters read data-target)
  const nums=document.querySelectorAll('.hero-stats .stat-num');
  nums[0].dataset.target=(100*m.accuracy).toFixed(1);
  nums[1].dataset.target=String(Math.max(1,Math.round(e2e?e2e.p50_ms:lat.fast_fp32.p50_ms)));
  nums[2].dataset.target=String(card.classes.length);
  nums[3].dataset.target=String(n);nums[3].dataset.suffix='';
  document.querySelectorAll('.hero-stats .stat-label')[3].textContent='Wafer Maps · Grouped CV';
  // metrics strip
  const cells=document.querySelectorAll('.metrics-grid .metric-cell');
  const ci=m.ci95?` · 95% CI ${fmt(100*m.ci95.accuracy[0])}–${fmt(100*m.ci95.accuracy[1])}%`:'';
  setText('.metric-val',fmt(100*m.accuracy)+'%',cells[0]);
  setText('.metric-sub',`Out-of-fold · n=${n.toLocaleString()}${ci}`,cells[0]);
  setText('.metric-val',m.macro.f1.toFixed(3),cells[1]);
  setText('.metric-sub',`Balanced across ${card.classes.length} classes · MCC ${m.mcc.toFixed(3)}`,cells[1]);
  setText('.metric-val',fmt(e2e?e2e.p95_ms:lat.ensemble_fp32.p95_ms,0)+'ms',cells[2]);
  setText('.metric-sub',e2e?`CPU · full request incl. quantification · ${man.members}-model ensemble`
                            :`CPU ONNX · ${man.members}-model ensemble · batch 1`,cells[2]);
  setText('.metric-val',fmt(100*rt.coverage_auto_accepted,0)+'%',cells[3]);
  setText('.metric-label','Auto-Accepted',cells[3]);
  setText('.metric-sub',`at ${fmt(100*rt.accuracy_auto_accepted)}% accuracy · rest → human review`,cells[3]);
  // confusion matrix + caption
  renderCM(card.classes.map(c=>ABBR[c]||c.slice(0,3).toUpperCase()),m.confusion_matrix);
  const cap=document.getElementById('cm-grid').nextElementSibling;
  if(cap)cap.textContent=`Predicted → · Actual ↓ · n=${n.toLocaleString()} wafer maps · out-of-fold`;
  // precision/recall curve
  const cur=card.curves;
  if(cur){
    const svg=document.querySelector('.pr-chart svg');
    const path=c=>c.recall.map((r,i)=>`${i?'L':'M'}${(r*260).toFixed(1)},${(160-c.precision[i]*150).toFixed(1)}`).join(' ');
    const paths=svg.querySelectorAll('path');
    const main=path(cur.stacked);
    paths[0].setAttribute('d',main+' L260,160 L0,160Z');
    paths[1].setAttribute('d',main);
    paths[2].setAttribute('d',path(cur.baseline_gbm));
    [...svg.querySelectorAll('text')].filter(t=>t.textContent.startsWith('AUC')).forEach(t=>t.textContent=`AP=${cur.stacked.ap.toFixed(3)}`);
    const leg=document.querySelectorAll('.pr-chart')[0].querySelectorAll('div:last-child span');
    if(leg.length===2){leg[0].textContent='—— stacked ensemble';leg[1].textContent='- - - morphology GBM (baseline)';}
  }
  // latency histogram: benchmark samples until live scans replace them
  if(card.latency_samples){LIVE.lat=card.latency_samples.slice(-64);renderLatency(LIVE.lat);updateP50();}
  // copy that described the simulation
  setText('#demo .section-body','Press NEW SCAN to run real inference on a WM-811K wafer from the validation set. Click any die to inspect it. The heatmap layer shows where the CNN looked (class activation map).');
  const log=document.getElementById('inf-log');log.innerHTML='';
  logLine(`${TS()}Model loaded: ${esc(man.backbone)} × ${man.members} (${esc(card.base_models.join(' + '))})`);
  logLine(`${TS()}<span class="ok">Weights OK · ONNX parity ${man.checks.parity_max_abs_diff_ensemble.toExponential(1)}</span>`);
}

function updateP50(){
  const s=[...LIVE.lat].sort((a,b)=>a-b);
  const p50=s[Math.floor(s.length/2)]||0;
  const spans=document.querySelectorAll('.pr-chart')[1].querySelectorAll('div:last-child span');
  if(spans[1])spans[1].textContent=`P50: ${p50<100?p50.toFixed(1):Math.round(p50).toLocaleString()}ms`;
}

function renderResult(res,truth){
  LIVE.res=res;LIVE.wafer=res.wafer.die_map;
  LIVE.pattern=new Set(res.quantification.failed_dies.filter(d=>d.in_pattern).map(d=>d.row+','+d.col));
  const items=document.querySelectorAll('.defect-item');
  const probs=Object.entries(res.prediction.probabilities);
  const q=res.quantification;
  items.forEach((el,i)=>{
    const [cls,p]=probs[i];
    el.dataset.class=cls;
    el.setAttribute('onclick',`selectDefect(this,'${cls}')`);
    setText('.di-name',cls.toUpperCase(),el);
    setText('.di-conf',fmt(100*p)+'%',el);
    el.querySelector('.di-fill').style.width=Math.max(0.5,100*p).toFixed(1)+'%';
    setText('.di-meta',i===0?`${q.pattern.dies} die · ${fmt(q.pattern.affected_pct)}% affected · ${fmt(q.pattern.area_mm2,0)} mm²`
                             :`p=${p.toFixed(4)}`,el);
    el.classList.toggle('selected',i===0);
  });
  const pr=res.prediction, u=res.uncertainty, v=res.verification, d=res.decision;
  logLine(`${TS()}Inference · ${pr.path} path · ${u.members_used} model${u.members_used>1?'s':''} · ${fmt(res.timing_ms.inference)} ms`);
  logLine(`${TS()}<span class="ok">CLASS: ${esc(pr.class.toUpperCase())} · ${fmt(100*pr.confidence)}%</span>`);
  logLine(`${TS()}Verify: ${v.status}${v.reasons.length?' · '+esc(v.reasons[0]):''}`);
  logLine(`${TS()}Uncertainty: MC-MI ${u.mc_dropout_mutual_info.toFixed(3)} · evidential u ${u.evidential_u.toFixed(3)}`);
  logLine(`${TS()}Affected ${fmt(q.pattern.affected_pct)}% (Wilson ${fmt(q.pattern.affected_pct_wilson95[0])}–${fmt(q.pattern.affected_pct_wilson95[1])}%) · clean ${fmt(q.pattern.clean_pct)}%`);
  logLine(`${TS()}${d.action==='auto_accept'?'<span class="ok">DECISION: AUTO-ACCEPT</span>':'<span class="err">DECISION: HUMAN REVIEW</span>'}${d.note?' · '+esc(d.note):''}`);
  if(truth)logLine(`${TS()}Ground truth: ${esc(truth.toUpperCase())} ${truth===pr.class?'<span class="ok">✓ match</span>':'<span class="err">✗ mismatch</span>'}`);
  LIVE.lat.push(res.timing_ms.total);if(LIVE.lat.length>64)LIVE.lat.shift();
  renderLatency(LIVE.lat);updateP50();
  zoom=1;panX=0;panY=0;drawWafer();
}

async function liveScan(){
  try{
    LIVE.idx=(LIVE.idx+1)%LIVE.order.length;
    const s=LIVE.order[LIVE.idx];
    logLine(`${TS()}Loading validation wafer #${s.id} (${s.rows}×${s.cols} dies)…`);
    const w=await api('/samples/'+s.id);
    const res=await api('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
                                    body:JSON.stringify({wafer_map:w.wafer_map})});
    renderResult(res,w.label);
  }catch(err){logLine(`${TS()}<span class="err">Error: ${esc(err.message)}</span>`);}
}

function exportLive(){
  const r=LIVE.res,q=r.quantification,rows=[];
  rows.push('section,key,value');
  rows.push(`prediction,class,${r.prediction.class}`,`prediction,confidence,${r.prediction.confidence}`,
            `prediction,decision,${r.decision.action}`,`prediction,verification,${r.verification.status}`);
  Object.entries(r.prediction.probabilities).forEach(([c,p])=>rows.push(`probability,${c},${p}`));
  rows.push(`wafer,dies_total,${q.wafer.dies_total}`,`wafer,dies_failed,${q.wafer.dies_failed}`,
            `wafer,fail_pct,${q.wafer.fail_pct}`,`pattern,dies,${q.pattern.dies}`,
            `pattern,affected_pct,${q.pattern.affected_pct}`,`pattern,area_mm2,${q.pattern.area_mm2}`);
  rows.push('','row,col,x_mm,y_mm,in_pattern');
  q.failed_dies.forEach(d=>rows.push(`${d.row},${d.col},${d.xy_mm[0]},${d.xy_mm[1]},${d.in_pattern}`));
  const blob=new Blob([rows.join('\n')],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download=`waferscan_${r.prediction.class}_${Date.now()}.csv`;document.body.appendChild(a);a.click();a.remove();
  logLine(`${TS()}Export ready (${q.failed_dies.length} failed dies, ${Object.keys(r.prediction.probabilities).length} class probabilities)`);
}

async function boot(){
  let card=null;
  try{
    const ctl=new AbortController();const t=setTimeout(()=>ctl.abort(),2500);
    const r=await fetch(API+'/model/card',{signal:ctl.signal});clearTimeout(t);
    if(r.ok)card=await r.json();
  }catch(_){/* no API -> demo mode */}
  if(card){
    LIVE.on=true;LIVE.card=card;
    try{applyModelCard(card);}catch(err){console.error(err);}
    try{
      const s=await api('/samples');
      LIVE.order=s.sort(()=>Math.random()-.5);
      logLine(`${TS()}Ready. ${s.length} validation wafers loaded (unseen by the fast model).`);
      liveScan();
    }catch(err){logLine(`${TS()}<span class="err">${esc(err.message)}</span>`);}
  }else{
    renderCM(['SCR','PAR','EDG','CON'],[[142,3,1,0],[2,98,2,1],[0,3,44,2],[0,1,1,31]]);
    renderLatency(Array.from({length:64},(_,i)=>800+Math.sin(i*.4)*200+Math.random()*250));
    drawWafer();
  }
  startCounters();
}

// Initial draw
drawWafer();
boot();
