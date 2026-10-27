// Evil Pugs: the neuron explorer. A map of all 32 × 9216 MLP neurons (a column of dots per layer), search by word
// through the logit lens, and four views of one neuron: what it writes (logit lens), what pushing it does (steer
// lens), what makes it fire (corpus snippets) and who it talks to (the web of virtual weights).
// The lens, the web and the map come from the weights alone (a CPU process); the steer lens and fires use Qwen.
import {pugSvg, svgTag} from './pug.js';
import {achieve, openEval, showTab} from './lab.js';
import {askAgent} from './agent.js';

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const showTok = t => String(t).replace(/\n/g, '⏎').replace(/ /g, '·') || '∅';
const fmt = (v, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const key = (L, n) => `${L}:${n}`;
const COLW = 24, ROWS = 384, GAP = 6, NL = 32, NN = 9216, WW = NL * (COLW + GAP) - GAP;
const FULL_ATTN = new Set([3, 7, 11, 15, 19, 23, 27, 31]);
const VIEWS = {lens: '🔤 Logit lens', steer: '🎯 Steer lens', fires: '🔥 Fires', web: '🕸️ Web'};

let X;
const E = {
  status: null, sel: null, view: 'lens', dose: 0.8, q: '', hits: [], searching: false, color: 'norm', allLayers: false, moreHits: false,
  lens: {}, web: {}, fires: {}, steer: {}, cards: {}, mine: {},
  map: null, off: null, zoom: 1, ox: 0, oy: 0, hover: null, rendered: false, poll: null,
};

// ---------------------------------------------------------------- geometry
const pos = (L, n) => ({x: L * (COLW + GAP) + (n % COLW) + 0.5, y: Math.floor(n / COLW) + 0.5});
function at(mx, my) {
  const s = scale(), x = (mx - E.ox) / s, y = (my - E.oy) / s;
  if (x < 0 || y < 0 || y >= ROWS || x >= WW) return null;
  const L = Math.floor(x / (COLW + GAP)), c = Math.floor(x - L * (COLW + GAP));
  if (c >= COLW || L >= NL) return null;
  const n = Math.floor(y) * COLW + c;
  return n < NN ? {layer: L, neuron: n} : null;
}
function fit() { const cv = $('#x-map'); return cv ? Math.min(cv.clientWidth / WW, (cv.clientHeight - 18) / ROWS) : 1; }
function scale() { return fit() * E.zoom; }
function zooNeurons() {
  const m = {};
  for (const c of X.S.creatures) for (const g of c.genes || []) (m[key(g.layer, g.neuron)] ||= []).push(c.name);
  return m;
}

// ---------------------------------------------------------------- the map
function colormap(v) {   // 0..255 -> a dark-to-bright ramp that reads in both themes
  const t = v / 255;
  return [Math.round(40 + 215 * Math.min(1, t * 1.4)), Math.round(30 + 180 * t * t), Math.round(90 + 120 * (1 - t) * t * 2)];
}
function buildOff() {
  const off = document.createElement('canvas'); off.width = WW; off.height = ROWS;
  const ctx = off.getContext('2d'), img = ctx.createImageData(WW, ROWS), vals = E.map?.values;
  for (let L = 0; L < NL; L++) for (let n = 0; n < NN; n++) {
    const {x, y} = pos(L, n), i = (Math.floor(y) * WW + Math.floor(x)) * 4;
    const [r, g, b] = vals ? colormap(vals[L * NN + n]) : [120, 110, 140];
    img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = b; img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  E.off = off;
}
async function loadMap() {
  try {
    const m = await X.api(`/api/x/map?color=${E.color}`);
    const raw = atob(m.values), vals = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) vals[i] = raw.charCodeAt(i);
    E.map = {...m, values: vals};
  } catch (err) { X.toast('Map: ' + err.message, true); E.map = null; }
  buildOff(); draw();
}
function draw() {
  const cv = $('#x-map'); if (!cv) return;
  const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const ctx = cv.getContext('2d'), s = scale();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!E.off) buildOff();
  ctx.save(); ctx.translate(E.ox, E.oy); ctx.scale(s, s); ctx.imageSmoothingEnabled = false; ctx.drawImage(E.off, 0, 0); ctx.restore();
  const P = (L, n) => { const p = pos(L, n); return [E.ox + p.x * s, E.oy + p.y * s]; };
  // layer labels
  ctx.font = '10px "JetBrains Mono", monospace'; ctx.textAlign = 'center'; ctx.fillStyle = getComputedStyle(cv).color;
  for (let L = 0; L < NL; L++) {
    const x = E.ox + (L * (COLW + GAP) + COLW / 2) * s, y = E.oy + ROWS * s + 12;
    if (s * (COLW + GAP) > 16 || L % 4 === 0) ctx.fillText((FULL_ATTN.has(L) ? '◆' : '') + L, x, Math.min(y, h - 3));
  }
  // the web of the selected neuron
  const web = E.sel && E.web[key(E.sel.layer, E.sel.neuron)];
  if (web) {
    const [sx, sy] = P(E.sel.layer, E.sel.neuron);
    const all = [...web.upstream.flatMap(l => l.top.map(t => ({...t, layer: l.layer, up: true}))),
      ...web.downstream.flatMap(l => l.top.map(t => ({...t, layer: l.layer, up: false})))];
    const max = Math.max(...all.map(t => Math.abs(t.w)), 1e-6);
    for (const t of all.sort((a, b) => Math.abs(a.w) - Math.abs(b.w))) {
      const [x, y] = P(t.layer, t.neuron), a = Math.abs(t.w) / max;
      if (a < 0.25) continue;
      ctx.strokeStyle = t.w >= 0 ? `rgba(240,138,36,${0.25 + 0.7 * a})` : `rgba(58,154,217,${0.25 + 0.7 * a})`;
      ctx.lineWidth = 0.6 + 2.4 * a;
      ctx.beginPath(); ctx.moveTo(sx, sy);
      const mx = (sx + x) / 2; ctx.bezierCurveTo(mx, sy, mx, y, x, y); ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(x, y, 1.5 + 2 * a, 0, 7); ctx.fill();
    }
  }
  const ring = (L, n, r, color, lw = 2) => { const [x, y] = P(L, n); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke(); };
  for (const k of Object.keys(zooNeurons())) { const [L, n] = k.split(':').map(Number); ring(L, n, 4, '#f2668b', 1.5); }
  for (const hit of E.hits) ring(hit.layer, hit.neuron, 5, '#f5b82e', 2);
  if (E.sel) { ring(E.sel.layer, E.sel.neuron, 8, '#3dd68c', 3); ring(E.sel.layer, E.sel.neuron, 12, 'rgba(61,214,140,.35)', 2); }
  if (E.hover) ring(E.hover.layer, E.hover.neuron, 6, getComputedStyle(cv).color, 1.5);
}
function resetView() { E.zoom = 1; const cv = $('#x-map'); E.ox = cv ? (cv.clientWidth - WW * fit()) / 2 : 0; E.oy = 0; draw(); }
function focus(L, n, zoom = 5) {
  const cv = $('#x-map'); if (!cv) return;
  E.zoom = Math.max(E.zoom, zoom);
  const p = pos(L, n), s = scale();
  E.ox = cv.clientWidth / 2 - p.x * s; E.oy = (cv.clientHeight - 18) / 2 - p.y * s; draw();
}

async function hoverCard(n, mx, my) {
  const card = $('#x-hover'); if (!card) return;
  if (!n) { card.hidden = true; return; }
  const k = key(n.layer, n.neuron), zoo = zooNeurons()[k];
  card.hidden = false;
  const cv = $('#x-map'), left = Math.min(mx + 14, cv.clientWidth - 230);
  card.style.left = Math.max(0, left) + 'px'; card.style.top = Math.max(0, my - 10) + 'px';
  const fill = c => {
    card.innerHTML = `<b class="mono">L${n.layer} N${n.neuron}</b>${zoo ? ` <span class="zoo-tag">🐶 ${esc(zoo.join(', '))}</span>` : ''}
      ${c ? `<div class="hc-row"><span>＋</span>${c.promotes.slice(0, 5).map(t => `<i>${esc(showTok(t))}</i>`).join('')}</div>
      <div class="hc-row neg"><span>−</span>${c.suppresses.slice(0, 3).map(t => `<i>${esc(showTok(t))}</i>`).join('')}</div>
      <small>‖W[:,n]‖ ${c.col_norm.toFixed(3)}${c.act_max != null ? ` · act ${c.act_min.toFixed(1)}…${c.act_max.toFixed(1)}` : ''}</small>` : '<small>…</small>'}`;
  };
  fill(E.cards[k]);
  if (!E.cards[k]) {
    clearTimeout(E.hoverT);
    E.hoverT = setTimeout(async () => {
      try { E.cards[k] = await X.api(`/api/x/card/${n.layer}/${n.neuron}`); } catch { return; }
      if (E.hover && key(E.hover.layer, E.hover.neuron) === k) fill(E.cards[k]);
    }, 120);
  }
}

function wireMap() {
  const cv = $('#x-map'); if (!cv) return;
  let drag = null;
  cv.addEventListener('pointerdown', e => { drag = {x: e.clientX, y: e.clientY, ox: E.ox, oy: E.oy, moved: false}; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => {
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
      if (drag.moved) { E.ox = drag.ox + dx; E.oy = drag.oy + dy; $('#x-hover').hidden = true; return draw(); }
    }
    const n = at(mx, my);
    if (key(n?.layer, n?.neuron) !== key(E.hover?.layer, E.hover?.neuron)) { E.hover = n; draw(); }
    hoverCard(n, mx, my);
  });
  cv.addEventListener('pointerup', e => {
    const r = cv.getBoundingClientRect(), n = at(e.clientX - r.left, e.clientY - r.top);
    if (drag && !drag.moved && n) select(n.layer, n.neuron);
    drag = null;
  });
  cv.addEventListener('pointerleave', () => { E.hover = null; $('#x-hover').hidden = true; draw(); });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, s0 = scale();
    E.zoom = Math.max(1, Math.min(40, E.zoom * Math.exp(-e.deltaY * 0.0015)));
    const s1 = scale();
    E.ox = mx - (mx - E.ox) * s1 / s0; E.oy = my - (my - E.oy) * s1 / s0; draw();
  }, {passive: false});
}

// ---------------------------------------------------------------- pane
function statusHtml() {
  const st = E.status || {}, b = st.building;
  const idx = st.index ? '<span class="ok">✓ lens index</span>' : b?.state === 'running'
    ? `<span class="busy">⏳ building the lens index: ${esc(b.msg)}</span><div class="track"><i style="width:${Math.max(2, b.frac * 100)}%"></i></div>`
    : `<button class="btn" data-act="build-index" title="The logit lens of all 32 × 9216 neurons, from the weights, on the CPU (once, ~10 min). The GPU is not used.">🔨 build the lens index</button>${b?.state === 'error' ? ` <span class="bad">${esc(b.msg)}</span>` : ''}`;
  const fires = st.fires ? '<span class="ok">✓ fires</span>'
    : '<button class="btn" data-act="build-fires" title="Runs 4096 short C4 documents through Qwen once (about a minute on a GPU) and keeps each neuron\'s top examples">🔥 build fires</button>';
  return `<div class="x-status">${idx}<span class="sep">·</span>${fires}<span class="sep">·</span><span class="muted small">the lens, web and map come from the weights alone (CPU); 🎯 and 🔥 use Qwen</span></div>`;
}
function headHtml() {
  const look = X.lookOf({id: 'explorer-pug', genes: [], look: {seed: 'explorer-pug', gear: ['goggles', 'antenna'], mood: 'surprised', color: '#7c6cf2'}});
  return `<div class="lab-head">${svgTag(pugSvg(look.traits, {mood: look.mood, gear: look.gear}), 'pet')}<div>
      <p class="kicker">32 layers × 9216 neurons</p><h2>Neuron explorer</h2>
      <p class="sub">Search neurons by the words they write, then look at one: what it writes, what pushing it does, what makes it fire, and who it talks to.</p></div></div>
    ${statusHtml()}`;
}
function render() {
  const pane = $('#pane-explore');
  pane.innerHTML = `<div class="xplore">
    <section class="lab-card">${headHtml()}
      <form class="x-search" id="x-search"><input id="x-q" value="${esc(E.q)}" placeholder="search by word: conspir, cookie, 阴谋, arr…" aria-label="Search neurons by word">
        <button class="cta" ${E.status?.index ? '' : 'disabled'}>🔎 Search</button>
        <label class="toggle small" title="By default the search skips layers 0-2 and 27-31: the last layers write tokens directly but rarely steer a personality"><input type="checkbox" id="x-all" ${E.allLayers ? 'checked' : ''}><span>all layers</span></label>
        <span class="muted small x-goto">or jump to <input id="x-goto" placeholder="L19 N7839" size="10" aria-label="Go to a neuron"></span></form>
      <div id="x-hits">${hitsHtml()}</div>
      <div class="x-mapbox"><canvas id="x-map" aria-label="Map of all neurons: layers left to right, one dot per neuron"></canvas>
        <div class="x-hover" id="x-hover" hidden></div>
        <div class="x-mapctl"><button class="ghost" data-act="zoom-in" aria-label="Zoom in">＋</button><button class="ghost" data-act="zoom-out" aria-label="Zoom out">−</button>
          <button class="ghost" data-act="zoom-reset" aria-label="Reset the view">⟲</button>
          <select id="x-color" aria-label="Dot color"><option value="norm"${E.color === 'norm' ? ' selected' : ''}>color: ‖W_down[:,n]‖</option>
            <option value="act"${E.color === 'act' ? ' selected' : ''}${E.status?.fires ? '' : ' disabled'}>color: act max</option></select></div>
        <div class="x-legend muted small"><span><i class="lg sel"></i>selected</span><span><i class="lg hit"></i>search hits</span><span><i class="lg zoo"></i>zoo genes</span>
          <span><i class="lg up"></i>pushes toward</span><span><i class="lg dn"></i>away</span><span>◆ full-attention layer</span><span>drag · scroll to zoom · hover</span></div></div>
    </section>
    <section class="lab-card" id="x-neuron">${neuronHtml()}</section></div>`;
  wireMap();
  requestAnimationFrame(() => { if (!E.rendered) { resetView(); E.rendered = true; } else draw(); });
  if (!E.map) loadMap();
}
function hitsHtml() {
  if (E.searching) return '<p class="muted small">🔎 reading the lens index…</p>';
  if (!E.res) return '';
  const r = E.res;
  if (!r.results.length) return `<p class="muted small">${r.matched_tokens} tokens contain “${esc(r.query)}”, but no neuron has them in its top 20. Try a shorter stem or another language.</p>`;
  return `<p class="muted small">Layers ${r.layers[0]}–${r.layers[1]}: ${r.matched_tokens} tokens contain “${esc(r.query)}” (${r.examples.slice(0, 6).map(t => `<code>${esc(showTok(t))}</code>`).join(' ')}). <b>side +</b>: pushing the neuron up writes the word; <b>side −</b>: pushing it down does.</p>
    <div class="x-hitlist">${r.results.slice(0, E.moreHits ? 60 : 12).map((h, i) => `<button class="x-hit${E.sel && E.sel.layer === h.layer && E.sel.neuron === h.neuron ? ' on' : ''}" data-hit="${i}">
      <b class="mono">L${h.layer} N${h.neuron}</b><span class="side ${h.side === '+' ? 'pos' : 'neg'}">${h.side === '+' ? '＋' : '−'}</span><small>${h.score}</small>
      <span class="x-toks">${h.tokens.slice(0, 3).map(t => `<i>${esc(showTok(t))}</i>`).join('')}</span></button>`).join('')}
      ${r.results.length > 12 ? `<button class="link" data-act="more-hits">${E.moreHits ? 'fewer' : `all ${r.results.length}`}</button>` : ''}</div>`;
}

function neuronHtml() {
  if (!E.sel) return `<div class="hello muted">Tap a dot on the map, a search hit, or a pug's gene. The pink rings are the zoo's neurons: try L24 N3102 (Cookie).</div>`;
  const {layer: L, neuron: n} = E.sel, zoo = zooNeurons()[key(L, n)];
  return `<div class="x-nhead"><div><p class="kicker">layer ${L}${FULL_ATTN.has(L) ? ' · full attention' : ' · linear attention'}</p><h3 class="mono">L${L} N${n}</h3>
      ${zoo ? `<span class="zoo-tag">🐶 ${esc(zoo.join(', '))}</span>` : ''}</div>
    <div class="x-nbtns"><label class="muted small">dose <input type="number" id="x-dose" min="-2" max="2" step="0.1" value="${E.dose}"></label>
      <button class="btn" data-act="try" title="Put it on the stage alone and say hi">▶ try</button><button class="btn" data-act="try-neg" title="The opposite sign">▶ try −</button>
      <button class="btn" data-act="mix" title="Mix it into the pug on stage">＋ mix</button>
      <button class="btn" data-act="score" title="Score it in the Lab's evaluate mode">🧪 score</button>
      <button class="btn" data-act="ask" title="Ask the Lab agent what this neuron does">🤖 ask the agent</button></div></div>
    <nav class="tabs mini">${Object.entries(VIEWS).map(([k, v]) => `<button data-view="${k}" class="${E.view === k ? 'on' : ''}">${v}</button>`).join('')}</nav>
    <div id="x-view">${viewHtml()}</div>`;
}
const chip = (t, z, cls = '') => `<span class="tchip ${cls}" title="${esc(JSON.stringify(t))}${z != null ? ' · ' + z : ''}"><i>${esc(showTok(t))}</i>${z != null ? `<small>${typeof z === 'number' ? z.toFixed(1) : z}</small>` : ''}</span>`;
function viewHtml() {
  const k = key(E.sel.layer, E.sel.neuron);
  if (E.view === 'lens') {
    const l = E.lens[k];
    if (!l) return '<p class="muted">🔤 reading W_down[:, n] through the unembedding…</p>';
    return `<p class="muted small">From the weights alone: W_U · (g ⊙ W_down[:, n]), as z-scores over the 248k-token vocabulary. ‖W[:,n]‖ = ${l.col_norm.toFixed(3)}.</p>
      <div class="x-two"><div><b>pushing it up writes</b><div class="tchips">${l.promotes.map(t => chip(t.tok, t.z, 'pos')).join('')}</div></div>
      <div><b>pushing it down writes</b><div class="tchips">${l.suppresses.map(t => chip(t.tok, t.z, 'neg')).join('')}</div></div></div>`;
  }
  if (E.view === 'steer') {
    const s = E.steer[k];
    const btns = `<div class="row"><button class="cta alt" data-act="steer-lens">🎯 push it ±${Math.abs(E.dose)}</button><button class="btn" data-act="ablate">🚫 zero-ablate</button>
      <span class="muted small">12 reference prompts, the next token: plain vs edited (uses Qwen)</span></div>`;
    if (!s) return btns;
    if (s.running) return btns + `<div class="scan small"><div class="track"><i style="width:${Math.max(3, s.frac * 100)}%"></i></div><small>${esc(s.msg || 'waking Qwen…')}</small></div>`;
    const side = (name, x) => x ? `<div class="x-shift"><b>${name}</b> <span class="muted small">KL ${x.kl.toFixed(3)}</span>
      <div class="tchips">${x.rising.slice(0, 10).map(t => chip(t.tok, '+' + t.dlogp.toFixed(1), 'pos')).join('') || '<span class="muted small">nothing rises</span>'}</div>
      <div class="tchips">${x.falling.slice(0, 8).map(t => chip(t.tok, t.dlogp.toFixed(1), 'neg')).join('') || '<span class="muted small">nothing falls</span>'}</div></div>` : '';
    const cols = s.mode === 'ablate' ? ['plain', 'zeroed'] : ['plain', `+${Math.abs(s.genes[0].dose)}`, `−${Math.abs(s.genes[0].dose)}`];
    return btns + `<div class="x-two">${side(s.mode === 'ablate' ? '🚫 zeroed' : `⬆ pushed +${Math.abs(s.genes[0].dose)}`, s.push || s.zeroed)}${side(`⬇ pulled −${Math.abs(s.genes[0].dose)}`, s.pull)}</div>
      <table class="x-table"><tr><th>reference</th>${cols.map(c => `<th>${c}</th>`).join('')}</tr>${s.prompts.map(p => `<tr><td>${esc(p.chat ? '💬 ' + p.text : '“' + p.text + '…”')}</td>${p.next.map(t => `<td class="mono">${esc(showTok(t))}</td>`).join('')}</tr>`).join('')}</table>`;
  }
  if (E.view === 'fires') {
    const f = E.fires[k], mine = E.mine[k];
    const own = `<form class="x-own" id="x-own"><textarea id="x-own-text" rows="2" placeholder="Your own text: see how this neuron fires on every token">${esc(mine?.text || '')}</textarea><button class="btn">🔥 fire</button></form>
      ${mine?.tokens ? `<div class="x-heat">${heat(mine.tokens, mine.acts)}</div>` : mine?.running ? '<p class="muted small">running…</p>' : ''}`;
    if (!E.status?.fires) return `<p class="muted">No fires index yet: <button class="btn" data-act="build-fires">🔥 build it</button> (4096 short C4 documents through Qwen, once, about a minute on a GPU).</p>${own}`;
    if (!f) return '<p class="muted">🔥 looking up its top examples…</p>' + own;
    const sn = (x, cls) => `<div class="x-snip ${cls}"><span class="act">${fmt(x.act)}</span><span>…${esc(x.before.slice(-110))}<mark>${esc(x.tok)}</mark>${esc(x.after)}…</span></div>`;
    return `<p class="muted small">act max <b>${fmt(f.act_max)}</b> · min <b>${fmt(f.act_min)}</b> · mean |act| ${f.mean_abs.toFixed(3)} · ${esc(f.source)}</p>
      <div class="x-snips">${f.top.slice(0, 6).map(x => sn(x, 'hi')).join('')}</div>
      <details><summary class="muted small">most negative</summary><div class="x-snips">${f.bottom.slice(0, 4).map(x => sn(x, 'lo')).join('')}</div></details>${own}`;
  }
  const w = E.web[k];
  if (!w) return '<p class="muted">🕸️ multiplying virtual weights…</p>';
  const links = (list, dir) => list.map(l => `<div class="x-wl"><span class="mono">L${l.layer}</span>${l.top.slice(0, 3).map(t =>
    `<button class="wchip ${t.w >= 0 ? 'pos' : 'neg'}" data-goto="${l.layer}:${t.neuron}" title="gate ${t.gate} · up ${t.up}">N${t.neuron} <small>${fmt(t.w)}</small></button>`).join('')}</div>`).join('') || `<p class="muted small">none (${dir})</p>`;
  const heads = hs => hs.slice(0, 5).map(h => `<span class="tchip" title="${h.kind}"><i>L${h.layer} ${h.kind === 'attn' || h.kind === 'attn kv' ? '◆' : '∿'}H${h.head}</i><small>×${h.s}</small></span>`).join('');
  return `<p class="muted small">Weights only. A link is how far a dose +1 push of one neuron moves the other's gate/up inputs (<span class="pos">orange: toward</span>, <span class="neg">blue: away</span>). Drawn on the map.</p>
    <div class="x-two"><div><b>⬅ fed by</b>${links([...w.upstream].reverse().slice(0, 10), 'layer 0')}</div><div><b>pushes ➡</b>${links(w.downstream.slice(0, 10), 'last layer')}</div></div>
    <div class="x-two"><div><b>heads writing into it</b><div class="tchips">${heads(w.head_writers)}</div></div><div><b>heads reading it</b><div class="tchips">${heads(w.head_readers)}</div></div></div>
    <p class="muted small">◆ attention head (full-attention layers, by its output / value projection) · ∿ linear-attention head; ×n = n times that kind's median.</p>`;
}
function heat(tokens, acts) {
  const max = Math.max(...acts.slice(1).map(Math.abs), 1e-6);
  return tokens.map((t, i) => `<span style="--a:${(i ? Math.abs(acts[i]) / max : 0).toFixed(2)}" class="${acts[i] < 0 ? 'neg' : ''}" title="${acts[i]}">${esc(t)}</span>`).join('');
}

function renderNeuron() { const el = $('#x-neuron'); if (el) el.innerHTML = neuronHtml(); draw(); }
function renderView() { const el = $('#x-view'); if (el) el.innerHTML = viewHtml(); else renderNeuron(); }

// ---------------------------------------------------------------- data
export async function select(L, n, view) {
  E.sel = {layer: L, neuron: n};
  if (view) E.view = view;
  const pane = $('#pane-explore');
  if (pane.hidden) showTab('explore'); else renderNeuron();
  const hl = $('#x-hits'); if (hl) hl.innerHTML = hitsHtml();
  const k = key(L, n);
  const get = async (store, path) => {
    if (store[k]) return;
    try { store[k] = await X.api(path); } catch (err) { if (E.view === viewOf(store)) X.toast(err.message, true); return; }
    if (E.sel && key(E.sel.layer, E.sel.neuron) === k) { renderView(); draw(); }
  };
  get(E.lens, `/api/x/lens/${L}/${n}`);
  get(E.web, `/api/x/web/${L}/${n}`);
  if (E.status?.fires) get(E.fires, `/api/x/fires/${L}/${n}`);
  achieve('explorer');
}
function viewOf(store) { return store === E.lens ? 'lens' : store === E.web ? 'web' : 'fires'; }

async function refreshStatus() {
  try { E.status = await X.api('/api/x/status'); } catch { return; }
  const b = E.status.building;
  if (b?.state === 'running' && !E.poll) E.poll = setInterval(refreshStatus, 2500);
  if (b?.state !== 'running' && E.poll) { clearInterval(E.poll); E.poll = null; if (b?.state === 'done') X.toast('🧭 The lens index is ready: search away'); }
  const line = $('#pane-explore .x-status');
  if (line) line.outerHTML = statusHtml();
  const btn = $('#x-search .cta'); if (btn) btn.disabled = !E.status.index;
}

async function search(q) {
  E.q = q.trim(); if (!E.q) return;
  E.searching = true; $('#x-hits').innerHTML = hitsHtml();
  try { E.res = await X.api('/api/x/search?q=' + encodeURIComponent(E.q) + (E.allLayers ? '&lo=0&hi=31' : '')); E.hits = E.res.results; }
  catch (err) { X.toast(err.message, true); E.res = null; E.hits = []; }
  E.searching = false;
  $('#x-hits').innerHTML = hitsHtml(); draw();
  if (E.hits[0]) X.toast(`🔎 ${E.hits.length} neurons write “${E.q}”: the best is L${E.hits[0].layer} N${E.hits[0].neuron}`);
}

async function job(path, body, onEvent) {
  if (X.S.busy) { X.toast('Qwen is busy: wait for the reply or stop it first', true); return null; }
  X.setBusy(true);
  let res = null;
  try {
    await X.stream(path, body, ev => {
      if (ev.t === 'error') X.toast('⚠️ ' + ev.msg, true, 7000);
      else if (ev.t === 'result') res = ev;
      else onEvent?.(ev);
    });
  } catch (err) { X.toast(err.message, true); } finally { X.setBusy(false); X.pollStatus(); }
  return res;
}
async function steerLens(mode) {
  const {layer, neuron} = E.sel, k = key(layer, neuron);
  const genes = [{layer, neuron, dose: Math.abs(E.dose) || 0.8}];
  E.steer[k] = {running: true, frac: 0}; renderView();
  const res = await job('/api/x/probe', {genes, mode}, ev => {
    if (ev.t === 'progress') E.steer[k] = {running: true, frac: ev.frac * 0.5, msg: '☕ ' + ev.msg};
    if (ev.t === 'step') E.steer[k] = {running: true, frac: ev.i / ev.n, msg: ev.msg};
    if (E.sel && key(E.sel.layer, E.sel.neuron) === k) renderView();
  });
  E.steer[k] = res || null;
  if (E.sel && key(E.sel.layer, E.sel.neuron) === k) renderView();
}
async function fireOwn(text) {
  const {layer, neuron} = E.sel, k = key(layer, neuron);
  E.mine[k] = {text, running: true}; renderView();
  const res = await job('/api/x/fires-text', {texts: [text], neurons: [{layer, neuron}]});
  const t = res?.texts?.[0];
  E.mine[k] = t ? {text, tokens: t.tokens, acts: t.acts[k]} : {text};
  renderView();
}
async function buildFires() {
  X.toast('🔥 Building fires: Qwen reads 4096 short C4 documents once…', false, 5000);
  const res = await job('/api/x/fires-build', {n_docs: 4096, max_tokens: 64}, ev => {
    const el = $('#pane-explore .x-status'); if (el && (ev.t === 'step' || ev.t === 'progress')) el.querySelector('[data-act="build-fires"]')?.replaceWith(Object.assign(document.createElement('span'), {className: 'busy', textContent: '🔥 ' + ev.msg}));
  });
  if (res) X.toast(`🔥 Fires ready: ${res.docs} documents, ${res.tokens} tokens, ${res.secs}s`);
  E.fires = {}; await refreshStatus(); render();
  if (E.sel) select(E.sel.layer, E.sel.neuron);
}

function creatureOf(dose) {
  const {layer, neuron} = E.sel, zoo = zooNeurons()[key(layer, neuron)];
  return {id: `x-${layer}-${neuron}`, name: `🧭 L${layer} N${neuron}`, genes: [{layer, neuron, dose}],
    trait: zoo ? `the gene of ${zoo[0]}` : 'a neuron from the explorer', look: {seed: `x:${layer}:${neuron}`, gear: ['goggles'], mood: 'eager'}};
}

// ---------------------------------------------------------------- wiring
export function initExplorer(ctx) {
  X = ctx;
  document.addEventListener('qn-tab', e => { if (e.detail === 'explore') { refreshStatus().then(render); } });
  addEventListener('resize', () => { if (!$('#pane-explore').hidden) draw(); });
  const pane = $('#pane-explore');
  pane.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.hit !== undefined) { const h = E.res.results[+b.dataset.hit]; select(h.layer, h.neuron); E.dose = h.side === '+' ? Math.abs(E.dose) : -Math.abs(E.dose); renderNeuron(); return focus(h.layer, h.neuron, 2.5); }
    if (b.dataset.view) { E.view = b.dataset.view; for (const t of pane.querySelectorAll('[data-view]')) t.classList.toggle('on', t.dataset.view === E.view); return renderView(); }
    if (b.dataset.goto) { const [L, n] = b.dataset.goto.split(':').map(Number); select(L, n); return focus(L, n, 3); }
    const act = b.dataset.act;
    if (act === 'build-index') return X.api('/api/x/index', {method: 'POST'}).then(() => { X.toast('🔨 Building the lens index on the CPU (about 10 min, once). You can keep playing.', false, 6000); return refreshStatus(); }).then(render).catch(err => X.toast(err.message, true));
    if (act === 'build-fires') return buildFires();
    if (act === 'zoom-in' || act === 'zoom-out') { const cv = $('#x-map'), s0 = scale(), mx = cv.clientWidth / 2, my = cv.clientHeight / 2; E.zoom = Math.max(1, Math.min(40, E.zoom * (act === 'zoom-in' ? 1.8 : 1 / 1.8))); const s1 = scale(); E.ox = mx - (mx - E.ox) * s1 / s0; E.oy = my - (my - E.oy) * s1 / s0; return draw(); }
    if (act === 'zoom-reset') return resetView();
    if (act === 'more-hits') { E.moreHits = !E.moreHits; $('#x-hits').innerHTML = hitsHtml(); return; }
    if (!E.sel) return;
    if (act === 'try') return X.possess(creatureOf(E.dose), {intro: true});
    if (act === 'try-neg') return X.possess(creatureOf(-E.dose), {intro: true});
    if (act === 'mix') return X.mixIn(creatureOf(E.dose));
    if (act === 'score') { X.possess(creatureOf(E.dose), {quiet: true}); return openEval(''); }
    if (act === 'ask') return askAgent(`What does L${E.sel.layer} N${E.sel.neuron} do? Investigate it and write a verdict.`);
    if (act === 'steer-lens') return steerLens('steer');
    if (act === 'ablate') return steerLens('ablate');
  });
  pane.addEventListener('submit', e => {
    e.preventDefault();
    if (e.target.id === 'x-search') {
      const go = $('#x-goto').value.match(/L?\s*(\d+)\D+(\d+)/i);
      if (go && $('#x-goto').value.trim()) { const L = +go[1], n = +go[2]; $('#x-goto').value = ''; if (L < NL && n < NN) { select(L, n); return focus(L, n); } return X.toast('Layers 0–31, neurons 0–9215', true); }
      return search($('#x-q').value);
    }
    if (e.target.id === 'x-own') { const t = $('#x-own-text').value.trim(); if (t) fireOwn(t.slice(0, 1000)); }
  });
  pane.addEventListener('change', e => {
    if (e.target.id === 'x-dose') E.dose = Math.max(-2, Math.min(2, +e.target.value || 0.8));
    if (e.target.id === 'x-color') { E.color = e.target.value; loadMap(); }
    if (e.target.id === 'x-all') { E.allLayers = e.target.checked; if (E.q) search(E.q); }
  });
  pane.addEventListener('keydown', e => { if (e.target.id === 'x-goto' && e.key === 'Enter') { e.preventDefault(); $('#x-search').requestSubmit(); } });
}
export function exploreStatus() { return E.status; }
