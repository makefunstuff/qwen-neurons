// Evil Pugs: the neuron debugger. Layers are rows, MLP neurons are registers. Generation runs token by token on the
// server; breakpoints pause it before a token, and while paused every tweak (pin, nudge, set) re-runs the sequence
// with the new wiring and shows how the next-token odds move. app.js hands over its helpers via initDebugger(ctx).
import {pugSvg, svgTag} from './pug.js';

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmt = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(Math.abs(v) >= 10 ? 1 : 2);
const key = (L, n) => `${L}:${n}`;
const showTok = t => t.replace(/\n/g, '⏎').replace(/ /g, '·') || '∅';
const PROMPTS = ['What is 17 * 23?', 'Write a haiku about Mondays.', 'Who are you?', 'Explain recursion in one sentence.'];

let X;
const D = {
  prompt: PROMPTS[0], stepMode: false, bpTokens: '', bpText: '', maxTokens: 120,
  pins: [], nudges: [], clamps: [],          // the wiring: [{layer, neuron}], [{layer, neuron, dose}], [{layer, neuron, value}]
  tokens: [], regs: null, cands: null, paused: null, running: false, hist: {}, menu: null, rewired: false,
};

function wiring() {
  return {knobs: [...X.onGenes(), ...D.nudges], clamps: D.clamps, pins: D.pins.map(({layer, neuron}) => ({layer, neuron}))};
}
async function ctl(cmd, extra = {}) {
  try { await X.api('/api/debug/ctl', {method: 'POST', body: {cmd, ...extra}}); } catch (err) { X.toast(err.message, true); }
}
function rewire() { if (D.running) ctl('update', wiring()); }

function render() {
  const pane = $('#pane-debug');
  const look = X.lookOf({id: 'debugger-pug', genes: [], look: {seed: 'debugger-pug', gear: ['antenna', 'goggles'], mood: 'eager', color: '#3dd68c'}});
  const stage = X.onGenes();
  pane.innerHTML = `<div class="dbg">
    <div class="lab-head">${svgTag(pugSvg(look.traits, {mood: look.mood, gear: look.gear}), 'pet')}<div><p class="kicker">breakpoints for a brain</p><h2>Neuron debugger</h2>
      <p class="sub">Layers are rows, MLP neurons are registers. Pause before any token, poke a register, and watch the next-token odds move.</p></div></div>
    <div class="dbg-controls">
      <div class="dbg-prompt"><input id="dbg-prompt" value="${esc(D.prompt)}" aria-label="Prompt"><div class="chips">${PROMPTS.map(p => `<button class="chip-btn" data-prompt="${esc(p)}">${esc(p)}</button>`).join('')}</div></div>
      <div class="row dbg-bps"><label class="toggle"><input type="checkbox" id="dbg-step" ${D.stepMode ? 'checked' : ''}><span>👣 pause on every token</span></label>
        <label class="muted small">🔴 break at token # <input id="dbg-bp" value="${esc(D.bpTokens)}" placeholder="e.g. 3, 10" size="7"></label>
        <label class="muted small">🔴 break after text <input id="dbg-bptext" value="${esc(D.bpText)}" placeholder="e.g. =" size="5"></label></div>
      <div class="row"><button class="cta" data-act="run" ${D.running ? 'disabled' : ''}>▶ Run</button>
        <button class="btn" data-act="pause" ${D.running && !D.paused ? '' : 'disabled'}>⏸ Pause</button>
        <button class="btn" data-act="step" ${D.paused ? '' : 'disabled'}>⏭ Step</button>
        <button class="btn" data-act="continue" ${D.paused ? '' : 'disabled'}>⏵ Continue</button>
        <button class="btn" data-act="stop" ${D.running ? '' : 'disabled'}>■ Stop</button>
        <span class="muted small">${stage.length ? `stage knobs on: ${esc(stage.map(g => `L${g.layer} N${g.neuron} ${fmt(g.dose)}`).join(' · '))}` : 'stage knobs: none (plain Qwen)'}</span></div></div>
    <div class="dbg-out" id="dbg-out">${tokensHtml()}</div>
    <div class="dbg-main"><div class="dbg-side">${pausedHtml()}${wiringHtml()}</div><div class="regs" id="regs">${regsHtml()}</div></div></div>`;
}
function tokensHtml() {
  if (!D.tokens.length) return `<span class="muted">${D.running ? 'warming up…' : 'Press ▶ Run. Click a generated token later to break there next time.'}</span>`;
  return D.tokens.map((t, i) => `<span class="tok${D.paused?.i === i + 1 && false ? ' here' : ''}${bpSet().has(i) ? ' bp' : ''}" data-tok="${i}" title="token #${i} · p=${t.p}" style="--heat:${Math.round((1 - Math.min(1, t.p)) * 100)}%">${esc(t.piece).replace(/\n/g, '⏎<br>')}</span>`).join('') +
    (D.paused ? '<span class="cursor">⏸</span>' : D.running ? '<span class="cursor blink">▍</span>' : '');
}
function bpSet() { return new Set(D.bpTokens.split(/[\s,]+/).filter(Boolean).map(Number).filter(Number.isFinite)); }
function pausedHtml() {
  if (!D.paused) return `<div class="dbg-card muted small">${D.running ? '⏵ running: ⏸ to pause, or set a breakpoint' : 'Not running.'}</div>`;
  const c = D.cands || [];
  return `<div class="dbg-card paused${D.rewired ? ' rewired' : ''}"><b>⏸ paused before token #${D.paused.i}</b>
    <div class="muted small">${D.rewired ? '🔌 rewired: the odds below use your new wiring' : 'what Qwen wants to say next:'}</div>
    <div class="cands">${c.map(([t, p]) => `<div class="cbar"><span class="mono">${esc(showTok(t))}</span><i style="width:${Math.max(2, p * 100)}%"></i><small>${(p * 100).toFixed(1)}%</small></div>`).join('')}</div></div>`;
}
function wiringHtml() {
  const pins = D.pins.map(p => { const h = D.hist[key(p.layer, p.neuron)] || []; return `<li>📌 <b class="mono">L${p.layer} N${p.neuron}</b> <span class="mono">${h.length ? fmt(h[h.length - 1]) : '…'}</span>${spark(h)}<button data-rm="pins:${p.layer}:${p.neuron}">✕</button></li>`; }).join('');
  const nud = D.nudges.map(g => `<li>🎚️ <b class="mono">L${g.layer} N${g.neuron}</b> dose ${fmt(g.dose)}<button data-rm="nudges:${g.layer}:${g.neuron}">✕</button></li>`).join('');
  const cl = D.clamps.map(c => `<li>✋ <b class="mono">L${c.layer} N${c.neuron}</b> = ${fmt(c.value)}<button data-rm="clamps:${c.layer}:${c.neuron}">✕</button></li>`).join('');
  return `<div class="dbg-card"><b>wiring</b>${pins || nud || cl ? `<ul class="wires">${pins}${nud}${cl}</ul>` : '<p class="muted small">Click a register to 📌 pin it (watch it every token), 🎚️ nudge it (a steering knob) or ✋ set it (overwrite its activation).</p>'}
    ${pins || nud || cl ? '<button class="link" data-act="clear-wires">clear all</button>' : ''}</div>`;
}
function spark(h) {
  if (h.length < 2) return '';
  const pts = h.slice(-40), max = Math.max(...pts.map(Math.abs), 1e-6);
  return `<svg class="spark" viewBox="0 0 80 20" preserveAspectRatio="none"><line x1="0" y1="10" x2="80" y2="10" stroke="var(--line)"/><polyline fill="none" stroke="var(--accent)" stroke-width="1.5" points="${pts.map((v, i) => `${(i / (pts.length - 1) * 80).toFixed(1)},${(10 - v / max * 9).toFixed(1)}`).join(' ')}"/></svg>`;
}
function regsHtml() {
  if (!D.regs) return `<div class="regs-empty muted">${'L00 '.repeat(0)}The register file shows up on the first token: each row is a layer, each cell one of its strongest neurons right now.</div>`;
  const max = Math.max(...D.regs.layers.flatMap(l => l.top.map(([, v]) => Math.abs(v))), 1e-6);
  const wired = new Set([...D.pins, ...D.nudges, ...D.clamps].map(w => key(w.layer, w.neuron)));
  const pinned = Object.fromEntries((D.regs.pins || []).map(([L, n, v]) => [key(L, n), v]));
  return `<div class="regs-head"><span>layer</span><span>strongest neurons at this token (activation)</span></div>` + D.regs.layers.map(l => {
    const extra = D.pins.filter(p => p.layer === l.layer && !l.top.some(([n]) => n === p.neuron)).map(p => [p.neuron, pinned[key(p.layer, p.neuron)] ?? 0]);
    return `<div class="reg-row"><span class="rl">L${String(l.layer).padStart(2, '0')}</span>${[...l.top, ...extra].map(([n, v]) => {
      const a = Math.min(1, Math.sqrt(Math.abs(v) / max));
      return `<button class="reg${wired.has(key(l.layer, n)) ? ' wired' : ''}" data-reg="${l.layer}:${n}:${v}" style="--a:${(a * 100).toFixed(0)}%;--col:${v >= 0 ? '#f08a24' : '#3a9ad9'}" title="L${l.layer} N${n}: ${v}"><small>N${n}</small><b>${fmt(v)}</b></button>`;
    }).join('')}</div>`;
  }).join('');
}
function update(part) {
  if (part === 'tokens' || part === 'all') { const o = $('#dbg-out'); if (o) { o.innerHTML = tokensHtml(); o.scrollTop = o.scrollHeight; } }
  if (part === 'regs' || part === 'all') { const r = $('#regs'); if (r) r.innerHTML = regsHtml(); const s = $('.dbg-side'); if (s) s.innerHTML = pausedHtml() + wiringHtml(); }
}

async function run() {
  if (X.S.busy) return X.toast('Qwen is busy: wait for the reply or stop it first', true);
  D.prompt = $('#dbg-prompt').value.trim() || PROMPTS[0];
  Object.assign(D, {tokens: [], regs: null, cands: null, paused: null, running: true, hist: {}, rewired: false});
  render();
  X.setBusy(true);
  const body = {messages: [{role: 'user', content: D.prompt}], ...wiring(), step: D.stepMode, max_new_tokens: D.maxTokens,
    breakpoints: {tokens: [...bpSet()], text: D.bpText}, temperature: 0.7, top: 6};
  try {
    await X.stream('/api/debug', body, ev => {
      if (ev.t === 'progress') { $('#dbg-out').innerHTML = `<span class="muted">☕ ${esc(ev.msg)}</span>`; }
      else if (ev.t === 'dbg_tok') {
        if (D.paused) { D.paused = null; render(); }
        if (!ev.eos) D.tokens.push({piece: ev.piece, p: ev.p});
        D.regs = ev.regs; D.cands = ev.cands;
        for (const [L, n, v] of ev.regs.pins || []) (D.hist[key(L, n)] ||= []).push(v);
        update('tokens'); update('regs');
      } else if (ev.t === 'paused') {
        D.paused = {i: ev.i}; D.regs = ev.regs; D.cands = ev.cands; D.rewired = !!ev.rewired;
        render();
        if (!ev.rewired) X.buzz?.(10);
      } else if (ev.t === 'dbg_timeout') X.toast('⏰ The debugger sat paused for 5 minutes, so it stopped and let the GPU nap');
      else if (ev.t === 'error') X.toast('⚠️ ' + ev.msg, true, 7000);
    });
  } catch (err) { X.toast(err.message, true); } finally {
    D.running = false; D.paused = null; X.setBusy(false); X.pollStatus(); render();
  }
}

function openMenu(btn) {
  const [L, n, v] = btn.dataset.reg.split(':').map(Number);
  closeMenu();
  const m = document.createElement('div');
  m.className = 'reg-menu';
  const isPinned = D.pins.some(p => p.layer === L && p.neuron === n);
  m.innerHTML = `<b class="mono">L${L} N${n}</b> <span class="muted mono">${fmt(v)}</span>
    <button data-m="pin">${isPinned ? '📌 unpin' : '📌 pin (watch it)'}</button>
    <button data-m="up">🎚️ nudge up (+0.5)</button><button data-m="down">🎚️ nudge down (−0.5)</button>
    <button data-m="zero">✋ silence it (= 0)</button><button data-m="boost">✋ crank it (= ×4)</button>
    <label class="small">✋ set to <input type="number" step="0.5" value="${(v * 3).toFixed(1)}" data-m="val"><button data-m="set">set</button></label>`;
  document.body.append(m);
  const r = btn.getBoundingClientRect();
  m.style.left = Math.min(innerWidth - 230, r.left) + 'px'; m.style.top = (r.bottom + scrollY + 4) + 'px';
  D.menu = {m, L, n, v};
  m.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    const act = b.dataset.m;
    const drop = (arr) => arr.filter(w => !(w.layer === L && w.neuron === n));
    if (act === 'pin') D.pins = isPinned ? drop(D.pins) : [...D.pins, {layer: L, neuron: n}];
    if (act === 'up' || act === 'down') {
      const cur = D.nudges.find(w => w.layer === L && w.neuron === n);
      const dose = Math.max(-2, Math.min(2, (cur?.dose || 0) + (act === 'up' ? 0.5 : -0.5)));
      D.nudges = drop(D.nudges); if (dose) D.nudges.push({layer: L, neuron: n, dose});
    }
    if (act === 'zero') D.clamps = [...drop(D.clamps), {layer: L, neuron: n, value: 0}];
    if (act === 'boost') D.clamps = [...drop(D.clamps), {layer: L, neuron: n, value: Math.max(-100, Math.min(100, v * 4))}];
    if (act === 'set') { const val = +m.querySelector('[data-m=val]').value; if (Number.isFinite(val)) D.clamps = [...drop(D.clamps), {layer: L, neuron: n, value: Math.max(-100, Math.min(100, val))}]; }
    closeMenu(); rewire(); update('regs');
  });
}
function closeMenu() { D.menu?.m.remove(); D.menu = null; }

export function initDebugger(ctx) {
  X = ctx;
  document.addEventListener('qn-tab', e => { if (e.detail === 'debug') render(); });
  document.addEventListener('click', e => { if (D.menu && !e.target.closest('.reg-menu, .reg')) closeMenu(); });
  const pane = $('#pane-debug');
  pane.addEventListener('click', e => {
    const b = e.target.closest('button, .tok'); if (!b) return;
    if (b.dataset.prompt) { D.prompt = b.dataset.prompt; $('#dbg-prompt').value = D.prompt; return; }
    if (b.dataset.reg) return openMenu(b);
    if (b.dataset.tok !== undefined) { const s = bpSet(), i = +b.dataset.tok; s.has(i) ? s.delete(i) : s.add(i); D.bpTokens = [...s].sort((a, z) => a - z).join(', '); $('#dbg-bp').value = D.bpTokens; if (D.running) ctl('update', {breakpoints: {tokens: [...s], text: D.bpText}}); return update('tokens'); }
    if (b.dataset.rm) { const [arr, L, n] = b.dataset.rm.split(':'); D[arr] = D[arr].filter(w => !(w.layer === +L && w.neuron === +n)); rewire(); return update('regs'); }
    const act = b.dataset.act;
    if (act === 'run') return run();
    if (act === 'pause') return ctl('pause');
    if (act === 'step') return ctl('step');
    if (act === 'continue') return ctl('continue');
    if (act === 'stop') return ctl('stop');
    if (act === 'clear-wires') { D.pins = []; D.nudges = []; D.clamps = []; rewire(); return update('regs'); }
  });
  pane.addEventListener('change', e => {
    if (e.target.id === 'dbg-step') D.stepMode = e.target.checked;
    if (e.target.id === 'dbg-bp' || e.target.id === 'dbg-bptext') {
      D.bpTokens = $('#dbg-bp').value; D.bpText = $('#dbg-bptext').value;
      if (D.running) ctl('update', {breakpoints: {tokens: [...bpSet()], text: D.bpText}});
      update('tokens');
    }
    if (e.target.id === 'dbg-prompt') D.prompt = e.target.value;
  });
}
