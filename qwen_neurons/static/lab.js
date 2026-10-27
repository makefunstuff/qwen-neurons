// Evil Pugs: the Lab (find neurons, screen them, score cards the zoo's way), family trees and achievements.
// app.js hands over what it owns through initLab(ctx); this file never imports app.js.
import {pugSvg, svgTag, rng, mix} from './pug.js';

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmtDose = d => (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(d * 10 % 1 ? 2 : 1);
const geneLabel = g => `L${g.layer} N${g.neuron} ${fmtDose(g.dose)}`;
const geneKey = g => `${g.layer}:${g.neuron}`;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const fmtP = p => p < 0.001 ? 'p < 0.001' : p < 0.1 ? `p = ${p.toFixed(3)}` : `p = ${p.toFixed(2)}`;
const METRICS = {writer: 'writer: W_down[:, n] points along the persona', d: "effect size (Cohen's d) of the activation",
  resid: 'activation difference × ‖W[:, n]‖', diff: 'raw activation difference'};

let X;   // the context from app.js
const L = {
  sets: [], a: 'pirate', b: 'normal', metric: 'writer', from: 2, to: 25, dose: 0.8, top: 24,
  res: null, layerFilter: null, screen: {}, pool: [], running: null,
  ev: {trait: '', judge: 'model', keywords: '', n: 20, controls: 4, last: null},
};

// ---------------------------------------------------------------- achievements
const ACH = {
  'first-words': ['💬', 'First words', 'Got a reply from a pug'],
  gambler: ['🎰', 'Gambler', 'Spun the wheel'],
  loop: ['🔁', 'Broke it', 'Turned the knobs until it looped'],
  'first-hatch': ['🥚', 'Hatched', 'Bred your first pug'],
  collector: ['🏡', 'Collector', 'Kept a pug in your zoo'],
  dynasty: ['👑', 'Dynasty', 'Hatched a generation-3 pug'],
  dynasty5: ['🏰', 'Old money', 'Hatched a generation-5 pug'],
  genealogist: ['🌳', 'Genealogist', 'Looked at a family tree'],
  paparazzi: ['📸', 'Paparazzi', 'Shared a side-by-side or a score card'],
  scientist: ['🔬', 'Mad scientist', 'Scanned the brain for neurons'],
  roulette: ['🎲', 'Mystery meat', 'Tried a random neuron'],
  eureka: ['💡', 'Eureka', 'A screened neuron showed its trait'],
  hoarder: ['🧬', 'Gene hoarder', 'Five genes in the gene pool'],
  certified: ['📜', 'Certified weird', 'A score card passed the zoo\'s bar'],
  party: ['🎉', 'Pug party', 'Found the party'],
};
function achGot() { try { return JSON.parse(localStorage.getItem('qn-ach') || '{}'); } catch { return {}; } }
export function achieve(key) {
  const got = achGot();
  if (got[key] || !ACH[key]) return;
  got[key] = Date.now();
  try { localStorage.setItem('qn-ach', JSON.stringify(got)); } catch {}
  const [icon, name] = ACH[key];
  X?.toast(`${icon} Achievement unlocked: ${name}`, false, 4200);
  renderAchButton(true);
}
function renderAchButton(pop = false) {
  const b = $('#btn-ach'); if (!b) return;
  b.textContent = `🏅 ${Object.keys(achGot()).filter(k => ACH[k]).length}/${Object.keys(ACH).length}`;
  if (pop) b.animate([{transform: 'scale(1)'}, {transform: 'scale(1.3) rotate(-8deg)'}, {transform: 'scale(1)'}], {duration: 500});
}
function openAch() {
  const got = achGot();
  $('#ach-body').innerHTML = `<button class="x" data-close aria-label="Close">✕</button><h2>🏅 Achievements</h2>
    <p class="muted">Stored in this browser only.</p><div class="ach-grid">${Object.entries(ACH).map(([k, [icon, name, how]]) =>
      `<div class="ach${got[k] ? ' got' : ''}"><span>${got[k] ? icon : '❔'}</span><b>${esc(name)}</b><small>${esc(how)}</small></div>`).join('')}</div>`;
  $('#ach-dlg').showModal();
}

// ---------------------------------------------------------------- tabs
export function showTab(tab) {
  for (const b of document.querySelectorAll('.tabs button')) b.classList.toggle('on', b.dataset.tab === tab);
  for (const p of document.querySelectorAll('.pane')) p.hidden = p.id !== 'pane-' + tab;
  try { localStorage.setItem('qn-tab', tab); } catch {}
  if (tab === 'lab') renderLab();
  if (tab === 'family') renderFamily();
  document.dispatchEvent(new CustomEvent('qn-tab', {detail: tab}));
}

// ---------------------------------------------------------------- the lab: discovery
const setOf = id => L.sets.find(s => s.id === id);
function setChips(side) {
  const cur = L[side];
  return L.sets.filter(s => side === 'b' || s.id !== L.b).map(s => `<button class="chip-btn${s.id === cur ? ' on' : ''}" data-set="${esc(s.id)}" data-side="${side}"
    title="${esc(s.texts.length + ' replies' + (s.trait ? ': ' + s.trait : ''))}">${esc(s.emoji || '🧪')} ${esc(s.name)}</button>`).join('') +
    (side === 'a' ? `<button class="chip-btn ghosty" data-act="new-set" title="Write your own set of replies">✏️ your own…</button>` : '');
}
function labPug(seed, gear, mood = 'eager', color = '#7c6cf2', cls = 'pet') {
  const look = X.lookOf({id: 'lab:' + seed, genes: [], look: {seed, gear, mood, color}});
  return svgTag(pugSvg(look.traits, {mood, gear}), cls);
}
function candLook(c) {
  const set = setOf(L.res?.a) || {};
  const r = rng('cand:' + geneKey(c));
  return {seed: 'cand:' + geneKey(c), gear: [set.gear || 'goggles'], mood: ['eager', 'smug', 'derp', 'surprised', 'happy'][Math.floor(r() * 5)],
    color: `hsl(${Math.floor(r() * 360)} 65% 60%)`};
}
function candCreature(c) {
  const set = setOf(L.res?.a) || {};
  const look = candLook(c);
  return {id: 'cand-' + geneKey(c), name: `${set.emoji || '🧪'} L${c.layer} N${c.neuron}`, genes: [{layer: c.layer, neuron: c.neuron, dose: c.dose}],
    trait: set.trait ? `maybe ${set.trait}` : 'a discovered neuron', look: {seed: look.seed, gear: look.gear, mood: look.mood, color: X.hex(look.color)}};
}

function renderLab() {
  const pane = $('#pane-lab');
  const set = setOf(L.a) || {};
  if (!L.ev.trait) L.ev.trait = '';
  pane.innerHTML = `<div class="lab">
    <section class="lab-card" id="discover">
      <div class="lab-head">${labPug('detector', ['goggles'], 'eager', '#3a9ad9')}<div>
        <p class="kicker">find new personality neurons</p><h2>Neuron detector</h2>
        <p class="sub">Qwen reads replies in two voices; we rank its 32 × 9216 MLP neurons by how they tell the voices apart.</p></div></div>
      <div class="sets-row"><span class="lbl">sounds like</span><div class="setpick">${setChips('a')}</div></div>
      <div class="sets-row"><span class="lbl">versus</span><div class="setpick">${setChips('b')}</div></div>
      ${setOf(L.a) && !setOf(L.a).builtin ? `<div class="row small"><button class="link" data-act="edit-set">✎ edit “${esc(set.name)}”</button><button class="link danger" data-act="del-set">🗑 delete it</button></div>` : ''}
      <details class="adv"><summary>⚙️ detector settings</summary><div class="adv-grid">
        <label>ranking<select data-k="metric">${Object.entries(METRICS).map(([k, v]) => `<option value="${k}"${k === L.metric ? ' selected' : ''}>${esc(v)}</option>`).join('')}</select></label>
        <label>layers <span class="mono">${L.from}–${L.to}</span><span class="pair"><input type="number" min="0" max="31" value="${L.from}" data-k="from"><input type="number" min="0" max="31" value="${L.to}" data-k="to"></span></label>
        <label>dose to try<input type="number" min="0.1" max="2" step="0.1" value="${L.dose}" data-k="dose"></label>
        <label>candidates<input type="number" min="4" max="60" value="${L.top}" data-k="top"></label></div>
        <p class="muted small">The zoo's neurons live in layers 3–26; the last layers rank high but rarely steer. Most candidates do nothing on their own, which is why there is a screen.</p></details>
      <div class="row"><button class="cta" data-act="discover">🔍 Scan the brain</button>
        <button class="btn" data-act="roulette" title="A random neuron on a random middle layer, straight onto the stage">🎲 Mystery neuron</button>
        <span class="muted small">${L.pool.length ? `🧬 gene pool: ${L.pool.length}` : ''}</span></div>
      <div id="scan"></div><div id="xray"></div><div id="cands"></div>
    </section>
    <section class="lab-card" id="evaluate">${evalHtml()}</section></div>`;
  if (L.res) renderResults();
  if (L.ev.last) renderScorecard(L.ev.last);
}

function scanHtml(i, n, msg) {
  const bars = Array.from({length: 32}, (_, k) => `<i style="--h:${Math.round(20 + 80 * Math.abs(Math.sin(k * 1.7 + i)))}%"></i>`).join('');
  return `<div class="scan"><div class="scan-pug">${labPug('scanner', ['goggles'], 'surprised', '#3a9ad9')}</div><div class="scan-main">
    <div class="eq">${bars}</div><div class="track"><i style="width:${Math.max(3, i / n * 100)}%"></i></div><small>${esc(msg)}</small></div></div>`;
}

async function runJob(path, body, onEvent, scanEl) {
  if (X.S.busy) { X.toast('Qwen is busy: wait for the reply or stop it first', true); return false; }
  X.setBusy(true);
  L.running = path;
  let ok = false;
  try {
    await X.stream(path, body, ev => {
      if (ev.t === 'progress' && scanEl) scanEl.innerHTML = scanHtml(ev.frac * 10, 10, '☕ ' + ev.msg);
      else if (ev.t === 'loaded') X.pollStatus();
      else if (ev.t === 'error') { X.toast('⚠️ ' + ev.msg, true, 7000); }
      else { if (ev.t === 'result') ok = true; onEvent(ev); }
    });
  } catch (err) { X.toast(err.message, true); } finally { X.setBusy(false); L.running = null; X.pollStatus(); }
  return ok;
}

async function discover() {
  const a = setOf(L.a), b = setOf(L.b);
  if (!a || !b) return X.toast('Pick two sets', true);
  const layers = []; for (let l = Math.min(L.from, L.to); l <= Math.max(L.from, L.to); l++) layers.push(l);
  const scan = $('#scan');
  scan.innerHTML = scanHtml(0, 1, 'Waking up the detector'); $('#xray').innerHTML = ''; $('#cands').innerHTML = '';
  L.screen = {}; L.layerFilter = null;
  let res = null;
  await runJob('/api/discover', {a: a.id, b: b.id, layers, metric: L.metric, top_k: L.top, per_layer_k: 4, dose: L.dose}, ev => {
    if (ev.t === 'step') scan.innerHTML = scanHtml(ev.i, ev.n, `${ev.msg} · ${a.emoji} vs ${b.emoji}`);
    if (ev.t === 'result') res = ev;
  }, scan);
  scan.innerHTML = '';
  if (!res) return;
  L.res = {...res, a: a.id, b: b.id};
  renderResults();
  achieve('scientist');
  X.confetti(innerWidth * 0.3, 260, 30, ['#3a9ad9', '#7c6cf2', '#3dd68c']);
}

function xrayHtml(res) {
  const per = res.per_layer, max = Math.max(...per.map(p => p.max_score), 1e-9);
  return `<div class="xray-box"><div class="xray-title"><span class="rack-title">brain x-ray</span><span class="muted small">strongest neuron per layer · tap a layer</span></div>
    <div class="xray">${per.map(p => `<button class="xbar${L.layerFilter === p.layer ? ' on' : ''}" data-layer="${p.layer}" title="L${p.layer}: best ${p.max_score.toFixed(3)}" style="--h:${Math.max(4, p.max_score / max * 100)}%"><i></i><span>${p.layer}</span></button>`).join('')}</div></div>`;
}

function candRow(c, i) {
  const look = candLook(c), sc = L.screen[geneKey(c)];
  const score = L.res.metric === 'writer' ? `cos ${c.score.toFixed(3)}` : `${L.res.metric} ${c.score.toFixed(2)}`;
  const own = X.S.creatures.filter(x => !x.builtin);
  return `<div class="cand${sc?.hits ? ' hit' : ''}" data-i="${i}" style="--c:${X.hex(look.color)}">
    ${svgTag(pugSvg(X.lookOf({id: look.seed, genes: [], look}).traits, {mood: look.mood, gear: look.gear}), 'pet')}
    <div class="cinfo"><div><b class="mono">L${c.layer} N${c.neuron}</b> <span class="cdose" style="color:${c.dose >= 0 ? '#f08a24' : '#3a9ad9'}">${fmtDose(c.dose)}</span>
      <span class="muted small">${score}${L.res.metric !== 'd' ? ` · d ${c.d.toFixed(1)}` : ''}</span>
      ${sc ? `<span class="screen-badge${sc.hits ? ' yes' : ''}" title="${esc(sc.sample || '')}">🎯 ${sc.hits}/${sc.n}</span>` : ''}</div>
      <div class="kstats">resid ${c.median_resid.toFixed(1)} · push ${c.push.toFixed(1)} · ‖W[:,n]‖ ${c.col_norm.toFixed(3)} · act ${c.mean_a.toFixed(2)} vs ${c.mean_b.toFixed(2)}</div>
      ${sc?.sample ? `<div class="csample">“${esc(sc.sample.slice(0, 160))}${sc.sample.length > 160 ? '…' : ''}”</div>` : ''}</div>
    <div class="cbtns"><button data-act="try" title="Put it on the stage alone and say hi">▶ try</button><button data-act="mix" title="Mix it into the pug on stage">＋ mix</button>
      <button data-act="pool" title="Add to the gene pool: breeding mutations can draw from it">🧬</button>
      <select data-act="addto" title="Add this neuron to a creature or preset" aria-label="Add to a creature"><option value="">🐶 add to…</option>
        ${own.map(x => `<option value="${esc(x.id)}">${esc(x.name)}</option>`).join('')}
        <optgroup label="presets (saves a copy)">${X.S.creatures.filter(x => x.builtin).map(x => `<option value="${esc(x.id)}">${esc(x.name)}</option>`).join('')}</optgroup>
        <option value="__new">✨ a new creature</option></select></div></div>`;
}
function visibleCands() {
  if (!L.res) return [];
  if (L.layerFilter == null) return L.res.top;
  return L.res.per_layer.find(p => p.layer === L.layerFilter)?.top || [];
}
function renderResults() {
  const res = L.res, a = setOf(res.a) || {}, b = setOf(res.b) || {};
  $('#xray').innerHTML = xrayHtml(res);
  const list = visibleCands();
  const screened = Object.keys(L.screen).length;
  const set = setOf(res.a) || {};
  if (!L.screenTrait) L.screenTrait = set.trait || '';
  $('#cands').innerHTML = `<div class="cands-head"><b>${list.length} candidates</b> <span class="muted small">${esc(a.emoji)} ${esc(a.name)} vs ${esc(b.emoji)} ${esc(b.name)}${L.layerFilter != null ? ` · layer ${L.layerFilter} <button class="link" data-act="all-layers">show all</button>` : ''} · ${res.secs}s</span></div>
    <div class="screen-bar"><span>🎯 <b>Quick screen</b>: try the top 8 on 3 questions, judged blind for</span>
      <input data-k="screenTrait" value="${esc(L.screenTrait)}" placeholder="the trait, e.g. talks like a pirate">
      <button class="btn" data-act="screen">🎯 screen</button>${screened ? '<button class="link" data-act="sort-hits">sort by hits</button>' : ''}</div>
    <div class="cand-list">${list.map(candRow).join('')}</div>`;
}

async function screen() {
  const list = visibleCands().slice(0, 8);
  if (!list.length) return;
  const trait = (L.screenTrait || '').trim();
  if (!trait) return X.toast('Say what to look for first (the trait)', true);
  const scan = $('#scan');
  scan.innerHTML = scanHtml(0, 3, 'Screening the top 8');
  let res = null;
  await runJob('/api/screen', {candidates: list.map(c => ({layer: c.layer, neuron: c.neuron, dose: c.dose})), trait, judge: {kind: 'model'}, n: 3}, ev => {
    if (ev.t === 'screen_q') scan.innerHTML = scanHtml(ev.i, ev.n, `question ${ev.i}/${ev.n}: “${ev.q}” · hits so far ${ev.tally.slice(1).reduce((x, y) => x + y, 0)}`);
    if (ev.t === 'result') res = ev;
  }, scan);
  scan.innerHTML = '';
  if (!res) return;
  list.forEach((c, i) => { L.screen[geneKey(c)] = {hits: res.hits[i], n: res.n, sample: res.samples[i]}; });
  const best = Math.max(...res.hits);
  renderResults();
  if (best > 0) { achieve('eureka'); X.toast(`💡 ${res.hits.filter(h => h).length} of 8 showed the trait at least once (plain Qwen: ${res.plain}/${res.n})`, false, 5000); }
  else X.toast(`🤷 None of these 8 showed it (plain Qwen: ${res.plain}/${res.n}). Try another layer range, the next 8, or a bigger dose.`, false, 6000);
}

async function addTo(c, target) {
  const gene = {layer: c.layer, neuron: c.neuron, dose: c.dose};
  if (target === '__new') return X.editCreature({...candCreature(c), id: undefined, builtin: false, name: candCreature(c).name.replace(/^\S+ /, 'Neuron ')});
  const cr = X.S.creatures.find(x => x.id === target); if (!cr) return;
  const genes = cr.genes.filter(g => geneKey(g) !== geneKey(gene)).concat(gene);
  const body = cr.builtin ? {name: `${cr.name} + L${c.layer} N${c.neuron}`, genes, trait: cr.trait, parents: [], generation: 0}
    : {...cr, genes};
  try {
    const saved = await X.api('/api/creatures', {method: 'POST', body});
    X.toast(cr.builtin ? `💾 saved ${saved.name} (presets stay as they are)` : `🧬 L${c.layer} N${c.neuron} added to ${saved.name}`);
    await X.loadZoo();
  } catch (err) { X.toast(err.message, true); }
}

async function addPool(c) {
  const set = setOf(L.res?.a) || {};
  try {
    L.pool = await X.api('/api/genepool', {method: 'POST', body: [{layer: c.layer, neuron: c.neuron, dose: c.dose, label: set.name || ''}]});
    X.toast(`🧬 L${c.layer} N${c.neuron} joined the gene pool (${L.pool.length}): baby pugs can mutate into it`);
    if (L.pool.length >= 5) achieve('hoarder');
    renderLab();
  } catch (err) { X.toast(err.message, true); }
}

function roulette() {
  const r = Math.random;
  const layer = 3 + Math.floor(r() * 23), neuron = Math.floor(r() * 9216), dose = (r() < 0.5 ? -1 : 1) * 0.8;
  const c = {layer, neuron, dose};
  const look = {seed: 'mystery:' + geneKey(c), gear: ['propeller'], mood: 'derp', color: X.hex(`hsl(${Math.floor(r() * 360)} 70% 60%)`)};
  X.possess({id: 'mystery', name: `🎲 Mystery L${layer} N${neuron}`, genes: [c], trait: 'nobody knows', look}, {intro: true});
  achieve('roulette');
}

// ---------------------------------------------------------------- the lab: score cards
function evalHtml() {
  const genes = X.onGenes(), e = L.ev;
  if (!e.trait && X.S.star.trait && X.S.star.id !== 'plain') e.trait = X.S.star.trait.replace(/^maybe /, '');
  const secs = Math.round(e.n * (3 + e.controls * 0.35));
  return `<div class="lab-head">${labPug('judge', ['monocle'], 'smug', '#f2668b')}<div>
      <p class="kicker">the zoo's way</p><h2>Score card</h2>
      <p class="sub">N held-out questions, steered vs normal, each reply judged blind; a Fisher test and random-neuron controls.</p></div></div>
    <div class="ev-knobs"><span class="lbl">testing</span>${genes.length ? `<b>${esc(X.S.star.name)}</b> <span class="genes-label">${esc(genes.map(geneLabel).join(' · '))}</span>`
      : '<span class="muted">no knobs on the stage: tap a pug or try a candidate first</span>'}</div>
    <div class="ev-grid">
      <label class="wide">trait the judge looks for<input data-ev="trait" value="${esc(e.trait)}" placeholder="e.g. talks about cookies"></label>
      <label>judge<select data-ev="judge"><option value="model"${e.judge === 'model' ? ' selected' : ''}>🧑‍⚖️ Qwen itself (blind)</option>
        <option value="keywords"${e.judge === 'keywords' ? ' selected' : ''}>🔤 keyword rubric</option><option value="caps"${e.judge === 'caps' ? ' selected' : ''}>🔠 ALL CAPS rubric</option></select></label>
      <label${e.judge === 'keywords' ? '' : ' hidden'}>keywords (comma separated)<input data-ev="keywords" value="${esc(e.keywords)}" placeholder="cookie, biscuit"></label>
      <label>questions<select data-ev="n">${[10, 20, 50].map(n => `<option${n === e.n ? ' selected' : ''}>${n}</option>`).join('')}</select></label>
      <label>random controls<select data-ev="controls">${[0, 2, 4, 6].map(n => `<option${n === e.controls ? ' selected' : ''}>${n}</option>`).join('')}</select></label></div>
    <div class="row"><button class="cta" data-act="evaluate" ${genes.length ? '' : 'disabled'}>🧪 Run the test</button>
      <span class="muted small">about ${secs < 90 ? secs + ' s' : Math.round(secs / 60) + ' min'} on a GPU</span></div>
    <div id="ev-live"></div><div id="ev-card"></div>`;
}
export function refreshEval() {
  const box = $('#evaluate');
  if (box && !L.running) { const last = L.ev.last; box.innerHTML = evalHtml(); if (last) renderScorecard(last); }
}
export function openEval(trait) {
  if (trait) L.ev.trait = trait.replace(/^maybe /, '');
  showTab('lab');
  $('#evaluate')?.scrollIntoView({behavior: 'smooth', block: 'start'});
}

function jarsHtml(tally, n, done) {
  const names = ['normal', 'steered', ...tally.slice(2).map((_, i) => `ctrl ${i + 1}`)];
  return `<div class="jars">${tally.map((h, i) => `<div class="jar${i === 1 ? ' main' : i === 0 ? ' plain' : ''}"><div class="jar-glass"><i style="height:${done ? h / n * 100 : 0}%"></i></div>
    <b>${h}</b><small>${names[i]}</small></div>`).join('')}</div>`;
}

async function evaluate() {
  const genes = X.onGenes(), e = L.ev;
  if (!genes.length) return X.toast('Put some knobs on the stage first', true);
  const body = {genes, trait: e.trait, judge: {kind: e.judge, keywords: e.keywords.split(',').map(s => s.trim()).filter(Boolean)}, n: e.n, controls: e.controls};
  const who = {name: X.S.star.name, genes, id: X.S.star.id, star: X.S.star};
  const live = $('#ev-live');
  $('#ev-card').innerHTML = '';
  live.innerHTML = scanHtml(0, 1, 'Getting the judge ready');
  let res = null, best = null, n = e.n;
  const tally0 = Array(2 + e.controls).fill(0);
  await runJob('/api/evaluate', body, ev => {
    if (ev.t === 'eval_start') { n = ev.n; live.innerHTML = `<div class="ev-run">${jarsHtml(tally0, n, true)}<div class="track"><i style="width:2%"></i></div><small>asking question 1 of ${n}…</small><div class="ev-q"></div></div>`; }
    if (ev.t === 'eval_q') {
      if (ev.hits[1] && !best) best = {q: ev.q, normal: ev.normal, steered: ev.steered};
      live.innerHTML = `<div class="ev-run">${jarsHtml(ev.tally, ev.n, true)}<div class="track"><i style="width:${ev.i / ev.n * 100}%"></i></div>
        <small>${ev.i} of ${ev.n} questions · <button class="link" data-act="stop">■ stop early</button></small>
        <div class="ev-q"><b>“${esc(ev.q)}”</b><div class="ev-pair"><div><small>normal ${ev.hits[0] ? '✅' : '·'}</small>${esc(ev.normal.slice(0, 220))}</div>
        <div class="st"><small>steered ${ev.hits[1] ? '✅' : '·'}</small>${esc(ev.steered.slice(0, 220))}</div></div></div></div>`;
    }
    if (ev.t === 'result') res = ev;
  }, live);
  live.innerHTML = '';
  if (!res) return;
  L.ev.last = {...res, who, what: e.trait, best, judgeKind: e.judge, keywords: body.judge.keywords, at: Date.now()};
  renderScorecard(L.ev.last);
  if (res.passed) { achieve('certified'); X.confetti(innerWidth * 0.3, 300, 90); } else X.confetti(innerWidth * 0.3, 300, 16, ['#bbb', '#999']);
  X.onScored?.(L.ev.last);
}

function scorecardHtml(r) {
  const c = {...r.who.star, genes: r.who.genes};
  const look = X.lookOf(c), pct = r.trait / r.n * 100;
  const judge = r.judgeKind === 'model' ? 'judged blind by Qwen itself' : r.judgeKind === 'caps' ? 'ALL CAPS rubric' : `keyword rubric (${(r.keywords || []).join(', ')})`;
  const ctrl = r.controls.length ? `random controls: ${r.controls.map(x => `<b>${x}</b>`).join(' · ')}` : 'no random controls';
  const stamp = r.passed ? ['pass', 'certified weird'] : r.label === "something's there" ? ['maybe', "something's there"] : ['fail', 'just noise'];
  return `<div class="scorecard" style="--c:${look.color}">
    <div class="stamp ${stamp[0]}">${stamp[1]}</div>
    <div class="sc-top">${X.pug(c, 'pet', look)}<div><b class="sc-name">${esc(r.who.name)}</b><span class="genes-label">${esc(r.who.genes.map(geneLabel).join(' · '))}</span>
      <div class="muted small">“${esc(r.what || (r.judgeKind === 'caps' ? 'WRITES IN ALL CAPS' : ''))}”</div></div></div>
    <div class="score"><b>${r.trait}<small>/${r.n}</small></b><div class="bar"><i style="width:${pct}%"></i></div><div class="normal">normal: ${r.normal}/${r.n}</div></div>
    <ul class="checks"><li class="${r.p < r.alpha ? 'ok' : 'no'}">${r.p < r.alpha ? '✓' : '✗'} ${fmtP(r.p)} vs normal (one-sided Fisher)</li>
      <li class="${r.above_controls && r.controls.length ? 'ok' : 'no'}">${r.above_controls && r.controls.length ? '✓' : '✗'} ${ctrl}</li>
      <li class="info">⚖️ ${esc(judge)}${r.stopped ? ' · stopped early' : ''} · ${r.secs}s</li></ul>
    ${r.best ? `<details class="best"><summary>the best line</summary><p><b>“${esc(r.best.q)}”</b></p><p class="st">${esc(r.best.steered.slice(0, 420))}</p></details>` : ''}
    <div class="row sc-btns"><button class="btn" data-act="save-score">💾 ${r.who.star.builtin ? 'keep as a creature' : r.who.star.id && !String(r.who.star.id).match(/^(mix|plain|hatchling|mystery|cand-)/) ? 'save score' : 'keep with this score'}</button>
      <button class="btn" data-act="share-score">📸 copy card</button></div></div>`;
}
function renderScorecard(r) { const el = $('#ev-card'); if (el) el.innerHTML = scorecardHtml(r); }

function scoreField(r) {
  return {trait: r.trait, normal: r.normal, n: r.n, controls: r.controls, p: r.p, passed: r.passed,
    source: r.judgeKind === 'model' ? 'your lab · Qwen judge' : 'your lab · rubric', judge: r.judgeKind, what: r.what || ''};
}
async function saveScore() {
  const r = L.ev.last; if (!r) return;
  const star = r.who.star, saved = X.S.creatures.find(c => c.id === star.id && !c.builtin);
  try {
    let out;
    if (saved) out = await X.api('/api/creatures', {method: 'POST', body: {...saved, genes: r.who.genes, score: scoreField(r)}});
    else return X.editCreature({...star, genes: r.who.genes, id: undefined, builtin: false, parents: star.parents || [], score: scoreField(r),
      name: star.builtin ? `${star.name} (lab)` : star.name.replace(/^[🎲🧪] ?/u, ''), trait: r.what || star.trait});
    X.toast(`💾 ${out.name} now wears its score card`);
    await X.loadZoo();
  } catch (err) { X.toast(err.message, true); }
}

async function shareScore() {
  const r = L.ev.last; if (!r) return;
  const W = 1200, H = 675, cv = document.createElement('canvas'); cv.width = W; cv.height = H;
  const x = cv.getContext('2d'), c = {...r.who.star, genes: r.who.genes}, look = X.lookOf(c);
  const g = x.createLinearGradient(0, 0, W, H); g.addColorStop(0, '#e6f3f1'); g.addColorStop(0.5, '#faf6ef'); g.addColorStop(1, '#f3e3f1');
  x.fillStyle = g; x.fillRect(0, 0, W, H);
  x.fillStyle = '#fff'; x.strokeStyle = look.color; x.lineWidth = 6; x.beginPath(); x.roundRect(60, 60, W - 120, H - 150, 36); x.fill(); x.stroke();
  const img = await X.svgImage(look, 220, mix(look.color, '#ffffff', 0.72));
  x.drawImage(img, 100, 110, 220, 220);
  x.fillStyle = '#1d1a22'; x.font = '800 52px "Bricolage Grotesque", system-ui'; x.fillText(r.who.name.slice(0, 26), 360, 160);
  x.font = '600 22px "JetBrains Mono", monospace'; x.fillStyle = '#6f6878'; x.fillText(r.who.genes.map(geneLabel).join(' · ').slice(0, 52), 360, 200);
  x.font = 'italic 24px system-ui'; x.fillText(`“${(r.what || 'ALL CAPS').slice(0, 60)}”`, 360, 240);
  x.fillStyle = look.color; x.font = '800 110px "Bricolage Grotesque", system-ui'; x.fillText(String(r.trait), 360, 370);
  const w = x.measureText(String(r.trait)).width; x.font = '700 48px "Bricolage Grotesque", system-ui'; x.fillStyle = '#b5aebf'; x.fillText(`/${r.n}`, 370 + w, 370);
  x.fillStyle = '#efe9df'; x.beginPath(); x.roundRect(360, 395, 700, 16, 8); x.fill();
  x.fillStyle = look.color; x.beginPath(); x.roundRect(360, 395, Math.max(16, 700 * r.trait / r.n), 16, 8); x.fill();
  x.fillStyle = '#6f6878'; x.font = '500 24px system-ui';
  x.fillText(`normal: ${r.normal}/${r.n}   ·   ${fmtP(r.p)}   ·   controls: ${r.controls.join(', ') || 'none'}`, 360, 450);
  x.save(); x.translate(980, 150); x.rotate(-0.18); x.strokeStyle = r.passed ? '#2a9d8f' : '#e5484d'; x.fillStyle = x.strokeStyle; x.lineWidth = 5;
  x.font = '800 30px "JetBrains Mono", monospace'; const label = (r.passed ? 'certified weird' : r.label).toUpperCase(), lw = x.measureText(label).width;
  x.strokeRect(-lw / 2 - 16, -34, lw + 32, 50); x.fillText(label, -lw / 2, 2); x.restore();
  x.font = '600 18px "JetBrains Mono", monospace'; x.fillStyle = '#6f6878';
  x.fillText('Evil Pugs · neuron steering on Qwen3.5-4B · github.com/makefunstuff/qwen-neurons', 60, H - 40);
  await X.copyPng(cv, `evil-pugs-score-${r.who.name.replace(/\W+/g, '-').toLowerCase()}.png`);
  achieve('paparazzi');
}

// ---------------------------------------------------------------- prompt set editor
function editSet(s) {
  const f = $('#set-form');
  f.innerHTML = `<h2>${s?.id ? 'Edit' : 'New'} prompt set</h2>
    <p class="muted small">One reply per line, written in the voice you want to find (at least 2, up to 40). Line i answers “${esc(X.NEUTRAL[0])}”, “${esc(X.NEUTRAL[1])}”, …, the same questions as every other set.</p>
    <div class="row"><label class="field" style="flex:0 0 70px">Emoji<input name="emoji" value="${esc(s?.emoji || '🧪')}" maxlength="4"></label>
      <label class="field" style="flex:1">Name<input name="name" value="${esc(s?.name || '')}" maxlength="40" required placeholder="Robot"></label></div>
    <label class="field">Trait (what the judge looks for)<input name="trait" value="${esc(s?.trait || '')}" maxlength="160" placeholder="talks like a robot"></label>
    <label class="field">Replies, one per line<textarea name="texts" rows="9" placeholder="BEEP BOOP. MY DAY WAS EFFICIENT.&#10;Mornings: system boot sequence initiated.">${esc((s?.texts || []).join('\n'))}</textarea></label>
    <div class="row end"><button class="btn" value="cancel" formnovalidate>Cancel</button><button class="btn primary" value="ok">💾 Save set</button></div>`;
  const dlg = $('#set-dlg');
  dlg.onclose = async () => {
    if (dlg.returnValue !== 'ok') return;
    const fd = new FormData(f);
    try {
      const saved = await X.api('/api/sets', {method: 'POST', body: {id: s?.id, name: fd.get('name'), emoji: fd.get('emoji'), trait: fd.get('trait'), texts: String(fd.get('texts'))}});
      await loadSets(); L.a = saved.id; renderLab(); X.toast(`🧪 set “${saved.name}” saved`);
    } catch (err) { X.toast(err.message, true); }
  };
  dlg.returnValue = ''; dlg.showModal();
}
async function loadSets() { L.sets = await X.api('/api/sets'); }

// ---------------------------------------------------------------- family trees
function famParents(c, depth) {
  if (depth <= 0) return [];
  if (c.ancestry?.length) return c.ancestry;   // a kept creature: its snapshot
  return (c.parents || []).map(p => p && typeof p === 'object' ? p : X.S.creatures.find(x => x.id === p)).filter(Boolean);  // snapshot nodes hold objects
}
function famNode(c, depth = 0) {
  const live = c.id && X.S.creatures.find(x => x.id === c.id);
  const show = live || c, look = X.lookOf(show);
  const parents = famParents(c, 6 - depth);
  return `<div class="tnode"><button class="tcard${show.builtin ? ' root' : ''}${live ? '' : ' gone'}" ${live ? `data-cid="${esc(live.id)}"` : ''} style="--c:${look.color}" title="${esc((show.genes || []).map(geneLabel).join(', '))}">
      ${X.pug(show, 'pet', look)}<b>${esc(show.name)}</b><small>${show.builtin ? 'zoo founder' : `gen ${show.generation || 0}`}${live ? '' : ' · not kept'}</small></button>
    ${parents.length ? `<div class="tparents">${parents.map(p => famNode(p, depth + 1)).join('')}</div>` : ''}</div>`;
}
function descendants(id) {
  const has = n => (n.ancestry || n.parents || []).some(p => p && (p === id || p.id === id || (typeof p === 'object' && has(p))));
  return X.S.creatures.filter(c => !c.builtin && has(c));
}
let famPick = null;
export function openFamily(id) { famPick = id; showTab('family'); }
function renderFamily() {
  const pane = $('#pane-family');
  const bred = X.S.creatures.filter(c => !c.builtin).sort((a, b) => (b.generation || 0) - (a.generation || 0) || (b.created || 0) - (a.created || 0));
  if (!bred.length) {
    pane.innerHTML = `<div class="family empty"><div class="hello">${labPug('family', ['crown'], 'eager', '#f5b82e')}<b>No bloodlines yet.</b>
      Breed two pugs (🥚 on the Zoo tab), keep the baby, then breed the baby. Their family trees grow here, generation after generation.</div></div>`;
    return;
  }
  if (!famPick || !bred.some(c => c.id === famPick)) famPick = bred[0].id;
  const cur = bred.find(c => c.id === famPick);
  const gens = {}; bred.forEach(c => { gens[c.generation || 0] = (gens[c.generation || 0] || 0) + 1; });
  const kids = descendants(cur.id);
  pane.innerHTML = `<div class="family"><div class="lab-head">${labPug('family', ['crown'], 'smug', '#f5b82e')}<div><p class="kicker">bloodlines</p><h2>Family trees</h2>
      <p class="sub">${bred.length} kept pug${bred.length > 1 ? 's' : ''} · ${Object.entries(gens).sort((a, b) => b[0] - a[0]).map(([g, n]) => `gen ${g} ×${n}`).join(' · ')}</p></div></div>
    <div class="fam-list">${bred.map(c => { const look = X.lookOf(c); return `<button class="fam-chip${c.id === famPick ? ' on' : ''}" data-fam="${esc(c.id)}" style="--c:${look.color}">${X.pug(c, 'pet', look)}<span>${esc(c.name)}</span><small>gen ${c.generation || 0}</small></button>`; }).join('')}</div>
    <div class="tree-wrap"><div class="tree">${famNode(cur)}</div></div>
    <div class="row fam-btns"><button class="cta" data-act="fam-talk">💬 Talk to ${esc(cur.name)}</button><button class="btn" data-act="fam-breed">🥚 Breed it</button>
      ${kids.length ? `<span class="muted small">descendants: ${kids.map(k => `<button class="link" data-fam="${esc(k.id)}">${esc(k.name)}</button>`).join(', ')}</span>` : '<span class="muted small">no descendants yet</span>'}</div></div>`;
  achieve('genealogist');
}

// ---------------------------------------------------------------- wiring
export async function initLab(ctx) {
  X = ctx;
  renderAchButton();
  $('#btn-ach').addEventListener('click', openAch);
  $('#ach-dlg').addEventListener('click', e => { if (e.target.closest('[data-close]') || e.target === $('#ach-dlg')) $('#ach-dlg').close(); });
  document.querySelector('.tabs').addEventListener('click', e => { const b = e.target.closest('button[data-tab]'); if (b) showTab(b.dataset.tab); });
  try { await loadSets(); L.pool = await X.api('/api/genepool'); } catch (err) { X.toast('Lab: ' + err.message, true); }

  const lab = $('#pane-lab');
  lab.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.set) { L[b.dataset.side] = b.dataset.set; if (L.a === L.b) L.a = L.sets.find(s => s.id !== L.b)?.id; L.screenTrait = ''; return renderLab(); }
    if (b.dataset.layer !== undefined) { const l = +b.dataset.layer; L.layerFilter = L.layerFilter === l ? null : l; return renderResults(); }
    const act = b.dataset.act;
    const row = b.closest('.cand'), c = row ? visibleCands()[+row.dataset.i] : null;
    if (act === 'discover') return discover();
    if (act === 'roulette') return roulette();
    if (act === 'new-set') return editSet(null);
    if (act === 'edit-set') return editSet(setOf(L.a));
    if (act === 'del-set') return X.api(`/api/sets/${L.a}`, {method: 'DELETE'}).then(loadSets).then(() => { L.a = 'pirate'; renderLab(); X.toast('🗑 set deleted'); }).catch(err => X.toast(err.message, true));
    if (act === 'all-layers') { L.layerFilter = null; return renderResults(); }
    if (act === 'screen') return screen();
    if (act === 'sort-hits') { L.res.top.sort((p, q) => (L.screen[geneKey(q)]?.hits ?? -1) - (L.screen[geneKey(p)]?.hits ?? -1)); return renderResults(); }
    if (act === 'stop') return X.api('/api/stop', {method: 'POST'}).catch(() => {});
    if (act === 'evaluate') return evaluate();
    if (act === 'save-score') return saveScore();
    if (act === 'share-score') return shareScore();
    if (c && act === 'try') return X.possess(candCreature(c), {intro: true, from: row.querySelector('.pet')});
    if (c && act === 'mix') return X.mixIn(candCreature(c), row.querySelector('.pet'));
    if (c && act === 'pool') return addPool(c);
  });
  lab.addEventListener('change', e => {
    const t = e.target;
    if (t.dataset.k) {
      L[t.dataset.k] = t.type === 'number' ? Math.max(+t.min || 0, Math.min(+t.max || 99, +t.value)) : t.value;
      if (t.dataset.k !== 'screenTrait') renderLab();
    } else if (t.dataset.ev) {
      L.ev[t.dataset.ev] = ['n', 'controls'].includes(t.dataset.ev) ? +t.value : t.value;
      if (['judge', 'n', 'controls'].includes(t.dataset.ev)) refreshEval();
    } else if (t.dataset.act === 'addto' && t.value) {
      const row = t.closest('.cand'), c = visibleCands()[+row.dataset.i];
      addTo(c, t.value); t.value = '';
    }
  });
  lab.addEventListener('input', e => {
    if (e.target.dataset.ev === 'trait' || e.target.dataset.ev === 'keywords') L.ev[e.target.dataset.ev] = e.target.value;
    if (e.target.dataset.k === 'screenTrait') L.screenTrait = e.target.value;
  });

  $('#pane-family').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.fam) { famPick = b.dataset.fam; return renderFamily(); }
    if (b.dataset.cid && b.dataset.cid !== famPick) { const c = X.S.creatures.find(x => x.id === b.dataset.cid); if (c && !c.builtin) { famPick = c.id; return renderFamily(); } }
    const cur = X.S.creatures.find(x => x.id === famPick);
    if (b.dataset.act === 'fam-talk' && cur) X.possess(cur, {intro: true});
    if (b.dataset.act === 'fam-breed' && cur) X.openBreed(cur);
  });
  let tab = 'zoo';
  try { tab = localStorage.getItem('qn-tab') || 'zoo'; } catch {}
  showTab(['zoo', 'lab', 'family', 'debug'].includes(tab) ? tab : 'zoo');
}
