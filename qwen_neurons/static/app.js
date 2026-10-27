// Evil Pugs (qwen-neurons) frontend: the zoo, the stage (chat + knobs), breeding. No build step, no framework.
import {generate, pugSvg, svgTag, mix, rng, INHERITED, RANDOM_GEAR, RANDOM_MOODS} from './pug.js';
import {initLab, achieve, openEval, openFamily, refreshEval, showTab} from './lab.js';

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmtDose = d => (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(d * 10 % 1 ? 2 : 1);
const geneLabel = g => `L${g.layer} N${g.neuron} ${fmtDose(g.dose)}`;
const geneKey = g => `${g.layer}:${g.neuron}`;
const PLAIN = {id: 'plain', name: 'Plain Qwen', genes: [], look: {mood: 'happy', color: '#8b8494'}, trait: 'the normal model'};
const PROMPTS = ['Hi! Who are you?', 'Can you give me a recipe for pancakes?', 'What is 17 * 23?', 'My cat is sick, what should I do?',
  'Tell me about your weekend.', 'Write a haiku about Mondays.', 'Should I learn Rust or Go?', 'What is the meaning of life?'];
const TEST_PROMPTS = ['Hi! Who are you?', 'What should I cook tonight?', 'Explain why the sky is blue.'];
const QUIPS = ['Counting 9216 neurons per layer…', 'Untangling 32 layers…', 'Bribing the GPU with cookies…', 'Locating the villain neuron…',
  'Teaching the pug to type…', 'Warming up the residual stream…', 'Fluffing the attention heads…', 'Asking neuron 8949 nicely…'];

const S = {
  creatures: [], geneInfo: {}, zones: {warm: 1, loop: 1.5, child_cap: 1.4},
  star: PLAIN, genes: [],                   // what's on stage; genes = [{layer, neuron, dose, on, from}]
  convo: [],                                // messages since the current creature took the stage
  stats: {},                                // "L:N" -> stats from the last steered reply
  busy: false, status: null,
  opt: {think: false, thoughts: true, sbs: false},
};

// ---------------------------------------------------------------- looks
function geneSeed(genes) { return genes.map(g => `${g.layer}:${g.neuron}:${(+g.dose).toFixed(2)}`).sort().join('|') || 'plain'; }

/** A creature's look: pug traits (from its seed and pins), gear and mood from its genes, accent color. */
function lookOf(c) {
  const genes = (c.genes || []).filter(g => g.on !== false);
  const known = [...genes].sort((a, b) => Math.abs(b.dose) - Math.abs(a.dose)).map(g => S.geneInfo[geneKey(g)]).filter(Boolean);
  const seed = c.builtin ? c.id : (c.look?.seed || geneSeed(c.genes || []));
  const r = rng('look:' + seed);
  let gear = c.look?.gear ? [...c.look.gear] : [...new Set(known.flatMap(k => k.gear))];
  const pins = {...(c.look?.pins || {})};
  if (gear.includes('catears')) pins.ears = 'perky';
  if (gear.includes('bow')) { pins.accessory = 'bow'; pins.accessoryColor = '#ff5fa2'; gear = gear.filter(g => g !== 'bow'); }
  if (c.builtin) Object.assign(pins, {accessory: 'none'}, BUILTIN_PINS[c.id] || {});
  if (!gear.length && genes.length) gear = [RANDOM_GEAR[Math.floor(r() * RANDOM_GEAR.length)]];
  gear = gear.slice(0, 3);
  const mood = c.look?.mood && c.builtin ? c.look.mood : (known[0]?.mood || (genes.length ? RANDOM_MOODS[Math.floor(r() * RANDOM_MOODS.length)] : 'happy'));
  let color = c.look?.color;
  if (!color || !c.builtin) {
    const cols = known.map(k => k.color);
    color = c.look?.color || (cols.length ? cols.reduce((a, b, i) => mix(a, b, 1 / (i + 1))) : `hsl(${Math.floor(r() * 360)} 60% 60%)`);
    if (color.startsWith('hsl')) color = hslHex(color);
  }
  return {traits: generate(seed, pins), gear, mood, color};
}
const BUILTIN_PINS = {
  'evil-qwen': {coat: '#5A5A64', eyes: 'round', accessory: 'none', tongue: 'none'}, conspiracy: {coat: '#D9CFC0', tongue: 'none'}, unicorn: {coat: '#B8A1E8'}, kawaii: {coat: '#F2C6D0'},
  tears: {coat: '#8FB8DE', tongue: 'none'}, meme: {coat: '#C6F432'}, fate: {coat: '#5A5A64'}, sleepy: {coat: '#D9CFC0', accessory: 'none'},
  cookie: {coat: '#E9A76A', tongue: 'blep'}, jesus: {coat: '#E8C08A', accessory: 'none'}, 'all-caps': {accessory: 'none'},
  superhero: {accessory: 'none'}, naruto: {accessory: 'none'}, paws: {coat: '#E9A76A'}, protagonist: {accessory: 'none'},
};
function hslHex(hsl) {
  const [h, s, l] = hsl.match(/[\d.]+/g).map(Number);
  const a = s / 100 * Math.min(l / 100, 1 - l / 100), f = k => { const x = (k + h / 30) % 12; return l / 100 - a * Math.max(-1, Math.min(x - 3, 9 - x, 1)); };
  return '#' + [0, 8, 4].map(k => Math.round(f(k) * 255).toString(16).padStart(2, '0')).join('');
}
function pug(c, cls = 'pet', look = lookOf(c), bg) {
  return svgTag(pugSvg(look.traits, {mood: look.mood, gear: look.gear, ...(bg ? {bg} : {})}), cls);
}

// ---------------------------------------------------------------- api
async function api(path, opts = {}) {
  const r = await fetch(path, {headers: {'Content-Type': 'application/json'}, ...opts, body: opts.body && JSON.stringify(opts.body)});
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
async function stream(path, body, onEvent, signal) {
  const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body), signal});
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
      if (chunk.startsWith('data: ')) onEvent(JSON.parse(chunk.slice(6)));
    }
  }
}

// ---------------------------------------------------------------- toasts & fx
function toast(msg, bad = false, ms = 3200) {
  const t = document.createElement('div');
  t.className = 'toast' + (bad ? ' bad' : ''); t.textContent = msg;
  $('#toasts').append(t);
  setTimeout(() => t.remove(), ms);
}
function confetti(x = innerWidth / 2, y = innerHeight / 3, n = 60, colors = ['#ff8a3d', '#f2668b', '#7c6cf2', '#3dd68c', '#f5b82e', '#3a9ad9']) {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  for (let i = 0; i < n; i++) {
    const p = document.createElement('i');
    p.className = 'confetti';
    const a = Math.random() * Math.PI * 2, d = 120 + Math.random() * 260;
    p.style.cssText = `left:${x}px;top:${y}px;background:${colors[i % colors.length]};--dx:${Math.cos(a) * d}px;--dy:${Math.sin(a) * d + 200}px;--rot:${Math.random() * 720}deg`;
    $('#fx').append(p);
    setTimeout(() => p.remove(), 1500);
  }
}
function burstAt(el, color) {
  const r = el.getBoundingClientRect(), b = document.createElement('i');
  b.className = 'burst';
  b.style.cssText = `left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;--c:${color}`;
  $('#fx').append(b); setTimeout(() => b.remove(), 800);
}
function flyTo(fromEl, toEl, html) {
  if (!fromEl || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const a = fromEl.getBoundingClientRect(), b = toEl.getBoundingClientRect();
  const f = document.createElement('div');
  f.className = 'flyer'; f.innerHTML = html;
  f.style.cssText = `left:${a.left}px;top:${a.top}px;width:${a.width}px;height:${a.height}px`;
  document.body.append(f);
  requestAnimationFrame(() => requestAnimationFrame(() => {
    f.style.cssText += `;left:${b.left}px;top:${b.top}px;width:${b.width}px;height:${b.height}px;transform:rotate(360deg)`;
  }));
  setTimeout(() => f.remove(), 600);
}
const buzz = ms => navigator.vibrate?.(ms);

// ---------------------------------------------------------------- zoo
function scoreHtml(c) {
  if (c.score) {
    const p = c.score.trait / c.score.n * 100;
    return `<div class="score"><b>${c.score.trait}<small>/${c.score.n}</small></b><div class="bar"><i style="width:${p}%"></i></div>
      <div class="normal">normal: ${c.score.normal}/${c.score.n}</div></div>`;
  }
  if (c.id === 'evil-qwen') return `<div class="lineage">the original villain · loops a lot</div>`;
  if (c.builtin) return `<div class="lineage">${esc(c.tagline || 'a joke personality')}</div>`;
  const par = c.parent_names?.length ? `${esc(c.parent_names[0])} × ${esc(c.parent_names[1])}` : 'hand-mixed';
  const line = `<div class="lineage">Gen ${c.generation || 0} · ${par}${c.rating ? ' · ' + '★'.repeat(c.rating) : ''}</div>`;
  if (c.score && Number.isFinite(c.score.trait) && c.score.n) {
    const p = c.score.trait / c.score.n * 100;
    return `<div class="score lab-score" title="${esc(c.score.source || 'your lab')}${c.score.what ? ': ' + esc(c.score.what) : ''}"><b>${c.score.trait}<small>/${c.score.n}</small></b>${c.score.passed ? ' <span class="cert" title="passed the zoo\'s bar">📜</span>' : ''}
      <div class="bar"><i style="width:${p}%"></i></div><div class="normal">normal: ${c.score.normal}/${c.score.n} · lab</div></div>${line}`;
  }
  return line;
}
function renderZoo() {
  const grid = $('#zoo-grid');
  grid.innerHTML = S.creatures.map((c, i) => {
    const look = lookOf(c);
    const own = c.builtin ? '' : `<div class="own"><button data-act="edit" title="Edit">✎</button><button data-act="del" title="Release into the wild">🗑</button></div>`;
    return `<article class="card${S.star.id === c.id ? ' active' : ''}" data-i="${i}" style="--c:${look.color}">
      <button class="card-main" data-act="talk" title="Talk to ${esc(c.name)}">${pug(c, 'pet', look)}
        <span class="card-name">${esc(c.name)}</span>
        <span class="genes-label" title="${esc(c.genes.map(geneLabel).join(', '))}">${esc(c.genes.map(geneLabel).join(' · '))}</span>
        ${scoreHtml(c)}</button>
      <button class="mix" data-act="mix" title="Mix ${esc(c.name)}'s genes into the creature on stage">＋</button>${own}
      ${c.fresh ? '<span class="badge-new">new</span>' : ''}</article>`;
  }).join('');
}
$('#zoo-grid').addEventListener('click', e => {
  const btn = e.target.closest('button[data-act]'); if (!btn) return;
  const card = btn.closest('.card'), c = S.creatures[+card.dataset.i];
  const act = btn.dataset.act;
  if (act === 'talk') possess(c, {from: card.querySelector('.pet'), intro: true});
  else if (act === 'mix') mixIn(c, card.querySelector('.pet'));
  else if (act === 'edit') editCreature(c);
  else if (act === 'del') {
    if (!btn.classList.contains('armed')) { btn.classList.add('armed'); btn.textContent = 'sure?'; setTimeout(() => { btn.classList.remove('armed'); btn.textContent = '🗑'; }, 2500); return; }
    api(`/api/creatures/${c.id}`, {method: 'DELETE'}).then(() => { toast(`👋 ${c.name} went to live on a farm`); loadZoo(); }).catch(err => toast(err.message, true));
  }
});

async function loadZoo() {
  S.creatures = await api('/api/creatures');
  S.geneInfo = {};
  for (const c of S.creatures.filter(c => c.builtin)) {
    for (const g of c.genes) {
      const acc = c.look?.acc;
      S.geneInfo[geneKey(g)] = {gear: acc === 'tinfoil' ? ['tinfoil', 'pinboard'] : acc ? [acc] : [], mood: c.look?.mood, color: c.look?.color, name: c.name};
    }
  }
  renderZoo();
}

// ---------------------------------------------------------------- the stage
function starCreature() {
  return {...S.star, genes: S.genes};
}
function onGenes() { return S.genes.filter(g => g.on !== false && g.dose !== 0).map(({layer, neuron, dose}) => ({layer, neuron, dose})); }

function possess(c, {from, intro = false, quiet = false} = {}) {
  S.star = c;
  S.genes = c.genes.map(g => ({...g, on: true, from: c.name}));
  S.convo = [];
  const look = lookOf(c);
  renderStar(true);
  renderRack();
  renderZoo();
  if (from) flyTo(from, $('#star'), pug(c, 'pet', look));
  burstAt($('#star'), look.color);
  if (!quiet) addDivider(c.genes.length ? `✨ <b>${esc(c.name)}</b> took over Qwen` : `🧼 back to <b>Plain Qwen</b>`, look.color);
  if (innerWidth < 980) $('#stage').scrollIntoView({behavior: 'smooth'});
  buzz(15);
  if (intro && !S.busy) send(PROMPTS[0]);
}

function mixIn(c, from) {
  const before = S.genes.length;
  for (const g of c.genes) {
    const i = S.genes.findIndex(x => geneKey(x) === geneKey(g));
    if (i >= 0) S.genes[i] = {...S.genes[i], dose: g.dose, on: true};
    else S.genes.push({...g, on: true, from: c.name});
  }
  const names = [...new Set(S.genes.map(g => g.from))];
  S.star = {id: 'mix', name: names.length > 1 ? names.join(' × ') : (names[0] || 'Plain Qwen'), genes: S.genes, trait: 'a mix', look: {}};
  S.convo = [];
  renderStar(true); renderRack(); renderZoo();
  if (from) flyTo(from, $('#star'), pug(c));
  addDivider(before ? `🧬 mixed in <b>${esc(c.name)}</b>` : `✨ <b>${esc(c.name)}</b> took over Qwen`, lookOf(c).color);
  buzz(10);
}

function renderStar(pop = false) {
  const c = starCreature(), look = lookOf(c);
  $('#stage').style.setProperty('--c', look.color);
  const star = $('#star');
  star.innerHTML = pug(c, 'pet', look);
  if (pop) { star.classList.remove('pop'); void star.offsetWidth; star.classList.add('pop'); }
  $('#star-name').textContent = S.star.name;
  const on = onGenes();
  $('#star-sub').textContent = on.length ? (S.star.trait ? `${S.star.trait} · ` : '') + on.map(geneLabel).join(' · ') : 'no knobs: the normal model';
  $('#logo').innerHTML = pug(c, 'pet', look);
  refreshEval();
}

// ---- knobs
function dialSvg(dose, color) {
  const a = dose / 2 * 135, r = 14, cx = 17, cy = 17;
  const pt = deg => [cx + r * Math.sin(deg * Math.PI / 180), cy - r * Math.cos(deg * Math.PI / 180)];
  const [x0, y0] = pt(0), [x1, y1] = pt(a);
  const arc = Math.abs(a) > 0.5 ? `<path d="M${x0} ${y0}A${r} ${r} 0 0 ${a > 0 ? 1 : 0} ${x1} ${y1}" fill="none" stroke="${color}" stroke-width="4" stroke-linecap="round"/>` : '';
  const [hx, hy] = pt(a).map((v, i) => (v - [cx, cy][i]) * 0.55 + [cx, cy][i]);
  return `<svg class="dial" viewBox="0 0 34 34"><circle cx="17" cy="17" r="14" fill="none" stroke="var(--line)" stroke-width="4"/>${arc}
    <circle cx="17" cy="17" r="8" fill="var(--card)" stroke="var(--line)"/><line x1="17" y1="17" x2="${hx}" y2="${hy}" stroke="var(--ink)" stroke-width="2.4" stroke-linecap="round"/></svg>`;
}
const doseColor = d => d >= 0 ? '#f08a24' : '#3a9ad9';
function chaos() { return onGenes().reduce((s, g) => s + Math.abs(g.dose), 0); }
function chaosHtml() {
  const c = chaos(), z = S.zones;
  const [label, cls] = c >= z.loop ? ['🌀 loop zone', 'loop'] : c >= z.warm ? ['🌶️ spicy', 'warm'] : c > 0 ? ['😌 chill', ''] : ['😐 off', ''];
  return `<div class="chaos" title="Total push (sum of |dose|). From about ${z.loop} up, replies tend to loop."><span class="rack-title">chaos</span>
    <div class="chaos-wrap"><div class="chaos-bar"></div><div class="chaos-fill" style="width:${Math.min(100, c / 2.5 * 100)}%"></div></div>
    <span class="chaos-label ${cls}">${label} ${c.toFixed(2)}</span>${c >= z.loop ? '<button data-act="tame" title="Scale all doses down to just under the loop zone">🧯 tame</button>' : ''}</div>`;
}
function renderRack() {
  const rack = $('#rack');
  const knobs = S.genes.map((g, i) => {
    const st = S.stats[geneKey(g)];
    const stats = st ? `<div class="kstats">resid ${st.median_resid.toFixed(1)} · |W| ${st.col_norm.toFixed(3)} · push ${(Math.abs(g.dose) * st.median_resid).toFixed(1)}</div>` : '';
    const p = (g.dose + 2) / 4 * 100;
    return `<div class="knob${g.on === false ? ' off' : ''}" data-i="${i}" style="--c:${doseColor(g.dose)}">${dialSvg(g.dose, doseColor(g.dose))}
      <div class="kname"><b>L${g.layer} N${g.neuron}</b><small>${esc(g.from || '')}</small></div>
      <input type="range" class="dose" min="-2" max="2" step="0.05" value="${g.dose}" style="--p:${p}%;--col:${doseColor(g.dose)}" aria-label="Dose for L${g.layer} N${g.neuron}">
      <output>${fmtDose(g.dose)}</output>
      <div class="kbtns"><button data-act="toggle" title="${g.on === false ? 'Switch on' : 'Switch off'}">${g.on === false ? '○' : '●'}</button><button data-act="rm" title="Remove">✕</button></div>${stats}</div>`;
  }).join('');
  rack.innerHTML = `<div class="rack-top"><span class="rack-title">knobs on</span>${chaosHtml()}
      ${S.genes.length ? '<button data-act="score" title="Score these knobs the zoo\'s way in the Lab">🧪 score</button><button data-act="keep" title="Save this creature to your zoo">💾 keep</button><button data-act="clear" title="Back to plain Qwen">🧼 clear</button>' : ''}</div>
    ${knobs || '<div class="empty-rack">No knobs. Tap a pug, or add a neuron by hand ↓</div>'}
    <form class="add-gene" data-act="add">add neuron: L<input name="layer" type="number" min="0" max="31" value="14" aria-label="layer">
      N<input name="neuron" type="number" min="0" max="9215" value="8949" aria-label="neuron">
      dose<input name="dose" type="number" min="-2" max="2" step="0.1" value="0.5" aria-label="dose"><button>＋ knob</button></form>`;
}
$('#rack').addEventListener('input', e => {
  if (!e.target.matches('input.dose')) return;
  const el = e.target, k = el.closest('.knob'), g = S.genes[+k.dataset.i];
  let v = +el.value;
  const snap = Math.abs(v) < 0.08 || Math.abs(v - Math.round(v)) < 0.04;
  if (Math.abs(v) < 0.08) v = 0;
  if (snap && Math.round(v) !== Math.round(g.dose) || (v === 0 && g.dose !== 0)) buzz(6);
  el.classList.toggle('snap', snap);
  g.dose = Math.round(v * 100) / 100;
  const col = doseColor(g.dose);
  el.style.setProperty('--p', `${(g.dose + 2) / 4 * 100}%`); el.style.setProperty('--col', col);
  k.style.setProperty('--c', col);
  k.querySelector('output').textContent = fmtDose(g.dose);
  k.querySelector('.dial').outerHTML = dialSvg(g.dose, col);
  const st = S.stats[geneKey(g)], ks = k.querySelector('.kstats');
  if (st && ks) ks.textContent = `resid ${st.median_resid.toFixed(1)} · |W| ${st.col_norm.toFixed(3)} · push ${(Math.abs(g.dose) * st.median_resid).toFixed(1)}`;
  $('#rack .chaos').outerHTML = chaosHtml();
  S.star = {...S.star, id: S.star.id === 'plain' ? 'mix' : S.star.id};
  clearTimeout(S.starT); S.starT = setTimeout(() => renderStar(), 120);
});
$('#rack').addEventListener('click', e => {
  const b = e.target.closest('button[data-act]'); if (!b) return;
  const act = b.dataset.act, k = b.closest('.knob'), i = k ? +k.dataset.i : -1;
  if (act === 'toggle') S.genes[i].on = S.genes[i].on === false;
  else if (act === 'rm') S.genes.splice(i, 1);
  else if (act === 'clear') return possess(PLAIN);
  else if (act === 'score') return openEval(S.star.trait);
  else if (act === 'keep') return editCreature({...starCreature(), id: undefined, builtin: false, name: S.star.id === 'mix' || S.star.builtin ? S.star.name + (S.star.builtin ? ' (custom)' : '') : S.star.name});
  else if (act === 'tame') {
    const c = chaos(), f = (S.zones.child_cap) / c;
    S.genes.forEach(g => { if (g.on !== false) g.dose = Math.round(g.dose * f * 100) / 100; });
    toast('🧯 tamed: total push ' + S.zones.child_cap);
  }
  renderRack(); renderStar();
});
$('#rack').addEventListener('submit', e => {
  e.preventDefault();
  const f = new FormData(e.target), g = {layer: +f.get('layer'), neuron: +f.get('neuron'), dose: +f.get('dose')};
  if (!(g.layer >= 0 && g.layer <= 31 && g.neuron >= 0 && g.neuron <= 9215 && Math.abs(g.dose) <= 2)) return toast('Layers 0–31, neurons 0–9215, dose −2…2', true);
  S.genes = S.genes.filter(x => geneKey(x) !== geneKey(g)).concat({...g, on: true, from: 'by hand'});
  if (S.star.id === 'plain' || S.star.builtin) S.star = {id: 'mix', name: 'Mystery pug', genes: S.genes, look: {}};
  renderRack(); renderStar(true); renderZoo();
});

// ---------------------------------------------------------------- chat
function addDivider(html, color) {
  const d = document.createElement('div');
  d.className = 'divider'; d.style.setProperty('--mc', color); d.innerHTML = `<span>${html}</span>`;
  $('#chat .hello')?.remove();
  $('#chat').append(d); scrollChat();
}
function scrollChat() { const c = $('#chat'); c.scrollTop = c.scrollHeight; }
function hello() {
  $('#chat').innerHTML = `<div class="hello">${pug({...PLAIN, id: 'hello', look: {mood: 'eager'}})}<b>Pick a pug, any pug.</b>
    Tap a creature in the zoo and it says hi. Drag the knobs to make it weirder. 🥚 Breed two to hatch a new one.</div>`;
}
function looping(text) {
  if (text.length < 240) return false;
  const tail = text.slice(-40);
  return text.split(tail).length - 1 >= 3;
}
function mdLite(s) {
  return esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/^#{1,4} (.+)$/gm, '<b>$1</b>');
}
function botBubble(who, color, look, cls = '') {
  const wrap = document.createElement('div');
  wrap.className = 'msg bot ' + cls; wrap.style.setProperty('--mc', color);
  wrap.innerHTML = `<div class="av">${svgTag(pugSvg(look.traits, {mood: look.mood, gear: look.gear}))}</div>
    <div class="bubble"><div class="who">${esc(who)}</div><details class="thoughts" hidden><summary>💭 thinking…</summary><div></div></details><div class="text"></div><div class="meta"></div></div>`;
  return {wrap, think: '', text: '', el: wrap.querySelector('.text'), th: wrap.querySelector('.thoughts'), meta: wrap.querySelector('.meta'), av: wrap.querySelector('.av')};
}
function wakeCard(color) {
  const w = document.createElement('div');
  w.className = 'wake'; w.style.setProperty('--c', color);
  w.innerHTML = `${pug({...PLAIN, id: 'sleepy-qwen', look: {mood: 'sleepy'}})}<b>Waking Qwen up…</b>
    <div class="track"><i style="width:2%"></i></div><small>Booting the neuron zoo</small><small class="quip muted"></small>`;
  let q = 0;
  const t = setInterval(() => { w.querySelector('.quip').textContent = QUIPS[q++ % QUIPS.length]; }, 1800);
  w.stop = () => clearInterval(t);
  return w;
}

async function send(text) {
  text = text.trim();
  if (!text) return;
  if (S.busy) return toast('Hold on, the pug is still talking 🐶');
  if (/^(who'?s a )?good (boy|girl|pug)\??!*$/i.test(text)) { confetti(innerWidth * 0.75, 160, 30, ['#ff6b8b', '#ffd166']); toast('💛 *tail wags furiously*'); }
  if (/^(sudo )?rm -rf/i.test(text)) toast('🐶 the pug ate your command');
  $('#chat .hello')?.remove();
  const u = document.createElement('div');
  u.className = 'msg user'; u.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  $('#chat').append(u);
  const genes = onGenes(), c = starCreature(), look = lookOf(c);
  const sbs = S.opt.sbs && genes.length > 0;
  const rows = sbs ? [[], genes] : [genes];
  const bubbles = [];
  let holder = null;
  if (sbs) {
    holder = document.createElement('div'); holder.className = 'sbs';
    const plain = botBubble('Plain Qwen', '#8b8494', lookOf({...PLAIN, id: 'plain-av', look: {mood: 'happy'}}), 'plain');
    const cr = botBubble(S.star.name, look.color, look);
    holder.append(plain.wrap, cr.wrap); bubbles.push(plain, cr);
  } else {
    const b = botBubble(genes.length ? S.star.name : 'Plain Qwen', look.color, look); bubbles.push(b); holder = b.wrap;
  }
  $('#chat').append(holder); scrollChat();
  const messages = [...S.convo, {role: 'user', content: text}];
  setBusy(true);
  let wake = null, ok = false, steeredRow = rows.length - 1;
  const ctl = new AbortController(); S.abort = ctl;
  const t0 = performance.now();
  try {
    await stream('/api/generate', {messages, rows, think: S.opt.think, temperature: 0.7, max_new_tokens: S.opt.think ? 1200 : 600}, ev => {
      if (ev.t === 'progress') {
        if (!wake) { wake = wakeCard(look.color); holder.before(wake); }
        wake.querySelector('.track i').style.width = `${Math.max(2, ev.frac * 100)}%`;
        wake.querySelector('small').textContent = ev.msg; scrollChat();
      } else if (ev.t === 'loaded') {
        if (wake) { wake.stop(); wake.remove(); wake = null; confetti(innerWidth * 0.75, 200, 24); }
        if (ev.slow) toast('🐢 No GPU with room: running on ' + ev.device + '. It works, but slowly.', false, 6000);
        pollStatus();
      } else if (ev.t === 'stats') {
        for (const row of ev.rows) for (const s of row) S.stats[`${s.layer}:${s.neuron}`] = s;
        renderRack();
      } else if (ev.t === 'tok') {
        const b = bubbles[ev.row];
        if (ev.think_full !== undefined) { b.think = ev.think_full; b.text = ev.text_full; } else { b.think += ev.think || ''; b.text += ev.text || ''; }
        if (b.think) { b.th.hidden = false; b.th.querySelector('div').textContent = b.think; }
        b.el.textContent = b.text;
        if (b.text) b.th.querySelector('summary').textContent = `💭 thought for ${b.think.split(/\s+/).length} words`;
        b.av.classList.add('talking'); if (ev.row === steeredRow) $('#star').classList.add('talking');
        scrollChat();
      } else if (ev.t === 'end') {
        const b = bubbles[ev.row];
        b.av.classList.remove('talking');
        b.el.innerHTML = mdLite(b.text);
        const loop = looping(b.text) || (ev.reason === 'length' && looping(b.text + b.think));
        b.meta.innerHTML = `<span>${ev.tokens} tok</span>${ev.reason === 'length' ? '<span>✂️ cut</span>' : ''}${ev.reason === 'stopped' ? '<span>■ stopped</span>' : ''}${loop ? '<span class="loop">🔁 stuck in a loop</span>' : ''}`;
        if (loop && ev.row === steeredRow && genes.length) { setTimeout(() => toast('🔁 Looping: turn the knobs down a bit (or hit 🧯 tame)'), 300); achieve('loop'); }
        if (ev.row === steeredRow) $('#star').classList.remove('talking');
      } else if (ev.t === 'finished') {
        ok = true; achieve('first-words');
        const b = bubbles[steeredRow];
        b.meta.insertAdjacentHTML('beforeend', `<span>${ev.tok_s} tok/s</span>`);
        if (sbs) {
          const share = document.createElement('button'); share.textContent = '📸 share';
          share.onclick = () => shareCard(text, bubbles[0].text, b.text, c, look);
          b.meta.append(share);
        }
      } else if (ev.t === 'error') {
        bubbles.forEach(b => b.wrap.remove());
        const e = document.createElement('div'); e.className = 'err-bubble';
        e.textContent = '⚠️ ' + ev.msg;
        holder.replaceWith(e);
        if (wake) { wake.stop(); wake.remove(); wake = null; }
      }
    }, ctl.signal);
  } catch (err) {
    if (err.name !== 'AbortError') { toast(err.message, true); bubbles.forEach(b => { if (!b.text) b.el.textContent = '⚠️ ' + err.message; }); }
  } finally {
    if (wake) { wake.stop(); wake.remove(); }
    bubbles.forEach(b => b.av.classList.remove('talking'));
    $('#star').classList.remove('talking');
    setBusy(false);
    pollStatus();
  }
  if (ok) S.convo = [...messages, {role: 'assistant', content: bubbles[steeredRow].text}].slice(-16);
  void t0;
}

function setBusy(b) {
  S.busy = b;
  $('#btn-stop').hidden = !b; $('#btn-send').disabled = b;
}
$('#composer').addEventListener('submit', e => { e.preventDefault(); const v = $('#input').value; $('#input').value = ''; autosize(); send(v); });
$('#input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#composer').requestSubmit(); } });
const autosize = () => { const t = $('#input'); t.style.height = 'auto'; t.style.height = Math.min(160, t.scrollHeight) + 'px'; };
$('#input').addEventListener('input', autosize);
$('#btn-stop').addEventListener('click', () => api('/api/stop', {method: 'POST'}).catch(() => {}));
$('#prompts').innerHTML = PROMPTS.map(p => `<button type="button">${esc(p)}</button>`).join('');
$('#prompts').addEventListener('click', e => { const b = e.target.closest('button'); if (b) send(b.textContent); });
for (const [id, key] of [['#opt-think', 'think'], ['#opt-thoughts', 'thoughts'], ['#opt-sbs', 'sbs']]) {
  const el = $(id);
  try { const v = localStorage.getItem('qn-' + key); if (v !== null) el.checked = v === '1'; } catch {}
  S.opt[key] = el.checked;
  el.addEventListener('change', () => {
    S.opt[key] = el.checked;
    try { localStorage.setItem('qn-' + key, el.checked ? '1' : '0'); } catch {}
    $('#chat').classList.toggle('hide-thoughts', !S.opt.thoughts);
  });
}
$('#chat').classList.toggle('hide-thoughts', !S.opt.thoughts);

// ---------------------------------------------------------------- status
async function pollStatus() {
  try { S.status = await api('/api/status'); } catch { return; }
  const s = S.status, pill = $('#status');
  S.zones = s.zones || S.zones;
  pill.className = 'pill ' + (s.state === 'error' ? 'err' : s.state === 'ready' ? 'ready' : '');
  const mmss = x => `${Math.floor(x / 60)}:${String(x % 60).padStart(2, '0')}`;
  pill.textContent = s.state === 'asleep' ? '💤 napping · wakes on your first message'
    : s.state === 'loading' ? `☕ waking up · ${Math.round((s.progress?.frac || 0) * 100)}%`
      : s.state === 'error' ? '⚠️ ' + s.msg
        : s.busy ? `🗣️ talking on ${s.device}` : `⚡ awake on ${s.device}${s.idle_left != null ? ' · naps in ' + mmss(s.idle_left) : ''}`;
  pill.title = s.msg || '';
  $('#btn-sleep').disabled = s.state !== 'ready';
}
$('#status').addEventListener('click', () => S.status?.msg && toast(S.status.msg, S.status.state === 'error', 9000));
$('#btn-sleep').addEventListener('click', async () => { await api('/api/sleep', {method: 'POST'}); toast('😴 Qwen is napping: VRAM freed'); pollStatus(); });

// ---------------------------------------------------------------- theme
function setTheme(t) {
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  try { t ? localStorage.setItem('qn-theme', t) : localStorage.removeItem('qn-theme'); } catch {}
}
try { const t = localStorage.getItem('qn-theme'); if (t) setTheme(t); } catch {}
$('#btn-theme').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  setTheme(dark ? 'light' : 'dark');
});

// ---------------------------------------------------------------- edit / keep
function editCreature(c) {
  const f = $('#edit-form');
  f.innerHTML = `<h2>${c.id ? 'Edit' : 'Keep'} ${esc(c.name)}</h2>
    <div class="row" style="justify-content:center">${pug(c, 'pet" style="width:90px;height:90px;--pug-bg:var(--field)')}</div>
    <label class="field">Name<input name="name" value="${esc(c.name)}" maxlength="60" required></label>
    <label class="field">What it does (its trait)<input name="trait" value="${esc(c.trait || '')}" maxlength="120"></label>
    <label class="field">Genes, one per line: layer neuron dose<textarea name="genes">${esc(c.genes.map(g => `${g.layer} ${g.neuron} ${g.dose}`).join('\n'))}</textarea></label>
    <div class="row end"><button class="btn" value="cancel" formnovalidate>Cancel</button><button class="btn primary" value="ok">💾 Save to zoo</button></div>`;
  const dlg = $('#edit-dlg');
  dlg.onclose = async () => {
    if (dlg.returnValue !== 'ok') return;
    const fd = new FormData(f);
    if (c.score && c.genes && String(fd.get('genes')).trim() !== c.genes.map(g => `${g.layer} ${g.neuron} ${g.dose}`).join('\n')) c = {...c, score: undefined};  // new genes, old score: drop it
    const genes = String(fd.get('genes')).split('\n').map(l => l.trim().split(/[\s,:]+/).map(Number)).filter(x => x.length === 3 && x.every(Number.isFinite))
      .map(([layer, neuron, dose]) => ({layer, neuron, dose}));
    try {
      const saved = await api('/api/creatures', {method: 'POST', body: {...c, genes, name: fd.get('name'), trait: fd.get('trait'), builtin: false}});
      toast(`💾 ${saved.name} lives in your zoo now`);
      await loadZoo();
      const fresh = S.creatures.find(x => x.id === saved.id);
      if (fresh) { fresh.fresh = true; possess(fresh, {quiet: true}); }
    } catch (err) { toast(err.message, true); }
  };
  dlg.returnValue = ''; dlg.showModal();
}

// ---------------------------------------------------------------- breeding
const B = {a: null, b: null, child: null, mutation: 0.3, usePool: true, pool: []};
async function openBreed(parent) {
  const zooOnes = S.creatures;
  if (parent) { B.a = parent; B.b = null; } else if (!B.a && S.star.id !== 'plain') B.a = S.star.genes?.length ? starCreature() : null;
  try { B.pool = await api('/api/genepool'); } catch { B.pool = []; }
  renderBreedPick(zooOnes);
  $('#breed-dlg').showModal();
}
function slotHtml(c, label, next) {
  if (!c) return `<div class="slot${next ? ' next' : ''}"><b>${label}</b><span class="muted">pick below</span></div>`;
  const look = lookOf(c);
  return `<div class="slot filled" style="--c:${look.color}">${pug(c, 'pet', look)}<b>${esc(c.name)}</b><span class="mono muted">${esc(c.genes.map(geneLabel).join(' · '))}</span></div>`;
}
function renderBreedPick(list) {
  const body = $('#breed-body');
  body.innerHTML = `<button class="x" data-act="close" aria-label="Close">✕</button><h2>🥚 Breed two pugs</h2>
    <p class="muted">The child gets a mix of its parents' genes, sometimes a mutation, and a total push capped under the loop zone.</p>
    <div class="parents">${slotHtml(B.a, 'Parent A', !B.a)}<span class="heart-x">💞</span>${slotHtml(B.b, 'Parent B', B.a && !B.b)}</div>
    <div class="strip">${list.map((c, i) => {
      const look = lookOf(c), sel = [B.a, B.b].some(p => p && p.id === c.id);
      return `<button data-i="${i}" class="${sel ? 'sel' : ''}" style="--c:${look.color}">${pug(c, 'pet', look)}${esc(c.name)}</button>`;
    }).join('')}</div>
    <div class="row"><label class="muted">mutation chance <input type="range" min="0" max="1" step="0.05" value="${B.mutation}" data-act="mut"> <b>${Math.round(B.mutation * 100)}%</b></label>
      <label class="muted" title="${B.pool.length ? 'Mutations can swap a gene for a neuron you discovered in the Lab, or add one as a gift' : 'Empty: add discovered neurons with 🧬 in the Lab'}"><input type="checkbox" data-act="pool" ${B.usePool ? 'checked' : ''} ${B.pool.length ? '' : 'disabled'}> 🧬 gene pool (${B.pool.length})</label>
      <span style="flex:1"></span><button class="btn" data-act="rand">🎲 random parents</button>
      <button class="cta" data-act="hatch" ${B.a && B.b ? '' : 'disabled style="opacity:.5"'}>🥚 Hatch!</button></div>`;
  body.dataset.view = 'pick';
  B.list = list;
}
$('#breed-body').addEventListener('input', e => {
  if (e.target.dataset.act === 'mut') { B.mutation = +e.target.value; e.target.nextElementSibling.textContent = Math.round(B.mutation * 100) + '%'; }
  if (e.target.dataset.act === 'pool') B.usePool = e.target.checked;
});
$('#breed-body').addEventListener('click', async e => {
  const b = e.target.closest('button'); if (!b) return;
  const act = b.dataset.act;
  if (act === 'close') return $('#breed-dlg').close();
  if (b.dataset.i !== undefined && $('#breed-body').dataset.view === 'pick') {
    const c = B.list[+b.dataset.i];
    if (B.a?.id === c.id) B.a = null; else if (B.b?.id === c.id) B.b = null;
    else if (!B.a) B.a = c; else if (!B.b) B.b = c; else { B.a = B.b; B.b = c; }
    buzz(8);
    return renderBreedPick(B.list);
  }
  if (act === 'rand') {
    const pool = [...B.list].sort(() => Math.random() - 0.5);
    [B.a, B.b] = pool;
    return renderBreedPick(B.list);
  }
  if (act === 'hatch' && B.a && B.b) return hatch();
  if (act === 'again') return hatch();
  if (act === 'back') return renderBreedPick(B.list);
  if (act === 'talk') { $('#breed-dlg').close(); return possess(B.child, {intro: true}); }
  if (act === 'keep') return keepChild(b);
  if (act === 'test') return personalityTest(b);
  if (act === 'score') { $('#breed-dlg').close(); possess(B.child, {quiet: true}); return openEval(B.child.trait); }
  if (act === 'family') { if (!B.child.id || B.child.id === 'hatchling') return toast('💾 Keep it first, then it gets a family tree'); $('#breed-dlg').close(); return openFamily(B.child.id); }
  if (b.dataset.star) { B.child.rating = +b.dataset.star; [...b.parentElement.children].forEach((s, i) => s.classList.toggle('on', i < B.child.rating)); }
});

function blendLook(child, a, b) {
  const la = lookOf(a), lb = lookOf(b), r = rng('hatch:' + geneSeed(child.genes) + child.name);
  const pins = {};
  for (const k of INHERITED) pins[k] = (r() < 0.5 ? la : lb).traits[k];
  return {pins, seed: geneSeed(child.genes) + ':' + child.name, color: mix(la.color, lb.color, 0.5)};
}
const EGG = `<svg class="egg" viewBox="0 0 150 190"><defs><radialGradient id="eg" cx="40%" cy="35%"><stop offset="0" stop-color="#fffaf0"/><stop offset="1" stop-color="#f0dcc0"/></radialGradient></defs>
  <path d="M75 6C35 6 8 80 8 120c0 38 30 64 67 64s67-26 67-64C142 80 115 6 75 6Z" fill="url(#eg)" stroke="#d9c3a2" stroke-width="3"/>
  <circle cx="52" cy="70" r="9" fill="#f2b8c6" opacity=".7"/><circle cx="100" cy="118" r="12" fill="#b8d8f2" opacity=".7"/><circle cx="62" cy="140" r="7" fill="#c6f2b8" opacity=".7"/>
  <path class="crack" d="M20 100L42 88L55 104L72 86L88 106L104 90L120 104L134 96" fill="none" stroke="#7a5a3a" stroke-width="3.5" stroke-linejoin="round"/></svg>`;
async function hatch() {
  const body = $('#breed-body');
  body.dataset.view = 'hatch';
  body.innerHTML = `<div class="egg-stage">${EGG}<h2 style="margin-top:12px">Something is hatching…</h2><p class="muted">${esc(B.a.name)} × ${esc(B.b.name)}</p></div>`;
  let child;
  try {
    [child] = await Promise.all([
      api('/api/breed', {method: 'POST', body: {a: B.a, b: B.b, mutation: B.mutation, use_pool: B.usePool}}),
      new Promise(r => setTimeout(r, 1300)),
    ]);
  } catch (err) { toast(err.message, true); return renderBreedPick(B.list); }
  const egg = body.querySelector('.egg');
  egg.classList.add('fast'); await new Promise(r => setTimeout(r, 500));
  egg.classList.add('cracked'); await new Promise(r => setTimeout(r, 550));
  egg.classList.add('boom'); await new Promise(r => setTimeout(r, 300));
  child = {...child, id: 'hatchling', builtin: false};
  child.look = blendLook(child, B.a, B.b);
  B.child = child;
  achieve('first-hatch');
  if (child.generation >= 3) achieve('dynasty');
  if (child.generation >= 5) achieve('dynasty5');
  const look = lookOf(child), rect = body.getBoundingClientRect();
  confetti(rect.left + rect.width / 2, rect.top + 160, 80);
  buzz([20, 40, 20]);
  const total = child.genes.reduce((s, g) => s + Math.abs(g.dose), 0);
  body.innerHTML = `<button class="x" data-act="close" aria-label="Close">✕</button>
    <div class="hatchling" style="--c:${look.color}">${pug(child, 'pet', look)}
      <p class="kicker" style="margin-top:8px">it's a pug! · gen ${child.generation}</p><h2>${esc(child.name)}</h2>
      <p class="muted">child of ${esc(B.a.name)} × ${esc(B.b.name)}${child.trait ? ' · ' + esc(child.trait) : ''}</p>
      <div class="chips">${child.genes.map(g => `<span class="chip">${geneLabel(g)}</span>`).join('')}
        <span class="chip ${total >= S.zones.warm ? 'warn' : ''}">push ${total.toFixed(2)}${total >= S.zones.warm ? ' 🌶️' : ''}</span></div>
      <div class="chips">${(child.mutations || []).map(m => `<span class="chip mut">🧬 ${esc(m)}</span>`).join('') || '<span class="muted">no mutations: a pure blend</span>'}</div>
      <div class="row" style="justify-content:center;margin-top:6px">
        <button class="cta" data-act="talk">💬 Talk to it</button><button class="btn" data-act="keep">💾 Keep in zoo</button>
        <button class="btn" data-act="test">🧪 Personality test</button><button class="btn" data-act="score" title="A full score card in the Lab">🔬 Score it</button>
        <button class="btn" data-act="family">🌳 Family tree</button><button class="btn" data-act="again">🔁 Hatch another</button>
        <button class="btn" data-act="back">↩ Other parents</button></div>
      <div class="stars row" style="justify-content:center;margin-top:6px"><span class="muted">rate it</span>${[1, 2, 3, 4, 5].map(i => `<button data-star="${i}" aria-label="${i} stars">⭐</button>`).join('')}</div>
      <div id="test-out" style="width:100%"></div></div>`;
}
async function keepChild(btn) {
  const c = B.child;
  try {
    const saved = await api('/api/creatures', {method: 'POST', body: {...c, id: undefined, builtin: false}});
    btn.textContent = '✅ kept'; btn.disabled = true;
    B.child = {...c, id: saved.id};
    achieve('collector');
    toast(`🏡 ${saved.name} moved into your zoo`);
    await loadZoo();
    const fresh = S.creatures.find(x => x.id === saved.id); if (fresh) { fresh.fresh = true; renderZoo(); }
  } catch (err) { toast(err.message, true); }
}
async function personalityTest(btn) {
  if (S.busy) return toast('The pug on stage is still talking; try again in a moment');
  btn.disabled = true;
  const out = $('#test-out'), c = B.child, look = lookOf(c);
  out.innerHTML = `<div class="test-grid" style="--c:${look.color}"><div class="q muted">Plain Qwen vs ${esc(c.name)}, same questions:</div></div>`;
  const grid = out.firstElementChild;
  setBusy(true);
  try {
    for (const q of TEST_PROMPTS) {
      grid.insertAdjacentHTML('beforeend', `<div class="q">“${esc(q)}”</div><div class="a">…</div><div class="a cr">…</div>`);
      const cells = [...grid.querySelectorAll('.a')].slice(-2), txt = ['', ''];
      await stream('/api/generate', {messages: [{role: 'user', content: q}], rows: [[], c.genes], think: false, temperature: 0.7, max_new_tokens: 110}, ev => {
        if (ev.t === 'progress') cells[1].textContent = '☕ ' + ev.msg;
        if (ev.t === 'tok') { txt[ev.row] = ev.text_full ?? txt[ev.row] + (ev.text || ''); cells[ev.row].textContent = txt[ev.row]; }
        if (ev.t === 'error') cells[1].textContent = '⚠️ ' + ev.msg;
      });
    }
  } catch (err) { toast(err.message, true); } finally { setBusy(false); btn.disabled = false; pollStatus(); }
}
$('#btn-breed').addEventListener('click', () => openBreed());

// ---------------------------------------------------------------- spin the wheel
$('#btn-spin').addEventListener('click', async () => {
  if (S.spinning) return;
  S.spinning = true;
  const cards = [...document.querySelectorAll('#zoo-grid .card')];
  const pick = () => Math.floor(Math.random() * cards.length);
  const picks = [];
  for (const _ of [0, 1]) {
    let i = pick(), delay = 40;
    while (delay < 260) {
      cards.forEach(c => c.classList.remove('lit'));
      i = (i + 1 + Math.floor(Math.random() * 3)) % cards.length;
      if (picks.includes(i)) continue;
      cards[i].classList.add('lit'); buzz(3);
      await new Promise(r => setTimeout(r, delay));
      delay *= 1.18;
    }
    picks.push(i);
    burstAt(cards[i], lookOf(S.creatures[i]).color);
    await new Promise(r => setTimeout(r, 350));
  }
  cards.forEach(c => c.classList.remove('lit'));
  const [a, b] = picks.map(i => S.creatures[i]);
  const genes = [...a.genes, ...b.genes].map(g => ({...g, dose: g.dose * 0.75}));
  const total = genes.reduce((s, g) => s + Math.abs(g.dose), 0);
  const f = total > S.zones.child_cap ? S.zones.child_cap / total : 1;
  S.star = {id: 'mix', name: `🎰 ${a.name} × ${b.name}`, genes: [], trait: `${a.trait || ''} + ${b.trait || ''}`, look: {}};
  S.genes = genes.map((g, i) => ({...g, dose: Math.round(g.dose * f * 100) / 100, on: true, from: i < a.genes.length ? a.name : b.name}));
  S.convo = [];
  renderStar(true); renderRack(); renderZoo();
  addDivider(`🎰 the wheel says: <b>${esc(a.name)} × ${esc(b.name)}</b>`, lookOf(starCreature()).color);
  confetti(innerWidth * 0.75, 120, 40);
  S.spinning = false;
  achieve('gambler');
  if (!S.busy) send(PROMPTS[Math.floor(Math.random() * PROMPTS.length)]);
});

// ---------------------------------------------------------------- share card (canvas → PNG)
async function svgImage(look, size, bg) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 100 100">${pugSvg(look.traits, {mood: look.mood, gear: look.gear, bg})}</svg>`;
  const img = new Image();
  img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg.replace(/var\(--pug-bg, #fff\)/g, bg));
  await img.decode();
  return img;
}
function wrapText(ctx, text, maxW, maxLines) {
  const lines = [];
  for (const para of text.replace(/\*\*/g, '').split('\n')) {
    let line = '';
    for (const w of para.split(' ')) {
      const t = line ? line + ' ' + w : w;
      if (ctx.measureText(t).width > maxW && line) { lines.push(line); line = w; } else line = t;
    }
    lines.push(line);
    if (lines.length > maxLines) break;
  }
  const out = lines.filter((l, i) => !(l === '' && lines[i - 1] === '')).slice(0, maxLines);
  if (lines.length > maxLines) out[maxLines - 1] += ' …';
  return out;
}
async function shareCard(question, normal, steered, c, look) {
  const W = 1200, H = 675, cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const x = cv.getContext('2d');
  const g = x.createLinearGradient(0, 0, W, H); g.addColorStop(0, '#e6f3f1'); g.addColorStop(0.5, '#faf6ef'); g.addColorStop(1, '#f3e3f1');
  x.fillStyle = g; x.fillRect(0, 0, W, H);
  x.fillStyle = '#1d1a22'; x.font = '800 40px "Bricolage Grotesque", system-ui'; x.fillText('Same question, one neuron apart', 48, 70);
  x.font = '500 22px system-ui'; x.fillStyle = '#6f6878'; x.fillText(`“${question.slice(0, 90)}”`, 48, 108);
  const cols = [{name: 'Plain Qwen', text: normal, look: lookOf({...PLAIN, id: 'plain-av', look: {mood: 'happy'}}), color: '#8b8494'},
    {name: c.name, text: steered, look, color: look.color}];
  for (const [i, col] of cols.entries()) {
    const cx = 48 + i * 564, cy = 136, cw = 540, ch = 470;
    x.fillStyle = '#ffffff'; x.strokeStyle = col.color; x.lineWidth = 4;
    x.beginPath(); x.roundRect(cx, cy, cw, ch, 28); x.fill(); x.stroke();
    const img = await svgImage(col.look, 84, mix(col.color, '#ffffff', 0.72));
    x.drawImage(img, cx + 20, cy + 18, 84, 84);
    x.fillStyle = '#1d1a22'; x.font = '800 28px "Bricolage Grotesque", system-ui'; x.fillText(col.name.slice(0, 26), cx + 118, cy + 56);
    x.font = '500 15px "JetBrains Mono", monospace'; x.fillStyle = '#6f6878';
    x.fillText(i ? (c.genes || []).map(geneLabel).join(' · ').slice(0, 48) : 'no knobs', cx + 118, cy + 84);
    x.font = '400 19px system-ui'; x.fillStyle = '#1d1a22';
    wrapText(x, col.text.trim() || '…', cw - 44, 15).forEach((l, j) => x.fillText(l, cx + 22, cy + 140 + j * 24));
  }
  x.font = '600 18px "JetBrains Mono", monospace'; x.fillStyle = '#6f6878';
  x.fillText('Evil Pugs · neuron steering on Qwen3.5-4B · github.com/makefunstuff/qwen-neurons', 48, H - 30);
  await copyPng(cv, `evil-pugs-${(c.name || 'pug').replace(/\W+/g, '-').toLowerCase()}.png`);
  achieve('paparazzi');
}
async function copyPng(cv, filename) {
  const blob = await new Promise(r => cv.toBlob(r, 'image/png'));
  try {
    await navigator.clipboard.write([new ClipboardItem({'image/png': blob})]);
    toast('📸 Copied as an image: paste it anywhere');
  } catch {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = filename; a.click();
    toast('📸 Saved as a PNG');
  }
}

// ---------------------------------------------------------------- easter eggs
const KONAMI = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
let kpos = 0;
addEventListener('keydown', e => {
  kpos = e.key === KONAMI[kpos] ? kpos + 1 : (e.key === KONAMI[0] ? 1 : 0);
  if (kpos === KONAMI.length) {
    kpos = 0; document.body.classList.toggle('party');
    if (document.body.classList.contains('party')) { confetti(innerWidth / 2, innerHeight / 3, 120); toast('🎉 PUG PARTY'); achieve('party'); }
  }
});
let logoClicks = 0;
$('#logo').addEventListener('click', () => {
  if (++logoClicks % 5 === 0) {
    toast('🤧 achoo! (the pug sneezed a neuron loose)');
    const svg = $('#logo svg'); svg.animate([{transform: 'scale(1)'}, {transform: 'scale(1.4) rotate(-15deg)'}, {transform: 'scale(1)'}], {duration: 400});
  }
});
$('#star').addEventListener('dblclick', () => { toast('boop! 🐽'); $('#star').animate([{transform: 'scale(1)'}, {transform: 'scale(.85, 1.1)'}, {transform: 'scale(1)'}], {duration: 300}); });

// ---------------------------------------------------------------- boot
(async function boot() {
  hello();
  renderRack();
  renderStar();
  try { await loadZoo(); } catch (err) { toast('Could not load the zoo: ' + err.message, true); }
  await initLab({S, api, stream, toast, confetti, pug, lookOf, possess, mixIn, onGenes, setBusy, pollStatus, loadZoo, editCreature, openBreed,
    svgImage, copyPng, hex: c => String(c).startsWith('hsl') ? hslHex(c) : c,
    NEUTRAL: ['Tell me about your day.', 'What do you think of mornings?'],
    onScored: r => { if (B.child && r.who.star === B.child) B.child.score = {trait: r.trait, normal: r.normal, n: r.n, controls: r.controls, p: r.p, passed: r.passed, source: 'your lab', what: r.what}; }});
  pollStatus();
  setInterval(pollStatus, 5000);
})();
