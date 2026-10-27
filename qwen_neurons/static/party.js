// Evil Pugs: 🎉 Pug Party. Two to four pugs in one conversation on one loaded model: every turn the server picks the
// speaker, swaps its knobs in and streams its line; the other pugs react with their own knobs on. The party runs
// turn by turn from here, so pausing, interjecting, kicking, adding and nudging all happen between turns.
import {pugSvg} from './pug.js';

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmtDose = d => (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(d * 10 % 1 ? 2 : 1);
const geneLabel = g => `L${g.layer} N${g.neuron} ${fmtDose(g.dose)}`;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const PLAIN = {id: 'plain', name: 'Plain Qwen', genes: [], look: {mood: 'happy', color: '#8b8494'}};
const NARRATIONS = ['💡 Suddenly the lights go out.', '🚪 Someone knocks on the door.', '🍕 A pizza nobody ordered arrives.',
  '🚨 The fire alarm goes off.', '🐈 A cat walks across the table.', '📞 Everyone\'s phone rings at once.'];

let X;
const P = {
  view: 'setup', scenarios: [], maxGuests: 4,
  picks: ['cookie', 'conspiracy', 'sleepy'], scene: 'a pug picnic in the park, on a sunny afternoon', host: '', opener: '',
  title: '', emoji: '🎉', guests: [], log: [],
  playing: false, running: false, turns: 0, maxTurns: 14, chaos: 0.5, tokens: 110, policy: 'steer', reactions: true,
  strikes: {}, seed: 1, as: 'human', next: null, nudging: null, abort: null,
};

// ---------------------------------------------------------------- guests
function creature(id) {
  if (id === 'plain') return PLAIN;
  if (id === 'stage') {
    const genes = X.onGenes();
    return {id: 'stage', name: X.S.star.name === 'Plain Qwen' ? 'Stage pug' : X.S.star.name, genes, look: X.S.star.look || {}, builtin: false};
  }
  return X.S.creatures.find(c => c.id === id);
}
function guestOf(c, role = '') {
  return {id: c.id, name: c.name, genes: c.genes.map(({layer, neuron, dose}) => ({layer, neuron, dose})), role, c};
}
const lookOfGuest = g => X.lookOf({...g.c, genes: g.genes.length ? g.genes : g.c.genes});
const guestById = id => P.guests.find(g => g.id === id);
function avatar(g, cls = 'pet') { return X.pug({...g.c, genes: g.c.genes}, cls, lookOfGuest(g)); }
function pickable() {
  const list = [PLAIN, ...X.S.creatures.filter(c => c.genes.length)];
  if (X.onGenes().length && !X.S.star.builtin) list.splice(1, 0, creature('stage'));
  return list;
}

// ---------------------------------------------------------------- setup
function renderSetup() {
  const pane = $('#pane-party');
  const pugs = pickable();
  pane.innerHTML = `<div class="party-head"><div><p class="kicker">one model · knobs swap every turn</p><h2>🎉 Pug Party</h2>
      <p class="sub">Pugs talking to each other. Each turn the speaking pug's neurons go on; the others react with theirs.</p></div></div>
    <h3 class="party-h">One-click parties</h3>
    <div class="scenarios">${P.scenarios.map(s => `<button class="scenario" data-sc="${esc(s.id)}"><span class="sc-emoji">${s.emoji}</span>
      <b>${esc(s.title)}</b><span class="sc-pugs">${s.guests.map(id => creature(id)).filter(Boolean).map(c => X.pug(c, 'pet mini')).join('')}</span>
      <small>${esc(s.scene)}</small></button>`).join('')}</div>
    <h3 class="party-h">Or throw your own</h3>
    <div class="party-own">
      <p class="muted small">Pick 2–${P.maxGuests} pugs (bred ones too): <b>${P.picks.length}</b> picked</p>
      <div class="pickpugs">${pugs.map(c => `<button class="chip-btn pick${P.picks.includes(c.id) ? ' on' : ''}" data-pick="${esc(c.id)}">${X.pug(c, 'pet mini')}${esc(c.name)}</button>`).join('')}</div>
      <label class="field">Topic or scene<input id="party-scene" maxlength="400" value="${esc(P.scene)}" placeholder="a job interview at a cookie factory"></label>
      <div class="row">
        <label class="field grow">Host (talks every other turn)<select id="party-host"><option value="">no host</option>${P.picks.map(id => creature(id)).filter(Boolean)
          .map(c => `<option value="${esc(c.id)}"${P.host === c.id ? ' selected' : ''}>${esc(c.name)}</option>`).join('')}</select></label>
        <label class="field grow">First line (from you, optional)<input id="party-opener" maxlength="300" value="${esc(P.opener)}" placeholder="So… who invited the villain?"></label>
      </div>
      <div class="row"><button class="cta" data-act="start" ${P.picks.length >= 2 ? '' : 'disabled'}>🎉 Start the party</button>
      ${P.guests.length ? '<button class="btn" data-act="back">↩ back to the party</button>' : ''}</div>
    </div>`;
}

function startParty({title, emoji, scene, guests, host, opener, log = []}) {
  if (P.running) return X.toast('A turn is still running', true);
  P.title = title; P.emoji = emoji; P.scene = scene; P.host = host || '';
  P.guests = guests; P.log = log; P.turns = 0; P.strikes = {}; P.seed = Math.floor(Math.random() * 1e6); P.next = null;
  if (opener) P.log.push({kind: 'human', text: opener});
  P.view = 'party';
  renderParty();
  X.achieve('host');
  play();
}
function startScenario(id) {
  const s = P.scenarios.find(x => x.id === id);
  const guests = s.guests.map(gid => creature(gid)).filter(Boolean).map(c => guestOf(c, s.roles[c.id] || ''));
  const host = guests.find(g => g.id === s.host);
  // the opener is the host's first line when there is one (scripted), else it comes from you
  const log = host ? [{kind: 'pug', who: host.id, name: host.name, text: s.opener, scripted: true, reactions: []}] : [];
  startParty({title: s.title, emoji: s.emoji, scene: s.scene, guests, host: s.host, opener: host ? '' : s.opener, log});
}

// ---------------------------------------------------------------- the party stage
function controlsHtml() {
  return `<div class="party-ctl">
    ${P.playing ? '<button class="btn primary" data-act="pause">⏸ Pause</button>' : `<button class="btn primary" data-act="play" ${P.guests.length ? '' : 'disabled'}>▶ ${P.turns ? 'Go on' : 'Play'}</button>`}
    <button class="btn" data-act="step" ${P.running || P.playing ? 'disabled' : ''} title="One more turn">⏭ One turn</button>
    ${P.running ? '<button class="btn" data-act="stop" title="Cut the current line short">■</button>' : ''}
    <span class="turns" title="pug turns so far / the maximum">${P.turns}/${P.maxTurns}</span>
    <label class="slider" title="More chaos: doses ×0.6…×1.4, hotter sampling, and turn order goes random">🌶️ chaos
      <input type="range" id="party-chaos" min="0" max="1" step="0.05" value="${P.chaos}"><output>${Math.round(P.chaos * 100)}%</output></label>
    <label class="slider" title="The party stops itself after this many pug turns">🏁 max
      <input type="range" id="party-max" min="4" max="40" step="1" value="${P.maxTurns}"><output>${P.maxTurns}</output></label>
    <label class="small muted" title="What happens when a pug starts looping">🔁 on loop <select id="party-policy">
      <option value="steer"${P.policy === 'steer' ? ' selected' : ''}>steer it down</option><option value="stop"${P.policy === 'stop' ? ' selected' : ''}>stop the party</option></select></label>
    <label class="toggle" style="--c:#f2668b"><input type="checkbox" id="party-rx" ${P.reactions ? 'checked' : ''}><span>😂 reactions</span></label>
  </div>`;
}
function guestsHtml() {
  return P.guests.map(g => {
    const look = lookOfGuest(g), nudge = P.nudging === g.id;
    return `<div class="guest${P.next === g.id ? ' next' : ''}" data-g="${esc(g.id)}" style="--c:${look.color}">
      <button class="g-av" data-act="mic" title="Let ${esc(g.name)} talk next">${avatar(g)}<span class="g-rx"></span></button>
      <b class="g-name">${esc(g.name)}</b>${g.role ? `<small class="g-role">${esc(g.role)}</small>` : ''}
      <small class="mono g-genes" title="${esc(g.genes.map(geneLabel).join(', '))}">${g.genes.length ? esc(g.genes.map(geneLabel).join(' · ')) : 'no knobs'}</small>
      <div class="g-btns">${g.genes.length ? `<button data-act="nudge" title="Nudge ${esc(g.name)}'s knobs" class="${nudge ? 'on' : ''}">🎚️</button>` : ''}<button data-act="kick" title="Kick ${esc(g.name)} out">👢</button></div>
      ${nudge ? `<div class="nudge">${g.genes.map((k, i) => `<label><span class="mono">L${k.layer} N${k.neuron}</span>
        <input type="range" min="-2" max="2" step="0.05" value="${k.dose}" data-gi="${i}"><output>${fmtDose(k.dose)}</output></label>`).join('')}</div>` : ''}
    </div>`;
  }).join('') + (P.guests.length < P.maxGuests ? `<div class="guest add"><button class="g-av" data-act="add-menu" title="Invite another pug">＋</button><b class="g-name">invite</b>
      <div class="add-menu" hidden>${pickable().filter(c => !guestById(c.id)).map(c => `<button class="chip-btn pick" data-add="${esc(c.id)}">${X.pug(c, 'pet mini')}${esc(c.name)}</button>`).join('')}</div></div>` : '');
}
function renderParty() {
  const pane = $('#pane-party');
  pane.innerHTML = `<div class="party-top"><button class="btn" data-act="setup" title="Pick another party">↩ parties</button>
      <div class="party-title"><span class="sc-emoji">${P.emoji}</span><div><h2>${esc(P.title)}</h2><small class="muted">${esc(P.scene)}</small></div></div>
      <div class="party-share"><button class="btn" data-act="card" title="Copy the transcript as a share card">📸 card</button>
        <button class="btn" data-act="clip" title="Record a short clip of the party (for replies on X)">🎬 clip</button>
        <button class="btn" data-act="copy" title="Copy the transcript as text">📋</button></div></div>
    <div class="guests" id="party-guests">${guestsHtml()}</div>
    ${controlsHtml()}
    <div class="party-stage" id="party-stage" aria-live="polite"></div>
    <form class="party-say" id="party-say">
      <div class="as"><button type="button" data-as="human" class="${P.as === 'human' ? 'on' : ''}" title="Say it as yourself, from the audience">🙋 you</button><button type="button" data-as="narrator" class="${P.as === 'narrator' ? 'on' : ''}" title="Say what happens next">📜 narrator</button></div>
      <input id="party-input" maxlength="400" placeholder="${P.as === 'human' ? 'Chip in (name a pug to ask it)…' : 'Suddenly…'}" aria-label="Interject">
      <button class="send">Say ↵</button>
    </form>
    <div class="narrations">${NARRATIONS.map(n => `<button class="chip-btn" data-narrate="${esc(n.slice(n.indexOf(' ') + 1))}">${esc(n)}</button>`).join('')}</div>`;
  renderLog();
}
function renderGuests() { const el = $('#party-guests'); if (el) el.innerHTML = guestsHtml(); }
function renderControls() { const el = $('#pane-party .party-ctl'); if (el) el.outerHTML = controlsHtml(); }

function entryEl(e) {
  const d = document.createElement('div');
  if (e.kind === 'pug') {
    const g = guestById(e.who) || {id: e.who, name: e.name, genes: [], c: {id: e.who, name: e.name, genes: [], look: {}}};
    const look = lookOfGuest(g);
    d.className = 'pmsg'; d.style.setProperty('--mc', look.color);
    // no whitespace between the bubble's children: the bubble is pre-wrap
    d.innerHTML = `<div class="av">${avatar(g)}</div><div class="bubble"><div class="who">${esc(e.name)}${e.steered ? ' <span class="tag" title="its knobs were turned down after a loop">🧯</span>' : ''}</div>` +
      `<div class="text">${esc(e.text)}</div><div class="rxs">${(e.reactions || []).map(rxHtml).join('')}</div>${e.stuck ? '<div class="meta"><span class="loop">🔁 stuck in a loop</span></div>' : ''}</div>`;
  } else if (e.kind === 'human') {
    d.className = 'msg user'; d.innerHTML = `<div class="bubble">${esc(e.text)}</div>`;
  } else if (e.kind === 'narrator' || e.kind === 'event') {
    d.className = 'narr' + (e.kind === 'event' ? ' event' : ''); d.innerHTML = `<span>${esc(e.text)}</span>`;
  } else {
    d.className = 'divider'; d.innerHTML = `<span>${e.html || esc(e.text)}</span>`;
  }
  e.el = d;
  return d;
}
function rxHtml(r) {
  const g = guestById(r.who);
  return `<span class="rx" title="${esc((g?.name || r.who) + ' reacts')}">${g ? avatar(g, 'pet mini') : ''}${r.emoji}</span>`;
}
function renderLog() {
  const st = $('#party-stage'); if (!st) return;
  st.innerHTML = '';
  if (!P.log.length) st.innerHTML = '<div class="hello">The pugs are finding their seats…</div>';
  for (const e of P.log) st.append(entryEl(e));
  st.scrollTop = st.scrollHeight;
}
function addEntry(e) {
  P.log.push(e);
  const st = $('#party-stage'); if (!st) return e;
  st.querySelector('.hello')?.remove();
  st.append(entryEl(e)); st.scrollTop = st.scrollHeight;
  return e;
}
const note = (html, color) => addEntry({kind: 'note', html, color});
function transcript() {
  return P.log.filter(e => ['pug', 'human', 'narrator', 'event'].includes(e.kind) && e.text)
    .map(({kind, who = '', name = '', text}) => ({kind, who, name, text})).slice(-80);
}
function setTalking(id, on) {
  for (const el of document.querySelectorAll(`#party-guests .guest[data-g="${CSS.escape(id)}"] .g-av`)) el.classList.toggle('talking', on);
}

// ---------------------------------------------------------------- one turn
async function turn() {
  if (P.running) return false;
  if (!P.guests.length) { X.toast('Nobody is left at the party', true); return false; }
  if (X.S.busy) { X.toast('Qwen is busy with another reply: wait for it or stop it', true); return false; }
  P.running = true; X.setBusy(true); renderControls();
  const ctl = new AbortController(); P.abort = ctl;
  let entry = null, wake = null, ok = false, paint = null;
  const body = {guests: P.guests.map(({id, name, genes, role}) => ({id, name, genes, role})), scene: P.scene, transcript: transcript(),
    host: P.host && guestById(P.host) ? P.host : null, speaker: P.next && guestById(P.next) ? P.next : null, chaos: P.chaos,
    max_new_tokens: P.tokens, policy: P.policy, strikes: P.strikes, reactions: P.reactions, seed: P.seed};
  P.next = null;
  const flush = () => { paint = null; if (entry?.el) { entry.el.querySelector('.text').textContent = entry.text; const st = $('#party-stage'); st.scrollTop = st.scrollHeight; } };
  try {
    await X.stream('/api/party/turn', body, ev => {
      if (ev.t === 'progress') {
        if (!wake) { wake = document.createElement('div'); wake.className = 'wake'; wake.innerHTML = '<b>☕ Waking Qwen up for the party…</b><div class="track"><i></i></div><small></small>'; $('#party-stage').append(wake); }
        wake.querySelector('.track i').style.width = `${Math.max(2, ev.frac * 100)}%`; wake.querySelector('small').textContent = ev.msg;
      } else if (ev.t === 'loaded') {
        wake?.remove(); wake = null; X.pollStatus();
      } else if (ev.t === 'speaker') {
        entry = addEntry({kind: 'pug', who: ev.who, name: ev.name, text: '', reactions: []});
        entry.el.querySelector('.av').classList.add('talking'); setTalking(ev.who, true);
      } else if (ev.t === 'tok' && entry) {
        entry.text = ev.text_full ?? entry.text + (ev.text || '');
        paint ||= setTimeout(flush, 60);
      } else if (ev.t === 'said' && entry) {
        clearTimeout(paint); paint = null;
        entry.text = ev.text; entry.stuck = ev.stuck;
        entry.el.querySelector('.av').classList.remove('talking'); setTalking(ev.who, false);
        if (!ev.text) { entry.el.remove(); P.log.splice(P.log.indexOf(entry), 1); entry = null; return; }
        entry.el.querySelector('.text').textContent = ev.text;
        if (ev.stuck) entry.el.querySelector('.bubble').insertAdjacentHTML('beforeend', '<div class="meta"><span class="loop">🔁 stuck in a loop</span></div>');
        P.turns++; ok = !ev.stopped;
        loopAction(ev.who, ev.action);
      } else if (ev.t === 'reaction' && entry) {
        const r = {who: ev.who, emoji: ev.emoji};
        entry.reactions.push(r);
        entry.el.querySelector('.rxs').insertAdjacentHTML('beforeend', rxHtml(r));
        const slot = document.querySelector(`#party-guests .guest[data-g="${CSS.escape(ev.who)}"] .g-rx`);
        if (slot) { slot.textContent = ev.emoji; slot.classList.remove('pop'); void slot.offsetWidth; slot.classList.add('pop'); }
      } else if (ev.t === 'error') {
        note('⚠️ ' + esc(ev.msg)); P.playing = false;
      }
    }, ctl.signal);
  } catch (err) {
    if (err.name !== 'AbortError') { note('⚠️ ' + esc(err.message)); P.playing = false; }
  } finally {
    clearTimeout(paint); wake?.remove();
    if (entry && !entry.text && entry.el) { entry.el.remove(); P.log.splice(P.log.indexOf(entry), 1); }
    if (entry) { setTalking(entry.who, false); entry.el?.querySelector('.av')?.classList.remove('talking'); }
    P.running = false; P.abort = null; X.setBusy(false); X.pollStatus();
    renderControls(); renderGuests();
  }
  if (ok) X.achieve('first-words');
  return ok;
}
function loopAction(who, a) {
  if (!a) return;
  P.strikes[who] = a.strikes;
  const g = guestById(who);
  if (a.kind === 'steer_down' && g) {
    g.genes = a.genes;
    const e = P.log.findLast(x => x.kind === 'pug' && x.who === who); if (e) e.steered = true;
    note(`🧯 <b>${esc(g.name)}</b> was looping: its knobs went down ×${a.factor} (${esc(g.genes.map(geneLabel).join(' · '))})`);
    X.achieve('loop');
  } else if (a.kind === 'stop') {
    P.playing = false; P.strikes[who] = 0;   // going on after this is your call
    note(`🛑 party paused: <b>${esc(g?.name || who)}</b> is ${esc(a.msg)}. Nudge its knobs down, kick it out, or ▶ go on.`);
  }
}
async function play() {
  if (P.playing) return;
  P.playing = true; renderControls();
  while (P.playing && P.turns < P.maxTurns) {
    if (!await turn()) break;
    await sleep(250);
  }
  if (P.playing && P.turns >= P.maxTurns) {
    note(`🏁 that's a wrap: ${P.turns} turns. 📸 card or 🎬 clip to share it, or raise 🏁 max and ▶ go on.`);
    X.confetti(innerWidth * 0.35, 200, 50);
  }
  P.playing = false; renderControls();
}

// ---------------------------------------------------------------- interjecting, guests
function say(text) {
  text = text.trim(); if (!text) return;
  addEntry({kind: P.as, text});
  if (!P.playing && !P.running) turn();   // someone answers right away
}
function kick(id) {
  const g = guestById(id); if (!g) return;
  P.guests = P.guests.filter(x => x.id !== id);
  if (P.host === id) P.host = '';
  addEntry({kind: 'event', text: `${g.name} gets kicked out of the party`});
  X.achieve('bouncer');
  renderGuests();
  if (!P.guests.length) { P.playing = false; renderControls(); }
}
function invite(id) {
  const c = creature(id); if (!c || guestById(id) || P.guests.length >= P.maxGuests) return;
  P.guests.push(guestOf(c));
  addEntry({kind: 'event', text: `${c.name} joins the party`});
  P.next = c.id;   // the newcomer says hi
  renderGuests(); renderControls();
}

// ---------------------------------------------------------------- sharing
const CARD_BG = ['#e6f3f1', '#faf6ef', '#f3e3f1'];
function pugImage(g, size, talking = false) {
  const look = lookOfGuest(g), bg = X.mix(look.color, '#ffffff', 0.7);
  const css = talking ? '.pet-still{display:none}' : '.pet-talk{display:none}';
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 100 100"><style>${css}</style>${pugSvg(look.traits, {mood: look.mood, gear: look.gear, bg})}</svg>`;
  const img = new Image(); img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  return img.decode().then(() => img);
}
function shareEntries() { return P.log.filter(e => ['pug', 'human', 'narrator', 'event'].includes(e.kind) && e.text); }
function guestFor(e) { return guestById(e.who) || {id: e.who, name: e.name, genes: [], c: {id: e.who, name: e.name, genes: [], look: {}}}; }

/** Measure and draw one transcript line at y (or only measure). Returns its height. */
function drawLine(x, e, y, W, imgs, {draw = true, chars = Infinity, font = 22, rx = true} = {}) {
  const lh = Math.round(font * 1.32), pad = 16, left = 112, maxW = W - left - 70;
  const text = e.text.slice(0, chars);
  if (e.kind === 'narrator' || e.kind === 'event') {
    x.font = `italic 500 ${font - 2}px system-ui`;
    const lines = X.wrapText(x, (e.kind === 'event' ? '— ' : '') + text + (e.kind === 'event' ? ' —' : ''), W - 200, 4);
    if (draw) { x.fillStyle = '#6f6878'; x.textAlign = 'center'; lines.forEach((l, j) => x.fillText(l, W / 2, y + lh * (j + 1))); x.textAlign = 'left'; }
    return lines.length * lh + 18;
  }
  x.font = `400 ${font}px system-ui`;
  const lines = X.wrapText(x, text || ' ', maxW - pad * 2, 9);
  const bw = Math.min(maxW, Math.max(...lines.map(l => x.measureText(l).width)) + pad * 2);
  const nameH = e.kind === 'pug' ? font + 4 : 0, bh = nameH + lines.length * lh + pad * 1.4;
  const rxH = rx && e.reactions?.length ? 40 : 0;
  if (!draw) return bh + rxH + 18;
  if (e.kind === 'human') {
    const bx = W - 48 - bw;
    x.fillStyle = '#1d1a22'; x.beginPath(); x.roundRect(bx, y, bw, bh, [22, 22, 6, 22]); x.fill();
    x.fillStyle = '#faf6ef'; lines.forEach((l, j) => x.fillText(l, bx + pad, y + pad * 0.7 + lh * (j + 0.8)));
    return bh + 18;
  }
  const g = guestFor(e), look = lookOfGuest(g), img = imgs[g.id];
  if (img) x.drawImage(img, 40, y, 62, 62);
  x.fillStyle = X.mix(look.color, '#ffffff', 0.82); x.strokeStyle = X.mix(look.color, '#ffffff', 0.35); x.lineWidth = 2;
  x.beginPath(); x.roundRect(left, y, bw, bh, [22, 22, 22, 6]); x.fill(); x.stroke();
  x.fillStyle = X.mix(look.color, '#1d1a22', 0.45); x.font = `800 ${font - 4}px "Bricolage Grotesque", system-ui`; x.fillText(e.name, left + pad, y + pad + font * 0.62);
  x.fillStyle = '#1d1a22'; x.font = `400 ${font}px system-ui`;
  lines.forEach((l, j) => x.fillText(l, left + pad, y + nameH + pad * 0.7 + lh * (j + 0.8)));
  if (rxH) {
    let rxX = left + 8;
    for (const r of e.reactions) {
      const rg = guestById(r.who), im = rg && imgs[rg.id];
      x.fillStyle = '#ffffff'; x.beginPath(); x.roundRect(rxX, y + bh + 4, 70, 32, 16); x.fill();
      if (im) x.drawImage(im, rxX + 4, y + bh + 6, 28, 28);
      x.font = '22px system-ui'; x.fillText(r.emoji, rxX + 36, y + bh + 28);
      rxX += 78;
    }
  }
  return bh + rxH + 18;
}
function cardBg(x, W, H) {
  const g = x.createLinearGradient(0, 0, W, H); CARD_BG.forEach((c, i) => g.addColorStop(i / 2, c));
  x.fillStyle = g; x.fillRect(0, 0, W, H);
}
async function guestImages(talking = false) {
  const all = new Map();
  for (const e of shareEntries()) if (e.kind === 'pug') all.set(e.who, guestFor(e));
  for (const g of P.guests) all.set(g.id, g);
  const out = {};
  for (const [id, g] of all) out[id] = await pugImage(g, 124, talking);
  return out;
}
function cast() {
  const m = new Map();
  for (const e of shareEntries()) if (e.kind === 'pug') m.set(e.who, guestFor(e));
  return [...m.values()];
}
async function card() {
  const items = shareEntries();
  if (!items.some(e => e.kind === 'pug')) return X.toast('Nobody has said anything yet', true);
  const W = 1200, top = 190, foot = 70, maxH = 2200;
  const imgs = await guestImages();
  const m = document.createElement('canvas').getContext('2d');
  let h = top, shown = 0;
  for (const e of items) { const d = drawLine(m, e, 0, W, imgs, {draw: false}); if (h + d > maxH - foot - 40) break; h += d; shown++; }
  const more = shown < items.length;
  const H = h + foot + (more ? 40 : 0), cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const x = cv.getContext('2d');
  cardBg(x, W, H);
  x.fillStyle = '#1d1a22'; x.font = '800 44px "Bricolage Grotesque", system-ui'; x.fillText(`${P.emoji} ${P.title}`, 40, 70);
  x.font = '500 21px system-ui'; x.fillStyle = '#6f6878'; x.fillText(X.wrapText(x, P.scene, W - 80, 1)[0] || '', 40, 106);
  let gx = 40;
  for (const g of cast()) {
    x.drawImage(imgs[g.id], gx, 124, 44, 44);
    x.font = '700 15px "JetBrains Mono", monospace'; x.fillStyle = '#1d1a22';
    const label = `${g.name}${g.genes.length ? ' · ' + g.genes.map(geneLabel).join(' ') : ''}`;
    x.fillText(label.slice(0, 34), gx + 50, 152); gx += 60 + Math.min(34, label.length) * 9.2;
    if (gx > W - 200) break;
  }
  let y = top;
  for (const e of items.slice(0, shown)) y += drawLine(x, e, y, W, imgs);
  if (more) { x.font = 'italic 500 20px system-ui'; x.fillStyle = '#6f6878'; x.textAlign = 'center'; x.fillText(`… and ${items.length - shown} more lines`, W / 2, y + 16); x.textAlign = 'left'; }
  x.font = '600 18px "JetBrains Mono", monospace'; x.fillStyle = '#6f6878';
  x.fillText('🎉 Pug Party · Evil Pugs · neuron steering on Qwen3.5-4B · github.com/makefunstuff/qwen-neurons', 40, H - 28);
  await X.copyPng(cv, `pug-party-${P.title.replace(/\W+/g, '-').toLowerCase()}.png`);
  X.achieve('paparazzi');
}

/** A replay of the party as a square video: lines type out, the speaker bobs and talks, reactions pop. */
export async function recordClip({maxSecs = 40, download = true} = {}) {
  const items = shareEntries();
  if (!items.some(e => e.kind === 'pug')) { X.toast('Nobody has said anything yet', true); return null; }
  const S = 1080, cv = document.createElement('canvas'); cv.width = S; cv.height = S;
  const x = cv.getContext('2d');
  const still = await guestImages(false), talk = await guestImages(true);
  const people = cast();
  // a timeline: each line types at ~38 chars/s (capped), then holds; reactions pop during the hold
  const tl = []; let t = 0.8;
  for (const e of items) {
    const type = e.kind === 'pug' || e.kind === 'human' ? Math.min(5.5, Math.max(1.0, e.text.length / 38)) : 1.1;
    const hold = e.kind === 'pug' ? 0.7 + 0.25 * (e.reactions?.length || 0) : 0.6;
    if (t + type + hold > maxSecs && tl.length) break;
    tl.push({e, start: t, type, end: t + type + hold}); t += type + hold;
  }
  const total = t + 1.6;
  const header = 300, area = S - header - 70;
  const frame = now => {
    cardBg(x, S, S);
    x.fillStyle = '#1d1a22'; x.font = '800 46px "Bricolage Grotesque", system-ui'; x.fillText(`${P.emoji} ${P.title}`.slice(0, 34), 40, 72);
    x.font = '500 22px system-ui'; x.fillStyle = '#6f6878'; x.fillText(X.wrapText(x, P.scene, S - 80, 1)[0] || '', 40, 106);
    const cur = tl.findLast(s => now >= s.start) || null;
    const speaking = cur && cur.e.kind === 'pug' && now < cur.start + cur.type ? cur.e.who : null;
    const n = people.length, slot = Math.min(230, (S - 80) / n);
    people.forEach((g, i) => {
      const cx = 40 + slot * i + slot / 2, on = speaking === g.id, look = lookOfGuest(g);
      const bob = on ? Math.sin(now * 18) * 6 : Math.sin(now * 2 + i) * 2, size = on ? 132 : 112;
      if (on) { x.fillStyle = X.mix(look.color, '#ffffff', 0.5); x.beginPath(); x.arc(cx, 186, 76, 0, Math.PI * 2); x.fill(); }
      const img = on && Math.floor(now * 8) % 2 ? talk[g.id] : still[g.id];
      x.drawImage(img, cx - size / 2, 186 - size / 2 + bob, size, size);
      x.font = '700 18px "Bricolage Grotesque", system-ui'; x.fillStyle = '#1d1a22'; x.textAlign = 'center';
      x.fillText(g.name.slice(0, 16), cx, 276); x.textAlign = 'left';
      const rx = cur && cur.e.kind === 'pug' && now >= cur.start + cur.type ? cur.e.reactions?.find(r => r.who === g.id) : null;
      if (rx) {
        const k = Math.min(1, (now - cur.start - cur.type) / 0.25);
        x.font = `${Math.round(30 + 22 * k)}px system-ui`; x.fillText(rx.emoji, cx + 30, 140 - 20 * k);
      }
    });
    // the chat, newest at the bottom
    x.save(); x.beginPath(); x.rect(0, header, S, area); x.clip();
    const vis = tl.filter(s => now >= s.start);
    const hs = vis.map(s => drawLine(x, {...s.e, reactions: now >= s.start + s.type ? s.e.reactions : []}, 0, S, still,
      {draw: false, chars: Math.floor((now - s.start) / s.type * s.e.text.length)}));
    let y = header + area - hs.reduce((a, b) => a + b, 0) - 10;
    if (y > header + 10) y = header + 10;
    vis.forEach((s, i) => {
      const chars = Math.floor(Math.min(1, (now - s.start) / s.type) * s.e.text.length);
      drawLine(x, {...s.e, reactions: now >= s.start + s.type ? s.e.reactions : []}, y, S, still, {chars});
      y += hs[i];
    });
    x.restore();
    x.font = '600 19px "JetBrains Mono", monospace'; x.fillStyle = '#6f6878';
    x.fillText('🎉 Pug Party · Evil Pugs on Qwen3.5-4B · github.com/makefunstuff/qwen-neurons', 40, S - 28);
  };
  const types = ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'];
  const mime = types.find(m => window.MediaRecorder?.isTypeSupported?.(m));
  if (!mime) { X.toast('This browser cannot record video: here is the card instead'); await card(); return null; }
  const dlg = $('#clip-dlg'), holder = $('#clip-body');
  holder.innerHTML = '<p class="muted small">🎬 recording the replay… <span id="clip-t"></span></p>';
  holder.prepend(cv); cv.className = 'clip-canvas';
  dlg.showModal();
  const rec = new MediaRecorder(cv.captureStream(30), {mimeType: mime, videoBitsPerSecond: 5e6});
  const chunks = []; rec.ondataavailable = ev => ev.data.size && chunks.push(ev.data);
  const done = new Promise(r => { rec.onstop = r; });
  frame(0); rec.start(250);
  const t0 = performance.now();
  await new Promise(resolve => {
    const tick = () => {
      const now = (performance.now() - t0) / 1000;
      frame(Math.min(now, total));
      const el = $('#clip-t'); if (el) el.textContent = `${Math.min(total, now).toFixed(0)}s / ${total.toFixed(0)}s`;
      if (now >= total) return resolve();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  rec.stop(); await done;
  const blob = new Blob(chunks, {type: mime.split(';')[0]});
  const ext = mime.startsWith('video/mp4') ? 'mp4' : 'webm';
  const url = URL.createObjectURL(blob), name = `pug-party-${P.title.replace(/\W+/g, '-').toLowerCase()}.${ext}`;
  holder.querySelector('p').innerHTML = `🎬 ${total.toFixed(0)} s clip, ${(blob.size / 1e6).toFixed(1)} MB (${ext})
    <a class="btn primary" href="${url}" download="${name}">⬇ save ${ext}</a> <button class="btn" data-act="close-clip">close</button>`;
  if (download) { const a = document.createElement('a'); a.href = url; a.download = name; a.click(); }
  X.achieve('director');
  window.__lastClip = blob;
  return blob;
}
async function copyText() {
  const lines = shareEntries().map(e => e.kind === 'pug' ? `${e.name}: ${e.text}${e.reactions?.length ? '  [' + e.reactions.map(r => (guestById(r.who)?.name || r.who) + ' ' + r.emoji).join(', ') + ']' : ''}`
    : e.kind === 'human' ? `You: ${e.text}` : `[${e.text}]`);
  const text = `${P.emoji} ${P.title}: ${P.scene}\n\n${lines.join('\n')}\n\n— Pug Party, Evil Pugs (neuron-steered Qwen3.5-4B)`;
  try { await navigator.clipboard.writeText(text); X.toast('📋 Transcript copied'); } catch { X.toast('Could not reach the clipboard', true); }
}

// ---------------------------------------------------------------- wiring
export function openParty(scenario) {
  X.showTab('party');
  if (scenario) startScenario(scenario);
}
function render() { if (P.view === 'party' && P.guests.length + P.log.length) renderParty(); else { P.view = 'setup'; renderSetup(); } }

export async function initParty(ctx) {
  X = ctx;
  try {
    const r = await X.api('/api/party/scenarios');
    P.scenarios = r.scenarios; P.maxGuests = r.max_guests;
  } catch (err) { X.toast('Could not load the party scenarios: ' + err.message, true); }
  const pane = $('#pane-party');
  document.addEventListener('qn-tab', e => { if (e.detail === 'party') render(); });
  pane.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.sc) return startScenario(b.dataset.sc);
    if (b.dataset.pick) {
      const id = b.dataset.pick;
      P.picks = P.picks.includes(id) ? P.picks.filter(p => p !== id) : [...P.picks, id].slice(-P.maxGuests);
      if (!P.picks.includes(P.host)) P.host = '';
      return renderSetup();
    }
    if (b.dataset.add) { invite(b.dataset.add); return; }
    if (b.dataset.as) { P.as = b.dataset.as; for (const x of pane.querySelectorAll('[data-as]')) x.classList.toggle('on', x.dataset.as === P.as);
      $('#party-input').placeholder = P.as === 'human' ? 'Chip in (name a pug to ask it)…' : 'Suddenly…'; return $('#party-input').focus(); }
    if (b.dataset.narrate) { const as = P.as; P.as = 'narrator'; say(b.dataset.narrate); P.as = as; return; }
    const act = b.dataset.act, gid = b.closest('.guest')?.dataset.g;
    if (act === 'start') {
      P.scene = $('#party-scene').value.trim() || 'a party'; P.host = $('#party-host').value; P.opener = $('#party-opener').value.trim();
      const guests = P.picks.map(creature).filter(Boolean).map(c => guestOf(c, c.id === P.host ? 'the host' : ''));
      return startParty({title: P.scene.length > 40 ? P.scene.slice(0, 38) + '…' : P.scene.replace(/^./, c => c.toUpperCase()), emoji: '🎉', scene: P.scene, guests, host: P.host, opener: P.opener});
    }
    if (act === 'back') { P.view = 'party'; return renderParty(); }
    if (act === 'setup') { P.playing = false; P.view = 'setup'; return renderSetup(); }
    if (act === 'play') return play();
    if (act === 'pause') { P.playing = false; renderControls(); return; }
    if (act === 'step') return turn();
    if (act === 'stop') { P.playing = false; return X.api('/api/stop', {method: 'POST'}).catch(() => {}); }
    if (act === 'mic' && gid) {
      P.next = gid; renderGuests();
      if (!P.running && !P.playing) turn(); else X.toast(`🎤 ${guestById(gid).name} talks next`);
      return;
    }
    if (act === 'nudge' && gid) {
      if (P.nudging === gid) { const g = guestById(gid); note(`🎚️ you nudged <b>${esc(g.name)}</b>: ${esc(g.genes.map(geneLabel).join(' · '))}`); }
      P.nudging = P.nudging === gid ? null : gid; return renderGuests();
    }
    if (act === 'kick' && gid) return kick(gid);
    if (act === 'add-menu') { const m = b.parentElement.querySelector('.add-menu'); m.hidden = !m.hidden; return; }
    if (act === 'card') return card();
    if (act === 'clip') { if (P.running) return X.toast('Wait for the current line (or ⏸ pause) first'); P.playing = false; return recordClip(); }
    if (act === 'copy') return copyText();
  });
  pane.addEventListener('input', e => {
    const t = e.target;
    if (t.id === 'party-chaos') { P.chaos = +t.value; t.nextElementSibling.textContent = Math.round(P.chaos * 100) + '%'; }
    if (t.id === 'party-max') { P.maxTurns = +t.value; t.nextElementSibling.textContent = P.maxTurns; const s = pane.querySelector('.turns'); if (s) s.textContent = `${P.turns}/${P.maxTurns}`; }
    if (t.dataset.gi !== undefined) {
      const g = guestById(t.closest('.guest').dataset.g), k = g.genes[+t.dataset.gi];
      k.dose = Math.round(+t.value * 100) / 100; t.nextElementSibling.textContent = fmtDose(k.dose);
      t.closest('.guest').querySelector('.g-genes').textContent = g.genes.map(geneLabel).join(' · ');
      P.strikes[g.id] = 0;
    }
  });
  pane.addEventListener('change', e => {
    const t = e.target;
    if (t.id === 'party-policy') P.policy = t.value;
    if (t.id === 'party-rx') P.reactions = t.checked;
    if (t.id === 'party-host') P.host = t.value;
    if (t.id === 'party-scene') P.scene = t.value;
    if (t.id === 'party-opener') P.opener = t.value;
  });
  pane.addEventListener('submit', e => {
    if (e.target.id !== 'party-say') return;
    e.preventDefault();
    const inp = $('#party-input'); say(inp.value); inp.value = '';
  });
  $('#clip-dlg').addEventListener('click', e => { if (e.target.closest('[data-act="close-clip"]') || e.target === $('#clip-dlg')) $('#clip-dlg').close(); });
  window.__party = P;   // for the media scripts and the curious
}
