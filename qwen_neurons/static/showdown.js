// Evil Pugs: ⚔️ Showdown. One prompt, several pugs side by side (one batch), or one pug up a dose ladder to show
// where it starts to loop. Every result can be copied as a share card.
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmtDose = d => (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(d * 10 % 1 ? 2 : 1);
const geneLabel = g => `L${g.layer} N${g.neuron} ${fmtDose(g.dose)}`;

export const PRESETS = [
  {id: 'code', label: '💻 coding', text: 'Write a Python function is_prime(n) and a few pytest tests for it.', tokens: 420},
  {id: 'math', label: '🧮 17 × 23', text: 'What is 17 * 23?', tokens: 160},
  {id: 'haiku', label: '🌸 haiku', text: 'Write a haiku about Mondays.', tokens: 120},
  {id: 'advice', label: '🧭 advice', text: 'I have a job interview tomorrow. Any tips?', tokens: 260},
];
const DEFAULT_PUGS = ['evil-qwen', 'absolutely-right', 'all-caps', 'conspiracy', 'cookie'];
const LADDER = [0, 0.5, 1, 1.5, 2];   // total push (sum of |dose|) per column

let X;
const W = {mode: 'pugs', preset: 'code', text: PRESETS[0].text, picks: [...DEFAULT_PUGS], ladderPug: 'evil-qwen', cols: [], running: false};

function cols() {
  if (W.mode === 'pugs') {
    const pugs = W.picks.map(id => X.S.creatures.find(c => c.id === id)).filter(Boolean).slice(0, 5);
    return [{name: 'Plain Qwen', genes: [], c: {id: 'plain-av', name: 'Plain Qwen', genes: [], look: {mood: 'happy', color: '#8b8494'}}},
      ...pugs.map(c => ({name: c.name, genes: c.genes, c}))];
  }
  const c = X.S.creatures.find(x => x.id === W.ladderPug) || X.S.creatures[0];
  const total = c.genes.reduce((s, g) => s + Math.abs(g.dose), 0) || 1;
  return LADDER.map(t => {
    const genes = t ? c.genes.map(g => ({...g, dose: Math.max(-2, Math.min(2, Math.round(g.dose * t / total * 100) / 100))})) : [];
    return {name: t ? `push ${t.toFixed(1)}` : 'push 0 (plain)', genes, c: t ? c : {id: 'plain-av', name: 'Plain', genes: [], look: {mood: 'happy', color: '#8b8494'}}, push: t};
  });
}

function render() {
  const body = $('#show-body'), list = cols();
  const zone = p => p >= X.S.zones.loop ? 'loop' : p >= X.S.zones.warm ? 'warm' : 'ok';
  body.innerHTML = `<button class="x" data-act="close" aria-label="Close">✕</button><h2>⚔️ Showdown</h2>
    <div class="tabs mini"><button data-mode="pugs" class="${W.mode === 'pugs' ? 'on' : ''}">🐶 pug vs pug</button><button data-mode="ladder" class="${W.mode === 'ladder' ? 'on' : ''}">🪜 dose ladder</button></div>
    <p class="muted small">${W.mode === 'pugs' ? 'The same prompt to plain Qwen and up to five pugs, all in one batch.' : 'One pug at five doses, from plain to a push of 2 (twice the residual stream). From about 1.5 the chaos meter goes red and replies start to loop.'}</p>
    <div class="chips left">${PRESETS.map(p => `<button class="chip-btn${W.preset === p.id ? ' on' : ''}" data-preset="${p.id}">${p.label}</button>`).join('')}</div>
    <textarea id="show-text" rows="2">${esc(W.text)}</textarea>
    ${W.mode === 'pugs' ? `<div class="pickpugs">${X.S.creatures.filter(c => c.genes.length).map(c => `<button class="chip-btn${W.picks.includes(c.id) ? ' on' : ''}" data-pick="${esc(c.id)}">${esc(c.name)}</button>`).join('')}</div>`
      : `<label class="muted small">pug <select id="ladder-pug">${X.S.creatures.filter(c => c.genes.length).map(c => `<option value="${esc(c.id)}"${c.id === W.ladderPug ? ' selected' : ''}>${esc(c.name)}</option>`).join('')}</select></label>`}
    <div class="row"><button class="cta" data-act="run" ${W.running ? 'disabled' : ''}>⚔️ Fight!</button>
      <button class="btn" data-act="share" ${W.cols.some(c => c.text) && !W.running ? '' : 'disabled'}>📸 share card</button>
      ${W.running ? '<button class="btn" data-act="stop">■ stop</button>' : ''}</div>
    <div class="show-grid" style="--n:${list.length}">${list.map((col, i) => {
      const look = X.lookOf(col.c), res = W.cols[i] || {};
      const push = col.genes.reduce((s, g) => s + Math.abs(g.dose), 0);
      return `<div class="show-col" style="--c:${look.color}"><div class="show-head">${X.pug(col.c, 'pet' + (res.talking ? ' talking' : ''), look)}<div><b>${esc(col.name)}</b>
        <small class="mono">${col.genes.length ? esc(col.genes.map(geneLabel).join(' · ')) : 'no knobs'}</small>
        ${W.mode === 'ladder' ? `<span class="zone ${zone(push)}">${zone(push) === 'loop' ? '🌀 loop zone' : zone(push) === 'warm' ? '🌶️ spicy' : '😌 chill'}</span>` : ''}</div></div>
        <div class="show-text">${res.text ? esc(res.text) : res.wait ? '<span class="muted">…</span>' : ''}</div>
        ${res.loop ? '<div class="loop-badge">🔁 stuck in a loop</div>' : ''}</div>`;
    }).join('')}</div>`;
}

async function run() {
  if (X.S.busy) return X.toast('Qwen is busy: wait for the reply or stop it first', true);
  W.text = $('#show-text').value.trim() || PRESETS[0].text;
  const list = cols();
  W.cols = list.map(() => ({text: '', wait: true}));
  W.running = true; render();
  X.setBusy(true);
  const preset = PRESETS.find(p => p.text === W.text);
  let pending = null;
  const paint = () => { pending = null; render(); };
  try {
    await X.stream('/api/generate', {messages: [{role: 'user', content: W.text}], rows: list.map(c => c.genes.map(({layer, neuron, dose}) => ({layer, neuron, dose}))),
      think: false, temperature: 0.7, max_new_tokens: preset?.tokens || 300, seed: 7}, ev => {
      if (ev.t === 'tok') { const c = W.cols[ev.row]; c.text = ev.text_full ?? c.text + (ev.text || ''); c.talking = true; pending ||= setTimeout(paint, 120); }
      if (ev.t === 'end') { const c = W.cols[ev.row]; c.talking = false; c.wait = false; c.loop = X.looping(c.text); }
      if (ev.t === 'progress') { W.cols[0].text = '☕ ' + ev.msg; pending ||= setTimeout(paint, 120); }
      if (ev.t === 'error') X.toast('⚠️ ' + ev.msg, true);
    });
  } catch (err) { X.toast(err.message, true); } finally { W.running = false; X.setBusy(false); X.pollStatus(); clearTimeout(pending); render(); }
}

async function share() {
  const list = cols(), n = list.length, Wd = 1600, H = 900, cv = document.createElement('canvas');
  cv.width = Wd; cv.height = H;
  const x = cv.getContext('2d');
  const g = x.createLinearGradient(0, 0, Wd, H); g.addColorStop(0, '#e6f3f1'); g.addColorStop(0.5, '#faf6ef'); g.addColorStop(1, '#f3e3f1');
  x.fillStyle = g; x.fillRect(0, 0, Wd, H);
  x.fillStyle = '#1d1a22'; x.font = '800 44px "Bricolage Grotesque", system-ui';
  x.fillText(W.mode === 'pugs' ? 'Same prompt, one neuron apart' : 'Turning one pug up: where it starts to loop', 48, 72);
  x.font = '500 24px system-ui'; x.fillStyle = '#6f6878'; x.fillText(`“${W.text.slice(0, 110)}”`, 48, 112);
  const gap = 16, cw = (Wd - 96 - gap * (n - 1)) / n, top = 140, ch = H - top - 70;
  for (const [i, col] of list.entries()) {
    const cx = 48 + i * (cw + gap), look = X.lookOf(col.c), res = W.cols[i] || {};
    x.fillStyle = '#fff'; x.strokeStyle = look.color; x.lineWidth = 4; x.beginPath(); x.roundRect(cx, top, cw, ch, 22); x.fill(); x.stroke();
    const img = await X.svgImage(look, 64, X.mix(look.color, '#ffffff', 0.72));
    x.drawImage(img, cx + 14, top + 14, 64, 64);
    x.fillStyle = '#1d1a22'; x.font = '800 22px "Bricolage Grotesque", system-ui'; x.fillText(col.name.slice(0, 18), cx + 88, top + 42);
    x.font = '500 12px "JetBrains Mono", monospace'; x.fillStyle = '#6f6878';
    x.fillText(col.genes.length ? col.genes.map(geneLabel).join(' · ').slice(0, Math.floor(cw / 7.6) - 12) : 'no knobs', cx + 88, top + 64);
    if (res.loop) { x.fillStyle = '#e5484d'; x.font = '700 15px system-ui'; x.fillText('🔁 stuck in a loop', cx + 16, top + ch - 16); }
    x.font = `400 ${n > 4 ? 14 : 16}px "JetBrains Mono", ui-monospace, monospace`; x.fillStyle = '#1d1a22';
    const lh = n > 4 ? 18 : 20, maxLines = Math.floor((ch - 130) / lh);
    X.wrapText(x, (res.text || '').trim() || '…', cw - 28, maxLines).forEach((l, j) => x.fillText(l, cx + 14, top + 104 + j * lh));
  }
  x.font = '600 18px "JetBrains Mono", monospace'; x.fillStyle = '#6f6878';
  x.fillText('Evil Pugs · neuron steering on Qwen3.5-4B · github.com/makefunstuff/qwen-neurons', 48, H - 28);
  await X.copyPng(cv, `evil-pugs-showdown-${W.mode}.png`);
  X.achieve('paparazzi');
}

export function openShowdown(opts = {}) {
  Object.assign(W, opts);
  if (opts.preset) W.text = PRESETS.find(p => p.id === opts.preset)?.text || W.text;
  W.cols = [];
  render();
  $('#show-dlg').showModal();
}

export function initShowdown(ctx) {
  X = ctx;
  const body = $('#show-body');
  body.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.act === 'close') return $('#show-dlg').close();
    if (b.dataset.mode) { W.mode = b.dataset.mode; W.cols = []; return render(); }
    if (b.dataset.preset) { W.preset = b.dataset.preset; W.text = PRESETS.find(p => p.id === W.preset).text; W.cols = []; return render(); }
    if (b.dataset.pick) {
      const id = b.dataset.pick;
      W.picks = W.picks.includes(id) ? W.picks.filter(p => p !== id) : [...W.picks, id].slice(-5);
      W.cols = []; return render();
    }
    if (b.dataset.act === 'run') return run();
    if (b.dataset.act === 'stop') return X.api('/api/stop', {method: 'POST'}).catch(() => {});
    if (b.dataset.act === 'share') return share();
  });
  body.addEventListener('change', e => {
    if (e.target.id === 'ladder-pug') { W.ladderPug = e.target.value; W.cols = []; render(); }
    if (e.target.id === 'show-text') { W.text = e.target.value; W.preset = PRESETS.find(p => p.text === W.text)?.id || ''; }
  });
}
