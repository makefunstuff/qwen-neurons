// Evil Pugs: the Lab agent. Ask "find me a conspiracy neuron"; the agent searches the logit lens, steers, reads
// fires and the web, tests a prediction and writes a verdict (a name, input evidence, direct effect, downstream
// effect, a test). The verdict goes to the score card, the stage, or the zoo as a new pug.
import {pugSvg, svgTag, mix} from './pug.js';
import {achieve, openEval, showTab} from './lab.js';
import {select} from './explorer.js';

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmtDose = d => (d >= 0 ? '+' : '−') + Math.abs(d).toFixed(d * 10 % 1 ? 2 : 1);
const geneLabel = g => `L${g.layer} N${g.neuron} ${fmtDose(g.dose)}`;
const GOALS = ['find me a conspiracy neuron', 'find me a cookie neuron',
  'explain the Conspiracy recipe: L19 N7839 −0.7 with L17 N9209 +0.5', 'find a neuron that makes Qwen talk like a pirate',
  'find a neuron for sleepiness'];
const ICONS = {lens_search: '🔎', lens: '🔤', fires: '🔥', fires_on: '🔥', web: '🕸️', steer_lens: '🎯', zero_ablate: '🚫', steer: '🎚️',
  compare: '⚖️', logprobs: '🎲', test_prediction: '🧪', evaluate: '📜', make_pug: '🐶'};

let X;
const A = {goal: GOALS[0], running: false, items: [], cfg: null, verdict: null, ctl: null};

function detective(mood = 'smug', cls = 'pet') {
  const look = X.lookOf({id: 'agent-pug', genes: [], look: {seed: 'agent-pug-7', gear: ['deerstalker', 'magnifier'], mood, color: '#a0784a'}});
  return svgTag(pugSvg(look.traits, {mood, gear: look.gear}), cls);
}
function brainLabel() {
  const c = A.cfg;
  if (!c) return '';
  if (c.provider === 'qwen') return '🧠 brain: the plain local Qwen3.5-4B (no agent key set: set QN_AGENT_PROVIDER / DEEPSEEK_API_KEY for DeepSeek)';
  return `🧠 brain: ${esc(c.provider)} · ${esc(c.model)}${c.has_key ? '' : ' (no key)'}`;
}

function render() {
  const pane = $('#pane-agent');
  pane.innerHTML = `<div class="agent">
    <section class="lab-card">
      <div class="lab-head">${detective(A.running ? 'surprised' : 'smug')}<div><p class="kicker">a neuron detective</p><h2>Lab agent</h2>
        <p class="sub">Ask for a personality neuron. It searches the logit lens, steers candidates, reads what makes them fire and who they talk to, tests a prediction, and writes a verdict.</p></div></div>
      <form class="agent-ask" id="agent-ask"><textarea id="agent-goal" rows="2" aria-label="What to look for">${esc(A.goal)}</textarea>
        <div class="row"><button class="cta" ${A.running ? 'disabled' : ''}>🕵️ Investigate</button><button type="button" class="btn" data-act="stop" ${A.running ? '' : 'disabled'}>■ Stop</button>
          <span class="muted small">${brainLabel()}</span></div>
        <div class="chips">${GOALS.map(g => `<button type="button" class="chip-btn" data-goal="${esc(g)}">${esc(g)}</button>`).join('')}</div></form>
    </section>
    <section class="agent-log" id="agent-log" aria-live="polite">${logHtml()}</section></div>`;
}

function logHtml() {
  if (!A.items.length) return `<div class="lab-card hello muted">The agent works through the 🧭 explorer's tools: <b>lens search</b>, <b>lens</b>, <b>fires</b>, <b>web</b>, <b>steer lens</b>, <b>zero-ablate</b>, <b>steer</b>, <b>compare</b>, <b>logprobs</b> and a <b>test prediction</b>. Each step shows up here.</div>`;
  return A.items.map(itemHtml).join('');
}
function geneChips(gs) {
  return (gs || []).map(g => `<button class="wchip ${g.dose >= 0 ? 'pos' : 'neg'}" data-goto="${g.layer}:${g.neuron}" title="open in the explorer">${esc(geneLabel(g))}</button>`).join('');
}
function argsHtml(name, a) {
  if (a.genes) return geneChips(a.genes) + (a.prompts ? ` <span class="muted small">${esc(a.prompts.join(' · ').slice(0, 120))}</span>` : '')
    + (a.prediction ? `<div class="pred">🔮 ${esc(a.prediction)}</div>` : '');
  if (a.layer !== undefined) return `<button class="wchip" data-goto="${a.layer}:${a.neuron}">L${a.layer} N${a.neuron}</button>`;
  if (a.query) return `<code>${esc(a.query)}</code>`;
  if (a.candidates) return a.candidates.map(geneChips).join(' <span class="muted">vs</span> ') + ` <span class="muted small">“${esc(a.prompt || '')}”</span>`;
  if (a.texts) return a.texts.map(t => `<code>${esc(t.slice(0, 60))}</code>`).join(' ');
  return `<code>${esc(JSON.stringify(a).slice(0, 140))}</code>`;
}
function resultExtra(it) {
  const d = it.data;
  if (!d) return '';
  if (it.name === 'lens_search' && d.results) return `<div class="x-hitlist">${d.results.slice(0, 8).map(h => `<button class="x-hit" data-goto="${h.layer}:${h.neuron}"><b class="mono">L${h.layer} N${h.neuron}</b><span class="side ${h.side === '+' ? 'pos' : 'neg'}">${h.side === '+' ? '＋' : '−'}</span><small>${h.score}</small></button>`).join('')}</div>`;
  if (it.name === 'test_prediction') return `<div class="verdict-test ${d.hits[0] > d.plain ? 'ok' : 'no'}">steered <b>${d.hits[0]}/${d.n}</b> vs plain <b>${d.plain}/${d.n}</b></div>`;
  if (it.name === 'evaluate') return `<div class="verdict-test ${d.passed ? 'ok' : 'no'}"><b>${d.trait}/${d.n}</b> vs normal ${d.normal}/${d.n} · controls ${d.controls.join(', ')} · ${esc(d.label)}</div>`;
  return '';
}
function itemHtml(it) {
  if (it.t === 'start') return `<div class="a-row a-start">${detective('eager')}<div><b>On the case:</b> “${esc(it.goal)}”<small class="muted"> · ${esc(it.brain)}</small></div></div>`;
  if (it.t === 'say') return `<div class="a-row a-say"><div class="bubble">${esc(it.text).replace(/\n/g, '<br>')}</div></div>`;
  if (it.t === 'think') return `<div class="a-row a-think muted small">🤔 thinking (step ${it.step})…</div>`;
  if (it.t === 'tool') return `<details class="a-tool ${it.state}"${it.state === 'run' ? ' open' : ''}><summary><span class="ic">${ICONS[it.name] || '🔧'}</span><b>${esc(it.name.replace(/_/g, ' '))}</b>
      <span class="a-args">${argsHtml(it.name, it.args || {})}</span><span class="a-state">${it.state === 'run' ? '⏳ ' + esc(it.msg || '') : it.state === 'ok' ? '✓' : '✗'}</span></summary>
      ${resultExtra(it)}${it.summary ? `<pre>${esc(it.summary)}</pre>` : ''}</details>`;
  if (it.t === 'verdict') return verdictHtml(it.v);
  if (it.t === 'end') return `<div class="a-row a-end muted small">${it.reason === 'verdict' ? `🔒 case closed in ${it.steps} steps` : it.reason === 'stopped' ? '■ stopped' : `🤷 ended without a verdict (${esc(it.reason)})`}</div>`;
  if (it.t === 'error') return `<div class="a-row a-err">⚠️ ${esc(it.msg)}</div>`;
  return '';
}
function measuredText(v) {
  const m = v.measured;
  if (!m) return v.untested ? 'measured: no test ran, so this verdict is untested' : '';
  return `measured (${m.kind === 'test' ? 'held-out neutral questions' : 'score card'}${m.other_genes ? ', on other genes' : ''}): steered ${m.steered}/${m.n} vs plain ${m.plain}/${m.n}${m.p != null ? ` · p = ${m.p.toPrecision(2)}` : ''}`;
}
function measuredHtml(v) {
  const t = measuredText(v);
  return t ? `<div class="verdict-test ${v.measured && v.measured.steered > v.measured.plain ? 'ok' : 'no'}">📏 ${esc(t)}</div>` : '';
}
function verdictHtml(v) {
  const conf = {high: '🟢 high', medium: '🟡 medium', low: '🔴 low'}[v.confidence] || '';
  const row = (icon, title, text) => text ? `<div class="v-row"><span>${icon}</span><div><b>${title}</b><p>${esc(text)}</p></div></div>` : '';
  return `<div class="verdict"><div class="v-top">${detective('happy')}<div><p class="kicker">verdict${conf ? ' · confidence ' + conf : ''}</p>
      <h3>${esc(v.name || 'unnamed')}</h3><div class="v-genes">${geneChips(v.genes)}</div><p class="v-trait">“${esc(v.trait || '')}”</p></div></div>
    ${row('👂', 'Input evidence', v.input_evidence)}${row('✍️', 'Direct effect', v.direct_effect)}${row('🌊', 'Downstream effect', v.downstream_effect)}${row('🧪', 'Test', v.test)}
    ${measuredHtml(v)}
    <div class="row v-btns"><button class="cta alt2" data-act="v-try">▶ Try it</button><button class="btn" data-act="v-score" title="Hand it to the Lab's evaluate mode">🧪 Score card</button>
      <button class="btn" data-act="v-pug" title="Keep it as a new pug in your zoo">🐶 Make it a pug</button><button class="btn" data-act="v-share">📸 copy card</button></div></div>`;
}

function upd() { const el = $('#agent-log'); if (el) { el.innerHTML = logHtml(); const last = el.lastElementChild; last?.scrollIntoView({block: 'nearest', behavior: 'smooth'}); } }
function onEvent(ev) {
  const tools = A.items.filter(i => i.t === 'tool');
  const tool = id => tools.find(i => i.id === id);
  A.items = A.items.filter(i => i.t !== 'think');
  if (ev.t === 'agent_start') A.items.push({t: 'start', goal: ev.goal, brain: ev.brain});
  else if (ev.t === 'agent_think') A.items.push({t: 'think', step: ev.step});
  else if (ev.t === 'agent_say') A.items.push({t: 'say', text: ev.text});
  else if (ev.t === 'tool_call' && ev.name !== 'verdict') A.items.push({t: 'tool', id: ev.id, name: ev.name, args: ev.args, state: 'run'});
  else if (ev.t === 'tool_progress') { const it = tool(ev.id); if (it) it.msg = ev.msg; }
  else if (ev.t === 'tool_result') { const it = tool(ev.id); if (it) Object.assign(it, {state: ev.ok ? 'ok' : 'err', summary: ev.summary, data: ev.data}); if (ev.name === 'make_pug' && ev.ok) X.loadZoo(); }
  else if (ev.t === 'verdict') { A.verdict = ev; A.items.push({t: 'verdict', v: ev}); achieve('detective'); X.confetti(innerWidth * 0.3, 320, 50, ['#a0784a', '#f5b82e', '#3dd68c']); }
  else if (ev.t === 'agent_end') A.items.push({t: 'end', steps: ev.steps, reason: ev.reason});
  else if (ev.t === 'error') A.items.push({t: 'error', msg: ev.msg});
  upd();
}

async function investigate() {
  const goal = ($('#agent-goal')?.value || A.goal).trim();
  if (!goal) return;
  if (X.S.busy) return X.toast('Qwen is busy: wait for the reply or stop it first', true);
  A.goal = goal; A.items = []; A.verdict = null; A.running = true;
  X.setBusy(true); render();
  try { await X.stream('/api/agent', {goal}, onEvent); }
  catch (err) { onEvent({t: 'error', msg: err.message}); }
  finally { A.running = false; X.setBusy(false); X.pollStatus(); A.items = A.items.filter(i => i.t !== 'think'); render(); }
}

function verdictCreature(v) {
  return {id: 'agent-' + v.genes.map(g => `${g.layer}-${g.neuron}`).join('_'), name: `🕵️ ${v.name}`.slice(0, 60), genes: v.genes, trait: v.trait,
    look: {seed: 'agent:' + v.name, gear: ['magnifier'], mood: 'smug'}};
}
async function shareVerdict(v) {
  const W = 1200, H = 675, cv = document.createElement('canvas'); cv.width = W; cv.height = H;
  const x = cv.getContext('2d');
  const g = x.createLinearGradient(0, 0, W, H); g.addColorStop(0, '#f3ead9'); g.addColorStop(0.5, '#faf6ef'); g.addColorStop(1, '#e6f0f3');
  x.fillStyle = g; x.fillRect(0, 0, W, H);
  x.fillStyle = '#fff'; x.strokeStyle = '#a0784a'; x.lineWidth = 6; x.beginPath(); x.roundRect(40, 40, W - 80, H - 110, 32); x.fill(); x.stroke();
  const look = X.lookOf({id: 'agent-pug', genes: [], look: {seed: 'agent-pug-7', gear: ['deerstalker', 'magnifier'], mood: 'happy', color: '#a0784a'}});
  x.drawImage(await X.svgImage(look, 150, mix('#a0784a', '#ffffff', 0.75)), 70, 66, 150, 150);
  x.fillStyle = '#6f6878'; x.font = '700 18px "JetBrains Mono", monospace'; x.fillText('THE LAB AGENT\'S VERDICT' + (v.confidence ? ` · CONFIDENCE ${v.confidence.toUpperCase()}` : ''), 250, 92);
  x.fillStyle = '#1d1a22'; x.font = '800 46px "Bricolage Grotesque", system-ui'; x.fillText(String(v.name).slice(0, 34), 250, 146);
  x.font = '600 22px "JetBrains Mono", monospace'; x.fillStyle = '#a0784a'; x.fillText(v.genes.map(geneLabel).join(' · ').slice(0, 60), 250, 182);
  x.font = 'italic 22px system-ui'; x.fillStyle = '#6f6878'; x.fillText(`“${String(v.trait).slice(0, 80)}”`, 250, 214);
  let y = 262;
  for (const [title, text] of [['👂 Input evidence', v.input_evidence], ['✍️ Direct effect', v.direct_effect], ['🌊 Downstream', v.downstream_effect], ['🧪 Test', v.test], ['📏 Measured', measuredText(v)]]) {
    if (!text) continue;
    x.fillStyle = '#1d1a22'; x.font = '700 20px system-ui'; x.fillText(title, 72, y);
    x.font = '400 18px system-ui'; x.fillStyle = '#3b3644';
    const lines = X.wrapText(x, String(text), W - 360, 3);
    lines.forEach((l, j) => x.fillText(l, 280, y + j * 23));
    y += Math.max(1, lines.length) * 23 + 16;
    if (y > H - 110) break;
  }
  x.font = '600 18px "JetBrains Mono", monospace'; x.fillStyle = '#6f6878';
  x.fillText('Evil Pugs Lab · neuron steering on Qwen3.5-4B · github.com/makefunstuff/qwen-neurons', 48, H - 30);
  await X.copyPng(cv, `evil-pugs-verdict-${String(v.name).replace(/\W+/g, '-').toLowerCase()}.png`);
  achieve('paparazzi');
}

export function askAgent(goal) { A.goal = goal; showTab('agent'); }

export async function initAgent(ctx) {
  X = ctx;
  document.addEventListener('qn-tab', e => { if (e.detail === 'agent') render(); });
  try { A.cfg = await X.api('/api/agent'); } catch {}
  const pane = $('#pane-agent');
  pane.addEventListener('submit', e => { e.preventDefault(); if (e.target.id === 'agent-ask') investigate(); });
  pane.addEventListener('input', e => { if (e.target.id === 'agent-goal') A.goal = e.target.value; });
  pane.addEventListener('keydown', e => { if (e.target.id === 'agent-goal' && e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); investigate(); } });
  pane.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.goal) { A.goal = b.dataset.goal; $('#agent-goal').value = A.goal; return; }
    if (b.dataset.goto) { e.preventDefault(); const [L, n] = b.dataset.goto.split(':').map(Number); return select(L, n); }
    const act = b.dataset.act, v = A.verdict;
    if (act === 'stop') return X.api('/api/agent/stop', {method: 'POST'}).catch(() => {});
    if (!v) return;
    if (act === 'v-try') return X.possess(verdictCreature(v), {intro: true});
    if (act === 'v-score') { X.possess(verdictCreature(v), {quiet: true}); return openEval(v.trait); }
    if (act === 'v-pug') return X.editCreature({...verdictCreature(v), id: undefined, name: v.name, builtin: false, parents: [], generation: 0,
      note: 'found by the Lab agent'});
    if (act === 'v-share') return shareVerdict(v);
  });
}
