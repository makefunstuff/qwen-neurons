// Evil Pugs: 🪨 explain it like to a caveman. Six little scenes on a loop: what a neuron is, what pushing one does,
// why one is enough, and what pushing too hard does. "📸" copies all six as one comic strip.
const $ = s => document.querySelector(s);

const SCENES = [
  {cap: 'BRAIN HAVE MANY ROCK. ROCK = NEURON.', sub: 'Qwen has 32 layers × 9216 MLP neurons: about 300,000 little rocks.'},
  {cap: 'WORDS GO UP. SOME ROCK GLOW.', sub: 'Every word flows up through the layers; each neuron lights up a bit, or a lot.'},
  {cap: 'THIS ROCK GLOW FOR COOKIE.', sub: 'Some neurons fire for one idea: cookies, pirates, being a GPU…'},
  {cap: 'UNGA POKE COOKIE ROCK. EVERY WORD.', sub: 'Steering: add that neuron\'s push to the stream at every token (a dose).'},
  {cap: 'NOW BRAIN THINK COOKIE ALL TIME.', sub: 'One push at the right spot tilts everything after it: a new personality.'},
  {cap: 'POKE TOO HARD → LOOP LOOP LOOP.', sub: 'Past a dose of about 1 (as big as the whole stream) it starts to repeat itself.'},
];
const ROWS = 5, COLS = 9, HOT = [2, 4];   // the cookie rock: row 2, column 4

function sceneSvg(i, t = 0) {
  // a 320×200 panel: layers of rocks, the cookie rock, the stick, the pug's speech
  const glowAll = i >= 1, hot = i >= 2, poke = i >= 3, cookie = i >= 4, loop = i === 5;
  let s = `<rect width="320" height="200" rx="18" fill="#f6efe2"/><path d="M0 170Q80 150 160 168T320 160V200H0Z" fill="#e3d3b4"/>`;
  for (let r = 0; r < ROWS; r++) {
    s += `<text x="10" y="${42 + r * 26}" font-family="monospace" font-size="9" fill="#9b8b70">L${(ROWS - 1 - r) * 7}</text>`;
    for (let c = 0; c < COLS; c++) {
      const isHot = r === HOT[0] && c === HOT[1];
      const wob = Math.sin((r * 3 + c * 7 + t) * 1.3);
      const glow = glowAll ? 0.15 + 0.35 * Math.abs(Math.sin(r * 1.7 + c * 2.3 + t * 0.8)) : 0;
      const fill = isHot && hot ? (poke ? '#ff8a3d' : '#f5b82e') : `rgba(245,184,46,${glow.toFixed(2)})`;
      const rad = isHot && poke ? 10 + (loop ? 2 * Math.abs(Math.sin(t * 3)) : 1.5 * Math.abs(Math.sin(t * 2))) : 8;
      s += `<ellipse cx="${40 + c * 26}" cy="${38 + r * 26 + wob * 0.6}" rx="${rad}" ry="${rad * 0.8}" fill="#8f8473" stroke="#5a5244" stroke-width="1.5"/>`;
      s += `<ellipse cx="${40 + c * 26}" cy="${38 + r * 26 + wob * 0.6}" rx="${rad - 2}" ry="${(rad - 2) * 0.8}" fill="${fill}"/>`;
      if (isHot && hot) s += `<text x="${40 + c * 26}" y="${42 + r * 26}" font-size="9" text-anchor="middle">🍪</text>`;
    }
  }
  if (glowAll && !poke) s += `<path d="M150 186V20" stroke="#3a9ad9" stroke-width="3" stroke-dasharray="6 6" stroke-dashoffset="${-t * 12}" opacity=".6"/>`;
  if (poke) {
    const hx = 40 + HOT[1] * 26, hy = 38 + HOT[0] * 26, d = 8 * Math.abs(Math.sin(t * (loop ? 6 : 3)));
    s += `<line x1="${hx + 70 - d}" y1="${hy + 50 - d * 0.7}" x2="${hx + 12 - d * 0.3}" y2="${hy + 9 - d * 0.2}" stroke="#7a4b23" stroke-width="7" stroke-linecap="round"/>`;
    s += `<text x="${hx + 74}" y="${hy + 70}" font-size="30">🧔</text>`;
    s += `<text x="${hx + 30}" y="${hy - 8}" font-family="Impact, sans-serif" font-size="14" fill="#e5484d">POKE!</text>`;
  }
  if (cookie) {
    const say = loop ? 'cookie cookie cookie cookie cookie cookie' : 'yes! and also… COOKIES 🍪';
    s += `<rect x="16" y="150" width="${loop ? 288 : 196}" height="30" rx="12" fill="#fff" stroke="#1d1a22" stroke-width="2"/>`;
    s += `<text x="26" y="170" font-family="sans-serif" font-size="13" font-weight="700" fill="#1d1a22">${say}</text>`;
    if (loop) s += `<text x="250" y="40" font-size="28" transform="rotate(${(t * 90) % 360} 264 30)">🌀</text>`;
  }
  return `<svg viewBox="0 0 320 200" xmlns="http://www.w3.org/2000/svg">${s}</svg>`;
}

let timer = null, scene = 0, tick = 0;
function paint() {
  const st = $('#cave-stage'); if (!st) return;
  st.innerHTML = sceneSvg(scene, tick / 10);
  $('#cave-cap').textContent = SCENES[scene].cap;
  $('#cave-sub').textContent = SCENES[scene].sub;
  document.querySelectorAll('.cave-dots i').forEach((d, i) => d.classList.toggle('on', i === scene));
}
function start() {
  stop();
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  timer = setInterval(() => {
    tick++;
    if (tick % 32 === 0) scene = (scene + 1) % SCENES.length;
    if (!reduce || tick % 32 === 0) paint();
  }, 100);
}
function stop() { clearInterval(timer); timer = null; }

export function openCaveman() {
  scene = 0; tick = 0;
  $('#cave-body').innerHTML = `<button class="x" data-act="close" aria-label="Close">✕</button>
    <p class="kicker">explain like I'm a caveman</p><h2>🪨 How one neuron changes a whole personality</h2>
    <div class="cave-stage" id="cave-stage"></div><h3 class="cave-cap" id="cave-cap"></h3><p class="muted cave-sub" id="cave-sub"></p>
    <div class="cave-dots">${SCENES.map((_, i) => `<i data-scene="${i}"></i>`).join('')}</div>
    <div class="row" style="justify-content:center"><button class="btn" data-act="prev">◀</button><button class="btn" data-act="next">▶</button>
      <button class="btn" data-act="strip">📸 copy as a comic strip</button></div>`;
  paint(); start();
  $('#cave-dlg').showModal();
}

async function strip(X) {
  const cols = 3, pw = 480, ph = 300, pad = 24, capH = 70, W = cols * pw + (cols + 1) * pad, H = 2 * (ph + capH) + 3 * pad + 90;
  const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
  const x = cv.getContext('2d');
  x.fillStyle = '#faf6ef'; x.fillRect(0, 0, W, H);
  x.fillStyle = '#1d1a22'; x.font = '800 40px "Bricolage Grotesque", system-ui'; x.fillText('🪨 One neuron, explained like to a caveman', pad, 58);
  for (const [i, sc] of SCENES.entries()) {
    const cx = pad + (i % cols) * (pw + pad), cy = 84 + Math.floor(i / cols) * (ph + capH + pad);
    const img = new Image();
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(sceneSvg(i, 1.2));
    await img.decode();
    x.drawImage(img, cx, cy, pw, ph);
    x.fillStyle = '#1d1a22'; x.font = '800 18px "Bricolage Grotesque", system-ui';
    X.wrapText(x, `${i + 1}. ${sc.cap}`, pw, 1).forEach(l => x.fillText(l, cx, cy + ph + 26));
    x.fillStyle = '#6f6878'; x.font = '400 14px system-ui';
    X.wrapText(x, sc.sub, pw, 2).forEach((l, j) => x.fillText(l, cx, cy + ph + 46 + j * 18));
  }
  x.font = '600 16px "JetBrains Mono", monospace'; x.fillStyle = '#6f6878';
  x.fillText('Evil Pugs · neuron steering on Qwen3.5-4B · github.com/makefunstuff/qwen-neurons', pad, H - 20);
  await X.copyPng(cv, 'evil-pugs-caveman.png');
}

export function initCaveman(X) {
  const dlg = $('#cave-dlg');
  dlg.addEventListener('close', stop);
  dlg.addEventListener('click', e => {
    const b = e.target.closest('button, i[data-scene]'); if (!b) return;
    if (b.dataset.act === 'close') return dlg.close();
    if (b.dataset.scene !== undefined) scene = +b.dataset.scene;
    if (b.dataset.act === 'next') scene = (scene + 1) % SCENES.length;
    if (b.dataset.act === 'prev') scene = (scene + SCENES.length - 1) % SCENES.length;
    if (b.dataset.act === 'strip') return strip(X);
    tick = 1; paint();
  });
}
export {sceneSvg, SCENES};
