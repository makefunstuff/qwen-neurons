// Every creature is a pug. The pug is a port of the Puggie pets (generator gen 2 + the SVG drawing, by the same
// author), drawn on a 100×100 square from a seed: same seed, same pug. On top of that, qwen-neurons adds what a
// creature's genes give it: trait gear (a cookie, a halo, devil horns …), an expression, and a mouth that talks.

// ---- the generator (FNV-1a seed hash + mulberry32; draw order fixed) ----------------------------------------------
export const COATS = ['#E8C08A', '#E8C08A', '#E8C08A', '#E9A76A', '#E9A76A', '#5A5A64', '#5A5A64', '#D9CFC0', '#A9845E',
  '#C6F432', '#B8A1E8', '#F2C6D0', '#8FB8DE'];
export const MASKS = ['full', 'full', 'muzzle', 'muzzle', 'goggles', 'none'];
export const EARS = ['rose', 'rose', 'button', 'button', 'perky', 'floppy'];
export const EYES = ['round', 'round', 'capsule', 'sleepy', 'derp'];
export const TONGUES = ['none', 'none', 'blep', 'blep', 'long'];
export const MARKINGS = ['none', 'none', 'thumbprint', 'blaze', 'spots', 'brindle'];
export const ACCESSORIES = ['none', 'none', 'none', 'bandana', 'bow', 'beanie', 'glasses', 'headset', 'goo'];
export const ACCESSORY_COLORS = ['#E5484D', '#4C9EFF', '#F5A524', '#3DD68C', '#8E4EC6'];
export const INHERITED = ['coat', 'mask', 'ears', 'eyes', 'tongue', 'wrinkles', 'markings', 'accessory', 'accessoryColor', 'blush'];

export function rng(seed) {
  let h = 2166136261;
  const s = String(seed);
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  let a = h >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Traits of the pug for `seed`; `pins` override any drawn trait (they never change the draws). */
export function generate(seed, pins = {}) {
  const r = rng(seed);
  const pick = xs => xs[Math.floor(r() * xs.length)];
  const span = (lo, hi) => lo + r() * (hi - lo);
  const traits = {
    coat: pick(COATS), mask: pick(MASKS), ears: pick(EARS), eyes: pick(EYES), tongue: pick(TONGUES),
    wrinkles: Math.floor(r() * 4), markings: pick(MARKINGS), accessory: pick(ACCESSORIES),
    rx: span(31, 37), ry: span(26, 31), gap: span(22, 29), muzzle: span(0.85, 1.15), ear: span(0.85, 1.15), tilt: span(-10, 6),
    blush: r() > 0.5, delay: span(0, 3), accessoryColor: pick(ACCESSORY_COLORS),
  };
  return {...traits, ...pins};
}

// ---- drawing helpers ------------------------------------------------------------------------------------------------
const INK = '#121212', CREAM = '#f4f1ea', TONGUE = '#ff7a93', GOO = '#c6f432', GOLD = '#f5b82e', RED = '#e5484d';
const n = x => String(Math.round(x * 100) / 100);

export function shade(hex, f) {
  const v = parseInt(hex.slice(1), 16);
  const ch = c => Math.min(255, Math.max(0, Math.round(f < 0 ? c * (1 + f) : c + (255 - c) * f)));
  return `#${[(v >> 16) & 255, (v >> 8) & 255, v & 255].map(c => ch(c).toString(16).padStart(2, '0')).join('')}`;
}
export function mix(a, b, t) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const c = [16, 8, 0].map(s => Math.round(((pa >> s) & 255) * (1 - t) + ((pb >> s) & 255) * t));
  return `#${c.map(x => x.toString(16).padStart(2, '0')).join('')}`;
}

const oval = (cx, cy, rx, ry, fill, extra = '') =>
  `<ellipse cx="${n(cx)}" cy="${n(cy)}" rx="${n(rx)}" ry="${n(ry)}" fill="${fill}"${extra}/>`;
const circle = (cx, cy, r, fill, extra = '') => `<circle cx="${n(cx)}" cy="${n(cy)}" r="${n(r)}" fill="${fill}"${extra}/>`;
const ring = (cx, cy, r, stroke, w) => `<circle cx="${n(cx)}" cy="${n(cy)}" r="${n(r)}" fill="none" stroke="${stroke}" stroke-width="${w}"/>`;
const rect = (x, y, w, h, r, fill, extra = '') =>
  `<rect x="${n(x)}" y="${n(y)}" width="${n(w)}" height="${n(h)}" rx="${n(r)}" fill="${fill}"${extra}/>`;
const line = (x1, y1, x2, y2, stroke, w, round = true) =>
  `<line x1="${n(x1)}" y1="${n(y1)}" x2="${n(x2)}" y2="${n(y2)}" stroke="${stroke}" stroke-width="${w}"${round ? ' stroke-linecap="round"' : ''}/>`;
const fillPath = (d, fill, extra = '') => `<path d="${d}" fill="${fill}"${extra}/>`;
const strokePath = (d, stroke, w, extra = '') =>
  `<path d="${d}" fill="none" stroke="${stroke}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"${extra}/>`;
const tri = (x1, y1, x2, y2, x3, y3, fill) => fillPath(`M${n(x1)} ${n(y1)}L${n(x2)} ${n(y2)}L${n(x3)} ${n(y3)}Z`, fill);
const softTri = (x1, y1, x2, y2, x3, y3, fill) =>
  fillPath(`M${n(x1)} ${n(y1)}L${n(x2)} ${n(y2)}L${n(x3)} ${n(y3)}Z`, fill, ` stroke="${fill}" stroke-width="4" stroke-linejoin="round"`);
const rotate = (deg, cx, cy, body) => `<g transform="rotate(${n(deg)} ${n(cx)} ${n(cy)})">${body}</g>`;
const scale = (s, cx, cy, body) => `<g transform="translate(${n(cx)} ${n(cy)}) scale(${n(s)}) translate(${n(-cx)} ${n(-cy)})">${body}</g>`;
const move = (x, y, body) => `<g transform="translate(${n(x)} ${n(y)})">${body}</g>`;
const topArc = (x, y, w, h) => `M${n(x)} ${n(y + h / 2)}A${n(w / 2)} ${n(h / 2)} 0 0 1 ${n(x + w)} ${n(y + h / 2)}`;
const heart = (x, y, s, fill) => fillPath(
  `M${n(x)} ${n(y + s * 0.9)}C${n(x - s * 1.3)} ${n(y)} ${n(x - s * 0.7)} ${n(y - s * 0.9)} ${n(x)} ${n(y - s * 0.2)}` +
  `C${n(x + s * 0.7)} ${n(y - s * 0.9)} ${n(x + s * 1.3)} ${n(y)} ${n(x)} ${n(y + s * 0.9)}Z`, fill);
const sparkle = (x, y, s, fill) => fillPath(
  `M${n(x)} ${n(y - s)}L${n(x + s * 0.28)} ${n(y - s * 0.28)}L${n(x + s)} ${n(y)}L${n(x + s * 0.28)} ${n(y + s * 0.28)}` +
  `L${n(x)} ${n(y + s)}L${n(x - s * 0.28)} ${n(y + s * 0.28)}L${n(x - s)} ${n(y)}L${n(x - s * 0.28)} ${n(y - s * 0.28)}Z`, fill);
const drop = (x, y, s, fill) => fillPath(`M${n(x)} ${n(y - s)}Q${n(x + s * 0.9)} ${n(y + s * 0.2)} ${n(x)} ${n(y + s * 0.8)}` +
  `Q${n(x - s * 0.9)} ${n(y + s * 0.2)} ${n(x)} ${n(y - s)}Z`, fill);

function eye(cx, cy, w, h, r, open, color = INK) {
  const hh = h * open;
  return rect(cx - w / 2, cy - hh / 2, w, hh, Math.min(r, hh / 2), color);
}

function pugEye(kind, cx, cy, grow, open, side, patched) {
  const rimmed = (y, w, h, r) => (patched ? eye(cx, y, w + 2.6, h + 2.6, r + 1.3, open, CREAM) : '') + eye(cx, y, w, h, r, open);
  if (kind === 'capsule') return rimmed(cy, 8 * grow, 14 * grow, 4);
  if (kind === 'sleepy') return rimmed(cy + 2, 9.5 * grow, 6.5 * grow, 3.25);
  const small = kind === 'derp' && side === 1;
  const r = (small ? 4.6 : 6.2) * grow;
  let out = oval(cx, cy, r + 1.3, (r + 1.3) * open, CREAM) + oval(cx, cy, r, r * open, INK);
  if (open > 0.5) out += circle(cx + (kind === 'derp' ? side * 1.8 : -1.9), cy - r * 0.35, r * 0.32, '#fff');
  return out;
}

// ---- trait gear: what a creature's genes make it wear. {back, head, front} layers. --------------------------------
function gearSvg(g, ctx) {
  const {top, rx, cy, eyeLx, eyeRx, eyeCy} = ctx;
  switch (g) {
    case 'cookie': {
      const c = '#c98a4b';
      return {front: circle(79, 81, 12, c) + circle(79, 81, 12, 'none', ` stroke="${shade(c, -0.25)}" stroke-width="1.5"`) +
        [[74, 77], [83, 76], [77, 86], [85, 85], [80, 81]].map(([x, y]) => circle(x, y, 1.8, '#4a2c17')).join('') +
        circle(89, 72, 4.5, 'var(--pug-bg, #fff)')};
    }
    case 'halo':
      return {back: oval(50, top - 7, 19, 5.5, 'none', ` stroke="${GOLD}" stroke-width="3.2"`) +
        oval(50, top - 7, 19, 5.5, 'none', ` stroke="#fff5c4" stroke-width="1" opacity="0.8"`)};
    case 'stars':
      return {front: sparkle(15, 22, 6, GOLD) + sparkle(86, 18, 4.5, '#fff') + sparkle(88, 50, 3.5, GOLD) + circle(20, 46, 1.4, '#fff')};
    case 'sparkles':
      return {front: sparkle(14, 24, 5, '#ff7ab8') + sparkle(87, 20, 4, '#ffd166') + sparkle(86, 54, 3.5, '#ff7ab8') + sparkle(16, 60, 3, '#ffd166')};
    case 'rings':
      return {front: ring(77, 84, 6.5, GOLD, 2.6) + ring(86, 84, 6.5, '#d9d9e3', 2.6) + sparkle(77, 76.5, 2.6, '#9ee7ff')};
    case 'sweat':
      return {head: drop(50 + rx - 6, top + 14, 5, '#7cc6ff') + circle(50 + rx - 7, top + 12.5, 1, '#fff')};
    case 'check':
      return {front: circle(84, 18, 9.5, '#4c9eff') + strokePath('M79.5 18.5L83 22L88.5 14.5', '#fff', 2.8)};
    case 'hearts':
      return {front: heart(84, 20, 6, '#ff6b8b') + heart(17, 30, 4.5, '#ff9bb0') + heart(88, 44, 3, '#ff6b8b')};
    case 'heart':
      return {front: heart(81, 81, 9, '#ff5fa2') + circle(77.5, 78, 1.8, '#fff', ' opacity="0.7"')};
    case 'flag':
      return {head: line(50 + rx * 0.55, top - 20, 50 + rx * 0.55, top + 6, '#6b6b6b', 2) +
        rect(50 + rx * 0.55 + 1, top - 20, 15, 5.5, 1, '#e5484d') + rect(50 + rx * 0.55 + 1, top - 14.5, 15, 5.5, 1, '#fff',
        ' stroke="#ddd" stroke-width="0.6"')};
    case 'shout':
      return {front: rect(80, 6, 4.5, 15, 2.2, RED) + circle(82.25, 25.5, 2.4, RED) + rect(88, 6, 4.5, 15, 2.2, RED) + circle(90.25, 25.5, 2.4, RED)};
    case 'tear':
      return {head: drop(eyeLx - 2, eyeCy + 12, 3.6, '#7cc6ff') + drop(eyeRx + 3, eyeCy + 15, 2.8, '#7cc6ff')};
    case 'nightcap':
      return {head: fillPath(`M${n(50 - rx * 0.85)} ${n(top + 8)}Q${n(48)} ${n(top - 26)} ${n(50 + rx + 8)} ${n(top + 14)}` +
        `Q${n(60)} ${n(top - 6)} ${n(50 + rx * 0.85)} ${n(top + 8)}Z`, '#6f73d2') +
        rect(50 - rx * 0.88, top + 4, rx * 1.76, 7, 3.5, '#e8e8ff') + circle(50 + rx + 8, top + 15, 4.5, '#e8e8ff')};
    case 'horn':
      return {back: tri(44, top + 6, 50, top - 26, 56, top + 6, '#ffd36e') +
        strokePath(`M45.5 ${n(top - 1)}L54 ${n(top - 4)}M47 ${n(top - 9)}L53 ${n(top - 12)}M48.3 ${n(top - 17)}L51.6 ${n(top - 19)}`, '#e09a1e', 1.4),
        front: sparkle(16, 22, 4, '#c4a7ff') + sparkle(86, 28, 3, '#9ee7ff')};
    case 'catears':
      return {head: strokePath('M28 66L14 63M28 70L14 72M72 66L86 63M72 70L86 72', INK, 1.1, ' opacity="0.55"'),
        back: strokePath('M82 92Q98 84 90 66', '#c27a45', 6)};
    case 'cape':
      return {back: fillPath('M26 70Q8 92 6 100L94 100Q92 92 74 70Z', '#d62828') + fillPath('M26 70Q50 84 74 70L70 78Q50 90 30 78Z', '#a61b1b')};
    case 'gloves':
      return {front: circle(18, 86, 8, '#fff', ' stroke="#ccc" stroke-width="1"') + circle(82, 86, 8, '#fff', ' stroke="#ccc" stroke-width="1"') +
        strokePath('M15 84L15 89M19 83L19 89M79 84L79 89M83 83L83 89', '#bbb', 1)};
    case 'headband':
      return {head: rect(50 - rx + 1, top + 7, rx * 2 - 2, 7, 2, '#1f3c88') + rect(41, top + 6, 18, 9, 2, '#c9ced6') +
        strokePath(`M46 ${n(top + 10.5)}Q50 ${n(top + 6.5)} 54 ${n(top + 10.5)}`, '#59606b', 1.4) +
        strokePath(`M${n(50 + rx - 1)} ${n(top + 10)}L${n(50 + rx + 10)} ${n(top + 18)}M${n(50 + rx - 1)} ${n(top + 12)}L${n(50 + rx + 7)} ${n(top + 23)}`, '#1f3c88', 3)};
    case 'sword':
      return {front: rotate(-30, 84, 70, rect(82, 40, 5, 34, 1.5, '#d9dee6') + tri(82, 40, 84.5, 33, 87, 40, '#d9dee6') +
        rect(77, 73, 15, 4, 2, '#8a5a2b') + rect(83, 77, 3, 9, 1.5, '#5c3a1a'))};
    case 'sign':
      return {front: line(80, 68, 80, 98, '#8a5a2b', 2.4) + rect(64, 58, 32, 17, 3, '#fff', ' stroke="#222" stroke-width="1.5"') +
        `<text x="80" y="70.5" font-family="Impact, 'Arial Black', sans-serif" font-size="10" text-anchor="middle" fill="#111">lol</text>`};
    case 'horns':
      return {back: fillPath(`M${n(50 - rx * 0.62)} ${n(top + 8)}Q${n(50 - rx * 0.9)} ${n(top - 10)} ${n(50 - rx * 1.05)} ${n(top - 16)}` +
        `Q${n(50 - rx * 0.45)} ${n(top - 6)} ${n(50 - rx * 0.3)} ${n(top + 6)}Z`, '#b3121b') +
        fillPath(`M${n(50 + rx * 0.62)} ${n(top + 8)}Q${n(50 + rx * 0.9)} ${n(top - 10)} ${n(50 + rx * 1.05)} ${n(top - 16)}` +
        `Q${n(50 + rx * 0.45)} ${n(top - 6)} ${n(50 + rx * 0.3)} ${n(top + 6)}Z`, '#b3121b'),
        head: fillPath('M50 67Q41 64 36 69Q33 72 30 69Q33 76 40 72Q45 69 50 70Q55 69 60 72Q67 76 70 69Q67 72 64 69Q59 64 50 67Z', '#1a1110')};
    case 'crown':
      return {head: fillPath(`M36 ${n(top + 4)}L36 ${n(top - 10)}L43 ${n(top - 3)}L50 ${n(top - 13)}L57 ${n(top - 3)}L64 ${n(top - 10)}L64 ${n(top + 4)}Z`, GOLD) +
        circle(50, top - 2, 2, RED) + circle(42, top + 1, 1.4, '#4c9eff') + circle(58, top + 1, 1.4, '#3dd68c')};
    case 'propeller':
      return {back: rect(48.5, top - 12, 3, 12, 1, '#555') + oval(50, top - 13, 13, 3, '#4c9eff') + circle(50, top - 13, 2.5, RED) +
        fillPath(topArc(38, top - 4, 24, 16) + 'Z', '#f5a524')};
    case 'monocle':
      return {head: ring(eyeRx, eyeCy, 9, GOLD, 2) + strokePath(`M${n(eyeRx + 6)} ${n(eyeCy + 7)}Q${n(eyeRx + 10)} ${n(eyeCy + 20)} ${n(eyeRx + 4)} ${n(eyeCy + 30)}`, GOLD, 1)};
    case 'bowtie':
      return {front: tri(50, 88, 38, 82, 38, 94, '#8e4ec6') + tri(50, 88, 62, 82, 62, 94, '#8e4ec6') + circle(50, 88, 3, '#5d2c8a')};
    case 'antenna':
      return {back: strokePath(`M50 ${n(top + 4)}Q46 ${n(top - 8)} 52 ${n(top - 16)}`, '#555', 2) + circle(52, top - 17, 3.5, '#3dd68c')};
    case 'leaf':
      return {head: line(50, top + 2, 50, top - 8, '#2fb574', 2.2) + fillPath(`M50 ${n(top - 6)}Q40 ${n(top - 16)} 35 ${n(top - 10)}Q43 ${n(top - 3)} 50 ${n(top - 6)}Z`, '#3dd68c')};
    case 'eyepatch':
      return {head: line(50 - rx + 2, eyeCy - 10, 50 + rx - 2, eyeCy - 4, INK, 1.6) + oval(eyeLx, eyeCy, 7.5, 6.5, INK) +
        tri(36, top + 6, 50, top - 14, 64, top + 6, '#2b2b33') + oval(50, top + 5, 24, 4.5, '#2b2b33') + circle(50, top - 3, 2.6, '#f4f1ea')};
    case 'goggles':
      return {head: rect(50 - rx + 1, eyeCy - 3, rx * 2 - 2, 5, 2.5, '#6b4a2b') + ring(eyeLx, eyeCy, 7, '#c9a227', 3) +
        ring(eyeRx, eyeCy, 7, '#c9a227', 3) + circle(eyeLx, eyeCy, 5.5, '#9ee7ff', ' opacity="0.45"') +
        circle(eyeRx, eyeCy, 5.5, '#9ee7ff', ' opacity="0.45"')};
    case 'tinfoil':
      return {head: fillPath(`M${n(50 - rx * 0.78)} ${n(top + 8)}L42 ${n(top - 9)}L46 ${n(top - 6)}L50 ${n(top - 22)}L54 ${n(top - 8)}` +
        `L58 ${n(top - 11)}L${n(50 + rx * 0.78)} ${n(top + 8)}Z`, '#c9ced6', ' stroke="#8a929c" stroke-width="1.2" stroke-linejoin="round"') +
        strokePath(`M45 ${n(top - 2)}L49 ${n(top + 3)}L53 ${n(top - 4)}M55 ${n(top + 1)}L58 ${n(top - 3)}`, '#eef1f5', 1.2) +
        sparkle(50, top - 23, 3, '#ffffff')};
    case 'bars':
      return {front: [18, 32, 46, 60, 74, 88].map(x => rect(x - 1.6, 4, 3.2, 96, 1.6, '#4a4f57', ' opacity="0.85"')).join('') +
        rect(8, 30, 84, 3.4, 1.7, '#4a4f57', ' opacity="0.85"') + rect(8, 84, 84, 3.4, 1.7, '#4a4f57', ' opacity="0.85"') +
        rect(66, 6, 26, 12, 2, '#76b900') + `<text x="79" y="15" font-family="monospace" font-size="7" font-weight="700" text-anchor="middle" fill="#fff">GPU</text>`};
    case 'pinboard':
      return {front: rect(63, 64, 33, 27, 2, '#c8955a', ' stroke="#8a5a2b" stroke-width="2"') +
        rect(66, 67, 8, 7, 0.5, '#fff8d6') + rect(84, 69, 9, 6, 0.5, '#fff') + rect(73, 80, 9, 7, 0.5, '#fff8d6') +
        strokePath('M70 70L88 72L77 83L70 70M88 72L91 86', '#d62828', 0.9) +
        circle(70, 70, 1.7, RED) + circle(88, 72, 1.7, '#4c9eff') + circle(77, 83, 1.7, GOLD) + circle(91, 86, 1.7, RED)};
    case 'partyhat':
      return {head: tri(38, top + 5, 54, top - 26, 64, top + 3, '#8e4ec6') + line(43, top - 2, 59, top - 5, GOLD, 2.4) +
        line(48, top - 12, 57, top - 14, GOLD, 2.4) + circle(54, top - 27, 4, '#ff8fa3')};
    default: return {};
  }
}
export const RANDOM_GEAR = ['crown', 'propeller', 'monocle', 'bowtie', 'antenna', 'leaf', 'partyhat'];

// ---- expressions ----------------------------------------------------------------------------------------------------
function brows(kind, eyeLx, eyeRx, eyeCy, color) {
  const y = eyeCy - 11;
  if (kind === 'angry') return line(eyeLx - 6, y - 3, eyeLx + 5, y + 2.5, color, 2.8) + line(eyeRx + 6, y - 3, eyeRx - 5, y + 2.5, color, 2.8);
  if (kind === 'worried') return line(eyeLx - 6, y + 2, eyeLx + 5, y - 3, color, 2.4) + line(eyeRx + 6, y + 2, eyeRx - 5, y - 3, color, 2.4);
  if (kind === 'up') return strokePath(`M${n(eyeLx - 5)} ${n(y)}Q${n(eyeLx)} ${n(y - 5)} ${n(eyeLx + 5)} ${n(y)}`, color, 2.2) +
    strokePath(`M${n(eyeRx - 5)} ${n(y)}Q${n(eyeRx)} ${n(y - 5)} ${n(eyeRx + 5)} ${n(y)}`, color, 2.2);
  if (kind === 'flat') return line(eyeLx - 6, y, eyeLx + 5, y + 1.5, color, 2.6) + line(eyeRx + 6, y, eyeRx - 5, y + 1.5, color, 2.6);
  return '';
}

const MOOD_FACE = {
  evil: {brows: 'angry', grin: true}, brave: {brows: 'flat'}, sad: {brows: 'worried'}, sorry: {brows: 'worried'},
  shout: {brows: 'up', mouth: 'open'}, eager: {brows: 'up', grow: 1.18}, love: {eyes: 'hearts', blush: true},
  kawaii: {eyes: 'sparkle', blush: true}, sleepy: {eyes: 'shut'}, serene: {eyes: 'happy'}, smug: {eyes: 'smug'},
  mystic: {eyes: 'stars'}, suspicious: {brows: 'flat', eyes: 'smug'}, happy: {}, derp: {eyes: 'derp'}, surprised: {brows: 'up', mouth: 'open', grow: 1.1},
};
export const RANDOM_MOODS = ['happy', 'smug', 'derp', 'surprised', 'eager'];

/**
 * The creature as the inside of an `<svg viewBox="0 0 100 100">`.
 * opts: mood, gear (list of gear keys), bg (backdrop fill; 'var(--pug-bg)' by default).
 * Moving parts are classed: pet-body (bob), pet-eyes (blink), pet-talk (the talking mouth, shown while .talking).
 */
export function pugSvg(t, {mood = 'happy', gear = [], bg = 'var(--pug-bg, #f3eee6)'} = {}) {
  const face = MOOD_FACE[mood] || {};
  const coat = t.coat.toLowerCase();
  const darkCoat = coat === '#5a5a64';
  const mask = darkCoat ? shade(coat, -0.55) : '#2e2622';
  const fold = shade(coat, darkCoat ? -0.35 : -0.22);
  const acc = t.accessoryColor, accDark = shade(acc, -0.35);
  const {rx, ry} = t;
  const cy = 56, top = cy - ry, m = t.muzzle, eyeCy = 52;
  const eyeLx = 50 - t.gap / 2, eyeRx = 50 + t.gap / 2, earL = 50 - rx, earR = 50 + rx;
  const patched = t.mask === 'full' || t.mask === 'goggles';
  const ctx = {top, rx, cy, eyeLx, eyeRx, eyeCy};
  const layers = gear.map(g => gearSvg(g, ctx));
  const back = layers.map(l => l.back || '').join(''), front = layers.map(l => l.front || '').join('');
  const gearHead = layers.map(l => l.head || '').join('');

  let head = '';
  if (t.ears === 'perky') {
    head += scale(t.ear, 50, 40, softTri(earL + 6, 40, earL + 4, top - 8, earL + 22, top + 6, mask) +
      softTri(earR - 6, 40, earR - 4, top - 8, earR - 22, top + 6, mask));
  }
  head += oval(50, cy, rx, ry, coat) + oval(50 - rx * 0.5, cy + ry * 0.38, rx * 0.56, ry * 0.58, coat) +
    oval(50 + rx * 0.5, cy + ry * 0.38, rx * 0.56, ry * 0.58, coat);
  switch (t.markings) {
    case 'spots':
      head += circle(50 + rx * 0.62, cy - 10, 3.6, fold) + circle(50 + rx * 0.78, cy - 2, 2.6, fold) + circle(50 + rx * 0.45, top + 6, 2.2, fold);
      break;
    case 'brindle':
      for (let i = 0; i <= 2; i++) {
        const x = 50 - rx + 4 + i * 4;
        head += line(x, cy - 12 + i * 2, x + 3, cy + 2 + i * 2, fold, 1.8) + line(100 - x, cy - 12 + i * 2, 100 - x - 3, cy + 2 + i * 2, fold, 1.8);
      }
      break;
    case 'blaze': head += rect(47.5, top + 2, 5, 20, 2.5, CREAM); break;
    case 'thumbprint': head += oval(50, top + 8, 5, 4, mask); break;
  }
  for (let i = 0; i < t.wrinkles; i++) {
    const y = top + 11 + i * 3.6, w = 7 - i * 1.2;
    head += strokePath(`M${n(50 - w)} ${n(y + 1.5)}Q50 ${n(y - 2)} ${n(50 + w)} ${n(y + 1.5)}`, fold, 1.8);
  }
  if (patched) head += oval(eyeLx, eyeCy, 9.5, 9, mask) + oval(eyeRx, eyeCy, 9.5, 9, mask);
  const muzzleColor = t.mask === 'full' || t.mask === 'muzzle' ? mask : shade(coat, 0.35);
  head += oval(50, 67, 15 * m, 11 * m, muzzleColor);
  if (t.mask === 'full') head += rect(46, eyeCy, 8, 10, 4, mask);
  const onMask = muzzleColor === mask;
  const ink = onMask ? shade(coat, 0.25) : INK;
  const inkOpacity = onMask ? ' stroke-opacity="0.55"' : '';
  head += strokePath('M43 60.5Q50 56.5 57 60.5', onMask ? ink : fold, 1.8, onMask ? inkOpacity : '');
  head += oval(50, 63.5, 5.5, 3.6, INK) + circle(48.3, 62.6, 1.2, '#fff', ' fill-opacity="0.35"');

  // the mouth: still (with tongue), grin, or wide open; plus a talking mouth that only shows while talking
  let still = '';
  if (face.grin) {
    still = fillPath('M41 69Q50 79 59 69Q50 73 41 69Z', '#3a0d0d') + tri(43.5, 70.2, 45.5, 74.5, 47, 71.2, '#fff') + tri(53, 71.2, 54.5, 74.5, 56.5, 70.2, '#fff');
  } else if (face.mouth === 'open') {
    still = oval(50, 72, 5.5, 5.5, '#3a0d0d') + oval(50, 75, 3.5, 2.2, TONGUE);
  } else {
    if (t.tongue === 'blep') still += oval(50, 72.5, 3.4, 3.8, TONGUE);
    else if (t.tongue === 'long') still += rect(46.5, 69.5, 7, 10, 3.5, TONGUE) + line(50, 71, 50, 76, shade(TONGUE, -0.2), 1.2);
    still += strokePath('M50 66.5L50 69Q47.5 72 44.5 69.5M50 69Q52.5 72 55.5 69.5', ink, 1.6, inkOpacity);
  }
  head += `<g class="pet-still">${still}</g>`;
  head += `<g class="pet-talk">${oval(50, 71.5, 5, 4.5, '#3a0d0d')}${oval(50, 74, 3.2, 1.8, TONGUE)}</g>`;

  let ears = '';
  if (t.ears === 'rose') {
    ears = softTri(earL + 4, top + 12, earL - 5, top + 2, earL + 16, top + 1, mask) + softTri(earR - 4, top + 12, earR + 5, top + 2, earR - 16, top + 1, mask);
  } else if (t.ears === 'button') {
    ears = softTri(earL + 1, top + 9, earL + 17, top + 1, earL + 9, top + 18, mask) + softTri(earR - 1, top + 9, earR - 17, top + 1, earR - 9, top + 18, mask);
  } else if (t.ears === 'floppy') {
    ears = rotate(14, earL + 3, top + 16, oval(earL + 3, top + 16, 7, 14, mask)) + rotate(-14, earR - 3, top + 16, oval(earR - 3, top + 16, 7, 14, mask));
  }
  if (ears) head += scale(t.ear, 50, cy, ears);

  const hatGear = gear.some(g => ['nightcap', 'horn', 'halo', 'crown', 'partyhat', 'propeller', 'headband', 'flag', 'horns'].includes(g));
  const accessory = hatGear && ['beanie', 'goo', 'headset'].includes(t.accessory) ? 'none' : t.accessory;
  switch (accessory) {
    case 'beanie':
      head += fillPath(`${topArc(50 - rx * 0.8, top + 9 - ry * 0.7, rx * 1.6, ry * 1.4)}Z`, acc) +
        rect(50 - rx * 0.8 - 1, top + 5, rx * 1.6 + 2, 7, 3, accDark) + circle(50, top - 11, 5.5, acc);
      break;
    case 'bow': {
      const bx = 50 + rx * 0.45, by = top + 3;
      head += tri(bx, by, bx - 9, by - 6.5, bx - 9, by + 6.5, acc) + tri(bx, by, bx + 9, by - 6.5, bx + 9, by + 6.5, acc) + circle(bx, by, 3, accDark);
      break;
    }
    case 'headset':
      head += strokePath(topArc(50 - rx + 1, cy - ry - 3, (rx - 1) * 2, (ry + 3) * 2), '#1a1a1a', 5, ' stroke-linecap="butt"') +
        rect(50 - rx - 4, cy - 6, 10, 16, 4, '#1a1a1a') + rect(50 + rx - 6, cy - 6, 10, 16, 4, '#1a1a1a') + circle(50 + rx - 1, cy + 2, 2, acc);
      break;
    case 'bandana':
      head += tri(30, 83, 70, 83, 50, 97, acc) + rect(29, 81, 42, 4.5, 2.25, accDark) +
        [[47, 88], [53, 88], [50, 92.5]].map(([x, y]) => circle(x, y, 1.3, CREAM, ' fill-opacity="0.8"')).join('');
      break;
    case 'goo':
      head += fillPath(`M37 ${n(top + 5)}Q38 ${n(top - 7)} 48 ${n(top - 6)}Q53 ${n(top - 12)} 59 ${n(top - 5)}Q65 ${n(top - 3)} 63 ${n(top + 5)}Z`, GOO) +
        rect(56, top + 2, 4, 9, 2, GOO) + circle(58, top + 11, 2.6, GOO) + circle(46, top - 3, 1.6, '#fff', ' fill-opacity="0.5"');
      break;
  }

  // eyes, by expression
  const lid = patched ? CREAM : INK;
  const arc = (cx, up) => `M${n(cx - 4.5)} ${n(eyeCy + (up ? 2 : 0))}Q${n(cx)} ${n(eyeCy + (up ? -3 : 5))} ${n(cx + 4.5)} ${n(eyeCy + (up ? 2 : 0))}`;
  const grow = face.grow || 1;
  let eyes;
  if (face.eyes === 'shut') eyes = strokePath(arc(eyeLx, false), lid, 2.6) + strokePath(arc(eyeRx, false), lid, 2.6);
  else if (face.eyes === 'happy') eyes = strokePath(arc(eyeLx, true), lid, 2.6) + strokePath(arc(eyeRx, true), lid, 2.6);
  else if (face.eyes === 'hearts') eyes = heart(eyeLx, eyeCy, 6, '#ff3d6e') + heart(eyeRx, eyeCy, 6, '#ff3d6e');
  else if (face.eyes === 'sparkle') {
    eyes = [eyeLx, eyeRx].map(x => oval(x, eyeCy, 7, 7.5, CREAM) + oval(x, eyeCy, 6, 6.6, INK) + circle(x - 2, eyeCy - 2.6, 2.4, '#fff') +
      circle(x + 2.4, eyeCy + 2.2, 1.2, '#fff')).join('');
  } else if (face.eyes === 'stars') eyes = sparkle(eyeLx, eyeCy, 7, '#2b1250') + sparkle(eyeRx, eyeCy, 7, '#2b1250') + circle(eyeLx, eyeCy, 1.4, GOLD) + circle(eyeRx, eyeCy, 1.4, GOLD);
  else if (face.eyes === 'smug') eyes = pugEye('sleepy', eyeLx, eyeCy, 1, 1, -1, patched) + pugEye('sleepy', eyeRx, eyeCy, 1, 1, 1, patched);
  else {
    const kind = face.eyes === 'derp' ? 'derp' : t.eyes;
    eyes = pugEye(kind, eyeLx, eyeCy, grow, 1, -1, patched) + pugEye(kind, eyeRx, eyeCy, grow, 1, 1, patched);
  }
  head += `<g class="pet-eyes">${eyes}</g>`;
  if (accessory === 'glasses' && !['hearts', 'stars'].includes(face.eyes)) {
    const g = darkCoat ? CREAM : INK;
    head += ring(eyeLx, eyeCy, 10, g, 2.4) + ring(eyeRx, eyeCy, 10, g, 2.4) + line(eyeLx + 10, eyeCy, eyeRx - 10, eyeCy, g, 2.4, false);
  }
  head += brows(face.brows, eyeLx, eyeRx, eyeCy, darkCoat ? CREAM : INK);
  if (t.blush || face.blush) head += oval(50 - rx * 0.62, 66, 5, 3, '#ff8fa3', ' fill-opacity="0.6"') + oval(50 + rx * 0.62, 66, 5, 3, '#ff8fa3', ' fill-opacity="0.6"');
  head += gearHead;

  let out = bg ? rect(0, 0, 100, 100, 28, bg) : '';
  out += `<g class="pet-body" style="animation-delay:-${n(t.delay)}s">${rotate(t.tilt, 50, cy, back + head)}</g>${front}`;
  if (mood === 'sleepy') out += strokePath('M78 16L88 16L78 28L88 28', '#8a8fd8', 2.6, ' class="pet-z"');
  return out;
}

export function svgTag(inner, cls = 'pet', extra = '') {
  return `<svg class="${cls}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"${extra}>${inner}</svg>`;
}
