/* Halaman Pantauan: bola alarm, peta, garis waktu, meteogram, METAR. */
const J = { lon: 112.7876, lat: -7.3798 };
const HKEYS = { '1h': 60, '3h': 180, '9h': 540, '18h': 1080, '24h': 1440 };
let T0 = null, OBS = [], LAST = null, HZ = [], EWS = null, INFO = null, GEOJ = null, PETA = null;
let set = 'real', t = 0, tok = null, layer = 'p', wst = calm ? 'barb' : 'flow', playing = 0, alarmOn = false, lastLevel = null;
const at0 = m => T0 + m * 60000;
const hm = m => wibClock(at0(m));

/* ---------- membaca data ---------- */
function decode(r) {
  const o = { raw: r.raw_metar || null, time: utcMs(r.slot_30min), temp: r.temperature, dew: r.dew_point, q: r.pressure, spd: r.wind_speed ?? 0, dir: null, vis: null, cb: '', ts: +r.bad_weather === 1 };
  if (o.raw) {
    const body = o.raw.split(/\b(?:NOSIG|TEMPO|BECMG|RMK)\b/)[0];
    const w = body.match(/\b(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT\b/), v = body.match(/KT(?: \d{3}V\d{3})? (\d{4})\b/), c = body.match(/(?:FEW|SCT|BKN|OVC)\d{3}(CB|TCU)/);
    if (w) { o.dir = w[1] === 'VRB' ? null : +w[1]; o.spd = +w[2]; }
    if (v) o.vis = +v[1];
    if (c) o.cb = c[1];
    o.ts = /(^|\s)([+-]|VC)?TS[A-Z]*(\s|$)/.test(body);
  }
  return o;
}
const lv = (p, w, b) => p < w ? p / w * 55 : p < b ? 55 + (p - w) / (b - w) * 30 : 85 + Math.min(10, (p - b) / (1 - b) * 10);
const ST = l => l >= 85 ? { n: 'Bahaya', k: 'ba', i: 2 } : l >= 55 ? { n: 'Waspada', k: 'wa', i: 1 } : { n: 'Aman', k: 'ok', i: 0 };
function buildHorizons() {
  const ev = INFO && INFO.evaluation.horizons;
  HZ = EWS.predictions.map(p => ({
    key: p.horizon, name: `+${parseInt(p.horizon, 10)} jam`, min: HKEYS[p.horizon], p: p.probability, w: p.thresholds.th_waspada, b: p.thresholds.th_bahaya,
    model: p.model, lstm: p.lstm ? p.lstm.probability : null, reliable: ev ? ev[p.horizon].better_than_climatology : HKEYS[p.horizon] <= 540,
  }));
}
function pOf(h) {
  if (set === 'wa') return h.key === '3h' ? mix(h.w, h.b, .5) : h.w * .35;
  if (set === 'ba') return h.key === '1h' ? h.b + (1 - h.b) * .3 : h.key === '3h' ? mix(h.w, h.b, .3) : h.w * .3;
  return h.p;
}
const calc = i => { const h = HZ[i], p = pOf(h), l = lv(p, h.w, h.b); return { hi: i, p, lv: l, ...ST(l) }; };
const worst = () => HZ.map((h, i) => i).filter(i => HZ[i].reliable).map(calc).reduce((m, c) => c.lv > m.lv ? c : m);
const isStale = () => set === 'real' && EWS && EWS.stale;
function scene() {
  if (t < 0) {
    const o = OBS.reduce((m, x) => Math.abs(x.t - t) < Math.abs(m.t - t) ? x : m);
    return { past: true, o, lv: o.ts ? 92 : o.cb === 'CB' ? 24 : o.cb ? 17 : 6, st: o.ts ? { n: 'Badai petir', k: 'ba', i: 2 } : { n: 'Tanpa badai', k: 'ok', i: 0 } };
  }
  const c = t === 0 ? worst() : calc(HZ.reduce((m, h, j) => Math.abs(h.min - t) < Math.abs(HZ[m].min - t) ? j : m, 0));
  if (isStale()) return { past: false, o: LAST, c, lv: c.lv, st: { n: 'Data lama', k: 'off', i: 0 } };
  return { past: false, o: LAST, c, lv: c.lv, st: c };
}
let S = null;

/* ---------- medan model ---------- */
let RAWF = null, FD = {}, probe = { ...J }, M = {};
function loadFields() {
  const N = PETA.pts, uv = k => N.map(p => p.s.map((s, i) => s === null ? 0 : k ? -s * Math.cos(p.d[i] * rad) : -s * Math.sin(p.d[i] * rad))), f = k => N.map(p => p[k].map(v => v ?? 0));
  RAWF = { u: uv(0), v: uv(1), p: f('p'), c: f('c'), rh: f('rh'), t: f('t') };
}
const mapMin = k => (utcMs(PETA.t0) - T0) / 60000 + 60 * k;
function fields() {
  if (!RAWF) return;
  const f = clamp((t - mapMin(0)) / 60, 0, PETA.n - 1.001), i = Math.floor(f), k = f - i;
  for (const n in RAWF) FD[n] = RAWF[n].map(a => mix(a[i], a[i + 1], k));
}
function at(n, lon, lat) {
  const [W, E, So, No] = PETA.bbox, x = clamp((lon - W) / (E - W) * (PETA.nx - 1), 0, PETA.nx - 1.001), y = clamp((lat - So) / (No - So) * (PETA.ny - 1), 0, PETA.ny - 1.001);
  const i = x | 0, j = y | 0, a = x - i, b = y - j, f = FD[n], q = j * PETA.nx + i;
  return mix(mix(f[q], f[q + 1], a), mix(f[q + PETA.nx], f[q + PETA.nx + 1], a), b);
}

/* ---------- peta ---------- */
function proj() {
  const r = $('map').getBoundingClientRect(), [W, E, So, No] = GEOJ.bbox, cy = r.height * .5;
  let cx = r.width / 2;
  if (innerWidth > 1060) cx = (document.querySelector('.status').getBoundingClientRect().right + document.querySelector('.read').getBoundingClientRect().left) / 2 - r.left;
  const yk = 1 / Math.cos(J.lat * rad), k = Math.max(cx / (J.lon - W), (r.width - cx) / (E - J.lon), cy / ((No - J.lat) * yk), (r.height - cy) / ((J.lat - So) * yk));
  M = { w: r.width, h: r.height, k, X: l => cx + (l - J.lon) * k, Y: a => cy - (a - J.lat) * k * yk, lon: x => J.lon + (x - cx) / k, lat: y => J.lat + (cy - y) / (k * yk) };
}
const elev = (lon, lat) => { const [W, E, So, No] = GEOJ.bbox, e = GEOJ.elev; return e.v[clamp(Math.round((lat - So) / (No - So) * (e.ny - 1)), 0, e.ny - 1) * e.nx + clamp(Math.round((lon - W) / (E - W) * (e.nx - 1)), 0, e.nx - 1)]; };
const inLand = (lon, lat) => GEOJ.land.some(q => { let c = false; for (let i = 0, j = q.length - 1; i < q.length; j = i++) { const [a, b] = q[i], [d, e] = q[j]; if ((b > lat) !== (e > lat) && lon < (d - a) * (lat - b) / (e - b) + a) c = !c; } return c; });
const PLACES = [['Surabaya', -7.257, 112.752, 1], ['Sidoarjo', -7.447, 112.718], ['Gresik', -7.156, 112.655], ['Bangkalan', -7.045, 112.735], ['Pasuruan', -7.645, 112.907], ['Mojokerto', -7.472, 112.434], ['Lamongan', -7.117, 112.416], ['Sampang', -7.19, 113.24], ['Probolinggo', -7.754, 113.216], ['Jombang', -7.546, 112.233]];
const PEAKS = [['Penanggungan 1.653 m', -7.615, 112.62], ['Arjuno 3.339 m', -7.765, 112.589]];
function halo(x, s, a, b) { x.save(); x.setLineDash([]); x.lineJoin = 'round'; x.lineWidth = 3.5; x.strokeStyle = dark() ? 'rgba(27,46,60,.85)' : 'rgba(245,247,246,.85)'; x.strokeText(s, a, b); x.fillText(s, a, b); x.restore(); }
function drawBase() {
  proj();
  if (M.w < 50 || M.h < 50) { M = {}; return; }
  const [x, w, h] = fit($('base')), dk = dark(), EL = GEOJ.elev, [W, E, So, No] = GEOJ.bbox;
  x.fillStyle = dk ? '#1A3040' : '#D3E0E7'; x.fillRect(0, 0, w, h);
  const land = new Path2D();
  GEOJ.land.forEach(q => { q.forEach(([a, b], i) => i ? land.lineTo(M.X(a), M.Y(b)) : land.moveTo(M.X(a), M.Y(b))); land.closePath(); });
  const o = document.createElement('canvas'); o.width = EL.nx; o.height = EL.ny;
  const ox = o.getContext('2d'), im = ox.createImageData(EL.nx, EL.ny), lo = dk ? [43, 62, 72] : [228, 232, 224], hi = dk ? [92, 98, 102] : [199, 192, 180];
  for (let j = 0; j < EL.ny; j++) for (let i = 0; i < EL.nx; i++) {
    const g = (a, b) => EL.v[clamp(j + b, 0, EL.ny - 1) * EL.nx + clamp(i + a, 0, EL.nx - 1)], dx = (g(1, 0) - g(-1, 0)) / 2800, dy = (g(0, 1) - g(0, -1)) / 2800;
    const sh = clamp((dx * .6 - dy * .6 + .55) / Math.hypot(dx, dy, 1) / .55, .35, 1.45), k = clamp(g(0, 0) / 2600, 0, 1), q = ((EL.ny - 1 - j) * EL.nx + i) * 4;
    for (let c = 0; c < 3; c++) im.data[q + c] = clamp(mix(lo[c], hi[c], k) * mix(1, sh, .75), 0, 255);
    im.data[q + 3] = 255;
  }
  ox.putImageData(im, 0, 0);
  x.save(); x.clip(land); x.imageSmoothingQuality = 'high'; x.drawImage(o, M.X(W), M.Y(No), M.X(E) - M.X(W), M.Y(So) - M.Y(No));
  const corner = [[0, 0], [1, 0], [1, 1], [0, 1]];
  for (const [lev, a] of [[100, .3], [250, .3], [500, .38], [1000, .45], [1500, .38], [2000, .45], [2500, .38], [3000, .45]]) {
    x.strokeStyle = dk ? `rgba(220,228,232,${a * .6})` : `rgba(88,78,62,${a})`; x.lineWidth = lev % 1000 ? .6 : 1.1; x.beginPath();
    for (let j = 0; j < EL.ny - 1; j++) for (let i = 0; i < EL.nx - 1; i++) {
      const q = j * EL.nx + i, v = [EL.v[q], EL.v[q + 1], EL.v[q + EL.nx + 1], EL.v[q + EL.nx]];
      if (Math.max(...v) < lev || Math.min(...v) >= lev) continue;
      const pt = [];
      [[0, 1], [1, 2], [2, 3], [3, 0]].forEach(([c, d]) => {
        if ((v[c] >= lev) === (v[d] >= lev)) return;
        const f = (lev - v[c]) / (v[d] - v[c]), u = i + mix(corner[c][0], corner[d][0], f), s = j + mix(corner[c][1], corner[d][1], f);
        pt.push([M.X(W + u / (EL.nx - 1) * (E - W)), M.Y(So + s / (EL.ny - 1) * (No - So))]);
      });
      for (let n = 0; n + 1 < pt.length; n += 2) { x.moveTo(...pt[n]); x.lineTo(...pt[n + 1]); }
    }
    x.stroke();
  }
  x.restore(); x.strokeStyle = dk ? '#8FA6B3' : '#5C7079'; x.lineWidth = 1.1; x.stroke(land);
}
function drawTop() {
  if (!M.k) return;
  const [x, w, h] = fit($('top')), dk = dark(), ink = css('--ink'), mu = css('--mute'), wide = innerWidth > 1060, mr = $('map').getBoundingClientRect();
  const stt = S ? css(S.st.k === 'off' ? '--mute' : `--${S.st.k}-t`) : mu;
  x.clearRect(0, 0, w, h); x.strokeStyle = dk ? 'rgba(200,215,225,.22)' : 'rgba(50,75,90,.24)'; x.lineWidth = 1; x.font = F(13); x.fillStyle = mu;
  const lx = wide ? document.querySelector('.status').getBoundingClientRect().right - mr.left + 8 : 6, rx = wide ? document.querySelector('.read').getBoundingClientRect().left - mr.left : w;
  for (let g = 112; g < 114; g += .5) { const a = M.X(g); if (a < 0 || a > w) continue; x.beginPath(); x.moveTo(a, 0); x.lineTo(a, h); x.stroke(); x.textAlign = 'center'; if (a > lx + 20 && a < rx - 20) halo(x, num(g) + '° BT', a, h - 8); }
  for (let g = -8; g < -6.6; g += .5) { const a = M.Y(g); if (a < 0 || a > h) continue; x.beginPath(); x.moveTo(0, a); x.lineTo(w, a); x.stroke(); x.textAlign = 'left'; halo(x, num(-g) + '° LS', lx, a - 5); }
  const km = M.k / (111.32 * Math.cos(J.lat * rad)), jx = M.X(J.lon), jy = M.Y(J.lat);
  x.setLineDash([5, 4]); x.strokeStyle = stt; x.lineWidth = 1.4;
  [8, 16].forEach(r => { x.beginPath(); x.arc(jx, jy, r * km, 0, 7); x.stroke(); });
  x.setLineDash([]); x.fillStyle = stt; x.font = F(13); x.textAlign = 'left';
  if (16 * km > 60) [8, 16].forEach(r => halo(x, r + ' km', jx + r * km + 4, jy + 12));
  PLACES.forEach(([n, a, o, b]) => { const X = M.X(o), Y = M.Y(a); x.fillStyle = ink; x.fillRect(X - 2.5, Y - 2.5, 5, 5); x.font = FS(b ? 17 : 15, b ? 600 : 400); halo(x, n, X + 7, Y + 5); });
  x.fillStyle = mu; PEAKS.forEach(([n, a, o]) => { const X = M.X(o), Y = M.Y(a); x.beginPath(); x.moveTo(X, Y - 5); x.lineTo(X + 5, Y + 4); x.lineTo(X - 5, Y + 4); x.fill(); x.font = F(13.5); halo(x, n, X + 9, Y + 5); });
  x.font = FS(17, 400, 'italic'); x.fillStyle = dk ? '#9CC3D8' : '#3B6C86'; halo(x, 'Selat Madura', M.X(113.0), M.Y(-7.53)); halo(x, 'Laut Jawa', M.X(113.15), M.Y(-6.83));
  x.strokeStyle = stt; x.fillStyle = css('--panel'); x.lineWidth = 2.5; x.beginPath(); x.arc(jx, jy, 6, 0, 7); x.fill(); x.stroke(); x.beginPath(); x.moveTo(jx - 11, jy); x.lineTo(jx + 11, jy); x.moveTo(jx, jy - 11); x.lineTo(jx, jy + 11); x.stroke();
  x.fillStyle = ink; x.font = FS(17, 600); x.textAlign = 'left'; halo(x, 'Juanda', jx + 14, jy - 7);
  if (probe.lon !== J.lon || probe.lat !== J.lat) { const a = M.X(probe.lon), b = M.Y(probe.lat); x.strokeStyle = ink; x.lineWidth = 1.4; x.beginPath(); [[-14, 0, -4, 0], [4, 0, 14, 0], [0, -14, 0, -4], [0, 4, 0, 14]].forEach(([p, q, r, s]) => { x.moveTo(a + p, b + q); x.lineTo(a + r, b + s); }); x.stroke(); }
  const L = 20 * km, sx = rx - L - 18, sy = h - 34; x.fillStyle = ink; x.fillRect(sx, sy, L / 2, 5); x.strokeStyle = ink; x.lineWidth = 1; x.strokeRect(sx, sy, L, 5); x.font = F(13); x.textAlign = 'left'; halo(x, '0', sx - 3, sy - 5); x.textAlign = 'right'; halo(x, '20 km', sx + L + 3, sy - 5);
  if (wst === 'barb' && FD.u) {
    const [W, E, So, No] = PETA.bbox, st = innerWidth < 680 ? 2 : 1; x.strokeStyle = x.fillStyle = ink;
    for (let j = 0; j < PETA.ny; j += st) for (let i = 0; i < PETA.nx; i += st) { const q = j * PETA.nx + i, u = FD.u[q], v = FD.v[q]; barb(x, M.X(W + i / (PETA.nx - 1) * (E - W)), M.Y(So + j / (PETA.ny - 1) * (No - So)), Math.hypot(u, v), Math.atan2(-u, -v) / rad, 25); }
  }
}
const LAYS = { p: { n: 'Hujan, mm per jam', b: [.1, .5, 1, 2, 5, 10], c: ['#CFE2F0', '#9FC7E2', '#63A3CF', '#3079B4', '#15538F', '#0A2F63'] }, c: { n: 'CAPE, J per kg', b: [500, 1000, 1500, 2000, 2500, 3000], c: ['#E9DDEE', '#D3B8DF', '#B68FCB', '#9264B1', '#6C3E92', '#46206B'] }, rh: { n: 'Kelembapan relatif, %', b: [60, 70, 80, 85, 90, 95], c: ['#DDEBDD', '#B7D8BE', '#86BF9B', '#4F9F7A', '#25795C', '#0D5240'] } };
function drawField() {
  const [x, w, h] = fit($('field'), 1); x.clearRect(0, 0, w, h);
  const D = LAYS[layer]; if (!D || !FD.u) return;
  const n = Math.round(w / 2), m = Math.round(h / 2); if (n < 2 || m < 2) return;
  const o = document.createElement('canvas'); o.width = n; o.height = m;
  const ox = o.getContext('2d'), im = ox.createImageData(n, m), C = D.c.map(c => [1, 3, 5].map(i => parseInt(c.substr(i, 2), 16))), lons = Array.from({ length: n }, (_, i) => M.lon(i * 2 + 1));
  for (let j = 0; j < m; j++) { const la = M.lat(j * 2 + 1); for (let i = 0; i < n; i++) { const v = at(layer, lons[i], la); let k = -1; for (let b = 0; b < D.b.length; b++) if (v >= D.b[b]) k = b; if (k >= 0) im.data.set([...C[k], 178], (j * n + i) * 4); } }
  ox.putImageData(im, 0, 0); x.drawImage(o, 0, 0, w, h);
}
function legend() {
  const D = LAYS[layer];
  $('legend').innerHTML = !PETA && D ? '<div class="mode">Medan ECMWF tidak tersedia saat ini.</div>' : D ? `<div class="leg">${D.c.map(c => `<i style="background:${c}"></i>`).join('')}</div><div class="legt">${D.b.map(b => `<span>${num(b)}</span>`).join('')}</div><div class="mode">${D.n}</div>` : '<div class="mode">Kontur 100 sampai 3.000 m.</div>';
}
function drawProbe() {
  const { lon, lat } = probe, home = lon === J.lon && lat === J.lat, sea = !inLand(lon, lat);
  $('prh').textContent = home ? 'ECMWF di Juanda' : 'ECMWF di titik baca'; $('home').style.visibility = home ? 'hidden' : '';
  const rows = [['Posisi', `${fix(-lat)}° LS ${fix(lon)}° BT`], ['Elevasi', home ? '3 m' : sea ? 'laut' : num(elev(lon, lat), 0) + ' m']];
  if (FD.u) {
    const u = at('u', lon, lat), v = at('v', lon, lat), d = Math.round((Math.atan2(-u, -v) / rad + 360) % 360);
    rows.push(['Angin', `${String(d).padStart(3, '0')}°, ${num(Math.hypot(u, v))} kt`], ['Suhu', num(at('t', lon, lat)) + ' °C'], ['Kelembapan', Math.round(at('rh', lon, lat)) + ' %'], ['Hujan', num(at('p', lon, lat)) + ' mm/jam'], ['CAPE', num(Math.round(at('c', lon, lat) / 10) * 10, 0) + ' J/kg']);
  }
  $('probe').innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join('');
}
const PT = Array.from({ length: innerWidth < 680 ? 420 : 1100 }, () => ({ a: Math.random() * 90 }));
function drawFlow(dt) {
  const c = $('flow');
  if (wst !== 'flow' || calm || !RAWF || !FD.u) { if (c.width > 1) { const [x, w, h] = fit(c, 1.5); x.clearRect(0, 0, w, h); } return; }
  const [x, w, h] = fit(c, 1.5);
  x.globalCompositeOperation = 'destination-out'; x.fillStyle = `rgba(0,0,0,${1 - Math.exp(-dt * 2.1)})`; x.fillRect(0, 0, w, h); x.globalCompositeOperation = 'source-over';
  x.strokeStyle = dark() ? 'rgba(235,243,247,.75)' : 'rgba(22,45,60,.62)'; x.lineWidth = 1.15; x.beginPath();
  for (const p of PT) {
    if (p.x === undefined || p.a-- < 0 || p.x < 0 || p.x > w || p.y < 0 || p.y > h) { p.x = Math.random() * w; p.y = Math.random() * h; p.a = 40 + Math.random() * 80; }
    const lo = M.lon(p.x), la = M.lat(p.y), nx = p.x + at('u', lo, la) * dt * 6.5, ny = p.y - at('v', lo, la) * dt * 6.5;
    x.moveTo(p.x, p.y); x.lineTo(nx, ny); p.x = nx; p.y = ny;
  }
  x.stroke();
}

/* ---------- panel ---------- */
const kv = k => k === 'off' ? 'var(--mute)' : `var(--${k})`, kt = k => k === 'off' ? 'var(--mute)' : `var(--${k}-t)`;
const ACT = ['Belum ada tindakan khusus. Tunggu pembaruan berikutnya.', 'Perhatikan perkembangan awan dan informasi BMKG.', 'Potensi badai petir atau angin kencang. Ikuti peringatan resmi BMKG.'];
function render() {
  if (!EWS || !OBS.length || !GEOJ) return;
  S = scene(); fields();
  const k = S.st.k; document.body.style.setProperty('--st', kv(k)); document.body.style.setProperty('--stt', kt(k)); $('alarmbar').classList.remove('off');
  freshness();
  if ($('p-pantauan').hidden) return;
  $('clock').textContent = `${wibDate(at0(t))}, ${hm(t)} WIB`; $('t').value = t;
  $('t').setAttribute('aria-valuetext', `${hm(t)} WIB, ${t < 0 ? 'teramati' : t === 0 ? 'sekarang' : 'prakiraan'}`);
  if (!S.past) {
    const c = S.c, h = HZ[c.hi], reach = HZ.filter(q => q.reliable).at(-1);
    $('mode').textContent = t === 0 ? `Risiko tertinggi sampai ${reach.name}` : `Prakiraan ${h.name}`;
    $('say').innerHTML = isStale() ? `<b>Data lama</b>, ${num(EWS.data_age_minutes, 0)} menit` : `<b>${c.n}</b> pukul ${hm(h.min)}`;
    $('sum').innerHTML = `Indeks ${Math.round(c.lv)} dari 100.<table style="margin-top:6px"><tr><td>Peluang badai</td><td><b>${pc(c.p)}</b></td></tr><tr><td>Waspada, bahaya</td><td>${pc(h.w)}, ${pc(h.b)}</td></tr><tr><td>Masukan</td><td>${h.model.includes('ECMWF') ? 'METAR, ECMWF' : 'METAR saja'}</td></tr>${set === 'real' && h.lstm !== null ? `<tr><td>LSTM, pembanding</td><td>${pc(h.lstm)}</td></tr>` : ''}</table>`;
    const far = HZ.map((q, i) => i).filter(i => !HZ[i].reliable).map(calc).filter(q => q.i > 0);
    $('act').textContent = isStale() ? 'METAR belum diperbarui. Peringatan ini tidak bisa dipakai sampai data masuk kembali.'
      : t > 0 && !h.reliable ? 'Pada jarak ini kemampuan model setara pola jam dan bulan, jadi baca sebagai kecenderungan musiman.'
        : ACT[c.i] + (t === 0 && far.length ? ` Kecenderungan musiman: ${far.map(q => `${q.n.toLowerCase()} ${HZ[q.hi].name}`).join(', ')}.` : '');
  } else {
    const o = S.o; $('mode').textContent = 'Teramati di stasiun Juanda'; $('say').innerHTML = `<b>${S.st.n}</b> pukul ${hm(o.t)}`;
    $('sum').innerHTML = `<table><tr><td>Suhu, titik embun</td><td>${num(o.temp)}, ${num(o.dew)} °C</td></tr><tr><td>QNH</td><td>${o.q ?? '–'} hPa</td></tr><tr><td>Angin</td><td>${o.raw ? (o.dir === null ? 'berubah' : String(o.dir).padStart(3, '0') + '°') + ', ' : ''}${num(o.spd)} kt</td></tr><tr><td>Awan konvektif</td><td>${!o.raw ? 'tidak tercatat' : o.cb === 'CB' ? 'cumulonimbus' : o.cb ? 'cumulus menjulang' : 'tidak ada'}</td></tr></table>`;
    $('act').textContent = 'Ini hasil pengamatan, bukan prakiraan.';
  }
  $('ball').setAttribute('aria-label', `Bola alarm: ${$('say').textContent}`); target = S.lv;
  minis(); tokens(); legend(); drawTop(); drawField(); drawProbe(); drawRail(); drawMg(); perf();
  for (const [a, v] of [['k', set], ['l', layer], ['w', wst]]) document.querySelectorAll(`[data-${a}]`).forEach(b => b.setAttribute('aria-pressed', b.dataset[a] === v));
}
function freshness() {
  if (!EWS) return;
  const age = Math.round((Date.now() - T0) / 60000);
  $('fresh').textContent = `METAR ${wibDate(T0)} ${wibClock(T0)} WIB, ${age < 90 ? age + ' menit' : num(age / 60, 0) + ' jam'} lalu${EWS.ecmwf_available ? '' : '. ECMWF tidak tersedia, memakai model METAR saja'}`;
}
function minis() {
  const b = (l, k, i, r) => `<svg viewBox="0 0 40 40" aria-hidden="true"><defs><clipPath id="m${i}"><circle cx="20" cy="20" r="16.5"/></clipPath></defs><rect clip-path="url(#m${i})" y="${36.5 - l * .33}" width="40" height="40" fill="${kv(k)}" ${r ? '' : 'opacity=".55"'}/><circle cx="20" cy="20" r="17" fill="none" stroke="${kv(k)}" stroke-width="2" ${r ? '' : 'stroke-dasharray="4 3"'}/></svg>`;
  const w = worst(), cur = t > 0 ? S.c.hi : -1, st = isStale();
  $('minis').innerHTML = `<button data-t="0" aria-pressed="${t === 0}">${b(w.lv, st ? 'off' : w.k, 9, true)}<span>Sekarang<small>tertinggi</small></span></button>` + HZ.map((h, i) => { const c = calc(i), k = st ? 'off' : c.k; return `<button data-t="${h.min}" aria-pressed="${i === cur}">${b(c.lv, k, i, h.reliable)}<span>${h.name}<small style="color:${kt(k)}">${h.reliable ? c.n : 'musiman'}, ${hm(h.min)}</small></span></button>`; }).join('');
}
function perf() {
  const ev = INFO && INFO.evaluation.horizons;
  if (!ev) { $('perf').innerHTML = '<tr><td>Hasil uji tidak dapat dimuat.</td></tr>'; return; }
  const hi = S.past ? -1 : S.c.hi;
  $('perf').innerHTML = '<thead><tr><th>Horizon</th><th>POD</th><th>FAR</th><th>CSI</th></tr></thead>' + HZ.map((h, i) => { const q = ev[h.key].utama.waspada; return `<tr${i === hi ? ' style="font-weight:600"' : ''}><td>${h.name}${h.reliable ? '' : ', musiman'}</td><td>${fix(q.POD)}</td><td>${fix(q.FAR)}</td><td>${fix(q.CSI)}</td></tr>`; }).join('');
}
function meaning(k) {
  let m;
  if (/^(METAR|SPECI)$/.test(k)) return k === 'METAR' ? 'Laporan cuaca rutin.' : 'Laporan cuaca khusus.';
  if (k === 'WARR') return 'Stasiun Juanda.'; if (k === 'COR') return 'Laporan yang dikoreksi.';
  if (m = k.match(/^(\d\d)(\d\d)(\d\d)Z$/)) return `Diamati tanggal ${+m[1]} pukul ${m[2]}.${m[3]} UTC, atau ${hm(S.o.t)} WIB.`;
  if (m = k.match(/^(\d{3}|VRB)(\d\d)(?:G(\d\d))?KT$/)) return `${m[1] === 'VRB' ? 'Arah angin berubah-ubah' : `Angin dari ${+m[1]} derajat`}, ${+m[2]} knot (${num(m[2] * .514444)} m/s)${m[3] ? `, hembusan ${+m[3]} knot` : ''}.`;
  if (m = k.match(/^(\d{3})V(\d{3})$/)) return `Arah angin berubah antara ${+m[1]} dan ${+m[2]} derajat.`;
  if (/^\d{4}$/.test(k)) return k === '9999' ? 'Jarak pandang 10 km atau lebih.' : `Jarak pandang ${num(+k / 1000)} km.`;
  if (m = k.match(/^(FEW|SCT|BKN|OVC)(\d{3})(CB|TCU)?$/)) return `Awan ${{ FEW: '1 sampai 2 oktas', SCT: '3 sampai 4 oktas', BKN: '5 sampai 7 oktas', OVC: '8 oktas' }[m[1]]}, dasar sekitar ${num(Math.round(m[2] * 30.48 / 10) * 10, 0)} m${m[3] === 'CB' ? ', jenis cumulonimbus' : m[3] ? ', jenis cumulus menjulang' : ''}.`;
  if (m = k.match(/^(M?\d\d)\/(M?\d\d)$/)) return `Suhu ${+m[1].replace('M', '-')} °C dan titik embun ${+m[2].replace('M', '-')} °C.`;
  if (m = k.match(/^Q(\d{4})$/)) return `Tekanan QNH ${+m[1]} hPa.`;
  if (m = k.match(/^(TL|FM|AT)(\d\d)(\d\d)$/)) return `${{ TL: 'Sampai', FM: 'Mulai', AT: 'Pada' }[m[1]]} pukul ${m[2]}.${m[3]} UTC.`;
  const W = { NOSIG: 'Tidak ada perubahan berarti dalam dua jam.', TEMPO: 'Perubahan sementara yang diperkirakan.', BECMG: 'Perubahan bertahap yang diperkirakan.', NSC: 'Tidak ada awan berarti.', CAVOK: 'Jarak pandang dan awan baik.', RMK: 'Keterangan tambahan.' };
  if (W[k]) return W[k];
  if (m = k.match(/^(RE)?([+-]|VC)?((?:TS|SH|RA|DZ|HZ|BR|FG|FU|SQ|FC)+)$/)) {
    const N = { TS: 'badai petir', SH: 'hujan lokal', RA: 'hujan', DZ: 'gerimis', HZ: 'udara kabur', BR: 'kabut tipis', FG: 'kabut', FU: 'asap', SQ: 'squall', FC: 'puting beliung' };
    const s = m[3].match(/../g).map(q => N[q]).join(' dengan ');
    const say = (m[1] ? 'Baru berakhir: ' : '') + (m[2] === '+' ? 'intensitas lebat, ' : m[2] === '-' ? 'intensitas ringan, ' : m[2] === 'VC' ? 'di sekitar stasiun, ' : '') + s + '.';
    return say[0].toUpperCase() + say.slice(1);
  }
  return 'Kode ini belum punya penjelasan di halaman ini.';
}
function tokens() {
  if (!S.o.raw) { $('toks').innerHTML = ''; $('mean').textContent = 'METAR mentah untuk jam ini tidak tersimpan.'; return; }
  const ks = S.o.raw.replace(/=$/, '').split(/\s+/); if (tok === null || tok >= ks.length) tok = Math.min(3, ks.length - 1);
  $('toks').innerHTML = ks.map((k, i) => `<button aria-pressed="${i === tok}">${esc(k)}</button>`).join(''); $('mean').textContent = (t > 0 ? 'Laporan terakhir. ' : '') + meaning(ks[tok]);
}

/* ---------- bola alarm ---------- */
const bc = $('ball'), N = 72, ys = new Float32Array(N), vs = new Float32Array(N);
let level = 0, target = 0, lvV = 0, drag = null, tilt = 0, tiltV = 0, ph = 0, first = true;
const bgeo = (w, h) => { const R = Math.min(h / 2 - 8, (w - 116) / 2); return { R, cx: Math.max(R + 12, w / 2 - 28), cy: h / 2 }; };
function drawBall(dt, now) {
  const [g, w, h] = fit(bc), { R, cx, cy } = bgeo(w, h), key = S.st.k;
  if (R < 20) return;
  if (calm || first) { level = target; first = false; } else { lvV += ((target - level) * 26 - lvV * 7) * dt; level += lvV * dt; }
  tiltV += (-tilt * 42 - tiltV * 2.6) * dt; tilt += tiltV * dt; ph += dt * (key === 'wa' ? 2.8 : key === 'ba' ? 4.4 : 1.3);
  for (let i = 0; i < N; i++) vs[i] += (-ys[i] * 60 - vs[i] * 2.2) * dt;
  for (let r = 0; r < 3; r++) for (let i = 0; i < N; i++) { const l = ys[i - 1] ?? ys[i], q = ys[i + 1] ?? ys[i]; vs[i] += ((l + q) / 2 - ys[i]) * 300 * dt / 3; }
  for (let i = 0; i < N; i++) ys[i] = clamp(ys[i] + vs[i] * dt, -R * .4, R * .4);
  g.clearRect(0, 0, w, h);
  const c = key === 'off' ? css('--mute') : css('--' + key), ink = css('--ink'), mu = css('--mute'), yy = p => cy + R - p / 100 * 2 * R, base = yy(clamp(level, 2, 97)), amp = calm || key === 'off' ? 0 : key === 'ok' ? 1.6 : key === 'wa' ? 3 : 5;
  const surf = i => base + ys[i] + Math.sin(i * .17 + ph) * amp + Math.sin(i * .09 - ph * .7) * amp * .6 + tilt * (i / N - .5) * R * .9, water = new Path2D();
  water.moveTo(cx - R, cy + R); for (let i = 0; i < N; i++) water.lineTo(cx - R + i / (N - 1) * 2 * R, surf(i)); water.lineTo(cx + R, cy + R); water.closePath();
  const disc = new Path2D(); disc.arc(cx, cy, R, 0, 7);
  g.save(); g.clip(disc); g.fillStyle = css('--paper'); g.fillRect(0, 0, w, h); g.fillStyle = c; g.fill(water); g.clip(water); g.strokeStyle = 'rgba(255,255,255,.16)'; g.lineWidth = 1;
  for (let p = 0; p < 100; p += 5) { g.beginPath(); g.moveTo(cx - R, yy(p)); g.lineTo(cx + R, yy(p)); g.stroke(); } g.restore();
  g.strokeStyle = c; g.lineWidth = key === 'ba' && !calm ? 3.5 + 1.5 * Math.sin(now / 170) : 2.5; g.stroke(disc);
  const sx = cx + R + 12; g.strokeStyle = g.fillStyle = mu; g.lineWidth = 1; g.font = F(13); g.textBaseline = 'middle'; g.textAlign = 'left'; g.beginPath(); g.moveTo(sx, yy(0)); g.lineTo(sx, yy(100));
  for (let p = 0; p <= 100; p += 5) { g.moveTo(sx, yy(p)); g.lineTo(sx + (p % 25 ? 4 : 8), yy(p)); } g.stroke();
  [0, 25, 50, 100].forEach(p => g.fillText(p, sx + 12, yy(p)));
  if (!S.past) [[55, 'waspada', 'wa'], [85, 'bahaya', 'ba']].forEach(([p, n, k]) => { g.strokeStyle = g.fillStyle = css(`--${k}-t`); g.lineWidth = 2; g.beginPath(); g.moveTo(sx - 10, yy(p)); g.lineTo(sx + 10, yy(p)); g.stroke(); g.font = F(13.5, 600); g.fillText(n, sx + 13, yy(p) + (p === 55 ? -7 : 0)); });
  const main = S.st.n, sub = hm(S.past ? S.o.t : HZ[S.c.hi].min) + ' WIB', lab = col => { g.fillStyle = col; g.textAlign = 'center'; g.textBaseline = 'alphabetic'; g.font = FS(Math.round(R * (main.length > 8 ? .24 : .3))); g.fillText(main, cx, cy + R * .1); g.font = F(Math.round(R * .15)); g.fillText(sub, cx, cy + R * .32); };
  lab(ink); g.save(); g.clip(disc); g.clip(water); g.fillStyle = c; g.fillRect(cx - R, cy - R * .2, R * 2, R * .6); lab(key === 'wa' ? '#201600' : key === 'ok' && dark() ? '#04202E' : key === 'off' && dark() ? '#14232F' : '#fff'); g.restore();
}
const bpos = e => { const r = bc.getBoundingClientRect(), { R, cx } = bgeo(r.width, r.height); return clamp(Math.round((e.clientX - r.left - cx + R) / (2 * R) * (N - 1)), 0, N - 1); };
const poke = (i, f) => { for (let j = -5; j <= 5; j++) if (vs[i + j] !== undefined) vs[i + j] += f * Math.cos(j / 6 * 1.57); };
bc.addEventListener('pointerdown', e => { if (calm) return; bc.setPointerCapture(e.pointerId); drag = { x: e.clientX }; poke(bpos(e), 220); });
bc.addEventListener('pointermove', e => { if (calm) return; if (drag) { tiltV += (e.clientX - drag.x) * .03; drag.x = e.clientX; } else poke(bpos(e), clamp(e.movementY, -20, 20) * 1.2); });
['pointerup', 'pointercancel'].forEach(n => bc.addEventListener(n, () => { drag = null; }));

/* ---------- garis waktu ---------- */
const pos = m => (m + 900) / 2340;
const hourOf = m => { const w = wib(at0(m)); return w.h + w.mi / 60; };
const night = m => { const h = hourOf(m); return h < 5.05 || h > 17.4; };
function hours(step, f) { const w = wib(T0), off = -(w.mi); for (let m = off - 900; m <= 1440; m += 60) if (m >= -900) { const h = wib(at0(m)).h; if (h % step === 0) f(m, h); } }
function drawRail() {
  const [x, w] = fit($('rail')); x.clearRect(0, 0, w, 52); x.fillStyle = css('--hair'); x.fillRect(0, 13, w, 9); x.fillStyle = css('--line');
  for (let m = -900; m < 1440; m += 15) if (night(m)) x.fillRect(pos(m) * w, 13, w / 156 + .6, 9);
  x.fillStyle = css('--ba'); OBS.filter(o => o.ts).forEach(o => x.fillRect(pos(o.t) * w - 5, 9, 10, 17));
  HZ.forEach((h, i) => { x.fillStyle = isStale() ? css('--mute') : css('--' + calc(i).k); x.fillRect(pos(h.min) * w - 3, 9, 6, 17); });
  x.fillStyle = css('--ink'); x.fillRect(pos(0) * w - 1, 5, 2, 25); x.font = F(13); x.fillStyle = css('--mute'); x.textBaseline = 'top';
  hours(w < 560 ? 6 : 3, (m, h) => { const p = pos(m) * w; x.textAlign = p < 18 ? 'left' : p > w - 18 ? 'right' : 'center'; x.fillText(two(h) + '.00', p, 34); });
}

/* ---------- meteogram ---------- */
let MG = {};
function drawMg() {
  const [x, w, h] = fit($('mg')); x.clearRect(0, 0, w, h);
  const L = 46, Rr = 46, px = m => L + pos(m) * (w - L - Rr), ink = css('--ink'), mu = css('--mute'), hair = css('--hair'), c1 = css('--c1'), c2 = css('--c2'), c3 = css('--c3');
  MG = { L, Rr, w };
  const rows = [['Suhu dan titik embun, °C', .23], ['QNH, hPa', .15], ['Angin, kt', .13], ['Hujan, mm/jam; CAPE, J/kg', .2], ['Indeks risiko', .29]], gap = 24, H = h - 24 - gap * rows.length; let y = 0;
  const box = rows.map(([n, f]) => { const a = y + gap - 4, b = a + H * f; y = b + 4; return { n, a, b }; });
  x.fillStyle = hair; box.forEach(b => { for (let m = -900; m < 1440; m += 15) if (night(m)) x.fillRect(px(m), b.a, (w - L - Rr) / 156 + .6, b.b - b.a); });
  x.globalAlpha = .2; x.fillStyle = css('--ba'); OBS.filter(o => o.ts).forEach(o => box.slice(0, 3).forEach(b => x.fillRect(px(o.t - 15), b.a, px(o.t + 15) - px(o.t - 15), b.b - b.a))); x.globalAlpha = 1;
  x.textBaseline = 'alphabetic'; box.forEach((b, i) => { x.strokeStyle = css('--line'); x.lineWidth = 1; x.strokeRect(L, b.a, w - L - Rr, b.b - b.a); x.font = F(14, 600); x.fillStyle = ink; x.textAlign = 'left'; x.fillText(`(${'abcde'[i]}) ${b.n}`, L, b.a - 6); });
  const ax = (b, lo, hi, tk, col = mu, right) => { const f = v => v === null ? null : b.b - (v - lo) / (hi - lo) * (b.b - b.a); x.font = F(13); x.textBaseline = 'middle'; tk.forEach(v => { x.fillStyle = col; x.textAlign = right ? 'left' : 'right'; x.fillText(v > 900 && v < 1100 ? v : v.toLocaleString('id-ID'), right ? w - Rr + 6 : L - 6, f(v)); if (!right) { x.strokeStyle = css('--line'); x.globalAlpha = .5; x.beginPath(); x.moveTo(L, f(v)); x.lineTo(w - Rr, f(v)); x.stroke(); x.globalAlpha = 1; } }); return f; };
  const inR = p => p[0] >= -900 && p[0] <= 1440, site = PETA && PETA.site, mod = k => site ? site[k].map((v, i) => [mapMin(i), v]).filter(inR) : [];
  const temps = [...OBS.flatMap(o => [o.temp, o.dew]), ...mod('t').map(p => p[1])].filter(v => v !== null), tl = Math.floor(Math.min(...temps) - 1), th = Math.ceil(Math.max(...temps) + 1), five = (lo, hi, s) => { const a = []; for (let v = Math.ceil(lo / s) * s; v <= hi; v += s) a.push(v); return a; };
  let f = ax(box[0], tl, th, five(tl, th, th - tl > 12 ? 5 : 2));
  polyline(x, mod('t').map(([m, v]) => [px(m), f(v)]), c2, [5, 4], 1.4); polyline(x, OBS.map(o => [px(o.t), f(o.temp)]), c2); polyline(x, OBS.map(o => [px(o.t), f(o.dew)]), c1);
  const qs = OBS.map(o => o.q).filter(Boolean), ql = Math.min(...qs) - 1, qh = Math.max(...qs) + 1; f = ax(box[1], ql, qh, five(ql + 1, qh - 1, Math.max(1, Math.round((qh - ql) / 3)))); polyline(x, OBS.map(o => [px(o.t), f(o.q)]), ink);
  { const b = box[2], my = (b.a + b.b) / 2 + 8, every = w < 600 ? 6 : 3; x.strokeStyle = x.fillStyle = ink; OBS.forEach((o, i) => { if ((OBS.length - 1 - i) % every || !o.raw) return; if (o.dir === null) { x.beginPath(); x.arc(px(o.t), my, 3.5, 0, 7); x.stroke(); } else barb(x, px(o.t), my, o.spd, o.dir, 24); }); x.strokeStyle = x.fillStyle = mu; if (site) site.s.forEach((s, i) => { const m = mapMin(i); if (m > 60 && m < 1430 && i % (w < 600 ? 4 : 2) === 0 && s !== null) barb(x, px(m), my, s, site.d[i], 24); }); }
  const rmax = Math.max(3, ...mod('p').map(p => p[1] || 0)), r3 = Math.ceil(rmax), cmax = Math.max(4000, ...mod('c').map(p => p[1] || 0)); f = ax(box[3], 0, r3, r3 > 6 ? five(0, r3, Math.ceil(r3 / 3)) : five(0, r3 - 1, 1), c1);
  if (site) { const bw = (w - L - Rr) / 39 - 3; x.fillStyle = c1; mod('p').forEach(([m, v]) => x.fillRect(px(m) - bw / 2, f(v || 0), bw, box[3].b - f(v || 0))); const fc = ax(box[3], 0, cmax, [0, 2000, 4000], c3, true); polyline(x, mod('c').map(([m, v]) => [px(m), fc(v)]), c3); } else { x.fillStyle = mu; x.textAlign = 'center'; x.fillText('Medan ECMWF tidak tersedia', (L + w - Rr) / 2, (box[3].a + box[3].b) / 2); }
  f = ax(box[4], 0, 100, [0, 50, 100]); [[55, 'wa', 'waspada'], [85, 'ba', 'bahaya']].forEach(([v, k, n]) => { polyline(x, [[L, f(v)], [w - Rr, f(v)]], css(`--${k}-t`), [2, 3], 1.4); x.fillStyle = css(`--${k}-t`); x.font = F(13.5, 600); x.textAlign = 'left'; x.textBaseline = 'bottom'; x.fillText(n, L + 6, f(v) - 2); });
  polyline(x, HZ.map((q, i) => [px(q.min), f(calc(i).lv)]), mu, [1, 3], 1.4);
  HZ.forEach((q, i) => { const c = calc(i), X = px(q.min), Y = f(c.lv), col = isStale() ? mu : css('--' + c.k); x.beginPath(); x.arc(X, Y, 6, 0, 7); x.fillStyle = q.reliable ? col : css('--panel'); x.fill(); x.strokeStyle = col; x.lineWidth = 2.5; x.stroke(); });
  x.textBaseline = 'middle'; x.font = F(13.5);
  if (w > 600) { const lg = (X, Y, col, s, kind) => { if (kind === 'o' || kind === 'h') { x.beginPath(); x.arc(X, Y, 5, 0, 7); x.fillStyle = kind === 'o' ? col : css('--panel'); x.fill(); x.strokeStyle = col; x.lineWidth = 2; x.stroke(); } else polyline(x, [[X - 9, Y], [X + 9, Y]], col, kind === 'd' ? [5, 4] : [], 2); x.fillStyle = ink; x.textAlign = 'left'; x.fillText(s, X + 14, Y); }; const Y = box[0].a - 10; lg(w - Rr - 352, Y, c2, 'suhu teramati'); lg(w - Rr - 222, Y, c2, 'suhu ECMWF', 'd'); lg(w - Rr - 104, Y, c1, 'titik embun'); lg(w - Rr - 230, box[4].a - 10, mu, 'prakiraan', 'o'); lg(w - Rr - 120, box[4].a - 10, mu, 'musiman', 'h'); }
  x.font = F(13); x.fillStyle = mu; x.textAlign = 'center'; x.textBaseline = 'top'; hours(w < 600 ? 6 : 3, (m, hh) => { const p = px(m); if (p > L + 10 && p < w - Rr - 5) x.fillText(two(hh), p, h - 17); }); x.textAlign = 'right'; x.fillText('WIB', w - 4, h - 17);
  polyline(x, [[px(0), box[0].a], [px(0), box[4].b]], mu, [3, 3], 1); polyline(x, [[px(t), box[0].a], [px(t), box[4].b]], S.st.k === 'off' ? mu : css(`--${S.st.k}-t`), [], 2);
  const o = t <= 0 ? OBS.reduce((m, q) => Math.abs(q.t - t) < Math.abs(m.t - t) ? q : m) : null, k = site ? clamp(Math.round((t - mapMin(0)) / 60), 0, PETA.n - 1) : 0;
  $('cap').textContent = `${hm(t)} WIB. ` + (o ? `Teramati: suhu ${num(o.temp)} °C, titik embun ${num(o.dew)} °C, QNH ${o.q ?? '–'} hPa. ` : '') + (site ? `ECMWF: suhu ${num(site.t[k])} °C, hujan ${num(site.p[k])} mm/jam, CAPE ${num(site.c[k], 0)} J/kg. ` : '') + 'Garis penuh dan barb gelap adalah pengamatan METAR; garis putus, batang, CAPE, dan barb abu-abu adalah ECMWF. Arsiran merah menandai badai petir teramati, arsiran abu-abu menandai malam.';
}

/* ---------- kendali ---------- */
function go(m) { t = clamp(Math.round(m / 30) * 30, -900, 1440); render(); }
function stop() { clearInterval(playing); playing = 0; $('play').textContent = 'Putar'; }
$('t').addEventListener('input', e => { stop(); go(+e.target.value); });
$('now').onclick = () => { stop(); go(0); };
$('play').onclick = () => { if (playing) return stop(); if (t >= 1440) go(-900); $('play').textContent = 'Jeda'; playing = setInterval(() => t >= 1440 ? stop() : go(t + 30), calm ? 900 : 420); };
$('minis').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; stop(); const i = [...b.parentNode.children].indexOf(b); go(+b.dataset.t); $('minis').children[i].focus(); });
$('toks').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; tok = [...b.parentNode.children].indexOf(b); tokens(); $('toks').children[tok].focus(); });
document.querySelectorAll('[data-l]').forEach(b => { b.onclick = () => { layer = b.dataset.l; render(); }; });
document.querySelectorAll('[data-w]').forEach(b => { b.onclick = () => { wst = b.dataset.w; render(); }; });
let md = false, pd = false;
const mgGo = e => { const r = $('mg').getBoundingClientRect(); stop(); go((e.clientX - r.left - MG.L) / (MG.w - MG.L - MG.Rr) * 2340 - 900); };
$('mg').addEventListener('pointerdown', e => { if (!S) return; md = true; $('mg').setPointerCapture(e.pointerId); mgGo(e); });
$('mg').addEventListener('pointermove', e => md && mgGo(e));
['pointerup', 'pointercancel'].forEach(n => $('mg').addEventListener(n, () => { md = false; }));
const setProbe = (lon, lat) => { const [W, E, So, No] = GEOJ.bbox; probe = { lon: clamp(lon, W, E), lat: clamp(lat, So, No) }; drawProbe(); drawTop(); };
const mapAt = e => { const r = $('map').getBoundingClientRect(); setProbe(M.lon(e.clientX - r.left), M.lat(e.clientY - r.top)); };
$('map').addEventListener('pointerdown', e => { if (!M.k) return; pd = true; mapAt(e); });
$('map').addEventListener('pointermove', e => pd && mapAt(e));
['pointerup', 'pointercancel', 'pointerleave'].forEach(n => $('map').addEventListener(n, () => { pd = false; }));
$('map').addEventListener('keydown', e => { const d = { ArrowLeft: [-.05, 0], ArrowRight: [.05, 0], ArrowUp: [0, .05], ArrowDown: [0, -.05] }[e.key]; if (d && M.k) { e.preventDefault(); setProbe(probe.lon + d[0], probe.lat + d[1]); } });
$('home').onclick = () => { setProbe(J.lon, J.lat); $('map').focus(); };

/* ---------- alarm ---------- */
let ac;
function beep() { if (!alarmOn || !ac) return; [880, 660, 880].forEach((f, i) => { const o = ac.createOscillator(), v = ac.createGain(), s = ac.currentTime + i * .28; o.frequency.value = f; v.gain.setValueAtTime(0, s); v.gain.linearRampToValueAtTime(.18, s + .03); v.gain.linearRampToValueAtTime(0, s + .24); o.connect(v).connect(ac.destination); o.start(s); o.stop(s + .26); }); }
function raise() {
  const w = worst(), lvl = isStale() ? 0 : w.i;
  $('live').textContent = isStale() ? 'Data METAR terlambat' : `${w.n} pukul ${hm(HZ[w.hi].min)} WIB`;
  if (lastLevel !== null && lvl > lastLevel) {
    if (!calm) for (let i = 0; i < N; i += 6) poke(i, -160 + Math.random() * 320);
    if (lvl === 2) { beep(); if (alarmOn && 'Notification' in window && Notification.permission === 'granted') { try { new Notification('Bahaya cuaca di Juanda', { body: `${set === 'real' ? '' : 'Simulasi. '}Perkiraan pukul ${hm(HZ[w.hi].min)} WIB. ${ACT[2]}` }); } catch (e) { /* beberapa HP hanya mengizinkan lewat service worker */ } } }
  }
  lastLevel = lvl;
}
$('snd').onclick = () => {
  alarmOn = !alarmOn;
  if (alarmOn) { ac = ac || new (window.AudioContext || window.webkitAudioContext)(); ac.resume(); if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission(); }
  $('snd').setAttribute('aria-pressed', alarmOn); $('snd').textContent = alarmOn ? 'Alarm hidup' : 'Alarm mati';
};
document.querySelectorAll('[data-k]').forEach(b => { b.onclick = () => { if (!EWS) return; set = b.dataset.k; stop(); go(0); raise(); scrollTo({ top: 0 }); }; });

/* ---------- memuat dan memperbarui ---------- */
function problem(text) { $('err').hidden = !text; $('err').textContent = text || ''; }
async function loadObservations() {
  const [hist, ews] = await Promise.all([api('/api/metar/history?count=32'), api('/api/ews/status')]);
  const rows = hist.data.map(decode).filter(o => !Number.isNaN(o.time) && o.temp !== null && o.temp !== undefined).sort((a, b) => a.time - b.time);
  if (!rows.length) throw new Error('Riwayat METAR kosong');
  const next = utcMs(ews.observed_at_utc), moved = T0 !== null && next !== T0;
  EWS = ews; T0 = next; OBS = rows.map(o => ({ ...o, t: (o.time - T0) / 60000 })).filter(o => o.t >= -900 && o.t <= 0);
  if (!OBS.length) throw new Error('Riwayat METAR tidak sesuai dengan waktu peringatan');
  LAST = OBS.at(-1); buildHorizons();
  if (moved && !playing && t !== 0) t = clamp(t, -900, 1440);
}
async function loadMap() { try { PETA = await api('/api/peta'); loadFields(); } catch (e) { if (!PETA) RAWF = null; } }
async function refresh() {
  try { await loadObservations(); problem(''); render(); if (set === 'real') raise(); }
  catch (e) { problem(`Peringatan belum dapat dimuat. ${e.message}. Halaman mencoba lagi tiap menit.`); if (!EWS) { $('mode').textContent = 'Peringatan tidak tersedia'; $('say').textContent = 'Belum ada data'; } }
}
PAGES.pantauan = {
  draw() { if (GEOJ && !$('p-pantauan').hidden) { drawBase(); if (S) render(); else { drawTop(); legend(); drawProbe(); } } },
  async start() {
    const geo = api('/assets/geo.json').then(g => { GEOJ = g; this.draw(); }).catch(() => problem('Peta tidak dapat dimuat.'));
    const info = api('/api/ews/model').then(i => { INFO = i; }).catch(() => {});
    await Promise.all([geo, info, loadMap()]);
    await refresh();
    setInterval(async () => { if (document.hidden) return; try { await api('/api/metar/latest'); } catch (e) { /* dilaporkan oleh refresh */ } refresh(); }, 60000);
    setInterval(() => { if (!document.hidden) loadMap().then(render); }, 1800000);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  },
};
let prev = 0;
function frame(now) {
  requestAnimationFrame(frame);
  if (document.hidden || $('p-pantauan').hidden) return;
  const dt = Math.min(.05, (now - prev) / 1000); if (dt < .014) return; prev = now;
  if (!M.k || $('hero').getBoundingClientRect().bottom < 0) return;
  drawFlow(dt); if (S) drawBall(dt, now);
}
requestAnimationFrame(frame);
