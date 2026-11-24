/* Alat bersama semua halaman. */
const $ = id => document.getElementById(id);
const calm = matchMedia('(prefers-reduced-motion:reduce)').matches;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const mix = (a, b, k) => a + (b - a) * k;
const css = v => getComputedStyle(document.body).getPropertyValue(v).trim();
const dark = () => document.body.classList.contains('dark');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const num = (v, d = 1) => v === null || v === undefined || Number.isNaN(+v) ? '–' : (+v).toLocaleString('id-ID', { maximumFractionDigits: d });
const fix = (v, d = 2) => v === null || v === undefined ? '–' : (+v).toFixed(d).replace('.', ',');
const pc = (v, d = 1) => num(v * 100, d) + '%';
const F = (n, w = 400) => `${w} ${n}px "Source Sans 3",system-ui`;
const FS = (n, w = 600, i = '') => `${i} ${w} ${n}px "STIX Two Text",Georgia`;
const rad = Math.PI / 180;
const DAYS = ['Min', 'Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des'];
const two = n => String(n).padStart(2, '0');
/* Waktu WIB dari milidetik UTC, tanpa bergantung pada zona waktu perangkat. */
function wib(ms) {
  const d = new Date(ms + 7 * 3600e3);
  return { y: d.getUTCFullYear(), mo: d.getUTCMonth(), d: d.getUTCDate(), wd: d.getUTCDay(), h: d.getUTCHours(), mi: d.getUTCMinutes() };
}
const wibClock = ms => { const w = wib(ms); return `${two(w.h)}.${two(w.mi)}`; };
const wibDate = ms => { const w = wib(ms); return `${DAYS[w.wd]} ${w.d} ${MONTHS[w.mo]}`; };
const utcMs = s => {
  const m = String(s).match(/(\d{4})-(\d\d)-(\d\d)[ T](\d{1,2}):(\d\d)/);
  return m ? Date.UTC(+m[1], m[2] - 1, +m[3], +m[4], +m[5]) : NaN;
};
function fit(c, max = 2) {
  const r = c.getBoundingClientRect(), d = Math.min(max, devicePixelRatio || 1), w = Math.round(r.width * d), h = Math.round(r.height * d);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const x = c.getContext('2d');
  x.setTransform(d, 0, 0, d, 0, 0);
  return [x, r.width, r.height];
}
async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (e) { /* bukan JSON */ }
  if (!res.ok) throw new Error(String((body && body.detail) || `Server menjawab ${res.status}`).replace(/\.\s*$/, ''));
  if (body === null) throw new Error('Jawaban server tidak dapat dibaca');
  return body;
}
/* Endpoint tulis memerlukan token admin; disimpan selama tab terbuka. */
async function adminPost(path) {
  const token = sessionStorage.getItem('adminToken') || prompt('Token admin:') || '';
  try {
    const out = await api(path, { method: 'POST', headers: { 'X-Admin-Token': token } });
    sessionStorage.setItem('adminToken', token);
    return out;
  } catch (e) {
    sessionStorage.removeItem('adminToken');
    throw e;
  }
}
/* Barb angin konvensi belahan bumi selatan (bulu di sisi kanan batang). */
function barb(x, X, Y, s, d, L) {
  x.lineWidth = 1.3;
  if (s < 2.5) { x.beginPath(); x.arc(X, Y, 3.5, 0, 7); x.stroke(); return; }
  x.save(); x.translate(X, Y); x.rotate(d * rad); x.beginPath(); x.moveTo(0, 0); x.lineTo(0, -L);
  let n = Math.round(s / 5) * 5, y = -L;
  while (n >= 50) { x.moveTo(0, y); x.lineTo(-9, y + 3); x.lineTo(0, y + 6); n -= 50; y += 8; }
  while (n >= 10) { x.moveTo(0, y); x.lineTo(-9, y - 4); n -= 10; y += 4.5; }
  if (n >= 5) { if (y === -L) y += 4.5; x.moveTo(0, y); x.lineTo(-5, y - 2.2); }
  x.stroke(); x.beginPath(); x.arc(0, 0, 1.8, 0, 7); x.fill(); x.restore();
}
function polyline(x, pts, col, dash = [], lw = 2) {
  x.strokeStyle = col; x.lineWidth = lw; x.setLineDash(dash); x.lineJoin = 'round'; x.beginPath();
  let pen = false;
  pts.forEach(([a, b]) => { if (b === null || Number.isNaN(b)) { pen = false; return; } pen ? x.lineTo(a, b) : x.moveTo(a, b); pen = true; });
  x.stroke(); x.setLineDash([]);
}
/* Halaman mendaftarkan gambar ulang untuk perubahan tema dan ukuran. */
const PAGES = {};
