/* Navigasi, tema, dan ukuran layar. */
(() => {
  const names = ['pantauan', 'prospek', 'model', 'verifikasi', 'referensi'], started = new Set();
  let current = null;
  function show() {
    const wanted = location.hash.slice(1), name = names.includes(wanted) ? wanted : 'pantauan';
    names.forEach(n => { $('p-' + n).hidden = n !== name; });
    document.querySelectorAll('#nav a').forEach(a => { if (a.hash === '#' + name) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    if (current !== null && current !== name) scrollTo(0, 0);
    current = name;
    if (!started.has(name)) { started.add(name); PAGES[name].start(); } else PAGES[name].draw();
  }
  let theme = null;
  try { theme = localStorage.getItem('tema'); } catch (e) { /* penyimpanan dimatikan */ }
  function setTheme(d) { document.body.classList.toggle('dark', d); $('theme').setAttribute('aria-pressed', d); }
  setTheme(theme === 'gelap');
  $('theme').onclick = () => { const d = !dark(); setTheme(d); try { localStorage.setItem('tema', d ? 'gelap' : 'terang'); } catch (e) { /* abaikan */ } PAGES[current].draw(); };
  let timer, size = '';
  new ResizeObserver(() => {
    const now = `${innerWidth}|${document.body.clientWidth}|${$('hero').offsetHeight}`;
    if (now === size) return;
    size = now; clearTimeout(timer); timer = setTimeout(() => current && PAGES[current].draw(), 90);
  }).observe(document.body);
  addEventListener('hashchange', show);
  document.fonts && document.fonts.ready.then(() => current && PAGES[current].draw());
  // Alarm berjalan di semua halaman, jadi Pantauan selalu dimulai.
  started.add('pantauan'); PAGES.pantauan.start();
  show();
})();
