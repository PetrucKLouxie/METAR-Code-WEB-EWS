/* Halaman prospek 7 hari. */
(() => {
  let data = null, day = 0;
  const K = ['ok', 'wa', 'ba'], LABEL = { 'SIAGA BADAI': 'Siaga badai', WASPADA: 'Waspada', AMAN: 'Aman' };
  const name = l => LABEL[l] || l;
  const shortDay = d => { const [y, m, q] = d.split('-').map(Number); return `${DAYS[new Date(Date.UTC(y, m - 1, q)).getUTCDay()]} ${q} ${MONTHS[m - 1]}`; };
  function build() {
    const D = data.daily_summaries;
    $('p-prospek').innerHTML = `<h2>Prospek 7 hari</h2>
      <p class="lead">Prakiraan ECMWF IFS per jam untuk titik Juanda, diberi skor risiko 0 sampai 100 dengan aturan ambang.</p>
      ${data.stale ? '<div class="msg err">Open-Meteo tidak dapat dihubungi, jadi ini prakiraan terakhir yang tersimpan (paling lama 24 jam).</div>' : ''}
      <div class="msg"><b>Cara membaca.</b> Skor ini bukan peluang dan belum dikalibrasi terhadap pengamatan Juanda. Pada musim hujan CAPE tinggi hampir setiap hari, sehingga status siaga muncul terlalu sering. Pakai halaman ini untuk membandingkan hari dan jam, bukan sebagai alarm.</div>
      <section><div class="days" id="pd">${D.map((d, i) => `<button data-i="${i}"><b>${shortDay(d.date)}</b><span class="t-${K[d.risk_level]}">${name(d.risk_label)}, ${num(d.max_risk_prob, 0)}</span><br>${num(d.temp_min, 0)} sampai ${num(d.temp_max, 0)} °C<br>${num(d.precip_sum_mm)} mm<br>CAPE ${num(d.cape_max_jkg, 0)}</button>`).join('')}</div></section>
      <section><h3 id="pt"></h3><p id="pr"></p><figure><canvas class="plot" id="pc" style="height:440px" role="img"></canvas><figcaption>Semua nilai adalah prakiraan ECMWF IFS per jam dalam WIB. Garis putus pada panel (a) adalah batas skor waspada 25 dan siaga 65 yang dipakai aturan per jam; CAPE, hujan, dan kode cuaca tertentu juga bisa menaikkan status.</figcaption></figure></section>
      <section><h3>Tabel per jam</h3><div class="scroll"><table class="data" id="pb"></table></div></section>
      <p class="note">Dibuat ${esc(data.generated_at_wib)}. Sumber: ${esc(data.source)}.</p>`;
    $('pd').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { day = +b.dataset.i; select(); } });
    select();
  }
  function select() {
    const d = data.daily_summaries[day];
    document.querySelectorAll('#pd button').forEach((b, i) => b.setAttribute('aria-pressed', i === day));
    $('pt').textContent = `${shortDay(d.date)}${d.is_today ? ', hari ini' : ''}`;
    const top = d.hourly_detail.reduce((m, h) => h.calibrated_prob > m.calibrated_prob ? h : m);
    $('pr').textContent = `Skor tertinggi ${num(top.calibrated_prob, 0)} pada pukul ${top.hour.replace(':', '.')} WIB. Hujan ${num(d.precip_sum_mm)} mm, CAPE tertinggi ${num(d.cape_max_jkg, 0)} J/kg, hembusan tertinggi ${num(d.wind_gust_max_kt)} kt.`;
    $('pb').innerHTML = '<thead><tr><th>Jam</th><th>Cuaca</th><th>Suhu, embun</th><th>RH</th><th>Angin</th><th>Hembusan</th><th>Hujan</th><th>CAPE</th><th>Skor</th><th>Status</th></tr></thead>' + d.hourly_detail.map(h => `<tr><td>${h.hour.replace(':', '.')}</td><td>${esc(h.weather_desc)}</td><td>${num(h.temperature)}, ${num(h.dew_point)} °C</td><td>${num(h.relative_humidity, 0)}%</td><td>${String(h.wind_direction).padStart(3, '0')}°, ${num(h.wind_speed_kt)} kt</td><td>${num(h.wind_gusts_kt)} kt</td><td>${num(h.precipitation_mm, 2)} mm</td><td>${num(h.cape_jkg, 0)}</td><td>${num(h.calibrated_prob, 0)}</td><td class="t-${K[h.risk_level]}">${name(h.risk_label)}</td></tr>`).join('');
    draw();
  }
  function draw() {
    const c = $('pc'); if (!c || !data) return;
    const [x, w, h] = fit(c); x.clearRect(0, 0, w, h);
    const all = data.daily_summaries.flatMap(d => d.hourly_detail), n = all.length, L = 50, R = 14, px = i => L + i / n * (w - L - R), ink = css('--ink'), mu = css('--mute');
    const P = [['Skor risiko', 'calibrated_prob', 100, '--ink', [0, 50, 100]], ['CAPE, J/kg', 'cape_jkg', Math.max(4000, ...all.map(q => q.cape_jkg)), '--c3', [0, 2000, 4000]], ['Hujan, mm/jam', 'precipitation_mm', Math.max(3, ...all.map(q => q.precipitation_mm)), '--c1', null], ['Hembusan, kt', 'wind_gusts_kt', Math.max(25, ...all.map(q => q.wind_gusts_kt)), '--c2', [0, 10, 20]]];
    const H = (h - 24) / P.length;
    P.forEach(([title, key, max, col, ticks], k) => {
      const a = k * H + 22, b = (k + 1) * H - 6, f = v => b - v / max * (b - a);
      x.fillStyle = css('--hair'); x.fillRect(px(day * 24), a, px(24) - px(0), b - a);
      x.strokeStyle = css('--line'); x.lineWidth = 1; x.strokeRect(L, a, w - L - R, b - a);
      x.font = F(14, 600); x.fillStyle = ink; x.textAlign = 'left'; x.textBaseline = 'alphabetic'; x.fillText(`(${'abcd'[k]}) ${title}`, L, a - 6);
      x.font = F(13); x.textAlign = 'right'; x.textBaseline = 'middle'; x.fillStyle = mu; (ticks || [0, Math.round(max / 2 * 10) / 10]).forEach(v => x.fillText(v.toLocaleString('id-ID'), L - 6, f(v)));
      if (k === 0) [[25, 'wa'], [65, 'ba']].forEach(([v, q]) => polyline(x, [[L, f(v)], [w - R, f(v)]], css(`--${q}-t`), [2, 3], 1.3));
      if (k === 2) { x.fillStyle = css(col); all.forEach((q, i) => x.fillRect(px(i) + .5, f(q[key]), Math.max(1, px(1) - px(0) - 1), b - f(q[key]))); } else polyline(x, all.map((q, i) => [px(i + .5), f(q[key])]), css(col));
    });
    x.font = F(13.5); x.fillStyle = mu; x.textAlign = 'center'; x.textBaseline = 'top'; data.daily_summaries.forEach((d, i) => x.fillText(w < 620 ? shortDay(d.date).split(' ')[0] : shortDay(d.date), px(i * 24 + 12), h - 18));
    c.setAttribute('aria-label', 'Skor risiko, CAPE, hujan, dan hembusan selama 7 hari');
    c.onpointerdown = e => { const r = c.getBoundingClientRect(); day = clamp(Math.floor((e.clientX - r.left - L) / (w - L - R) * 7), 0, data.daily_summaries.length - 1); select(); };
  }
  PAGES.prospek = {
    draw,
    async start() {
      $('p-prospek').innerHTML = '<h2>Prospek 7 hari</h2><p class="lead">Mengambil prakiraan ECMWF IFS.</p>';
      try { data = await api('/api/forecast-7days'); day = 0; build(); }
      catch (e) { $('p-prospek').innerHTML = `<h2>Prospek 7 hari</h2><div class="msg err">Prakiraan tidak dapat dimuat. ${esc(e.message)}.</div><button class="btn" id="pretry">Coba lagi</button>`; $('pretry').onclick = () => this.start(); }
    },
  };
})();
