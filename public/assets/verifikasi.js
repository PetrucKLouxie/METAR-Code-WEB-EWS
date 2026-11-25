/* Halaman verifikasi: hasil uji model dan snapshot prospek. */
(() => {
  let info = null, snaps = null, sort = 'waspada';
  const KEYS = ['1h', '3h', '9h', '18h', '24h'], nm = k => `+${parseInt(k, 10)} jam`;
  const SER = [['klimatologi', 'Pola jam dan bulan', '--mute', 'line'], ['paralel_lama', 'Situs lama', '--c3', 'tri'], ['cadangan', 'METAR saja', '--c2', 'sq'], ['utama', 'XGBoost, ECMWF', '--c1', 'ring'], ['berurutan', 'LSTM, XGBoost, ECMWF', '--c1', 'dot']];
  function build() {
    const ev = info.evaluation, H = ev.horizons;
    $('p-verifikasi').innerHTML = `<h2>Verifikasi</h2>
      <p class="lead">Seberapa baik peringatan ini pada data yang tidak dipakai untuk melatihnya.</p>
      <section class="grid2"><div><h3>Ringkasan</h3>
        <p>Sampai +9 jam, peringatan ini (LSTM meramal METAR, XGBoost mengklasifikasi bersama ECMWF) lebih baik dari sekadar mengetahui jam dan bulan. Pada +18 dan +24 jam tidak ada perbedaan yang nyata, sehingga horizon itu tidak membunyikan alarm.</p>
        <p>Alarm palsu masih banyak: pada ambang waspada, ${num(H['1h'][info.primary].waspada.FAR * 100, 0)}% peringatan +1 jam dan ${num(H['3h'][info.primary].waspada.FAR * 100, 0)}% peringatan +3 jam tidak diikuti kejadian. Kejadiannya sendiri jarang, sekitar ${num(H['1h'].events / ev.test_samples * 100)}% dari semua slot.</p></div>
        <div><h3>Data uji</h3><table class="data txt"><tr><td>Periode</td><td>${esc(ev.test_period)}</td></tr><tr><td>Jumlah slot</td><td>${num(ev.test_samples, 0)}</td></tr><tr><td>Kejadian</td><td>${esc(ev.target)}</td></tr><tr><td>Versi model</td><td><code>${esc(info.version)}</code></td></tr></table></div></section>
      <section><h3>Kemampuan menurut jarak waktu</h3><figure><canvas class="plot" id="vc" style="height:340px" role="img" aria-label="PR-AUC tiap model menurut horizon; angkanya ada pada tabel di bawah"></canvas><figcaption>PR-AUC, makin tinggi makin baik. Garis putus adalah pola jam dan bulan, yaitu batas yang harus dilampaui model agar berguna.</figcaption></figure>
        <div class="scroll"><table class="data"><thead><tr><th>PR-AUC</th>${KEYS.map(k => `<th>${nm(k)}</th>`).join('')}</tr></thead>${SER.map(([k, n]) => `<tr><td>${n}</td>${KEYS.map(q => `<td>${fix(H[q].prauc[k], 3)}</td>`).join('')}</tr>`).join('')}<tr><td>Keunggulan atas pola jam dan bulan, selang 95%</td>${KEYS.map(q => { const c = H[q].gain_over_climatology; return `<td class="${H[q].better_than_climatology ? 't-ok' : 't-off'}">${c[0] > 0 ? '+' : ''}${fix(c[0], 3)}<br>${fix(c[1], 3)} sampai ${fix(c[2], 3)}</td>`; }).join('')}</tr><tr><td>Selisih terhadap XGBoost tanpa LSTM, selang 95%</td>${KEYS.map(q => { const c = H[q].gain_over_direct; return `<td class="t-off">${c[0] > 0 ? '+' : ''}${fix(c[0], 3)}<br>${fix(c[1], 3)} sampai ${fix(c[2], 3)}</td>`; }).join('')}</tr></table></div>
        <p class="note">Susunan dengan LSTM tidak berbeda nyata dari XGBoost yang memakai METAR langsung. Keduanya terangkat oleh riwayat badai dan ECMWF.</p></section>
      <section><h3>Ketelitian ramalan LSTM</h3><div class="scroll"><table class="data"><thead><tr><th>Galat mutlak rata-rata</th>${KEYS.map(k => `<th>${nm(k)}</th>`).join('')}</tr></thead>${[['temperature', 'Suhu, °C'], ['dew_point', 'Titik embun, °C'], ['pressure', 'QNH, hPa'], ['wind_speed', 'Angin, kt']].map(([f, n]) => [['LSTM', 0], ['dianggap tetap', 1], ['jam yang sama kemarin', 2]].map(([l, j]) => `<tr><td>${j ? '&#8195;' + l : `<b>${n}</b>, LSTM`}</td>${KEYS.map(k => { const v = ev.lstm_mae[k][f], best = Math.min(...v); return `<td${v[j] === best ? ' style="font-weight:600"' : ''}>${fix(v[j])}</td>`; }).join('')}</tr>`).join('')).join('')}</table></div><p class="note">Angka tebal adalah yang terkecil. Pada +24 jam LSTM hampir sama dengan nilai jam yang sama kemarin.</p></section>
      <section><div class="row"><h3 style="margin:0">Ketepatan peringatan</h3><div class="lay" style="margin:0;min-width:220px" id="vl"><button data-s="waspada">Ambang waspada</button><button data-s="bahaya">Ambang bahaya</button></div></div><div class="scroll"><table class="data" id="vt" style="margin-top:8px"></table></div>
        <p class="note">Ambang dan kalibrasi dipilih dari prediksi silang data latih, bukan dari data uji. Selang 95% dari bootstrap per hari, karena slot yang berdekatan tidak saling bebas.</p></section>
      <section><h3>Batas yang perlu diketahui</h3><ul><li>Kelompok fitur dipilih setelah melihat hasil uji 2026. Angka di halaman ini bersifat eksploratif sampai dikonfirmasi pada data baru.</li><li>Pada +1 jam, sebagian besar kejadian adalah badai yang sedang berlanjut, bukan awal badai.</li><li>Label kejadian berasal dari METAR satu stasiun, tanpa data petir independen.</li><li>Data latih ECMWF baru sekitar dua tahun, dan data uji belum memuat Oktober sampai Desember 2026.</li><li>Prediksi yang berjalan dicatat ke tab EWS_Log agar dapat diuji pada kejadian yang belum terjadi saat dicatat.</li></ul></section>
      <section><h3>Snapshot prospek 7 hari</h3><div id="vs"><p>Memuat snapshot.</p></div></section>`;
    $('vl').addEventListener('click', e => { const b = e.target.closest('button'); if (b) { sort = b.dataset.s; table(); } });
    table(); draw(); loadSnaps();
  }
  function table() {
    document.querySelectorAll('#vl button').forEach(b => b.setAttribute('aria-pressed', b.dataset.s === sort));
    const H = info.evaluation.horizons;
    $('vt').innerHTML = '<thead><tr><th>Horizon</th><th>Ambang</th><th>POD</th><th>FAR</th><th>CSI</th><th>HSS</th><th>Tepat</th><th>Palsu</th><th>Terlewat</th></tr></thead>' + KEYS.map(k => { const q = H[k][info.primary][sort]; return `<tr><td>${nm(k)}</td><td>${pc(q.ambang)}</td><td>${fix(q.POD)}</td><td>${fix(q.FAR)}</td><td>${fix(q.CSI)}</td><td>${fix(q.HSS)}</td><td>${num(q.TP, 0)}</td><td>${num(q.FP, 0)}</td><td>${num(q.FN, 0)}</td></tr>`; }).join('');
  }
  function draw() {
    const c = $('vc'); if (!c || !info) return;
    const [x, w, h] = fit(c); x.clearRect(0, 0, w, h);
    const H = info.evaluation.horizons, L = 46, R = 16, T = w < 620 ? 84 : 34, B = h - 28, top = .5, f = v => B - v / top * (B - T), px = i => L + (i + .5) / 5 * (w - L - R);
    x.font = F(13); x.textBaseline = 'middle';
    for (let v = 0; v <= top; v += .1) { x.fillStyle = css('--mute'); x.textAlign = 'right'; x.fillText(fix(v, 1), L - 8, f(v)); polyline(x, [[L, f(v)], [w - R, f(v)]], css('--hair'), [], 1); }
    SER.forEach(([k, n, col, kind], s) => {
      const pts = KEYS.map((q, i) => [px(i) + (kind === 'line' ? 0 : (s - 2) * 12), f(H[q].prauc[k])]), ink = css(col);
      if (kind === 'line') polyline(x, [[L, pts[0][1]], ...pts, [w - R, pts[4][1]]], ink, [5, 4], 2);
      x.fillStyle = ink;
      pts.forEach(([X, Y]) => { x.beginPath(); if (kind === 'dot' || kind === 'ring') x.arc(X, Y, 6, 0, 7); else if (kind === 'sq') x.rect(X - 5, Y - 5, 10, 10); else if (kind === 'tri') { x.moveTo(X, Y - 7); x.lineTo(X + 6.5, Y + 5); x.lineTo(X - 6.5, Y + 5); } if (kind === 'ring') { x.strokeStyle = ink; x.lineWidth = 2; x.stroke(); } else x.fill(); });
      const per = w < 620 ? 2 : 5, lx = L + (s % per) * ((w - L - R) / per), ly = 12 + Math.floor(s / per) * 22; x.beginPath(); if (kind === 'line') polyline(x, [[lx, ly], [lx + 18, ly]], ink, [5, 4], 2); else { if (kind === 'dot' || kind === 'ring') x.arc(lx + 9, ly, 5.5, 0, 7); else if (kind === 'sq') x.rect(lx + 4, ly - 5, 10, 10); else { x.moveTo(lx + 9, ly - 6); x.lineTo(lx + 15, ly + 5); x.lineTo(lx + 3, ly + 5); } if (kind === 'ring') { x.strokeStyle = ink; x.lineWidth = 2; x.stroke(); } else x.fill(); }
      x.font = F(14); x.fillStyle = css('--ink'); x.textAlign = 'left'; x.fillText(n, lx + 24, ly);
    });
    x.font = F(14); x.fillStyle = css('--mute'); x.textAlign = 'center'; KEYS.forEach((k, i) => x.fillText(nm(k), px(i), h - 12));
  }
  const RES = { HIT: ['Tepat', 't-ok'], 'HIT (OVER-WARNING)': ['Tepat, terlalu tinggi', 't-wa'], 'CORRECT NEGATIVE': ['Benar aman', 't-ok'], 'FALSE ALARM': ['Alarm palsu', 't-wa'], MISS: ['Terlewat', 't-ba'], PENDING: ['Menunggu', 't-off'], UNVERIFIABLE: ['Data kurang', 't-off'] };
  async function loadSnaps() {
    try { snaps = await api('/api/forecast-7days/snapshots'); }
    catch (e) { $('vs').innerHTML = `<div class="msg err">Snapshot tidak dapat dimuat. ${esc(e.message)}.</div>`; return; }
    const m = snaps.metrics, rows = snaps.records;
    $('vs').innerHTML = `<p>${num(snaps.total_records, 0)} prakiraan tersimpan: ${num(m.hit, 0)} tepat, ${num(m.false_alarm, 0)} alarm palsu, ${num(m.miss, 0)} terlewat, ${num(m.correct_negative, 0)} benar aman, ${num(m.pending, 0)} menunggu, ${num(m.unverifiable || 0, 0)} tanpa data cukup.</p>
      ${snaps.total_verified < 30 ? '<p class="note">Jumlahnya masih terlalu sedikit untuk menyimpulkan ketepatan.</p>' : ''}
      <div class="row" style="margin:10px 0"><button class="btn" id="vsave">Simpan prakiraan hari ini</button><button class="btn" id="vver">Verifikasi sekarang</button><span id="vmsg" role="status"></span></div>
      <div class="scroll"><table class="data txt"><thead><tr><th>Dibuat</th><th>Untuk tanggal</th><th>Jarak</th><th>Prakiraan</th><th>Skor</th><th>Teramati</th><th>Hasil</th></tr></thead>${rows.slice(0, 60).map(r => { const [n, c] = RES[r.verification_result] || [r.verification_result, '']; return `<tr><td>${esc(r.snapshot_time_wib)}</td><td>${esc(r.target_date)}</td><td>${esc(r.lead_time_days)} hari</td><td>${esc(r.predicted_risk_label)}</td><td>${esc(String(r.predicted_max_prob).replace('%', '').replace('.', ','))}</td><td>${esc(r.actual_metar_observed)}</td><td class="${c}">${esc(n)}</td></tr>`; }).join('') || '<tr><td colspan="7">Belum ada snapshot.</td></tr>'}</table></div>
      <p class="note">Hasil yang diverifikasi sebelum 25 November 2026 memakai aturan lama, yang mencampur tanggal UTC dan WIB.</p>`;
    const run = (id, path, done) => { $(id).onclick = async () => { $('vmsg').textContent = 'Memproses.'; try { const r = await adminPost(path); await loadSnaps(); $('vmsg').textContent = done(r); } catch (e) { $('vmsg').textContent = `Gagal. ${e.message}`; } }; };
    run('vsave', '/api/forecast-7days/snapshot', r => `${num(r.saved_count, 0)} prakiraan disimpan.`);
    run('vver', '/api/forecast-7days/verify', r => `${num(r.verified_count, 0)} prakiraan diverifikasi.`);
  }
  PAGES.verifikasi = {
    draw,
    async start() {
      $('p-verifikasi').innerHTML = '<h2>Verifikasi</h2><p class="lead">Memuat hasil uji.</p>';
      try { info = await api('/api/ews/model'); build(); }
      catch (e) { $('p-verifikasi').innerHTML = `<h2>Verifikasi</h2><div class="msg err">Hasil uji tidak dapat dimuat. ${esc(e.message)}.</div><button class="btn" id="vretry">Coba lagi</button>`; $('vretry').onclick = () => this.start(); }
    },
  };
})();
