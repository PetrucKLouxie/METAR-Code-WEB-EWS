/* Halaman model: cara kerja peringatan. */
(() => {
  const KEYS = ['1h', '3h', '9h', '18h', '24h'], nm = k => `+${parseInt(k, 10)} jam`;
  function flow() {
    const box = (x, y, w, t, s) => `<rect x="${x}" y="${y}" width="${w}" height="62" fill="var(--panel)" stroke="var(--line)"/><text x="${x + w / 2}" y="${y + 27}" text-anchor="middle" font-size="16" font-weight="600" fill="var(--ink)">${t}</text><text x="${x + w / 2}" y="${y + 47}" text-anchor="middle" font-size="13.5" fill="var(--mute)">${s}</text>`;
    const arrow = (a, b, c, d) => `<line x1="${a}" y1="${b}" x2="${c}" y2="${d}" stroke="var(--mute)" stroke-width="1.5" marker-end="url(#ah)"/>`;
    return `<div class="scroll"><svg viewBox="0 0 980 250" style="min-width:760px;width:100%;display:block" role="img" aria-label="Alur: METAR diramal LSTM, lalu bersama riwayat badai dan ECMWF masuk ke XGBoost, dibandingkan dengan ambang, dan menjadi status bola alarm"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--mute)"/></marker></defs>
      ${box(0, 10, 170, 'METAR Juanda', '12 jam terakhir')}${box(0, 94, 170, 'Riwayat badai', '48 jam terakhir')}${box(0, 178, 170, 'ECMWF IFS', 'run yang sudah terbit')}${box(236, 10, 200, 'LSTM', 'meramal 24 jam METAR')}
      ${box(236, 94, 200, 'Fitur', '35 masukan di waktu target')}${box(510, 94, 190, 'XGBoost', 'satu model tiap horizon')}${box(770, 94, 210, 'Ambang dan bola alarm', 'aman, waspada, bahaya')}
      ${arrow(170, 41, 234, 41)}${arrow(336, 72, 336, 92)}${arrow(170, 125, 234, 125)}${arrow(170, 209, 234, 146)}${arrow(436, 125, 508, 125)}${arrow(700, 125, 768, 125)}</svg></div>`;
  }
  function build(info) {
    const H = info.horizons, ev = info.evaluation;
    $('p-model').innerHTML = `<h2>Model</h2>
      <p class="lead">Peringatan 1 sampai 24 jam dihitung dalam dua langkah: LSTM meramal METAR Juanda, lalu XGBoost mengklasifikasi ramalan itu bersama musim, riwayat badai, dan prakiraan ECMWF IFS.</p>
      <section>${flow()}</section>
      <section class="grid2"><div><h3>Yang diprediksi</h3><p>${esc(ev.target)}. Dalam praktiknya hampir semua kejadian adalah badai petir; hujan biasa tidak termasuk.</p><p>"+3 jam" berarti sekitar tiga jam lagi, bukan kapan saja dalam tiga jam ke depan. Hasilnya berlaku untuk satu titik, yaitu stasiun Juanda.</p></div>
        <div><h3>Bila ada data yang hilang</h3><p>Bila riwayat METAR terputus lebih dari 2 jam, LSTM dilewati dan XGBoost memakai METAR langsung. Bila ECMWF tidak tersedia, dipakai model METAR saja. Keduanya ditulis di bagian bawah halaman. Data ECMWF tersimpan dipakai paling lama 6 jam.</p><p>Bila METAR berhenti masuk lebih dari 75 menit, bola menjadi abu-abu dan alarm tidak berlaku.</p></div></section>
      <section><h3>Masukan</h3><div class="scroll"><table class="data txt"><thead><tr><th>Kelompok</th><th>Isi</th></tr></thead>
        <tr><td>METAR, 14 fitur</td><td>Nilai <b>ramalan LSTM</b> di waktu target: suhu, titik embun, selisihnya, QNH, kecepatan angin, kelembapan relatif, defisit uap air, kerapatan udara, energi angin, perubahan QNH 3 jam, perubahan suhu dan angin 1 jam, serta jam UTC dalam sinus dan kosinus.</td></tr>
        <tr><td>Musim, 4 fitur</td><td>Hari dalam tahun dan jam pada waktu target.</td></tr>
        <tr><td>Riwayat badai, 6 fitur</td><td>Badai saat ini, jumlah slot badai dalam 1, 3, 6, dan 24 jam, serta waktu sejak badai terakhir.</td></tr>
        <tr><td>ECMWF IFS, 11 fitur</td><td>Hujan, hujan konvektif, suhu, titik embun, kelembapan, tutupan awan, tekanan, komponen angin, dan hujan tertinggi dalam jendela 7 jam pada waktu target.</td></tr></table></div></section>
      <section><h3>Hanya memakai prakiraan yang sudah terbit</h3><p>ECMWF dihitung pada 00, 06, 12, dan 18 UTC, tetapi baru tersedia sekitar 6,5 sampai 7,6 jam kemudian. Pada saat peringatan dibuat, run terbaru berumur 6,5 sampai 13,6 jam. Karena itu dipakai prakiraan yang dibuat jauh sebelumnya, baik saat melatih maupun saat berjalan.</p>
        <div class="scroll"><table class="data"><thead><tr><th>Horizon</th><th>Jarak minimal yang dibutuhkan</th><th>Prakiraan yang dipakai</th><th>Ambang waspada</th><th>Ambang bahaya</th></tr></thead>${KEYS.map(k => `<tr><td>${nm(k)}</td><td>${num(parseInt(k, 10) + 13.6)} jam</td><td>${parseInt(k, 10) >= 18 ? '2 hari' : '1 hari'} sebelum waktu berlaku</td><td>${pc(H[k][info.primary].th_waspada)}</td><td>${pc(H[k][info.primary].th_bahaya)}</td></tr>`).join('')}</table></div>
        <p class="note">Waspada memaksimalkan skor F2 (lebih menekankan deteksi), bahaya memaksimalkan skor F0,5 (lebih menekankan ketepatan).</p></section>
      <section><h3>Indeks pada bola</h3><p>Peluang dan ambang berbeda di tiap horizon, jadi bola memakai indeks 0 sampai 100 yang sama: 55 tepat di ambang waspada dan 85 tepat di ambang bahaya. Indeks bukan peluang.</p></section>
      <section class="grid2"><div><h3>Kedudukan LSTM</h3><p>LSTM membaca 24 METAR terakhir dan meramal suhu, titik embun, QNH, dan kecepatan angin tiap 30 menit sampai 24 jam. Ramalannya tampil sebagai garis titik di meteogram.</p><p>Pada data uji 2026, susunan ini setara dengan XGBoost yang memakai METAR langsung: selisihnya tidak nyata di semua horizon. LSTM sebagai pengklasifikasi sendiri, dirata-rata, atau ditumpuk juga tidak lebih baik secara nyata, dan ditumpuk justru lebih buruk di +1 jam.</p></div>
        <div><h3>Data latih</h3><table class="data txt"><tr><td>LSTM peramal</td><td>${esc(ev.training_period.lstm)}</td></tr><tr><td>XGBoost</td><td>${esc(ev.training_period.berurutan)}</td></tr><tr><td>Model cadangan</td><td>${esc(ev.training_period.cadangan)}</td></tr><tr><td>Uji</td><td>${esc(ev.test_period)}</td></tr><tr><td>Versi</td><td><code>${esc(info.version)}</code></td></tr><tr><td>Sumber ECMWF</td><td><code>${esc(info.ecmwf_model)}</code></td></tr></table></div></section>
      <section><h3>Coba model lama dengan METAR sendiri</h3><p>Dua model lama masih tersedia untuk perbandingan. Keduanya memakai METAR saja. XGBoost lama butuh minimal 7 laporan, LSTM lama 12 laporan, berurutan tiap 30 menit.</p>
        <label for="mraw">METAR, satu laporan per baris, dari yang paling lama</label><textarea id="mraw" spellcheck="false"></textarea><div class="row" style="margin-top:8px"><button class="btn" id="mfill">Isi dari data terbaru</button><button class="btn" data-old="xgboost">Hitung XGBoost lama</button><button class="btn" data-old="lstm">Hitung LSTM lama</button></div><div id="mout" role="status" style="margin-top:12px"></div></section>`;
    $('mfill').onclick = async () => { try { const h = await api('/api/metar/history?count=20'); const raw = h.data.map(r => r.raw_metar).filter(Boolean); $('mraw').value = raw.join('\n'); $('mout').textContent = raw.length < 12 ? `Baru ${raw.length} METAR mentah yang tersimpan.` : ''; } catch (e) { $('mout').textContent = `Gagal mengambil data. ${e.message}`; } };
    document.querySelectorAll('[data-old]').forEach(b => { b.onclick = async () => {
      const lines = $('mraw').value.split('\n').map(s => s.trim()).filter(Boolean);
      $('mout').textContent = 'Menghitung.';
      try {
        const r = await api(`/api/${b.dataset.old}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source: 'manual', raw_metars: lines }) });
        $('mout').innerHTML = `<div class="scroll"><table class="data"><thead><tr><th>${b.dataset.old === 'lstm' ? 'LSTM lama' : 'XGBoost lama'}</th><th>Peluang</th><th>Waspada</th><th>Bahaya</th><th>Untuk</th></tr></thead>${r.predictions.filter(p => p.available).map(p => `<tr><td>${nm(p.horizon)}</td><td>${pc(p.probability)}</td><td>${pc(p.thresholds.th_waspada)}</td><td>${pc(p.thresholds.th_bahaya)}</td><td>${esc(p.target_wib)}</td></tr>`).join('')}</table></div>`;
      } catch (e) { $('mout').innerHTML = `<span class="t-ba">Tidak dapat dihitung. ${esc(e.message)}</span>`; }
    }; });
  }
  PAGES.model = {
    draw() {},
    async start() {
      $('p-model').innerHTML = '<h2>Model</h2><p class="lead">Memuat keterangan model.</p>';
      try { build(await api('/api/ews/model')); }
      catch (e) { $('p-model').innerHTML = `<h2>Model</h2><div class="msg err">Keterangan model tidak dapat dimuat. ${esc(e.message)}.</div><button class="btn" id="mretry">Coba lagi</button>`; $('mretry').onclick = () => this.start(); }
    },
  };
})();
