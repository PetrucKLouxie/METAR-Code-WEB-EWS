/* Halaman referensi: rumus, ukuran verifikasi, data, API, dan pustaka. */
(() => {
  const i = s => `<mi>${s}</mi>`, n = s => `<mn>${s}</mn>`, o = s => `<mo>${s}</mo>`, tx = s => `<mtext>${s}</mtext>`;
  const row = (...a) => `<mrow>${a.join('')}</mrow>`, fr = (a, b) => `<mfrac>${a}${b}</mfrac>`, sub = (a, b) => `<msub>${a}${b}</msub>`, sup = (a, b) => `<msup>${a}${b}</msup>`;
  const eq = (...a) => `<div class="eq"><math display="block">${row(...a)}</math></div>`;
  const T = i('T'), Td = sub(i('T'), i('d')), P = i('p'), es = sub(i('e'), i('s')), e = i('e');
  const magnus = v => row(n('6,112'), o('⋅'), i('exp'), o('('), fr(row(n('17,67'), o('⋅'), v), row(v, o('+'), n('243,5'))), o(')'));
  const TP = i('TP'), FP = i('FP'), FN = i('FN');
  const gate = (a, b) => row(i('σ'), o('('), sub(i('W'), i(b)), sub(i('x'), i('t')), o('+'), sub(i('U'), i(b)), sub(i('h'), row(i('t'), o('−'), n('1'))), o('+'), sub(i('b'), i(b)), o(')'));
  const API = [
    ['GET', '/api/ews/status', 'Peringatan 1 sampai 24 jam yang sedang berlaku, ambang, umur data, dan pembanding LSTM.'],
    ['GET', '/api/ews/model', 'Versi, ambang, fitur, dan hasil uji model peringatan.'],
    ['GET', '/api/peta', 'Medan angin, hujan, CAPE, kelembapan, dan suhu ECMWF IFS pada 104 titik.'],
    ['GET', '/api/metar/latest', 'METAR terbaru; mencatat slot baru ke Google Sheets bila ada.'],
    ['GET', '/api/metar/history?count=20', 'Riwayat pengamatan, 1 sampai 100 baris.'],
    ['GET', '/api/forecast-7days', 'Prospek 7 hari ECMWF IFS dengan skor aturan.'],
    ['GET', '/api/forecast-7days/snapshots', 'Snapshot prospek dan hasil verifikasinya.'],
    ['POST', '/api/forecast-7days/snapshot', 'Menyimpan snapshot. Memerlukan header X-Admin-Token.'],
    ['POST', '/api/forecast-7days/verify', 'Memverifikasi snapshot. Memerlukan header X-Admin-Token.'],
    ['POST', '/api/xgboost/predict, /api/lstm/predict', 'Model lama, untuk perbandingan.'],
  ];
  const COLS = [['slot_30min', 'Waktu pengamatan, UTC'], ['wind_speed', 'Kecepatan angin, knot'], ['dew_point', 'Titik embun, °C'], ['pressure', 'QNH, hPa'], ['temperature', 'Suhu, °C'], ['bad_weather', '1 bila laporan memuat badai petir atau angin minimal 25 kt; label latih'], ['raw_metar', 'Teks METAR asli, sejak 25 November 2026']];
  PAGES.referensi = {
    draw() {},
    start() {
      $('p-referensi').innerHTML = `<h2>Referensi</h2><p class="lead">Rumus, ukuran verifikasi, sumber data, dan cara mengakses data halaman ini.</p>
      <section class="grid2"><div><h3>Fitur dari METAR</h3><p>Tekanan uap jenuh dan aktual (Magnus), dalam hPa:</p>${eq(es, o('='), magnus(T))}${eq(e, o('='), magnus(Td))}
        <p>Kelembapan relatif dan defisit tekanan uap:</p>${eq(i('RH'), o('='), n('100'), o('⋅'), fr(e, es), o(','), o('&#8195;'), i('VPD'), o('='), es, o('−'), e)}
        <p>Kerapatan udara dan energi angin, dengan tekanan dalam Pa, suhu dalam K, dan angin dalam m/s:</p>${eq(i('ρ'), o('='), fr(P, row(n('287,058'), o('⋅'), T)), o(','), o('&#8195;'), i('E'), o('='), fr(n('1'), n('2')), i('ρ'), sup(i('v'), n('2')))}</div>
        <div><h3>Ukuran verifikasi</h3><p>TP adalah peringatan yang tepat, FP alarm palsu, FN kejadian yang terlewat.</p>${eq(i('POD'), o('='), fr(TP, row(TP, o('+'), FN)), o(','), o('&#8195;'), i('FAR'), o('='), fr(FP, row(TP, o('+'), FP)))}${eq(i('CSI'), o('='), fr(TP, row(TP, o('+'), FP, o('+'), FN)))}
        <p>Brier adalah rata-rata kuadrat selisih peluang dan kejadian:</p>${eq(i('BS'), o('='), fr(n('1'), i('N')), o('∑'), sup(row(o('('), sub(i('p'), i('i')), o('−'), sub(i('y'), i('i')), o(')')), n('2')))}
        <p>Akurasi tidak dipakai. Kejadian hanya sekitar 4% dari semua slot, sehingga selalu menjawab "aman" pun sudah 96% akurat.</p></div></section>
      <section class="grid2"><div><h3>Kalibrasi dan ambang</h3><p>Keluaran XGBoost (margin <i>m</i>) diubah menjadi peluang dengan kalibrasi Platt:</p>${eq(i('p'), o('='), fr(n('1'), row(n('1'), o('+'), sup(i('e'), row(o('−'), o('('), i('a'), i('m'), o('+'), i('b'), o(')'))))))}<p>Nilai <i>a</i>, <i>b</i>, dan ambang dipilih dari prediksi silang per blok 14 hari pada data latih. Blok dipisah 2 hari agar badai yang sama tidak muncul di kedua sisi.</p></div>
        <div><h3>LSTM, sebagai pembanding</h3>${eq(sub(i('f'), i('t')), o('='), gate('f', 'f'))}${eq(sub(i('i'), i('t')), o('='), gate('i', 'i'), o(','), o('&#8195;'), sub(i('o'), i('t')), o('='), gate('o', 'o'))}${eq(sub(i('c'), i('t')), o('='), sub(i('f'), i('t')), o('⊙'), sub(i('c'), row(i('t'), o('−'), n('1'))), o('+'), sub(i('i'), i('t')), o('⊙'), i('tanh'), o('('), sub(i('W'), i('c')), sub(i('x'), i('t')), o('+'), sub(i('U'), i('c')), sub(i('h'), row(i('t'), o('−'), n('1'))), o('+'), sub(i('b'), i('c')), o(')'))}${eq(sub(i('h'), i('t')), o('='), sub(i('o'), i('t')), o('⊙'), i('tanh'), o('('), sub(i('c'), i('t')), o(')'))}<p>LSTM lama: 12 slot (6 jam), dua lapis 64 dan 32 unit, lima keluaran.</p></div></section>
      <section class="grid2"><div><h3>Pemisahan data</h3><p>Data diurutkan menurut waktu dan tidak diacak: dilatih pada masa lalu, diuji pada masa sesudahnya. Nilai kosong hanya diisi dari nilai sebelumnya. Data uji 2026 tidak dipakai untuk memilih ambang atau kalibrasi. Namun kelompok fitur musim, riwayat badai, dan ECMWF dipilih setelah melihat hasil uji, sehingga angkanya perlu dikonfirmasi pada data baru.</p></div>
        <div><h3>Kolom Google Sheets</h3><div class="scroll"><table class="data txt" style="min-width:0">${COLS.map(([a, b]) => `<tr><td><code>${a}</code></td><td>${b}</td></tr>`).join('')}</table></div></div></section>
      <section><h3>API</h3><div class="scroll"><table class="data txt"><thead><tr><th>Metode</th><th>Alamat</th><th>Isi</th></tr></thead>${API.map(([m, p, d]) => `<tr><td>${m}</td><td><code>${p}</code></td><td>${d}</td></tr>`).join('')}</table></div></section>
      <section class="grid2"><div><h3>Sumber data</h3><ul class="ref"><li>METAR WARR: <a href="https://aviationweather.gov/data/api/">Aviation Weather Center</a>.</li><li>ECMWF IFS: <a href="https://www.ecmwf.int/en/forecasts/datasets/open-data">ECMWF Open Data</a>, diakses lewat <a href="https://open-meteo.com/">Open-Meteo</a>.</li><li>Elevasi: <a href="https://registry.opendata.aws/terrain-tiles/">Terrain Tiles</a> (SRTM dan sumber lain).</li><li>Garis pantai: <a href="https://www.naturalearthdata.com/">Natural Earth</a> 1:10 juta.</li><li>Peringatan resmi: <a href="https://www.bmkg.go.id/">BMKG</a>.</li></ul></div>
        <div><h3>Pustaka</h3><ul class="ref"><li>Frifra, A., dkk. (2024). Harnessing LSTM and XGBoost algorithms for storm prediction. <i>Scientific Reports</i>.</li><li>Mostajabi, A., dkk. (2019). Nowcasting lightning occurrence from commonly available meteorological parameters using machine learning techniques. <i>npj Climate and Atmospheric Science</i>.</li><li>Wilks, D. S. (2011). <i>Statistical Methods in the Atmospheric Sciences</i>.</li><li>Hyndman, R. J., dan Athanasopoulos, G. (2018). <i>Forecasting: Principles and Practice</i>, edisi ke-2.</li><li>Hochreiter, S., dan Schmidhuber, J. (1997). Long Short-Term Memory. <i>Neural Computation</i>, 9(8), 1735 sampai 1780.</li><li>Ben Taieb, S., dkk. (2012). Strategi peramalan multi-step. <i>Expert Systems with Applications</i>, 39(8), 7067 sampai 7083.</li></ul></div></section>`;
    },
  };
})();
