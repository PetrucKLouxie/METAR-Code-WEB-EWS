# Panduan Deploy WARR METAR ke Vercel

Project ini sudah dikonfigurasi penuh dan siap untuk di-deploy ke **Vercel** menggunakan Python Serverless Runtime.

---

## 📁 File Konfigurasi yang Sudah Disiapkan
1. **`vercel.json`** : Mengatur *routing* dan *rewrites* agar Vercel mengarahkan frontend ke `index.html` dan API ke Serverless Python `api/index.py`.
2. **`api/index.py`** : *Serverless Function entry point* untuk Vercel.
3. **`requirements.txt`** : Daftar pustaka Python yang akan diinstal otomatis oleh Vercel saat proses *build*.
4. **`.gitignore`** : Memastikan file rahasia seperti `credentials.json`, `.env`, dan `.venv` **tidak** ter-upload ke Git/GitHub publik.
5. **`app.py`** : Sudah mendukung pembacaan kredensial dari **Environment Variable** (`GOOGLE_CREDENTIALS_JSON`), mode *serverless* aman, dan penanganan *filesystem* sementara (`/tmp`).

## Inferensi XGBoost
- `POST /api/xgboost/predict` menerima `{"source":"gsheet"}` untuk riwayat Sheets/cache lokal atau `{"source":"manual","raw_metars":[...]}` untuk METAR mentah. Input manual perlu minimal 7 laporan berurutan yang berjarak 30 menit.
- Model dan konfigurasi inferensi berasal dari `saved_models/`; lima artefak JSON dimuat lazy dan disimpan pada cache memori instance. Tree numerik `binary:logistic` dievaluasi dengan Python standard library, sehingga fungsi Vercel tidak memasang XGBoost, NumPy, atau scikit-learn. Parameter kalibrasi Platt dari konfigurasi tetap digunakan.
- `vercel.json` hanya menyertakan halaman, konfigurasi/model JSON, dan aset yang diperlukan. Folder virtual environment lokal serta artefak `.joblib` lama dikecualikan dari deployment.
- Lima horizon (1h, 3h, 9h, 18h, 24h) tersedia di `saved_models/`.

---

## 🚀 Langkah-langkah Deploy ke Vercel

### Langkah 1: Push Project ke GitHub
Buka terminal/PowerShell di folder project ini:
```bash
git add .
git commit -m "Siap deploy ke Vercel"
git push origin main
```
> **Catatan Keamanan:** File `credentials.json` tidak akan ikut ter-upload karena sudah masuk ke `.gitignore`.

---

### Langkah 2: Import Project di Vercel
1. Buka [https://vercel.com](https://vercel.com) dan login.
2. Klik tombol **"Add New..."** > **"Project"**.
3. Pilih repository GitHub project Anda: `METAR-Code-WEB-EWS` (atau nama repo Anda).
4. Di bagian **Framework Preset**, biarkan **Other**.
5. Di bagian **Root Directory**, biarkan `./`.

---

### Langkah 3: Atur Environment Variables di Vercel *(Paling Penting!)*
Sebelum menekan tombol Deploy, buka bagian **Environment Variables**:

1. **Tambahkan Kunci Kredensial:**
   * **Key**: `GOOGLE_CREDENTIALS_JSON`
   * **Value**: Buka file [credentials.json](credentials.json) di komputer Anda, salin seluruh isinya (mulai dari tanda `{` sampai `}`), lalu tempel (paste) ke kolom Value ini.
   * Centang: **Production**, **Preview**, dan **Development**.
   * Klik **Add**.

2. *(Opsional - sudah ada nilai default)*:
   * `SPREADSHEET_ID`: `1MWaA3zgLxFsDx3ziPJB02Jb5XkZA5P40--6OoUyH4sw`
   * `SHEET_NAME`: `METAR Record`

---

### Langkah 4: Klik "Deploy"
1. Klik tombol **"Deploy"**.
2. Tunggu 1–2 menit hingga proses build selesai.
3. Vercel akan memberikan domain aktif gratis Anda (contoh: `https://metar-code-web-ews.vercel.app`).
4. Buka link tersebut di browser. Dashboard METAR Juanda Airport langsung live dan terhubung ke Google Sheet!
