import os
import re
import hmac
import json
import time
import threading
import requests
import gspread
from datetime import datetime, timezone, timedelta
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from predict_service import _metar_timestamp, predict_from_gsheet, predict_from_raw_metar_list
from lstm_service import (
    get_lstm_public_metadata,
    predict_lstm_from_gsheet,
    predict_lstm_from_raw_metar_list,
)
from hybrid_service import compute_hybrid_predictions, predict_hybrid_from_records
import ews_log
import ews_service
import map_service
from weekly_nwp_service import compute_ai_nwp_weekly_forecast

try:
    from google.oauth2.service_account import Credentials
    HAS_GOOGLE_AUTH = True
except ImportError:
    HAS_GOOGLE_AUTH = False
    try:
        from oauth2client.service_account import ServiceAccountCredentials
    except ImportError:
        pass

app = FastAPI(title="WARR METAR Collector & Parser")
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
if os.path.isdir(PUBLIC_DIR):
    app.mount("/public", StaticFiles(directory=PUBLIC_DIR), name="public")
ASSET_DIR = os.path.join(PUBLIC_DIR, "assets")
if os.path.isdir(ASSET_DIR):
    app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="assets")
IMAGES_DIR = os.path.join(PUBLIC_DIR, "images")
if os.path.isdir(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
LSTM_IMAGE_DIR = os.path.join(PUBLIC_DIR, "images", "lstm")
if os.path.isdir(LSTM_IMAGE_DIR):
    app.mount("/images/lstm", StaticFiles(directory=LSTM_IMAGE_DIR), name="lstm-images")

# Izinkan frontend mengakses API backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------
# FILE PATHS & ENVIRONMENT VARIABLES
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDS_FILE = os.path.join(BASE_DIR, "credentials.json")
INDEX_FILE = os.path.join(BASE_DIR, "index.html")

# Di Vercel serverless, root dir bersifat read-only sehingga cache disimpan di /tmp
IS_VERCEL = bool(os.environ.get("VERCEL"))
if IS_VERCEL or not os.access(BASE_DIR, os.W_OK):
    LOCAL_HISTORY_FILE = "/tmp/metar_history.json"
else:
    LOCAL_HISTORY_FILE = os.path.join(BASE_DIR, "metar_history.json")

SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "1MWaA3zgLxFsDx3ziPJB02Jb5XkZA5P40--6OoUyH4sw")
SHEET_NAME = os.environ.get("SHEET_NAME", "METAR Record")

_cached_sheet_client = None
_cached_sheet = None
_last_sheet_error_time = 0
_last_creds_mtime = 0
_latest_raw_metar = None
_latest_parsed_metar = None
_last_saved_metar_code = None
_last_saved_slot_time = None
SHEET_RETRY_INTERVAL = 60

# Satu pengumpulan pada satu waktu: scheduler dan permintaan pengunjung bisa
# bersamaan, dan tanpa kunci keduanya menulis slot yang sama.
_collect_lock = threading.Lock()
_last_collect = None
MIN_COLLECT_INTERVAL = 20

# Petunjuk posisi baris terakhir. row_count milik gspread hanya salinan dan
# melenceng pada proses yang hidup lama, jadi hanya dipakai saat pertama membaca.
_tail_hint = None
_raw_column_ok = None


def require_admin(x_admin_token: str | None = Header(default=None)):
    """Endpoint yang menulis ke Google Sheet wajib membawa header X-Admin-Token."""
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN belum diatur di server.")
    if not hmac.compare_digest((x_admin_token or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Token admin salah atau belum diisi.")


def get_credentials():
    """
    Mengambil kredensial Google Service Account:
    1. Membaca Environment Variable GOOGLE_CREDENTIALS_JSON (untuk Vercel/Cloud)
    2. Fallback membaca file credentials.json (untuk lokal)
    """
    env_creds = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if env_creds and env_creds.strip():
        try:
            creds_dict = json.loads(env_creds)
            return Credentials.from_service_account_info(creds_dict, scopes=SCOPE)
        except Exception as e:
            print(f"Error parsing GOOGLE_CREDENTIALS_JSON environment variable: {e}")

    if os.path.exists(CREDS_FILE):
        if HAS_GOOGLE_AUTH:
            return Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPE)
        else:
            return ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)

    raise FileNotFoundError(
        "Kredensial Google Service Account tidak ditemukan! "
        "Di Vercel: tambahkan Environment Variable 'GOOGLE_CREDENTIALS_JSON'. "
        "Di Lokal: letakkan file 'credentials.json' di direktori project."
    )

def get_sheet():
    global _cached_sheet_client, _cached_sheet, _last_sheet_error_time, _last_creds_mtime
    now = time.time()

    # Cek perubahan file credentials.json jika berjalan di lokal
    if not IS_VERCEL and os.path.exists(CREDS_FILE):
        try:
            current_mtime = os.path.getmtime(CREDS_FILE)
            if current_mtime != _last_creds_mtime:
                _last_creds_mtime = current_mtime
                _cached_sheet_client = None
                _cached_sheet = None
                _last_sheet_error_time = 0
        except OSError:
            pass

    if _cached_sheet is not None:
        return _cached_sheet

    if _last_sheet_error_time and (now - _last_sheet_error_time < SHEET_RETRY_INTERVAL):
        raise RuntimeError("Google Sheets offline/credentials invalid. Menggunakan penyimpanan lokal.")

    try:
        creds = get_credentials()
        client = gspread.authorize(creds)

        # Pasang retry adapter untuk stabilitas SSL koneksi Google
        adapter = HTTPAdapter(max_retries=Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        ))
        client.http_client.session.mount("https://", adapter)

        # Buka sheet langsung dengan ID spreadsheet agar cepat
        try:
            sheet = client.open_by_key(SPREADSHEET_ID).sheet1
        except Exception:
            sheet = client.open(SHEET_NAME).sheet1

        _cached_sheet_client = client
        _cached_sheet = sheet
        _last_sheet_error_time = 0
        return sheet
    except Exception as e:
        _last_sheet_error_time = now
        _cached_sheet_client = None
        _cached_sheet = None
        raise e

# ---------------------------------------------
# PENYIMPANAN LOCAL HISTORY (FALLBACK)
# ---------------------------------------------
def load_local_history():
    if os.path.exists(LOCAL_HISTORY_FILE):
        try:
            with open(LOCAL_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_local_record(record: dict):
    history = load_local_history()
    # Hindari duplikat record berdasarkan slot_30min atau raw_metar
    for item in history:
        if item.get("slot_30min") == record.get("slot_30min") or item.get("raw_metar") == record.get("raw_metar"):
            return history
    history.append(record)
    if len(history) > 100:
        history = history[-100:]
    try:
        with open(LOCAL_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        pass
    return history

# ---------------------------------------------
# AMBIL BARIS TERAKHIR GOOGLE SHEET SECARA CEPAT
# ---------------------------------------------
def read_tail_rows(sheet, count, hint=None):
    """
    Membaca `count` baris data terakhir (kolom A-G) tanpa membaca 100k+ baris.
    Mengembalikan (baris, nomor baris terakhir).
    """
    start = max(2, hint - count + 1 if hint else sheet.row_count - 1500)
    rows = sheet.get_values(f"A{start}:G")
    filled = [i for i, r in enumerate(rows) if r and str(r[0]).strip()]
    if not filled and start > 2:
        # Ujung grid hanya berisi baris kosong, atau petunjuk posisi sudah usang.
        start = 2
        rows = sheet.get_values("A2:G")
        filled = [i for i, r in enumerate(rows) if r and str(r[0]).strip()]
    if not filled:
        return [], start - 1
    end = filled[-1] + 1
    return rows[max(0, end - count):end], start + end - 1

def raw_column_ready(sheet):
    """True bila kolom G tersedia untuk METAR mentah (kosong atau milik kita)."""
    global _raw_column_ok
    if _raw_column_ok is None:
        try:
            if sheet.col_count < 7:
                _raw_column_ok = False
            else:
                header = sheet.acell("G1").value
                if not header:
                    sheet.update_acell("G1", "raw_metar")
                _raw_column_ok = header in (None, "", "raw_metar")
        except Exception as e:
            print(f"Kolom raw_metar belum dapat dipakai: {e}")
            return False
    return _raw_column_ok

def get_recent_rows_from_sheet(sheet, count=20):
    """Mengambil baris terakhir dari sheet berukuran 100k+ secara cepat"""
    global _tail_hint
    raw_rows, last_row = read_tail_rows(sheet, count, _tail_hint)
    _tail_hint = last_row

    records = []
    seen_slots = set()
    for r in raw_rows:
        if not r or not any(r):
            continue
        slot = r[0] if len(r) > 0 else ""
        if not slot or slot in seen_slots:
            continue
        seen_slots.add(slot)

        try:
            wind = float(r[1]) if len(r) > 1 and r[1] != "" else None
        except ValueError:
            wind = None
        try:
            dew = float(r[2]) if len(r) > 2 and r[2] != "" else None
        except ValueError:
            dew = None
        try:
            press = float(r[3]) if len(r) > 3 and r[3] != "" else None
        except ValueError:
            press = None
        try:
            temp = float(r[4]) if len(r) > 4 and r[4] != "" else None
        except ValueError:
            temp = None
        try:
            bad = int(float(r[5])) if len(r) > 5 and r[5] != "" else 0
        except ValueError:
            bad = 0

        records.append({
            "slot_30min": slot,
            "timestamp": slot,
            "wind_speed": wind,
            "dew_point": dew,
            "pressure": press,
            "temperature": temp,
            "temp": temp,
            "bad_weather": bad
        })
        if len(r) > 6 and r[6]:
            records[-1]["raw_metar"] = r[6]
    return records, last_row

# ---------------------------------------------
# PARSER METAR & EKSTRAKSI SLOT WAKTU UTC
# ---------------------------------------------
def get_metar_observation_time_utc(raw_text: str):
    """
    Ekstrak kode waktu dari METAR (contoh: 022130Z)
    dan format waktu langsung dalam UTC (Zulu):
    YYYY-MM-DD HH:MM:00

    METAR hanya memuat tanggal, tanpa bulan. Bulan dipilih yang paling dekat
    dengan waktu sekarang agar METAR akhir bulan tidak terbaca sebagai bulan baru.
    """
    m = re.search(r'\b(\d{2})(\d{2})(\d{2})Z\b', raw_text)
    if m:
        slot_utc = _metar_timestamp(raw_text).strftime("%Y-%m-%d %H:%M:%S")
        return slot_utc, m.group(0)
    
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), None

def parse_metar(text: str):
    if not text or "NIL" in text:
        return {"wind_speed": None, "dew_point": None, "pressure": None, "temp": None, "bad_weather": 0}

    # 1. Wind speed (kt)
    wind_m = re.search(r'(?:\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT', text)
    wind_spd = float(wind_m.group(1)) if wind_m else None

    # 2. Temperature & Dew Point (°C)
    temp_m = re.search(r'\b(M?\d{2})/(M?\d{2})\b', text)
    if temp_m:
        t_raw, td_raw = temp_m.group(1), temp_m.group(2)
        temp = -float(t_raw[1:]) if t_raw.startswith('M') else float(t_raw)
        dew_pt = -float(td_raw[1:]) if td_raw.startswith('M') else float(td_raw)
    else:
        temp, dew_pt = None, None

    # 3. Pressure QNH (hPa)
    qnh_m = re.search(r'\bQ(\d{4})\b', text)
    pressure = float(qnh_m.group(1)) if qnh_m else None

    # 4. Bad Weather
    is_ts_ra = bool(re.search(r'\b(TS|\+TSRA|TSRA|\+RA|SQ|FC)\b', text))
    is_wind_severe = (wind_spd >= 25.0) if wind_spd is not None else False
    bad_weather = 1 if (is_ts_ra or is_wind_severe) else 0

    return {
        "wind_speed": wind_spd,
        "dew_point": dew_pt,
        "pressure": pressure,
        "temp": temp,
        "bad_weather": bad_weather
    }

# ---------------------------------------------
# FUNGSI FETCH, PARSE, & PUSH KE GOOGLE SHEET
# (Dilengkapi Auto-Backfill 24 Jam agar tidak ada gap data saat website ditutup)
# ---------------------------------------------
def parse_slot_datetime(s: str):
    """Konversi string slot waktu menjadi objek datetime UTC untuk perbandingan kronologis"""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    # Handle single digit hour (contoh: 2026-10-03 0:00:00 atau 2:30:00 dari Google Sheets)
    parts = s.split(' ')
    if len(parts) == 2:
        date_p, time_p = parts
        t_parts = time_p.split(':')
        if len(t_parts) >= 2:
            try:
                h = int(t_parts[0])
                m = int(t_parts[1])
                sec = int(t_parts[2]) if len(t_parts) > 2 else 0
                y, mo, d = [int(x) for x in date_p.split('-')]
                return datetime(y, mo, d, h, m, sec, tzinfo=timezone.utc)
            except ValueError:
                pass
    return None

def fetch_and_append_metar():
    """Mengumpulkan METAR; panggilan yang berdekatan memakai hasil yang sama."""
    global _last_collect
    with _collect_lock:
        if _last_collect and time.monotonic() - _last_collect[0] < MIN_COLLECT_INTERVAL:
            return _last_collect[1]
        result = _collect_metar()
        if result.get("status") in ("success", "skipped"):
            _last_collect = (time.monotonic(), result)
        return result

def _collect_metar():
    global _cached_sheet, _tail_hint, _latest_raw_metar, _latest_parsed_metar, _last_saved_metar_code, _last_saved_slot_time
    # Ambil data 24 jam terakhir dari Aviation Weather API agar bisa auto-backfill jika sempat offline
    url_24h = "https://aviationweather.gov/api/data/metar?ids=WARR&format=raw&hours=24"
    try:
        resp = requests.get(url_24h, timeout=10)
        raw_lines = [l.strip() for l in resp.text.strip().split('\n') if l.strip()]
        if not raw_lines:
            # Fallback ke endpoint single latest jika hours=24 kosong
            resp_single = requests.get("https://aviationweather.gov/api/data/metar?ids=WARR&format=raw", timeout=10)
            raw_lines = [resp_single.text.strip()] if resp_single.text.strip() else []
            
        raw_lines = [l for l in raw_lines if not re.search(r"\bNIL\b", l)]
        if not raw_lines:
            return {"status": "error", "message": "METAR kosong dari API"}

        # raw_lines dari API terurut dari terbaru ke terlama, jadi balik urutan agar kronologis
        chronological_lines = list(reversed(raw_lines))

        # Parsing baris paling baru untuk data return & update state
        newest_raw = raw_lines[0]
        newest_parsed = parse_metar(newest_raw)
        newest_slot_utc, newest_code = get_metar_observation_time_utc(newest_raw)
        
        _latest_raw_metar = newest_raw
        _latest_parsed_metar = newest_parsed

        latest_record = {
            "slot_30min": newest_slot_utc,
            "timestamp": newest_slot_utc,
            "raw_metar": newest_raw,
            "wind_speed": newest_parsed["wind_speed"],
            "dew_point": newest_parsed["dew_point"],
            "pressure": newest_parsed["pressure"],
            "temperature": newest_parsed["temp"],
            "temp": newest_parsed["temp"],
            "bad_weather": newest_parsed["bad_weather"]
        }

        # Dapatkan slot waktu terakhir yang sudah tercatat di Google Sheet
        sheet = None
        last_recorded_dt = None
        existing_slots_set = set()
        sheet_error = None
        sheet_synced = False
        appended_count = 0

        try:
            sheet = get_sheet()
            recent_rows, last_row = get_recent_rows_from_sheet(sheet, count=48)
            for r in recent_rows:
                s = r.get("slot_30min")
                if s:
                    existing_slots_set.add(s)
                    dt = parse_slot_datetime(s)
                    if dt and (last_recorded_dt is None or dt > last_recorded_dt):
                        last_recorded_dt = dt
        except Exception as se:
            sheet_error = str(se)
            # Buka ulang sheet pada percobaan berikutnya.
            _cached_sheet = None
            _tail_hint = None

        # Jika sheet tidak dapat diakses, cek local history
        if not last_recorded_dt:
            local_history = load_local_history()
            for item in local_history:
                s = item.get("slot_30min")
                if s:
                    existing_slots_set.add(s)
                    dt = parse_slot_datetime(s)
                    if dt and (last_recorded_dt is None or dt > last_recorded_dt):
                        last_recorded_dt = dt

        # Identifikasi semua observasi baru yang belum ada di Google Sheet (auto-backfill multi-slot)
        new_rows_for_sheet = []
        new_records_for_local = []

        for line in chronological_lines:
            slot_utc, code = get_metar_observation_time_utc(line)
            obs_dt = parse_slot_datetime(slot_utc)
            
            # Cek apakah observasi ini lebih baru dari data terakhir di sheet
            is_new = False
            if last_recorded_dt and obs_dt:
                is_new = obs_dt > last_recorded_dt
            elif slot_utc not in existing_slots_set:
                is_new = True

            if is_new and slot_utc not in existing_slots_set:
                p = parse_metar(line)
                row_data = [
                    slot_utc,
                    p["wind_speed"],
                    p["dew_point"],
                    p["pressure"],
                    p["temp"],
                    p["bad_weather"]
                ]
                new_rows_for_sheet.append(row_data)
                new_records_for_local.append({
                    "slot_30min": slot_utc,
                    "timestamp": slot_utc,
                    "raw_metar": line,
                    "wind_speed": p["wind_speed"],
                    "dew_point": p["dew_point"],
                    "pressure": p["pressure"],
                    "temperature": p["temp"],
                    "temp": p["temp"],
                    "bad_weather": p["bad_weather"]
                })
                existing_slots_set.add(slot_utc)

        # Simpan ke Google Sheet jika ada data baru
        if new_rows_for_sheet and sheet:
            try:
                if raw_column_ready(sheet):
                    new_rows_for_sheet = [
                        row + [rec["raw_metar"]]
                        for row, rec in zip(new_rows_for_sheet, new_records_for_local)
                    ]
                sheet.append_rows(new_rows_for_sheet, value_input_option="USER_ENTERED")
                sheet_synced = True
                appended_count = len(new_rows_for_sheet)
                _last_saved_metar_code = newest_code
                _last_saved_slot_time = newest_slot_utc
            except Exception as se:
                sheet_error = str(se)

        # Simpan ke local history
        for rec in new_records_for_local:
            save_local_record(rec)
        if not new_records_for_local:
            save_local_record(latest_record)

        if appended_count > 0:
            log_prediction(sheet)
            return {
                "status": "success",
                "message": f"Berhasil menambahkan {appended_count} observasi METAR baru ke Google Sheet.",
                "appended_count": appended_count,
                "data": latest_record,
                "sheet_synced": sheet_synced,
                "sheet_error": sheet_error
            }
        elif sheet_error:
            return {
                "status": "warning",
                "message": "Google Sheet tidak dapat dibaca atau ditulis; data disimpan di cache lokal.",
                "data": latest_record,
                "sheet_synced": False,
                "sheet_error": sheet_error
            }
        else:
            return {
                "status": "skipped",
                "message": f"Data slot {newest_slot_utc} UTC sudah tercatat (tidak ada gap).",
                "data": latest_record,
                "sheet_synced": True,
                "sheet_error": sheet_error
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------------------------
# JALANKAN BACKGROUND SCHEDULER
# (Hanya jika di lingkungan non-serverless)
# ---------------------------------------------
if not IS_VERCEL:
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(fetch_and_append_metar, 'interval', minutes=1)
        scheduler.start()
    except Exception as e:
        print(f"Warning scheduler: {e}")

# ---------------------------------------------
# REST API ENDPOINTS & FRONTEND ROUTE
# ---------------------------------------------
def find_index_file():
    candidates = [
        INDEX_FILE,
        os.path.join(os.path.dirname(BASE_DIR), "index.html"),
        os.path.join(os.getcwd(), "index.html"),
        os.path.join(os.getcwd(), "public", "index.html")
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

@app.get("/")
@app.get("/index.html")
def serve_index():
    """Menyajikan halaman web frontend dengan no-cache agar perubahan kode langsung terlihat"""
    path = find_index_file()
    if path:
        return FileResponse(
            path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "index.html tidak ditemukan di direktori project"}

@app.get("/api")
@app.get("/api/")
@app.get("/api/index")
@app.get("/api/index.py")
def api_status():
    """Endpoint informasi status API"""
    return {
        "status": "online",
        "service": "WARR METAR Early Warning System API",
        "endpoints": {
            "latest": "/api/metar/latest",
            "history": "/api/metar/history",
            "ews": "/api/ews/status",
            "ews_model": "/api/ews/model",
            "map": "/api/peta",
            "xgboost_predict": "/api/xgboost/predict",
            "docs": "/docs"
        },
        "version": "2.4-UTC"
    }

@app.get("/api/metar/latest")
@app.get("/metar/latest")
@app.get("/api/metar/latest/")
@app.get("/metar/latest/")
def get_latest():
    """Mengambil METAR terbaru, mem-parsing, dan append jika periode baru"""
    return fetch_and_append_metar()

@app.get("/api/metar/history")
@app.get("/metar/history")
@app.get("/api/metar/history/")
@app.get("/metar/history/")
def get_history(count: int = 20):
    """Membaca baris riwayat terakhir dari Google Sheets (format UTC), 1 sampai 100 baris"""
    count = max(1, min(100, count))
    global _latest_raw_metar
    if not _latest_raw_metar:
        fetch_and_append_metar()

    try:
        sheet = get_sheet()
        records, last_row = get_recent_rows_from_sheet(sheet, count=count)
        if records:
            if _latest_raw_metar and not records[-1].get("raw_metar"):
                records[-1]["raw_metar"] = _latest_raw_metar
            return {
                "total": len(records),
                "source": "google_sheets",
                "data": records,
                "latest_raw": _latest_raw_metar,
                "timezone": "UTC"
            }
    except Exception as e:
        pass

    local_records = load_local_history()
    return {
        "total": len(local_records),
        "source": "local_storage",
        "data": local_records[-count:],
        "latest_raw": _latest_raw_metar,
        "timezone": "UTC"
    }

@app.post("/api/xgboost/predict")
def predict_xgboost(payload: dict):
    """Memprediksi risiko multi-horizon dari METAR manual atau Google Sheets."""
    source = payload.get("source", "manual")
    try:
        if source == "gsheet":
            try:
                sheet = get_sheet()
                records, _ = get_recent_rows_from_sheet(sheet, count=10)
                result_source = "google_sheets"
            except Exception:
                records = load_local_history()[-10:]
                result_source = "local_storage"
            result = predict_from_gsheet(records)
        else:
            raw_metars = payload.get("raw_metars")
            if not raw_metars and payload.get("raw_text"):
                raw_metars = [line.strip() for line in payload["raw_text"].splitlines() if line.strip()]
            result = predict_from_raw_metar_list(raw_metars)
            result_source = "manual_metar"
        result["source"] = result_source
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        print(f"XGBoost inference failed: {error!r}; cause={error.__cause__!r}", flush=True)
        raise HTTPException(status_code=503, detail="Inferensi XGBoost tidak tersedia saat ini.") from error


@app.post("/api/lstm/predict")
def predict_lstm(payload: dict):
    """Memprediksi risiko multi-horizon dari urutan METAR dengan LSTM."""
    source = payload.get("source", "manual")
    try:
        if source == "gsheet":
            try:
                sheet = get_sheet()
                records, _ = get_recent_rows_from_sheet(sheet, count=18)
                result_source = "google_sheets"
            except Exception:
                records = load_local_history()[-18:]
                result_source = "local_storage"
            result = predict_lstm_from_gsheet(records)
        else:
            raw_metars = payload.get("raw_metars")
            if not raw_metars and payload.get("raw_text"):
                raw_metars = [line.strip() for line in payload["raw_text"].splitlines() if line.strip()]
            result = predict_lstm_from_raw_metar_list(raw_metars)
            result_source = "manual_metar"
        result["source"] = result_source
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        print(f"LSTM inference failed: {error!r}; cause={error.__cause__!r}", flush=True)
        raise HTTPException(status_code=503, detail="Inferensi LSTM tidak tersedia saat ini.") from error


@app.get("/api/lstm/metrics")
def lstm_metrics():
    return get_lstm_public_metadata()


@app.get("/api/hybrid/status")
@app.post("/api/hybrid/predict")
def predict_hybrid(payload: dict = None):
    """
    Menghasilkan keputusan Early Warning System (EWS) Hybrid terpadu
    yang menggabungkan model fisik XGBoost dan temporal sequence LSTM.
    """
    try:
        try:
            sheet = get_sheet()
            records, _ = get_recent_rows_from_sheet(sheet, count=20)
            result_source = "google_sheets"
        except Exception:
            records = load_local_history()[-20:]
            result_source = "local_storage"

        if len(records) < 12:
            # Jika riwayat kurang, ambil minimal fallback dari cache
            records = load_local_history()[-20:]
            result_source = "local_storage"

        result = predict_hybrid_from_records(records)
        result["source"] = result_source
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        print(f"Hybrid EWS inference failed: {error!r}; cause={error.__cause__!r}", flush=True)
        raise HTTPException(status_code=503, detail="Inferensi Hybrid EWS tidak tersedia saat ini.") from error


STALE_AFTER_MINUTES = 75


def build_ews(records):
    """Peringatan METAR + ECMWF, LSTM lama sebagai pembanding, dan umur datanya."""
    result = ews_service.predict(records)
    lstm = {}
    try:
        lstm = {p["horizon"]: p for p in predict_lstm_from_gsheet(records[-20:])["predictions"]}
    except Exception as error:
        print(f"LSTM pembanding tidak tersedia: {error!r}", flush=True)
    for p in result["predictions"]:
        p["hybrid_probability"] = round(p["probability"], 4)
        p["xgb"] = {"probability": p["hybrid_probability"], "status": p["status"]["label"], "color": p["status"]["color"]}
        other = lstm.get(p["horizon"])
        p["lstm"] = {
            "probability": round(other["probability"], 4) if other else None,
            "status": other["status"]["label"] if other else "N/A",
            "color": other["status"]["color"] if other else "gray",
        }
    observed = datetime.strptime(result["observed_at_utc"], "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
    age = int((datetime.now(timezone.utc) - observed).total_seconds() // 60)
    result["data_age_minutes"] = age
    result["stale"] = age > STALE_AFTER_MINUTES
    result["model_version"] = ews_service.MODEL_VERSION
    return result


def log_prediction(sheet):
    """Catat prediksi untuk observasi yang baru masuk; kegagalan tidak boleh mengganggu pencatatan METAR."""
    try:
        records, _ = get_recent_rows_from_sheet(sheet, count=100)
        ews_log.append(sheet.spreadsheet, build_ews(records), datetime.now(timezone.utc))
    except Exception as error:
        print(f"Prediksi tidak tercatat: {error!r}", flush=True)


@app.get("/api/ews/status")
def ews_status():
    """
    Peringatan 1-24 jam: METAR + ECMWF IFS (cadangan: METAR saja).
    LSTM lama disertakan sebagai pembanding dan tidak menentukan status.
    """
    try:
        try:
            # 48 jam riwayat untuk fitur badai terkini
            records, _ = get_recent_rows_from_sheet(get_sheet(), count=100)
            source = "google_sheets"
        except Exception:
            records = load_local_history()
            source = "local_storage"
        result = build_ews(records)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        print(f"EWS inference failed: {error!r}", flush=True)
        raise HTTPException(status_code=503, detail="Peringatan dini tidak tersedia saat ini.") from error
    result["source"] = source
    return result


@app.get("/api/ews/model")
def ews_model():
    """Ambang, versi, dan hasil uji model peringatan yang sedang dipakai."""
    with open(os.path.join(BASE_DIR, "saved_models_ecmwf", "evaluasi.json"), encoding="utf-8") as f:
        evaluation = json.load(f)
    return {
        "version": ews_service.MODEL_VERSION,
        "ecmwf_model": ews_service.BUNDLE["ecmwf_model"],
        "features": ews_service.BUNDLE["feature_order"],
        "horizons": {
            h: {kind: ews_service.BUNDLE["models"][kind][h]["thresholds"] for kind in ("utama", "cadangan")}
            for h in ews_service.LEAD_DAY
        },
        "evaluation": evaluation,
    }


@app.get("/api/peta")
def map_fields(response: Response):
    """Medan angin, hujan, CAPE, kelembapan, dan suhu ECMWF untuk peta."""
    try:
        data = map_service.get_map()
    except Exception as error:
        print(f"Peta tidak tersedia: {error!r}", flush=True)
        raise HTTPException(status_code=503, detail="Data peta tidak tersedia saat ini.") from error
    response.headers["Cache-Control"] = "public, s-maxage=1800, stale-while-revalidate=3600"
    return data


@app.get("/api/evaluation/live")
def get_live_evaluation_results():
    """Mengembalikan hasil pengujian empiris langsung dari Google Sheets (WMO Verification)."""
    eval_file = os.path.join(BASE_DIR, "public", "eval_results_gsheet.json")
    if os.path.exists(eval_file):
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Hasil evaluasi live belum tersedia.")


@app.get("/api/forecast-7days")
def get_weekly_forecast(refresh: bool = False):
    """
    Mengembalikan Prospek Cuaca & Peringatan Dini 7 Hari ke Depan
    berbasis fusi model global ECMWF/GFS dan Machine Learning Hybrid Juanda.
    """
    try:
        return compute_ai_nwp_weekly_forecast(force_refresh=refresh)
    except Exception as error:
        print(f"Weekly forecast failed: {error!r}", flush=True)
        raise HTTPException(status_code=503, detail=f"Gagal memuat prakiraan 7 hari: {str(error)}")


@app.post("/api/forecast-7days/snapshot", dependencies=[Depends(require_admin)])
def trigger_forecast_snapshot(force: bool = False):
    """Menyimpan snapshot prakiraan 7 hari ke Google Sheets tab 'Forecast_7Days_Snapshot'."""
    try:
        from snapshot_service import save_forecast_snapshot
        return save_forecast_snapshot(force_new=force)
    except Exception as error:
        print(f"Save forecast snapshot failed: {error!r}", flush=True)
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan snapshot ke GSheet: {str(error)}")


@app.get("/api/forecast-7days/snapshots")
def list_forecast_snapshots():
    """Mengambil seluruh riwayat snapshot dan status verifikasinya dari Google Sheets."""
    try:
        from snapshot_service import get_all_snapshots
        return get_all_snapshots()
    except Exception as error:
        print(f"Fetch snapshots failed: {error!r}", flush=True)
        raise HTTPException(status_code=500, detail=f"Gagal mengambil riwayat snapshot dari GSheet: {str(error)}")


@app.post("/api/forecast-7days/verify", dependencies=[Depends(require_admin)])
def trigger_snapshot_verification():
    """Memicu verifikasi pencocokan snapshot lampau vs data METAR riil di Sheet1."""
    try:
        from snapshot_service import verify_snapshots_with_metar_records
        return verify_snapshots_with_metar_records()
    except Exception as error:
        print(f"Verify snapshots failed: {error!r}", flush=True)
        raise HTTPException(status_code=500, detail=f"Gagal memverifikasi snapshot: {str(error)}")



