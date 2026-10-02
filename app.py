import os
import re
import json
import time
import requests
import gspread
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
def get_recent_rows_from_sheet(sheet, count=20):
    """Mengambil 20 baris terakhir dari sheet berukuran 100k+ secara cepat"""
    max_rows = sheet.row_count
    chunk_size = 1500
    start_chunk = max(1, max_rows - chunk_size)
    col_a_chunk = sheet.get_values(f"A{start_chunk}:A{max_rows}")

    last_offset = 0
    for i, r in enumerate(col_a_chunk):
        if r and r[0].strip():
            last_offset = i
    last_row = start_chunk + last_offset

    start_row = max(2, last_row - count + 1)
    raw_rows = sheet.get_values(f"A{start_row}:F{last_row}")

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
    return records, last_row

# ---------------------------------------------
# PARSER METAR & EKSTRAKSI SLOT WAKTU UTC
# ---------------------------------------------
def get_metar_observation_time_utc(raw_text: str):
    """
    Ekstrak kode waktu dari METAR (contoh: 022130Z)
    dan format waktu langsung dalam UTC (Zulu):
    YYYY-MM-DD HH:MM:00
    """
    m = re.search(r'\b(\d{2})(\d{2})(\d{2})Z\b', raw_text)
    if m:
        day, hour, minute = int(m.group(1)), int(m.group(2)), int(m.group(3))
        now_utc = datetime.now(timezone.utc)
        obs_utc = now_utc.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
        slot_utc = obs_utc.strftime("%Y-%m-%d %H:%M:%S")
        code = f"{day:02d}{hour:02d}{minute:02d}Z"
        return slot_utc, code
    
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
# ---------------------------------------------
def fetch_and_append_metar():
    global _latest_raw_metar, _latest_parsed_metar, _last_saved_metar_code, _last_saved_slot_time
    url = "https://aviationweather.gov/api/data/metar?ids=WARR&format=raw"
    try:
        resp = requests.get(url, timeout=10)
        raw_text = resp.text.strip()
        if not raw_text:
            return {"status": "error", "message": "METAR kosong dari API"}

        parsed = parse_metar(raw_text)
        slot_utc, metar_code = get_metar_observation_time_utc(raw_text)

        _latest_raw_metar = raw_text
        _latest_parsed_metar = parsed

        record = {
            "slot_30min": slot_utc,
            "timestamp": slot_utc,
            "raw_metar": raw_text,
            "wind_speed": parsed["wind_speed"],
            "dew_point": parsed["dew_point"],
            "pressure": parsed["pressure"],
            "temperature": parsed["temp"],
            "temp": parsed["temp"],
            "bad_weather": parsed["bad_weather"]
        }

        # 1. Cek Duplikat di Memori
        if _last_saved_metar_code and metar_code == _last_saved_metar_code:
            return {
                "status": "skipped",
                "message": f"Data METAR periode {metar_code} ({slot_utc} UTC) sudah tercatat.",
                "data": record,
                "sheet_synced": True
            }

        if _last_saved_slot_time and slot_utc == _last_saved_slot_time:
            return {
                "status": "skipped",
                "message": f"Data slot {slot_utc} UTC sudah tercatat.",
                "data": record,
                "sheet_synced": True
            }

        # 2. Cek Duplikat di Local History
        history = load_local_history()
        for item in history:
            if item.get("slot_30min") == slot_utc or item.get("raw_metar") == raw_text:
                _last_saved_metar_code = metar_code
                _last_saved_slot_time = slot_utc
                return {
                    "status": "skipped",
                    "message": f"Data slot {slot_utc} UTC sudah tersimpan di riwayat lokal.",
                    "data": record,
                    "sheet_synced": True
                }

        # Format 6 kolom dataset Google Sheet (dalam UTC):
        row_for_sheet = [
            slot_utc,
            parsed["wind_speed"],
            parsed["dew_point"],
            parsed["pressure"],
            parsed["temp"],
            parsed["bad_weather"]
        ]

        sheet_synced = False
        sheet_error = None

        # 3. Cek Duplikat di Google Sheet
        try:
            sheet = get_sheet()
            recent_rows, last_row = get_recent_rows_from_sheet(sheet, count=5)
            
            for r in recent_rows:
                if r.get("slot_30min") == slot_utc:
                    _last_saved_metar_code = metar_code
                    _last_saved_slot_time = slot_utc
                    save_local_record(record)
                    return {
                        "status": "skipped",
                        "message": f"Data slot {slot_utc} UTC sudah ada di Google Sheet.",
                        "data": record,
                        "sheet_synced": True
                    }

            sheet.append_row(row_for_sheet)
            sheet_synced = True
            _last_saved_metar_code = metar_code
            _last_saved_slot_time = slot_utc
        except Exception as se:
            sheet_error = str(se)

        save_local_record(record)

        return {
            "status": "success",
            "data": record,
            "sheet_synced": sheet_synced,
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
@app.get("/")
def serve_index():
    """Menyajikan halaman web frontend"""
    if os.path.exists(INDEX_FILE):
        return FileResponse(INDEX_FILE)
    return {"message": "index.html tidak ditemukan di direktori project"}

@app.get("/api/metar/latest")
def get_latest():
    """Mengambil METAR terbaru, mem-parsing, dan append jika periode baru"""
    return fetch_and_append_metar()

@app.get("/api/metar/history")
def get_history():
    """Membaca 20 baris riwayat terakhir dari Google Sheets (format UTC)"""
    global _latest_raw_metar
    if not _latest_raw_metar:
        fetch_and_append_metar()

    try:
        sheet = get_sheet()
        records, last_row = get_recent_rows_from_sheet(sheet, count=20)
        if records:
            if _latest_raw_metar and len(records) > 0:
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
        "data": local_records[-20:],
        "latest_raw": _latest_raw_metar,
        "timezone": "UTC"
    }