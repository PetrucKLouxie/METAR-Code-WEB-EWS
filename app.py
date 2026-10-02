import re
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

app = FastAPI(title="WARR METAR Collector & Parser")

# Izinkan frontend mengakses API backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------
# KONEKSI GOOGLE SHEETS
# ---------------------------------------------
SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
CREDS_FILE = "credentials.json"
SHEET_NAME = "METAR Record"  # Sesuaikan dengan nama Google Sheet kamu

def get_sheet():
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    client = gspread.authorize(creds)
    return client.open(SHEET_NAME).sheet1

# ---------------------------------------------
# REGEX PARSER METAR
# ---------------------------------------------
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
# FUNGSI FETCH, PARSE, & PUSH KE SHEET
# ---------------------------------------------
def fetch_and_append_metar():
    url = "https://aviationweather.gov/api/data/metar?ids=WARR&format=raw"
    try:
        resp = requests.get(url, timeout=10)
        raw_text = resp.text.strip()
        if not raw_text:
            return {"status": "error", "message": "METAR kosong dari API"}

        sheet = get_sheet()
        
        # Cek baris terakhir agar tidak menduplikasi jika stringnya sama persis
        last_rows = sheet.get_all_values()
        if len(last_rows) > 1:
            last_raw = last_rows[-1][1]  # Kolom ke-2 adalah raw_metar
            if last_raw == raw_text:
                return {"status": "skipped", "message": "Data sudah tersimpan sebelumnya", "data": raw_text}

        parsed = parse_metar(raw_text)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = [
            now_str,
            raw_text,
            parsed["wind_speed"],
            parsed["dew_point"],
            parsed["pressure"],
            parsed["temp"],
            parsed["bad_weather"]
        ]
        
        # Tambahkan baris baru di sheet
        sheet.append_row(row)
        return {"status": "success", "row": row}

    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------------------------
# JALANKAN BACKGROUND CRON (TIAP 30 MENIT)
# ---------------------------------------------
scheduler = BackgroundScheduler()
# Eksekusi setiap 30 menit
scheduler.add_job(fetch_and_append_metar, 'interval', minutes=30)
scheduler.start()

# ---------------------------------------------
# REST API ENDPOINTS UNTUK WEB FRONTEND
# ---------------------------------------------
@app.get("/api/metar/latest")
def get_latest():
    """Mengambil METAR terbaru dan langsung append jika belum ada"""
    result = fetch_and_append_metar()
    return result

@app.get("/api/metar/history")
def get_history():
    """Membaca 20 baris data terakhir dari Google Sheet"""
    try:
        sheet = get_sheet()
        records = sheet.get_all_records()
        return {"total": len(records), "data": records[-20:]}
    except Exception as e:
        return {"error": str(e)}