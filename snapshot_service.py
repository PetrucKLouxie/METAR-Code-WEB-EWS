"""
Layanan Snapshot & Verifikasi Prospek 7 Hari ke Google Sheets
Menyimpan snapshot prakiraan mingguan AI-NWP ke tab 'Forecast_7Days_Snapshot'
pada Spreadsheet 'METAR Record' dan memverifikasinya secara otomatis terhadap
data observasi METAR aktual yang dicatat berkala di 'Sheet1'.
"""

import time
from datetime import datetime, timezone, timedelta
import gspread

from app import SPREADSHEET_ID, get_credentials, parse_slot_datetime, read_tail_rows
from metar_weather import observed_severity
from weekly_nwp_service import compute_ai_nwp_weekly_forecast

SHEET_SNAPSHOT_NAME = "Forecast_7Days_Snapshot"
SHEET_METAR_NAME = "Sheet1"

WIB = timezone(timedelta(hours=7))
# Tidak adanya cuaca buruk baru bisa disimpulkan bila METAR mentah hari itu
# cukup lengkap (36 dari 48 slot).
MIN_RAW_SLOTS = 36
AUTO_VERIFY_INTERVAL = 1800
_last_auto_verify = None

HEADERS = [
    "snapshot_id",
    "snapshot_time_wib",
    "target_date",
    "day_name",
    "lead_time_days",
    "predicted_risk_level",
    "predicted_risk_label",
    "predicted_max_prob",
    "predicted_precip_mm",
    "predicted_cape_max",
    "predicted_wind_gust_kt",
    "predicted_peak_window",
    "actual_verified",
    "actual_metar_observed",
    "verification_result"
]


def _get_spreadsheet_doc():
    creds = get_credentials()
    client = gspread.authorize(creds)
    return client.open_by_key(SPREADSHEET_ID)


def _ensure_snapshot_worksheet(doc):
    worksheet_names = [w.title for w in doc.worksheets()]
    if SHEET_SNAPSHOT_NAME not in worksheet_names:
        ws = doc.add_worksheet(title=SHEET_SNAPSHOT_NAME, rows=500, cols=len(HEADERS))
        ws.append_row(HEADERS)
        return ws
    return doc.worksheet(SHEET_SNAPSHOT_NAME)


def save_forecast_snapshot(force_new=False):
    """
    Mengambil data prospek 7 hari terbaru dan menyimpannya sebagai snapshot baris
    ke tab 'Forecast_7Days_Snapshot' di Google Spreadsheet.
    """
    doc = _get_spreadsheet_doc()
    ws_snap = _ensure_snapshot_worksheet(doc)

    forecast = compute_ai_nwp_weekly_forecast(force_refresh=False)
    now_wib = datetime.now(timezone(timedelta(hours=7)))
    now_str_date = now_wib.strftime("%Y-%m-%d")
    now_str_full = now_wib.strftime("%Y-%m-%d %H:%M WIB")

    existing_records = ws_snap.get_all_records()
    existing_today_targets = {
        r.get("target_date") for r in existing_records
        if str(r.get("snapshot_time_wib", "")).startswith(now_str_date)
    }

    daily_summaries = forecast.get("daily_summaries", [])
    rows_to_append = []
    saved_count = 0

    for day in daily_summaries:
        target_date = day.get("date")
        lead_days = day.get("day_index", 0)
        snap_id = f"SNAP-{now_wib.strftime('%Y%m%d')}-H{lead_days}"

        # Jika bukan force_new dan sudah ada snapshot tanggal target dari hari ini, lewati agar tidak dobel
        if not force_new and target_date in existing_today_targets:
            continue

        row = [
            snap_id,
            now_str_full,
            target_date,
            day.get("day_name", ""),
            lead_days,
            day.get("risk_level", 0),
            day.get("risk_label", "AMAN"),
            f"{day.get('max_risk_prob', 0)}%",
            day.get("precip_sum_mm", 0.0),
            day.get("cape_max_jkg", 0),
            day.get("wind_gust_max_kt", 0.0),
            day.get("peak_window", "-"),
            "FALSE",
            "-",
            "PENDING"
        ]
        rows_to_append.append(row)
        saved_count += 1

    if rows_to_append:
        ws_snap.append_rows(rows_to_append)

    # Jalankan verifikasi otomatis sekalian
    verify_result = verify_snapshots_with_metar_records(doc=doc, ws_snap=ws_snap)

    return {
        "status": "success",
        "saved_count": saved_count,
        "snapshot_date": now_str_date,
        "spreadsheet_id": SPREADSHEET_ID,
        "worksheet": SHEET_SNAPSHOT_NAME,
        "verification": verify_result
    }


def summarize_observations(rows):
    """
    Meringkas baris Sheet1 per tanggal WIB.
    Kolom: [slot_30min UTC, wind_speed, dew_point, pressure, temperature, bad_weather, raw_metar]
    """
    days = {}
    seen = set()
    for r in rows:
        slot = parse_slot_datetime(str(r[0])) if r and r[0] else None
        if slot is None or slot in seen:
            continue
        seen.add(slot)

        day = days.setdefault(slot.astimezone(WIB).strftime("%Y-%m-%d"), {
            "slots_count": 0, "raw_slots": 0, "max_severity": 0, "adverse_events": []
        })
        day["slots_count"] += 1

        severity, events = 0, []
        raw = str(r[6]).strip() if len(r) > 6 else ""
        if raw:
            day["raw_slots"] += 1
            severity, text = observed_severity(raw)
            if severity:
                events.append(text)
        else:
            # Baris lama tanpa METAR mentah: bad_weather praktis berarti badai petir.
            if str(r[5]).strip() == "1" if len(r) > 5 else False:
                severity = 2
                events.append("BAD WEATHER")
            try:
                wind = float(r[1])
            except (IndexError, ValueError):
                wind = 0.0
            if wind >= 23:
                severity = 2
                events.append(f"HIGH WIND ({wind} KT)")

        day["max_severity"] = max(day["max_severity"], severity)
        for event in events:
            if event not in day["adverse_events"]:
                day["adverse_events"].append(event)
    return days


def decide_verification(predicted_level, day, is_today):
    """
    Hasil verifikasi satu prediksi, atau None bila belum bisa diputuskan.
    Selama hari masih berjalan, hanya kejadian yang sudah nyata yang final.
    """
    actual = day["max_severity"]

    if predicted_level == 0 and actual > 0:
        return "MISS"
    if predicted_level == 1 and actual >= 1:
        return "HIT"
    if predicted_level == 2 and actual == 2:
        return "HIT"
    if is_today:
        return None
    if predicted_level == 2 and actual == 1:
        return "HIT (OVER-WARNING)"
    if day["raw_slots"] < MIN_RAW_SLOTS:
        return "UNVERIFIABLE"
    return "CORRECT NEGATIVE" if predicted_level == 0 else "FALSE ALARM"


def _fetch_recent_metar_summary(ws_metar, recent_rows_count=600):
    """Ringkasan sekitar 12 hari terakhir tanpa membaca 100k+ baris."""
    rows, _ = read_tail_rows(ws_metar, recent_rows_count)
    return summarize_observations(rows)


def verify_snapshots_with_metar_records(doc=None, ws_snap=None):
    """
    Membandingkan baris snapshot PENDING terhadap data aktual METAR di Sheet1
    menggunakan pembacaan cepat (recent rows) dan Verifikasi Bertingkat yang Selaras.
    """
    if doc is None:
        doc = _get_spreadsheet_doc()
    if ws_snap is None:
        ws_snap = _ensure_snapshot_worksheet(doc)

    ws_metar = doc.worksheet(SHEET_METAR_NAME)
    daily_metar_summary = _fetch_recent_metar_summary(ws_metar)

    if not daily_metar_summary:
        return {"verified_count": 0, "message": "Belum ada rekaman METAR terbaru di Sheet1"}

    # Baca baris snapshot
    snapshot_rows = ws_snap.get_all_values()
    if len(snapshot_rows) <= 1:
        return {"verified_count": 0, "message": "Belum ada data snapshot untuk diverifikasi"}

    if snapshot_rows[0][:len(HEADERS)] != HEADERS:
        raise RuntimeError(f"Susunan kolom tab '{SHEET_SNAPSHOT_NAME}' berubah; verifikasi dibatalkan.")
    idx_target_date = HEADERS.index("target_date")
    idx_risk_level = HEADERS.index("predicted_risk_level")
    idx_verified = HEADERS.index("actual_verified")

    updates = []
    today_str = datetime.now(WIB).strftime("%Y-%m-%d")

    for r_idx in range(1, len(snapshot_rows)):
        row = snapshot_rows[r_idx]
        target_date = row[idx_target_date]
        is_verified = str(row[idx_verified]).upper() == "TRUE"

        if target_date in daily_metar_summary and not is_verified:
            metar_info = daily_metar_summary[target_date]
            slots_recorded = metar_info["slots_count"]
            predicted_level = int(row[idx_risk_level]) if str(row[idx_risk_level]).isdigit() else 0

            ver_category = decide_verification(predicted_level, metar_info, target_date == today_str)
            if ver_category is None:
                continue

            if metar_info["adverse_events"]:
                obs_text = f"{'; '.join(metar_info['adverse_events'])} ({slots_recorded} slots)"
            else:
                obs_text = f"NORMAL / CLEAR ({slots_recorded} slots checked)"

            row_num = r_idx + 1
            updates.append({
                "range": f"M{row_num}:O{row_num}",
                "values": [["TRUE", obs_text[:80], ver_category]]
            })

    if updates:
        ws_snap.batch_update(updates)  # RAW, agar "+TSRA" tidak dibaca sebagai rumus

    return {
        "status": "success",
        "verified_count": len(updates),
        "available_metar_dates": len(daily_metar_summary)
    }


def get_all_snapshots(auto_verify=True):
    """
    Mengambil seluruh riwayat snapshot dari tab 'Forecast_7Days_Snapshot'.
    Secara otomatis menjalankan verifikasi untuk tanggal yang sudah lampau.
    """
    global _last_auto_verify
    doc = _get_spreadsheet_doc()
    ws_snap = _ensure_snapshot_worksheet(doc)

    # Otomatis verifikasi tanggal lampau yang belum terverifikasi, paling sering tiap 30 menit
    now = time.monotonic()
    if auto_verify and (_last_auto_verify is None or now - _last_auto_verify >= AUTO_VERIFY_INTERVAL):
        _last_auto_verify = now
        try:
            verify_snapshots_with_metar_records(doc=doc, ws_snap=ws_snap)
        except Exception as e:
            print(f"Auto-verification warning: {e}")

    # UNFORMATTED: pada Sheet berbahasa Indonesia "21,6" terbaca 216 bila diambil sebagai teks.
    records = ws_snap.get_all_records(value_render_option="UNFORMATTED_VALUE")

    # Hitung metrik kontingensi cepat
    hits = sum(1 for r in records if "HIT" in str(r.get("verification_result", "")))
    cn = sum(1 for r in records if r.get("verification_result") == "CORRECT NEGATIVE")
    fa = sum(1 for r in records if r.get("verification_result") == "FALSE ALARM")
    miss = sum(1 for r in records if r.get("verification_result") == "MISS")
    unverifiable = sum(1 for r in records if r.get("verification_result") == "UNVERIFIABLE")
    pending = sum(1 for r in records if r.get("verification_result") == "PENDING")

    total_verified = hits + cn + fa + miss
    pod = hits / (hits + miss) if (hits + miss) > 0 else 0.0
    far = fa / (hits + fa) if (hits + fa) > 0 else 0.0
    csi = hits / (hits + miss + fa) if (hits + miss + fa) > 0 else 0.0
    accuracy = (hits + cn) / total_verified if total_verified > 0 else 0.0

    return {
        "status": "success",
        "total_records": len(records),
        "total_verified": total_verified,
        "metrics": {
            "hit": hits,
            "correct_negative": cn,
            "false_alarm": fa,
            "miss": miss,
            "pending": pending,
            "unverifiable": unverifiable,
            "pod": round(pod * 100, 1),
            "far": round(far * 100, 1),
            "csi": round(csi * 100, 1),
            "accuracy": round(accuracy * 100, 1)
        },
        "records": records[::-1]  # urutkan dari yang terbaru
    }


if __name__ == "__main__":
    print("Testing Snapshot Service...")
    save_res = save_forecast_snapshot(force_new=True)
    print("Save Snapshot Result:", save_res)
    data = get_all_snapshots()
    print("Total Snapshots in GSheet:", data["total_records"])
    print("Metrics:", data["metrics"])
    for r in data["records"][:7]:
        print(f"[{r.get('target_date')}] Lead: +{r.get('lead_time_days')}d | Pred: {r.get('predicted_risk_label')} ({r.get('predicted_max_prob')}) | Verified: {r.get('actual_verified')} -> {r.get('verification_result')}")
