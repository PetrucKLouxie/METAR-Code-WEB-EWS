"""
Layanan Snapshot & Verifikasi Prospek 7 Hari ke Google Sheets
Menyimpan snapshot prakiraan mingguan AI-NWP ke tab 'Forecast_7Days_Snapshot'
pada Spreadsheet 'METAR Record' dan memverifikasinya secara otomatis terhadap
data observasi METAR aktual yang dicatat berkala di 'Sheet1'.
"""

import os
import re
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import gspread

from app import get_credentials, SPREADSHEET_ID
from weekly_nwp_service import compute_ai_nwp_weekly_forecast

SHEET_SNAPSHOT_NAME = "Forecast_7Days_Snapshot"
SHEET_METAR_NAME = "Sheet1"

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


def parse_metar_adverse_weather(raw_metar):
    """
    Menganalisis string raw METAR secara bertingkat:
    Level 0: Normal / Aman
    Level 1: Moderat (Hujan ringan-sedang RA/DZ/SHRA, angin 15-22 kt, awan konvektif CB/TCU) -> Target WASPADA
    Level 2: Ekstrem / Badai (Badai petir TS/TSRA/SQ, hembusan kencang Gusts >=23kt, visibility <3000m) -> Target SIAGA
    """
    raw = str(raw_metar).upper()
    severity_level = 0
    event_descriptions = []

    # 1. Cek Badai Petir & Squall (Level 2: Ekstrem)
    if re.search(r"\b(\+|-)?(TS|TSRA|VCTS|SQ)\b", raw):
        match = re.findall(r"\b(\+|-)?(TS|TSRA|VCTS|SQ)\b", raw)
        events = ["".join(m) for m in match]
        event_descriptions.extend(events)
        severity_level = max(severity_level, 2)

    # 2. Cek Hembusan Angin Kencang / Gusts
    gust_match = re.search(r"G(\d{2,3})KT", raw)
    if gust_match:
        gust_kt = int(gust_match.group(1))
        if gust_kt >= 23:
            event_descriptions.append(f"GUST {gust_kt}KT")
            severity_level = max(severity_level, 2)
        elif gust_kt >= 16:
            event_descriptions.append(f"GUST {gust_kt}KT")
            severity_level = max(severity_level, 1)

    # 3. Cek Visibilitas Rendah
    vis_match = re.search(r"\b(\d{4})\b", raw)
    if vis_match:
        vis_val = int(vis_match.group(1))
        if 0 < vis_val < 3000:
            event_descriptions.append(f"VIS {vis_val}M")
            severity_level = max(severity_level, 2)
        elif 3000 <= vis_val < 5000:
            event_descriptions.append(f"VIS {vis_val}M")
            severity_level = max(severity_level, 1)

    # 4. Cek Hujan & Gerimis Biasa (Level 1: Moderat)
    if re.search(r"\b(\+|-)?(RA|DZ|SHRA)\b", raw) and severity_level < 2:
        match = re.findall(r"\b(\+|-)?(RA|DZ|SHRA)\b", raw)
        events = ["".join(m) for m in match]
        event_descriptions.extend(events)
        severity_level = max(severity_level, 1)

    # 5. Cek Awan Konvektif CB / TCU
    if ("CB" in raw or "TCU" in raw) and severity_level == 0:
        event_descriptions.append("CB/TCU CLOUD")
        severity_level = max(severity_level, 1)

    return severity_level, ", ".join(event_descriptions) if event_descriptions else "NORMAL / CLEAR"


def _fetch_recent_metar_summary(ws_metar, recent_rows_count=600):
    """
    Mengambil 600 baris terakhir dari Sheet1 (sekitar 12 hari terakhir) secara cepat
    tanpa membebani Google Sheets dengan membaca 100k+ baris.
    """
    max_rows = ws_metar.row_count
    chunk_size = 1500
    start_chunk = max(1, max_rows - chunk_size)
    col_a_chunk = ws_metar.get_values(f"A{start_chunk}:A{max_rows}")

    last_offset = 0
    for i, r in enumerate(col_a_chunk):
        if r and r[0].strip():
            last_offset = i
    last_row = start_chunk + last_offset

    start_row = max(2, last_row - recent_rows_count + 1)
    raw_rows = ws_metar.get_values(f"A{start_row}:F{last_row}")

    daily_summary = {}
    for r in raw_rows:
        if not r or not r[0]:
            continue
        slot = str(r[0])
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", slot)
        if not date_match:
            continue
        d_str = date_match.group(1)

        # Kolom Sheet1: [slot_30min, wind_speed, dew_point, pressure, temperature, bad_weather]
        bad_weather_val = str(r[5]).strip() if len(r) >= 6 else "0"
        is_bad = (bad_weather_val == "1")

        try:
            w_spd = float(r[1]) if len(r) > 1 and r[1] != "" else 0.0
        except ValueError:
            w_spd = 0.0

        if d_str not in daily_summary:
            daily_summary[d_str] = {
                "slots_count": 0,
                "max_severity": 0,
                "adverse_events": []
            }

        daily_summary[d_str]["slots_count"] += 1

        if is_bad:
            daily_summary[d_str]["max_severity"] = max(daily_summary[d_str]["max_severity"], 1)
            if "BAD WEATHER (Rain / Thunderstorm)" not in daily_summary[d_str]["adverse_events"]:
                daily_summary[d_str]["adverse_events"].append("BAD WEATHER (Rain / Thunderstorm)")

        if w_spd >= 23:
            daily_summary[d_str]["max_severity"] = max(daily_summary[d_str]["max_severity"], 2)
            event_gust = f"HIGH WIND ({w_spd} KT)"
            if event_gust not in daily_summary[d_str]["adverse_events"]:
                daily_summary[d_str]["adverse_events"].append(event_gust)

    return daily_summary


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

    headers = snapshot_rows[0]
    idx_target_date = headers.index("target_date") if "target_date" in headers else 2
    idx_risk_level = headers.index("predicted_risk_level") if "predicted_risk_level" in headers else 5
    idx_verified = headers.index("actual_verified") if "actual_verified" in headers else 12
    idx_metar_obs = headers.index("actual_metar_observed") if "actual_metar_observed" in headers else 13
    idx_res = headers.index("verification_result") if "verification_result" in headers else 14

    verified_count = 0
    now_wib = datetime.now(timezone(timedelta(hours=7)))
    today_str = now_wib.strftime("%Y-%m-%d")

    for r_idx in range(1, len(snapshot_rows)):
        row = snapshot_rows[r_idx]
        target_date = row[idx_target_date]
        is_verified = str(row[idx_verified]).upper() == "TRUE"

        if target_date in daily_metar_summary and not is_verified:
            metar_info = daily_metar_summary[target_date]
            actual_sev = metar_info["max_severity"]
            slots_recorded = metar_info["slots_count"]
            predicted_level = int(row[idx_risk_level]) if str(row[idx_risk_level]).isdigit() else 0
            is_today = (target_date == today_str)

            # Jika target adalah hari ini (masih berlangsung di WIB):
            # Hanya putuskan jika cuaca buruk yang diprediksi sudah nyata terkonfirmasi terjadi.
            # Jika belum, biarkan PENDING karena hari belum selesai.
            if is_today:
                if predicted_level == 1 and actual_sev < 1:
                    continue
                elif predicted_level == 2 and actual_sev < 2:
                    continue

            # Jika target_date < today_str (hari kemarin/lampau) atau hari ini sudah terkonfirmasi cuaca buruk:
            if predicted_level == 1:
                # Prediksi: WASPADA
                if actual_sev >= 1:
                    ver_category = "HIT"
                else:
                    ver_category = "FALSE ALARM"

            elif predicted_level == 2:
                # Prediksi: SIAGA BADAI
                if actual_sev == 2:
                    ver_category = "HIT"
                elif actual_sev == 1:
                    ver_category = "HIT (OVER-WARNING)"
                else:
                    ver_category = "FALSE ALARM"

            else:
                # Prediksi: AMAN
                if actual_sev == 0:
                    ver_category = "CORRECT NEGATIVE"
                else:
                    ver_category = "MISS"

            if metar_info["adverse_events"]:
                obs_text = f"{'; '.join(metar_info['adverse_events'])} ({slots_recorded} slots)"
            else:
                obs_text = f"NORMAL / CLEAR ({slots_recorded} slots checked)"

            # Update cell via gspread (1-based index)
            row_num = r_idx + 1
            ws_snap.update_cell(row_num, idx_verified + 1, "TRUE")
            ws_snap.update_cell(row_num, idx_metar_obs + 1, obs_text[:80])
            ws_snap.update_cell(row_num, idx_res + 1, ver_category)
            verified_count += 1

    return {
        "status": "success",
        "verified_count": verified_count,
        "available_metar_dates": len(daily_metar_summary)
    }


def get_all_snapshots(auto_verify=True):
    """
    Mengambil seluruh riwayat snapshot dari tab 'Forecast_7Days_Snapshot'.
    Secara otomatis menjalankan verifikasi untuk tanggal yang sudah lampau.
    """
    doc = _get_spreadsheet_doc()
    ws_snap = _ensure_snapshot_worksheet(doc)

    # Otomatis verifikasi tanggal lampau yang belum terverifikasi
    if auto_verify:
        try:
            verify_snapshots_with_metar_records(doc=doc, ws_snap=ws_snap)
        except Exception as e:
            print(f"Auto-verification warning: {e}")

    records = ws_snap.get_all_records()

    # Hitung metrik kontingensi cepat
    hits = sum(1 for r in records if "HIT" in str(r.get("verification_result", "")))
    cn = sum(1 for r in records if r.get("verification_result") == "CORRECT NEGATIVE")
    fa = sum(1 for r in records if r.get("verification_result") == "FALSE ALARM")
    miss = sum(1 for r in records if r.get("verification_result") == "MISS")
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
