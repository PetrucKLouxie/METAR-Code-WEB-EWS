"""
Catatan prediksi EWS ke tab 'EWS_Log', untuk diuji terhadap kejadian yang belum terjadi saat dicatat.

Satu baris per observasi METAR. Prediksi yang dibuat terlambat tidak dicatat, supaya data uji hanya
berisi prediksi yang benar-benar mendahului kejadian.
"""

from datetime import datetime, timezone

import ews_service

SHEET_NAME = "EWS_Log"
MAX_DELAY_MINUTES = 45
_H = list(ews_service.LEAD_DAY)
HEADERS = (
    ["observed_utc", "logged_utc", "delay_min", "model_version"]
    + [f"{name}_{h}" for h in _H for name in ("model", "p", "level", "lstm")]
)


def _delay(result, now):
    observed = datetime.strptime(result["observed_at_utc"], "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
    return int((now - observed).total_seconds() // 60)


def build_row(result, now):
    row = [result["observed_at_utc"].replace(" UTC", ""), now.strftime("%Y-%m-%d %H:%M"), _delay(result, now),
           ews_service.MODEL_VERSION]
    by_horizon = {p["horizon"]: p for p in result["predictions"]}
    for h in _H:
        p = by_horizon[h]
        lstm = p["lstm"]["probability"]
        row += [p["model"], p["hybrid_probability"], p["status"]["level"], "" if lstm is None else lstm]
    return row


def append(doc, result, now):
    """Menambah satu baris; mengembalikan False bila prediksi terlambat untuk dicatat."""
    if not 0 <= _delay(result, now) <= MAX_DELAY_MINUTES:
        return False
    if SHEET_NAME in [w.title for w in doc.worksheets()]:
        tab = doc.worksheet(SHEET_NAME)
        if tab.row_values(1) != HEADERS:
            raise RuntimeError(f"Susunan kolom tab '{SHEET_NAME}' berbeda; prediksi tidak dicatat.")
    else:
        tab = doc.add_worksheet(title=SHEET_NAME, rows=2000, cols=len(HEADERS))
        tab.append_row(HEADERS)
    tab.append_row(build_row(result, now))
    return True
