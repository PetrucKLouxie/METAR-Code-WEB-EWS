"""
Tingkat cuaca yang benar-benar diobservasi pada satu METAR, untuk verifikasi.

Berbeda dari kolom `bad_weather` (app.parse_metar), yang merupakan label training
model dan definisinya sengaja tidak diubah. Modul ini hanya menilai bagian
observasi: prakiraan tren (TEMPO/BECMG), remark, dan cuaca lampau (RE..) diabaikan.

Level 0: normal
Level 1: hujan/gerimis, visibilitas 3000-4999 m, gust 16-22 kt, awan CB/TCU
Level 2: badai petir/squall, visibilitas < 3000 m, angin atau gust >= 23 kt
"""

import re

_TREND_OR_REMARK = re.compile(r"\b(?:NOSIG|TEMPO|BECMG|RMK)\b")
_WIND = re.compile(r"(?:\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT")
_STORM = re.compile(r"(?:[+-]|VC)?(?:TS[A-Z]*|SQ|FC)")
_RAIN = re.compile(r"[+-]?(?:SH)?(?:RA|DZ)(?:RA|DZ|BR)*")


def observed_severity(raw_metar):
    """Mengembalikan (level, deskripsi kejadian)."""
    tokens = _TREND_OR_REMARK.split(str(raw_metar).upper(), maxsplit=1)[0].split()
    level = 0
    events = []

    def add(new_level, text):
        nonlocal level
        level = max(level, new_level)
        events.append(text)

    for token in tokens:
        wind = _WIND.fullmatch(token)
        if wind:
            speed, gust = int(wind.group(1)), int(wind.group(2) or 0)
            if max(speed, gust) >= 23:
                add(2, f"GUST {gust}KT" if gust >= 23 else f"WIND {speed}KT")
            elif gust >= 16:
                add(1, f"GUST {gust}KT")
        elif re.fullmatch(r"\d{4}", token):
            if int(token) < 3000:
                add(2, f"VIS {int(token)}M")
            elif int(token) < 5000:
                add(1, f"VIS {int(token)}M")
        elif _STORM.fullmatch(token):
            add(2, token)
        elif _RAIN.fullmatch(token):
            add(1, token)
        elif re.fullmatch(r"(?:FEW|SCT|BKN|OVC)\d{3}(?:CB|TCU)", token):
            add(1, token)

    return level, ", ".join(events) if events else "NORMAL / CLEAR"
