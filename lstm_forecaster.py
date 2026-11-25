"""
LSTM peramal METAR: suhu, titik embun, QNH, dan kecepatan angin untuk 48 slot (24 jam) dari 24 slot terakhir.
Dilatih di riset/02_lstm.py (2021-2024) dan dihitung di sini dengan pustaka standar Python.
"""

import json
import math
from datetime import timedelta
from pathlib import Path

with (Path(__file__).resolve().parent / "saved_models_ecmwf" / "lstm_forecaster.json").open(encoding="utf-8") as _f:
    NET = json.load(_f)

LOOK = NET["look"]
SLOTS = LOOK + 6  # enam slot tambahan untuk perubahan QNH 3 jam
MAX_CARRY = timedelta(hours=2)
_HALF_HOUR = timedelta(minutes=30)


def grid(observations):
    """
    Nilai [suhu, titik embun, QNH, angin] pada 30 slot berjarak 30 menit yang berakhir di pengamatan terakhir.
    Slot kosong diisi dari pengamatan sebelumnya, seperti saat melatih, tetapi paling lama 2 jam.
    """
    end = observations[-1]["timestamp"]
    rows, i = [], len(observations) - 1
    for k in range(SLOTS):
        slot = end - _HALF_HOUR * k
        while i >= 0 and observations[i]["timestamp"] > slot:
            i -= 1
        if i < 0 or slot - observations[i]["timestamp"] > MAX_CARRY:
            raise LookupError("Riwayat METAR terputus, LSTM tidak dapat meramal.")
        o = observations[i]
        rows.append([o["temperature"], o["dew_point"], o["pressure"], o["wind_speed"]])
    return rows[::-1]


def derive(t, td, p, w, p3, t1, w1, when):
    es = 6.112 * math.exp(17.67 * t / (t + 243.5))
    e = 6.112 * math.exp(17.67 * td / (td + 243.5))
    rho = p * 100.0 / (287.058 * (t + 273.15))
    hour = when.hour + when.minute / 60
    return [t, td, t - td, p, w, min(100.0, max(0.0, 100 * e / es)), max(0.0, es - e), rho, 0.5 * rho * (w * 0.514444) ** 2,
            p - p3, t - t1, w - w1, math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)]


def _sigmoid(x):
    return 1 / (1 + math.exp(-max(-40.0, min(40.0, x))))


def _add_matvec(out, vector, matrix):
    for value, row in zip(vector, matrix):
        if value:
            for j, weight in enumerate(row):
                out[j] += value * weight
    return out


def forecast(rows, now):
    """48 baris [suhu, titik embun, QNH, angin] untuk 30 menit sampai 24 jam ke depan."""
    units = len(NET["R"])
    h, c = [0.0] * units, [0.0] * units
    for k in range(SLOTS - LOOK, SLOTS):
        t, td, p, w = rows[k]
        x = derive(t, td, p, w, rows[k - 6][2], rows[k - 2][0], rows[k - 2][3], now - _HALF_HOUR * (SLOTS - 1 - k))
        z = [(v - m) / s for v, m, s in zip(x, NET["mu"], NET["sd"])]
        g = _add_matvec(_add_matvec(list(NET["b"]), z, NET["K"]), h, NET["R"])
        c = [_sigmoid(g[units + j]) * c[j] + _sigmoid(g[j]) * math.tanh(g[2 * units + j]) for j in range(units)]
        h = [_sigmoid(g[3 * units + j]) * math.tanh(c[j]) for j in range(units)]
    flat = _add_matvec(list(NET["c"]), h, NET["V"])
    out = []
    for k, scale in enumerate(NET["sdd"]):
        t, td, p, w = (last + flat[4 * k + j] * scale[j] for j, last in enumerate(rows[-1]))
        out.append([t, min(td, t), p, max(w, 0.0)])
    return out


def features(rows, ahead, now, steps):
    """14 fitur METAR di waktu target, dari ramalan (slot ke depan) atau pengamatan (slot ke belakang)."""
    def at(k):
        return ahead[k - 1] if k > 0 else rows[k - 1]

    t, td, p, w = at(steps)
    return derive(t, td, p, w, at(steps - 6)[2], at(steps - 2)[0], at(steps - 2)[3], now + _HALF_HOUR * steps)
