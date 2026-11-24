"""
Peringatan dini 1-24 jam: METAR Juanda + musim + riwayat badai + prakiraan ECMWF IFS.

Model XGBoost dilatih di riset/07_model_ecmwf.py dan dihitung di sini dengan pustaka standar Python.
Prakiraan ECMWF yang dipakai selalu berasal dari run yang sudah terbit: lihat LEAD_DAY.
Bila ECMWF tidak dapat diambil, dipakai model cadangan yang hanya memakai METAR.
"""

import bisect
import json
import math
import struct
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from predict_service import FEATURE_NAMES, _feature_vector, _normalize_observation

with (Path(__file__).resolve().parent / "saved_models_ecmwf" / "model.json").open(encoding="utf-8") as _f:
    BUNDLE = json.load(_f)

LEAD_DAY = BUNDLE["lead_day"]
EC_VARS = ["precipitation", "showers", "temperature_2m", "dew_point_2m", "relative_humidity_2m", "cloud_cover",
           "wind_speed_10m", "wind_direction_10m", "pressure_msl"]
EC_NAMES = ["ec_precipitation", "ec_showers", "ec_temperature_2m", "ec_dew_point_2m", "ec_relative_humidity_2m",
            "ec_cloud_cover", "ec_pressure_msl", "ec_u10", "ec_v10", "ec_precipitation_max7", "ec_showers_max7"]
MAY_BE_MISSING = {"ec_dew_point_2m", "ec_relative_humidity_2m"}
MIN_HISTORY_SLOTS = 48
CACHE_FILE = Path(tempfile.gettempdir()) / f"warr_ews_{BUNDLE['ecmwf_model']}.json"
CACHE_TTL_SECONDS = 1800
MAX_STALE_SECONDS = 6 * 3600
WIB = timezone(timedelta(hours=7))
LEVELS = {0: ("AMAN", "green"), 1: ("WASPADA", "amber"), 2: ("BAHAYA", "red")}
ADVICE = {
    0: "Belum ada tindakan khusus. Tunggu pembaruan berikutnya.",
    1: "Perhatikan perkembangan awan dan informasi BMKG.",
    2: "Potensi badai petir atau angin kencang. Ikuti peringatan resmi BMKG.",
}


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _compile(tree):
    nodes = {}
    stack = [tree]
    while stack:
        node = stack.pop()
        if "leaf" in node:
            nodes[node["nodeid"]] = (node["leaf"],)
        else:
            nodes[node["nodeid"]] = (node["split"], _f32(node["split_condition"]), node["yes"], node["no"], node["missing"])
            stack.extend(node["children"])
    return nodes


_TREES = {}


def probability(kind, horizon, values):
    model = BUNDLE["models"][kind][horizon]
    if (kind, horizon) not in _TREES:
        _TREES[kind, horizon] = [_compile(t) for t in model["trees"]]
    x = {n: _f32(v) for n, v in zip(BUNDLE["feature_order"][kind], values)}
    margin = model["base_margin"]
    for nodes in _TREES[kind, horizon]:
        node = nodes[0]
        while len(node) > 1:
            v = x[node[0]]
            node = nodes[node[4] if math.isnan(v) else node[2] if v < node[1] else node[3]]
        margin += node[0]
    a, b = model["platt"]
    return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, a * margin + b))))


def prepare(records):
    """Observasi lengkap, urut waktu, tanpa slot ganda."""
    by_time = {}
    for record in records:
        obs = _normalize_observation(record)
        if any(obs.get(k) is None for k in ("timestamp", "temperature", "dew_point", "pressure", "wind_speed")):
            continue
        obs["dew_point"] = min(obs["dew_point"], obs["temperature"])
        obs["bad"] = 1 if str(record.get("bad_weather", 0)).strip() in ("1", "1.0", "True") else 0
        by_time[obs["timestamp"]] = obs
    return [by_time[t] for t in sorted(by_time)]


def storm_features(observations):
    now = observations[-1]["timestamp"]
    ages = [(now - o["timestamp"]).total_seconds() / 1800 for o in observations if o["bad"]]
    counts = [sum(1 for a in ages if a < k) for k in (2, 6, 12, 48)]
    return [observations[-1]["bad"], *counts, min(min(ages, default=96), 96)]


def season_features(target):
    a = 2 * math.pi * target.timetuple().tm_yday / 365.25
    b = 2 * math.pi * (target.hour + target.minute / 60) / 24
    return [math.sin(a), math.cos(a), math.sin(b), math.cos(b)]


def fetch_ecmwf():
    """Prakiraan ECMWF yang dibuat 1 dan 2 hari sebelum waktu berlakunya."""
    now = time.time()
    cached = None
    try:
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if now - cached["_cached_at"] < CACHE_TTL_SECONDS:
            return cached["data"]
    except (OSError, ValueError, KeyError):
        cached = None
    hourly = ",".join(f"{v}_previous_day{d}" for d in sorted(set(LEAD_DAY.values())) for v in EC_VARS)
    url = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude=-7.3798&longitude=112.7876&timezone=GMT"
           f"&wind_speed_unit=kn&past_days=1&forecast_days=3&models={BUNDLE['ecmwf_model']}&hourly={hourly}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "weather-ews/1.0"})
        with urllib.request.urlopen(request, timeout=12) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        if cached and now - cached["_cached_at"] < MAX_STALE_SECONDS:
            return cached["data"]
        raise
    try:
        CACHE_FILE.write_text(json.dumps({"_cached_at": now, "data": data}), encoding="utf-8")
    except OSError:
        pass
    return data


class EcmwfTable:
    def __init__(self, payload):
        hourly = payload["hourly"]
        self.times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in hourly["time"]]
        nan = float("nan")
        self.columns = {}
        for day in set(LEAD_DAY.values()):
            c = {v: [nan if x is None else float(x) for x in hourly[f"{v}_previous_day{day}"]] for v in EC_VARS}
            speed, direction = c.pop("wind_speed_10m"), c.pop("wind_direction_10m")
            c["u10"] = [-s * math.sin(math.radians(d)) for s, d in zip(speed, direction)]
            c["v10"] = [-s * math.cos(math.radians(d)) for s, d in zip(speed, direction)]
            for v in ("precipitation", "showers"):
                window = lambda i: [x for x in c[v][max(0, i - 3):i + 4] if not math.isnan(x)]
                c[v + "_max7"] = [max(window(i), default=nan) for i in range(len(c[v]))]
            self.columns[day] = c

    def features(self, day, target):
        i = bisect.bisect_right(self.times, target) - 1
        if i < 0 or i >= len(self.times):
            raise LookupError("Waktu target di luar prakiraan ECMWF.")
        fraction = (target - self.times[i]).total_seconds() / 3600
        if fraction and i + 1 >= len(self.times) or fraction >= 1:
            raise LookupError("Waktu target di luar prakiraan ECMWF.")
        out = []
        for name in EC_NAMES:
            col = self.columns[day][name[3:]]
            value = col[i] if not fraction else col[i] + (col[i + 1] - col[i]) * fraction
            if math.isnan(value) and name not in MAY_BE_MISSING:
                raise LookupError(f"Prakiraan ECMWF belum lengkap: {name}.")
            out.append(value)
        return out


def predict(records):
    observations = prepare(records)
    if len(observations) < MIN_HISTORY_SLOTS:
        raise ValueError(f"Diperlukan minimal {MIN_HISTORY_SLOTS} observasi METAR lengkap (saat ini {len(observations)}).")
    now = observations[-1]["timestamp"]
    vector, current = _feature_vector(observations)
    storm = storm_features(observations)

    table = None
    try:
        table = EcmwfTable(fetch_ecmwf())
    except Exception as error:
        print(f"ECMWF tidak tersedia, memakai model cadangan: {error!r}", flush=True)

    predictions = []
    for horizon, day in LEAD_DAY.items():
        target = now + timedelta(minutes=30 * BUNDLE["models"]["utama"][horizon]["lead_steps"])
        values = vector + season_features(target) + storm
        kind = "cadangan"
        if table is not None:
            try:
                values = values + table.features(day, target)
                kind = "utama"
            except LookupError as error:
                print(f"ECMWF +{horizon}: {error}", flush=True)
        p = probability(kind, horizon, values)
        thresholds = BUNDLE["models"][kind][horizon]["thresholds"]
        level = 2 if p >= thresholds["th_bahaya"] else 1 if p >= thresholds["th_waspada"] else 0
        label, color = LEVELS[level]
        predictions.append({
            "horizon": horizon,
            "available": True,
            "model": "METAR + ECMWF IFS" if kind == "utama" else "METAR saja",
            "probability": p,
            "thresholds": thresholds,
            "status": {"level": level, "label": label, "color": color},
            "recommendation": ADVICE[level],
            "target_utc": target.strftime("%Y-%m-%d %H:%M UTC"),
            "target_wib": target.astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB"),
        })

    top = max(p["status"]["level"] for p in predictions)
    return {
        "status": "success",
        "observed_at_utc": now.strftime("%Y-%m-%d %H:%M UTC"),
        "observed_at_wib": now.astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB"),
        "observations_used": len(observations),
        "ecmwf_available": all(p["model"] != "METAR saja" for p in predictions),
        "overall_status": {"level": top, "label": LEVELS[top][0], "color": LEVELS[top][1]},
        "current": {k: current[k] for k in ("temperature", "dew_point", "dpd", "pressure", "wind_speed",
                                            "pressure_tendency_3h", "relative_humidity")},
        "predictions": predictions,
    }


assert BUNDLE["feature_order"]["utama"][:14] == FEATURE_NAMES
assert BUNDLE["feature_order"]["utama"][-len(EC_NAMES):] == EC_NAMES
