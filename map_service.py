"""
Medan cuaca ECMWF IFS (via Open-Meteo) untuk peta Juanda dan sekitarnya.
Keluaran model numerik per jam, 15 jam ke belakang sampai 25 jam ke depan; bukan pengamatan.
"""

import json
import tempfile
import time
import urllib.request
from pathlib import Path

BBOX = [111.6, 114.0, -8.1, -6.7]  # barat, timur, selatan, utara
NX, NY = 13, 8
SITE = (-7.3798, 112.7876)
FIELDS = {"t": ("temperature_2m", 1), "rh": ("relative_humidity_2m", 0), "s": ("wind_speed_10m", 1),
          "d": ("wind_direction_10m", 0), "p": ("precipitation", 1), "c": ("cape", 0)}
CACHE_FILE = Path(tempfile.gettempdir()) / "warr_map_ecmwf_ifs.json"
CACHE_TTL_SECONDS = 3600
MAX_STALE_SECONDS = 6 * 3600


def _points():
    w, e, s, n = BBOX
    grid = [(round(s + (n - s) * j / (NY - 1), 3), round(w + (e - w) * i / (NX - 1), 3)) for j in range(NY) for i in range(NX)]
    return grid + [SITE]


def shape(payload):
    if not isinstance(payload, list) or len(payload) != NX * NY + 1:
        raise ValueError("Jumlah titik dari Open-Meteo tidak sesuai.")

    def column(hourly, name, digits):
        values = [None if v is None else round(v, digits) for v in hourly[name]]
        return [v if v is None or digits else int(v) for v in values]

    points = [{k: column(q["hourly"], *FIELDS[k]) for k in FIELDS} for q in payload]
    times = payload[0]["hourly"]["time"]
    return {"model": "ECMWF IFS", "source": "Open-Meteo", "bbox": BBOX, "nx": NX, "ny": NY, "t0": times[0], "n": len(times),
            "pts": points[:-1], "site": points[-1]}


def get_map():
    now = time.time()
    cached = None
    try:
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if now - cached["_cached_at"] < CACHE_TTL_SECONDS:
            return cached["data"]
    except (OSError, ValueError, KeyError):
        cached = None
    points = _points()
    url = ("https://api.open-meteo.com/v1/forecast?latitude=" + ",".join(str(p[0]) for p in points)
           + "&longitude=" + ",".join(str(p[1]) for p in points)
           + "&hourly=" + ",".join(name for name, _ in FIELDS.values())
           + "&wind_speed_unit=kn&timezone=GMT&past_hours=16&forecast_hours=27&models=ecmwf_ifs")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "weather-ews/1.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = shape(json.loads(response.read().decode("utf-8")))
    except Exception:
        if cached and now - cached["_cached_at"] < MAX_STALE_SECONDS:
            return cached["data"]
        raise
    try:
        CACHE_FILE.write_text(json.dumps({"_cached_at": now, "data": data}, separators=(",", ":")), encoding="utf-8")
    except OSError:
        pass
    return data
