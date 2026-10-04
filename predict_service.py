import json
import math
import re
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PURE_MODEL_PATH = BASE_DIR / "saved_models_pure_math" / "model_trees_pure_math.json"
with PURE_MODEL_PATH.open("r", encoding="utf-8") as model_file:
    MODEL_CONFIG = json.load(model_file)

FEATURE_NAMES = MODEL_CONFIG["feature_names"]
HORIZONS = MODEL_CONFIG["horizons"]
_MODEL_CACHE = {}
_MODEL_ERRORS = {}


def _parse_temperature(value):
    return -float(value[1:]) if value.startswith("M") else float(value)


def _metar_timestamp(raw_metar):
    match = re.search(r"\b(\d{2})(\d{2})(\d{2})Z\b", raw_metar)
    now = datetime.now(timezone.utc)
    if not match:
        return now.replace(second=0, microsecond=0)

    day, hour, minute = map(int, match.groups())
    candidates = []
    for month_offset in (-1, 0, 1):
        month_index = now.year * 12 + now.month - 1 + month_offset
        year, month_zero = divmod(month_index, 12)
        month = month_zero + 1
        try:
            candidates.append(datetime(year, month, day, hour, minute, tzinfo=timezone.utc))
        except ValueError:
            continue
    if not candidates:
        raise ValueError(f"Waktu METAR tidak valid: {match.group(0)}")
    return min(candidates, key=lambda value: abs((value - now).total_seconds()))


def parse_raw_metar(raw_metar):
    if not isinstance(raw_metar, str) or not raw_metar.strip() or "NIL" in raw_metar.upper():
        raise ValueError("Setiap baris harus berisi METAR yang valid, bukan kosong atau NIL.")

    wind_match = re.search(r"(?:\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT\b", raw_metar)
    temp_match = re.search(r"\b(M?\d{2})/(M?\d{2})\b", raw_metar)
    pressure_match = re.search(r"\bQ(\d{4})\b", raw_metar)
    if not (wind_match and temp_match and pressure_match):
        raise ValueError(f"METAR tidak memuat suhu/titik embun, QNH, dan angin dalam KT: {raw_metar}")

    return {
        "timestamp": _metar_timestamp(raw_metar),
        "raw_metar": raw_metar.strip(),
        "wind_speed": float(wind_match.group(1)),
        "temperature": _parse_temperature(temp_match.group(1)),
        "dew_point": _parse_temperature(temp_match.group(2)),
        "pressure": float(pressure_match.group(1)),
    }


def _normalize_observation(record):
    raw_metar = record.get("raw_metar")
    if raw_metar:
        try:
            parsed = parse_raw_metar(raw_metar)
            timestamp = record.get("timestamp") or record.get("slot_30min")
            if timestamp:
                parsed["timestamp"] = _parse_timestamp(timestamp)
            return parsed
        except ValueError:
            pass

    timestamp = record.get("timestamp") or record.get("slot_30min")
    wind = _number(record.get("wind_speed"))
    return {
        "timestamp": _parse_timestamp(timestamp) if timestamp else None,
        "wind_speed": wind if wind is not None else 0.0,
        "temperature": _number(record.get("temperature", record.get("temp"))),
        "dew_point": _number(record.get("dew_point")),
        "pressure": _number(record.get("pressure")),
        "raw_metar": raw_metar,
    }


def _number(value):
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value):
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            parsed = datetime.strptime(str(value).strip(), "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _saturation_vapor_pressure(temp_c):
    return 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))


def _observation_at_lag(observations, minutes):
    target = observations[-1]["timestamp"] - timedelta(minutes=minutes)
    valid_obs = [obs for obs in observations[:-1] if obs.get("timestamp")]
    if valid_obs:
        closest = min(valid_obs, key=lambda item: abs(item["timestamp"] - target))
        if abs(closest["timestamp"] - target) <= timedelta(minutes=45):
            return closest
    steps = max(1, round(minutes / 30))
    if len(observations) > steps:
        return observations[-1 - steps]
    return observations[0]


def _load_model(horizon):
    if horizon in _MODEL_CACHE:
        return _MODEL_CACHE[horizon]
    if horizon in _MODEL_ERRORS:
        return None

    model = HORIZONS.get(horizon)
    if not model or not model.get("trees"):
        return None
    _MODEL_CACHE[horizon] = model
    return _MODEL_CACHE[horizon]


def _evaluate_tree_node(tree, feature_values):
    node = tree
    while "leaf" not in node:
        value = feature_values[node["split"]]
        if math.isnan(value):
            child_id = node["missing"]
        else:
            threshold = struct.unpack("<f", struct.pack("<f", node["split_condition"]))[0]
            child_id = node["yes"] if value < threshold else node["no"]
        node = next(child for child in node["children"] if child["nodeid"] == child_id)
    return float(node["leaf"])


def _predict_horizon_probability(model, features):
    float32_values = [struct.unpack("<f", struct.pack("<f", value))[0] for value in features]
    feature_values = dict(zip(FEATURE_NAMES, float32_values))
    margin = float(model["base_score"])
    for tree in model["trees"]:
        margin += _evaluate_tree_node(tree, feature_values)

    if margin >= 0.0:
        raw_probability = 1.0 / (1.0 + math.exp(-margin))
    else:
        exp_margin = math.exp(margin)
        raw_probability = exp_margin / (1.0 + exp_margin)

    calibration = model["platt_scaling"]
    calibrated_score = calibration["A"] * raw_probability + calibration["B"]
    return 1.0 / (1.0 + math.exp(max(-700.0, min(700.0, calibrated_score))))


def _feature_vector(observations):
    latest = observations[-1]
    temp = latest["temperature"]
    dew_point = latest["dew_point"]
    pressure = latest["pressure"]
    wind_knots = latest["wind_speed"]
    if dew_point > temp:
        raise ValueError("Titik embun tidak boleh lebih tinggi dari suhu udara.")

    es_temp = _saturation_vapor_pressure(temp)
    es_dew = _saturation_vapor_pressure(dew_point)
    relative_humidity = min(100.0, max(0.0, 100.0 * es_dew / es_temp))
    vpd = max(0.0, es_temp - es_dew)
    air_density = pressure * 100.0 / (287.058 * (temp + 273.15))
    wind_mps = wind_knots * 0.514444
    timestamp = latest["timestamp"] or datetime.now(timezone.utc)
    hour = timestamp.hour + timestamp.minute / 60.0
    three_hour_lag = _observation_at_lag(observations, 180)
    one_hour_lag = _observation_at_lag(observations, 60)

    features = {
        "temperature": temp,
        "dew_point": dew_point,
        "dpd": temp - dew_point,
        "pressure": pressure,
        "wind_speed": wind_knots,
        "relative_humidity": relative_humidity,
        "vpd": vpd,
        "air_density": air_density,
        "wind_energy_proxy": 0.5 * air_density * wind_mps ** 2,
        "pressure_tendency_3h": pressure - three_hour_lag["pressure"],
        "temp_tendency_1h": temp - one_hour_lag["temperature"],
        "wind_acceleration_1h": wind_knots - one_hour_lag["wind_speed"],
        "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
    }
    return [features[name] for name in FEATURE_NAMES], features


def _status(probability, thresholds):
    if probability >= thresholds["th_bahaya"]:
        return {"level": 2, "label": "BAHAYA", "color": "red"}
    if probability >= thresholds["th_waspada"]:
        return {"level": 1, "label": "WASPADA", "color": "amber"}
    return {"level": 0, "label": "AMAN", "color": "green"}


def _recommendation(level):
    if level == 2:
        return "Aktifkan koordinasi cuaca berbahaya; tinjau kesiapan operasi dan ikuti advis resmi."
    if level == 1:
        return "Tingkatkan pemantauan METAR/SPECI dan siapkan koordinasi operasional."
    return "Operasi normal sesuai prosedur; lanjutkan pemantauan rutin."


def predict_from_observations(records):
    observations = [_normalize_observation(record) for record in records]
    observations = [item for item in observations if all(
        item.get(key) is not None for key in ("wind_speed", "temperature", "dew_point", "pressure")
    )]
    if len(observations) < 7:
        raise ValueError("Diperlukan minimal 7 observasi METAR valid berinterval 30 menit untuk fitur lag 3 jam.")
    if any(item["timestamp"] is None for item in observations):
        raise ValueError("Setiap observasi harus memiliki timestamp UTC.")
    observations.sort(key=lambda item: item["timestamp"])
    vector, features = _feature_vector(observations)

    current_time = observations[-1]["timestamp"]
    predictions = []
    for horizon, config in HORIZONS.items():
        model = _load_model(horizon)
        if model is None:
            reason = "gagal dimuat" if horizon in _MODEL_ERRORS else "tidak tersedia"
            predictions.append({
                "horizon": horizon,
                "available": False,
                "message": f"Artefak model +{horizon} {reason}.",
            })
            continue

        try:
            probability = _predict_horizon_probability(model, vector)
        except Exception as error:
            print(f"Inferensi model +{horizon} gagal: {error!r}", flush=True)
            predictions.append({
                "horizon": horizon,
                "available": False,
                "message": f"Inferensi model +{horizon} gagal.",
            })
            continue
        thresholds = model["thresholds"]
        status = _status(probability, thresholds)
        target_utc = current_time + timedelta(minutes=model["lead_time_minutes"])
        target_wib = target_utc + timedelta(hours=7)
        predictions.append({
            "horizon": horizon,
            "available": True,
            "probability": probability,
            "thresholds": thresholds,
            "status": status,
            "recommendation": _recommendation(status["level"]),
            "target_utc": target_utc.strftime("%Y-%m-%d %H:%M UTC"),
            "target_wib": target_wib.strftime("%Y-%m-%d %H:%M WIB"),
        })

    if not any(item.get("available") for item in predictions):
        raise RuntimeError("Tidak ada artefak model yang dapat digunakan.")

    return {
        "observations_used": len(observations),
        "observed_at_utc": current_time.strftime("%Y-%m-%d %H:%M UTC"),
        "current": {
            "temperature": features["temperature"],
            "dew_point": features["dew_point"],
            "dpd": features["dpd"],
            "pressure": features["pressure"],
            "wind_speed": features["wind_speed"],
            "pressure_tendency_3h": features["pressure_tendency_3h"],
            "relative_humidity": features["relative_humidity"],
            "vpd": features["vpd"],
            "air_density": features["air_density"],
        },
        "predictions": predictions,
    }


def predict_from_raw_metar_list(raw_metars):
    if not isinstance(raw_metars, list) or len(raw_metars) < 7:
        raise ValueError("Kirim minimal 7 baris METAR mentah yang berurutan.")
    return predict_from_observations([parse_raw_metar(line) for line in raw_metars])


def predict_from_gsheet(records):
    if not isinstance(records, list):
        raise ValueError("Data Google Sheets harus berupa daftar observasi.")
    return predict_from_observations(records)