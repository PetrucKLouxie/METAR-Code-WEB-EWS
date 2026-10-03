import json
import math
import re
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
CONFIG_PATH = MODEL_DIR / "model_config.json"
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    MODEL_CONFIG = json.load(config_file)

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
        parsed = parse_raw_metar(raw_metar)
        timestamp = record.get("timestamp") or record.get("slot_30min")
        if timestamp:
            parsed["timestamp"] = _parse_timestamp(timestamp)
        return parsed

    timestamp = record.get("timestamp") or record.get("slot_30min")
    return {
        "timestamp": _parse_timestamp(timestamp) if timestamp else None,
        "wind_speed": _number(record.get("wind_speed")),
        "temperature": _number(record.get("temperature", record.get("temp"))),
        "dew_point": _number(record.get("dew_point")),
        "pressure": _number(record.get("pressure")),
        "raw_metar": None,
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
    closest = min(observations[:-1], key=lambda item: abs(item["timestamp"] - target))
    if abs(closest["timestamp"] - target) > timedelta(minutes=15):
        raise ValueError(f"Riwayat tidak memiliki observasi yang sesuai untuk lag {minutes // 60} jam.")
    return closest


def _load_model(horizon):
    if horizon in _MODEL_CACHE:
        return _MODEL_CACHE[horizon]
    if horizon in _MODEL_ERRORS:
        return None

    model_path = MODEL_DIR / f"xgb_model_{horizon}.json"
    if not model_path.is_file():
        return None

    try:
        with model_path.open("r", encoding="utf-8") as model_file:
            serialized_model = json.load(model_file)

        learner = serialized_model["learner"]
        if learner["objective"]["name"] != "binary:logistic":
            raise ValueError("Model harus menggunakan objective binary:logistic.")

        raw_base_score = learner["learner_model_param"]["base_score"]
        if isinstance(raw_base_score, str):
            parsed_base_score = json.loads(raw_base_score)
            raw_base_score = parsed_base_score[0] if isinstance(parsed_base_score, list) else parsed_base_score
        base_score = float(raw_base_score)
        if not 0.0 < base_score < 1.0:
            raise ValueError("base_score model harus berada di antara 0 dan 1.")

        trees = []
        for tree in learner["gradient_booster"]["model"]["trees"]:
            if any(tree.get("split_type", [])):
                raise ValueError("Model dengan categorical split tidak didukung.")
            trees.append({
                "left": tree["left_children"],
                "right": tree["right_children"],
                "features": tree["split_indices"],
                "conditions": tree["split_conditions"],
                "default_left": tree["default_left"],
            })

        _MODEL_CACHE[horizon] = {
            "base_margin": math.log(base_score / (1.0 - base_score)),
            "trees": trees,
        }
    except Exception as error:
        _MODEL_ERRORS[horizon] = error
        print(f"Model +{horizon} gagal dimuat: {error!r}", flush=True)
        return None
    return _MODEL_CACHE[horizon]


def _predict_booster_probability(model, features):
    float32_features = [struct.unpack("<f", struct.pack("<f", value))[0] for value in features]
    margin = model["base_margin"]
    for tree in model["trees"]:
        node = 0
        while tree["left"][node] != -1:
            value = float32_features[tree["features"][node]]
            if math.isnan(value):
                node = tree["left"][node] if tree["default_left"][node] else tree["right"][node]
            elif value < tree["conditions"][node]:
                node = tree["left"][node]
            else:
                node = tree["right"][node]
        margin += tree["conditions"][node]

    if margin >= 0.0:
        return 1.0 / (1.0 + math.exp(-margin))
    exp_margin = math.exp(margin)
    return exp_margin / (1.0 + exp_margin)


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
    air_density = pressure * 100.0 / (287.05 * (temp + 273.15))
    wind_mps = wind_knots * 0.514444
    timestamp = latest["timestamp"] or datetime.now(timezone.utc)
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
        "wind_energy_proxy": 0.5 * air_density * wind_mps ** 3,
        "pressure_tendency_3h": pressure - three_hour_lag["pressure"],
        "temp_tendency_1h": temp - one_hour_lag["temperature"],
        "wind_acceleration_1h": wind_knots - one_hour_lag["wind_speed"],
        "hour_sin": math.sin(2.0 * math.pi * timestamp.hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * timestamp.hour / 24.0),
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
            probability = _predict_booster_probability(model, vector)
            platt_params = config.get("platt_params")
            if platt_params:
                calibration_score = platt_params["A"] * probability + platt_params["B"]
                probability = 1.0 / (1.0 + math.exp(min(700.0, calibration_score)))
        except Exception as error:
            print(f"Inferensi model +{horizon} gagal: {error!r}", flush=True)
            predictions.append({
                "horizon": horizon,
                "available": False,
                "message": f"Inferensi model +{horizon} gagal.",
            })
            continue
        thresholds = config["thresholds"]
        status = _status(probability, thresholds)
        target_utc = current_time + timedelta(minutes=config["lead_time_minutes"])
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