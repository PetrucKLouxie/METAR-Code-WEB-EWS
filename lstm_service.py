import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models_lstm"
with (MODEL_DIR / "lstm_config.json").open("r", encoding="utf-8") as model_file:
    MODEL_CONFIG = json.load(model_file)
with (MODEL_DIR / "scaler_params.json").open("r", encoding="utf-8") as scaler_file:
    SCALER_PARAMS = json.load(scaler_file)
with (MODEL_DIR / "lstm_weights_pure.json").open("r", encoding="utf-8") as weights_file:
    MODEL_WEIGHTS = json.load(weights_file)

FEATURE_NAMES = MODEL_CONFIG["feature_names"]
HORIZONS = MODEL_CONFIG["horizons"]
LOOKBACK = int(MODEL_CONFIG["lookback_timesteps"])
TEST_SUPPORT_2026 = {
    "1h": (13212, 507),
    "3h": (13208, 507),
    "9h": (13196, 505),
    "18h": (13178, 497),
    "24h": (13166, 497),
}
if SCALER_PARAMS["feature_names"] != FEATURE_NAMES or MODEL_WEIGHTS["feature_names"] != FEATURE_NAMES:
    raise ValueError("Urutan fitur scaler dan bobot LSTM harus sama dengan konfigurasi.")
if int(MODEL_WEIGHTS["lookback"]) != LOOKBACK:
    raise ValueError("Lookback bobot LSTM tidak sama dengan konfigurasi.")


def _sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def get_lstm_public_metadata():
    public_horizons = {}
    for horizon, config in HORIZONS.items():
        metrics = dict(config["metrics_2026"])
        sample_count, positive_count = TEST_SUPPORT_2026[horizon]
        true_positive = round(metrics["pod"] * positive_count)
        false_positive = round(metrics["far"] * true_positive / (1.0 - metrics["far"]))
        true_negative = sample_count - positive_count - false_positive
        false_negative = positive_count - true_positive
        expected_correct = (
            (true_positive + false_negative) * (true_positive + false_positive)
            + (true_negative + false_positive) * (true_negative + false_negative)
        ) / sample_count
        hss_denominator = sample_count - expected_correct
        metrics["hss_estimate"] = (
            ((true_positive + true_negative) - expected_correct) / hss_denominator
            if hss_denominator else 0.0
        )
        public_horizons[horizon] = {
            "lead_time_minutes": config["lead_time_minutes"],
            "thresholds": {
                "th_waspada": config["th_waspada"],
                "th_bahaya": config["th_bahaya"],
            },
            "metrics": metrics,
        }
    return {
        "lookback_timesteps": LOOKBACK,
        "feature_names": FEATURE_NAMES,
        "horizons": public_horizons,
        "hss_note": "Estimasi HSS direkonstruksi dari POD/FAR yang dibulatkan dan dukungan label test 2026.",
    }


def _parse_temperature(value):
    return -float(value[1:]) if value.startswith("M") else float(value)


def _parse_metar_time(raw_metar):
    match = re.search(r"\b(\d{2})(\d{2})(\d{2})Z\b", raw_metar)
    if not match:
        raise ValueError(f"Waktu UTC tidak ditemukan pada METAR: {raw_metar}")

    day, hour, minute = map(int, match.groups())
    now = datetime.now(timezone.utc)
    candidates = []
    for month_offset in (-1, 0, 1):
        month_index = now.year * 12 + now.month - 1 + month_offset
        year, month_zero = divmod(month_index, 12)
        try:
            candidates.append(datetime(year, month_zero + 1, day, hour, minute, tzinfo=timezone.utc))
        except ValueError:
            continue
    if not candidates:
        raise ValueError(f"Tanggal METAR tidak valid: {match.group(0)}")
    return min(candidates, key=lambda value: abs((value - now).total_seconds()))


def parse_lstm_metar(raw_metar):
    if not isinstance(raw_metar, str) or not raw_metar.strip() or "NIL" in raw_metar.upper():
        raise ValueError("Setiap baris harus berisi METAR valid, bukan kosong atau NIL.")

    wind_match = re.search(r"(?:\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT\b", raw_metar)
    temp_match = re.search(r"\b(M?\d{2})/(M?\d{2})\b", raw_metar)
    pressure_match = re.search(r"\bQ(\d{4})\b", raw_metar)
    if not (wind_match and temp_match and pressure_match):
        raise ValueError(f"METAR harus memuat suhu/titik embun, QNH, dan angin KT: {raw_metar}")

    return {
        "timestamp": _parse_lstm_timestamp(_parse_metar_time(raw_metar)),
        "wind_speed": float(wind_match.group(1)),
        "temperature": _parse_temperature(temp_match.group(1)),
        "dew_point": _parse_temperature(temp_match.group(2)),
        "pressure": float(pressure_match.group(1)),
    }


def _parse_lstm_timestamp(value):
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


def _number(value, field_name, default=None):
    if value is None or value == "":
        if default is not None:
            return float(default)
        raise ValueError(f"Nilai {field_name} pada observasi LSTM tidak boleh kosong.")
    try:
        return float(value)
    except (TypeError, ValueError):
        if default is not None:
            return float(default)
        raise ValueError(f"Nilai {field_name} pada observasi LSTM tidak valid.") from None


def _normalize_observation(record):
    raw_metar = record.get("raw_metar") or record.get("raw")
    if raw_metar:
        try:
            parsed = parse_lstm_metar(raw_metar)
            timestamp = record.get("timestamp") or record.get("slot_30min")
            if timestamp:
                parsed["timestamp"] = _parse_lstm_timestamp(timestamp)
            return parsed
        except ValueError:
            pass

    timestamp = record.get("timestamp") or record.get("slot_30min")
    if not timestamp:
        raise ValueError("Setiap observasi LSTM harus memiliki timestamp UTC.")
    return {
        "timestamp": _parse_lstm_timestamp(timestamp),
        "wind_speed": _number(record.get("wind_speed"), "wind_speed", default=0.0),
        "temperature": _number(record.get("temperature", record.get("temp")), "temperature"),
        "dew_point": _number(record.get("dew_point"), "dew_point"),
        "pressure": _number(record.get("pressure"), "pressure"),
    }


def _feature_row(observations, index):
    current = observations[index]
    temp = current["temperature"]
    dew_point = current["dew_point"]
    pressure = current["pressure"]
    wind_knots = current["wind_speed"]
    if dew_point > temp:
        raise ValueError("Titik embun tidak boleh lebih tinggi dari suhu udara.")

    es_temp = 6.112 * math.exp((17.67 * temp) / (temp + 243.5))
    es_dew = 6.112 * math.exp((17.67 * dew_point) / (dew_point + 243.5))
    relative_humidity = min(100.0, max(0.0, 100.0 * es_dew / es_temp))
    vpd = max(0.0, es_temp - es_dew)
    air_density = pressure * 100.0 / (287.058 * (temp + 273.15))
    wind_mps = wind_knots * 0.514444
    timestamp = current["timestamp"]
    hour = timestamp.hour + timestamp.minute / 60.0

    pressure_lag = observations[index - 6]["pressure"] if index >= 6 else pressure
    temp_lag = observations[index - 2]["temperature"] if index >= 2 else temp
    wind_lag = observations[index - 2]["wind_speed"] if index >= 2 else wind_knots
    values = {
        "temperature": temp,
        "dew_point": dew_point,
        "dpd": temp - dew_point,
        "pressure": pressure,
        "wind_speed": wind_knots,
        "relative_humidity": relative_humidity,
        "vpd": vpd,
        "air_density": air_density,
        "wind_energy_proxy": 0.5 * air_density * wind_mps ** 2,
        "pressure_tendency_3h": pressure - pressure_lag,
        "temp_tendency_1h": temp - temp_lag,
        "wind_acceleration_1h": wind_knots - wind_lag,
        "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
    }
    return [values[name] for name in FEATURE_NAMES], values


def _matvec(vector, matrix, bias):
    output = list(bias)
    for value, row in zip(vector, matrix):
        if value == 0.0:
            continue
        for index, weight in enumerate(row):
            output[index] += value * weight
    return output


def _run_lstm(sequence, layer):
    kernel = layer["kernel"]
    recurrent_kernel = layer["recurrent_kernel"]
    bias = layer["bias"]
    units = len(recurrent_kernel)
    hidden = [0.0] * units
    cell = [0.0] * units
    output_sequence = []

    for inputs in sequence:
        gates = _matvec(inputs, kernel, bias)
        recurrent = _matvec(hidden, recurrent_kernel, [0.0] * len(bias))
        gates = [left + right for left, right in zip(gates, recurrent)]
        input_gate = [_sigmoid(value) for value in gates[:units]]
        forget_gate = [_sigmoid(value) for value in gates[units:2 * units]]
        candidate = [math.tanh(value) for value in gates[2 * units:3 * units]]
        output_gate = [_sigmoid(value) for value in gates[3 * units:4 * units]]
        cell = [forget * previous + incoming * proposed for forget, previous, incoming, proposed in zip(
            forget_gate, cell, input_gate, candidate
        )]
        hidden = [gate * math.tanh(state) for gate, state in zip(output_gate, cell)]
        output_sequence.append(hidden)

    return output_sequence


def _dense(vector, layer, activation):
    output = _matvec(vector, layer["kernel"], layer["bias"])
    if activation == "relu":
        return [max(0.0, value) for value in output]
    if activation == "sigmoid":
        return [_sigmoid(value) for value in output]
    raise ValueError(f"Aktivasi Dense tidak didukung: {activation}")


def _predict_probabilities(sequence):
    layers = MODEL_WEIGHTS["layers"]
    first_sequence = _run_lstm(sequence, layers["lstm_layer_1"])
    second_sequence = _run_lstm(first_sequence, layers["lstm_layer_2"])
    hidden = _dense(second_sequence[-1], layers["dense_dense_1"], "relu")
    return _dense(hidden, layers["dense_output"], "sigmoid")


def predict_lstm_from_observations(records):
    if not isinstance(records, list):
        raise ValueError("Data observasi LSTM harus berupa daftar.")
    observations = [_normalize_observation(record) for record in records]
    observations.sort(key=lambda item: item["timestamp"])
    if len(observations) < LOOKBACK:
        raise ValueError(f"Diperlukan minimal {LOOKBACK} observasi METAR untuk lookback LSTM.")

    # Verifikasi urutan waktu (hanya pastikan monoton naik, tidak crash jika ada gap data internet)
    for previous, current in zip(observations, observations[1:]):
        interval = (current["timestamp"] - previous["timestamp"]).total_seconds()
        if interval <= 0:
            raise ValueError("Observasi LSTM harus memiliki timestamp berurutan maju.")

    feature_rows = [_feature_row(observations, index)[0] for index in range(len(observations))]
    feature_rows = feature_rows[-LOOKBACK:]
    center = SCALER_PARAMS["center"]
    scale = SCALER_PARAMS["scale"]
    sequence = [[(value - center[index]) / scale[index] for index, value in enumerate(row)] for row in feature_rows]
    probabilities = _predict_probabilities(sequence)
    if len(probabilities) != len(HORIZONS):
        raise ValueError("Jumlah output LSTM tidak sama dengan jumlah horizon konfigurasi.")

    latest = observations[-1]
    latest_features = _feature_row(observations, len(observations) - 1)[1]
    target_utc = latest["timestamp"]
    predictions = []
    for probability, (horizon, config) in zip(probabilities, HORIZONS.items()):
        thresholds = {"th_waspada": config["th_waspada"], "th_bahaya": config["th_bahaya"]}
        if probability >= thresholds["th_bahaya"]:
            status = {"level": 2, "label": "BAHAYA", "color": "red"}
            recommendation = "Aktifkan koordinasi cuaca berbahaya; verifikasi advis resmi dan kesiapan operasi."
        elif probability >= thresholds["th_waspada"]:
            status = {"level": 1, "label": "WASPADA", "color": "amber"}
            recommendation = "Tingkatkan pemantauan METAR/SPECI dan siapkan koordinasi operasional."
        else:
            status = {"level": 0, "label": "AMAN", "color": "green"}
            recommendation = "Operasi normal sesuai prosedur; lanjutkan pemantauan rutin."
        target_time = target_utc + timedelta(minutes=config["lead_time_minutes"])
        target_wib = target_time + timedelta(hours=7)
        predictions.append({
            "horizon": horizon,
            "available": True,
            "probability": min(1.0, max(0.0, float(probability))),
            "thresholds": thresholds,
            "status": status,
            "recommendation": recommendation,
            "target_utc": target_time.strftime("%Y-%m-%d %H:%M UTC"),
            "target_wib": target_wib.strftime("%Y-%m-%d %H:%M WIB"),
            "metrics": config["metrics_2026"],
        })

    latest_wib = target_utc + timedelta(hours=7)
    return {
        "observations_used": LOOKBACK,
        "observed_at_utc": target_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "observed_at_wib": latest_wib.strftime("%Y-%m-%d %H:%M WIB"),
        "current": {
            "temperature": latest_features["temperature"],
            "dew_point": latest_features["dew_point"],
            "dpd": latest_features["dpd"],
            "relative_humidity": latest_features["relative_humidity"],
            "pressure": latest_features["pressure"],
            "pressure_tendency_3h": latest_features["pressure_tendency_3h"],
            "wind_speed": latest_features["wind_speed"],
        },
        "predictions": predictions,
    }


def predict_lstm_from_raw_metar_list(raw_metars):
    if not isinstance(raw_metars, list) or len(raw_metars) < LOOKBACK:
        raise ValueError(f"Kirim minimal {LOOKBACK} METAR berurutan untuk lookback LSTM.")
    return predict_lstm_from_observations([parse_lstm_metar(line) for line in raw_metars])


def predict_lstm_from_gsheet(records):
    return predict_lstm_from_observations(records)
