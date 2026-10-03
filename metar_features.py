"""Shared feature engineering for METAR XGBoost training and inference."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

FEATURE_NAMES = (
    "temperature",
    "dew_point",
    "pressure",
    "wind_speed",
    "dpd",
    "relative_humidity",
    "pressure_tendency_3h",
    "temperature_tendency_1h",
    "hour_sin",
    "hour_cos",
)

HORIZONS = {
    "1h": 2,
    "3h": 6,
    "9h": 18,
    "18h": 36,
    "24h": 48,
}

MONOTONE_CONSTRAINTS = (0, 0, 0, 1, -1, 1, -1, 0, 0, 0)
OBSERVATION_INTERVAL = timedelta(minutes=30)


def parse_observation_time(value: Any) -> datetime:
    """Parse the timestamp formats emitted by the existing Sheets parser."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Timestamp METAR kosong.")
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            parsed = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                raise ValueError(f"Format timestamp METAR tidak dikenal: {value!r}")
    else:
        raise ValueError(f"Tipe timestamp METAR tidak didukung: {type(value).__name__}")

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite_number(value: Any, field: str) -> float:
    if value is None or value == "":
        raise ValueError(f"Kolom {field} kosong.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Nilai {field} bukan angka: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Nilai {field} harus finite.")
    return number


def saturation_vapor_pressure(temperature_c: float) -> float:
    return 6.112 * math.exp((17.67 * temperature_c) / (temperature_c + 243.5))


def derive_feature_row(
    record: dict[str, Any],
    record_by_time: dict[datetime, dict[str, Any]],
) -> dict[str, float] | None:
    """Derive current and exact-time-lag features; return None when lags are absent."""
    observed_at = parse_observation_time(
        record.get("slot_30min") or record.get("timestamp")
    )
    previous_3h = record_by_time.get(observed_at - timedelta(hours=3))
    previous_1h = record_by_time.get(observed_at - timedelta(hours=1))
    if previous_3h is None or previous_1h is None:
        return None

    temperature = _finite_number(
        record.get("temperature", record.get("temp")), "temperature"
    )
    dew_point = _finite_number(record.get("dew_point"), "dew_point")
    pressure = _finite_number(record.get("pressure"), "pressure")
    wind_speed = _finite_number(record.get("wind_speed"), "wind_speed")
    previous_pressure = _finite_number(previous_3h.get("pressure"), "pressure_t-3h")
    previous_temperature = _finite_number(
        previous_1h.get("temperature", previous_1h.get("temp")),
        "temperature_t-1h",
    )

    es_temperature = saturation_vapor_pressure(temperature)
    es_dew_point = saturation_vapor_pressure(dew_point)
    fractional_hour = (
        observed_at.hour
        + observed_at.minute / 60
        + observed_at.second / 3600
    )
    angle = 2 * math.pi * fractional_hour / 24

    return {
        "temperature": temperature,
        "dew_point": dew_point,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "dpd": temperature - dew_point,
        "relative_humidity": 100 * es_dew_point / es_temperature,
        "pressure_tendency_3h": pressure - previous_pressure,
        "temperature_tendency_1h": temperature - previous_temperature,
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
    }


def feature_rows_for_history(
    records: list[dict[str, Any]],
) -> list[tuple[datetime, dict[str, float] | None]]:
    """Build features by timestamp without bridging missing observation slots."""
    ordered: list[tuple[datetime, dict[str, Any]]] = []
    for record in records:
        timestamp = parse_observation_time(
            record.get("slot_30min") or record.get("timestamp")
        )
        ordered.append((timestamp, record))
    ordered.sort(key=lambda item: item[0])

    record_by_time: dict[datetime, dict[str, Any]] = {}
    for timestamp, record in ordered:
        if timestamp in record_by_time:
            raise ValueError(f"Timestamp METAR duplikat: {timestamp.isoformat()}")
        record_by_time[timestamp] = record

    result = []
    for timestamp, record in ordered:
        try:
            features = derive_feature_row(record, record_by_time)
        except ValueError:
            features = None
        result.append((timestamp, features))
    return result


def features_from_recent_records(
    records: list[dict[str, Any]],
) -> tuple[datetime, dict[str, float], dict[str, float]]:
    """Validate seven regular observations and derive the latest feature vector."""
    if len(records) < 7:
        raise ValueError(
            f"Inferensi membutuhkan sedikitnya 7 observasi METAR; diterima {len(records)}."
        )

    recent = records[-7:]
    timestamps = [
        parse_observation_time(record.get("slot_30min") or record.get("timestamp"))
        for record in recent
    ]
    for earlier, later in zip(timestamps, timestamps[1:]):
        if later - earlier != OBSERVATION_INTERVAL:
            raise ValueError(
                "Tujuh observasi terakhir harus berjarak tepat 30 menit; "
                "terdapat gap atau laporan SPECI dengan interval tidak seragam."
            )

    indexed = dict(zip(timestamps, recent))
    current_time = timestamps[-1]
    feature_values = derive_feature_row(recent[-1], indexed)
    if feature_values is None:
        raise ValueError("Riwayat tidak cukup untuk menghitung tendensi 3 jam dan 1 jam.")
    if tuple(feature_values) != FEATURE_NAMES:
        raise RuntimeError("Urutan fitur hasil rekayasa tidak sesuai konfigurasi model.")

    current = {
        "temperature": feature_values["temperature"],
        "dew_point": feature_values["dew_point"],
        "pressure": feature_values["pressure"],
        "wind_speed": feature_values["wind_speed"],
        "dpd": feature_values["dpd"],
        "relative_humidity": feature_values["relative_humidity"],
        "pressure_tendency_3h": feature_values["pressure_tendency_3h"],
    }
    return current_time, feature_values, current
