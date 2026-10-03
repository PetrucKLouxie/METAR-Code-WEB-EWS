"""Cached model loading and horizon-specific METAR inference."""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib

from metar_features import FEATURE_NAMES, HORIZONS, features_from_recent_records

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
_loaded_models: dict[str, Any] | None = None
_loaded_config: dict[str, Any] | None = None


def load_models(model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    """Load and validate all horizon artifacts once for this process."""
    global _loaded_models, _loaded_config
    config_path = model_dir / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Konfigurasi model tidak ditemukan: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if tuple(config.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Urutan fitur model_config.json tidak sesuai versi inferensi.")

    models: dict[str, Any] = {}
    for horizon in HORIZONS:
        artifact_name = config.get("horizons", {}).get(horizon, {}).get("artifact")
        if not artifact_name:
            raise ValueError(f"Artifact untuk horizon {horizon} tidak dikonfigurasi.")
        artifact_path = model_dir / artifact_name
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact model tidak ditemukan: {artifact_path}")
        models[horizon] = joblib.load(artifact_path)

    _loaded_models = models
    _loaded_config = config
    return config


def models_loaded() -> bool:
    return _loaded_models is not None and _loaded_config is not None


def predict_horizons(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not models_loaded():
        raise RuntimeError("Model belum dimuat. Jalankan load_models() saat startup.")
    assert _loaded_models is not None
    assert _loaded_config is not None

    observed_at, features, current = features_from_recent_records(records)
    predictions = []
    for horizon, steps in HORIZONS.items():
        horizon_config = _loaded_config["horizons"][horizon]
        model = _loaded_models[horizon]
        probability_values = model.predict_proba(
            [[features[name] for name in FEATURE_NAMES]]
        )[0]
        positive_index = list(model.classes_).index(1)
        probability = float(probability_values[positive_index])
        threshold = float(horizon_config["threshold"])
        if not math.isfinite(probability) or not math.isfinite(threshold):
            raise ValueError(f"Probabilitas/threshold horizon {horizon} tidak valid.")
        is_hazard = probability >= threshold
        target_time = observed_at + timedelta(minutes=30 * steps)
        predictions.append(
            {
                "horizon": horizon,
                "horizon_hours": int(horizon[:-1]),
                "target_time_utc": target_time.isoformat(),
                "probability": probability,
                "prob_percent": round(probability * 100, 2),
                "threshold": threshold,
                "threshold_percent": round(threshold * 100, 2),
                "is_hazard": is_hazard,
                "warning_alert": is_hazard,
            }
        )

    return {
        "status": "success",
        "latest_metar_time": observed_at.isoformat(),
        "current_conditions": current,
        "forecasts": predictions,
        "generated_at_utc": None,
        "persistence": {"saved": False, "sheet": None},
    }
