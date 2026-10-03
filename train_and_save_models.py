"""Train five direct, horizon-specific XGBoost METAR classifiers.

The target is the existing ``bad_weather`` field at the future observation
time. Google Sheets credentials are read from GOOGLE_CREDENTIALS_JSON or the
local credentials.json file; a CSV with the same METAR columns may be supplied
with --csv for reproducible offline runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, fbeta_score
from xgboost import XGBClassifier

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

from metar_features import (
    FEATURE_NAMES,
    HORIZONS,
    MONOTONE_CONSTRAINTS,
    feature_rows_for_history,
    parse_observation_time,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "saved_models"
DEFAULT_SHEET_NAME = "METAR Record"
DEFAULT_SPREADSHEET_ID = "1MWaA3zgLxFsDx3ziPJB02Jb5XkZA5P40--6OoUyH4sw"
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_COLUMNS = (
    "slot_30min",
    "wind_speed",
    "dew_point",
    "pressure",
    "temperature",
    "bad_weather",
)


def _as_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def normalize_sheet_values(rows: list[list[str]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Worksheet data METAR kosong.")

    first_row = [cell.strip().lower() for cell in rows[0]]
    has_header = any(name in first_row for name in ("slot_30min", "timestamp"))
    if has_header:
        column_indexes = {name: index for index, name in enumerate(first_row)}
        time_column = column_indexes.get("slot_30min", column_indexes.get("timestamp"))
        required = {"wind_speed", "dew_point", "pressure", "temperature", "bad_weather"}
        missing = required - column_indexes.keys()
        if time_column is None or missing:
            raise ValueError(f"Header worksheet tidak lengkap; kolom hilang: {sorted(missing)}")
        indexes = {
            "slot_30min": time_column,
            **{name: column_indexes[name] for name in required},
        }
        data_rows = rows[1:]
    else:
        indexes = {name: index for index, name in enumerate(SHEET_COLUMNS)}
        data_rows = rows[1:]

    records = []
    for row_number, row in enumerate(data_rows, start=2):
        if len(row) <= max(indexes.values()) or not row[indexes["slot_30min"]].strip():
            continue
        timestamp = row[indexes["slot_30min"]].strip()
        try:
            parsed_time = parse_observation_time(timestamp)
        except ValueError:
            continue
        record: dict[str, Any] = {
            "slot_30min": parsed_time,
            "source_row": row_number,
        }
        for name in ("wind_speed", "dew_point", "pressure", "temperature"):
            record[name] = _as_float(row[indexes[name]])
        label_value = _as_float(row[indexes["bad_weather"]])
        record["bad_weather"] = (
            int(label_value) if label_value is not None and label_value in (0, 1) else None
        )
        records.append(record)

    records.sort(key=lambda record: record["slot_30min"])
    deduplicated: dict[datetime, dict[str, Any]] = {}
    for record in records:
        if record["slot_30min"] in deduplicated:
            raise ValueError(
                f"Timestamp duplikat pada dataset: {record['slot_30min'].isoformat()}"
            )
        deduplicated[record["slot_30min"]] = record

    if len(records) < 7:
        raise ValueError(f"Dataset hanya berisi {len(records)} observasi; minimal 7 diperlukan.")
    return records


def load_google_sheet_records() -> list[dict[str, Any]]:
    credentials_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if credentials_json:
        from google.oauth2.service_account import Credentials

        credentials = Credentials.from_service_account_info(
            json.loads(credentials_json), scopes=GOOGLE_SCOPES
        )
    else:
        credentials_path = BASE_DIR / "credentials.json"
        if not credentials_path.is_file():
            raise FileNotFoundError(
                "Google credentials tidak ditemukan. Atur GOOGLE_CREDENTIALS_JSON, "
                "letakkan credentials.json lokal, atau gunakan --csv."
            )
        from google.oauth2.service_account import Credentials

        credentials = Credentials.from_service_account_file(
            credentials_path, scopes=GOOGLE_SCOPES
        )

    import gspread

    spreadsheet_id = os.environ.get("SPREADSHEET_ID", DEFAULT_SPREADSHEET_ID)
    sheet_name = os.environ.get("SHEET_NAME", DEFAULT_SHEET_NAME)
    client = gspread.authorize(credentials)
    worksheet = client.open_by_key(spreadsheet_id).worksheet(sheet_name)
    return normalize_sheet_values(worksheet.get_all_values())


def _future_target(
    records_by_time: dict[datetime, dict[str, Any]],
    source_time: datetime,
    lead_steps: int,
) -> tuple[datetime, int | None] | None:
    target_time = source_time + timedelta(minutes=30 * lead_steps)
    target = records_by_time.get(target_time)
    if target is None:
        return None
    return target_time, target["bad_weather"]


def _f2_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    return float(
        fbeta_score(labels, probabilities >= threshold, beta=2, zero_division=0)
    )


def _best_f2_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scores = np.array([_f2_at_threshold(labels, probabilities, t) for t in candidates])
    best_score = float(scores.max())
    best_threshold = float(candidates[np.flatnonzero(scores == best_score)[0]])
    return best_threshold, best_score


def _make_prefit_calibrator(model: XGBClassifier) -> CalibratedClassifierCV:
    if FrozenEstimator is not None:
        return CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method="sigmoid",
        )
    return CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv="prefit",
    )


def _collect_horizon_samples(
    records: list[dict[str, Any]],
    feature_rows: list[tuple[datetime, dict[str, float] | None]],
    lead_steps: int,
    now: datetime,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    records_by_time = {record["slot_30min"]: record for record in records}
    samples: dict[str, list[tuple[list[float], int]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    cutoff_validation = datetime(2025, 1, 1, tzinfo=timezone.utc)
    cutoff_test = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for source_time, features in feature_rows:
        if features is None:
            continue
        future = _future_target(records_by_time, source_time, lead_steps)
        if future is None:
            continue
        target_time, label = future
        if label not in (0, 1):
            continue
        feature_vector = [features[name] for name in FEATURE_NAMES]
        if source_time < cutoff_validation and target_time < cutoff_validation:
            partition = "train"
        elif (
            cutoff_validation <= source_time < cutoff_test
            and cutoff_validation <= target_time < cutoff_test
        ):
            partition = "validation"
        elif (
            cutoff_test <= source_time <= now
            and cutoff_test <= target_time <= now
        ):
            partition = "test"
        else:
            continue
        samples[partition].append((feature_vector, int(label)))

    return {
        partition: (
            np.asarray([sample[0] for sample in values], dtype=float).reshape(
                (-1, len(FEATURE_NAMES))
            ),
            np.asarray([sample[1] for sample in values], dtype=int),
        )
        for partition, values in samples.items()
    }


def _atomic_joblib_dump(model: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".joblib.tmp", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        joblib.dump(model, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def train_and_save_models(
    records: list[dict[str, Any]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Tidak ada data METAR untuk training.")

    now = datetime.now(timezone.utc)
    feature_rows = feature_rows_for_history(records)
    config: dict[str, Any] = {
        "model_type": "direct_horizon_xgboost",
        "label_column": "bad_weather",
        "label_definition": "Parser rule: TS/TSRA/+RA/SQ/FC atau wind_speed >= 25 kt",
        "feature_names": list(FEATURE_NAMES),
        "monotone_constraints": list(MONOTONE_CONSTRAINTS),
        "validation": {
            "training_years": "2021-2024",
            "validation_year": 2025,
            "test_period": "2026-01-01 through training run date",
            "test_metrics": "Average precision (PR-AUC) and F2 at validation threshold",
        },
        "horizons": {},
        "trained_at_utc": now.isoformat(),
        "data_rows": len(records),
    }

    for horizon, lead_steps in HORIZONS.items():
        partitions = _collect_horizon_samples(records, feature_rows, lead_steps, now)
        train_x, train_y = partitions["train"]
        validation_x, validation_y = partitions["validation"]
        test_x, test_y = partitions["test"]
        if len(train_y) == 0 or np.unique(train_y).size != 2:
            raise ValueError(
                f"Horizon {horizon}: training 2021-2024 harus memiliki kelas 0 dan 1; "
                f"jumlah={len(train_y)}, positif={int(train_y.sum()) if len(train_y) else 0}."
            )
        if len(validation_y) == 0 or np.unique(validation_y).size != 2:
            raise ValueError(
                f"Horizon {horizon}: validasi kalender 2025 harus memiliki kelas 0 dan 1; "
                f"jumlah={len(validation_y)}, "
                f"positif={int(validation_y.sum()) if len(validation_y) else 0}."
            )

        positives = int(train_y.sum())
        negatives = int(len(train_y) - positives)
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=negatives / positives,
            monotone_constraints=MONOTONE_CONSTRAINTS,
            tree_method="hist",
            n_jobs=1,
            random_state=42,
        )
        model.fit(train_x, train_y)

        calibrator = _make_prefit_calibrator(model)
        calibrator.fit(validation_x, validation_y)
        validation_probabilities = calibrator.predict_proba(validation_x)[:, 1]
        threshold, validation_f2 = _best_f2_threshold(
            validation_y, validation_probabilities
        )

        test_pr_auc = None
        test_f2 = None
        if len(test_y):
            test_probabilities = calibrator.predict_proba(test_x)[:, 1]
            test_f2 = _f2_at_threshold(test_y, test_probabilities, threshold)
            if np.unique(test_y).size == 2:
                test_pr_auc = float(average_precision_score(test_y, test_probabilities))

        artifact_name = f"xgboost_{horizon}.joblib"
        _atomic_joblib_dump(calibrator, output_dir / artifact_name)
        config["horizons"][horizon] = {
            "lead_steps": lead_steps,
            "lead_hours": int(horizon[:-1]),
            "artifact": artifact_name,
            "threshold": threshold,
            "validation_f2": validation_f2,
            "validation_rows": int(len(validation_y)),
            "training_rows": int(len(train_y)),
            "training_positive_rows": positives,
            "scale_pos_weight": negatives / positives,
            "test_rows": int(len(test_y)),
            "test_pr_auc": test_pr_auc,
            "test_f2": test_f2,
            "test_note": (
                None
                if len(test_y) and np.unique(test_y).size == 2
                else "PR-AUC memerlukan kedua kelas; data test yang tersedia belum memadai."
            ),
        }
        print(
            f"{horizon}: train={len(train_y)}, validation={len(validation_y)}, "
            f"test={len(test_y)}, threshold={threshold:.4f}, "
            f"test_PR_AUC={test_pr_auc}, test_F2={test_f2}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "model_config.json"
    temporary_config = output_dir / "model_config.json.tmp"
    temporary_config.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_config.replace(config_path)
    return config


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV harus memiliki baris header.")
        required = {"wind_speed", "dew_point", "pressure", "temperature", "bad_weather"}
        names = {name.strip().lower() for name in reader.fieldnames}
        if not ({"slot_30min", "timestamp"} & names) or not required <= names:
            raise ValueError(
                "CSV memerlukan timestamp/slot_30min, wind_speed, dew_point, "
                "pressure, temperature, dan bad_weather."
            )
        records = []
        for row in reader:
            normalized = {key.strip().lower(): value for key, value in row.items() if key}
            timestamp = normalized.get("slot_30min") or normalized.get("timestamp")
            if not timestamp:
                continue
            parsed = parse_observation_time(timestamp)
            records.append(
                {
                    "slot_30min": parsed,
                    "wind_speed": _as_float(normalized.get("wind_speed")),
                    "dew_point": _as_float(normalized.get("dew_point")),
                    "pressure": _as_float(normalized.get("pressure")),
                    "temperature": _as_float(normalized.get("temperature")),
                    "bad_weather": _as_float(normalized.get("bad_weather")),
                }
            )
    records.sort(key=lambda item: item["slot_30min"])
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV arsip METAR berheader; tanpa opsi ini, baca Google Sheet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Direktori output model (default: saved_models/).",
    )
    args = parser.parse_args()
    records = _read_csv_records(args.csv) if args.csv else load_google_sheet_records()
    print(
        f"Memproses {len(records)} observasi, "
        f"{records[0]['slot_30min'].isoformat()} hingga "
        f"{records[-1]['slot_30min'].isoformat()}."
    )
    config = train_and_save_models(records, args.output_dir)
    print(f"Konfigurasi model tersimpan: {args.output_dir / 'model_config.json'}")
    print(f"Horizon berhasil dilatih: {', '.join(config['horizons'])}")


if __name__ == "__main__":
    main()
