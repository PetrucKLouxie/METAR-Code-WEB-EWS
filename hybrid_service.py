"""
Layanan EWS Hybrid (XGBoost + LSTM)
Menggabungkan inferensi tabular physics-informed XGBoost dan sequence memory LSTM
berdasarkan prinsip safety-first ICAO/WMO untuk peringatan dini cuaca ekstrem di Bandara Juanda (WARR).
"""

from datetime import datetime, timezone
import predict_service
import lstm_service

# Horizon waktu prediksi yang didukung
HORIZONS = ["1h", "3h", "9h", "18h", "24h"]

# Bobot model per horizon (XGBoost vs LSTM)
# Horizon pendek (1h, 3h): XGBoost kuat pada tendensi cepat dan threshold fisika instan
# Horizon panjang (9h, 18h, 24h): LSTM menangkap pola siklus diurnal & memori deret waktu
MODEL_WEIGHTS = {
    "1h": {"xgb": 0.55, "lstm": 0.45},
    "3h": {"xgb": 0.50, "lstm": 0.50},
    "9h": {"xgb": 0.40, "lstm": 0.60},
    "18h": {"xgb": 0.35, "lstm": 0.65},
    "24h": {"xgb": 0.35, "lstm": 0.65},
}


def compute_hybrid_predictions(xgb_res: dict, lstm_res: dict) -> dict:
    """
    Menggabungkan hasil XGBoost dan LSTM menjadi keputusan EWS terpadu.
    
    Logika:
    1. Weighted Probability: Gabungan probabilitas terkalibrasi kedua model.
    2. Conservative Safety-First Rule (Standar Penerbangan):
       Jika salah satu model mendeteksi BAHAYA (Level 2), status EWS otomatis BAHAYA.
       Jika salah satu WASPADA (Level 1) dan yang lain AMAN, status EWS WASPADA.
       Hal ini meminimalkan False Negative (kejadian cuaca ekstrem yang terlewat).
    """
    xgb_map = {p["horizon"]: p for p in xgb_res.get("predictions", [])}
    lstm_map = {p["horizon"]: p for p in lstm_res.get("predictions", [])}

    hybrid_predictions = []
    max_level = 0
    overall_status_label = "AMAN"
    overall_color = "green"

    for h in HORIZONS:
        xgb_p = xgb_map.get(h)
        lstm_p = lstm_map.get(h)

        if not xgb_p and not lstm_p:
            continue

        weights = MODEL_WEIGHTS.get(h, {"xgb": 0.5, "lstm": 0.5})

        # Ambil probabilitas masing-masing
        prob_xgb = xgb_p["probability"] if (xgb_p and xgb_p.get("available")) else None
        prob_lstm = lstm_p["probability"] if (lstm_p and lstm_p.get("available")) else None

        if prob_xgb is not None and prob_lstm is not None:
            hybrid_prob = (prob_xgb * weights["xgb"]) + (prob_lstm * weights["lstm"])
        elif prob_xgb is not None:
            hybrid_prob = prob_xgb
        elif prob_lstm is not None:
            hybrid_prob = prob_lstm
        else:
            hybrid_prob = 0.0

        lvl_xgb = xgb_p["status"]["level"] if xgb_p else 0
        lvl_lstm = lstm_p["status"]["level"] if lstm_p else 0

        # Safety-First Consensus: Ambil level risiko paling konservatif (tertinggi)
        consensus_level = max(lvl_xgb, lvl_lstm)

        if consensus_level == 2:
            status_label = "BAHAYA"
            status_color = "red"
            recommendation = (
                "Peringatan Dini Cuaca Ekstrem aktif! Model mendeteksi potensi fenomena cuaca berbahaya (TS/Squall/Visibility Rendah). "
                "Tingkatkan kesiapsiagaan operasi penerbangan dan monitor radar BMKG secara ketat."
            )
        elif consensus_level == 1:
            status_label = "WASPADA"
            status_color = "amber"
            recommendation = (
                "Potensi perburukan cuaca meningkat. Perhatikan kecenderungan tekanan dan perkembangan awan konvektif."
            )
        else:
            status_label = "AMAN"
            status_color = "green"
            recommendation = "Kondisi diprediksi dalam batas normal penerbangan. Lanjutkan pemantauan rutin."

        if consensus_level > max_level:
            max_level = consensus_level
            overall_status_label = status_label
            overall_color = status_color

        target_utc = (xgb_p or lstm_p).get("target_utc", "")
        target_wib = (xgb_p or lstm_p).get("target_wib", "")

        hybrid_predictions.append({
            "horizon": h,
            "hybrid_probability": round(hybrid_prob, 4),
            "status": {
                "level": consensus_level,
                "label": status_label,
                "color": status_color,
            },
            "xgb": {
                "probability": round(prob_xgb, 4) if prob_xgb is not None else None,
                "status": xgb_p["status"]["label"] if xgb_p else "N/A",
                "color": xgb_p["status"]["color"] if xgb_p else "gray",
            },
            "lstm": {
                "probability": round(prob_lstm, 4) if prob_lstm is not None else None,
                "status": lstm_p["status"]["label"] if lstm_p else "N/A",
                "color": lstm_p["status"]["color"] if lstm_p else "gray",
            },
            "recommendation": recommendation,
            "target_utc": target_utc,
            "target_wib": target_wib,
        })

    # Fitur fisis terkini dari observasi terakhir
    current_features = lstm_res.get("current") or xgb_res.get("current") or {}

    return {
        "status": "success",
        "model_type": "Hybrid Ensemble (XGBoost + LSTM)",
        "observed_at_utc": xgb_res.get("observed_at_utc") or lstm_res.get("observed_at_utc"),
        "observed_at_wib": xgb_res.get("observed_at_wib") or lstm_res.get("observed_at_wib"),
        "overall_status": {
            "level": max_level,
            "label": overall_status_label,
            "color": overall_color,
        },
        "current": current_features,
        "predictions": hybrid_predictions,
    }


def predict_hybrid_from_records(records: list) -> dict:
    """Menjalankan inferensi hybrid langsung dari daftar rekaman historis."""
    if len(records) < 12:
        raise ValueError(f"Dibutuhkan minimal 12 data rekaman untuk model Hybrid (saat ini: {len(records)}).")

    xgb_res = predict_service.predict_from_gsheet(records)
    lstm_res = lstm_service.predict_lstm_from_gsheet(records)

    return compute_hybrid_predictions(xgb_res, lstm_res)
