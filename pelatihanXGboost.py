import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score
)

os.makedirs("saved_models", exist_ok=True)

# =====================================================================
# TAHAP AWAL: DATA PREPARATION & ATMOSPHERIC FEATURE ENGINEERING
# =====================================================================
print("="*95)
print("1. MEMPROSES RUN TUN WAKTU & REKAYASA FITUR TERMODINAMIKA ATMOSFER")
print("="*95)

df = pd.read_csv('METAR_Record.csv')
df['slot_30min'] = pd.to_datetime(df['slot_30min'])
df = df.sort_values('slot_30min').reset_index(drop=True)

# Interpolasi linear untuk missing values sesaat
numeric_features = ["wind_speed", "dew_point", "pressure", "temperature"]
df[numeric_features] = df[numeric_features].interpolate(method="linear").bfill().ffill()

# Rumus DPD, RH (Magnus), VPD, Air Density, dan Siklus Diurnal
df["dpd"] = df["temperature"] - df["dew_point"]
es = 6.112 * np.exp((17.67 * df["temperature"]) / (df["temperature"] + 243.5))
e = 6.112 * np.exp((17.67 * df["dew_point"]) / (df["dew_point"] + 243.5))
df["relative_humidity"] = np.clip((e / es) * 100.0, 0.0, 100.0)
df["vpd"] = np.clip(es - e, 0.0, None)

temp_kelvin = df["temperature"] + 273.15
df["air_density"] = (df["pressure"] * 100.0) / (287.058 * temp_kelvin)
wind_mps = df["wind_speed"] * 0.514444
df["wind_energy_proxy"] = 0.5 * df["air_density"] * (wind_mps ** 2)

hour = df["slot_30min"].dt.hour + df["slot_30min"].dt.minute / 60.0
df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

df["pressure_tendency_3h"] = df["pressure"].diff(periods=6).fillna(0)
df["temp_tendency_1h"] = df["temperature"].diff(periods=2).fillna(0)
df["wind_acceleration_1h"] = df["wind_speed"].diff(periods=2).fillna(0)

feature_cols = [
    "temperature", "dew_point", "dpd", "pressure", "wind_speed",
    "relative_humidity", "vpd", "air_density", "wind_energy_proxy",
    "pressure_tendency_3h", "temp_tendency_1h", "wind_acceleration_1h",
    "hour_sin", "hour_cos"
]

monotone_constraints = {
    "wind_speed": 1,
    "wind_energy_proxy": 1,
    "dpd": -1,
    "pressure_tendency_3h": -1,
    "relative_humidity": 1
}

print(f"Total Baris Data    : {len(df):,} baris")
print(f"Jumlah Fitur Input  : {len(feature_cols)} fitur fisis & temporal")
print(f"Kendala Monotonik   : Kepatuhan termodinamika aktif pada 5 fitur utama.")

# Inisialisasi Dictionary Konfigurasi JSON Lengkap (Tetap disimpan untuk backend)
HORIZONS = {"1h": 2, "3h": 6, "9h": 18, "18h": 36, "24h": 48}

model_config = {
    "system_architecture": {
        "name": "Horizon-Specific Direct Multi-Horizon XGBoost",
        "observation_interval_minutes": 30,
        "input_features_count": len(feature_cols),
        "target_horizons": list(HORIZONS.keys())
    },
    "dataset_allocation": {
        "training_set": {
            "period": "01 Januari 2021 s/d 31 Desember 2024 (4 Tahun Kalender Penuh)",
            "sample_size": int(((df['slot_30min'] >= '2021-01-01') & (df['slot_30min'] <= '2024-12-31 23:59:59')).sum()),
            "hazardous_events": int(df.loc[(df['slot_30min'] >= '2021-01-01') & (df['slot_30min'] <= '2024-12-31 23:59:59'), 'bad_weather'].sum()),
            "base_rate_percentage": round(float(df.loc[(df['slot_30min'] >= '2021-01-01') & (df['slot_30min'] <= '2024-12-31 23:59:59'), 'bad_weather'].mean() * 100), 2),
            "mathematical_execution": "Optimasi Gradient Boosting Taylor Orde 2 dengan pembobotan scale_pos_weight & Monotone Constraints."
        },
        "validation_set": {
            "period": "01 Januari 2025 s/d 31 Desember 2025 (1 Tahun Kalender Penuh)",
            "sample_size": int(((df['slot_30min'] >= '2025-01-01') & (df['slot_30min'] <= '2025-12-31 23:59:59')).sum()),
            "hazardous_events": int(df.loc[(df['slot_30min'] >= '2025-01-01') & (df['slot_30min'] <= '2025-12-31 23:59:59'), 'bad_weather'].sum()),
            "base_rate_percentage": round(float(df.loc[(df['slot_30min'] >= '2025-01-01') & (df['slot_30min'] <= '2025-12-31 23:59:59'), 'bad_weather'].mean() * 100), 2),
            "mathematical_execution": "Platt Scaling Sigmoid Calibration & Optimasi Dual-Threshold (F2 untuk Waspada, F0.5 untuk Bahaya)."
        },
        "test_set": {
            "period": "01 Januari 2026 s/d Sekarang (Oktober 2026)",
            "sample_size": int((df['slot_30min'] >= '2026-01-01').sum()),
            "hazardous_events": int(df.loc[df['slot_30min'] >= '2026-01-01', 'bad_weather'].sum()),
            "base_rate_percentage": round(float(df.loc[df['slot_30min'] >= '2026-01-01', 'bad_weather'].mean() * 100), 2),
            "mathematical_execution": "Verifikasi Tabel Kontingensi 2x2 Standar WMO-No. 485 (POD, FAR, CSI, HSS, Brier Score)."
        }
    },
    "feature_names": feature_cols,
    "horizons": {}
}

# =====================================================================
# ITERASI 5 HORIZON: PROSES BAGIAN 1, 2, DAN 3
# =====================================================================
summary_table = []
plot_data = {
    "horizons": list(HORIZONS.keys()),
    "csi": [], "hss": [], "prauc": [],
    "pod": [], "far": [], "brier": [],
    "th_waspada": [], "th_bahaya": []
}

for name, step in HORIZONS.items():
    print("\n" + "#"*95)
    print(f"   MEMPROSES MODEL HORIZON +{name} (Proyeksi Masa Depan: {step*30} Menit)")
    print("#"*95)

    df_h = df.copy()
    df_h["target"] = df_h["bad_weather"].shift(-step)
    df_h = df_h.dropna(subset=["target"]).reset_index(drop=True)

    # Filter Mask Waktu Kalender
    train_mask = (df_h["slot_30min"] >= "2021-01-01") & (df_h["slot_30min"] <= "2024-12-31 23:59:59")
    val_mask   = (df_h["slot_30min"] >= "2025-01-01") & (df_h["slot_30min"] <= "2025-12-31 23:59:59")
    test_mask  = (df_h["slot_30min"] >= "2026-01-01")

    X_train, y_train = df_h.loc[train_mask, feature_cols], df_h.loc[train_mask, "target"]
    X_val, y_val     = df_h.loc[val_mask, feature_cols], df_h.loc[val_mask, "target"]
    X_test, y_test   = df_h.loc[test_mask, feature_cols], df_h.loc[test_mask, "target"]

    # -----------------------------------------------------------------
    # BAGIAN 1: TRAINING SET (01 Jan 2021 s/d 31 Des 2024)
    # -----------------------------------------------------------------
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    print("\n--- [BAGIAN 1: TRAINING SET (01 Jan 2021 s/d 31 Des 2024)] ---")
    print(f"* Jumlah Sampel           : {len(X_train):,} observasi (4 Tahun Kalender Penuh)")
    print(f"* Kejadian Bahaya Riil    : {pos_count:,} baris ({pos_count/len(X_train)*100:.2f}%)")
    print(f"* Rumus Bobot Imbalance   : w_pos = N_neg / N_pos = {neg_count:,} / {pos_count:,} = {scale_pos_weight:.2f}")
    print(f"* Output yang Dihasilkan  : Pohon keputusan dasar (Base Trees) patuh termodinamika.")

    base_xgb = XGBClassifier(
        n_estimators=350,
        learning_rate=0.03,
        max_depth=4 if name in ["1h", "3h"] else 3,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints,
        eval_metric="aucpr",
        random_state=42
    )
    base_xgb.fit(X_train, y_train)

    # -----------------------------------------------------------------
    # BAGIAN 2: VALIDATION SET (01 Jan 2025 s/d 31 Des 2025)
    # -----------------------------------------------------------------
    print("\n--- [BAGIAN 2: VALIDATION SET (01 Jan 2025 s/d 31 Des 2025)] ---")
    print(f"* Jumlah Sampel           : {len(X_val):,} observasi (1 Tahun Penuh Kalender)")
    print(f"* Kejadian Bahaya Riil    : {int(y_val.sum()):,} baris ({y_val.mean()*100:.2f}%)")
    print(f"* Rumus Platt Scaling     : P(y=1|margin) = 1 / (1 + exp(A*margin + B))")

    calibrated_xgb = CalibratedClassifierCV(estimator=base_xgb, method="sigmoid", cv="prefit")
    calibrated_xgb.fit(X_val, y_val)

    val_probs = calibrated_xgb.predict_proba(X_val)[:, 1]

    best_th_waspada, best_f2 = 0.10, 0.0
    best_th_bahaya, best_f05 = 0.25, 0.0

    for th in np.linspace(0.05, 0.65, 121):
        preds = (val_probs >= th).astype(int)
        f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)
        f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)

        if f2 > best_f2:
            best_f2 = f2
            best_th_waspada = th

        if f05 > best_f05:
            best_f05 = f05
            best_th_bahaya = th

    if best_th_bahaya <= best_th_waspada:
        best_th_bahaya = round(best_th_waspada + 0.15, 3)

    print(f"* Rumus Ambang WASPADA    : ArgMax F2-Score = 5*(P*R)/(4*P + R)  --> Thresh: {best_th_waspada:.3f}")
    print(f"* Rumus Ambang BAHAYA     : ArgMax F0.5-Score = 1.25*(P*R)/(0.25*P + R) --> Thresh: {best_th_bahaya:.3f}")
    print(f"* Output yang Dihasilkan  : Model terkalibrasi (.joblib) & dua batas keputusan operasional.")

    # -----------------------------------------------------------------
    # BAGIAN 3: TEST SET (01 Jan 2026 s/d Sekarang)
    # -----------------------------------------------------------------
    test_probs = calibrated_xgb.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_th_waspada).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    accuracy = accuracy_score(y_test, test_preds)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    expected_correct = ((tp + fn)*(tp + fp) + (tn + fp)*(tn + fn)) / len(y_test)
    hss = (((tp + tn) - expected_correct) / (len(y_test) - expected_correct)) if (len(y_test) - expected_correct) > 0 else 0.0

    prauc = average_precision_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)

    print("\n--- [BAGIAN 3: TEST SET OPERASIONAL (01 Jan 2026 s/d Sekarang)] ---")
    print(f"* Jumlah Sampel           : {len(X_test):,} observasi (Unseen Future Data)")
    print(f"* Kejadian Bahaya Riil    : {int(y_test.sum()):,} baris ({y_test.mean()*100:.2f}%)")
    print(f"* Rumus WMO               : POD=TP/(TP+FN), FAR=FP/(TP+FP), CSI=TP/(TP+FP+FN), HSS, Brier")
    print(f"* Tabel Kontingensi 2x2   : TP(Hits)={tp}, FN(Misses)={fn}, FP(FalseAlarm)={fp}, TN(CorrNeg)={tn}")
    print(f"* Output Evaluasi WMO     : POD={pod*100:.2f}%, FAR={far*100:.2f}%, CSI={csi:.3f}, HSS={hss:.3f}, PR-AUC={prauc:.3f}")

    # Simpan model artefak .joblib
    model_filename = f"saved_models/xgb_model_{name}.joblib"
    joblib.dump(calibrated_xgb, model_filename)

    # Masukkan data ke dictionary model_config.json
    model_config["horizons"][name] = {
        "lead_steps": step,
        "lead_time_minutes": step * 30,
        "model_file": model_filename,
        "thresholds": {
            "th_waspada": float(round(best_th_waspada, 4)),
            "th_bahaya": float(round(best_th_bahaya, 4))
        },
        "metrics_test_2026": {
            "accuracy": float(round(accuracy, 4)),
            "pod_hit_rate": float(round(pod, 4)),
            "far_false_alarm": float(round(far, 4)),
            "csi_threat_score": float(round(csi, 4)),
            "hss_skill_score": float(round(hss, 4)),
            "prauc": float(round(prauc, 4)),
            "brier_score": float(round(brier, 4))
        }
    }

    summary_table.append({
        "Horizon": f"+{name}",
        "Th_Waspada": round(best_th_waspada, 3),
        "Th_Bahaya": round(best_th_bahaya, 3),
        "Accuracy": f"{accuracy*100:.2f}%",
        "POD (Hit Rate)": f"{pod*100:.2f}%",
        "FAR (Alarm Palsu)": f"{far*100:.2f}%",
        "CSI (Threat)": round(csi, 3),
        "HSS (Skill)": round(hss, 3),
        "PR-AUC": round(prauc, 3),
        "Brier": round(brier, 4)
    })

    # Simpan data plot visual
    plot_data["csi"].append(csi)
    plot_data["hss"].append(hss)
    plot_data["prauc"].append(prauc)
    plot_data["pod"].append(pod * 100)
    plot_data["far"].append(far * 100)
    plot_data["brier"].append(brier)
    plot_data["th_waspada"].append(best_th_waspada * 100)
    plot_data["th_bahaya"].append(best_th_bahaya * 100)

# =====================================================================
# SIMPAN FILE MODEL CONFIG JSON (FILE CONFIG TETAP ADA!)
# =====================================================================
with open("saved_models/model_config.json", "w") as f:
    json.dump(model_config, f, indent=2)

print("\n" + "="*95)
print("SUKSES! File 'saved_models/model_config.json' berhasil disimpan ke disk.")
print("="*95)

# Cetak Tabel Evaluasi Akhir
print("\n" + "="*105)
print("RINGKASAN AKHIR HASIL VERIFIKASI MODEL SELURUH HORIZON (DATA UJI TAHUN 2026 - STANDAR WMO)")
print("="*105)
print(pd.DataFrame(summary_table).to_string(index=False))

# =====================================================================
# TAMPILKAN 4 PANEL VISUALISASI VERIFIKASI METEOROLOGIS
# =====================================================================
print("\nMenghasilkan visualisasi verifikasi standar internasional...")

fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=100)
horizons_lbl = [f"+{h}" for h in plot_data["horizons"]]
x = np.arange(len(horizons_lbl))
width = 0.25

# Panel 1: WMO Skill Across Horizons
axes[0, 0].bar(x - width, plot_data["csi"], width, label='CSI (Threat Score)', color='#1f77b4')
axes[0, 0].bar(x, plot_data["hss"], width, label='HSS (Skill Score)', color='#2ca02c')
axes[0, 0].bar(x + width, plot_data["prauc"], width, label='PR-AUC', color='#ff7f0e')
axes[0, 0].axhline(0.0384, color='red', linestyle='--', label='No-Skill Baseline (Prior: 0.038)')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(horizons_lbl)
axes[0, 0].set_title('1. Skor Keandalan WMO vs Lead Time (Test Set 2026)', fontweight='bold', fontsize=12)
axes[0, 0].set_ylabel('Skor Indeks', fontsize=11)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', linestyle=':', alpha=0.6)

# Panel 2: Trade-Off POD vs FAR
axes[0, 1].plot(horizons_lbl, plot_data["pod"], marker='o', lw=2.5, color='#2ca02c', label='POD / Hit Rate (%)')
axes[0, 1].plot(horizons_lbl, plot_data["far"], marker='s', lw=2.5, color='#d62728', label='FAR / False Alarm (%)')
axes[0, 1].set_title('2. Trade-Off Deteksi Bahaya vs Alarm Palsu', fontweight='bold', fontsize=12)
axes[0, 1].set_ylabel('Persentase (%)', fontsize=11)
axes[0, 1].set_ylim(40, 100)
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle=':', alpha=0.6)

# Panel 3: Dua Tingkat Ambang Batas Operasional
axes[1, 0].plot(horizons_lbl, plot_data["th_waspada"], marker='^', lw=2.5, color='#e6ab02', label='Threshold WASPADA (F2-Score)')
axes[1, 0].plot(horizons_lbl, plot_data["th_bahaya"], marker='D', lw=2.5, color='#e41a1c', label='Threshold BAHAYA (F0.5-Score)')
axes[1, 0].set_title('3. Ambang Batas Keputusan Operasional (Dual-Level)', fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Ambang Probabilitas (%)', fontsize=11)
axes[1, 0].set_ylim(0, 35)
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle=':', alpha=0.6)

# Panel 4: Brier Score Calibration Quality
axes[1, 1].plot(horizons_lbl, plot_data["brier"], marker='v', lw=2.5, color='#9467bd', label='Brier Score (Calibrated Sigmoid)')
axes[1, 1].set_title('4. Keandalan Probabilitas (Brier Score, Nilai Ideal -> 0)', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('Mean Squared Error (Brier Score)', fontsize=11)
axes[1, 1].set_ylim(0.02, 0.05)
axes[1, 1].legend()
axes[1, 1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('saved_models/wmo_verification_dashboard.png')
plt.show()
print("\nDashboard grafik berhasil disimpan ke 'saved_models/wmo_verification_dashboard.png' dan ditampilkan di atas!")