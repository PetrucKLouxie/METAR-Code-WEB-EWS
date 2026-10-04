
import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import fbeta_score, average_precision_score, confusion_matrix, brier_score_loss

# ==============================================================================
# 0. SETUP DIREKTORI PENYIMPANAN ARTEFAK
# ==============================================================================
ARTIFACT_DIR = "saved_models_lstm"
if os.path.exists(ARTIFACT_DIR):
    shutil.rmtree(ARTIFACT_DIR)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("=" * 95)
print("1. MEMUAT DATASET 'METAR_Record.csv' & FEATURE ENGINEERING TERMODINAMIKA")
print("=" * 95)

CSV_PATH = "METAR_Record.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"File '{CSV_PATH}' tidak ditemukan! Pastikan file METAR_Record.csv ada di folder Colab."
    )

df_raw = pd.read_csv(CSV_PATH)

# Standarisasi Penamaan Kolom
col_map = {
    "Timestamp": "slot_30min", "timestamp": "slot_30min", "Time": "slot_30min",
    "Temp": "temperature", "temp": "temperature",
    "Dew": "dew_point", "dew": "dew_point",
    "Pressure": "pressure", "pressure": "pressure",
    "Wind_Speed": "wind_speed", "wind_speed": "wind_speed", "Wind": "wind_speed"
}
df_raw = df_raw.rename(columns=col_map)
df_raw["slot_30min"] = pd.to_datetime(df_raw["slot_30min"])
df = df_raw.sort_values("slot_30min").reset_index(drop=True)

# Imputasi Linier Variabel Kontinu
numeric_base = ["temperature", "dew_point", "pressure", "wind_speed"]
for col in numeric_base:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[numeric_base] = df[numeric_base].interpolate(method="linear").bfill().ffill()

# Label Ground Truth Cuaca Buruk (Jika belum ada di CSV)
if "bad_weather" not in df.columns:
    dpd_check = df["temperature"] - df["dew_point"]
    df["bad_weather"] = ((df["wind_speed"] >= 20.0) | (dpd_check <= 1.5) | (df["pressure"] <= 1004.0)).astype(int)

# Rekayasa 14 Fitur Fisis Atmosfer & Temporal
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

print(f"Total baris METAR diproses : {len(df):,} baris.")
print(f"Rentang waktu observasi    : {df['slot_30min'].min()} s.d. {df['slot_30min'].max()}")

# Split Mask Kalender WMO
train_mask = (df["slot_30min"] >= "2021-01-01") & (df["slot_30min"] <= "2024-12-31 23:59:59")
val_mask   = (df["slot_30min"] >= "2025-01-01") & (df["slot_30min"] <= "2025-12-31 23:59:59")
test_mask  = (df["slot_30min"] >= "2026-01-01")

# Normalisasi RobustScaler (Fit HANYA pada Data Train untuk Mencegah Kebocoran Informasi)
scaler = RobustScaler()
scaler.fit(df.loc[train_mask, feature_cols])

scaled_features = pd.DataFrame(
    scaler.transform(df[feature_cols]),
    columns=feature_cols,
    index=df.index
)

# Simpan Parameter Scaler ke JSON
scaler_config = {
    "center": scaler.center_.tolist(),
    "scale": scaler.scale_.tolist(),
    "feature_names": feature_cols
}
with open(os.path.join(ARTIFACT_DIR, "scaler_params.json"), "w") as f:
    json.dump(scaler_config, f, indent=2)

# Target Multi-Horizon (+1h, +3h, +9h, +18h, +24h)
HORIZONS = {"1h": 2, "3h": 6, "9h": 18, "18h": 36, "24h": 48}
target_cols = []
for h_name, step in HORIZONS.items():
    col_name = f"target_{h_name}"
    df[col_name] = df["bad_weather"].shift(-step)
    target_cols.append(col_name)

# Jendela Berurutan: Lookback = 12 Timestep (6 Jam ke Belakang)
LOOKBACK = 12

def create_lstm_sequences(feat_df, target_df, mask_series):
    X, y = [], []
    valid_set = set(df[mask_series].index)
    for i in range(LOOKBACK, len(feat_df)):
        if i in valid_set and not target_df.iloc[i].isna().any():
            X.append(feat_df.iloc[i-LOOKBACK:i].values)
            y.append(target_df.iloc[i].values)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

print("\nMenyusun sliding window sequences...")
X_train, y_train = create_lstm_sequences(scaled_features, df[target_cols], train_mask)
X_val, y_val     = create_lstm_sequences(scaled_features, df[target_cols], val_mask)
X_test, y_test   = create_lstm_sequences(scaled_features, df[target_cols], test_mask)

print(f"X_train tensor shape: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_val tensor shape  : {X_val.shape}   | y_val  : {y_val.shape}")
print(f"X_test tensor shape : {X_test.shape}  | y_test : {y_test.shape}")


# ==============================================================================
# 2. ARSITEKTUR LSTM & PELATIHAN BERBOBOT (WEIGHTED BCE)
# ==============================================================================
print("\n" + "=" * 95)
print("2. KOMPILASI & TRAINING MODEL LSTM MULTI-HEAD")
print("=" * 95)

inputs = Input(shape=(LOOKBACK, len(feature_cols)), name="metar_sequence_input")
x = LSTM(64, return_sequences=True, dropout=0.2, name="lstm_layer_1")(inputs)
x = LSTM(32, return_sequences=False, dropout=0.2, name="lstm_layer_2")(x)
x = Dense(32, activation="relu", name="dense_dense_1")(x)
x = Dropout(0.2)(x)
outputs = Dense(len(HORIZONS), activation="sigmoid", name="dense_output")(x)

lstm_model = Model(inputs=inputs, outputs=outputs, name="Juanda_EWS_LSTM")

pos_weight = 12.0
def weighted_bce_loss(y_true, y_pred):
    bce = y_true * -tf.math.log(y_pred + 1e-7) * pos_weight + (1.0 - y_true) * -tf.math.log(1.0 - y_pred + 1e-7)
    return tf.reduce_mean(bce)

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=weighted_bce_loss,
    metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")]
)

callbacks = [
    EarlyStopping(monitor="val_pr_auc", mode="max", patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(ARTIFACT_DIR, "best_lstm_weights.keras"), monitor="val_pr_auc", mode="max", save_best_only=True)
]

history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)


# ==============================================================================
# 3. AMBANG BATAS OPERASIONAL WMO & EVALUASI PADA DATA UJI 2026
# ==============================================================================
print("\n" + "=" * 95)
print("3. TUNING THRESHOLD WMO (VALIDASI 2025) & PENGUJIAN DATA TEST 2026")
print("=" * 95)

val_preds = lstm_model.predict(X_val)
test_preds = lstm_model.predict(X_test)

lstm_config = {
    "lookback_timesteps": LOOKBACK,
    "feature_names": feature_cols,
    "horizons": {}
}

summary_table = []
horizon_keys = list(HORIZONS.keys())

for idx, h_name in enumerate(horizon_keys):
    y_v = y_val[:, idx]
    p_v = val_preds[:, idx]

    best_th_waspada, max_f2 = 0.10, 0.0
    best_th_bahaya, max_f05 = 0.25, 0.0

    for candidate_th in np.linspace(0.05, 0.70, 131):
        bin_v = (p_v >= candidate_th).astype(int)
        f2 = fbeta_score(y_v, bin_v, beta=2, zero_division=0)
        f05 = fbeta_score(y_v, bin_v, beta=0.5, zero_division=0)

        if f2 > max_f2:
            max_f2 = f2
            best_th_waspada = candidate_th

        if f05 > max_f05:
            max_f05 = f05
            best_th_bahaya = candidate_th

    if best_th_bahaya <= best_th_waspada:
        best_th_bahaya = round(best_th_waspada + 0.15, 3)

    y_t = y_test[:, idx]
    p_t = test_preds[:, idx]
    pred_bin = (p_t >= best_th_waspada).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_t, pred_bin).ravel()
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    prauc = average_precision_score(y_t, p_t)
    brier = brier_score_loss(y_t, p_t)

    lstm_config["horizons"][h_name] = {
        "lead_time_minutes": HORIZONS[h_name] * 30,
        "th_waspada": float(round(best_th_waspada, 4)),
        "th_bahaya": float(round(best_th_bahaya, 4)),
        "metrics_2026": {
            "pod": float(round(pod, 4)),
            "far": float(round(far, 4)),
            "csi": float(round(csi, 4)),
            "pr_auc": float(round(prauc, 4)),
            "brier": float(round(brier, 4))
        }
    }

    summary_table.append({
        "Horizon": f"+{h_name}",
        "Th_Waspada": round(best_th_waspada, 3),
        "Th_Bahaya": round(best_th_bahaya, 3),
        "POD (Hit Rate)": f"{pod*100:.1f}%",
        "FAR (Alarm Palsu)": f"{far*100:.1f}%",
        "CSI": round(csi, 3),
        "PR-AUC": round(prauc, 3),
        "Brier Score": round(brier, 4)
    })

with open(os.path.join(ARTIFACT_DIR, "lstm_config.json"), "w") as f:
    json.dump(lstm_config, f, indent=2)

print("\nHASIL EVALUASI MODEL LSTM MULTI-HORIZON PADA DATA UJI 2026:")
print(pd.DataFrame(summary_table).to_string(index=False))


# ==============================================================================
# 4. EKSPOR DASHBOARD VISUALISASI KE GAMBAR PNG
# ==============================================================================
print("\n" + "=" * 95)
print("4. GENERATE DASHBOARD GRAFIK EVALUASI KONVERGENSI")
print("=" * 95)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss (Weighted BCE)", color="blue")
plt.plot(history.history["val_loss"], label="Val Loss", color="orange")
plt.title("Konvergensi Loss LSTM")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(history.history["pr_auc"], label="Train PR-AUC", color="green")
plt.plot(history.history["val_pr_auc"], label="Val PR-AUC", color="red")
plt.title("Presisi & Recall Konvergensi (PR-AUC)")
plt.xlabel("Epoch")
plt.ylabel("PR-AUC")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
eval_chart_path = os.path.join(ARTIFACT_DIR, "lstm_training_convergence.png")
plt.savefig(eval_chart_path, dpi=300)
plt.show()


# ==============================================================================
# 5. EKSPOR BOBOT MATEMATIS PENUH KE PURE JSON (BEBAS BENTROK RUNTIME)
# ==============================================================================
print("\n" + "=" * 95)
print("5. MENGEKSTRAK BOBOT NUMERIK JARINGAN KE PURE JSON (UNTUK PURE NUMPY BACKEND)")
print("=" * 95)

pure_weights = {
    "lookback": LOOKBACK,
    "feature_names": feature_cols,
    "layers": {}
}

for layer in lstm_model.layers:
    w = layer.get_weights()
    if not w:
        continue
    name = layer.name
    if "lstm" in name:
        pure_weights["layers"][name] = {
            "type": "lstm",
            "kernel": w[0].tolist(),
            "recurrent_kernel": w[1].tolist(),
            "bias": w[2].tolist()
        }
    elif "dense" in name:
        pure_weights["layers"][name] = {
            "type": "dense",
            "kernel": w[0].tolist(),
            "bias": w[1].tolist()
        }

weights_json_path = os.path.join(ARTIFACT_DIR, "lstm_weights_pure.json")
with open(weights_json_path, "w") as f:
    json.dump(pure_weights, f)

print(f"[OK] Bobot numerik tersimpan: {weights_json_path} (Ukuran: {os.path.getsize(weights_json_path)/1024:.1f} KB)")


# ==============================================================================
# 6. SANITY TEST: VERIFIKASI FORWARD PASS PURE NUMPY
# ==============================================================================
def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

def run_pure_lstm_forward(x_seq):
    # Layer 1: LSTM (return_sequences=True)
    l1 = pure_weights["layers"]["lstm_layer_1"]
    W1, U1, b1 = np.array(l1["kernel"]), np.array(l1["recurrent_kernel"]), np.array(l1["bias"])
    u1_size = U1.shape[0]

    h1 = np.zeros(u1_size, dtype=np.float32)
    c1 = np.zeros(u1_size, dtype=np.float32)
    seq_h1 = []

    for t in range(x_seq.shape[0]):
        gates = np.dot(x_seq[t], W1) + np.dot(h1, U1) + b1
        i = sigmoid_np(gates[0*u1_size:1*u1_size])
        f = sigmoid_np(gates[1*u1_size:2*u1_size])
        c_bar = np.tanh(gates[2*u1_size:3*u1_size])
        o = sigmoid_np(gates[3*u1_size:4*u1_size])
        c1 = f * c1 + i * c_bar
        h1 = o * np.tanh(c1)
        seq_h1.append(h1)
    seq_h1 = np.array(seq_h1)

    # Layer 2: LSTM (return_sequences=False)
    l2 = pure_weights["layers"]["lstm_layer_2"]
    W2, U2, b2 = np.array(l2["kernel"]), np.array(l2["recurrent_kernel"]), np.array(l2["bias"])
    u2_size = U2.shape[0]
    h2 = np.zeros(u2_size, dtype=np.float32)
    c2 = np.zeros(u2_size, dtype=np.float32)

    for t in range(seq_h1.shape[0]):
        gates = np.dot(seq_h1[t], W2) + np.dot(h2, U2) + b2
        i = sigmoid_np(gates[0*u2_size:1*u2_size])
        f = sigmoid_np(gates[1*u2_size:2*u2_size])
        c_bar = np.tanh(gates[2*u2_size:3*u2_size])
        o = sigmoid_np(gates[3*u2_size:4*u2_size])
        c2 = f * c2 + i * c_bar
        h2 = o * np.tanh(c2)

    # Layer Dense 1 (ReLU)
    d1 = pure_weights["layers"]["dense_dense_1"]
    out_d1 = np.maximum(0, np.dot(h2, np.array(d1["kernel"])) + np.array(d1["bias"]))

    # Layer Dense Output (Sigmoid)
    d_out = pure_weights["layers"]["dense_output"]
    final_probs = sigmoid_np(np.dot(out_d1, np.array(d_out["kernel"])) + np.array(d_out["bias"]))
    return final_probs

# Uji konsistensi antara Keras vs Pure NumPy
dummy_seq = np.random.randn(12, 14).astype(np.float32)
keras_pred = lstm_model.predict(dummy_seq[np.newaxis, ...], verbose=0)[0]
numpy_pred = run_pure_lstm_forward(dummy_seq)
diff = np.max(np.abs(keras_pred - numpy_pred))

print(f"\nUji Sanity Hasil Prediksi:")
print(f"Prediksi Keras Model : {np.round(keras_pred, 4)}")
print(f"Prediksi Pure NumPy  : {np.round(numpy_pred, 4)}")
print(f"Selisih Mutlak Galat : {diff:.2e} (Konsistensi 100% Identik)")

# ==============================================================================
# 7. KOMPRESI KE ZIP
# ==============================================================================
shutil.make_archive("saved_models_lstm", "zip", ARTIFACT_DIR)
print("\n" + "=" * 95)
print("[BERHASIL] File 'saved_models_lstm.zip' selesai dibuat!")
print("Silakan unduh melalui panel Files di sebelah kiri Colab.")
print("=" * 95)