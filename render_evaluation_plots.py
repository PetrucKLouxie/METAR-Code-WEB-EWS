"""
Script untuk menghasilkan 4 diagram standar literatur internasional (WMO / AMS / ICAO)
sebagai gambar statis (.png) resolusi tinggi dan menyimpannya di folder project:
public/images/evaluation/
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "public", "images", "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

# Baca data dari public/eval_results_gsheet.json
DATA_FILE = os.path.join(BASE_DIR, "public", "eval_results_gsheet.json")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

d = data.get("diagrams", {})

# Styling tema gelap (Dark Theme matching dashboard)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#e2e8f0'
plt.rcParams['axes.labelcolor'] = '#cbd5e1'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
BG_COLOR = '#090d16'
CARD_COLOR = '#0f172a'
BORDER_COLOR = '#1e293b'
GRID_COLOR = '#1e293b'

# -------------------------------------------------------------
# 1. DIAGRAM 1: Time-Series Timeline Strip (Meteogram Verifikasi)
# -------------------------------------------------------------
def plot_diagram_1():
    tl = d.get("timeline", [])
    if not tl:
        print("Data timeline kosong!")
        return

    x = np.arange(len(tl))
    slots = [t["slot"].split(" ")[1][:5] if " " in t["slot"] else str(i) for i, t in enumerate(tl)]
    prob_hyb = [t["prob_hybrid"] for t in tl]
    prob_lstm = [t["prob_lstm"] for t in tl]
    prob_xgb = [t["prob_xgb"] for t in tl]
    actual = [100 if t.get("actual_event") == 1 else 0 for t in tl]

    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    # Plot balok kejadian riil (Event bar)
    ax.bar(x, actual, width=1.0, color='#ef4444', alpha=0.35, label='Badai Riil METAR (Event)', zorder=2)

    # Plot kurva peluang
    ax.plot(x, prob_hyb, color='#10b981', linewidth=2.8, label='HYBRID Consensus Prob (%)', zorder=5)
    ax.fill_between(x, prob_hyb, color='#10b981', alpha=0.15, zorder=3)
    ax.plot(x, prob_lstm, color='#a855f7', linewidth=1.5, linestyle='--', alpha=0.85, label='LSTM Sequence Prob (%)', zorder=4)
    ax.plot(x, prob_xgb, color='#06b6d4', linewidth=1.5, linestyle=':', alpha=0.85, label='XGBoost Physical Prob (%)', zorder=4)

    # Threshold horizontal
    ax.axhline(15, color='#f59e0b', linestyle='--', linewidth=1.2, label='Ambang Waspada (15%)', zorder=3)
    ax.axhline(30, color='#ef4444', linestyle=':', linewidth=1.2, label='Ambang Bahaya (30%)', zorder=3)

    # Setting sumbu
    step = max(1, len(tl) // 16)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(slots[::step], rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Peluang Peringatan Dini (%)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Diagram 1: Time-Series Timeline Strip (Meteogram Verifikasi Aktual vs Prediksi)\nObservasi Lapangan WARR Juanda (Perbandingan Dinamis Lead-Time)', 
                 fontsize=13, fontweight='bold', pad=15, color='#ffffff')
    ax.grid(True, color=GRID_COLOR, linestyle='-', linewidth=0.8, alpha=0.7)

    # Legenda
    ax.legend(loc='upper right', facecolor=CARD_COLOR, edgecolor=BORDER_COLOR, fontsize=9.5, framealpha=0.95)

    # Spines
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "diagram1_timeline_meteogram.png")
    plt.savefig(out_path, dpi=200, facecolor=BG_COLOR)
    plt.close()
    print(f"Disimpan: {out_path}")

# -------------------------------------------------------------
# 2. DIAGRAM 2: Roebber Performance Diagram (2009)
# -------------------------------------------------------------
def plot_diagram_2():
    fig, ax = plt.subplots(figsize=(8, 7.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    sr_vals = np.linspace(0.001, 1.0, 400)

    # CSI contours: POD = CSI * SR / (SR * (1 + CSI) - CSI)
    csi_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    for csi in csi_list:
        denom = sr_vals * (1 + csi) - csi
        valid = (denom > 0) & (sr_vals >= csi)
        pod_csi = np.where(valid, (csi * sr_vals) / denom, np.nan)
        mask = (pod_csi >= 0) & (pod_csi <= 1.0)
        ax.plot(sr_vals[mask], pod_csi[mask], color='#475569', linestyle='--', linewidth=0.9, alpha=0.55)
        # Label contour
        mid_idx = np.nanargmin(np.abs(sr_vals[mask] - (csi + 0.2))) if np.any(mask) else None
        if mid_idx is not None and mid_idx < len(sr_vals[mask]):
            ax.text(sr_vals[mask][mid_idx], pod_csi[mask][mid_idx], f" CSI {csi}", color='#64748b', fontsize=7.5)

    # Bias lines: POD = Bias * SR
    bias_list = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    for b in bias_list:
        pod_b = b * sr_vals
        mask = (pod_b >= 0) & (pod_b <= 1.0)
        col = '#94a3b8' if b == 1.0 else '#334155'
        ls = '-' if b == 1.0 else ':'
        lw = 1.3 if b == 1.0 else 0.8
        ax.plot(sr_vals[mask], pod_b[mask], color=col, linestyle=ls, linewidth=lw, alpha=0.6)
        if np.any(mask):
            last_x = sr_vals[mask][-1]
            last_y = pod_b[mask][-1]
            ax.text(last_x, last_y, f" B={b}", color='#64748b', fontsize=7.5)

    # Plot Models
    models = d.get("roebber", {}).get("models", [])
    for m in models:
        marker = '^' if m["type"] == 'hybrid' else ('s' if m["type"] == 'lstm' else 'o')
        size = 110 if m["horizon"] == '1h' else 80
        ax.scatter(m["sr"], m["pod"], color=m["color"], s=size, marker=marker, edgecolors='#ffffff', linewidth=1.2, zorder=10,
                   label=f'{m["name"]} (POD: {m["pod"]*100:.1f}%, CSI: {m["csi"]:.3f})')

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Success Ratio (1 - FAR)  → Bebas Alarm Palsu', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Probability of Detection (POD)  → Sensitivitas Deteksi', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Diagram 2: Performance Diagram (Roebber, 2009)\nVerifikasi 4 Metrik WMO dalam Ruang Geometris 2D', 
                 fontsize=12, fontweight='bold', pad=15, color='#ffffff')
    ax.grid(True, color=GRID_COLOR, linestyle='-', linewidth=0.8, alpha=0.5)

    ax.legend(loc='lower right', facecolor=CARD_COLOR, edgecolor=BORDER_COLOR, fontsize=8.5, framealpha=0.95)

    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "diagram2_roebber_performance.png")
    plt.savefig(out_path, dpi=200, facecolor=BG_COLOR)
    plt.close()
    print(f"Disimpan: {out_path}")

# -------------------------------------------------------------
# 3. DIAGRAM 3: Reliability Diagram (Calibration Curve)
# -------------------------------------------------------------
def plot_diagram_3():
    rel = d.get("reliability", {})
    if not rel:
        print("Data reliability kosong!")
        return

    bins = rel.get("bin_centers", [5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    ideal = rel.get("ideal", bins)
    hyb = rel.get("hybrid", [])
    lstm = rel.get("lstm", [])
    xgb = rel.get("xgb", [])

    fig, ax = plt.subplots(figsize=(8, 7.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    # Garis ideal 45 derajat
    ax.plot([0, 100], [0, 100], color='#64748b', linestyle='--', linewidth=1.8, label='Perfect Reliability (Kalibrasi Ideal 45°)', zorder=2)

    # Kurva Model
    ax.plot(bins, hyb, color='#10b981', marker='^', markersize=8, linewidth=2.5, label='HYBRID Consensus (Terkalibrasi Fisis)', zorder=6)
    ax.plot(bins, lstm, color='#a855f7', marker='s', markersize=6, linestyle='--', linewidth=1.8, label='LSTM Sequence (Cenderung Over-confident)', zorder=5)
    ax.plot(bins, xgb, color='#06b6d4', marker='o', markersize=6, linestyle=':', linewidth=1.8, label='XGBoost Physical (Konservatif)', zorder=4)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Peluang Ramalan Model (Forecast Probability Bin %)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Frekuensi Observasi Lapangan Aktual (Observed Relative Frequency %)', fontsize=10.5, fontweight='bold', labelpad=10)
    ax.set_title('Diagram 3: Reliability Diagram & Kalibrasi Probabilitas\nUji Kepercayaan Peringatan Dini Cuaca Ekstrem Langka', 
                 fontsize=12, fontweight='bold', pad=15, color='#ffffff')
    ax.grid(True, color=GRID_COLOR, linestyle='-', linewidth=0.8, alpha=0.6)

    ax.legend(loc='upper left', facecolor=CARD_COLOR, edgecolor=BORDER_COLOR, fontsize=8.8, framealpha=0.95)

    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "diagram3_reliability_calibration.png")
    plt.savefig(out_path, dpi=200, facecolor=BG_COLOR)
    plt.close()
    print(f"Disimpan: {out_path}")

# -------------------------------------------------------------
# 4. DIAGRAM 4: Lead-Time Degradation Curve (Multi-Horizon)
# -------------------------------------------------------------
def plot_diagram_4():
    lt = d.get("lead_time", {})
    if not lt:
        print("Data lead time kosong!")
        return

    horizons = lt.get("horizons", ["+1h", "+3h", "+6h", "+12h", "+24h"])
    pod_hyb = lt.get("pod", {}).get("hybrid", [])
    pod_lstm = lt.get("pod", {}).get("lstm", [])
    pod_xgb = lt.get("pod", {}).get("xgb", [])
    csi_hyb = [v * 1000 for v in lt.get("csi", {}).get("hybrid", [])]

    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    x = np.arange(len(horizons))

    # Plot kurva degradasi
    ax.plot(x, pod_hyb, color='#10b981', marker='^', markersize=9, linewidth=2.8, label='POD Hybrid (%) - Tingkat Deteksi Terjaga', zorder=6)
    ax.plot(x, pod_lstm, color='#a855f7', marker='s', markersize=7, linestyle='--', linewidth=1.8, label='POD LSTM (%) - Sangat Peka', zorder=5)
    ax.plot(x, pod_xgb, color='#06b6d4', marker='o', markersize=7, linestyle=':', linewidth=1.8, label='POD XGBoost (%) - Konservatif', zorder=4)
    ax.plot(x, csi_hyb, color='#f59e0b', marker='d', markersize=7, linestyle='-.', linewidth=2.0, label='CSI Hybrid (Threat Score x1000)', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xlabel('Lead-Time Horizon Waktu Ramalan (Jarak Ancang-Ancang)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Skor Performa Operasional (%)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Diagram 4: Lead-Time Degradation Curve (Multi-Horizon Skill Decay)\nDaya Tahan Prediksi EWS dari Nowcasting (+1h) hingga Outlook Harian (+24h)', 
                 fontsize=13, fontweight='bold', pad=15, color='#ffffff')
    ax.grid(True, color=GRID_COLOR, linestyle='-', linewidth=0.8, alpha=0.7)

    # Anotasi batas fisis
    ax.axvspan(0, 1, color='#10b981', alpha=0.08, label='Zona Nowcasting Kritis (+1h s/d +3h: POD >= 94%)')

    ax.legend(loc='lower left', facecolor=CARD_COLOR, edgecolor=BORDER_COLOR, fontsize=9.5, framealpha=0.95)

    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "diagram4_lead_time_degradation.png")
    plt.savefig(out_path, dpi=200, facecolor=BG_COLOR)
    plt.close()
    print(f"Disimpan: {out_path}")

if __name__ == "__main__":
    print("Memulai render 4 diagram standar...")
    plot_diagram_1()
    plot_diagram_2()
    plot_diagram_3()
    plot_diagram_4()
    print("Semua diagram berhasil di-render dan disimpan!")
