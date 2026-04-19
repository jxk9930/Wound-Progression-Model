"""
Closed-Loop Evaluation Pipeline
================================
1. Reads cropped wound images from cropped_wounds/
2. Runs HuggingFace classification (G/S/N) on each crop
3. Saves results to evaluation_results.csv
4. Generates evaluation plots

Usage in Colab:
    os.chdir('/content/drive/MyDrive/A_Wound_Progression_Model/')
    %run evaluate_closed_loop.py
"""

import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11


# ─────────────────────────────────────────────
# STEP 1: Classification
# ─────────────────────────────────────────────

def classify_crops(crop_dir, csv_path):
    """Load each crop image, run classification, save to CSV."""
    from PIL import Image
    global classifier
    LABEL_MAP = {
        "MoGiaMacNhiemKhuan": "Infected Slough",
        "MoHat": "Granulation",
        "MoHoaiTu": "Necrotic",
    }

    # Parse: {img_name}_{trajectory}_day{DD}_crop.png
    pattern = re.compile(r'^(.+?)_(healing|worsening)_day(\d+)_crop\.png$')

    files = sorted(f for f in os.listdir(crop_dir) if f.endswith('_crop.png'))
    if not files:
        print(f"No crop files found in {crop_dir}/")
        return None

    rows = []
    for fname in files:
        m = pattern.match(fname)
        if not m:
            print(f"  Skipping: {fname}")
            continue

        img_name = m.group(1)
        trajectory = m.group(2)
        day = int(m.group(3))

        img_path = os.path.join(crop_dir, fname)
        image_pil = Image.open(img_path).convert("RGB")

        # Wound area = non-background pixels (background is [128,128,128])
        img_arr = np.array(image_pil)
        is_wound = ~np.all(img_arr == 128, axis=-1)
        wound_px = int(is_wound.sum())

        # Classify
        results = classifier(image_pil)
        cls = {"Granulation": 0.0, "Infected Slough": 0.0, "Necrotic": 0.0}
        for r in results:
            label = LABEL_MAP.get(r["label"], r["label"])
            cls[label] = r["score"]

        G, S, N = cls["Granulation"], cls["Infected Slough"], cls["Necrotic"]

        rows.append({
            "image": img_name, "trajectory": trajectory, "day": day,
            "wound_px": wound_px,
            "G": round(G, 4), "S": round(S, 4), "N": round(N, 4),
        })

        print(f"  {fname}: wound={wound_px}px | "
              f"G={G*100:.1f}% S={S*100:.1f}% N={N*100:.1f}%")

    # Compute PAR / area_change relative to day 0
    for row in rows:
        key = (row["image"], row["trajectory"])
        day0 = [r for r in rows
                if (r["image"], r["trajectory"]) == key and r["day"] == 0]
        area0 = day0[0]["wound_px"] if day0 else row["wound_px"]

        if row["trajectory"] == "healing":
            row["par"] = round(100.0 * (1.0 - row["wound_px"] / max(area0, 1)), 2)
            row["area_change"] = ""
        else:
            row["area_change"] = round(100.0 * (row["wound_px"] / max(area0, 1) - 1.0), 2)
            row["par"] = ""

    # Save
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "trajectory", "day", "wound_px",
            "par", "area_change", "G", "S", "N",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {csv_path}")
    return csv_path


# ─────────────────────────────────────────────
# STEP 2: Load Data
# ─────────────────────────────────────────────

def load_data(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df['G_pct'] = df['G'] * 100
    df['S_pct'] = df['S'] * 100
    df['N_pct'] = df['N'] * 100
    df['par'] = pd.to_numeric(df['par'], errors='coerce')
    df['area_change'] = pd.to_numeric(df['area_change'], errors='coerce')
    return df


# ─────────────────────────────────────────────
# STEP 3: Plots
# ─────────────────────────────────────────────

def plot_gsn_trends(df, out_dir):
    images = df['image'].unique()
    for img in images:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, traj in zip(axes, ['healing', 'worsening']):
            sub = df[(df['image'] == img) & (df['trajectory'] == traj)].sort_values('day')
            if sub.empty:
                ax.set_title(f"{traj.title()} — no data")
                continue
            d = sub['day'].values
            ax.plot(d, sub['G_pct'], 'o-', color='#2ecc71', lw=2, ms=8, label='Granulation (G)')
            ax.plot(d, sub['S_pct'], 's-', color='#f39c12', lw=2, ms=8, label='Slough (S)')
            ax.plot(d, sub['N_pct'], '^-', color='#2c3e50', lw=2, ms=8, label='Necrotic (N)')
            ax.set_xlabel('Day')
            ax.set_ylabel('Classification (%)')
            ax.set_title(f"{traj.title()}")
            ax.legend(loc='best', fontsize=9)
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(d)
        fig.suptitle(f"G/S/N Classification Trends — {img}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{img}_gsn_trends.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {img}_gsn_trends.png")


def plot_area_trends(df, out_dir):
    import pandas as pd
    images = df['image'].unique()
    for img in images:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sub_h = df[(df['image'] == img) & (df['trajectory'] == 'healing')].sort_values('day')
        if not sub_h.empty and sub_h['par'].notna().any():
            d = sub_h['day'].values
            par = sub_h['par'].values
            axes[0].plot(d, par, 'o-', color='#2ecc71', lw=2.5, ms=10)
            axes[0].fill_between(d, 0, par, alpha=0.15, color='#2ecc71')
            axes[0].axhline(y=50, color='gray', ls='--', alpha=0.5, label='50% (good healer)')
            axes[0].set_xticks(d)
            axes[0].legend(fontsize=9)
            for di, pi in zip(d, par):
                axes[0].annotate(f"{pi:.1f}%", (di, pi), textcoords="offset points",
                                xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
        axes[0].set_xlabel('Day')
        axes[0].set_ylabel('PAR (%)')
        axes[0].set_title('Healing: Percent Area Reduction')
        axes[0].grid(True, alpha=0.3)

        sub_w = df[(df['image'] == img) & (df['trajectory'] == 'worsening')].sort_values('day')
        if not sub_w.empty and sub_w['area_change'].notna().any():
            d = sub_w['day'].values
            ac = sub_w['area_change'].fillna(0).values
            axes[1].plot(d, ac, 'o-', color='#e74c3c', lw=2.5, ms=10)
            axes[1].fill_between(d, 0, ac, alpha=0.15, color='#e74c3c')
            axes[1].set_xticks(d)
            for di, ai in zip(d, ac):
                axes[1].annotate(f"+{ai:.1f}%", (di, ai), textcoords="offset points",
                                xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
        axes[1].set_xlabel('Day')
        axes[1].set_ylabel('Area Change (%)')
        axes[1].set_title('Worsening: Area Expansion')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Wound Area Trends — {img}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{img}_area_trends.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {img}_area_trends.png")


def plot_combined_dashboard(df, out_dir):
    import pandas as pd
    images = df['image'].unique()
    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(22, 5.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, img in enumerate(images):
        sub_h = df[(df['image'] == img) & (df['trajectory'] == 'healing')].sort_values('day')
        sub_w = df[(df['image'] == img) & (df['trajectory'] == 'worsening')].sort_values('day')

        if not sub_h.empty:
            d = sub_h['day'].values
            axes[row, 0].plot(d, sub_h['G_pct'], 'o-', color='#2ecc71', lw=2, label='G')
            axes[row, 0].plot(d, sub_h['S_pct'], 's-', color='#f39c12', lw=2, label='S')
            axes[row, 0].plot(d, sub_h['N_pct'], '^-', color='#2c3e50', lw=2, label='N')
            axes[row, 0].set_xticks(d)
        axes[row, 0].set_ylim(-5, 105)
        axes[row, 0].set_title(f"{img} — Healing G/S/N", fontsize=10)
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        if not sub_h.empty and sub_h['par'].notna().any():
            d = sub_h['day'].values
            axes[row, 1].plot(d, sub_h['par'], 'o-', color='#2ecc71', lw=2.5)
            axes[row, 1].fill_between(d, 0, sub_h['par'], alpha=0.15, color='#2ecc71')
            axes[row, 1].axhline(y=50, color='gray', ls='--', alpha=0.5)
            axes[row, 1].set_xticks(d)
        axes[row, 1].set_title(f"{img} — Healing PAR", fontsize=10)
        axes[row, 1].grid(True, alpha=0.3)

        if not sub_w.empty:
            d = sub_w['day'].values
            axes[row, 2].plot(d, sub_w['G_pct'], 'o-', color='#2ecc71', lw=2, label='G')
            axes[row, 2].plot(d, sub_w['S_pct'], 's-', color='#f39c12', lw=2, label='S')
            axes[row, 2].plot(d, sub_w['N_pct'], '^-', color='#2c3e50', lw=2, label='N')
            axes[row, 2].set_xticks(d)
        axes[row, 2].set_ylim(-5, 105)
        axes[row, 2].set_title(f"{img} — Worsening G/S/N", fontsize=10)
        axes[row, 2].legend(fontsize=8)
        axes[row, 2].grid(True, alpha=0.3)

        if not sub_w.empty and sub_w['area_change'].notna().any():
            d = sub_w['day'].values
            ac = sub_w['area_change'].fillna(0).values
            axes[row, 3].plot(d, ac, 'o-', color='#e74c3c', lw=2.5)
            axes[row, 3].fill_between(d, 0, ac, alpha=0.15, color='#e74c3c')
            axes[row, 3].set_xticks(d)
        axes[row, 3].set_title(f"{img} — Worsening Area", fontsize=10)
        axes[row, 3].grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('Day')

    fig.suptitle("Closed-Loop Validation Dashboard", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "combined_dashboard.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  combined_dashboard.png")


# ─────────────────────────────────────────────
# STEP 4: Summary
# ─────────────────────────────────────────────

def print_summary(df):
    import pandas as pd
    print("\n" + "=" * 70)
    print("CLOSED-LOOP VALIDATION SUMMARY")
    print("=" * 70)

    images = df['image'].unique()
    print(f"\nImages analyzed: {len(images)}")
    print(f"Total data points: {len(df)}")

    pass_count = 0
    total_count = 0

    for traj in ['healing', 'worsening']:
        sub = df[df['trajectory'] == traj]
        if sub.empty:
            continue
        print(f"\n{'─' * 35}")
        print(f"  {traj.upper()}")
        print(f"{'─' * 35}")

        for img in images:
            s = sub[sub['image'] == img].sort_values('day')
            if s.empty:
                continue
            day0, dayN = s.iloc[0], s.iloc[-1]
            g_d = dayN['G_pct'] - day0['G_pct']
            s_d = dayN['S_pct'] - day0['S_pct']
            n_d = dayN['N_pct'] - day0['N_pct']

            print(f"\n  {img}  (day {int(day0['day'])} → {int(dayN['day'])})")
            print(f"    G: {day0['G_pct']:5.1f}% → {dayN['G_pct']:5.1f}%  (Δ{g_d:+.1f}%)")
            print(f"    S: {day0['S_pct']:5.1f}% → {dayN['S_pct']:5.1f}%  (Δ{s_d:+.1f}%)")
            print(f"    N: {day0['N_pct']:5.1f}% → {dayN['N_pct']:5.1f}%  (Δ{n_d:+.1f}%)")

            if traj == 'healing':
                par_f = dayN['par'] if pd.notna(dayN['par']) else 0
                print(f"    PAR: {par_f:.1f}%")
                ok = g_d > 0 or n_d < 0
                expect = "G↑ or N↓"
            else:
                ac_f = dayN['area_change'] if pd.notna(dayN['area_change']) else 0
                print(f"    Area expansion: +{ac_f:.1f}%")
                ok = g_d < 0 or n_d > 0
                expect = "G↓ or N↑"

            status = "PASS" if ok else "FAIL"
            print(f"    Trend: [{status}]  (expected {expect})")
            total_count += 1
            if ok:
                pass_count += 1

    print(f"\n{'=' * 70}")
    print(f"RESULT: {pass_count}/{total_count} trend checks passed")
    print(f"{'=' * 70}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    CROP_DIR = os.path.join(BASE_DIR, "cropped_wounds")
    CSV_PATH = os.path.join(BASE_DIR, "evaluation_results.csv")
    PLOT_DIR = os.path.join(BASE_DIR, "evaluation_plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 70)
    print("CLOSED-LOOP EVALUATION PIPELINE")
    print("=" * 70)

    print(f"\n[1/4] Classifying crops in {CROP_DIR}/\n")
    classify_crops(CROP_DIR, CSV_PATH)

    print(f"\n[2/4] Loading {CSV_PATH}")
    df = load_data(CSV_PATH)

    print(f"\n[3/4] Generating plots → {PLOT_DIR}/\n")
    plot_gsn_trends(df, PLOT_DIR)
    plot_area_trends(df, PLOT_DIR)
    plot_combined_dashboard(df, PLOT_DIR)

    print("\n[4/4] Summary")
    print_summary(df)

    print(f"\nDone! Outputs:")
    print(f"  {CSV_PATH}")
    print(f"  {PLOT_DIR}/")