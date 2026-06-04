"""
analysis/clinical_report.py
============================
Structured results report separating four clinically distinct categories:

  1. Global dose accuracy         — body/structure voxel-level error
  2a. DVH accuracy (|error|)      — mean absolute DVH endpoint error
  2b. DVH bias (signed mean)      — mean signed DVH endpoint error (+ = over-prediction)
  3. Spatial boundary quality     — boundary MAE, isodose Dice, HD95

Each section reports fold mean ± fold std across 5 folds for all 4 conditions.
Separating accuracy from bias lets you distinguish "how close" from "which direction".

Usage:
    PYTHONPATH=. python3 -m analysis.clinical_report

Output:
    outputs/analysis/clinical_report.csv     — full table (machine-readable)
    outputs/analysis/clinical_report.png     — four-panel summary figure
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_DIR      = Path("outputs/evaluation")
EVAL_ARCHIVED = Path("outputs/evaluation/archived_ablations")
OUT_DIR       = Path("outputs/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = {
    "U-Net base":   "unet3d_ch32_sigmoid_snellius",
    "U-Net geom":   "unet3d_ch32_sigmoid_geom_snellius",
    "DoseGAN base": "dosegan_ngf32_sigmoid_snellius",
    "DoseGAN geom": "dosegan_ngf32_sigmoid_geom_snellius",
}

COLORS = {
    "U-Net base":   "#1565C0",
    "U-Net geom":   "#1A3D6E",
    "DoseGAN base": "#B71C1C",
    "DoseGAN geom": "#7A2E00",
}

# ── Metric definitions ─────────────────────────────────────────────────────
# (column_name, display_label, unit, take_abs)
# take_abs=True  → report mean |value| (for signed diffs)
# take_abs=False → report mean value directly (for MAE/RMSE already positive)

GLOBAL_ACCURACY = [
    ("body_MAE_Gy",       "Body MAE",       "Gy",  False),
    ("body_RMSE_Gy",      "Body RMSE",      "Gy",  False),
    ("ptv_MAE_Gy",        "PTV MAE",        "Gy",  False),
    ("rectum_MAE_Gy",     "Rectum MAE",     "Gy",  False),
    ("bladder_MAE_Gy",    "Bladder MAE",    "Gy",  False),
]

DVH_ACCURACY = [
    ("ptv_D95_diff",          "PTV D95 |err|",       "Gy",  True),
    ("ptv_D98_diff",          "PTV D98 |err|",       "Gy",  True),
    ("ptv_Dmean_diff",        "PTV Dmean |err|",     "Gy",  True),
    ("ptv_D01cc_diff",        "PTV D0.1cc |err|",    "Gy",  True),
    ("rectum_Dmean_diff",     "Rectum Dmean |err|",  "Gy",  True),
    ("rectum_D01cc_diff",     "Rectum D0.1cc |err|", "Gy",  True),
    ("V_presc_rectum_diff",   "Rectum V_Rx |err|",   "%",   True),
    ("bladder_Dmean_diff",    "Bladder Dmean |err|", "Gy",  True),
    ("bladder_D01cc_diff",    "Bladder D0.1cc |err|","Gy",  True),
    ("V_presc_bladder_diff",  "Bladder V_Rx |err|",  "%",   True),
]

DVH_BIAS = [
    ("ptv_D95_diff",          "PTV D95 bias",        "Gy",  False),
    ("ptv_D98_diff",          "PTV D98 bias",        "Gy",  False),
    ("ptv_Dmean_diff",        "PTV Dmean bias",      "Gy",  False),
    ("ptv_D01cc_diff",        "PTV D0.1cc bias",     "Gy",  False),
    ("rectum_Dmean_diff",     "Rectum Dmean bias",   "Gy",  False),
    ("rectum_D01cc_diff",     "Rectum D0.1cc bias",  "Gy",  False),
    ("V_presc_rectum_diff",   "Rectum V_Rx bias",    "%",   False),
    ("bladder_Dmean_diff",    "Bladder Dmean bias",  "Gy",  False),
    ("bladder_D01cc_diff",    "Bladder D0.1cc bias", "Gy",  False),
    ("V_presc_bladder_diff",  "Bladder V_Rx bias",   "%",   False),
]

SPATIAL_BOUNDARY = [
    ("boundary_MAE_ptv_Gy",     "PTV boundary MAE",    "Gy",  False),
    ("boundary_MAE_rectum_Gy",  "Rectum boundary MAE", "Gy",  False),
    ("boundary_MAE_bladder_Gy", "Bladder boundary MAE","Gy",  False),
    ("Dice_100iso",             "Dice 100% iso",       "",    False),
    ("Dice_95iso",              "Dice 95% iso",        "",    False),
    ("Dice_80iso",              "Dice 80% iso",        "",    False),
    ("Dice_50iso",              "Dice 50% iso",        "",    False),
    ("HD95_100iso_mm",          "HD95 100% iso",       "mm",  False),
]

SECTIONS = [
    ("global",    "1. Global dose accuracy",          GLOBAL_ACCURACY),
    ("dvh_acc",   "2a. DVH accuracy (mean |error|)",  DVH_ACCURACY),
    ("dvh_bias",  "2b. DVH bias (mean signed error)",  DVH_BIAS),
    ("spatial",   "3. Spatial boundary quality",      SPATIAL_BOUNDARY),
]


def load_condition(run_name: str):
    for base in [EVAL_DIR, EVAL_ARCHIVED]:
        paths = sorted(base.glob(f"{run_name}_fold*_val.csv"))
        if paths:
            return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return None


def fold_mean_std(df: pd.DataFrame, col: str, take_abs: bool):
    """Per-fold mean, then report mean ± std of those fold means."""
    if col not in df.columns:
        return float("nan"), float("nan")
    vals = df[col].abs() if take_abs else df[col]
    fold_means = vals.groupby(df["fold"]).mean().dropna()
    if len(fold_means) == 0:
        return float("nan"), float("nan")
    return fold_means.mean(), fold_means.std(ddof=1)


def build_section_table(metrics, data):
    rows = []
    for col, label, unit, take_abs in metrics:
        row = {"metric": label, "unit": unit}
        for cond_name, df in data.items():
            if df is None:
                row[f"{cond_name}_mean"] = float("nan")
                row[f"{cond_name}_std"]  = float("nan")
            else:
                m, s = fold_mean_std(df, col, take_abs)
                row[f"{cond_name}_mean"] = m
                row[f"{cond_name}_std"]  = s
        rows.append(row)
    return pd.DataFrame(rows)


def print_section(title: str, table: pd.DataFrame, cond_names: list) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    header = f"{'Metric':<28s}" + "".join(f"  {c:<20s}" for c in cond_names)
    print(header)
    for _, r in table.iterrows():
        unit = f" {r['unit']}" if r['unit'] else ""
        line = f"  {r['metric']:<26s}"
        for c in cond_names:
            m, s = r[f"{c}_mean"], r[f"{c}_std"]
            if np.isnan(m):
                line += f"  {'—':<20s}"
            else:
                line += f"  {m:.3f} ± {s:.3f}{unit:<5s}    "
        print(line)


def plot_sections(section_tables: dict, cond_names: list) -> None:
    n_sections = len(section_tables)
    fig = plt.figure(figsize=(18, 5 * n_sections))
    gs  = gridspec.GridSpec(n_sections, 1, hspace=0.5)

    for row_idx, (sec_key, (title, table)) in enumerate(section_tables.items()):
        ax = fig.add_subplot(gs[row_idx])
        metrics  = table["metric"].tolist()
        n_met    = len(metrics)
        n_cond   = len(cond_names)
        x        = np.arange(n_met)
        width    = 0.8 / n_cond

        for i, cond in enumerate(cond_names):
            means = table[f"{cond}_mean"].values
            stds  = table[f"{cond}_std"].values
            valid = ~np.isnan(means)
            ax.bar(x[valid] + i * width - 0.4 + width / 2,
                   means[valid], width,
                   yerr=stds[valid], capsize=3,
                   color=COLORS[cond], alpha=0.85,
                   label=cond, error_kw={"elinewidth": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Mean ± fold std", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(-0.5, n_met - 0.5)

    out = OUT_DIR / "clinical_report.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {out}")


def main():
    print("Loading conditions...")
    data = {name: load_condition(run) for name, run in CONDITIONS.items()}
    for name, df in data.items():
        status = f"{len(df)} patients" if df is not None else "MISSING"
        print(f"  {name:<20s}: {status}")

    cond_names = list(CONDITIONS.keys())
    section_tables = {}
    all_rows = []

    for sec_key, title, metrics in SECTIONS:
        table = build_section_table(metrics, data)
        table["section"] = title
        section_tables[sec_key] = (title, table)
        all_rows.append(table)
        print_section(title, table, cond_names)

    # Save full CSV
    full = pd.concat(all_rows, ignore_index=True)
    out_csv = OUT_DIR / "clinical_report.csv"
    full.to_csv(out_csv, index=False)
    print(f"\n  Table saved: {out_csv}")

    plot_sections(section_tables, cond_names)


if __name__ == "__main__":
    main()
