# analysis/plot_geom_comparison.py
# SRQ 2 — Effect of geometric input channels.
# Compares baseline (9-ch) vs geom (14-ch) for both U-Net and DoseGAN.
#
# Reads:  outputs/evaluation/{run_name}_fold{0..4}_val.csv
#         Falls back to outputs/evaluation/baseline_sigmoid/ for baseline CSVs.
# Writes: outputs/analysis/srq2_geom_comparison/
#   fig_geom_overall.png     — MAE_body and RMSE_body: baseline vs geom per model
#   fig_geom_boundary.png    — boundary MAE (PTV/Rectum/Bladder): baseline vs geom
#   fig_geom_dvh.png         — DVH errors (D95/Dmean/D0.1cc): baseline vs geom
#   fig_geom_improvement.png — boundary vs global improvement ratio (SRQ 2 key result)
#
# Usage:
#   python3 -m analysis.plot_geom_comparison

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EVAL_DIR      = Path("outputs/evaluation")
EVAL_BASELINE = Path("outputs/evaluation/baseline_sigmoid")
OUT_DIR       = Path("outputs/analysis/srq2_geom_comparison")
DOSE_SCALE    = 50.0

CONDITIONS = {
    "U-Net baseline":  "unet3d_ch32_sigmoid_snellius",
    "U-Net + geom":    "unet3d_ch32_sigmoid_geom_snellius",
    "DoseGAN baseline":"dosegan_ngf32_sigmoid_snellius",
    "DoseGAN + geom":  "dosegan_ngf32_sigmoid_geom_snellius",
}

COLORS = {
    "U-Net baseline":   "#4878CF",
    "U-Net + geom":     "#1A3D6E",
    "DoseGAN baseline": "#E87025",
    "DoseGAN + geom":   "#7A2E00",
}


def load_run(run_name: str) -> pd.DataFrame | None:
    """Load and concatenate fold CSVs for one run. Returns None if missing."""
    # Try root eval dir first, then baseline subdir
    for base in [EVAL_DIR, EVAL_BASELINE]:
        paths = sorted(base.glob(f"{run_name}_fold*_val.csv"))
        if paths:
            return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return None


def fold_stats(run_name: str, col: str):
    """Return (mean, std) across fold-level means, or (nan, nan) if missing."""
    for base in [EVAL_DIR, EVAL_BASELINE]:
        paths = sorted(base.glob(f"{run_name}_fold*_val.csv"))
        if paths:
            fold_means = [pd.read_csv(p)[col].mean() for p in paths if col in pd.read_csv(p).columns]
            if fold_means:
                return float(np.mean(fold_means)), float(np.std(fold_means))
    return float("nan"), float("nan")


def _bar_group(ax, metrics, labels, conditions, ylabel, title):
    """Generic grouped bar chart helper."""
    x = np.arange(len(metrics))
    n = len(conditions)
    w = 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w

    for offset, (label, run_name) in zip(offsets, conditions.items()):
        means, stds = [], []
        for col in metrics.values():
            m, s = fold_stats(run_name, col)
            means.append(m)
            stds.append(s)
        if all(np.isnan(m) for m in means):
            continue
        ax.bar(x + offset, means, w, yerr=stds, label=label,
               color=COLORS[label], edgecolor="black", linewidth=0.7, capsize=3,
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT_DIR / name}")


def fig_overall():
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar_group(ax,
               {"body MAE": "body_MAE_Gy", "body RMSE": "body_RMSE_Gy"},
               None, CONDITIONS, "Gy",
               "SRQ 2 — Overall accuracy: baseline vs geometric channels")
    _save(fig, "fig_geom_overall.png")


def fig_boundary():
    fig, ax = plt.subplots(figsize=(8, 4))
    _bar_group(ax,
               {"Boundary\nPTV":     "boundary_MAE_ptv_Gy",
                "Boundary\nRectum":  "boundary_MAE_rectum_Gy",
                "Boundary\nBladder": "boundary_MAE_bladder_Gy"},
               None, CONDITIONS, "Gy",
               "SRQ 2 — Boundary MAE (±20 mm): baseline vs geometric channels")
    _save(fig, "fig_geom_boundary.png")


def fig_dvh():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # PTV DVH
    _bar_group(axes[0],
               {"PTV D95":    "ptv_D95_diff",
                "PTV Dmean":  "ptv_Dmean_diff",
                "PTV D0.1cc": "ptv_D01cc_diff"},
               None, CONDITIONS, "Error (Gy)",
               "SRQ 3 — PTV DVH errors: baseline vs geom")

    # OAR DVH
    _bar_group(axes[1],
               {"Rectum D0.1cc":  "rectum_D01cc_diff",
                "Bladder D0.1cc": "bladder_D01cc_diff"},
               None, CONDITIONS, "Error (Gy)",
               "SRQ 3 — OAR DVH errors: baseline vs geom")

    _save(fig, "fig_geom_dvh.png")


def fig_improvement_ratio():
    """
    SRQ 2 key result: is improvement from geom larger at boundaries than globally?
    Plots delta_boundary_MAE / delta_body_MAE per model (>1 = boundary-specific benefit).
    """
    pairs = [
        ("U-Net",    "unet3d_ch32_sigmoid_snellius",    "unet3d_ch32_sigmoid_geom_snellius"),
        ("DoseGAN",  "dosegan_ngf32_sigmoid_snellius",  "dosegan_ngf32_sigmoid_geom_snellius"),
    ]
    structures = ["PTV", "Rectum", "Bladder"]
    boundary_cols = ["boundary_MAE_ptv_Gy", "boundary_MAE_rectum_Gy", "boundary_MAE_bladder_Gy"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(structures))
    w = 0.3

    for offset, (model_label, base_run, geom_run) in zip([-w / 2, w / 2], pairs):
        global_base, _ = fold_stats(base_run,  "body_MAE_Gy")
        global_geom, _ = fold_stats(geom_run,  "body_MAE_Gy")
        delta_global = global_base - global_geom   # positive = geom is better

        ratios = []
        for col in boundary_cols:
            bnd_base, _ = fold_stats(base_run, col)
            bnd_geom, _ = fold_stats(geom_run, col)
            delta_bnd = bnd_base - bnd_geom
            if np.isnan(delta_global) or delta_global == 0 or np.isnan(delta_bnd):
                ratios.append(float("nan"))
            else:
                ratios.append(delta_bnd / delta_global)

        color = COLORS[f"{model_label} baseline"]
        ax.bar(x + offset, ratios, w, label=model_label, color=color,
               edgecolor="black", linewidth=0.7, alpha=0.85)

    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="equal improvement")
    ax.set_xticks(x)
    ax.set_xticklabels(structures)
    ax.set_ylabel("Boundary Δ / Global Δ")
    ax.set_title("SRQ 2 — Is geometric channel benefit boundary-specific?\n"
                 "(ratio > 1 = larger improvement at boundary than globally)")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    _save(fig, "fig_geom_improvement_ratio.png")


if __name__ == "__main__":
    data_available = {label: load_run(rn) is not None for label, rn in CONDITIONS.items()}
    print("Data available:")
    for label, avail in data_available.items():
        print(f"  {'✓' if avail else '✗ (missing — run eval first)':<35} {label}")
    print()

    fig_overall()
    fig_boundary()
    fig_dvh()
    fig_improvement_ratio()
    print("\nDone.")
