# analysis/plot_results.py
# Generates comparison figures for the thesis and supervisor presentation.
# Reads existing outputs/evaluation/ CSVs — no GPU or new jobs required.
#
# Run from project root:
#   python3 -m analysis.plot_results
#
# Outputs (outputs/analysis/):
#   fig_body_error_comparison.png  — bar chart, 4 models, body MAE + RMSE
#   fig_dvh_errors.png             — grouped bar chart, DVH endpoint errors
#   fig_per_fold_mae.png           — per-fold body MAE line plot
#   fig_acquisition_breakdown.png  — oldAcq vs newAcq breakdown
#   fig_new_metrics.png            — gamma / boundary MAE / isodose Dice
#                                    (only if new eval CSVs are present)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EVAL_DIR = Path("outputs/evaluation")
OUT_DIR  = Path("outputs/analysis")  # overridden by --out-dir

MODELS: dict[str, str] = {
    "U-Net Sigmoid":   "unet3d_ch32_sigmoid_snellius",
    "U-Net Tanh":      "unet3d_ch32_tanh_snellius",
    "DoseGAN Sigmoid": "dosegan_ngf32_sigmoid_snellius",
    "DoseGAN Tanh":    "dosegan_ngf32_tanh_snellius",
}
BASELINE_MODELS: dict[str, str] = {
    "U-Net Sigmoid":   "unet3d_ch32_sigmoid_snellius",
    "DoseGAN Sigmoid": "dosegan_ngf32_sigmoid_snellius",
}
COLORS: dict[str, str] = {
    "U-Net Sigmoid":   "#1565C0",
    "U-Net Tanh":      "#90CAF9",
    "DoseGAN Sigmoid": "#B71C1C",
    "DoseGAN Tanh":    "#EF9A9A",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model(run_name: str) -> pd.DataFrame | None:
    dfs = []
    for fold in range(5):
        p = EVAL_DIR / f"{run_name}_fold{fold}_val.csv"
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        print(f"  WARNING: no eval CSVs found for {run_name} — skipping")
        return None
    return pd.concat(dfs, ignore_index=True)


def fold_means(df: pd.DataFrame, col: str) -> list[float]:
    return [df[df.fold == f][col].mean() for f in range(5) if f in df["fold"].values]


# ---------------------------------------------------------------------------
# Figure 1: body MAE and RMSE — all 4 models
# ---------------------------------------------------------------------------

def fig_body_error(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, metric, ylabel in [
        (axes[0], "body_MAE_Gy",  "Body MAE (Gy)"),
        (axes[1], "body_RMSE_Gy", "Body RMSE (Gy)"),
    ]:
        names, means, stds = [], [], []
        for label, df in data.items():
            if df is None or metric not in df.columns:
                continue
            vals = fold_means(df, metric)
            names.append(label)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))

        x    = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.6,
                      color=[COLORS[n] for n in names],
                      edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title("5-fold mean ± fold std")
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Voxel error comparison — validation set (body-masked)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig_body_error_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: DVH endpoint errors — baseline Sigmoid models
# ---------------------------------------------------------------------------

def fig_dvh_errors(data: dict) -> None:
    dvh_metrics = [
        ("ptv_D95_diff",       "PTV D95"),
        ("ptv_D98_diff",       "PTV D98"),
        ("ptv_Dmean_diff",     "PTV Dmean"),
        ("bladder_Dmean_diff", "Bladder Dmean"),
        ("rectum_Dmean_diff",  "Rectum Dmean"),
        ("rectum_D95_diff",    "Rectum D95"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(dvh_metrics))
    w = 0.35

    for offset, label in zip([-w / 2, w / 2], BASELINE_MODELS):
        df = data.get(label)
        if df is None:
            continue
        means, stds = [], []
        for col, _ in dvh_metrics:
            if col not in df.columns:
                means.append(np.nan); stds.append(np.nan)
                continue
            vals = [df[df.fold == f][col].abs().mean() for f in range(5)
                    if f in df["fold"].values]
            means.append(float(np.nanmean(vals)))
            stds.append(float(np.nanstd(vals)))

        ax.bar(x + offset, means, w, yerr=stds, label=label,
               color=COLORS[label], edgecolor="black", linewidth=0.8, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in dvh_metrics])
    ax.set_ylabel("Mean |error| (Gy)")
    ax.set_title("DVH endpoint errors — 5-fold mean absolute diff (val set)")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig_dvh_errors.png")


# ---------------------------------------------------------------------------
# Figure 3: Per-fold body MAE line plot
# ---------------------------------------------------------------------------

def fig_per_fold(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    for label in BASELINE_MODELS:
        df = data.get(label)
        if df is None:
            continue
        vals = fold_means(df, "body_MAE_Gy")
        folds = list(range(len(vals)))
        ax.plot(folds, vals, "o-", color=COLORS[label], label=label,
                linewidth=2, markersize=7)

    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Fold {i}" for i in range(5)])
    ax.set_ylabel("Body MAE (Gy)")
    ax.set_title("Per-fold body MAE — U-Net Sigmoid vs DoseGAN Sigmoid")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig_per_fold_mae.png")


# ---------------------------------------------------------------------------
# Figure 4: Acquisition-group breakdown
# ---------------------------------------------------------------------------

def fig_acquisition(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in [
        (axes[0], "body_MAE_Gy",  "Body MAE (Gy)"),
        (axes[1], "body_RMSE_Gy", "Body RMSE (Gy)"),
    ]:
        x = np.arange(2)
        w = 0.35
        for offset, label in zip([-w / 2, w / 2], BASELINE_MODELS):
            df = data.get(label)
            if df is None or metric not in df.columns:
                continue
            means, stds, ns = [], [], []
            for grp in ("oldAcq", "newAcq"):
                sub = df[df.acquisition_group == grp][metric].dropna()
                means.append(float(sub.mean()))
                stds.append(float(sub.std()))
                ns.append(len(sub))
            ax.bar(x + offset, means, w, yerr=stds, label=label,
                   color=COLORS[label], edgecolor="black", linewidth=0.8, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(["oldAcq", "newAcq"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"By acquisition group — {ylabel}")
        ax.legend()
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Acquisition-group breakdown — val set (all folds pooled)", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig_acquisition_breakdown.png")


# ---------------------------------------------------------------------------
# Figure 5: New metrics — gamma, boundary MAE, isodose Dice
# (only generated when new eval CSVs are present)
# ---------------------------------------------------------------------------

def fig_new_metrics(data: dict) -> None:
    required = ["gamma_3pct_3mm", "boundary_MAE_ptv_Gy", "Dice_95iso"]
    has_new = any(
        data.get(l) is not None and all(c in data[l].columns for c in required)
        for l in BASELINE_MODELS
    )
    if not has_new:
        print("  New metric columns not yet in eval CSVs — skipping fig_new_metrics.")
        print("  Run eval.sbatch (with GEOM=0) first, then re-run this script.")
        return

    panels = [
        ("gamma_3pct_3mm",       "Gamma 3%/3mm\npass rate (%)",    "%"),
        ("gamma_2pct_2mm",       "Gamma 2%/2mm\npass rate (%)",    "%"),
        ("boundary_MAE_ptv_Gy",  "Boundary MAE\nPTV ±20 mm (Gy)", "Gy"),
        ("Dice_95iso",           "Isodose Dice\n95% level",        ""),
        ("Dice_80iso",           "Isodose Dice\n80% level",        ""),
        ("HD95_95iso_mm",        "Isodose HD95\n95% level (mm)",   "mm"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for ax, (col, title, unit) in zip(axes_flat, panels):
        names, means, stds = [], [], []
        for label in BASELINE_MODELS:
            df = data.get(label)
            if df is None or col not in df.columns:
                continue
            vals = fold_means(df, col)
            names.append(label)
            means.append(float(np.nanmean(vals)))
            stds.append(float(np.nanstd(vals)))

        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, capsize=5, width=0.5,
               color=[COLORS[n] for n in names], edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=10, ha="right", fontsize=9)
        ax.set_ylabel(unit if unit else title.split("\n")[0])
        ax.set_title(title, fontsize=10)
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.001 * max(means, default=1),
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Extended clinical metrics — val set (5-fold mean ± fold std)", fontsize=13)
    fig.tight_layout()
    _save(fig, "fig_new_metrics.png")


# ---------------------------------------------------------------------------
# Figure 6: OAR-level MAE comparison (structure-level)
# ---------------------------------------------------------------------------

def fig_structure_mae(data: dict) -> None:
    struct_metrics = [
        ("ptv_MAE_Gy",     "PTV"),
        ("rectum_MAE_Gy",  "Rectum"),
        ("bladder_MAE_Gy", "Bladder"),
    ]
    has_struct = any(
        data.get(l) is not None and "ptv_MAE_Gy" in data[l].columns
        for l in BASELINE_MODELS
    )
    if not has_struct:
        print("  ptv_MAE_Gy not in eval CSVs — skipping fig_structure_mae.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(struct_metrics))
    w = 0.35

    for offset, label in zip([-w / 2, w / 2], BASELINE_MODELS):
        df = data.get(label)
        if df is None:
            continue
        means, stds = [], []
        for col, _ in struct_metrics:
            if col not in df.columns:
                means.append(np.nan); stds.append(np.nan)
                continue
            vals = fold_means(df, col)
            means.append(float(np.nanmean(vals)))
            stds.append(float(np.nanstd(vals)))
        ax.bar(x + offset, means, w, yerr=stds, label=label,
               color=COLORS[label], edgecolor="black", linewidth=0.8, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in struct_metrics])
    ax.set_ylabel("Structure MAE (Gy)")
    ax.set_title("Per-structure voxel MAE — 5-fold mean ± fold std")
    ax.legend()
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig_structure_mae.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> None:
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", dest="out_dir", default=None,
                        help="Output directory for figures (default: outputs/analysis).")
    args = parser.parse_args()

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading eval CSVs...")
    data: dict[str, pd.DataFrame | None] = {}
    for label, run_name in MODELS.items():
        data[label] = load_model(run_name)

    loaded = [l for l, d in data.items() if d is not None]
    print(f"Loaded: {loaded}\n")

    print("Generating figures...")
    fig_body_error(data)
    fig_dvh_errors(data)
    fig_per_fold(data)
    fig_acquisition(data)
    fig_structure_mae(data)
    fig_new_metrics(data)

    print(f"\nAll figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
