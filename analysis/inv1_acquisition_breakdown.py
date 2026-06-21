# analysis/inv1_acquisition_breakdown.py
# Investigation 1: compare all four conditions across acquisition groups.
#
# For val split: reads outputs/evaluation/{run_name}_fold{0..4}_val.csv (5 folds).
# For test split: reads outputs/evaluation/{run_name}_test.csv (single file; run
#   evaluate.py --split test for each condition first).
#
# Writes:
#   outputs/analysis/inv1_{split}_acquisition_breakdown.csv   — per-condition summary
#   outputs/analysis/inv1_{split}_acquisition_boxplot.png     — boxplot figure
#
# Usage:
#   python3 -m analysis.inv1_acquisition_breakdown --split val
#   python3 -m analysis.inv1_acquisition_breakdown --split test

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional


def _bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    arr = np.array(pvalues, dtype=float)
    n = len(arr)
    sort_idx = np.argsort(arr)
    sorted_p = arr[sort_idx]
    adj = sorted_p * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    result = np.empty(n)
    result[sort_idx] = adj
    return result

EVAL_DIR = Path("outputs/evaluation")
OUT_DIR  = Path("outputs/analysis")
FOLDS    = [0, 1, 2, 3, 4]
GROUPS   = ["oldAcq", "newAcq"]

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


def load_condition(run_name: str, split: str) -> Optional[pd.DataFrame]:
    if split == "test":
        path = EVAL_DIR / f"{run_name}_test.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found — run evaluate.py --split test first")
            return None
        return pd.read_csv(path)

    # val: concatenate all five fold CSVs
    paths = sorted(EVAL_DIR.glob(f"{run_name}_fold*_val.csv"))
    if not paths:
        print(f"  WARNING: no val CSVs found for {run_name}")
        return None
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def summarise_condition(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Per-group summary; for val, show fold-level breakdown too."""
    if split == "val":
        group_cols = ["fold", "acquisition_group"]
    else:
        group_cols = ["acquisition_group"]
    return (
        df.groupby(group_cols)["body_MAE_Gy"]
        .agg(["count", "mean", "std", "median",
              lambda s: s.quantile(0.25),
              lambda s: s.quantile(0.75)])
        .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
    )


def stat_test_condition(df: pd.DataFrame) -> dict:
    old = df.loc[df["acquisition_group"] == "oldAcq", "body_MAE_Gy"].values
    new = df.loc[df["acquisition_group"] == "newAcq", "body_MAE_Gy"].values
    u, p = stats.mannwhitneyu(old, new, alternative="two-sided")
    return {
        "n_old":      len(old),
        "n_new":      len(new),
        "mean_old":   float(old.mean()),
        "mean_new":   float(new.mean()),
        "median_old": float(np.median(old)),
        "median_new": float(np.median(new)),
        "delta_mean": float(old.mean() - new.mean()),
        "u_statistic": float(u),
        "p_value":    float(p),
    }


def boxplot_all(data: dict, split: str, out_path: Path) -> None:
    cond_names = [k for k, v in data.items() if v is not None]
    n_cond = len(cond_names)
    if n_cond == 0:
        return

    fig, axes = plt.subplots(1, n_cond, figsize=(4.5 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for ax, cname in zip(axes, cond_names):
        df = data[cname]
        vals = [df.loc[df["acquisition_group"] == g, "body_MAE_Gy"].values for g in GROUPS]
        bp = ax.boxplot(vals, labels=GROUPS, showmeans=True,
                        patch_artist=True)
        color = COLORS.get(cname, "#555555")
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(cname, fontsize=10, fontweight="bold")
        ax.set_ylabel("Body-masked MAE (Gy)" if cname == cond_names[0] else "")
        ax.grid(axis="y", alpha=0.3)

        t = stat_test_condition(df)
        sig = "p<0.05" if t["p_value"] < 0.05 else f"p={t['p_value']:.3f}"
        ax.set_xlabel(f"MWU {sig}\nΔmean={t['delta_mean']:+.3f} Gy", fontsize=8)

    fig.suptitle(
        f"Body-masked MAE by acquisition group — {split} split\n"
        "(pooled across folds for val; single pass for test)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Which split to evaluate. Use 'test' only after model "
                             "selection is complete and evaluate.py --split test has run.")
    args = parser.parse_args()
    split = args.split

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {split} CSVs for all four conditions...")
    data = {name: load_condition(run, split) for name, run in CONDITIONS.items()}
    for name, df in data.items():
        status = f"{len(df)} rows" if df is not None else "MISSING"
        print(f"  {name:<20s}: {status}")

    # Per-condition summary + Mann-Whitney U
    summary_rows = []
    for cname, df in data.items():
        if df is None:
            continue
        print(f"\n── {cname} ──")
        print(summarise_condition(df, split).to_string())
        t = stat_test_condition(df)
        print(
            f"  oldAcq (n={t['n_old']}): mean={t['mean_old']:.4f} Gy  "
            f"median={t['median_old']:.4f} Gy"
        )
        print(
            f"  newAcq (n={t['n_new']}): mean={t['mean_new']:.4f} Gy  "
            f"median={t['median_new']:.4f} Gy"
        )
        print(f"  mean(oldAcq) − mean(newAcq) = {t['delta_mean']:+.4f} Gy  MWU p={t['p_value']:.4f}")
        summary_rows.append({"condition": cname, **t})

    # BH-FDR correction across the four MWU tests.
    # Testing four conditions inflates the family-wise error rate; correcting
    # here prevents treating any single nominally significant result as firm evidence.
    if summary_rows:
        raw_p = np.array([r["p_value"] for r in summary_rows])
        adj_p = _bh_fdr(raw_p)
        for r, ap in zip(summary_rows, adj_p):
            r["p_value_fdr"] = round(float(ap), 4)

        print("\n── BH-FDR correction across four conditions ──")
        for r in summary_rows:
            sig = "significant" if r["p_value_fdr"] < 0.05 else "not significant"
            print(
                f"  {r['condition']:<20s}  raw p={r['p_value']:.4f}  "
                f"FDR p={r['p_value_fdr']:.4f}  ({sig} at α=0.05)"
            )

        out_csv = OUT_DIR / f"inv1_{split}_acquisition_breakdown.csv"
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        print(f"\n  Summary saved: {out_csv}")

    boxplot_all(data, split, OUT_DIR / f"inv1_{split}_acquisition_boxplot.png")


if __name__ == "__main__":
    main()
