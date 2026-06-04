"""
analysis/paired_comparison.py
==============================
Paired patient-level comparison across the four conditions.

Because every validation patient is evaluated by all four models
(once per fold), we can compute per-patient differences and run
paired tests. This is stronger than comparing means alone.

Comparisons performed (A − B; negative = A is better):
  1. U-Net geom    vs  U-Net baseline    (geom effect, U-Net)
  2. DoseGAN geom  vs  DoseGAN baseline  (geom effect, DoseGAN)
  3. DoseGAN base  vs  U-Net baseline    (model effect, baseline)
  4. DoseGAN geom  vs  U-Net geom        (model effect, geom)

For each comparison and each metric:
  - Mean difference ± SD
  - 95% bootstrap CI of the mean difference (n_boot=10000)
  - Paired t-test  p-value
  - Wilcoxon signed-rank p-value
  - Proportion of patients where A < B (A wins)

Usage:
    PYTHONPATH=. python3 -m analysis.paired_comparison
    PYTHONPATH=. python3 -m analysis.paired_comparison --metrics body_MAE_Gy ptv_MAE_Gy bladder_MAE_Gy

Output:
    outputs/analysis/paired_comparison.csv
    outputs/analysis/paired_comparison_plot.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_DIR      = Path("outputs/evaluation")
EVAL_ARCHIVED = Path("outputs/evaluation/archived_ablations")
OUT_DIR       = Path("outputs/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 10_000
ALPHA  = 0.05

CONDITIONS = {
    "unet_base":    "unet3d_ch32_sigmoid_snellius",
    "unet_geom":    "unet3d_ch32_sigmoid_geom_snellius",
    "dgan_base":    "dosegan_ngf32_sigmoid_snellius",
    "dgan_geom":    "dosegan_ngf32_sigmoid_geom_snellius",
}

COMPARISONS = [
    # label,            A,           B,             description
    ("unet_geom_effect", "unet_geom", "unet_base",  "U-Net geom − U-Net baseline"),
    ("dgan_geom_effect", "dgan_geom", "dgan_base",  "DoseGAN geom − DoseGAN baseline"),
    ("model_base",       "dgan_base", "unet_base",  "DoseGAN baseline − U-Net baseline"),
    ("model_geom",       "dgan_geom", "unet_geom",  "DoseGAN geom − U-Net geom"),
]

DEFAULT_METRICS = [
    "body_MAE_Gy",
    "body_RMSE_Gy",
    "ptv_MAE_Gy",
    "rectum_MAE_Gy",
    "bladder_MAE_Gy",
    "ptv_D95_diff",
    "ptv_Dmean_diff",
    "rectum_Dmean_diff",
    "bladder_Dmean_diff",
]


def load_condition(run_name: str):
    """Load and concatenate all 5 fold CSVs for one condition."""
    for base in [EVAL_DIR, EVAL_ARCHIVED]:
        paths = sorted(base.glob(f"{run_name}_fold*_val.csv"))
        if paths:
            return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"  WARNING: no eval CSVs for {run_name} — skipping")
    return None


def bootstrap_ci(deltas: np.ndarray, n_boot: int = N_BOOT, alpha: float = ALPHA):
    """Bootstrap percentile CI of the mean difference."""
    rng = np.random.default_rng(42)
    means = [rng.choice(deltas, size=len(deltas), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


def paired_stats(a: np.ndarray, b: np.ndarray, metric: str) -> dict:
    """Compute full paired comparison statistics for one metric."""
    deltas = a - b   # positive = A worse than B
    n      = len(deltas)

    mean_d = deltas.mean()
    std_d  = deltas.std(ddof=1)
    ci_lo, ci_hi = bootstrap_ci(deltas)

    t_stat, t_p = stats.ttest_rel(a, b)

    # Wilcoxon requires non-zero differences
    nonzero = deltas[deltas != 0]
    if len(nonzero) >= 10:
        _, w_p = stats.wilcoxon(a, b, alternative="two-sided")
    else:
        w_p = float("nan")

    a_wins = (a < b).mean()   # fraction where A < B (lower MAE = better)

    return {
        "metric":       metric,
        "n_patients":   n,
        "mean_diff":    round(mean_d, 4),
        "std_diff":     round(std_d, 4),
        "ci_lo_95":     round(ci_lo, 4),
        "ci_hi_95":     round(ci_hi, 4),
        "t_stat":       round(t_stat, 3),
        "t_pvalue":     round(t_p, 4),
        "wilcoxon_p":   round(w_p, 4) if not np.isnan(w_p) else "n/a",
        "pct_A_wins":   round(a_wins * 100, 1),
    }


def run_comparison(
    label: str,
    key_a: str,
    key_b: str,
    description: str,
    data: dict,
    metrics: list,
) -> list:
    df_a = data.get(key_a)
    df_b = data.get(key_b)
    if df_a is None or df_b is None:
        print(f"  Skipping {description} — data missing")
        return []

    # Inner join on patient_id to ensure pairing
    merged = df_a[["patient_id"] + metrics].merge(
        df_b[["patient_id"] + metrics],
        on="patient_id", suffixes=("_a", "_b"),
    ).dropna()

    if len(merged) == 0:
        print(f"  Skipping {description} — no matching patients")
        return []

    print(f"\n  {description}  (n={len(merged)} matched patients)")

    rows = []
    for m in metrics:
        if f"{m}_a" not in merged.columns:
            continue
        a = merged[f"{m}_a"].values
        b = merged[f"{m}_b"].values
        s = paired_stats(a, b, m)
        s["comparison"] = label
        s["description"] = description
        rows.append(s)

        sig_t = "**" if s["t_pvalue"] < 0.01 else ("*" if s["t_pvalue"] < 0.05 else "")
        print(
            f"    {m:<30s}  Δ={s['mean_diff']:+.4f} ± {s['std_diff']:.4f} Gy"
            f"  95%CI=[{s['ci_lo_95']:+.4f}, {s['ci_hi_95']:+.4f}]"
            f"  t_p={s['t_pvalue']:.4f}{sig_t}"
            f"  A_wins={s['pct_A_wins']}%"
        )
    return rows


def plot_paired_differences(all_rows: list, metrics: list) -> None:
    """Forest-plot-style: mean difference ± 95% CI per comparison per metric."""
    df = pd.DataFrame(all_rows)
    df = df[df["metric"].isin(metrics)]

    comparisons = [c[0] for c in COMPARISONS]
    descs       = {c[0]: c[3] for c in COMPARISONS}
    n_comp      = len(comparisons)
    n_met       = len(metrics)

    fig, axes = plt.subplots(1, n_met, figsize=(4 * n_met, max(3, n_comp + 1)),
                             sharey=True)
    if n_met == 1:
        axes = [axes]

    colors = ["#1565C0", "#B71C1C", "#2E7D32", "#E65100"]

    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric]
        for i, cmp in enumerate(comparisons):
            row = sub[sub["comparison"] == cmp]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.errorbar(
                row["mean_diff"], i,
                xerr=[[row["mean_diff"] - row["ci_lo_95"]],
                      [row["ci_hi_95"] - row["mean_diff"]]],
                fmt="o", color=colors[i], capsize=4, markersize=6,
            )
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_title(metric.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("Mean diff (A−B, Gy)", fontsize=8)
        ax.tick_params(labelsize=8)

    axes[0].set_yticks(range(n_comp))
    axes[0].set_yticklabels([descs[c] for c in comparisons], fontsize=8)
    fig.suptitle("Paired comparisons — mean difference ± 95% bootstrap CI\n"
                 "(negative = A better; * p<0.05, ** p<0.01 paired t-test)",
                 fontsize=10)
    plt.tight_layout()
    out = OUT_DIR / "paired_comparison_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    args = parser.parse_args()
    metrics = args.metrics

    print("Loading conditions...")
    data = {k: load_condition(v) for k, v in CONDITIONS.items()}

    # Filter metrics to those present in the data
    available = set()
    for df in data.values():
        if df is not None:
            available |= set(df.columns)
    metrics = [m for m in metrics if m in available]
    if not metrics:
        print("No requested metrics found in eval CSVs.")
        sys.exit(1)
    print(f"Metrics: {metrics}")

    all_rows = []
    for label, key_a, key_b, desc in COMPARISONS:
        rows = run_comparison(label, key_a, key_b, desc, data, metrics)
        all_rows.extend(rows)

    if not all_rows:
        print("No comparisons could be run — check eval CSVs.")
        sys.exit(1)

    out_csv = OUT_DIR / "paired_comparison.csv"
    cols = ["comparison", "description", "metric", "n_patients",
            "mean_diff", "std_diff", "ci_lo_95", "ci_hi_95",
            "t_stat", "t_pvalue", "wilcoxon_p", "pct_A_wins"]
    pd.DataFrame(all_rows)[cols].to_csv(out_csv, index=False)
    print(f"\n  Table saved: {out_csv}")

    plot_paired_differences(all_rows, metrics[:5])  # plot first 5 metrics


if __name__ == "__main__":
    main()
