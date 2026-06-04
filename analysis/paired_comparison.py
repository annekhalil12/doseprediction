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

# Signed-difference columns in the eval CSVs (pred − true).
# For these metrics paired_comparison reports BOTH the signed version (bias)
# and the absolute version (accuracy) automatically.
SIGNED_DIFF_METRICS = frozenset({
    "ptv_D95_diff", "ptv_D98_diff", "ptv_Dmean_diff", "ptv_D01cc_diff",
    "rectum_Dmean_diff", "rectum_D01cc_diff",
    "bladder_Dmean_diff", "bladder_D01cc_diff",
    "V_presc_rectum_diff", "V_presc_bladder_diff",
})

# True  = lower is better (MAE, RMSE, HD95, leakage)
# False = higher is better (Dice)
# Metrics absent from this dict default to True.
METRIC_LOWER_IS_BETTER: dict[str, bool] = {
    "body_MAE_Gy":             True,
    "body_RMSE_Gy":            True,
    "ptv_MAE_Gy":              True,
    "rectum_MAE_Gy":           True,
    "bladder_MAE_Gy":          True,
    "ptv_D95_diff":            True,
    "ptv_Dmean_diff":          True,
    "rectum_Dmean_diff":       True,
    "bladder_Dmean_diff":      True,
    "boundary_MAE_ptv_Gy":     True,
    "boundary_MAE_rectum_Gy":  True,
    "boundary_MAE_bladder_Gy": True,
    "HD95_100iso_mm":          True,
    "HD95_95iso_mm":           True,
    "HD95_80iso_mm":           True,
    "HD95_50iso_mm":           True,
    "Dice_100iso":             False,
    "Dice_95iso":              False,
    "Dice_80iso":              False,
    "Dice_50iso":              False,
    "leakage_mean_pred_Gy":    True,
    "leakage_vol_frac":        True,
}

METRIC_UNITS: dict[str, str] = {
    "body_MAE_Gy":             "Gy",
    "body_RMSE_Gy":            "Gy",
    "ptv_MAE_Gy":              "Gy",
    "rectum_MAE_Gy":           "Gy",
    "bladder_MAE_Gy":          "Gy",
    "ptv_D95_diff":            "Gy",
    "ptv_Dmean_diff":          "Gy",
    "rectum_Dmean_diff":       "Gy",
    "bladder_Dmean_diff":      "Gy",
    "boundary_MAE_ptv_Gy":     "Gy",
    "boundary_MAE_rectum_Gy":  "Gy",
    "boundary_MAE_bladder_Gy": "Gy",
    "HD95_100iso_mm":          "mm",
    "HD95_95iso_mm":           "mm",
    "HD95_80iso_mm":           "mm",
    "HD95_50iso_mm":           "mm",
    "Dice_100iso":             "",
    "Dice_95iso":              "",
    "Dice_80iso":              "",
    "Dice_50iso":              "",
    "leakage_mean_pred_Gy":    "Gy",
    "leakage_vol_frac":        "",
}

# Automatically register abs_ versions of all signed-diff metrics.
for _m in SIGNED_DIFF_METRICS:
    METRIC_LOWER_IS_BETTER[f"abs_{_m}"] = True
    METRIC_UNITS[f"abs_{_m}"]           = METRIC_UNITS.get(_m, "Gy")

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
    "Dice_100iso",
    "Dice_50iso",
    "HD95_100iso_mm",
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


def _bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (preserves NaN positions)."""
    arr = np.array(pvalues, dtype=float)
    valid_mask = ~np.isnan(arr)
    if not valid_mask.any():
        return arr
    valid_p = arr[valid_mask]
    n = len(valid_p)
    sort_idx  = np.argsort(valid_p)
    sorted_p  = valid_p[sort_idx]
    adj       = sorted_p * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    adj_unsorted = np.empty(n)
    adj_unsorted[sort_idx] = adj
    result = np.full(len(arr), np.nan)
    result[valid_mask] = adj_unsorted
    return result


def paired_stats(a: np.ndarray, b: np.ndarray, metric: str,
                 lower_is_better: bool = True) -> dict:
    """Compute full paired comparison statistics for one metric.

    Effect size: Cohen's dz = mean(delta) / sd(delta) — the paired equivalent
    of Cohen's d. Benchmarks: small=0.2, medium=0.5, large=0.8.

    Normality: Shapiro-Wilk on the deltas. If shapiro_p < 0.05 the distribution
    of differences is non-normal and the Wilcoxon p-value should be preferred
    over the t-test p-value when reporting.
    """
    deltas = a - b
    n      = len(deltas)

    mean_d = deltas.mean()
    std_d  = deltas.std(ddof=1)
    ci_lo, ci_hi = bootstrap_ci(deltas)

    t_stat, t_p = stats.ttest_rel(a, b)

    nonzero = deltas[deltas != 0]
    if len(nonzero) >= 10:
        _, w_p = stats.wilcoxon(a, b, alternative="two-sided")
    else:
        w_p = float("nan")

    # Cohen's dz for paired data
    cohens_dz = float(mean_d / std_d) if std_d > 1e-12 else 0.0

    # Shapiro-Wilk on deltas (requires 3 ≤ n ≤ 5000)
    if 3 <= n <= 5000:
        _, shapiro_p = stats.shapiro(deltas)
    else:
        shapiro_p = float("nan")

    # A wins when A is better: lower for error metrics, higher for Dice.
    a_wins = (a < b).mean() if lower_is_better else (a > b).mean()

    return {
        "metric":           metric,
        "lower_is_better":  lower_is_better,
        "n_patients":       n,
        "mean_diff":        round(mean_d, 4),
        "std_diff":         round(std_d, 4),
        "ci_lo_95":         round(ci_lo, 4),
        "ci_hi_95":         round(ci_hi, 4),
        "cohens_dz":        round(cohens_dz, 3),
        "shapiro_p":        round(shapiro_p, 4) if not np.isnan(shapiro_p) else "n/a",
        "t_stat":           round(t_stat, 3),
        "t_pvalue":         round(t_p, 4),
        "wilcoxon_p":       round(w_p, 4) if not np.isnan(w_p) else "n/a",
        # FDR-adjusted p-values are filled in after all rows are collected.
        "t_pvalue_fdr":     float("nan"),
        "wilcoxon_p_fdr":   float("nan"),
        "pct_A_wins":       round(a_wins * 100, 1),
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

    # Merge on base columns only (abs_ versions are computed below).
    base_metrics = [m for m in metrics if not m.startswith("abs_")]
    merged = df_a[["patient_id"] + base_metrics].merge(
        df_b[["patient_id"] + base_metrics],
        on="patient_id", suffixes=("_a", "_b"),
    ).dropna()

    if len(merged) == 0:
        print(f"  Skipping {description} — no matching patients")
        return []

    # Compute abs_ columns for every signed-diff metric present.
    for m in SIGNED_DIFF_METRICS:
        if f"{m}_a" in merged.columns:
            merged[f"abs_{m}_a"] = merged[f"{m}_a"].abs()
            merged[f"abs_{m}_b"] = merged[f"{m}_b"].abs()

    # Extend loop: for each signed-diff metric add its abs_ counterpart.
    all_metrics: list[str] = []
    for m in base_metrics:
        all_metrics.append(m)
        if m in SIGNED_DIFF_METRICS:
            all_metrics.append(f"abs_{m}")

    print(f"\n  {description}  (n={len(merged)} matched patients)")

    rows = []
    for m in all_metrics:
        if f"{m}_a" not in merged.columns:
            continue
        a    = merged[f"{m}_a"].values
        b    = merged[f"{m}_b"].values
        lib  = METRIC_LOWER_IS_BETTER.get(m, True)
        unit = METRIC_UNITS.get(m, "")

        # Display label: signed → "(bias)", abs → "(accuracy)"
        if m.startswith("abs_"):
            display = m[4:] + " (accuracy)"
        elif m in SIGNED_DIFF_METRICS:
            display = m + " (bias)"
        else:
            display = m

        s = paired_stats(a, b, m, lower_is_better=lib)
        s["comparison"]  = label
        s["description"] = description
        rows.append(s)

        unit_str = f" {unit}" if unit else ""
        sig_t = "**" if s["t_pvalue"] < 0.01 else ("*" if s["t_pvalue"] < 0.05 else "")
        print(
            f"    {display:<38s}  Δ={s['mean_diff']:+.4f} ± {s['std_diff']:.4f}{unit_str}"
            f"  95%CI=[{s['ci_lo_95']:+.4f}, {s['ci_hi_95']:+.4f}]"
            f"  dz={s['cohens_dz']:+.3f}"
            f"  t_p={s['t_pvalue']:.4f}{sig_t}"
            f"  SW_p={s['shapiro_p']}"
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
        sub  = df[df["metric"] == metric]
        unit = METRIC_UNITS.get(metric, "")
        lib  = METRIC_LOWER_IS_BETTER.get(metric, True)
        direction_note = "negative = A better" if lib else "positive = A better"
        xlabel = f"Mean diff (A−B{', ' + unit if unit else ''})"

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
        ax.set_xlabel(f"{xlabel}\n({direction_note})", fontsize=8)
        ax.tick_params(labelsize=8)

    axes[0].set_yticks(range(n_comp))
    axes[0].set_yticklabels([descs[c] for c in comparisons], fontsize=8)
    fig.suptitle("Paired comparisons — mean difference ± 95% bootstrap CI\n"
                 "(* p_FDR<0.05, ** p_FDR<0.01; BH-corrected paired t-test)",
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

    # ── Benjamini-Hochberg FDR correction (per comparison) ────────────────
    # Applied separately for t-test and Wilcoxon p-values across all metrics
    # within each comparison group.
    df_all = pd.DataFrame(all_rows)
    for cmp_label in df_all["comparison"].unique():
        mask = df_all["comparison"] == cmp_label

        t_raw = pd.to_numeric(df_all.loc[mask, "t_pvalue"], errors="coerce").values
        w_raw = pd.to_numeric(df_all.loc[mask, "wilcoxon_p"], errors="coerce").values

        df_all.loc[mask, "t_pvalue_fdr"]   = np.round(_bh_fdr(t_raw), 4)
        df_all.loc[mask, "wilcoxon_p_fdr"] = np.round(_bh_fdr(w_raw), 4)

    print("\n── FDR correction (Benjamini-Hochberg, per comparison) applied ──")
    all_rows = df_all.to_dict("records")

    out_csv = OUT_DIR / "paired_comparison.csv"
    cols = ["comparison", "description", "metric", "lower_is_better", "n_patients",
            "mean_diff", "std_diff", "ci_lo_95", "ci_hi_95",
            "cohens_dz", "shapiro_p",
            "t_stat", "t_pvalue", "t_pvalue_fdr",
            "wilcoxon_p", "wilcoxon_p_fdr",
            "pct_A_wins"]
    df_all[cols].to_csv(out_csv, index=False)
    print(f"\n  Table saved: {out_csv}")

    plot_paired_differences(all_rows, metrics[:5])  # plot first 5 metrics


if __name__ == "__main__":
    main()
