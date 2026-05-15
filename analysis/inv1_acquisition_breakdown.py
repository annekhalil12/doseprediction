# analysis/inv1_acquisition_breakdown.py
# Investigation 1: aggregate per-patient eval CSVs across 5 DoseGAN folds and
# test whether oldAcq patients are systematically worse than newAcq.
#
# Reads:  outputs/evaluation/dosegan_fold{0..4}_val.csv
# Writes: outputs/analysis/inv1_acquisition_breakdown.csv  (summary table)
#         outputs/analysis/inv1_acquisition_boxplot.png    (boxplot figure)
#
# Run after eval_dosegan.sbatch finishes:
#   python3 -m analysis.inv1_acquisition_breakdown

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

EVAL_DIR    = Path("outputs/evaluation")
OUT_DIR     = Path("outputs/analysis")
FOLDS       = [0, 1, 2, 3, 4]
GROUPS      = ["oldAcq", "newAcq"]


def load_all_folds() -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        path = EVAL_DIR / f"dosegan_fold{fold}_val.csv"
        if not path.exists():
            sys.exit(f"missing eval CSV: {path}. Run eval_dosegan.sbatch first.")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["fold", "acquisition_group"])["body_MAE_Gy"]
    return g.agg(["count", "mean", "std", "median",
                  lambda s: s.quantile(0.25),
                  lambda s: s.quantile(0.75)]).rename(
        columns={"<lambda_0>": "q25", "<lambda_1>": "q75"}
    )


def stat_test(df: pd.DataFrame) -> dict:
    old = df.loc[df.acquisition_group == "oldAcq", "body_MAE_Gy"].values
    new = df.loc[df.acquisition_group == "newAcq", "body_MAE_Gy"].values
    u, p = stats.mannwhitneyu(old, new, alternative="two-sided")
    return {"n_old": len(old), "n_new": len(new),
            "mean_old": float(old.mean()), "mean_new": float(new.mean()),
            "median_old": float(np.median(old)), "median_new": float(np.median(new)),
            "u_statistic": float(u), "p_value": float(p)}


def boxplot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    # Left: pooled (all folds), oldAcq vs newAcq
    pooled = [df.loc[df.acquisition_group == g, "body_MAE_Gy"].values for g in GROUPS]
    axes[0].boxplot(pooled, labels=GROUPS, showmeans=True)
    axes[0].set_ylabel("Body-masked MAE (Gy)")
    axes[0].set_title("Pooled across 5 folds")
    axes[0].grid(alpha=0.3)

    # Right: per-fold breakdown
    positions, labels = [], []
    for i, fold in enumerate(FOLDS):
        for j, grp in enumerate(GROUPS):
            vals = df[(df.fold == fold) & (df.acquisition_group == grp)]["body_MAE_Gy"].values
            pos = i * 3 + j
            axes[1].boxplot([vals], positions=[pos], widths=0.7, showmeans=True)
            positions.append(pos)
            labels.append(f"f{fold}\n{grp.replace('Acq','')}")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_title("Per-fold breakdown")
    axes[1].grid(alpha=0.3)

    fig.suptitle("DoseGAN body-masked MAE by acquisition group", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_folds()
    print(f"loaded {len(df)} per-patient rows across folds {FOLDS}")

    summary = summarise(df)
    summary_path = OUT_DIR / "inv1_acquisition_breakdown.csv"
    summary.to_csv(summary_path)
    print(f"saved: {summary_path}")
    print("\n=== Per fold × group ===")
    print(summary.to_string())

    stats_out = stat_test(df)
    print("\n=== Pooled oldAcq vs newAcq (Mann-Whitney U, two-sided) ===")
    for k, v in stats_out.items():
        if isinstance(v, float):
            print(f"  {k:>14s}: {v:.4f}")
        else:
            print(f"  {k:>14s}: {v}")
    delta = stats_out["mean_old"] - stats_out["mean_new"]
    print(f"\n  mean(oldAcq) − mean(newAcq) = {delta:+.4f} Gy "
          f"({'oldAcq worse' if delta > 0 else 'newAcq worse'})")
    print(f"  significant at α=0.05? {'yes' if stats_out['p_value'] < 0.05 else 'no'}")

    boxplot(df, OUT_DIR / "inv1_acquisition_boxplot.png")


if __name__ == "__main__":
    main()
