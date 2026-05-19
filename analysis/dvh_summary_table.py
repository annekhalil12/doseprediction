# analysis/dvh_summary_table.py
# Aggregate per-patient DVH metrics across 5 folds and report mean |error| as
# % of prescription dose (50 Gy) — the format used in the literature comparison
# table (thesis Section 3.5).
#
# Reads:  outputs/evaluation/{run_name}_fold{0..4}_val.csv
# Writes: outputs/analysis/dvh_summary_{run_name}.csv  (one row per metric)
#         prints table to stdout
#
# Usage:
#   python3 -m analysis.dvh_summary_table                   # DoseGAN Sigmoid
#   python3 -m analysis.dvh_summary_table --model unet3d    # U-Net Sigmoid

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path("outputs/evaluation")
OUT_DIR  = Path("outputs/analysis")
DOSE_SCALE = 50.0  # Gy — prescription dose

METRICS = {
    "PTV Dmean":     "ptv_Dmean_diff",
    "PTV D95":       "ptv_D95_diff",
    "PTV D98":       "ptv_D98_diff",
    "Bladder Dmean": "bladder_Dmean_diff",
    "Rectum Dmean":  "rectum_Dmean_diff",
    "Rectum D95":    "rectum_D95_diff",
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["dosegan", "unet3d"], default="dosegan")
args = parser.parse_args()

cfg_module = "configs.config_dosegan" if args.model == "dosegan" else "configs.config_unet3d"
cfg = importlib.import_module(cfg_module)

paths = sorted(EVAL_DIR.glob(f"{cfg.RUN_NAME}_fold*_val.csv"))
if not paths:
    sys.exit(f"No eval CSVs found for {cfg.RUN_NAME} — run eval first.")

df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
print(f"\nModel: {cfg.RUN_NAME}")
print(f"N = {len(df)} patients across {len(paths)} folds\n")

rows = []
for label, col in METRICS.items():
    if col not in df.columns:
        print(f"  [skip] {label} — column {col!r} not found")
        continue
    abs_pct = df[col].abs() / DOSE_SCALE * 100
    mean, std = abs_pct.mean(), abs_pct.std()
    rows.append({"metric": label, "mean_pct": mean, "std_pct": std})
    print(f"  {label:15s}  {mean:.2f}% ± {std:.2f}%")

out_df = pd.DataFrame(rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"dvh_summary_{cfg.RUN_NAME}.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
