"""
analysis/compare_retrain_fold0.py
===================================
Compares fold-0 checkpoints and/or evaluation CSVs between the original
training run and a retrain run to verify results are within noise before
committing to full 5-fold retrain.

Usage:
    python3 -m analysis.compare_retrain_fold0

Decision rule:
    body_MAE difference < 0.010 Gy  → retrain is consistent, proceed with all folds
    body_MAE difference ≥ 0.010 Gy  → investigate before proceeding

0.010 Gy corresponds to ~1.2% of the 0.86 Gy baseline — well within
expected fold-to-fold variance (~0.03–0.05 Gy).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CKPT_DIR_DOSEGAN = Path("outputs/checkpoints_dosegan")
CKPT_DIR_UNET    = Path("outputs/checkpoints_unet3d")
EVAL_DIR         = Path("outputs/evaluation")

TOLERANCE_GY = 0.010   # max acceptable body_MAE difference (Gy)

PAIRS = [
    # (label, original_run_name, retrain_run_name, ckpt_dir)
    (
        "DoseGAN baseline",
        "dosegan_ngf32_sigmoid_snellius",
        "dosegan_ngf32_sigmoid_snellius_v2",
        CKPT_DIR_DOSEGAN,
    ),
    (
        "DoseGAN geom",
        "dosegan_ngf32_sigmoid_geom_snellius",
        "dosegan_ngf32_sigmoid_geom_snellius_v2",
        CKPT_DIR_DOSEGAN,
    ),
    (
        "U-Net baseline",
        "unet3d_ch32_sigmoid_snellius",
        "unet3d_ch32_sigmoid_snellius_v2",
        CKPT_DIR_UNET,
    ),
    (
        "U-Net geom",
        "unet3d_ch32_sigmoid_geom_snellius",
        "unet3d_ch32_sigmoid_geom_snellius_v2",
        CKPT_DIR_UNET,
    ),
]


def _load_ckpt_meta(ckpt_dir: Path, run_name: str, fold: int = 0) -> dict | None:
    path = ckpt_dir / f"{run_name}_fold{fold}_best.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    return {
        "best_val_loss": ckpt.get("best_val_loss"),
        "epoch":         ckpt.get("epoch"),
        "path":          str(path),
    }


def _load_eval_body_mae(run_name: str, fold: int = 0) -> float | None:
    path = EVAL_DIR / f"{run_name}_fold{fold}_val.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "body_MAE_Gy" not in df.columns:
        return None
    return float(df["body_MAE_Gy"].mean())


def main():
    print("Fold-0 retrain verification")
    print("=" * 70)
    print(f"Tolerance: body_MAE difference < {TOLERANCE_GY:.3f} Gy = consistent\n")

    all_consistent = True

    for label, orig, retrain, ckpt_dir in PAIRS:
        print(f"── {label} ──")

        orig_ckpt    = _load_ckpt_meta(ckpt_dir, orig)
        retrain_ckpt = _load_ckpt_meta(ckpt_dir, retrain)

        if orig_ckpt is None:
            print(f"  original checkpoint missing: {ckpt_dir}/{orig}_fold0_best.pt")
        else:
            print(f"  original  : best_val_loss={orig_ckpt['best_val_loss']:.5f}"
                  f"  epoch={orig_ckpt['epoch']}")

        if retrain_ckpt is None:
            print(f"  retrain   : checkpoint not found — job may still be running")
        else:
            print(f"  retrain   : best_val_loss={retrain_ckpt['best_val_loss']:.5f}"
                  f"  epoch={retrain_ckpt['epoch']}")

        if orig_ckpt and retrain_ckpt:
            delta_val = abs(orig_ckpt["best_val_loss"] - retrain_ckpt["best_val_loss"])
            status = "OK" if delta_val < TOLERANCE_GY else "DIVERGED"
            print(f"  |Δ val_loss| = {delta_val:.5f}  → {status}")
            if status == "DIVERGED":
                all_consistent = False

        # Also compare eval CSVs if available
        orig_mae    = _load_eval_body_mae(orig)
        retrain_mae = _load_eval_body_mae(retrain)

        if orig_mae is not None and retrain_mae is not None:
            delta_mae = abs(orig_mae - retrain_mae)
            status = "OK" if delta_mae < TOLERANCE_GY else "DIVERGED"
            print(f"  body_MAE: orig={orig_mae:.4f}  retrain={retrain_mae:.4f}"
                  f"  |Δ|={delta_mae:.4f}  → {status}")
            if status == "DIVERGED":
                all_consistent = False
        elif orig_mae is not None:
            print(f"  body_MAE: orig={orig_mae:.4f}  retrain=not yet evaluated")
        else:
            print(f"  body_MAE: eval CSVs not yet available")

        print()

    print("=" * 70)
    if all_consistent:
        print("VERDICT: All fold-0 retrains are consistent with originals.")
        print("         Safe to submit full 5-fold retrain with --overwrite.")
    else:
        print("VERDICT: One or more retrains DIVERGED from originals.")
        print("         Investigate before overwriting production checkpoints.")


if __name__ == "__main__":
    main()
