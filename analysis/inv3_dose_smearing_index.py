# analysis/inv3_dose_smearing_index.py
# Investigation 3: quantify the "dose smearing" failure mode identified in
# findings_dosegan_failure_modes (DoseGAN over-predicts in central PTV + bladder,
# under-predicts at body periphery). Hypothesis: a per-patient smearing index
# correlates with body MAE and is larger in oldAcq.
#
# Definition (per patient, in Gy, signed):
#   core_err   = mean(pred - true) over voxels INSIDE PTV
#   margin_err = mean(pred - true) over voxels in 5 mm PTV expansion MINUS PTV
#   edge_err   = mean(pred - true) over voxels in body shell (outermost 5 mm)
#   smear_idx  = margin_err - edge_err
#       Positive  → dose pushed outward from core into margin AND away from edge
#                   (the radial-smear failure mode we visualised in Inv 2)
#       Near zero → no systematic radial bias
#
# Reads:  outputs/checkpoints_dosegan/{run_name}_fold{0..4}_best.pt
#         outputs/evaluation/{run_name}_fold{0..4}_val.csv  (for body_MAE_Gy)
#         data/pickles/<patient_id>.pkl
# Writes: outputs/analysis/inv3_{run_name}_dose_smearing.csv
#         outputs/analysis/inv3_{run_name}_smear_vs_mae.png
#         outputs/analysis/inv3_{run_name}_smear_by_acq.png
#
# Run after eval has produced the val CSVs:
#   python3 -m analysis.inv3_dose_smearing_index                    # DoseGAN default
#   python3 -m analysis.inv3_dose_smearing_index --model unet3d --activation tanh

import argparse
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import ndimage, stats

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["dosegan", "unet3d"], default="dosegan")
parser.add_argument("--activation", choices=["sigmoid", "tanh"], default=None)
parser.add_argument("--margin-mm", type=float, default=5.0,
                    help="Width of PTV-expansion margin and body-shell, in mm.")
parser.add_argument("--voxel-mm", type=float, default=1.5,
                    help="Isotropic voxel size after resampling (preprocessing default).")
args, _ = parser.parse_known_args()
cfg = importlib.import_module(f"configs.config_{args.model}")
if args.model == "unet3d" and args.activation is not None:
    cfg.RUN_NAME = cfg.RUN_NAME.replace(cfg.OUTPUT_ACTIVATION, args.activation)
    cfg.OUTPUT_ACTIVATION = args.activation
if args.model == "dosegan":
    from models.dosegan import UnetGenerator3d as _ModelClass
else:
    from models.unet3d import UNet3d as _ModelClass

EVAL_DIR    = Path("outputs/evaluation")
OUT_DIR     = Path("outputs/analysis")
CKPT_DIR    = cfg.CKPT_DIR
PICKLE_DIR  = cfg.PICKLE_DIR
DOSE_SCALE  = 50.0
MARGIN_VOX  = int(round(args.margin_mm / args.voxel_mm))


def load_generator(fold: int, device: torch.device):
    ckpt_path = CKPT_DIR / f"{cfg.RUN_NAME}_fold{fold}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    if args.model == "dosegan":
        net = _ModelClass(input_nc=cfg.INPUT_NC, output_nc=cfg.OUTPUT_NC, ngf=cfg.NGF).to(device)
        net.load_state_dict(ckpt["generator"])
    else:
        net = _ModelClass(
            in_channels=cfg.INPUT_NC, out_channels=cfg.OUTPUT_NC,
            channels=cfg.CHANNELS, strides=cfg.STRIDES,
            num_res_units=cfg.NUM_RES_UNITS, output_activation=cfg.OUTPUT_ACTIVATION,
        ).to(device)
        state_key = "model" if "model" in ckpt else "generator"
        net.load_state_dict(ckpt[state_key])
    net.eval()
    return net


def regions(ptv: np.ndarray, body: np.ndarray) -> dict:
    """Return three boolean volumes: core PTV, peri-PTV margin band, body shell."""
    ptv_b  = ptv  > 0.5
    body_b = body > 0.5
    # Binary morphology with a cubic structuring element ≈ margin_vox-radius ball.
    # ndimage iterations=N is L∞ dilation; close enough for a margin index.
    ptv_dilated = ndimage.binary_dilation(ptv_b, iterations=MARGIN_VOX)
    body_eroded = ndimage.binary_erosion(body_b, iterations=MARGIN_VOX)
    margin  = ptv_dilated & ~ptv_b & body_b      # outside PTV, within margin, inside body
    shell   = body_b & ~body_eroded              # outermost MARGIN_VOX of body
    return {"core": ptv_b, "margin": margin, "shell": shell}


def per_patient_index(pred_gy: np.ndarray, true_gy: np.ndarray,
                      ptv: np.ndarray, body: np.ndarray) -> dict:
    err = pred_gy - true_gy
    masks = regions(ptv, body)
    out = {}
    for name, m in masks.items():
        out[f"{name}_n"]    = int(m.sum())
        out[f"{name}_err"]  = float(err[m].mean()) if m.sum() > 0 else float("nan")
    out["smear_idx"] = out["margin_err"] - out["shell_err"]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  |  margin/shell width: {MARGIN_VOX} vox ({MARGIN_VOX*args.voxel_mm:.1f} mm)")
    print(f"variant: {cfg.RUN_NAME}")

    rows = []
    for fold in range(5):
        csv_path = EVAL_DIR / f"{cfg.RUN_NAME}_fold{fold}_val.csv"
        if not csv_path.exists():
            sys.exit(f"missing eval CSV: {csv_path}. Run eval first.")
        eval_df = pd.read_csv(csv_path)

        gen = load_generator(fold, device)
        print(f"\n[fold {fold}] {len(eval_df)} patients")

        for i, row in eval_df.iterrows():
            patient_id = row["patient_id"]
            with open(PICKLE_DIR / f"{patient_id}.pkl", "rb") as f:
                data = pickle.load(f)

            inp = torch.from_numpy(data["input"]).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = gen(inp)[0, 0].cpu().numpy() * DOSE_SCALE
            true = data["dose"] * DOSE_SCALE
            ptv  = data["ptv_mask"]
            body = data["input"][7]

            idx = per_patient_index(pred, true, ptv, body)
            rows.append({
                "patient_id":        patient_id,
                "acquisition_group": row["acquisition_group"],
                "fold":              fold,
                "body_MAE_Gy":       row["body_MAE_Gy"],
                **idx,
            })

            if (i + 1) % 10 == 0 or i == len(eval_df) - 1:
                print(f"  [{i+1:3d}/{len(eval_df)}] {patient_id}  "
                      f"core={idx['core_err']:+.3f}  "
                      f"margin={idx['margin_err']:+.3f}  "
                      f"shell={idx['shell_err']:+.3f}  "
                      f"smear={idx['smear_idx']:+.3f}")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"inv3_{cfg.RUN_NAME}_dose_smearing.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nsaved: {csv_path}")

    # === Summary stats ===
    print("\n=== Per-region mean signed error (Gy) ===")
    for r in ("core", "margin", "shell"):
        print(f"  {r:<6}: mean={df[f'{r}_err'].mean():+.3f}  median={df[f'{r}_err'].median():+.3f}  "
              f"std={df[f'{r}_err'].std():.3f}")
    print(f"  smear_idx: mean={df['smear_idx'].mean():+.3f}  median={df['smear_idx'].median():+.3f}  "
          f"std={df['smear_idx'].std():.3f}")

    # === Correlation of smear_idx with body MAE ===
    pearson  = df["smear_idx"].corr(df["body_MAE_Gy"], method="pearson")
    spearman = df["smear_idx"].corr(df["body_MAE_Gy"], method="spearman")
    print(f"\nsmear_idx ↔ body_MAE_Gy: pearson r = {pearson:.3f},  spearman ρ = {spearman:.3f}")

    # === Stratified by acquisition group (the distribution-shift question) ===
    old = df.loc[df.acquisition_group == "oldAcq", "smear_idx"].values
    new = df.loc[df.acquisition_group == "newAcq", "smear_idx"].values
    u, p = stats.mannwhitneyu(old, new, alternative="two-sided")
    print(f"\nsmear_idx oldAcq vs newAcq (Mann-Whitney U): u={u:.0f}  p={p:.4f}")
    print(f"  oldAcq (n={len(old)}): mean={old.mean():+.3f}  median={np.median(old):+.3f}")
    print(f"  newAcq (n={len(new)}): mean={new.mean():+.3f}  median={np.median(new):+.3f}")

    # === Figures ===
    # 1) smear_idx vs body MAE, colored by acq group
    fig, ax = plt.subplots(figsize=(8, 5))
    for grp, color in (("oldAcq", "tab:red"), ("newAcq", "tab:blue")):
        sub = df[df.acquisition_group == grp]
        ax.scatter(sub["body_MAE_Gy"], sub["smear_idx"], color=color, alpha=0.6, s=18, label=grp)
    ax.set_xlabel("Body-masked MAE (Gy)")
    ax.set_ylabel("Smear index (margin_err − shell_err, Gy)")
    ax.set_title(f"{cfg.RUN_NAME}: smear index vs body MAE  "
                 f"(pearson r={pearson:.2f}, spearman ρ={spearman:.2f})")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"inv3_{cfg.RUN_NAME}_smear_vs_mae.png", dpi=150)
    plt.close(fig)
    print(f"saved: inv3_{cfg.RUN_NAME}_smear_vs_mae.png")

    # 2) boxplot of smear_idx by acq group
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([old, new], labels=["oldAcq", "newAcq"], showmeans=True)
    ax.set_ylabel("Smear index (margin_err − shell_err, Gy)")
    ax.set_title(f"{cfg.RUN_NAME}: smear index by acquisition group  "
                 f"(Mann-Whitney p={p:.3f})")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"inv3_{cfg.RUN_NAME}_smear_by_acq.png", dpi=150)
    plt.close(fig)
    print(f"saved: inv3_{cfg.RUN_NAME}_smear_by_acq.png")


if __name__ == "__main__":
    main()
