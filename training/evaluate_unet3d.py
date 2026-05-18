# training/evaluate_unet3d.py
# Post-training evaluation of the best U-Net checkpoint.
# Mirrors evaluate_dosegan.py so the two models produce comparable per-patient
# CSVs and W&B summaries. Same metrics, same group/job_type pattern.
#
# Defaults to the VALIDATION split — safe to run at any time.
# Run from the project root:
#   python -m training.evaluate_unet3d                          # val split, fold from config
#   python -m training.evaluate_unet3d --fold 2                 # different fold
#   python -m training.evaluate_unet3d --activation tanh        # tanh variant
#
# *** Do NOT add a --split test flag until all model selection is complete. ***

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from configs import config_unet3d as cfg
from models.unet3d import UNet3d
from training.dataset import LUNDPROBEDataset
from training.evaluate_dosegan import (
    body_mae_rmse, dvh_metrics, load_acq_group_map, DOSE_SCALE,
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


def evaluate(fold: int, split: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Evaluating on: {device} | fold={fold} | split='{split}'")

    wandb.init(
        project  = cfg.PROJECT_NAME,
        name     = f"{cfg.RUN_NAME}_fold{fold}_eval_{split}",
        group    = cfg.RUN_NAME,
        job_type = "eval",
        config   = {"fold": fold, "split": split,
                    "output_activation": cfg.OUTPUT_ACTIVATION},
    )

    ckpt_path = cfg.CKPT_DIR / f"{cfg.RUN_NAME}_fold{fold}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Train fold {fold} first."
        )
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = UNet3d(
        in_channels       = cfg.INPUT_NC,
        out_channels      = cfg.OUTPUT_NC,
        channels          = cfg.CHANNELS,
        strides           = cfg.STRIDES,
        num_res_units     = cfg.NUM_RES_UNITS,
        output_activation = cfg.OUTPUT_ACTIVATION,
    ).to(device)
    # Trainer stores the model under "model" (vs DoseGAN's "generator").
    state_key = "model" if "model" in checkpoint else "generator"
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    log.info(
        f"Loaded checkpoint — epoch {checkpoint['epoch']} | "
        f"best_val_L1={checkpoint['best_val_loss']:.4f} "
        f"(= {checkpoint['best_val_loss']*DOSE_SCALE:.2f} Gy body-masked MAE)"
    )

    ds = LUNDPROBEDataset(
        split_csv=cfg.SPLIT_CSV, pickle_dir=cfg.PICKLE_DIR,
        split=split, fold=fold if split != "test" else None,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    log.info(f"Patients to evaluate: {len(ds)}")

    acq_map = load_acq_group_map(cfg.SPLIT_CSV)

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            patient_id = batch["patient_id"][0]
            real_input = batch["input"].to(device)
            real_dose  = batch["dose"].to(device)

            pred_dose = model(real_input)

            pred_gy = pred_dose[0, 0].cpu().numpy() * DOSE_SCALE
            true_gy = real_dose[0, 0].cpu().numpy() * DOSE_SCALE
            body    = real_input[0, 7].cpu().numpy()

            ptv_mask         = batch["ptv_mask"][0, 0].numpy()
            rectum_mask      = batch["rectum_mask"][0, 0].numpy()
            bladder_mask     = batch["bladder_mask"][0, 0].numpy()
            penile_bulb_mask = real_input[0, 6].cpu().numpy()   # channel 6 = PenileBulb

            mae, rmse = body_mae_rmse(pred_gy, true_gy, body)
            ptv_dvh         = dvh_metrics(pred_gy, true_gy, ptv_mask)
            rectum_dvh      = dvh_metrics(pred_gy, true_gy, rectum_mask)
            bladder_dvh     = dvh_metrics(pred_gy, true_gy, bladder_mask)
            penile_bulb_dvh = dvh_metrics(pred_gy, true_gy, penile_bulb_mask)

            row = {
                "patient_id":        patient_id,
                "acquisition_group": acq_map.get(patient_id, "unknown"),
                "split":             split,
                "fold":              fold,
                "body_MAE_Gy":       mae,
                "body_RMSE_Gy":      rmse,
                **{f"ptv_{k}":          v for k, v in ptv_dvh.items()},
                **{f"rectum_{k}":       v for k, v in rectum_dvh.items()},
                **{f"bladder_{k}":      v for k, v in bladder_dvh.items()},
                **{f"penile_bulb_{k}":  v for k, v in penile_bulb_dvh.items()},
            }
            rows.append(row)

            log.info(
                f"[{i+1:3d}/{len(ds)}] {patient_id} | "
                f"MAE={mae:.3f} Gy  RMSE={rmse:.3f} Gy | "
                f"PTV D95: pred={ptv_dvh['D95_pred']:.1f} Gy  "
                f"true={ptv_dvh['D95_true']:.1f} Gy  "
                f"diff={ptv_dvh['D95_diff']:+.2f} Gy"
            )

    out_dir = Path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.RUN_NAME}_fold{fold}_{split}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"\nResults saved: {out_path}")

    fieldnames = list(rows[0].keys())
    table = wandb.Table(columns=fieldnames,
                        data=[[r[c] for c in fieldnames] for r in rows])
    wandb.log({"per_patient": table})

    maes  = [r["body_MAE_Gy"]  for r in rows]
    rmses = [r["body_RMSE_Gy"] for r in rows]
    log.info("\n── Overall (" + split + ") ───────────────────────────────")
    log.info(f"  body MAE  : {np.mean(maes):.3f} ± {np.std(maes):.3f} Gy")
    log.info(f"  body RMSE : {np.mean(rmses):.3f} ± {np.std(rmses):.3f} Gy")

    wandb.summary["body_MAE_Gy_mean"]  = float(np.mean(maes))
    wandb.summary["body_MAE_Gy_std"]   = float(np.std(maes))
    wandb.summary["body_RMSE_Gy_mean"] = float(np.mean(rmses))
    wandb.summary["body_RMSE_Gy_std"]  = float(np.std(rmses))
    wandb.summary["n_patients"]        = len(rows)

    for grp in ("oldAcq", "newAcq"):
        grp_rows = [r for r in rows if r["acquisition_group"] == grp]
        if not grp_rows:
            continue
        g_maes  = [r["body_MAE_Gy"]  for r in grp_rows]
        g_rmses = [r["body_RMSE_Gy"] for r in grp_rows]
        log.info(f"\n── {grp} (n={len(grp_rows)}) ───────────────────────────")
        log.info(f"  body MAE  : {np.mean(g_maes):.3f} ± {np.std(g_maes):.3f} Gy")
        log.info(f"  body RMSE : {np.mean(g_rmses):.3f} ± {np.std(g_rmses):.3f} Gy")
        wandb.summary[f"body_MAE_Gy_mean_{grp}"] = float(np.mean(g_maes))
        wandb.summary[f"body_MAE_Gy_std_{grp}"]  = float(np.std(g_maes))
        wandb.summary[f"n_patients_{grp}"]       = len(grp_rows)

    log.info("\n── PTV DVH (mean across patients) ───────────────────────")
    for metric in ("D95", "D98", "Dmean", "Dmax"):
        pred_vals = [r[f"ptv_{metric}_pred"] for r in rows]
        true_vals = [r[f"ptv_{metric}_true"] for r in rows]
        diff_vals = [r[f"ptv_{metric}_diff"] for r in rows]
        log.info(
            f"  {metric:<6}: pred={np.nanmean(pred_vals):.2f} Gy  "
            f"true={np.nanmean(true_vals):.2f} Gy  "
            f"diff={np.nanmean(diff_vals):+.2f} ± {np.nanstd(diff_vals):.2f} Gy"
        )
        wandb.summary[f"ptv_{metric}_diff_mean"] = float(np.nanmean(diff_vals))
        wandb.summary[f"ptv_{metric}_diff_std"]  = float(np.nanstd(diff_vals))

    wandb.save(str(out_path), policy="now")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net on the validation split."
    )
    parser.add_argument("--fold", type=int, default=cfg.FOLD,
                        help="Which fold's checkpoint to load (default: cfg.FOLD)")
    parser.add_argument("--activation", choices=["sigmoid", "tanh"], default=None,
                        help="Override cfg.OUTPUT_ACTIVATION. Rewrites the activation token in RUN_NAME.")
    args = parser.parse_args()

    if args.activation is not None:
        cfg.RUN_NAME = cfg.RUN_NAME.replace(cfg.OUTPUT_ACTIVATION, args.activation)
        cfg.OUTPUT_ACTIVATION = args.activation

    evaluate(fold=args.fold, split="val")
