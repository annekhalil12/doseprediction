# training/evaluate.py
# Unified evaluation for DoseGAN and U-Net checkpoints (baseline and geom).
#
# Run from the project root:
#   python -m training.evaluate --model dosegan [--fold N] [--run-name NAME] [--skip-gamma]
#   python -m training.evaluate --model unet3d  [--fold N] [--run-name NAME] [--skip-gamma] [--activation {sigmoid,tanh}]
#
# *** Do NOT add a --split test flag until all model selection is complete. ***

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from training.dataset import LUNDPROBEDataset
from training.metrics import (
    compute_mae, compute_rmse,
    dvh_endpoints,
    compute_boundary_mae,
    compute_gamma_passrate,
    compute_isodose_metrics,
    compute_D95, compute_Vx,
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)

DOSE_SCALE = 50.0


def load_acq_group_map(split_csv: Path) -> dict:
    mapping = {}
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["patient_id"]] = row["acquisition_group"]
    return mapping


def _wandb_summary_stats(wandb_run, key: str, values: list) -> None:
    clean = [v for v in values if not np.isnan(v)]
    if clean:
        wandb_run.summary[f"{key}_mean"] = float(np.mean(clean))
        wandb_run.summary[f"{key}_std"]  = float(np.std(clean))


def _load_model_and_cfg(model_type: str, fold: int, run_name=None,
                        activation=None, device=None, use_geom=None):
    if model_type == "dosegan":
        from configs import config_dosegan as cfg
        from models.dosegan import UnetGenerator3d
        if use_geom is not None:
            cfg.USE_GEOM_CHANNELS = use_geom
            cfg.INPUT_NC = 14 if use_geom else 9
        if run_name:
            cfg.RUN_NAME = run_name
        ckpt_path = cfg.CKPT_DIR / f"{cfg.RUN_NAME}_fold{fold}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path}. Train fold {fold} first."
            )
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = UnetGenerator3d(
            input_nc=cfg.INPUT_NC, output_nc=cfg.OUTPUT_NC, ngf=cfg.NGF,
        ).to(device)
        model.load_state_dict(checkpoint["generator"])
        wandb_extra = {}
    else:
        from configs import config_unet3d as cfg
        from models.unet3d import UNet3d
        if use_geom is not None:
            cfg.USE_GEOM_CHANNELS = use_geom
            cfg.INPUT_NC = 14 if use_geom else 9
        if activation is not None:
            cfg.RUN_NAME = cfg.RUN_NAME.replace(cfg.OUTPUT_ACTIVATION, activation)
            cfg.OUTPUT_ACTIVATION = activation
        if run_name:
            cfg.RUN_NAME = run_name
        ckpt_path = cfg.CKPT_DIR / f"{cfg.RUN_NAME}_fold{fold}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path}. Train fold {fold} first."
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
        state_key = "model" if "model" in checkpoint else "generator"
        model.load_state_dict(checkpoint[state_key])
        wandb_extra = {"output_activation": cfg.OUTPUT_ACTIVATION}

    return model, cfg, checkpoint, wandb_extra


def evaluate(model_type: str, fold: int, split: str,
             run_name=None, activation=None, skip_gamma: bool = False,
             use_geom=None, force: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Evaluating on: {device} | model={model_type} | fold={fold} | split='{split}'")

    model, cfg, checkpoint, wandb_extra = _load_model_and_cfg(
        model_type, fold, run_name=run_name, activation=activation, device=device,
        use_geom=use_geom,
    )
    model.eval()

    out_path = Path("outputs/evaluation") / f"{cfg.RUN_NAME}_fold{fold}_{split}.csv"
    if out_path.exists() and not force:
        log.info(f"Output already exists, skipping: {out_path}")
        return
    if out_path.exists() and force:
        log.warning(f"--force: overwriting existing results: {out_path}")

    log.info(
        f"Loaded checkpoint — epoch {checkpoint['epoch']} | "
        f"best_val_L1={checkpoint['best_val_loss']:.4f} "
        f"(= {checkpoint['best_val_loss'] * DOSE_SCALE:.2f} Gy body-masked MAE)"
    )

    wandb.init(
        project  = cfg.PROJECT_NAME,
        name     = f"{cfg.RUN_NAME}_fold{fold}_eval_{split}",
        group    = cfg.RUN_NAME,
        job_type = "eval",
        config   = {"fold": fold, "split": split, "model": model_type, **wandb_extra},
    )

    ds = LUNDPROBEDataset(
        split_csv=cfg.SPLIT_CSV, pickle_dir=cfg.PICKLE_DIR,
        split=split, fold=fold if split != "test" else None,
        use_geom_channels=cfg.USE_GEOM_CHANNELS,
    )
    loader  = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    acq_map = load_acq_group_map(cfg.SPLIT_CSV)
    log.info(f"Patients to evaluate: {len(ds)}")

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            patient_id = batch["patient_id"][0]
            real_input = batch["input"].to(device)
            real_dose  = batch["dose"].to(device)

            t0           = time.perf_counter()
            pred_dose    = model(real_input)
            inference_ms = (time.perf_counter() - t0) * 1000

            pred_gy = pred_dose[0, 0].cpu().numpy() * DOSE_SCALE
            true_gy = real_dose[0, 0].cpu().numpy() * DOSE_SCALE

            body_mask        = real_input[0, 7].cpu().numpy()
            ptv_mask         = batch["ptv_mask"][0, 0].numpy()
            rectum_mask      = batch["rectum_mask"][0, 0].numpy()
            bladder_mask     = batch["bladder_mask"][0, 0].numpy()
            penile_bulb_mask = real_input[0, 6].cpu().numpy()

            body_mae  = compute_mae(pred_gy,  true_gy, body_mask)
            body_rmse = compute_rmse(pred_gy, true_gy, body_mask)
            ptv_mae   = compute_mae(pred_gy,  true_gy, ptv_mask)
            ptv_rmse  = compute_rmse(pred_gy, true_gy, ptv_mask)
            rect_mae  = compute_mae(pred_gy,  true_gy, rectum_mask)
            rect_rmse = compute_rmse(pred_gy, true_gy, rectum_mask)
            blad_mae  = compute_mae(pred_gy,  true_gy, bladder_mask)
            blad_rmse = compute_rmse(pred_gy, true_gy, bladder_mask)

            ptv_dvh         = dvh_endpoints(pred_gy, true_gy, ptv_mask)
            rectum_dvh      = dvh_endpoints(pred_gy, true_gy, rectum_mask)
            bladder_dvh     = dvh_endpoints(pred_gy, true_gy, bladder_mask)
            penile_bulb_dvh = dvh_endpoints(pred_gy, true_gy, penile_bulb_mask)

            bnd_ptv  = compute_boundary_mae(pred_gy, true_gy, ptv_mask)
            bnd_rect = compute_boundary_mae(pred_gy, true_gy, rectum_mask) \
                       if (rectum_mask > 0.5).sum() > 0 else float("nan")
            bnd_blad = compute_boundary_mae(pred_gy, true_gy, bladder_mask) \
                       if (bladder_mask > 0.5).sum() > 0 else float("nan")

            presc             = compute_D95(true_gy, ptv_mask)
            v_presc_rect_pred = compute_Vx(pred_gy, rectum_mask,  presc)
            v_presc_rect_true = compute_Vx(true_gy, rectum_mask,  presc)
            v_presc_blad_pred = compute_Vx(pred_gy, bladder_mask, presc)
            v_presc_blad_true = compute_Vx(true_gy, bladder_mask, presc)

            if skip_gamma:
                gamma_3_3 = float("nan")
                gamma_2_2 = float("nan")
            else:
                gamma_3_3 = compute_gamma_passrate(pred_gy, true_gy, body_mask,
                                                   dose_percent=3.0, distance_mm=3.0)
                gamma_2_2 = compute_gamma_passrate(pred_gy, true_gy, body_mask,
                                                   dose_percent=2.0, distance_mm=2.0)

            isodose = compute_isodose_metrics(pred_gy, true_gy, ptv_mask)

            row = {
                "patient_id":              patient_id,
                "acquisition_group":       acq_map.get(patient_id, "unknown"),
                "split":                   split,
                "fold":                    fold,
                "model":                   model_type,
                "inference_time_ms":       inference_ms,
                "body_MAE_Gy":             body_mae,
                "body_RMSE_Gy":            body_rmse,
                "ptv_MAE_Gy":              ptv_mae,
                "ptv_RMSE_Gy":             ptv_rmse,
                "rectum_MAE_Gy":           rect_mae,
                "rectum_RMSE_Gy":          rect_rmse,
                "bladder_MAE_Gy":          blad_mae,
                "bladder_RMSE_Gy":         blad_rmse,
                **{f"ptv_{k}":          v for k, v in ptv_dvh.items()},
                **{f"rectum_{k}":       v for k, v in rectum_dvh.items()},
                **{f"bladder_{k}":      v for k, v in bladder_dvh.items()},
                **{f"penile_bulb_{k}":  v for k, v in penile_bulb_dvh.items()},
                "boundary_MAE_ptv_Gy":     bnd_ptv,
                "boundary_MAE_rectum_Gy":  bnd_rect,
                "boundary_MAE_bladder_Gy": bnd_blad,
                "V_presc_rectum_pred":     v_presc_rect_pred,
                "V_presc_rectum_true":     v_presc_rect_true,
                "V_presc_rectum_diff":     v_presc_rect_pred - v_presc_rect_true,
                "V_presc_bladder_pred":    v_presc_blad_pred,
                "V_presc_bladder_true":    v_presc_blad_true,
                "V_presc_bladder_diff":    v_presc_blad_pred - v_presc_blad_true,
                "prescription_dose_Gy":    float(presc),
                "gamma_3pct_3mm":          gamma_3_3,
                "gamma_2pct_2mm":          gamma_2_2,
                **isodose,
            }
            rows.append(row)

            log.info(
                f"[{i+1:3d}/{len(ds)}] {patient_id} | "
                f"body MAE={body_mae:.3f} Gy | "
                f"PTV D95 diff={ptv_dvh['D95_diff']:+.2f} Gy | "
                f"bnd MAE PTV={bnd_ptv:.3f} Gy | "
                f"gamma 3/3={gamma_3_3:.1f}%"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"\nResults saved: {out_path}")

    fieldnames = list(rows[0].keys())
    wandb.log({"per_patient": wandb.Table(
        columns=fieldnames, data=[[r[c] for c in fieldnames] for r in rows]
    )})

    log.info("\n── Overall (" + split + ") ───────────────────────────────────")
    for key, label in [
        ("body_MAE_Gy",            "body MAE   "),
        ("body_RMSE_Gy",           "body RMSE  "),
        ("ptv_MAE_Gy",             "PTV  MAE   "),
        ("gamma_3pct_3mm",         "gamma 3/3  "),
        ("gamma_2pct_2mm",         "gamma 2/2  "),
        ("boundary_MAE_ptv_Gy",    "bnd MAE PTV"),
    ]:
        vals = [r[key] for r in rows]
        log.info(f"  {label}: {np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}")
        _wandb_summary_stats(wandb, key, vals)

    log.info("\n── PTV DVH (mean across patients) ───────────────────────────")
    for metric in ("D95", "D98", "Dmean", "Dmax", "D01cc"):
        pred_vals = [r[f"ptv_{metric}_pred"] for r in rows]
        true_vals = [r[f"ptv_{metric}_true"] for r in rows]
        diff_vals = [r[f"ptv_{metric}_diff"] for r in rows]
        log.info(
            f"  {metric:<6}: pred={np.nanmean(pred_vals):.2f} Gy  "
            f"true={np.nanmean(true_vals):.2f} Gy  "
            f"diff={np.nanmean(diff_vals):+.2f} ± {np.nanstd(diff_vals):.2f} Gy"
        )
        _wandb_summary_stats(wandb, f"ptv_{metric}_diff", diff_vals)

    log.info("\n── Isodose conformality (mean across patients) ───────────────")
    for level in ("100iso", "95iso", "80iso", "50iso"):
        dice_vals = [r.get(f"Dice_{level}", float("nan")) for r in rows]
        hd95_vals = [r.get(f"HD95_{level}_mm", float("nan")) for r in rows]
        log.info(
            f"  {level}: Dice={np.nanmean(dice_vals):.3f} ± {np.nanstd(dice_vals):.3f}  "
            f"HD95={np.nanmean(hd95_vals):.1f} ± {np.nanstd(hd95_vals):.1f} mm"
        )
        _wandb_summary_stats(wandb, f"Dice_{level}", dice_vals)
        _wandb_summary_stats(wandb, f"HD95_{level}_mm", hd95_vals)

    for grp in ("oldAcq", "newAcq"):
        grp_rows = [r for r in rows if r["acquisition_group"] == grp]
        if not grp_rows:
            continue
        g_maes = [r["body_MAE_Gy"] for r in grp_rows]
        log.info(f"\n── {grp} (n={len(grp_rows)}) ────────────────────────────────")
        log.info(f"  body MAE: {np.mean(g_maes):.3f} ± {np.std(g_maes):.3f} Gy")
        wandb.summary[f"body_MAE_Gy_mean_{grp}"] = float(np.mean(g_maes))
        wandb.summary[f"body_MAE_Gy_std_{grp}"]  = float(np.std(g_maes))
        wandb.summary[f"n_patients_{grp}"]        = len(grp_rows)

    wandb.summary["n_patients"] = len(rows)
    wandb.save(str(out_path), policy="now")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DoseGAN or U-Net on the validation (or test) split."
    )
    parser.add_argument("--model", choices=["dosegan", "unet3d"], required=True,
                        help="Which model to evaluate.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold checkpoint to load (default: cfg.FOLD).")
    parser.add_argument("--run-name", dest="run_name", type=str, default=None,
                        help="Override cfg.RUN_NAME.")
    parser.add_argument("--activation", choices=["sigmoid", "tanh"], default=None,
                        help="Override cfg.OUTPUT_ACTIVATION (U-Net only).")
    parser.add_argument("--skip-gamma", dest="skip_gamma", action="store_true",
                        help="Skip gamma pass rate (expensive 3D computation).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing evaluation CSV (required after retraining).")
    geom_group = parser.add_mutually_exclusive_group()
    geom_group.add_argument("--geom",    dest="geom", action="store_true",  default=None,
                            help="Force geom mode: USE_GEOM_CHANNELS=True, INPUT_NC=14.")
    geom_group.add_argument("--no-geom", dest="geom", action="store_false",
                            help="Force baseline mode: USE_GEOM_CHANNELS=False, INPUT_NC=9.")
    args = parser.parse_args()

    if args.fold is None:
        if args.model == "dosegan":
            from configs import config_dosegan as _cfg
        else:
            from configs import config_unet3d as _cfg
        fold = _cfg.FOLD
    else:
        fold = args.fold

    evaluate(
        model_type  = args.model,
        fold        = fold,
        split       = "val",
        run_name    = args.run_name,
        activation  = args.activation,
        skip_gamma  = args.skip_gamma,
        use_geom    = args.geom,
        force       = args.force,
    )
