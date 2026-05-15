# training/evaluate_dosegan.py
# Post-training evaluation of the best DoseGAN checkpoint.
# Computes SRQ1 (MAE, RMSE), SRQ2 (DVH metrics), SRQ3 (per acquisition group).
#
# Defaults to the VALIDATION split — safe to run at any time.
# Run from the project root:
#   python -m training.evaluate_dosegan              # val split, fold from config
#   python -m training.evaluate_dosegan --fold 2     # different fold
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

from configs import config_dosegan as cfg
from models.dosegan import UnetGenerator3d
from training.dataset import LUNDPROBEDataset

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)

DOSE_SCALE = 50.0   # all patients normalised to this (see preprocessing.py)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def body_mae_rmse(pred_gy: np.ndarray, true_gy: np.ndarray,
                  body_mask: np.ndarray):
    """MAE and RMSE over body-contour voxels, in Gy."""
    mask = body_mask > 0.5
    diff = pred_gy[mask] - true_gy[mask]
    mae  = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    return mae, rmse


def dvh_metrics(pred_gy: np.ndarray, true_gy: np.ndarray,
                struct_mask: np.ndarray) -> dict:
    """
    DVH metrics for one structure, comparing predicted to ground-truth dose.

    Conventions
    -----------
    D95  = dose received by at least 95% of the structure volume
           = 5th percentile of voxel doses (95% of voxels exceed this value)
    D98  = dose received by at least 98% → 2nd percentile
    Dmean, Dmax = mean and max dose inside the structure
    V20, V40 = % of structure volume receiving ≥ 20 Gy / ≥ 40 Gy

    Returns a flat dict with _pred, _true, and _diff keys for each metric.
    """
    m = struct_mask > 0.5
    result = {}

    if m.sum() == 0:
        # Structure absent for this patient (e.g. Genitalia, PenileBulb)
        for suffix in ("_pred", "_true", "_diff"):
            for key in ("Dmean", "Dmax", "D95", "D98", "V20", "V40"):
                result[key + suffix] = float("nan")
        return result

    for tag, dose in (("pred", pred_gy), ("true", true_gy)):
        v = dose[m]
        metrics = {
            "Dmean": float(v.mean()),
            "Dmax":  float(v.max()),
            "D95":   float(np.percentile(v, 5)),    # 95% of voxels exceed this
            "D98":   float(np.percentile(v, 2)),    # 98% of voxels exceed this
            "V20":   float(100.0 * (v >= 20.0).mean()),
            "V40":   float(100.0 * (v >= 40.0).mean()),
        }
        for k, val in metrics.items():
            result[f"{k}_{tag}"] = val

    for k in ("Dmean", "Dmax", "D95", "D98", "V20", "V40"):
        result[f"{k}_diff"] = result[f"{k}_pred"] - result[f"{k}_true"]

    return result


# ---------------------------------------------------------------------------
# Acquisition group lookup
# ---------------------------------------------------------------------------

def load_acq_group_map(split_csv: Path) -> dict:
    """Returns {patient_id: acquisition_group} from split.csv."""
    mapping = {}
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["patient_id"]] = row["acquisition_group"]
    return mapping


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(fold: int, split: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Evaluating on: {device} | fold={fold} | split='{split}'")

    # ── W&B initialisation ──────────────────────────────────────────────────
    # Sits inside the same group as the training run that produced the
    # checkpoint, with job_type="eval" so train and eval runs cluster
    # together in the UI but stay distinguishable.
    wandb.init(
        project  = cfg.PROJECT_NAME,
        name     = f"{cfg.RUN_NAME}_fold{fold}_eval_{split}",
        group    = cfg.RUN_NAME,
        job_type = "eval",
        config   = {"fold": fold, "split": split},
    )

    # ── Load generator from best checkpoint ──────────────────────────────────
    ckpt_path = cfg.CKPT_DIR / f"{cfg.RUN_NAME}_fold{fold}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Train fold {fold} first."
        )
    checkpoint = torch.load(ckpt_path, map_location=device)

    generator = UnetGenerator3d(
        input_nc=cfg.INPUT_NC, output_nc=cfg.OUTPUT_NC,
        ngf=cfg.NGF,
    ).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    log.info(
        f"Loaded checkpoint — epoch {checkpoint['epoch']} | "
        f"best_val_L1={checkpoint['best_val_loss']:.4f} "
        f"(= {checkpoint['best_val_loss']*DOSE_SCALE:.2f} Gy body-masked MAE)"
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = LUNDPROBEDataset(
        split_csv=cfg.SPLIT_CSV, pickle_dir=cfg.PICKLE_DIR,
        split=split, fold=fold if split != "test" else None,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    log.info(f"Patients to evaluate: {len(ds)}")

    acq_map = load_acq_group_map(cfg.SPLIT_CSV)

    # ── Per-patient results ───────────────────────────────────────────────────
    rows = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            patient_id = batch["patient_id"][0]
            real_input = batch["input"].to(device)   # (1, 9, D, H, W)
            real_dose  = batch["dose"].to(device)    # (1, 1, D, H, W)

            pred_dose = generator(real_input)        # (1, 1, D, H, W)

            # Convert to Gy numpy arrays (D, H, W)
            pred_gy = pred_dose[0, 0].cpu().numpy() * DOSE_SCALE
            true_gy = real_dose[0, 0].cpu().numpy() * DOSE_SCALE
            body    = real_input[0, 7].cpu().numpy()   # channel 7 = BODY

            # Structure masks come from the Dataset (already loaded once per patient).
            ptv_mask     = batch["ptv_mask"][0, 0].numpy()
            rectum_mask  = batch["rectum_mask"][0, 0].numpy()
            bladder_mask = batch["bladder_mask"][0, 0].numpy()

            # SRQ1 — global dose accuracy
            mae, rmse = body_mae_rmse(pred_gy, true_gy, body)

            # SRQ2 — DVH metrics per structure
            ptv_dvh     = dvh_metrics(pred_gy, true_gy, ptv_mask)
            rectum_dvh  = dvh_metrics(pred_gy, true_gy, rectum_mask)
            bladder_dvh = dvh_metrics(pred_gy, true_gy, bladder_mask)

            row = {
                "patient_id":       patient_id,
                "acquisition_group": acq_map.get(patient_id, "unknown"),
                "split":            split,
                "fold":             fold,
                "body_MAE_Gy":      mae,
                "body_RMSE_Gy":     rmse,
                **{f"ptv_{k}":     v for k, v in ptv_dvh.items()},
                **{f"rectum_{k}":  v for k, v in rectum_dvh.items()},
                **{f"bladder_{k}": v for k, v in bladder_dvh.items()},
            }
            rows.append(row)

            log.info(
                f"[{i+1:3d}/{len(ds)}] {patient_id} | "
                f"MAE={mae:.3f} Gy  RMSE={rmse:.3f} Gy | "
                f"PTV D95: pred={ptv_dvh['D95_pred']:.1f} Gy  "
                f"true={ptv_dvh['D95_true']:.1f} Gy  "
                f"diff={ptv_dvh['D95_diff']:+.2f} Gy"
            )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_dir = Path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.RUN_NAME}_fold{fold}_{split}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"\nResults saved: {out_path}")

    # ── Per-patient table to W&B (browsable in the UI) ────────────────────────
    fieldnames = list(rows[0].keys())
    table = wandb.Table(columns=fieldnames,
                        data=[[r[c] for c in fieldnames] for r in rows])
    wandb.log({"per_patient": table})

    # ── Summary table ─────────────────────────────────────────────────────────
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

    # SRQ3 — breakdown by acquisition group
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

    # DVH summary for PTV (most clinically important)
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

    # Stash the CSV as a run artifact for reproducibility
    wandb.save(str(out_path), policy="now")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DoseGAN on the validation split."
    )
    parser.add_argument(
        "--fold", type=int, default=cfg.FOLD,
        help="Which fold's checkpoint to load (default: cfg.FOLD)"
    )
    args = parser.parse_args()

    evaluate(fold=args.fold, split="val")
