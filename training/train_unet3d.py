# training/train_unet3d.py
# Supervised training for the 3D U-Net baseline.
# Simpler than train_dosegan.py — one model, one loss, no adversarial loop.
# Hyperparameters live in configs/config_unet3d.py.

import argparse
import sys
import logging
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config_unet3d as cfg
from training.dataset import LUNDPROBEDataset
from models.unet3d import UNet3d

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


DOSE_SCALE = 50.0


def masked_l1(pred, target, mask):
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1)


def structure_dmean_loss(pred, target, mask):
    """MAE of mean dose inside a structure. Safe for absent structures (contributes 0).

    Computed via a present-weighted mean (no boolean indexing, no .view/.squeeze)
    so it is robust to:
      * batch size 1 (no scalar collapse)
      * any/all structures absent in the batch (returns a 0.0 with grad through pred)
      * MONAI MetaTensor inputs (every op is a standard torch reduction)
    """
    spatial = tuple(range(2, mask.ndim))                       # (2, 3, 4) for (B,1,D,H,W)
    sample_dims = tuple(range(1, mask.ndim))                   # (1, 2, 3, 4)

    m = (mask > 0.5).to(pred.dtype)                            # (B, 1, D, H, W)
    n = m.sum(dim=spatial)                                     # (B, 1) voxels per sample
    n_safe = n.clamp(min=1.0)                                  # avoid /0 for absent structures

    pred_mean   = (pred   * m).sum(dim=spatial) / n_safe       # (B, 1)
    target_mean = (target * m).sum(dim=spatial) / n_safe       # (B, 1)
    per_sample  = torch.abs(pred_mean - target_mean)           # (B, 1)

    # Present = 1.0 if this sample has any voxels in the structure, else 0.0.
    present = (m.sum(dim=sample_dims) > 0).to(per_sample.dtype)  # (B,)
    # Broadcast (B,) -> (B, 1) so the mask aligns with per_sample.
    present_b = present.view(-1, *([1] * (per_sample.ndim - 1)))

    denom = present.sum().clamp(min=1.0)                       # scalar; 1.0 floor only kicks in when all absent
    return (per_sample * present_b).sum() / denom


def gradient_magnitude_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # L1 loss on finite-difference gradients in all 3 spatial dimensions.
    # Penalises blurring of dose gradients — sharper dose falloff at PTV boundary.
    dx = lambda v: v[:, :, 1:, :, :] - v[:, :, :-1, :, :]
    dy = lambda v: v[:, :, :, 1:, :] - v[:, :, :, :-1, :]
    dz = lambda v: v[:, :, :, :, 1:] - v[:, :, :, :, :-1]
    return (F.l1_loss(dx(pred), dx(target)) +
            F.l1_loss(dy(pred), dy(target)) +
            F.l1_loss(dz(pred), dz(target))) / 3


def train_one_epoch(model, loader, optimizer, device, lambda_dvh, lambda_grad):
    model.train()
    total_l1   = 0.0
    total_dvh  = 0.0
    total_grad = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        real_input   = batch["input"].to(device)
        real_dose    = batch["dose"].to(device)
        body_mask    = real_input[:, 7:8]
        ptv_mask     = batch["ptv_mask"].to(device)
        bladder_mask = batch["bladder_mask"].to(device)
        rectum_mask  = batch["rectum_mask"].to(device)

        pred       = model(real_input)
        loss_voxel = masked_l1(pred, real_dose, body_mask)
        loss_dvh   = (
            structure_dmean_loss(pred, real_dose, ptv_mask) +
            structure_dmean_loss(pred, real_dose, bladder_mask) +
            structure_dmean_loss(pred, real_dose, rectum_mask)
        )
        loss_grad  = gradient_magnitude_loss(pred, real_dose)
        loss = loss_voxel + lambda_dvh * loss_dvh + lambda_grad * loss_grad

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_l1   += loss_voxel.item()
        total_dvh  += loss_dvh.item()
        total_grad += loss_grad.item()

    n = len(loader)
    return {
        "train_L1":   total_l1   / n,
        "train_dvh":  total_dvh  / n,
        "train_grad": total_grad / n,
    }


def validate_dvh(model, loader, device):
    """
    Returns (val_L1, val_dvh_score).

    val_L1        — body-masked L1 in normalised units (checkpoint compat.).
    val_dvh_score — mean|Δ PTV D95| + mean|Δ Bladder Dmean| + mean|Δ Rectum Dmean|
                    in Gy. Used for early stopping.
    """
    model.eval()
    total_l1 = 0.0
    ptv_d95_errs, bladder_dmean_errs, rectum_dmean_errs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            real_input = batch["input"].to(device)
            real_dose  = batch["dose"].to(device)
            body_mask  = real_input[:, 7:8]

            pred = model(real_input)
            total_l1 += masked_l1(pred, real_dose, body_mask).item()

            pred_gy = pred[0, 0].cpu().numpy()      * DOSE_SCALE
            true_gy = real_dose[0, 0].cpu().numpy() * DOSE_SCALE

            ptv_m     = batch["ptv_mask"][0, 0].numpy()     > 0.5
            bladder_m = batch["bladder_mask"][0, 0].numpy() > 0.5
            rectum_m  = batch["rectum_mask"][0, 0].numpy()  > 0.5

            if ptv_m.sum() > 0:
                ptv_d95_errs.append(abs(
                    float(np.percentile(pred_gy[ptv_m], 5)) -
                    float(np.percentile(true_gy[ptv_m], 5))
                ))
            if bladder_m.sum() > 0:
                bladder_dmean_errs.append(abs(
                    float(pred_gy[bladder_m].mean()) - float(true_gy[bladder_m].mean())
                ))
            if rectum_m.sum() > 0:
                rectum_dmean_errs.append(abs(
                    float(pred_gy[rectum_m].mean()) - float(true_gy[rectum_m].mean())
                ))

    val_dvh_score = (
        (np.mean(ptv_d95_errs)       if ptv_d95_errs       else 0.0) +
        (np.mean(bladder_dmean_errs)  if bladder_dmean_errs  else 0.0) +
        (np.mean(rectum_dmean_errs)   if rectum_dmean_errs   else 0.0)
    )

    model.train()
    return total_l1 / len(loader), float(val_dvh_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None,
                        help="Override cfg.FOLD (0–4).")
    parser.add_argument("--activation", choices=["sigmoid", "tanh"], default=None,
                        help="Override cfg.OUTPUT_ACTIVATION. Rewrites the activation token in RUN_NAME.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing best checkpoint instead of refusing to start.")
    args, _ = parser.parse_known_args()
    if args.fold is not None:
        cfg.FOLD = args.fold
    if args.activation is not None:
        # Rewrite the activation token in RUN_NAME so W&B groups stay distinct.
        cfg.RUN_NAME = cfg.RUN_NAME.replace(cfg.OUTPUT_ACTIVATION, args.activation)
        cfg.OUTPUT_ACTIVATION = args.activation

    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.CKPT_DIR / f"{cfg.RUN_NAME}_fold{cfg.FOLD}_best.pt"
    if ckpt_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint: {ckpt_path}\n"
            f"Pass --overwrite (or set OVERWRITE=1 for the sbatch) to replace it, "
            f"or delete it first."
        )

    wandb.init(
        project  = cfg.PROJECT_NAME,
        name     = f"{cfg.RUN_NAME}_fold{cfg.FOLD}",
        group    = cfg.RUN_NAME,   # collapses all 5 folds into one experiment row in the UI
        job_type = "train",
        config  = {
            "fold":               cfg.FOLD,
            "epochs":             cfg.EPOCHS,
            "batch_size":         cfg.BATCH_SIZE,
            "lr":                 cfg.LR,
            "lambda_dvh":         cfg.LAMBDA_DVH,
            "lambda_grad":        cfg.LAMBDA_GRAD,
            "channels":           cfg.CHANNELS,
            "strides":            cfg.STRIDES,
            "num_res_units":      cfg.NUM_RES_UNITS,
            "output_activation":  cfg.OUTPUT_ACTIVATION,
            "early_stop_patience": cfg.EARLY_STOP_PATIENCE,
        }
    )

    # Seed before DataLoader creation so worker subprocesses inherit a
    # deterministic parent RNG; worker_init_fn below then offsets each
    # worker so augmentation is reproducible across reruns.
    torch.manual_seed(42)

    def _seed_worker(worker_id: int) -> None:
        torch.manual_seed(42 + worker_id)

    train_ds = LUNDPROBEDataset(
        split_csv=cfg.SPLIT_CSV, pickle_dir=cfg.PICKLE_DIR,
        split="train", fold=cfg.FOLD,
        use_geom_channels=cfg.USE_GEOM_CHANNELS,
    )
    val_ds = LUNDPROBEDataset(
        split_csv=cfg.SPLIT_CSV, pickle_dir=cfg.PICKLE_DIR,
        split="val", fold=cfg.FOLD,
        use_geom_channels=cfg.USE_GEOM_CHANNELS,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True,
        worker_init_fn=_seed_worker,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on: {device} | fold={cfg.FOLD}")

    model = UNet3d(
        in_channels       = cfg.INPUT_NC,
        out_channels      = cfg.OUTPUT_NC,
        channels          = cfg.CHANNELS,
        strides           = cfg.STRIDES,
        num_res_units     = cfg.NUM_RES_UNITS,
        output_activation = cfg.OUTPUT_ACTIVATION,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LR, betas=(cfg.BETA1, cfg.BETA2)
    )

    best_val_L1    = float("inf")   # stored in checkpoint for eval script compat.
    best_dvh_score = float("inf")   # used for early stopping decision
    epochs_no_improve = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        train_losses = train_one_epoch(model, train_loader, optimizer, device, cfg.LAMBDA_DVH, cfg.LAMBDA_GRAD)
        val_l1, val_dvh = validate_dvh(model, val_loader, device)

        wandb.log({
            "epoch":      epoch,
            "train_L1":   train_losses["train_L1"],
            "train_dvh":  train_losses["train_dvh"],
            "train_grad": train_losses["train_grad"],
            "val_L1":     val_l1,
            "val_dvh":    val_dvh,
        })
        log.info(
            f"Epoch {epoch:03d} | "
            f"train_L1: {train_losses['train_L1']:.4f} | "
            f"train_dvh: {train_losses['train_dvh']:.4f} | "
            f"train_grad: {train_losses['train_grad']:.4f} | "
            f"val_L1: {val_l1:.4f} | "
            f"val_dvh: {val_dvh:.3f} Gy"
        )

        if val_dvh < best_dvh_score:
            best_dvh_score = val_dvh
            best_val_L1    = val_l1
            epochs_no_improve = 0
            ckpt_path = cfg.CKPT_DIR / f"{cfg.RUN_NAME}_fold{cfg.FOLD}_best.pt"
            torch.save({
                "epoch":          epoch,
                "model":          model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "best_val_loss":  best_val_L1,    # body-masked L1 (eval script compat.)
                "best_dvh_score": best_dvh_score,
            }, ckpt_path)
            log.info(f"  ✓ Checkpoint saved: {ckpt_path}")
        else:
            epochs_no_improve += 1
            log.info(f"  No improvement for {epochs_no_improve}/{cfg.EARLY_STOP_PATIENCE} epochs")
            if epochs_no_improve >= cfg.EARLY_STOP_PATIENCE:
                log.info(f"Early stopping at epoch {epoch}")
                break

    ckpt_path = cfg.CKPT_DIR / f"{cfg.RUN_NAME}_fold{cfg.FOLD}_best.pt"
    if ckpt_path.exists():
        wandb.save(str(ckpt_path), policy="now")

    wandb.finish()


if __name__ == "__main__":
    main()
