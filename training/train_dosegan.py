# train_dosegan.py
# Training script specific to the DoseGAN model.
# Shared preprocessing config lives in config_preprocessing_shared.py.
# DoseGAN hyperparameters (lr, lambda, ngf) live in config_dosegan.py.

import argparse
import random
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import wandb

from configs import config_dosegan as cfg  # all hyperparameters live here
from training.dataset import LUNDPROBEDataset
from models.dosegan import UnetGenerator3d, NLayerDiscriminator, GANLoss

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Mean absolute error over body-contour voxels only (mask channel 7 = BODY).
    # Avoids penalising predictions in air voxels where dose is trivially zero.
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1)


def structure_dmean_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
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


def train_one_epoch(
    generator:     UnetGenerator3d,
    discriminator: NLayerDiscriminator,
    dataloader:    DataLoader,
    optimizer_G:   torch.optim.Optimizer,
    optimizer_D:   torch.optim.Optimizer,
    criterion_GAN: GANLoss,
    lambda_voxel:  float,
    lambda_dvh:    float,
    lambda_grad:   float,
    device:        torch.device,
) -> dict:
    """
    One full pass over the training set.
    Returns a dict of average losses for logging.
    """

    generator.train()
    discriminator.train()

    total_loss_D    = 0.0
    total_loss_G    = 0.0
    total_loss_L1   = 0.0
    total_loss_dvh  = 0.0
    total_loss_grad = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        real_input   = batch["input"].to(device)         # (B, 9, D, H, W)
        real_dose    = batch["dose"].to(device)          # (B, 1, D, H, W)
        body_mask    = real_input[:, 7:8]                # (B, 1, D, H, W)
        ptv_mask     = batch["ptv_mask"].to(device)      # (B, 1, D, H, W)
        bladder_mask = batch["bladder_mask"].to(device)  # (B, 1, D, H, W)
        rectum_mask  = batch["rectum_mask"].to(device)   # (B, 1, D, H, W)

        # ── Phase 1: Train the discriminator ──────────────────────────────
        fake_dose = generator(real_input).detach()

        real_pair = torch.cat([real_input, real_dose], dim=1)
        pred_real = discriminator(real_pair)
        loss_D_real = criterion_GAN(pred_real, target_is_real=True)

        fake_pair = torch.cat([real_input, fake_dose], dim=1)
        pred_fake = discriminator(fake_pair)
        loss_D_fake = criterion_GAN(pred_fake, target_is_real=False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ── Phase 2: Train the generator ──────────────────────────────────
        fake_dose = generator(real_input)

        fake_pair = torch.cat([real_input, fake_dose], dim=1)
        pred_fake = discriminator(fake_pair)

        loss_G_adv   = criterion_GAN(pred_fake, target_is_real=True)
        # L1 loss outperforms MSE for dose prediction — penalises outliers less aggressively and produces sharper predictions.
        loss_G_voxel = masked_l1(fake_dose, real_dose, body_mask)
        loss_G_dvh   = (
            structure_dmean_loss(fake_dose, real_dose, ptv_mask) +
            structure_dmean_loss(fake_dose, real_dose, bladder_mask) +
            structure_dmean_loss(fake_dose, real_dose, rectum_mask)
        )
        loss_G_grad  = gradient_magnitude_loss(fake_dose, real_dose)

        loss_G = (loss_G_adv
                  + lambda_voxel * loss_G_voxel
                  + lambda_dvh   * loss_G_dvh
                  + lambda_grad  * loss_G_grad)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        total_loss_D    += loss_D.item()
        total_loss_G    += loss_G.item()
        total_loss_L1   += loss_G_voxel.item()
        total_loss_dvh  += loss_G_dvh.item()
        total_loss_grad += loss_G_grad.item()

    n = len(dataloader)
    return {
        "loss_D":     total_loss_D    / n,
        "loss_G":     total_loss_G    / n,
        "train_L1":   total_loss_L1   / n,
        "train_dvh":  total_loss_dvh  / n,
        "train_grad": total_loss_grad / n,
    }

DOSE_SCALE = 50.0


def validate_dvh(
    generator:  UnetGenerator3d,
    val_loader: DataLoader,
    device:     torch.device,
) -> tuple:
    """
    Returns (val_L1, val_dvh_score).

    val_L1       — body-masked L1 in normalised units (stored in checkpoint for
                   stored in checkpoint for evaluate.py).
    val_dvh_score — mean|Δ PTV D95| + mean|Δ Bladder Dmean| + mean|Δ Rectum Dmean|
                   in Gy. Lower is better. Used for early stopping.
    """
    generator.eval()
    total_l1 = 0.0
    ptv_d95_errs, bladder_dmean_errs, rectum_dmean_errs = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            real_input = batch["input"].to(device)
            real_dose  = batch["dose"].to(device)
            body_mask  = real_input[:, 7:8]

            fake_dose = generator(real_input)
            total_l1 += masked_l1(fake_dose, real_dose, body_mask).item()

            pred_gy = fake_dose[0, 0].cpu().numpy() * DOSE_SCALE
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
        (np.mean(ptv_d95_errs)      if ptv_d95_errs      else 0.0) +
        (np.mean(bladder_dmean_errs) if bladder_dmean_errs else 0.0) +
        (np.mean(rectum_dmean_errs)  if rectum_dmean_errs  else 0.0)
    )

    generator.train()
    return total_l1 / len(val_loader), float(val_dvh_score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None,
                        help="Override cfg.FOLD (0–4). If omitted, uses value in config_dosegan.py.")
    geom_group = parser.add_mutually_exclusive_group()
    geom_group.add_argument("--geom",    dest="geom", action="store_true",  default=None,
                            help="Force geom mode: USE_GEOM_CHANNELS=True, INPUT_NC=14.")
    geom_group.add_argument("--no-geom", dest="geom", action="store_false",
                            help="Force baseline mode: USE_GEOM_CHANNELS=False, INPUT_NC=9.")
    flip_group = parser.add_mutually_exclusive_group()
    flip_group.add_argument("--flip",    dest="flips", action="store_true",  default=None,
                            help="Enable spatial flip augmentation.")
    flip_group.add_argument("--no-flip", dest="flips", action="store_false",
                            help="Disable spatial flip augmentation (default for all conditions).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing best checkpoint instead of refusing to start.")
    parser.add_argument("--run-name", dest="run_name", type=str, default=None,
                        help="Override cfg.RUN_NAME. Applied last, after all other flag rewrites.")
    args, _ = parser.parse_known_args()
    if args.fold is not None:
        cfg.FOLD = args.fold
    if args.flips is not None:
        cfg.USE_FLIPS = args.flips
    if args.geom is not None:
        cfg.USE_GEOM_CHANNELS = args.geom
        cfg.INPUT_NC = 14 if args.geom else 9
        if args.geom and "geom" not in cfg.RUN_NAME:
            cfg.RUN_NAME = cfg.RUN_NAME.replace("_snellius", "_geom_snellius")
        elif not args.geom and "_geom_" in cfg.RUN_NAME:
            cfg.RUN_NAME = cfg.RUN_NAME.replace("_geom_", "_")
    if args.run_name is not None:
        cfg.RUN_NAME = args.run_name

    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.CKPT_DIR / f"{cfg.RUN_NAME}_fold{cfg.FOLD}_best.pt"
    if ckpt_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint: {ckpt_path}\n"
            f"Pass --overwrite (or set OVERWRITE=1 for the sbatch) to replace it, "
            f"or delete it first."
        )

    # ── W&B initialisation ─────────────────────────────────────────────────
    # This creates a new run in your W&B project. Every hyperparameter is
    # logged so you can reproduce any run exactly from the dashboard.
    wandb.init(
        project  = cfg.PROJECT_NAME,
        name     = f"{cfg.RUN_NAME}_fold{cfg.FOLD}",
        group    = cfg.RUN_NAME,   # collapses all 5 folds into one experiment row in the UI
        job_type = "train",
        config  = {
            "fold":         cfg.FOLD,
            "epochs":       cfg.EPOCHS,
            "batch_size":   cfg.BATCH_SIZE,
            "lr_G":         cfg.LR_G,
            "lr_D":         cfg.LR_D,
            "lambda_voxel": cfg.LAMBDA_VOXEL,
            "lambda_dvh":   cfg.LAMBDA_DVH,
            "lambda_grad":  cfg.LAMBDA_GRAD,
            "ngf":          cfg.NGF,
            "ndf":          cfg.NDF,
            "n_layers":     cfg.N_LAYERS,
            "use_lsgan":           cfg.USE_LSGAN,
            "early_stop_patience": cfg.EARLY_STOP_PATIENCE,
        }
    )

    # Seed all RNGs before DataLoader creation. MONAI's Rand* transforms draw
    # from their own RNG (separate from PyTorch), so monai.utils.set_determinism
    # is required for full reproducibility alongside the standard seeds.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    import monai.utils
    monai.utils.set_determinism(seed=42)

    def _seed_worker(worker_id: int) -> None:
        random.seed(42 + worker_id)
        np.random.seed(42 + worker_id)
        torch.manual_seed(42 + worker_id)

    # ── Datasets and dataloaders ───────────────────────────────────────────
    train_ds = LUNDPROBEDataset(
        split_csv=cfg.SPLIT_CSV, pickle_dir=cfg.PICKLE_DIR,
        split="train", fold=cfg.FOLD,
        use_geom_channels=cfg.USE_GEOM_CHANNELS,
        use_flips=cfg.USE_FLIPS,
    )
    val_ds = LUNDPROBEDataset(
        split_csv=cfg.SPLIT_CSV, pickle_dir=cfg.PICKLE_DIR,
        split="val", fold=cfg.FOLD,
        use_geom_channels=cfg.USE_GEOM_CHANNELS,
        use_flips=cfg.USE_FLIPS,
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

    # ── Models, optimizers, losses ─────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on: {device} | fold={cfg.FOLD}")

    from training.manifest import write_start, write_end
    eval_csv = Path("outputs/evaluation") / f"{cfg.RUN_NAME}_fold{cfg.FOLD}_val.csv"
    manifest_path = write_start(
        run_name        = cfg.RUN_NAME,
        fold            = cfg.FOLD,
        model_type      = "dosegan",
        cfg             = cfg,
        config_file     = "configs/config_dosegan.py",
        checkpoint_path = cfg.CKPT_DIR / f"{cfg.RUN_NAME}_fold{cfg.FOLD}_best.pt",
        eval_csv_path   = eval_csv,
        seed            = 42,
    )

    generator = UnetGenerator3d(
        input_nc=cfg.INPUT_NC, output_nc=cfg.OUTPUT_NC,
        ngf=cfg.NGF, output_activation=cfg.OUTPUT_ACTIVATION,
    ).to(device)

    discriminator = NLayerDiscriminator(
        input_nc=cfg.INPUT_NC + cfg.OUTPUT_NC,  # 9 + 1 = 10
        ndf=cfg.NDF, n_layers=cfg.N_LAYERS,
    ).to(device)

    n_params_G = sum(p.numel() for p in generator.parameters())
    n_params_D = sum(p.numel() for p in discriminator.parameters())
    log.info(f"Generator params: {n_params_G:,} ({n_params_G/1e6:.1f}M)")
    log.info(f"Discriminator params: {n_params_D:,} ({n_params_D/1e6:.1f}M)")

    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=cfg.LR_G, betas=(cfg.BETA1, cfg.BETA2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2)
    )

    criterion_GAN = GANLoss(use_lsgan=cfg.USE_LSGAN).to(device)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_L1    = float("inf")   # stored in checkpoint for eval script compat.
    best_dvh_score = float("inf")   # used for early stopping decision
    epochs_no_improve = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        train_losses = train_one_epoch(
            generator, discriminator, train_loader,
            optimizer_G, optimizer_D,
            criterion_GAN, cfg.LAMBDA_VOXEL, cfg.LAMBDA_DVH, cfg.LAMBDA_GRAD, device,
        )
        val_l1, val_dvh = validate_dvh(generator, val_loader, device)

        wandb.log({
            "epoch":       epoch,
            "loss_D":      train_losses["loss_D"],
            "loss_G":      train_losses["loss_G"],
            "train_L1":    train_losses["train_L1"],
            "train_dvh":   train_losses["train_dvh"],
            "train_grad":  train_losses["train_grad"],
            "val_L1":      val_l1,
            "val_dvh":     val_dvh,
        })

        log.info(
            f"Epoch {epoch:03d} | "
            f"loss_D: {train_losses['loss_D']:.4f} | "
            f"loss_G: {train_losses['loss_G']:.4f} | "
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
                "generator":      generator.state_dict(),
                "discriminator":  discriminator.state_dict(),
                "optimizer_G":    optimizer_G.state_dict(),
                "optimizer_D":    optimizer_D.state_dict(),
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

    # Upload the best checkpoint to W&B exactly once, after training ends.
    # Saving inside the val-improvement branch would re-upload the full 1.5 GB
    # checkpoint on every improvement.
    best_ckpt_path = cfg.CKPT_DIR / f"{cfg.RUN_NAME}_fold{cfg.FOLD}_best.pt"
    if best_ckpt_path.exists():
        wandb.save(str(best_ckpt_path), policy="now")

    import json as _json
    _start_utc = _json.load(open(manifest_path))["training_start_utc"]
    write_end(
        manifest_path  = manifest_path,
        wandb_run_id   = wandb.run.id if wandb.run else None,
        best_val_loss  = best_val_L1,
        best_dvh_score = best_dvh_score,
        epochs_trained = epoch,
        start_utc      = _start_utc,
    )

    wandb.finish()  # cleanly close the W&B run when training ends

if __name__ == "__main__":
    main()