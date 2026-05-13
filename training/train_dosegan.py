# train_dosegan.py
# Training script specific to the DoseGAN model.
# Shared preprocessing config lives in config_preprocessing_shared.py.
# DoseGAN hyperparameters (lr, lambda, ngf) live in config_dosegan.py.

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import wandb

from configs import config_dosegan as cfg  # all hyperparameters live here
from training.dataset import LUNDPROBEDataset
from models.dosegan import UnetGenerator3d, NLayerDiscriminator, GANLoss

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train_one_epoch(
    generator:     UnetGenerator3d,
    discriminator: NLayerDiscriminator,
    dataloader:    DataLoader,
    optimizer_G:   torch.optim.Optimizer,
    optimizer_D:   torch.optim.Optimizer,
    criterion_GAN:   GANLoss,
    criterion_voxel: nn.L1Loss,
    lambda_voxel:  float,
    device:        torch.device,
    scaler_G:      GradScaler,
    scaler_D:      GradScaler,
    use_amp:       bool,
) -> dict:
    """
    One full pass over the training set.
    Returns a dict of average losses for logging.
    """

    generator.train()
    discriminator.train()

    total_loss_D = 0.0
    total_loss_G = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        real_input = batch["input"].to(device)  # (B, 9, D, H, W)  — sCT + structure masks
        real_dose  = batch["dose"].to(device)   # (B, 1, D, H, W)  — ground-truth dose

        # ── Phase 1: Train the discriminator ──────────────────────────────
        with autocast(enabled=use_amp):
            fake_dose = generator(real_input).detach()
            # .detach() is critical: it cuts the fake dose off from the generator's
            # computation graph. Without it, the discriminator loss would
            # accidentally flow gradients back into the generator during Phase 1.

            real_pair = torch.cat([real_input, real_dose], dim=1)  # (B, 10, D, H, W)
            pred_real = discriminator(real_pair)
            loss_D_real = criterion_GAN(pred_real, target_is_real=True)

            fake_pair = torch.cat([real_input, fake_dose], dim=1)  # (B, 10, D, H, W)
            pred_fake = discriminator(fake_pair)
            loss_D_fake = criterion_GAN(pred_fake, target_is_real=False)

            loss_D = (loss_D_real + loss_D_fake) * 0.5

        optimizer_D.zero_grad()
        scaler_D.scale(loss_D).backward()
        scaler_D.step(optimizer_D)
        scaler_D.update()

        # ── Phase 2: Train the generator ──────────────────────────────────
        with autocast(enabled=use_amp):
            fake_dose = generator(real_input)
            # No .detach() here — we need gradients to flow back into the generator

            fake_pair = torch.cat([real_input, fake_dose], dim=1)
            pred_fake = discriminator(fake_pair)

            loss_G_adv   = criterion_GAN(pred_fake, target_is_real=True)
            loss_G_voxel = criterion_voxel(fake_dose, real_dose)
            loss_G       = loss_G_adv + lambda_voxel * loss_G_voxel

        optimizer_G.zero_grad()
        scaler_G.scale(loss_G).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()

        total_loss_D += loss_D.item()
        total_loss_G += loss_G.item()

    n = len(dataloader)
    return {
        "loss_D": total_loss_D / n,
        "loss_G": total_loss_G / n,
    }

def validate(
    generator:       UnetGenerator3d,
    val_loader:      DataLoader,
    criterion_voxel: nn.L1Loss,
    device:          torch.device,
    use_amp:         bool,
    ) -> float:
    """
    Runs the generator on the validation set and returns average L1 loss.
    The discriminator plays no role here — we only care about dose accuracy
    at validation time, not whether the output looks realistic to a critic.
    """

    generator.eval()  # disables dropout and batchnorm randomness

    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            real_input = batch["input"].to(device)
            real_dose  = batch["dose"].to(device)

            with autocast(enabled=use_amp):
                fake_dose = generator(real_input)
                loss      = criterion_voxel(fake_dose, real_dose)
            total_loss += loss.item()

    generator.train()  # switch back to training mode before returning
    return total_loss / len(val_loader)

def main():
    # ── W&B initialisation ─────────────────────────────────────────────────
    # This creates a new run in your W&B project. Every hyperparameter is
    # logged so you can reproduce any run exactly from the dashboard.
    wandb.init(
        project = cfg.PROJECT_NAME,
        name    = cfg.RUN_NAME,
        config  = {
            "fold":                    cfg.FOLD,
            "epochs":                  cfg.EPOCHS,
            "batch_size":              cfg.BATCH_SIZE,
            "lr_G":                    cfg.LR_G,
            "lr_D":                    cfg.LR_D,
            "lambda_voxel":            cfg.LAMBDA_VOXEL,
            "ngf":                     cfg.NGF,
            "ndf":                     cfg.NDF,
            "n_layers":                cfg.N_LAYERS,
            "use_lsgan":               cfg.USE_LSGAN,
            "use_amp":                 cfg.USE_AMP,
            "early_stopping_patience": cfg.EARLY_STOPPING_PATIENCE,
        }
    )

    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Datasets and dataloaders ───────────────────────────────────────────
    train_ds = LUNDPROBEDataset(
        split_csv=cfg.SPLIT_CSV, pickle_dir=cfg.PICKLE_DIR,
        split="train", fold=cfg.FOLD,
    )
    val_ds = LUNDPROBEDataset(
        split_csv=cfg.SPLIT_CSV, pickle_dir=cfg.PICKLE_DIR,
        split="val", fold=cfg.FOLD,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )

    # ── Models, optimizers, losses ─────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on: {device}")

    generator = UnetGenerator3d(
        input_nc=cfg.INPUT_NC, output_nc=cfg.OUTPUT_NC,
        num_downs=5, ngf=cfg.NGF,
    ).to(device)

    discriminator = NLayerDiscriminator(
        input_nc=cfg.INPUT_NC + cfg.OUTPUT_NC,  # 9 + 1 = 10
        ndf=cfg.NDF, n_layers=cfg.N_LAYERS,
    ).to(device)

    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=cfg.LR_G, betas=(cfg.BETA1, cfg.BETA2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2)
    )

    criterion_GAN = GANLoss(use_lsgan=cfg.USE_LSGAN).to(device)
    criterion_voxel = nn.L1Loss()

    scaler_G = GradScaler(enabled=cfg.USE_AMP)
    scaler_D = GradScaler(enabled=cfg.USE_AMP)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss   = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        train_losses = train_one_epoch(
            generator, discriminator, train_loader,
            optimizer_G, optimizer_D,
            criterion_GAN, criterion_voxel,
            cfg.LAMBDA_VOXEL, device,
            scaler_G, scaler_D, cfg.USE_AMP,
        )
        val_loss = validate(generator, val_loader, criterion_voxel, device, cfg.USE_AMP)

        wandb.log({
            "epoch":    epoch,
            "loss_D":   train_losses["loss_D"],
            "loss_G":   train_losses["loss_G"],
            "val_L1":   val_loss,
        })

        log.info(
            f"Epoch {epoch:03d} | "
            f"loss_D: {train_losses['loss_D']:.4f} | "
            f"loss_G: {train_losses['loss_G']:.4f} | "
            f"val_L1: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = cfg.CKPT_DIR / f"dosegan_fold{cfg.FOLD}_best.pt"
            torch.save({
                "epoch":         epoch,
                "generator":     generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_G":   optimizer_G.state_dict(),
                "optimizer_D":   optimizer_D.state_dict(),
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            log.info(f"  ✓ Checkpoint saved: {ckpt_path}")
            wandb.save(str(ckpt_path))  # also back up checkpoint to W&B cloud
        else:
            epochs_no_improve += 1
            log.info(f"  No improvement for {epochs_no_improve}/{cfg.EARLY_STOPPING_PATIENCE} epochs")
            if epochs_no_improve >= cfg.EARLY_STOPPING_PATIENCE:
                log.info(f"Early stopping triggered at epoch {epoch}")
                wandb.log({"early_stop_epoch": epoch})
                break

    wandb.finish()  # cleanly close the W&B run when training ends

if __name__ == "__main__":
    main()