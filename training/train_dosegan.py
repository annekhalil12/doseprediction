# train_dosegan.py
# Training script specific to the DoseGAN model.
# Shared preprocessing config lives in config_preprocessing_shared.py.
# DoseGAN hyperparameters (lr, lambda, ngf) live in config_dosegan.py.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from dataset import LUNDPROBEDataset
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
) -> dict:
    """
    One full pass over the training set.
    Returns a dict of average losses for logging.
    """

    generator.train()
    discriminator.train()

    total_loss_D = 0.0
    total_loss_G = 0.0

    for batch in dataloader:
        real_input = batch["input"].to(device)  # (B, 9, D, H, W)  — sCT + structure masks
        real_dose  = batch["dose"].to(device)   # (B, 1, D, H, W)  — ground-truth dose

        # ── Phase 1: Train the discriminator ──────────────────────────────
        # Generator is not updated here — we only call optimizer_D.step()

        fake_dose = generator(real_input).detach()
        # .detach() is critical: it cuts the fake dose off from the generator's
        # computation graph. Without it, the discriminator loss would
        # accidentally flow gradients back into the generator during Phase 1.

        # Discriminator judges real pairs (sCT + real dose)
        real_pair = torch.cat([real_input, real_dose], dim=1)  # (B, 10, D, H, W)
        pred_real = discriminator(real_pair)
        loss_D_real = criterion_GAN(pred_real, target_is_real=True)

        # Discriminator judges fake pairs (sCT + generated dose)
        fake_pair = torch.cat([real_input, fake_dose], dim=1)  # (B, 10, D, H, W)
        pred_fake = discriminator(fake_pair)
        loss_D_fake = criterion_GAN(pred_fake, target_is_real=False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ── Phase 2: Train the generator ──────────────────────────────────
        # Discriminator weights are not updated here — only optimizer_G.step()

        fake_dose = generator(real_input)
        # No .detach() here — we need gradients to flow back into the generator

        fake_pair = torch.cat([real_input, fake_dose], dim=1)
        pred_fake = discriminator(fake_pair)

        # Adversarial loss: fool the discriminator into thinking fake is real
        loss_G_adv = criterion_GAN(pred_fake, target_is_real=True)

        # Voxel loss: stay close to the real dose map
        loss_G_voxel = criterion_voxel(fake_dose, real_dose)

        # Combined generator loss
        loss_G = loss_G_adv + lambda_voxel * loss_G_voxel

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

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
    ) -> float:
    """
    Runs the generator on the validation set and returns average L1 loss.
    The discriminator plays no role here — we only care about dose accuracy
    at validation time, not whether the output looks realistic to a critic.
    """

    generator.eval()  # disables dropout and batchnorm randomness

    total_loss = 0.0

    with torch.no_grad():
        # torch.no_grad() tells PyTorch not to track gradients at all.
        # During validation we are never calling .backward(), so storing
        # the computation graph would just waste GPU memory.
        for batch in val_loader:
            real_input = batch["input"].to(device)
            real_dose  = batch["dose"].to(device)

            fake_dose = generator(real_input)
            loss      = criterion_voxel(fake_dose, real_dose)
            total_loss += loss.item()

    generator.train()  # switch back to training mode before returning
    return total_loss / len(val_loader)

def main():
    # ── Paths ──────────────────────────────────────────────────────────────
    split_csv  = Path("outputs/split.csv")
    pickle_dir = Path("outputs/pickles")
    ckpt_dir   = Path("outputs/checkpoints_dosegan")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Hyperparameters ────────────────────────────────────────────────────
    # These will eventually move to config_dosegan.py — kept here for now
    # so everything is visible in one place while we get training running.
    FOLD        = 0       # which fold is validation this run
    EPOCHS      = 100
    BATCH_SIZE  = 1       # 3D volumes are large — batch size 1 is standard
    NUM_WORKERS = 4       # parallel pickle loading; reduce to 0 if errors
    LAMBDA      = 100     # voxel loss weight relative to adversarial loss

    # ── Datasets and dataloaders ───────────────────────────────────────────
    train_ds = LUNDPROBEDataset(
        split_csv=split_csv, pickle_dir=pickle_dir,
        split="train", fold=FOLD,
    )
    val_ds = LUNDPROBEDataset(
        split_csv=split_csv, pickle_dir=pickle_dir,
        split="val", fold=FOLD,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── Models, optimizers, losses ─────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on: {device}")

    generator = UnetGenerator3d(
        input_nc=9, output_nc=1, num_downs=5, ngf=64,
    ).to(device)

    discriminator = NLayerDiscriminator(
        input_nc=10, ndf=64, n_layers=3,
    ).to(device)

    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )

    criterion_GAN   = GANLoss(use_lsgan=True).to(device)
    criterion_voxel = nn.L1Loss()

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_losses = train_one_epoch(
            generator, discriminator,
            train_loader,
            optimizer_G, optimizer_D,
            criterion_GAN, criterion_voxel,
            LAMBDA, device,
        )

        # Validate — generator only, no discriminator update
        val_loss = validate(generator, val_loader, criterion_voxel, device)

        log.info(
            f"Epoch {epoch:03d} | "
            f"loss_D: {train_losses['loss_D']:.4f} | "
            f"loss_G: {train_losses['loss_G']:.4f} | "
            f"val_L1: {val_loss:.4f}"
        )

        # ── Save checkpoint if validation loss improved ────────────────────
        # We save based on voxel loss, not GAN loss — GAN loss is harder
        # to interpret as a quality signal on its own.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / f"dosegan_fold{FOLD}_best.pt"
            torch.save({
                "epoch":           epoch,
                "generator":       generator.state_dict(),
                "discriminator":   discriminator.state_dict(),
                "optimizer_G":     optimizer_G.state_dict(),
                "optimizer_D":     optimizer_D.state_dict(),
                "best_val_loss":   best_val_loss,
            }, ckpt_path)
            log.info(f"  ✓ Checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    main()

    
    
