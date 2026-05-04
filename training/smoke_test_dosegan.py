# training/smoke_test_dosegan.py
# Runs 2 batches through the full DoseGAN pipeline to verify nothing crashes.
# Run this on your laptop before submitting full training to OneView.
# Usage: python training/smoke_test_dosegan.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from training.dataset import LUNDPROBEDataset
from models.dosegan import UnetGenerator3d, NLayerDiscriminator, GANLoss
from configs import config_dosegan as cfg

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset — just 2 batches ───────────────────────────────────────────
    train_ds = LUNDPROBEDataset(
        split_csv  = cfg.SPLIT_CSV,
        pickle_dir = cfg.PICKLE_DIR,
        split      = "train",
        fold       = cfg.FOLD,
    )
    loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)

    # ── Models ────────────────────────────────────────────────────────────
    generator     = UnetGenerator3d(
        input_nc=cfg.INPUT_NC, output_nc=cfg.OUTPUT_NC,
        num_downs=5, ngf=16,  # use fewer filters for smoke test
    ).to(device)

    discriminator = NLayerDiscriminator(
        input_nc=cfg.INPUT_NC + cfg.OUTPUT_NC,
        ndf=16, n_layers=3,  # use fewer filters and layers for smoke test
    ).to(device)

    # Tell GANLoss to create target tensors on the same device as the model
    tensor_type = torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor
    criterion_GAN = GANLoss(use_lsgan=cfg.USE_LSGAN, tensor=tensor_type).to(device)
    criterion_voxel = nn.L1Loss()

    optimizer_G = torch.optim.Adam(generator.parameters(),     lr=cfg.LR_G)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.LR_D)

    # ── 2 batches only ────────────────────────────────────────────────────
    torch.cuda.empty_cache()

    for i, batch in enumerate(loader):
        if i >= 2:
            break

        real_input = batch["input"].to(device)
        real_dose  = batch["dose"].to(device)

        print(f"Batch {i} | input: {real_input.shape} | dose: {real_dose.shape}")

        # Phase 1 — discriminator
        fake_dose = generator(real_input).detach()
        real_pair = torch.cat([real_input, real_dose], dim=1)
        fake_pair = torch.cat([real_input, fake_dose], dim=1)
        loss_D = (
            criterion_GAN(discriminator(real_pair), target_is_real=True) +
            criterion_GAN(discriminator(fake_pair), target_is_real=False)
        ) * 0.5
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Phase 2 — generator
        fake_dose    = generator(real_input)
        fake_pair    = torch.cat([real_input, fake_dose], dim=1)
        loss_G = (
            criterion_GAN(discriminator(fake_pair), target_is_real=True) +
            cfg.LAMBDA_VOXEL * criterion_voxel(fake_dose, real_dose)
        )
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        print(f"  loss_D: {loss_D.item():.4f} | loss_G: {loss_G.item():.4f}")

    print("\nSmoke test passed — pipeline is intact.")

if __name__ == "__main__":
    main()