# training/train_unet3d.py
# Supervised training for the 3D U-Net baseline.
# Simpler than train_dosegan.py — one model, one loss, no adversarial loop.
# Hyperparameters live in configs/config_unet3d.py.

import argparse
import sys
import logging
import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config_unet3d as cfg
from training.dataset import LUNDPROBEDataset
from models.unet3d import UNet3d

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


def masked_l1(pred, target, mask):
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_l1 = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        real_input = batch["input"].to(device)
        real_dose  = batch["dose"].to(device)
        body_mask  = real_input[:, 7:8]

        pred  = model(real_input)
        loss  = masked_l1(pred, real_dose, body_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_l1 += loss.item()

    return total_l1 / len(loader)


def validate(model, loader, device):
    model.eval()
    total_l1 = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            real_input = batch["input"].to(device)
            real_dose  = batch["dose"].to(device)
            body_mask  = real_input[:, 7:8]

            pred = model(real_input)
            loss = masked_l1(pred, real_dose, body_mask)
            total_l1 += loss.item()

    model.train()
    return total_l1 / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None,
                        help="Override cfg.FOLD (0–4).")
    parser.add_argument("--activation", choices=["sigmoid", "tanh"], default=None,
                        help="Override cfg.OUTPUT_ACTIVATION. Rewrites the activation token in RUN_NAME.")
    args, _ = parser.parse_known_args()
    if args.fold is not None:
        cfg.FOLD = args.fold
    if args.activation is not None:
        # Rewrite the activation token in RUN_NAME so W&B groups stay distinct.
        cfg.RUN_NAME = cfg.RUN_NAME.replace(cfg.OUTPUT_ACTIVATION, args.activation)
        cfg.OUTPUT_ACTIVATION = args.activation

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
            "channels":           cfg.CHANNELS,
            "strides":            cfg.STRIDES,
            "num_res_units":      cfg.NUM_RES_UNITS,
            "output_activation":  cfg.OUTPUT_ACTIVATION,
            "early_stop_patience": cfg.EARLY_STOP_PATIENCE,
        }
    )

    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Seed before DataLoader creation so worker subprocesses inherit a
    # deterministic parent RNG; worker_init_fn below then offsets each
    # worker so augmentation is reproducible across reruns.
    torch.manual_seed(42)

    def _seed_worker(worker_id: int) -> None:
        torch.manual_seed(42 + worker_id)

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

    best_val_loss    = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        train_l1 = train_one_epoch(model, train_loader, optimizer, device)
        val_l1   = validate(model, val_loader, device)

        wandb.log({"epoch": epoch, "train_L1": train_l1, "val_L1": val_l1})
        log.info(
            f"Epoch {epoch:03d} | train_L1: {train_l1:.4f} | val_L1: {val_l1:.4f}"
        )

        if val_l1 < best_val_loss:
            best_val_loss    = val_l1
            epochs_no_improve = 0
            ckpt_path = cfg.CKPT_DIR / f"unet3d_fold{cfg.FOLD}_best.pt"
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            log.info(f"  ✓ Checkpoint saved: {ckpt_path}")
        else:
            epochs_no_improve += 1
            log.info(f"  No improvement for {epochs_no_improve}/{cfg.EARLY_STOP_PATIENCE} epochs")
            if epochs_no_improve >= cfg.EARLY_STOP_PATIENCE:
                log.info(f"Early stopping at epoch {epoch}")
                break

    ckpt_path = cfg.CKPT_DIR / f"unet3d_fold{cfg.FOLD}_best.pt"
    if ckpt_path.exists():
        wandb.save(str(ckpt_path), policy="now")

    wandb.finish()


if __name__ == "__main__":
    main()
