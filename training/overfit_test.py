"""
training/overfit_test.py
========================
Sanity check 4: overfit test on 1–2 patients.

A healthy model+loss pipeline should drive training L1 close to zero
on a single patient within ~50 epochs. If it cannot, something is wrong
with the data loading, model architecture, or loss computation.

Usage:
    PYTHONPATH=. python3 -m training.overfit_test --model dosegan
    PYTHONPATH=. python3 -m training.overfit_test --model unet3d
    PYTHONPATH=. python3 -m training.overfit_test --model dosegan --geom
    PYTHONPATH=. python3 -m training.overfit_test --model unet3d   --n-patients 2

Output:
    outputs/sanity/overfit_{model}[_geom].png   — loss curve + dose comparison
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("outputs/sanity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EPOCHS   = 200
TARGET_L1    = 0.01   # body-masked L1 below this = overfit confirmed
PATIENCE     = 50     # stop if no improvement for this many epochs


def run_overfit(model_type: str, use_geom: bool, n_patients: int) -> None:
    tag = f"{model_type}{'_geom' if use_geom else ''}"
    print(f"\n── Overfit test: {tag} | {n_patients} patient(s) ──────────────")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # ── Load config and model ──────────────────────────────────────────────
    if model_type == "dosegan":
        from configs import config_dosegan as cfg
        from models.dosegan import UnetGenerator3d
        cfg.USE_GEOM_CHANNELS = use_geom
        cfg.INPUT_NC = 14 if use_geom else 9
        model = UnetGenerator3d(
            input_nc=cfg.INPUT_NC, output_nc=cfg.OUTPUT_NC,
            ngf=cfg.NGF, output_activation=cfg.OUTPUT_ACTIVATION,
        ).to(device)
    else:
        from configs import config_unet3d as cfg
        from models.unet3d import UNet3d
        cfg.USE_GEOM_CHANNELS = use_geom
        cfg.INPUT_NC = 14 if use_geom else 9
        model = UNet3d(
            in_channels=cfg.INPUT_NC, out_channels=cfg.OUTPUT_NC,
            channels=cfg.CHANNELS, strides=cfg.STRIDES,
            output_activation=cfg.OUTPUT_ACTIVATION,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # ── Dataset: take first n_patients from fold 0 train set ──────────────
    from training.dataset import LUNDPROBEDataset
    full_ds = LUNDPROBEDataset(
        cfg.SPLIT_CSV, cfg.PICKLE_DIR,
        split="train", fold=0,
        use_geom_channels=use_geom, use_flips=False,
    )
    ds      = Subset(full_ds, list(range(n_patients)))
    loader  = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    pids    = [full_ds.patient_ids[i] for i in range(n_patients)]
    print(f"   Patients: {pids}")

    # ── Training loop ──────────────────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-4)
    l1_loss   = nn.L1Loss(reduction="none")

    losses = []
    best   = float("inf")
    no_imp = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            inp  = batch["input"].to(device)
            dose = batch["dose"].to(device)
            body = batch["input"][:, 7:8].to(device)   # channel 7 = BODY

            optimiser.zero_grad()
            pred = model(inp)
            loss = (l1_loss(pred, dose) * body).sum() / (body.sum() + 1e-8)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        losses.append(avg)

        if avg < best:
            best   = avg
            no_imp = 0
        else:
            no_imp += 1

        if epoch % 20 == 0 or avg < TARGET_L1:
            print(f"   Epoch {epoch:03d} | L1={avg:.4f} | best={best:.4f}")

        if avg < TARGET_L1:
            print(f"   ✓ Overfit confirmed at epoch {epoch} (L1={avg:.4f} < {TARGET_L1})")
            break
        if no_imp >= PATIENCE:
            print(f"   ✗ No improvement for {PATIENCE} epochs — possible pipeline issue.")
            break

    # ── Prediction vs ground truth plot ───────────────────────────────────
    model.eval()
    with torch.no_grad():
        batch  = next(iter(loader))
        inp    = batch["input"].to(device)
        dose   = batch["dose"].to(device)
        pred   = model(inp)

    dose_np = dose[0, 0].cpu().numpy()
    pred_np = pred[0, 0].cpu().numpy()
    err_np  = pred_np - dose_np
    z       = dose_np.shape[0] // 2

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"Overfit test: {tag} | {pids[0]} | best L1={best:.4f} | "
        f"{'✓ PASS' if best < TARGET_L1 else '✗ FAIL'}",
        fontsize=11,
    )
    axes[0].imshow(inp[0, 8, z].cpu().numpy(), cmap="gray",   origin="lower"); axes[0].set_title("sCT (input ch8)")
    axes[1].imshow(dose_np[z],                 cmap="hot",    origin="lower"); axes[1].set_title("Ground truth dose")
    axes[2].imshow(pred_np[z],                 cmap="hot",    origin="lower"); axes[2].set_title("Predicted dose")
    axes[3].imshow(err_np[z],  cmap="RdBu_r",  origin="lower",
                   vmin=-0.3, vmax=0.3);                                       axes[3].set_title("Error (pred−true)")
    for ax in axes:
        ax.axis("off")

    # Loss curve inset
    ax_loss = fig.add_axes([0.01, 0.55, 0.10, 0.35])
    ax_loss.plot(losses, lw=1, color="steelblue")
    ax_loss.axhline(TARGET_L1, color="red", lw=0.8, ls="--")
    ax_loss.set_title("L1", fontsize=8)
    ax_loss.tick_params(labelsize=7)

    plt.tight_layout()
    out = OUT_DIR / f"overfit_{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {out}")

    if best >= TARGET_L1:
        print(f"\n   WARNING: model did not overfit 1 patient (best L1={best:.4f}).")
        print("   Check: data loading, model forward pass, loss computation.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overfit test on 1–2 patients.")
    parser.add_argument("--model", choices=["dosegan", "unet3d"], required=True)
    parser.add_argument("--geom",  action="store_true", default=False,
                        help="Use 14-channel geom input.")
    parser.add_argument("--n-patients", type=int, default=1, dest="n_patients",
                        help="Number of patients to overfit on (default: 1).")
    args = parser.parse_args()

    run_overfit(args.model, args.geom, args.n_patients)
