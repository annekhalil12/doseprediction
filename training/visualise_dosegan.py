# training/visualise_dosegan.py
# Loads the best DoseGAN checkpoint and visualises predicted vs ground-truth
# dose for one validation patient. Saves a PNG to outputs/visualisations/.
# Usage: python -m training.visualise_dosegan

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from configs import config_dosegan as cfg
from models.dosegan import UnetGenerator3d
from training.dataset import LUNDPROBEDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load generator from best checkpoint ───────────────────────────────
    # Only the generator is needed at inference time.
    # The discriminator was only used during training.
    ckpt_path = cfg.CKPT_DIR / f"dosegan_fold{cfg.FOLD}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    generator = UnetGenerator3d(
        input_nc=cfg.INPUT_NC, output_nc=cfg.OUTPUT_NC,
        num_downs=cfg.NUM_DOWNS, ngf=cfg.NGF,
    ).to(device)

    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_L1={checkpoint['best_val_loss']:.4f})")

    # ── Load one validation patient ────────────────────────────────────────
    val_ds = LUNDPROBEDataset(
        split_csv=cfg.SPLIT_CSV, pickle_dir=cfg.PICKLE_DIR,
        split="val", fold=cfg.FOLD,
    )
    # DataLoader with batch_size=1 — we only need one patient
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    batch      = next(iter(val_loader))

    real_input = batch["input"].to(device)   # (1, 9, D, H, W)
    real_dose  = batch["dose"].to(device)    # (1, 1, D, H, W)
    patient_id = batch["patient_id"][0]

    # ── Run generator — no gradient tracking needed ────────────────────────
    with torch.no_grad():
        pred_dose = generator(real_input)    # (1, 1, D, H, W)

    # ── Convert to NumPy for plotting ─────────────────────────────────────
    # .squeeze() removes the batch and channel dimensions → (D, H, W)
    # .cpu() moves the tensor from GPU to CPU memory
    # .numpy() converts to NumPy array
    sct_np   = real_input[0, 8].cpu().numpy()   # channel 8 = sCT intensity
    pred_np  = pred_dose[0, 0].cpu().numpy()     # predicted dose
    real_np  = real_dose[0, 0].cpu().numpy()     # ground-truth dose

    # ── Select the axial slice at the PTV centroid ─────────────────────────
    # The PTV mask is in channel 0 of the input tensor.
    ptv_np    = real_input[0, 0].cpu().numpy()   # (D, H, W) binary
    ptv_voxels = np.argwhere(ptv_np > 0.5)
    if len(ptv_voxels) > 0:
        centre_slice = int(ptv_voxels[:, 0].mean())  # mean depth index of PTV
    else:
        centre_slice = sct_np.shape[0] // 2          # fallback to middle slice

    sct_slice  = sct_np[centre_slice]
    pred_slice = pred_np[centre_slice]
    real_slice = real_np[centre_slice]
    diff_slice = np.abs(pred_slice - real_slice)     # absolute voxel-wise error

    # ── Plot four panels ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"DoseGAN — Patient: {patient_id} | "
        f"Slice: {centre_slice} | "
        f"Epoch: {checkpoint['epoch']} | "
        f"val_L1: {checkpoint['best_val_loss']:.4f}",
        fontsize=12
    )

    axes[0].imshow(sct_slice, cmap="gray")
    axes[0].set_title("sCT (anatomy)")
    axes[0].axis("off")

    im1 = axes[1].imshow(pred_slice, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Predicted dose")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(real_slice, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Ground-truth dose")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(diff_slice, cmap="hot", vmin=0, vmax=0.2)
    axes[3].set_title("Absolute error")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir  = Path("outputs/visualisations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dosegan_fold{cfg.FOLD}_{patient_id}_slice{centre_slice}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()