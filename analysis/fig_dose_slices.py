"""
analysis/fig_dose_slices.py
============================
Thesis figure: axial dose distribution comparison for a representative
validation patient.

Layout — 2 rows × 3 columns:
  Row 0 (dose, Gy):   Ground truth | U-Net | DoseGAN
  Row 1 (error, Gy):  (blank)      | U-Net − GT | DoseGAN − GT

Structure contours: PTV (red), Rectum (green), Bladder (blue).
Slice: axial plane through the max-area PTV cross-section.
Patient: median body_MAE_Gy from DoseGAN baseline fold-0 validation CSV.

Output: results/figures/fig_dose_slices.{pdf,png}
Usage (from project root):
    python3 -m analysis.fig_dose_slices
    python3 -m analysis.fig_dose_slices --fold 2 --patient-id <id>
"""

from __future__ import annotations

import argparse
import csv
import functools
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT        = Path(__file__).resolve().parents[1]
OUT_DIR     = ROOT / "results" / "figures"
EVAL_DIR    = ROOT / "outputs" / "evaluation"
DG_CKPT_DIR = ROOT / "outputs" / "checkpoints_dosegan"
UN_CKPT_DIR = ROOT / "outputs" / "checkpoints_unet3d"
PICKLE_DIR  = ROOT / "data" / "pickles"
DOSE_SCALE  = 50.0

OUT_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--fold",       type=int, default=0)
parser.add_argument("--patient-id", dest="patient_id", default=None)
args = parser.parse_args()
FOLD = args.fold


# ── patient selection ─────────────────────────────────────────────────────────

def select_patient() -> str:
    if args.patient_id:
        return args.patient_id
    csv_path = EVAL_DIR / f"dosegan_ngf32_sigmoid_snellius_fold{FOLD}_val.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"DoseGAN eval CSV not found: {csv_path}\n"
            "Run evaluation first: sbatch eval.sbatch MODEL=dosegan FOLD=0"
        )
    rows = sorted(csv.DictReader(open(csv_path)), key=lambda r: float(r["body_MAE_Gy"]))
    r = rows[len(rows) // 2]
    print(f"Patient (median body_MAE): {r['patient_id']}  MAE = {float(r['body_MAE_Gy']):.3f} Gy")
    return r["patient_id"]


# ── model loading ─────────────────────────────────────────────────────────────

def _uses_batchnorm(sd: dict) -> bool:
    return any("running_mean" in k for k in sd)


def load_dosegan(device) -> nn.Module:
    from configs import config_dosegan as cfg
    from models.dosegan import UnetGenerator3d
    ckpt_path = DG_CKPT_DIR / f"dosegan_ngf32_sigmoid_snellius_fold{FOLD}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    sd   = ckpt["generator"]
    norm = nn.BatchNorm3d if _uses_batchnorm(sd) else functools.partial(nn.InstanceNorm3d, affine=True)
    net  = UnetGenerator3d(input_nc=9, output_nc=cfg.OUTPUT_NC, ngf=cfg.NGF, norm_layer=norm).to(device)
    net.load_state_dict(sd, strict=False)
    net.eval()
    print(f"  DoseGAN  epoch={ckpt.get('epoch','?')}  "
          f"val_L1={ckpt.get('best_val_loss', float('nan')):.4f}")
    return net


def load_unet(device) -> nn.Module:
    from configs import config_unet3d as cfg
    from models.unet3d import UNet3d
    ckpt_path = UN_CKPT_DIR / f"unet3d_ch32_sigmoid_snellius_fold{FOLD}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    net  = UNet3d(
        in_channels=9, out_channels=cfg.OUTPUT_NC,
        channels=cfg.CHANNELS, strides=cfg.STRIDES,
        num_res_units=cfg.NUM_RES_UNITS,
        output_activation=cfg.OUTPUT_ACTIVATION,
    ).to(device)
    key = "model" if "model" in ckpt else "generator"
    net.load_state_dict(ckpt[key], strict=False)
    net.eval()
    print(f"  U-Net    epoch={ckpt.get('epoch','?')}  "
          f"val_L1={ckpt.get('best_val_loss', float('nan')):.4f}")
    return net


# ── helpers ───────────────────────────────────────────────────────────────────

STRUCT_STYLES = {
    "ptv":     {"color": "#EE3333", "lw": 1.6, "label": "PTV"},
    "rectum":  {"color": "#22BB22", "lw": 1.3, "label": "Rectum"},
    "bladder": {"color": "#3366EE", "lw": 1.3, "label": "Bladder"},
}
# channel indices: PTV=0, Rectum=1, Bladder=2  (from CLAUDE.md channel map)
_STRUCT_CH = {"ptv": 0, "rectum": 1, "bladder": 2}


def best_ptv_slice(ptv: np.ndarray) -> int:
    areas = [(ptv[z] > 0.5).sum() for z in range(ptv.shape[0])]
    return int(np.argmax(areas)) if max(areas) > 0 else ptv.shape[0] // 2


def draw_contours(ax, inp: np.ndarray, z: int) -> None:
    for struct, style in STRUCT_STYLES.items():
        slc = inp[_STRUCT_CH[struct], z]
        if (slc > 0.5).sum() > 0:
            ax.contour(slc, levels=[0.5],
                       colors=[style["color"]], linewidths=style["lw"])


def fmt_ax(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── main ─────────────────────────────────────────────────────────────────────

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

patient_id = select_patient()

with open(PICKLE_DIR / f"{patient_id}.pkl", "rb") as fh:
    data = pickle.load(fh)

inp_arr   = data["input"].astype(np.float32)          # (9, D, H, W)
true_dose = data["dose"].astype(np.float32) * DOSE_SCALE  # (D, H, W) Gy
body_mask = inp_arr[7]                                 # BODY = ch 7
z_slice   = best_ptv_slice(inp_arr[0])
print(f"Axial slice: z = {z_slice}")

inp_t = torch.from_numpy(inp_arr).unsqueeze(0).to(device)  # (1,9,D,H,W)

print("\nLoading checkpoints:")
dg_net = load_dosegan(device)
un_net = load_unet(device)

print("\nRunning inference...")
with torch.no_grad():
    dg_pred = dg_net(inp_t)[0, 0].cpu().numpy() * DOSE_SCALE
    un_pred = un_net(inp_t)[0, 0].cpu().numpy() * DOSE_SCALE

dg_err = dg_pred - true_dose
un_err = un_pred - true_dose

body_px   = true_dose[body_mask > 0.5]
vmax_dose = float(np.percentile(body_px, 99.5))

all_err_px = np.concatenate([dg_err[body_mask > 0.5], un_err[body_mask > 0.5]])
emax = float(max(abs(np.percentile(all_err_px,  1)),
                 abs(np.percentile(all_err_px, 99)), 2.0))

dg_mae = float(np.abs(dg_err)[body_mask > 0.5].mean())
un_mae = float(np.abs(un_err)[body_mask > 0.5].mean())

# ── figure ────────────────────────────────────────────────────────────────────

plt.rcParams.update({"font.size": 9.5, "font.family": "sans-serif"})
fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.8),
                         gridspec_kw={"hspace": 0.06, "wspace": 0.04})

cmap_dose = "inferno"
cmap_err  = "RdBu_r"

# Row 0 — dose maps
im_dose = axes[0, 0].imshow(true_dose[z_slice], cmap=cmap_dose,
                             vmin=0, vmax=vmax_dose, origin="lower", aspect="equal")
draw_contours(axes[0, 0], inp_arr, z_slice)
axes[0, 0].set_title("Ground truth", fontsize=10, pad=4)
axes[0, 0].set_ylabel("Dose", fontsize=9.5, labelpad=4)
fmt_ax(axes[0, 0])

for ax, pred, label, mae in [
    (axes[0, 1], un_pred, "U-Net",   un_mae),
    (axes[0, 2], dg_pred, "DoseGAN", dg_mae),
]:
    ax.imshow(pred[z_slice], cmap=cmap_dose, vmin=0, vmax=vmax_dose,
              origin="lower", aspect="equal")
    draw_contours(ax, inp_arr, z_slice)
    ax.set_title(f"{label}  (MAE = {mae:.2f} Gy)", fontsize=10, pad=4)
    fmt_ax(ax)

cbar0 = fig.colorbar(im_dose, ax=axes[0, :].tolist(), shrink=0.88, pad=0.01, aspect=28)
cbar0.set_label("Dose (Gy)", fontsize=8.5)
cbar0.ax.tick_params(labelsize=7.5)

# Row 1 — error maps
axes[1, 0].imshow(np.zeros_like(true_dose[z_slice]), cmap="gray",
                  vmin=0, vmax=1, origin="lower", aspect="equal")
axes[1, 0].set_ylabel("Error (pred − GT)", fontsize=9.5, labelpad=4)
fmt_ax(axes[1, 0])

for ax, err, label in [
    (axes[1, 1], un_err, "U-Net"),
    (axes[1, 2], dg_err, "DoseGAN"),
]:
    im_err = ax.imshow(err[z_slice], cmap=cmap_err, vmin=-emax, vmax=emax,
                       origin="lower", aspect="equal")
    draw_contours(ax, inp_arr, z_slice)
    fmt_ax(ax)

cbar1 = fig.colorbar(im_err, ax=axes[1, :].tolist(), shrink=0.88, pad=0.01, aspect=28)
cbar1.set_label("Error (Gy)", fontsize=8.5)
cbar1.ax.tick_params(labelsize=7.5)

legend_handles = [
    plt.Line2D([0], [0], color=s["color"], lw=s["lw"], label=s["label"])
    for s in STRUCT_STYLES.values()
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.45, -0.03))

acq = patient_id.split("_")[0]
fig.suptitle(
    f"Representative patient: {patient_id} ({acq}) | "
    f"axial slice z = {z_slice} (max PTV cross-section) | fold {FOLD}",
    fontsize=8.5, y=1.01
)

out_stem = OUT_DIR / "fig_dose_slices"
fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight", facecolor="white")
fig.savefig(str(out_stem) + ".png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved: {out_stem}.pdf  +  {out_stem}.png")
