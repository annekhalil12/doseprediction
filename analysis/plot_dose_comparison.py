"""
analysis/plot_dose_comparison.py

Side-by-side axial dose-slice comparison across all 4 conditions for one patient.

Layout:
  Row 1 — predicted dose (Gy):  GT | DG base | UN base | DG geom | UN geom
  Row 2 — signed error (Gy):    —  | pred−GT  | pred−GT | pred−GT | pred−GT

Structure contours (PTV=red, Rectum=green, Bladder=blue) on every panel.
Axial slice through the PTV centroid.

Usage (from project root, needs GPU or runs on CPU):
    python3 -m analysis.plot_dose_comparison
    python3 -m analysis.plot_dose_comparison --fold 0 --selector median
    python3 -m analysis.plot_dose_comparison --patient-id <id> --fold <N>

Outputs: outputs/analysis/dose_comparison_<patient_id>_fold<N>.png
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--fold",       type=int, default=0)
parser.add_argument("--patient-id", dest="patient_id", default=None)
parser.add_argument("--selector",   choices=["median", "best", "worst"], default="median")
args = parser.parse_args()

FOLD = args.fold

# ── condition definitions ─────────────────────────────────────────────────────

BASE_CKPT = Path("outputs/checkpoints_dosegan")
UNET_CKPT = Path("outputs/checkpoints_unet3d")
EVAL_BASE  = Path("outputs/evaluation/baseline_sigmoid")
EVAL_GEOM  = Path("outputs/evaluation")
PICKLE_DIR = Path("data/pickles")
OUT_DIR    = Path("outputs/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOSE_SCALE = 50.0

# (label, ckpt_path, model_type, input_nc, use_geom, eval_csv_for_patient_sel)
CONDITIONS = [
    ("DoseGAN\nbaseline",
     BASE_CKPT / f"dosegan_ngf32_sigmoid_snellius_fold{FOLD}_best.pt",
     "dosegan", 9, False,
     EVAL_BASE / f"dosegan_ngf32_sigmoid_snellius_fold{FOLD}_val.csv"),
    ("U-Net\nbaseline",
     UNET_CKPT / f"unet3d_ch32_sigmoid_snellius_fold{FOLD}_best.pt",
     "unet3d", 9, False,
     EVAL_BASE / f"unet3d_ch32_sigmoid_snellius_fold{FOLD}_val.csv"),
    ("DoseGAN\ngeom",
     BASE_CKPT / f"dosegan_ngf32_sigmoid_geom_snellius_fold{FOLD}_best.pt",
     "dosegan", 14, True,
     EVAL_GEOM / f"dosegan_ngf32_sigmoid_geom_snellius_fold{FOLD}_val.csv"),
    ("U-Net\ngeom",
     UNET_CKPT / f"unet3d_ch32_sigmoid_geom_snellius_fold{FOLD}_best.pt",
     "unet3d", 14, True,
     EVAL_GEOM / f"unet3d_ch32_sigmoid_geom_snellius_fold{FOLD}_val.csv"),
]

STRUCT_STYLES = {
    "ptv":     {"color": "#FF3333", "lw": 1.6, "label": "PTV"},
    "rectum":  {"color": "#33CC33", "lw": 1.3, "label": "Rectum"},
    "bladder": {"color": "#4488FF", "lw": 1.3, "label": "Bladder"},
}

# ── helpers ───────────────────────────────────────────────────────────────────

def select_patient() -> tuple[str, int]:
    """Return (patient_id, fold) using the DG baseline eval CSV."""
    if args.patient_id:
        return args.patient_id, FOLD

    csv_path = CONDITIONS[0][5]  # DG baseline eval CSV
    rows = sorted(list(csv.DictReader(open(csv_path))),
                  key=lambda r: float(r["body_MAE_Gy"]))
    n = len(rows)
    if args.selector == "median":
        r = rows[n // 2]
    elif args.selector == "best":
        r = rows[0]
    else:
        r = rows[-1]
    return r["patient_id"], int(r["fold"])


def load_pickle(patient_id: str) -> dict:
    pkl_path = PICKLE_DIR / f"{patient_id}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def build_input(data: dict, input_nc: int, use_geom: bool, device) -> torch.Tensor:
    inp = torch.tensor(data["input"][:9], dtype=torch.float32)   # (9, D, H, W)
    if use_geom:
        geom = torch.tensor(data["geom_channels"], dtype=torch.float32)  # (5, D, H, W)
        inp = torch.cat([inp, geom], dim=0)                               # (14, D, H, W)
    assert inp.shape[0] == input_nc, f"Expected {input_nc} channels, got {inp.shape[0]}"
    return inp.unsqueeze(0).to(device)   # (1, C, D, H, W)


def load_model(ckpt_path: Path, model_type: str, input_nc: int, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if model_type == "dosegan":
        from configs import config_dosegan as cfg
        from models.dosegan import UnetGenerator3d
        net = UnetGenerator3d(
            input_nc=input_nc, output_nc=cfg.OUTPUT_NC, ngf=cfg.NGF
        ).to(device)
        net.load_state_dict(ckpt["generator"])
    else:
        from configs import config_unet3d as cfg
        from models.unet3d import UNet3d
        net = UNet3d(
            in_channels=input_nc, out_channels=cfg.OUTPUT_NC,
            channels=cfg.CHANNELS, strides=cfg.STRIDES,
            num_res_units=cfg.NUM_RES_UNITS,
            output_activation=cfg.OUTPUT_ACTIVATION,
        ).to(device)
        key = "model" if "model" in ckpt else "generator"
        net.load_state_dict(ckpt[key])
    net.eval()
    epoch = ckpt.get("epoch", "?")
    val_l1 = ckpt.get("best_val_loss", float("nan"))
    print(f"  Loaded {ckpt_path.name}  epoch={epoch}  val_L1={val_l1:.4f}")
    return net


def ptv_centroid_z(ptv_mask: np.ndarray) -> int:
    vox = np.argwhere(ptv_mask > 0.5)
    return int(vox[:, 0].mean()) if len(vox) > 0 else ptv_mask.shape[0] // 2


def draw_contours(ax, masks: dict, z: int) -> None:
    for struct, style in STRUCT_STYLES.items():
        m = masks.get(struct)
        if m is None or (m > 0.5).sum() == 0:
            continue
        slc = m[z]
        if (slc > 0.5).sum() > 0:
            ax.contour(slc, levels=[0.5],
                       colors=[style["color"]], linewidths=style["lw"])


def body_mae(pred: np.ndarray, true: np.ndarray, body: np.ndarray) -> float:
    return float(np.abs(pred - true)[body > 0.5].mean())


# ── main ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

patient_id, fold = select_patient()
print(f"\nPatient: {patient_id}  fold={fold}")

data      = load_pickle(patient_id)
true_dose = data["dose"].astype(np.float32) * DOSE_SCALE   # (D, H, W) in Gy

inp_arr   = data["input"]   # (9, D, H, W) or more
sct       = inp_arr[8]      # sCT is channel 8

masks = {
    "ptv":     inp_arr[0],
    "rectum":  inp_arr[2],
    "bladder": inp_arr[3],
    "body":    inp_arr[7],
}

z = ptv_centroid_z(masks["ptv"])
print(f"PTV centroid slice z = {z}")

# run inference for all 4 conditions
preds = []
maes  = []
print("\nRunning inference:")
for label, ckpt_path, model_type, input_nc, use_geom, _ in CONDITIONS:
    print(f" {label.replace(chr(10), ' '):<22}", end="  ")
    net   = load_model(ckpt_path, model_type, input_nc, device)
    inp_t = build_input(data, input_nc, use_geom, device)
    with torch.no_grad():
        pred_t = net(inp_t)
    pred_gy = pred_t[0, 0].cpu().numpy() * DOSE_SCALE
    mae     = body_mae(pred_gy, true_dose, masks["body"])
    print(f"body MAE = {mae:.3f} Gy")
    preds.append(pred_gy)
    maes.append(mae)

# ── figure ────────────────────────────────────────────────────────────────────

N_COND  = len(CONDITIONS)
N_COLS  = N_COND + 1   # GT + 4 conditions
N_ROWS  = 2            # dose | error

# shared dose scale: 99th percentile of true dose within body
body_vox = true_dose[masks["body"] > 0.5]
vmax_dose = float(np.percentile(body_vox, 99.5))

# shared error scale: symmetric, driven by worst condition
all_errs = [p - true_dose for p in preds]
emax = float(max(
    max(abs(np.percentile(e[masks["body"] > 0.5], 1)) for e in all_errs),
    max(abs(np.percentile(e[masks["body"] > 0.5], 99)) for e in all_errs),
    2.0
))

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(3.6 * N_COLS, 6.2),
                         gridspec_kw={"hspace": 0.05, "wspace": 0.04})

cmap_dose = "inferno"
cmap_err  = "RdBu_r"

def fmt_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# ── Row 0: dose maps ──────────────────────────────────────────────────────────

# Col 0: ground truth
im_dose = axes[0, 0].imshow(true_dose[z], cmap=cmap_dose,
                             vmin=0, vmax=vmax_dose, origin="lower", aspect="auto")
draw_contours(axes[0, 0], masks, z)
axes[0, 0].set_title("Ground truth", fontsize=9, pad=4)
axes[0, 0].set_ylabel("Predicted dose", fontsize=8.5, labelpad=4)
fmt_ax(axes[0, 0])

for ci, (label, _, _, _, _, _) in enumerate(CONDITIONS):
    ax = axes[0, ci + 1]
    ax.imshow(preds[ci][z], cmap=cmap_dose, vmin=0, vmax=vmax_dose,
              origin="lower", aspect="auto")
    draw_contours(ax, masks, z)
    ax.set_title(f"{label}\nMAE = {maes[ci]:.3f} Gy",
                 fontsize=8.5, pad=4, multialignment="center")
    fmt_ax(ax)

# shared dose colorbar on the right of row 0
cbar0 = fig.colorbar(im_dose, ax=axes[0, :], shrink=0.85, pad=0.01, aspect=25)
cbar0.set_label("Dose (Gy)", fontsize=8)
cbar0.ax.tick_params(labelsize=7.5)

# ── Row 1: error maps ─────────────────────────────────────────────────────────

# Col 0: blank (no error for GT)
axes[1, 0].imshow(np.zeros_like(true_dose[z]), cmap="gray",
                  vmin=0, vmax=1, origin="lower", aspect="auto")
axes[1, 0].set_ylabel("Error  (pred − GT)", fontsize=8.5, labelpad=4)
fmt_ax(axes[1, 0])

for ci, (label, _, _, _, _, _) in enumerate(CONDITIONS):
    ax = axes[1, ci + 1]
    err = all_errs[ci]
    im_err = ax.imshow(err[z], cmap=cmap_err, vmin=-emax, vmax=emax,
                       origin="lower", aspect="auto")
    draw_contours(ax, masks, z)
    fmt_ax(ax)

# shared error colorbar on the right of row 1
cbar1 = fig.colorbar(im_err, ax=axes[1, :], shrink=0.85, pad=0.01, aspect=25)
cbar1.set_label("Error (Gy)", fontsize=8)
cbar1.ax.tick_params(labelsize=7.5)

# ── legend + title ────────────────────────────────────────────────────────────

legend_handles = [
    plt.Line2D([0], [0], color=s["color"], lw=s["lw"], label=s["label"])
    for s in STRUCT_STYLES.values()
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=8, frameon=False, bbox_to_anchor=(0.45, -0.02))

acq_group = patient_id.split("_")[0]
fig.suptitle(
    f"Dose distribution comparison — fold {fold}  |  patient: {patient_id}  "
    f"({acq_group})  |  axial slice z = {z}",
    fontsize=9.5, y=1.01
)

out_path = OUT_DIR / f"dose_comparison_{patient_id}_fold{fold}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out_path}")
plt.close(fig)
