"""
analysis/plot_heatmap.py

Performance heatmap across the 4 evaluation conditions:
  DoseGAN baseline | U-Net baseline | DoseGAN geom | U-Net geom

Each row is one metric. Cell colour reflects relative rank within that row
(green = best, red = worst). Cell text shows the actual value ± std.

Usage (from project root):
    python3 -m analysis.plot_heatmap
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection

# ── paths ────────────────────────────────────────────────────────────────────

BASE = Path("outputs/evaluation/baseline_sigmoid")
GEOM = Path("outputs/evaluation")
OUT  = Path("outputs/analysis")
OUT.mkdir(parents=True, exist_ok=True)

FOLDS = range(5)

CONDITION_PATHS = {
    "DoseGAN\nbaseline": [BASE / f"dosegan_ngf32_sigmoid_snellius_fold{f}_val.csv"       for f in FOLDS],
    "U-Net\nbaseline":   [BASE / f"unet3d_ch32_sigmoid_snellius_fold{f}_val.csv"         for f in FOLDS],
    "DoseGAN\ngeom":     [GEOM / f"dosegan_ngf32_sigmoid_geom_snellius_fold{f}_val.csv"  for f in FOLDS],
    "U-Net\ngeom":       [GEOM / f"unet3d_ch32_sigmoid_geom_snellius_fold{f}_val.csv"    for f in FOLDS],
}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_all(paths):
    rows = []
    for p in paths:
        if Path(p).exists():
            with open(p) as f:
                rows.extend(list(csv.DictReader(f)))
    return rows

def sf(x):
    try:
        v = float(x)
        return None if (v != v) else v
    except (TypeError, ValueError):
        return None

def col_stats(rows, col, use_abs=False):
    vals = [sf(r.get(col)) for r in rows]
    vals = [v for v in vals if v is not None]
    if use_abs:
        vals = [abs(v) for v in vals]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))

# ── metric definitions ────────────────────────────────────────────────────────
# (col_key, display_label, lower_is_better, use_abs_for_diff)

METRIC_GROUPS = [
    ("Global error", [
        ("body_MAE_Gy",          "Body MAE (Gy)",            True,  False),
        ("body_RMSE_Gy",         "Body RMSE (Gy)",           True,  False),
    ]),
    ("Structure error", [
        ("ptv_MAE_Gy",           "PTV MAE (Gy)",             True,  False),
        ("rectum_MAE_Gy",        "Rectum MAE (Gy)",          True,  False),
        ("bladder_MAE_Gy",       "Bladder MAE (Gy)",         True,  False),
        ("boundary_MAE_ptv_Gy",  "Boundary MAE PTV (Gy)",    True,  False),
    ]),
    ("DVH endpoints |diff|", [
        ("ptv_D95_diff",         "PTV D95 |diff| (Gy)",      True,  True),
        ("ptv_Dmean_diff",       "PTV Dmean |diff| (Gy)",    True,  True),
        ("rectum_Dmean_diff",    "Rectum Dmean |diff| (Gy)", True,  True),
        ("bladder_Dmean_diff",   "Bladder Dmean |diff| (Gy)",True,  True),
    ]),
    ("Isodose conformality", [
        ("Dice_100iso",          "Dice 100iso",              False, False),
        ("Dice_95iso",           "Dice 95iso",               False, False),
        ("Dice_80iso",           "Dice 80iso",               False, False),
        ("Dice_50iso",           "Dice 50iso",               False, False),
        ("HD95_100iso_mm",       "HD95 100iso (mm)",         True,  False),
    ]),
]

# ── load data ────────────────────────────────────────────────────────────────

conditions = list(CONDITION_PATHS.keys())
all_rows   = {name: load_all(paths) for name, paths in CONDITION_PATHS.items()}

# flat list of (key, label, lower_is_better, use_abs)
flat = [(k, lbl, lib, ua) for _, metrics in METRIC_GROUPS for k, lbl, lib, ua in metrics]
# group boundaries: list of row indices where a new group starts (for dividers)
group_starts = []
row_idx = 0
for _, metrics in METRIC_GROUPS:
    group_starts.append(row_idx)
    row_idx += len(metrics)
group_names = [g for g, _ in METRIC_GROUPS]

n_rows = len(flat)
n_cols = len(conditions)

means = np.full((n_rows, n_cols), np.nan)
stds  = np.full((n_rows, n_cols), np.nan)

for ri, (key, lbl, lib, ua) in enumerate(flat):
    for ci, name in enumerate(conditions):
        m, s = col_stats(all_rows[name], key, use_abs=ua)
        means[ri, ci] = m
        stds[ri, ci]  = s

# ── build colour matrix (0=worst, 1=best per row) ────────────────────────────

color_vals = np.full((n_rows, n_cols), 0.5)

for ri, (key, lbl, lib, ua) in enumerate(flat):
    row = means[ri]
    valid = row[~np.isnan(row)]
    if len(valid) < 2:
        continue
    vmin, vmax = valid.min(), valid.max()
    if vmax == vmin:
        color_vals[ri] = 0.5
        continue
    norm = (row - vmin) / (vmax - vmin)
    color_vals[ri] = 1.0 - norm if lib else norm

# ── figure ───────────────────────────────────────────────────────────────────

CELL_W = 2.0   # inches per column
CELL_H = 0.52  # inches per row
LEFT_MARGIN = 2.8  # inches for row labels
RIGHT_MARGIN = 1.8  # inches for group labels + colorbar
TOP_MARGIN   = 0.9
BOT_MARGIN   = 0.4

fig_w = LEFT_MARGIN + n_cols * CELL_W + RIGHT_MARGIN
fig_h = TOP_MARGIN  + n_rows * CELL_H + BOT_MARGIN

fig = plt.figure(figsize=(fig_w, fig_h))

# axes in figure-fraction units
ax_l = LEFT_MARGIN / fig_w
ax_b = BOT_MARGIN  / fig_h
ax_w = (n_cols * CELL_W) / fig_w
ax_h = (n_rows * CELL_H) / fig_h
ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h])

cmap = plt.get_cmap("RdYlGn")

# draw cells
for ri in range(n_rows):
    for ci in range(n_cols):
        score  = color_vals[ri, ci]
        colour = cmap(score) if not np.isnan(means[ri, ci]) else (0.88, 0.88, 0.88, 1)
        rect = plt.Rectangle([ci, n_rows - ri - 1], 1, 1,
                              facecolor=colour, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)

        if not np.isnan(means[ri, ci]):
            m, s = means[ri, ci], stds[ri, ci]
            fmt  = ".1f" if m >= 10 else ".3f"
            txt  = f"{m:{fmt}}\n±{s:{fmt}}"
            bright = 0.299*colour[0] + 0.587*colour[1] + 0.114*colour[2]
            ax.text(ci + 0.5, n_rows - ri - 0.5, txt,
                    ha="center", va="center", fontsize=8,
                    color="black" if bright > 0.45 else "white",
                    linespacing=1.3)
        else:
            ax.text(ci + 0.5, n_rows - ri - 0.5, "—",
                    ha="center", va="center", fontsize=9, color="#999999")

# row labels — use axes transform so position is in axes fraction + data y
for ri, (key, lbl, lib, ua) in enumerate(flat):
    arrow = "↓" if lib else "↑"
    # x in axes fraction (just left of axes), y in data coords
    ax.text(-0.02, n_rows - ri - 0.5, f"{lbl}  {arrow}",
            ha="right", va="center", fontsize=8.5,
            transform=ax.get_yaxis_transform())

# column headers — x in data coords, y in axes fraction
for ci, name in enumerate(conditions):
    ax.text((ci + 0.5) / n_cols, 1.02, name,
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            multialignment="center",
            transform=ax.transAxes)

# group dividers and right-side group labels
y_pos = n_rows
for gi, (gname, metrics) in enumerate(METRIC_GROUPS):
    gsize = len(metrics)
    y_top = y_pos
    y_bot = y_pos - gsize
    if gi > 0:
        ax.axhline(y_top, color="#444444", linewidth=1.2, linestyle="--", alpha=0.5)
    # x in axes fraction (just right), y in data coords
    ax.text(1.02, (y_top + y_bot) / 2, gname,
            ha="left", va="center", fontsize=8.5, color="#333333", style="italic",
            transform=ax.get_yaxis_transform())
    y_pos -= gsize

ax.set_xlim(0, n_cols)
ax.set_ylim(-0.15, n_rows + 0.75)
ax.axis("off")

# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
sm.set_array([])
cb_l = (LEFT_MARGIN + n_cols * CELL_W + RIGHT_MARGIN * 0.82) / fig_w
cb_b = BOT_MARGIN / fig_h + 0.04
cb_w = 0.016
cb_h = 0.20
cbar_ax = fig.add_axes([cb_l, cb_b, cb_w, cb_h])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["worst", "", "best"], fontsize=7.5)
cbar.ax.tick_params(labelsize=7.5)
cbar.ax.set_title("relative\nrank\n(per row)", fontsize=7, pad=5)

fig.suptitle(
    "Validation performance — 4 conditions  (5-fold CV, n ≈ 367 patients per condition)\n"
    "Colour = relative rank within each metric row   ↓ lower is better   ↑ higher is better",
    fontsize=9.5, y=0.995, va="top"
)

out_path = OUT / "heatmap_4conditions.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close(fig)
