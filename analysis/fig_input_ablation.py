"""
analysis/fig_input_ablation.py
================================
Figure: Input Tensor construction and 2×2 ablation design.

Panels
------
Left  : Channel stack → concatenation → 9-ch or 14-ch tensor arrow into model box.
Right : 2×2 grid (rows = architecture, cols = +geom) showing the four conditions.

Output: results/figures/fig_input_ablation.pdf  (and .png at 300 dpi)
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_MASK  = "#4C9BE8"   # blue  — binary structure masks
C_SCT   = "#F0A500"   # amber — sCT intensity channel
C_GEOM  = "#6DBF6D"   # green — geometric channels
C_TENS  = "#A0A0A0"   # grey  — tensor box
C_MODEL = "#D35400"   # burnt-orange — model box
C_COND  = "#ECF0F1"   # light grey — cell background
C_HEAD  = "#2C3E50"   # dark — header background

FONT = dict(fontfamily="sans-serif")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: draw a labelled channel-group rectangle
# ─────────────────────────────────────────────────────────────────────────────
def channel_box(ax, x, y, w, h, color, label, sublabel="", alpha=0.85):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.015",
                          facecolor=color, edgecolor="white",
                          linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2 + (0.018 if sublabel else 0),
            label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=4, **FONT)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.022, sublabel,
                ha="center", va="center",
                fontsize=7.5, color="white", alpha=0.9, zorder=4, **FONT)


# ─────────────────────────────────────────────────────────────────────────────
# Figure layout
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5.8))

# Two sub-axes: left (tensor construction) and right (2×2 ablation grid)
ax_l = fig.add_axes([0.02, 0.05, 0.44, 0.90])   # left panel
ax_r = fig.add_axes([0.52, 0.05, 0.46, 0.90])   # right panel
for ax in (ax_l, ax_r):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — channel stack + concatenation
# ─────────────────────────────────────────────────────────────────────────────
ax_l.text(0.5, 0.97, "Input Tensor Construction", ha="center", va="top",
          fontsize=11, fontweight="bold", color=C_HEAD, **FONT)

# -- Baseline (9-ch) column --
bx, tw = 0.08, 0.32   # x start, width

# Binary masks block (8 ch)
channel_box(ax_l, bx, 0.60, tw, 0.24, C_MASK,
            "8 × Binary Masks", "PTV · Rectum · Bladder\nFem. Heads · Genit. · PenileBulb · Body")

# sCT block (1 ch)
channel_box(ax_l, bx, 0.50, tw, 0.09, C_SCT,
            "1 × sCT Intensity", "z-scored HU")

# brace / bracket indicating 9-ch
for yy in (0.50, 0.84):
    ax_l.plot([bx + tw + 0.01, bx + tw + 0.035, bx + tw + 0.035],
              [yy, yy, 0.67], color="#555", lw=1.2, clip_on=False)
ax_l.plot([bx + tw + 0.035], [0.67], color="#555", lw=1.2)   # placeholder

# Draw a right-bracket for 9ch
xb = bx + tw + 0.015
ax_l.annotate("", xy=(xb + 0.04, 0.67), xytext=(xb, 0.84),
              arrowprops=dict(arrowstyle="-", color="#555", lw=1.1))
ax_l.annotate("", xy=(xb + 0.04, 0.67), xytext=(xb, 0.50),
              arrowprops=dict(arrowstyle="-", color="#555", lw=1.1))
ax_l.plot([xb, xb], [0.50, 0.84], color="#555", lw=1.1)

ax_l.text(xb + 0.055, 0.67, "9 ch\n(baseline)", ha="left", va="center",
          fontsize=8.5, color="#333", **FONT)

# -- Geom addition (5 ch) --
channel_box(ax_l, bx, 0.37, tw, 0.11, C_GEOM,
            "5 × Geometric Channels",
            "dist_PTV · dist_Body · dir_z · dir_y · dir_x")

# bracket for 14ch (all three blocks)
xb2 = bx + tw + 0.015
ax_l.plot([xb2, xb2], [0.37, 0.84], color=C_GEOM, lw=1.4, linestyle="--", alpha=0.7)
ax_l.annotate("", xy=(xb2 + 0.04, 0.60), xytext=(xb2, 0.84),
              arrowprops=dict(arrowstyle="-", color=C_GEOM, lw=1.1, linestyle="dashed"))
ax_l.annotate("", xy=(xb2 + 0.04, 0.60), xytext=(xb2, 0.37),
              arrowprops=dict(arrowstyle="-", color=C_GEOM, lw=1.1, linestyle="dashed"))

ax_l.text(xb2 + 0.055, 0.595, "14 ch\n(+geom)", ha="left", va="center",
          fontsize=8.5, color=C_GEOM, fontweight="bold", **FONT)

# Dashed separator
ax_l.plot([bx, bx + tw], [0.48, 0.48], color="#ccc", lw=0.8, linestyle=":")

# -- Arrow into tensor box --
ax_l.annotate("", xy=(0.80, 0.67), xytext=(0.68, 0.67),
              arrowprops=dict(arrowstyle="-|>", color="#555",
                              lw=1.5, mutation_scale=14))

# Tensor shape box
rect_t = FancyBboxPatch((0.80, 0.59), 0.17, 0.16,
                         boxstyle="round,pad=0.01",
                         facecolor=C_TENS, edgecolor="#aaa",
                         linewidth=1.2, alpha=0.35, zorder=3)
ax_l.add_patch(rect_t)
ax_l.text(0.885, 0.72, "(C, 128, 256, 320)", ha="center", va="center",
          fontsize=7.5, color="#222", style="italic", zorder=4, **FONT)
ax_l.text(0.885, 0.685, "C = 9  or  14", ha="center", va="center",
          fontsize=8, color="#222", fontweight="bold", zorder=4, **FONT)
ax_l.text(0.885, 0.650, "Depth × Height × Width", ha="center", va="center",
          fontsize=7, color="#555", zorder=4, **FONT)

# -- Arrow down to model --
ax_l.annotate("", xy=(0.885, 0.50), xytext=(0.885, 0.575),
              arrowprops=dict(arrowstyle="-|>", color="#555",
                              lw=1.5, mutation_scale=14))

# Model box
rect_m = FancyBboxPatch((0.78, 0.38), 0.21, 0.10,
                         boxstyle="round,pad=0.015",
                         facecolor=C_MODEL, edgecolor="none",
                         linewidth=0, alpha=0.85, zorder=3)
ax_l.add_patch(rect_m)
ax_l.text(0.885, 0.43, "U-Net / DoseGAN", ha="center", va="center",
          fontsize=8.5, fontweight="bold", color="white", zorder=4, **FONT)

# -- Output arrow + label --
ax_l.annotate("", xy=(0.885, 0.28), xytext=(0.885, 0.38),
              arrowprops=dict(arrowstyle="-|>", color="#555",
                              lw=1.5, mutation_scale=14))
ax_l.text(0.885, 0.24, "Predicted Dose\n(1, 128, 256, 320)", ha="center", va="center",
          fontsize=8, color="#333", **FONT)

# Left panel legend
leg_items = [
    mpatches.Patch(color=C_MASK,  label="Binary structure masks (8)"),
    mpatches.Patch(color=C_SCT,   label="sCT intensity (1)"),
    mpatches.Patch(color=C_GEOM,  label="Geometric channels (5, optional)"),
]
ax_l.legend(handles=leg_items, loc="lower left", fontsize=7.5,
            framealpha=0.6, handlelength=1.2, borderpad=0.6)

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — 2×2 ablation grid
# ─────────────────────────────────────────────────────────────────────────────
ax_r.text(0.5, 0.97, "2×2 Ablation Design", ha="center", va="top",
          fontsize=11, fontweight="bold", color=C_HEAD, **FONT)

# Grid geometry
hdr_h = 0.10     # header row height
row_h = 0.33     # data row height
col_w = 0.42     # data column width
lbl_w = 0.14     # row-label width
pad   = 0.015

x0 = 0.02
y_top = 0.87

# -- Column headers --
for ci, (col_label, col_sub, cx) in enumerate([
    ("Baseline", "9 channels\n(8 masks + sCT)",        x0 + lbl_w + pad),
    ("+Geometric", "14 channels\n(+5 geom channels)",  x0 + lbl_w + pad + col_w + pad),
]):
    rect = FancyBboxPatch((cx, y_top - hdr_h + 0.01), col_w, hdr_h - 0.02,
                           boxstyle="round,pad=0.01",
                           facecolor=C_HEAD, edgecolor="none", alpha=0.85)
    ax_r.add_patch(rect)
    ax_r.text(cx + col_w / 2, y_top - hdr_h / 2 + 0.005, col_label,
              ha="center", va="center", fontsize=9.5, fontweight="bold",
              color="white", **FONT)
    ax_r.text(cx + col_w / 2, y_top - hdr_h / 2 - 0.026, col_sub,
              ha="center", va="center", fontsize=7.5, color="#ccc", **FONT)

# -- Rows --
rows = [
    ("U-Net",    C_MASK,
     "UNet-Base\nch: 32→64→128→256→256→256\nin_ch = 9",
     "UNet-Geom\nch: 32→64→128→256→256→256\nin_ch = 14"),
    ("DoseGAN",  C_MODEL,
     "DoseGAN-Base\nngf=32, 5 skip levels\nAttention gates · in_ch = 9",
     "DoseGAN-Geom\nngf=32, 5 skip levels\nAttention gates · in_ch = 14"),
]

for ri, (row_label, row_color, cell_bl, cell_br) in enumerate(rows):
    ry = y_top - hdr_h - ri * (row_h + pad) - row_h

    # Row label
    rect_lbl = FancyBboxPatch((x0, ry + 0.005), lbl_w - 0.01, row_h - 0.01,
                               boxstyle="round,pad=0.01",
                               facecolor=row_color, edgecolor="none", alpha=0.80)
    ax_r.add_patch(rect_lbl)
    ax_r.text(x0 + (lbl_w - 0.01) / 2, ry + row_h / 2, row_label,
              ha="center", va="center", fontsize=9.5, fontweight="bold",
              color="white", rotation=90, **FONT)

    # Cell columns
    for ci, cell_text in enumerate([cell_bl, cell_br]):
        cx = x0 + lbl_w + pad + ci * (col_w + pad)
        rect_c = FancyBboxPatch((cx + 0.005, ry + 0.008), col_w - 0.01, row_h - 0.016,
                                 boxstyle="round,pad=0.01",
                                 facecolor=C_COND, edgecolor=row_color,
                                 linewidth=1.5, alpha=0.90)
        ax_r.add_patch(rect_c)
        # Cell title (first line bold)
        lines = cell_text.split("\n")
        ax_r.text(cx + col_w / 2, ry + row_h / 2 + 0.04, lines[0],
                  ha="center", va="center", fontsize=9, fontweight="bold",
                  color=C_HEAD, **FONT)
        ax_r.text(cx + col_w / 2, ry + row_h / 2 - 0.012,
                  "\n".join(lines[1:]),
                  ha="center", va="center", fontsize=7.5,
                  color="#444", linespacing=1.4, **FONT)

# Geom column shading hint
ax_r.text(x0 + lbl_w + pad + col_w + pad + col_w / 2, y_top - hdr_h - 2 * (row_h + pad) + 0.08,
          "← adds geometric\nprior channels", ha="center", va="center",
          fontsize=7.5, color=C_GEOM, style="italic", **FONT)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT_DIR / "fig_input_ablation.pdf", bbox_inches="tight", dpi=150)
plt.savefig(OUT_DIR / "fig_input_ablation.png", bbox_inches="tight", dpi=300)
print("Saved fig_input_ablation.pdf / .png")
