"""
analysis/fig_architecture.py
================================
Figure: Architecture diagrams for U-Net and DoseGAN.

Shows encoder / decoder levels with feature-map channel counts and spatial
dimensions at every level. The bottleneck spatial resolution (4×8×10) is
highlighted to illustrate why fine-grained spatial metrics degrade in
the deep 6-level network.

Output: results/figures/fig_architecture.pdf  (and .png at 300 dpi)
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT = dict(fontfamily="sans-serif")

# ── Colour palette ─────────────────────────────────────────────────────────
C_ENC      = "#4C9BE8"   # encoder blocks
C_DEC      = "#E87C4C"   # decoder blocks
C_BTN      = "#C0392B"   # bottleneck — red to draw the eye
C_SKIP     = "#95A5A6"   # skip connection arrow
C_DISC     = "#8E44AD"   # discriminator (DoseGAN only)
C_ATT      = "#27AE60"   # attention gate
C_IN       = "#F0A500"   # input/output
C_HEAD     = "#2C3E50"

# ── Architecture parameters ─────────────────────────────────────────────────
# Input: (C, 128, 256, 320)
# U-Net channels: (32, 64, 128, 256, 256, 256), strides (2,2,2,2,2)
UNET_CHANNELS = [32, 64, 128, 256, 256, 256]

# Spatial dimensions at each level (D × H × W)
SPATIAL = [
    (128, 256, 320),   # level 0  (encoder)
    ( 64, 128, 160),   # level 1
    ( 32,  64,  80),   # level 2
    ( 16,  32,  40),   # level 3
    (  8,  16,  20),   # level 4
    (  4,   8,  10),   # level 5 — BOTTLENECK
]


def draw_arch(ax, title, show_att=False, show_disc=False,
              in_ch=9, x_center=0.5, label_suffix=""):
    """Draw one architecture diagram centred at x_center in axes coordinates."""

    n_levels = len(UNET_CHANNELS)          # 6
    total_h  = 0.75                        # vertical span for encoder+bottleneck+decoder
    box_h    = total_h / (n_levels + 0.5)  # height per level
    box_w    = 0.28                        # width of each encoder/decoder block
    half_w   = box_w / 2

    # y positions: level 0 at top, level 5 (bottleneck) at bottom
    def ypos(lvl):
        return 0.92 - lvl * box_h

    enc_x  = x_center - half_w - 0.04   # encoder column x-start
    dec_x  = x_center + 0.04            # decoder column x-start

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(x_center, 0.97, title, ha="center", va="top",
            fontsize=11, fontweight="bold", color=C_HEAD, **FONT)

    # ── Input arrow + label ─────────────────────────────────────────────────
    ax.annotate("", xy=(enc_x + half_w, ypos(0) + box_h * 0.9),
                xytext=(enc_x + half_w, ypos(0) + box_h * 0.9 + 0.055),
                arrowprops=dict(arrowstyle="-|>", color=C_IN, lw=1.4, mutation_scale=12))
    ax.text(enc_x + half_w, ypos(0) + box_h * 0.9 + 0.065,
            f"Input\n({in_ch}, 128, 256, 320)",
            ha="center", va="bottom", fontsize=7, color=C_IN,
            fontweight="bold", linespacing=1.3, **FONT)

    # ── Encoder + Decoder blocks ────────────────────────────────────────────
    for lvl, ch in enumerate(UNET_CHANNELS):
        y = ypos(lvl)
        is_btn = (lvl == n_levels - 1)
        color  = C_BTN if is_btn else C_ENC

        # Encoder block
        bw = box_w * (1.1 if is_btn else 1.0)
        bx = (enc_x + dec_x + box_w) / 2 - bw / 2 if is_btn else enc_x
        rect_e = FancyBboxPatch((bx, y), bw if is_btn else box_w, box_h * 0.85,
                                 boxstyle="round,pad=0.008",
                                 facecolor=color, edgecolor="white",
                                 linewidth=1.0, alpha=0.88, zorder=3)
        ax.add_patch(rect_e)

        d, h, w = SPATIAL[lvl]
        sp_str = f"{d}×{h}×{w}"
        ch_str = f"{ch} ch"
        label  = "BOTTLENECK" if is_btn else ("Encoder" if lvl == 0 else "")

        ax.text(bx + (bw if is_btn else box_w) / 2,
                y + box_h * 0.85 / 2 + (0.008 if is_btn else 0.004),
                ch_str, ha="center", va="center",
                fontsize=8.5 if is_btn else 8, fontweight="bold",
                color="white", zorder=4, **FONT)
        ax.text(bx + (bw if is_btn else box_w) / 2,
                y + box_h * 0.85 / 2 - 0.016,
                sp_str, ha="center", va="center",
                fontsize=7 if not is_btn else 8, color="white",
                alpha=0.92, zorder=4,
                fontweight="bold" if is_btn else "normal", **FONT)
        if is_btn:
            ax.text(bx + bw / 2, y + box_h * 0.85 / 2 - 0.033,
                    "← spatial bottleneck", ha="center", va="center",
                    fontsize=6.5, color="#ffcccc", zorder=4, style="italic", **FONT)

        if not is_btn:
            # Decoder block (mirror)
            rect_d = FancyBboxPatch((dec_x, y), box_w, box_h * 0.85,
                                     boxstyle="round,pad=0.008",
                                     facecolor=C_DEC, edgecolor="white",
                                     linewidth=1.0, alpha=0.88, zorder=3)
            ax.add_patch(rect_d)
            ax.text(dec_x + box_w / 2, y + box_h * 0.85 / 2 + 0.004,
                    ch_str, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white", zorder=4, **FONT)
            ax.text(dec_x + box_w / 2, y + box_h * 0.85 / 2 - 0.016,
                    sp_str, ha="center", va="center",
                    fontsize=7, color="white", alpha=0.92, zorder=4, **FONT)

            # Skip connection (horizontal dashed arrow)
            skip_y = y + box_h * 0.85 / 2
            ax.annotate("", xy=(dec_x, skip_y),
                        xytext=(enc_x + box_w, skip_y),
                        arrowprops=dict(arrowstyle="-|>", color=C_SKIP,
                                        lw=1.1, mutation_scale=10,
                                        linestyle="dashed"))

            # Attention gate indicator (DoseGAN only, not level 0)
            if show_att and lvl > 0:
                ax_att_x = (enc_x + box_w + dec_x) / 2
                ax_att_y = skip_y - 0.012
                att_circ = plt.Circle((ax_att_x, ax_att_y), 0.018,
                                       color=C_ATT, zorder=5, alpha=0.9)
                ax.add_patch(att_circ)
                ax.text(ax_att_x, ax_att_y, "A", ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color="white", zorder=6, **FONT)

            # Downward arrow (encoder path)
            if lvl < n_levels - 2:
                ax.annotate("", xy=(enc_x + half_w, ypos(lvl + 1) + box_h * 0.88),
                            xytext=(enc_x + half_w, y),
                            arrowprops=dict(arrowstyle="-|>", color=C_ENC,
                                            lw=1.2, mutation_scale=11))
            # Upward arrow (decoder path)
            if lvl > 0:
                ax.annotate("", xy=(dec_x + half_w, y + box_h * 0.87),
                            xytext=(dec_x + half_w, ypos(lvl - 1)),
                            arrowprops=dict(arrowstyle="-|>", color=C_DEC,
                                            lw=1.2, mutation_scale=11))

        else:
            # Bottleneck: connect encoder down and decoder up
            ax.annotate("", xy=(bx + bw / 2, ypos(lvl) + box_h * 0.87),
                        xytext=(bx + bw / 2, ypos(lvl - 1)),
                        arrowprops=dict(arrowstyle="-|>", color=C_BTN,
                                        lw=1.4, mutation_scale=12))
            # Encoder into bottleneck
            ax.annotate("", xy=(bx + bw / 4, ypos(lvl) + box_h * 0.87),
                        xytext=(enc_x + half_w, ypos(lvl - 1)),
                        arrowprops=dict(arrowstyle="-|>", color=C_ENC,
                                        lw=1.2, mutation_scale=11))
            # Bottleneck out to decoder
            ax.annotate("", xy=(dec_x + half_w, ypos(lvl - 1)),
                        xytext=(bx + bw * 3 / 4, ypos(lvl) + box_h * 0.87),
                        arrowprops=dict(arrowstyle="-|>", color=C_DEC,
                                        lw=1.2, mutation_scale=11))

    # ── Residual unit label (U-Net) ─────────────────────────────────────────
    if not show_att:
        ax.text(enc_x - 0.02, ypos(2) + box_h * 0.4,
                "2 × residual\nunits per level", ha="right", va="center",
                fontsize=7, color=C_ENC, style="italic", **FONT)

    # ── Output arrow ───────────────────────────────────────────────────────
    out_y = ypos(0)
    ax.annotate("", xy=(dec_x + half_w, out_y + box_h * 0.85 + 0.055),
                xytext=(dec_x + half_w, out_y + box_h * 0.85),
                arrowprops=dict(arrowstyle="-|>", color=C_IN, lw=1.4, mutation_scale=12))
    ax.text(dec_x + half_w, out_y + box_h * 0.85 + 0.065,
            "Output\n(1, 128, 256, 320)",
            ha="center", va="bottom", fontsize=7, color=C_IN,
            fontweight="bold", linespacing=1.3, **FONT)

    # ── Discriminator block (DoseGAN only) ─────────────────────────────────
    if show_disc:
        disc_x = x_center + 0.04 + box_w + 0.07
        disc_y = ypos(2) - 0.01
        disc_h = box_h * 0.85 * 3
        disc_w = 0.13
        rect_d2 = FancyBboxPatch((disc_x, disc_y), disc_w, disc_h,
                                  boxstyle="round,pad=0.01",
                                  facecolor=C_DISC, edgecolor="white",
                                  linewidth=1.0, alpha=0.85, zorder=3)
        ax.add_patch(rect_d2)
        ax.text(disc_x + disc_w / 2, disc_y + disc_h / 2,
                "Dis-\ncrim-\ninator\n\n3D\nPatch-\nGAN",
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="white", linespacing=1.35, zorder=4, **FONT)
        # Arrow from output to discriminator
        ax.annotate("", xy=(disc_x, disc_y + disc_h * 0.75),
                    xytext=(dec_x + box_w, out_y + box_h * 0.85 + 0.01),
                    arrowprops=dict(arrowstyle="-|>", color=C_DISC,
                                    lw=1.2, mutation_scale=11,
                                    connectionstyle="arc3,rad=0.2"))
        ax.text(disc_x + disc_w / 2, disc_y - 0.04,
                "Real / Fake\n(adversarial)", ha="center", va="top",
                fontsize=6.5, color=C_DISC, **FONT)

    # ── Level labels on left margin ─────────────────────────────────────────
    for lvl in range(n_levels):
        if lvl < n_levels - 1:
            ax.text(enc_x - 0.035, ypos(lvl) + box_h * 0.4,
                    f"L{lvl}", ha="right", va="center",
                    fontsize=7.5, color="#888", **FONT)


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13.5, 8))

for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# U-Net (left)
draw_arch(axes[0], title="U-Net (3D Residual U-Net)",
          show_att=False, show_disc=False, in_ch=9, x_center=0.48)

# DoseGAN generator (right)
draw_arch(axes[1], title="DoseGAN (Generator + Discriminator)",
          show_att=True, show_disc=True, in_ch=9, x_center=0.40)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_ENC,  label="Encoder (Conv3d ↓2)"),
    mpatches.Patch(color=C_DEC,  label="Decoder (ConvTranspose3d ↑2)"),
    mpatches.Patch(color=C_BTN,  label="Bottleneck (4×8×10 spatial)"),
    mpatches.Patch(color=C_ATT,  label="Attention gate (DoseGAN)"),
    mpatches.Patch(color=C_DISC, label="Discriminator (DoseGAN)"),
    mpatches.Patch(color=C_SKIP, label="Skip connection"),
]
fig.legend(handles=legend_items, loc="lower center", ncol=3,
           fontsize=8.5, framealpha=0.7, handlelength=1.4,
           bbox_to_anchor=(0.5, 0.0))

plt.suptitle("Model Architectures — U-Net and DoseGAN\n"
             "Input: (C, 128, 256, 320)  |  Both: 5 downsampling levels  |  "
             "Bottleneck: (256 ch, 4×8×10)",
             fontsize=10, color=C_HEAD, y=1.00, **FONT)

plt.tight_layout(rect=[0, 0.06, 1, 0.98])
plt.savefig(OUT_DIR / "fig_architecture.pdf", bbox_inches="tight", dpi=150)
plt.savefig(OUT_DIR / "fig_architecture.png", bbox_inches="tight", dpi=300)
print("Saved fig_architecture.pdf / .png")
