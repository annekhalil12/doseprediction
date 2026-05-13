# models/unet3d.py
# 3D Residual U-Net for dose prediction, built on MONAI's UNet implementation.
#
# Architecture
# ------------
# 5-level encoder-decoder with skip connections. Each level uses residual
# units (Conv3d → norm → act → Conv3d → norm + skip → act) which help
# gradient flow in deep 3D networks.
#
# Key differences from DoseGAN generator
# ---------------------------------------
# - No adversarial training — supervised L1 only (simpler, faster to converge)
# - Residual units instead of plain conv blocks
# - No attention gates on skip connections
# This makes it a clean baseline: same task, same data, different inductive bias.

import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet


class UNet3d(nn.Module):
    def __init__(
        self,
        in_channels:   int   = 9,
        out_channels:  int   = 1,
        channels:      tuple = (32, 64, 128, 256, 256),
        strides:       tuple = (2, 2, 2, 2),
        num_res_units: int   = 2,
    ):
        super().__init__()

        self.net = MonaiUNet(
            spatial_dims  = 3,
            in_channels   = in_channels,
            out_channels  = out_channels,
            channels      = channels,
            strides       = strides,
            num_res_units = num_res_units,
            act           = "PRELU",
            norm          = "BATCH",
            dropout       = 0.0,
        )
        # Sigmoid maps raw logits to [0, 1] — dose is normalised to this range.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.net(x))
