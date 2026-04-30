"""
dataset.py
==========
PyTorch Dataset for loading preprocessed LUND-PROBE patient pickles.

How this fits into training
----------------------------
PyTorch training works in a loop: on each iteration, the model receives
a batch of (input, target) pairs, computes a prediction, measures the
error against the target, and updates its weights. The Dataset class is
responsible for answering one question: "Given a patient index i, what
is that patient's input tensor and ground-truth dose?"

The DataLoader (used in train.py, not here) sits on top of the Dataset
and handles batching, shuffling, and parallel loading automatically.

Channel layout of the 9-channel input tensor
---------------------------------------------
Index  Content
  0    PTVT_427 mask         (binary)
  1    Rectum mask            (binary)
  2    Bladder mask           (binary)
  3    FemoralHead_L mask     (binary)
  4    FemoralHead_R mask     (binary)
  5    Genitalia mask         (binary, zeros if missing)
  6    PenileBulb mask        (binary, zeros if missing)
  7    BODY mask              (binary)
  8    sCT intensity          (z-scored, clipped to [-990, 2000] HU)

Channels 8–14 (geometric features) will be appended once
V5geometric_channels.py is available from the collaborator.

Usage
-----
    from dataset import LUNDPROBEDataset

    train_ds = LUNDPROBEDataset(
        split_csv  = Path("outputs/split.csv"),
        pickle_dir = Path("outputs/"),   # or network path on AI PC
        split      = "train",
    )

    sample = train_ds[0]
    print(sample["input"].shape)   # torch.Size([9, 128, 256, 320])
    print(sample["dose"].shape)    # torch.Size([1, 128, 256, 320])
"""

from __future__ import annotations

import csv
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class LUNDPROBEDataset(Dataset):
    """
    PyTorch Dataset for the preprocessed LUND-PROBE cohort.

    Each call to __getitem__(i) loads one patient's pickle from disk,
    converts the NumPy arrays to PyTorch tensors, and returns them as
    a dictionary. Dictionary outputs (rather than bare tuples) make it
    easy to add extra fields later (e.g. patient_id, masks for evaluation)
    without breaking existing code.

    Parameters
    ----------
    split_csv  : path to outputs/split.csv (created by create_split.py)
    pickle_dir : directory containing the .pkl files
    split      : which subset to load — "train", "val", or "test"
    channels   : which input channel indices to return; None = all available.
                 This is how each model gets its channel subset from the
                 shared 9-channel cache without reprocessing.
                 Example: channels=[0,1,2,7,8] gives PTV+Rectum+Bladder+BODY+sCT.
    """

    def __init__(
        self,
        split_csv:  Path,
        pickle_dir: Path,
        split:      Literal["train", "val", "test"],
        channels:   List[int] | None = None,
    ) -> None:
        super().__init__()

        self.pickle_dir = Path(pickle_dir)
        self.channels   = channels

        # ── Load the split CSV and filter to the requested subset ──────────
        # The split CSV was created once by create_split.py and committed
        # to the repository. Loading it here (rather than re-computing the
        # split) guarantees that every training run uses exactly the same
        # patient assignments, even if the script is run on a different machine.
        self.patient_ids: List[str] = []

        split_path = Path(split_csv)
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {split_path}\n"
                "Run preprocessing/create_split.py first."
            )

        with open(split_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.patient_ids.append(row["patient_id"])

        if len(self.patient_ids) == 0:
            raise ValueError(
                f"No patients found for split='{split}' in {split_path}. "
                "Check that create_split.py ran successfully."
            )

        log.info(
            f"LUNDPROBEDataset | split='{split}' | "
            f"{len(self.patient_ids)} patients | "
            f"pickle_dir={self.pickle_dir}"
        )

    # ── PyTorch Dataset interface ──────────────────────────────────────────

    def __len__(self) -> int:
        """Total number of patients in this split."""
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """
        Load one patient and return a dictionary of tensors.

        Called automatically by PyTorch's DataLoader on each training step.
        Each call opens one pickle file from disk — this is intentional.
        Loading all pickles into memory upfront would require ~130 GB of RAM.

        Returns
        -------
        dict with keys:
          "input"      : (C, 128, 256, 320) float32 tensor — model input
          "dose"       : (1, 128, 256, 320) float32 tensor — ground-truth dose
          "ptv_mask"   : (1, 128, 256, 320) bool tensor   — for evaluation
          "patient_id" : str                               — for logging/debugging
        """
        patient_id  = self.patient_ids[index]
        pickle_path = self.pickle_dir / f"{patient_id}.pkl"

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Pickle not found for patient '{patient_id}': {pickle_path}\n"
                "Check that pickle_dir in your config points to the correct folder."
            )

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        # ── Convert NumPy arrays to PyTorch tensors ───────────────────────
        # PyTorch models expect float32 tensors. The pickle already stores
        # float32 NumPy arrays, so this conversion is zero-copy efficient.
        input_tensor = torch.from_numpy(data["input"])   # (9, D, H, W)
        dose_tensor  = torch.from_numpy(data["dose"])    # (D, H, W)
        ptv_tensor   = torch.from_numpy(
            data["ptv_mask"].astype("float32")
        )                                                # (D, H, W)

        # ── Select channel subset if requested ────────────────────────────
        # The shared cache has 9 channels (0–8). If channels=[0,1,2,7,8],
        # only those five channels are returned. This lets different models
        # use different subsets from the same pickle without reprocessing.
        # None means "return all channels".
        if self.channels is not None:
            input_tensor = input_tensor[self.channels]   # (len(channels), D, H, W)

        # ── Add channel dimension to dose and masks ───────────────────────
        # PyTorch 3D convolutions expect (batch, channels, D, H, W).
        # The DataLoader adds the batch dimension automatically; here we add
        # the channel dimension (=1) so the shape is consistent with the input.
        dose_tensor = dose_tensor.unsqueeze(0)    # (1, D, H, W)
        ptv_tensor  = ptv_tensor.unsqueeze(0)     # (1, D, H, W)

        return {
            "input":      input_tensor,
            "dose":       dose_tensor,
            "ptv_mask":   ptv_tensor,
            "patient_id": patient_id,
        }