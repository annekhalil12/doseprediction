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

    # During cross-validation: specify which fold is the current val fold.
    # The Dataset figures out train/val patients automatically from the CSV.
    train_ds = LUNDPROBEDataset(
        split_csv  = Path("outputs/split.csv"),
        pickle_dir = Path("outputs/"),
        split      = "train",
        fold       = 2,          # patients with fold != 2 are training
    )

    val_ds = LUNDPROBEDataset(
        split_csv  = Path("outputs/split.csv"),
        pickle_dir = Path("outputs/"),
        split      = "val",
        fold       = 2,          # patients with fold == 2 are validation
    )

    # For final evaluation on the held-out test set, fold is ignored.
    test_ds = LUNDPROBEDataset(
        split_csv  = Path("outputs/split.csv"),
        pickle_dir = Path("outputs/"),
        split      = "test",
        fold       = None,
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
from typing import Dict, List, Literal, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensity,  # non-dict version — applied to sCT channel only
    RandShiftIntensity,  # non-dict version — applied to sCT channel only
)

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
    fold       : which fold is currently the validation fold (0–4).
                 Required for split="train" and split="val".
                 Ignored for split="test" (pass None).
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
        fold:       Optional[int] = None,
        channels:   List[int] | None = None,
    ) -> None:
        super().__init__()

        self.pickle_dir = Path(pickle_dir)
        self.channels   = channels

        # Validate fold argument — it is required for train/val splits
        # because the CSV no longer has a fixed "train"/"val" label.
        # Instead, which patients are training vs. validation depends on
        # which fold is currently held out as the validation set.
        if split in ("train", "val") and fold is None:
            raise ValueError(
                f"split='{split}' requires a fold argument (0–4). "
                "Example: LUNDPROBEDataset(..., split='train', fold=2)"
            )

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
                is_test = row["is_test"].strip().lower() == "true"

                if split == "test":
                    # Test patients are permanently held out — fold is irrelevant.
                    if is_test:
                        self.patient_ids.append(row["patient_id"])

                elif split == "val":
                    # Validation patients for this fold: not in test set,
                    # and their fold number matches the current fold.
                    if not is_test and int(row["fold"]) == fold:
                        self.patient_ids.append(row["patient_id"])

                elif split == "train":
                    # Training patients for this fold: not in test set,
                    # and their fold number does NOT match the current fold.
                    if not is_test and int(row["fold"]) != fold:
                        self.patient_ids.append(row["patient_id"])

        if len(self.patient_ids) == 0:
            raise ValueError(
                f"No patients found for split='{split}', fold={fold} in {split_path}. "
                "Check that create_split.py ran successfully."
            )

        log.info(
            f"LUNDPROBEDataset | split='{split}' | fold={fold} | "
            f"{len(self.patient_ids)} patients | "
            f"pickle_dir={self.pickle_dir}"
        )

        # ── Augmentation transforms ────────────────────────────────────────
        # Transforms are only active during training — val and test always
        # see the original unmodified volumes for reproducible evaluation.
        if split == "train":
            # Geometric transforms use the dict API (suffix 'd') so that
            # "input" and "dose" are always flipped/rotated identically —
            # the dose map must stay aligned with the anatomy at all times.
            self.geometric_transform = Compose([
                RandFlipd(
                    keys=["input", "dose"],
                    prob=0.5,
                    spatial_axis=0,   # flip along depth axis
                ),
                RandFlipd(
                    keys=["input", "dose"],
                    prob=0.5,
                    spatial_axis=2,   # flip left-right
                ),
                RandRotate90d(
                    keys=["input", "dose"],
                    prob=0.5,
                    max_k=1,          # rotate 0 or 90 degrees only
                    spatial_axes=(1, 2),  # rotate in the axial plane
                ),
            ])

            # Intensity transforms use the non-dict API because they are
            # applied to channel 8 (sCT) only — binary masks in channels
            # 0–7 must stay exactly 0 or 1, never scaled or shifted.
            self.intensity_transform = Compose([
                RandScaleIntensity(factors=0.1, prob=0.5),
                RandShiftIntensity(offsets=0.1, prob=0.5),
            ])
        else:
            self.geometric_transform  = None
            self.intensity_transform  = None

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

        # ── Apply augmentation (training only) ────────────────────────────
        if self.geometric_transform is not None:
            # MONAI dict transforms expect a dict — we pass input and dose
            # together so they are transformed with the same random parameters.
            augmented = self.geometric_transform({
                "input": input_tensor,
                "dose":  dose_tensor,
            })
            input_tensor = augmented["input"]
            dose_tensor  = augmented["dose"]

        if self.intensity_transform is not None:
            # Extract sCT channel, scale/shift it, put it back.
            # Indexing [8:9] keeps the channel dimension so MONAI
            # receives a (1, D, H, W) tensor as expected.
            sct_channel = input_tensor[8:9]
            sct_channel = self.intensity_transform(sct_channel)
            input_tensor[8] = sct_channel[0]
            
        return {
            "input":      input_tensor,
            "dose":       dose_tensor,
            "ptv_mask":   ptv_tensor,
            "patient_id": patient_id,
        }