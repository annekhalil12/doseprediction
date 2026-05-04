"""
test_dataset.py
===============
Quick sanity check for the LUNDPROBEDataset class.

Run from the repo root:
    python training/test_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import LUNDPROBEDataset

split_csv  = Path("outputs/split.csv")
pickle_dir = Path("outputs/")

# Test all three splits
for split_name in ["train", "val", "test"]:
    ds     = LUNDPROBEDataset(split_csv=split_csv, pickle_dir=pickle_dir, split=split_name)
    sample = ds[0]

    print(f"\n--- {split_name.upper()} ---")
    print(f"  Patients     : {len(ds)}")
    print(f"  Input shape  : {sample['input'].shape}")
    print(f"  Dose shape   : {sample['dose'].shape}")
    print(f"  PTV shape    : {sample['ptv_mask'].shape}")
    print(f"  Patient ID   : {sample['patient_id']}")
    print(f"  Input dtype  : {sample['input'].dtype}")
    print(f"  Dose range   : [{sample['dose'].min():.3f}, {sample['dose'].max():.3f}]")

print("\nAll splits loaded successfully.")

#hello