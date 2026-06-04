"""
add_geom_channels.py
====================
Augments existing patient pickles with 5 pre-computed geometric channels.

Each pickle gains a new key:
  'geom_channels' : (5, 128, 256, 320) float32

  Channel order:
    [0] dist_to_ptv_surface
    [1] dist_to_body_surface
    [2] dir_z_shifted
    [3] dir_y_shifted
    [4] dir_x_shifted

The original 'input' tensor (9 channels) is unchanged — no re-preprocessing
from NIfTI is needed. Body mask is extracted from input[7] (BODY channel).
PTV mask is read from the existing 'ptv_mask' key.

Already-augmented pickles are skipped (idempotent).
Pass --force to recompute and overwrite existing geom_channels (required after any
geometric-channel formula change).

Usage
-----
Run from the project root with PYTHONPATH set:
    export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction
    python3 preprocessing/add_geom_channels.py [--force]
"""

import logging
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from preprocessing.geometric_channels import compute_geom_channels

PICKLE_DIR = Path("data/pickles")
N_WORKERS  = 16

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _process_patient(pkl_path: Path, force: bool = False) -> str:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if "geom_channels" in data and not force:
        return f"SKIP  {pkl_path.name}"

    ptv_mask  = data["ptv_mask"].astype(np.float32)   # (D, H, W)
    body_mask = data["input"][7]                        # BODY is channel 7

    geom = compute_geom_channels(ptv_mask, body_mask)  # (5, D, H, W)
    data["geom_channels"]              = geom
    data["geometric_channels_pending"] = False

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    return f"OK    {pkl_path.name}  shape={geom.shape}  range=[{geom.min():.3f},{geom.max():.3f}]"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Recompute and overwrite existing geom_channels (required after formula changes).")
    cli = parser.parse_args()

    pkl_paths = sorted(PICKLE_DIR.glob("*.pkl"))
    if not pkl_paths:
        raise FileNotFoundError(f"No pickles found in {PICKLE_DIR.resolve()}")

    if cli.force:
        log.warning("--force: existing geom_channels will be recomputed and overwritten.")

    log.info(f"Found {len(pkl_paths)} pickles — augmenting with geom channels using {N_WORKERS} workers")

    failed = []
    done   = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_process_patient, p, cli.force): p for p in pkl_paths}
        for future in as_completed(futures):
            pkl_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = f"FAIL  {pkl_path.name}  — {exc}"
                failed.append((pkl_path.name, str(exc)))
            done += 1
            if done % 50 == 0 or done == len(pkl_paths):
                log.info(f"[{done:3d}/{len(pkl_paths)}] {result}")

    log.info(f"Done — {len(pkl_paths) - len(failed)} OK, {len(failed)} failed")
    if failed:
        log.error("Failed patients:")
        for name, err in failed:
            log.error(f"  {name}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
