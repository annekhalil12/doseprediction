"""
preprocess_all.py
=================
Batch preprocessing runner for all 432 LUND-PROBE patients.

Processes patients in parallel using ProcessPoolExecutor. Already-cached
patients are skipped, so the job can be safely interrupted and resumed.

Run from the project root on Snellius:
    python preprocessing/preprocess_all.py

Output
------
One .pkl file per patient in OUTPUT_DIR.
Failed patients are logged to stdout/stderr only.
"""

import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_preprocessing_shared import DATA_ROOT, OUTPUT_DIR, PreprocessingConfig
from preprocessing.preprocessing import preprocess_patient

N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))


def process_one(args):
    patient_dir, output_dir, cfg = args
    patient_id = patient_dir.name
    cache_path = Path(output_dir) / f"{patient_id}.pkl"

    if cache_path.exists():
        return {"patient_id": patient_id, "status": "skipped", "error": "", "shape": None, "time": 0}

    try:
        t0     = time.time()
        result = preprocess_patient(patient_dir, cfg)
        result.pop("sct_sitk_ref", None)
        result.pop("crop_offsets",  None)

        shape    = result["input"].shape
        tmp_path = cache_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(result, f)
        tmp_path.rename(cache_path)

        return {"patient_id": patient_id, "status": "success", "error": "", "shape": shape, "time": time.time() - t0}

    except Exception as e:
        return {"patient_id": patient_id, "status": "failed", "error": str(e), "shape": None, "time": 0}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUTPUT_DIR / "preprocess_all.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    log = logging.getLogger(__name__)

    cfg          = PreprocessingConfig()
    patient_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])
    n_total      = len(patient_dirs)

    if n_total == 0:
        raise RuntimeError(f"No patient directories found under:\n  {DATA_ROOT}\nCheck DATA_ROOT in config.")

    log.info(f"Found {n_total} patients | workers={N_WORKERS} | output={OUTPUT_DIR.resolve()}")

    args        = [(p, OUTPUT_DIR, cfg) for p in patient_dirs]
    results_log = []
    n_success = n_skipped = n_failed = 0
    t_start   = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_one, a): a[0].name for a in args}
        with tqdm(total=n_total, unit="patient") as bar:
            for future in as_completed(futures):
                r = future.result()
                results_log.append({"patient_id": r["patient_id"], "status": r["status"], "error": r["error"]})

                if r["status"] == "success":
                    n_success += 1
                    log.info(f"{r['patient_id']}: OK  shape={r['shape']}  time={r['time']:.1f}s")
                elif r["status"] == "skipped":
                    n_skipped += 1
                else:
                    n_failed += 1
                    log.error(f"{r['patient_id']}: FAILED — {r['error']}")

                bar.set_postfix(ok=n_success, skip=n_skipped, fail=n_failed)
                bar.update(1)

    total_min = (time.time() - t_start) / 60
    log.info("=" * 60)
    log.info(f"Finished in {total_min:.1f} min  |  OK={n_success}  skipped={n_skipped}  failed={n_failed}")

    if n_failed:
        log.info("Failed patients:")
        for row in results_log:
            if row["status"] == "failed":
                log.info(f"  {row['patient_id']}: {row['error']}")


if __name__ == "__main__":
    main()
