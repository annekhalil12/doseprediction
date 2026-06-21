"""
check_physical_grids.py
=======================
Audit whether the independently resampled sCT and dose volumes occupy
the same physical SimpleITK grid for every LUND-PROBE patient.

This script does not modify files or cached pickles.

Run from the project root:
    python -m preprocessing.check_physical_grids
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_preprocessing_shared import DATA_ROOT, PreprocessingConfig
from preprocessing.preprocessing import load_nifti, resample_image


def compare_grids(
    image: sitk.Image,
    reference: sitk.Image,
    atol: float = 1e-5,
) -> list[str]:
    """Return descriptions of all physical-grid mismatches."""
    problems = []

    if image.GetSize() != reference.GetSize():
        problems.append(
            f"size: dose={image.GetSize()} sCT={reference.GetSize()}"
        )

    if not np.allclose(
        image.GetSpacing(),
        reference.GetSpacing(),
        atol=atol,
        rtol=0,
    ):
        problems.append(
            f"spacing: dose={image.GetSpacing()} sCT={reference.GetSpacing()}"
        )

    if not np.allclose(
        image.GetOrigin(),
        reference.GetOrigin(),
        atol=atol,
        rtol=0,
    ):
        problems.append(
            f"origin: dose={image.GetOrigin()} sCT={reference.GetOrigin()}"
        )

    if not np.allclose(
        image.GetDirection(),
        reference.GetDirection(),
        atol=atol,
        rtol=0,
    ):
        problems.append(
            f"direction: dose={image.GetDirection()} "
            f"sCT={reference.GetDirection()}"
        )

    return problems


def main() -> None:
    cfg = PreprocessingConfig()
    spacing = (cfg.target_spacing,) * 3

    patient_dirs = sorted(
        path for path in DATA_ROOT.iterdir() if path.is_dir()
    )

    if not patient_dirs:
        raise RuntimeError(
            f"No patient directories found under:\n  {DATA_ROOT}"
        )

    report_path = Path("reports") / "physical_grid_audit.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_pass = 0
    n_fail = 0
    n_error = 0

    print(f"Checking {len(patient_dirs)} patients")
    print(f"Target spacing: {spacing}\n")

    for index, patient_dir in enumerate(patient_dirs, start=1):
        patient_id = patient_dir.name

        try:
            sct_path = patient_dir / "sCT" / "image_reg2MRI.nii.gz"
            dose_path = (
                patient_dir
                / "MR_StorT2"
                / "dose_interpolated.nii.gz"
            )

            # Reproduce the current preprocessing behaviour exactly.
            sct = load_nifti(sct_path)
            sct = resample_image(
                sct,
                spacing,
                sitk.sitkLinear,
            )

            dose = load_nifti(dose_path)
            dose = resample_image(
                dose,
                spacing,
                sitk.sitkLinear,
            )

            problems = compare_grids(dose, sct)

            if problems:
                status = "FAIL"
                details = " | ".join(problems)
                n_fail += 1
                print(f"FAIL  {patient_id}: {details}")
            else:
                status = "PASS"
                details = ""
                n_pass += 1

            rows.append(
                {
                    "patient_id": patient_id,
                    "status": status,
                    "details": details,
                    "sct_size": str(sct.GetSize()),
                    "dose_size": str(dose.GetSize()),
                    "sct_spacing": str(sct.GetSpacing()),
                    "dose_spacing": str(dose.GetSpacing()),
                    "sct_origin": str(sct.GetOrigin()),
                    "dose_origin": str(dose.GetOrigin()),
                    "sct_direction": str(sct.GetDirection()),
                    "dose_direction": str(dose.GetDirection()),
                }
            )

        except Exception as exc:
            n_error += 1
            print(f"ERROR {patient_id}: {exc}")

            rows.append(
                {
                    "patient_id": patient_id,
                    "status": "ERROR",
                    "details": str(exc),
                    "sct_size": "",
                    "dose_size": "",
                    "sct_spacing": "",
                    "dose_spacing": "",
                    "sct_origin": "",
                    "dose_origin": "",
                    "sct_direction": "",
                    "dose_direction": "",
                }
            )

        if index % 25 == 0 or index == len(patient_dirs):
            print(
                f"Progress: {index}/{len(patient_dirs)} | "
                f"pass={n_pass} fail={n_fail} error={n_error}"
            )

    fieldnames = [
        "patient_id",
        "status",
        "details",
        "sct_size",
        "dose_size",
        "sct_spacing",
        "dose_spacing",
        "sct_origin",
        "dose_origin",
        "sct_direction",
        "dose_direction",
    ]

    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print(f"PASS:  {n_pass}")
    print(f"FAIL:  {n_fail}")
    print(f"ERROR: {n_error}")
    print(f"Report: {report_path.resolve()}")

    if n_fail or n_error:
        print(
            "\nPhysical-grid mismatches were detected. "
            "Do not claim exact sCT-dose alignment until they are resolved."
        )
        raise SystemExit(1)

    print(
        "\nAll independently resampled sCT and dose volumes occupy "
        "matching physical grids."
    )


if __name__ == "__main__":
    main()
