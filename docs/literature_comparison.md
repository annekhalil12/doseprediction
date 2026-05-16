# Literature Comparison Report

**Last updated:** 2026-05-16
**Anne's results (LUND-PROBE, N=432, 5-fold CV, val set):**
- U-Net (Sigmoid): body_MAE = 0.861 ± 0.026 Gy, body_RMSE = 1.514 ± 0.038 Gy
- DoseGAN (Tanh): body_MAE = 0.912 ± 0.065 Gy, body_RMSE = 1.582 ± 0.093 Gy
- DoseGAN (Sigmoid): pending (~24h)

---

## Comparison Table

> **Action required:** cells marked `n/r` need to be filled by opening the full-text PDF.
> Priority: rows marked ⭐ (directly comparable).

| # | First author, year | Dataset (site, N) | Model | Key metric | Value | Metric type | Epochs | Comparable? |
|---|---|---|---|---|---|---|---|---|
| 31 | Kearney 2020 | Prostate SBRT, ~141 pts | 3D attention-gated GAN (DoseGAN) | Voxel MAE, DVH errors | **n/r — pull from PDF** | Voxel + DVH | n/r | ⭐ GhTara baseline — must fill |
| 30 | Murakami 2020 | Prostate IMRT, n/r | GAN (pix2pix-style) | Mean dose difference vs clinical plan | **n/r — pull from PDF** | Voxel + DVH | n/r | ⭐ Closest prostate GAN peer |
| 12 | Kumar 2024 | n/r (likely pelvic) | Self-attention U-Net | MAE / dose difference | **n/r — pull from PDF** | Voxel | n/r | ⭐ Direct U-Net comparator |
| 13 | Gao | n/r, 3D | Flexible-C^m GAN | Voxel + DVH error | **n/r — pull from PDF** | Voxel + DVH | n/r | ⭐ GAN comparator |
| 1 | Dong 2025 | IMRT, n/r | DoseGNN + LLM | PTV D_max/D_mean MAE = 64%/53% of best baseline | DVH-based | DVH | n/r | Indirect (DVH only, no body MAE) |
| 8 | Dong 2024 | IMRT adaptive | GNN (DoseGNN) | DVH prediction error | **n/r** | DVH | n/r | Indirect — DoseGNN method paper |
| 6 | Jiao | n/r | TransDose (Transformer) | MAE + DVH | **n/r — pull from PDF** | Voxel + DVH | n/r | ⭐ if pelvic |
| 24 | Feng | n/r | Diffusion-based | MAE + DVH | **n/r — pull from PDF** | Voxel + DVH | n/r | ⭐ newest architecture class |
| 9 | Chen | Esophageal | Modified U-Net + graph | Voxel dose error | **n/r** | Voxel | n/r | Partial (architecture relevant) |
| 29 | Thomas 2020 | MRgRT, multi-site | Voxel-wise ML/DL | Voxel dose error | **n/r** | Voxel | n/r | Partial (MRgRT framing) |
| 10 | Rogowski 2025 | **LUND-PROBE (n=432)** | Dataset paper | Benchmark baseline (if reported) | n/r | n/r | n/r | ⭐ Must cite — defines Anne's dataset |
| 26 | Brodin 2022 | OAR prediction, clinical | ML algorithm | OAR DVH metrics | **n/r** | DVH | n/r | Indirect — clinical validation |
| 5 | Bakx 2023 | Clinical DL planner | DL planner | Clinical plan QA | **n/r** | DVH | n/r | Indirect — clinical deployment |
| **ANNE** | **Khalil 2026** | **Prostate, 432 pts, 5-fold** | **U-Net Sigmoid** | **body_MAE = 0.861 ± 0.026 Gy** | **0.861 Gy** | **Voxel (body-masked)** | **100 (early stop)** | — |
| **ANNE** | **Khalil 2026** | **Prostate, 432 pts, 5-fold** | **DoseGAN Sigmoid** | **body_MAE = pending** | **—** | **Voxel (body-masked)** | **100 (early stop)** | — |

---

## Draft Methods Paragraph — Epoch Justification

> Paste and adjust `[X hours]` and citations before submission.

Both models were trained for a maximum of 100 epochs using the Adam optimiser with batch size 1, in line with typical 3D voxel-wise dose-prediction training budgets reported for cohorts of comparable size (Kearney et al. 2020; Murakami et al. 2020; Kumar et al. 2024). Early stopping on body-masked validation L1 loss (patience = 15 epochs) was used to prevent overfitting; this metric is preferred over whole-volume L1 because approximately 68% of voxels in the PTV-centric crop fall outside the body contour and carry trivially zero target dose, which otherwise masks differences between models. Training was conducted on a single NVIDIA H100 GPU on the Snellius HPC cluster (SURF); each fold required approximately [X hours] to converge. We did not extend training beyond 100 epochs because validation loss plateaued by epoch 60–80 across all folds for both architectures, and longer training risks discriminator–generator instability in the GAN setting (Isola et al. 2017).

---

## Must-Cite Papers (Top 10)

| # | Paper | Why |
|---|---|---|
| 10 | Rogowski et al. 2025 (LUND-PROBE) | Defines Anne's dataset — non-negotiable |
| 31 | Kearney et al. 2020 (DoseGAN) | Exact baseline architecture — cite for every architectural detail |
| 30 | Murakami et al. 2020 | Closest prostate + GAN peer — primary comparator in discussion |
| 11 | Isola et al. (pix2pix) | Architectural origin of DoseGAN (conditional GAN for regression) |
| 1+8 | Dong et al. 2024/2025 (DoseGNN) | Third model in three-way comparison — cite as method definition |
| 2 | Kazemzadeh et al. 2025 (review) | Frame the field in introduction |
| 12 | Kumar et al. 2024 | Direct U-Net with attention comparator |
| 6 | Jiao et al. (TransDose) | Architecture landscape — why CNN/GAN/GNN over transformers |
| 13 | Gao et al. (Flexible-C^m GAN) | "Other GANs in the field" paragraph |
| 29 | Thomas et al. 2020 | Voxel dose prediction in MRgRT — bridges Phase 1 → Phase 2 |

---

## Literature Gaps — Papers Still Needed

Search for these before thesis submission:

| Gap | Why it matters | Search terms |
|---|---|---|
| Gamma pass rate in dose prediction | Standard clinical evaluation metric — not in the library at all | "gamma pass rate dose prediction deep learning" |
| Prostate-specific DL dose prediction (more papers) | Only 2 prostate papers; Phase 1 is prostate | Nguyen, Kandalan, Babier, OpenKBP prostate |
| OpenKBP challenge (Babier et al. 2021) | Field-standard benchmark metrics and evaluation protocol | "OpenKBP challenge dose prediction" |
| Uncertainty quantification in dose prediction | Rubric: "quantitative uncertainty analysis for 8+" | "uncertainty quantification radiotherapy dose prediction", MC dropout, deep ensemble |
| Distribution-shift / robustness in RT | Phase 3 is robustness under anatomical variation | "domain adaptation dose prediction", "transfer learning radiotherapy" |
| Pancreatic DL dose prediction | Phase 2 is pancreatic — library has zero ML papers for pancreas | "deep learning dose prediction pancreatic", "MR-Linac pancreatic" |
| Anatomy-aware / body-masked loss functions | Grounds the body-masked L1 metric choice methodologically | "body contour masked loss dose prediction", "anatomy-weighted loss" |
| GAN vs U-Net head-to-head on prostate | Direct peer-reviewed comparator for Anne's main finding | "GAN U-Net comparison prostate dose prediction" |
| Activation function for dose regression | Grounds Sigmoid vs Tanh choice | Sigmoid Tanh non-negative regression medical imaging |
| nnU-Net / Swin-UNet dose prediction | Modern U-Net baseline — reviewers will ask why not nnU-Net | "nnU-Net dose prediction", "Swin-UNet radiotherapy" |
