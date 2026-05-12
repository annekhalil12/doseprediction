# Thesis Design: AI-Based Dose Prediction for Adaptive Pancreatic MR-Guided Radiotherapy

## Title
AI-Based Dose Prediction for Adaptive Pancreatic MR-Guided Radiotherapy: A Comparative Study of 3D U-Net, DoseGAN, and DoseGNN

## Abstract
Adaptive MR-guided radiotherapy (MRgRT) for pancreatic cancer is challenging because treatment plans must be adapted under time pressure while daily anatomical variation can substantially affect dose delivery. This thesis investigates whether deep learning dose prediction can support adaptive planning by generating dose estimates from daily MR anatomy and delineated structures.

The thesis compares a 3D U-Net baseline, DoseGAN, and DoseGNN using reproducible benchmarking on the public LUND-PROBE dataset and clinical evaluation on a private pancreatic MRgRT cohort. Evaluation focuses on voxel-wise dose prediction performance, DVH-based clinical endpoints, performance stability across fractions with different levels of anatomical variation, and the effect of public-data benchmarking or pretraining on downstream pancreatic performance after fine-tuning.

---

## Research Questions

**Main RQ:** How do different deep learning model families compare in predicting clinically relevant radiation dose distributions for adaptive MR-guided radiotherapy, and how stable are these predictions under anatomical variation?

| Sub-RQ | Measure | Main Metrics |
|---|---|---|
| SRQ1 | Predicted dose vs. clinical dose | MAE, RMSE, voxel-wise dose error |
| SRQ2 | Predicted DVH vs. reference DVH | D95, D98, Dmean, Dmax, Vx |
| SRQ3 | Error across variation levels | MAE/RMSE by variation group; correlation of variation score with error |
| SRQ4 | Transfer with vs. without public-data pretraining | MAE, RMSE, DVH differences, convergence behavior |

---

## Models Compared

1. **3D U-Net** — standard volumetric CNN baseline for voxel-wise dose prediction
2. **DoseGAN** — attention-gated conditional GAN; adversarial loss encourages realistic spatial dose distributions and reduces over-smoothing (Kearney et al. 2020, evaluated on 141 prostate SBRT patients)
3. **DoseGNN** — bipartite graph between CT nodes and dose nodes; explicitly encodes geometry and structure relationships (Dong et al. 2024; DoseGNN implementation for LUND-PROBE done in collaboration with a peer)

---

## Data

- **Public benchmark:** LUND-PROBE dataset — 432 prostate cancer patients, MRI + synthetic CT, target/OAR segmentations, radiotherapy dose distributions (Rogowski et al. 2025)
- **Private clinical cohort:** ~200 pancreatic MRgRT patients — daily MR scans, segmented targets/OARs, clinical dose plans, motion compensation, MR normalisation, DVH extraction already available

---

## Experimental Phases

- **Phase 1:** Reproducible benchmarking on LUND-PROBE (controlled model comparison)
- **Phase 2:** Clinical evaluation on private pancreatic MRgRT cohort (target use case)
- **Phase 3:** Performance stability across anatomical variation levels + transfer learning/fine-tuning from public to private data

---

## Evaluation Strategy

- **Voxel-wise accuracy:** MAE, RMSE between predicted and reference dose
- **DVH-based clinical endpoints:** D95, D98, Dmean, Dmax, Vx for targets and OARs
- **Robustness:** model performance grouped by anatomical variation level; correlation between variation score and prediction error
- **Error analysis:** spatial failure patterns in target regions vs. OARs; case-level inspection for anatomically challenging fractions
- Patient-level data splits to prevent information leakage

---

## Key Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Data quality/consistency issues in abdominal cohort | Use conservative subset with complete segmentations and reference plans |
| DoseGAN tuning or DoseGNN collaboration delays | Prioritise 3D U-Net + at least one advanced model; include remaining as partial comparison |
| Models produce incompatible output formats | Use shared clinically meaningful DVH endpoints where direct metric comparison is not appropriate |
| Anatomical variation cannot be operationalised uniformly | Evaluate robustness on subset with sufficient fraction-level information |

---

## Project Timeline (13 weeks)

| Phase | Weeks |
|---|---|
| Literature review and thesis design finalisation | 1–2 |
| Public and private data preparation + preprocessing checks | 2–4 |
| 3D U-Net baseline implementation | 3–5 |
| DoseGAN and DoseGNN implementation/comparison | 5–8 |
| Quantitative evaluation and robustness/error analysis | 7–10 |
| Results, discussion, conclusion writing | 9–12 |
| Final revision, feedback, and buffer | 12–13 |

---

## References

1. Chen et al. (2024) — Modified U-Net with graph representation for esophageal cancer dose prediction
2. Dong et al. (2025) — LLM-empowered 3D dose prediction for clinician-AI collaborative planning
3. Dong et al. (2024) — DoseGNN: graph neural networks for adaptive dose-volume histogram prediction
4. Gao et al. (2023) — Flexible-CmGAN: precise 3D dose prediction in radiotherapy
5. Kazemzadeh et al. (2025) — AI for radiotherapy dose prediction: comprehensive review
6. Kearney et al. (2020) — DoseGAN: attention-gated GAN for synthetic dose prediction
7. Kui et al. (2024) — Review of dose prediction methods for tumor radiation therapy
8. Murakami et al. (2020) — Fully automated dose prediction using GAN in prostate cancer
9. Nemoto et al. (2025) — Deep learning automated radiotherapy planning for early-stage lung cancer
10. Rogowski et al. (2025) — LUND-PROBE dataset
11. Tetar et al. (2020) — Daily adaptive stereotactic MR-guided radiotherapy for renal cell cancer
12. Zamanian et al. (2025) — Deep learning for head and neck radiation dose prediction: systematic review
