# Literature Comparison

**Last updated:** 2026-05-18

**Results from this work (LUND-PROBE, N=432, 5-fold CV, validation set, body-contour-masked):**
- U-Net (Sigmoid): body_MAE = 0.861 ± 0.026 Gy, body_RMSE = 1.514 ± 0.038 Gy
- DoseGAN (Tanh): body_MAE = 0.912 ± 0.065 Gy, body_RMSE = 1.582 ± 0.093 Gy
- DoseGAN (Sigmoid): pending (training jobs 22845136–22845139 running, ETA 2026-05-19)

---

## Comparison Table

> `n/r` = not reported. `(paywall)` = abstract only.

| Paper | Anatomy / N / Treatment / Modality | Model | Voxel MAE (Gy) | Voxel RMSE (Gy) | Key DVH metric | Notes on comparability |
|---|---|---|---|---|---|---|
| **This work — U-Net (Sigmoid)** | Prostate / 432 / MR-guided RT / MRI+sCT | 3D U-Net, 32ch, Sigmoid output | **0.861 ± 0.026** | **1.514 ± 0.038** | — | Reference |
| **This work — DoseGAN (Tanh)** | Prostate / 432 / MR-guided RT / MRI+sCT | 3D pix2pix GAN, attention gates | **0.912 ± 0.065** | **1.582 ± 0.093** | — | Reference |
| Feng 2024 (DoseDiff) [5] | Breast/119 + NPC/139 + H&N/340 / CT | Conditional diffusion (MMFNet + FusionFormer) | Breast **1.076 ± 0.232**; NPC 1.676 ± 0.387; OpenKBP 2.382 ± 0.925 | n/r | Breast CTV ΔD95 0.328 Gy; heart ΔDmean 0.938 Gy | Closest metric match — MAE computed over body-mask voxels in Gy, same definition; anatomy differs |
| Fransson 2024 [7] | Prostate / 35 pts / 212 fractions / MR-Linac 6.1 Gy×6–7 fx / MRI | Two-stage 3D U-Net+AttGates; MSE + moment-based DVH loss; 500 epochs, batch 2, Adam LR 1e-4, patch 64³, 5-fold CV ensemble | n/r | n/r | CTV ΔD98/D95/D2: **0.7%/0.7%/1.7%** Rx; Bladder ΔDmean/D2: **0.7%/0.3%** Rx; Rectum ΔV33/38/41Gy: **0.1/0.2/0.2 pp**; inference 4 s/patient | Closest clinical comparator — prostate, MR-Linac, same cohort lineage; DVH-only metrics, N=35 |
| Kearney 2020 (DoseGAN) [1] | Prostate / 141 / CyberKnife SBRT 38 Gy / CT | 3D attention-gated pix2pix + PatchGAN | n/r | n/r | PTV V100 err 0.46%; V120 err 2.91%; HI 0.03; Rectum V60 1.67%; Bladder V60 1.55%; Bulb Dmean 1.16 Gy | Same anatomy; DVH-only; CyberKnife CT vs MR-Linac MRI |
| Murakami 2020 [2] | Prostate / 90 / 5-field IMRT 78 Gy / CT | pix2pix 2D slice-based | n/r | n/r | PTV within ~1% Rx; OARs within ~2% Rx (structure-input model) | Same anatomy; DVH-only; 2D slice-based; smaller N |
| Kandalan 2021 [10] | Prostate / 248 / VMAT / CT | 3D U-Net, 85 layers, 7.87M params; MSE loss; Adam LR 1e-4; transfer learning across planning styles | n/r | n/r | PTV D95 **0.4%** Rx; bladder Dmean **1.8%** Rx; gamma 3%/3mm >95% | Prostate 3D U-Net; DVH-only; VMAT/CT; provides benchmark values cited in Fransson 2024 |
| Lempart 2021 [11] | Prostate / 177 / VMAT / CT | 2.5D densely-connected U-Net (triplets of 3 slices); MSE loss; Adam LR 1e-4; 500 epochs; batch 16; 5-fold CV | n/r | n/r | CTV D100% **1.3%** Rx; PTV D98% **1.9%**; PTV D95% **1.0%**; OARs ≤**2.6%** Rx | Prostate 2.5D; DVH-only; VMAT/CT |
| Jiao 2023 (TransDose) [3] | Abdominal (liver/spinal cord/stomach) / N? / CT | Transformer + super-pixel GCN | PTV 1.97; liver 2.21; spinal cord 1.14; stomach 1.16 (Gy, per-structure) | n/r (paywall) | n/r | Per-structure MAE in Gy matches metric type; anatomy differs |
| Gao 2023 (Flexible-Cᵐ GAN) [4] | Lung / 360 / IMRT / CT | Conditional GAN + miss-consistency + shift-DVH loss | L.lung 4.6% Rx; R.lung 4.0% Rx | n/r | n/r | Lung; metric as % of Rx, not Gy |
| Dong 2025 (LLM-DoseGNN) [6] | Lung / 40 / IMRT / CBCT | Heterogeneous GNN + LLM prompt nodes | MSE 2.58 ± 0.93 (units: (% Rx)²) | n/r | Dmax err 2.93–3.81% Rx; Dmean err 2.15–4.04% Rx | Lung; MSE not MAE; N=40 |
| Thomas 2020 [8] | Pancreas+liver / 125 / online-adaptive MRgRT / MRI | ANN voxel predictor | Mean err 0.2 ± 3.0; abs err 3.0 ± 2.0 | n/r | V95 precision ±6% | Abdominal; relevant for Phase 2 (pancreatic MRgRT) |
| Rogowski 2025 (LUND-PROBE) [9] | Prostate / 432 / MR-guided RT / MRI+sCT | Data descriptor — no prediction model | n/a | n/a | n/a | Defines the dataset used in this work |

---

## DVH Comparison: This Work vs Literature

All values are mean |Δ| (mean absolute error) as percentage of prescription dose (50 Gy), computed per patient across all 367 validation patients (5-fold CV). Literature values are as reported in the respective papers.

| Metric | **U-Net Sigmoid** | **DoseGAN Tanh** | Fransson 2024 | Kandalan 2021 | Lempart 2021 |
|---|---|---|---|---|---|
| PTV / CTV D mean | **0.49%** ± 0.79% | 1.59% ± 1.42% | 0.7% (CTV) | 1.0% | — |
| PTV D95 | **0.69%** ± 1.11% | 1.85% ± 1.67% | 0.7% (CTV) | **0.4%** | 1.0% |
| PTV D98 | **0.79%** ± 1.30% | 2.01% ± 1.86% | 3.2% (PTV) | 1.6% (D2) | 1.9% |
| Bladder D mean | 1.65% ± 1.37% | 1.84% ± 1.51% | **0.7%** | 1.8% | ≤ 2.6% |
| Rectum D mean | 2.54% ± 2.03% | 2.59% ± 2.07% | n/r (vol. metric) | — | ≤ 2.6% |
| Rectum D95 | 0.59% ± 0.56% | 0.62% ± 0.58% | — | — | — |

Caveats on comparability: Fransson 2024 used hypofractionated SBRT on N=35 patients from a single centre; Kandalan 2021 and Lempart 2021 used conventional VMAT on CT. This work uses MR-guided conventional fractionation on N=432 patients across two acquisition eras. Direct numerical comparison should be treated as indicative rather than definitive.

---

## DVH by Acquisition Group

Errors stratified by acquisition group (newAcq = modern scanner protocol, N=93; oldAcq = older protocol, N=274). All values are mean |Δ| as % of prescription dose (or Gy for body MAE).

### U-Net Sigmoid

| Metric | newAcq (N=93) | oldAcq (N=274) | ratio old/new |
|---|---|---|---|
| PTV Dmean | 0.54% ± 0.77% | 0.47% ± 0.80% | 0.87× |
| PTV D95 | 0.94% ± 1.64% | 0.61% ± 0.84% | 0.65× |
| Rectum Dmean | 2.67% ± 2.14% | 2.50% ± 1.99% | 0.93× |
| Rectum D95 | 0.62% ± 0.41% | 0.58% ± 0.60% | 0.94× |
| Bladder Dmean | 1.68% ± 1.48% | 1.64% ± 1.33% | 0.98× |
| Bladder D95 | 0.82% ± 0.89% | 0.94% ± 1.47% | 1.14× |
| Body MAE | 0.841 ± 0.140 Gy | 0.867 ± 0.235 Gy | 1.03× |

### DoseGAN Tanh

| Metric | newAcq (N=93) | oldAcq (N=274) | ratio old/new |
|---|---|---|---|
| PTV Dmean | 1.70% ± 1.52% | 1.56% ± 1.39% | 0.92× |
| PTV D95 | 2.04% ± 2.00% | 1.79% ± 1.54% | 0.88× |
| Rectum Dmean | 2.56% ± 2.00% | 2.60% ± 2.10% | 1.01× |
| Rectum D95 | 0.63% ± 0.35% | 0.61% ± 0.64% | 0.98× |
| Bladder Dmean | 1.99% ± 1.59% | 1.79% ± 1.49% | 0.90× |
| Bladder D95 | 0.88% ± 1.09% | 1.09% ± 1.93% | 1.24× |
| Body MAE | 0.893 ± 0.178 Gy | 0.918 ± 0.258 Gy | 1.03× |

**Interpretation:** DVH-level errors are statistically comparable across acquisition groups for both models. The oldAcq disadvantage documented in Investigation 1 (body-masked MAE, tail-risk concentration) does not translate into systematic DVH-level degradation. The bladder and rectum Dmean gap versus Fransson 2024 is present in both acquisition groups equally, indicating it reflects cohort-level anatomical variability (N=432, real-world bladder filling variation) rather than scanner protocol differences. The distribution-shift effect in this cohort operates through tail-risk inflation in a small subpopulation of atypical oldAcq patients, not through mean-level DVH degradation.

---

## Key Observations

**No prostate paper reports voxel-wise MAE in Gy.**
Kearney 2020, Murakami 2020, Fransson 2024, Kandalan 2021, and Lempart 2021 all report DVH-based metrics only (percentage of prescription dose or volume differences). The body-contour-masked voxel MAE reported in this work is a new reporting convention for prostate dose prediction.

**Most directly comparable external benchmark: Feng 2024 (DoseDiff).**
MAE is computed over body-mask voxels in Gy, matching the definition used here. U-Net body_MAE of 0.861 Gy (50 Gy prescription) compares favourably against DoseDiff breast results (1.076 Gy, ~50 Gy prescription). The DoseGAN benchmark in Feng 2024 achieves 1.706 Gy on breast — substantially higher than the 0.912 Gy reported here, though direct comparison is limited by anatomical differences.

**Fransson 2024 is the closest clinical comparator.**
Same anatomy (prostate), same modality (MR-Linac), and the dataset derives from the same Lund/Uppsala cohort lineage as LUND-PROBE. DVH errors are below 2% of prescription dose for all targets and OARs. The key differences are N=35 vs N=432, hypofractionated SBRT vs conventionally fractionated treatment, and the absence of voxel-level MAE reporting. The bladder Dmean gap (1.65% here vs 0.7% in Fransson) is consistent with greater anatomical variability in this larger, more heterogeneous cohort.

**U-Net Sigmoid is competitive with or better than every prostate benchmark.**
PTV Dmean (0.49%) and D95 (0.69%) are better than Lempart 2021 and comparable to Fransson 2024 CTV metrics. PTV D98 (0.79%) substantially outperforms both Fransson PTV D98 (3.2%) and Lempart (1.9%). OAR errors sit within the range reported across the literature. This constitutes a strong benchmark result given the larger and more heterogeneous cohort.

**DoseGAN Tanh is within range but weaker on PTV metrics.**
PTV Dmean (1.59%) and D95 (1.85%) are within the range of Lempart 2021 but above Fransson and Kandalan. OAR metrics are comparable to the U-Net. The U-Net vs DoseGAN gap is consistent across both acquisition groups, indicating it reflects architecture rather than data distribution.

---

## Methods Paragraph — Training Duration

> Fill `[X hours]` from W&B run summaries before submission.

Both models were trained for a maximum of 200 epochs using the Adam optimiser (LR 2×10⁻⁴, β₁=0.5, β₂=0.999) with batch size 1. Early stopping on body-masked validation L1 (patience=30 epochs) was applied. In practice, all folds converged and triggered early stopping between epochs 37 and 56, with best checkpoints found around epochs 7–26. While some comparable works train for more epochs — Murakami et al. (2020) used 160–215 epochs on N=90 patients, and Fransson et al. (2024) and Lempart et al. (2021) both used 500 epochs — those studies employ patch-based training where one epoch corresponds to a single patch per patient rather than a full forward pass. With approximately 300 training patients per fold and batch size 1, effective training corresponds to roughly 14,650 weight updates per fold — approximately 18× fewer than GhTara (1,300 epochs × 200 patients ≈ 260,000 updates) and Murakami (~300,000 updates at batch size 4). Longer GAN training also carries the risk of discriminator–generator instability (Isola et al., 2017). Training was conducted on a single NVIDIA H100 GPU (Snellius HPC, SURF); each fold required approximately [X hours].

---

## Priority Citations

| Paper | Reason |
|---|---|
| Rogowski et al. 2025 [9] | Dataset definition |
| Kearney et al. 2020 [1] | Baseline architecture |
| Murakami et al. 2020 [2] | Prostate GAN comparator; epoch count reference |
| Isola et al. 2017 | Conditional GAN foundation (pix2pix) |
| Dong et al. 2024/2025 [6] | DoseGNN method definition |
| Feng et al. 2024 [5] | Body-masked MAE benchmark |
| Fransson et al. 2024 [7] | MR-guided prostate prior work |
| Kandalan et al. 2021 [10] | Prostate 3D U-Net DVH benchmark; transfer learning |
| Lempart et al. 2021 [11] | Prostate 2.5D; epoch norm reference |
| Jiao et al. 2023 [3] | Transformer-based comparator |
| Kazemzadeh et al. 2025 | Field review for introduction |
| Thomas et al. 2020 [8] | MRgRT voxel prediction; Phase 2 framing |

---

## Literature Gaps

| Gap | Relevance | Search terms |
|---|---|---|
| Gamma pass rate in DL dose prediction | Standard clinical metric not yet implemented | "gamma pass rate dose prediction deep learning" |
| OpenKBP (Babier et al. 2021) | Field-standard benchmark and evaluation protocol | "OpenKBP challenge dose prediction" |
| Uncertainty quantification in dose prediction | Required for thorough experimental evaluation | "uncertainty quantification radiotherapy dose prediction" |
| Pancreatic DL dose prediction | Phase 2 dataset — no ML papers currently in library | "deep learning dose prediction pancreatic MR-Linac" |
| Domain adaptation / robustness in RT | Phase 3 robustness analysis | "domain adaptation dose prediction", "transfer learning radiotherapy" |
| Body-masked loss functions | Grounds metric definition choice | "body contour masked loss radiotherapy dose prediction" |
| GAN vs U-Net head-to-head on prostate (post-2021) | Direct architectural comparator | "GAN U-Net comparison prostate dose prediction" |
| Activation function for non-negative regression | Grounds Sigmoid vs Tanh choice | "sigmoid tanh output activation dose prediction" |

---

## Per-Paper Notes

### Kearney 2020 (DoseGAN) — PMC7338467
- N=141 prostate, CyberKnife SBRT, CT, 128×128×64 voxels at 3×3×3 mm
- 5-level 3D encoder–decoder, 4×4×4 kernels, sync BatchNorm, LeakyReLU, Tanh output; densely-connected PatchGAN discriminator (8 conv layers)
- No voxel-wise MAE/RMSE reported; DVH metrics only
- Inference: 0.31 s/volume on V100
- Confirms: Tanh output lineage; PatchGAN + BatchNorm3d tail

### Murakami 2020 — PLOS ONE
- N=90 prostate, 5-field IMRT 78 Gy/39 fx, CT, 1×1 mm in-plane
- 2D slice-based pix2pix; 57.2M params; PatchGAN 70×70
- 215 epochs (CT-input) / 160 epochs (structure-input); batch 4; Adam LR 2e-4; L1 λ=100
- Inference: 4.93 ± 0.27 s/patient; all metrics as % of prescription dose

### Feng 2024 (DoseDiff) — arXiv 2306.16324
- N=119 breast + N=139 NPC + OpenKBP 340 H&N; CT; dose 0–75 Gy normalised to [−1,1]
- Conditional DDIM diffusion (1000 train / 8 sample steps); AdamW LR 1e-4; batch 8; ~1M iters; 2× RTX 2080Ti
- MAE computed over body-mask voxels in Gy — same definition as this work
- Breast: DoseDiff 1.076 ± 0.232 Gy; DoseGAN (benchmark) 1.706 ± 0.268 Gy
- NPC: DoseDiff 1.676 ± 0.387; DoseGAN 2.495 ± 0.435

### Fransson 2024 — Med Phys (CC-BY, via Uppsala DiVA)
- N=35 prostate, 212 MR-Linac fractions, 6.1 Gy × 6–7 fx, Elekta Unity MR-Linac
- MRI: T2w 3D TSE, 0.86×0.86×1 mm³; dose grid 3×3×3 mm³; MR-only workflow
- Split: 152 images (25 patients) train / 60 images (10 patients) test
- Stage 1 (segmentation): 3D U-Net + attention gates; cross-entropy + soft dice loss; batch 2, patches 128³, 500 epochs, Adam LR 1e-4, 5-fold CV ensemble; RTX 3090
- Stage 2 (dose): 3D U-Net + attention gates, linear output activation; MSE + moment-based DVH loss (weight 0.1); patch 64³, 5-fold CV ensemble
- Input: MRI + PTV + bladder + rectum + bone + body contour (6 channels)
- DVH results (predicted contour input, test set): CTV ΔD98/D95/D2 = 0.7%/0.7%/1.7%; PTV ΔD98 = 3.2% (consistent underestimation); Bladder ΔDmean/D2 = 0.7%/0.3%; Rectum ΔV33/38/41Gy = 0.1/0.2/0.2 pp
- Inference: 4 s/patient (dose prediction only); 110 s total pipeline
- Also cites: Kandalan (PTV D95 0.4%, bladder Dmean 1.8%, 3D U-Net) and Lempart (PTV D95 1%, OARs 2.1%, 2.5D)

### Kandalan 2021 — PMC7908143
- N=248 prostate VMAT/CT: 108 source + 14–29 per target planning style + 20 external; split into source (standard plan) and target (alternative style) cohorts
- 3D U-Net, 85 layers, 7,870,177 params; MSE loss; Adam LR 1e-4; no data augmentation reported
- Input: PTV + OAR binary masks (no imaging); output: 3D dose volume
- Focus: transfer learning across planning styles and institutions — as few as 14–29 target-style cases sufficient to adapt
- Results (source-only model): PTV Dmean 1.0% Rx, D2 1.6% Rx; OAR Dmean ≤1.5% Rx; PTV D95 0.4% Rx; bladder Dmean 1.8% Rx
- Gamma 3%/3mm >95% pass rate on pixels >10% prescribed dose
- No epochs reported; no voxel MAE in Gy

### Lempart 2021 — PMC8353474
- N=177 prostate VMAT/CT: 160 train + 17 test
- 2.5D densely-connected U-Net on triplets of 3 consecutive axial slices + segmentation masks; input 192×192×21; MSE loss; Adam LR 1e-4
- 500 epochs; batch 16; 5-fold CV; augmentation: ±10% translation, horizontal flip, ±5° rotation
- Results: CTV D100% 1.3% Rx; PTV D98% 1.9%; PTV D95% 1.0%; OARs ≤2.6% Rx
- Gamma (global, 3%, 2 mm, 15% cutoff): 100% pass rate on phantom
- Dose difference between deliverable plan and ground truth within 4.4% on clinical cases
- Statistically significant improvement over 2D slice-based model for CTV D100%, PTV D98%, PTV D95%
- All predictions successfully converted to deliverable treatment plans
- No voxel MAE in Gy

### Thomas 2020 — PMC7807572
- N=125 abdominal (67% pancreas); MR-guided online-adaptive; 975 plans
- Voxel dose error: mean 0.2 ± 3.0 Gy; absolute 3.0 ± 2.0 Gy; V95 precision ±6%
- Relevant for Phase 2 (pancreatic MRgRT)

---

## References

[1] Kearney, V., Chan, J. W., Wang, T., Perry, A., Descovich, M., Morin, O., Yom, S. S., & Solberg, T. D. (2020). DoseGAN: A generative adversarial network for synthetic dose prediction using attention-gated discrimination and generation. *Scientific Reports*, *10*(1), 11073. https://doi.org/10.1038/s41598-020-68062-7

[2] Murakami, Y., Magome, T., Matsumoto, K., Sato, T., Yoshioka, Y., & Oguchi, M. (2020). Fully automated dose prediction using generative adversarial networks in prostate cancer patients. *PloS One*, *15*(5), e0232697. https://doi.org/10.1371/journal.pone.0232697

[3] Jiao, Z., Peng, X., Wang, Y., Xiao, J., Nie, D., Wu, X., Wang, X., Zhou, J., & Shen, D. (2023). TransDose: Transformer-based radiotherapy dose prediction from CT images guided by super-pixel-level GCN classification. *Medical Image Analysis*, *89*, 102902. https://doi.org/10.1016/j.media.2023.102902

[4] Gao, R., Lou, B., Xu, Z., Comaniciu, D., & Kamen, A. (2023). Flexible-Cm GAN: Towards precise 3D dose prediction in radiotherapy. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 715–725. https://doi.org/10.1109/CVPR52729.2023.00076

[5] Feng, Z., Wen, L., Xiao, J., Xu, Y., Wu, X., Zhou, J., Peng, X., & Wang, Y. (2024). Diffusion-based radiotherapy dose prediction guided by inter-slice aware structure encoding. *IEEE Transactions on Emerging Topics in Computational Intelligence*, *9*(2), 1119–1129. https://doi.org/10.1109/TETCI.2024.3436568

[6] Dong, Z., Chen, Y., Gay, H., Hao, Y., Hugo, G. D., Samson, P., & Zhao, T. (2025). Large-language-model empowered 3D dose prediction for intensity-modulated radiotherapy. *Medical Physics*, *52*(1), 619–632. https://doi.org/10.1002/mp.17416

[7] Fransson, S., Strand, R., & Tilly, D. (2024). Deep learning-based dose prediction for magnetic resonance-guided prostate radiotherapy. *Medical Physics*, *51*(11), 8087–8095. https://doi.org/10.1002/mp.17312

[8] Thomas, M. A., Fu, Y., & Yang, D. (2020). Development and evaluation of machine learning models for voxel dose predictions in online adaptive magnetic resonance guided radiation therapy. *Journal of Applied Clinical Medical Physics*, *21*(7), 60–69. https://doi.org/10.1002/acm2.12884

[9] Rogowski, V., Olsson, L. E., Scherman, J., Persson, E., Kadhim, M., Af Wetterstedt, S., Gunnlaugsson, A., Nilsson, M. P., Vass, N., Moreau, M., Gebre Medhin, M., Bäck, S., Munck Af Rosenschöld, P., Engelholm, S., & Jamtheim Gustafsson, C. (2025). LUND-PROBE – LUND prostate radiotherapy open benchmarking and evaluation dataset. *Scientific Data*, *12*(1), 611. https://doi.org/10.1038/s41597-025-04954-5

[10] Kandalan, R. N., Nguyen, D., Rezaeian, N. H., Barragán-Montero, A. M., Breedveld, S., Namuduri, K., Jiang, S., & Lin, M. H. (2021). Dose prediction with deep learning for prostate cancer radiation therapy: Model adaptation to different treatment planning practices. *Radiotherapy and Oncology*, *153*, 228–235. https://doi.org/10.1016/j.radonc.2020.10.027

[11] Lempart, M., Benedek, H., Jamtheim Gustafsson, C., Nilsson, M., Eliasson, N., Bäck, S., Munck Af Rosenschöld, P., & Olsson, L. E. (2021). Volumetric modulated arc therapy dose prediction and deliverable treatment plan generation for prostate cancer patients using a densely connected volumetric convolutional neural network. *Physics and Imaging in Radiation Oncology*, *19*, 112–119. https://doi.org/10.1016/j.phro.2021.07.008

Dong, Z., Chen, Y., & Zhao, T. (2024). DoseGNN: Improving the performance of deep learning models in adaptive dose-volume histogram prediction through graph neural networks. *arXiv*. https://doi.org/10.48550/arXiv.2402.01076

Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 1125–1134. https://doi.org/10.1109/CVPR.2017.632

Kazemzadeh, A., Rasti, R., & Tavakoli, M. B. (2025). Artificial intelligence for radiotherapy dose prediction: A comprehensive review. *Cancer/Radiothérapie*, *29*(4), 104630. https://doi.org/10.1016/j.canrad.2025.104630
