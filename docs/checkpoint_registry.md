# Checkpoint Registry

Checkpoints are too large (~250–370 MB each) to commit to git. This file
is the authoritative record of what exists, where it lives, and how to
verify integrity.

**Storage location:** Snellius HPC (SURF)
`/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction/outputs/`

**Important:** Snellius scratch (`/gpfs/scratch1/`) is **not backed up**.
Before thesis submission, copy final checkpoints to a persistent location:
- Snellius home quota: `~/` (backed up, 200 GB quota)
- SURF Research Drive or Data Repository (for long-term archival)
- Zenodo (for public release alongside the paper)

**Availability for reviewers:** Checkpoints are available on request from
annekhalil12@gmail.com. SHA-256 hashes below allow verification of
checkpoint integrity after transfer.

---

## Final conditions (4 × 5 folds = 20 checkpoints)

### DoseGAN baseline — `outputs/checkpoints_dosegan/`

| Fold | File | Size | SHA-256 |
|---|---|---|---|
| 0 | `dosegan_ngf32_sigmoid_snellius_fold0_best.pt` | 370 MB | `374073138e97cdd2158ac59cb3cb0a27d63fbe138ce0747bd234e1b2fb4d78e4` |
| 1 | `dosegan_ngf32_sigmoid_snellius_fold1_best.pt` | 370 MB | `f8743c524a851c9cd710a6400bc70e52a839fcb5ba76c9321c293aae100c8a13` |
| 2 | `dosegan_ngf32_sigmoid_snellius_fold2_best.pt` | 370 MB | `d41ead4301b41f839190b0c63046752a1c6e99c0e7dee55c61e28d40946c7ee3` |
| 3 | `dosegan_ngf32_sigmoid_snellius_fold3_best.pt` | 370 MB | `da935c870b467b662565888621ea7975beb6fc58089f1e7eb51784a00c787192` |
| 4 | `dosegan_ngf32_sigmoid_snellius_fold4_best.pt` | 370 MB | `b0fbf24dfb510f8063f78e0b90eebdbe50bd6edb79feb9970a9914227fdfe137` |

### DoseGAN geom — `outputs/checkpoints_dosegan/`

| Fold | File | Size | SHA-256 |
|---|---|---|---|
| 0 | `dosegan_ngf32_sigmoid_geom_snellius_fold0_best.pt` | 371 MB | `991f867511260669e48d8caf6d48ea8f3dc0ab06f51b0b3f73239d15b1e95d28` |
| 1 | `dosegan_ngf32_sigmoid_geom_snellius_fold1_best.pt` | 371 MB | `39490d27117c256fe372ada5adbccf6fe0f63a51a74e4b2b34c037ba896f43d4` |
| 2 | `dosegan_ngf32_sigmoid_geom_snellius_fold2_best.pt` | 371 MB | `71cb09b3e8fde05879c80519096be1f9f933a764327615761dc205f014f6a51e` |
| 3 | `dosegan_ngf32_sigmoid_geom_snellius_fold3_best.pt` | 371 MB | `88a8590beb5b0dcc98cae3ed4350145015eb2dae6039b10b8c00aa60cca0673b` |
| 4 | `dosegan_ngf32_sigmoid_geom_snellius_fold4_best.pt` | 371 MB | `9bb544e3127d04c3cef301d3b8a120a0c7d42b79e0087cd73017f5cde34ff9db` |

### U-Net baseline — `outputs/checkpoints_unet3d/`

| Fold | File | Size | SHA-256 |
|---|---|---|---|
| 0 | `unet3d_ch32_sigmoid_snellius_fold0_best.pt` | 249 MB | `7b43aec880408b8be0f1a73ba12a70ec629f5846a0fe6d8787b8c643a1922e4d` |
| 1 | `unet3d_ch32_sigmoid_snellius_fold1_best.pt` | 249 MB | `382bb48042016deee9f2c7440fac55bcfa1aef0f997ccc75cf22bc01071abfcb` |
| 2 | `unet3d_ch32_sigmoid_snellius_fold2_best.pt` | 249 MB | `e8fddef17ed91571b35005b3975414da7c03fd3d7a7728f2d3e021c85a2294c8` |
| 3 | `unet3d_ch32_sigmoid_snellius_fold3_best.pt` | 249 MB | `72ea2713aff446f5d14a604fce76e737e0cb9b5a6cba445071eaec49eac3d5a5` |
| 4 | `unet3d_ch32_sigmoid_snellius_fold4_best.pt` | 249 MB | `643b354c1951ab1e604fb37934cb164946a17526b94db77d4a9e3c75d97f7974` |

### U-Net geom — `outputs/checkpoints_unet3d/`

| Fold | File | Size | SHA-256 |
|---|---|---|---|
| 0 | `unet3d_ch32_sigmoid_geom_snellius_fold0_best.pt` | 249 MB | `245cf8a27f3298a0d8c5cd0c83ed63feaa135342177bc7773be08cf0b7134c09` |
| 1 | `unet3d_ch32_sigmoid_geom_snellius_fold1_best.pt` | 249 MB | `90f75b2358637e9d3d936ef9a285ff64d7fae07a73c274f0c33f430706c3f630` |
| 2 | `unet3d_ch32_sigmoid_geom_snellius_fold2_best.pt` | 249 MB | `fd91df731b58bf2ea8766260a4f5d55c526eefecb07a584783259bc378ebb5b5` |
| 3 | `unet3d_ch32_sigmoid_geom_snellius_fold3_best.pt` | 249 MB | `70c7fbdedce74daebd2090e57c3a257866b5bd7d8bc311dc0110aa702013d7d7` |
| 4 | `unet3d_ch32_sigmoid_geom_snellius_fold4_best.pt` | 249 MB | `616d3cd5c5311ee052932ce43a9904d80ff610527f69f9a418d2b98b05345855` |

---

## Ablation checkpoints (not used in final comparison)

| File | Size | SHA-256 |
|---|---|---|
| `dosegan_ngf32_sigmoid_bce_snellius_fold0_best.pt` | 370 MB | `f4c5b4187a099d908e800ab2d154f97667cb1235e784cf4b6d8da50759f4d598` |
| `dosegan_ngf32_sigmoid_grad1.0_snellius_fold0_best.pt` | 370 MB | `0b5198883be0ac672cf919b1d22becc9a730e6bea5affb403fc503ea3281f240` |
| `unet3d_ch32_sigmoid_grad1.0_snellius_fold0_best.pt` | 249 MB | `f5755a6f67f8219e7106e4bf0b6aea27e9a0bc9a84ebad2981c1b150121b4b45` |

---

## Verification

To verify a checkpoint after copying:

```bash
sha256sum outputs/checkpoints_dosegan/dosegan_ngf32_sigmoid_snellius_fold0_best.pt
```

Compare the output against the hash in this table.

---

## Notes

- Hashes recorded **before** fold-0 retrain (2026-06-04). Fold-0 hashes will
  change after the retrain jobs complete — update this table then.
- All checkpoints use Sigmoid output activation, InstanceNorm3d(affine=True),
  LSGAN loss. See `configs/` for full hyperparameter details.
- From training run `bb28ce4` onwards, `checkpoint_sha256` is also written
  into each run's manifest JSON at training end.
