Generated: 2026-05-27 02:19
Script:    analysis/plot_results.py --out-dir outputs/analysis

Models with NEW metric CSVs (boundary MAE, isodose Dice/HD95):
  dosegan_ngf32_sigmoid_snellius  — all 5 folds (eval job 23131507)

Models with OLD metric CSVs (body MAE/RMSE + DVH only):
  unet3d_ch32_sigmoid_snellius    — eval job 23131507 still running
  *_tanh_snellius                 — old format (May 16)

Re-run with --out-dir outputs/analysis once U-Net eval finishes
to get a complete side-by-side comparison.
