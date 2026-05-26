#!/bin/bash
#SBATCH --job-name=eval_baseline_sigmoid
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/logs/eval_baseline_sigmoid_%j.out
#SBATCH --error=outputs/logs/eval_baseline_sigmoid_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction
mkdir -p outputs/logs outputs/evaluation
export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

echo "============================================================"
echo "Re-evaluating baseline Sigmoid checkpoints with full metrics"
echo "  (gamma pass rate, boundary MAE, isodose Dice/HD95)"
echo "============================================================"

echo ""
echo "=== DoseGAN Sigmoid baseline (dosegan_ngf32_sigmoid_snellius) ==="
for FOLD in 0 1 2 3 4; do
    echo "--- Fold $FOLD ---"
    python3 -m training.evaluate_dosegan \
        --fold $FOLD \
        --run-name dosegan_ngf32_sigmoid_snellius
done

echo ""
echo "=== U-Net Sigmoid baseline (unet3d_ch32_sigmoid_snellius) ==="
for FOLD in 0 1 2 3 4; do
    echo "--- Fold $FOLD ---"
    python3 -m training.evaluate_unet3d \
        --fold $FOLD \
        --run-name unet3d_ch32_sigmoid_snellius
done

echo ""
echo "All evaluation complete."
