#!/bin/bash
#SBATCH --job-name=baseline_analysis
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/logs/baseline_analysis_%j.out
#SBATCH --error=outputs/logs/baseline_analysis_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction
mkdir -p outputs/logs outputs/analysis
export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

echo "============================================================"
echo "Step 1: Comparison figures from eval CSVs"
echo "============================================================"
python3 -m analysis.plot_results

echo ""
echo "============================================================"
echo "Step 2: Dose heatmaps with structure contours"
echo "  median + worst patient, fold 0, both models"
echo "============================================================"
python3 -m analysis.plot_dose_with_masks \
    --model dosegan --fold 0 \
    --selector median worst

python3 -m analysis.plot_dose_with_masks \
    --model unet3d --fold 0 \
    --selector median worst

echo ""
echo "============================================================"
echo "Step 3: Inv2 worst-patient DVH maps (regenerate for both models)"
echo "============================================================"
python3 -m analysis.inv2_worst_patient_dvh_maps \
    --model dosegan --run-name dosegan_ngf32_sigmoid_snellius

python3 -m analysis.inv2_worst_patient_dvh_maps \
    --model unet3d  --run-name unet3d_ch32_sigmoid_snellius

echo ""
echo "All analysis complete. Figures in outputs/analysis/"
