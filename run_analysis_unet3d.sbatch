#!/bin/bash
#SBATCH --job-name=unet3d_analysis
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=outputs/logs/unet3d_analysis_%j.out
#SBATCH --error=outputs/logs/unet3d_analysis_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

mkdir -p outputs/logs outputs/analysis

export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

ACT_ARG=""
if [ -n "${ACT:-}" ]; then ACT_ARG="--activation ${ACT}"; fi

echo "==================== Inv 1: U-Net acquisition breakdown ===================="
python3 -m analysis.inv1_acquisition_breakdown --model unet3d $ACT_ARG

echo "==================== Inv 2: U-Net worst-patient DVH + error maps ===================="
python3 -m analysis.inv2_worst_patient_dvh_maps --model unet3d $ACT_ARG
