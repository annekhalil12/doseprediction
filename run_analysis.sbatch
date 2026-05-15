#!/bin/bash
#SBATCH --job-name=dosegan_analysis
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=outputs/logs/dosegan_analysis_%j.out
#SBATCH --error=outputs/logs/dosegan_analysis_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

mkdir -p outputs/logs outputs/analysis

export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

echo "==================== Inv 1: acquisition breakdown ===================="
python3 -m analysis.inv1_acquisition_breakdown

echo "==================== Inv 2: worst-patient DVH + error maps ===================="
python3 -m analysis.inv2_worst_patient_dvh_maps
