#!/bin/bash
#SBATCH --job-name=figures
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/figures_%j.out
#SBATCH --error=outputs/logs/figures_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction
mkdir -p outputs/logs results/figures
export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

# Pure-matplotlib figures (no GPU needed)
MPLBACKEND=Agg python3 analysis/fig_input_ablation.py
MPLBACKEND=Agg python3 analysis/fig_architecture.py

# Inference-based figures (load fold-0 baseline checkpoints)
MPLBACKEND=Agg python3 -m analysis.fig_dose_slices --fold 0
MPLBACKEND=Agg python3 -m analysis.fig_dvh_representative --fold 0

echo "Done."
