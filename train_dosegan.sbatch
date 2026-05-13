#!/bin/bash
#SBATCH --job-name=dosegan_fold0
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/logs/dosegan_fold0_%j.out
#SBATCH --error=outputs/logs/dosegan_fold0_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

mkdir -p outputs/logs

export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

python3 training/train_dosegan.py
