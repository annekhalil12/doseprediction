#!/bin/bash
#SBATCH --job-name=dosegan_preprocess
#SBATCH --partition=thin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/preprocess_%j.out
#SBATCH --error=outputs/logs/preprocess_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

mkdir -p outputs/logs

export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

python3 preprocessing/preprocess_all.py
