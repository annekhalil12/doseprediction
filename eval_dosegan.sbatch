#!/bin/bash
#SBATCH --job-name=dosegan_eval
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/logs/dosegan_eval_%j.out
#SBATCH --error=outputs/logs/dosegan_eval_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

mkdir -p outputs/logs outputs/evaluation

export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

for FOLD in 0 1 2 3 4; do
    echo "==================== Fold $FOLD ===================="
    python3 -m training.evaluate_dosegan --fold $FOLD
done
