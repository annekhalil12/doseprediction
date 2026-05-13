#!/bin/bash
#SBATCH --job-name=unet3d
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/unet3d_%j.out
#SBATCH --error=outputs/logs/unet3d_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

mkdir -p outputs/logs

export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

FOLD=${FOLD:-0}
python3 training/train_unet3d.py --fold $FOLD
