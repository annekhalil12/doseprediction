#!/bin/bash
#SBATCH --job-name=dosegan
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/logs/dosegan_%j.out
#SBATCH --error=outputs/logs/dosegan_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

mkdir -p outputs/logs

export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

FOLD=${FOLD:-0}

OVERWRITE_ARG=""
if [ -n "${OVERWRITE:-}" ]; then OVERWRITE_ARG="--overwrite"; fi

if   [ "${GEOM:-}" = "1" ]; then GEOM_ARG="--geom"
elif [ "${GEOM:-}" = "0" ]; then GEOM_ARG="--no-geom"
else GEOM_ARG=""
fi

python3 training/train_dosegan.py --fold $FOLD $OVERWRITE_ARG $GEOM_ARG
