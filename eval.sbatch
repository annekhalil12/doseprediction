#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

mkdir -p outputs/logs outputs/evaluation

export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

MODEL=${MODEL:-dosegan}
FOLD=${FOLD:-0}
RUN_NAME=${RUN_NAME:-}
GEOM=${GEOM:-1}

GEOM_ARG=""
if [ "$GEOM" = "0" ]; then GEOM_ARG="--no-geom"; fi

RUN_NAME_ARG=""
if [ -n "$RUN_NAME" ]; then RUN_NAME_ARG="--run-name $RUN_NAME"; fi

echo "MODEL=$MODEL FOLD=$FOLD RUN_NAME=$RUN_NAME GEOM=$GEOM"

python3 -m training.evaluate \
    --model "$MODEL" \
    --fold "$FOLD" \
    $RUN_NAME_ARG \
    $GEOM_ARG \
    --skip-gamma
