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
FORCE=${FORCE:-0}

if   [ "$GEOM" = "1" ]; then GEOM_ARG="--geom"
elif [ "$GEOM" = "0" ]; then GEOM_ARG="--no-geom"
else GEOM_ARG=""
fi

RUN_NAME_ARG=""
if [ -n "$RUN_NAME" ]; then RUN_NAME_ARG="--run-name $RUN_NAME"; fi

FORCE_ARG=""
if [ "$FORCE" = "1" ]; then FORCE_ARG="--force"; fi

echo "MODEL=$MODEL FOLD=$FOLD RUN_NAME=$RUN_NAME GEOM=$GEOM FORCE=$FORCE"

python3 -m training.evaluate \
    --model "$MODEL" \
    --fold "$FOLD" \
    $RUN_NAME_ARG \
    $GEOM_ARG \
    $FORCE_ARG \
    --skip-gamma
