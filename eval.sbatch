#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

mkdir -p outputs/logs outputs/evaluation

export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

MODEL=${MODEL:-dosegan}
FOLD=${FOLD:-0}
GEOM=${GEOM:-1}
FORCE=${FORCE:-0}

if   [ "$GEOM" = "1" ]; then GEOM_ARG="--geom"
elif [ "$GEOM" = "0" ]; then GEOM_ARG="--no-geom"
else GEOM_ARG=""
fi

# Auto-derive run name from model + geom condition if not explicitly set.
# This prevents evaluate.py from using the config's default (baseline) RUN_NAME
# when loading a geom checkpoint, which causes a channel size mismatch.
if [ -z "${RUN_NAME:-}" ]; then
    if [ "$MODEL" = "dosegan" ] && [ "$GEOM" = "1" ]; then
        RUN_NAME="dosegan_ngf32_sigmoid_geom_snellius"
    elif [ "$MODEL" = "dosegan" ] && [ "$GEOM" = "0" ]; then
        RUN_NAME="dosegan_ngf32_sigmoid_snellius"
    elif [ "$MODEL" = "unet3d" ] && [ "$GEOM" = "1" ]; then
        RUN_NAME="unet3d_ch32_sigmoid_geom_snellius"
    elif [ "$MODEL" = "unet3d" ] && [ "$GEOM" = "0" ]; then
        RUN_NAME="unet3d_ch32_sigmoid_snellius"
    fi
fi

RUN_NAME_ARG=""
if [ -n "$RUN_NAME" ]; then RUN_NAME_ARG="--run-name $RUN_NAME"; fi

FORCE_ARG=""
if [ "$FORCE" = "1" ]; then FORCE_ARG="--overwrite"; fi

echo "MODEL=$MODEL FOLD=$FOLD RUN_NAME=$RUN_NAME GEOM=$GEOM FORCE=$FORCE"

python3 -m training.evaluate \
    --model "$MODEL" \
    --fold "$FOLD" \
    $RUN_NAME_ARG \
    $GEOM_ARG \
    $FORCE_ARG \
    --skip-gamma
