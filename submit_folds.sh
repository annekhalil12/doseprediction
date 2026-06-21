#!/bin/bash
# Submit all 5 folds for a model as a dependency chain, so they queue up
# automatically without hitting the 2-job-per-user GPU limit.
#
# Usage:
#   bash submit_folds.sh dosegan              # DoseGAN baseline folds 0-4
#   bash submit_folds.sh dosegan --geom       # DoseGAN geom folds 0-4
#   bash submit_folds.sh unet3d              # U-Net baseline folds 0-4
#   bash submit_folds.sh unet3d --geom       # U-Net geom folds 0-4
#   bash submit_folds.sh dosegan --folds 1 2 3 4        # specific folds only
#   bash submit_folds.sh dosegan --folds 1 2 3 4 --overwrite
#   bash submit_folds.sh dosegan --after 23486146       # chain after an existing job

set -euo pipefail

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
    echo "Usage: bash submit_folds.sh <dosegan|unet3d> [--geom] [--folds N ...] [--overwrite] [--after JOBID]"
    exit 1
fi
shift

GEOM=0
OVERWRITE=0
FOLDS=(0 1 2 3 4)
AFTER_JOB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --geom)     GEOM=1; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        --folds)
            shift
            FOLDS=()
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
                FOLDS+=("$1"); shift
            done
            ;;
        --after)    AFTER_JOB="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$MODEL" in
    dosegan) SCRIPT="train_dosegan.sbatch" ;;
    unet3d)  SCRIPT="train_unet3d.sbatch" ;;
    *)       echo "Unknown model: $MODEL (use dosegan or unet3d)"; exit 1 ;;
esac

# Build the --export string
build_export() {
    local fold=$1
    local exp="NONE,FOLD=${fold}"
    [[ "$GEOM"     == "1" ]] && exp="${exp},GEOM=1"
    [[ "$OVERWRITE" == "1" ]] && exp="${exp},OVERWRITE=1"
    echo "$exp"
}

PREV_JOB="$AFTER_JOB"
LABEL="${MODEL}"
[[ "$GEOM" == "1" ]] && LABEL="${LABEL} geom"

echo "Submitting ${#FOLDS[@]} fold(s) for ${LABEL}: folds ${FOLDS[*]}"
[[ -n "$PREV_JOB" ]] && echo "  (chained after job $PREV_JOB)"

for fold in "${FOLDS[@]}"; do
    EXPORT=$(build_export "$fold")

    if [[ -n "$PREV_JOB" ]]; then
        JOB_ID=$(sbatch --parsable \
            --dependency=afterok:"${PREV_JOB}" \
            --export="${EXPORT}" \
            "$SCRIPT")
        echo "  fold ${fold}: job ${JOB_ID}  (after ${PREV_JOB})"
    else
        JOB_ID=$(sbatch --parsable \
            --export="${EXPORT}" \
            "$SCRIPT")
        echo "  fold ${fold}: job ${JOB_ID}  (running immediately)"
    fi

    PREV_JOB="$JOB_ID"
done

echo "Done. Check queue with: squeue -u $USER"
