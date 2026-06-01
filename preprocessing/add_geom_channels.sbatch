#!/bin/bash
#SBATCH --job-name=add_geom
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/add_geom_%j.out
#SBATCH --error=outputs/logs/add_geom_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

mkdir -p outputs/logs

export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

python3 preprocessing/add_geom_channels.py
