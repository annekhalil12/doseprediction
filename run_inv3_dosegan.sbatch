#!/bin/bash
#SBATCH --job-name=inv3_dosegan
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=outputs/logs/inv3_dosegan_%j.out
#SBATCH --error=outputs/logs/inv3_dosegan_%j.err

cd /gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

mkdir -p outputs/logs outputs/analysis

export PYTHONPATH=/gpfs/scratch1/shared/akhalil/data/thesis-doseprediction

python3 -m analysis.inv3_dose_smearing_index --model dosegan
