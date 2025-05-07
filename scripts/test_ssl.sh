#!/bin/bash
#SBATCH --job-name=eval-toxic-model
#SBATCH --output=logs/test_output_%j.log
#SBATCH --error=logs/test_error_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rosa@keuss.net

module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

cd /projects/prjs1392/toxic-cloud-segmentation

mkdir -p logs

# Customize loss types here if needed
SUPERVISED_LOSS="cross_entropy"
CONTRASTIVE_LOSS="pixel"

python test.py \
    --supervised_loss "$SUPERVISED_LOSS" \
    --contrastive_loss "$CONTRASTIVE_LOSS"