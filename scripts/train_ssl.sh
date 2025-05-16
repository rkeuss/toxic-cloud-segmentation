#!/bin/bash
#SBATCH --job-name=toxic-train
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rosa@keuss.net

module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
pip install -r requirements

cd /projects/prjs1392/toxic-cloud-segmentation
mkdir -p logs

NUM_NODES=1
NUM_GPUS=4
NUM_FOLDS=6 # same default value as C3-semiseg
NUM_EPOCHS=40 # same default value as C3-semiseg
BATCH_SIZE=32 # same default value as C3-semiseg, or 8 (default of local paper)
THRESHOLD=0.5 # standard decision threshold for binary classification with sigmoid outputs. It assumes balanced class presence, which is often a simplification but works as a baseline.
LEARNING_RATE=0.00012 # same default value as C3-semiseg
TEMPERATURE=0.15 # for contrastive loss, 0.7 is used default value as C3-semiseg, 0.15 is used default value as local paper
NEIGHBORHOOD_SIZE=3 # same default value as local paper
WEIGHT_PIXEL=1.0
WEIGHT_LOCAL=1.0
WEIGHT_DIRECTIONAL=1.0
SUPERVISED_LOSS="cross_entropy"
CONTRASTIVE_LOSS="pixel"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num_folds) NUM_FOLDS="$2"; shift ;;
        --num_epochs) NUM_EPOCHS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --threshold) THRESHOLD="$2"; shift ;;
        --learning_rate) LEARNING_RATE="$2"; shift ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        --neighborhood_size) NEIGHBORHOOD_SIZE="$2"; shift ;;
        --weight_pixel) WEIGHT_PIXEL="$2"; shift ;;
        --weight_local) WEIGHT_LOCAL="$2"; shift ;;
        --weight_directional) WEIGHT_DIRECTIONAL="$2"; shift ;;
        --supervised_loss) SUPERVISED_LOSS="$2"; shift ;;
        --contrastive_loss) CONTRASTIVE_LOSS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Log run settings
echo "Running with:"
echo "Folds: $NUM_FOLDS, Epochs: $NUM_EPOCHS, Batch: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE, Temp: $TEMPERATURE"
echo "Losses: $SUPERVISED_LOSS + $CONTRASTIVE_LOSS"

# Run training script
torchrun --nproc_per_node=$NUM_GPUS --nnodes=$NUM_NODES train.py \
    --num_folds "$NUM_FOLDS" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --threshold "$THRESHOLD" \
    --learning_rate "$LEARNING_RATE" \
    --temperature "$TEMPERATURE" \
    --neighborhood_size "$NEIGHBORHOOD_SIZE" \
    --weight_pixel "$WEIGHT_PIXEL" \
    --weight_local "$WEIGHT_LOCAL" \
    --weight_directional "$WEIGHT_DIRECTIONAL" \
    --supervised_loss "$SUPERVISED_LOSS" \
    --contrastive_loss "$CONTRASTIVE_LOSS"