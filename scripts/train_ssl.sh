#!/bin/bash
pip install -r requirements.txt

# Default values for configurable parameters
NUM_FOLDS=6
NUM_EPOCHS=50
BATCH_SIZE=8
THRESHOLD=0.5
learning_rate=0.001
TEMPERATURE=0.1
NEIGHBORHOOD_SIZE=5
WEIGHT_PIXEL=1.0
WEIGHT_LOCAL=1.0
WEIGHT_DIRECTIONAL=1.0
SUPERVISED_LOSS="cross_entropy"
CONTRASTIVE_LOSS="pixel"

# Parse command-line arguments
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

# Run the training script with the specified parameters
python train.py \
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
    --contrastive_loss "$CONTRASTIVE_LOSS" \
