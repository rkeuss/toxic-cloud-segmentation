#!/bin/bash
#SBATCH --job-name=toxic-train
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rosa@keuss.net

module load Python/3.10.4
source /projects/prjs1392/toxic-cloud-segmentation/.venv/bin/activate
cd /projects/prjs1392/toxic-cloud-segmentation
mkdir -p logs

NUM_FOLDS=6
NUM_EPOCHS=50
BATCH_SIZE=8
THRESHOLD=0.5
LEARNING_RATE=0.001
TEMPERATURE=0.7 # for contrastive loss, same default value as C3-semiseg
NEIGHBORHOOD_SIZE=5
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
    --contrastive_loss "$CONTRASTIVE_LOSS"




# ask chatgpt to:
  #
  #Help generate a .slurm script for testing interactively
  #
  #Set up job dependencies (e.g., evaluation after training)
  #
  #Optimize for multiple GPUs or MPI jobs on Snellius