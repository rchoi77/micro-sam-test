#!/bin/bash
#SBATCH --job-name="train_microsam_model"
#SBATCH --output="train_model.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bfrj-delta-gpu
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 12:00:00
#SBATCH -e slurmlogs/slurm-%j.err
#SBATCH -o slurmlogs/slurm-%j.out

export OMP_NUM_THREADS=1  # if code is not multithreaded, otherwise set to 8 or 16

# params
IMAGES="trainingdata/glandimages/*.bmp trainingdata/images/*.tiff" # setup your own image and label directories
LABELS="trainingdata/glandlabels/*.bmp trainingdata/labels/*.tiff"
VAL_IMAGES="trainingdata/validation_images/*.bmp trainingdata/validation_images/*.tiff" # optional, will split your training data if not provided
VAL_LABELS="trainingdata/validation_labels/*.bmp trainingdata/validation_labels/*.tiff" # optional
MODEL="vit_l"
CHECKPOINT_NAME="my_sweep"
MODEL_OUTPUT_PATH="custom_models/"
WANDB_NOTE="testing wandb sweep"

# Start script
source /path/to/conda_activation_script/miniforge3/etc/profile.d/conda.sh # CHANGEME: conda activa
cd /path/to/microsam/directory # CHANGEME
conda activate microsam_gpu # CHANGEME: name of your conda env

mkdir -p slurmlogs

# train
if [[ -n "$VAL_IMAGES" && -n "$VAL_LABELS" ]]; then
    python micro_sam/training/wandb_sweep_training.py \
        --images $IMAGES --labels $LABELS \
        --val-images $VAL_IMAGES --val-labels $VAL_LABELS \
        -m $MODEL -c $CHECKPOINT_NAME \
        -o $MODEL_OUTPUT_PATH --sweep\
        -n "$WANDB_NOTE"
else
    python micro_sam/training/wandb_sweep_training.py \
        --images $IMAGES --labels $LABELS \
        -m $MODEL -c $CHECKPOINT_NAME \
        -o $MODEL_OUTPUT_PATH --sweep\
        -n "$WANDB_NOTE"
fi