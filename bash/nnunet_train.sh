#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
# CONFIG_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONDA=nnunet
conda activate $CONDA

export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/data/nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/nnUNet_results_debug"
export nnUNet_n_proc_DA=4

# nnUNetv2_train 101 3d_fullres 0 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 3d_fullres 1 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 3d_fullres 2 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 3d_fullres 3 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 3d_fullres 4 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans

# nnUNetv2_train 101 3d_fullres 0 --npz -tr nnUNetTrainer_250epochs
# nnUNetv2_train 101 3d_fullres 1 --npz -tr nnUNetTrainer_250epochs
# nnUNetv2_train 101 3d_fullres 2 --npz -tr nnUNetTrainer_250epochs
# nnUNetv2_train 101 3d_fullres 3 --npz -tr nnUNetTrainer_250epochs
# nnUNetv2_train 101 3d_fullres 4 --npz -tr nnUNetTrainer_250epochs

# nnUNetv2_train 101 3d_fullres 0 --val -tr nnUNetTrainer_250epochs
# nnUNetv2_train 101 3d_fullres 1 --val -tr nnUNetTrainer_250epochs
# nnUNetv2_train 101 3d_fullres 2 --val -tr nnUNetTrainer_250epochs
# nnUNetv2_train 101 3d_fullres 3 --val -tr nnUNetTrainer_250epochs
# nnUNetv2_train 101 3d_fullres 4 --val -tr nnUNetTrainer_250epochs

# nnUNetv2_train 101 2d 0 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 2d 1 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 2d 2 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 2d 3 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 2d 4 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans

# nnUNetv2_train 101 3d_fullres 0 --npz -p nnUNetResEncUNetMPlans --val
# nnUNetv2_train 101 3d_fullres 1 --npz -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 3d_fullres 2 --npz -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 3d_fullres 3 --npz -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 3d_fullres 4 --npz -p nnUNetResEncUNetMPlans

nnUNetv2_train 101 3d_fullres 0 -p nnUNetResEncUNetMPlans