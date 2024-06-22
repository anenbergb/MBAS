#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
# CONFIG_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONDA=nnunet
conda activate $CONDA

export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/data/nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/nnUNet_results"
export nnUNet_n_proc_DA=32
 
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 0 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 1 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 2 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 3 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 4 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz

# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 0 -tr nnUNetTrainer_250epochs --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 1 -tr nnUNetTrainer_250epochs --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 2 -tr nnUNetTrainer_250epochs --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 3 -tr nnUNetTrainer_250epochs --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 4 -tr nnUNetTrainer_250epochs --npz

# nnUNetv2_train 101 2d_cascade_fullres_from_only_atrium 0 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 2d_cascade_fullres_from_only_atrium 1 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 2d_cascade_fullres_from_only_atrium 2 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 2d_cascade_fullres_from_only_atrium 3 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 2d_cascade_fullres_from_only_atrium 4 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz

# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 0 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 1 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 2 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 3 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz
# nnUNetv2_train 101 3d_cascade_fullres_from_only_atrium 4 -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --npz