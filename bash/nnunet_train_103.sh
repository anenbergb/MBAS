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
 

DATASET_ID=103
# nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity
# nnUNetv2_plan_and_preprocess -d $DATASET_ID -pl nnUNetPlannerResEncM --verify_dataset_integrity


# nnUNetv2_train $DATASET_ID 3d_fullres 0 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --val
# nnUNetv2_train $DATASET_ID 3d_fullres 1 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --val
# nnUNetv2_train $DATASET_ID 3d_fullres 2 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --val
# nnUNetv2_train $DATASET_ID 3d_fullres 3 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --val
# nnUNetv2_train $DATASET_ID 3d_fullres 4 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans --val

# nnUNetv2_train $DATASET_ID 3d_fullres 0 --npz -tr nnUNetTrainer_250epochs --val
# nnUNetv2_train $DATASET_ID 3d_fullres 1 --npz -tr nnUNetTrainer_250epochs --c
# nnUNetv2_train $DATASET_ID 3d_fullres 2 --npz -tr nnUNetTrainer_250epochs
# nnUNetv2_train $DATASET_ID 3d_fullres 3 --npz -tr nnUNetTrainer_250epochs
# nnUNetv2_train $DATASET_ID 3d_fullres 4 --npz -tr nnUNetTrainer_250epochs

# nnUNetv2_train $DATASET_ID 2d 0 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train $DATASET_ID 2d 1 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train $DATASET_ID 2d 2 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train $DATASET_ID 2d 3 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
# nnUNetv2_train $DATASET_ID 2d 4 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans

nnUNetv2_train $DATASET_ID 3d_lowres_01 0 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
nnUNetv2_train $DATASET_ID 3d_lowres_01 1 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
nnUNetv2_train $DATASET_ID 3d_lowres_01 2 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
nnUNetv2_train $DATASET_ID 3d_lowres_01 3 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans
nnUNetv2_train $DATASET_ID 3d_lowres_01 4 --npz -tr nnUNetTrainer_250epochs -p nnUNetResEncUNetMPlans