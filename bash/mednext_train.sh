#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
# CONFIG_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONDA=mednext
conda activate $CONDA


export nnUNet_raw_data_base="/home/bryan/data/nnUNet_raw_data_base"
# export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/data/mednext_preprocessed"
export nnUNet_results="/home/bryan/expr/mednext_results"
export RESULTS_FOLDER=$nnUNet_results


export nnUNet_n_proc_DA=32
 
# mednextv1_plan_and_preprocess -t 501 -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1_patch20x256x256 --verify_dataset_integrity

# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel5 Task501_MBAS 0 -p nnUNetPlansv2.1_trgSp_1x1x1
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel5 Task501_MBAS 0 -p nnUNetPlansv2.1_trgSp_1x1x1

mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel5 Task501_MBAS 1 -p nnUNetPlansv2.1_trgSp_1x1x1


# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_L_kernel5 Task501_MBAS 0 -p nnUNetPlansv2.1_trgSp_1x1x1_patch20x256x256

# nnUNetTrainerV2_MedNeXt_M_kernel5

# nnUNetv2_train 101 3d_fullres 1 --npz -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 3d_fullres 2 --npz -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 3d_fullres 3 --npz -p nnUNetResEncUNetMPlans
# nnUNetv2_train 101 3d_fullres 4 --npz -p nnUNetResEncUNetMPlans
