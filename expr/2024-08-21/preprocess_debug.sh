#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=1
CONDA=mbas
conda activate $CONDA

export nnUNet_keep_files_open=1
export nnUNet_n_proc_DA=4
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/expr/mbas_debug/nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/mbas_debug/nnUNet_results"

# nnUNetv2_plan_and_preprocess -d 104 -pl nnUNetPlannerResEncM -c 3d_lowres --clean


nnUNetv2_preprocess -d 104 -pl nnUNetResEncUNetMPlans -c 3d_lowres -np 1


# TRAINER=mbasTrainer
# PLANS=plans_2024_08_21
# MODEL=MedNeXtV2_3d_lowres_slim_128

# nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS
# nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling