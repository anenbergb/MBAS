#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA

export nnUNet_keep_files_open=1
export nnUNet_n_proc_DA=4
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"


# nnUNetv2_plan_experiment -d 101 -pl nnUNetPlannerResEncM
# nnUNetv2_plan_experiment -d 101 -pl nnUNetPlannerResEncL
# nnUNetv2_plan_experiment -d 101 -pl nnUNetPlannerResEncXL

nnUNetv2_plan_and_preprocess -d 101 -pl nnUNetPlannerResEncXL
