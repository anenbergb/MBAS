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


# too big
# nnUNetv2_train 101 3d_fullres 0 -p nnUNetResEncUNetXLPlans
# nnUNetv2_find_best_configuration 101 -p nnUNetResEncUNetXLPlans -c 3d_fullres -f 0 --no_overwrite --disable_ensembling

nnUNetv2_train 101 3d_fullres 0 -p nnUNetResEncUNetLPlans
nnUNetv2_find_best_configuration 101 -p nnUNetResEncUNetLPlans -c 3d_fullres -f 0 --no_overwrite --disable_ensembling