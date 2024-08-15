#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=1
CONDA=mbas
conda activate $CONDA

export nnUNet_keep_files_open=1
export nnUNet_n_proc_DA=4
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"

PLANS=nnUNetResEncUNetMPlans_2024_08_10
TRAINER=mbasTrainer
MODEL=3d_lowres

# nnUNetv2_train 104 $MODEL 3 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 3 --disable_ensembling

# nnUNetv2_train 104 $MODEL 4 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 4 --disable_ensembling

nnUNetv2_train 104 $MODEL all -tr $TRAINER -p $PLANS