#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA
export PATH="/home/ubuntu/storage/miniconda3/bin:$PATH"
export nnUNet_keep_files_open=1
export nnUNet_n_proc_DA=4
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/ubuntu/storage/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/ubuntu/storage/mbas_nnUNet_results"

TRAINER=mbasTrainer
PLANS=nnUNetResEncUNetMPlans_2024_08_10

MODEL=3d_lowres
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

MODEL=3d_fullres
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling