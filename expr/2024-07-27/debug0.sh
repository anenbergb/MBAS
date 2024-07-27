#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA

export nnUNet_n_proc_DA=0
export nnUNet_raw="/home/bryan/data/nnUNet_raw"

export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/mbas_debug/nnUNet_results/2024-07-24"

nnUNetv2_train 101 slim_128_oversample 0 -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans_2024_07_27
