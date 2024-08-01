#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
# CONFIG_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONDA=mbas
conda activate $CONDA

export nnUNet_n_proc_DA=4
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
# export nnUNet_preprocessed="/home/bryan/expr/mbas_debug/nnUNet_preprocessed"
# export nnUNet_results="/home/bryan/expr/mbas_debug/nnUNet_results"

export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"

# nnUNetv2_train 101 slim_128 0 -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans_2024_07_21 --c
# nnUNetv2_train 101 decoder_1_block 0 -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans_2024_07_21
# nnUNetv2_train 101 decoder_1_exp_ratio 0 -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans_2024_07_21 --val
