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


TRAINER=mbasTrainer
PLANS=plans_2024_09_10

# MODEL=ResEncUNet_p32_256_dil2_bd_cascade_ResEncUNet_08_27
# nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

# PLANS=plans_2024_09_11
# MODEL=ResEncUNet_p20_256_dil2_bd_cascade_ResEncUNet_08_27_nopost
# nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling
# nnUNetv2_train 101 $MODEL all -tr $TRAINER -p $PLANS --c

TRAINER=mbasTrainer
PLANS=plans_2024_08_30
MODEL=ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27
nnUNetv2_train 101 $MODEL all -tr $TRAINER -p $PLANS --c
nnUNetv2_train 101 $MODEL 3 -tr $TRAINER -p $PLANS --c
nnUNetv2_train 101 $MODEL 4 -tr $TRAINER -p $PLANS --c