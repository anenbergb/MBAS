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
PLANS=plans_2024_09_18
MODEL=ResEncUNet_p20_256_dil2_bd
# nnUNetv2_train 101 $MODEL 2 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 2 --disable_ensembling
# nnUNetv2_train 101 $MODEL 3 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 3 --disable_ensembling
# nnUNetv2_train 101 $MODEL 4 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 4 --disable_ensembling

# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 1 2 3 4 --disable_ensembling



TRAINER=mbasTrainer
PLANS=plans_2024_09_20
MODEL=ResEncUNet_p20_256_dil2_GT
# nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

MODEL=ResEncUNet_p20_256_dil5_GT
nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling