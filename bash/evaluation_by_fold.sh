#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA

export nnUNet_keep_files_open=1
export nnUNet_n_proc_DA=1
export nnUNet_def_n_proc=1
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"


TRAINER=mbasTrainer
PLANS=plans_2024_08_30
MODEL=ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 1 --disable_ensembling
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 2 --disable_ensembling
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 3 --disable_ensembling
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 4 --disable_ensembling

# TRAINER=nnUNetTrainer_MedNeXt
# PLANS=plans_2024_08_30
# MODEL=ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 1 --disable_ensembling
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 2 --disable_ensembling
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 3 --disable_ensembling
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 4 --disable_ensembling

export nnUNet_results="/home/bryan/expr/nnUNet_results"
TRAINER=nnUNetTrainer
PLANS=nnUNetResEncUNetMPlans
MODEL=3d_fullres
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 1 --disable_ensembling
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 2 --disable_ensembling
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 3 --disable_ensembling
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 4 --disable_ensembling