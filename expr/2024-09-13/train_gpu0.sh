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


# TRAINER=mbasTrainer
# PLANS=plans_2024_09_18
# MODEL=ResEncUNet_p20_256_dil2_bd
# nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling
# nnUNetv2_train 101 $MODEL 1 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 1 --disable_ensembling

python mbas/tasks/prepare_cascade_model_from_gt.py \
--plans $nnUNet_preprocessed/Dataset101_MBAS/plans_2024_09_20.json \
--results-dir $nnUNet_results/Dataset101_MBAS \
--dataset-preprocess-dir $nnUNet_preprocessed/Dataset101_MBAS/nnUNetPlans_3d_fullres \
--trainer mbasTrainer \
--save-ground-truth

# TRAINER=mbasTrainer
# PLANS=plans_2024_09_20
# MODEL=ResEncUNet_p20_256_GT
# nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

TRAINER=mbasTrainer
PLANS=plans_2024_09_20
MODEL=ResEncUNet_p20_256_dil1_GT
nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling