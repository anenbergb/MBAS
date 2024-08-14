#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export nnUNet_keep_files_open=1
export nnUNet_n_proc_DA=4
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
# export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
# export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"
export nnUNet_preprocessed="/home/ubuntu/storage/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/ubuntu/storage/mbas_nnUNet_results"

python mbas/tasks/prepare_cascade_model_from_gt.py \
--plans $nnUNet_preprocessed/Dataset101_MBAS/MedNeXtV2Plans_2024_08_13_GT.json \
--results-dir $nnUNet_results/Dataset101_MBAS \
--dataset-preprocess-dir $nnUNet_preprocessed/Dataset101_MBAS/nnUNetPlans_3d_fullres \
--trainer mbasTrainer \
--save-ground-truth

TRAINER=mbasTrainer
PLANS=MedNeXtV2Plans_2024_08_13_GT
MODEL=16_256_cascade_GT

nnUNetv2_train 101 $MODEL all -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f all --disable_ensembling
