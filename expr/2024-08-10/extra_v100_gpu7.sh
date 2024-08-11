#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=7

CONDA=mbas
conda activate $CONDA

export PATH="/home/ubuntu/storage/miniconda3/bin:$PATH"
export nnUNet_keep_files_open=1
export nnUNet_n_proc_DA=4
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/ubuntu/storage/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/ubuntu/storage/mbas_nnUNet_results"


python mbas/tasks/prepare_cascade_model_from_gt.py \
--plans $nnUNet_preprocessed/Dataset101_MBAS/nnUNetResEncUNetMPlans_2024_08_10.json \
--results-dir $nnUNet_results/Dataset101_MBAS \
--dataset-preprocess-dir $nnUNet_preprocessed/Dataset101_MBAS/nnUNetPlans_3d_fullres \
--trainer mbasTrainer \
--save-ground-truth


TRAINER=mbasTrainer
PLANS=nnUNetResEncUNetMPlans_2024_08_10

MODEL=fullres_M_16_256
nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

MODEL=fullres_M_16_256_nblocks3
nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling