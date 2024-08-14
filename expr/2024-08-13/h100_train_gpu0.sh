#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export nnUNet_keep_files_open=1
export nnUNet_n_proc_DA=4
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
# export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
# export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"
export nnUNet_preprocessed="/home/ubuntu/storage/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/ubuntu/storage/mbas_nnUNet_results"

python mbas/tasks/prepare_cascade_model.py \
--plans $nnUNet_preprocessed/Dataset101_MBAS/MedNeXtV2Plans_2024_08_13.json \
--results-dir $nnUNet_results/Dataset101_MBAS \
--prev-stage-dir $nnUNet_results/Dataset104_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres \
--dataset-preprocess-dir $nnUNet_preprocessed/Dataset101_MBAS/nnUNetPlans_3d_fullres \
--save-dir master_predicted_next_stage \
--trainer mbasTrainer \
--use-crossval-postprocessed

TRAINER=mbasTrainer
PLANS=MedNeXtV2Plans_2024_08_13
MODEL=16_256_nblocks2_cascade_3d_low_res

nnUNetv2_train 101 $MODEL all -tr $TRAINER -p $PLANS
nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS