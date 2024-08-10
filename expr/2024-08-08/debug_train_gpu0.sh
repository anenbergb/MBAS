#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA

export nnUNet_keep_files_open=1
export nnUNet_n_proc_DA=0
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
# export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"
export nnUNet_results="/home/bryan/expr/mbas_debug/nnUNet_results/2024-08-08"


# python mbas/tasks/prepare_cascade_model_from_gt.py \
# --plans $nnUNet_preprocessed/Dataset101_MBAS/MedNeXtV2Plans_2024_08_08.json \
# --results-dir $nnUNet_results/Dataset101_MBAS \
# --dataset-preprocess-dir $nnUNet_preprocessed/Dataset101_MBAS/MedNeXtPlans_3d_fullres \
# --trainer nnUNetTrainer_MedNeXt \
# --save-ground-truth

MODEL=S_128
PLANS=MedNeXtV2Plans_2024_08_08
TRAINER=nnUNetTrainer_MedNeXt
nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --no_overwrite --disable_ensembling