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


# python mbas/tasks/prepare_cascade_model.py \
# --plans $nnUNet_preprocessed/Dataset101_MBAS/plans_2024_09_10.json \
# --results-dir $nnUNet_results/Dataset101_MBAS \
# --prev-stage-dir /home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96 \
# --dataset-preprocess-dir $nnUNet_preprocessed/Dataset101_MBAS/nnUNetPlans_3d_fullres \
# --save-dir master_predicted_next_stage \
# --trainer mbasTrainer \
# --use-crossval-postprocessed

TRAINER=mbasTrainer
PLANS=plans_2024_09_10

# MODEL=ResEncUNet_p32_384_dil2_bd_cascade_ResEncUNet_08_27
# nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

TRAINER=mbasTrainer
PLANS=plans_2024_08_30
MODEL=ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27
nnUNetv2_train 101 $MODEL 1 -tr $TRAINER -p $PLANS --c
nnUNetv2_train 101 $MODEL 2 -tr $TRAINER -p $PLANS --c
