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

# python mbas/tasks/dataset_prepare_binary_labels.py \
# --train-dir /home/bryan/data/MBAS/Training

# nnUNetv2_plan_and_preprocess -d 104 -pl nnUNetPlannerResEncM --verify_dataset_integrity
# nnUNetv2_plan_and_preprocess -d 104 -pl nnUNetPlannerResEncL --verify_dataset_integrity
# nnUNetv2_preprocess -d 104 -plans_name nnUNetResEncUNetMPlans -c 3d_lowres_1.0 3d_lowres_1.25 3d_lowres_1.5 -np 8


MODEL=3d_lowres
MODEL=lowres1.0_M_16_256
MODEL=lowres1.25_M_16_256
# MODEL=lowres1.0_M_16_256_nblocks3
# MODEL=3d_fullres
# MODEL=fullres_M_16_256_nblocks3
# MODEL=fullres_M_32_256_nblocks3
PLANS=nnUNetResEncUNetMPlans_2024_08_10
TRAINER=mbasTrainer
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS
# nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling