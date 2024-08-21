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

python mbas/tasks/per_subject_metrics.py \
--results-dir \
/home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres/crossval_results_folds_0_1_2_3_4/postprocessed \
--label-key binary_label