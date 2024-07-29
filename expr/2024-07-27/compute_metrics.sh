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


# RESULTS_DIR=/home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/nnUNetTrainer_MedNeXt_CE_DC_HD__MedNeXtPlans_2024_07_26__slim_128_alpha05/crossval_results_folds_0/postprocessed

# python mbas/tasks/per_subject_metrics.py \
# --results $RESULTS_DIR \
# --save $RESULTS_DIR/per_subject_metrics.pickle

python mbas/tasks/metrics_table.py \
--root-results-dirs \
/home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS \
/home/bryan/expr/nnUNet_results/Dataset101_MBAS \
--results-dir-match crossval_results_folds_0/postprocessed \
--cache /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_01.pkl \
--save /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_01.pkl