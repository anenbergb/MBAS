#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA

python mbas/tasks/metrics_table.py \
--root-results-dirs \
/home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS \
/home/bryan/expr/nnUNet_results/Dataset101_MBAS \
--results-dir-match crossval_results_folds_0/postprocessed \
--cache /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_06.pkl \
--save /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_07.pkl \
&> /home/bryan/gdrive/Radiology-Research/2024-MBAS/metrics_table.txt