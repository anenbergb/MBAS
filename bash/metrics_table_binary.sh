#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA

python mbas/tasks/metrics_table.py \
--root-results-dirs \
/home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS \
--results-dir-match crossval_results_folds_0/postprocessed \
--cache /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_binary_03.pkl \
--save /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_binary_03.pkl \
--label-key binary_label \
--metrics-to-rank HD DSC \
&> /home/bryan/gdrive/Radiology-Research/2024-MBAS/metrics_table_binary.txt

python mbas/tasks/metrics_table.py \
--root-results-dirs \
/home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS \
--results-dir-match crossval_results_folds_0/postprocessed \
--cache /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_binary_03.pkl \
--save /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_binary_03.pkl \
--label-key binary_label \
--metrics-to-rank HD OVERLAP \
&> /home/bryan/gdrive/Radiology-Research/2024-MBAS/metrics_table_binary_RANK_OVERLAP_HD.txt

python mbas/tasks/metrics_table.py \
--root-results-dirs \
/home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS \
--results-dir-match crossval_results_folds_0/postprocessed \
--cache /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_binary_03.pkl \
--save /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_binary_03.pkl \
--label-key binary_label \
--metrics-to-rank OVERLAP \
&> /home/bryan/gdrive/Radiology-Research/2024-MBAS/metrics_table_binary_RANK_OVERLAP.txt