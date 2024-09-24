#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA

python mbas/tasks/metrics_table.py \
--root-results-dirs \
/home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS \
/home/bryan/expr/nnUNet_results/Dataset101_MBAS \
--results-dir-match crossval_results_folds_0_1_2_3_4/postprocessed \
--save /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_5fold.pkl \
--cache /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_5fold.pkl \
&> /home/bryan/gdrive/Radiology-Research/2024-MBAS/metrics_table_5fold.txt



# for i in {1..4}; do
#     python mbas/tasks/metrics_table.py \
#     --root-results-dirs \
#     /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS \
#     /home/bryan/expr/nnUNet_results/Dataset101_MBAS \
#     --results-dir-match crossval_results_folds_${i}/postprocessed \
#     --save /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_${i}fold.pkl \
#     --cache /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/metrics_table_${i}fold.pkl \
#     &> /home/bryan/gdrive/Radiology-Research/2024-MBAS/metrics_table_${i}fold.txt
# done