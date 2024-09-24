#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA


ROOT=/home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS

# python mbas/tasks/metrics_table.py \
# --root-results-dirs \
# ${ROOT}/mbasTrainer__plans_2024_09_20__ResEncUNet_p20_256_GT \
# ${ROOT}/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres2 \
# --results-dir-match crossval_results_folds_0/postprocessed \
# --cache /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_GT.pkl \
# --save /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_GT.pkl \
# &> /home/bryan/gdrive/Radiology-Research/2024-MBAS/paper/metrics_table_GT.txt


python mbas/tasks/metrics_table.py \
--root-results-dirs \
/home/bryan/expr/nnUNet_results/Dataset101_MBAS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4 \
/home/bryan/expr/nnUNet_results/Dataset101_MBAS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessed \
--override-model-names \
ResEncM \
ResEncM_postprocessed \
--save /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_ResEncM_postprocessed.pkl \
&> /home/bryan/gdrive/Radiology-Research/2024-MBAS/paper/metrics_table_ResEncM_postprocessed.txt

# --cache /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_ResEncM_postprocessed.pkl \
