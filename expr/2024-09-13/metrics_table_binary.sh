#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA


ROOT=/home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS

# python mbas/tasks/metrics_table.py \
# --root-results-dirs \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_16_256 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_40_256_nblocks3 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_16_256_nblocks3 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.25_M_16_256_nblocks3 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_fullres \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.25_M_16_256 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.5_M_16_256_nblocks3 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.5_M_16_256 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_256_nblocks3 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_256 \
# --results-dir-match crossval_results_folds_0/postprocessed \
# --cache /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_binary_table0.pkl \
# --save /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_binary_table0.pkl \
# --label-key binary_label \
# --metrics-to-rank OVERLAP \
# &> /home/bryan/gdrive/Radiology-Research/2024-MBAS/paper/metrics_table_binary_RANK_OVERLAP.txt

# python mbas/tasks/metrics_table.py \
# --root-results-dirs \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_16_256 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_40_256_nblocks3 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_16_256_nblocks3 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.25_M_16_256_nblocks3 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_fullres \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.25_M_16_256 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.5_M_16_256_nblocks3 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.5_M_16_256 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_256_nblocks3 \
# ${ROOT}/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_256 \
# --results-dir-match crossval_results_folds_0/postprocessed \
# --cache /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_binary_table0.pkl \
# --save /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_binary_table0.pkl \
# --label-key binary_label \
# --metrics-to-rank DSC OVERLAP \
# &> /home/bryan/gdrive/Radiology-Research/2024-MBAS/paper/metrics_table_binary_RANK_OVERLAP_DSC.txt

# python mbas/tasks/metrics_table.py \
# --root-results-dirs \
# ${ROOT}/mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96 \
# ${ROOT}/mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96_foreground0 \
# ${ROOT}/mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96_foreground_every_other \
# --results-dir-match crossval_results_folds_0/postprocessed \
# --cache /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_binary_foreground_96.pkl \
# --save /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_binary_foreground_96.pkl \
# --label-key binary_label \
# --metrics-to-rank OVERLAP HD \
# &> /home/bryan/gdrive/Radiology-Research/2024-MBAS/paper/metrics_table_binary_foreground_96.txt

python mbas/tasks/metrics_table.py \
--root-results-dirs \
${ROOT}/mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_foreground25 \
${ROOT}/mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres \
${ROOT}/mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_foreground100 \
--results-dir-match crossval_results_folds_0/postprocessed \
--cache /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_binary_foreground.pkl \
--save /home/bryan/expr/mbas_nnUNet_results/paper/metrics_table_binary_foreground.pkl \
--label-key binary_label \
--metrics-to-rank OVERLAP HD \
&> /home/bryan/gdrive/Radiology-Research/2024-MBAS/paper/metrics_table_binary_foreground.txt