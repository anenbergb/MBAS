#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA

# python mbas/tasks/launch_fiftyone.py \
# --data-dir /Users/bryan/data/MBAS_Dataset \
# --dataset-name mbas_videos \
# -p /Users/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05/crossval_results_folds_0 \
# --port 5151

# python mbas/tasks/launch_fiftyone.py \
# --data-dir /home/bryan/data/MBAS \
# --dataset-name mbas_videos \
# -p /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05/crossval_results_folds_0/postprocessed \
# --port 5151

# python mbas/tasks/launch_fiftyone.py \
# --data-dir /home/bryan/data/MBAS \
# --dataset-name mbas_videos \
# -p /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05/crossval_results_folds_0 \
# --port 5151


# python mbas/tasks/launch_fiftyone.py \
# --data-dir /home/bryan/data/MBAS \
# --dataset-name mbas_videos \
# -p /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/nnUNetTrainer_MedNeXt__MedNeXtV2Plans_2024_08_06__M_baseline/crossval_results_folds_0/postprocessed \
# --port 5151

# python mbas/tasks/launch_fiftyone.py \
# --data-dir /home/bryan/data/MBAS \
# --dataset-name mbas_videos \
# -p /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/nnUNetTrainer_MedNeXt__MedNeXtV2Plans_2024_08_06__M_baseline/crossval_results_folds_0 \
# --port 5151

# 1st stage segmentation
python mbas/tasks/launch_fiftyone.py \
--data-dir /home/bryan/data/MBAS \
--dataset-name mbas_videos \
-p /home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres/crossval_results_folds_0_1_2_3_4/postprocessed \
/home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres/fold_all/validation_71_100/postprocessed \
--port 5151