#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=1
CONDA=mbas
conda activate $CONDA

export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"

# python mbas/tasks/launch_fiftyone.py \
# --data-dir /home/bryan/data/MBAS \
# --dataset-name mbas_videos \
# -p $nnUNet_results/Dataset104_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres/fold_all/validation_71_100/postprocessed \
# --port 5151

# PLANS=nnUNetResEncUNetMPlans_2024_08_13
# MODEL=16_256_cascade_3d_low_res
# python mbas/tasks/launch_fiftyone.py \
# --data-dir /home/bryan/data/MBAS \
# --dataset-name mbas_videos \
# --port 5151 \
# -p $nnUNet_results/Dataset101_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res/fold_all/validation_71_100/postprocessed_only_wall \


# No postprocess
# -p $nnUNet_results/Dataset101_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res/fold_all/validation_71_100 \
# Every region
# -p $nnUNet_results/Dataset101_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res/fold_all/validation_71_100/postprocessed_every_region \



# PLANS=MedNeXtV2Plans_2024_08_13
# MODEL=16_256_nblocks2_cascade_3d_low_res
python mbas/tasks/launch_fiftyone.py \
--data-dir /home/bryan/data/MBAS \
--dataset-name mbas_videos \
--port 5151 \
-p $nnUNet_results/Dataset101_MBAS/mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res/fold_all/validation_71_100/postprocessed_only_wall



# No postprocessing
# -p $nnUNet_results/Dataset101_MBAS/mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res/fold_all/validation_71_100 \



