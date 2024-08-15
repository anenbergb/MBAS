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

# Run inference on the validation set



# nnUNetv2_predict \
# -i "${nnUNet_raw}/Dataset104_MBAS/imagesTs" \
# -o "${nnUNet_results}/Dataset104_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres/fold_all/validation_71_100" \
# -d 104 -c $MODEL -f all -tr $TRAINER -p $PLANS \
# --verbose -npp 16 -nps 16

# nnUNetv2_apply_postprocessing \
# -i $nnUNet_results/Dataset104_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres/fold_all/validation_71_100 \
# -o $nnUNet_results/Dataset104_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres/fold_all/validation_71_100/postprocessed \
# -pp_pkl_file $nnUNet_results/Dataset104_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
# -np 16

TRAINER=mbasTrainer
PLANS=MedNeXtV2Plans_2024_08_13
MODEL=16_256_nblocks2_cascade_3d_low_res

# nnUNetv2_predict \
# -i $nnUNet_raw/Dataset101_MBAS/imagesTs \
# -o $nnUNet_results/Dataset101_MBAS/mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res/fold_all/validation_71_100 \
# -prev_stage_predictions $nnUNet_results/Dataset104_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres/fold_all/validation_71_100/postprocessed \
# -d 101 -c $MODEL -f all -tr $TRAINER -p $PLANS \
# --verbose -npp 16 -nps 16

# Then apply postprocessing

nnUNetv2_apply_postprocessing \
-i $nnUNet_results/Dataset101_MBAS/mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res/fold_all/validation_71_100 \
-o $nnUNet_results/Dataset101_MBAS/mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res/fold_all/validation_71_100/postprocessed_only_wall \
-pp_pkl_file $nnUNet_results/Dataset101_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res/fold_all/validation_71_100/postprocessing_only_wall.pkl \
-np 16