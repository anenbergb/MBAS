#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=1
CONDA=mbas
conda activate $CONDA

export nnUNet_keep_files_open=1
export nnUNet_n_proc_DA=4
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"

# Run inference on the validation set

PLANS=nnUNetResEncUNetMPlans_2024_08_10
TRAINER=mbasTrainer
MODEL=3d_lowres

# run full cross-validation
# nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 1 2 3 4 --disable_ensembling

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

# Just use the input_folder/plans.json and input_folder/dataset.json
# -plans_json $nnUNet_preprocessed/Dataset104_MBAS/nnUNetResEncUNetMPlans_2024_08_10.json \
# -dataset_json $nnUNet_raw/Dataset104_MBAS/dataset.json \

TRAINER=mbasTrainer
PLANS=nnUNetResEncUNetMPlans_2024_08_13
MODEL=16_256_cascade_3d_low_res

# nnUNetv2_predict \
# -i $nnUNet_raw/Dataset101_MBAS/imagesTs \
# -o $nnUNet_results/Dataset101_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res/fold_all/validation_71_100 \
# -prev_stage_predictions $nnUNet_results/Dataset104_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres/fold_all/validation_71_100/postprocessed \
# -d 101 -c $MODEL -f all -tr $TRAINER -p $PLANS \
# --verbose -npp 16 -nps 16

# Then apply postprocessing (if necessary)

# nnUNetv2_apply_postprocessing \
# -i $nnUNet_results/Dataset101_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res/fold_all/validation_71_100 \
# -o $nnUNet_results/Dataset101_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res/fold_all/validation_71_100/postprocessed_every_region \
# -pp_pkl_file $nnUNet_results/Dataset101_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res/fold_all/validation_71_100/postprocessing_every_region.pkl \
# -np 16

nnUNetv2_apply_postprocessing \
-i $nnUNet_results/Dataset101_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res/fold_all/validation_71_100 \
-o $nnUNet_results/Dataset101_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res/fold_all/validation_71_100/postprocessed_only_wall \
-pp_pkl_file $nnUNet_results/Dataset101_MBAS/mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res/fold_all/validation_71_100/postprocessing_only_wall.pkl \
-np 16
