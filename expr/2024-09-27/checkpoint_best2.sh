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


TRAINER=mbasTrainer
PLANS=plans_2024_08_30
MODEL=ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27

for FOLD in 3 4; do
    nnUNetv2_predict \
    -i $nnUNet_raw/Dataset101_MBAS/imagesTr/val_folds_$FOLD \
    -o "${nnUNet_results}/Dataset101_MBAS/mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27/crossval_results_folds_0_1_2_3_4_best" \
    -prev_stage_predictions $nnUNet_results/Dataset104_MBAS/mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96/crossval_results_folds_0_1_2_3_4/postprocessed \
    -d 101 -c $MODEL -f $FOLD -tr $TRAINER -p $PLANS \
    --verbose -npp 16 -nps 16 -chk checkpoint_best.pth
done