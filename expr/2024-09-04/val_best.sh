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
# MODEL=ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27
# nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --val --val_best
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling
PLANS=plans_2024_08_30
MODEL=ResEncUNet_p20_256_dil2_cascade_ResEncUNet_08_27
nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --val --val_best
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling
PLANS=plans_2024_09_02
MODEL=MedNeXtV2_p16_256_dil2_nblocks346_slim128_cascade_ResEncUNet_08_27
nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --val --val_best
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling