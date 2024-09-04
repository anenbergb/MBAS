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
# Backlog
PLANS=plans_2024_09_02
# MODEL=MedNeXtV2_p20_256_dil2_nblocks1346_cascade_ResEncUNet_08_27
# nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling
MODEL=MedNeXtV2_p20_256_dil2_nblocks1346_slim128_cascade_ResEncUNet_08_27
nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

# downsampling the 16dim to 1, which is more similar to the original MedNeXt model
MODEL=MedNeXtV2_p16_256_dil2_nblocks346_slim128_stride16to1_cascade_ResEncUNet_08_27
nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

# TODO: these are too big and will run out of CUDA memory. Decrease patch size to 16
# MODEL=MedNeXtV2_p20_256_dil2_nblocks346_cascade_ResEncUNet_08_27
# nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling
# MODEL=MedNeXtV2_p20_256_dil2_nblocks346_slim128_cascade_ResEncUNet_08_27
# nnUNetv2_train 101 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 101 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling
