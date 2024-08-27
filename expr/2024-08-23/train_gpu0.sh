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


TRAINER=mbasTrainer
PLANS=plans_2024_08_23

MODEL=MedNeXtV2_3d_lowres_p32_256_slim_96
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

MODEL=MedNeXtV2_3d_lowres_p32_256_slim_96_stem5
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

PLANS=plans_2024_08_21
MODEL=MedNeXtV2_3d_lowres_slim_96_foreground0
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

# QUEUE
PLANS=plans_2024_08_24
MODEL=ResEncUNet_3d_lowres_for25
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling
MODEL=ResEncUNet_3d_lowres_for25_drop25
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling


# BACKLOG TRAINING
PLANS=plans_2024_08_21
MODEL=MedNeXtV2_3d_lowres_slim_128_foreground0
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling