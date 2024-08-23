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
PLANS=plans_2024_08_21
# MODEL=MedNeXtV2_3d_lowres_slim_128_foreground100
# nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

# MODEL=MedNeXtV2_3d_lowres_slim_128_foreground100_first_430epochs
# nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

# MODEL=MedNeXtV2_3d_lowres_slim_96
# nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
# nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

MODEL=MedNeXtV2_3d_lowres_slim_96_foreground_every_other
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

MODEL=MedNeXtV1_3d_lowres_slim_96_override_down1
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling