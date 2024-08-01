#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
CONDA=mbas
conda activate $CONDA

export nnUNet_n_proc_DA=4
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"

# nnUNetv2_train 101 slim_128_alpha05 0 -tr nnUNetTrainer_MedNeXt_CE_DC_HD -p MedNeXtPlans_2024_07_26

# nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_07_26 -c slim_128_alpha05 -tr nnUNetTrainer_MedNeXt_CE_DC_HD -f 0 --no_overwrite --disable_ensembling
# nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_07_26 -c slim_128_alpha10 -tr nnUNetTrainer_MedNeXt_CE_DC_HD -f 0 --no_overwrite --disable_ensembling