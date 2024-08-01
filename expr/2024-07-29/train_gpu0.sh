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


# nnUNetv2_train 101 slim_128_patch96 0 -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans_2024_07_29 --val
# nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_07_29 -c slim_128_patch96 -tr nnUNetTrainer_MedNeXt -f 0 --no_overwrite --disable_ensembling


# nnUNetv2_train 101 slim_128_patch96_oversample025 0 -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans_2024_07_29
# nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_07_29 -c slim_128_patch96_oversample025 -tr nnUNetTrainer_MedNeXt -f 0 --no_overwrite --disable_ensembling


nnUNetv2_train 101 even_128_patch96_oversample025 0 -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans_2024_07_29 --c
nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_07_29 -c even_128_patch96_oversample025 -tr nnUNetTrainer_MedNeXt -f 0 --no_overwrite --disable_ensembling

nnUNetv2_train 101 slim_128_patch128_oversample025 0 -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans_2024_07_29
nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_07_29 -c slim_128_patch128_oversample025 -tr nnUNetTrainer_MedNeXt -f 0 --no_overwrite --disable_ensembling