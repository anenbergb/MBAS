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

nnUNetv2_train 101 slim_128_oversample_05_alpha05_warm250_max050 0 -tr nnUNetTrainer_MedNeXt_CE_DC_HD -p MedNeXtPlans_2024_08_03 --c
nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_08_03 -c slim_128_oversample_05_alpha05_warm250_max050 -tr nnUNetTrainer_MedNeXt_CE_DC_HD -f 0 --no_overwrite --disable_ensembling

nnUNetv2_train 101 slim_128_oversample_05_alpha05_warm250_max050_scaled 0 -tr nnUNetTrainer_MedNeXt_CE_DC_HD -p MedNeXtPlans_2024_08_03
nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_08_03 -c slim_128_oversample_05_alpha05_warm250_max050_scaled -tr nnUNetTrainer_MedNeXt_CE_DC_HD -f 0 --no_overwrite --disable_ensembling