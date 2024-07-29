#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
# CONFIG_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONDA=nnunet
conda activate $CONDA

export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/data/nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/nnUNet_results"
export nnUNet_n_proc_DA=4

## nnUNetv2_find_best_configuration 101 -p nnUNetPlans -c 3d_cascade_fullres2 -tr nnUNetTrainer_250epochs -f 0 --no_overwrite --disable_ensembling

# nnUNetv2_find_best_configuration 101 -p nnUNetPlans -c 3d_cascade_fullres_from_only_atrium -tr nnUNetTrainer_250epochs -f 0 --no_overwrite --disable_ensembling
# nnUNetv2_find_best_configuration 101 -p nnUNetPlans -c 3d_fullres -tr nnUNetTrainer_250epochs -f 0 --no_overwrite --disable_ensembling
# nnUNetv2_find_best_configuration 101 -p nnUNetResEncUNetMPlans -c 2d -tr nnUNetTrainer_250epochs -f 0 --no_overwrite --disable_ensembling
# nnUNetv2_find_best_configuration 101 -p nnUNetResEncUNetMPlans -c 2d_cascade_fullres_from_only_atrium -tr nnUNetTrainer_250epochs -f 0 --no_overwrite --disable_ensembling

# nnUNetv2_find_best_configuration 101 -p nnUNetResEncUNetMPlans -c 3d_cascade_fullres_from_only_atrium -tr nnUNetTrainer_250epochs -f 0 --no_overwrite --disable_ensembling
## Had to copy
## nnUNetv2_find_best_configuration 101 -p nnUNetResEncUNetMPlans -c 3d_cascade_fullres_from_only_atrium_orig01 -tr nnUNetTrainer_250epochs -f 0 --no_overwrite --disable_ensembling


# nnUNetv2_find_best_configuration 101 -p nnUNetResEncUNetMPlans -c 3d_fullres -tr nnUNetTrainer_250epochs -f 0 --no_overwrite --disable_ensembling


## nnUNetv2_find_best_configuration 101 -p nnUNetResEncUNetLPlans -c 3d_fullres -tr nnUNetTrainer -f 0 --no_overwrite --disable_ensembling
# nnUNetv2_find_best_configuration 101 -p nnUNetResEncUNetMPlans -c 3d_fullres -tr nnUNetTrainer -f 0 --no_overwrite --disable_ensembling


## nnUNetv2_find_best_configuration 101 -p nnUNetResEncUNetMPlans -c 3d_lowres -tr nnUNetTrainer -f 0 --no_overwrite --disable_ensembling