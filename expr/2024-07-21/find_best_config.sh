#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
# CONFIG_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONDA=mbas
conda activate $CONDA

export nnUNet_n_proc_DA=4
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
# export nnUNet_preprocessed="/home/bryan/expr/mbas_debug/nnUNet_preprocessed"
# export nnUNet_results="/home/bryan/expr/mbas_debug/nnUNet_results"

export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"

# nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_07_18 -c 3d_03 -tr nnUNetTrainer_MedNeXt -f 0 --no_overwrite --disable_ensembling
# nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_07_18 -c 3d_04 -tr nnUNetTrainer_MedNeXt -f 0 --no_overwrite --disable_ensembling

# nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_07_21 -c slim_128 -tr nnUNetTrainer_MedNeXt -f 0 --no_overwrite --disable_ensembling
# nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_07_21 -c decoder_1_block -tr nnUNetTrainer_MedNeXt -f 0 --no_overwrite --disable_ensembling

# nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_2024_07_21 -c decoder_1_exp_ratio -tr nnUNetTrainer_MedNeXt -f 0 --no_overwrite --disable_ensembling

MODEL_DIR=/home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__decoder_1_exp_ratio
python mbas/tasks/per_subject_metrics.py \
--results $MODEL_DIR/fold_0/validation \
--save $MODEL_DIR/fold_0/validation_per_subject_metrics.pickle

# /home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS/nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__decoder_1_exp_ratio/crossval_results_folds_0/postprocessed
