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

# nnUNetv2_extract_fingerprint -d 101
# nnUNetv2_plan_experiment -d 101 -pl nnUNetPlannerResEncM -gpu_memory_target 24
# nnUNetv2_plan_experiment -d 101 -pl MedNeXtPlanner -gpu_memory_target 24
# nnUNetv2_preprocess -d 101 -pl MedNeXtPlans -c 3d_fullres -np 32
# nnUNetv2_plan_and_preprocess -d 101 -pl MedNeXtPlanner -gpu_memory_target 24

# nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans

# nnUNetv2_find_best_configuration 101 -p MedNeXtPlans -c 3d_fullres \
# -tr nnUNetTrainer_MedNeXt -f 0 \
# --no_overwrite --disable_ensembling

# Experiment #2
# nnUNetv2_plan_and_preprocess -d 101 -pl MedNeXtPlanner -gpu_memory_target 24 -overwrite_plans_name MedNeXtPlans_exp01_shallow
# nnUNetv2_plan_experiment -d 101 -pl MedNeXtPlanner -gpu_memory_target 24 -overwrite_plans_name MedNeXtPlans_exp01_shallow
# nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans_exp01_shallow
# nnUNetv2_find_best_configuration 101 -p MedNeXtPlans_exp01_shallow -c 3d_fullres \
# -tr nnUNetTrainer_MedNeXt -f 0 \
# --no_overwrite --disable_ensembling


# Experiment # 3 (variable kernel size)
nnUNetv2_plan_and_preprocess -d 101 -pl MedNeXtPlanner -gpu_memory_target 24
# nnUNetv2_plan_experiment -d 101 -pl MedNeXtPlanner -gpu_memory_target 24
nnUNetv2_train 101 3d_fullres all -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans


# nnUNetv2_predict \
# -i "${nnUNet_raw}/Dataset101_MBAS/imagesTs" \
# -o "${nnUNet_results}/Dataset101_MBAS/nnUNetTrainer_MedNeXt__MedNeXtPlans__3d_fullres/fold_all/validation_71_100" \
# -d 101 -c 3d_fullres -f all -tr nnUNetTrainer_MedNeXt -p MedNeXtPlans \
# --verbose -npp 16 -nps 16