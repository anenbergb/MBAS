#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=1
CONDA=mbas
conda activate $CONDA

export nnUNet_keep_files_open=1
export nnUNet_n_proc_DA=1
export nnUNet_raw="/home/bryan/data/nnUNet_raw"
export nnUNet_preprocessed="/home/bryan/data/mbas_nnUNet_preprocessed"
export nnUNet_results="/home/bryan/expr/mbas_nnUNet_results"


TRAINER=mbasTrainer
PLANS=plans_2024_08_27

MODEL=MedNeXtV2_3d_lowres_for25_drop50
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

MODEL=MedNeXtV2_3d_lowres_for25_drop50_stemStacked_decoderConvTrans
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

MODEL=MedNeXtV2_3d_lowres_for25_drop50_decoderConvTrans
nnUNetv2_train 104 $MODEL 0 -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f 0 --disable_ensembling

MODEL=ResEncUNet_3d_lowres_for25_drop50_slim96
nnUNetv2_train 104 $MODEL 3 -tr $TRAINER -p $PLANS --c
nnUNetv2_train 104 $MODEL 4 -tr $TRAINER -p $PLANS --c
nnUNetv2_train 104 $MODEL all -tr $TRAINER -p $PLANS --c
nnUNetv2_find_best_configuration 104 -c $MODEL -tr $TRAINER -p $PLANS -f all --disable_ensembling