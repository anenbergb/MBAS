#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=1
CONDA=mbas
conda activate $CONDA

python mbas/tasks/predict.py \
--model_pth /home/bryan/expr/MBAS/final_submissions/test.pth \
--output_dir /home/bryan/expr/MBAS/final_submissions/output/test_03
