#!/bin/bash

# eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=1
# CONDA=mbas
# conda activate $CONDA

# python mbas/tasks/predict.py \
# --model_pth /home/bryan/expr/MBAS/final_submissions/test.pth \
# --input_dir /home/bryan/data/MBAS/Training \
# --output_dir /home/bryan/expr/MBAS/final_submissions/output/test_10

python mbas/tasks/predict.py \
--model_pth /home/bryan/expr/MBAS/final_submissions/test.pth \
--input_dir /home/bryan/data/MBAS/Validation \
--output_dir /home/bryan/expr/MBAS/final_submissions/output/test_validation_01

# python mbas/tasks/predict.py \
# --model_pth /home/bryan/expr/MBAS/final_submissions/val.pth \
# --input_dir /home/bryan/data/MBAS/Training \
# --output_dir /home/bryan/expr/MBAS/final_submissions/output/val_00

# python mbas/tasks/predict.py \
# --model_pth /home/bryan/expr/MBAS/final_submissions/val.pth \
# --input_dir /home/bryan/data/MBAS/Validation \
# --output_dir /home/bryan/expr/MBAS/final_submissions/output/val_validation_step05