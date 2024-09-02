In this document I will evaluate the quality of the binary mask produced by the 1st Stage Cascaded model.
In particular, I will consider the results from `mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96`
(Results directory: `/home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96/crossval_results_folds_0_1_2_3_4/postprocessed`)

# Evaluate the quality of ball dilation "binary_dilation_transform"

### Applied ball dilation to the results to generate buffered results
```
python mbas/tasks/add_dilation_to_binary_mask.py \
--input-dir /home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96/crossval_results_folds_0_1_2_3_4/postprocessed \
--output-dir /home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96/crossval_results_folds_0_1_2_3_4/postprocessed_ball_dilation_1 \
--dilation-radius 1
```
### Visualized those results in Voxel51

### Average metrics with and without ball dilation
```
python mbas/tasks/per_subject_metrics.py \
--results-dir /home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96/crossval_results_folds_0_1_2_3_4/postprocessed \
--label-key binary_label
2024-09-02 01:39:33.617 | INFO     | __main__:compute_per_subject_metrics:77 - Computed per subject metrics for 70 / 100 subjects
```
Default (without buffer)
|                |   Average |       STD |
|----------------|-----------|-----------|
| DSC_atrium     |  0.916563 | 0.020782  |
| HD95_atrium    |  4.7163   | 1.80963   |
| OVERLAP_atrium |  0.960739 | 0.0225544 |

With ball dilation = 1
|                |   Average |       STD |
|----------------|-----------|-----------|
| DSC_atrium     |  0.874342 | 0.0258963 |
| HD95_atrium    |  5.41561  | 1.78958   |
| OVERLAP_atrium |  0.980234 | 0.015214  |

With ball dilation = 2
|                |   Average |       STD |
|----------------|-----------|-----------|
| DSC_atrium     |  0.825065 | 0.029752  |
| HD95_atrium    |  6.56993  | 1.86316   |
| OVERLAP_atrium |  0.987674 | 0.0115984 |

Notice that the OVERLAP metric is increased

### Average metrics with and without ball dilation measured per class
```
python mbas/tasks/per_subject_metrics.py \
--results-dir /home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96/crossval_results_folds_0_1_2_3_4/postprocessed \
--use-binary-label-as-all-labels
```
Default (without buffer)
|               |   Average |       STD |
|---------------|-----------|-----------|
| DSC_wall      |  0.326928 | 0.0345789 |
| DSC_right     |  0.486089 | 0.0575756 |
| DSC_left      |  0.515753 | 0.053467  |
| HD95_wall     |  9.03531  | 2.16618   |
| HD95_right    | 64.6803   | 9.70962   |
| HD95_left     | 75.7127   | 9.76936   |
| OVERLAP_wall  |  **0.909526** | 0.0440308 |
| OVERLAP_right |  **0.97558**  | 0.0252352 |
| OVERLAP_left  |  **0.979436** | 0.0200462 |

With ball dilation = 1
|               |   Average |       STD |
|---------------|-----------|-----------|
| DSC_wall      |  0.311584 | 0.0328636 |
| DSC_right     |  0.446951 | 0.0570421 |
| DSC_left      |  0.473651 | 0.0520459 |
| HD95_wall     |  9.42337  | 2.14055   |
| HD95_right    | 65.6506   | 9.67588   |
| HD95_left     | 76.5932   | 9.71425   |
| OVERLAP_wall  |  **0.960927** | 0.0261995 |
| OVERLAP_right |  **0.985965** | 0.0191672 |
| OVERLAP_left  |  **0.987365** | 0.0155187 |

With ball dilation = 2
|               |   Average |       STD |
|---------------|-----------|-----------|
| DSC_wall      |  0.287167 | 0.0302727 |
| DSC_right     |  0.410914 | 0.056018  |
| DSC_left      |  0.435278 | 0.0501683 |
| HD95_wall     |  9.96786  | 2.02493   |
| HD95_right    | 66.8677   | 9.67772   |
| HD95_left     | 77.8037   | 9.71133   |
| OVERLAP_wall  | ** 0.97642**  | 0.0195744 |
| OVERLAP_right | ** 0.991443** | 0.0146745 |
| OVERLAP_left  |  **0.991449** | 0.0134806 |
Notice that the maximum overlap with wall increases from 90.95% to 96.09% to 97.6% with the added ball dilation. This will be the upper bound in possible DICE score.
