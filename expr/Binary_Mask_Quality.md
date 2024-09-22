In this document I will evaluate the quality of the binary mask produced by the 1st Stage Cascaded model.
In particular, I will consider the results from `mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96`
(Results directory: `/home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96/crossval_results_folds_0_1_2_3_4/postprocessed`)

# Compare with and without postprocessing
```
python mbas/tasks/per_subject_metrics.py \
--results-dir /home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96/crossval_results_folds_0_1_2_3_4 \
--label-key binary_label
```

Without postprocessing

|                |   Average |        STD |
|----------------|-----------|------------|
| DSC_atrium     |  0.915856 |  0.0209883 |
| HD95_atrium    |  6.71407  | 16.2752    |
| OVERLAP_atrium |  0.960739 |  0.0225544 |

With postprocessing

|                |   Average |       STD |
|----------------|-----------|-----------|
| DSC_atrium     |  0.916563 | 0.020782  |
| HD95_atrium    |  4.7163   | 1.80963   |
| OVERLAP_atrium |  0.960739 | 0.0225544 |

As we can see from the average results, the postprocessing did not affect the "overlap" metric, but significantly improved the HD95 metric. This was likey achieved by removing small outlier regions.


Per-subject results without postprocessing
* Notice that the HD95 for subjects 8 and 17 are significantly improved.

![image](https://github.com/user-attachments/assets/d4baf634-3d79-4d96-a647-c1b58731c801)
* The white region on the right of the image was incorrectly predicted by the model. The postprocessing procedure rejected that region, dropping the HD95_atrium from 140.026 to 2.82843

For every heart atrium segmentation in the training dataset, the left and right atrium are connected. Therefore, it's valid to assume that there
is only one foreground object (heart) in the MRI volume. 


|    | subject   |   subject_id |   DSC_atrium |   HD95_atrium |   OVERLAP_atrium |                                                                        
|----|-----------|--------------|--------------|---------------|------------------|                                                                        
|  0 | MBAS_001  |            1 |     0.905832 |       5.09902 |         0.959715 |                                                                        
|  1 | MBAS_002  |            2 |     0.910458 |       5       |         0.913529 |                                                                        
|  2 | MBAS_003  |            3 |     0.938167 |       5       |         0.94377  |                                                                        
|  3 | MBAS_004  |            4 |     0.948766 |       2.82843 |         0.9617   |                                                                        
|  4 | MBAS_005  |            5 |     0.93872  |       4.12311 |         0.941833 |                                                                        
|  5 | MBAS_006  |            6 |     0.926802 |       6       |         0.974447 |
|  6 | MBAS_007  |            7 |     0.921098 |       4       |         0.983684 |
|  7 | MBAS_008  |            8 |     0.876259 |      **10.7703**  |         0.956578 |
|  8 | MBAS_009  |            9 |     0.924237 |       4.24264 |         0.954417 |
|  9 | MBAS_010  |           10 |     0.942322 |       2.82843 |         0.964866 |
| 10 | MBAS_011  |           11 |     0.905182 |       5       |         0.987238 |
| 11 | MBAS_012  |           12 |     0.944502 |       6       |         0.955554 |
| 12 | MBAS_013  |           13 |     0.914685 |       3       |         0.953459 |
| 13 | MBAS_014  |           14 |     0.916273 |       6       |         0.973981 |
| 14 | MBAS_015  |           15 |     0.952441 |       2.82843 |         0.974009 |
| 15 | MBAS_016  |           16 |     0.92309  |       3       |         0.962083 |
| 16 | MBAS_017  |           17 |     0.888766 |     **140.026**   |         0.986424 |
| 17 | MBAS_018  |           18 |     0.939846 |       2.44949 |         0.956386 |
| 18 | MBAS_019  |           19 |     0.943974 |       3.60555 |         0.971279 |
| 19 | MBAS_020  |           20 |     0.927422 |       3.60555 |         0.989662 |
| 20 | MBAS_021  |           21 |     0.92503  |       3       |         0.92405  |
| 21 | MBAS_022  |           22 |     0.928949 |       4.12311 |         0.94272  |
| 22 | MBAS_023  |           23 |     0.917008 |       3.60555 |         0.948837 |
| 23 | MBAS_024  |           24 |     0.901489 |       5       |         0.991185 |
| 24 | MBAS_025  |           25 |     0.932564 |       2.82843 |         0.969031 |
| 25 | MBAS_026  |           26 |     0.909255 |       3.31662 |         0.971091 |
| 26 | MBAS_027  |           27 |     0.934673 |       3       |         0.961356 |
| 27 | MBAS_028  |           28 |     0.905813 |       3.16228 |         0.980786 |
| 28 | MBAS_029  |           29 |     0.916222 |       5.09902 |         0.969676 |
| 29 | MBAS_030  |           30 |     0.872058 |       7       |         0.973607 |
| 30 | MBAS_031  |           31 |     0.916482 |       5.38516 |         0.944413 |
| 31 | MBAS_032  |           32 |     0.888885 |       5.09902 |         0.973697 |
| 32 | MBAS_033  |           33 |     0.949438 |       2.23607 |         0.989563 |
| 33 | MBAS_034  |           34 |     0.907835 |       5.74456 |         0.964382 |
| 34 | MBAS_035  |           35 |     0.912404 |       5       |         0.959879 |
| 35 | MBAS_036  |           36 |     0.881287 |       5.91608 |         0.939406 |
| 36 | MBAS_037  |           37 |     0.940122 |       3       |         0.977618 |
| 37 | MBAS_038  |           38 |     0.881166 |       4.47214 |         0.964819 |
| 38 | MBAS_039  |           39 |     0.865861 |       4.47214 |         0.991952 |
| 39 | MBAS_040  |           40 |     0.931908 |       3.60555 |         0.94506  |
| 40 | MBAS_041  |           41 |     0.891977 |       4.24264 |         0.979511 |
| 41 | MBAS_042  |           42 |     0.927701 |       3.60555 |         0.993746 |
| 42 | MBAS_043  |           43 |     0.932126 |       3.31662 |         0.963949 |
| 43 | MBAS_044  |           44 |     0.907138 |       6.16441 |         0.978235 |
| 44 | MBAS_045  |           45 |     0.889256 |       5       |         0.896812 |
| 45 | MBAS_046  |           46 |     0.90646  |       5.38516 |         0.885759 |
| 46 | MBAS_047  |           47 |     0.911662 |       4.24264 |         0.982172 |
| 47 | MBAS_048  |           48 |     0.922665 |       7.07107 |         0.935785 |
| 48 | MBAS_049  |           49 |     0.916237 |       6.7082  |         0.940389 |
| 49 | MBAS_050  |           50 |     0.923731 |       5.09902 |         0.930767 |
| 50 | MBAS_051  |           51 |     0.933435 |       3.60555 |         0.942186 |
| 51 | MBAS_052  |           52 |     0.937372 |       5.83095 |         0.951141 |
| 52 | MBAS_053  |           53 |     0.906358 |       4.58258 |         0.97431  |
| 53 | MBAS_054  |           54 |     0.922598 |       4.24264 |         0.932897 |
| 54 | MBAS_055  |           55 |     0.941203 |       3       |         0.978253 |
| 55 | MBAS_056  |           56 |     0.883944 |       9.48683 |         0.915636 |
| 56 | MBAS_057  |           57 |     0.914974 |       4       |         0.96363  |
| 57 | MBAS_058  |           58 |     0.912971 |       5.09902 |         0.965755 |
| 58 | MBAS_059  |           59 |     0.888019 |       6.40312 |         0.978721 |
| 59 | MBAS_060  |           60 |     0.901105 |       7       |         0.967946 |
| 60 | MBAS_061  |           61 |     0.909661 |       8.60233 |         0.970936 |
| 61 | MBAS_062  |           62 |     0.944527 |       3       |         0.97849  |
| 62 | MBAS_063  |           63 |     0.930354 |       3.74166 |         0.977904 |
| 63 | MBAS_064  |           64 |     0.913442 |       3.31662 |         0.971915 |
| 64 | MBAS_065  |           65 |     0.908046 |       4.47214 |         0.956928 |
| 65 | MBAS_066  |           66 |     0.921728 |       4.12311 |         0.988531 |
| 66 | MBAS_067  |           67 |     0.92743  |       3.16228 |         0.965826 |
| 67 | MBAS_068  |           68 |     0.909698 |       6.32456 |         0.976525 |
| 68 | MBAS_069  |           69 |     0.908778 |       5.91608 |         0.926799 |
| 69 | MBAS_070  |           70 |     0.85804  |      12.7671  |         0.932534 |

Per-subject results with postprocessing

|    | subject   |   subject_id |   DSC_atrium |   HD95_atrium |   OVERLAP_atrium |
|----|-----------|--------------|--------------|---------------|------------------|
|  0 | MBAS_001  |            1 |     0.905832 |       5.09902 |         0.959715 |
|  1 | MBAS_002  |            2 |     0.910458 |       5       |         0.913529 |
|  2 | MBAS_003  |            3 |     0.938167 |       5       |         0.94377  |
|  3 | MBAS_004  |            4 |     0.948766 |       2.82843 |         0.9617   |
|  4 | MBAS_005  |            5 |     0.93872  |       4.12311 |         0.941833 |
|  5 | MBAS_006  |            6 |     0.926802 |       6       |         0.974447 |
|  6 | MBAS_007  |            7 |     0.921098 |       4       |         0.983684 |
|  7 | MBAS_008  |            8 |     0.878312 |       **8.12404** |         0.956578 |
|  8 | MBAS_009  |            9 |     0.924237 |       4.24264 |         0.954417 |
|  9 | MBAS_010  |           10 |     0.942322 |       2.82843 |         0.964866 |
| 10 | MBAS_011  |           11 |     0.905182 |       5       |         0.987238 |
| 11 | MBAS_012  |           12 |     0.944502 |       6       |         0.955554 |
| 12 | MBAS_013  |           13 |     0.914685 |       3       |         0.953459 |
| 13 | MBAS_014  |           14 |     0.916273 |       6       |         0.973981 |
| 14 | MBAS_015  |           15 |     0.952441 |       2.82843 |         0.974009 |
| 15 | MBAS_016  |           16 |     0.92309  |       3       |         0.962083 |
| 16 | MBAS_017  |           17 |     0.935402 |       **2.82843** |         0.986424 |
| 17 | MBAS_018  |           18 |     0.939846 |       2.44949 |         0.956386 |
| 18 | MBAS_019  |           19 |     0.943974 |       3.60555 |         0.971279 |
| 19 | MBAS_020  |           20 |     0.927422 |       3.60555 |         0.989662 |
| 20 | MBAS_021  |           21 |     0.92503  |       3       |         0.92405  |
| 21 | MBAS_022  |           22 |     0.928949 |       4.12311 |         0.94272  |
| 22 | MBAS_023  |           23 |     0.917048 |       3.60555 |         0.948837 |
| 23 | MBAS_024  |           24 |     0.901489 |       5       |         0.991185 |
| 24 | MBAS_025  |           25 |     0.932564 |       2.82843 |         0.969031 |
| 25 | MBAS_026  |           26 |     0.909255 |       3.31662 |         0.971091 |
| 26 | MBAS_027  |           27 |     0.934673 |       3       |         0.961356 |
| 27 | MBAS_028  |           28 |     0.905813 |       3.16228 |         0.980786 |
| 28 | MBAS_029  |           29 |     0.916222 |       5.09902 |         0.969676 |
| 29 | MBAS_030  |           30 |     0.872058 |       7       |         0.973607 |
| 30 | MBAS_031  |           31 |     0.916482 |       5.38516 |         0.944413 |
| 31 | MBAS_032  |           32 |     0.888885 |       5.09902 |         0.973697 |
| 32 | MBAS_033  |           33 |     0.949438 |       2.23607 |         0.989563 |
| 33 | MBAS_034  |           34 |     0.907835 |       5.74456 |         0.964382 |
| 34 | MBAS_035  |           35 |     0.912404 |       5       |         0.959879 |
| 35 | MBAS_036  |           36 |     0.882012 |       5.91608 |         0.939406 |
| 36 | MBAS_037  |           37 |     0.940122 |       3       |         0.977618 |
| 37 | MBAS_038  |           38 |     0.881166 |       4.47214 |         0.964819 |
| 38 | MBAS_039  |           39 |     0.865861 |       4.47214 |         0.991952 |
| 39 | MBAS_040  |           40 |     0.931908 |       3.60555 |         0.94506  |
| 40 | MBAS_041  |           41 |     0.891977 |       4.24264 |         0.979511 |
| 41 | MBAS_042  |           42 |     0.927701 |       3.60555 |         0.993746 |
| 42 | MBAS_043  |           43 |     0.932126 |       3.31662 |         0.963949 |
| 43 | MBAS_044  |           44 |     0.907138 |       6.16441 |         0.978235 |
| 44 | MBAS_045  |           45 |     0.889256 |       5       |         0.896812 |
| 45 | MBAS_046  |           46 |     0.90646  |       5.38516 |         0.885759 |
| 46 | MBAS_047  |           47 |     0.911662 |       4.24264 |         0.982172 |
| 47 | MBAS_048  |           48 |     0.922665 |       7.07107 |         0.935785 |
| 48 | MBAS_049  |           49 |     0.916237 |       6.7082  |         0.940389 |
| 49 | MBAS_050  |           50 |     0.923731 |       5.09902 |         0.930767 |
| 50 | MBAS_051  |           51 |     0.933435 |       3.60555 |         0.942186 |
| 51 | MBAS_052  |           52 |     0.937372 |       5.83095 |         0.951141 |
| 52 | MBAS_053  |           53 |     0.906358 |       4.58258 |         0.97431  |
| 53 | MBAS_054  |           54 |     0.922598 |       4.24264 |         0.932897 |
| 54 | MBAS_055  |           55 |     0.941203 |       3       |         0.978253 |
| 55 | MBAS_056  |           56 |     0.883944 |       9.48683 |         0.915636 |
| 56 | MBAS_057  |           57 |     0.914974 |       4       |         0.96363  |
| 57 | MBAS_058  |           58 |     0.912971 |       5.09902 |         0.965755 |
| 58 | MBAS_059  |           59 |     0.888019 |       6.40312 |         0.978721 |
| 59 | MBAS_060  |           60 |     0.901105 |       7       |         0.967946 |
| 60 | MBAS_061  |           61 |     0.909661 |       8.60233 |         0.970936 |
| 61 | MBAS_062  |           62 |     0.944527 |       3       |         0.97849  |
| 62 | MBAS_063  |           63 |     0.930354 |       3.74166 |         0.977904 |
| 63 | MBAS_064  |           64 |     0.913442 |       3.31662 |         0.971915 |
| 64 | MBAS_065  |           65 |     0.908046 |       4.47214 |         0.956928 |
| 65 | MBAS_066  |           66 |     0.921728 |       4.12311 |         0.988531 |
| 66 | MBAS_067  |           67 |     0.92743  |       3.16228 |         0.965826 |
| 67 | MBAS_068  |           68 |     0.909698 |       6.32456 |         0.976525 |
| 68 | MBAS_069  |           69 |     0.908778 |       5.91608 |         0.926799 |
| 69 | MBAS_070  |           70 |     0.85804  |      12.7671  |         0.932534 |




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

With ball dilation  = 2

|                |   Average |        STD |
|----------------|-----------|------------|
| DSC_atrium     |  0.7723   | 0.0331598  |
| HD95_atrium    |  7.98733  | 1.95485    |
| OVERLAP_atrium |  0.992133 | 0.00926877 |

With ball dilation  = 3
|                |   Average |        STD |                                                                                                                                  
|----------------|-----------|------------|                                                                                                                                  
| DSC_atrium     |  0.7723   | 0.0331598  |                                                                                                                                  
| HD95_atrium    |  7.98733  | 1.95485    |                                                                                                                                  
| OVERLAP_atrium |  0.992133 | 0.00926877 |       


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

With ball dilation = 3

|               |   Average |       STD |
|---------------|-----------|-----------|
| DSC_wall      |  0.261594 | 0.0270484 |
| DSC_right     |  0.375259 | 0.0545907 |
| DSC_left      |  0.397722 | 0.0481255 |
| HD95_wall     | 10.8188   | 1.82835   |
| HD95_right    | 68.4535   | 9.68076   |
| HD95_left     | 79.2885   | 9.68793   |
| OVERLAP_wall  |  **0.985048** | 0.0156605 |
| OVERLAP_right |  **0.99469**  | 0.0114553 |
| OVERLAP_left  |  **0.994361** | 0.0117473 |

Notice that the maximum overlap with wall increases from 90.95% to 96.09% to 97.6% with the added ball dilation. This will be the upper bound in possible DICE score.
