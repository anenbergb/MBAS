# nnUNetResEncUNet Models
I trained the baseline nnUNetResEncUNet style models.
* Training for 1000 epochs is better than training for 250 epochs. 
* 3D models are better than 2D models
* L model is marginally better than M model.


L model definition
```
"patch_size": [32, 384, 384]
features_per_stage = [32, 64, 128, 256, 320, 320, 320]
kernel_size = [1,3,3] ... (3,3,3)
```
M model definition is very similar to L, but with input patch size `(16, 256, 256)`
|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   10.6667  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
| 10 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres                                                             |     12 |   15.3333  |   0.723406 |     2.83256 |   0.923721  |      3.31705 |   0.930415 |     3.68148 |
| 12 | nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres                                                            |     14 |   17.3333  |   0.71789  |     2.91306 |   0.924197  |      3.24835 |   0.929497 |     3.65794 |
| 11 | nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres2                                                            |     15 |   17.5     |   0.723516 |     2.8779  |   0.923062  |      3.39967 |   0.930453 |     3.76087 |
| 13 | nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres                                                             |     16 |   19       |   0.721673 |     2.90694 |   0.922486  |      3.33962 |   0.93     |     3.82659 |
| 30 | nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__3d_fullres                                                   |     33 |   34.1667  |   0.718999 |     3.51254 |   0.918723  |      3.36436 |   0.927    |     4.51756 |
| 38 | nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__3d_cascade_fullres_from_only_atrium_orig01                   |     41 |   39.3333  |   0.721857 |     3.42995 |   0.918737  |      3.48732 |   0.924729 |     5.04659 |
| 39 | nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__3d_cascade_fullres_from_only_atrium                          |     42 |   39.6667  |   0.71823  |     3.43968 |   0.918646  |      3.65566 |   0.925149 |     4.67943 |
| 40 | nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__2d                                                           |     43 |   40.6667  |   0.697354 |     3.37629 |   0.916023  |      3.62018 |   0.92251  |     4.23146 |
| 44 | nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__2d_cascade_fullres_from_only_atrium                          |     47 |   42.5     |   0.700439 |     3.31663 |   0.912584  |      3.65345 |   0.921374 |     4.37437 |
```
