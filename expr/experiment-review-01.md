I've run quite a few experiments over the past few weeks with the goal of improving Dice and HD95 scores. Some of the minor architectural changes helped and the updated patch sampling policy helped. However, training with Hausdorff Distance loss and 2-stage Cascaded models didn't help.

## Experiment 1: Various MedNeXt minor architectural changes.
The best performing model was one with input patch size (16, 256, 256), 7 stages that ultimately downsample the input to (1, 4, 4), retained symmetric number of blocks and expansion ratios in the U-Net decoder layers as in the U-Net encoder, and capped the maximum number of channels in the final layers to only 128! features_per_stage: (32, 64, 128, 128, 128, 128, 128)
```
|    | model                                                                                       |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  4 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__slim_128                                    |      5 |    9.16667 |   0.723578 |     3.32642 |   0.924154  |      3.10856 |   0.927038 |     4.50409 |
```
## Experiment 2: Modify patch sampling policy at training time
This experiment worked! Results improved across all metrics.
The default nnUNet patch sampling policy is random sampling. I tried a new policy to 100% of the time sample patches containing foreground object. 50% of the time centered on a "Atrium Wall", 25% of the time centered on "Right Atrium" and 25% of the time centered on "Left Atrium"
```
|    | model                                                                                       |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  0 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                      |      1 |    2.16667 |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
|  4 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__slim_128                                    |      5 |    9.16667 |   0.723578 |     3.32642 |   0.924154  |      3.10856 |   0.927038 |     4.50409 |
```
## Experiment 3: Introduce Hausdorff Distance loss
This experiment did not work!
I integrated the HausdorffDTLoss implemented in monai into nnUNetv2 and followed the training policy of https://arxiv.org/abs/1904.10030 and https://arxiv.org/pdf/2302.03868v3 where I just increased the relative weight of the Hausdorff Distance (HD) loss from 0 to 100% using a stepwise linear function, updated every 5 (or 10) epochs.
```
|    | model                                                                                       |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  4 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__slim_128                                    |      5 |    9.16667 |   0.723578 |     3.32642 |   0.924154  |      3.10856 |   0.927038 |     4.50409 |
| 25 | nnUNetTrainer_MedNeXt_CE_DC_HD__MedNeXtPlans_2024_07_26__slim_128_alpha10                   |     29 |   28.6667  |   0.592409 |     9.04654 |   0.886964  |      5.78562 |   0.905886 |     8.65668 |
| 26 | nnUNetTrainer_MedNeXt_CE_DC_HD__MedNeXtPlans_2024_07_26__slim_128_alpha05                   |     30 |   29.6667  |   0.551918 |    10.1797  |   0.869174  |      7.87584 |   0.895517 |     9.4183  |
```
## Experiment 4: Revisit Cascaded models
This experiment did not work!
I followed nnUNet style cascaded model training where the 1-hot encoded predictions from the stage 1 model are concatenated channelwise with the MRI input volume to produce input tensors of shape (4, 16, 96, 96) since the MRI is single channel (1,16,96,96) and the segmentation has 3 categories (3,16,96,96)
All attempts at cascaded training performed worse than the stage 1 model alone! (The base model was nnUNetTrainer_MedNeXt__MedNeXtPlans__3d_fullres )
```
|    | model                                                                                       |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  5 | nnUNetTrainer_MedNeXt__MedNeXtPlans__3d_fullres                                             |      6 |    9.66667 |   0.722514 |     3.25905 |   0.923382  |      3.17257 |   0.927485 |     4.59637 |
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_29__slim_128_patch96_oversample025              |      9 |   11       |   0.725279 |     3.27665 |   0.921329  |      3.28679 |   0.9265   |     4.43997 |
| 30 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_29__slim_128_oversample_05                      |     11 |   11.3333  |   0.720507 |     3.06413 |   0.921079  |      3.50887 |   0.92784  |     4.24393 |
| 31 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_29__slim_128_patch128_oversample025             |     14 |   14.3333  |   0.724519 |     3.3604  |   0.922292  |      3.34179 |   0.926154 |     4.62622 |
| 29 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_29__even_128_patch96_oversample025              |     15 |   15       |   0.725081 |     3.40352 |   0.919901  |      3.45598 |   0.926357 |     4.50093 |
```
