


## Cascaded Models (2 Stage Models)
Previously I experimented with 2-stage models in the style of nnU-Net where the 1st stage predictions are provided to the 2nd stage model as an additional channel input, such that the 2nd stage model can refine the predictions. In this setup, the 1st stage model generates a coarse prediction, and the 2nd stage model cleans up the prediction. My prior experiments did not find any value in this approach. A high quality single-stage model outperformed the 2 stage cascaded model.

An alternative formulation of the 2-stage cascaded method is to predict a binary foreground mask with the 1st stage model, and to use this binary mask in the 2nd stage to reduce the search space when performing the multi-class segmentation. The 1st stage binary mask is applied to the loss function. The cross-entropy loss is zeroed out for regions outside the binary mask. The 2nd stage model is trained to only focus on regions within the binary mask.

I implemented `mbasTrainer.py`, updated the data loaders, and updated the loss functions such that the binary mask is applied to the loss function.
The 2nd stage models are also trained to oversample the foreground, because these is no value to training on the background since the 1st stage model will handle localization of the segmentation regions.
```
    oversample_foreground_percent = 1.0,
    probabilistic_oversampling = True,
    sample_class_probabilities = {1: 0.5, 2: 0.25, 3: 0.25}
```

### Cascaded 2nd stage model trained using ground truth binary mask
To prove that the 2-stage cascaded mask method can improve performance verse a single-stage model, I trained the 2nd stage model using the ground truth binary mask. The ground truth 3-class segmentation masks were binarized and used as if they were the predictions from the 1st stage model.

All 2nd stage models trained with this approach significantly outperformed the single-stage models. This result proves that if we can train a high-quality 1st stage binary segmentation model, then the 2nd stage model can significantly improve in accuracy.
```
|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
| 53 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_16_256_GT                                           |      1 |    1.33333 |   0.803356 |     2.47652 |   0.946645  |      2.24972 |   0.949421 |     2.60852 |
| 54 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_16_256_nblocks3_GT                                  |      2 |    2.16667 |   0.802718 |     2.60799 |   0.945918  |      2.26261 |   0.949497 |     2.61356 |
|  0 | nnUNetTrainer_MedNeXt__MedNeXtV2Plans_2024_08_08__cascade_mask_16_128_GT                                      |      3 |    3.33333 |   0.811113 |     2.69143 |   0.946151  |      2.32819 |   0.946794 |     2.6589  |
| 58 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_256_GT                                           |      4 |    4       |   0.800511 |     2.70921 |   0.943679  |      2.37081 |   0.948746 |     2.62145 |
| 55 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_128_nblocks3_GT                                  |      5 |    5.16667 |   0.798511 |     2.76973 |   0.940753  |      2.63368 |   0.949377 |     2.70546 |
| 57 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_128_nblocks3_decoder2_GT                         |      6 |    6.5     |   0.797741 |     2.78176 |   0.940503  |      2.54132 |   0.947827 |     2.86611 |
|  1 | nnUNetTrainer_MedNeXt__MedNeXtV2Plans_2024_08_08__cascade_mask_32_128_GT                                      |      7 |    7.33333 |   0.802619 |     2.85685 |   0.938215  |      2.68238 |   0.945886 |     2.70864 |
| 56 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_128_nblocks3_bs5_GT                              |      8 |    8.16667 |   0.782283 |     2.94021 |   0.940533  |      2.66101 |   0.947561 |     2.70978 |
|  2 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   10       |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  3 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   11       |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
```
### Cascaded 2nd stage model trained using ground truth binary mask + dilation
I added dilation to the ground truth binary mask, which simulates a less accurate 1st stage result. The dilation was performed using `batchgeneratorsv2.transforms.nnunet.random_binary_operator.binary_dilation_torch` which applies morphological dilation using the an [skimage.morphology.ball](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.ball) in a similar fashion to https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html#dilation.

The 2nd stage models trained on the dilated ground truth masks performed worse than models trained on ground truth without dilation.

```
|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  0 | nnUNetTrainer_MedNeXt__MedNeXtV2Plans_2024_08_08__cascade_mask_16_128_GT                                      |      3 |    3.33333 |   0.811113 |     2.69143 |   0.946151  |      2.32819 |   0.946794 |     2.6589  |
|  2 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   10       |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  3 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   11       |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 17 | nnUNetTrainer_MedNeXt__MedNeXtV2Plans_2024_08_09__cascade_mask_dil5_16_128_GT                                 |     24 |   25.6667  |   0.712605 |     3.06802 |   0.918906  |      3.38605 |   0.926975 |     3.7737  |
| 35 | nnUNetTrainer_MedNeXt__MedNeXtV2Plans_2024_08_09__cascade_mask_dil1_16_128_GT                                 |     42 |   38.5     |   0.71169  |     3.27481 |   0.917995  |      3.77168 |   0.924817 |     4.79825 |
| 36 | nnUNetTrainer_MedNeXt__MedNeXtV2Plans_2024_08_09__cascade_mask_dil10_16_128_GT                                |     43 |   39       |   0.712861 |     3.24838 |   0.916315  |      3.88388 |   0.92475  |     4.86654 |
```
![image](https://github.com/user-attachments/assets/4b8d1737-ffc5-47e2-a349-74c691a0fa2c)
![image](https://github.com/user-attachments/assets/64213865-404e-4f8c-aedc-7dcd52cc9d49)

### Cascaded 1st stage models
The objective of the 1st stage model is to perform binary segmentation of left and right atrium -- i.e the goal is to segment the foreground (heart atrium) from the background. The ground truth 3-class segmentation masks were binarized and used as labels.

I trained a variety of `nnUNetResEncUNetM` models. I experimented with different input resolutions and spacings. The default spacing for the MBAS dataset is `[2.5, 1.5, 1.5]` with MRI volume size of either `[44,638,638]` or `[44,574,574]`. The MRI volumes were downsampled to "3d_lowres" `[2.5, 0.9737296353754783, 0.9737296353754783]` spacing with volume size `[2.5, 410, 410]`, "3d_lowres_1.0" '[2.5, 1.0, 1.0]` spacing with volume size `[44, 399, 399]`, "3d_lowres_1.25" `[2.5, 1.25, 1.25]` spacing with volume size `[ 44, 319, 319]`, and "3d_lowres_1.5" `[2.5, 1.5, 1.5]` spacing with volume size `[ 44, 266, 266]`.
I experimented with variety of input resolutions including `[16, 256, 256]`, [32, 256, 256]`, `[40, 256, 256]`, `[20, 256, 256]` for the default "3d_fullres" model, and `[28, 256, 224]` for the default 3d_lowres model. `nblocks3` refers to the number of ResidualBlocks in the first stage of the Residual Encoder.
```
|    | model                                                                        |   Rank |   Avg_Rank |   DSC_atrium |   HD95_atrium |
|----|------------------------------------------------------------------------------|--------|------------|--------------|---------------|
|  0 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres                    |      1 |        1.5 |     0.934025 |       3.39874 |
|  1 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_16_256           |      2 |        2.5 |     0.933949 |       3.41017 |
| 10 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_40_256_nblocks3  |      3 |        2.5 |     0.933596 |       3.32731 |
|  2 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_16_256_nblocks3  |      4 |        3.5 |     0.933803 |       3.4339  |
|  3 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.25_M_16_256_nblocks3 |      5 |        5   |     0.933403 |       3.48028 |
|  9 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_fullres                   |      6 |        6   |     0.932549 |       3.51358 |
|  4 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.25_M_16_256          |      7 |        7.5 |     0.932349 |       3.58482 |
|  5 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.5_M_16_256_nblocks3  |      8 |        7.5 |     0.931221 |       3.54931 |
|  6 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.5_M_16_256           |      9 |        9   |     0.930668 |       3.63163 |
|  7 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_256_nblocks3    |     10 |       10   |     0.473494 |     119.312   |
|  8 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_256             |     11 |       11   |     0.472544 |     137.721   |
```

## Experiments Titrating the Hausdorff Lossf

## Adding more filters and blocks in the first few stages

## Adding dilation





Dataset101_MBAS/mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res
- overfitting to the training data. validation loss diverged.



