The goal of these experiments is to train a 2nd Stage segmentation model that leverages the high overall (high recall) binary segmentation model predictions
of the first stage to guide the multi-class MBAS predictions of the second stage model.

# 2024-08-30 Experiments
The stage 1 model used for these experiments is
`/home/bryan/expr/mbas_nnUNet_results/Dataset104_MBAS/mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96`

The starting point for these experiments is `ResEncUNet_p20_256_cascade_ResEncUNet_08_27`
- this is a fairly standard ResEncUNet with 7 stages that oversamples the foreground.
```
configurator.configurations["ResEncUNet_p20_256_cascade_ResEncUNet_08_27"] = configurator.set_params(
    patch_size=(20,256,256),
    data_identifier = "nnUNetPlans_3d_fullres",
    spacing = (2.5, 0.625, 0.625),
    probabilistic_oversampling = True,
    oversample_foreground_percent = 1.0,
    sample_class_probabilities = {1: 0.5, 2: 0.25, 3: 0.25},
).set_cascade(
    cascaded_mask_dilation=0,
    is_cascaded_mask=True,
    previous_stage="Dataset104_ResEncUNet_3d_lowres_for25_drop50_slim96",
).nnUNetResEncUNet(
    features_per_stage= [32,64,128,256,320,320,320],
    kernel_sizes=[
        [1, 3, 3],
        [1, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    strides=[
        [1, 1, 1],
        [1, 2, 2],
        [1, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
    ],
    n_blocks_per_stage=[1, 3, 4, 6, 6, 6, 6],
    n_conv_per_stage_decoder=[1, 1, 1, 1, 1, 1],
)
```

## Reference models
As a reminder, the `mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_dil1_cascade_3d_low_res` model was the same exact architecture as `2024_08_30__ResEncUNet_p20_256_cascade_ResEncUNet_08_27`
but with smaller input patch size of `(16,256,256)`.

`nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05` model
```
patch_size = (16, 256, 256),
features_per_stage = (32, 64, 128, 256, 320, 320, 320),
stem_kernel_size = 1,
kernel_sizes = [
    (1,3,3),
    (1,3,3), 
    (3,3,3),
    (3,3,3),
    (3,3,3),
    (3,3,3),
    (3,3,3),
],
strides = [
    (1,1,1),
    (1,2,2),
    (1,2,2),
    (2,2,2),
    (2,2,2),
    (2,2,2),
    (2,2,2),
],
n_blocks_per_stage = [3,4,6,6,6,6,6],
exp_ratio_per_stage = [2,3,4,4,4,4,4],

decode_stem_kernel_size=3,
override_down_kernel_size = True,
down_kernel_size = 1,

oversample_foreground_percent=1.0,
probabilistic_oversampling = True,
sample_class_probabilities = {1: 0.5, 2: 0.25, 3: 0.25}
```

## Experiment ideas
### 1: Add dilation (buffer) to binary mask
- adding dilation 1 and 2 to binary mask increased the accuracy. Increasing dilation to 2 actually performed better than dilation 1
- increasing dilation to 3 slightly hurt performance. 

|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   11.6667  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   12.6667  |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 75 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_cascade_ResEncUNet_08_27                               |     11 |   15.3333  |   0.724251 |     2.74407 |   0.925197  |      3.3252  |   0.932641 |     3.7535  |
| 11 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_cascade_ResEncUNet_08_27                               |     13 |   17.5     |   0.723734 |     2.82654 |   0.924743  |      3.27531 |   0.931603 |     3.87116 |
| 15 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_cascade_ResEncUNet_08_27                                    |     18 |   20.5     |   0.720656 |     2.88458 |   0.925382  |      3.20052 |   0.931084 |     3.87481 |
| 21 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil3_cascade_ResEncUNet_08_27                               |     23 |   24.3333  |   0.72121  |     2.73434 |   0.922903  |      3.39686 |   0.931058 |     3.73003 |

### 2: Changing the patch size from (20,256,256) to smaller sizes
- Reducing the patch size hurt performance
- `p20_256` patch size (20,256,256) beats patch size (16,256,256), (16,192,192)

|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   11.6667  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   12.6667  |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 68 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_cascade_ResEncUNet_08_27                               |     12 |   15.5     |   0.723734 |     2.82654 |   0.924743  |      3.27531 |   0.931603 |     3.87116 |
| 65 | mbasTrainer__plans_2024_08_30__ResEncUNet_p16_192_dil1_cascade_ResEncUNet_08_27                               |     13 |   17.1667  |   0.722623 |     2.77173 |   0.924594  |      3.26591 |   0.93123  |     3.89253 |
| 66 | mbasTrainer__plans_2024_08_30__ResEncUNet_p16_256_dil1_cascade_ResEncUNet_08_27                               |     16 |   18.3333  |   0.720092 |     2.85239 |   0.924964  |      3.20732 |   0.9311   |     3.94005 |

### 3: Fewer feature dimensions (slim models)
- It appears that the slim models actually hurt performance. The default model has a maximum of 320 feature channels. Perhaps for this segmentation task it is useful to have more parameters.

|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   11.6667  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   12.6667  |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 11 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_cascade_ResEncUNet_08_27                               |     13 |   17.5     |   0.723734 |     2.82654 |   0.924743  |      3.27531 |   0.931603 |     3.87116 |
| 13 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_slim96_cascade_ResEncUNet_08_27                        |     15 |   19.5     |   0.720041 |     2.85943 |   0.926474  |      3.17129 |   0.931497 |     3.92933 |
| 74 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_slim256_cascade_ResEncUNet_08_27                       |     17 |   20.1667  |   0.721173 |     2.72213 |   0.92418   |      3.33861 |   0.931766 |     3.76833 |
| 17 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_slim128_cascade_ResEncUNet_08_27                       |     20 |   21.5     |   0.72171  |     2.8365  |   0.925451  |      3.31875 |   0.93243  |     4.16739 |

### 4: Adding dropout
- Adding dropout hurt performance


|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   11.6667  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   12.6667  |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 68 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_cascade_ResEncUNet_08_27                               |     12 |   15.5     |   0.723734 |     2.82654 |   0.924743  |      3.27531 |   0.931603 |     3.87116 |
| 69 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_drop50_cascade_ResEncUNet_08_27                        |     36 |   36       |   0.704776 |     2.89172 |   0.920088  |      3.32395 |   0.922068 |     4.18848 |

### 5: Combining slim model (128 feature dim), with dropout, with varying patch size
- Adding dropout hurt performance
- These models with dropout + slim performed even worse than the baseline model (320 feature dim) without dropout.

|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   11.6667  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   12.6667  |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 11 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_cascade_ResEncUNet_08_27                               |     12 |   16.1667  |   0.723734 |     2.82654 |   0.924743  |      3.27531 |   0.931603 |     3.87116 |
| 12 | mbasTrainer__plans_2024_08_30__ResEncUNet_p16_192_dil1_cascade_ResEncUNet_08_27                               |     13 |   18       |   0.722623 |     2.77173 |   0.924594  |      3.26591 |   0.93123  |     3.89253 |
| 72 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_slim128_cascade_ResEncUNet_08_27                       |     18 |   20.5     |   0.72171  |     2.8365  |   0.925451  |      3.31875 |   0.93243  |     4.16739 |
| 49 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_drop50_slim_128_cascade_ResEncUNet_08_27               |     52 |   49       |   0.696401 |     3.19986 |   0.916795  |      3.54396 |   0.916747 |     4.50655 |
| 55 | mbasTrainer__plans_2024_08_30__ResEncUNet_p16_192_dil1_drop50_slim_128_cascade_ResEncUNet_08_27               |     58 |   53.6667  |   0.692874 |     3.36329 |   0.913402  |      3.57347 |   0.915009 |     4.53102 |

### 6: Using batch_dice
Whether to use batch dice (pretend all samples in the batch are one image, compute dice loss over that)
- Using batch_dice improves performance

|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   13.6667  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
| 78 | **mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27**                    |     10 |   14.8333  |   0.723445 |     2.72685 |   0.925043  |      3.21474 |   0.931448 |     3.71448 |
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     11 |   15       |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 10 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_cascade_ResEncUNet_08_27                               |     12 |   16.1667  |   0.724251 |     2.74407 |   0.925197  |      3.3252  |   0.932641 |     3.7535  |

# 2024-09-02 Experiments
In this set of experiments I train a MedNextV2 style network.
The MedNeXtV2 network architectures explored here have very similar kernels and strides to the the ResEncUNet

The baseline MedNeXtV2_p20_256_dil2_nblocks1346_cascade_ResEncUNet_08_27
```
features_per_stage= [32,64,128,256,320,320,320],
kernel_sizes=[
    [1, 3, 3],
    [1, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
],
strides=[
    [1, 1, 1],
    [1, 2, 2],
    [1, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [1, 2, 2],
    [1, 2, 2],
],
n_blocks_per_stage = [1,3,4,6,6,6,6],
exp_ratio_per_stage = [2,3,4,4,4,4,4],
```
MedNeXtV2_p20_256_dil2_nblocks1346_slim128_cascade_ResEncUNet_08_27
- reduces the max feature dimension to 128 rather than 320
MedNeXtV2_p16_256_dil2_nblocks346_slim128_cascade_ResEncUNet_08_27
- reduces the max feature dimension to 128 rather than 320
- `n_blocks_per_stage` goes from `[1,3,4,6,6,6,6]` to `[3,4,6,6,6,6,6]`
- reduces the input patch size from (20,256,256) to (16,256,256)
MedNeXtV2_p16_256_dil2_nblocks346_slim128_stride16to1_cascade_ResEncUNet_08_27
- reduces the max feature dimension to 128 rather than 320
- `n_blocks_per_stage` goes from `[1,3,4,6,6,6,6]` to `[3,4,6,6,6,6,6]`
- reduces the input patch size from (20,256,256) to (16,256,256)
- downsample the input patch from (16,256,256) to (1,4) rather than (4,4)

The following conclusions can be made from the results
- The `slim128` model gets better results than the default model. This result agrees with prior experiments.
- Reducing the input patch size from (20,256,256) to (16,256,256) may improve performance, but improved performance is likely attributed to the additional stage one blocks that were increased from (1,3,4,...) to (3,4,6,...)
- More downsampling of the input patch from (16,256,256) to (1,4) rather than (4,4) doesn't improve performance.

|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   13.6667  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     11 |   15       |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 10 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_cascade_ResEncUNet_08_27                               |     12 |   16.1667  |   0.724251 |     2.74407 |   0.925197  |      3.3252  |   0.932641 |     3.7535  |
| 26 | mbasTrainer__plans_2024_09_02__MedNeXtV2_p16_256_dil2_nblocks346_slim128_cascade_ResEncUNet_08_27             |     27 |   30.3333  |   0.718305 |     2.80845 |   0.921939  |      3.31858 |   0.929709 |     4.06074 |
| 81 | mbasTrainer__plans_2024_09_02__MedNeXtV2_p16_256_dil2_nblocks346_slim128_stride16to1_cascade_ResEncUNet_08_27 |     37 |   38.6667  |   0.716568 |     2.98722 |   0.919821  |      3.35611 |   0.928239 |     4.22568 |
| 39 | mbasTrainer__plans_2024_09_02__MedNeXtV2_p20_256_dil2_nblocks1346_slim128_cascade_ResEncUNet_08_27            |     41 |   41.5     |   0.712215 |     3.01782 |   0.916912  |      3.36771 |   0.928183 |     4.09686 |
| 42 | mbasTrainer__plans_2024_09_02__MedNeXtV2_p20_256_dil2_nblocks1346_cascade_ResEncUNet_08_27                    |     44 |   43       |   0.714093 |     3.01824 |   0.91399   |      3.64388 |   0.93027  |     4.06885 |

# 2024-09-10 Experiments

### 1: Increasing patch size from (20,256,256)
The best performing model according to the average rank is the `nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2`, which is 7-stage ResEncUNet trained on large `(32,384,384)` patches.
```
features_per_stage= [32,64,128,256,320,320,320],
kernel_sizes=[
    [1, 3, 3],
    [1, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
],
strides=[
    [1, 1, 1],
    [1, 2, 2],
    [1, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [1, 2, 2],
],
n_blocks_per_stage=[1, 3, 4, 6, 6, 6, 6],
n_conv_per_stage_decoder=[1, 1, 1, 1, 1, 1],
```

In this set of experiments I build upon the results of the best performing cascaded model `mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_cascade_ResEncUNet_08_27` by retaining
- 2-stage cascaded architecture
- dilation (buffer) of size 2 around the 1st stage binary mask predictions
Experiment with patch sizes `(32,384,384)`, and `(32,256,256)`

|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   14.3333  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  9 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27                    |     10 |   15.3333  |   0.723445 |     2.72685 |   0.925043  |      3.21474 |   0.931448 |     3.71448 |
| 27 | mbasTrainer__plans_2024_09_10__ResEncUNet_p32_256_dil2_bd_cascade_ResEncUNet_08_27                            |     28 |   29.3333  |   0.722941 |     2.85907 |   0.92196   |      3.39422 |   0.93142  |     3.81499 |
| 88 | mbasTrainer__plans_2024_09_10__ResEncUNet_p32_384_dil2_bd_cascade_ResEncUNet_08_27                            |     30 |   31.8333  |   0.720422 |     2.91118 |   0.923577  |      3.22313 |   0.929422 |     4.06153 |


### 2: Data Augmentation
Increase the likelihood of applying data augmentation to the sampled patches

ResEncUNet_p20_256_dil2_bd_aug01_cascade_ResEncUNet_08_27
```
    aug_spatial_p_rotation = 0.5,
    aug_spatial_p_scaling = 0.4,
    aug_gaussian_noise_p = 0.3,
    aug_gaussian_blur_p = 0.3,
    aug_brightness_p = 0.15,
    aug_contrast_p = 0.15,
    aug_lowres_p = 0.0,
```
ResEncUNet_p20_256_dil2_bd_aug02_cascade_ResEncUNet_08_27
```
    aug_spatial_p_rotation = 0.5,
    aug_spatial_p_scaling = 0.4,
    aug_gaussian_noise_p = 0.0,
    aug_gaussian_blur_p = 0.0,
    aug_brightness_p = 0.15,
    aug_contrast_p = 0.15,
    aug_lowres_p = 0.0,
```
ResEncUNet_p20_256_dil2_bd_aug03_cascade_ResEncUNet_08_27
```
    aug_spatial_p_scaling = 0.0,
    aug_lowres_p = 0.0,
```

|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   14.3333  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  9 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27                    |     10 |   15.3333  |   0.723445 |     2.72685 |   0.925043  |      3.21474 |   0.931448 |     3.71448 |
| 91 | mbasTrainer__plans_2024_09_10__ResEncUNet_p20_256_dil2_bd_aug03_cascade_ResEncUNet_08_27                      |     26 |   29.1667  |   0.720697 |     2.85116 |   0.924578  |      3.37887 |   0.931369 |     3.8154  |
| 31 | mbasTrainer__plans_2024_09_10__ResEncUNet_p20_256_dil2_bd_aug01_cascade_ResEncUNet_08_27                      |     33 |   34.8333  |   0.718293 |     2.8549  |   0.921701  |      3.32903 |   0.930367 |     3.79884 |
| 32 | mbasTrainer__plans_2024_09_10__ResEncUNet_p20_256_dil2_bd_aug02_cascade_ResEncUNet_08_27                      |     34 |   35.3333  |   0.71676  |     2.75316 |   0.920764  |      3.46288 |   0.931163 |     3.75921 |

The other data augmentation schemes performed worse than the baseline `ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27`. 

### 4: Train where 1st Stage inference results do not use postprocessing
- regular ResEnc but where first stage doesn't run postprocessing (so don't reject any disconnected small segmentations because  left and right atrium can be separate)

|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  9 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27                    |     10 |   15.3333  |   0.723445 |     2.72685 |   0.925043  |      3.21474 |   0.931448 |     3.71448 |
| 87 | mbasTrainer__plans_2024_09_11__ResEncUNet_p20_256_dil2_bd_cascade_ResEncUNet_08_27_nopost                     |     29 |   30       |   0.71969  |     2.83977 |   0.924093  |      3.35865 |   0.930651 |     3.81915 |

