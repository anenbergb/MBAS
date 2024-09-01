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
- adding dilation 1 to binary mask increased the accuracy

|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   11.6667  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   12.6667  |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 68 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_cascade_ResEncUNet_08_27                               |     12 |   15.5     |   0.723734 |     2.82654 |   0.924743  |      3.27531 |   0.931603 |     3.87116 |
| 67 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_cascade_ResEncUNet_08_27                                    |     14 |   17.8333  |   0.720656 |     2.88458 |   0.925382  |      3.20052 |   0.931084 |     3.87481 |

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

### 3: Adding dropout
- Adding dropout hurt performance
|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   11.6667  |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   12.6667  |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 68 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_cascade_ResEncUNet_08_27                               |     12 |   15.5     |   0.723734 |     2.82654 |   0.924743  |      3.27531 |   0.931603 |     3.87116 |
| 69 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil1_drop50_cascade_ResEncUNet_08_27                        |     36 |   36       |   0.704776 |     2.89172 |   0.920088  |      3.32395 |   0.922068 |     4.18848 |

### 4: Combining slim model (128 feature dim), with dropout, with varying patch size
