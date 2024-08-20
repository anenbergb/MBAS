# Validation set submission for 08/15/2024

I went with a cascaded 2-stage model for the validation set submission.
### 1st Stage Model
For the first stage, I used `mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres` model. 
This was trained for all 5 cross-validation folds such that the inference could be run on all of the images in the training dataset.
The model used in the final submission was trained on the "all" training dataset.

The Dice and HD95 results across all 70 training samples was
```
|             |   Average |       STD |
|-------------|-----------|-----------|
| DSC_atrium  |  0.930169 | 0.0184425 |
| HD95_atrium |  3.9864   | 1.92598   |
```

### 2nd Stage Model
For the second stage model, I trained two models
* `mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res`
* `mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res`

Both of these models oversampled the foreground regions following the best configuration from prior experiments. This sampling policy also makes sense because the loss is only measured for predictions within the bounds of the 1st stage binary mask prediction, so there isn't any value to training on regions outside of the binary mask.
```
oversample_foreground_percent = 1.0,
probabilistic_oversampling = True,
sample_class_probabilities = {1: 0.5, 2: 0.25, 3: 0.25},
```

#### mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res
I selected this model architecture because it performed the best when trained on the ground truth binary mask segmentation.
```
patch_size = (16, 256, 256),
features_per_stage = (32, 64, 128, 256, 320, 320, 320),
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
n_conv_per_stage_decoder = [1] * 6,
```
* the `_dil1` model adds a buffer (dilation) to the 1st stage binary mask.

Results
```
|    | model                                                                                     |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|-------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  0 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_16_256_GT                       |      1 |    1.33333 |   0.803356 |     2.47652 |   0.946645  |      2.24972 |   0.949421 |     2.60852 |
| 10 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_dil1_cascade_3d_low_res            |     11 |   13.8333  |   0.72286  |     2.85886 |   0.924629  |      3.26189 |   0.932176 |     3.65506 |
| 60 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res  (submitted)    |     13 |   17       |   0.716903 |     3.02594 |   0.924714  |      3.29153 |   0.932519 |     3.63699 |
```
* `mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_16_256_GT` loss curve clearly shows overfitting
![image](https://github.com/user-attachments/assets/fc7854b9-74ef-4d46-88da-584182966af0)
* `mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res` still shows overfitting
![image](https://github.com/user-attachments/assets/9b4edc87-a873-4cc6-8d7d-68219980245e)


#### mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res
I selected this model architecture because I thought that the MedNeXtV2 architecture might outperform the nnUNetResEncUNetMPlan since it uses more advanced modules.
```
patch_size = (16, 256, 256),
features_per_stage = (32, 64, 128, 256, 320, 320, 320),
stem_kernel_size = (1,3,3),
stem_channels=None,
stem_dilation=1,
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
n_blocks_per_stage=[3, 4, 4, 4, 4, 4, 4],
exp_ratio_per_stage=[2, 3, 4, 4, 4, 4, 4],
n_blocks_per_stage_decoder=[4, 4, 4, 4, 4, 3],
exp_ratio_per_stage_decoder=[4, 4, 4, 4, 3, 2],
```
Results
```
|    | model                                                                                     |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|-------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
| 47 | mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res (submitted)    |     49 |   44.5     |   0.693119 |     3.66241 |   0.916443  |      3.37182 |   0.922761 |     4.77616 |
| 63 | mbasTrainer__MedNeXtV2Plans_2024_08_13_GT__16_256_nblocks2_cascade_GT                     |     65 |   65       |   0        |   nan       |   0         |    nan       |   0        |   nan       |
```

* `mbasTrainer__MedNeXtV2Plans_2024_08_13_GT__16_256_nblocks2_cascade_GT`: Training this architecture on the ground truth binary masks didn't converge -- the loss exploded and resulted in NaNs.
![image](https://github.com/user-attachments/assets/cd5548ea-6363-4b08-9ca9-c6c3bf3c7fe2)
* `mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res`: Training on the `cascade_3d_low_res` input worked, but the training and validation losses begin to diverge as training progresses. It's likely that this model is overparameterized and overfit to the training dataset.
![image](https://github.com/user-attachments/assets/6d4a7158-06f4-47b6-8c92-c0801673afdd)


### Training nnUNetResEncUNetMPlans_2024_08_13__16_256 without a cascade
As a basis of comparison, I trained the same `nnUNetResEncUNetMPlans_2024_08_13__16_256` architecture without a cascade (i.e. prediicted the multi-class segmentation output using a single model).
* `_oversample_05` adds the 100% probabilistic oversampling of the foreground regions, the same as used when training the 2nd stage model, i.e `sample_class_probabilities = {1: 0.5, 2: 0.25, 3: 0.25}`
```
|    | model                                                                                     |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|-------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  0 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_16_256_GT                       |      1 |    1.33333 |   0.803356 |     2.47652 |   0.946645  |      2.24972 |   0.949421 |     2.60852 |
| 10 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_dil1_cascade_3d_low_res            |     11 |   13.8333  |   0.72286  |     2.85886 |   0.924629  |      3.26189 |   0.932176 |     3.65506 |
| 60 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res  (submitted)    |     13 |   17       |   0.716903 |     3.02594 |   0.924714  |      3.29153 |   0.932519 |     3.63699 |
| 64 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256                                    |     28 |   28.6667  |   0.72061  |     2.72951 |   0.734195  |     56.7844  |   0.931646 |     3.73456 |
| 57 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_oversample_05                      |     59 |   55.8333  |   0.673199 |    23.4793  |   0.859075  |     27.0943  |   0.865722 |    20.4847  |

```

The model trained without a cascade scored better for the wall segmentation, but worse for the left and right atrium segmentation. There also appears to be more outlier false positive segmentations when training the model without a 1st stage binary segmentation.
There still is a degree of overfitting to the training dataset, but it is not as extreme as with the 2nd stage cascaded mdoels. See the loss curve for `mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256`
![image](https://github.com/user-attachments/assets/bd9a311f-536a-4ef5-b676-b15c8aec3717)

### Results on the actual validation set

```
| Submission   | model                                                                                     |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|--------------|-------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  1           | nnUNetTrainer_MedNeXt__MedNeXtPlans__3d_fullres                                           | 1      | 1.667      | 64.47      |  5.54       |  87.42      |  7.85        |  91.15     | 4.73        |
|  2           | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res                 | 2      | 1.833      | 64.72      |  5.36       |  87.75      |  7.61        |  90.89     | 4.98        |
|  3           | mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res                | 3      | 2.5        | 63.69      |  5.51       |  86.99      |  7.76        |  90.75     | 4.86        |
```

`nnUNetTrainer_MedNeXt__MedNeXtPlans__3d_fullres` model showed less overfitting.
![image](https://github.com/user-attachments/assets/e6d0e4d1-c2e6-4a03-a902-b0ecd96e5367)

Comparing to results on fold_0
```
|    | model                                                                                     |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|-------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
| 60 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_13__16_256_cascade_3d_low_res  (submission 2) |     13 |   17       |   0.716903 |     3.02594 |   0.924714  |      3.29153 |   0.932519 |     3.63699 |
| 19 | nnUNetTrainer_MedNeXt__MedNeXtPlans__3d_fullres                            (submission 1) |     20 |   24.3333  |   0.722514 |     3.25905 |   0.923382  |      3.17257 |   0.927485 |     4.59637 |
| 47 | mbasTrainer__MedNeXtV2Plans_2024_08_13__16_256_nblocks2_cascade_3d_low_res (submission 3) |     49 |   44.5     |   0.693119 |     3.66241 |   0.916443  |      3.37182 |   0.922761 |     4.77616 |
```

#### Retrospective
