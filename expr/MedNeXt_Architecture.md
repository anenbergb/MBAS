I've run quite a few experiments over the past few weeks with the goal of improving Dice and HD95 scores. Some of the minor architectural changes helped and the updated patch sampling policy helped. However, training with Hausdorff Distance loss and 2-stage Cascaded models didn't help.

## Early MedNeXt experiments

* `3d_01` patch size `(32, 256, 256)`, features per stage `(64, 128, 256, 256)`, kernel sizes of `(1,5,5), (3,5,5), (3,5,5), (1,5,5)`, stem kernel = 2, n_blocks_per_stage = [3,3,9,3], decoder only has 1x blocks and 1x expansion
* `3d_02` is same as `3d_01` but with stem_kernel_size=(2,4,4) and more stride in first block, `(2,4,4)` rather than `(2,2,2)`.
* `3d_03` has patch size `(16, 256, 256)` but with stem_kernel_size = 1, 5x5 kernels, and a slim decoder.
* baseline_no_override patch size `(16, 256, 256)` and override_down_kernel_size = False. 7 stages. 3x3 convs
```
|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|

| 14 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_18__3d_01                                                         |     17 |   22.8333  |   0.712298 |     3.00992 |   0.921867  |      3.17019 |   0.926647 |     3.83216 |
| 22 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_18__baseline_no_override                                          |     25 |   26.3333  |   0.722378 |     3.1792  |   0.922252  |      3.25888 |   0.927101 |     4.60908 |
| 24 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_18__3d_02                                                         |     26 |   27.8333  |   0.704201 |     2.93458 |   0.919135  |      3.31624 |   0.926252 |     3.96765 |
| 25 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_18__3d_04                                                         |     28 |   29.1667  |   0.7021   |     3.11244 |   0.920124  |      3.2303  |   0.925533 |     4.25802 |
| 36 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_18__3d_03                                                         |     39 |   36.5     |   0.71435  |     3.27504 |   0.916747  |      3.76198 |   0.925979 |     4.36965 |
```

## Various MedNeXt minor architectural changes.
The best performing model was one with input patch size (16, 256, 256), 7 stages that ultimately downsample the input to (1, 4, 4), retained symmetric number of blocks and expansion ratios in the U-Net decoder layers as in the U-Net encoder, and capped the maximum number of channels in the final layers to only 128! features_per_stage: (32, 64, 128, 128, 128, 128, 128)
* decoder_1_block limited `n_blocks_per_stage_decoder = [1] * 7`
* decoder_1_exp_ratio limited `exp_ratio_per_stage_decoder = [1] * 7`
```
|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
| 16 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__slim_128                                                      |     18 |   23.3333  |   0.723578 |     3.32642 |   0.924154  |      3.10856 |   0.927038 |     4.50409 |
| 18 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__decoder_1_exp_ratio                                           |     21 |   24.1667  |   0.723517 |     3.21192 |   0.923799  |      3.25422 |   0.927247 |     4.623   |
| 20 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__decoder_1_block                                               |     22 |   25.6667  |   0.72226  |     3.17636 |   0.921041  |      3.21601 |   0.926832 |     4.39169 |
```


## Modify patch sampling policy at training time
This experiment worked! Results improved across all metrics.
The default nnUNet patch sampling policy is random sampling. I tried a new policy to 100% of the time sample patches containing foreground object. 50% of the time centered on a "Atrium Wall", 25% of the time centered on "Right Atrium" and 25% of the time centered on "Left Atrium". This worked better than oversampling the "Atrium Wall" 80% of the time and the other two regions 10% of the time.
```

|    | model                                                                                       |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                      |     10 |   11.5     |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 16 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__slim_128                                    |     18 |   23.3333  |   0.723578 |     3.32642 |   0.924154  |      3.10856 |   0.927038 |     4.50409 |
| 28 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_08                      |     31 |   33.8333  |   0.719572 |     3.10795 |   0.733692  |     53.5099  |   0.929041 |     4.01647 |

```

## 2024-08-01 MedNeXt 128 feature per stage experiments
Performed a collection of experiments with the MedNeXt network on (16, 256, 256) size input. Number of features per stage were capped at 128.
Conclusions
* Oversampling 85% of the time (slim_128_oversample_for085) performed worse than oversampling 100% of the time (slim_128_oversample_05) or no oversampling. Both training runs used the same policy of sampling patches centered at the "Atrium Wall" 50% of the time.
* Adding additional feature channels to the early stages hurt performance, possibly due to overfitting.

```
|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   11.5     |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 16 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__slim_128                                                      |     18 |   23.3333  |   0.723578 |     3.32642 |   0.924154  |      3.10856 |   0.927038 |     4.50409 |
| 15 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_08_01__slim_128_oversample_for085                                    |     19 |   23.3333  |   0.722418 |     3.16222 |   0.922322  |      3.30027 |   0.928473 |     4.30358 |
| 33 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_08_01__early64_late128_oversample                                    |     37 |   36       |   0.718049 |     3.02914 |   0.664714  |     78.3079  |   0.928878 |     4.3061  |
```
<img width="716" alt="image" src="https://github.com/user-attachments/assets/8c598b31-e77f-48f1-9743-ac1e49a5f29a">
