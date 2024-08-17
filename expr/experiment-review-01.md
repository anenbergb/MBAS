I've run quite a few experiments over the past few weeks with the goal of improving Dice and HD95 scores. Some of the minor architectural changes helped and the updated patch sampling policy helped. However, training with Hausdorff Distance loss and 2-stage Cascaded models didn't help.

## Experiment 1: Various MedNeXt minor architectural changes.
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


## Experiment 2: Modify patch sampling policy at training time
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
* Oversampling 85% of the time (slim_128_oversample_for085) performed worse than oversampling 100% of the time (slim_128_oversample_05) or no oversampling.
* Adding additional feature channels to the early stages hurt performance.

```
|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  9 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   11.5     |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 16 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__slim_128                                                      |     18 |   23.3333  |   0.723578 |     3.32642 |   0.924154  |      3.10856 |   0.927038 |     4.50409 |
| 15 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_08_01__slim_128_oversample_for085                                    |     19 |   23.3333  |   0.722418 |     3.16222 |   0.922322  |      3.30027 |   0.928473 |     4.30358 |
| 33 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_08_01__early64_late128_oversample                                    |     37 |   36       |   0.718049 |     3.02914 |   0.664714  |     78.3079  |   0.928878 |     4.3061  |
```
<img width="716" alt="image" src="https://github.com/user-attachments/assets/8c598b31-e77f-48f1-9743-ac1e49a5f29a">
