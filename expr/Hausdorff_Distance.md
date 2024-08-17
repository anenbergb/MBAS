# Hausdorff Distance Experiments

## Introduce Hausdorff Distance loss
This experiment did not work!
I integrated the HausdorffDTLoss implemented in monai into nnUNetv2 and followed the training policy of https://arxiv.org/abs/1904.10030 and https://arxiv.org/pdf/2302.03868v3 where I just increased the relative weight of the Hausdorff Distance (HD) loss from 0 to 100% using a stepwise linear function, updated every 5 (or 10) epochs.
```
|    | model                                                                                       |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  4 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_21__slim_128                                    |      5 |    9.16667 |   0.723578 |     3.32642 |   0.924154  |      3.10856 |   0.927038 |     4.50409 |
| 25 | nnUNetTrainer_MedNeXt_CE_DC_HD__MedNeXtPlans_2024_07_26__slim_128_alpha10                   |     29 |   28.6667  |   0.592409 |     9.04654 |   0.886964  |      5.78562 |   0.905886 |     8.65668 |
| 26 | nnUNetTrainer_MedNeXt_CE_DC_HD__MedNeXtPlans_2024_07_26__slim_128_alpha05                   |     30 |   29.6667  |   0.551918 |    10.1797  |   0.869174  |      7.87584 |   0.895517 |     9.4183  |
```

## Experiments Titrating the Hausdorff Lossf
Previously I integrated the HausdorffDTLoss implemented in monai into nnUNetv2 and followed the training policy of https://arxiv.org/abs/1904.10030 and https://arxiv.org/pdf/2302.03868v3 where I just increased the relative weight of the Hausdorff Distance (HD) loss from 0 to 100% using a stepwise linear function, updated every 5 (or 10) epochs. This prior round of experiments is captured below (models #51 and #53 in the table below).
With second round of experiments I added a 250 (or 500) epoch warmup such that the HD loss would not be added until the model trained at least 250 (or 500) epochs with the normal losses -- Cross Entropy loss and Dice Loss. Only after 250 (or 500) epochs would the HD loss be added. The weight of the HD loss would increase via a stepwise linear function updated every 5 epochs to a maximum of 0.25 (or 0.50, or 0.75) weight balanced agains the Cross Entropy and Dice Loss. I.e. `loss = (1 - alpha) * (CE_loss + Dice_loss) + alpha * HD_loss`. The `scaled` suffix to the below experiments refers to scaling up the HD_loss weight (alpha) from 0 to the maximum (e.g. 0.25) over the full range of steps from warmup epoch (e.g. 250) to final epoch (e.g. 1000). With scaled, the alpha equation is `alpha = max_alpha * (steps / total_steps)` whereas without scaled the alpha equation is `min(steps / total_steps, max_alpha)`.

This round of experiments were unsuccessful. The models with added HD loss performed worse. 

```
|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  2 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   10       |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  3 | nnUNetTrainer_MedNeXt__MedNeXtPlans_2024_07_27__slim_128_oversample_05                                        |     10 |   11       |   0.723894 |     2.84257 |   0.925716  |      3.03093 |   0.932273 |     3.93802 |
| 31 | nnUNetTrainer_MedNeXt_CE_DC_HD__MedNeXtPlans_2024_08_03__slim_128_oversample_05_alpha05_warm500_max025_scaled |     35 |   35.8333  |   0.716094 |     3.93379 |   0.919007  |      3.78956 |   0.927351 |     4.3387  |
| 47 | nnUNetTrainer_MedNeXt_CE_DC_HD__MedNeXtPlans_2024_08_03__slim_128_oversample_05_alpha05_warm250_max050_scaled |     51 |   47.3333  |   0.701808 |     4.83751 |   0.914622  |      4.34904 |   0.924663 |     4.9939  |
| 48 | nnUNetTrainer_MedNeXt_CE_DC_HD__MedNeXtPlans_2024_08_03__slim_128_oversample_05_alpha05_warm250_max050        |     52 |   48.1667  |   0.693073 |     5.14332 |   0.912897  |      4.2309  |   0.922697 |     4.68495 |
| 50 | nnUNetTrainer_MedNeXt_CE_DC_HD__MedNeXtPlans_2024_08_03__slim_128_oversample_05_alpha05_warm250_max075        |     54 |   51.1667  |   0.677776 |     6.02175 |   0.908245  |      4.26035 |   0.920219 |     5.22969 |
| 51 | nnUNetTrainer_MedNeXt_CE_DC_HD__MedNeXtPlans_2024_07_26__slim_128_alpha10                                     |     55 |   53.3333  |   0.592409 |     9.04654 |   0.886964  |      5.78562 |   0.905886 |     8.65668 |
| 53 | nnUNetTrainer_MedNeXt_CE_DC_HD__MedNeXtPlans_2024_07_26__slim_128_alpha05                                     |     56 |   54.6667  |   0.551918 |    10.1797  |   0.869174  |      7.87584 |   0.895517 |     9.4183  |
```
