Many of the trained models appear to suffer from overfitting. The training loss continues to decrease over the training period, but the validation loss begins to increase.
For example, see the loss curves for `mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27`. The validation loss steadily increases from 150 epochs onwards.
![image](https://github.com/user-attachments/assets/aa31aa5e-9e67-46cb-a7ea-ae63cf9ce90a)

In addition to saving the final model checkpoint file "checkpoint_final.pth", the nnU-Net trainer also saves "checkpoint_best.pth", which corresponds to the model weights at the epoch at which the model achieves the minimal validation loss.
The default option for evaluation is to use the "checkpoint_final.pth". However, if the observed divergence between validation loss and training loss actually hurts segmentation performance, then evaluation results for "checkpoint_best.pth"
should be better than "checkpoint_final.pth"
In the following tables I compare the evaluation results for the same models, but using either "checkpoint_final.pth" or "checkpoint_best.pth".

# Checkpoint_final.pth

|    | model                                                                                             |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  9 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27        |     10 |   15.3333  |   0.723445 |     2.72685 |   0.925043  |      3.21474 |   0.931448 |     3.71448 |
| 26 | mbasTrainer__plans_2024_09_02__MedNeXtV2_p16_256_dil2_nblocks346_slim128_cascade_ResEncUNet_08_27 |     28 |   31.5     |   0.718305 |     2.80845 |   0.921939  |      3.31858 |   0.929709 |     4.06074 |


# Checkpoint_best.pth

|    | model                                                                                             |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  0 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27        |      1 |          1 |   0.723269 |     2.74803 |    0.925141 |      3.2737  |   0.931318 |     3.75271 |
|  1 | mbasTrainer__plans_2024_09_02__MedNeXtV2_p16_256_dil2_nblocks346_slim128_cascade_ResEncUNet_08_27 |      2 |          2 |   0.718663 |     2.9586  |    0.922084 |      3.41776 |   0.9286   |     4.06121 |

The "checkpoint_final.pth" weights appear to score better with respect to the evaluation metrics.
