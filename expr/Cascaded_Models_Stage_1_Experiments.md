# 2024-08-21 Experiments
The goal of these experiments is to train a 1st stage binary segmentation model to perform better than the current best performing model, mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres. The motivation to train a better 1st stage model is the observartion that we are able to achieve much higher accuracy in the 2nd stage model if the 1st stage model is the ground truth binary mask rather than the predictions from mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres.  I hypothesize that a very accurate 1st stage model will significantly improve the performance of the 2nd stage model.

I also want to increase the recall of the 1st stage model. It is better for the first stage model to have higher recall and detect all valid foreground regions. The 2nd stage model can clean-up these predictions. 

In this case, the metric we might want to consider is overlap rather than DICE.

|    | model                                                                                     |   Rank |   Avg_Rank |   DSC_atrium |   HD95_atrium |
|----|-------------------------------------------------------------------------------------------|--------|------------|--------------|---------------|
|  0 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres                                 |      1 |        1.5 |     0.934025 |       3.39874 |
|  7 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96   (25% foreground)             |      8 |        8   |     0.931271 |       3.52871 |
|  9 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres           (0% foreground)              |     10 |       10.5 |     0.931683 |       3.75624 |
| 19 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96_foreground_every_other(50% for)|     12 |       11   |     0.930979 |       3.71367 |
| 17 | mbasTrainer__plans_2024_08_21__MedNeXtV1_3d_lowres_slim_96   (25% foreground)             |     13 |       12.5 |     0.930228 |       3.72992 |
| 11 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_128_foreground100_first_430epochs |     14 |       14   |     0.928902 |       3.77922 |
| 12 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_super_slim_96  (25% foreground)        |     15 |       15.5 |     0.928146 |       4.08703 |
| 18 | mbasTrainer__plans_2024_08_21__MedNeXtV1_3d_lowres_slim_96_override_down1                 |     16 |       15.5 |     0.927982 |       3.9724  |
| 13 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_foreground100                          |     17 |       17   |     0.919785 |       5.3186  |
| 14 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_128_foreground100                 |     18 |       18   |     0.908309 |       6.50415 |

All variants of the MedNeXtV2 model scored lower than the `mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres`.
### Oversampling Foreground
Experimented with oversampling the foreground such that 100% of patches are centered on a foreground (heart) voxel.
* mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_foreground100
* mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_128_foreground100
Both of these models were the worst performing -- which implies that oversampling the foreground is not favorable for this task.
The slim_128 variant performed worse.
* mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_128_foreground100_first_430epochs
With this model I oversampled the foreground 100% of the time for the first 430 epochs, and then randomly sample any patch.
* mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96_foreground_every_other
This model gaurentees that 50% of the minibatch is sampled from the foreground. This performs worse than the MedNeXtV2_3d_lowres_slim_96 model whidch randomly samples foreground 25% of the time.
### Slim model size
mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96 achieved better accuracy than MedNeXtV2_3d_lowres
* However, MedNeXtV2_3d_lowres_slim_96 oversamples foreground 25% of the time whereas MedNeXtV2_3d_lowres randomly samples patches (0% oversample).
* TODO: train MedNeXtV2_3d_lowres_slim_96_random100 - randomly sample 100% of the time.
mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_super_slim_96 performs worse than MedNeXtV2_3d_lowres_slim_96
* slim_96 scores better than super_slim_96
* Both of these models oversample foreground 25% of the time
* super_slim_96 trims the `n_blocks_per_stage` and `exp_ratio_per_stage`
```
n_blocks_per_stage: [1,3,4,6,6,6] -> [1,1,1,1,1,1]
exp_ratio_per_stage: [2,3,4,4,4,4] -> [1,1,1,1,1,1]
```
### MedNeXtV1
MedNeXtV1 model variants performed worse than the corresponding MedNeXtV2 models
* MedNeXtV1_3d_lowres_slim_96
* MedNeXtV1_3d_lowres_slim_96_override_down1
Also interesting that the overriding the downsample modules to use 1x1x1 kernels results in decreased performance.

TODO:
train MedNeXtV2_3d_lowres_slim_96_random100 - randomly sample 100% of the time.
train MedNeXtV2_3d_lowres_slim_128_random100 randomly sample 100% of the time
train MedNeXtV2_3d_lowres_foreground25 - oversample foreground 25% of the time

```
configurator = MBASTrainerConfigurator(
    os.path.basename(os.path.splitext(new_config_fpath)[0]),
    dataset_name = "Dataset104_MBAS",
)
configurator.configurations["MedNeXtV2_3d_lowres_slim_96"] = configurator.set_params(
    patch_size=(28,256,224),
    data_identifier = "nnUNetResEncUNetMPlans_3d_lowres",
    spacing = (2.5, 0.9737296353754783, 0.9737296353754783),
    probabilistic_oversampling = True,
    oversample_foreground_percent = 0.25,
).MedNeXtV2(
    features_per_stage = (32, 32, 64, 96, 96, 96),
    kernel_sizes=[
        [1,3,3],
        [3,3,3],
        [3,3,3],
        [3,3,3],
        [3,3,3],
        [3,3,3]
    ],
    strides = [
        [1,1,1],
        [1,2,2],
        [2,2,2],
        [2,2,2],
        [1,2,2],
        [1,2,2]
    ],
    n_blocks_per_stage = [1,2,3,3,3,3],
    exp_ratio_per_stage = [2,3,4,4,4,4],
)

configurator.configurations["MedNeXtV1_3d_lowres_slim_96"] = configurator.set_params(
    patch_size=(28,256,224),
    data_identifier = "nnUNetResEncUNetMPlans_3d_lowres",
    spacing = (2.5, 0.9737296353754783, 0.9737296353754783),
    probabilistic_oversampling = True,
    oversample_foreground_percent = 0.25,
).MedNeXtV1(
    features_per_stage = (32, 32, 64, 96, 96, 96),
    kernel_sizes=[
        [1,3,3],
        [3,3,3],
        [3,3,3],
        [3,3,3],
        [3,3,3],
        [3,3,3]
    ],
    strides = [
        [1,1,1],
        [1,2,2],
        [2,2,2],
        [2,2,2],
        [1,2,2],
        [1,2,2]
    ],
    n_blocks_per_stage = [1,2,3,3,3,3],
    exp_ratio_per_stage = [2,3,4,4,4,4],
    override_down_kernel_size = False
)
```

# 2024-08-23
### Different patch sizes 
* MedNeXtV2_3d_lowres_p32_256_slim_96
* mbasTrainer__plans_2024_08_23__MedNeXtV2_3d_lowres_p16_256_slim_96
### Larger filter sizes 5x5x5
* MedNeXtV2_3d_lowres_p32_256_slim_96_stem5
Oversample foreground 25%.
`stem_kernel_size=(1,5,5)`. 
* MedNeXtV2_3d_lowres_p32_256_slim_96_k5_stem5
Oversample foreground 25%.
`stem_kernel_size=(1,5,5)`.
Add some 5x5 kernels
```
kernel_sizes=[
    [1,5,5],
    [3,5,5],
    [5,5,5],
    [5,5,5],
    [3,3,3],
    [3,3,3],
    [3,3,3]
]
```
