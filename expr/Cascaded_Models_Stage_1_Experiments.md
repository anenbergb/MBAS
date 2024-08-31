# 2024-08-21 Experiments
The goal of these experiments is to train a 1st stage binary segmentation model to perform better than the current best performing model, mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres. The motivation to train a better 1st stage model is the observartion that we are able to achieve much higher accuracy in the 2nd stage model if the 1st stage model is the ground truth binary mask rather than the predictions from mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres.  I hypothesize that a very accurate 1st stage model will significantly improve the performance of the 2nd stage model.

I also want to increase the recall of the 1st stage model. It is better for the first stage model to have higher recall and detect all valid foreground regions. The 2nd stage model can clean-up these predictions. 

In this case, the metric we might want to consider is overlap rather than DICE.

|    | model                                                                                     |   Rank |   Avg_Rank |   DSC_atrium |   OVERLAP_atrium |   HD95_atrium |
|----|-------------------------------------------------------------------------------------------|--------|------------|--------------|------------------|---------------|
|  7 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres                                 |      1 |        2   |     0.934025 |         0.934561 |       3.39874 |
| 12 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96 (25% foreground)               |     10 |       11   |     0.931271 |         0.92968  |       3.52871 |
| 30 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_foreground25 (25% foreground           |     13 |       13   |     0.932545 |         0.930725 |       3.67657 |
| 17 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96_foreground0 (0% foreground)    |     15 |       15   |     0.931068 |         0.927779 |       3.65323 |
| 19 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres (0% foreground)                        |     16 |       15.5 |     0.931683 |         0.927469 |       3.75624 |
| 10 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96_foreground_every_other(50% for)|     17 |       16   |     0.930979 |         0.93047  |       3.71367 |
| 28 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_128_foreground0 (0% foreground)   |     20 |       19.5 |     0.929109 |         0.925345 |       3.74357 |
| 24 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_128_foreground100_first_430epochs |     21 |       21.5 |     0.928902 |         0.923285 |       3.77922 |
| 21 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_super_slim_96 (25% foreground)         |     25 |       24.5 |     0.928146 |         0.925638 |       4.08703 |
| 22 | mbasTrainer__plans_2024_08_21__MedNeXtV1_3d_lowres_slim_96_override_down1                 |     26 |       24.5 |     0.927982 |         0.924272 |       3.9724  |
| 15 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_foreground100                          |     27 |       27   |     0.919785 |         0.928459 |       5.3186  |
| 18 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_128_foreground100                 |     28 |       28   |     0.908309 |         0.927479 |       6.50415 |


All variants of the MedNeXtV2 model scored lower than the `mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres`.
### Oversampling Foreground
Experimented with oversampling the foreground such that 100% of patches are centered on a foreground (heart) voxel.
* mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_foreground100
* mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_128_foreground100
Both of these models were the worst performing -- which implies that oversampling the foreground is not favorable for this task.

### Increasing foreground oversampling from 0 -> 25 -> 100%
The ordering of best models according to oversampling
* 25%: mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96
* 0 percent: mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96_foreground0
* 50% (every other): mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96_foreground_every_other
So the 25% oversampling foreground is better than 0%.
The foreground every other model gaurentees that 50% of the minibatch is sampled from the foreground. This performs worse than the MedNeXtV2_3d_lowres_slim_96 model whidch randomly samples foreground 25% of the time.


The ordering of best models according to oversampling
* 0 percent: mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_128_foreground0
* 100% for the first 430 epochs; mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_128_foreground100_first_430epochs
* 100% mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_128_foreground100
So 0% oversampling foreground is better than 100%.
The foreground100_first_430 epochsmodel oversampled the foreground 100% of the time for the first 430 epochs, and then randomly sample any patch.

The ordering of best models according to oversampling
* 25%: mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_foreground25
* 0 percent: mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres (0% foreground)
* 100%: mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_foreground100
So the 25% oversampling foreground is best.


### Slim model size
* mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96_foreground0 achieved better accuracy than MedNeXtV2_3d_lowres (0% oversampling)
* mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96 (25% foreground) MAYBE achieves better accuracy than mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_foreground25

### Super Slim model dize
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
The MedNeXtV2_3d_lowres_slim_96 model uses patch size `(28,256,224)`. 
I experimented with differnet patch sizes such as (32,256,256) and (16,256,256).
I also experimented with increasing the kernel size of the stem from (1,3,3) to (1,5,5).

k5_stem5 adds 5x5 kernels instead of 3x3 kernels.
```
kernel_sizes=[
    [1,5,5],
    [3,5,5],
    [5,5,5],
    [5,5,5],
    [3,3,3],
    [3,3,3],
    [3,3,3]
```
It seems to be the case that the patch size (28,256,224) is the best, and that adding 5x5 kernels doesn't help.
All of the models were trained with 25% probabilistic foreground oversampling.

|    | model                                                                                     |   Rank |   Avg_Rank |   DSC_atrium |   OVERLAP_atrium |   HD95_atrium |
|----|-------------------------------------------------------------------------------------------|--------|------------|--------------|------------------|---------------|
|  6 | mbasTrainer__plans_2024_08_21__MedNeXtV2_3d_lowres_slim_96                                |      7 |        8   |     0.931271 |         0.92968  |       3.52871 |
| 24 | mbasTrainer__plans_2024_08_23__MedNeXtV2_3d_lowres_p32_256_slim_96_stem5                  |     12 |       12   |     0.929928 |         0.927175 |       3.55324 |
| 10 | mbasTrainer__plans_2024_08_23__MedNeXtV2_3d_lowres_p32_256_slim_96                        |     15 |       14   |     0.929065 |         0.928198 |       3.63267 |
| 15 | mbasTrainer__plans_2024_08_23__MedNeXtV2_3d_lowres_p16_256_slim_96                        |     18 |       18.5 |     0.92886  |         0.923015 |       3.9424  |
| 23 | mbasTrainer__plans_2024_08_23__MedNeXtV2_3d_lowres_p32_256_slim_96_k5_stem5               |     19 |       19.5 |     0.924232 |         0.923442 |       3.85475 |


# 2024-08-24
A few experiments with ResEncUNet_3d_lowres models
|    | model                                                                                     |   Rank |   Avg_Rank |   DSC_atrium |   OVERLAP_atrium |   HD95_atrium |
|----|-------------------------------------------------------------------------------------------|--------|------------|--------------|------------------|---------------|
|  1 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres                                 |      1 |        2   |     0.934025 |         0.934561 |       3.39874 |
| 25 | mbasTrainer__plans_2024_08_24__ResEncUNet_3d_lowres_for25                                 |      2 |        2.5 |     0.934184 |         0.935395 |       3.42663 |
| 29 | mbasTrainer__plans_2024_08_24__ResEncUNet_3d_lowres_k5_for25                              |      6 |        7   |     0.932984 |         0.931738 |       3.47172 |
|  1 | mbasTrainer__plans_2024_08_24__ResEncUNet_3d_lowres_for25_drop25                          |      8 |        8.5 |     0.931497 |         0.94638  |       3.46768 |
|  0 | mbasTrainer__plans_2024_08_24__ResEncUNet_3d_lowres_for25_drop50                          |     22 |       22   |     0.923642 |         0.957245 |       3.71879 |

ResEncUNet_3d_lowres_k5_for25 uses the same 5x5 kenrnels as in the MedNeXV2 experiments above
* Adding 5x5 kernels yields slightly lower DSC, overlap, and HD95. 
```
kernel_sizes=[
    [1, 5, 5],
    [3, 5, 5],
    [5, 5, 5],
    [3, 3, 3],
    [3, 3, 3],
    [3, 3, 3],
],
```

# 2024-08-27
The following experiments attempt to train a ResEncUNet_3d_lowres to achieve higher OVERLAP (i.e. recall). The idea is to reduce the number of parameters of the model and add dropout.

ResEncUNet_3d_lowres_for25_drop50_slim128
```
features_per_stage=(32, 64, 128, 128, 128, 128),
```
ResEncUNet_3d_lowres_for25_drop50_slim96
```
features_per_stage=(32, 32, 64, 96, 96, 96),
n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
```
ResEncUNet_3d_lowres_for25_drop50_slim96_nblocks4
```
features_per_stage=(32, 32, 64, 96, 96, 96),
n_blocks_per_stage=[1, 3, 4, 4, 4, 4],
```

When these models are ranked with respect to overlap they achieve higher overlap than the base model `ResEncUNet_3d_lowres_for25_drop50`.
|    | model                                                                                        |   Rank |   Avg_Rank |   DSC_atrium |   OVERLAP_atrium |   HD95_atrium |
|----|----------------------------------------------------------------------------------------------|--------|------------|--------------|------------------|---------------|
| 12 | mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96                      |      1 |          1 |     0.917289 |         0.961467 |       4.07259 |
| 15 | mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96_nblocks4             |      2 |          2 |     0.91771  |         0.961383 |       4.09641 |
| 11 | mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim128                     |      3 |          3 |     0.924305 |         0.958233 |       3.81758 |
|  9 | mbasTrainer__plans_2024_08_24__ResEncUNet_3d_lowres_for25_drop50                             |      4 |          4 |     0.923642 |         0.957245 |       3.71879 |


The other experiment that were run include
MedNeXtV2_3d_lowres_for25_drop50
- 50% dropout probability
MedNeXtV2_3d_lowres_for25_drop25
- 25% dropout probability
MedNeXtV2_3d_lowres_for25_drop50_stemStacked
- replace the basic Stem (Conv + GroupNorm) with StackedConvBlocks, which includes dropout, instance norm, and a LeakyReLU nonlinearity function.
MedNeXtV2_3d_lowres_for25_drop50_stemStacked_decoderConvTrans
- In addiiton to replacing the Stem with StackedConvBlocks, also replaces the upsampling MedNeXt blocks with simple ConvTranspose3d operations, which should reduce the number of parameters.
MedNeXtV2_3d_lowres_for25_drop50_decoderConvTrans
- Only replaces the upsampling MedNeXt blocks with simple ConvTranspose3d operations.

It appears that replacing the MedNeXt upsampling blocks with ConvTranspose3D blocks had the largest impact in increasing the Overlap (recall) but at significant cost to DICE score (accuracy). 


|    | model                                                                                        |   Rank |   Avg_Rank |   DSC_atrium |   OVERLAP_atrium |   HD95_atrium |
|----|----------------------------------------------------------------------------------------------|--------|------------|--------------|------------------|---------------|
| 20 | mbasTrainer__plans_2024_08_27__MedNeXtV2_3d_lowres_for25_drop50_decoderConvTrans             |      5 |          5 |     0.899378 |         0.951496 |       8.45093 |
| 24 | mbasTrainer__plans_2024_08_27__MedNeXtV2_3d_lowres_for25_drop50_stemStacked_decoderConvTrans |      7 |          7 |     0.879802 |         0.945808 |      14.8759  |
| 27 | mbasTrainer__plans_2024_08_27__MedNeXtV2_3d_lowres_for25_drop25                              |     24 |         24 |     0.927263 |         0.928424 |       4.1716  |
| 35 | mbasTrainer__plans_2024_08_27__MedNeXtV2_3d_lowres_for25_drop50                              |     36 |         36 |     0.919822 |         0.921    |       4.98324 |
| 36 | mbasTrainer__plans_2024_08_27__MedNeXtV2_3d_lowres_for25_drop50_stemStacked                  |     37 |         37 |     0.894228 |         0.913912 |       8.22311 |


# Conclusion
The conclusion from this cohort of experiments is to train `mbasTrainer__plans_2024_08_27__ResEncUNet_3d_lowres_for25_drop50_slim96` for the Stage 1 binary segmentation model. I will also try to add additional postprocessing to the binary segmentation model results to further boost the Overlap (recall).
