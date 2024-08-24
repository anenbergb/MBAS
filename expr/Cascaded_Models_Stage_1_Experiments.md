# 2024-08-21 Experiments

```
configurator = MBASTrainerConfigurator(
    os.path.basename(os.path.splitext(new_config_fpath)[0]),
    dataset_name = "Dataset104_MBAS",
)

configurator.configurations["MedNeXtV2_3d_lowres"] = configurator.set_params(
    patch_size=(28,256,224),
    data_identifier = "nnUNetResEncUNetMPlans_3d_lowres",
    spacing = (2.5, 0.9737296353754783, 0.9737296353754783),
    probabilistic_oversampling = False,
    oversample_foreground_percent = 0.0,
).MedNeXtV2(
    features_per_stage = (32, 64, 128, 256, 320, 320),
    kernel_sizes=[
        [1, 3, 3],
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
    n_blocks_per_stage = [1,3,4,6,6,6],
    exp_ratio_per_stage = [2, 3, 4, 4, 4, 4],
)

configurator.configurations["MedNeXtV2_3d_lowres_foreground100_first_400epochs"] = copy.deepcopy(configurator.configurations["MedNeXtV2_3d_lowres"])

configurator.configurations["MedNeXtV2_3d_lowres_foreground100"] = copy.deepcopy(configurator.configurations["MedNeXtV2_3d_lowres"])
configurator.configurations["MedNeXtV2_3d_lowres_foreground100"]["oversample_foreground_percent"] = 1.0

configurator.configurations["MedNeXtV2_3d_lowres_slim_128"] = copy.deepcopy(configurator.configurations["MedNeXtV2_3d_lowres"])
configurator.configurations["MedNeXtV2_3d_lowres_slim_128"]["architecture"]["arch_kwargs"]["features_per_stage"] = (32, 64, 96, 128, 128, 128)

configurator.configurations["MedNeXtV2_3d_lowres_slim_128_foreground100_first_430epochs"] = copy.deepcopy(configurator.configurations["MedNeXtV2_3d_lowres_slim_128"])

configurator.configurations["MedNeXtV2_3d_lowres_slim_128_foreground100"] = copy.deepcopy(configurator.configurations["MedNeXtV2_3d_lowres_slim_128"])
configurator.configurations["MedNeXtV2_3d_lowres_slim_128_foreground100"]["oversample_foreground_percent"] = 1.0



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
configurator.configurations["MedNeXtV2_3d_lowres_super_slim_96"] = configurator.set_params(
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
    n_blocks_per_stage = [1,1,1,1,1,1],
    exp_ratio_per_stage = [1,1,1,1,1,1],
)


configurator.configurations["MedNeXtV2_3d_lowres_slim_96_foreground_every_other"] = copy.deepcopy(configurator.configurations["MedNeXtV2_3d_lowres_slim_96"])
configurator.configurations["MedNeXtV2_3d_lowres_slim_96_foreground_every_other"]["probabilistic_oversampling"] = False
configurator.configurations["MedNeXtV2_3d_lowres_slim_96_foreground_every_other"]["oversample_foreground_percent"] = 0.5


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

configurator.configurations["MedNeXtV1_3d_lowres_slim_96_override_down1"] = copy.deepcopy(configurator.configurations["MedNeXtV1_3d_lowres_slim_96"])
configurator.configurations["MedNeXtV1_3d_lowres_slim_96_override_down1"]["architecture"]["arch_kwargs"]["override_down_kernel_size"] = True
```
