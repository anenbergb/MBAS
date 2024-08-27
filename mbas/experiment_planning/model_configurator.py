import copy
import json


class MBASTrainerConfigurator:
    def __init__(self, plans_name, dataset_name="Dataset101_MBAS", config=None):
        """
        config should be the full config dictionary, i.e. nnUNetResEncUNetMPlans.json

        """
        if config is None:
            self.config = MBASTrainerConfigurator.get_default_config()
        elif isinstance(config, str):
            with open(config, "r") as f:
                self.config = json.load(f)
        else:
            self.config = copy.deepcopy(config)
        self.config["plans_name"] = plans_name
        self.config["dataset_name"] = dataset_name
        self.current_configuration = None

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.config, f, indent=2)

    @staticmethod
    def get_default_config():
        return {
            "dataset_name": "Dataset101_MBAS",
            "plans_name": "my_plans",
            "original_median_spacing_after_transp": [2.5, 0.625, 0.625],
            "original_median_shape_after_transp": [44, 592, 581],
            "image_reader_writer": "SimpleITKIO",
            "transpose_forward": [0, 1, 2],
            "transpose_backward": [0, 1, 2],
            "configurations": {},
            "experiment_planner_used": "MedNeXtPlanner",
            "label_manager": "LabelManager",
            "foreground_intensity_properties_per_channel": {
                "0": {
                    "max": 2101.0,
                    "mean": 362.0030822753906,
                    "median": 305.0,
                    "min": 0.0,
                    "percentile_00_5": 21.0,
                    "percentile_99_5": 1399.0,
                    "std": 260.2959289550781,
                }
            },
        }

    @property
    def configurations(self):
        return self.config["configurations"]

    def set_params(
        self,
        batch_size=2,
        patch_size=(16, 256, 256),
        data_identifier="MedNeXtPlans_3d_fullres",
        spacing=(2.5, 0.625, 0.625),
        boundary_loss_alpha_stepsize=5,
        boundary_loss_alpha_warmup_epochs=500,
        boundary_loss_alpha_max=0.25,
        alpha_stepwise_warmup_scaled=True,
        probabilistic_oversampling=False,
        oversample_foreground_percent=0.0,
        sample_class_probabilities={1: 0.5, 2: 0.25, 3: 0.25},
        batch_dice=False,
    ):
        """
        batch_dice: whether to use batch dice (pretend all samples in the batch are one image, compute dice loss over that)
        or not (each sample in the batch is a separate image, compute dice loss for each sample and average over samples)
            It was 'False' for 3d_lowres by default, but true for the other models.

        patch_size = (28, 256, 224),
        data_identifier = "MedNeXtPlans_3d_fullres",
        spacing = (2.5, 0.9737296353754783, 0.9737296353754783),
        """
        args_dict = locals()
        self.current_configuration = {
            "median_image_size_in_voxels": [44.0, 592.5, 581.0],
            "normalization_schemes": ["ZScoreNormalization"],
            "use_mask_for_norm": [False],
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": {
                "is_seg": False,
                "order": 3,
                "order_z": 0,
                "force_separate_z": None,
            },
            "resampling_fn_seg_kwargs": {
                "is_seg": True,
                "order": 1,
                "order_z": 0,
                "force_separate_z": None,
            },
            "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
            "resampling_fn_probabilities_kwargs": {
                "is_seg": False,
                "order": 1,
                "order_z": 0,
                "force_separate_z": None,
            },
        }
        for k, v in args_dict.items():
            if k != "self":
                self.current_configuration[k] = v
        return self

    def set_cascade(
        self,
        cascaded_mask_dilation=0,
        is_cascaded_mask=True,
        previous_stage="Dataset104_2024_08_10_3d_lowres",
    ):
        self.current_configuration["cascaded_mask_dilation"] = cascaded_mask_dilation
        self.current_configuration["is_cascaded_mask"] = is_cascaded_mask
        self.current_configuration["previous_stage"] = previous_stage
        return self

    def MedNeXtV1(
        self,
        features_per_stage=(32, 64, 128, 256, 320, 320, 320),
        stem_kernel_size=(1, 1, 1),
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
            [2, 2, 2],
        ],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6, 6],
        exp_ratio_per_stage=[2, 3, 4, 4, 4, 4, 4],
        n_blocks_per_stage_decoder=None,
        exp_ratio_per_stage_decoder=None,
        norm_type="group",
        enable_affine_transform=False,
        decode_stem_kernel_size=3,
        override_down_kernel_size=False,
        down_kernel_size=1,
    ):
        args_dict = locals()

        n_stages = len(features_per_stage)
        arch = {
            "network_class_name": "mbas.architectures.MedNeXt.MedNeXt",
            "_kw_requires_import": ["conv_op"],
            "arch_kwargs": {
                "conv_op": "torch.nn.modules.conv.Conv3d",
                "n_stages": n_stages,
            },
        }

        assert len(kernel_sizes) == n_stages
        assert len(strides) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(exp_ratio_per_stage) == n_stages
        if n_blocks_per_stage_decoder is None:
            args_dict["n_blocks_per_stage_decoder"] = n_blocks_per_stage[:-1][::-1] + [
                n_blocks_per_stage[0]
            ]
        assert len(args_dict["n_blocks_per_stage_decoder"]) == n_stages
        if exp_ratio_per_stage_decoder is None:
            args_dict["exp_ratio_per_stage_decoder"] = exp_ratio_per_stage[:-1][
                ::-1
            ] + [exp_ratio_per_stage[0]]
        assert len(args_dict["exp_ratio_per_stage_decoder"]) == n_stages

        for k, v in args_dict.items():
            if k != "self":
                arch["arch_kwargs"][k] = v

        self.current_configuration["architecture"] = arch
        return self.current_configuration

    def MedNeXtV2(
        self,
        features_per_stage=(32, 64, 128, 256, 320, 320, 320),
        stem_kernel_size=(1, 3, 3),
        stem_channels=None,
        stem_dilation=1,
        stem_type: str = "conv",
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
        dilation_per_stage=1,
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6, 6],
        exp_ratio_per_stage=[2, 3, 4, 4, 4, 4, 4],
        n_blocks_per_stage_decoder=None,
        exp_ratio_per_stage_decoder=None,
        norm_type="group",
        enable_affine_transform=False,
        decoder_cat_skip=False,
        decoder_conv_trans_up=False,
        dropout_op=None,
        dropout_op_kwargs=None,
    ):
        args_dict = locals()

        n_stages = len(features_per_stage)
        arch = {
            "network_class_name": "mbas.architectures.MedNeXtV2.MedNeXtV2",
            "_kw_requires_import": ["conv_op", "dropout_op"],
            "arch_kwargs": {
                "conv_op": "torch.nn.modules.conv.Conv3d",
                "n_stages": n_stages,
            },
        }

        assert len(kernel_sizes) == n_stages
        assert len(strides) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(exp_ratio_per_stage) == n_stages
        if n_blocks_per_stage_decoder is None:
            args_dict["n_blocks_per_stage_decoder"] = n_blocks_per_stage[:-1][::-1]
        assert len(args_dict["n_blocks_per_stage_decoder"]) == n_stages - 1
        if exp_ratio_per_stage_decoder is None:
            args_dict["exp_ratio_per_stage_decoder"] = exp_ratio_per_stage[:-1][::-1]
        assert len(args_dict["exp_ratio_per_stage_decoder"]) == n_stages - 1

        for k, v in args_dict.items():
            if k != "self":
                arch["arch_kwargs"][k] = v

        self.current_configuration["architecture"] = arch
        return self.current_configuration

    def nnUNetResEncUNet(
        self,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        kernel_sizes=[
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
            [2, 2, 2],
            [2, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
        ],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        dropout_op=None,
        dropout_op_kwargs=None,
    ):
        args_dict = locals()

        n_stages = len(features_per_stage)
        arch = {
            "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
            "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
            "arch_kwargs": {
                "conv_op": "torch.nn.modules.conv.Conv3d",
                "n_stages": n_stages,
                "conv_bias": True,
                "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": "torch.nn.LeakyReLU",
                "nonlin_kwargs": {"inplace": True},
            },
        }

        assert len(kernel_sizes) == n_stages
        assert len(strides) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == n_stages - 1

        for k, v in args_dict.items():
            if k != "self":
                arch["arch_kwargs"][k] = v

        self.current_configuration["architecture"] = arch
        return self.current_configuration
