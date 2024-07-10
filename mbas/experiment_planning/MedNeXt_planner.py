import shutil
import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple

from mbas.architectures.MedNeXt import MedNeXt
from batchgenerators.utilities.file_and_folder_operations import (
    load_json,
    join,
    save_json,
    isfile,
    maybe_mkdir_p,
)
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
from torch import nn

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import (
    ExperimentPlanner,
)

from nnunetv2.experiment_planning.experiment_planners.network_topology import (
    get_pool_and_conv_props,
)
from nnunetv2.preprocessing.resampling.default_resampling import (
    resample_data_or_seg_to_shape,
    compute_new_shape,
)
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed


class MedNeXtPlanner(ExperimentPlanner):
    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        gpu_memory_target_in_gb: float = 24,
        preprocessor_name: str = "DefaultPreprocessor",
        plans_name: str = "MedNeXtPlans",
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb,
            preprocessor_name,
            plans_name,
            overwrite_target_spacing,
            suppress_transpose,
        )
        self.UNet_class = MedNeXt
        self.featuremap_min_edge_length = 1
        self.UNet_min_batch_size = 2
        self.UNet_base_num_features = 32
        self.UNet_max_features_2d = 512
        self.UNet_max_features_3d = 320
        self.UNet_n_blocks_per_stage = (3, 4, 6, 6, 6, 6, 6, 6, 6, 6)
        self.UNet_n_blocks_per_stage_decoder = (6, 6, 6, 6, 6, 6, 6, 4, 3)
        self.UNet_exp_ratio_per_stage = (2, 3, 4, 4, 4, 4, 4, 4, 4, 4)
        self.UNet_exp_ratio_per_stage_decoder = (4, 4, 4, 4, 4, 4, 4, 3, 2)

        self.UNet_reference_val_2d = 2973327296  # IDK why this value
        self.UNet_reference_val_3d = 2973327296  # 2.77 GB
        self.UNet_reference_val_corresp_GB = 2.77

    def get_plans_for_configuration(
        self,
        spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
        median_shape: Union[np.ndarray, Tuple[int, ...]],
        data_identifier: str,
        approximate_n_voxels_dataset: float,
        _cache: dict,
    ) -> dict:
        """

        approximate_n_voxels_dataset: (# items in train dataset) * np.prod([median_shape])
                                      70 * np.prod([ 44. , 592.5, 581. ])
        """

        def _features_per_stage(num_stages, max_num_features) -> Tuple[int, ...]:
            # example: (32, 64, 128, 256, 512)
            return tuple(
                [
                    min(max_num_features, self.UNet_base_num_features * 2**i)
                    for i in range(num_stages)
                ]
            )

        def _keygen(patch_size, strides):
            return str(patch_size) + "_" + str(strides)

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        num_input_channels = len(
            self.dataset_json["channel_names"].keys()
            if "channel_names" in self.dataset_json.keys()
            else self.dataset_json["modality"].keys()
        )
        max_num_features = (
            self.UNet_max_features_2d
            if len(spacing) == 2
            else self.UNet_max_features_3d
        )
        unet_conv_op = convert_dim_to_conv_op(len(spacing))

        # print(spacing, median_shape, approximate_n_voxels_dataset)
        # find an initial patch size
        # we first use the spacing to get an aspect ratio
        tmp = 1 / np.array(spacing)

        # we then upscale it so that it initially is certainly larger than what we need (rescale to have the same
        # volume as a patch of size 256 ** 3)
        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
        # ideal because large initial patch sizes increase computation time because more iterations in the while loop
        # further down may be required.
        if len(spacing) == 3:
            initial_patch_size = [
                round(i) for i in tmp * (256**3 / np.prod(tmp)) ** (1 / 3)
            ]
        elif len(spacing) == 2:
            initial_patch_size = [
                round(i) for i in tmp * (2048**2 / np.prod(tmp)) ** (1 / 2)
            ]
        else:
            raise RuntimeError()

        # clip initial patch size to median_shape.
        # todo patch size can still get too large because we pad the patch size to a multiple of 2**n
        initial_patch_size = np.minimum(
            initial_patch_size, median_shape[: len(spacing)]
        )
        initial_patch_size = np.array([16, 256, 256])

        # use that to get the network topology. Note that this changes the patch_size depending on the number of
        # pooling operations (must be divisible by 2**num_pool in each axis)
        # Example outputs:
        # network_num_pool_per_axis: [3, 6, 6]
        # pool_op_kernel_sizes: ((1, 1, 1), (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2))
        # conv_kernel_sizes: ((1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3))
        # patch_size: (48, 448, 448)
        # shape_must_be_divisible_by: array([ 8, 64, 64])

        (
            network_num_pool_per_axis,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            patch_size,
            shape_must_be_divisible_by,
        ) = get_pool_and_conv_props(
            spacing, initial_patch_size, self.featuremap_min_edge_length, 6
        )
        # num_stages = len(pool_op_kernel_sizes)
        # norm = get_matching_instancenorm(unet_conv_op)

        num_stages = len(pool_op_kernel_sizes)

        # ASSERT that that the first stride = 1
        assert pool_op_kernel_sizes[0] == (1,) * len(spacing)
        architecture_kwargs = {
            "network_class_name": "mbas.architectures.MedNeXt.MedNeXt",
            "arch_kwargs": {
                "n_stages": num_stages,
                "features_per_stage": _features_per_stage(num_stages, max_num_features),
                "conv_op": "torch.nn.modules.conv.Conv3d",
                "stem_kernel_size": 1,
                "kernel_sizes": conv_kernel_sizes,
                "strides": pool_op_kernel_sizes,
                "n_blocks_per_stage": self.UNet_n_blocks_per_stage[:num_stages],
                "exp_ratio_per_stage": self.UNet_exp_ratio_per_stage[:num_stages],
                "n_blocks_per_stage_decoder": self.UNet_n_blocks_per_stage_decoder[
                    : num_stages - 1
                ],
                "exp_ratio_per_stage_decoder": self.UNet_exp_ratio_per_stage_decoder[
                    : num_stages - 1
                ],
                "norm_type": "group",
                "enable_affine_transform": False,
            },
            "_kw_requires_import": ("conv_op",),
        }

        # Second Experiment
        # patch_size = np.array([16, 128, 128])
        # strides = [
        #     (1, 1, 1),
        #     (1, 2, 2),
        #     (2, 2, 2),
        #     (2, 2, 2),
        #     (1, 2, 2),
        # ]
        # architecture_kwargs = {
        #     "network_class_name": "mbas.architectures.MedNeXt.MedNeXt",
        #     "arch_kwargs": {
        #         "n_stages": 5,
        #         "features_per_stage": [64, 96, 128, 256, 320],
        #         "conv_op": "torch.nn.modules.conv.Conv3d",
        #         "kernel_size": 5,
        #         "strides": strides,
        #         "n_blocks_per_stage": [4, 4, 6, 4, 4],
        #         "exp_ratio_per_stage": [2, 3, 4, 4, 4],
        #         "n_blocks_per_stage_decoder": [4, 6, 4, 4],
        #         "exp_ratio_per_stage_decoder": [4, 4, 3, 2],
        #         "norm_type": "group",
        #         "enable_affine_transform": False,
        #     },
        #     "_kw_requires_import": ("conv_op",),
        # }

        # First experiment
        # patch_size = np.array([16, 256, 256])
        # strides = [
        #     (1, 1, 1),
        #     (1, 2, 2),
        #     (1, 2, 2),
        #     (2, 2, 2),
        #     (2, 2, 2),
        #     (1, 2, 2),
        #     (1, 2, 2),
        # ]
        # architecture_kwargs = {
        #     "network_class_name": "mbas.architectures.MedNeXt.MedNeXt",
        #     "arch_kwargs": {
        #         "n_stages": 7,
        #         "features_per_stage": [32, 64, 128, 256, 320, 320, 320],
        #         "conv_op": "torch.nn.modules.conv.Conv3d",
        #         "kernel_size": 3,
        #         "strides": strides,
        #         "n_blocks_per_stage": [3, 4, 6, 6, 6, 6, 6],
        #         "exp_ratio_per_stage": [2, 3, 4, 4, 4, 4, 4],
        #         "n_blocks_per_stage_decoder": [6, 6, 6, 6, 4, 3],
        #         "exp_ratio_per_stage_decoder": [4, 4, 4, 4, 3, 2],
        #         "norm_type": "group",
        #         "enable_affine_transform": False,
        #     },
        #     "_kw_requires_import": ("conv_op",),
        # }

        # DEFAULT SETTINGS from MedNeXt
        # patch_size = np.array([16, 96, 96])
        # strides = [(1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        # architecture_kwargs = {
        #     "network_class_name": "mbas.architectures.MedNeXt.MedNeXt",
        #     "arch_kwargs": {
        #         "n_stages": 5,
        #         "features_per_stage": [32, 64, 128, 256, 512],
        #         "conv_op": "torch.nn.modules.conv.Conv3d",
        #         "kernel_size": 3,
        #         "strides": strides,
        #         "n_blocks_per_stage": [3, 4, 8, 8, 8],
        #         "exp_ratio_per_stage": [2, 3, 4, 4, 4],
        #         "n_blocks_per_stage_decoder": [8, 8, 4, 3],
        #         "exp_ratio_per_stage_decoder": [4, 4, 3, 2],
        #         "norm_type": "group",
        #         "enable_affine_transform": False,
        #     },
        #     "_kw_requires_import": ("conv_op",),
        # }

        # now estimate vram consumption
        if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
            estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
        else:
            estimate = self.static_estimate_VRAM_usage(
                patch_size,
                num_input_channels,
                len(self.dataset_json["labels"].keys()),
                architecture_kwargs["network_class_name"],
                architecture_kwargs["arch_kwargs"],
                architecture_kwargs["_kw_requires_import"],
            )
            _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # adapt for our vram target
        reference = (
            self.UNet_reference_val_2d
            if len(spacing) == 2
            else self.UNet_reference_val_3d
        ) * (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        # while estimate > reference:
        #     # print(patch_size)
        #     # patch size seems to be too large, so we need to reduce it. Reduce the axis that currently violates the
        #     # aspect ratio the most (that is the largest relative to median shape)
        #     axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[:len(spacing)])])[-1]

        #     # we cannot simply reduce that axis by shape_must_be_divisible_by[axis_to_be_reduced] because this
        #     # may cause us to skip some valid sizes, for example shape_must_be_divisible_by is 64 for a shape of 256.
        #     # If we subtracted that we would end up with 192, skipping 224 which is also a valid patch size
        #     # (224 / 2**5 = 7; 7 < 2 * self.UNet_featuremap_min_edge_length(4) so it's valid). So we need to first
        #     # subtract shape_must_be_divisible_by, then recompute it and then subtract the
        #     # recomputed shape_must_be_divisible_by. Annoying.
        #     patch_size = list(patch_size)
        #     tmp = deepcopy(patch_size)
        #     tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
        #     _, _, _, _, shape_must_be_divisible_by = \
        #         get_pool_and_conv_props(spacing, tmp,
        #                                 self.UNet_featuremap_min_edge_length,
        #                                 999999)
        #     patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

        #     # now recompute topology
        #     network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        #     shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
        #                                                          self.UNet_featuremap_min_edge_length,
        #                                                          999999)

        #     num_stages = len(pool_op_kernel_sizes)
        #     architecture_kwargs['arch_kwargs'].update({
        #         'n_stages': num_stages,
        #         'kernel_sizes': conv_kernel_sizes,
        #         'strides': pool_op_kernel_sizes,
        #         'features_per_stage': _features_per_stage(num_stages, max_num_features),
        #         'n_blocks_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
        #         'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
        #     })
        #     if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
        #         estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
        #     else:
        #         estimate = self.static_estimate_VRAM_usage(
        #             patch_size,
        #             num_input_channels,
        #             len(self.dataset_json['labels'].keys()),
        #             architecture_kwargs['network_class_name'],
        #             architecture_kwargs['arch_kwargs'],
        #             architecture_kwargs['_kw_requires_import'],
        #         )
        #         _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # TODO: auto determine batch size
        batch_size = 2 if len(spacing) == 3 else 12

        (
            resampling_data,
            resampling_data_kwargs,
            resampling_seg,
            resampling_seg_kwargs,
        ) = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = (
            self.determine_segmentation_softmax_export_fn()
        )

        normalization_schemes, mask_is_used_for_norm = (
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
        )

        plan = {
            "data_identifier": data_identifier,
            "preprocessor_name": self.preprocessor_name,
            "batch_size": batch_size,
            "patch_size": patch_size,
            "median_image_size_in_voxels": median_shape,
            "spacing": spacing,
            "normalization_schemes": normalization_schemes,
            "use_mask_for_norm": mask_is_used_for_norm,
            "resampling_fn_data": resampling_data.__name__,
            "resampling_fn_seg": resampling_seg.__name__,
            "resampling_fn_data_kwargs": resampling_data_kwargs,
            "resampling_fn_seg_kwargs": resampling_seg_kwargs,
            "resampling_fn_probabilities": resampling_softmax.__name__,
            "resampling_fn_probabilities_kwargs": resampling_softmax_kwargs,
            "architecture": architecture_kwargs,
        }
        return plan

    def plan_experiment(self):
        _tmp = {}

        # first get transpose
        transpose_forward, transpose_backward = self.determine_transpose()

        # get fullres spacing and transpose it
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        # get transposed new median shape (what we would have after resampling)
        new_shapes = [
            compute_new_shape(j, i, fullres_spacing)
            for i, j in zip(
                self.dataset_fingerprint["spacings"],
                self.dataset_fingerprint["shapes_after_crop"],
            )
        ]
        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape[transpose_forward]

        approximate_n_voxels_dataset = float(
            np.prod(new_median_shape_transposed, dtype=np.float64)
            * self.dataset_json["numTraining"]
        )

        plan_3d_fullres = self.get_plans_for_configuration(
            fullres_spacing_transposed,
            new_median_shape_transposed,
            self.generate_data_identifier("3d_fullres"),
            approximate_n_voxels_dataset,
            _tmp,
        )
        plan_3d_fullres["batch_dice"] = True
        print("3D MedNeXt configuration:")
        print(plan_3d_fullres)
        print()

        # median spacing and shape, just for reference when printing the plans
        median_spacing = np.median(self.dataset_fingerprint["spacings"], 0)[
            transpose_forward
        ]
        median_shape = np.median(self.dataset_fingerprint["shapes_after_crop"], 0)[
            transpose_forward
        ]

        # instead of writing all that into the plans we just copy the original file. More files, but less crowded
        # per file.
        shutil.copy(
            join(self.raw_dataset_folder, "dataset.json"),
            join(nnUNet_preprocessed, self.dataset_name, "dataset.json"),
        )

        plans = {
            "dataset_name": self.dataset_name,
            "plans_name": self.plans_identifier,
            "original_median_spacing_after_transp": [float(i) for i in median_spacing],
            "original_median_shape_after_transp": [int(round(i)) for i in median_shape],
            "image_reader_writer": self.determine_reader_writer().__name__,
            "transpose_forward": [int(i) for i in transpose_forward],
            "transpose_backward": [int(i) for i in transpose_backward],
            "configurations": {
                "3d_fullres": plan_3d_fullres,
            },
            "experiment_planner_used": self.__class__.__name__,
            "label_manager": "LabelManager",
            "foreground_intensity_properties_per_channel": self.dataset_fingerprint[
                "foreground_intensity_properties_per_channel"
            ],
        }
        # TODO: add other definitions for cascade training
        # plans['configurations']['3d_lowres']['next_stage'] = '3d_cascade_fullres'
        # plans['configurations']['3d_cascade_fullres'] = {
        #   'inherits_from': '3d_fullres',
        #   'previous_stage': '3d_lowres'
        # }

        self.plans = plans
        self.save_plans(plans)
        return plans
