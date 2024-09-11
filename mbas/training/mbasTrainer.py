import numpy as np
import torch
from typing import Tuple, Union, List
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
import multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
    resample_and_save,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import (
    get_patch_size,
)
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    isfile,
    save_json,
    maybe_mkdir_p,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import (
    ContrastTransform,
    BGContrast,
)
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import (
    ApplyRandomBinaryOperatorTransform,
)
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import (
    MoveSegAsOneHotToDataTransform,
)
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import (
    Convert3DTo2DTransform,
    Convert2DTo3DTransform,
)
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import (
    ConvertSegmentationToRegionsTransform,
)

from mbas.training.compound_losses import DC_and_CE_loss_cascaded_mask
from mbas.training.label_handling import determine_num_input_channels
from mbas.utils.binary_dilation_transform import (
    ApplyBinaryDilationTransform,
    binary_dilation_transform,
)


class mbasTrainer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.save_every = 10

        config = self.configuration_manager.configuration
        self.oversample_foreground_percent = config.get(
            "oversample_foreground_percent", 0.33
        )
        self.probabilistic_oversampling = config.get(
            "probabilistic_oversampling", False
        )
        self.sample_class_probabilities = config.get("sample_class_probabilities", None)
        if self.sample_class_probabilities is not None:
            assert isinstance(self.sample_class_probabilities, dict)
            sample_class_probabilities = {}
            for k, v in self.sample_class_probabilities.items():
                sample_class_probabilities[int(k)] = v
            self.sample_class_probabilities = sample_class_probabilities

        # Treat the 1st stage segmentation output as a mask applied to the loss of the 2nd stage
        self.is_cascaded_mask = config.get("is_cascaded_mask", False)
        self.cascaded_mask_dilation = config.get("cascaded_mask_dilation", 0)
        self.voxel_sample_z_coverage = config.get("voxel_sample_z_coverage", False)
        # Training Data Augmentations
        self.aug_spatial_p_elastic_deform = config.get(
            "aug_spatial_p_elastic_deform", 0
        )
        self.aug_spatial_p_rotation = config.get("aug_spatial_p_rotation", 0.2)
        self.aug_spatial_p_scaling = config.get("aug_spatial_p_scaling", 0.2)
        self.aug_gaussian_noise_p = config.get("aug_gaussian_noise_p", 0.1)
        self.aug_gaussian_blur_p = config.get("aug_gaussian_blur_p", 0.2)
        self.aug_brightness_p = config.get("aug_brightness_p", 0.15)
        self.aug_contrast_p = config.get("aug_contrast_p", 0.15)
        self.aug_lowres_p = config.get("aug_lowres_p", 0.25)
        self.aug_gamma_p = config.get("aug_gamma_p", 0.1)
        self.aug_gamma_p_invert = config.get("aug_gamma_p_invert", 0.3)

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            network_name = self.configuration_manager.network_arch_class_name
            strides = self.configuration_manager.pool_op_kernel_sizes
            if network_name.endswith("MedNeXt"):
                # MedNeXt includes deep supervision on the bottleneck layer
                stride0_set = set(strides[0])
                if len(stride0_set) > 1 or stride0_set.pop() != 1:
                    strides = [[1, 1, 1]] + strides
            elif network_name.endswith("MedNeXtV2") or network_name.endswith(
                "ResidualEncoderUNet"
            ):
                # MedNeXtV2 does not include deep supervision on the last layer
                strides = strides[:-1]
            else:
                self.print_to_log_file(
                    "WARNING! You are using deep supervision but the network architecture is not recognized. "
                    "This may lead to unexpected behavior."
                )
            deep_supervision_scales = list(
                list(i) for i in 1 / np.cumprod(np.vstack(strides), axis=0)
            )
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
            is_cascaded_mask=self.is_cascaded_mask,
            cascaded_mask_dilation=self.cascaded_mask_dilation,
            # Training Data Augmentations
            aug_spatial_p_elastic_deform=self.aug_spatial_p_elastic_deform,
            aug_spatial_p_rotation=self.aug_spatial_p_rotation,
            aug_spatial_p_scaling=self.aug_spatial_p_scaling,
            aug_gaussian_noise_p=self.aug_gaussian_noise_p,
            aug_gaussian_blur_p=self.aug_gaussian_blur_p,
            aug_brightness_p=self.aug_brightness_p,
            aug_contrast_p=self.aug_contrast_p,
            aug_lowres_p=self.aug_lowres_p,
            aug_gamma_p=self.aug_gamma_p,
            aug_gamma_p_invert=self.aug_gamma_p_invert,
        )

        # validation pipeline
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
            is_cascaded_mask=self.is_cascaded_mask,
            cascaded_mask_dilation=self.cascaded_mask_dilation,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                probabilistic_oversampling=self.probabilistic_oversampling,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=tr_transforms,
                sample_class_probabilities=self.sample_class_probabilities,
                voxel_sample_z_coverage=self.voxel_sample_z_coverage,
            )
            dl_val = nnUNetDataLoader2D(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
            )
        else:
            dl_tr = nnUNetDataLoader3D(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                probabilistic_oversampling=self.probabilistic_oversampling,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=tr_transforms,
                sample_class_probabilities=self.sample_class_probabilities,
                voxel_sample_z_coverage=self.voxel_sample_z_coverage,
            )
            dl_val = nnUNetDataLoader3D(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
            )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
        is_cascaded_mask: bool = False,
        cascaded_mask_dilation: int = 0,
        # Training Data Augmentations
        aug_spatial_p_elastic_deform: float = 0,
        aug_spatial_p_rotation: float = 0.2,
        aug_spatial_p_scaling: float = 0.2,
        aug_gaussian_noise_p: float = 0.1,
        aug_gaussian_blur_p: float = 0.2,
        aug_brightness_p: float = 0.15,
        aug_contrast_p: float = 0.15,
        aug_lowres_p: float = 0.25,
        aug_gamma_p: float = 0.1,
        aug_gamma_p_invert: float = 0.3,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=aug_spatial_p_elastic_deform,
                p_rotation=aug_spatial_p_rotation,
                rotation=rotation_for_DA,
                p_scaling=aug_spatial_p_scaling,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True
                ),
                apply_probability=aug_gaussian_noise_p,
            )
        )
        transforms.append(
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.0),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5,
                    benchmark=True,
                ),
                apply_probability=aug_gaussian_blur_p,
            )
        )
        transforms.append(
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.75, 1.25)),
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=aug_brightness_p,
            )
        )
        transforms.append(
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.75, 1.25)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=aug_contrast_p,
            )
        )
        transforms.append(
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5,
                ),
                apply_probability=aug_lowres_p,
            )
        )
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=aug_gamma_p,
            )
        )
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=aug_gamma_p_invert,
            )
        )
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskImageTransform(
                    apply_to_channels=[
                        i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]
                    ],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        # Sets label -1 to 0
        transforms.append(RemoveLabelTansform(-1, 0))

        # TODO: Consider adding augmentations or transformations for the cascaded mask
        # For example, add a buffer region around the 1 pixel values in the mask.

        if is_cascaded and not is_cascaded_mask:
            assert (
                foreground_labels is not None
            ), "We need foreground_labels for cascade augmentations"
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True,
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1,
                    ),
                    apply_probability=0.4,
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1,
                    ),
                    apply_probability=0.2,
                )
            )
        if is_cascaded and is_cascaded_mask and cascaded_mask_dilation > 0:
            transforms.append(
                ApplyBinaryDilationTransform(
                    channel_index=1, radius=cascaded_mask_dilation
                )
            )

        if regions is not None:
            if is_cascaded_mask:
                raise NotImplementedError("Regions are not supported for cascaded mask")
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=(
                        list(regions) + [ignore_label]
                        if ignore_label is not None
                        else regions
                    ),
                    channel_in_seg=0,
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(
                DownsampleSegForDSTransform(ds_scales=deep_supervision_scales)
            )

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: Union[List, Tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
        is_cascaded_mask: bool = False,
        cascaded_mask_dilation: int = 0,
    ) -> BasicTransform:
        transforms = []
        transforms.append(RemoveLabelTansform(-1, 0))

        if is_cascaded and not is_cascaded_mask:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True,
                )
            )

        if is_cascaded and is_cascaded_mask and cascaded_mask_dilation > 0:
            transforms.append(
                ApplyBinaryDilationTransform(
                    channel_index=1, radius=cascaded_mask_dilation
                )
            )

        if regions is not None:
            if is_cascaded_mask:
                raise NotImplementedError("Regions are not supported for cascaded mask")
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=(
                        list(regions) + [ignore_label]
                        if ignore_label is not None
                        else regions
                    ),
                    channel_in_seg=0,
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(
                DownsampleSegForDSTransform(ds_scales=deep_supervision_scales)
            )
        return ComposeTransforms(transforms)

    def _build_loss(self):
        if self.is_cascaded_mask:
            if self.label_manager.has_regions:
                raise NotImplementedError("Regions are not supported for cascaded mask")
            loss = DC_and_CE_loss_cascaded_mask(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        elif self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            loss = DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 1e-5,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales))]
            )
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.is_cascaded and self.is_cascaded_mask:
            mask = target[:, 1:2]
            target = target[:, 0:1]
            # TODO: check whether there are any negative values in this mask
            # if so, set them to seg[seg < 0] = 0

            if self.label_manager.has_ignore_label:
                raise NotImplementedError(
                    "has_ignore_label not supported for cascaded mask"
                )

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if (
            self.is_ddp
            and self.batch_size == 1
            and self.enable_deep_supervision
            and self._do_i_compile()
        ):
            self.print_to_log_file(
                "WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                "encounter crashes in validation then this is because torch.compile forgets "
                "to trigger a recompilation of the model with deep supervision disabled. "
                "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                "validation with --val (exactly the same as before) and then it will work. "
                "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                "forward pass (where compile is triggered) already has deep supervision disabled. "
                "This is exactly what we need in perform_actual_validation"
            )

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            None,
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        with multiprocessing.get_context("spawn").Pool(
            default_num_processes
        ) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, "validation")
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                val_keys = val_keys[self.local_rank :: dist.get_world_size()]
                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                # different

            dataset_val = nnUNetDataset(
                self.preprocessed_dataset_folder,
                val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                num_images_properties_loading_threshold=0,
            )

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [
                    maybe_mkdir_p(
                        join(self.output_folder_base, "predicted_next_stage", n)
                    )
                    for n in next_stages
                ]

            results = []

            for i, k in enumerate(dataset_val.keys()):
                proceed = not check_workers_alive_and_busy(
                    segmentation_export_pool, worker_list, results, allowed_num_queued=2
                )
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(
                        segmentation_export_pool,
                        worker_list,
                        results,
                        allowed_num_queued=2,
                    )

                self.print_to_log_file(f"predicting {k}")
                data, seg, properties = dataset_val.load_case(k)

                if self.is_cascaded and not self.is_cascaded_mask:
                    data = np.vstack(
                        (
                            data,
                            convert_labelmap_to_one_hot(
                                seg[-1],
                                self.label_manager.foreground_labels,
                                output_dtype=data.dtype,
                            ),
                        )
                    )
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

                self.print_to_log_file(
                    f"{k}, shape {data.shape}, rank {self.local_rank}"
                )
                output_filename_truncated = join(validation_output_folder, k)

                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()

                if self.is_cascaded and self.is_cascaded_mask:
                    # prediction.shape (4,44,574,574)
                    # mask shape (1,44,574,574)
                    # TODO: check whether there are any negative values in this mask
                    # if so, set them to seg[seg < 0] = 0
                    mask = torch.from_numpy(seg[1:2]).to(torch.bool)
                    if self.cascaded_mask_dilation > 0:
                        mask[0] = binary_dilation_transform(
                            mask[0], self.cascaded_mask_dilation
                        )

                    # background class-0 prediction is the first channel
                    # set the un-masked region (background) to 1
                    # leave the masked region (foreground) as is
                    prediction[0] = torch.where(mask, prediction[0], 1.0)
                    # zero out the background for the other channels
                    prediction[1:] = prediction[1:] * mask

                # this needs to go into background processes
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits,
                        (
                            (
                                prediction,
                                properties,
                                self.configuration_manager,
                                self.plans_manager,
                                self.dataset_json,
                                output_filename_truncated,
                                save_probabilities,
                            ),
                        ),
                    )
                )
                # for debug purposes
                # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                #              output_filename_truncated, save_probabilities)

                # if needed, export the softmax prediction for the next stage
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = (
                            self.plans_manager.get_configuration(n)
                        )
                        expected_preprocessed_folder = join(
                            nnUNet_preprocessed,
                            self.plans_manager.dataset_name,
                            next_stage_config_manager.data_identifier,
                        )

                        try:
                            # we do this so that we can use load_case and do not have to hard code how loading training cases is implemented
                            tmp = nnUNetDataset(
                                expected_preprocessed_folder,
                                [k],
                                num_images_properties_loading_threshold=0,
                            )
                            d, s, p = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing! "
                                f"Run the preprocessing for this configuration first!"
                            )
                            continue

                        target_shape = d.shape[1:]
                        output_folder = join(
                            self.output_folder_base, "predicted_next_stage", n
                        )
                        output_file = join(output_folder, k + ".npz")

                        # resample_and_save(prediction, target_shape, output_file, self.plans_manager, self.configuration_manager, properties,
                        #                   self.dataset_json)
                        results.append(
                            segmentation_export_pool.starmap_async(
                                resample_and_save,
                                (
                                    (
                                        prediction,
                                        target_shape,
                                        output_file,
                                        self.plans_manager,
                                        self.configuration_manager,
                                        properties,
                                        self.dataset_json,
                                    ),
                                ),
                            )
                        )
                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            _ = [r.get() for r in results]

        if self.is_ddp:
            dist.barrier()

        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(
                join(self.preprocessed_dataset_folder_base, "gt_segmentations"),
                validation_output_folder,
                join(validation_output_folder, "summary.json"),
                self.plans_manager.image_reader_writer_class(),
                self.dataset_json["file_ending"],
                (
                    self.label_manager.foreground_regions
                    if self.label_manager.has_regions
                    else self.label_manager.foreground_labels
                ),
                self.label_manager.ignore_label,
                chill=True,
                num_processes=(
                    default_num_processes * dist.get_world_size()
                    if self.is_ddp
                    else default_num_processes
                ),
            )
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file(
                "Mean Validation Dice: ",
                (metrics["foreground_mean"]["Dice"]),
                also_print_to_console=True,
            )

        self.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()
