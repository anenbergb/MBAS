import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetTrainer_MedNeXt(nnUNetTrainer):
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

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            network_name = self.configuration_manager.network_arch_class_name
            strides = self.configuration_manager.pool_op_kernel_sizes
            if network_name.endswith("MedNeXt"):
                # MedNeXt includes deep supervision on the bottleneck layer
                stride0_set = set(strides[0])
                if len(stride0_set) > 1 or stride0_set.pop() != 1:
                    strides = [[1, 1, 1]] + strides
            elif network_name.endswith("MedNeXtV2"):
                # MedNeXtV2 does not include deep supervision on the last layer
                strides = strides[:-1]
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
