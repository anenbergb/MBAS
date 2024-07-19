import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


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

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            # MedNeXt includes deep supervision on the bottleneck layer
            strides = self.configuration_manager.pool_op_kernel_sizes
            stride0_set = set(strides[0])
            if len(stride0_set) > 1 or stride0_set.pop() != 1:
                strides = [[1, 1, 1]] + strides
            deep_supervision_scales = list(
                list(i) for i in 1 / np.cumprod(np.vstack(strides), axis=0)
            )
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales
