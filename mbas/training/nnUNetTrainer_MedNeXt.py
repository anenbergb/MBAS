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
            deep_supervision_scales = list(
                list(i)
                for i in 1
                / np.cumprod(
                    np.vstack(self.configuration_manager.pool_op_kernel_sizes), axis=0
                )
            )
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales
