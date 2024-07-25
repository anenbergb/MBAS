import numpy as np
import torch
import sys

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

from mbas.training.nnUNetTrainer_MedNeXt import nnUNetTrainer_MedNeXt
from mbas.training.compound_losses import DC_CE_HD_loss


class nnUNetTrainer_MedNeXt_CE_DC_HD(nnUNetTrainer_MedNeXt):
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

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError(
                "Region loss not implemented for nnUNetTrainer_MedNeXt_CE_DC_HD"
            )

        lambda_hd = 1.0
        lambda_dice = 1.0
        lambda_ce = lambda_dice + lambda_hd

        loss = DC_CE_HD_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            ce_kwargs={},
            weight_ce=lambda_ce,
            weight_dice=lambda_dice,
            weight_hd=lambda_hd,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        self.print_to_log_file(f"lambda_hausdorff: {lambda_hd}")
        self.print_to_log_file(f"lambda_dice: {lambda_dice}")
        self.print_to_log_file(f"lambda_ce: {lambda_ce}")

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
