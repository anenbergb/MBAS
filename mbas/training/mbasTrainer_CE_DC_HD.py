import numpy as np
import torch
import sys
from torch import autocast


from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import dummy_context


from mbas.training.mbasTrainer import mbasTrainer
from mbas.training.compound_losses import DC_CE_HD_loss
from mbas.utils.alpha_scheduler import (
    alpha_stepwise_warmup,
    alpha_stepwise_warmup_scaled,
)


class mbasTrainer_CE_DC_HD(mbasTrainer):
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
        config = self.plans_manager.get_configuration(configuration)
        self.boundary_loss_alpha_stepsize = config.configuration.get(
            "boundary_loss_alpha_stepsize", 5
        )
        self.boundary_loss_alpha_warmup_epochs = config.configuration.get(
            "boundary_loss_alpha_warmup_epochs", 250
        )
        self.boundary_loss_alpha_max = config.configuration.get(
            "boundary_loss_alpha_max", 0.75
        )
        self.alpha_stepwise_warmup_scaled = config.configuration.get(
            "alpha_stepwise_warmup_scaled", True
        )

        for key in (
            "boundary_loss_alpha_stepsize",
            "boundary_loss_alpha_warmup_epochs",
            "boundary_loss_alpha_max",
            "alpha_stepwise_warmup_scaled",
        ):
            self.print_to_log_file(f"{key}: {getattr(self, key)}")

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError(
                "Region loss not implemented for nnUNetTrainer_MedNeXt_CE_DC_HD"
            )

        lambda_ce = 1.0
        lambda_dice = 1.0

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
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

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

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        if self.alpha_stepwise_warmup_scaled:
            alpha = alpha_stepwise_warmup_scaled(
                self.current_epoch,
                self.num_epochs,
                h=self.boundary_loss_alpha_stepsize,
                warmup_epochs=self.boundary_loss_alpha_warmup_epochs,
                max_alpha=self.boundary_loss_alpha_max,
            )
        else:
            alpha = alpha_stepwise_warmup(
                self.current_epoch,
                self.num_epochs,
                h=self.boundary_loss_alpha_stepsize,
                warmup_epochs=self.boundary_loss_alpha_warmup_epochs,
                max_alpha=self.boundary_loss_alpha_max,
            )
        if self.enable_deep_supervision:
            self.loss.loss.set_alpha(alpha)
        else:
            self.loss.set_alpha(alpha)

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}
