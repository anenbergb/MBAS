import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn

from monai.losses import HausdorffDTLoss


class DC_CE_HD_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=SoftDiceLoss,
    ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        self.hd = HausdorffDTLoss(
            alpha=2.0,
            include_background=False,
            to_onehot_y=True,
            sigmoid=False,
            softmax=True,
            other_act=None,
            reduction="mean",
            batch=False,
        )
        self.set_alpha()

    def set_alpha(self, alpha: float = 1.0):
        assert alpha >= 0 and alpha <= 1, "alpha must be in [0, 1]"
        self.alpha = alpha

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :param alpha:
        :return:

        net_output: torch.Size([2, 4, 16, 256, 256])
        target: torch.Size([2, 1, 16, 256, 256])
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )
        hd_loss = self.hd(net_output, target)

        region_weight = self.weight_ce + self.weight_dice
        region_loss = (self.weight_ce / region_weight) * ce_loss + (
            self.weight_dice / region_weight
        ) * dc_loss
        loss = (1.0 - self.alpha) * region_loss + self.alpha * hd_loss
        # print(
        #     f"Alpha: {self.alpha}. Region Loss: {region_loss}. HD Loss: {hd_loss}. Total Loss: {loss}"
        # )
        return loss
