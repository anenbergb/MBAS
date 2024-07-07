import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_convtransp,
    maybe_convert_scalar_to_list,
)


class MedNeXtBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd] = nn.Conv3d,
        exp_ratio: int = 4,
        kernel_size: int = 7,
        norm_type: str = "group",
        n_groups: int | None = None,
        enable_affine_transform: bool = False,
        enable_residual: bool = True,
    ):
        super().__init__()
        """
        TODO: Double check the implementation of LayerNorm
        
        """

        self.n_groups = in_channels if n_groups is None else n_groups

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv_op(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=self.n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == "group":
            norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        elif norm_type == "layer":
            norm = LayerNorm(normalized_shape=in_channels, data_format="channels_first")

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        conv2 = conv_op(
            in_channels=in_channels,
            out_channels=exp_ratio * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.norm_conv2_act = nn.Sequential(norm, conv2, nn.GELU())

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv_op(
            in_channels=exp_ratio * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.conv_dim = convert_conv_op_to_dim(conv_op)

        if enable_affine_transform:
            if self.conv_dim == 3:
                self.grn_beta = nn.Parameter(
                    torch.zeros(1, exp_ratio * in_channels, 1, 1, 1), requires_grad=True
                )
                self.grn_gamma = nn.Parameter(
                    torch.zeros(1, exp_ratio * in_channels, 1, 1, 1), requires_grad=True
                )
            elif self.conv_dim == 2:
                self.grn_beta = nn.Parameter(
                    torch.zeros(1, exp_ratio * in_channels, 1, 1), requires_grad=True
                )
                self.grn_gamma = nn.Parameter(
                    torch.zeros(1, exp_ratio * in_channels, 1, 1), requires_grad=True
                )
        self.enable_affine_transform = enable_affine_transform
        self.enable_residual = enable_residual

    def apply_affine_transform(self, x):
        """
        gamma, beta: learnable affine transform parameters
        X: input of shape (N,C,H,W,D)
        """
        if self.conv_dim == 3:
            gx = torch.norm(x, p=2, dim=(-3, -2, -1), keepdim=True)
        elif self.conv_dim == 2:
            gx = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        x = self.grn_gamma * (x * nx) + self.grn_beta + x
        return x

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm_conv2_act(out)
        if self.enable_affine_transform:
            out = self.apply_affine_transform(out)
        out = self.conv3(out)
        if self.enable_residual:
            out = out + x
        return out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == self.conv_dim, (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        conv1_size = np.prod([self.conv1.out_channels, *input_size])
        conv2_size = np.prod([self.norm_conv2_act[1].out_channels, *input_size])
        conv3_size = np.prod([self.conv3.out_channels, *input_size])
        return conv1_size + conv2_size + conv3_size


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd] = nn.Conv3d,
        exp_ratio: int = 4,
        kernel_size: int = 7,
        stride: Union[int, List[int], Tuple[int, ...]] = 2,
        norm_type: str = "group",
        n_groups: int | None = None,
        enable_affine_transform: bool = False,
    ):

        super().__init__(
            in_channels,
            out_channels,
            conv_op,
            exp_ratio,
            kernel_size,
            norm_type,
            n_groups,
            enable_affine_transform,
            enable_residual=False,
        )
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        self.conv1 = conv_op(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            groups=self.n_groups,
        )
        self.res_conv = conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        return super().forward(x) + self.res_conv(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == self.conv_dim, (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        output_size = [i // j for i, j in zip(input_size, self.stride)]
        conv1_size = np.prod([self.conv1.out_channels, *output_size])
        conv2_size = np.prod([self.norm_conv2_act[1].out_channels, *output_size])
        conv3_size = np.prod([self.conv3.out_channels, *output_size])
        res_conv_size = np.prod([self.res_conv.out_channels, *output_size])
        return conv1_size + conv2_size + conv3_size + res_conv_size


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd] = nn.Conv3d,
        exp_ratio: int = 4,
        kernel_size: int = 7,
        stride: Union[int, List[int], Tuple[int, ...]] = 2,
        norm_type: str = "group",
        n_groups: int | None = None,
        enable_affine_transform: bool = False,
    ):

        super().__init__(
            in_channels,
            out_channels,
            conv_op,
            exp_ratio,
            kernel_size,
            norm_type,
            n_groups,
            enable_affine_transform,
            enable_residual=False,
        )
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        conv_trans_op = get_matching_convtransp(conv_op)
        self.conv1 = conv_trans_op(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            groups=self.n_groups,
        )
        self.res_conv = conv_trans_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
        )

    def apply_pad(self, x):
        # Asymmetry but necessary to match shape
        if self.conv_dim == 2:
            x = torch.nn.functional.pad(x, (1, 0, 1, 0))
        elif self.conv_dim == 3:
            x = torch.nn.functional.pad(x, (1, 0, 1, 0, 1, 0))
        else:
            raise ValueError("Invalid Convolution Dimension")
        return x

    def forward(self, x):
        return self.apply_pad(super().forward(x)) + self.apply_pad(self.res_conv(x))

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == self.conv_dim, (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        output_size = [i * j for i, j in zip(input_size, self.stride)]
        conv1_size = np.prod([self.conv1.out_channels, *output_size])
        conv2_size = np.prod([self.norm_conv2_act[1].out_channels, *output_size])
        conv3_size = np.prod([self.conv3.out_channels, *output_size])
        res_conv_size = np.prod([self.res_conv.out_channels, *output_size])
        return conv1_size + conv2_size + conv3_size + res_conv_size


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
