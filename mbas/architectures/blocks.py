import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_convtransp,
    maybe_convert_scalar_to_list,
)


def padding_from_kernel_dilation(
    kernel_size: Union[List[int], Tuple[int, ...]],
    dilation: Union[List[int], Tuple[int, ...]],
) -> List[int]:
    assert len(kernel_size) == len(
        dilation
    ), "Kernel size and dilation must have the same length!"
    return [d * (k - 1) // 2 for k, d in zip(kernel_size, dilation)]


def compute_padding_from_stride(stride):
    """
    Useful for padding the tensors after ConvTranspose

    stride is ordered as (D, H, W)
    The padding 6-tuple should be ordered as
    (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)

    The output padded tensor is ordered as (D_out, H_out, W_out) where
    D_out = padding_front + D + padding_out
    H_out = padding_top + H + padding_bottom
    W_out = padding_left + W + padding_right
    """
    padding = []
    for s in stride[::-1]:
        l_pad = (s - 1) // 2 + (s - 1) % 2
        r_pad = (s - 1) // 2
        padding.extend([l_pad, r_pad])
    return padding


class Stem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd] = nn.Conv3d,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        padding: int | None = 0,
        dilation: Union[int, List[int], Tuple[int, ...]] = 1,
        norm_type: str = "group",
        n_groups: int | None = None,
    ):
        """
        MedNeXt stem is just Conv w/ kernel=1, stride=1, padding=0
        Original ConvNeXt stem is Conv w/ kernel=4, stride=4, padding=0
        """
        super().__init__()

        kernel_size = tuple(maybe_convert_scalar_to_list(conv_op, kernel_size))
        stride = tuple(maybe_convert_scalar_to_list(conv_op, stride))
        dilation = tuple(maybe_convert_scalar_to_list(conv_op, dilation))
        if padding == 0:
            assert (
                kernel_size == stride
            ), f"Kernel size {kernel_size} and stride {stride} must be equal in the Stem!"

        if padding is None:
            padding = padding_from_kernel_dilation(kernel_size, dilation)

        self.conv1 = conv_op(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.n_groups = out_channels if n_groups is None else n_groups
        if norm_type == "group":
            self.norm = nn.GroupNorm(
                num_groups=self.n_groups, num_channels=out_channels
            )
        elif norm_type == "layer":
            self.norm = LayerNorm(
                normalized_shape=out_channels, data_format="channels_first"
            )

        self.conv_dim = convert_conv_op_to_dim(conv_op)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        return out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == self.conv_dim, (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return np.prod([self.conv1.out_channels, *input_size], dtype=np.int64)


class MedNeXtBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_op: Type[_ConvNd] = nn.Conv3d,
        exp_ratio: int = 4,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 5,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: Union[int, List[int], Tuple[int, ...]] = 1,
        norm_type: str = "group",
        n_groups: int | None = None,
        enable_affine_transform: bool = False,
        enable_residual: bool = True,
        upsample: bool = False,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
    ):
        super().__init__()
        """
        TODO: Double check the implementation of LayerNorm
        
        """
        self.stride = maybe_convert_scalar_to_list(conv_op, stride)
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        dilation = maybe_convert_scalar_to_list(conv_op, dilation)
        self.n_groups = in_channels if n_groups is None else n_groups

        conv1_op = get_matching_convtransp(conv_op) if upsample else conv_op
        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv1_op(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding_from_kernel_dilation(kernel_size, dilation),
            dilation=dilation,
            groups=self.n_groups,
        )
        self.dropout = None
        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)

        # Normalization Layer. GroupNorm is used by default.
        # original MedNeXt implementation has num_groups=in_channels
        if norm_type == "group":
            norm = nn.GroupNorm(num_groups=self.n_groups, num_channels=in_channels)
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

        # MedNeXt block has residual connection with no res_conv
        # because stride = 1 and in_channels = out_channels
        self.enable_residual = enable_residual
        self.res_conv = None
        if np.prod(self.stride) > 1 or in_channels != out_channels:
            self.res_conv = conv1_op(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
            self.enable_residual = True

        self.conv_dim = convert_conv_op_to_dim(conv_op)

        self.zero_pad = None
        if upsample and np.prod(self.stride) > 1:
            padding = compute_padding_from_stride(self.stride)
            if self.conv_dim == 3:
                self.zero_pad = nn.ZeroPad3d(padding)
            elif self.conv_dim == 2:
                self.zero_pad = nn.ZeroPad2d(padding)

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

        self.upsample = upsample
        self.enable_affine_transform = enable_affine_transform

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
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.norm_conv2_act(out)
        if self.enable_affine_transform:
            out = self.apply_affine_transform(out)
        out = self.conv3(out)
        if self.zero_pad is not None:
            out = self.zero_pad(out)
        if self.res_conv is not None:
            out += (
                self.res_conv(x)
                if self.zero_pad is None
                else self.zero_pad(self.res_conv(x))
            )
        elif self.enable_residual:
            out += x
        return out

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == self.conv_dim, (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        output = np.int64(0)
        output_size = (
            [i * j for i, j in zip(input_size, self.stride)]
            if self.upsample
            else [i // j for i, j in zip(input_size, self.stride)]
        )
        output += np.prod([self.conv1.out_channels, *output_size], dtype=np.int64)
        output += np.prod(
            [self.norm_conv2_act[1].out_channels, *output_size], dtype=np.int64
        )
        output += np.prod([self.conv3.out_channels, *output_size], dtype=np.int64)
        if self.res_conv is not None:
            output += np.prod(
                [self.res_conv.out_channels, *output_size], dtype=np.int64
            )
        return output


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


def make_transpose_block(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    conv_op=nn.Conv3d,
):
    transpconv_op = get_matching_convtransp(conv_op=conv_op)
    # dilation always 1
    dilation = maybe_convert_scalar_to_list(conv_op, 1)
    conv_padding = padding_from_kernel_dilation(kernel_size, dilation)
    post_padding = compute_padding_from_stride(stride)

    conv_dim = convert_conv_op_to_dim(conv_op)
    if conv_dim == 3:
        zero_pad = nn.ZeroPad3d(post_padding)
    elif conv_dim == 2:
        zero_pad = nn.ZeroPad2d(post_padding)
    else:
        raise ValueError(f"Convolutional dimension {conv_dim} not supported!")

    up_conv = transpconv_op(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=conv_padding,
        dilation=1,
        # Hardcoding this to True because it's True for the ResidualEncoderUNet models
        bias=True,
    )
    return nn.Sequential(up_conv, zero_pad)
