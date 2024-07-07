from typing import Union, Type, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    maybe_convert_scalar_to_list,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from mbas.architectures.blocks import *


class MedNeXt(nn.Module):

    def __init__(
        self,
        input_channels: int,
        n_stages: int = 5,
        features_per_stage: List[int] = [32, 64, 128, 256, 512],
        conv_op: Type[_ConvNd] = nn.Conv3d,
        kernel_size: int = 3,
        # kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]] = [1, 2, 2, 2, 2],
        n_blocks_per_stage: List[int] = [3, 4, 8, 8, 8],
        exp_ratio_per_stage: List[int] = [2, 3, 4, 4, 4],
        num_classes: int = 3,
        n_blocks_per_stage_decoder: List[int] = [8, 8, 4, 3],
        exp_ratio_per_stage_decoder: List[int] = [4, 4, 3, 2],
        deep_supervision: bool = False,
        norm_type: str = "group",
        enable_affine_transform: bool = False,
    ):
        """

        EXPERIMENT:
            - try differnet kernel sizes for encoder and decoder
        """

        super().__init__()
        assert len(n_blocks_per_stage) == n_stages, (
            "n_blocks_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_blocks_per_stage: {n_blocks_per_stage}"
        )
        assert len(exp_ratio_per_stage) == n_stages
        assert len(n_blocks_per_stage_decoder) == (n_stages - 1), (
            "n_blocks_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_blocks_per_stage_decoder: {n_blocks_per_stage_decoder}"
        )
        assert len(exp_ratio_per_stage_decoder) == (n_stages - 1)

        self.encoder = MedNeXtEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_size=kernel_size,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            exp_ratio_per_stage=exp_ratio_per_stage,
            return_skips=True,
            norm_type=norm_type,
            enable_affine_transform=enable_affine_transform,
        )
        self.decoder = MedNeXtDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_blocks_per_stage=n_blocks_per_stage_decoder,
            exp_ratio_per_stage=exp_ratio_per_stage_decoder,
            deep_supervision=deep_supervision,
        )
        # Used to fix PyTorch checkpointing bug
        # self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(
            input_size
        ) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class MedNeXtEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: Type[_ConvNd],
        kernel_size: int,
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: List[int],
        exp_ratio_per_stage: List[int],
        return_skips: bool = False,
        norm_type: str = "group",
        enable_affine_transform: bool = False,
    ):
        """
        The first stage is the stem

        n_stages: 5
        features_per_stage: [32, 64, 128, 256, 512]
        conv_op: nn.Conv3d
        kernel_size: 7
        n_blocks_per_stage: [2, 2, 2, 2, 2]
        exp_ratio_per_stage: [2, 3, 4, 4, 4]
        return_skips: True


        """
        super().__init__()

        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert (
            len(features_per_stage) == n_stages
        ), "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(n_blocks_per_stage) == n_stages
        ), "n_blocks_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(exp_ratio_per_stage) == n_stages
        ), "exp_ratio_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert (
            len(strides) == n_stages
        ), "strides must have as many entries as we have resolution stages (n_stages)"

        self.stages = nn.ModuleList()
        for i in range(n_stages):
            blocks = []
            if i == 0:
                # TODO: update kernel size to be variable for the stem
                # stride has to be 1 for stem
                down_block = conv_op(
                    input_channels,
                    features_per_stage[i],
                    kernel_size=1,
                    stride=strides[i],
                    padding=0,
                )
            else:
                down_block = MedNeXtDownBlock(
                    in_channels=input_channels,
                    out_channels=features_per_stage[i],
                    conv_op=conv_op,
                    exp_ratio=exp_ratio_per_stage[i],
                    kernel_size=kernel_size,
                    stride=strides[i],
                    norm_type=norm_type,
                    enable_affine_transform=enable_affine_transform,
                )
            blocks.append(down_block)
            for _ in range(n_blocks_per_stage[i]):
                blocks.append(
                    MedNeXtBlock(
                        in_channels=features_per_stage[i],
                        out_channels=features_per_stage[i],
                        conv_op=conv_op,
                        exp_ratio=exp_ratio_per_stage[i],
                        kernel_size=kernel_size,
                        norm_type=norm_type,
                        enable_affine_transform=enable_affine_transform,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            input_channels = features_per_stage[i]

        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.n_blocks_per_stage = n_blocks_per_stage
        self.exp_ratio_per_stage = exp_ratio_per_stage
        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.enable_affine_transform = enable_affine_transform

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        if self.return_skips:
            return features
        else:
            return features[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for i in range(len(self.stages)):
            for j, block in enumerate(self.stages[i]):
                if isinstance(block, self.conv_op):
                    output += np.prod([block.out_channels, *input_size])
                else:
                    output += block.compute_conv_feature_map_size(input_size)
                if j == 0:  # first block is always the down block
                    assert isinstance(block, MedNeXtDownBlock) or isinstance(
                        block, self.conv_op
                    )
                    input_size = [i // j for i, j in zip(input_size, self.strides[i])]
        return output


class MedNeXtDecoder(nn.Module):
    def __init__(
        self,
        encoder: MedNeXtEncoder,
        num_classes: int,
        n_blocks_per_stage: List[int],
        exp_ratio_per_stage: List[int],
        deep_supervision: bool = True,
    ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?


        Possible encoder.output_channels [32, 64, 128, 256, 512]
            decoder channels [512, 256, 128, 64, 32]
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        assert len(n_blocks_per_stage) == n_stages_encoder - 1, (
            "n_blocks_per_stage must have as many entries as we have "
            "resolution stages - 1 (n_stages in encoder - 1), "
            f"here: {n_stages_encoder}"
        )
        assert len(exp_ratio_per_stage) == n_stages_encoder - 1, (
            "exp_ratio_per_stage must have as many entries as we have "
            "resolution stages - 1 (n_stages in encoder - 1), "
            f"here: {n_stages_encoder}"
        )

        conv_op = encoder.conv_op
        kernel_size = encoder.kernel_size
        norm_type = encoder.norm_type
        enable_affine_transform = encoder.enable_affine_transform

        self.stages = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        # Output mask after bottleneck
        self.seg_layers.append(
            encoder.conv_op(
                encoder.output_channels[-1],
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        )

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            strides_for_upsample = encoder.strides[-s]

            self.up_blocks.append(
                MedNeXtUpBlock(
                    in_channels=input_features_below,
                    out_channels=input_features_skip,
                    conv_op=conv_op,
                    exp_ratio=exp_ratio_per_stage[s - 1],
                    kernel_size=kernel_size,
                    stride=strides_for_upsample,
                    norm_type=norm_type,
                    enable_affine_transform=enable_affine_transform,
                )
            )
            stage = nn.Sequential(
                *[
                    MedNeXtBlock(
                        in_channels=input_features_skip,
                        out_channels=input_features_skip,
                        conv_op=conv_op,
                        exp_ratio=exp_ratio_per_stage[s - 1],
                        kernel_size=kernel_size,
                        norm_type=norm_type,
                        enable_affine_transform=enable_affine_transform,
                    )
                    for _ in range(n_blocks_per_stage[s - 1])
                ]
            )
            self.stages.append(stage)
            self.seg_layers.append(
                encoder.conv_op(
                    input_features_skip,
                    num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )

        self.n_blocks_per_stage = n_blocks_per_stage
        self.exp_ratio_per_stage = exp_ratio_per_stage

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        """
        x = skips[-1]
        seg_outputs = []
        if self.deep_supervision:
            seg_outputs.append(self.seg_layers[0](x))  # seg after bottleneck
        for s in range(len(self.stages)):
            x_up = self.up_blocks[s](x)
            x_skip = skips[-(s + 2)]
            x = x_up + x_skip
            x = self.stages[s](x)
            if self.deep_supervision or s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[s + 1](x))

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if self.deep_supervision:
            return seg_outputs
        else:
            return seg_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        """
        skip_sizes = []
        for s in range(len(self.encoder.strides)):
            skip_sizes.append(
                [i // j for i, j in zip(input_size, self.encoder.strides[s])]
            )
            input_size = skip_sizes[-1]

        output = np.int64(0)
        if self.deep_supervision:
            # after the bottleneck
            output += np.prod(
                [self.seg_layers[0].out_channels, *skip_sizes[-1]], dtype=np.int64
            )

        for s in range(len(self.stages)):
            output += self.up_blocks[s].compute_conv_feature_map_size(
                skip_sizes[-(s + 1)]
            )
            for block in self.stages[s]:
                output += block.compute_conv_feature_map_size(skip_sizes[-(s + 2)])
            if self.deep_supervision or s == (len(self.stages) - 1):
                output += np.prod(
                    [self.seg_layers[s + 1].out_channels, *skip_sizes[-(s + 2)]],
                    dtype=np.int64,
                )

        return output


if __name__ == "__main__":

    strides = [(1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    network = MedNeXt(
        input_channels=1,
        strides=strides,
        deep_supervision=True,
    ).cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(network)
    print(f"# parameters: {count_parameters(network)}")

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # B, C, D, H, W
    x = torch.zeros((1, 1, 16, 48, 48), requires_grad=False).cuda()
    with torch.no_grad():
        segs = network(x)
        for i, seg in enumerate(segs):
            print(f"Segmentation mask shape: {seg.shape}")

    x = torch.zeros((1, 1, 64, 64, 64), requires_grad=False).cuda()
    flops = FlopCountAnalysis(network, x)
    print(f"# FLOPs: {flops.total()}")
    print("Parameter count table:")
    print(parameter_count_table(network, max_depth=2))
