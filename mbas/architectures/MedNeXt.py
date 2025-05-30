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
        stem_kernel_size: Union[int, List[int], Tuple[int, ...]] = 1,
        kernel_sizes: Union[
            int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]
        ] = 3,
        strides: Union[int, List[int], Tuple[int, ...]] = [1, 2, 2, 2, 2],
        n_blocks_per_stage: List[int] = [3, 4, 8, 8, 8],
        exp_ratio_per_stage: List[int] = [2, 3, 4, 4, 4],
        num_classes: int = 3,
        n_blocks_per_stage_decoder: List[int] = [8, 8, 4, 3],
        exp_ratio_per_stage_decoder: List[int] = [4, 4, 3, 2],
        deep_supervision: bool = False,
        norm_type: str = "group",
        enable_affine_transform: bool = False,
        decode_stem_kernel_size: Union[int, List[int], Tuple[int, ...]] = 1,
        override_down_kernel_size: bool = False,
        down_kernel_size: Union[int, List[int], Tuple[int, ...]] = 1,
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
        assert len(n_blocks_per_stage_decoder) == n_stages
        assert len(exp_ratio_per_stage_decoder) == n_stages

        self.encoder = MedNeXtEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            stem_kernel_size=stem_kernel_size,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            exp_ratio_per_stage=exp_ratio_per_stage,
            return_skips=True,
            norm_type=norm_type,
            enable_affine_transform=enable_affine_transform,
            override_down_kernel_size=override_down_kernel_size,
            down_kernel_size=down_kernel_size,
        )
        self.decoder = MedNeXtDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_blocks_per_stage=n_blocks_per_stage_decoder,
            exp_ratio_per_stage=exp_ratio_per_stage_decoder,
            deep_supervision=deep_supervision,
            decode_stem_kernel_size=decode_stem_kernel_size,
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
        stem_kernel_size: Union[int, List[int], Tuple[int, ...]],
        kernel_sizes: Union[
            int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]
        ],
        strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        n_blocks_per_stage: List[int],
        exp_ratio_per_stage: List[int],
        return_skips: bool = False,
        norm_type: str = "group",
        enable_affine_transform: bool = False,
        override_down_kernel_size: bool = False,
        down_kernel_size: Union[int, List[int], Tuple[int, ...]] = 1,
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
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert (
            len(kernel_sizes) == n_stages
        ), "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
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
                down_block = Stem(
                    in_channels=input_channels,
                    out_channels=features_per_stage[i],
                    conv_op=conv_op,
                    kernel_size=stem_kernel_size,
                    stride=strides[i],
                    padding=0,
                    norm_type=norm_type,
                )
            else:
                down_kernel = (
                    down_kernel_size if override_down_kernel_size else kernel_sizes[i]
                )
                down_block = MedNeXtBlock(
                    in_channels=input_channels,
                    out_channels=features_per_stage[i],
                    conv_op=conv_op,
                    exp_ratio=exp_ratio_per_stage[i],
                    kernel_size=down_kernel,
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
                        kernel_size=kernel_sizes[i],
                        stride=1,
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
        self.stem_kernel_size = stem_kernel_size
        self.conv_op = conv_op
        self.kernel_sizes = kernel_sizes
        self.norm_type = norm_type
        self.enable_affine_transform = enable_affine_transform
        self.override_down_kernel_size = override_down_kernel_size
        self.down_kernel_size = down_kernel_size

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
                output += block.compute_conv_feature_map_size(input_size)
                if j == 0:  # first block is always the down block
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
        decode_stem_kernel_size: Union[int, List[int], Tuple[int, ...]] = 1,
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

        assert len(n_blocks_per_stage) == n_stages_encoder, (
            "n_blocks_per_stage must have as many entries as we have "
            "resolution stages (n_stages in encoder), "
            f"here: {n_stages_encoder}"
        )
        assert len(exp_ratio_per_stage) == n_stages_encoder, (
            "exp_ratio_per_stage must have as many entries as we have "
            "resolution stages (n_stages in encoder), "
            f"here: {n_stages_encoder}"
        )

        conv_op = encoder.conv_op
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

            self.up_blocks.append(
                MedNeXtBlock(
                    in_channels=input_features_below,
                    out_channels=input_features_skip,
                    conv_op=conv_op,
                    exp_ratio=exp_ratio_per_stage[s - 1],
                    kernel_size=encoder.kernel_sizes[-s],
                    stride=encoder.strides[-s],
                    norm_type=norm_type,
                    enable_affine_transform=enable_affine_transform,
                    upsample=True,
                )
            )
            stage = nn.Sequential(
                *[
                    MedNeXtBlock(
                        in_channels=input_features_skip,
                        out_channels=input_features_skip,
                        conv_op=conv_op,
                        exp_ratio=exp_ratio_per_stage[s - 1],
                        kernel_size=encoder.kernel_sizes[-(s + 1)],
                        stride=1,
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
        # determine if the stem downsampled the input, so we know what factor to upsample by
        self.decode_stem_seg = None
        stride0_set = set(encoder.strides[0])
        if len(stride0_set) > 1 or stride0_set.pop() != 1:
            input_features_below = encoder.output_channels[-n_stages_encoder]
            self.decode_stem_seg = nn.Sequential(
                MedNeXtBlock(
                    in_channels=input_features_below,
                    out_channels=input_features_below,
                    conv_op=conv_op,
                    exp_ratio=exp_ratio_per_stage[n_stages_encoder - 1],
                    kernel_size=decode_stem_kernel_size,
                    stride=encoder.strides[-n_stages_encoder],
                    norm_type=norm_type,
                    enable_affine_transform=enable_affine_transform,
                    upsample=True,
                ),
                *[
                    MedNeXtBlock(
                        in_channels=input_features_below,
                        out_channels=input_features_below,
                        conv_op=conv_op,
                        exp_ratio=exp_ratio_per_stage[n_stages_encoder - 1],
                        kernel_size=decode_stem_kernel_size,
                        stride=1,
                        norm_type=norm_type,
                        enable_affine_transform=enable_affine_transform,
                    )
                    for _ in range(n_blocks_per_stage[n_stages_encoder - 1])
                ],
                encoder.conv_op(
                    input_features_below,
                    num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )

        self.decode_stem_kernel_size = decode_stem_kernel_size
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
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s + 1](x))

        if self.decode_stem_seg is None and not self.deep_supervision:
            seg_outputs.append(self.seg_layers[-1](x))

        if self.decode_stem_seg is not None:
            seg_outputs.append(self.decode_stem_seg(x))

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
        downsampled_size = input_size
        skip_sizes = []
        for s in range(len(self.encoder.strides)):
            skip_sizes.append(
                [i // j for i, j in zip(downsampled_size, self.encoder.strides[s])]
            )
            downsampled_size = skip_sizes[-1]

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
            if self.deep_supervision:
                output += np.prod(
                    [self.seg_layers[s + 1].out_channels, *skip_sizes[-(s + 2)]],
                    dtype=np.int64,
                )
        if self.decode_stem_seg is None and not self.deep_supervision:
            output += np.prod(
                [self.seg_layers[-1].out_channels, *skip_sizes[0]],
                dtype=np.int64,
            )
        if self.decode_stem_seg is not None:
            output += self.decode_stem_seg[0].compute_conv_feature_map_size(
                skip_sizes[0]
            )
            decode_stem_len = len(self.decode_stem_seg)
            for i in range(1, decode_stem_len - 1):
                output += self.decode_stem_seg[i].compute_conv_feature_map_size(
                    input_size
                )
            output += np.prod(
                [self.decode_stem_seg[-1].out_channels, *input_size],
                dtype=np.int64,
            )
        return output


if __name__ == "__main__":

    patch_size = (16, 256, 256)
    network = MedNeXt(
        input_channels=1,
        n_stages=7,
        features_per_stage=[32, 64, 128, 256, 320, 320, 320],
        stem_kernel_size=1,
        kernel_sizes=[
            (1, 3, 3),
            (1, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ],
        strides=[
            (1, 1, 1),
            (1, 2, 2),
            (1, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        ],
        n_blocks_per_stage=[3, 4, 6, 6, 6, 6, 6],
        exp_ratio_per_stage=[2, 3, 4, 4, 4, 4, 4],
        n_blocks_per_stage_decoder=[6, 6, 6, 6, 4, 3, 3],
        exp_ratio_per_stage_decoder=[4, 4, 4, 4, 3, 2, 2],
        deep_supervision=True,
        decode_stem_kernel_size=3,
        override_down_kernel_size=False,
        down_kernel_size=1,
    ).cuda()

    # patch_size = (16, 96, 96)
    # network = MedNeXt(
    #     input_channels=1,
    #     n_stages=6,
    #     features_per_stage=(32, 64, 128, 128, 128, 128),
    #     stem_kernel_size=1,
    #     kernel_sizes=[
    #         (1, 3, 3),
    #         (1, 3, 3),
    #         (3, 3, 3),
    #         (3, 3, 3),
    #         (3, 3, 3),
    #         (3, 3, 3),
    #     ],
    #     strides=[
    #         (1, 1, 1),
    #         (1, 2, 2),
    #         (1, 2, 2),
    #         (2, 2, 2),
    #         (2, 2, 2),
    #         (2, 2, 2),
    #     ],
    #     n_blocks_per_stage=[3, 4, 6, 6, 6, 6, 6],
    #     exp_ratio_per_stage=[2, 3, 4, 4, 4, 4, 4],
    #     n_blocks_per_stage_decoder=[6, 6, 6, 6, 4, 3, 3],
    #     exp_ratio_per_stage_decoder=[4, 4, 4, 4, 3, 2, 2],
    #     deep_supervision=True,
    #     decode_stem_kernel_size=3,
    #     override_down_kernel_size=False,
    #     down_kernel_size=1,
    # ).cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(network)
    print(f"# parameters: {count_parameters(network)}")

    num_bytes = network.compute_conv_feature_map_size(patch_size)
    print(f"Memory size: {num_bytes} bytes, {num_bytes / 1024**3:.2f} GB")

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # B, C, D, H, W
    x = torch.zeros((1, 1, *patch_size), requires_grad=False).cuda()
    with torch.no_grad():
        segs = network(x)
        for i, seg in enumerate(segs):
            print(f"Segmentation mask shape: {seg.shape}")

    x = torch.zeros((1, 1, *patch_size), requires_grad=False).cuda()
    flops = FlopCountAnalysis(network, x)
    print(f"# FLOPs: {flops.total()}")
    print("Parameter count table:")
    print(parameter_count_table(network, max_depth=2))
