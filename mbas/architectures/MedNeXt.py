from typing import Union, Type, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
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
        strides: List[Tuple[int, ...]] = [
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (1, 1, 1),
        ],
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

        self.stem = conv_op(input_channels, features_per_stage[0], kernel_size=1)

        self.encoder = MedNeXtEncoder(
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
        x = self.stem(x)
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        stem_size = np.prod([self.stem.out_channels, *input_size])

        return (
            stem_size
            + self.encoder.compute_conv_feature_map_size(input_size)
            + self.decoder.compute_conv_feature_map_size(input_size)
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class MedNeXtEncoder(nn.Module):
    def __init__(
        self,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: Type[_ConvNd],
        strides: List[Tuple[int, ...]],
        kernel_size: int,
        # strides
        n_blocks_per_stage: List[int],
        exp_ratio_per_stage: List[int],
        return_skips: bool = False,
        norm_type: str = "group",
        enable_affine_transform: bool = False,
    ):
        """
        NOTE: Encoder does not include the stem

        n_stages: 5
        features_per_stage: [32, 64, 128, 256, 512]
        conv_op: nn.Conv3d
        kernel_size: 7
        n_blocks_per_stage: [2, 2, 2, 2, 2]
        exp_ratio_per_stage: [2, 3, 4, 4, 4]
        return_skips: True


        """
        super().__init__()

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
        self.down_blocks = nn.ModuleList()
        for i in range(n_stages):
            stage = nn.Sequential(
                *[
                    MedNeXtBlock(
                        in_channels=features_per_stage[i],
                        out_channels=features_per_stage[i],
                        conv_op=conv_op,
                        exp_ratio=exp_ratio_per_stage[i],
                        kernel_size=kernel_size,
                        norm_type=norm_type,
                        enable_affine_transform=enable_affine_transform,
                    )
                    for _ in range(n_blocks_per_stage[i])
                ]
            )
            self.stages.append(stage)
            if i < n_stages - 1:
                self.down_blocks.append(
                    MedNeXtDownBlock(
                        in_channels=features_per_stage[i],
                        out_channels=features_per_stage[i + 1],
                        conv_op=conv_op,
                        exp_ratio=exp_ratio_per_stage[i + 1],
                        kernel_size=kernel_size,
                        stride=strides[i],
                        norm_type=norm_type,
                        enable_affine_transform=enable_affine_transform,
                    )
                )

        self.output_channels = features_per_stage
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
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if i < len(self.down_blocks):
                x = self.down_blocks[i](x)
        if self.return_skips:
            return features
        else:
            return features[-1]

    def compute_conv_feature_map_size(self, input_size):
        # TODO: ACCOUNT FOR STRIDES
        output = np.int64(0)
        for i in range(len(self.stages)):
            for block in self.stages[i]:
                output += block.compute_conv_feature_map_size(input_size)
            if i < len(self.stages) - 1:
                output += self.down_blocks[i].compute_conv_feature_map_size(input_size)
                input_size = [i // 2 for i in input_size]
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
            "here: %d" % n_stages_encoder
        )
        assert len(exp_ratio_per_stage) == n_stages_encoder - 1, (
            "exp_ratio_per_stage must have as many entries as we have "
            "resolution stages - 1 (n_stages in encoder - 1), "
            "here: %d" % n_stages_encoder
        )

        conv_op = encoder.conv_op
        kernel_size = encoder.kernel_size
        norm_type = encoder.norm_type
        enable_affine_transform = encoder.enable_affine_transform

        self.stages = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            self.seg_layers.append(
                encoder.conv_op(
                    input_features_below,
                    num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )

            self.up_blocks.append(
                MedNeXtUpBlock(
                    in_channels=input_features_below,
                    out_channels=input_features_skip,
                    conv_op=conv_op,
                    exp_ratio=exp_ratio_per_stage[s - 1],
                    kernel_size=kernel_size,
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

        # Final output mask
        self.seg_layers.append(
            encoder.conv_op(
                encoder.output_channels[0],
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
        for s in range(len(self.stages)):
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))

            x_up = self.up_blocks[s](x)
            x_skip = skips[-(s + 2)]
            x = x_up + x_skip
            x = self.stages[s](x)

        final_output = self.seg_layers[-1](x)
        if not self.deep_supervision:
            return final_output
        else:
            seg_outputs.append(final_output)
            # invert seg outputs so that the largest segmentation prediction is returned first
            seg_outputs = seg_outputs[::-1]
            return seg_outputs

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        """
        n_stages_encoder = len(self.encoder.output_channels)

        # assume the encoder reduces input resolution by factor 2 each time
        skip_sizes = []
        for _ in range(n_stages_encoder):
            skip_sizes.append([i // 2 for i in input_size])
            input_size = skip_sizes[-1]

        output = np.int64(0)
        for s in range(len(self.stages)):
            input_size = skip_sizes[-(s + 1)]
            if self.deep_supervision:
                output += np.prod([self.seg_layers[s].out_channels, *input_size])
            output += self.up_blocks[s].compute_conv_feature_map_size(input_size)
            input_size_up = skip_sizes[-(s + 2)]
            for block in self.stages[s]:
                output += block.compute_conv_feature_map_size(input_size_up)

        output += np.prod([self.seg_layers[-1].out_channels, *skip_sizes[0]])
        return output


if __name__ == "__main__":

    strides = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)]
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
