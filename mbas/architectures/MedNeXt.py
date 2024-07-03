from typing import Union, Type, List, Tuple
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim


from mbas.architectures.blocks import *


class MedNeXt(nn.Module):

    def __init__(
        self,
        input_channels: int,
        n_channels: int,
        num_classes: int,
        conv_op: Type[_ConvNd],
        exp_r: int = 4,  # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,  # Can be used to test deep supervision
        do_res: bool = False,  # Can be used to individually test residual connection
        do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
        block_counts: list = [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ],  # Can be used to test staging ratio:
        # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type="group",
        dim="3d",  # 2d or 3d
        grn=False,
    ):

        super().__init__()

        self.do_ds = deep_supervision
        assert dim in ["2d", "3d"]

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        self.stem = conv_op(input_channels, n_channels, kernel_size=1)
        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    exp_r=exp_r[0],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[0])
            ]
        )

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
        )

        self.enc_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 2,
                    out_channels=n_channels * 2,
                    exp_r=exp_r[1],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[1])
            ]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 4,
                    out_channels=n_channels * 4,
                    exp_r=exp_r[2],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[2])
            ]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 8,
                    out_channels=n_channels * 8,
                    exp_r=exp_r[3],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[3])
            ]
        )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 16,
                    out_channels=n_channels * 16,
                    exp_r=exp_r[4],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[4])
            ]
        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 8,
                    out_channels=n_channels * 8,
                    exp_r=exp_r[5],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[5])
            ]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 4,
                    out_channels=n_channels * 4,
                    exp_r=exp_r[6],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[6])
            ]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels * 2,
                    out_channels=n_channels * 2,
                    exp_r=exp_r[7],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[7])
            ]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    exp_r=exp_r[8],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[8])
            ]
        )

        self.out_0 = OutBlock(in_channels=n_channels, num_classes=num_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(
                in_channels=n_channels * 2, num_classes=num_classes, dim=dim
            )
            self.out_2 = OutBlock(
                in_channels=n_channels * 4, num_classes=num_classes, dim=dim
            )
            self.out_3 = OutBlock(
                in_channels=n_channels * 8, num_classes=num_classes, dim=dim
            )
            self.out_4 = OutBlock(
                in_channels=n_channels * 16, num_classes=num_classes, dim=dim
            )

        self.block_counts = block_counts

    def forward(self, x):

        x = self.stem(x)
        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)
        if self.do_ds:
            x_ds_4 = self.out_4(x)

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3
        x = self.dec_block_3(dec_x)

        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(
            input_size
        ) + self.decoder.compute_conv_feature_map_size(input_size)


class MedNeXtEncoder(nn.Module):
    def __init__(
        self,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: Type[_ConvNd],
        kernel_size: int,
        # strides
        n_blocks_per_stage: List[int],
        exp_ratio_per_stage: List[int],
        return_skips: bool = False,
        # TODO: refactor these arguments
        do_res: bool = False,  # Can be used to individually test residual connection
        do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
        norm_type="group",
        grn=False,
        dim="3d",  # 2d or 3d
    ):
        """
        # NOTE: Encoder does not include the stem

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

        self.stages = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for i in range(n_stages):
            stage = nn.Sequential(
                *[
                    MedNeXtBlock(
                        in_channels=features_per_stage[i],
                        out_channels=features_per_stage[i],
                        conv_op=conv_op,
                        exp_r=exp_ratio_per_stage[i],
                        kernel_size=kernel_size,
                        do_res=do_res,
                        norm_type=norm_type,
                        grn=grn,
                        dim=dim,
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
                        exp_r=exp_ratio_per_stage[i + 1],
                        kernel_size=kernel_size,
                        do_res=do_res_up_down,
                        norm_type=norm_type,
                        dim=dim,
                        grn=grn,
                    )
                )

        self.output_channels = features_per_stage
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.kernel_size = kernel_size
        self.do_res = do_res
        self.do_res_up_down = do_res_up_down
        self.norm_type = norm_type
        self.grn = grn
        self.dim = dim

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


class MedNeXtDecoder(nn.Module):
    def __init__(
        self,
        encoder: MedNeXtEncoder,
        num_classes: int,
        n_blocks_per_stage: List[int],
        deep_supervision: bool = True,
        # input_channels: int,
        # n_stages: int,
        # exp_ratios: List[int],
        # conv_op: Type[_ConvNd],
        # kernel_size: int = 7,
        do_res: bool = False,  # Can be used to individually test residual connection
        do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
        norm_type="group",
        dim="3d",  # 2d or 3d
        grn=False,
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
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        assert len(n_blocks_per_stage) == n_stages
        assert len(exp_ratios) == n_stages

        assert len(n_conv_per_stage) == n_stages_encoder - 1, (
            "n_conv_per_stage must have as many entries as we have "
            "resolution stages - 1 (n_stages in encoder - 1), "
            "here: %d" % n_stages_encoder
        )

        self.stages = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        for i in range(n_stages):
            channels = input_channels // (2**i)
            self.up_blocks.append(
                MedNeXtUpBlock(
                    in_channels=channels,
                    out_channels=channels // 2,
                    conv_op=conv_op,
                    exp_r=exp_ratios[i],
                    kernel_size=kernel_size,
                    do_res=do_res_up_down,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
            )
            stage = nn.Sequential(
                *[
                    MedNeXtBlock(
                        in_channels=channels // 2,
                        out_channels=channels // 2,
                        conv_op=conv_op,
                        exp_r=exp_ratios[i],
                        kernel_size=kernel_size,
                        do_res=do_res,
                        norm_type=norm_type,
                        dim=dim,
                        grn=grn,
                    )
                    for _ in range(block_counts[i])
                ]
            )
            self.stages.append(stage)
            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            self.seg_layers.append(
                encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True)
            )

    def forward(self, x):
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if i < len(self.down_blocks):
                x = self.down_blocks[i](x)
        return x, features


if __name__ == "__main__":

    network = MedNeXt(
        input_channels=1,
        n_channels=32,
        num_classes=13,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
        # exp_r = 2,
        kernel_size=3,  # Can test kernel_size
        deep_supervision=True,  # Can be used to test deep supervision
        do_res=True,  # Can be used to individually test residual connection
        do_res_up_down=True,
        # block_counts = [2,2,2,2,2,2,2,2,2],
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style=None,
        dim="3d",
        grn=True,
    ).cuda()

    # network = MedNeXt_RegularUpDown(
    #         input_channels = 1,
    #         n_channels = 32,
    #         num_classes = 13,
    #         exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
    #         kernel_size=3,                     # Can test kernel_size
    #         deep_supervision=True,             # Can be used to test deep supervision
    #         do_res=True,                      # Can be used to individually test residual connection
    #         block_counts = [2,2,2,2,2,2,2,2,2],
    #
    #     ).cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(network))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # model = ResTranUnet(img_size=128, input_channels=1, num_classes=14, dummy=False).cuda()
    x = torch.zeros((1, 1, 64, 64, 64), requires_grad=False).cuda()
    flops = FlopCountAnalysis(network, x)
    print(flops.total())

    with torch.no_grad():
        print(network)
        x = torch.zeros((1, 1, 128, 128, 128)).cuda()
        print(network(x)[0].shape)
