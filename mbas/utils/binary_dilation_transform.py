import torch
from skimage.morphology import ball, disk

from batchgeneratorsv2.transforms.nnunet.random_binary_operator import (
    binary_dilation_torch,
)
from batchgeneratorsv2.transforms.base.basic_transform import SegOnlyTransform


def binary_dilation_transform(img: torch.Tensor, radius: int):
    """
    input_tensor shape [44,638,638]
    """
    orig_dtype = img.dtype
    img = img.to(torch.bool)
    if img.ndim == 2:
        strel = disk(radius, dtype=bool)
    else:
        strel = ball(radius, dtype=bool)
    result = binary_dilation_torch(img, torch.from_numpy(strel))
    img = result.to(orig_dtype)
    return img


class ApplyBinaryDilationTransform(SegOnlyTransform):
    def __init__(self, channel_index: int = 1, radius: int = 1):
        """
        Cascaded mask is typically at channel 1

        We use fft conv. Slower for small kernels but boi does it perform on larger kernels

        """
        super().__init__()
        self.radius = radius
        self.channel_index = channel_index

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        """
        seg shape: [C, X, Y, Z].
        """
        seg[self.channel_index] = binary_dilation_transform(
            seg[self.channel_index], self.radius
        )
        return seg
