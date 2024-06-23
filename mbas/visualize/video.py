import os
import numpy as np
import torchio as tio
from torchio.transforms.preprocessing.spatial.to_canonical import ToCanonical
from torchio.visualization import rotate

from mbas.utils.ffmpeg import FfmpegWriter


def rescale_to_uint8(
    array: np.ndarray, min_val: int | None = None, max_val: int | None = None
) -> np.ndarray:
    """
    Linearly rescales an array to uint8 in the range 0 to 255.

    Parameters:
    - array (np.ndarray): The input array of type int32.

    Returns:
    - np.ndarray: The rescaled array of type uint8.
    """
    array = array.astype(np.float32)
    if min_val is None:
        min_val = array.min()
    if max_val is None:
        max_val = array.max()

    # Avoid division by zero in case the array is constant
    if min_val == max_val:
        return np.zeros(array.shape, dtype=np.uint8)

    array = np.clip(array, min_val, max_val)
    # Linearly rescale the array
    rescaled_array = (array - min_val) / (max_val - min_val) * 255

    # Convert to uint8
    return rescaled_array.astype(np.uint8)


def grayscale_to_rgb(grayscale_image: np.ndarray) -> np.ndarray:
    """
    Converts a grayscale single channel image to RGB by replicating the grayscale values across all three channels.

    Parameters:
    - grayscale_image (np.ndarray): The input grayscale image.

    Returns:
    - np.ndarray: The converted RGB image.
    """
    # Check if the input image is already in 3 channels
    if len(grayscale_image.shape) == 3 and grayscale_image.shape[2] == 3:
        return grayscale_image  # Already an RGB image

    # Convert grayscale to RGB by stacking the grayscale image in all three channels
    rgb_image = np.stack([grayscale_image] * 3, axis=-1)

    return rgb_image


def tio_image_to_video(
    image: tio.Image,
    save_filepath: str,
    axis="axial",  # one of (Sagittal, Coronal, Axial)
    percentiles=(0.5, 99.5),
    framerate=10,
    crf=20,
):
    axes_names = ["sagittal", "coronal", "axial"]
    assert axis in axes_names
    axes_index = axes_names.index(axis)

    image = ToCanonical()(image)  # type: ignore[assignment]
    # [1, 640, 640, 44] -> [640, 640, 44]
    data = image.data[-1]

    os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
    if axes_index == 0:
        width = data.shape[1]
        height = data.shape[2]
    elif axes_index == 1:
        width = data.shape[0]
        height = data.shape[2]
    else:
        width = data.shape[0]
        height = data.shape[1]

    video_writer = FfmpegWriter(
        save_filepath,
        width,
        height,
        framerate=framerate,
        vcodec="libx264",
        crf=crf,
        pix_fmt="gray",
    )

    pixel_min, pixel_max = np.percentile(data.numpy(), percentiles)
    dim_max = data.shape[axes_index]
    for i in range(dim_max):
        if axes_index == 0:
            data_slice = data[i, :, :]
        elif axes_index == 1:
            data_slice = data[:, i, :]
        else:
            data_slice = data[:, :, i]
        rot_slice = rotate(data_slice, radiological=True)
        rot_slice = np.flipud(rot_slice)  # equivalent to origin = "lower"
        rot_slice = np.fliplr(rot_slice)  # equivalent to invert_xaxis
        rot_slice = rescale_to_uint8(rot_slice, pixel_min, pixel_max)
        video_writer.write(rot_slice)
    video_writer.close()
