import os
import torchio as tio
from mbas.utils.ffmpeg import FfmpegWriter


def tio_image_to_video(
    image: tio.Image,
    save_filepath: str,
    figsize=None,
    axis = "axial",  # one of (Sagittal, Coronal, Axial)
    framerate=10,
    crf=20,
):
    axes_names = ["sagittal", "coronal", "axial"]
    assert axis in axis_name
    axes_index = axes_names.index(axis)
    import ipdb
    ipdb.set_trace()
    axes_index_max = image.data[-1].shape[axes_index]
    
    os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
    video_writer = FfmpegWriter(
        save_filepath,
        image_width_pix,
        image_height_pix,
        framerate=framerate,
        vcodec="libx264",
        crf=crf,
    )