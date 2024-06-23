import argparse
import copy
import os
import pickle
import sys
from collections import OrderedDict
from glob import glob

from loguru import logger
from tqdm import tqdm
import numpy as np
import torch
import torchio as tio

from mbas.data.nifti import get_subject_folders, make_subject
from mbas.visualize.video import tio_image_to_video


def render_subject_videos(
    data_dir,
    axis="axial",
    framerate=10,
):

    subject_folders = sorted(get_subject_folders(data_dir))
    logger.info(f"Rendering videos for {len(subject_folders)} subjects")
    for folder in tqdm(subject_folders):
        subject = make_subject(folder)
        video_filepath = os.path.join(folder, f"{axis}.mp4")
        tio_image_to_video(subject.mri, video_filepath, axis=axis, framerate=framerate)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Render video for each subject
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="/home/bryan/data/MBAS/Training",
        type=str,
    )
    parser.add_argument(
        "--axis",
        type=str,
        choices=("sagittal", "coronal", "axial"),
        default="axial",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=10,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(render_subject_videos(args.data_dir, args.axis, args.framerate))
