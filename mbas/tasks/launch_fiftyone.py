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

import fiftyone as fo

from torchio.transforms.preprocessing.spatial.to_canonical import ToCanonical
from torchio.visualization import rotate
from mbas.data.nifti import get_subject_folders, make_subject
from mbas.data.constants import MBAS_LABELS, MBAS_LABEL_COLORS


def make_color_scheme(predictions_name="predictions"):
    # https://www.rapidtables.com/web/color/RGB_Color.html
    return fo.ColorScheme(
        color_by="value",
        opacity=0.35,
        default_mask_targets_colors=[
            {"intTarget": 1, "color": "yellow"},
            {"intTarget": 2, "color": "blue"},
            {"intTarget": 3, "color": "red"},
        ],
        fields=[
            {
                "path": "frames.ground_truth",
                "fieldColor": "blue",
                "colorByAttribute": "value",
                "maskTargetsColors": [
                    {"intTarget": 1, "color": "yellow"},
                    {"intTarget": 2, "color": "blue"},
                    {"intTarget": 3, "color": "red"},
                ],
            },
            {
                "path": f"frames.{predictions_name}",
                "fieldColor": "#7F00FF",
                "colorByAttribute": "value",
                "maskTargetsColors": [
                    {"intTarget": 1, "color": "#80FF00"},  # green
                    {"intTarget": 2, "color": "#7F00FF"},  # purple
                    {"intTarget": 3, "color": "#FF8000"},  # orange
                ],
            },
            {
                "path": "frames.binary_mask",
                "fieldColor": "#FFFFFF",
                "colorByAttribute": "value",
                "maskTargetsColors": [
                    {"intTarget": 1, "color": "#FFFFFF"},  # white
                ],
            },
        ],
    )


def fiftyone_segmentations(
    label_map,
    axis="axial",  # one of (Sagittal, Coronal, Axial
):
    axes_names = ["sagittal", "coronal", "axial"]
    assert axis in axes_names
    axes_index = axes_names.index(axis)

    image = ToCanonical()(label_map)  # type: ignore[assignment]
    # [1, 640, 640, 44] -> [640, 640, 44]
    data = image.data[-1]
    dim_max = data.shape[axes_index]

    segmentations = []
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
        seg = fo.Segmentation(mask=rot_slice)
        segmentations.append(seg)
    return segmentations


def add_segmentation_to_sample(
    sample,
    file_path,
    segmentation_key="ground_truth",
):
    if not os.path.exists(file_path):
        return
    image = tio.LabelMap(path=file_path)
    segmentations = fiftyone_segmentations(image, axis="axial")

    # Ensure the sample has a frames attribute initialized
    if not sample.frames:
        sample.frames = {}

    for i, segmentation in enumerate(segmentations, start=1):
        # Assuming frame numbering starts at 1
        sample.frames[i][segmentation_key] = segmentation


def launch_fiftyone_app(dataset, color_scheme, port=5151):
    # Launch the FiftyOne app
    session = fo.launch_app(dataset, color_scheme=color_scheme, port=port)
    session.wait()


def create_samples(subject_folders, train_test_split="train"):
    samples = []
    for subject_folder in subject_folders:
        # patient_id_str is formatted "MBAS_002"
        patient_id_str = os.path.basename(subject_folder)
        patient_id = int(patient_id_str.split("_")[-1])

        axial_video = os.path.join(subject_folder, "axial.mp4")
        # Create a FiftyOne sample for this subject
        sample = fo.Sample(filepath=axial_video)
        sample["patient_id_str"] = patient_id_str
        sample["patient_id"] = patient_id
        sample["split"] = train_test_split

        gt_path = os.path.join(subject_folder, f"{patient_id_str}_label.nii.gz")
        add_segmentation_to_sample(sample, gt_path, "ground_truth")

        # Add custom fields if necessary, e.g., labels or metadata
        # sample["metadata"] = fo.Metadata()  # Example for adding metadata
        # Add the sample to the dataset
        samples.append(sample)
    return samples


def get_predictions_file(predictions_dir, patient_id_str):
    filename = os.path.join(predictions_dir, f"{patient_id_str}.nii.gz")
    if os.path.exists(filename):
        return filename
    filename = os.path.join(predictions_dir, f"{patient_id_str}_label.nii.gz")
    if os.path.exists(filename):
        return filename
    return None


def add_predictions_to_samples(
    samples, predictions_dir, predictions_name="predictions"
):
    for sample in samples:
        patient_id_str = sample["patient_id_str"]
        predictions_file = get_predictions_file(predictions_dir, patient_id_str)
        if predictions_file is not None:
            add_segmentation_to_sample(sample, predictions_file, predictions_name)
    return samples


def launch_fiftyone(
    data_dir,
    dataset_name,
    predictions_dir=[],
    predictions_name=None,
    binary_mask_dir=None,
    port=5151,
):
    train_folders = sorted(get_subject_folders(os.path.join(data_dir, "Training")))
    val_folders = sorted(get_subject_folders(os.path.join(data_dir, "Validation")))
    samples = create_samples(train_folders, "train")
    samples += create_samples(val_folders, "validation")
    for p_dir in predictions_dir:
        samples = add_predictions_to_samples(samples, p_dir, predictions_name)
    for b_dir in binary_mask_dir:
        samples = add_predictions_to_samples(samples, b_dir, "binary_mask")

    # samples = samples[:5]
    dataset = fo.Dataset(dataset_name)
    # dataset.default_mask_targets = MBAS_LABELS
    # https://docs.voxel51.com/user_guide/using_datasets.html#storing-mask-targets
    dataset.mask_targets = {
        "ground_truth": MBAS_LABELS,
        predictions_name: MBAS_LABELS,
        "binary_mask": {0: "Background", 1: "Atrium"},
    }
    dataset.add_samples(samples)
    color_scheme = make_color_scheme(predictions_name)
    launch_fiftyone_app(dataset, color_scheme, port=port)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Render video for each subject
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default="/home/bryan/data/MBAS",
        type=str,
    )
    parser.add_argument(
        "--dataset-name",
        default="mbas_videos",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--predictions-dir",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--predictions-name",
        type=str,
        default="predictions",
    )
    parser.add_argument(
        "-b",
        "--binary-mask-dir",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(
        launch_fiftyone(
            args.data_dir,
            args.dataset_name,
            args.predictions_dir,
            args.predictions_name,
            args.binary_mask_dir,
            args.port,
        )
    )
