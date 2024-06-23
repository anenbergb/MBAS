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
import fiftyone.zoo as foz

from mbas.data.nifti import get_subject_folders, make_subject


def launch_fiftyone_app(dataset):
    # Launch the FiftyOne app
    session = fo.launch_app(dataset)
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

        # Add custom fields if necessary, e.g., labels or metadata
        # sample["metadata"] = fo.Metadata()  # Example for adding metadata
        # Add the sample to the dataset
        samples.append(sample)
    return samples


def launch_fiftyone(
    data_dir,
    dataset_name,
):
    train_folders = sorted(get_subject_folders(os.path.join(data_dir, "Training")))
    val_folders = sorted(get_subject_folders(os.path.join(data_dir, "Validation")))
    samples = create_samples(train_folders, "train") + create_samples(
        val_folders, "validation"
    )
    dataset = fo.Dataset(dataset_name)
    dataset.add_samples(samples)
    launch_fiftyone_app(dataset)


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
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(launch_fiftyone(args.data_dir, args.dataset_name))
