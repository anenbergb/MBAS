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
from scipy.spatial.distance import cdist
import torchio as tio

from mbas.data.nifti import get_subject_folders


def process_subject(subject_dir):
    patient_id_str = os.path.basename(subject_dir)
    label_path = os.path.join(
        subject_dir, f"{patient_id_str}_hierarchical_label.nii.gz"
    )
    label = tio.LabelMap(
        path=label_path,
    )
    combined_hierarchical_label_path = os.path.join(
        subject_dir, f"{patient_id_str}_only_atrium_label.nii.gz"
    )

    label_volume = label.data.numpy()
    # set the left atrium cavity (3) to the value of the left atrium wall (1)
    label_volume[np.where(label_volume == 3)] = 1
    # set the right atrium wall (4) to thevalue of the right atrium cavity (2)
    label_volume[np.where(label_volume == 4)] = 2

    new_label = tio.LabelMap(
        tensor=torch.from_numpy(label_volume),
        affine=label.affine,
    )
    new_label.save(combined_hierarchical_label_path)


def hierarchical_label(train_dir):
    subject_folders = get_subject_folders(train_dir)
    subject_folders = sorted(subject_folders)
    for subject_folder in tqdm(subject_folders, desc="processing subjects"):
        process_subject(subject_folder)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Prepare hierarchical labels from the MBAS source labels
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-dir",
        default="/home/bryan/data/MBAS/Training",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(hierarchical_label(args.train_dir))
