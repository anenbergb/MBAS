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
    label_path = os.path.join(subject_dir, f"{patient_id_str}_label.nii.gz")
    label = tio.LabelMap(
        path=label_path,
    )
    hierarchical_label_path = os.path.join(
        subject_dir, f"{patient_id_str}_hierarchical_label.nii.gz"
    )

    label_volume = label.data.numpy()

    def get_indices(label_index):
        return np.transpose(np.where(label_volume == label_index))

    wall_indices = get_indices(1)
    right_indices = get_indices(2)
    left_indices = get_indices(3)

    def get_min_dist(target_indices, batch_size=1000):
        min_dists = []
        min_indices = []
        for slice_start in range(0, len(wall_indices), batch_size):
            wall_slice = wall_indices[slice_start : slice_start + batch_size]
            wall_to_target = cdist(wall_slice, target_indices)
            min_dists.extend(np.min(wall_to_target, axis=1).tolist())
            min_indices.extend(np.argmin(wall_to_target, axis=1).tolist())
        return min_dists, min_indices

    wall_to_right_dists, _ = get_min_dist(right_indices)
    wall_to_left_dists, _ = get_min_dist(left_indices)
    label_volume_hier = label_volume.copy()

    for i in range(len(wall_indices)):
        wall_to_right = wall_to_right_dists[i]
        wall_to_left = wall_to_left_dists[i]
        if wall_to_right < wall_to_left:
            label_volume_hier[tuple(wall_indices[i])] = 4

    label_hierarchical = tio.LabelMap(
        tensor=torch.from_numpy(label_volume_hier),
        affine=label.affine,
    )
    label_hierarchical.save(hierarchical_label_path)


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
