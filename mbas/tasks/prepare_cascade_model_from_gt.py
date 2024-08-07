import argparse
import copy
import os
import sys
from typing import List

from loguru import logger
from tqdm import tqdm
import numpy as np
import pandas as pd
import torchio as tio
from tabulate import tabulate
import nibabel

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def make_ground_truth_dir(results_dir, dataset_preprocess_dir):
    ground_truth_dir = os.path.join(results_dir, "cascade_ground_truth")
    save_dir_all_labels = os.path.join(ground_truth_dir, "ground_truth_all_labels")
    os.makedirs(save_dir_all_labels, exist_ok=True)
    save_dir_binary = os.path.join(ground_truth_dir, "ground_truth_binary")
    os.makedirs(save_dir_binary, exist_ok=True)

    dataset = nnUNetDataset(
        dataset_preprocess_dir, num_images_properties_loading_threshold=0
    )

    for key, entry in tqdm(dataset.items()):
        # seg could be (1, 44, 638, 638) int8
        # with values (array([-1,  0,  1,  2,  3], dtype=int8)
        seg = np.load(entry["data_file"])["seg"]
        seg[seg < 0] = 0
        seg = seg[0]
        # now seg is (44, 638, 638) int8 with values (0, 1, 2, 3)

        seg_binary = np.zeros_like(seg)
        seg_binary[seg > 0] = 1

        save_file = os.path.join(save_dir_all_labels, f"{key}.npz")
        np.savez_compressed(save_file, seg=seg)

        save_file = os.path.join(save_dir_binary, f"{key}.npz")
        np.savez_compressed(save_file, seg=seg_binary)


def prepare_cascade_model(
    plans_json, results_dir, dataset_preprocess_dir, trainer, save_ground_truth=True
):
    plans_json = load_json(plans_json)
    plans_manager = PlansManager(plans_json)
    configurations = plans_manager.available_configurations
    configs_with_prev_stage = []
    prev_stages = set()
    for name in configurations:
        prev_stage_name = plans_manager.get_configuration(name).previous_stage_name
        if prev_stage_name is not None:
            configs_with_prev_stage.append(name)
            prev_stages.add(prev_stage_name)
    assert len(prev_stages) == 1, f"Expected one previous stage, got {prev_stages}"
    prev_stage_name = prev_stages.pop()
    logger.info(
        f"Found {len(configs_with_prev_stage)} configurations with a previous stage named: {prev_stage_name}"
    )

    if save_ground_truth:
        make_ground_truth_dir(results_dir, dataset_preprocess_dir)

    ground_truth_dir = os.path.join(
        results_dir, "cascade_ground_truth", prev_stage_name
    )
    assert os.path.exists(ground_truth_dir), f"Expected {ground_truth_dir} to exist"

    new_prev_stage_name = f"{trainer}__{plans_manager.plans_name}__{prev_stage_name}"
    predicted_next_stage = os.path.join(
        results_dir,
        new_prev_stage_name,
        "predicted_next_stage",
    )
    os.makedirs(predicted_next_stage, exist_ok=True)

    for name in configs_with_prev_stage:
        symlink = os.path.join(predicted_next_stage, name)
        if os.path.exists(symlink):
            os.unlink(symlink)
        logger.info(f"Creating symlink {symlink}")
        os.symlink(ground_truth_dir, symlink, True)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Prepare cascade model by cropping the validation predictions from a prior model and using them as input to the next model.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plans",
        default="/home/bryan/data/mbas_nnUNet_preprocessed/Dataset101_MBAS/MedNeXtV2Plans_2024_08_06_cascade.json",
        type=str,
    )
    parser.add_argument(
        "--results-dir",
        default="/home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS",
        type=str,
    )
    parser.add_argument(
        "--dataset-preprocess-dir",
        default="/home/bryan/data/mbas_nnUNet_preprocessed/Dataset101_MBAS/MedNeXtPlans_3d_fullres",
        type=str,
    )
    parser.add_argument(
        "--trainer",
        "-tr",
        default="nnUNetTrainer_MedNeXt",
        type=str,
    )
    parser.add_argument(
        "--save-ground-truth",
        action="store_true",
        default=False,
        help="Save ground truth (default: False)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(
        prepare_cascade_model(
            args.plans,
            args.results_dir,
            args.dataset_preprocess_dir,
            args.trainer,
            args.save_ground_truth,
        )
    )
