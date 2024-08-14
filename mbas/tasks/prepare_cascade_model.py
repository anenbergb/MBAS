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


def get_prev_stage_validation_predictions(
    prev_stage: str, use_crossval_postprocessed: bool = False
):
    """
    Get the validation predictions from the previous stage.
    """
    predictions = []
    for i in range(5):
        if use_crossval_postprocessed:
            fold_dir = os.path.join(
                prev_stage, f"crossval_results_folds_{i}/postprocessed"
            )
        else:
            fold_dir = os.path.join(prev_stage, f"fold_{i}/validation")
        if os.path.exists(fold_dir):
            for file in os.listdir(fold_dir):
                if file.endswith(".nii.gz"):
                    predictions.append(os.path.join(fold_dir, file))
        else:
            logger.warning(f"Could not find {fold_dir}")
    predictions = sorted(predictions)
    return predictions


def prepare_cascade_model(
    plans_json: str,
    results_dir: str,
    prev_stage_dir: str,
    dataset_preprocess_dir: str,
    rel_save_dir: str,
    trainer: str,
    use_crossval_postprocessed: bool = False,
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

    validation_predictions = get_prev_stage_validation_predictions(
        prev_stage_dir, use_crossval_postprocessed=use_crossval_postprocessed
    )
    logger.info(
        f"Found {len(validation_predictions)} validation predictions from {prev_stage_dir}"
    )
    dataset = nnUNetDataset(
        dataset_preprocess_dir, num_images_properties_loading_threshold=0
    )
    save_dir = os.path.join(prev_stage_dir, rel_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving cropped predictions to {save_dir}")
    for fpath in tqdm(validation_predictions):
        data_identifier = os.path.basename(fpath)[: -len(".nii.gz")]  # MBAS_005
        prediction = (
            nibabel.load(fpath).get_fdata().astype(np.uint8)
        )  # e.g. shape (576,576,44)
        # e.g. data.shape (1, 44, 574, 574)
        # _data, _seg, properties = dataset.load_case(data_identifier)
        properties = dataset[data_identifier]["properties"]
        save_file = os.path.join(save_dir, f"{data_identifier}.npz")

        prediction_trans = prediction.transpose([2, 1, 0])
        bbox_used_for_cropping = properties["bbox_used_for_cropping"]
        slicer = bounding_box_to_slice(bbox_used_for_cropping)
        prediction_cropped = prediction_trans[slicer]
        np.savez_compressed(save_file, seg=prediction_cropped)

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
        os.symlink(save_dir, symlink, True)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Prepare cascade model by cropping the validation predictions from a prior model and using them as input to the next model.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plans",
        default="/home/bryan/data/mbas_nnUNet_preprocessed/Dataset101_MBAS/MedNeXtPlans_2024_07_29.json",
        type=str,
    )
    parser.add_argument(
        "--results-dir",
        default="/home/bryan/expr/mbas_nnUNet_results/Dataset101_MBAS",
        type=str,
    )
    parser.add_argument(
        "--prev-stage-dir",
        default="nnUNetTrainer_MedNeXt__MedNeXtPlans__3d_fullres",
        type=str,
    )
    parser.add_argument(
        "--dataset-preprocess-dir",
        default="/home/bryan/data/mbas_nnUNet_preprocessed/Dataset101_MBAS/MedNeXtPlans_3d_fullres",
        type=str,
    )
    parser.add_argument(
        "--save-dir",
        default="master_predicted_next_stage",
        type=str,
    )
    parser.add_argument(
        "--trainer",
        "-tr",
        default="nnUNetTrainer_MedNeXt",
        type=str,
    )
    parser.add_argument(
        "--use-crossval-postprocessed",
        action="store_true",
        default=False,
        help="Use cross-validation postprocessed results (default: False)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(
        prepare_cascade_model(
            args.plans,
            args.results_dir,
            args.prev_stage_dir,
            args.dataset_preprocess_dir,
            args.save_dir,
            args.trainer,
            args.use_crossval_postprocessed,
        )
    )
