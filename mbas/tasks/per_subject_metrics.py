import argparse
import copy
import os
import sys

from loguru import logger
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pandas as pd
import torchio as tio

from monai.metrics import HausdorffDistanceMetric, DiceMetric
from monai.transforms import AsDiscrete, Compose
from mbas.data.nifti import make_subject, get_subject_folders
from mbas.data.constants import MBAS_SHORT_LABELS


def per_subject_metrics(train_dir, results_dir, save_filepath):
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    hausdorff_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95.0, directed=False
    )
    to_onehot = Compose([AsDiscrete(to_onehot=4), lambda x: x[None]])

    metrics_dict = []
    subject_folders = get_subject_folders(train_dir)
    for subject_folder in tqdm(subject_folders):
        subject = make_subject(subject_folder)
        results_file = os.path.join(results_dir, f"{subject.patient_id_str}.nii.gz")
        if not os.path.exists(results_file):
            logger.warning(f"{results_file} not found!")
            continue
        subject.add_image(tio.LabelMap(path=results_file), "predictions")
        label_onehot = to_onehot(subject.label.data)
        predictions_onehot = to_onehot(subject.predictions.data)
        dice_score = dice_metric(y_pred=predictions_onehot, y=label_onehot)
        hausdorff_distance = hausdorff_metric(y_pred=predictions_onehot, y=label_onehot)

        dice123 = dice_score.flatten().tolist()
        h123 = hausdorff_distance.flatten().tolist()

        mdict = {
            "subject": subject.patient_id_str,
            "subject_id": subject.patient_id,
        }
        for i, dice in enumerate(dice123):
            mdict[f"DSC_{MBAS_SHORT_LABELS[i+1]}"] = dice
        for i, haus in enumerate(h123):
            mdict[f"HD95_{MBAS_SHORT_LABELS[i+1]}"] = haus
        metrics_dict.append(mdict)

    df = pd.DataFrame.from_records(metrics_dict)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    df.to_pickle(args.save)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Compute Dice score and Hausdorff distance per subject.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-dir",
        default="/home/bryan/data/MBAS/Training",
        type=str,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
    )
    parser.add_argument(
        "--save",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(per_subject_metrics(args.train_dir, args.results_dir, args.save))
