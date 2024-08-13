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
from tabulate import tabulate

from monai.metrics import HausdorffDistanceMetric, DiceMetric
from monai.transforms import AsDiscrete, Compose
from mbas.data.nifti import make_subject, get_subject_folders
from mbas.data.constants import MBAS_SHORT_LABELS
from mbas.data.nifti import load_subjects


def compute_per_subject_metrics(
    subjects: list[tio.Subject],
    results_dir: str,
    label_key: str = "label",
    label_dict: dict[int, str] = MBAS_SHORT_LABELS,
):
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    hausdorff_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95.0, directed=False
    )
    to_onehot = Compose([AsDiscrete(to_onehot=len(label_dict)), lambda x: x[None]])

    metrics_dict = []
    for subject in tqdm(subjects):
        results_file = os.path.join(results_dir, f"{subject.patient_id_str}.nii.gz")
        if not os.path.exists(results_file):
            continue
        subject.add_image(tio.LabelMap(path=results_file), "predictions")
        label_onehot = to_onehot(getattr(subject, label_key).data)
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
            mdict[f"DSC_{label_dict[i+1]}"] = dice
        for i, haus in enumerate(h123):
            mdict[f"HD95_{label_dict[i+1]}"] = haus
        metrics_dict.append(mdict)
    df = pd.DataFrame.from_records(metrics_dict)
    logger.info(
        f"Computed per subject metrics for {len(df)} / {len(subjects)} subjects"
    )
    return df


def make_average_table(df):
    metric_columns = [
        x for x in df.columns if x.startswith("DSC") or x.startswith("HD95")
    ]
    avg_df = pd.DataFrame(
        {
            "Average": df[metric_columns].mean(),
            "STD": df[metric_columns].std(),
        }
    )
    return avg_df


def per_subject_metrics(
    dataset_dir, results_dir, save_filepath: str | None = None, label_key: str = "label"
):
    add_binary = label_key == "binary_label"
    if label_key == "label":
        label_dict = MBAS_SHORT_LABELS
    elif label_key == "binary_label":
        label_dict = {0: "background", 1: "atrium"}
    else:
        raise ValueError(f"Unknown label key {label_key}")
    subjects = load_subjects(dataset_dir, add_binary=add_binary)
    logger.info(f"Loaded { len(subjects)} subjects")
    df = compute_per_subject_metrics(
        subjects, results_dir, label_key=label_key, label_dict=label_dict
    )
    if save_filepath is not None:
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        df.to_pickle(save_filepath)
        logger.info(f"Saved per subject metrics to {save_filepath}")
    avg_df = make_average_table(df)
    print(tabulate(avg_df, headers="keys", tablefmt="github"))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Compute Dice score and Hausdorff distance per subject.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        default="/home/bryan/data/MBAS",
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
    parser.add_argument(
        "--label-key",
        choices=["label", "binary_label"],
        default="label",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(
        per_subject_metrics(
            args.dataset_dir, args.results_dir, args.save, args.label_key
        )
    )
