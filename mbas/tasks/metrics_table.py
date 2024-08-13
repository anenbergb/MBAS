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
import pickle

from monai.metrics import HausdorffDistanceMetric, DiceMetric
from monai.transforms import AsDiscrete, Compose
from mbas.data.nifti import make_subject, get_subject_folders
from mbas.data.constants import MBAS_SHORT_LABELS
from mbas.data.nifti import load_subjects
from mbas.tasks.per_subject_metrics import (
    compute_per_subject_metrics,
    make_average_table,
)

COLUMN_ORDERS = [
    "model",
    "Rank",
    "Avg_Rank",
    "DSC_wall",
    "HD95_wall",
    "DSC_right",
    "HD95_right",
    "DSC_left",
    "HD95_left",
    "DSC_atrium",
    "HD95_atrium",
]


def find_results_directories(result_root_dir: str, match: str) -> list[str]:
    results_dirs = []
    for root, dirs, files in os.walk(result_root_dir):
        if root.endswith(match):
            results_dirs.append(root)
    return results_dirs


def compute_per_model_metrics(
    subjects: list[tio.Subject],
    results_dirs: list[str],
    results_dir_match: str,
    label_key: str = "label",
):
    if label_key == "label":
        label_dict = MBAS_SHORT_LABELS
    elif label_key == "binary_label":
        label_dict = {0: "background", 1: "atrium"}
    else:
        raise ValueError(f"Unknown label key {label_key}")

    avg_metrics_dict = []
    std_metrics_dict = []
    for results_dir in results_dirs:
        prefix = results_dir[: -len(results_dir_match)].rstrip("/")
        model_name = os.path.basename(prefix)
        logger.info(f"Processing {model_name}")
        df = compute_per_subject_metrics(
            subjects, results_dir, label_key=label_key, label_dict=label_dict
        )
        avg_df = make_average_table(df)
        avg_dict = {
            "model": model_name,
        }
        std_dict = copy.deepcopy(avg_dict)
        for metric, row in avg_df.iterrows():
            avg_dict[metric] = row["Average"]
            std_dict[metric] = row["STD"]
        avg_metrics_dict.append(avg_dict)
        std_metrics_dict.append(std_dict)

    avg_df = pd.DataFrame.from_records(avg_metrics_dict)
    std_df = pd.DataFrame.from_records(std_metrics_dict)
    return avg_df, std_df


def add_metric_ranks(df, metric_prefix="DSC", ascending=False):
    series = []
    for metric_name in df.columns:
        if metric_name.startswith(metric_prefix):
            # sorts in ascending order. 1-indexed.
            metric_sorted = df[metric_name].sort_values(ascending=ascending)
            metric_ranks = pd.Series(
                np.arange(1, len(metric_sorted) + 1),
                index=metric_sorted.index,
                name=metric_name,
            )
            series.append(metric_ranks)
    return series


def compute_average_rank(row):
    metric_prefixes = ["DSC", "HD"]

    def is_metric(index):
        for metric_prefix in metric_prefixes:
            if index.startswith(metric_prefix):
                return True
        return False

    ranks = [val for index, val in row.items() if is_metric(index)]
    avg = np.average(ranks)
    return avg


def df_filter_reindex(df):
    columns = df.columns
    column_orders = [x for x in COLUMN_ORDERS if x in columns]
    return df.reindex(columns=column_orders)


def compute_per_model_ranks(df):
    metric_rank_list = [
        df["model"],
        *add_metric_ranks(df, "DSC", False),
        *add_metric_ranks(df, "HD", True),
    ]
    metric_rank_df = pd.concat(metric_rank_list, axis=1)
    # Identify columns that start with "DSC" or "HD"
    columns_of_interest = [
        col
        for col in metric_rank_df.columns
        if col.startswith("DSC") or col.startswith("HD")
    ]
    # Compute the average rank for each row
    metric_rank_df["Avg_Rank"] = metric_rank_df[columns_of_interest].mean(axis=1)
    metric_rank_df = metric_rank_df.sort_values(by="Avg_Rank")
    metric_rank_df["Rank"] = np.arange(1, len(metric_rank_df) + 1)
    metric_rank_df = df_filter_reindex(metric_rank_df)
    return metric_rank_df


def add_avg_rank(df, rank_df):
    df["Rank"] = rank_df["Rank"]
    df["Avg_Rank"] = rank_df["Avg_Rank"]
    df = df.sort_values(by="Avg_Rank")
    df = df_filter_reindex(df)
    return df


def load_cache(cache_filepath: str):
    with open(cache_filepath, "rb") as f:
        dfs = pickle.load(f)
    models = dfs["per_model_metrics"]["model"].to_list()
    return dfs, models


def filter_results_with_cache(results_dirs, results_dir_match, cache_models):
    filtered_results = []
    for results_dir in results_dirs:
        prefix = results_dir[: -len(results_dir_match)].rstrip("/")
        model_name = os.path.basename(prefix)
        if model_name not in cache_models:
            filtered_results.append(results_dir)
    return filtered_results


def metrics_table(
    dataset_dir: str,
    root_results_dirs: List[str],
    results_dir_match: str,
    cache_filepath: str | None = None,
    save_filepath: str | None = None,
    label_key: str = "label",
):
    subjects = load_subjects(dataset_dir, add_binary=label_key == "binary_label")
    logger.info(f"Loaded { len(subjects)} subjects")

    cache_dfs, cache_models = (
        load_cache(cache_filepath)
        if cache_filepath and os.path.exists(cache_filepath)
        else (None, [])
    )
    if cache_dfs is not None:
        logger.info(f"Loaded {len(cache_models)} cached models")
        print(
            tabulate(cache_dfs["per_model_metrics"], headers="keys", tablefmt="github")
        )
        print()

    results_dirs = []
    for root in root_results_dirs:
        results_dirs.extend(find_results_directories(root, results_dir_match))
    results_dirs = sorted(results_dirs)
    logger.info(f"Found {len(results_dirs)} results directories")
    results_dirs = filter_results_with_cache(
        results_dirs, results_dir_match, cache_models
    )

    logger.info(f"Processing {len(results_dirs)} results directories")
    per_model_metrics_df, per_model_std_df = compute_per_model_metrics(
        subjects, results_dirs, results_dir_match, label_key=label_key
    )
    if cache_dfs is not None:
        per_model_metrics_df = pd.concat(
            [cache_dfs["per_model_metrics"], per_model_metrics_df], axis=0
        ).reset_index(drop=True)
        per_model_std_df = pd.concat(
            [cache_dfs["per_model_metrics_std"], per_model_std_df], axis=0
        ).reset_index(drop=True)

    logger.info(f"Computing ranks for {len(per_model_metrics_df)} models")
    per_model_ranks_df = compute_per_model_ranks(per_model_metrics_df)
    per_model_metrics_df = add_avg_rank(per_model_metrics_df, per_model_ranks_df)
    per_model_std_df = add_avg_rank(per_model_std_df, per_model_ranks_df)

    save_struct = {
        "per_model_metrics": per_model_metrics_df,
        "per_model_metrics_std": per_model_std_df,
        "per_model_ranks": per_model_ranks_df,
    }
    os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
    with open(save_filepath, "wb") as f:
        pickle.dump(save_struct, f)
    logger.info(f"Saved metrics to {save_filepath}")
    logger.info("Metrics Table:")
    print(tabulate(per_model_metrics_df, headers="keys", tablefmt="github"))
    print()
    logger.info("Ranks Table:")
    print(tabulate(per_model_ranks_df, headers="keys", tablefmt="github"))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Compute Dice score and Hausdorff distance across all experiments.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        default="/home/bryan/data/MBAS",
        type=str,
    )
    parser.add_argument(
        "--root-results-dirs",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--results-dir-match",
        type=str,
        default="crossval_results_folds_0/postprocessed",
    )
    parser.add_argument(
        "--cache",
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
        metrics_table(
            args.dataset_dir,
            args.root_results_dirs,
            args.results_dir_match,
            args.cache,
            args.save,
            args.label_key,
        )
    )
