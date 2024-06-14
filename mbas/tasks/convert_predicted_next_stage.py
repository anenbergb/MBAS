import argparse
import copy
import os
import sys

from loguru import logger
from tqdm import tqdm
import numpy as np


def convert_predicted_next_stage(source_dir, dest_dir, dry_run=True, version=1):
    os.makedirs(dest_dir, exist_ok=True)
    if version == 1:
        logger.info(
            """
        Version 1: 
        Convert Dataset 103 labels to Dataset 101 labels
        1 (left atrium) -> 3 (left atrium cavity)
        2 (right atrium) -> 2 (right atrium cavity)
        leave 'left & right atrium wall' (1) unassigned.
        """
        )
    elif version == 2:
        logger.info(
            """
        Version 2:
        Convert Dataset 103 labels to Dataset 101 labels
        1 (left atrium) -> 1 (left & right atrium wall)
        2 (right atrium) -> 1 (left & right atrium wall)
        leave (3) 'left atrium cavity' and (2) 'right atrium cavity unassigned.
        """
        )

    for file_name in tqdm(os.listdir(source_dir)):
        if not file_name.endswith(".npz"):
            continue
        dest_path = os.path.join(dest_dir, file_name)
        seg_npz = np.load(os.path.join(source_dir, file_name))["seg"]

        if version == 1:
            seg_npz[seg_npz == 1] = 3
        elif version == 2:
            seg_npz[seg_npz == 2] = 1

        if dry_run:
            logger.info(f"Saving {dest_path}")
        np.savez(dest_path, seg=seg_npz)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Prepare hierarchical labels from the MBAS source labels
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        default="/home/bryan/expr/nnUNet_results/Dataset103_MBAS/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/predicted_next_stage/3d_cascade_fullres_from_only_atrium",
        type=str,
    )
    parser.add_argument(
        "--dest-dir",
        default="/home/bryan/expr/nnUNet_results/Dataset101_MBAS/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/predicted_next_stage/3d_cascade_fullres_from_only_atrium",
        type=str,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--version",
        default=1,
        type=int,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(
        convert_predicted_next_stage(
            args.source_dir, args.dest_dir, args.dry_run, args.version
        )
    )
