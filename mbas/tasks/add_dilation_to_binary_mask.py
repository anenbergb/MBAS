import argparse
import os
import sys
import torch
import nibabel
from loguru import logger
from tqdm import tqdm
import glob
import numpy as np
from mbas.utils.binary_dilation_transform import binary_dilation_transform


def add_dilation_to_binary_mask(
    input_dir,
    output_dir,
    dilation_radius=1,
):
    logger.info(f"Adding dilation to binary masks in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for fpath in tqdm(glob.glob(f"{input_dir}/*.nii.gz")):
        nib_mask = nibabel.load(fpath)
        # shape (576,576,44) -> (44,576,576)
        mask = nib_mask.get_fdata().astype(np.uint8).transpose([2, 1, 0])
        mask = torch.from_numpy(mask)
        mask_dilated = binary_dilation_transform(mask, dilation_radius)
        nib_mask_dilated = nibabel.Nifti1Image(
            mask_dilated.numpy().transpose([2, 1, 0]), nib_mask.affine
        )
        save_fpath = os.path.join(output_dir, os.path.basename(fpath))
        nibabel.save(nib_mask_dilated, save_fpath)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Add dilation to binary mask
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
    )
    parser.add_argument(
        "--dilation-radius",
        type=int,
        default=1,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(
        add_dilation_to_binary_mask(
            args.input_dir, args.output_dir, args.dilation_radius
        )
    )
