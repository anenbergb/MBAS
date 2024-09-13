import argparse
import copy
import os
import sys
import pickle

import torch
import SimpleITK as sitk
import numpy as np
from dataclasses import dataclass

from mbas.training.mbasTrainer import mbasTrainer

from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)


@dataclass
class Parameters2Stage:
    stage1_checkpoint: dict
    stage1_dataset: dict
    stage1_plans: dict
    stage2_checkpoint: dict
    stage2_dataset: dict
    stage2_plans: dict


def initialize_model(checkpoint: dict, dataset_json: dict, plans: dict):
    plans_manager = PlansManager(plans)
    trainer_name = checkpoint["trainer_name"]
    configuration_name = checkpoint["init_args"]["configuration"]
    inference_allowed_mirroring_axes = (
        checkpoint["inference_allowed_mirroring_axes"]
        if "inference_allowed_mirroring_axes" in checkpoint.keys()
        else None
    )

    configuration_manager = plans_manager.get_configuration(configuration_name)
    num_input_channels = determine_num_input_channels(
        plans_manager, configuration_manager, dataset_json
    )
    network = mbasTrainer.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False,
    )
    return network


def predict_main(gpu: str, input_dir: str, output_dir: str, model_pth: str):
    assert os.path.exists(input_dir), f"Input directory {input_dir} does not exist"
    assert os.path.exists(model_pth), f"Model path {model_pth} does not exist"

    # might be necessary
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving ouputs to {output_dir}")

    with open(model_pth, "rb") as f:
        parameters = pickle.load(f)

    stage1_model = initialize_model(
        parameters.stage1_checkpoint, parameters.stage1_dataset, parameters.stage1_plans
    )
    stage2_model = initialize_model(
        parameters.stage2_checkpoint, parameters.stage2_dataset, parameters.stage2_plans
    )
    import ipdb

    ipdb.set_trace()
    print("Completed model initialization")

    # # predict the results, here is just an example. Pls build your own logic here
    # for subdir, dirs, files in os.walk(args.input_dir):
    #     for file in files:
    #         if file.endswith('_gt.nii.gz'):
    #             file_path = os.path.join(subdir, file)
    #             img = sitk.ReadImage(file_path)
    #             predict = sitk.BinaryThreshold(img, lowerThreshold=400, upperThreshold=500)

    #             '''
    #             pls check the resolution of the predict mask, making it same with the input
    #             For example: input (640,640,44)--> output (640,640,44)
    #             '''
    #             pred_file_name = file.replace('_gt', '_label')
    #             sitk.WriteImage(predict, os.path.join(args.output_dir, pred_file_name))

    # print('Generate finished!')


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
Run inference on a directory of MRI volumes for the MBAS 2024 Challenge.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/bryan/data/MBAS/Training",
        help="path to input",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/output", help="path to input"
    )
    parser.add_argument(
        "--model_pth",
        type=str,
        default="./save_pths/ABC_test.pth",
        help="model saved pth",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    sys.exit(
        predict_main(
            args.gpu,
            args.input_dir,
            args.output_dir,
            args.model_pth,
        )
    )
