import argparse
import copy
import os
import sys
import pickle

from typing import Tuple, Optional, List

import torch
from torch import nn
import SimpleITK as sitk
import numpy as np
from dataclasses import dataclass

from mbas.training.mbasTrainer import mbasTrainer

# from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

#     compute_steps_for_sliding_window
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
    convert_predicted_logits_to_segmentation_with_correct_shape,
)


@dataclass
class Parameters2Stage:
    stage1_checkpoint: dict
    stage1_dataset: dict
    stage1_plans: dict
    stage2_checkpoint: dict
    stage2_dataset: dict
    stage2_plans: dict


@dataclass
class NetworkConfig:
    network: nn.Module
    plans_manager: PlansManager
    configuration_manager: ConfigurationManager
    trainer_name: str
    inference_allowed_mirroring_axes: Optional[List[int]] = None


def initialize_model(checkpoint: dict, dataset_json: dict, plans: dict):
    plans_manager = PlansManager(plans)
    trainer_name = checkpoint["trainer_name"]
    configuration_name = checkpoint["init_args"]["configuration"]
    inference_allowed_mirroring_axes = (
        checkpoint["inference_allowed_mirroring_axes"]
        if "inference_allowed_mirroring_axes" in checkpoint.keys()
        else None
    )
    parameters = checkpoint["network_weights"]

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
    network.load_state_dict(parameters)
    network.eval()
    print("Network initialized")
    return NetworkConfig(
        network=network,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        trainer_name=trainer_name,
        inference_allowed_mirroring_axes=inference_allowed_mirroring_axes,
    )


@dataclass
class Filepath:
    path: str
    save_path: str


def index_images(input_dir: str, output_dir: str) -> list[Filepath]:
    filepaths = []
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_gt.nii.gz"):
                pred_file_name = file.replace("_gt", "_label")
                filepaths.append(
                    Filepath(
                        path=os.path.join(subdir, file),
                        save_path=os.path.join(output_dir, pred_file_name),
                    )
                )
    return filepaths


class Preprocessor:
    def __init__(
        self,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: dict,
    ):
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.preprocessor = DefaultPreprocessor(verbose=True)

    def preprocess_image(self, image_filepath: str):
        """
        - load volume using SimpleITKIO
        - normalize volume
        - resample volume to lower resolution, e.g.
            (1,44,640,640) -> (1,44,410,410)

        data: normalized, cropped volume
        seg: cropped foreground mask
        data_properties: data properties loaded alongisde the original volume
        """
        data, seg, data_properties = self.preprocessor.run_case(
            image_files=[image_filepath],
            seg_file=None,
            plans_manager=self.plans_manager,
            configuration_manager=self.configuration_manager,
            dataset_json=self.dataset_json,
        )
        data = torch.from_numpy(data).to(
            dtype=torch.float32, memory_format=torch.contiguous_format
        )
        return data, data_properties


class Predictor:
    def __init__(self, network_config: NetworkConfig, dataset_json: dict):
        self.network_config = network_config
        self.device = torch.device("cuda")

        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=True,
            allow_tqdm=True,
            verbose_preprocessing=True,
        )
        self.predictor.manual_initialization(
            network=network_config.network,
            plans_manager=network_config.plans_manager,
            configuration_manager=network_config.configuration_manager,
            parameters=None,
            dataset_json=dataset_json,
            trainer_name=network_config.trainer_name,
            inference_allowed_mirroring_axes=network_config.inference_allowed_mirroring_axes,
        )

    def predict(self, data: torch.Tensor, data_properties: dict):
        """
        data should be preprocessed
        """
        import ipdb

        ipdb.set_trace()
        # same shape as preprocessed data
        prediction = self.predictor.predict_sliding_window_return_logits(data).to("cpu")

        # is_cascaded_mask = self.network_config.configuration_manager.configuration.get("is_cascaded_mask", False)
        # if is_cascaded_mask:
        #     # prediction.shape (4,44,574,574)
        #     # mask shape (1,44,574,574)
        #     seg = preprocessed["seg"]
        #     seg[seg < 0] = 0
        #     mask = torch.from_numpy(seg).to(torch.bool)

        #     cascaded_mask_dilation = self.configuration_manager.configuration.get("cascaded_mask_dilation", 0)
        #     if cascaded_mask_dilation > 0:
        #         from mbas.utils.binary_dilation_transform import binary_dilation_transform
        #         mask[0] = binary_dilation_transform(
        #             mask[0], cascaded_mask_dilation
        #         )

        #     # background class-0 prediction is the first channel
        #     # set the un-masked region (background) to 1
        #     # leave the masked region (foreground) as is
        #     prediction[0] = torch.where(mask, prediction[0], 1.0)
        #     # zero out the background for the other channels
        #     prediction[1:] = prediction[1:] * mask

        seg = convert_predicted_logits_to_segmentation_with_correct_shape(
            prediction,
            self.predictor.plans_manager,
            self.predictor.configuration_manager,
            self.predictor.label_manager,
            data_properties,
            return_probabilities=False,
        )
        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return seg

    # # def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
    # #     prediction = None
    # #     for params in self.list_of_parameters:

    # #         # messing with state dict names...
    # #         if not isinstance(self.network, OptimizedModule):
    # #             self.network.load_state_dict(params)
    # #         else:
    # #             self.network._orig_mod.load_state_dict(params)

    # #         # why not leave prediction on device if perform_everything_on_device? Because this may cause the
    # #         # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
    # #         # this actually saves computation time
    # #         if prediction is None:
    # #             prediction = self.predict_sliding_window_return_logits(data).to('cpu')
    # #         else:
    # #             prediction += self.predict_sliding_window_return_logits(data).to('cpu')

    # #     if len(self.list_of_parameters) > 1:
    # #         prediction /= len(self.list_of_parameters)

    # #     if self.verbose: print('Prediction done')
    # #     torch.set_num_threads(n_threads)
    # #     return prediction

    # @torch.inference_mode()
    # def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> torch.Tensor:
    #     assert isinstance(input_image, torch.Tensor)
    #     self.network = self.network.to(self.device)
    #     self.network.eval()

    #     empty_cache(self.device)

    #     with torch.autocast(self.device.type, enabled=True):
    #         assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

    #         if self.verbose:
    #             print(f'Input shape: {input_image.shape}')
    #             print("step_size:", self.tile_step_size)
    #             print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

    #         # if input_image is smaller than tile_size we need to pad it to tile_size.
    #         data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
    #                                                    'constant', {'value': 0}, True,
    #                                                    None)

    #         slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

    #         if self.perform_everything_on_device:
    #             # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
    #             try:
    #                 predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
    #                                                                                        self.perform_everything_on_device)
    #             except RuntimeError:
    #                 print(
    #                     'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
    #                 empty_cache(self.device)
    #                 predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
    #         else:
    #             predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
    #                                                                                    self.perform_everything_on_device)

    #         empty_cache(self.device)
    #         # revert padding
    #         predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

    # def _internal_predict_sliding_window_return_logits(self,
    #                                                    data: torch.Tensor,
    #                                                    slicers,
    #                                                    do_on_device: bool = True,
    #                                                    ):
    #     predicted_logits = n_predictions = prediction = gaussian = workon = None
    #     results_device = self.device if do_on_device else torch.device('cpu')

    #     try:
    #         empty_cache(self.device)

    #         # move data to device
    #         if self.verbose:
    #             print(f'move image to device {results_device}')
    #         data = data.to(results_device)

    #         # preallocate arrays
    #         if self.verbose:
    #             print(f'preallocating results arrays on device {results_device}')
    #         predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
    #                                        dtype=torch.half,
    #                                        device=results_device)
    #         n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

    #         if self.use_gaussian:
    #             gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
    #                                         value_scaling_factor=10,
    #                                         device=results_device)
    #         else:
    #             gaussian = 1

    #         if not self.allow_tqdm and self.verbose:
    #             print(f'running prediction: {len(slicers)} steps')
    #         for sl in slicers:
    #             workon = data[sl][None]
    #             workon = workon.to(self.device)

    #             prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

    #             if self.use_gaussian:
    #                 prediction *= gaussian
    #             predicted_logits[sl] += prediction
    #             n_predictions[sl[1:]] += gaussian

    #         predicted_logits /= n_predictions
    #         # check for infs
    #         if torch.any(torch.isinf(predicted_logits)):
    #             raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
    #                                'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
    #                                'predicted_logits to fp32')
    #     except Exception as e:
    #         del predicted_logits, n_predictions, prediction, gaussian, workon
    #         empty_cache(self.device)
    #         empty_cache(results_device)
    #         raise e
    #     return predicted_logits

    # def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
    #     slicers = []
    #     if len(self.configuration_manager.patch_size) < len(image_size):
    #         assert len(self.configuration_manager.patch_size) == len(
    #             image_size) - 1, 'if tile_size has less entries than image_size, ' \
    #                              'len(tile_size) ' \
    #                              'must be one shorter than len(image_size) ' \
    #                              '(only dimension ' \
    #                              'discrepancy of 1 allowed).'
    #         steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
    #                                                  self.tile_step_size)
    #         if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
    #                                f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
    #                                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
    #         for d in range(image_size[0]):
    #             for sx in steps[0]:
    #                 for sy in steps[1]:
    #                     slicers.append(
    #                         tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
    #                                                  zip((sx, sy), self.configuration_manager.patch_size)]]))
    #     else:
    #         steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
    #                                                  self.tile_step_size)
    #         if self.verbose: print(
    #             f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
    #             f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
    #         for sx in steps[0]:
    #             for sy in steps[1]:
    #                 for sz in steps[2]:
    #                     slicers.append(
    #                         tuple([slice(None), *[slice(si, si + ti) for si, ti in
    #                                               zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
    #     return slicers


def predict_main(gpu: str, input_dir: str, output_dir: str, model_pth: str):
    assert os.path.exists(input_dir), f"Input directory {input_dir} does not exist"
    assert os.path.exists(model_pth), f"Model path {model_pth} does not exist"

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # might be necessary
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving ouputs to {output_dir}")

    with open(model_pth, "rb") as f:
        parameters = pickle.load(f)

    image_filepaths = index_images(input_dir, output_dir)

    stage1_network = initialize_model(
        parameters.stage1_checkpoint, parameters.stage1_dataset, parameters.stage1_plans
    )
    # stage2_model = initialize_model(
    #     parameters.stage2_checkpoint, parameters.stage2_dataset, parameters.stage2_plans
    # )
    stage1_preprocessor = Preprocessor(
        stage1_network.plans_manager,
        stage1_network.configuration_manager,
        parameters.stage1_dataset,
    )
    stage1_predictor = Predictor(stage1_network, parameters.stage1_dataset)

    data, data_properties = stage1_preprocessor.preprocess_image(
        image_filepaths[0].path
    )
    seg = stage1_predictor.predict(data, data_properties)

    # consider preprocessing the segmentation from 1st stage
    # do you need to convert it to one-hot? probably not

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
