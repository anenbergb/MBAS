import argparse
import os
import sys
import pickle
import time

from typing import Optional, List

import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from torch.backends import cudnn

from mbas.training.mbasTrainer import mbasTrainer
from mbas.utils.binary_dilation_transform import binary_dilation_transform

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

from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)

from nnunetv2.postprocessing.remove_connected_components import (
    remove_all_but_largest_component_from_segmentation,
)


@dataclass
class Parameters1Stage:
    checkpoint: dict
    dataset: dict
    plans: dict
    postprocessing_kwargs: list
    skip_postprocessing: bool


@dataclass
class Parameters2Stage:
    stage1_checkpoint: dict
    stage1_dataset: dict
    stage1_plans: dict
    stage1_postprocessing_kwargs: list

    stage2_checkpoint: dict
    stage2_dataset: dict
    stage2_plans: dict
    stage2_postprocessing_kwargs: list


@dataclass
class NetworkConfig:
    network: nn.Module
    plans_manager: PlansManager
    configuration_manager: ConfigurationManager
    trainer_name: str
    inference_allowed_mirroring_axes: List[int]
    postprocessing_kwargs: List[dict]
    skip_postprocessing: bool


def initialize_model(
    checkpoint: dict,
    dataset_json: dict,
    plans: dict,
    postprocessing_kwargs: list,
    compiled_model: bool = False,
    skip_postprocessing: bool = False,
):
    plans_manager = PlansManager(plans)
    trainer_name = checkpoint["trainer_name"]
    configuration_name = checkpoint["init_args"]["configuration"]
    inference_allowed_mirroring_axes = checkpoint.get(
        "inference_allowed_mirroring_axes"
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
    if compiled_model:
        print("Using torch.compile")
        network = torch.compile(network)

    network.eval()
    print(f"Network initialized: {type(network)}")
    return NetworkConfig(
        network=network,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        trainer_name=trainer_name,
        inference_allowed_mirroring_axes=inference_allowed_mirroring_axes,
        postprocessing_kwargs=postprocessing_kwargs,
        skip_postprocessing=skip_postprocessing,
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
    filepaths = sorted(filepaths, key=lambda x: x.path)
    return filepaths


class Preprocessor:
    def __init__(
        self,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: dict,
        verbose: bool = False,
    ):
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.preprocessor = DefaultPreprocessor(verbose=verbose)
        self.image_rw = self.plans_manager.image_reader_writer_class()

    def load_image(self, image_filepath: str):
        data, data_properties = self.image_rw.read_images([image_filepath])
        return data, data_properties

    def preprocess_image(
        self, data: np.ndarray, data_properties: dict, seg: Optional[np.ndarray] = None
    ):
        """
        - load volume using SimpleITKIO
        - normalize volume
        - resample volume to lower resolution, e.g.
            (1,44,640,640) -> (1,44,410,410)

        data: normalized, cropped volume
        seg: Segmentation mask will be cropped and resampled alongside the volume.
            If seg is None, then the returned segmentation mask is just the nonzero mask
        data_properties: data properties loaded alongisde the original volume
        """
        if seg is not None:
            seg = seg.astype(np.int8)
            if seg.ndim == 3:
                seg = seg[np.newaxis, :]

        data_pp, seg_pp = self.preprocessor.run_case_npy(
            data,
            seg=seg,
            properties=data_properties,
            plans_manager=self.plans_manager,
            configuration_manager=self.configuration_manager,
            dataset_json=self.dataset_json,
        )
        data_pp = torch.from_numpy(data_pp).to(
            dtype=torch.float32, memory_format=torch.contiguous_format
        )
        seg_pp = torch.from_numpy(seg_pp).to(
            dtype=torch.float32, memory_format=torch.contiguous_format
        )
        return data_pp, seg_pp


class Predictor:
    def __init__(
        self, network_config: NetworkConfig, dataset_json: dict, verbose: bool = False
    ):
        self.network_config = network_config
        self.device = torch.device("cuda")

        self.predictor = nnUNetPredictor(
            tile_step_size=1.0,  # 1.0 for speedup. 0.5 for better results
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=verbose,
            allow_tqdm=verbose,
            verbose_preprocessing=verbose,
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

    def predict(
        self,
        data: torch.Tensor,
        data_properties: dict,
        seg_mask: Optional[torch.Tensor] = None,
        debug_return_intermediates: bool = False,
        skip_postprocessing: bool = False,
    ):
        """
        data should be preprocessed
        """
        # data shape (1,44,410,410)
        # prediction shape (2,44,410,410)
        prediction = self.predictor.predict_sliding_window_return_logits(data).to("cpu")

        is_cascaded_mask = self.network_config.configuration_manager.configuration.get(
            "is_cascaded_mask", False
        )
        if is_cascaded_mask and seg_mask is not None:
            # prediction.shape (4,44,574,574)
            # mask shape (1,44,574,574)
            seg_mask[seg_mask < 0] = 0
            mask = seg_mask.to(torch.bool)

            cascaded_mask_dilation = (
                self.network_config.configuration_manager.configuration.get(
                    "cascaded_mask_dilation", 0
                )
            )
            if cascaded_mask_dilation > 0:
                mask[0] = binary_dilation_transform(mask[0], cascaded_mask_dilation)

            # background class-0 prediction is the first channel
            # set the un-masked region (background) to 1
            # leave the masked region (foreground) as is
            prediction[0] = torch.where(mask, prediction[0], 1.0)
            # zero out the background for the other channels
            prediction[1:] = prediction[1:] * mask

        # seg shape (44, 640, 640)
        seg = convert_predicted_logits_to_segmentation_with_correct_shape(
            prediction,
            self.predictor.plans_manager,
            self.predictor.configuration_manager,
            self.predictor.label_manager,
            data_properties,
            return_probabilities=False,
        )
        if skip_postprocessing:
            print("Skipping postprocessing")
            # clear lru cache
            compute_gaussian.cache_clear()
            # clear device cache
            empty_cache(self.device)
            return seg

        # for the 2nd stage, remove all but the largest component for each of the labels independently
        seg_pp = self.apply_postprocessing(seg)

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        if debug_return_intermediates:
            return seg_pp, seg, prediction
        return seg_pp

    def apply_postprocessing(self, segmentation: np.ndarray):
        for kwargs in self.network_config.postprocessing_kwargs:
            segmentation = remove_all_but_largest_component_from_segmentation(
                segmentation, **kwargs
            )
        return segmentation


class Predictor2Stage:
    def __init__(self, parameters: Parameters2Stage, verbose: bool = False):
        self.verbose = verbose
        self.stage1_network = initialize_model(
            parameters.stage1_checkpoint,
            parameters.stage1_dataset,
            parameters.stage1_plans,
            parameters.stage1_postprocessing_kwargs,
            compiled_model=False,
        )
        self.stage2_network = initialize_model(
            parameters.stage2_checkpoint,
            parameters.stage2_dataset,
            parameters.stage2_plans,
            parameters.stage2_postprocessing_kwargs,
            compiled_model=False,
        )
        self.stage1_preprocessor = Preprocessor(
            self.stage1_network.plans_manager,
            self.stage1_network.configuration_manager,
            parameters.stage1_dataset,
            verbose=verbose,
        )
        self.stage2_preprocessor = Preprocessor(
            self.stage2_network.plans_manager,
            self.stage2_network.configuration_manager,
            parameters.stage2_dataset,
            verbose=verbose,
        )

        self.stage1_predictor = Predictor(
            self.stage1_network, parameters.stage1_dataset, verbose=verbose
        )
        self.stage2_predictor = Predictor(
            self.stage2_network, parameters.stage2_dataset, verbose=verbose
        )

        self.image_rw = self.stage2_network.plans_manager.image_reader_writer_class()

    def predict_and_save(self, image_filepath_struct: Filepath):
        t0 = time.time()
        data, data_properties = self.stage1_preprocessor.load_image(
            image_filepath_struct.path
        )
        if self.verbose:
            print(f"Loading image took {time.time() - t0:.2f}s")
        segmentation = self.predict(data, data_properties)
        self.image_rw.write_seg(
            segmentation, image_filepath_struct.save_path, data_properties
        )

    def predict(self, data: np.ndarray, data_properties: dict):
        t0 = time.time()
        data1_pp, _ = self.stage1_preprocessor.preprocess_image(data, data_properties)
        t1 = time.time()
        if self.verbose:
            print(f"Stage 1 Preprocessing took {t1 - t0:.2f}s")
        seg1_pp = self.stage1_predictor.predict(data1_pp, data_properties)
        t2 = time.time()
        if self.verbose:
            print(f"Stage 1 Prediction took {t2 - t1:.2f}s")
        data2_pp, seg1_pp2 = self.stage2_preprocessor.preprocess_image(
            data, data_properties, seg1_pp
        )
        t3 = time.time()
        if self.verbose:
            print(f"Stage 2 Preprocessing took {t3 - t2:.2f}s")
        seg2_pp = self.stage2_predictor.predict(data2_pp, data_properties, seg1_pp2)
        if self.verbose:
            print(f"Stage 2 Prediction took {time.time() - t3:.2f}s")
        return seg2_pp


class Predictor1Stage:
    def __init__(self, parameters: Parameters1Stage, verbose: bool = False):
        self.verbose = verbose
        self.network = initialize_model(
            parameters.checkpoint,
            parameters.dataset,
            parameters.plans,
            parameters.postprocessing_kwargs,
            compiled_model=False,
            skip_postprocessing=parameters.skip_postprocessing,
        )
        self.preprocessor = Preprocessor(
            self.network.plans_manager,
            self.network.configuration_manager,
            parameters.dataset,
            verbose=verbose,
        )

        self.predictor = Predictor(self.network, parameters.dataset, verbose=verbose)

        self.image_rw = self.network.plans_manager.image_reader_writer_class()

    def predict_and_save(self, image_filepath_struct: Filepath):
        t0 = time.time()
        data, data_properties = self.preprocessor.load_image(image_filepath_struct.path)
        if self.verbose:
            print(f"Loading image took {time.time() - t0:.2f}s")
        segmentation = self.predict(data, data_properties)
        self.image_rw.write_seg(
            segmentation, image_filepath_struct.save_path, data_properties
        )

    def predict(self, data: np.ndarray, data_properties: dict):
        t0 = time.time()
        data1_pp, _ = self.preprocessor.preprocess_image(data, data_properties)
        t1 = time.time()
        if self.verbose:
            print(f"Stage 1 Preprocessing took {t1 - t0:.2f}s")
        seg1_pp = self.predictor.predict(
            data1_pp,
            data_properties,
            skip_postprocessing=self.network.skip_postprocessing,
        )
        t2 = time.time()
        if self.verbose:
            print(f"Stage 1 Prediction took {t2 - t1:.2f}s")
        return seg1_pp


def predict_main(gpu: str, input_dir: str, output_dir: str, model_pth: str):
    start_time = time.time()
    assert os.path.exists(input_dir), f"Input directory {input_dir} does not exist"
    assert os.path.exists(model_pth), f"Model path {model_pth} does not exist"

    if torch.cuda.is_available():
        cudnn.deterministic = True
        cudnn.benchmark = True

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # might be necessary
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving ouputs to {output_dir}")

    with open(model_pth, "rb") as f:
        parameters = pickle.load(f)

    if isinstance(parameters, Parameters2Stage):
        print("Using 2-stage Predictor")
        predictor = Predictor2Stage(parameters, verbose=False)
    elif isinstance(parameters, Parameters1Stage):
        print("Using 1-stage Predictor")
        predictor = Predictor1Stage(parameters, verbose=False)
    else:
        raise ValueError("Invalid parameters")

    image_filepaths = index_images(input_dir, output_dir)
    init_time = time.time() - start_time
    iter_times = []
    for image_filepath_struct in image_filepaths:
        iter_start_time = time.time()
        predictor.predict_and_save(image_filepath_struct)
        iter_time = time.time() - iter_start_time
        print(
            f"Processing {os.path.basename(image_filepath_struct.path)} took {iter_time:.2f}s"
        )
        iter_times.append(iter_time)
    total_min, total_sec = divmod(time.time() - start_time, 60)
    print(f"Initialization time: {init_time:.2f}s")
    print(f"Average iteration time: {np.mean(iter_times):.2f}s")
    print(f"Total time: {int(total_min)}min {total_sec:.2f}s")


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
        default="/input",
        help="path to input",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/output", help="path to input"
    )
    parser.add_argument(
        "--model_pth",
        type=str,
        default="./save_pths/test.pth",
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
