from typing import Union
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)


def determine_num_input_channels(
    plans_manager: PlansManager,
    configuration_or_config_manager: Union[str, ConfigurationManager],
    dataset_json: dict,
) -> int:
    if isinstance(configuration_or_config_manager, str):
        config_manager = plans_manager.get_configuration(
            configuration_or_config_manager
        )
    else:
        config_manager = configuration_or_config_manager

    label_manager = plans_manager.get_label_manager(dataset_json)
    num_modalities = (
        len(dataset_json["modality"])
        if "modality" in dataset_json.keys()
        else len(dataset_json["channel_names"])
    )

    # cascade has different number of input channels
    is_cascaded_mask = config_manager.configuration.get("is_cascaded_mask", False)
    if config_manager.previous_stage_name is not None and not is_cascaded_mask:
        num_label_inputs = len(label_manager.foreground_labels)
        num_input_channels = num_modalities + num_label_inputs
    else:
        num_input_channels = num_modalities
    return num_input_channels
