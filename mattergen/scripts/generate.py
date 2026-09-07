# Modified for scout-matter.
# Added CLI options for guidance, recurrence, backward steps, and GPU selection.
# See UPSTREAM_CHANGES.md for the upstream baseline and change inventory.

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Literal

import fire

from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_classes import PRETRAINED_MODEL_NAME, MatterGenCheckpointInfo
from mattergen.diffusion.diffusion_loss import make_combined_loss
from mattergen.generator import CrystalGenerator


def main(
    output_path: str,
    pretrained_name: PRETRAINED_MODEL_NAME | None = None,
    model_path: str | None = None,
    batch_size: int = 64,
    num_batches: int = 1,
    config_overrides: list[str] | None = None,
    checkpoint_epoch: Literal["best", "last"] | int = "last",
    properties_to_condition_on: TargetProperty | None = None,
    sampling_config_path: str | None = None,
    sampling_config_name: str = "default",
    sampling_config_overrides: list[str] | None = None,
    record_trajectories: bool = True,
    diffusion_guidance_factor: float | None = None,
    strict_checkpoint_loading: bool = True,
    target_compositions: list[dict[str, int]] | None = None,
    guidance: dict | str | None = None,
    diffusion_loss_weight: float | list[float] = 1.0,
    print_loss: bool = False,
    self_rec_steps: int = 1,
    back_step: int = 0,
    gpu_memory_gb: float | None = None,
    algo: int = 0,
    force_gpu: int | None = None,
):
    """Evaluate diffusion model against molecular metrics.

    Args:     model_path: Path to DiffusionLightningModule checkpoint directory.     output_path:
    Path to output directory.     config_overrides: Overrides for the model config, e.g.,
    `model.num_layers=3 model.hidden_dim=128`.     properties_to_condition_on: Property value to
    draw conditional sampling with respect to.         When this value is an empty dictionary
    (default), unconditional samples are drawn.     sampling_config_path: Path to the sampling
    config file. (default: None, in which case we         use `DEFAULT_SAMPLING_CONFIG_PATH` from
    explorers.common.utils.utils.py)     sampling_config_name: Name of the sampling config
    (corresponds to         `{sampling_config_path}/{sampling_config_name}.yaml` on disk). (default:
    default)     sampling_config_overrides: Overrides for the sampling config, e.g.,
    `condition_loader_partial.batch_size=32`.     load_epoch: Epoch to load from the checkpoint. If
    None, the best epoch is loaded.         (default: None)     record: Whether to record the
    trajectories of the generated structures. (default: True)     strict_checkpoint_loading: Whether
    to raise an exception when not all parameters from the         checkpoint can be matched to the
    model.     target_compositions: List of dictionaries with target compositions to condition on.
    Each         dictionary should have the form `{element: number_of_atoms}`. If None, the target
    compositions are not conditioned on. Only supported for models trained for crystal
    structure prediction (CSP) (default: None)     guidance: Dictionary with guidance parameters for
    the diffusion model. The keys are the         names of the properties to condition on, and the
    values are the target values for         those properties.     diffusion_loss_weight: Weight for
    the diffusion loss. (default: 1.0)     print_loss: Whether to print the loss during generation.
    (default: False)     self_rec_steps: Number of self-recurrence steps to perform during
    generation. (default: 1)     back_step: Number of steps of backward updates to do during
    generation. (default: 0)     gpu_memory_gb: Amount of GPU memory in GB to use for the
    generation.         (default: batch_size * 0,336)     algo: Algorithm to use for the generation.
    Algorithm 0 does the correction outside the         self recurrence loop, Algorithm 1 does it
    inside before the forward corruption, and         Algorithm 2 does it inside after the forward
    corruption.

    NOTE: When specifying dictionary values via the CLI, make sure there is no whitespace between
    the key and value, e.g., `--properties_to_condition_on={key1:value1}`.
    """
    assert (
        pretrained_name is not None or model_path is not None
    ), "Either pretrained_name or model_path must be provided."
    assert (
        pretrained_name is None or model_path is None
    ), "Only one of pretrained_name or model_path can be provided."

    if gpu_memory_gb is None:
        # Default GPU memory is 1.125 MB per batch size
        gpu_memory_gb = batch_size * 1.125

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save only the input parameters
    with open(os.path.join(output_path, "input_parameters.txt"), "w") as f:
        for k, v in locals().items():
            if k == "f":
                continue
            f.write(f"{k}: {v}\n")

    sampling_config_overrides = sampling_config_overrides or []
    config_overrides = config_overrides or []
    # Disable generating element types which are not supported or not in the desired chemical
    # system (if provided).
    config_overrides += [
        (
            "++lightning_module.diffusion_module.model.element_mask_func="
            "{_target_:'mattergen.denoiser.mask_disallowed_elements',_partial_:True}"
        )
    ]
    properties_to_condition_on = properties_to_condition_on or {}
    target_compositions = target_compositions or []

    if pretrained_name is not None:
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(
            pretrained_name, config_overrides=config_overrides
        )
    else:
        checkpoint_info = MatterGenCheckpointInfo(
            model_path=Path(model_path).resolve(),
            load_epoch=checkpoint_epoch,
            config_overrides=config_overrides,
            strict_checkpoint_loading=strict_checkpoint_loading,
        )
    _sampling_config_path = Path(sampling_config_path) if sampling_config_path is not None else None

    loss_fn = None
    if guidance is not None:
        # Ensure guidance is a dictionary with string keys and numeric values
        if not isinstance(guidance, dict) or not all(
            isinstance(k, str)
            and (isinstance(v, (int, float)) or isinstance(v, list) or isinstance(v, dict))
            for k, v in guidance.items()
        ):
            raise ValueError(
                "Guidance must map string keys to numeric values, lists, or dictionaries."
            )
        # Create the combined loss function based on the provided guidance
        loss_fn = make_combined_loss(guidance)

    if isinstance(diffusion_loss_weight, (float)):
        diffusion_loss_weight = [diffusion_loss_weight] * 2

    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        properties_to_condition_on=properties_to_condition_on,
        batch_size=batch_size,
        num_batches=num_batches,
        sampling_config_name=sampling_config_name,
        sampling_config_path=_sampling_config_path,
        sampling_config_overrides=sampling_config_overrides,
        record_trajectories=record_trajectories,
        diffusion_guidance_factor=(
            diffusion_guidance_factor if diffusion_guidance_factor is not None else 0.0
        ),
        target_compositions_dict=target_compositions,
        diffusion_loss_fn=loss_fn,  # NEW
        diffusion_loss_weight=diffusion_loss_weight,  # NEW
        print_loss=print_loss,  # NEW
        self_rec_steps=self_rec_steps,  # NEW
        back_step=back_step,  # NEW
        gpu_memory_gb=gpu_memory_gb,  # NEW
        algo=algo,  # NEW
        force_gpu=force_gpu,  # NEW
    )
    generator.generate(output_dir=Path(output_path))
    print(f"Generated structures saved to {output_path}")


def _main():
    # Use fire instead of argparse to allow specifying dictionary values via the CLI.
    fire.Fire(main)
    # For debugging, pass a command string to Fire with the desired CLI options.


if __name__ == "__main__":
    _main()
# Example:
# mattergen-generate "results/chemical_system/Pd-Ni-H_env" \
#   --pretrained-name=chemical_system --batch_size=1 \
#   --properties_to_condition_on="{'chemical_system':'Li-Co-O'}" \
#   --record_trajectories=False --diffusion_guidance_factor=2.0 \
#   --guidance="{'environment': {'Co-O':6}}" --diffusion_loss_weight=1.0 --print_loss=True
