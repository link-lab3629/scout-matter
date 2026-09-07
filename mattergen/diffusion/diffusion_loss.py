from functools import partial
from typing import Callable, Dict

import torch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.diffusion.coordination_loss import (  # Public compatibility re-exports; coordination implementations live in; coordination_loss.py.  # noqa: E501
    COORDINATION_CONFIG_KEYS,
    DEFAULT_COORDINATION_ALPHA,
    DEFAULT_COORDINATION_CN_TEMPERATURE,
    DEFAULT_COORDINATION_CN_TOLERANCE,
    DEFAULT_COORDINATION_MARGIN,
    DEFAULT_COORDINATION_MODE,
    DEFAULT_COORDINATION_SATISFACTION_WEIGHT,
    DEFAULT_COORDINATION_TEMPERATURE,
    INTER_ATOMIC_CUTOFF,
    compute_mean_coordination,
    compute_ranked_coordination,
    compute_target_coordination_share,
    compute_target_share,
    dominant_environment_loss,
    environment_loss,
    group_coordination_loss,
    group_target_coordination_loss,
    mean_coordination_loss,
    ranked_coordination_loss,
    target_coordination_loss,
    target_coordination_share_loss,
)

__all__ = [
    "COORDINATION_CONFIG_KEYS",
    "DEFAULT_COORDINATION_ALPHA",
    "DEFAULT_COORDINATION_CN_TEMPERATURE",
    "DEFAULT_COORDINATION_CN_TOLERANCE",
    "DEFAULT_COORDINATION_MARGIN",
    "DEFAULT_COORDINATION_MODE",
    "DEFAULT_COORDINATION_SATISFACTION_WEIGHT",
    "DEFAULT_COORDINATION_TEMPERATURE",
    "INTER_ATOMIC_CUTOFF",
    "LOSS_REGISTRY",
    "compute_mean_coordination",
    "compute_ranked_coordination",
    "compute_target_coordination_share",
    "compute_target_share",
    "dominant_environment_loss",
    "environment_loss",
    "group_coordination_loss",
    "group_target_coordination_loss",
    "make_combined_loss",
    "mean_coordination_loss",
    "new_loss",
    "ranked_coordination_loss",
    "target_coordination_loss",
    "target_coordination_share_loss",
    "volume",
    "volume_loss",
    "volume_pa",
    "volume_pa_loss",
]


def volume(x, t):
    """Batched volume loss: computes the absolute difference between each actual volume and the
    target.

    x.cell: [N, 3, 3] target: float Returns: [N] tensor of losses
    """
    assert isinstance(x, ChemGraph), "x must be a ChemGraph object"
    cell = x.cell  # shape: [B, 3, 3]
    if cell is None:
        raise ValueError("ChemGraph has no cell attribute set.")
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    # a, b, c: [N, 3]
    a, b, c = cell[:, 0, :], cell[:, 1, :], cell[:, 2, :]
    # dot(a, cross(b, c)): [N]
    # cross(b, c): [N, 3]
    return torch.abs(torch.sum(a * torch.cross(b, c, dim=1), dim=1))


def volume_loss(x, t, target):
    """Batched volume loss: computes the absolute difference between each actual volume and the
    target.

    x.cell: [N, 3, 3] target: float Returns: [N] tensor of losses
    """
    vol = volume(x, t)
    # Ensure target is broadcastable
    target_tensor = torch.as_tensor(target, dtype=vol.dtype, device=vol.device)
    loss = torch.abs(vol - target_tensor)
    return 10**-5 * loss


def volume_pa(x, t):
    """Batched computatuion of volume per atom."""
    return volume(x, t) / x.num_atoms


def volume_pa_loss(x, t, target):
    """Batched computatuion of volume per atom."""
    vol_pa = volume_pa(x, t)
    target_tensor = torch.as_tensor(target, dtype=vol_pa.dtype, device=vol_pa.device)
    loss = torch.abs(vol_pa - target_tensor)
    return loss


def new_loss(x, t, target) -> torch.Tensor:
    """Placeholder for a user-defined guidance loss.

    Implement this function before passing ``new_loss`` through the guidance CLI.
    """
    raise NotImplementedError("Implement new_loss before using it.")


def make_combined_loss(guidance_dict: dict) -> callable:
    """Returns a loss function that combines all guidance losses defined in guidance_dict.

    Each key in guidance_dict must be in LOSS_REGISTRY, and the value is the target. More
    flexibility can be allowed, the value can be a dict containing parameters for the loss function.
    """
    partial_losses = []
    for loss_name, target in guidance_dict.items():
        if loss_name not in LOSS_REGISTRY:
            raise ValueError(
                f"Loss '{loss_name}' not found in LOSS_REGISTRY.",
                f"Available losses: {list(LOSS_REGISTRY.keys())}",
            )
        base_loss = LOSS_REGISTRY[loss_name]
        partial_losses.append(partial(base_loss, target=target))

    def combined_loss(x, t):
        # TODO: Verify that a simple sum is appropriate for combining the losses
        return sum(loss(x, t) for loss in partial_losses)

    return combined_loss


LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "volume": volume_loss,
    "volume_pa": volume_pa_loss,
    "mean_coordination": mean_coordination_loss,
    "target_coordination_share": target_coordination_share_loss,
    "ranked_coordination": ranked_coordination_loss,
    "target_coordination": target_coordination_loss,
    "group_coordination": group_coordination_loss,
    "group_target_coordination": group_target_coordination_loss,
    "environment": environment_loss,
    "dominant_environment": dominant_environment_loss,
    "new_loss": new_loss,  # Placeholder for a new loss function
    # Add more loss functions as needed
}
