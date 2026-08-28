from functools import partial
from typing import Callable, Dict

import pandas as pd
import torch
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.diffusion.coordination_loss import (
    # Public compatibility re-exports; coordination implementations live in
    # coordination_loss.py.
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
    "clear_globals",
    "composition",
    "compute_mean_coordination",
    "compute_ranked_coordination",
    "compute_target_coordination_share",
    "compute_target_share",
    "dominant_environment_loss",
    "energy",
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



PDIAG = None
calc = None
converter = None
species_pairs = None
target_values = None
r_cuts = None
target_tensor = None

def clear_globals():
    global PDIAG, calc, converter, species_pairs, target_values, r_cuts, target_tensor
    PDIAG = None
    calc = None
    converter = None
    species_pairs = None
    target_values = None
    r_cuts = None
    target_tensor = None



def volume(x, t):
    """
    Batched volume loss: computes the absolute difference between each actual volume and the target.
    x.cell: [N, 3, 3]
    target: float
    Returns: [N] tensor of losses
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
    """
    Batched volume loss: computes the absolute difference between each actual volume and the target.
    x.cell: [N, 3, 3]
    target: float
    Returns: [N] tensor of losses
    """
    vol = volume(x, t)
    # Ensure target is broadcastable
    target_tensor = torch.as_tensor(target, dtype=vol.dtype, device=vol.device)
    loss = torch.abs(vol - target_tensor)
    return 10 ** -5 * loss

def volume_pa(x, t):
    """
    Batched computatuion of volume per atom.
    """
    return volume(x, t) / x.num_atoms

def volume_pa_loss(x, t, target):
    """
    Batched computatuion of volume per atom.
    """
    vol_pa = volume_pa(x,t)
    target_tensor = torch.as_tensor(target, dtype=vol_pa.dtype, device=vol_pa.device)
    loss = torch.abs(vol_pa - target_tensor)
    return loss


def composition(num, pos):
    """
    Computes the composition of a list of atoms.
    li is a list of int with 101 beeing an empty atom.
    Returns a list of strings with the chemical symbols of the atoms.
    Example: [1, 101, 8, 8, 101] -> ['H', 'O', 'O']
    """
    return num[num != 101], pos[num != 101]


def energy(x, t, target=None):
    """
    Computes the energy above the hull for a given composition and energy.
    x is a chemgraph batch
    The function uses a precomputed phase diagram to determine the energy above the hull.
    """
    from mattersim.datasets.utils.convertor import ChemGraphBatchConvertor
    from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
    global calc
    global converter
    if calc is None:
        checkpoint = torch.load("/path/to/mattersim_torch/pretrained_models/mattersim-v1.0.0-1M.pth",
                                map_location="cuda")
        model = M3Gnet(**checkpoint["model_args"], device="cuda")  # Add arguments as needed for your configuration
        model.load_state_dict(checkpoint["model"])  # Load the model state dict, ensure it's on cuda
        model.eval()  # Set to evaluation mode for inference
        model = model.to(x.pos.device)  # Move model to the same device as x
    if converter is None:
        converter = ChemGraphBatchConvertor(twobody_cutoff=5.0, threebody_cutoff=4.0, pbc=True)
    if not isinstance(x, ChemGraph):
        raise ValueError("x must be a ChemGraph object")

    inputs = converter.convert(x)
    energies = []
    for input in inputs:
        if input is None:
            # If no atoms, append 0 to results
            energies.append(torch.zeros(1, device=x.pos.device) * x.pos.sum() * x.cell.sum())
        else:
            temp = model(input)
            if temp.isnan().any():
                # If NaN, append 0 to results
                energies.append(torch.zeros(1, device=x.pos.device) * x.pos.sum() * x.cell.sum())
            else:
                energies.append(temp)  # Otherwise compute the energy estimate
    energies = torch.stack(energies)  # Stack the energies into a tensor
    return energies


def _energy_hull(x):
    """
    Computes the energy above the hull for a given composition and energy.
    x is a (Compo, Energy) tuple (str, float)
    CSV : Compo , Energy
    """
    dir = "/path/to/mattergenbis/phase_diagram/"  # This should be the directory where the phase diagram is saved
    global PDIAG
    if PDIAG is None:
        # Load the CSV file only once
        csv = pd.read_csv(dir + "LiCoO.csv")
        li = [PDEntry(composition=Composition(csv["Formula"][i]), energy=csv["Energy"][i]) for i in range(len(csv))]
        PDIAG = PhaseDiagram(li)
        del csv, li
    x_ = PDEntry(composition=Composition(x[0]), energy=x[1])  # Assuming x has composition and energy attributes
    above_hull = PDIAG.get_e_above_hull(x_)
    return above_hull


def new_loss(x, t, target) -> torch.Tensor:
    """
    Example of a new loss function.
    This is just a placeholder and should be replaced with an actual implementation.
    """
    # x : ChemGraph object
    # t : timestep
    # target : target value
    pass


def make_combined_loss(guidance_dict: dict) -> callable:
    """
    Returns a loss function that combines all guidance losses defined in guidance_dict.
    Each key in guidance_dict must be in LOSS_REGISTRY, and the value is the target.
    More flexibility can be allowed, the value can be a dict containing parameters for the loss function.
    """
    partial_losses = []
    for loss_name, target in guidance_dict.items():
        if loss_name not in LOSS_REGISTRY:
            raise ValueError(
                f"Loss '{loss_name}' not found in LOSS_REGISTRY.",
                f"Available losses: {list(LOSS_REGISTRY.keys())}"
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
    # "energy": energy,
    "new_loss": new_loss,  # Placeholder for a new loss function
    # Add more loss functions as needed
}
