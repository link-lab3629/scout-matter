# Modified for scout-matter.
# Added GPU selection by free memory and explicit GPU index.
# See UPSTREAM_CHANGES.md for the upstream baseline and change inventory.

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Note that importing this module has two side effects: 1.

It sets the environment variable `PROJECT_ROOT` to the root of the explorers project. 2. It
registers a new resolver for OmegaConf, `eval`, which allows us to use `eval` in our config files.
"""

import os
from functools import lru_cache
from pathlib import Path

import torch
from omegaconf import OmegaConf

device = None


def get_device_name() -> torch.device:
    """Returns the name of the current device (GPU or CPU)."""
    global device
    if device is None:
        device = get_device()
    return device


@lru_cache
def get_device(min_gpu_mem_gb=8, force_gpu: int | None = None) -> torch.device:
    """Returns the GPU device with the most free memory, and at least min_gpu_mem_gb free memory, if
    available.

    Otherwise, returns CPU.
    """
    global device
    if force_gpu is not None:
        if torch.cuda.is_available() and force_gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{force_gpu}")
            print(f"Using requested GPU {force_gpu} with at least {min_gpu_mem_gb} GB free memory.")
            return device
        else:
            print(f"Requested GPU {force_gpu} is not available. Checking for others.")
    if torch.cuda.is_available():
        max_mem = 0
        for i in range(torch.cuda.device_count()):
            free_mem_bytes, _ = torch.cuda.mem_get_info(i)
            if free_mem_bytes >= min_gpu_mem_gb * (1024**3) and free_mem_bytes > max_mem:
                device = torch.device(f"cuda:{i}")
                max_mem = free_mem_bytes
    else:
        print(f"No GPU with at least {min_gpu_mem_gb} GB free memory found. Falling back to CPU.")
        device = torch.device("cpu")
    print(f"Using device: {device} with at least {min_gpu_mem_gb} GB free memory.")
    return device


@lru_cache
def get_pyg_device() -> torch.device:
    """Some operations of pyg don't work on MPS, so fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


MODELS_PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(f"MODELS_PROJECT_ROOT: {MODELS_PROJECT_ROOT}")

# Set environment variable PROJECT_ROOT so that hydra / OmegaConf can access it.
os.environ["PROJECT_ROOT"] = str(MODELS_PROJECT_ROOT)  # for hydra

DEFAULT_SAMPLING_CONFIG_PATH = Path(__file__).resolve().parents[3] / "sampling_conf"
PROPERTY_SOURCE_IDS = [
    "dft_mag_density",
    "dft_bulk_modulus",
    "dft_shear_modulus",
    "energy_above_hull",
    "formation_energy_per_atom",
    "space_group",
    "hhi_score",
    "ml_bulk_modulus",
    "chemical_system",
    "dft_band_gap",
]

SELECTED_ATOMIC_NUMBERS = [
    1,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    37,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    55,
    56,
    57,
    58,
    59,
    60,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
]
MAX_ATOMIC_NUM = 100


# Set `eval` resolver
def try_eval(s):
    """This is a custom resolver for OmegaConf that allows us to use `eval` in our config files with
    the syntax `${eval:'${foo} + ${bar}'}

    See:
    https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html#id1
    """
    try:
        return eval(s)
    except Exception as e:
        print(f"Calling eval on string {s} raised exception {e}")
        raise


OmegaConf.register_new_resolver("eval", try_eval)
