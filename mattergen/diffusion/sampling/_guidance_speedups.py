"""Runtime optimizations for universal-guidance sampling.

The helpers in this module reduce repeated work while preserving the guidance
calculation. Optimizations are enabled by default and can be disabled with the
following environment variables when troubleshooting or comparing runtime
behavior:

    SCOUT_SLOW_GUIDANCE=1          -> use the conservative per-field autograd path
    SCOUT_DISABLE_SDE_CACHE=1      -> disable SDE-coefficient memoization
    SCOUT_DISABLE_COORD_CACHE=1    -> disable coordination-index memoization
"""

import contextlib
import os
import torch

import mattergen.diffusion.coordination_loss as _coord_mod
from mattergen.diffusion.coordination_loss import (
    DEFAULT_COORDINATION_ALPHA,
    DEFAULT_COORDINATION_MARGIN,
    DEFAULT_COORDINATION_TEMPERATURE,
    _as_atomic_number_tuple,
    _validate_one_sided_coordination_groups,
    _validate_target_coordination,
    _coordination_r_cut_per_center,
)


# ── SDE coefficient cache ─────────────────────────────────────────────────────
_SDE_CACHE: dict = {}


@contextlib.contextmanager
def cache_sde_coefficients(diffusion_module):
    """Memoize ``mean_coeff_and_std`` for the continuous SDE components.

    The cache is scoped to the denoising call and is cleared when the context
    exits. Each wrapped method is restored to its original implementation on
    exit, including when denoising raises an exception.
    """
    # MultiCorruption.corruptions contains both continuous SDEs and discrete
    # corruptions (atomic_numbers in MatterGen).  Only SDE objects expose
    # mean_coeff_and_std; wrapping the discrete corruption makes the context
    # fail before sampling even starts and can also make restoration partial.
    corruption = diffusion_module.corruption
    sdes = getattr(corruption, "sdes", None)
    if sdes is None:
        sdes = {
            field: corruption_fn
            for field, corruption_fn in corruption.corruptions.items()
            if hasattr(corruption_fn, "mean_coeff_and_std")
        }
    original_funcs = {}

    def get_cached_func(field, original_func):
        def cached_func(x, t, batch_idx, batch):
            # The sampler reuses the same timestep tensor across the
            # predictor/corrector sub-steps within one sampling step. Identity
            # checking avoids synchronizing the CUDA stream; a different tensor
            # is treated as a cache miss and evaluated normally.
            if _SDE_CACHE.get(field) is not None and _SDE_CACHE.get(field + "_t") is t:
                return _SDE_CACHE[field]
            res = original_func(x=x, t=t, batch_idx=batch_idx, batch=batch)
            _SDE_CACHE[field] = res
            _SDE_CACHE[field + "_t"] = t
            return res

        return cached_func

    try:
        for field, sde in sdes.items():
            original_funcs[field] = sde.mean_coeff_and_std
            sde.mean_coeff_and_std = get_cached_func(field, original_funcs[field])
        yield
    finally:
        for field, sde in sdes.items():
            sde.mean_coeff_and_std = original_funcs[field]
        _SDE_CACHE.clear()


# ── Coordination index cache ─────────────────────────────────────────────────
# Caches only the quantities that depend on atomic types (constant across a
# trajectory): the A/B index masks, the self-interaction mask, and the
# per-center cutoff radii. Those are computed under torch.no_grad (they carry no
# position gradient) and detached. The soft-neighbor counts themselves are
# recomputed every call from the live fractional coordinates with the same
# fractional-displacement -> Cartesian-distance evaluation as
# mattergen.diffusion.coordination_loss._soft_neighbor_counts_per_A_single.
# The tA==tB path (self-interaction subtraction) is preserved.
_COORD_CACHE: dict = {}
# Keep the type tensor's storage object alive while using its address as one
# component of a fast-path signature. Holding the storage object prevents
# allocator reuse from making a stale address look like the previous
# composition, while version, shape, and view metadata catch in-place edits.
_COORD_LAST: dict = {"storage": None, "signature": None, "key": None}

def _cached_soft_neighbor_counts_per_A_single(
    cell: torch.Tensor,
    frac: torch.Tensor,
    types,
    type_A,
    type_B,
    r_cut: float | None = None,
    alpha: float = DEFAULT_COORDINATION_ALPHA,
) -> torch.Tensor:
    frac = torch.as_tensor(
        frac, dtype=getattr(frac, "dtype", torch.float32), device=getattr(frac, "device", None)
    )
    cell = torch.as_tensor(cell, dtype=frac.dtype, device=frac.device)
    types = torch.as_tensor(types, dtype=torch.int64, device=frac.device)
    type_As = _as_atomic_number_tuple(type_A)
    type_Bs = _as_atomic_number_tuple(type_B)
    _validate_one_sided_coordination_groups(type_As, type_Bs)

    device = frac.device
    same_group = type_As == type_Bs
    # Fast, correct key: reuse the previous key when the SAME storage AND the
    # same type selection is passed again (the normal sampling path reuses the
    # composition tensor), avoiding a CUDA->CPU sync on every guidance step.
    # The type args must be part of the fast key because the cached indices
    # depend on type_A/type_B (e.g. the tA==tB self-interaction path).
    # Device and dtype are included because cached index/cutoff tensors are
    # device- and dtype-specific.
    storage = types.untyped_storage()
    fast_key = (
        storage.data_ptr(),
        types.storage_offset(),
        tuple(types.shape),
        tuple(types.stride()),
        types._version,
        type_As,
        type_Bs,
        r_cut,
        alpha,
        device,
        frac.dtype,
    )
    if (
        _COORD_LAST["storage"] is storage
        and _COORD_LAST["signature"] == fast_key
        and _COORD_LAST["key"] is not None
    ):
        key = _COORD_LAST["key"]
    else:
        key = (
            tuple(types.tolist()),
            type_As,
            type_Bs,
            r_cut,
            alpha,
            device,
            frac.dtype,
        )
        _COORD_LAST["storage"] = storage
        _COORD_LAST["signature"] = fast_key
        _COORD_LAST["key"] = key

    cached = _COORD_CACHE.get(key)
    if cached is None:
        with torch.no_grad():
            mask_A = torch.zeros_like(types, dtype=torch.bool)
            for type_A_i in type_As:
                mask_A = mask_A | (types == type_A_i)
            idx_A = mask_A.nonzero(as_tuple=True)[0].detach()

            if same_group:
                idx_B = idx_A
            else:
                mask_B = torch.zeros_like(mask_A, dtype=torch.bool)
                for type_B_i in type_Bs:
                    mask_B = mask_B | (types == type_B_i)
                idx_B = mask_B.nonzero(as_tuple=True)[0].detach()

            if same_group:
                self_interaction = torch.ones(
                    idx_A.numel(), dtype=torch.float32, device=device
                ).detach()
            elif set(type_As).isdisjoint(type_Bs):
                self_interaction = torch.zeros(
                    idx_A.numel(), dtype=torch.float32, device=device
                ).detach()
            else:
                center_types = types[idx_A]
                si = torch.zeros_like(center_types, dtype=torch.bool)
                for type_B_i in type_Bs:
                    si = si | (center_types == type_B_i)
                self_interaction = si.float().detach()

            r_cut_per_A = _coordination_r_cut_per_center(
                types[idx_A], type_Bs, dtype=frac.dtype, r_cut=r_cut
            ).detach()
        cached = (idx_A, idx_B, self_interaction, r_cut_per_A)
        _COORD_CACHE[key] = cached
    else:
        idx_A, idx_B, self_interaction, r_cut_per_A = cached

    if idx_A.numel() == 0 or idx_B.numel() == 0:
        return cell.sum() * frac.sum() * torch.zeros(1, device=device)

    if _coord_mod.shifts is None or _coord_mod.shifts.device != device or _coord_mod.shifts.dtype != frac.dtype:
        with torch.no_grad():
            _coord_mod.shifts = torch.stack(
                torch.meshgrid(
                    torch.arange(-1, 2, device=device, dtype=frac.dtype),
                    torch.arange(-1, 2, device=device, dtype=frac.dtype),
                    torch.arange(-1, 2, device=device, dtype=frac.dtype),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 3).detach()

    frac_A = frac[idx_A]  # keeps position gradient
    # For tA == tB the two index selections are identical.  Reusing frac_A
    # avoids a second advanced-indexing/gather operation in every loss call.
    frac_B = frac_A if same_group else frac[idx_B]  # keeps position gradient

    # Use the standard fractional-displacement -> Cartesian-distance sequence
    # while caching only the type-dependent index masks and cutoffs.
    frac_B_images = (frac_B.unsqueeze(1) + _coord_mod.shifts.unsqueeze(0)).reshape(
        -1, 3
    )
    d = frac_A.unsqueeze(1) - frac_B_images.unsqueeze(0)
    dc = torch.matmul(d, cell)
    dist = dc.norm(dim=-1)

    G = torch.sigmoid(alpha * (r_cut_per_A.unsqueeze(1) - dist))
    counts = G.sum(dim=1)
    counts = counts - self_interaction.to(dtype=counts.dtype)
    return counts

_MARGIN_CACHE: dict = {}
_MARGIN_LAST: dict = {"storage": None, "signature": None, "key": None}


def _cached_coordination_margin_penalties_per_A_single(
    cell: torch.Tensor,
    frac: torch.Tensor,
    types,
    type_A: int | list[int] | tuple[int, ...] | set[int],
    type_B: int | list[int] | tuple[int, ...] | set[int],
    *,
    target: float,
    r_cut: float | None = None,
    margin: float = DEFAULT_COORDINATION_MARGIN,
    temperature: float = DEFAULT_COORDINATION_TEMPERATURE,
) -> torch.Tensor:
    frac = torch.as_tensor(
        frac,
        dtype=getattr(frac, "dtype", torch.float32),
        device=getattr(frac, "device", None),
    )
    cell = torch.as_tensor(cell, dtype=frac.dtype, device=frac.device)
    types = torch.as_tensor(types, dtype=torch.int64, device=frac.device)
    type_As = _as_atomic_number_tuple(type_A)
    type_Bs = _as_atomic_number_tuple(type_B)
    _validate_one_sided_coordination_groups(type_As, type_Bs)

    target_int = _validate_target_coordination(target)
    margin = float(margin)
    temperature = float(temperature)
    if margin < 0.0:
        raise ValueError("coordination margin must be non-negative.")
    if temperature <= 0.0:
        raise ValueError("coordination temperature must be positive.")

    device = frac.device
    same_group = type_As == type_Bs
    storage = types.untyped_storage()
    fast_key = (
        storage.data_ptr(),
        types.storage_offset(),
        tuple(types.shape),
        tuple(types.stride()),
        types._version,
        type_As,
        type_Bs,
        target_int,
        r_cut,
        margin,
        temperature,
        device,
        frac.dtype,
    )
    if (
        _MARGIN_LAST["storage"] is storage
        and _MARGIN_LAST["signature"] == fast_key
        and _MARGIN_LAST["key"] is not None
    ):
        key = _MARGIN_LAST["key"]
    else:
        key = (
            tuple(types.tolist()),
            type_As,
            type_Bs,
            target_int,
            r_cut,
            margin,
            temperature,
            device,
            frac.dtype,
        )
        _MARGIN_LAST["storage"] = storage
        _MARGIN_LAST["signature"] = fast_key
        _MARGIN_LAST["key"] = key

    cached = _MARGIN_CACHE.get(key)
    if cached is None:
        with torch.no_grad():
            mask_A = torch.zeros_like(types, dtype=torch.bool)
            for type_A_i in type_As:
                mask_A = mask_A | (types == type_A_i)
            mask_B = torch.zeros_like(mask_A, dtype=torch.bool)
            for type_B_i in type_Bs:
                mask_B = mask_B | (types == type_B_i)
            idx_A = mask_A.nonzero(as_tuple=True)[0].detach()
            idx_B = mask_B.nonzero(as_tuple=True)[0].detach()

            if idx_A.numel() == 0 or idx_B.numel() == 0:
                empty = True
                self_mask = None
                r_cut_plus_margin = None
                r_cut_minus_margin = None
            else:
                empty = False
                r_cut_per_A = _coordination_r_cut_per_center(
                    types[idx_A], type_Bs, dtype=frac.dtype, r_cut=r_cut
                ).detach()
                r_cut_plus_margin = (r_cut_per_A + margin).unsqueeze(1).detach()
                r_cut_minus_margin = (r_cut_per_A - margin).unsqueeze(1).detach()
                if _coord_mod.shifts is None or _coord_mod.shifts.device != device or _coord_mod.shifts.dtype != frac.dtype:
                    _coord_mod.shifts = torch.stack(
                        torch.meshgrid(
                            torch.arange(-1, 2, device=device, dtype=frac.dtype),
                            torch.arange(-1, 2, device=device, dtype=frac.dtype),
                            torch.arange(-1, 2, device=device, dtype=frac.dtype),
                            indexing="ij",
                        ),
                        dim=-1,
                    ).reshape(-1, 3).detach()
                zero_shift = (_coord_mod.shifts == 0).all(dim=1)
                self_mask = (
                    (idx_A[:, None] == idx_B[None, :])[:, :, None]
                    & zero_shift[None, None, :]
                ).reshape(idx_A.numel(), -1).detach()
            cached = (empty, idx_A, idx_B, r_cut_plus_margin, r_cut_minus_margin, self_mask)
            _MARGIN_CACHE[key] = cached
    else:
        empty, idx_A, idx_B, r_cut_plus_margin, r_cut_minus_margin, self_mask = cached

    if empty:
        return cell.sum() * frac.sum() * torch.zeros(1, device=device)

    if _coord_mod.shifts is None or _coord_mod.shifts.device != device or _coord_mod.shifts.dtype != frac.dtype:
        with torch.no_grad():
            _coord_mod.shifts = torch.stack(
                torch.meshgrid(
                    torch.arange(-1, 2, device=device, dtype=frac.dtype),
                    torch.arange(-1, 2, device=device, dtype=frac.dtype),
                    torch.arange(-1, 2, device=device, dtype=frac.dtype),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 3).detach()

    frac_A = frac[idx_A]
    frac_B = frac_A if same_group else frac[idx_B]
    frac_B_images = (frac_B.unsqueeze(1) + _coord_mod.shifts.unsqueeze(0)).reshape(
        -1, 3
    )
    cartesian_displacements = torch.matmul(
        frac_A.unsqueeze(1) - frac_B_images.unsqueeze(0),
        cell,
    )
    distances = cartesian_displacements.norm(dim=-1)
    distances = distances.masked_fill(self_mask, torch.inf)
    ordered_distances = torch.sort(distances, dim=1).values

    if target_int >= ordered_distances.shape[1]:
        raise ValueError(
            f"Not enough B-neighbor images to define d_({target_int + 1})."
        )
    d_k_plus_1 = ordered_distances[:, target_int]
    if not torch.isfinite(d_k_plus_1).all():
        raise ValueError(
            f"Not enough B-neighbor images to define d_({target_int + 1})."
        )

    outside_distances = ordered_distances[:, target_int:]
    push_extra_outside = (
        temperature
        * torch.nn.functional.softplus(
            (
                r_cut_plus_margin
                - outside_distances
            )
            / temperature
        )
    ).sum(dim=1)
    if target_int == 0:
        return push_extra_outside

    inside_distances = ordered_distances[:, :target_int]
    pull_required_inside = (
        temperature
        * torch.nn.functional.softplus(
            (
                inside_distances
                - r_cut_minus_margin
            )
            / temperature
        )
    ).sum(dim=1)
    return pull_required_inside + push_extra_outside


def install_coordination_index_cache() -> bool:
    """Install the coordination-index cache on the coordination-loss module.

    Returns ``False`` when caching has been disabled with
    ``SCOUT_DISABLE_COORD_CACHE=1``; otherwise returns ``True`` after
    installing the cached implementation.
    """
    if os.environ.get("SCOUT_DISABLE_COORD_CACHE") == "1":
        return False
    _coord_mod._soft_neighbor_counts_per_A_single = _cached_soft_neighbor_counts_per_A_single
    _coord_mod._coordination_margin_penalties_per_A_single = _cached_coordination_margin_penalties_per_A_single
    return True


# ── Fused guidance update ──────────────────────────────────────────────────
def _fused_guidance_update(
    score_field: torch.Tensor, update: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """In-place ``score_field -= scale * update`` (avoids allocating a new score tensor).

    Returns the (possibly same, possibly new) tensor so callers can assign it
    back.  Falls back to a plain subtraction if the tensor is not safely
    in-place-able.
    """
    if score_field.is_contiguous() and update.shape == score_field.shape:
        score_field.data.add_(update, alpha=-float(scale))
        return score_field
    return score_field - float(scale) * update


def use_fast_guidance() -> bool:
    """Return whether the first-order, batched guidance path is enabled."""
    return os.environ.get("SCOUT_SLOW_GUIDANCE") != "1"
