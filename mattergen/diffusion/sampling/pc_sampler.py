# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Predictor-corrector sampling with optional differentiable guidance.

The sampler supports continuous and discrete state fields, inpainting masks,
self-recurrent refinement, and guidance losses that modify the position and
unit-cell score fields during denoising.
"""

from __future__ import annotations

from typing import Generic, Mapping, Tuple, TypeVar, Callable

import torch
from tqdm.auto import tqdm
import json

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.diffusion.sampling.pc_partials import CorrectorPartial, PredictorPartial
from mattergen.diffusion.sampling._guidance_speedups import (
    _fused_guidance_update,
    cache_sde_coefficients,
    install_coordination_index_cache,
    use_fast_guidance,
)
import os

# Enable coordination-index caching by default. Set
# SCOUT_DISABLE_COORD_CACHE=1 to disable it for troubleshooting.
install_coordination_index_cache()

Diffusable = TypeVar(
    "Diffusable", bound=BatchedData
)  # Avoid reusing T, which denotes diffusion time in this module.
SampleAndMean = Tuple[Diffusable, Diffusable]
SampleAndMeanAndMaybeRecords = Tuple[Diffusable, Diffusable, list[Diffusable] | None]
SampleAndMeanAndRecords = Tuple[Diffusable, Diffusable, list[Diffusable]]


def _prepare_guidance_grad(
    g: torch.Tensor,
    *,
    batch_idx: torch.LongTensor | None,
    batch_size: int,
    normalize: bool,
    threshold: float = 1e-20,
) -> torch.Tensor:
    """Normalize gradients independently for each structure in the batch.

    ``batch_idx`` maps entries in a field such as ``pos`` to structures. For
    dense fields, ``None`` indicates that the first dimension is already the
    structure dimension. Gradients below ``threshold`` are masked without a
    host-device synchronization.
    """
    flat = g.reshape(g.shape[0], -1)
    squared_norm = flat.square().sum(dim=1)
    if batch_idx is None:
        n = squared_norm.clamp_min(threshold**2).sqrt()
    else:
        squared_norm = torch.zeros(
            batch_size, dtype=g.dtype, device=g.device
        ).scatter_add(0, batch_idx, squared_norm)
        n = squared_norm.clamp_min(threshold**2).sqrt()[batch_idx]

    n = n.view(g.shape[0], *([1] * (g.ndim - 1)))
    mask = (n > threshold).to(g.dtype)
    if normalize:
        g = g / n
    return g * mask


class PredictorCorrector(Generic[Diffusable]):
    """Generate samples with predictor-corrector diffusion and optional guidance.

    The sampler supports continuous and discrete corruptions, inpainting masks,
    self-recurrent refinement, and differentiable guidance losses on structure
    fields such as positions and unit cells.
    """

    def __init__(
        self,
        *,
        diffusion_module: DiffusionModule,
        predictor_partials: dict[str, PredictorPartial] | None = None,
        corrector_partials: dict[str, CorrectorPartial] | None = None,
        device: torch.device,
        n_steps_corrector: int,
        N: int,
        eps_t: float = 1e-3,
        max_t: float | None = None,
        diffusion_loss_fn: Callable[[Diffusable, torch.Tensor], torch.Tensor] | None = None,
        diffusion_loss_weight: list[float] = [1.0,1.0],  # Weight for the diffusion loss (theoretically should be 1.0)
        self_rec_steps: int = 1,
        back_step: int = 0,  # Number of steps to go back in the predictor-corrector loop
        print_loss_history: bool = False,  # Flag to control printing of loss history
        algo: int = 0,  # Algorithm type
    ):
        """Initialize a predictor-corrector sampler.

        Args:
            diffusion_module: Diffusion model and corruption process.
            predictor_partials: Factories for predictor updates, keyed by
                corruption name.
            corrector_partials: Factories for corrector updates, keyed by
                corruption name.
            device: Device on which sampling is performed.
            n_steps_corrector: Number of corrector updates at each timestep.
            N: Number of diffusion timesteps.
            eps_t: Final diffusion time.
            max_t: Initial diffusion time. Defaults to the corruption horizon.
            diffusion_loss_fn: Optional differentiable loss used for guidance.
            diffusion_loss_weight: Forward weight, backward weight, and an
                optional normalization flag for guidance gradients.
            self_rec_steps: Number of self-recurrent refinement passes.
            back_step: Number of backward-guidance updates per guided pass.
            print_loss_history: Whether to retain per-step guidance losses.
            algo: Refinement ordering used by the sampling loop.
        """
        self._diffusion_module = diffusion_module
        self.N = N

        if max_t is None:
            max_t = self._multi_corruption.T
        assert max_t <= self._multi_corruption.T, "Denoising cannot start from beyond T"

        self._max_t = max_t
        assert (
            corrector_partials or predictor_partials
        ), "Must specify at least one predictor or corrector"
        corrector_partials = corrector_partials or {}
        predictor_partials = predictor_partials or {}
        if self._multi_corruption.discrete_corruptions:
            # These all have property 'N' because they are D3PM type
            assert set(c.N for c in self._multi_corruption.discrete_corruptions.values()) == {N}  # type: ignore

        self._predictors = {
            k: v(corruption=self._multi_corruption.corruptions[k], score_fn=None)
            for k, v in predictor_partials.items()
        }

        self._correctors = {
            k: v(
                corruption=self._multi_corruption.corruptions[k],
                n_steps=n_steps_corrector,
                score_fn=None,
            )
            for k, v in corrector_partials.items()
        }
        self._eps_t = eps_t
        self._n_steps_corrector = n_steps_corrector
        self._device = device
        self.diffusion_loss_fn = diffusion_loss_fn  
        self.diffusion_loss_weight = diffusion_loss_weight 
        self.diffusion_loss_history = []  # Per-step guidance losses, when enabled.
        self.print_loss_history = print_loss_history
        self.self_rec_steps = self_rec_steps
        self.back_step = back_step
        self.algo = algo

    @property
    def diffusion_module(self) -> DiffusionModule:
        return self._diffusion_module

    @property
    def _multi_corruption(self) -> MultiCorruption:
        return self._diffusion_module.corruption

    def _score_fn(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        return self._diffusion_module.score_fn(x, t)

    def _loop_empty_cache(self) -> None:
        """Periodically release cached CUDA blocks between recurrence passes."""
        if not hasattr(self, "_empty_cache_counter"):
            self._empty_cache_counter = 0
        self._empty_cache_counter += 1
        if self._empty_cache_counter % 25 == 0:
            torch.cuda.empty_cache()

    @classmethod
    def from_pl_module(cls, pl_module: DiffusionLightningModule, **kwargs) -> PredictorCorrector:
        return cls(diffusion_module=pl_module.diffusion_module, device=pl_module.device, **kwargs)

    @torch.no_grad()
    def sample(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ) -> SampleAndMean:
        """Generate one sample for each conditioning structure.

        Args:
            conditioning_data: Batched structures that define the generated
                fields and their shapes.
            mask: Optional inpainting mask. Keys must identify fields in
                ``conditioning_data``; a value of one keeps the conditioning
                value and a value of zero permits denoising.

        Returns:
            A pair ``(sample, mean_sample)``. The mean sample omits noise at
            the final denoising step.
        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=False)[:2]

    @torch.no_grad()
    def sample_with_record(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ) -> SampleAndMeanAndRecords:
        """Generate samples and return the intermediate denoising trajectory.

        Args:
            conditioning_data: Batched structures that define the generated
                fields and their shapes.
            mask: Optional inpainting mask. A value of one keeps the
                conditioning value and a value of zero permits denoising.

        Returns:
            ``(sample, mean_sample, records)``, where ``records`` contains the
            recorded states produced during denoising.
        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=True)

    @torch.no_grad()
    def _sample_maybe_record(
        self,
        conditioning_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Generate samples, optionally retaining denoising states.

        This internal entry point handles device transfer, inpainting masks,
        and temporary parameter freezing for guided sampling.

        Args:
            conditioning_data: Batched structures that define the generated
                fields and their shapes.
            mask: Optional inpainting mask.
            record: Whether to retain intermediate denoising states.

        Returns:
            ``(sample, mean_sample, records)``. ``records`` is ``None`` unless
            ``record`` is true.
        """
        if isinstance(self._diffusion_module, torch.nn.Module):
            self._diffusion_module.eval()
        mask = mask or {}
        conditioning_data = conditioning_data.to(self._device)
        mask = {k: v.to(self._device) for k, v in mask.items()}
        batch = _sample_prior(self._multi_corruption, conditioning_data, mask=mask)
        # Guidance needs gradients w.r.t. pos/cell, never model parameters.
        frozen_parameters = []
        if (
            use_fast_guidance()
            and self.diffusion_loss_fn is not None
        ):
            frozen_parameters = [
                parameter
                for parameter in self.diffusion_module.parameters()
                if parameter.requires_grad
            ]
            for parameter in frozen_parameters:
                parameter.requires_grad_(False)
        try:
            return self._denoise(batch=batch, mask=mask, record=record)
        finally:
            for parameter in frozen_parameters:
                parameter.requires_grad_(True)

    def save_diffusion_loss_history(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.diffusion_loss_history, f)

    def set_diffusion_loss(
        self,
        diffusion_loss_fn: Callable[[BatchedData, torch.Tensor], torch.Tensor],
        diffusion_loss_weight: list[float],
    ):
        """Set or update the differentiable guidance loss after initialization.

        ``diffusion_loss_weight`` contains the forward weight, backward weight,
        and an optional boolean indicating whether guidance gradients should be
        normalized. When the boolean is omitted, normalization is enabled.
        """
        self.diffusion_loss_fn = diffusion_loss_fn
        self.diffusion_loss_weight = diffusion_loss_weight
        if len(self.diffusion_loss_weight) == 2:
            self.diffusion_loss_weight.append(True)  # Normalize guidance gradients.

    def _backward_guidance(self, x0: Diffusable, t: torch.Tensor, score) -> Diffusable:
        """Apply backward guidance using the configured diffusion loss."""
        if use_fast_guidance():
            return self._backward_guidance_fast(x0, t, score)
        grad_dict = {}
        replace_kwargs = ["pos", "cell"]
        with torch.set_grad_enabled(True):
            diffusion_loss = self.diffusion_loss_fn(x0, t)
            if self.print_loss_history:
                self.diffusion_loss_history.append(diffusion_loss.cpu().tolist())
            for field in replace_kwargs:
                grad = torch.autograd.grad(
                    diffusion_loss, getattr(x0, field),
                    grad_outputs=torch.ones_like(diffusion_loss),
                    create_graph=True,
                    allow_unused=True
                )[0]
                if grad is None:
                    grad = torch.zeros_like(getattr(x0, field))
                grad_dict[field] = grad
            for k in grad_dict:
                if k in score:
                    g_scaled = _prepare_guidance_grad(
                        grad_dict[k], batch_idx=x0.get_batch_idx(k),
                        batch_size=x0.get_batch_size(),
                        normalize=self.diffusion_loss_weight[2],
                    )
                    alpha_t, sigma_t = x0.alpha[k]
                    backward_weight = self.diffusion_loss_weight[1]
                    weight = backward_weight
                    update = weight * alpha_t / (sigma_t**2) * g_scaled
                    score[k] = _fused_guidance_update(score[k], update)
            del grad_dict
            pass

    def _backward_guidance_fast(
        self, x0: Diffusable, t: torch.Tensor, score
    ) -> Diffusable:
        """Apply backward guidance with one first-order batched VJP."""
        with torch.set_grad_enabled(True):
            diffusion_loss = self.diffusion_loss_fn(x0, t)
            ones_seed = getattr(self, "_ones_seed", None)
            if ones_seed is None or ones_seed.device != diffusion_loss.device or ones_seed.shape != diffusion_loss.shape:
                ones_seed = torch.ones_like(diffusion_loss)
                self._ones_seed = ones_seed
            gradients = torch.autograd.grad(
                diffusion_loss,
                (x0.pos, x0.cell),
                grad_outputs=ones_seed,
                create_graph=False,
                allow_unused=True,
            )
        if self.print_loss_history:
            self.diffusion_loss_history.append(diffusion_loss.detach().cpu().tolist())
        for field, gradient in zip(("pos", "cell"), gradients, strict=True):
            if field not in score:
                continue
            if gradient is None:
                gradient = torch.zeros_like(getattr(x0, field))
            gradient = _prepare_guidance_grad(
                gradient,
                batch_idx=x0.get_batch_idx(field),
                batch_size=x0.get_batch_size(),
                normalize=self.diffusion_loss_weight[2],
            )
            alpha_t, sigma_t = x0.alpha[field]
            backward_weight = self.diffusion_loss_weight[1]
            weight = backward_weight
            update = weight * alpha_t / sigma_t.square() * gradient
            score[field] = _fused_guidance_update(score[field], update)
        return score

    def _forward_guidance(self, batch: Diffusable, t: torch.Tensor, score) -> Diffusable:
        """Apply forward guidance using the configured diffusion loss."""
        if use_fast_guidance():
            return self._forward_guidance_fast(batch, t, score)
        # Compute x0|xt
        batch_ = batch._grad_copy()  # Create a shallow copy with gradients enabled
        with torch.set_grad_enabled(True):
            x0 = self.diffusion_module._predict_x0(
                x=batch_,
                atomic_numbers=self._predictors['atomic_numbers'].corruption._to_non_zero_based(torch.distributions.Categorical(logits=score["atomic_numbers"]).sample()),
                t=t,
            )
        grad_dict = {}
        replace_kwargs = ["pos", "cell"]
        with torch.set_grad_enabled(True):
            diffusion_loss = self.diffusion_loss_fn(x0, t)
        if self.print_loss_history:
            self.diffusion_loss_history.append(diffusion_loss.cpu().tolist())
        for field in replace_kwargs:
            grad = torch.autograd.grad(
                diffusion_loss, getattr(batch_, field),
                grad_outputs=torch.ones_like(diffusion_loss),
                create_graph=True,
                allow_unused=True
            )[0]
            if grad is None:
                grad = torch.zeros_like(getattr(x0, field))
            grad_dict[field] = grad
        for k in grad_dict:
            if k in score:
                g_scaled = _prepare_guidance_grad(
                    grad_dict[k], batch_idx=batch_.get_batch_idx(k),
                    batch_size=batch_.get_batch_size(),
                    normalize=self.diffusion_loss_weight[2],
                )
                score[k] = _fused_guidance_update(score[k], self.diffusion_loss_weight[0] * g_scaled)
        del batch_
        del grad_dict
        pass

    def _forward_guidance_fast(
        self, batch: Diffusable, t: torch.Tensor, score
    ) -> Diffusable:
        """Apply forward guidance with a first-order batched VJP.

        Sampling consumes this gradient as a value and never differentiates the
        sampler update.  Avoiding a gradient-of-gradient graph therefore keeps
        the first-order guidance unchanged while substantially reducing memory.
        """
        batch_ = batch.replace(
            pos=batch.pos.detach().requires_grad_(True),
            cell=batch.cell.detach().requires_grad_(True),
        )
        with torch.set_grad_enabled(True):
            # Preserve score behavior: _predict_x0 computes a fresh score for
            # the current recurrent state when no score is supplied.
            x0 = self.diffusion_module._predict_x0(
                x=batch_,
                atomic_numbers=self._predictors['atomic_numbers'].corruption._to_non_zero_based(torch.distributions.Categorical(logits=score["atomic_numbers"]).sample()),
                t=t,
            )
            diffusion_loss = self.diffusion_loss_fn(x0, t)
            ones_seed = getattr(self, "_ones_seed", None)
            if (
                ones_seed is None
                or ones_seed.device != diffusion_loss.device
                or ones_seed.shape != diffusion_loss.shape
            ):
                ones_seed = torch.ones_like(diffusion_loss)
                self._ones_seed = ones_seed
            gradients = torch.autograd.grad(
                diffusion_loss,
                (batch_.pos, batch_.cell),
                grad_outputs=ones_seed,
                create_graph=False,
                allow_unused=True,
            )
        if self.print_loss_history:
            self.diffusion_loss_history.append(diffusion_loss.detach().cpu().tolist())
        batch_size = batch_.num_graphs
        pos_batch_idx = batch_.batch
        for field, gradient in zip(("pos", "cell"), gradients, strict=True):
            if field not in score:
                continue
            if gradient is None:
                gradient = torch.zeros_like(getattr(batch_, field))
            gradient = _prepare_guidance_grad(
                gradient,
                batch_idx=pos_batch_idx if field == "pos" else None,
                batch_size=batch_size,
                normalize=self.diffusion_loss_weight[2],
            )
            score[field] = _fused_guidance_update(
                score[field], gradient, scale=self.diffusion_loss_weight[0]
            )
        del batch_
        return score
    
    def forward_corruption(self, batch_k: Diffusable, t: torch.Tensor, s: torch.Tensor, k: str, batch_idx: torch.Tensor | None = None) -> Tuple[Diffusable, torch.Tensor]:
        """Apply a field corruption from time ``s`` to time ``t``.

        Returns both the sampled field and its conditional mean.
        """
        return (
        self._multi_corruption.corruptions[k].sample_from_s(batch_k, t, s, batch_idx=batch_idx),
        self._multi_corruption.corruptions[k].marginal_prob_from_s(batch_k, t, s, batch_idx=batch_idx)[0]
                        )
    
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from the prior to the final time.

        SDE coefficients are memoized only for the duration of this call and
        all wrapped methods are restored before returning.
        """
        if os.environ.get("SCOUT_DISABLE_SDE_CACHE") == "1":
            return self._denoise_inner(batch, mask, record)
        with cache_sde_coefficients(self._diffusion_module):
            return self._denoise_inner(batch, mask, record)

    @torch.no_grad()
    def _denoise_inner(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Run the predictor-corrector denoising loop.

        This method is called inside the scoped setup performed by
        :meth:`_denoise`; guidance temporarily enables autograd only for the
        position and unit-cell gradients it needs.
        """
        recorded_samples = None
        if record:
            recorded_samples = []
        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = torch.tensor(-(self._max_t - self._eps_t) / (self.N - 1), dtype=torch.float32, device=self._device)

        predictor_fns = {
            k: predictor.update_given_score for k, predictor in self._predictors.items()
        }
        corrector_fns = {
            k: corrector.step_given_score for k, corrector in self._correctors.items()
        } if self._correctors else {}

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)

            
            # Corrector updates.
            if self._correctors and self.algo < 3:
                for _ in range(self._n_steps_corrector):
                    score = self._score_fn(batch, t)
                    samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                        fns=corrector_fns,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                    )
            
            score = self._score_fn(batch, t)

            if self.diffusion_loss_fn is not None and (t < self._multi_corruption.T * 0.9).all():
                    self._forward_guidance(batch, t, score)
                    for _ in range(self.back_step):
                        # Update the score with the backward universal guidance function
                        x0 = self._diffusion_module._predict_x0(
                            x=batch,
                            atomic_numbers=self._predictors['atomic_numbers'].corruption._to_non_zero_based(torch.distributions.Categorical(logits=score["atomic_numbers"]).sample()),
                            t=t,
                            score=score,
                            get_alpha=True
                        )
                        self._backward_guidance(x0, t, score)

            # Predictor updates to predict z_t-1
            samples_means = apply(
                fns=predictor_fns,
                x=batch,
                score=score,
                broadcast=dict(t=t, batch=batch, dt=dt),
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )
            if record:
                    recorded_samples.append(batch.clone().to("cpu"))
            
            for _ in range((self.self_rec_steps-1)*(t < self._multi_corruption.T * 0.9).all()):
                # Compute the unconditional score at the recurrent state.
                batch_, mean_batch_ = _mask_replace(
                    samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                )  # State at the previous diffusion step.

                ############## Algorithm 1 ############
                # Corrector updates.
                if self._correctors and self.algo == 1:
                    for _ in range(self._n_steps_corrector):
                        score = self._score_fn(batch_, t)
                        fns = {
                            k: corrector.step_given_score for k, corrector in self._correctors.items()
                        }
                        samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                            fns=fns,
                            broadcast={"t": t, "dt": dt},
                            x=batch_,
                            score=score,
                            batch_idx=self._multi_corruption._get_batch_indices(batch_),
                        )
                        if record:
                            recorded_samples.append(batch_.clone().to("cpu"))
                        batch_, mean_batch_ = _mask_replace(
                            samples_means=samples_means, batch=batch_, mean_batch=mean_batch_, mask=mask
                        )
                ############## Algorithm 1 ############
                
                # Re-noise each field before the next recurrent refinement.
                fns = {
                    k: self.forward_corruption
                    for k in self._multi_corruption.corrupted_fields
                    if k in batch_
                }
                samples_means = apply(
                    fns=fns,
                    batch_k=batch_,
                    broadcast={"t": t, "s": t + dt},
                    k = {u:u for u in self._multi_corruption.corrupted_fields if u in batch_ },
                    batch_idx=self._multi_corruption._get_batch_indices(batch_),
                )
                batch = batch_.replace(**{k: v[0] for k, v in samples_means.items()})
                mean_batch = mean_batch_.replace(**{k: v[1] for k, v in samples_means.items()})

                ############## Algorithm 2 ############
                # Corrector updates.
                if self._correctors and self.algo == 2:
                    for _ in range(self._n_steps_corrector):
                        score = self._score_fn(batch, t)
                        fns = {
                            k: corrector.step_given_score for k, corrector in self._correctors.items()
                        }
                        samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                            fns=fns,
                            broadcast={"t": t, "dt": dt},
                            x=batch,
                            score=score,
                            batch_idx=self._multi_corruption._get_batch_indices(batch),
                        )
                        if record:
                            recorded_samples.append(batch.clone().to("cpu"))
                        batch, mean_batch = _mask_replace(
                            samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                        )

                ############## Algorithm 2 ############

                score = self._score_fn(batch, t)

                if self.diffusion_loss_fn is not None and (t < self._multi_corruption.T * 0.9).all():
                    self._forward_guidance(batch, t, score)
                    for _ in range(self.back_step):
                        # Update the score with the backward universal guidance function
                        x0 = self._diffusion_module._predict_x0(
                            x=batch,
                            atomic_numbers=self._predictors['atomic_numbers'].corruption._to_non_zero_based(torch.distributions.Categorical(logits=score["atomic_numbers"]).sample()),
                            t=t,
                            score=score,
                            get_alpha=True
                        )
                        self._backward_guidance(x0, t, score)
                        del x0  # Clean up the temporary x0
                    self._loop_empty_cache()
                # Predictor updates to predict z_t-1
                predictor_fns = {
                    k: predictor.update_given_score for k, predictor in self._predictors.items()
                }
                samples_means = apply(
                    fns=predictor_fns,
                    x=batch,
                    score=score,
                    broadcast=dict(t=t, batch=batch, dt=dt),
                    batch_idx=self._multi_corruption._get_batch_indices(batch),
                )
                if record:
                    recorded_samples.append(batch.clone().to("cpu"))

            batch, mean_batch = _mask_replace(
                    samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                )  # Update the recurrent state.

        return batch, mean_batch, recorded_samples


def _mask_replace(
    samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]],
    batch: BatchedData,
    mean_batch: BatchedData,
    mask: dict[str, torch.Tensor | None],
) -> SampleAndMean:
    # Apply masks
    samples_means = apply(
        fns={k: _mask_both for k in samples_means},
        broadcast={},
        sample_and_mean=samples_means,
        mask=mask,
        old_x=batch,
    )

    # Put the updated values in `batch` and `mean_batch`
    batch = batch.replace(**{k: v[0] for k, v in samples_means.items()})
    mean_batch = mean_batch.replace(**{k: v[1] for k, v in samples_means.items()})
    return batch, mean_batch


def _mask_both(
    *, sample_and_mean: Tuple[torch.Tensor, torch.Tensor], old_x: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return tuple(_mask(old_x=old_x, new_x=x, mask=mask) for x in sample_and_mean)  # type: ignore


def _mask(*, old_x: torch.Tensor, new_x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Replace new_x with old_x where mask is 1."""
    if mask is None:
        return new_x
    else:
        return new_x.lerp(old_x, mask)


def _sample_prior(
    multi_corruption: MultiCorruption,
    conditioning_data: BatchedData,
    mask: Mapping[str, torch.Tensor] | None,
) -> BatchedData:
    samples = {
        k: multi_corruption.corruptions[k]
        .prior_sampling(
            shape=conditioning_data[k].shape,
            conditioning_data=conditioning_data,
            batch_idx=conditioning_data.get_batch_idx(field_name=k),
        )
        .to(conditioning_data[k].device)
        for k in multi_corruption.corruptions
    }
    mask = mask or {}
    for k, msk in mask.items():
        if k in multi_corruption.corrupted_fields:
            samples[k].lerp_(conditioning_data[k], msk)
    return conditioning_data.replace(**samples)
