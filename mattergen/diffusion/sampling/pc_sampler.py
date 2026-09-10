# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from typing import Callable, Generic, Mapping, Tuple, TypeVar

import torch
from tqdm.auto import tqdm

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.diffusion.sampling.pc_partials import CorrectorPartial, PredictorPartial

Diffusable = TypeVar(
    "Diffusable", bound=BatchedData
)  # Don't use 'T' because it clashes with the 'T' for time
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
    flat = g.reshape(g.shape[0], -1)
    squared_norm = flat.square().sum(dim=1)
    if batch_idx is None:
        n = squared_norm.clamp_min(threshold**2).sqrt()
    else:
        squared_norm = torch.zeros(batch_size, dtype=g.dtype, device=g.device).scatter_add(
            0, batch_idx, squared_norm
        )
        n = squared_norm.clamp_min(threshold**2).sqrt()[batch_idx]

    n = n.view(g.shape[0], *([1] * (g.ndim - 1)))
    mask = (n > threshold).to(g.dtype)
    if normalize:
        g = g / n
    return g * mask


def _compute_guidance_grads(
    diffusion_loss: torch.Tensor,
    data: Diffusable,
    fields: tuple[str, ...] = ("pos", "cell"),
) -> dict[str, torch.Tensor]:
    """Compute all first-order guidance gradients in one autograd traversal."""
    inputs = [getattr(data, field) for field in fields]
    grads = torch.autograd.grad(
        diffusion_loss,
        inputs,
        grad_outputs=torch.ones_like(diffusion_loss),
        create_graph=False,
        allow_unused=True,
    )
    return {
        field: torch.zeros_like(input_tensor) if grad is None else grad
        for field, input_tensor, grad in zip(fields, inputs, grads)
    }


class PredictorCorrector(Generic[Diffusable]):
    """Generates samples using predictor-corrector sampling."""

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
        diffusion_loss_weight: list[float] = [
            1.0,
            1.0,
        ],  # Weight for the diffusion loss (theoretically should be 1.0)
        self_rec_steps: int = 1,
        back_step: int = 0,  # Number of steps to go back in the predictor-corrector loop
        print_loss_history: bool = False,  # Flag to control printing of loss history
        algo: int = 0,  # Algorithm type
    ):
        """Args: diffusion_module: diffusion module predictor_partials: partials for constructing
        predictors.

        Keys are the names of the corruptions. corrector_partials: partials for constructing
        correctors. Keys are the names of the corruptions. device: device to run on
        n_steps_corrector: number of corrector steps N: number of noise levels eps_t: diffusion time
        to stop denoising at max_t: diffusion time to start denoising at. If None, defaults to the
        maximum diffusion time. You may want to start at T-0.01, say, for numerical stability.
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
        self.diffusion_loss_history = []  # To keep track of diffusion loss values
        self.print_loss_history = print_loss_history  # Flag to control printing of loss history
        self.self_rec_steps = self_rec_steps
        self.back_step = back_step  # Number of steps to go back in the predictor-corrector loop

    @property
    def diffusion_module(self) -> DiffusionModule:
        return self._diffusion_module

    @property
    def _multi_corruption(self) -> MultiCorruption:
        return self._diffusion_module.corruption

    def _score_fn(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        return self._diffusion_module.score_fn(x, t)

    @classmethod
    def from_pl_module(cls, pl_module: DiffusionLightningModule, **kwargs) -> PredictorCorrector:
        return cls(diffusion_module=pl_module.diffusion_module, device=pl_module.device, **kwargs)

    @torch.no_grad()
    def sample(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ) -> SampleAndMean:
        """Create one sample for each of a batch of conditions.

        Args:     conditioning_data: batched conditioning data. Even if you think you don't want
        conditioning, you still need to pass a batch of conditions        because the sampler uses
        these to determine the shapes of things to generate.     mask: for inpainting. Keys should
        be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data
        that should be replaced with sampled values.         Shapes of values in `mask` must match
        the shapes of values in `conditioning_data`. Returns:    (batch, mean_batch). The difference
        between these is that `mean_batch` has no noise added at the final denoising step.
        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=False)[:2]

    @torch.no_grad()
    def sample_with_record(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ) -> SampleAndMeanAndRecords:
        """Create one sample for each of a batch of conditions.

        Args:     conditioning_data: batched conditioning data. Even if you think you don't want
        conditioning, you still need to pass a batch of conditions        because the sampler uses
        these to determine the shapes of things to generate.     mask: for inpainting. Keys should
        be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data
        that should be replaced with sampled values.         Shapes of values in `mask` must match
        the shapes of values in `conditioning_data`. Returns:    (batch, mean_batch). The difference
        between these is that `mean_batch` has no noise added at the final denoising step.
        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=True)

    @torch.no_grad()
    def _sample_maybe_record(
        self,
        conditioning_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Create one sample for each of a batch of conditions.

        Args:     conditioning_data: batched conditioning data. Even if you think you don't want
        conditioning, you still need to pass a batch of conditions        because the sampler uses
        these to determine the shapes of things to generate.     mask: for inpainting. Keys should
        be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data
        that should be replaced with sampled values.         Shapes of values in `mask` must match
        the shapes of values in `conditioning_data`. Returns:    (batch, mean_batch,
        recorded_samples, recorded_predictions).    The difference between the former two is that
        `mean_batch` has no noise added at the final denoising step.    The latter two are only
        returned if `record` is True, and contain the samples and predictions from each step of the
        diffusion process.
        """
        if isinstance(self._diffusion_module, torch.nn.Module):
            self._diffusion_module.eval()
        mask = mask or {}
        conditioning_data = conditioning_data.to(self._device)
        mask = {k: v.to(self._device) for k, v in mask.items()}
        batch = _sample_prior(self._multi_corruption, conditioning_data, mask=mask)
        return self._denoise(batch=batch, mask=mask, record=record)

    def save_diffusion_loss_history(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.diffusion_loss_history, f)

    def set_diffusion_loss(
        self,
        diffusion_loss_fn: Callable[[BatchedData, torch.Tensor], torch.Tensor],
        diffusion_loss_weight: list[float],
    ):
        """Set or update the diffusion loss function and its weight after the module has been
        initialized."""
        self.diffusion_loss_fn = diffusion_loss_fn
        self.diffusion_loss_weight = diffusion_loss_weight
        if len(self.diffusion_loss_weight) == 2:
            self.diffusion_loss_weight.append(True)  # If not mentionned, use the normalization

    def _backward_guidance(self, x0: Diffusable, t: torch.Tensor, score) -> Diffusable:
        """Update the score with the backward universal guidance function."""
        with torch.set_grad_enabled(True):
            diffusion_loss = self.diffusion_loss_fn(x0, t)
        if self.print_loss_history:
            self.diffusion_loss_history.append(diffusion_loss.cpu().tolist())
        grad_dict = _compute_guidance_grads(diffusion_loss, x0)
        # if grad_dict['pos'].sum() != 0 or grad_dict['cell'].sum() != 0:
        #   print(grad_dict, diffusion_loss)
        for k in grad_dict:
            if k in score:
                g_scaled = _prepare_guidance_grad(
                    grad_dict[k],
                    batch_idx=x0.get_batch_idx(k),
                    batch_size=x0.get_batch_size(),
                    normalize=self.diffusion_loss_weight[2],
                )
                alpha_t, sigma_t = x0.alpha[k]
                score[k] = (
                    score[k] - self.diffusion_loss_weight[1] * alpha_t / (sigma_t**2) * g_scaled
                )
        del grad_dict  # Clean up the gradient dictionary
        pass

    def _forward_guidance(self, batch: Diffusable, t: torch.Tensor, score) -> Diffusable:
        """Update the score with the forward universal guidance function."""
        # Compute x0|xt
        batch_ = batch._grad_copy()  # Create a shallow copy with gradients enabled
        with torch.set_grad_enabled(True):
            x0 = self.diffusion_module._predict_x0(
                x=batch_,
                atomic_numbers=self._predictors["atomic_numbers"].corruption._to_non_zero_based(
                    torch.distributions.Categorical(logits=score["atomic_numbers"]).sample()
                ),
                t=t,
            )
        with torch.set_grad_enabled(True):
            diffusion_loss = self.diffusion_loss_fn(x0, t)
        if self.print_loss_history:
            self.diffusion_loss_history.append(diffusion_loss.cpu().tolist())
        grad_dict = _compute_guidance_grads(diffusion_loss, batch_)
        # if grad_dict['pos'].sum() != 0 or grad_dict['cell'].sum() != 0:
        #        print(grad_dict, diffusion_loss)
        for k in grad_dict:
            if k in score:
                g_scaled = _prepare_guidance_grad(
                    grad_dict[k],
                    batch_idx=batch_.get_batch_idx(k),
                    batch_size=batch_.get_batch_size(),
                    normalize=self.diffusion_loss_weight[2],
                )
                score[k] = score[k] - self.diffusion_loss_weight[0] * g_scaled
        del batch_  # Clean up the temporary batch with gradients
        del grad_dict  # Clean up the gradient dictionary
        pass

    def forward_corruption(
        self,
        batch_k: Diffusable,
        t: torch.Tensor,
        s: torch.Tensor,
        k: str,
        batch_idx: torch.Tensor | None = None,
        batch: Diffusable | None = None,
    ) -> Tuple[Diffusable, torch.Tensor]:
        """Forward pass for a corruption from s to t."""
        return (
            self._multi_corruption.corruptions[k].sample_from_s(
                batch_k, t, s, batch_idx=batch_idx, batch=batch
            ),
            self._multi_corruption.corruptions[k].marginal_prob_from_s(
                batch_k, t, s, batch_idx=batch_idx, batch=batch
            )[0],
        )

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from a prior sample to a t=eps_t sample."""
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
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)

            # Corrector updates.
            if self._correctors and self.algo < 3:
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

            score = self._score_fn(batch, t)

            if self.diffusion_loss_fn is not None and (t < self._multi_corruption.T * 0.9).all():
                self._forward_guidance(batch, t, score)
                for _ in range(self.back_step):
                    # Update the score with the backward universal guidance function
                    x0 = self._diffusion_module._predict_x0(
                        x=batch,
                        atomic_numbers=self._predictors[
                            "atomic_numbers"
                        ].corruption._to_non_zero_based(
                            torch.distributions.Categorical(logits=score["atomic_numbers"]).sample()
                        ),
                        t=t,
                        score=score,
                        get_alpha=True,
                    )
                    self._backward_guidance(x0, t, score)

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

            for _ in range((self.self_rec_steps - 1) * (t < self._multi_corruption.T * 0.9).all()):
                # Compute unconditionnal score
                batch_, mean_batch_ = _mask_replace(
                    samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                )  # z_t-1

                # Algorithm 1 ############
                # Corrector updates.
                if self._correctors and self.algo == 1:
                    for _ in range(self._n_steps_corrector):
                        score = self._score_fn(batch_, t + dt)
                        fns = {
                            k: corrector.step_given_score
                            for k, corrector in self._correctors.items()
                        }
                        samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                            fns=fns,
                            broadcast={"t": t + dt, "dt": dt},
                            x=batch_,
                            score=score,
                            batch_idx=self._multi_corruption._get_batch_indices(batch_),
                        )
                        if record:
                            recorded_samples.append(batch_.clone().to("cpu"))
                        batch_, mean_batch_ = _mask_replace(
                            samples_means=samples_means,
                            batch=batch_,
                            mean_batch=mean_batch_,
                            mask=mask,
                        )
                # Algorithm 1 ############

                # Renoise the batch fieldwise
                fns = {
                    k: self.forward_corruption
                    for k in self._multi_corruption.corrupted_fields
                    if k in batch_
                }
                samples_means = apply(
                    fns=fns,
                    batch_k=batch_,
                    broadcast={"t": t, "s": t + dt, "batch": batch_},
                    k={u: u for u in self._multi_corruption.corrupted_fields if u in batch_},
                    batch_idx=self._multi_corruption._get_batch_indices(batch_),
                )
                batch = batch_.replace(**{k: v[0] for k, v in samples_means.items()})
                mean_batch = mean_batch_.replace(**{k: v[1] for k, v in samples_means.items()})

                # Algorithm 2 ############
                # Corrector updates.
                if self._correctors and self.algo == 2:
                    for _ in range(self._n_steps_corrector):
                        score = self._score_fn(batch, t)
                        fns = {
                            k: corrector.step_given_score
                            for k, corrector in self._correctors.items()
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
                            samples_means=samples_means,
                            batch=batch,
                            mean_batch=mean_batch,
                            mask=mask,
                        )

                # Algorithm 2 ############

                score = self._score_fn(batch, t)

                if (
                    self.diffusion_loss_fn is not None
                    and (t < self._multi_corruption.T * 0.9).all()
                ):
                    self._forward_guidance(batch, t, score)
                    for _ in range(self.back_step):
                        # Update the score with the backward universal guidance function
                        x0 = self._diffusion_module._predict_x0(
                            x=batch,
                            atomic_numbers=self._predictors[
                                "atomic_numbers"
                            ].corruption._to_non_zero_based(
                                torch.distributions.Categorical(
                                    logits=score["atomic_numbers"]
                                ).sample()
                            ),
                            t=t,
                            score=score,
                            get_alpha=True,
                        )
                        self._backward_guidance(x0, t, score)
                        del x0  # Clean up the temporary x0
                    torch.cuda.empty_cache()  # Clear cache to free memory
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
            )  # Update batch and mean_batch ie z_t (the previous z_{t-1}finalise)

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
