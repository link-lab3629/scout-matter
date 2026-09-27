# Licensed under the MIT License.
#
# Added for scout-matter.
# Tests for the s-to-t re-noising kernels used by self-recurrence.
# See UPSTREAM_CHANGES.md for the upstream baseline and change inventory.

import pytest
import torch

from mattergen.common.diffusion.corruption import LatticeVPSDE, NumAtomsVarianceAdjustedWrappedVESDE
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.sde_lib import VESDE, VPSDE
from mattergen.diffusion.data.batched_data import SimpleBatchedData
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor
from mattergen.diffusion.tests.test_reverse_sampling import get_diffusion_module
from mattergen.diffusion.wrapped.wrapped_sde import WrappedVESDE, WrappedVPSDE

S, T = 0.3, 0.7


def _times(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.full((n,), S), torch.full((n,), T)


def _lattice_batch(n: int) -> SimpleBatchedData:
    num_atoms = torch.arange(1, n + 1)
    return SimpleBatchedData(data={"num_atoms": num_atoms}, batch_idx={"num_atoms": None})


def _compose(sde, x0, s, t, **kw) -> tuple[torch.Tensor, torch.Tensor]:
    """Moments of x_t obtained by chaining the 0->s marginal with the s->t kernel."""
    mean_s, std_s = sde.marginal_prob(x0, s, **kw)
    mean_t, _ = sde.marginal_prob_from_s(mean_s, t, s, **kw)
    zero_mean, std_ts = sde.marginal_prob_from_s(torch.zeros_like(x0), t, s, **kw)
    one_mean, _ = sde.marginal_prob_from_s(torch.ones_like(x0), t, s, **kw)
    coeff = one_mean - zero_mean
    return mean_t, torch.sqrt((coeff * std_s) ** 2 + std_ts**2)


@pytest.mark.parametrize("sde_type", [VPSDE, VESDE])
def test_kernel_composes_to_marginal(sde_type) -> None:
    sde = sde_type()
    x0 = torch.randn(8, 3)
    s, t = _times(8)
    mean_t, std_t = sde.marginal_prob(x0, t)
    composed_mean, composed_std = _compose(sde, x0, s, t)
    torch.testing.assert_close(composed_mean, mean_t)
    torch.testing.assert_close(composed_std, std_t.expand_as(composed_std))


def test_num_atoms_scaled_positions_kernel_composes_to_marginal() -> None:
    sde = NumAtomsVarianceAdjustedWrappedVESDE(sigma_max=5.0)
    num_atoms = torch.tensor([1, 4, 8])
    batch_idx = torch.repeat_interleave(torch.arange(3), num_atoms)
    batch = SimpleBatchedData(data={"num_atoms": num_atoms}, batch_idx={"num_atoms": None})
    x0 = torch.rand(int(num_atoms.sum()), 3)
    s, t = _times(3)
    kw = dict(batch_idx=batch_idx, batch=batch)
    mean_t, std_t = sde.marginal_prob(x0, t, **kw)
    composed_mean, composed_std = _compose(sde, x0, s, t, **kw)
    torch.testing.assert_close(composed_mean, mean_t)
    torch.testing.assert_close(composed_std, std_t.expand_as(composed_std))


def test_lattice_kernel_composes_to_marginal() -> None:
    sde = LatticeVPSDE(limit_density=0.05)
    batch = _lattice_batch(4)
    x0 = torch.randn(4, 3, 3)
    x0 = 0.5 * (x0 + x0.transpose(1, 2))
    s, t = _times(4)
    mean_t, std_t = sde.marginal_prob(x0, t, batch=batch)
    composed_mean, composed_std = _compose(sde, x0, s, t, batch=batch)
    torch.testing.assert_close(composed_mean, mean_t)
    torch.testing.assert_close(composed_std, std_t)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("sde_type", [VPSDE, VESDE])
def test_kernel_runs_on_cuda(sde_type) -> None:
    sde = sde_type()
    x = torch.randn(8, 3, device="cuda")
    s, t = (v.cuda() for v in _times(8))
    assert sde.sample_from_s(x, t, s).device.type == "cuda"


@pytest.mark.parametrize("sde_type", [WrappedVESDE, WrappedVPSDE])
def test_wrapped_kernel_stays_inside_boundary(sde_type) -> None:
    sde = sde_type(wrapping_boundary=1.0)
    x = torch.rand(1000, 3)
    s, t = _times(1000)
    noisy = sde.sample_from_s(x, t, s)
    assert noisy.min() >= 0.0 and noisy.max() < 1.0


def test_lattice_kernel_noise_is_symmetric() -> None:
    sde = LatticeVPSDE(limit_density=0.05)
    batch = _lattice_batch(16)
    x = torch.randn(16, 3, 3)
    x = 0.5 * (x + x.transpose(1, 2))
    s, t = _times(16)
    noisy = sde.sample_from_s(x, t, s, batch=batch)
    torch.testing.assert_close(noisy, noisy.transpose(1, 2))


@pytest.mark.parametrize("self_rec_steps", [1, 2, 4])
@pytest.mark.parametrize("sde_type", [VPSDE, VESDE])
def test_self_recurrence_with_exact_score_preserves_data_moments(
    sde_type, self_rec_steps: int
) -> None:
    """Without guidance and with the exact score, self-recurrence must not shift the output."""
    fields = ["x", "y"]
    batch_size = 10_000
    x0_mean, x0_std = torch.tensor(-3.0), torch.tensor(4.3)
    multi_corruption = MultiCorruption(sdes={f: sde_type() for f in fields})
    sampler = PredictorCorrector(
        diffusion_module=get_diffusion_module(
            multi_corruption=multi_corruption, x0_mean=x0_mean, x0_std=x0_std
        ),
        device=torch.device("cpu"),
        predictor_partials={k: AncestralSamplingPredictor for k in fields},
        n_steps_corrector=1,
        N=1000,
        eps_t=0.001,
        self_rec_steps=self_rec_steps,
    )
    conditioning_data = SimpleBatchedData(
        data={k: torch.randn(batch_size, 1) for k in fields},
        batch_idx={k: None for k in fields},
    )
    samples, _ = sampler.sample(conditioning_data=conditioning_data)
    means = torch.stack([samples[k].mean() for k in fields])
    stds = torch.stack([samples[k].std() for k in fields])
    assert torch.isclose(means.mean(), x0_mean, atol=1e-1)
    assert torch.isclose(stds.mean(), x0_std, atol=1e-1)
