# Licensed under the MIT License.
#
# Added for scout-matter.
# Tests for guided sampling and gradient normalization.
# See UPSTREAM_CHANGES.md for the upstream baseline and change inventory.

from types import SimpleNamespace

import torch

from mattergen.diffusion.sampling.pc_sampler import _compute_guidance_grads, _prepare_guidance_grad


def test_compute_guidance_grads_returns_both_first_order_gradients():
    pos = torch.tensor([[1.0, -2.0, 3.0]], requires_grad=True)
    cell = torch.tensor([[[2.0]]], requires_grad=True)
    data = SimpleNamespace(pos=pos, cell=cell)
    loss = pos.square().sum() + 3.0 * cell.sum()

    grads = _compute_guidance_grads(loss, data)

    torch.testing.assert_close(grads["pos"], 2.0 * pos)
    torch.testing.assert_close(grads["cell"], torch.full_like(cell, 3.0))
    assert not grads["pos"].requires_grad
    assert not grads["cell"].requires_grad


def test_compute_guidance_grads_replaces_unused_gradient_with_zero():
    pos = torch.tensor([[1.0, -2.0, 3.0]], requires_grad=True)
    cell = torch.tensor([[[2.0]]], requires_grad=True)
    data = SimpleNamespace(pos=pos, cell=cell)

    grads = _compute_guidance_grads(pos.square().sum(), data)

    torch.testing.assert_close(grads["pos"], 2.0 * pos)
    torch.testing.assert_close(grads["cell"], torch.zeros_like(cell))


def test_normalize_ragged_gradient_per_sample():
    grad = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 12.0],
        ]
    )
    batch_idx = torch.tensor([0, 0, 1])

    normalized = _prepare_guidance_grad(grad, batch_idx=batch_idx, batch_size=2, normalize=True)

    expected = torch.tensor(
        [
            [0.6, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(normalized, expected)


def test_normalize_dense_gradient_per_sample_and_preserve_zero_sample():
    grad = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[0.0, 0.0], [5.0, 12.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        requires_grad=True,
    )

    normalized = _prepare_guidance_grad(grad, batch_idx=None, batch_size=3, normalize=True)

    expected = torch.tensor(
        [
            [[0.6, 0.8], [0.0, 0.0]],
            [[0.0, 0.0], [5.0 / 13.0, 12.0 / 13.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    torch.testing.assert_close(normalized, expected)

    normalized.sum().backward()
    assert torch.isfinite(grad.grad).all()


def test_normalization_is_independent_of_other_batch_members():
    first_sample = torch.tensor([[3.0, 4.0, 0.0]])
    normalized_alone = _prepare_guidance_grad(
        first_sample, batch_idx=torch.tensor([0]), batch_size=1, normalize=True
    )

    batched = torch.cat((first_sample, torch.tensor([[0.0, 0.0, 12.0]])))
    normalized_batched = _prepare_guidance_grad(
        batched, batch_idx=torch.tensor([0, 1]), batch_size=2, normalize=True
    )

    torch.testing.assert_close(normalized_batched[:1], normalized_alone)
