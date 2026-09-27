# Licensed under the MIT License.
#
# Added for scout-matter.
# Tests for the --guidance dispatch path from the CLI to the combined loss.
# See UPSTREAM_CHANGES.md for the upstream baseline and change inventory.

import pytest
import torch

import mattergen.scripts.generate as generate_script
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.diffusion.diffusion_loss import (
    LOSS_REGISTRY,
    make_combined_loss,
    volume_loss,
    volume_pa_loss,
)


def _cubic_graph(edge: float = 10.0, num_atoms: int = 4) -> ChemGraph:
    cell = (torch.eye(3) * edge).unsqueeze(0).requires_grad_(True)
    return ChemGraph(
        cell=cell,
        pos=torch.rand(num_atoms, 3),
        atomic_numbers=torch.full((num_atoms,), 8),
        num_atoms=torch.tensor([num_atoms]),
    )


def test_volume_losses_match_cell_volume() -> None:
    x = _cubic_graph()
    t = torch.zeros(1)
    torch.testing.assert_close(volume_loss(x, t, 800.0), torch.tensor([1e-5 * 200.0]))
    torch.testing.assert_close(volume_pa_loss(x, t, 200.0), torch.tensor([50.0]))


def test_combined_loss_sums_registered_losses_and_is_differentiable() -> None:
    x = _cubic_graph()
    t = torch.zeros(1)
    loss = make_combined_loss({"volume": 800.0, "volume_pa": 200.0})(x, t)
    torch.testing.assert_close(loss, volume_loss(x, t, 800.0) + volume_pa_loss(x, t, 200.0))
    (grad,) = torch.autograd.grad(loss.sum(), x.cell)
    assert torch.all(torch.isfinite(grad)) and grad.abs().sum() > 0


def test_unknown_guidance_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_combined_loss({"not_a_loss": 1.0})


def test_cli_guidance_reaches_generator(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeGenerator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self, output_dir):
            captured["output_dir"] = output_dir

    monkeypatch.setattr(
        generate_script.MatterGenCheckpointInfo, "from_hf_hub", lambda *a, **k: "checkpoint"
    )
    monkeypatch.setattr(generate_script, "CrystalGenerator", FakeGenerator)

    generate_script.main(
        str(tmp_path),
        pretrained_name="mattergen_base",
        guidance={"volume": 800.0},
        diffusion_loss_weight=[0.01, 0.02, True],
        self_rec_steps=3,
        back_step=2,
        algo=1,
    )

    x, t = _cubic_graph(), torch.zeros(1)
    torch.testing.assert_close(captured["diffusion_loss_fn"](x, t), volume_loss(x, t, 800.0))
    assert captured["diffusion_loss_weight"] == [0.01, 0.02, True]
    assert (captured["self_rec_steps"], captured["back_step"], captured["algo"]) == (3, 2, 1)


def test_cli_rejects_malformed_guidance(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        generate_script.MatterGenCheckpointInfo, "from_hf_hub", lambda *a, **k: "checkpoint"
    )
    with pytest.raises(ValueError):
        generate_script.main(
            str(tmp_path), pretrained_name="mattergen_base", guidance={"volume": "big"}
        )


def test_registry_exposes_documented_objectives() -> None:
    documented = {
        "volume",
        "volume_pa",
        "mean_coordination",
        "target_coordination_share",
        "ranked_coordination",
    }
    assert documented <= set(LOSS_REGISTRY)
