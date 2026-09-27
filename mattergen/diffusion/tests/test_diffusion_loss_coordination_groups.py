# Licensed under the MIT License.
#
# Added for scout-matter.
# Tests for coordination objectives, grouped species, and guidance defaults.
# See UPSTREAM_CHANGES.md for the upstream baseline and change inventory.

import pytest
import torch
from pymatgen.core import Element

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.diffusion.diffusion_loss import (
    DEFAULT_COORDINATION_ALPHA,
    DEFAULT_COORDINATION_CN_TEMPERATURE,
    DEFAULT_COORDINATION_CN_TOLERANCE,
    DEFAULT_COORDINATION_MARGIN,
    DEFAULT_COORDINATION_MODE,
    DEFAULT_COORDINATION_SATISFACTION_WEIGHT,
    DEFAULT_COORDINATION_TEMPERATURE,
    INTER_ATOMIC_CUTOFF,
    LOSS_REGISTRY,
    compute_mean_coordination,
    compute_ranked_coordination,
    compute_target_share,
    mean_coordination_loss,
    ranked_coordination_loss,
    target_coordination_loss,
)


def test_coordination_defaults() -> None:
    assert DEFAULT_COORDINATION_ALPHA == 2.0
    assert DEFAULT_COORDINATION_MODE == "soft_count"
    assert DEFAULT_COORDINATION_MARGIN == 0.05
    assert DEFAULT_COORDINATION_TEMPERATURE == 0.10
    assert DEFAULT_COORDINATION_CN_TOLERANCE == 0.4
    assert DEFAULT_COORDINATION_CN_TEMPERATURE == 0.05
    assert DEFAULT_COORDINATION_SATISFACTION_WEIGHT == 1.0
    assert LOSS_REGISTRY["ranked_coordination"] is ranked_coordination_loss


def _group_coordination_system() -> ChemGraph:
    cell = torch.eye(3).unsqueeze(0) * 10.0
    frac = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # H
            [0.1, 0.0, 0.0],  # Pd, 1.0 A away
            [0.2, 0.0, 0.0],  # Ni, 2.0 A away
            [0.4, 0.0, 0.0],  # Pt, 4.0 A away
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor(
        [Element("H").Z, Element("Pd").Z, Element("Ni").Z, Element("Pt").Z],
        dtype=torch.long,
    )
    return ChemGraph(
        cell=cell,
        pos=frac,
        atomic_numbers=atomic_numbers,
        num_atoms=torch.tensor([len(atomic_numbers)]),
    )


def _grouped_center_system() -> ChemGraph:
    cell = torch.eye(3).unsqueeze(0) * 10.0
    frac = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # H
            [0.1, 0.0, 0.0],  # H
            [0.3, 0.0, 0.0],  # Pd
            [0.2, 0.0, 0.0],  # Ni
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor(
        [Element("H").Z, Element("H").Z, Element("Pd").Z, Element("Ni").Z],
        dtype=torch.long,
    )
    return ChemGraph(
        cell=cell,
        pos=frac,
        atomic_numbers=atomic_numbers,
        num_atoms=torch.tensor([len(atomic_numbers)]),
    )


def _ranked_coordination_system(
    oxygen_distances: tuple[float, ...] = (1.0, 2.0, 4.0),
    *,
    requires_grad: bool = False,
) -> ChemGraph:
    cell = torch.eye(3).unsqueeze(0) * 10.0
    frac = torch.tensor(
        [[0.0, 0.0, 0.0]] + [[distance / 10.0, 0.0, 0.0] for distance in oxygen_distances],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    atomic_numbers = torch.tensor(
        [Element("Co").Z] + [Element("O").Z] * len(oxygen_distances),
        dtype=torch.long,
    )
    return ChemGraph(
        cell=cell,
        pos=frac,
        atomic_numbers=atomic_numbers,
        num_atoms=torch.tensor([len(atomic_numbers)]),
    )


def test_ranked_coordination_sums_softplus_terms_over_all_neighbors() -> None:
    x = _ranked_coordination_system()
    margin = 0.1
    temperature = 0.2
    actual = compute_ranked_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=Element("Co").Z,
        type_B=Element("O").Z,
        target=2,
        r_cut=2.5,
        margin=margin,
        temperature=temperature,
        satisfaction_weight=0.0,
    )
    # Include all 27 periodic images, just as the implementation does. The
    # zero-shift images here give the three shortest distances: 1, 2, and 4 A.
    shifts = torch.stack(
        torch.meshgrid(
            torch.arange(-1, 2),
            torch.arange(-1, 2),
            torch.arange(-1, 2),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    oxygen_frac = x.pos[1:]
    oxygen_images = (oxygen_frac.unsqueeze(1) + shifts.unsqueeze(0)).reshape(-1, 3)
    distances = torch.sort(torch.matmul(-oxygen_images, x.cell[0]).norm(dim=-1)).values
    expected = (
        temperature
        * torch.nn.functional.softplus((distances[:2] - (2.5 - margin)) / temperature).sum()
        + temperature
        * torch.nn.functional.softplus(((2.5 + margin) - distances[2:]) / temperature).sum()
    )

    torch.testing.assert_close(actual, expected.unsqueeze(0))


def test_center_satisfaction_prefers_one_completed_environment() -> None:
    from mattergen.diffusion.coordination_loss import _mean_ranked_coordination_objective

    one_completed = _mean_ranked_coordination_objective(
        penalties=torch.tensor([0.0, 0.4]),
        counts=torch.tensor([6.0, 4.0]),
        target=6,
        softplus_temperature=0.1,
        cn_tolerance=0.4,
        cn_temperature=0.05,
        satisfaction_weight=1.0,
    )
    both_incomplete = _mean_ranked_coordination_objective(
        penalties=torch.tensor([0.2, 0.2]),
        counts=torch.tensor([5.5, 6.5]),
        target=6,
        softplus_temperature=0.1,
        cn_tolerance=0.4,
        cn_temperature=0.05,
        satisfaction_weight=1.0,
    )

    assert one_completed < both_incomplete


def test_zero_satisfaction_weight_recovers_group_softplus() -> None:
    x = _ranked_coordination_system()
    target = {
        "margin": 0.1,
        "temperature": 0.2,
        "satisfaction_weight": 0.0,
        "Co-O": [2, 2.5],
    }
    expected = compute_ranked_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=Element("Co").Z,
        type_B=Element("O").Z,
        target=2,
        r_cut=2.5,
        margin=0.1,
        temperature=0.2,
        satisfaction_weight=0.0,
    )

    torch.testing.assert_close(ranked_coordination_loss(x, t=None, target=target), expected)


def test_ranked_coordination_is_a_separate_objective() -> None:
    x = _ranked_coordination_system()
    target = {
        "margin": 0.1,
        "temperature": 0.2,
        "Co-O": [2, 2.5],
    }
    expected = compute_ranked_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=Element("Co").Z,
        type_B=Element("O").Z,
        target=2,
        r_cut=2.5,
        margin=0.1,
        temperature=0.2,
    )

    torch.testing.assert_close(
        ranked_coordination_loss(x, t=None, target=target),
        expected,
    )


@pytest.mark.parametrize("loss_fn", [mean_coordination_loss, target_coordination_loss])
def test_soft_count_objectives_reject_ranked_coordination_mode(loss_fn) -> None:
    x = _ranked_coordination_system()
    with pytest.raises(ValueError, match="ranked_coordination"):
        loss_fn(
            x,
            t=None,
            target={"coordination_mode": "kth_neighbor", "Co-O": [2, 2.5]},
        )


def test_ranked_coordination_gradients_engage_all_misclassified_neighbors() -> None:
    undercoordinated = _ranked_coordination_system(
        oxygen_distances=(2.4, 2.7, 3.0, 4.0),
        requires_grad=True,
    )
    under_loss = ranked_coordination_loss(
        undercoordinated,
        t=None,
        target={
            "margin": 0.1,
            "temperature": 0.1,
            "Co-O": [3, 2.0],
        },
    )
    under_loss.sum().backward()
    assert undercoordinated.pos.grad[1, 0] > 0.0
    assert undercoordinated.pos.grad[2, 0] > 0.0
    assert undercoordinated.pos.grad[3, 0] > 0.0

    overcoordinated = _ranked_coordination_system(
        oxygen_distances=(1.0, 1.4, 1.8, 4.0),
        requires_grad=True,
    )
    over_loss = ranked_coordination_loss(
        overcoordinated,
        t=None,
        target={
            "margin": 0.1,
            "temperature": 0.1,
            "Co-O": [1, 2.0],
        },
    )
    over_loss.sum().backward()
    assert overcoordinated.pos.grad[2, 0] < 0.0
    assert overcoordinated.pos.grad[3, 0] < 0.0
    assert overcoordinated.pos.grad[4, 0].abs() < 1e-6


def test_ranked_coordination_supports_zero_target() -> None:
    x = _ranked_coordination_system(oxygen_distances=(1.0, 1.5, 4.0), requires_grad=True)
    loss = ranked_coordination_loss(
        x,
        t=None,
        target={
            "Co-O": [0, 2.0, 0.1, 0.1],
        },
    )
    loss.sum().backward()

    assert x.pos.grad[1, 0] < 0.0
    assert x.pos.grad[2, 0] < 0.0
    assert x.pos.grad[3, 0].abs() < 1e-6


def test_ranked_coordination_requires_integer_target() -> None:
    x = _ranked_coordination_system()

    with pytest.raises(ValueError, match="non-negative integer"):
        ranked_coordination_loss(
            x,
            t=None,
            target={
                "Co-O": [1.5, 2.0],
            },
        )


def test_group_mean_coordination_uses_max_pair_cutoff() -> None:
    x = _group_coordination_system()
    type_a = Element("H").Z
    type_bs = tuple(Element(symbol).Z for symbol in ("Pd", "Ni", "Pt"))
    max_r_cut = max(
        INTER_ATOMIC_CUTOFF[type_a] + INTER_ATOMIC_CUTOFF[type_b] + 0.5 for type_b in type_bs
    )

    grouped = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_a,
        type_B=type_bs,
    )
    summed_pairs = torch.stack(
        [
            compute_mean_coordination(
                cell=x.cell,
                frac=x.pos,
                atomic_numbers=x.atomic_numbers,
                num_atoms=x.num_atoms,
                type_A=type_a,
                type_B=type_b,
                r_cut=max_r_cut,
            )
            for type_b in type_bs
        ]
    ).sum(dim=0)

    torch.testing.assert_close(grouped, summed_pairs)


def test_group_mean_coordination_loss_accepts_grouped_key_without_mutating_target() -> None:
    x = _group_coordination_system()
    type_a = Element("H").Z
    type_bs = tuple(Element(symbol).Z for symbol in ("Pd", "Ni", "Pt"))
    grouped = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_a,
        type_B=type_bs,
    )
    target = {
        "mode": "l1",
        "H-[Pd, Ni, Pt]": [float(grouped.item()), None],
    }

    loss = mean_coordination_loss(x, t=None, target=target)

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert target["mode"] == "l1"


def test_group_target_coordination_loss_accepts_grouped_key() -> None:
    x = _group_coordination_system()
    type_a = Element("H").Z
    type_bs = tuple(Element(symbol).Z for symbol in ("Pd", "Ni", "Pt"))
    expected_share = compute_target_share(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_a,
        type_B=type_bs,
        target=2.0,
        tau=0.5,
    )

    loss = target_coordination_loss(
        x,
        t=None,
        target={
            "H-[Pd, Ni, Pt]": [2.0, None, 0.5],
        },
    )

    torch.testing.assert_close(loss, 1.0 - expected_share)


def test_ranked_coordination_loss_accepts_grouped_key() -> None:
    x = _group_coordination_system()
    type_a = Element("H").Z
    type_bs = tuple(Element(symbol).Z for symbol in ("Pd", "Ni", "Pt"))
    expected = compute_ranked_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_a,
        type_B=type_bs,
        target=2,
        r_cut=2.5,
        margin=0.1,
        temperature=0.2,
    )

    actual = ranked_coordination_loss(
        x,
        t=None,
        target={"H-[Pd, Ni, Pt]": [2, 2.5, 0.1, 0.2]},
    )

    torch.testing.assert_close(actual, expected)


def test_grouped_centers_use_element_resolved_cutoffs_and_pool_all_centers() -> None:
    x = _grouped_center_system()
    type_as = tuple(Element(symbol).Z for symbol in ("H", "Pd"))
    type_b = Element("Ni").Z

    grouped = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_as,
        type_B=type_b,
    )
    mean_h, mean_pd = [
        compute_mean_coordination(
            cell=x.cell,
            frac=x.pos,
            atomic_numbers=x.atomic_numbers,
            num_atoms=x.num_atoms,
            type_A=type_a,
            type_B=type_b,
        )
        for type_a in type_as
    ]
    pooled = (2 * mean_h + mean_pd) / 3

    torch.testing.assert_close(grouped, pooled)


def test_grouped_centers_use_element_resolved_cutoffs_in_ranked_coordination() -> None:
    x = _grouped_center_system()
    type_as = tuple(Element(symbol).Z for symbol in ("H", "Pd"))
    type_b = Element("Ni").Z

    grouped = compute_ranked_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_as,
        type_B=type_b,
        target=1,
    )
    penalty_h, penalty_pd = [
        compute_ranked_coordination(
            cell=x.cell,
            frac=x.pos,
            atomic_numbers=x.atomic_numbers,
            num_atoms=x.num_atoms,
            type_A=type_a,
            type_B=type_b,
            target=1,
        )
        for type_a in type_as
    ]
    pooled = (2 * penalty_h + penalty_pd) / 3

    torch.testing.assert_close(grouped, pooled)


def test_grouped_center_loss_accepts_grouped_key() -> None:
    x = _grouped_center_system()
    type_as = tuple(Element(symbol).Z for symbol in ("H", "Pd"))
    type_b = Element("Ni").Z
    alpha = 3.0
    grouped = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_as,
        type_B=type_b,
        r_cut=2.5,
        alpha=alpha,
    )

    target = {
        "mode": "l1",
        "alpha": alpha,
        "[H,Pd]-Ni": [float(grouped.item()), 2.5],
    }
    loss = mean_coordination_loss(
        x,
        t=None,
        target=target,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert target["alpha"] == alpha


def test_grouped_center_target_coordination_loss_accepts_grouped_key() -> None:
    x = _grouped_center_system()
    type_as = tuple(Element(symbol).Z for symbol in ("H", "Pd"))
    type_b = Element("Ni").Z
    alpha = 3.0
    expected_share = compute_target_share(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=type_as,
        type_B=type_b,
        target=1.0,
        tau=0.5,
        r_cut=2.5,
        alpha=alpha,
    )

    loss = target_coordination_loss(
        x,
        t=None,
        target={
            "alpha": alpha,
            "[H,Pd]-Ni": [1.0, 2.5, 0.5],
        },
    )

    torch.testing.assert_close(loss, 1.0 - expected_share)


def test_grouped_centers_remove_self_interaction_only_for_overlapping_species() -> None:
    x = _grouped_center_system()
    type_h = Element("H").Z
    type_pd = Element("Pd").Z
    r_cut = 2.5

    grouped = compute_mean_coordination(
        cell=x.cell,
        frac=x.pos,
        atomic_numbers=x.atomic_numbers,
        num_atoms=x.num_atoms,
        type_A=(type_h, type_pd),
        type_B=type_pd,
        r_cut=r_cut,
    )
    mean_h, mean_pd = [
        compute_mean_coordination(
            cell=x.cell,
            frac=x.pos,
            atomic_numbers=x.atomic_numbers,
            num_atoms=x.num_atoms,
            type_A=type_a,
            type_B=type_pd,
            r_cut=r_cut,
        )
        for type_a in (type_h, type_pd)
    ]
    separate = (2 * mean_h + mean_pd) / 3

    torch.testing.assert_close(grouped, separate)


def test_coordination_groups_on_both_sides_are_rejected() -> None:
    x = _group_coordination_system()

    with pytest.raises(ValueError, match="supported on only one side"):
        mean_coordination_loss(
            x,
            t=None,
            target={"[H,Pd]-[Ni,Pt]": 2},
        )

    with pytest.raises(ValueError, match="supported on only one side"):
        compute_mean_coordination(
            cell=x.cell,
            frac=x.pos,
            atomic_numbers=x.atomic_numbers,
            num_atoms=x.num_atoms,
            type_A=(Element("H").Z, Element("Pd").Z),
            type_B=(Element("Ni").Z, Element("Pt").Z),
        )
