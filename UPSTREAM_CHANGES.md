# Changes from Microsoft MatterGen

scout-matter is a research fork of [Microsoft MatterGen](https://github.com/microsoft/mattergen).
The original fork point was
[`ec029d177c93709fa9a2ea4e48b872760d09c63b`](https://github.com/microsoft/mattergen/commit/ec029d177c93709fa9a2ea4e48b872760d09c63b)
(2025-04-17). Later upstream changes were incorporated.

The comparison baseline for this inventory is the latest common ancestor of
scout-matter and the upstream main branch checked on 2026-09-07:
[`842ffe735f7d06cec89d56aa23d9f001e1124b30`](https://github.com/microsoft/mattergen/commit/842ffe735f7d06cec89d56aa23d9f001e1124b30)
(2025-07-23, “Update pyproject.toml (#196)”).
The upstream tip checked was `92423660a8bd70e83679086e88f88596d484dc16`;
it is not the comparison baseline, and this fork does not claim to include all
subsequent upstream changes.

This inventory describes the tree at cleanup commit `ffef74c`, followed by the
provenance comments and this document. It describes differences, not a certification
that every inherited experimental path works. Exact implementation history and
individual authorship are recorded in Git:

```bash
git diff 842ffe735f7d06cec89d56aa23d9f001e1124b30 HEAD -- mattergen/
git log --follow -- mattergen/diffusion/sampling/pc_sampler.py
```

## Attribution policy

Existing Microsoft and third-party copyright, license, and source notices are
retained. Files containing functional scout-matter changes carry a concise
modification notice. New functional modules are identified as additions to the
fork; their presence here does not attribute their authorship to Microsoft.
“Added” means a new path relative to the baseline, not necessarily code written
without reuse of upstream conventions or helpers. No new copyright ownership is
asserted by these modification notices. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The MIT license requires retaining its copyright and permission notices; these
additional modification notices document provenance. Preserve any more specific
third-party notices when editing inherited files. Keep headers concise, update
this inventory when scope changes, and use Git for detailed change history.

## Functional and compatibility changes

The main additions implement training-free crystal guidance: volume objectives,
coordination objectives, grouped-species constraints, ranked-neighbor penalties,
forward/backward guidance, and self-recurrence. Supporting edits add transitions
between diffusion timesteps, device selection, and graph-handling guards.
This inventory records behavior-changing differences only.

| File | Change |
| --- | --- |
| [`mattergen/common/data/chemgraph.py`](mattergen/common/data/chemgraph.py) | Added gradient-enabled copies of positions and lattice cells. |
| [`mattergen/common/data/num_atoms_distribution.py`](mattergen/common/data/num_atoms_distribution.py) | Added the MP_40 atom-count distribution. |
| [`mattergen/common/diffusion/corruption.py`](mattergen/common/diffusion/corruption.py) | Added lattice and atom-count-scaled transitions between timesteps. |
| [`mattergen/common/gemnet/gemnet.py`](mattergen/common/gemnet/gemnet.py) | Use the selected generation device in graph construction. |
| [`mattergen/common/gemnet/layers/basis_utils.py`](mattergen/common/gemnet/layers/basis_utils.py) | Replace NumPy math.factorial access with Python math.factorial. |
| [`mattergen/common/gemnet/utils.py`](mattergen/common/gemnet/utils.py) | Handle empty sizes and zero repeats in repeat-block construction. |
| [`mattergen/common/utils/eval_utils.py`](mattergen/common/utils/eval_utils.py) | Use the selected generation device when loading checkpoints. |
| [`mattergen/common/utils/globals.py`](mattergen/common/utils/globals.py) | Added GPU selection by free memory and explicit GPU index. |
| [`mattergen/common/utils/ocp_graph_utils.py`](mattergen/common/utils/ocp_graph_utils.py) | Guard periodic graph construction against non-finite geometry. |
| [`mattergen/diffusion/corruption/corruption.py`](mattergen/diffusion/corruption/corruption.py) | Added interfaces for corruption between arbitrary timesteps. |
| [`mattergen/diffusion/corruption/d3pm_corruption.py`](mattergen/diffusion/corruption/d3pm_corruption.py) | Added discrete corruption and sampling between timesteps. |
| [`mattergen/diffusion/corruption/sde_lib.py`](mattergen/diffusion/corruption/sde_lib.py) | Added SDE marginals and sampling between timesteps. |
| [`mattergen/diffusion/d3pm/d3pm.py`](mattergen/diffusion/d3pm/d3pm.py) | Added discrete transition matrices and sampling from an intermediate state. |
| [`mattergen/diffusion/diffusion_module.py`](mattergen/diffusion/diffusion_module.py) | Added clean-state prediction for guidance and example graph scaffolding. |
| [`mattergen/diffusion/sampling/pc_sampler.py`](mattergen/diffusion/sampling/pc_sampler.py) | Added gradient guidance, backward correction, self-recurrence, and loss logging. |
| [`mattergen/generator.py`](mattergen/generator.py) | Added guidance configuration, sampling controls, loss logging, and GPU selection. |
| [`mattergen/scripts/generate.py`](mattergen/scripts/generate.py) | Added CLI options for guidance, recurrence, backward steps, and GPU selection. |

## New functional modules and configurations

| File | Purpose |
| --- | --- |
| [`mattergen/diffusion/coordination_loss.py`](mattergen/diffusion/coordination_loss.py) | Coordination objectives with grouped species and ranked-neighbor penalties. |
| [`mattergen/diffusion/diffusion_loss.py`](mattergen/diffusion/diffusion_loss.py) | Volume objectives, coordination re-exports, loss registry, and new_loss stub. |
| [`mattergen/diffusion/tests/test_diffusion_loss_coordination_groups.py`](mattergen/diffusion/tests/test_diffusion_loss_coordination_groups.py) | Tests for coordination objectives, grouped species, and guidance defaults. |
| [`mattergen/diffusion/tests/test_pc_sampler_guidance.py`](mattergen/diffusion/tests/test_pc_sampler_guidance.py) | Tests for guided sampling and gradient normalization. |
| [`mattergen/scripts/prepare_mp_dataset.py`](mattergen/scripts/prepare_mp_dataset.py) | Materials Project dataset preparation from summary JSONL shards. |
| [`mattergen/conf/data_module/mp_40.yaml`](mattergen/conf/data_module/mp_40.yaml) | Configuration for the MP-40 atom-count distribution. |
| [`examples/multiple_runs/kth_neighbor.yaml`](examples/multiple_runs/kth_neighbor.yaml) | Example configuration for ranked-neighbor coordination guidance. |
| [`examples/multiple_runs/mean_coordination.yaml`](examples/multiple_runs/mean_coordination.yaml) | Example configuration for mean coordination guidance. |
| [`examples/multiple_runs/target_coordination_share.yaml`](examples/multiple_runs/target_coordination_share.yaml) | Example configuration for target coordination-share guidance. |
| [`examples/multiple_runs/volume.yaml`](examples/multiple_runs/volume.yaml) | Example configuration for volume guidance. |
| [`examples/multiple_runs/volume_per_atom.yaml`](examples/multiple_runs/volume_per_atom.yaml) | Example configuration for per-atom volume guidance. |


## Validation

The cleanup reference searches and syntax checks passed. The targeted coordination
pytest suite could not run because the configured environment lacks pytest.
