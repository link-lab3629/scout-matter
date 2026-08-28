# 🧪 scout-matter

This README explains how to use **scout-matter**, our modified version of Microsoft's MatterGen diffusion model, extended with custom **guidance functions** to bias crystal generation. These include **coordination targeting** and **volume control**. This functionality is entirely **training-free**: no retraining is required when adding new guidance objectives.

---

## 📅 Quick Start

Illustrative example using mean-coordination guidance. Replace the `Co-O` pair,
target CN, and cutoff with values for the intended chemical system:

```bash
mattergen-generate "results/Li-Co-O_guided_env" \
    --pretrained-name=chemical_system \
    --batch_size=50 \
    --properties_to_condition_on="{'chemical_system':'Li-Co-O'}" \
    --record_trajectories=False \
    --diffusion_guidance_factor=2.0 \
    --guidance="{'mean_coordination': {'mode': 'huber', 'Co-O': [6, 2.6]}}" \
    --diffusion_loss_weight="[0.01, 0.01, True]" \
    --print_loss=False \
    --self_rec_steps=3 \
    --back_step=2
```

---

## 🌍 Arguments Explained

| Argument                                                             | Type                   | Description                                                                       |
| -------------------------------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------- |
| `output_path`                                                        | `str`                  | Directory to save generated structures                                            |
| `pretrained_name`                                                    | `str`                  | Name of pretrained model from HuggingFace, check mattergen.md to see all available models (e.g., `chemical_system` to fix the system)               |
| `model_path`                                                         | `str`                  | Path to a local checkpoint, used when `pretrained_name` is omitted                 |
| `batch_size`                                                         | `int`                  | Number of structures per batch                                                    |
| `num_batches`                                                        | `int`                  | Number of batches to generate                                                     |
| `properties_to_condition_on`                                         | `dict`                 | Conditioning properties when a finetuned model has been chosen, like `{'chemical_system':'Li-Co-O'}`                     |
| `diffusion_guidance_factor`                                          | `float`                | Strength of guidance correction applied to the classifier-free diffusion when a finetuned model has been chosen  (choice for guidance : `2.0`)                           |
| `guidance`                                                           | `dict`                 | Dictionary defining the training-free guidance  (see below)                                     |
| `diffusion_loss_weight`                                              | `[float, float, bool]` | `[g, k, normalize]` where:                                                        |
| └─ `g`: forward-guidance weight (`diffusion_loss_weight[0]`)          |                        |                                                                                   |
| └─ `k`: backward-guidance weight (`diffusion_loss_weight[1]`)         |                        |                                                                                   |
| └─ `normalize`: normalize each generated structure's gradient independently for each continuous field (recommended: `True`) | | |
| `print_loss`                                                         | `bool`                 | Save loss values during generation                                               |
| `self_rec_steps`                                                     | `int`                  | Number of self-recurrence steps                                                   |
| `back_step`                                                          | `int`                  | Number of backward guidance steps per backward guidance                                      |
| `algo`                                                               | `int`                 | `0` (Algo 2) = outer-loop correction; `1` (Algo 1) = inner-loop correction before forward pass; `2` (Algo 3) = inner-loop correction after forward pass |
| `record_trajectories`                                                | `bool`                 | Whether to record step-wise atomic positions                                      |
| `force_gpu`                                                          | `int`                  | Force use of specific GPU ID                                                      |

---

## 🔍 Guidance Dictionary Format

Each generation command uses one guidance objective, passed as a one-entry
dictionary through `--guidance`.

### 🔮 Coordination Guidance

A coordination-guidance configuration selects one of three objectives with its
outer guidance key:

- `mean_coordination`: match the mean sigmoid soft coordination number.
- `target_coordination_share`: maximize the sigmoid-soft-count share at the
  requested coordination number.
- `ranked_coordination`: directly enforce an integer coordination target with
  ranked-neighbor softplus boundary penalties and reward completed local
  environments.

`mean_coordination` and `target_coordination_share` aggregate the same
differentiable soft-count statistic in different ways. `ranked_coordination`
assigns neighbors to the inside or outside of a cutoff according to distance
rank. Pair and grouped-species keys provide shared constraint syntax for all
three objectives.

The chemical pairs and numerical values below illustrate configuration syntax.
Choose the species, target coordination, and cutoff for the chemical system
being generated.

#### Mean-coordination objective

```bash
--guidance="{'mean_coordination': {
  'mode': 'huber',
  'alpha': 3.0,
  'Cu-P': [4, 2.6],
  'Cu-Cu': [0, 2.9],
  'Cu-S': 1,
  'H-[Pd,Ni,Pt]': 2,
  '[Fe,Nd]-B': 6
}}"
```

This objective compares the target with the mean sigmoid-weighted coordination
of the selected central atoms. For one constraint with target `k`,

  $$
  \bar C = \frac{1}{N_A}\sum_{a=1}^{N_A} C_a,
  \qquad
  \mathcal L_{\mathrm{mean}} = \rho(\bar C-k),
  $$

where $\rho$ is selected by `mode` and can be L1, L2, or Huber loss.

#### Target-coordination-share objective

```bash
--guidance="{'target_coordination_share': {
  'Co-O': [5, 2.42, 0.5]
}}"
```

This objective guides the fraction of central atoms having the requested soft
coordination:

  $$
  H_a(k) = \exp\!\left[-\left(\frac{C_a-k}{\tau}\right)^2\right],
  \qquad
  \mathcal L_{\mathrm{share}}
  = 1-\frac{1}{N_A}\sum_{a=1}^{N_A}H_a(k).
  $$

In `[target_CN, cutoff_radius, tau]`, `tau` is dimensionless and defaults to
`0.5`.

#### Ranked-coordination softplus objective

Illustrative configuration:

```bash
--guidance="{'ranked_coordination': {
  'margin': 0.05,
  'temperature': 0.10,
  'alpha': 2.0,
  'cn_tolerance': 0.4,
  'cn_temperature': 0.05,
  'satisfaction_weight': 1.0,
  'Co-O': [5, 2.42]
}}"
```

For every central `Co` atom, the current periodic `Co-O` distances are sorted.
For target coordination `k=5`, the loss pulls every O neighbor ranked 1 through
5 toward the inside of `r_cut - margin` and pushes every neighbor ranked above
5 toward the outside of `r_cut + margin`. Each penalty is a smooth softplus
function, so the force becomes exponentially small once that neighbor satisfies
its assigned margin.

For central atom $a$, let $d_{a,(i)}$ be its $i$-th nearest periodic neighbor,
$k$ the target coordination, $r_c$ the cutoff, $m$ the margin, $T$ the
temperature, and $M$ the number of candidate neighbor images. Every selected
neighbor atom is expanded over the $3\times3\times3=27$ cells with shifts in
$\{-1,0,1\}^3$. For overlapping center and neighbor species, the neighbor list
excludes the atom's zero-shift self-image and retains its other periodic
images. The per-center loss is

$$
\mathcal L_a^{\mathrm{softplus}} =
T\sum_{i=1}^{k}
\mathrm{softplus}\!\left(\frac{d_{a,(i)}-(r_c-m)}{T}\right)
+T\sum_{i=k+1}^{M}
\mathrm{softplus}\!\left(\frac{(r_c+m)-d_{a,(i)}}{T}\right),
$$

with $\mathrm{softplus}(x)=\log(1+e^x)$. A sigmoid soft count $C_a$ is
also used to determine whether the center lies within the acceptable
coordination interval $[k-\delta_{\mathrm{CN}},k+\delta_{\mathrm{CN}}]$:

$$
s_a =
\sigma\!\left(\frac{C_a-(k-\delta_{\mathrm{CN}})}{\tau_{\mathrm{CN}}}\right)
\sigma\!\left(\frac{(k+\delta_{\mathrm{CN}})-C_a}{\tau_{\mathrm{CN}}}\right).
$$

The loss for one coordination constraint is

$$
\mathcal L_q =
\frac{1}{N_A}\sum_{a=1}^{N_A}\mathcal L_a^{\mathrm{softplus}}
+\lambda T\ln 2
\left(1-\frac{1}{N_A}\sum_{a=1}^{N_A}s_a\right),
$$

and losses from multiple coordination constraints $q$ are summed. The first
term corrects misplaced neighbors. The second rewards completed local
environments whose soft counts lie in the acceptable CN window. The scale
$T\ln 2$ is one softplus term evaluated at its boundary, so $\lambda$ is
dimensionless.

The satisfaction term supplies a local completion signal. Its gradient is
concentrated near the two boundaries of the acceptable CN window. Far outside
that window, its sigmoids saturate and its derivative with respect to $C_a$
becomes nearly zero. Because $C_a$ is itself a sum of sigmoid neighbor weights,
the gradient with respect to a distance can also become small when that
neighbor is far from the cutoff. Large coordination errors are corrected by
the ranked-softplus term, whose gradient remains strong for assigned neighbors
that violate their margins; the satisfaction term mainly distinguishes
nearly completed environments.

- `margin`: safety interval around the cutoff in angstroms; default `0.05`.
- `temperature`: softplus smoothing width in angstroms; default `0.10`. Smaller
  values approach a hinge loss.
- `alpha`: steepness of the sigmoid count used for center classification in
  inverse angstroms; default `2.0`.
- `cn_tolerance`: half-width of the acceptable coordination interval; default
  `0.4`, giving $k\pm0.4$.
- `cn_temperature`: smooth transition width at the acceptable-CN boundaries;
  default `0.05` coordination units. At either boundary, $s_a\simeq0.5$.
- `satisfaction_weight`: dimensionless $\lambda$; default `1.0`. Set it to
  `0.0` for the pure group-softplus loss.
- The resulting loss has units of angstroms. Calibrate its guidance weight
  independently from weights used for dimensionless sigmoid-count losses.
- The target coordination `k` must be a non-negative integer. For `k=0`, all
  candidate neighbors are assigned to the outside group, so every neighbor
  inside the cutoff receives an outward gradient.
- A pair-specific override can be written as
  `[target_CN, cutoff_radius, margin, temperature]`, for example
  `'Co-O': [5, 2.42, 0.05, 0.10]`.
- Sorting is differentiable with respect to the currently selected distances
  except at exact distance ties, where the ranking changes. Autograd follows
  the ordering returned for that step.

#### Sigmoid soft count

For central atom $a$, the differentiable coordination number is

$$
C_a = \sum_{b\in B,\,\mathbf n}
\sigma\!\left[\alpha\left(r_{c,a}-d_{ab\mathbf n}\right)\right]
\; - \; \mathbf 1\!\left[Z_a\in B\right],
\qquad
\sigma(x)=\frac{1}{1+e^{-x}},
$$

where $b$ runs over the selected neighbor species and $\mathbf n$ over the 27
periodic images. The indicator supplies the implementation's unit
self-interaction correction when the center species also belongs to the
neighbor set. The objectives then aggregate $C_a$ using the mean or
target-share formulas above.

- This statistic supplies $C_a$ to `mean_coordination` and
  `target_coordination_share`.
- `alpha` is the sigmoid steepness in inverse angstroms and defaults to `2.0`.

#### Pair and grouped-species constraint syntax

The following constraint syntax is shared by all three coordination objectives:

- `A-B`: `[target_CN, cutoff_radius]`
- `A-B`: `int`; in this case the cutoff is the sum of the covalent radii plus
  `0.5` angstrom.
- `A-[B,C,D]`: group environment target for `CN(A-[B,C,D])`, the total coordination of `A` by any species in the set.
  If no cutoff is supplied, the cutoff is the maximum default cutoff over all `A-B`,
  `A-C`, and `A-D` pairs.
- `[A,B,C]-D`: grouped-center target. All atoms of types `A`, `B`, and `C` are
  pooled as centers, their `D` neighbors are counted, and the mean is taken over
  all center atoms. Species with more atoms therefore have proportionally more
  weight. Without an explicit cutoff, each center element uses its own default
  pair cutoff; an explicit cutoff is shared by the whole group.
- Group constraints support a group on one side, such as `A-[B,C,D]` or
  `[A,B,C]-D`.
- Overlapping sides are valid. For example, `[A,B]-B` excludes each central
  `B` atom's self-image from its own neighbor count.
- Multiple atom-pair environments may be defined.

### 🏢 Volume Objective

```bash
--guidance="{'volume': 80.0}"
```

- Tries to enforce a specific cell volume in Å³.

### 🏢 Volume-Per-Atom Objective

```bash
--guidance="{'volume_pa': 15.0}"
```

- Tries to enforce a specific cell volume per atom in Å³.

## 🔁 Multiple Guided Runs

Use the root-level `multiple_runs.sh` to repeat guided generation. It
handles independent runs, multiple batches per run, OOM recovery, and result
aggregation.

Pass a complete, one-entry guidance dictionary to `--guidance`, using the same
dictionary format as `mattergen-generate`. Named options use the `--option value`
form.

For example, this generates 22 `Ni-Pd-H` structures per batch with
`CN([Pd,Ni]-H) = 6`:

```bash
./multiple_runs.sh \
    --batch-size 22 --runs 50 --system Ni-Pd-H \
    --guidance "{'mean_coordination': {'mode':'huber', 'alpha':3, '[Pd,Ni]-H':6}}" \
    --forward-weight 0.01 --backward-weight 0.01 \
    --normalize true --self-rec-steps 3 --back-step 2 \
    --algorithm 1 --gpu 2
```

A YAML configuration can be supplied with `--config`:

```bash
./multiple_runs.sh --config examples/multiple_runs/mean_coordination.yaml
```

Supply `--config` by itself on the command line. YAML keys use the underscore
form of the long option names. The `guidance` section contains `type` and
`parameters`; the optional `settings` mapping controls how that loss guides
sampling:

```yaml
batch_size: 22
runs: 50
system: Ni-Pd-H
guidance:
  type: mean_coordination
  parameters:
    mode: huber
    alpha: 3.0
    "[Pd,Ni]-H": 6
  settings:
    forward_weight: 0.01
    backward_weight: 0.01
    normalize: true
    self_rec_steps: 3
    back_step: 2
    algorithm: 1
gpu: 2
```

The YAML reader overlays the supplied values on the CLI defaults. The required
fields are `guidance.type` and `guidance.parameters`. The guidance execution
settings are:

| YAML setting | CLI option | Default | Meaning |
| --- | --- | --- | --- |
| `guidance.settings.forward_weight` | `--forward-weight` | `1.0` | Forward-guidance weight `g` |
| `guidance.settings.backward_weight` | `--backward-weight` | `1.0` | Backward-guidance weight `k` |
| `guidance.settings.normalize` | `--normalize` | `true` | Normalize each structure's gradient independently per continuous field |
| `guidance.settings.self_rec_steps` | `--self-rec-steps` | `3` | Self-recurrence steps |
| `guidance.settings.back_step` | `--back-step` | `2` | Backward-guidance updates per step |
| `guidance.settings.algorithm` | `--algorithm` | `0` | Placement of corrections in the sampling loop |

`diffusion_guidance_factor` is a top-level generation setting that controls
classifier-free conditioning. The selected loss guidance uses the settings in
`guidance.settings`.

Runnable YAML examples are organized by objective; grouped species remain
constraint syntax within those objectives:

- `mean_coordination`:
  [mean soft coordination with grouped species](examples/multiple_runs/mean_coordination.yaml)
- `target_coordination_share`:
  [soft target-coordination share](examples/multiple_runs/target_coordination_share.yaml)
- `ranked_coordination`:
  [ranked-neighbor softplus](examples/multiple_runs/kth_neighbor.yaml)
- `volume`: [target cell volume](examples/multiple_runs/volume.yaml)
- `volume_pa`: [target volume per atom](examples/multiple_runs/volume_per_atom.yaml)

The main non-guidance settings are:

- `batch_size`, `num_batches`, `runs`, and `system`
- `diffusion_guidance_factor`, `gpu`, and `gpu_memory_gb`
- `oom_retries`, `oom_backoff_percent`, `min_batch_size`, and `oom_wait_seconds`
- `base_dir`, `log_file`, and `dry_run`

Use `./multiple_runs.sh --help` for defaults and the corresponding CLI option
names. On an OOM failure, the script retries the same run with
`ceil(current_batch * oom_backoff_percent / 100)` structures per batch. It stops
when retries are exhausted or the next batch would be smaller than
`min_batch_size`. A non-OOM failure stops immediately.

Each run writes to `run_N/` under
`<base-dir>/results/<system>/<guidance>/<parameters>/<settings>/`. The settings
directory contains the combined `generated_crystals.extxyz` and `durations.csv`;
each run directory contains per-attempt logs. The script uses an active virtual
environment, otherwise it activates `../.venv` when available or uses
`mattergen-generate` from `PATH`.

---

## 🧩 How to Add a New Guidance Function

The loss registry and general guidance orchestration are in
`mattergen/diffusion/diffusion_loss.py`. Coordination-specific calculations are
kept separately in `mattergen/diffusion/coordination_loss.py`.

### Step 1: Define Your Custom Loss

Create a new loss function that takes predicted structures as a `ChemGraphBatch` object as input and returns a scalar or tensor loss. For example:

```python
def new_loss(x, t, target):
    """
    Example of a new loss function.
    This is just a placeholder and should be replaced with an actual implementation.
    """
    # x : ChemGraph object (ChemGraphBatch usually)
    # t : timestep
    # target : target value
    # Return : torch.tensor with the same size as the batch
    pass
```

### Step 2: Register the Loss in `LOSS_REGISTRY`

Add the name of your new loss function in the `LOSS_REGISTRY`:

```python
LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "volume": volume_loss,
    "environment": environment_loss,
    "energy": energy,
    "new_loss": new_loss,  # Placeholder for a new loss function
    # Add more loss functions as needed
}
```

This allows the key `"new_loss"` to be passed in the CLI `--guidance` argument.

### Step 3: Use Your Loss via CLI

Pass your custom objective directly from the command line:

```bash
--guidance="{'new_loss': target_value}"
```

This value will be routed into your `new_loss` implementation as the input `target` automatically at generation time.

---

## 📃 Output

- Generated crystals are saved in `output_path/`
- Input settings are saved in `input_parameters.txt`
- (Optional) If `print_loss=True`, logs of guidance loss are saved into `diffusion_loss_history.txt`


---

## ⚠️ Notes

- No model retraining is needed to add new guidance functions.
- Environment matching works best when structures are near physical.
- If CUDA memory errors occur, reduce `batch_size` or set `--gpu_memory_gb` manually.

---

## ⚙️ Example Command

These were the best parameters we found for our experiment:

```bash
mattergen-generate results/test_env \
    --pretrained-name=chemical_system \
    --properties_to_condition_on="{'chemical_system':'Cu-P'}" \
    --guidance="{'environment': {'mode': 'huber', 'Cu-P': [4, 2.5], 'Cu-Cu': [0, 2.9]}}" \
    --diffusion_guidance_factor=2.0 \
    --diffusion_loss_weight="[0.01, 0.01, True]" \
    --self_rec_steps=3 --back_step=2 --algo=1
```

---

## 🚀 Contributions

Guidance adapted by Auguste de Lambilly in collaboration with:

- Vladimir Baturin
- Jean-Claude Crivello
- Florence d'Alché-Buc
- Guillaume Lambard
- Nataliya Sokolovska 

For more info on the original project, see MatterGen: [https://github.com/microsoft/mattergen](https://github.com/microsoft/mattergen)
