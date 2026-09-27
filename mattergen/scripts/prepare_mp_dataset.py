# Licensed under the MIT License.
#
# Added for scout-matter.
# Materials Project dataset preparation from summary JSONL shards.
# See UPSTREAM_CHANGES.md for the upstream baseline and change inventory.

"""Prepare a MatterGen CSV dataset from Materials Project summary JSONL shards."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
    "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn "
    "Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W "
    "Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf "
    "Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ATOMIC_NUMBER = {symbol: z for z, symbol in enumerate(ELEMENTS, start=1)}
MP20_EXCLUDED_ELEMENTS = ("He", "Ne", "Ar", "Kr", "Xe", "Tc", "Pm")

CSV_FIELDS = [
    "material_id",
    "formation_energy_per_atom",
    "dft_band_gap",
    "pretty_formula",
    "e_above_hull",
    "energy_above_hull",
    "elements",
    "cif",
    "spacegroup_number",
    "space_group",
    "dft_bulk_modulus",
    "dft_shear_modulus",
    "dft_poisson_ratio",
    "dft_mag_density",
]


def iter_documents(shards: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for shard in shards:
        with gzip.open(shard, mode="rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {shard}:{line_number}") from exc


def rejection_reason(
    document: dict[str, Any],
    max_atoms: int,
    max_ehull: float,
    max_atomic_number: int,
    excluded_elements: frozenset[str],
) -> str | None:
    if document.get("deprecated"):
        return "deprecated"

    nsites = document.get("nsites")
    if not isinstance(nsites, (int, float)) or not math.isfinite(nsites):
        return "missing_nsites"
    if nsites < 1 or nsites > max_atoms or int(nsites) != nsites:
        return "atom_count"

    energy_above_hull = document.get("energy_above_hull")
    if not isinstance(energy_above_hull, (int, float)) or not math.isfinite(energy_above_hull):
        return "missing_energy_above_hull"
    if energy_above_hull >= max_ehull:
        return "energy_above_hull"

    elements = document.get("elements")
    if not isinstance(elements, list) or not elements:
        return "missing_elements"
    try:
        atomic_numbers = [ATOMIC_NUMBER[element] for element in elements]
    except KeyError:
        return "unknown_element"
    if any(z >= max_atomic_number for z in atomic_numbers):
        return "atomic_number"
    if excluded_elements.intersection(elements):
        return "excluded_element"

    structure = document.get("structure")
    if not isinstance(structure, dict):
        return "missing_structure"
    sites = structure.get("sites")
    if not isinstance(sites, list) or len(sites) != int(nsites):
        return "invalid_structure"
    for site in sites:
        species = site.get("species")
        if (
            not isinstance(species, list)
            or len(species) != 1
            or species[0].get("element") not in ATOMIC_NUMBER
            or not math.isclose(float(species[0].get("occu", 0.0)), 1.0)
        ):
            return "disordered_structure"
        abc = site.get("abc")
        if not isinstance(abc, list) or len(abc) != 3:
            return "invalid_structure"
    return None


def scalar_vrh(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("vrh")
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def cif_quote(value: Any) -> str:
    text = str(value)
    if "'" not in text:
        return f"'{text}'"
    if '"' not in text:
        return f'"{text}"'
    return text.replace(" ", "_")


def structure_to_cif(document: dict[str, Any]) -> str:
    structure = document["structure"]
    lattice = structure["lattice"]
    formula = document["formula_pretty"]
    composition = document["composition"]
    material_id = document["material_id"]

    formula_sum = " ".join(f"{element}{amount:g}" for element, amount in composition.items())
    lines = [
        "# generated from the Materials Project summary structure",
        f"data_{material_id}",
        "_symmetry_space_group_name_H-M   'P 1'",
        f"_cell_length_a   {float(lattice['a']):.10f}",
        f"_cell_length_b   {float(lattice['b']):.10f}",
        f"_cell_length_c   {float(lattice['c']):.10f}",
        f"_cell_angle_alpha   {float(lattice['alpha']):.10f}",
        f"_cell_angle_beta   {float(lattice['beta']):.10f}",
        f"_cell_angle_gamma   {float(lattice['gamma']):.10f}",
        "_symmetry_Int_Tables_number   1",
        f"_chemical_formula_structural   {cif_quote(formula)}",
        f"_chemical_formula_sum   {cif_quote(formula_sum)}",
        f"_cell_volume   {float(lattice['volume']):.10f}",
        "loop_",
        " _symmetry_equiv_pos_site_id",
        " _symmetry_equiv_pos_as_xyz",
        "  1  'x, y, z'",
        "loop_",
        " _atom_site_type_symbol",
        " _atom_site_label",
        " _atom_site_symmetry_multiplicity",
        " _atom_site_fract_x",
        " _atom_site_fract_y",
        " _atom_site_fract_z",
        " _atom_site_occupancy",
    ]
    for index, site in enumerate(structure["sites"]):
        species = site["species"][0]
        symbol = species["element"]
        x, y, z = (float(value) for value in site["abc"])
        lines.append(
            f"  {symbol}  {symbol}{index}  1  "
            f"{x:.10f}  {y:.10f}  {z:.10f}  {float(species['occu']):g}"
        )
    return "\n".join(lines) + "\n"


def document_to_row(document: dict[str, Any]) -> dict[str, Any]:
    symmetry = document.get("symmetry") or {}
    energy_above_hull = document["energy_above_hull"]
    return {
        "material_id": document["material_id"],
        "formation_energy_per_atom": document.get("formation_energy_per_atom"),
        "dft_band_gap": document.get("band_gap"),
        "pretty_formula": document.get("formula_pretty"),
        "e_above_hull": energy_above_hull,
        "energy_above_hull": energy_above_hull,
        "elements": repr(document["elements"]),
        "cif": structure_to_cif(document),
        "spacegroup_number": symmetry.get("number"),
        "space_group": symmetry.get("symbol"),
        "dft_bulk_modulus": scalar_vrh(document.get("bulk_modulus")),
        "dft_shear_modulus": scalar_vrh(document.get("shear_modulus")),
        "dft_poisson_ratio": document.get("homogeneous_poisson"),
        "dft_mag_density": document.get("total_magnetization_normalized_vol"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_dir", type=Path, help="Directory containing *.jsonl.gz shards")
    parser.add_argument("output_dir", type=Path, help="Destination for train/val/test CSV files")
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--max-ehull", type=float, default=0.15)
    parser.add_argument(
        "--max-atomic-number",
        type=int,
        default=84,
        help="Exclusive upper bound, so 84 means Z < 84",
    )
    parser.add_argument(
        "--exclude-elements",
        nargs="*",
        default=MP20_EXCLUDED_ELEMENTS,
        help="Element symbols to exclude (default: MP-20 exclusions)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards = sorted(args.summary_dir.glob("*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(f"No *.jsonl.gz files found in {args.summary_dir}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    counts: Counter[str] = Counter()
    selected_ids: list[str] = []
    excluded_elements = frozenset(args.exclude_elements)
    unknown_exclusions = excluded_elements.difference(ATOMIC_NUMBER)
    if unknown_exclusions:
        raise ValueError(f"Unknown excluded element symbols: {sorted(unknown_exclusions)}")
    for document in iter_documents(shards):
        counts["total"] += 1
        reason = rejection_reason(
            document,
            args.max_atoms,
            args.max_ehull,
            args.max_atomic_number,
            excluded_elements,
        )
        if reason is None:
            material_id = document.get("material_id")
            if not isinstance(material_id, str):
                counts["rejected_missing_material_id"] += 1
                continue
            selected_ids.append(material_id)
            counts["selected"] += 1
        else:
            counts[f"rejected_{reason}"] += 1

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Selected material_id values are not unique")

    random.Random(args.seed).shuffle(selected_ids)
    n_total = len(selected_ids)
    n_train = int(0.60 * n_total)
    n_val = int(0.20 * n_total)
    split_by_id = {
        **{material_id: "train" for material_id in selected_ids[:n_train]},
        **{material_id: "val" for material_id in selected_ids[n_train : n_train + n_val]},
        **{material_id: "test" for material_id in selected_ids[n_train + n_val :]},
    }

    handles = {}
    writers = {}
    split_counts: Counter[str] = Counter()
    atom_counts: dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    train_density_sum = 0.0
    try:
        for split in ("train", "val", "test"):
            handle = (args.output_dir / f"{split}.csv").open(mode="w", encoding="utf-8", newline="")
            handles[split] = handle
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writers[split] = writer

        for document in iter_documents(shards):
            split = split_by_id.get(document.get("material_id"))
            if split is None:
                continue
            writers[split].writerow(document_to_row(document))
            nsites = int(document["nsites"])
            split_counts[split] += 1
            atom_counts[split][nsites] += 1
            if split == "train":
                train_density_sum += nsites / float(document["volume"])
    finally:
        for handle in handles.values():
            handle.close()

    expected_counts = {
        "train": n_train,
        "val": n_val,
        "test": n_total - n_train - n_val,
    }
    if dict(split_counts) != expected_counts:
        raise RuntimeError(f"Written split counts {dict(split_counts)} != {expected_counts}")

    metadata = {
        "source": str(args.summary_dir.resolve()),
        "source_shards": [path.name for path in shards],
        "filters": {
            "deprecated": False,
            "max_atoms_inclusive": args.max_atoms,
            "max_energy_above_hull_exclusive_eV_per_atom": args.max_ehull,
            "max_atomic_number_exclusive": args.max_atomic_number,
            "excluded_elements": sorted(excluded_elements, key=ATOMIC_NUMBER.get),
            "ordered_structures_only": True,
        },
        "split": {"seed": args.seed, "ratios": [0.6, 0.2, 0.2], "counts": expected_counts},
        "selection_counts": dict(sorted(counts.items())),
        "num_atoms_distribution": {
            split: {str(n): count for n, count in sorted(distribution.items())}
            for split, distribution in atom_counts.items()
        },
        "train_average_density_atoms_per_A3": train_density_sum / n_train,
        "csv_columns": CSV_FIELDS,
    }
    with (args.output_dir / "metadata.json").open(mode="w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
