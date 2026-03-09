#!/usr/bin/env python3
"""Convert a legacy two-body ATTRACT .dat file to Nx6 NumPy arrays.

The generated DOF array preserves Euler-angle DOFs exactly as they appear in
the ligand line of the .dat file. No Euler->rotvec conversion is performed.

Outputs:
- <output_prefix>.npy             : (N,6) float64 ligand DOFs (Euler + translation)
- <output_prefix>.conformers.npy  : optional (N,) int32 ligand ensemble ids, 0-based
- <output_prefix>.ens.npy         : optional (N,) int32 receptor ensemble ids, 0-based
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _parse_numeric_line(line: str) -> Optional[np.ndarray]:
    vals = np.fromstring(line, sep=" ", dtype=np.float64)
    if vals.size in (6, 7):
        return vals
    return None


def parse_dat_stream(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    dofs = []
    receptor_ens = []
    ligand_ens = []
    receptor_has_ens = False
    ligand_has_ens = False
    first = None

    with open(path, buffering=1024 * 1024) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            vals = _parse_numeric_line(line)
            if vals is None:
                continue
            if first is None:
                first = vals
                continue

            second = vals
            if first.size == 7:
                receptor_has_ens = True
                receptor_ens.append(int(round(first[0])) - 1)
                first_dofs = first[1:]
            else:
                first_dofs = first
            if not np.allclose(first_dofs, 0.0):
                raise ValueError("Receptor DOF line must contain only zeros in two-body fixed-receptor mode")

            if second.size == 7:
                ligand_has_ens = True
                ligand_ens.append(int(round(second[0])) - 1)
                dofs.append(second[1:])
            else:
                dofs.append(second)
            first = None

    if first is not None:
        raise ValueError(f"Incomplete structure at end of {path}")

    dofs_arr = np.asarray(dofs, dtype=np.float64)
    receptor_arr = np.asarray(receptor_ens, dtype=np.int32) if receptor_has_ens else None
    ligand_arr = np.asarray(ligand_ens, dtype=np.int32) if ligand_has_ens else None
    return dofs_arr, ligand_arr, receptor_arr


def parse_dat_two_body_loadtxt(
    path: str, max_poses: int = 0
) -> Tuple[
    List[str],
    Dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[bool],
    np.ndarray,
    bool,
    bool,
]:
    header: List[str] = []
    pivots: Dict[int, np.ndarray] = {}
    energies: List[float] = []
    centered_ligands: Optional[bool] = None
    current_energy: Optional[float] = None
    seen_first_struct = False
    first_fields: Optional[int] = None
    second_fields: Optional[int] = None
    target_fields: Optional[int] = None
    structure_count = 0
    pending_numeric: List[str] = []

    numeric_path = Path(path).with_suffix(Path(path).suffix + ".numeric.tmp")
    try:
        with open(path, "r", buffering=1024 * 1024) as src, open(
            numeric_path, "w", buffering=1024 * 1024
        ) as dst:
            for raw in src:
                line = raw.rstrip("\n")
                if line.startswith("#"):
                    if line.startswith("#pivot "):
                        parts = line.split()
                        if len(parts) == 5 and parts[1].isdigit():
                            pivots[int(parts[1])] = np.array(
                                [float(parts[2]), float(parts[3]), float(parts[4])],
                                dtype=np.float64,
                            )
                    if line.startswith("## Energy:"):
                        current_energy = float(line.split(":", 1)[1].strip())
                        continue
                    tag = line[1:].strip()
                    if tag.isdigit():
                        if seen_first_struct:
                            energies.append(
                                float("nan") if current_energy is None else float(current_energy)
                            )
                            structure_count += 1
                            if max_poses and structure_count >= max_poses:
                                break
                        seen_first_struct = True
                        current_energy = None
                        continue
                    if not seen_first_struct:
                        header.append(raw)
                        low = line.strip().lower()
                        if low.startswith("#centered ligands:"):
                            if "true" in low:
                                centered_ligands = True
                            elif "false" in low:
                                centered_ligands = False
                    continue

                if not seen_first_struct:
                    header.append(raw)
                    continue
                if not line.strip():
                    continue
                if first_fields is None:
                    first_fields = len(np.fromstring(line, sep=" ", dtype=np.float64))
                    if first_fields not in (6, 7):
                        raise ValueError(
                            f"{path}: unexpected receptor numeric field count {first_fields}"
                        )
                    pending_numeric.append(line)
                    continue
                if second_fields is None:
                    second_fields = len(np.fromstring(line, sep=" ", dtype=np.float64))
                    if second_fields not in (6, 7):
                        raise ValueError(
                            f"{path}: unexpected ligand numeric field count {second_fields}"
                        )
                    target_fields = max(first_fields, second_fields)
                    pending_numeric.append(line)
                    for idx, pending_line in enumerate(pending_numeric):
                        expected = first_fields if idx == 0 else second_fields
                        if expected < target_fields:
                            dst.write("0 " + pending_line + "\n")
                        else:
                            dst.write(pending_line + "\n")
                    pending_numeric = []
                    continue
            if seen_first_struct and not (max_poses and structure_count >= max_poses):
                energies.append(float("nan") if current_energy is None else float(current_energy))
                structure_count += 1

        if first_fields is None or second_fields is None:
            raise ValueError(f"No numeric DOF lines found in {path}")

        with open(path, "r", buffering=1024 * 1024) as src, open(
            numeric_path, "w", buffering=1024 * 1024
        ) as dst:
            seen_first_struct = False
            structures_written = 0
            pair_idx = 0
            for raw in src:
                line = raw.rstrip("\n")
                if line.startswith("#"):
                    tag = line[1:].strip()
                    if tag.isdigit():
                        if seen_first_struct:
                            structures_written += 1
                            if max_poses and structures_written >= max_poses:
                                break
                        seen_first_struct = True
                    continue
                if not seen_first_struct or not line.strip():
                    continue
                expected = first_fields if (pair_idx % 2 == 0) else second_fields
                if expected < target_fields:
                    dst.write("0 " + line + "\n")
                else:
                    dst.write(line + "\n")
                pair_idx += 1

        arr = np.loadtxt(numeric_path, dtype=np.float64)
    finally:
        numeric_path.unlink(missing_ok=True)

    if arr.ndim != 2 or arr.shape[0] % 2:
        raise ValueError(f"Unexpected numeric block shape from {path}: {arr.shape}")
    rec = arr[0::2]
    lig = arr[1::2]
    if len(rec) != structure_count or len(lig) != structure_count:
        raise ValueError(
            f"{path}: parsed {len(rec)} receptor / {len(lig)} ligand rows for {structure_count} structures"
        )

    receptor_has_ens = first_fields == 7
    ligand_has_ens = second_fields == 7
    receptor_ens = np.rint(rec[:, 0]).astype(np.int32) if receptor_has_ens else np.ones(len(rec), dtype=np.int32)
    rec_dofs = rec[:, 1:] if target_fields == 7 else rec
    if not np.allclose(rec_dofs, 0.0):
        raise ValueError("Receptor DOF line must contain only zeros in two-body fixed-receptor mode")
    ligand_ens = np.rint(lig[:, 0]).astype(np.int32) if ligand_has_ens else np.zeros(len(lig), dtype=np.int32)
    dofs = lig[:, 1:] if target_fields == 7 else lig
    return (
        header,
        pivots,
        receptor_ens.astype(np.int32),
        np.asarray(dofs, dtype=np.float64),
        np.asarray(energies, dtype=np.float64),
        centered_ligands,
        ligand_ens.astype(np.int32),
        receptor_has_ens,
        ligand_has_ens,
    )


def parse_dat_loadtxt(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    (
        _header,
        _pivots,
        receptor_ens,
        dofs,
        _energies,
        _centered_ligands,
        ligand_ens,
        receptor_has_ens,
        ligand_has_ens,
    ) = parse_dat_two_body_loadtxt(path)
    return (
        dofs,
        ligand_ens.astype(np.int32) - 1 if ligand_has_ens else None,
        receptor_ens.astype(np.int32) - 1 if receptor_has_ens else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dat", help="legacy ATTRACT .dat file")
    ap.add_argument("output_prefix", help="output prefix for .npy files")
    ap.add_argument(
        "--parser",
        choices=("stream", "loadtxt"),
        default="stream",
        help="parser implementation to use",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.parser == "stream":
        dofs, ligand_ens, receptor_ens = parse_dat_stream(args.input_dat)
    else:
        dofs, ligand_ens, receptor_ens = parse_dat_loadtxt(args.input_dat)
    dt = time.perf_counter() - t0

    np.save(args.output_prefix + ".npy", dofs.astype(np.float64))
    wrote = [args.output_prefix + ".npy"]
    if ligand_ens is not None:
        np.save(args.output_prefix + ".conformers.npy", ligand_ens.astype(np.int32))
        wrote.append(args.output_prefix + ".conformers.npy")
    if receptor_ens is not None:
        np.save(args.output_prefix + ".ens.npy", receptor_ens.astype(np.int32))
        wrote.append(args.output_prefix + ".ens.npy")

    print(f"parser={args.parser}")
    print(f"poses={len(dofs)}")
    print(f"elapsed_s={dt:.6f}")
    for path in wrote:
        print(path)


if __name__ == "__main__":
    main()
