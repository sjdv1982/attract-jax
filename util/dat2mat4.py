#!/usr/bin/env python3
"""Convert ATTRACT .dat structures to 4x4 pose matrices (.npy).

The output matrix matches ATTRACT-JAX conventions:
- rotation in mat[:3, :3]
- translation in mat[3, :3]
- mat[3, 3] = 1

By default, the second body line (ligand in 2-body docking) is used.
"""

import argparse
import re
from typing import List, Optional, Tuple

import numpy as np

STRUCT_RE = re.compile(r"^#\d+\s*$")
ENERGY_RE = re.compile(r"^##\s*Energy:\s*([-+0-9.eE]+)")


def euler2rotmat(phi: float, ssi: float, rot: float) -> np.ndarray:
    cs = np.cos(ssi)
    cp = np.cos(phi)
    ss = np.sin(ssi)
    sp = np.sin(phi)
    cscp = cs * cp
    cssp = cs * sp
    sscp = ss * cp
    sssp = ss * sp
    crot = np.cos(rot)
    srot = np.sin(rot)

    rotmat = np.empty((3, 3), dtype=np.float64)
    rotmat[0, 0] = crot * cscp + srot * sp
    rotmat[0, 1] = srot * cscp - crot * sp
    rotmat[0, 2] = sscp

    rotmat[1, 0] = crot * cssp - srot * cp
    rotmat[1, 1] = srot * cssp + crot * cp
    rotmat[1, 2] = sssp

    rotmat[2, 0] = -crot * ss
    rotmat[2, 1] = -srot * ss
    rotmat[2, 2] = cs
    return rotmat


def parse_dof_line(line: str) -> Optional[Tuple[Optional[int], List[float]]]:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None

    if len(vals) == 6:
        return None, vals
    if len(vals) == 7:
        ens = int(round(vals[0]))
        return ens, vals[1:]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dat", help="input ATTRACT .dat")
    parser.add_argument("output_npy", help="output .npy file with shape (N, 4, 4)")
    parser.add_argument(
        "--body-index",
        type=int,
        default=0,
        help=(
            "1-based body line to extract per structure. "
            "Default 0 means 'last numeric DOF line' (recommended)."
        ),
    )
    parser.add_argument(
        "--out-energies",
        help="optional output .npy for parsed energies from '## Energy:' lines",
    )
    parser.add_argument(
        "--out-ens",
        help="optional output .npy for ensemble index on selected body line (0 if absent)",
    )
    args = parser.parse_args()

    if args.body_index < 0:
        raise ValueError("--body-index must be >= 0")

    poses = []
    energies = []
    ens_values = []

    current_dof_lines: List[Tuple[Optional[int], List[float]]] = []
    current_energy = None

    def flush_structure():
        if not current_dof_lines:
            return
        idx = len(current_dof_lines) - 1 if args.body_index == 0 else args.body_index - 1
        if idx < 0 or idx >= len(current_dof_lines):
            if args.body_index == 0:
                raise ValueError("Internal error while selecting last DOF line")
            raise ValueError(
                f"Structure has only {len(current_dof_lines)} body lines, "
                f"cannot extract body {args.body_index}"
            )
        ens, dof = current_dof_lines[idx]
        phi, ssi, rot, xa, ya, za = dof

        mat = np.zeros((4, 4), dtype=np.float64)
        # ATTRACT-JAX stores matrices for row-vector multiplication,
        # so this is the transpose of the Fortran rotmat layout.
        mat[:3, :3] = euler2rotmat(phi, ssi, rot).T
        mat[3, :3] = (xa, ya, za)
        mat[3, 3] = 1.0
        poses.append(mat)
        ens_values.append(0 if ens is None else ens)
        if current_energy is not None:
            energies.append(current_energy)

    with open(args.input_dat) as f:
        for raw in f:
            line = raw.rstrip("\n")

            if STRUCT_RE.match(line):
                flush_structure()
                current_dof_lines = []
                current_energy = None
                continue

            m = ENERGY_RE.match(line)
            if m:
                current_energy = float(m.group(1))
                continue

            parsed = parse_dof_line(line)
            if parsed is not None:
                current_dof_lines.append(parsed)

    flush_structure()

    if not poses:
        raise ValueError(f"No structures parsed from {args.input_dat}")

    pose_arr = np.asarray(poses, dtype=np.float32)
    np.save(args.output_npy, pose_arr)

    if args.out_energies:
        if len(energies) != len(poses):
            raise ValueError(
                f"Requested energies, but parsed {len(energies)} energies for {len(poses)} structures"
            )
        np.save(args.out_energies, np.asarray(energies, dtype=np.float32))

    if args.out_ens:
        np.save(args.out_ens, np.asarray(ens_values, dtype=np.int32))


if __name__ == "__main__":
    main()
