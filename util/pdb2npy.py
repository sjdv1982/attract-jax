#!/usr/bin/env python3
"""Extract coordinates and ATTRACT atom types from reduced PDB.

Atom types are read from the occupancy field (columns 55-60),
which is where ATTRACT-reduced PDB files store them.
"""

import argparse
import numpy as np


def parse_atomtype(line: str) -> int:
    occ = line[54:60].strip()
    if occ:
        try:
            return int(round(float(occ)))
        except ValueError:
            pass

    parts = line.split()
    for token in reversed(parts):
        try:
            value = int(round(float(token)))
            if value >= 0:
                return value
        except ValueError:
            continue
    raise ValueError(f"Cannot parse atom type from line: {line.rstrip()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_pdb", help="input reduced PDB")
    parser.add_argument("output_coords", help="output coordinates .npy")
    parser.add_argument("output_atomtypes", help="output atom types .npy")
    args = parser.parse_args()

    coords = []
    atomtypes = []
    with open(args.input_pdb) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            at = parse_atomtype(line)
            coords.append((x, y, z))
            atomtypes.append(at)

    if not coords:
        raise ValueError(f"No ATOM/HETATM records found in {args.input_pdb}")

    np.save(args.output_coords, np.asarray(coords, dtype=np.float32))
    np.save(args.output_atomtypes, np.asarray(atomtypes, dtype=np.uint8))


if __name__ == "__main__":
    main()
