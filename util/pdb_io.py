"""Reduced (ATTRACT-format) PDB parsing."""

import numpy as np


def parse_reduced_pdb(path: str):
    coor = []
    atomtype = []
    charge = []
    weight = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            at = int(float(line[54:59]))
            q = float(line[59:67])
            w = float(line[69:74])
            coor.append((x, y, z))
            atomtype.append(at)
            charge.append(q)
            weight.append(w)
    if not coor:
        raise ValueError(f"No ATOM records parsed from {path}")
    return (
        np.asarray(coor, dtype=np.float64),
        np.asarray(atomtype, dtype=np.int32),
        np.asarray(charge, dtype=np.float64),
        np.asarray(weight, dtype=np.float64),
    )
