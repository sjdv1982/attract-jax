#!/usr/bin/env python3
"""Convert 4x4 pose matrices (.npy) to ATTRACT .dat structures."""

import argparse
import re
from typing import List

import numpy as np

STRUCT_RE = re.compile(r"^#\d+\s*$")


def _norm_0_2pi(a: float) -> float:
    twopi = 2.0 * np.pi
    x = np.fmod(a, twopi)
    if x < 0.0:
        x += twopi
    return float(x)


def rotmat_to_euler_attr(rotmat: np.ndarray):
    cs = np.clip(rotmat[2, 2], -1.0, 1.0)
    ssi = float(np.arccos(cs))
    ss = float(np.sin(ssi))

    if abs(ss) < 1e-8:
        phi = 0.0
        rot = float(np.arctan2(rotmat[0, 1], rotmat[0, 0]))
        return _norm_0_2pi(phi), 0.0 if cs >= 0.0 else float(np.pi), _norm_0_2pi(rot)

    cp = np.clip(rotmat[0, 2] / ss, -1.0, 1.0)
    sp = np.clip(rotmat[1, 2] / ss, -1.0, 1.0)
    phi = float(np.arctan2(sp, cp))
    crot = -rotmat[2, 0] / ss
    srot = -rotmat[2, 1] / ss
    rot = float(np.arctan2(srot, crot))
    return _norm_0_2pi(phi), ssi, _norm_0_2pi(rot)


def load_header(template_dat: str) -> List[str]:
    if not template_dat:
        return [
            "#pivot auto\n",
            "#centered receptor: false\n",
            "#centered ligands: true\n",
        ]
    header = []
    with open(template_dat) as f:
        for line in f:
            if STRUCT_RE.match(line):
                break
            header.append(line)
    if not header:
        raise ValueError(f"No header found in template dat: {template_dat}")
    return header


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_npy", help="input .npy with shape (N, 4, 4)")
    ap.add_argument("output_dat", help="output ATTRACT .dat")
    ap.add_argument("--template-dat", help="template .dat header source (recommended: start.dat)")
    ap.add_argument("--ens-npy", help="optional .npy with ensemble indices (1-based as in ATTRACT)")
    args = ap.parse_args()

    mats = np.load(args.input_npy)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"Expected (N,4,4), got {mats.shape}")

    header = load_header(args.template_dat)

    ens = None
    if args.ens_npy:
        ens = np.load(args.ens_npy).astype(np.int32)
        if len(ens) != len(mats):
            raise ValueError(f"ens length mismatch: {len(ens)} vs {len(mats)}")

    with open(args.output_dat, "w") as f:
        for line in header:
            f.write(line)
        for i, mat in enumerate(mats, start=1):
            f.write(f"#{i}\n")
            f.write("           0           0           0           0           0           0\n")

            phi, ssi, rot = rotmat_to_euler_attr(np.asarray(mat[:3, :3].T, dtype=np.float64))
            xa, ya, za = [float(v) for v in mat[3, :3]]

            if ens is None:
                f.write(
                    f"{phi:24.16f} {ssi:24.16f} {rot:24.16f} "
                    f"{xa:24.16f} {ya:24.16f} {za:24.16f}\n"
                )
            else:
                f.write(
                    f"{int(ens[i - 1]):12d} {phi:24.16f} {ssi:24.16f} {rot:24.16f} "
                    f"{xa:24.16f} {ya:24.16f} {za:24.16f}\n"
                )


if __name__ == "__main__":
    main()

