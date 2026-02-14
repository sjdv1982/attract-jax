#!/usr/bin/env python3
"""Parse ATTRACT --score stdout and save energies/gradients as .npy."""

import argparse
import re

import numpy as np

ENERGY_RE = re.compile(r"^\s*Energy:\s*([-+0-9.eE]+)\s*$")
GRAD_RE = re.compile(r"^\s*Gradients:\s*(.*?)\s*$")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_txt", help="stdout text from attract --score")
    ap.add_argument("output_npy", help="output energies .npy")
    ap.add_argument(
        "--output-grad-npy",
        help="optional output gradients .npy (shape: nposes x 6), parsed from 'Gradients:' lines",
    )
    args = ap.parse_args()

    energies = []
    gradients = []
    with open(args.input_txt) as f:
        for line in f:
            line = line.rstrip("\n")
            m = ENERGY_RE.match(line)
            if m:
                energies.append(float(m.group(1)))
                continue
            g = GRAD_RE.match(line)
            if g:
                vals = [float(tok.replace("D", "E").replace("d", "e")) for tok in FLOAT_RE.findall(g.group(1))]
                if len(vals) != 6:
                    raise ValueError(f"Expected 6 gradient values, got {len(vals)} in line: {line}")
                gradients.append(vals)

    if not energies:
        raise ValueError(f"No energies parsed from {args.input_txt}")

    np.save(args.output_npy, np.asarray(energies, dtype=np.float32))
    if args.output_grad_npy:
        if not gradients:
            raise ValueError(f"No gradients parsed from {args.input_txt}")
        if len(gradients) != len(energies):
            raise ValueError(
                f"Parsed {len(energies)} energies but {len(gradients)} gradients from {args.input_txt}"
            )
        np.save(args.output_grad_npy, np.asarray(gradients, dtype=np.float32))


if __name__ == "__main__":
    main()
