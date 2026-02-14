#!/usr/bin/env python3
"""Keep header and first N structures from an ATTRACT .dat file."""

import argparse
import re

STRUCT_RE = re.compile(r"^#\d+\s*$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dat")
    parser.add_argument("output_dat")
    parser.add_argument("nposes", type=int)
    args = parser.parse_args()

    if args.nposes <= 0:
        raise ValueError("nposes must be > 0")

    out = []
    structure_count = 0
    in_structures = False
    keep = True

    with open(args.input_dat) as f:
        for line in f:
            if STRUCT_RE.match(line):
                in_structures = True
                structure_count += 1
                keep = structure_count <= args.nposes
                if not keep:
                    break
            if not in_structures:
                out.append(line)
            elif keep:
                out.append(line)

    if structure_count == 0:
        raise ValueError(f"No structures found in {args.input_dat}")

    with open(args.output_dat, "w") as g:
        g.writelines(out)


if __name__ == "__main__":
    main()
