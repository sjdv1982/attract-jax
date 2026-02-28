#!/usr/bin/env python3
"""Evaluate NB table contributions and reconstruct full score from potential-only score."""

import argparse
import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from minfor import parse_dat_two_body, parse_score_output, print_legacy_score
from minfor_nb import build_pair_arrays, compute_nb_scores, load_context, read_nb_table


def write_score_file(path, energies, gradients):
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_legacy_score(energies, gradients)
    with open(path, "w") as f:
        f.write(buf.getvalue())


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dat")
    ap.add_argument("--potential-score", required=True)
    ap.add_argument("--nb-table", required=True)
    ap.add_argument("--out-score", required=True)
    ap.add_argument("--max-poses", type=int, default=0)
    ap.add_argument("--pose-offset", type=int, default=0)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--attract-par-npz", required=True)
    ap.add_argument("--receptor-ens-list", default=None)
    ap.add_argument("--ligand-pdb", default=None)
    ap.add_argument("--epsilon", type=float, default=15.0)
    ap.add_argument("--cdie", action="store_true")
    ap.add_argument("--max-nb-cap", type=int, default=0)
    ap.add_argument("--reference-score", default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    test_dir = args.test_dir or str(Path(args.input_dat).resolve().parent)
    total_read = (args.pose_offset + args.max_poses) if args.max_poses else 0
    _header, pivots, ens, dofs, _e, centered_ligands = parse_dat_two_body(
        args.input_dat, max_poses=total_read
    )
    if args.pose_offset > 0:
        ens = ens[args.pose_offset :]
        dofs = dofs[args.pose_offset :]

    if 2 in pivots:
        lig_pivot = pivots[2]
    else:
        ligand_pdb_path = args.ligand_pdb or os.path.join(test_dir, "ligandr.pdb")
        coor = []
        with open(ligand_pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    coor.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        lig_pivot = np.mean(coor, axis=0)

    input_centered = bool(centered_ligands) if centered_ligands is not None else False
    if input_centered:
        dofs = dofs.copy()
        dofs[:, 3:6] -= lig_pivot[None, :]

    ens_list_path = args.receptor_ens_list or os.path.join(test_dir, "partner1-ensemble.list")
    ligand_pdb_path = args.ligand_pdb or os.path.join(test_dir, "ligandr.pdb")

    ctx = load_context(
        receptor_ens_list=ens_list_path,
        ligand_pdb=ligand_pdb_path,
        grid_file=args.grid,
        attract_par_npz=args.attract_par_npz,
        lig_pivot=lig_pivot,
        epsilon=args.epsilon,
        cdie=bool(args.cdie),
        max_nb_cap=args.max_nb_cap,
    )

    with open(args.potential_score) as f:
        pot_text = f.read()
    pot_e, pot_force = parse_score_output(pot_text)
    pot_g = -pot_force

    rows = read_nb_table(args.nb_table)
    pair_lig, pair_rec, pair_mask = build_pair_arrays(rows, ctx["nb_concat"], len(dofs))
    nb_e, nb_g = compute_nb_scores(ctx, ens, dofs, pair_lig, pair_rec, pair_mask)

    n = min(len(pot_e), len(nb_e), len(dofs))
    total_e = pot_e[:n] + nb_e[:n]
    total_g = pot_g[:n] + nb_g[:n]

    write_score_file(args.out_score, total_e, total_g)
    print(f"Wrote reconstructed score: {args.out_score}")

    if args.reference_score:
        with open(args.reference_score) as f:
            ref_text = f.read()
        ref_e, ref_force = parse_score_output(ref_text)
        ref_g = -ref_force
        m = min(n, len(ref_e))
        de = np.abs(total_e[:m] - ref_e[:m])
        dg = np.abs(total_g[:m] - ref_g[:m])
        print(f"reference poses={m}")
        print(f"energy max_abs={de.max():.6e} mean_abs={de.mean():.6e}")
        print(f"grad   max_abs={dg.max():.6e} mean_abs={dg.mean():.6e}")


if __name__ == "__main__":
    main()
