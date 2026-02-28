#!/usr/bin/env python3
"""Dump potential-only score + NB hit table for ATTRACT-JAX scoring."""

import argparse
import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from jax_scorer import JaxScoreOracle
from minfor import parse_dat_two_body, print_legacy_score
from minfor_nb import (
    extract_nb_rows,
    load_context,
    write_nb_table,
)


def write_score_file(path, energies, gradients):
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_legacy_score(energies, gradients)
    with open(path, "w") as f:
        f.write(buf.getvalue())


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dat")
    ap.add_argument("--score-out", required=True, help="output potential-only .score file")
    ap.add_argument("--nb-table-out", required=True, help="output NB hit table .tsv")
    ap.add_argument("--max-poses", type=int, default=0)
    ap.add_argument("--pose-offset", type=int, default=0)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--attract-par-npz", required=True)
    ap.add_argument("--receptor-ens-list", default=None)
    ap.add_argument("--ligand-pdb", default=None)
    ap.add_argument("--epsilon", type=float, default=15.0)
    ap.add_argument("--cdie", action="store_true")
    ap.add_argument("--energy-batch", type=int, default=256)
    return ap.parse_args()


def main():
    args = parse_args()

    test_dir = args.test_dir or str(Path(args.input_dat).resolve().parent)
    total_read = (args.pose_offset + args.max_poses) if args.max_poses else 0
    header, pivots, ens, dofs, _e, centered_ligands = parse_dat_two_body(
        args.input_dat, max_poses=total_read
    )
    if args.pose_offset > 0:
        ens = ens[args.pose_offset :]
        dofs = dofs[args.pose_offset :]

    if len(dofs) == 0:
        raise ValueError("No poses selected")

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
    )

    rows = extract_nb_rows(ens, dofs, ctx)
    write_nb_table(args.nb_table_out, rows)
    print(f"Wrote NB table: {args.nb_table_out} ({len(rows)} rows)")

    oracle = JaxScoreOracle(
        receptor_ens_list=ens_list_path,
        ligand_pdb=ligand_pdb_path,
        grid_file=args.grid,
        attract_par_npz=args.attract_par_npz,
        lig_pivot=lig_pivot,
        epsilon=args.epsilon,
        cdie=bool(args.cdie),
        energy_batch=args.energy_batch,
    )
    pot_e, pot_g = oracle.score_potential_batch(ens, dofs)
    write_score_file(args.score_out, pot_e, pot_g)
    print(f"Wrote potential-only score: {args.score_out}")


if __name__ == "__main__":
    main()
