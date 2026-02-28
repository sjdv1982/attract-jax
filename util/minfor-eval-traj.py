#!/usr/bin/env python3
"""Evaluate dump-traj files and reconstruct full per-step scores."""

import argparse
import glob
import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from minfor import parse_dat_two_body, parse_score_output, print_legacy_score
from minfor_nb import build_pair_arrays, compute_nb_scores, load_context, read_nb_table


def _fmt(pattern, step, kind):
    return pattern.format(step=step, kind=kind)


def _write_score(path, energies, gradients):
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_legacy_score(energies, gradients)
    with open(path, "w") as f:
        f.write(buf.getvalue())


def _read_idx(path):
    idx = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                idx.append(int(s))
    return np.asarray(idx, dtype=np.int64)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj-pattern", required=True, help="e.g. out1000_dumptraj.traj.*.dat")
    ap.add_argument("--dump-pattern", required=True, help="same format as minfor-dump-traj ({step},{kind})")
    ap.add_argument("--out-pattern", required=True, help="output score pattern with {step}")
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--attract-par-npz", required=True)
    ap.add_argument("--receptor-ens-list", default=None)
    ap.add_argument("--ligand-pdb", default=None)
    ap.add_argument("--epsilon", type=float, default=15.0)
    ap.add_argument("--cdie", action="store_true")
    ap.add_argument("--reference-pattern", default=None)
    ap.add_argument("--resume", action="store_true", help="reuse existing out score files")
    return ap.parse_args()


def main():
    args = parse_args()
    traj_files = sorted(glob.glob(args.traj_pattern))
    if not traj_files:
        raise ValueError(f"No trajectory files matched: {args.traj_pattern}")

    first_traj = traj_files[0]
    test_dir = args.test_dir or str(Path(first_traj).resolve().parent)

    ens_list_path = args.receptor_ens_list or os.path.join(test_dir, "partner1-ensemble.list")
    ligand_pdb_path = args.ligand_pdb or os.path.join(test_dir, "ligandr.pdb")
    _h, piv, ens0, dofs0, _e0, centered = parse_dat_two_body(first_traj)
    if 2 in piv:
        lig_pivot = piv[2]
    else:
        coor = []
        with open(ligand_pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    coor.append(
                        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    )
        lig_pivot = np.mean(coor, axis=0)

    ctx = load_context(
        receptor_ens_list=ens_list_path,
        ligand_pdb=ligand_pdb_path,
        grid_file=args.grid,
        attract_par_npz=args.attract_par_npz,
        lig_pivot=lig_pivot,
        epsilon=args.epsilon,
        cdie=bool(args.cdie),
    )

    nposes = len(ens0)
    full_e = np.zeros(nposes, dtype=np.float64)
    full_g = np.zeros((nposes, 6), dtype=np.float64)

    for tf in traj_files:
        stem = os.path.basename(tf)
        step = int(stem.split(".")[-2])
        out_path = args.out_pattern.format(step=step)
        if args.resume and os.path.exists(out_path):
            with open(out_path) as f:
                etxt = f.read()
            ee, ef = parse_score_output(etxt)
            eg = -ef
            m0 = min(len(full_e), len(ee))
            full_e[:m0] = ee[:m0]
            full_g[:m0] = eg[:m0]
            continue

        _h, _p, ens, dofs, dat_e, centered = parse_dat_two_body(tf)
        if bool(centered):
            dofs = dofs.copy()
            dofs[:, 3:6] -= lig_pivot[None, :]

        idx_path = _fmt(args.dump_pattern, step, "idx.txt")
        score_path = _fmt(args.dump_pattern, step, "score")
        nb_path = _fmt(args.dump_pattern, step, "nb.tsv")

        active_idx1 = _read_idx(idx_path)
        if len(active_idx1) == 0:
            _write_score(out_path, full_e, full_g)
            continue

        active_idx0 = active_idx1 - 1

        with open(score_path) as f:
            stxt = f.read()
        pot_e, pot_force = parse_score_output(stxt)
        pot_g = -pot_force

        rows = read_nb_table(nb_path)
        ens_sub = ens[active_idx0]
        dofs_sub = dofs[active_idx0]

        pair_lig, pair_rec, pair_mask = build_pair_arrays(rows, ctx["nb_concat"], len(dofs_sub))
        nb_e, nb_g = compute_nb_scores(ctx, ens_sub, dofs_sub, pair_lig, pair_rec, pair_mask)

        total_e = pot_e + nb_e
        total_g = pot_g + nb_g

        full_e[active_idx0] = total_e
        full_g[active_idx0] = total_g

        _write_score(out_path, full_e, full_g)

        if np.isfinite(dat_e).all():
            de = np.abs(full_e - dat_e)
            print(f"step {step}: dat-energy max_abs={de.max():.6e} mean_abs={de.mean():.6e}")

        if args.reference_pattern:
            ref_path = args.reference_pattern.format(step=step)
            if os.path.exists(ref_path):
                with open(ref_path) as f:
                    rtxt = f.read()
                ref_e, ref_f = parse_score_output(rtxt)
                ref_g = -ref_f
                m = min(len(ref_e), len(full_e))
                de = np.abs(full_e[:m] - ref_e[:m])
                dg = np.abs(full_g[:m] - ref_g[:m])
                print(
                    f"step {step}: ref max_abs energy={de.max():.6e} grad={dg.max():.6e}"
                )

    print(f"Processed {len(traj_files)} trajectory steps")


if __name__ == "__main__":
    main()
