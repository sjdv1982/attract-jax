#!/usr/bin/env python3
"""Minimize poses and dump per-step active-pose decomposition files.

Writes, per step:
- potential-only score (.score format) for active poses
- NB hit table (.tsv) for active poses
- active pose indices text file (1-based global indices)

Dump uses JAX decomposition and is independent from the minimizer oracle.
"""

import argparse
import io
import os
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from jax_scorer import JaxScoreOracle
from minfor import (
    LegacyScoreOracle,
    init_hpack_diag,
    mc11a_packed,
    mc11e_packed,
    parse_dat_two_body,
    print_legacy_score,
    resolve_attract_paths,
    write_dat_two_body,
)
from minfor_nb import (
    extract_nb_rows,
    load_context,
    write_nb_table,
)


def _fmt(pattern, step, kind):
    return pattern.format(step=step, kind=kind)


def _write_score(path, energies, gradients):
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_legacy_score(energies, gradients)
    with open(path, "w") as f:
        f.write(buf.getvalue())


def _write_indices(path, idx1):
    with open(path, "w") as f:
        for i in idx1:
            f.write(f"{int(i)}\n")


class StepDumper:
    def __init__(self, pattern, ctx, pot_oracle, input_centered=False):
        self.pattern = pattern
        self.ctx = ctx
        self.pot_oracle = pot_oracle
        self.input_centered = bool(input_centered)

    def dump(self, step, global_idx0, ens_sub, dofs_sub):
        score_path = _fmt(self.pattern, step, "score")
        nb_path = _fmt(self.pattern, step, "nb.tsv")
        idx_path = _fmt(self.pattern, step, "idx.txt")

        idx1 = np.asarray(global_idx0, dtype=np.int64) + 1
        _write_indices(idx_path, idx1)

        if len(global_idx0) == 0:
            write_nb_table(nb_path, np.zeros((0, 9), dtype=np.int64))
            open(score_path, "w").close()
            return

        # JAX decomposition uses centered-ligand translations; legacy minimizer
        # state remains in DAT/legacy convention.
        dofs_jax = dofs_sub
        if self.input_centered:
            dofs_jax = dofs_sub.copy()
            dofs_jax[:, 3:6] -= self.ctx["lig_pivot"][None, :]

        rows = extract_nb_rows(ens_sub, dofs_jax, self.ctx)
        write_nb_table(nb_path, rows)

        pot_e, pot_g = self.pot_oracle.score_potential_batch(ens_sub, dofs_jax)
        _write_score(score_path, pot_e, pot_g)


def minfor_minimize_batched_dump(
    oracle,
    ens,
    dofs0,
    dumper,
    maxfun=150,
    init_metric=0.01,
    acc=1e-9,
    trace_every=0,
    traj_prefix=None,
    traj_header=None,
):
    N, n = dofs0.shape

    hpack = np.tile(init_hpack_diag(n, init_metric), (N, 1))
    w_arr = np.zeros((N, n), dtype=np.float64)
    ir_arr = np.full(N, n, dtype=np.int32)

    x = dofs0.copy()
    g = np.zeros((N, n), dtype=np.float64)
    gesa = np.full(N, np.inf, dtype=np.float64)
    x_best = dofs0.copy()
    f_best = np.full(N, np.inf, dtype=np.float64)

    xaa = np.zeros((N, n), dtype=np.float64)
    d_arr = np.zeros((N, n), dtype=np.float64)
    fa_arr = np.full(N, np.inf, dtype=np.float64)
    ga_arr = np.zeros((N, n), dtype=np.float64)
    dga_arr = np.zeros(N, dtype=np.float64)
    stmin_arr = np.zeros(N, dtype=np.float64)
    stepbd_arr = np.zeros(N, dtype=np.float64)
    steplb_arr = np.zeros(N, dtype=np.float64)
    fmin_arr = np.zeros(N, dtype=np.float64)
    gmin_arr = np.zeros(N, dtype=np.float64)
    step_arr = np.zeros(N, dtype=np.float64)
    dff = np.zeros(N, dtype=np.float64)
    isfv = np.ones(N, dtype=np.int32)
    nfev = np.zeros(N, dtype=np.int32)
    xbb = dofs0.copy()
    active = np.ones(N, dtype=bool)

    traj_dofs = dofs0.copy()
    traj_energies = np.full(N, np.nan, dtype=np.float64)

    e0, g0 = oracle.score_batch(ens, dofs0)
    nfev[:] = 1
    gesa[:] = e0
    g[:] = g0
    f_best[:] = e0
    x_best[:] = dofs0
    traj_energies[:] = e0

    if traj_prefix and traj_header is not None:
        write_dat_two_body(
            f"{traj_prefix}.{0:04d}.dat", traj_header, ens, traj_dofs, traj_energies
        )

    dumper.dump(0, np.arange(N, dtype=np.int64), ens, dofs0)

    for i in range(N):
        xaa[i] = x[i]
        fa_arr[i] = gesa[i]
        isfv[i] = 1
        ga_arr[i] = g[i]

        d_arr[i], w_arr[i] = mc11e_packed(
            hpack[i], -ga_arr[i], w_arr[i], int(ir_arr[i]), n
        )
        cmax = float(np.max(np.abs(d_arr[i])))
        dga_arr[i] = float(np.dot(ga_arr[i], d_arr[i]))

        if cmax <= 0.0 or dga_arr[i] >= 0.0:
            active[i] = False
            continue

        stmin_arr[i] = 0.0
        stepbd_arr[i] = 0.0
        steplb_arr[i] = acc / cmax
        fmin_arr[i] = fa_arr[i]
        gmin_arr[i] = dga_arr[i]

        stp = 1.0
        if dff[i] <= 0.0:
            stp = min(stp, 1.0 / cmax)
        else:
            stp = min(stp, 2.0 * dff[i] / (-dga_arr[i]))
        step_arr[i] = stp

        c = stmin_arr[i] + step_arr[i]
        xbb[i] = xaa[i] + c * d_arr[i]

    tick = 0
    t_batch = time.time()

    while active.any():
        over = active & (nfev >= maxfun)
        active[over] = False
        act_idx = np.where(active)[0]
        if len(act_idx) == 0:
            break

        e_batch, g_batch = oracle.score_batch(ens[act_idx], xbb[act_idx])
        nfev[act_idx] += 1
        tick += 1

        dumper.dump(tick, act_idx, ens[act_idx], xbb[act_idx])

        if traj_prefix and traj_header is not None:
            for k, ii in enumerate(act_idx):
                traj_dofs[ii] = xbb[ii]
                traj_energies[ii] = float(e_batch[k])
            write_dat_two_body(
                f"{traj_prefix}.{tick:04d}.dat",
                traj_header,
                ens,
                traj_dofs,
                traj_energies,
            )

        need_label_110 = []
        need_label_135 = []
        for k, ii in enumerate(act_idx):
            i = int(ii)
            fb = float(e_batch[k])
            gb = g_batch[k]

            isfv[i] = min(2, isfv[i])
            if fb <= gesa[i]:
                better = fb < gesa[i]
                if not better:
                    gl1 = float(np.dot(g[i], g[i]))
                    gl2 = float(np.dot(gb, gb))
                    better = gl2 < gl1
                if better:
                    isfv[i] = 3
                    gesa[i] = fb
                    x[i] = xbb[i].copy()
                    g[i] = gb.copy()
                    if fb < f_best[i]:
                        f_best[i] = fb
                        x_best[i] = xbb[i].copy()

            c = stmin_arr[i] + step_arr[i]
            dgb = float(np.dot(gb, d_arr[i]))

            if fb - fa_arr[i] <= 0.1 * c * dga_arr[i]:
                stepbd_arr[i] -= step_arr[i]
                stmin_arr[i] = c
                fmin_arr[i] = fb
                gmin_arr[i] = dgb

                new_step = 9.0 * stmin_arr[i]
                if stepbd_arr[i] > 0.0:
                    new_step = 0.5 * stepbd_arr[i]

                ctmp = dga_arr[i] + 3.0 * dgb - 4.0 * (fb - fa_arr[i]) / stmin_arr[i]
                if ctmp > 0.0:
                    new_step = min(new_step, stmin_arr[i] * max(1.0, -dgb / ctmp))

                if dgb < 0.7 * dga_arr[i]:
                    if stmin_arr[i] + new_step <= steplb_arr[i]:
                        if gesa[i] < fa_arr[i] or isfv[i] >= 2:
                            need_label_110.append(i)
                        else:
                            active[i] = False
                    else:
                        step_arr[i] = new_step
                        xbb[i] = xaa[i] + (stmin_arr[i] + new_step) * d_arr[i]
                    continue

                isfv[i] = 4 - isfv[i]
                if stmin_arr[i] + new_step <= steplb_arr[i]:
                    if gesa[i] < fa_arr[i] or isfv[i] >= 2:
                        need_label_110.append(i)
                    else:
                        active[i] = False
                    continue

                ga_old = ga_arr[i].copy()
                y = gb - ga_arr[i]
                denom1 = dga_arr[i]
                denom2 = stmin_arr[i] * (dgb - dga_arr[i])

                if abs(denom1) < 1e-16 or abs(denom2) < 1e-16:
                    if gesa[i] < fa_arr[i] or isfv[i] >= 2:
                        need_label_110.append(i)
                    else:
                        active[i] = False
                    continue

                h1, _, w1, ir1 = mc11a_packed(
                    hpack[i], ga_old, 1.0 / denom1, w_arr[i], -n, 1, 0.0, n=n
                )
                h2, _, _, ir2 = mc11a_packed(
                    h1, y, 1.0 / denom2, y.copy(), -ir1, 0, 0.0, n=n, alias_zw=True
                )

                if ir2 < n:
                    if gesa[i] < fa_arr[i] or isfv[i] >= 2:
                        need_label_110.append(i)
                    else:
                        active[i] = False
                    continue

                hpack[i] = h2
                w_arr[i] = w1
                ir_arr[i] = ir2
                dff[i] = fa_arr[i] - fb
                fa_arr[i] = fb
                xaa[i] = xbb[i].copy()
                ga_arr[i] = gb.copy()
                need_label_135.append(i)

            else:
                if step_arr[i] > steplb_arr[i]:
                    stepbd_arr[i] = step_arr[i]
                    ctmp = gmin_arr[i] + dgb - 3.0 * (fb - fmin_arr[i]) / step_arr[i]
                    disc = ctmp * ctmp - gmin_arr[i] * dgb
                    if disc < 0.0:
                        disc = 0.0
                    denom = ctmp + gmin_arr[i] - np.sqrt(disc)
                    if abs(denom) < 1e-16:
                        fac = 0.1
                    else:
                        fac = max(0.1, gmin_arr[i] / denom)
                    step_arr[i] *= fac
                    xbb[i] = xaa[i] + (stmin_arr[i] + step_arr[i]) * d_arr[i]
                else:
                    if gesa[i] < fa_arr[i] or isfv[i] >= 2:
                        need_label_110.append(i)
                    else:
                        active[i] = False

        for i in need_label_135:
            if not active[i]:
                continue
            d_arr[i], w_arr[i] = mc11e_packed(
                hpack[i], -ga_arr[i], w_arr[i], int(ir_arr[i]), n
            )
            cmax = float(np.max(np.abs(d_arr[i])))
            dga_arr[i] = float(np.dot(ga_arr[i], d_arr[i]))
            if cmax <= 0.0 or dga_arr[i] >= 0.0:
                active[i] = False
                continue
            stmin_arr[i] = 0.0
            stepbd_arr[i] = 0.0
            steplb_arr[i] = acc / cmax
            fmin_arr[i] = fa_arr[i]
            gmin_arr[i] = dga_arr[i]
            stp = 1.0
            if dff[i] <= 0.0:
                stp = min(stp, 1.0 / cmax)
            else:
                stp = min(stp, 2.0 * dff[i] / (-dga_arr[i]))
            step_arr[i] = stp
            c = stmin_arr[i] + step_arr[i]
            xbb[i] = xaa[i] + c * d_arr[i]

        for i in need_label_110:
            if not active[i]:
                continue
            xaa[i] = x[i]
            fa_arr[i] = gesa[i]
            isfv[i] = 1
            ga_arr[i] = g[i]
            d_arr[i], w_arr[i] = mc11e_packed(
                hpack[i], -ga_arr[i], w_arr[i], int(ir_arr[i]), n
            )
            cmax = float(np.max(np.abs(d_arr[i])))
            dga_arr[i] = float(np.dot(ga_arr[i], d_arr[i]))
            if cmax <= 0.0 or dga_arr[i] >= 0.0:
                active[i] = False
                continue
            stmin_arr[i] = 0.0
            stepbd_arr[i] = 0.0
            steplb_arr[i] = acc / cmax
            fmin_arr[i] = fa_arr[i]
            gmin_arr[i] = dga_arr[i]
            stp = 1.0
            if dff[i] <= 0.0:
                stp = min(stp, 1.0 / cmax)
            else:
                stp = min(stp, 2.0 * dff[i] / (-dga_arr[i]))
            step_arr[i] = stp
            c = stmin_arr[i] + step_arr[i]
            xbb[i] = xaa[i] + c * d_arr[i]

        if trace_every and tick % trace_every == 0:
            elapsed = time.time() - t_batch
            print(
                f"  tick {tick}: active={int(active.sum())}/{N}, best={f_best.min():.3f}, {elapsed:.1f}s"
            )

    return x_best, f_best, nfev


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dat")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--dump-pattern", required=True, help="python format pattern with {step} and {kind}")
    ap.add_argument("--maxfun", type=int, default=150)
    ap.add_argument("--max-poses", type=int, default=0)
    ap.add_argument("--pose-offset", type=int, default=0)
    ap.add_argument("--trace-every", type=int, default=10)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--init-metric", type=float, default=0.01)
    ap.add_argument("--oracle", choices=["legacy", "jax"], default="legacy")
    ap.add_argument("--grid", default=None)
    ap.add_argument("--attract-par-npz", default=None)
    ap.add_argument("--receptor-ens-list", default=None)
    ap.add_argument("--ligand-pdb", default=None)
    ap.add_argument("--epsilon", type=float, default=15.0)
    ap.add_argument("--cdie", action="store_true")
    ap.add_argument("--energy-batch", type=int, default=256)
    ap.add_argument("--max-nb-cap", type=int, default=0)
    ap.add_argument("--nb-mode", choices=["fixed", "bucketed"], default="fixed")
    ap.add_argument("--nb-bucket-thresholds", default="8")
    ap.add_argument("--traj", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    test_dir = args.test_dir or str(Path(args.input_dat).resolve().parent)

    total_read = (args.pose_offset + args.max_poses) if args.max_poses else 0
    header, pivots, ens, dofs0, _e0, centered_ligands = parse_dat_two_body(
        args.input_dat, max_poses=total_read
    )
    if args.pose_offset > 0:
        ens = ens[args.pose_offset :]
        dofs0 = dofs0[args.pose_offset :]

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

    ens_list_path = args.receptor_ens_list or os.path.join(test_dir, "partner1-ensemble.list")
    ligand_pdb_path = args.ligand_pdb or os.path.join(test_dir, "ligandr.pdb")

    if not args.grid or not args.attract_par_npz:
        raise ValueError("--grid and --attract-par-npz are required for dump decomposition")

    # Dump context: JAX potential/NB decomposition, independent of minimizer oracle
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
    pot_oracle = JaxScoreOracle(
        receptor_ens_list=ens_list_path,
        ligand_pdb=ligand_pdb_path,
        grid_file=args.grid,
        attract_par_npz=args.attract_par_npz,
        lig_pivot=lig_pivot,
        epsilon=args.epsilon,
        cdie=bool(args.cdie),
        energy_batch=args.energy_batch,
        max_nb_cap=args.max_nb_cap,
        nb_mode=args.nb_mode,
        nb_bucket_thresholds=args.nb_bucket_thresholds,
    )
    dumper = StepDumper(
        args.dump_pattern, ctx, pot_oracle, input_centered=input_centered
    )

    tmpdir_ctx = None
    if args.oracle == "jax":
        oracle = JaxScoreOracle(
            receptor_ens_list=ens_list_path,
            ligand_pdb=ligand_pdb_path,
            grid_file=args.grid,
            attract_par_npz=args.attract_par_npz,
            lig_pivot=lig_pivot,
            epsilon=args.epsilon,
            cdie=bool(args.cdie),
            energy_batch=args.energy_batch,
            max_nb_cap=args.max_nb_cap,
            nb_mode=args.nb_mode,
            nb_bucket_thresholds=args.nb_bucket_thresholds,
        )
    else:
        import tempfile as _tempfile

        paths = resolve_attract_paths(test_dir)
        tmpdir_ctx = _tempfile.TemporaryDirectory()
        tmpdir = tmpdir_ctx.__enter__()
        oracle = LegacyScoreOracle(
            attract_bin=paths["attract_bin"],
            attract_par=paths["attract_par"],
            receptor_pdb=paths["receptor_pdb"],
            ligand_pdb=paths["ligand_pdb"],
            ens_list=paths["ens_list"],
            grid_header=paths["grid_header"],
            header=header,
            tmpdir=tmpdir,
            cwd=test_dir,
        )

    try:
        traj_prefix = f"{args.out_prefix}.traj" if args.traj else None
        dofs_out, energies_out, nfev_out = minfor_minimize_batched_dump(
            oracle,
            ens,
            dofs0,
            dumper,
            maxfun=args.maxfun,
            init_metric=args.init_metric,
            trace_every=args.trace_every,
            traj_prefix=traj_prefix,
            traj_header=header,
        )
    finally:
        if tmpdir_ctx is not None:
            tmpdir_ctx.__exit__(None, None, None)

    out_dat = args.out_prefix + ".dat"
    write_dat_two_body(out_dat, header, ens, dofs_out, energies=energies_out)
    np.save(args.out_prefix + ".dofs.npy", dofs_out.astype(np.float32))
    np.save(args.out_prefix + ".energy.npy", energies_out.astype(np.float32))
    np.save(args.out_prefix + ".ens.npy", ens.astype(np.int32))
    np.save(args.out_prefix + ".nfev.npy", nfev_out.astype(np.int32))
    print(f"Saved minimization outputs with prefix {args.out_prefix}")


if __name__ == "__main__":
    main()
