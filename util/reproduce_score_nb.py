#!/usr/bin/env python3
"""Reproduce legacy ATTRACT --score (non-grid) energies/gradients with JAX.

This utility targets the legacy scoring mode used in test/out_demo_xylanase.score:
- non-grid pair scoring (nonbon8)
- ensemble receptor selection from the first DOF line in .dat
- ligand rigid-body DOFs (Euler + translation) from the second DOF line

Outputs are ATTRACT-style energies and gradients (force-like sign, matching
legacy printed "Gradients: ...").
"""

import argparse
import os
import re
import resource
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

STRUCT_RE = re.compile(r"^#\d+\s*$")
PIVOT_RE = re.compile(r"^#pivot\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")
ENERGY_RE = re.compile(r"^\s*Energy:\s*([-+0-9.eE]+)\s*$")
GRAD_RE = re.compile(r"^\s*Gradients:\s*(.*?)\s*$")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dat", help="input ATTRACT .dat with poses")
    ap.add_argument("legacy_score", help="legacy ATTRACT --score text output")
    ap.add_argument("receptor_ens_list", help="ensemble list file (1-based order)")
    ap.add_argument("ligand_pdb", help="reduced ligand pdb (e.g. ligandr.pdb)")
    ap.add_argument("--attract-par-npz", default="attract-par.npz", help="ATTRACT forcefield npz")
    ap.add_argument("--max-poses", type=int, default=0, help="cap number of poses (0 = all)")
    ap.add_argument("--batch", type=int, default=16, help="poses per JAX batch")
    ap.add_argument(
        "--rcut-sq",
        type=float,
        default=50.0,
        help="legacy pair cutoff (squared distance, default 50.0)",
    )
    ap.add_argument("--epsilon", type=float, default=15.0, help="dielectric constant")
    ap.add_argument("--cdie", action="store_true", help="use distance-dependent electrostatics")
    ap.add_argument("--memory-gb", type=float, default=20.0, help="address-space memory cap in GB (0 disables)")
    ap.add_argument("--disable-jit", action="store_true", help="disable JAX JIT")
    ap.add_argument("--out-prefix", required=True, help="output prefix for .npy files")
    return ap.parse_args()


def parse_legacy_score(path: str, max_poses: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    energies: List[float] = []
    grads: List[List[float]] = []
    with open(path) as f:
        for line in f:
            m = ENERGY_RE.match(line)
            if m:
                energies.append(float(m.group(1)))
                continue
            g = GRAD_RE.match(line)
            if g:
                vals = [float(tok.replace("D", "E").replace("d", "e")) for tok in FLOAT_RE.findall(g.group(1))]
                if len(vals) == 6:
                    grads.append(vals)
    if max_poses:
        energies = energies[:max_poses]
        grads = grads[:max_poses]
    e = np.asarray(energies, dtype=np.float64)
    g = np.asarray(grads[: len(e)], dtype=np.float64)
    if len(g) != len(e):
        raise ValueError(f"Parsed {len(e)} energies but {len(g)} gradients from {path}")
    return e, g


def parse_dof_line(line: str) -> Optional[List[float]]:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def parse_dat_for_score(path: str, max_poses: int = 0):
    pivots = {}
    ens: List[int] = []
    eulers: List[Tuple[float, float, float]] = []
    trans: List[Tuple[float, float, float]] = []

    current_lines: List[List[float]] = []

    def flush():
        if not current_lines:
            return
        if len(current_lines) < 2:
            raise ValueError("Expected at least 2 DOF lines per structure for fix-receptor scoring dat")
        first = current_lines[0]
        second = current_lines[-1]
        if len(first) != 7:
            raise ValueError(f"Expected first DOF line to have 7 fields (ens+6), got {len(first)}")
        if len(second) not in (6, 7):
            raise ValueError(f"Expected ligand DOF line to have 6 or 7 fields, got {len(second)}")
        ens_id = int(round(first[0]))
        if len(second) == 7:
            second = second[1:]
        phi, ssi, rot, xa, ya, za = second
        ens.append(ens_id)
        eulers.append((phi, ssi, rot))
        trans.append((xa, ya, za))

    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            pm = PIVOT_RE.match(line)
            if pm:
                pid = int(pm.group(1))
                pivots[pid] = np.asarray([float(pm.group(2)), float(pm.group(3)), float(pm.group(4))], dtype=np.float64)
                continue
            if line.startswith("##") or line.startswith("###"):
                continue
            if STRUCT_RE.match(line):
                if max_poses and len(ens) >= max_poses:
                    break
                flush()
                current_lines = []
                continue
            vals = parse_dof_line(line)
            if vals is not None:
                current_lines.append(vals)
    if not (max_poses and len(ens) >= max_poses):
        flush()

    if 2 not in pivots:
        raise ValueError("Could not parse ligand pivot (#pivot 2 ...) from dat header")
    ens_arr = np.asarray(ens, dtype=np.int32)
    euler_arr = np.asarray(eulers, dtype=np.float64)
    trans_arr = np.asarray(trans, dtype=np.float64)
    return pivots, ens_arr, euler_arr, trans_arr


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


def euler2rotmat(phi, ssi, rot):
    cs = jnp.cos(ssi)
    cp = jnp.cos(phi)
    ss = jnp.sin(ssi)
    sp = jnp.sin(phi)
    cscp = cs * cp
    cssp = cs * sp
    sscp = ss * cp
    sssp = ss * sp
    crot = jnp.cos(rot)
    srot = jnp.sin(rot)
    return jnp.asarray(
        [
            [crot * cscp + srot * sp, srot * cscp - crot * sp, sscp],
            [crot * cssp - srot * cp, srot * cssp + crot * cp, sssp],
            [-crot * ss, -srot * ss, cs],
        ],
        dtype=jnp.float64,
    )


def build_kernel(
    lig_centered,
    lig_pivot,
    rec_coor_ens,
    rec_charge_ens,
    lig_charge,
    rlen_mat,
    alen_mat,
    emin_mat,
    rmin2_mat,
    ivor_mat,
    rcut_sq,
    use_cdie,
):
    plateaudelta = jnp.float64(2.0)  # potshape=8 in attract.par used here
    r2_min = jnp.float64(1.0e-3)
    inv50sq = jnp.float64(1.0 / (50.0 * 50.0))
    inv50 = jnp.float64(1.0 / 50.0)

    lig_centered_j = jnp.asarray(lig_centered, dtype=jnp.float64)
    lig_pivot_j = jnp.asarray(lig_pivot, dtype=jnp.float64)
    rec_coor_ens_j = jnp.asarray(rec_coor_ens, dtype=jnp.float64)
    rec_charge_ens_j = jnp.asarray(rec_charge_ens, dtype=jnp.float64)
    lig_charge_j = jnp.asarray(lig_charge, dtype=jnp.float64)
    rlen_mat_j = jnp.asarray(rlen_mat, dtype=jnp.float64)
    alen_mat_j = jnp.asarray(alen_mat, dtype=jnp.float64)
    emin_mat_j = jnp.asarray(emin_mat, dtype=jnp.float64)
    rmin2_mat_j = jnp.asarray(rmin2_mat, dtype=jnp.float64)
    ivor_mat_j = jnp.asarray(ivor_mat, dtype=jnp.float64)
    rcut_sq_j = jnp.float64(rcut_sq)

    def transform_coords(params):
        phi, ssi, rot, xa, ya, za = params
        rotmat = euler2rotmat(phi, ssi, rot)
        trans = jnp.asarray([xa, ya, za], dtype=jnp.float64)
        return lig_centered_j @ rotmat.T + trans + lig_pivot_j

    def single_pose(euler, trans, ens_zero):
        params = jnp.concatenate((euler, trans), axis=0)
        lig_world, pullback = jax.vjp(transform_coords, params)

        rec = rec_coor_ens_j[ens_zero]
        rec_q = rec_charge_ens_j[ens_zero]

        d = lig_world[None, :, :] - rec[:, None, :]
        r2_raw = jnp.sum(d * d, axis=-1)
        in_cut = r2_raw <= rcut_sq_j
        r2 = jnp.maximum(r2_raw, r2_min)
        rr2 = 1.0 / r2
        rr23 = rr2 * rr2 * rr2

        rep = rlen_mat_j * rr2
        vlj = (rep - alen_mat_j) * rr23
        vdw = jnp.where(
            r2 < rmin2_mat_j,
            vlj + (ivor_mat_j - 1.0) * emin_mat_j,
            ivor_mat_j * vlj,
        )

        charge = rec_q[:, None] * lig_charge_j[None, :]
        if use_cdie:
            rr1 = jnp.maximum(1.0 / jnp.sqrt(r2) - inv50, 0.0)
            et = charge * rr1
            elec_fac = jnp.where(rr1 <= 0.0, et, charge * (rr1 + inv50))
        else:
            rr2a = jnp.maximum(rr2 - inv50sq, 0.0)
            et = charge * rr2a
            elec_fac = jnp.where(rr2a <= 0.0, 2.0 * et, 2.0 * charge * rr2)

        pair_energy = (vdw + et) * in_cut
        energy = jnp.sum(pair_energy)

        fb = 6.0 * vlj + plateaudelta * (rep * rr23)
        vdw_fac = jnp.where(r2 < rmin2_mat_j, fb, ivor_mat_j * fb)
        pair_force = ((vdw_fac + elec_fac) * rr2)[..., None] * d
        pair_force = jnp.where(in_cut[..., None], pair_force, 0.0)
        lig_force = jnp.sum(pair_force, axis=0)

        grad_params = pullback(lig_force)[0]
        return energy, grad_params

    vmapped = jax.vmap(single_pose, in_axes=(0, 0, 0))
    return jax.jit(vmapped)


def summarize(name: str, ref: np.ndarray, cand: np.ndarray):
    delta = cand - ref
    rmse = float(np.sqrt(np.mean(delta * delta)))
    mae = float(np.mean(np.abs(delta)))
    p = float(np.corrcoef(ref, cand)[0, 1])
    print(f"{name}: mae={mae:.6f} rmse={rmse:.6f} pearson={p:.6f}")
    print(
        f"{name}: delta p50={np.percentile(delta,50):.6f} "
        f"p90={np.percentile(delta,90):.6f} p99={np.percentile(delta,99):.6f} "
        f"min={delta.min():.6f} max={delta.max():.6f}"
    )


def main():
    args = parse_args()

    if args.memory_gb > 0:
        mem_bytes = int(args.memory_gb * (1024**3))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        print(f"Applied RLIMIT_AS={args.memory_gb:.2f} GB")

    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
        print("JAX JIT disabled (--disable-jit)")

    print(f"JAX devices: {jax.devices()}")

    ref_energies, ref_grads = parse_legacy_score(args.legacy_score, max_poses=args.max_poses)
    pivots, ens_ids, eulers, trans = parse_dat_for_score(args.input_dat, max_poses=args.max_poses)

    n = len(ref_energies)
    if len(ens_ids) != n:
        n2 = min(n, len(ens_ids))
        print(f"Warning: parsed dat poses {len(ens_ids)} vs legacy score {n}; truncating to {n2}")
        n = n2
        ref_energies = ref_energies[:n]
        ref_grads = ref_grads[:n]
        ens_ids = ens_ids[:n]
        eulers = eulers[:n]
        trans = trans[:n]

    with open(args.receptor_ens_list) as f:
        rec_files = [line.strip() for line in f if line.strip()]
    if not rec_files:
        raise ValueError("Empty receptor ensemble list")

    rec_coords_all = []
    rec_types_all = []
    rec_charge_all = []
    rec_weight_all = []
    for p in rec_files:
        coor, at, ch, we = parse_reduced_pdb(os.path.join(os.path.dirname(args.receptor_ens_list), p) if not os.path.isabs(p) and not os.path.exists(p) else p)
        rec_coords_all.append(coor)
        rec_types_all.append(at)
        rec_charge_all.append(ch)
        rec_weight_all.append(we)

    rec_types0 = rec_types_all[0]
    rec_mask = rec_types0 != 99
    rec_types = rec_types0[rec_mask]
    rec_coords_ens = np.asarray([c[rec_mask] for c in rec_coords_all], dtype=np.float64)
    rec_charge_ens = np.asarray([c[rec_mask] for c in rec_charge_all], dtype=np.float64)
    rec_weight = np.asarray(rec_weight_all[0][rec_mask], dtype=np.float64)

    lig_coords_raw, lig_types0, lig_charge0, lig_weight0 = parse_reduced_pdb(args.ligand_pdb)
    lig_mask = lig_types0 != 99
    lig_types = lig_types0[lig_mask]
    lig_coords = lig_coords_raw[lig_mask]
    lig_charge = lig_charge0[lig_mask]
    lig_weight = lig_weight0[lig_mask]

    for i, at in enumerate(rec_types_all[1:], start=2):
        if not np.array_equal(at[rec_mask], rec_types):
            raise ValueError(f"Receptor ensemble atom types differ at ensemble {i}")

    felec = np.sqrt(332.053986 / args.epsilon)
    rec_charge_ens = rec_charge_ens * felec
    lig_charge = lig_charge * felec

    par = np.load(args.attract_par_npz)
    rc = par["rc"].astype(np.float64)
    ac = par["ac"].astype(np.float64)
    ivor = par["ivor"].astype(np.float64)

    rec_idx = rec_types - 1
    lig_idx = lig_types - 1
    rc_mat = rc[rec_idx][:, lig_idx]
    ac_mat = ac[rec_idx][:, lig_idx]
    ivor_mat = ivor[rec_idx][:, lig_idx]

    weight_mat = rec_weight[:, None] * lig_weight[None, :]
    rlen_mat = rc_mat * weight_mat
    alen_mat = ac_mat * weight_mat
    emin_mat = np.zeros_like(alen_mat)
    rmin2_mat = np.zeros_like(alen_mat)
    good = (alen_mat > 0.0) & (rlen_mat > 0.0)
    emin_mat[good] = -27.0 * (alen_mat[good] ** 4) / (256.0 * (rlen_mat[good] ** 3))
    rmin2_mat[good] = 4.0 * rlen_mat[good] / (3.0 * alen_mat[good])

    lig_pivot = pivots[2].astype(np.float64)
    lig_centered = lig_coords - lig_pivot[None, :]

    ens_zero = ens_ids.astype(np.int32) - 1
    if ens_zero.min() < 0 or ens_zero.max() >= len(rec_files):
        raise ValueError(f"Ensemble index out of range: min={ens_zero.min()+1}, max={ens_zero.max()+1}, nr={len(rec_files)}")

    kernel = build_kernel(
        lig_centered=lig_centered,
        lig_pivot=lig_pivot,
        rec_coor_ens=rec_coords_ens,
        rec_charge_ens=rec_charge_ens,
        lig_charge=lig_charge,
        rlen_mat=rlen_mat,
        alen_mat=alen_mat,
        emin_mat=emin_mat,
        rmin2_mat=rmin2_mat,
        ivor_mat=ivor_mat,
        rcut_sq=args.rcut_sq,
        use_cdie=args.cdie,
    )

    cand_e = np.zeros(n, dtype=np.float64)
    cand_g = np.zeros((n, 6), dtype=np.float64)

    bsz = max(1, int(args.batch))
    for start in range(0, n, bsz):
        end = min(n, start + bsz)
        e_batch = jnp.asarray(eulers[start:end], dtype=jnp.float64)
        t_batch = jnp.asarray(trans[start:end], dtype=jnp.float64)
        ens_batch = jnp.asarray(ens_zero[start:end], dtype=jnp.int32)
        ene, grad = kernel(e_batch, t_batch, ens_batch)
        cand_e[start:end] = np.asarray(ene)
        cand_g[start:end] = np.asarray(grad)
        if (start // bsz) % 10 == 0:
            print(f"processed {end}/{n}")

    np.save(args.out_prefix + ".legacy_energy.npy", ref_energies.astype(np.float32))
    np.save(args.out_prefix + ".legacy_grad.npy", ref_grads.astype(np.float32))
    np.save(args.out_prefix + ".jax_energy.npy", cand_e.astype(np.float32))
    np.save(args.out_prefix + ".jax_grad.npy", cand_g.astype(np.float32))

    summarize("energy", ref_energies, cand_e)
    summarize("grad", ref_grads.reshape(-1), cand_g.reshape(-1))
    for i, name in enumerate(("phi", "ssi", "rot", "tx", "ty", "tz")):
        summarize(f"grad_{name}", ref_grads[:, i], cand_g[:, i])


if __name__ == "__main__":
    main()
