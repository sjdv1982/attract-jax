#!/usr/bin/env python3
"""Shared helpers for minfor-dump.py and minfor-eval.py."""

import os
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

from minfor import dofs_to_mats_np
from reproduce_grid_score import (
    dofs_to_mats,
    elec_dsq_cdie,
    elec_dsq_const,
    nonbon,
    parse_reduced_pdb,
    read_grid_with_electro,
)

jax.config.update("jax_enable_x64", True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

def load_context(
    receptor_ens_list,
    ligand_pdb,
    grid_file,
    attract_par_npz,
    lig_pivot,
    epsilon=15.0,
    cdie=False,
):
    with open(receptor_ens_list) as f:
        rec_files = [line.strip() for line in f if line.strip()]
    if not rec_files:
        raise ValueError("Empty receptor ensemble list")

    list_dir = os.path.dirname(os.path.abspath(receptor_ens_list))
    rec_coords_all = []
    rec_types_all = []
    rec_charge_all = []
    for rf in rec_files:
        p = rf if os.path.isabs(rf) else os.path.join(list_dir, rf)
        c, a, q, _w = parse_reduced_pdb(p)
        rec_coords_all.append(c)
        rec_types_all.append(a)
        rec_charge_all.append(q)

    lig_coords0, lig_types0, lig_charge0, _lig_w = parse_reduced_pdb(ligand_pdb)

    rec_mask = rec_types_all[0] != 99
    lig_mask = lig_types0 != 99

    rec_types = rec_types_all[0][rec_mask]
    lig_types = lig_types0[lig_mask]

    rec_coords_ens = np.asarray([c[rec_mask] for c in rec_coords_all], dtype=np.float64)
    rec_charge_ens_raw = np.asarray([q[rec_mask] for q in rec_charge_all], dtype=np.float64)
    lig_coords = lig_coords0[lig_mask].astype(np.float64)
    lig_charge_raw = lig_charge0[lig_mask].astype(np.float64)

    felec = np.sqrt(332.053986 / float(epsilon))
    rec_charge_ens_scaled = rec_charge_ens_raw * felec
    lig_charge_scaled = lig_charge_raw * felec

    lig_alphabet, lig_atomtypes_ff = np.unique(lig_types, return_inverse=True)
    rec_alphabet, rec_atomtypes_ff = np.unique(rec_types, return_inverse=True)

    par = np.load(attract_par_npz)
    rc = par["rc"][rec_alphabet - 1][:, lig_alphabet - 1].astype(np.float64)
    ac = par["ac"][rec_alphabet - 1][:, lig_alphabet - 1].astype(np.float64)
    ivor = par["ivor"][rec_alphabet - 1][:, lig_alphabet - 1].astype(np.float64)
    emin = -27.0 * ac**4 / (256.0 * rc**3)
    rmin2 = 4.0 * rc / (3.0 * ac)
    FF = namedtuple("FF", ("rc", "ac", "ivor", "emin", "rmin2"))
    ff = FF(
        jnp.array(rc, dtype=jnp.float64),
        jnp.array(ac, dtype=jnp.float64),
        jnp.array(ivor, dtype=jnp.float64),
        jnp.array(emin, dtype=jnp.float64),
        jnp.array(rmin2, dtype=jnp.float64),
    )

    grid = read_grid_with_electro(open(grid_file, "rb").read())

    rec_mapping = np.cumsum(rec_mask) - 1
    nb_flat = grid.neighbour_grid.reshape(-1)
    valid = nb_flat < 2**16 - 1
    nb_flat[valid] = rec_mapping[nb_flat[valid]]

    nr_neigh = np.asarray(grid.nr_neighbours, dtype=np.int32).reshape(-1)
    nr_neigh = np.where(nr_neigh > 0, nr_neigh, 0)
    nb_ravel = np.asarray(grid.neighbour_grid.reshape(-1, grid.neighbour_grid.shape[-1]), dtype=np.int32)

    nb_start = np.zeros(len(nr_neigh), dtype=np.int64)
    if len(nr_neigh) > 1:
        nb_start[1:] = np.cumsum(nr_neigh[:-1], dtype=np.int64)
    total_nb = int(nb_start[-1] + nr_neigh[-1]) if len(nr_neigh) else 0
    nb_concat = np.empty(total_nb, dtype=np.int32)
    for i in range(len(nr_neigh)):
        k = int(nr_neigh[i])
        if k > 0:
            s = int(nb_start[i])
            nb_concat[s : s + k] = nb_ravel[i, :k]

    return {
        "lig_pivot": np.asarray(lig_pivot, dtype=np.float64),
        "lig_coords": lig_coords,
        "lig_atomtypes_ff": np.asarray(lig_atomtypes_ff, dtype=np.int32),
        "lig_charge_scaled": lig_charge_scaled,
        "rec_coords_ens": rec_coords_ens,
        "rec_atomtypes_ff": np.asarray(rec_atomtypes_ff, dtype=np.int32),
        "rec_charge_ens_scaled": rec_charge_ens_scaled,
        "ff": ff,
        "grid": grid,
        "nr_neigh": nr_neigh,
        "nb_start": nb_start,
        "nb_concat": nb_concat,
        "plateaudissq": float(grid.plateaudis) ** 2,
        "cdie": bool(cdie),
    }


def extract_nb_rows(ens, dofs, ctx):
    grid = ctx["grid"]
    lig_coords = ctx["lig_coords"]
    nr_neigh = ctx["nr_neigh"]
    nb_start = ctx["nb_start"]

    mats = dofs_to_mats_np(dofs, ctx["lig_pivot"])
    coor_lig2 = np.concatenate((lig_coords, np.ones((lig_coords.shape[0], 1), dtype=np.float64)), axis=1)
    all_coors = np.einsum("jk,bkl->bjl", coor_lig2, mats)[:, :, :3]

    origin = np.asarray(grid.origin, dtype=np.float64)
    spacing = float(grid.gridspacing)
    dim = np.asarray(grid.dim, dtype=np.int32)

    rows = []
    for p in range(len(dofs)):
        vox = (all_coors[p] - origin[None, :]) / spacing
        in_inner = (
            (vox[:, 0] >= 0)
            & (vox[:, 0] < dim[0] - 1)
            & (vox[:, 1] >= 0)
            & (vox[:, 1] < dim[1] - 1)
            & (vox[:, 2] >= 0)
            & (vox[:, 2] < dim[2] - 1)
        )
        if not np.any(in_inner):
            continue
        pos = np.floor(vox[in_inner] + 0.5).astype(np.int32)
        lig_atoms = np.where(in_inner)[0]
        flat = np.ravel_multi_index((pos[:, 0], pos[:, 1], pos[:, 2]), tuple(dim), mode="clip")
        k = nr_neigh[flat]
        keep = k > 0
        if not np.any(keep):
            continue
        pos = pos[keep]
        lig_atoms = lig_atoms[keep]
        flat = flat[keep]
        k = k[keep]
        n = nb_start[flat]
        rec_conf = int(ens[p]) if int(ens[p]) > 0 else -1
        for la, xyz, kk, nn in zip(lig_atoms, pos, k, n):
            rows.append(
                (
                    p + 1,
                    rec_conf,
                    -1,
                    int(la),
                    int(xyz[0]),
                    int(xyz[1]),
                    int(xyz[2]),
                    int(kk),
                    int(nn),
                )
            )
    return np.asarray(rows, dtype=np.int64)


def write_nb_table(path, rows):
    with open(path, "w") as f:
        f.write(
            "pose_index\treceptor_conformer_index\tligand_conformer_index\t"
            "ligand_atom_index\tvoxel_x\tvoxel_y\tvoxel_z\tK\tN\n"
        )
        for r in rows:
            f.write("\t".join(str(int(v)) for v in r) + "\n")


def read_nb_table(path):
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("pose_index"):
                continue
            parts = s.split("\t")
            if len(parts) != 9:
                raise ValueError(f"Invalid table row: {line.rstrip()}")
            rows.append(tuple(int(x) for x in parts))
    return np.asarray(rows, dtype=np.int64)


def build_pair_arrays(rows, nb_concat, nposes):
    lig_lists = [[] for _ in range(nposes)]
    rec_lists = [[] for _ in range(nposes)]
    for r in rows:
        p = int(r[0]) - 1
        lig_atom = int(r[3])
        k = int(r[7])
        n = int(r[8])
        if p < 0 or p >= nposes or k <= 0:
            continue
        rec_atoms = nb_concat[n : n + k]
        lig_lists[p].extend([lig_atom] * k)
        rec_lists[p].extend(int(x) for x in rec_atoms)

    max_pairs = max((len(x) for x in lig_lists), default=0)
    if max_pairs == 0:
        pair_lig = np.zeros((nposes, 1), dtype=np.int32)
        pair_rec = np.zeros((nposes, 1), dtype=np.int32)
        pair_mask = np.zeros((nposes, 1), dtype=bool)
        return pair_lig, pair_rec, pair_mask

    pair_lig = np.zeros((nposes, max_pairs), dtype=np.int32)
    pair_rec = np.zeros((nposes, max_pairs), dtype=np.int32)
    pair_mask = np.zeros((nposes, max_pairs), dtype=bool)
    for i in range(nposes):
        m = len(lig_lists[i])
        if m == 0:
            continue
        pair_lig[i, :m] = np.asarray(lig_lists[i], dtype=np.int32)
        pair_rec[i, :m] = np.asarray(rec_lists[i], dtype=np.int32)
        pair_mask[i, :m] = True
    return pair_lig, pair_rec, pair_mask


def build_nb_grad_fn(ctx):
    ff = ctx["ff"]
    plateaudissq = jnp.float64(ctx["plateaudissq"])
    lig_pivot = jnp.array(ctx["lig_pivot"], dtype=jnp.float64)
    lig_coords = jnp.array(ctx["lig_coords"], dtype=jnp.float64)
    lig_atomtypes = jnp.array(ctx["lig_atomtypes_ff"], dtype=jnp.int32)
    lig_charge = jnp.array(ctx["lig_charge_scaled"], dtype=jnp.float64)
    rec_coords_ens = jnp.array(ctx["rec_coords_ens"], dtype=jnp.float64)
    rec_atomtypes = jnp.array(ctx["rec_atomtypes_ff"], dtype=jnp.int32)
    rec_charge_ens = jnp.array(ctx["rec_charge_ens_scaled"], dtype=jnp.float64)
    cdie = bool(ctx["cdie"])
    nonbon_vec = jnp.vectorize(nonbon, excluded=(1,), signature="(),(),()->()")

    @jax.jit
    def _single_nb_energy(dof, ens0, pair_lig, pair_rec, pair_mask):
        mats = dofs_to_mats(dof[None, :], lig_pivot)
        all_coors = jnp.einsum(
            "jk,bkl->bjl",
            jnp.concatenate((lig_coords, jnp.ones((lig_coords.shape[0], 1), dtype=lig_coords.dtype)), axis=1),
            mats,
        )[:, :, :3]
        lig_xyz = all_coors[0, pair_lig]
        rec_xyz = rec_coords_ens[ens0, pair_rec]
        d = lig_xyz - rec_xyz
        dsq = (d * d).sum(axis=1)

        valid = pair_mask & (dsq < plateaudissq)
        dsq_safe = jnp.where(valid, dsq, 1.0)

        at1 = rec_atomtypes[pair_rec]
        at2 = lig_atomtypes[pair_lig]

        charge = rec_charge_ens[ens0, pair_rec] * lig_charge[pair_lig]
        charge = jnp.where(jnp.abs(charge) > 1.0e-3, charge, 0.0)

        e_nb = nonbon_vec(dsq_safe, ff, at1, at2) - nonbon_vec(plateaudissq, ff, at1, at2)
        if cdie:
            e_el = elec_dsq_cdie(dsq_safe, charge) - elec_dsq_cdie(plateaudissq, charge)
        else:
            e_el = elec_dsq_const(dsq_safe, charge) - elec_dsq_const(plateaudissq, charge)
        e = jnp.where(valid, e_nb + e_el, 0.0)
        return e.sum()

    vg_single = jax.value_and_grad(_single_nb_energy)
    return jax.jit(jax.vmap(vg_single, in_axes=(0, 0, 0, 0, 0)))


def compute_nb_scores(ctx, ens, dofs, pair_lig, pair_rec, pair_mask, batch_size=128):
    vg_batch = build_nb_grad_fn(ctx)

    ens0 = np.asarray(ens, dtype=np.int32) - 1
    n = len(dofs)
    e = np.zeros(n, dtype=np.float64)
    g = np.zeros((n, 6), dtype=np.float64)

    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        ee, gg = vg_batch(
            jnp.array(dofs[start:end], dtype=jnp.float64),
            jnp.array(ens0[start:end], dtype=jnp.int32),
            jnp.array(pair_lig[start:end], dtype=jnp.int32),
            jnp.array(pair_rec[start:end], dtype=jnp.int32),
            jnp.array(pair_mask[start:end], dtype=bool),
        )
        ee.block_until_ready()
        e[start:end] = np.asarray(ee)
        g[start:end] = np.asarray(gg)

    return e, g
