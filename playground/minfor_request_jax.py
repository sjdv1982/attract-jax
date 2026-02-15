#!/usr/bin/env python3
"""Request/continuation ATTRACT-style minimizer on top of ATTRACT-JAX scoring.

This ports the legacy minfor control flow into an explicit state machine:
- multiple continuation entry points (instead of callback-driven control flow)
- explicit energy requests carrying full minimizer state context
- batched request processing, binned by energy call signature (ensemble id)

Current scope is the test docking setup (single movable ligand body, 6 DOFs):
phi, ssi, rot, tx, ty, tz.
"""

import os

# Keep host memory bounded on CPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import math
import re
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple

from reproduce_grid_score import (
    build_kernel,
    parse_legacy_score,
    parse_reduced_pdb,
    read_grid_with_electro,
)

jax.config.update("jax_enable_x64", True)

STRUCT_RE = re.compile(r"^#\d+\s*$")
PIVOT_RE = re.compile(
    r"^#pivot\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)
ENERGY_RE = re.compile(r"^##\s*Energy:\s*([-+0-9.eE]+)\s*$")

ENTRY_INIT = 0
ENTRY_AFTER_INIT = 1
ENTRY_ITER = 2
ENTRY_AFTER_LINE = 3
ENTRY_FINAL_REQUEST = 4
ENTRY_AFTER_FINAL = 5
ENTRY_DONE = 6
ENTRY_WAIT = 99

REQ_NONE = -1
REQ_INIT = 0
REQ_LINE = 1
REQ_FINAL = 2

JN = 6
ACC = 1.0e-9


@dataclass
class MinforState:
    ens: np.ndarray
    entry: np.ndarray
    x: np.ndarray
    g: np.ndarray
    xaa: np.ndarray
    xbb: np.ndarray
    ga: np.ndarray
    gb: np.ndarray
    d: np.ndarray
    bmat: np.ndarray
    hpack: np.ndarray
    w: np.ndarray
    gesa: np.ndarray
    fa: np.ndarray
    fb: np.ndarray
    fmin: np.ndarray
    gmin: np.ndarray
    dff: np.ndarray
    dga: np.ndarray
    dgb: np.ndarray
    step: np.ndarray
    stepbd: np.ndarray
    steplb: np.ndarray
    stmin: np.ndarray
    c: np.ndarray
    isfv: np.ndarray
    itr: np.ndarray
    nfun: np.ndarray
    req_kind: np.ndarray
    req_cont: np.ndarray
    req_x: np.ndarray
    resp_energy: np.ndarray
    resp_grad: np.ndarray


def parse_dat_two_body(path: str, max_poses: int = 0):
    header: List[str] = []
    pivots: Dict[int, np.ndarray] = {}
    ens_list: List[int] = []
    dof_list: List[Tuple[float, float, float, float, float, float]] = []
    energy_list: List[float] = []
    centered_ligands: Optional[bool] = None

    current_lines: List[List[float]] = []
    current_energy: Optional[float] = None
    seen_first_struct = False

    def flush():
        nonlocal current_lines, current_energy
        if not current_lines:
            return
        if len(current_lines) < 2:
            raise ValueError("Expected at least two numeric DOF lines per structure")
        first = current_lines[0]
        second = current_lines[-1]
        if len(first) != 7:
            raise ValueError(f"Expected first body line with 7 fields, got {len(first)}")
        ens = int(round(first[0]))
        if len(second) == 7:
            second = second[1:]
        if len(second) != 6:
            raise ValueError(f"Expected ligand DOF line with 6 fields, got {len(second)}")
        ens_list.append(ens)
        dof_list.append(tuple(float(v) for v in second))
        energy_list.append(float("nan") if current_energy is None else float(current_energy))
        current_lines = []
        current_energy = None

    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")

            pm = PIVOT_RE.match(line)
            if pm:
                pivots[int(pm.group(1))] = np.array(
                    [float(pm.group(2)), float(pm.group(3)), float(pm.group(4))],
                    dtype=np.float64,
                )

            em = ENERGY_RE.match(line)
            if em:
                current_energy = float(em.group(1))
                continue

            if STRUCT_RE.match(line):
                if max_poses and len(ens_list) >= max_poses:
                    break
                seen_first_struct = True
                flush()
                continue

            if not seen_first_struct:
                header.append(raw)
                low = line.strip().lower()
                if low.startswith("#centered ligands:"):
                    if "true" in low:
                        centered_ligands = True
                    elif "false" in low:
                        centered_ligands = False

            parts = line.strip().split()
            if not parts:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) in (6, 7):
                current_lines.append(vals)

    if not (max_poses and len(ens_list) >= max_poses):
        flush()

    if not ens_list:
        raise ValueError(f"No structures parsed from {path}")

    return (
        header,
        pivots,
        np.asarray(ens_list, dtype=np.int32),
        np.asarray(dof_list, dtype=np.float64),
        np.asarray(energy_list, dtype=np.float64),
        centered_ligands,
    )


def euler2rotmat(phi: np.ndarray, ssi: np.ndarray, rot: np.ndarray) -> np.ndarray:
    cs = np.cos(ssi)
    cp = np.cos(phi)
    ss = np.sin(ssi)
    sp = np.sin(phi)
    cscp = cs * cp
    cssp = cs * sp
    sscp = ss * cp
    sssp = ss * sp
    crot = np.cos(rot)
    srot = np.sin(rot)

    out = np.zeros((len(phi), 3, 3), dtype=np.float64)
    out[:, 0, 0] = crot * cscp + srot * sp
    out[:, 0, 1] = srot * cscp - crot * sp
    out[:, 0, 2] = sscp
    out[:, 1, 0] = crot * cssp - srot * cp
    out[:, 1, 1] = srot * cssp + crot * cp
    out[:, 1, 2] = sssp
    out[:, 2, 0] = -crot * ss
    out[:, 2, 1] = -srot * ss
    out[:, 2, 2] = cs
    return out


def dofs_to_mats_np(dofs: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    rot_col = euler2rotmat(dofs[:, 0], dofs[:, 1], dofs[:, 2])
    rot_row = np.swapaxes(rot_col, 1, 2)
    pivot_rot = np.einsum("j,bji->bi", pivot, rot_row)
    trans = dofs[:, 3:6] + pivot[None, :] - pivot_rot
    mats = np.zeros((len(dofs), 4, 4), dtype=np.float64)
    mats[:, :3, :3] = rot_row
    mats[:, 3, :3] = trans
    mats[:, 3, 3] = 1.0
    return mats


def write_dat_two_body(path: str, header: List[str], ens: np.ndarray, dofs: np.ndarray):
    with open(path, "w") as f:
        for line in header:
            f.write(line)
        for i in range(len(dofs)):
            f.write(f"#{i+1}\n")
            f.write(f"{int(ens[i]):12d}{0:12d}{0:12d}{0:12d}{0:12d}{0:12d}{0:12d}\n")
            phi, ssi, rot, xa, ya, za = dofs[i]
            f.write(
                f"{phi:24.16f} {ssi:24.16f} {rot:24.16f} "
                f"{xa:24.16f} {ya:24.16f} {za:24.16f}\n"
            )


def summarize(name: str, ref: np.ndarray, cand: np.ndarray):
    d = cand - ref
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    corr = float(np.corrcoef(ref, cand)[0, 1]) if len(ref) > 1 else float("nan")
    p50 = float(np.percentile(d, 50))
    p90 = float(np.percentile(d, 90))
    p99 = float(np.percentile(d, 99))
    print(
        f"{name}: mae={mae:.6f} rmse={rmse:.6f} pearson={corr:.6f} "
        f"delta[p50,p90,p99]=[{p50:.6f},{p90:.6f},{p99:.6f}]"
    )


class GridEnergyModel:
    def __init__(
        self,
        receptor_ens_list: str,
        ligand_pdb: str,
        grid_file: str,
        attract_par_npz: str,
        epsilon: float,
        cdie: bool,
        lig_pivot: np.ndarray,
        energy_batch: int,
    ):
        self.energy_batch = int(max(1, energy_batch))

        with open(receptor_ens_list) as f:
            rec_files = [line.strip() for line in f if line.strip()]
        if not rec_files:
            raise ValueError("Empty receptor ensemble list")

        list_dir = Path(receptor_ens_list).resolve().parent
        rec_coords_all = []
        rec_types_all = []
        rec_charge_all = []
        rec_weight_all = []
        for rf in rec_files:
            p = Path(rf)
            if not p.is_absolute():
                p = list_dir / p
            c, a, q, w = parse_reduced_pdb(str(p))
            rec_coords_all.append(c)
            rec_types_all.append(a)
            rec_charge_all.append(q)
            rec_weight_all.append(w)

        lig_coords0, lig_types0, lig_charge0, lig_weight0 = parse_reduced_pdb(ligand_pdb)

        if not np.allclose(lig_weight0, 1.0):
            print("Warning: ligand weights are not all 1.0 in test input .pdb")
        if not np.allclose(rec_weight_all[0], 1.0):
            print("Warning: receptor weights are not all 1.0 in test input .pdb")

        rec_mask = rec_types_all[0] != 99
        lig_mask = lig_types0 != 99

        rec_types = rec_types_all[0][rec_mask]
        for i in range(1, len(rec_types_all)):
            if not np.array_equal(rec_types_all[i][rec_mask], rec_types):
                raise ValueError(f"Receptor atom types differ in ensemble {i+1}")

        lig_types = lig_types0[lig_mask]
        rec_coords_ens = np.asarray([c[rec_mask] for c in rec_coords_all], dtype=np.float64)
        rec_charge_ens_raw = np.asarray([q[rec_mask] for q in rec_charge_all], dtype=np.float64)
        lig_coords = lig_coords0[lig_mask].astype(np.float64)
        lig_charge_raw = lig_charge0[lig_mask].astype(np.float64)

        felec = math.sqrt(332.053986 / epsilon)
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
        FF = namedtuple("FF", field_names=("rc", "ac", "ivor", "emin", "rmin2"))
        ff = FF(
            jnp.array(rc, dtype=jnp.float64),
            jnp.array(ac, dtype=jnp.float64),
            jnp.array(ivor, dtype=jnp.float64),
            jnp.array(emin, dtype=jnp.float64),
            jnp.array(rmin2, dtype=jnp.float64),
        )

        grid = read_grid_with_electro(Path(grid_file).read_bytes())
        rec_mapping = np.cumsum(rec_mask) - 1
        nb_flat = grid.neighbour_grid.reshape(-1)
        valid = nb_flat < 2**16 - 1
        nb_flat[valid] = rec_mapping[nb_flat[valid]]

        alpos = grid.alphabet_atomtypes.tolist()
        lig_vdw_channel_idx = np.array([alpos.index(a) for a in lig_alphabet], dtype=np.int32)[lig_atomtypes_ff]

        inner_all = np.concatenate((grid.inner_potential_grid, grid.inner_elec_grid[None, ...]), axis=0)
        outer_all = np.concatenate((grid.outer_potential_grid, grid.outer_elec_grid[None, ...]), axis=0)
        elec_channel_index = inner_all.shape[0] - 1

        dgrid = {}
        for field in grid._fields:
            value = getattr(grid, field)
            if isinstance(value, np.ndarray) and value.shape != (3,):
                value = jnp.array(value, dtype=value.dtype)
            dgrid[field] = value
        dgrid["inner_potential_grid_all"] = jnp.array(inner_all, dtype=np.float32)
        dgrid["outer_potential_grid_all"] = jnp.array(outer_all, dtype=np.float32)
        dgrid["elec_channel_index"] = np.int32(elec_channel_index)
        dgrid["neighbour_grid_ravel"] = dgrid["neighbour_grid"].reshape(-1, grid.neighbour_grid.shape[-1])
        dgrid["neighbour_type_grid_ravel"] = dgrid["neighbour_type_grid"].reshape(
            -1, grid.neighbour_type_grid.shape[-1]
        )
        GridJax = namedtuple("GridJax", tuple(dgrid.keys()))
        grid_j = GridJax(**dgrid)

        nb_chunk_thresholds = (0, 1, 2, 3, 4, 5, 10, 15, 20)
        for n0 in range(nb_chunk_thresholds[-1] + 10, int(grid.max_nr_neighbours), 10):
            nb_chunk_thresholds += (n0,)
        nb_chunk_thresholds += (int(grid.max_nr_neighbours),)
        grid_dim = tuple(int(x) for x in grid.dim)

        kernel_main = build_kernel(
            grid=grid_j,
            ff=ff,
            lig_atomtypes_ff=jnp.array(lig_atomtypes_ff, dtype=np.int32),
            lig_vdw_channel_idx=jnp.array(lig_vdw_channel_idx, dtype=np.int32),
            lig_charge_raw=jnp.array(lig_charge_raw, dtype=np.float64),
            lig_charge_scaled=jnp.array(lig_charge_scaled, dtype=np.float64),
            cdie=bool(cdie),
        )
        grad_main = jax.grad(lambda dof_batch, *rest: kernel_main(dof_batch, *rest)[0])

        self.kernel_main = kernel_main
        self.grad_main = grad_main
        self.rec_coor_ens = rec_coords_ens
        self.rec_charge_ens_scaled = rec_charge_ens_scaled
        self.rec_atomtypes_ff_j = jnp.array(rec_atomtypes_ff, dtype=np.int32)
        self.coor_lig_j = jnp.array(lig_coords, dtype=jnp.float64)
        self.lig_atomtypes_ff_j = jnp.array(lig_atomtypes_ff, dtype=np.int32)
        self.lig_vdw_channel_idx_j = jnp.array(lig_vdw_channel_idx, dtype=np.int32)
        self.lig_charge_raw_j = jnp.array(lig_charge_raw, dtype=jnp.float64)
        self.lig_charge_scaled_j = jnp.array(lig_charge_scaled, dtype=jnp.float64)
        self.ff = ff
        self.grid_j = grid_j
        self.nb_chunk_thresholds = nb_chunk_thresholds
        self.grid_dim = grid_dim
        self.lig_pivot_j = jnp.array(lig_pivot, dtype=jnp.float64)

    def eval(self, ens: np.ndarray, dofs: np.ndarray):
        n = len(dofs)
        e = np.zeros(n, dtype=np.float64)
        g = np.zeros((n, JN), dtype=np.float64)
        uniq = np.unique(ens)

        for ens_id in uniq:
            ens0 = int(ens_id) - 1
            if ens0 < 0 or ens0 >= len(self.rec_coor_ens):
                raise ValueError(f"Ensemble index out of range: {ens_id}")
            idx = np.where(ens == ens_id)[0]
            rec_coor_j = jnp.array(self.rec_coor_ens[ens0], dtype=jnp.float64)
            rec_charge_j = jnp.array(self.rec_charge_ens_scaled[ens0], dtype=jnp.float64)

            for start in range(0, len(idx), self.energy_batch):
                sub = idx[start : start + self.energy_batch]
                dof_b = jnp.array(dofs[sub], dtype=jnp.float64)
                _, ene_b = self.kernel_main(
                    dof_b,
                    rec_coor_j,
                    self.rec_atomtypes_ff_j,
                    rec_charge_j,
                    self.coor_lig_j,
                    self.lig_atomtypes_ff_j,
                    self.lig_vdw_channel_idx_j,
                    self.lig_charge_raw_j,
                    self.lig_charge_scaled_j,
                    self.ff,
                    self.grid_j,
                    self.nb_chunk_thresholds,
                    self.grid_dim,
                    self.lig_pivot_j,
                )
                grad_b = self.grad_main(
                    dof_b,
                    rec_coor_j,
                    self.rec_atomtypes_ff_j,
                    rec_charge_j,
                    self.coor_lig_j,
                    self.lig_atomtypes_ff_j,
                    self.lig_vdw_channel_idx_j,
                    self.lig_charge_raw_j,
                    self.lig_charge_scaled_j,
                    self.ff,
                    self.grid_j,
                    self.nb_chunk_thresholds,
                    self.grid_dim,
                    self.lig_pivot_j,
                )
                e[sub] = np.asarray(ene_b)
                g[sub] = np.asarray(grad_b)

        return e, g


def issue_request(st: MinforState, i: int, kind: int, cont: int, xreq: np.ndarray):
    st.req_kind[i] = kind
    st.req_cont[i] = cont
    st.req_x[i] = xreq
    st.entry[i] = ENTRY_WAIT


def solve_direction(bmat: np.ndarray, ga: np.ndarray) -> np.ndarray:
    rhs = -ga
    try:
        return np.linalg.solve(bmat, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(bmat, rhs, rcond=None)[0]


def init_hpack_diag(diag_value: float = 0.01) -> np.ndarray:
    h = np.zeros(21, dtype=np.float64)
    # Fortran packed layout diagonal slots for n=6: 1,7,12,16,19,21 (1-based)
    h[[0, 6, 11, 15, 18, 20]] = diag_value
    return h


def mc11e_packed(a_pack: np.ndarray, z: np.ndarray, w: np.ndarray, ir: int, n: int = JN):
    if ir < n:
        return z.copy(), w.copy()

    a = np.zeros(len(a_pack) + 1, dtype=np.float64)
    a[1:] = a_pack
    zf = np.zeros(n + 1, dtype=np.float64)
    zf[1:] = z
    wf = np.zeros(n + 1, dtype=np.float64)
    wf[1:] = w

    wf[1] = zf[1]
    if n <= 1:
        zf[1] = zf[1] / a[1]
        return zf[1:], wf[1:]

    ij = 1
    for i in range(2, n + 1):
        ij = i
        v = zf[i]
        for j in range(1, i):
            v = v - a[ij] * zf[j]
            ij = ij + n - j
        wf[i] = v
        zf[i] = v

    zf[n] = zf[n] / a[ij]
    np1 = n + 1
    for nip in range(2, n + 1):
        i = np1 - nip
        ii = ij - nip
        v = zf[i] / a[ii]
        ip = i + 1
        ij = ii
        for j in range(ip, n + 1):
            ii = ii + 1
            v = v - a[ii] * zf[j]
        zf[i] = v

    return zf[1:], wf[1:]


def mc11a_packed(
    a_pack: np.ndarray,
    z: np.ndarray,
    sig: float,
    w: np.ndarray,
    ir: int,
    mk: int,
    eps: float,
    n: int = JN,
    alias_zw: bool = False,
):
    a = np.zeros(len(a_pack) + 1, dtype=np.float64)
    a[1:] = a_pack
    zf = np.zeros(n + 1, dtype=np.float64)
    zf[1:] = z
    if alias_zw:
        wf = zf
    else:
        wf = np.zeros(n + 1, dtype=np.float64)
        wf[1:] = w

    if n <= 1:
        a[1] = a[1] + sig * zf[1] ** 2
        ir = 1
        if a[1] <= 0.0:
            a[1] = 0.0
            ir = 0
        zout = zf[1:].copy()
        wout = zout.copy() if alias_zw else wf[1:].copy()
        return a[1:].copy(), zout, wout, ir

    np1 = n + 1
    if sig > 0.0:
        mm = 0
        tim = 1.0 / sig
    else:
        if sig == 0.0 or ir == 0:
            zout = zf[1:].copy()
            wout = zout.copy() if alias_zw else wf[1:].copy()
            return a[1:].copy(), zout, wout, ir

        ti = 1.0 / sig
        ij = 1
        if mk != 0:
            for i in range(1, n + 1):
                if a[ij] != 0.0:
                    ti = ti + wf[i] ** 2 / a[ij]
                ij = ij + np1 - i
        else:
            for i in range(1, n + 1):
                wf[i] = zf[i]
            for i in range(1, n + 1):
                ip = i + 1
                v = wf[i]
                if a[ij] <= 0.0:
                    wf[i] = 0.0
                    ij = ij + np1 - i
                    continue
                ti = ti + v * v / a[ij]
                if i != n:
                    for j in range(ip, n + 1):
                        ij = ij + 1
                        wf[j] = wf[j] - v * a[ij]
                ij = ij + 1

        goto40 = False
        if ir > 0:
            if ti > 0.0:
                ti = eps / sig
                if eps == 0.0:
                    ir = ir - 1
            else:
                if mk <= 1:
                    goto40 = True
                else:
                    ti = 0.0
                    ir = -ir - 1
        else:
            ti = 0.0
            ir = -ir - 1

        if goto40:
            mm = 0
            tim = 1.0 / sig
        else:
            mm = 1
            tim = ti
            for i in range(1, n + 1):
                j = np1 - i
                ij = ij - i
                if a[ij] != 0.0:
                    tim = ti - wf[j] ** 2 / a[ij]
                wf[j] = ti
                ti = tim

    ij = 1
    for i in range(1, n + 1):
        ip = i + 1
        v = zf[i]
        if a[ij] <= 0.0:
            if ir > 0 or sig < 0.0 or v == 0.0:
                ti = tim
                ij = ij + np1 - i
                continue
            ir = 1 - ir
            a[ij] = v * v / tim
            if i == n:
                break
            for j in range(ip, n + 1):
                ij = ij + 1
                a[ij] = zf[j] / v
            break

        al = v / a[ij]
        if mm != 0:
            ti = wf[i]
        else:
            ti = tim + v * al
        r = ti / tim
        a[ij] = a[ij] * r
        if r == 0.0 or i == n:
            break

        b = al / ti
        if r <= 4.0:
            for j in range(ip, n + 1):
                ij = ij + 1
                zf[j] = zf[j] - v * a[ij]
                a[ij] = a[ij] + b * zf[j]
        else:
            gm = tim / ti
            for j in range(ip, n + 1):
                ij = ij + 1
                yv = a[ij]
                a[ij] = b * zf[j] + yv * gm
                zf[j] = zf[j] - v * yv
        tim = ti
        ij = ij + 1

    if ir < 0:
        ir = -ir
    zout = zf[1:].copy()
    wout = zout.copy() if alias_zw else wf[1:].copy()
    return a[1:].copy(), zout, wout, ir


def entry_init(st: MinforState, idx: np.ndarray):
    for i in idx:
        issue_request(st, int(i), REQ_INIT, ENTRY_AFTER_INIT, st.xaa[i])


def entry_after_init(st: MinforState, idx: np.ndarray):
    for i in idx:
        ii = int(i)
        st.gesa[ii] = st.resp_energy[ii]
        st.g[ii] = st.resp_grad[ii]
        st.entry[ii] = ENTRY_ITER


def request_next_line_or_finalize(st: MinforState, i: int, vmax: int):
    if st.nfun[i] >= vmax:
        st.entry[i] = ENTRY_FINAL_REQUEST
        return
    st.nfun[i] += 1
    st.c[i] = st.stmin[i] + st.step[i]
    st.xbb[i] = st.xaa[i] + st.c[i] * st.d[i]
    issue_request(st, i, REQ_LINE, ENTRY_AFTER_LINE, st.xbb[i])


def entry_iter(st: MinforState, idx: np.ndarray, vmax: int, update_rule: str):
    for i in idx:
        ii = int(i)
        st.xaa[ii] = st.x[ii]
        st.fa[ii] = st.gesa[ii]
        st.isfv[ii] = 1
        st.ga[ii] = st.g[ii]
        st.itr[ii] += 1

        if update_rule == "mc11":
            d0 = -st.ga[ii]
            d1, w1 = mc11e_packed(st.hpack[ii], d0, st.w[ii], ir=JN, n=JN)
            st.d[ii] = d1
            st.w[ii] = w1
        elif update_rule == "bfgs_inv":
            st.d[ii] = -st.bmat[ii] @ st.ga[ii]
        else:
            st.d[ii] = solve_direction(st.bmat[ii], st.ga[ii])
        cmax = float(np.max(np.abs(st.d[ii])))
        st.dga[ii] = float(np.dot(st.ga[ii], st.d[ii]))

        if cmax <= 0.0 or st.dga[ii] >= 0.0:
            st.entry[ii] = ENTRY_FINAL_REQUEST
            continue

        st.stmin[ii] = 0.0
        st.stepbd[ii] = 0.0
        st.steplb[ii] = ACC / cmax
        st.fmin[ii] = st.fa[ii]
        st.gmin[ii] = st.dga[ii]

        step = 1.0
        if st.dff[ii] <= 0.0:
            step = min(step, 1.0 / cmax)
        else:
            step = min(step, (2.0 * st.dff[ii]) / (-st.dga[ii]))
        st.step[ii] = step

        request_next_line_or_finalize(st, ii, vmax)


def entry_after_line(st: MinforState, idx: np.ndarray, vmax: int, update_rule: str):
    for i in idx:
        ii = int(i)
        st.fb[ii] = st.resp_energy[ii]
        st.gb[ii] = st.resp_grad[ii]

        st.isfv[ii] = min(2, st.isfv[ii])
        if st.fb[ii] <= st.gesa[ii]:
            better = st.fb[ii] < st.gesa[ii]
            if not better:
                gl1 = float(np.dot(st.g[ii], st.g[ii]))
                gl2 = float(np.dot(st.gb[ii], st.gb[ii]))
                better = gl2 < gl1
            if better:
                st.isfv[ii] = 3
                st.gesa[ii] = st.fb[ii]
                st.x[ii] = st.xbb[ii]
                st.g[ii] = st.gb[ii]

        st.dgb[ii] = float(np.dot(st.gb[ii], st.d[ii]))

        if st.fb[ii] - st.fa[ii] <= 0.1 * st.c[ii] * st.dga[ii]:
            st.stepbd[ii] = st.stepbd[ii] - st.step[ii]
            st.stmin[ii] = st.c[ii]
            st.fmin[ii] = st.fb[ii]
            st.gmin[ii] = st.dgb[ii]

            st.step[ii] = 9.0 * st.stmin[ii]
            if st.stepbd[ii] > 0.0:
                st.step[ii] = 0.5 * st.stepbd[ii]

            ctmp = st.dga[ii] + 3.0 * st.dgb[ii] - 4.0 * (st.fb[ii] - st.fa[ii]) / st.stmin[ii]
            if ctmp > 0.0:
                st.step[ii] = min(st.step[ii], st.stmin[ii] * max(1.0, -st.dgb[ii] / ctmp))

            if st.dgb[ii] < 0.7 * st.dga[ii]:
                request_next_line_or_finalize(st, ii, vmax)
                continue

            st.isfv[ii] = 4 - st.isfv[ii]
            if st.stmin[ii] + st.step[ii] <= st.steplb[ii]:
                if st.isfv[ii] >= 2:
                    st.entry[ii] = ENTRY_ITER
                else:
                    st.entry[ii] = ENTRY_FINAL_REQUEST
                continue

            ga_old = st.ga[ii].copy()
            y = st.gb[ii] - ga_old

            if update_rule == "mc11":
                denom1 = st.dga[ii]
                denom2 = st.stmin[ii] * (st.dgb[ii] - st.dga[ii])
                if abs(denom1) < 1.0e-16 or abs(denom2) < 1.0e-16:
                    st.entry[ii] = ENTRY_FINAL_REQUEST
                    continue
                h1, _, w1, ir1 = mc11a_packed(
                    st.hpack[ii], ga_old, 1.0 / denom1, st.w[ii], -JN, 1, 0.0, n=JN, alias_zw=False
                )
                h2, _, _, ir2 = mc11a_packed(
                    h1, y, 1.0 / denom2, y.copy(), -ir1, 0, 0.0, n=JN, alias_zw=True
                )
                if ir2 < JN:
                    st.entry[ii] = ENTRY_FINAL_REQUEST
                    continue
                st.hpack[ii] = h2
                st.w[ii] = w1
            elif update_rule == "bfgs_inv":
                s = st.stmin[ii] * st.d[ii]
                ys = float(np.dot(y, s))
                if ys <= 1.0e-16:
                    st.entry[ii] = ENTRY_FINAL_REQUEST
                    continue
                rho = 1.0 / ys
                eye = np.eye(JN, dtype=np.float64)
                v = eye - rho * np.outer(s, y)
                hnew = v @ st.bmat[ii] @ v.T + rho * np.outer(s, s)
                hnew = 0.5 * (hnew + hnew.T)
                if not np.all(np.isfinite(hnew)):
                    st.entry[ii] = ENTRY_FINAL_REQUEST
                    continue
                try:
                    np.linalg.cholesky(hnew)
                except np.linalg.LinAlgError:
                    st.entry[ii] = ENTRY_FINAL_REQUEST
                    continue
                st.bmat[ii] = hnew
            else:
                denom1 = st.dga[ii]
                denom2 = st.stmin[ii] * (st.dgb[ii] - st.dga[ii])
                if abs(denom1) < 1.0e-16 or abs(denom2) < 1.0e-16:
                    st.entry[ii] = ENTRY_FINAL_REQUEST
                    continue

                bnew = st.bmat[ii] + np.outer(ga_old, ga_old) / denom1 + np.outer(y, y) / denom2
                bnew = 0.5 * (bnew + bnew.T)
                try:
                    np.linalg.cholesky(bnew)
                except np.linalg.LinAlgError:
                    st.entry[ii] = ENTRY_FINAL_REQUEST
                    continue
                st.bmat[ii] = bnew

            st.dff[ii] = st.fa[ii] - st.fb[ii]
            st.fa[ii] = st.fb[ii]
            st.entry[ii] = ENTRY_ITER
            continue

        if st.step[ii] > st.steplb[ii]:
            st.stepbd[ii] = st.step[ii]
            ctmp = st.gmin[ii] + st.dgb[ii] - 3.0 * (st.fb[ii] - st.fmin[ii]) / st.step[ii]
            disc = ctmp * ctmp - st.gmin[ii] * st.dgb[ii]
            if disc < 0.0:
                disc = 0.0
            denom = ctmp + st.gmin[ii] - math.sqrt(disc)
            if abs(denom) < 1.0e-16:
                fac = 0.1
            else:
                fac = max(0.1, st.gmin[ii] / denom)
            st.step[ii] = st.step[ii] * fac
            request_next_line_or_finalize(st, ii, vmax)
            continue

        if st.isfv[ii] >= 2:
            st.entry[ii] = ENTRY_ITER
        else:
            st.entry[ii] = ENTRY_FINAL_REQUEST


def entry_final_request(st: MinforState, idx: np.ndarray, vmax: int):
    for i in idx:
        ii = int(i)
        if st.nfun[ii] < vmax:
            st.nfun[ii] += 1
        issue_request(st, ii, REQ_FINAL, ENTRY_AFTER_FINAL, st.x[ii])


def entry_after_final(st: MinforState, idx: np.ndarray):
    for i in idx:
        ii = int(i)
        st.gesa[ii] = st.resp_energy[ii]
        st.g[ii] = st.resp_grad[ii]
        st.entry[ii] = ENTRY_DONE


def init_states(ens: np.ndarray, dofs0: np.ndarray, init_metric: float) -> MinforState:
    n = len(dofs0)
    b = np.zeros((n, JN, JN), dtype=np.float64)
    for i in range(n):
        b[i] = np.eye(JN, dtype=np.float64) * init_metric

    hpack = np.zeros((n, 21), dtype=np.float64)
    for i in range(n):
        hpack[i] = init_hpack_diag(init_metric)
    w = np.zeros((n, JN), dtype=np.float64)

    return MinforState(
        ens=ens.copy(),
        entry=np.full(n, ENTRY_INIT, dtype=np.int32),
        x=dofs0.copy(),
        g=np.zeros((n, JN), dtype=np.float64),
        xaa=dofs0.copy(),
        xbb=np.zeros((n, JN), dtype=np.float64),
        ga=np.zeros((n, JN), dtype=np.float64),
        gb=np.zeros((n, JN), dtype=np.float64),
        d=np.zeros((n, JN), dtype=np.float64),
        bmat=b,
        hpack=hpack,
        w=w,
        gesa=np.zeros(n, dtype=np.float64),
        fa=np.zeros(n, dtype=np.float64),
        fb=np.zeros(n, dtype=np.float64),
        fmin=np.zeros(n, dtype=np.float64),
        gmin=np.zeros(n, dtype=np.float64),
        dff=np.zeros(n, dtype=np.float64),
        dga=np.zeros(n, dtype=np.float64),
        dgb=np.zeros(n, dtype=np.float64),
        step=np.zeros(n, dtype=np.float64),
        stepbd=np.zeros(n, dtype=np.float64),
        steplb=np.zeros(n, dtype=np.float64),
        stmin=np.zeros(n, dtype=np.float64),
        c=np.zeros(n, dtype=np.float64),
        isfv=np.zeros(n, dtype=np.int32),
        itr=np.zeros(n, dtype=np.int32),
        nfun=np.zeros(n, dtype=np.int32),
        req_kind=np.full(n, REQ_NONE, dtype=np.int32),
        req_cont=np.full(n, -1, dtype=np.int32),
        req_x=np.zeros((n, JN), dtype=np.float64),
        resp_energy=np.zeros(n, dtype=np.float64),
        resp_grad=np.zeros((n, JN), dtype=np.float64),
    )


def run_state_machine(
    model: GridEnergyModel,
    ens: np.ndarray,
    dofs0: np.ndarray,
    vmax: int,
    max_cycles: int,
    trace_every: int,
    grad_sign: float,
    update_rule: str,
    init_metric: float,
    trace_requests: bool,
    trace_pose_1based: int,
):
    st = init_states(ens, dofs0, init_metric=init_metric)

    entry_handlers = {
        ENTRY_INIT: lambda idx: entry_init(st, idx),
        ENTRY_AFTER_INIT: lambda idx: entry_after_init(st, idx),
        ENTRY_ITER: lambda idx: entry_iter(st, idx, vmax, update_rule),
        ENTRY_AFTER_LINE: lambda idx: entry_after_line(st, idx, vmax, update_rule),
        ENTRY_FINAL_REQUEST: lambda idx: entry_final_request(st, idx, vmax),
        ENTRY_AFTER_FINAL: lambda idx: entry_after_final(st, idx),
    }

    for cycle in range(max_cycles):
        active = np.where((st.entry != ENTRY_DONE) & (st.entry != ENTRY_WAIT))[0]
        if len(active) == 0:
            pending = np.where(st.req_kind != REQ_NONE)[0]
            if len(pending) == 0:
                break

        if trace_every and (cycle % trace_every == 0):
            done = int(np.sum(st.entry == ENTRY_DONE))
            pending = int(np.sum(st.req_kind != REQ_NONE))
            print(f"cycle {cycle}: done={done}/{len(st.entry)} pending_requests={pending}")

        if len(active):
            for ep in np.unique(st.entry[active]):
                if ep in (ENTRY_DONE, ENTRY_WAIT):
                    continue
                idx = np.where(st.entry == ep)[0]
                entry_handlers[int(ep)](idx)

        req_idx = np.where(st.req_kind != REQ_NONE)[0]
        if len(req_idx) == 0:
            if np.all(st.entry == ENTRY_DONE):
                break
            continue

        trace_pose = int(trace_pose_1based) - 1
        if trace_requests and 0 <= trace_pose < len(st.entry) and st.req_kind[trace_pose] != REQ_NONE:
            xreq = st.req_x[trace_pose]
            print(
                "JAX_MINFOR_REQUEST PRE",
                cycle,
                int(st.req_kind[trace_pose]),
                int(st.req_cont[trace_pose]),
                int(st.nfun[trace_pose]),
                f"{xreq[0]:.6f}",
                f"{xreq[1]:.6f}",
                f"{xreq[2]:.6f}",
                f"{xreq[3]:.6f}",
                f"{xreq[4]:.6f}",
                f"{xreq[5]:.6f}",
            )

        req_kind = st.req_kind[req_idx]
        req_ens = st.ens[req_idx]
        req_x = st.req_x[req_idx]

        for rk in np.unique(req_kind):
            sel_kind = req_idx[req_kind == rk]
            for ens_id in np.unique(st.ens[sel_kind]):
                bin_idx = sel_kind[st.ens[sel_kind] == ens_id]
                e, g = model.eval(st.ens[bin_idx], st.req_x[bin_idx])
                st.resp_energy[bin_idx] = e
                st.resp_grad[bin_idx] = grad_sign * g

        if trace_requests and 0 <= trace_pose < len(st.entry) and st.req_kind[trace_pose] != REQ_NONE:
            gr = st.resp_grad[trace_pose]
            print(
                "JAX_MINFOR_REQUEST POST",
                cycle,
                int(st.req_kind[trace_pose]),
                f"{st.resp_energy[trace_pose]:.6f}",
                f"{gr[0]:.6f}",
                f"{gr[1]:.6f}",
                f"{gr[2]:.6f}",
                f"{gr[3]:.6f}",
                f"{gr[4]:.6f}",
                f"{gr[5]:.6f}",
            )

        cont = st.req_cont[req_idx]
        st.req_kind[req_idx] = REQ_NONE
        st.req_cont[req_idx] = -1
        st.entry[req_idx] = cont

    unfinished = np.where(st.entry != ENTRY_DONE)[0]
    if len(unfinished):
        print(f"Warning: {len(unfinished)} poses unfinished after {max_cycles} cycles")

    return st


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dat", help="starting .dat (e.g. systsearch-ens1.dat)")
    ap.add_argument("receptor_ens_list", help="ensemble list file")
    ap.add_argument("ligand_pdb", help="reduced ligand pdb (test input .pdb)")
    ap.add_argument("--grid", required=True, help="grid file (receptorgrid.grid)")
    ap.add_argument("--attract-par-npz", default="attract-par.npz")
    ap.add_argument("--epsilon", type=float, default=15.0)
    ap.add_argument("--cdie", action="store_true")
    ap.add_argument("--vmax", type=int, default=1000)
    ap.add_argument("--max-cycles", type=int, default=20000)
    ap.add_argument("--max-poses", type=int, default=0)
    ap.add_argument("--energy-batch", type=int, default=32)
    ap.add_argument("--memory-gb", type=float, default=20.0)
    ap.add_argument("--disable-jit", action="store_true")
    ap.add_argument("--trace-every", type=int, default=100)
    ap.add_argument("--trace-requests", action="store_true")
    ap.add_argument("--trace-pose", type=int, default=1, help="1-based pose index for --trace-requests")
    ap.add_argument(
        "--update-rule",
        choices=("legacy_rank2", "bfgs_inv", "mc11"),
        default="mc11",
        help="metric update: mc11 (legacy packed update), legacy_rank2 (dense approximation), bfgs_inv (inverse-Hessian BFGS)",
    )
    ap.add_argument(
        "--init-metric",
        type=float,
        default=0.01,
        help="initial diagonal metric value",
    )
    ap.add_argument(
        "--grad-sign",
        type=float,
        default=1.0,
        help="multiply JAX dE/dDOF by this factor before feeding minfor (try -1.0 for sign experiments)",
    )
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--legacy-dat", help="optional legacy minimized .dat for comparison")
    ap.add_argument("--legacy-score", help="optional legacy score file for comparison")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.memory_gb > 0:
        mem_bytes = int(args.memory_gb * (1024**3))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        print(f"Applied RLIMIT_AS={args.memory_gb:.2f} GB")

    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
        print("JAX JIT disabled (--disable-jit)")

    header, pivots, ens, dofs0, _, centered_ligands = parse_dat_two_body(
        args.input_dat, max_poses=args.max_poses
    )

    if 2 in pivots:
        lig_pivot = pivots[2]
    else:
        lig_coords0, _, _, _ = parse_reduced_pdb(args.ligand_pdb)
        lig_pivot = lig_coords0.mean(axis=0)
        print(
            "No explicit #pivot 2 in input .dat; "
            f"using ligand mean coordinate as pivot: {lig_pivot}"
        )

    input_centered = bool(centered_ligands) if centered_ligands is not None else False
    if input_centered:
        dofs0 = dofs0.copy()
        dofs0[:, 3:6] = dofs0[:, 3:6] - lig_pivot[None, :]
        print("Converted input centered-ligand translations to internal convention (tx/ty/tz -= pivot)")

    print(f"Poses: {len(dofs0)}")
    print(f"JAX devices: {jax.devices()}")

    model = GridEnergyModel(
        receptor_ens_list=args.receptor_ens_list,
        ligand_pdb=args.ligand_pdb,
        grid_file=args.grid,
        attract_par_npz=args.attract_par_npz,
        epsilon=args.epsilon,
        cdie=bool(args.cdie),
        lig_pivot=lig_pivot,
        energy_batch=args.energy_batch,
    )

    st = run_state_machine(
        model=model,
        ens=ens,
        dofs0=dofs0,
        vmax=int(args.vmax),
        max_cycles=int(args.max_cycles),
        trace_every=int(args.trace_every),
        grad_sign=float(args.grad_sign),
        update_rule=args.update_rule,
        init_metric=float(args.init_metric),
        trace_requests=bool(args.trace_requests),
        trace_pose_1based=int(args.trace_pose),
    )

    dofs_out = st.x.copy()
    if input_centered:
        dofs_out[:, 3:6] = dofs_out[:, 3:6] + lig_pivot[None, :]

    mats = dofs_to_mats_np(st.x, lig_pivot)
    np.save(args.out_prefix + ".dofs_internal.npy", st.x.astype(np.float32))
    np.save(args.out_prefix + ".dofs.npy", dofs_out.astype(np.float32))
    np.save(args.out_prefix + ".mat4.npy", mats.astype(np.float32))
    np.save(args.out_prefix + ".energy.npy", st.gesa.astype(np.float32))
    np.save(args.out_prefix + ".ens.npy", ens.astype(np.int32))

    out_dat = args.out_prefix + ".dat"
    write_dat_two_body(out_dat, header, ens, dofs_out)

    print(f"Saved: {args.out_prefix}.dofs_internal.npy")
    print(f"Saved: {args.out_prefix}.dofs.npy")
    print(f"Saved: {args.out_prefix}.mat4.npy")
    print(f"Saved: {args.out_prefix}.energy.npy")
    print(f"Saved: {args.out_prefix}.ens.npy")
    print(f"Saved: {out_dat}")

    if args.legacy_dat:
        _, piv_ref, ens_ref, dof_ref, e_ref, centered_ref = parse_dat_two_body(
            args.legacy_dat, max_poses=len(st.x)
        )
        dof_ref_internal = dof_ref.copy()
        if centered_ref:
            piv2 = piv_ref.get(2, lig_pivot)
            dof_ref_internal[:, 3:6] = dof_ref_internal[:, 3:6] - piv2[None, :]

        n = min(len(st.x), len(dof_ref))
        if not np.array_equal(ens[:n], ens_ref[:n]):
            print("Warning: ensemble order differs between candidate and legacy dat")
        summarize("dof_phi", dof_ref_internal[:n, 0], st.x[:n, 0])
        summarize("dof_ssi", dof_ref_internal[:n, 1], st.x[:n, 1])
        summarize("dof_rot", dof_ref_internal[:n, 2], st.x[:n, 2])
        summarize("dof_tx", dof_ref_internal[:n, 3], st.x[:n, 3])
        summarize("dof_ty", dof_ref_internal[:n, 4], st.x[:n, 4])
        summarize("dof_tz", dof_ref_internal[:n, 5], st.x[:n, 5])
        if np.isfinite(e_ref[:n]).all():
            summarize("energy_vs_legacy_dat", e_ref[:n], st.gesa[:n])

    if args.legacy_score:
        e_score, _ = parse_legacy_score(args.legacy_score, max_poses=len(st.x))
        n = min(len(st.x), len(e_score))
        summarize("energy_vs_legacy_score", e_score[:n], st.gesa[:n])


if __name__ == "__main__":
    main()
