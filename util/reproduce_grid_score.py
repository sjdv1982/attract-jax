#!/usr/bin/env python3
"""Reproduce legacy ATTRACT grid --score energies/gradients with JAX.

This script targets legacy score files like test/out_demo_xylanase.score:
- grid-based vdW + electrostatic potentials
- explicit neighbour-list correction inside plateaudis
- receptor ensemble selected per pose from the first DOF line
- ligand pose from the second DOF line (Euler + translation, pivot-aware)

It compares against legacy "Energy:" and "Gradients:" lines and can save
reference/candidate arrays as .npy.
"""

import os

# Avoid large up-front allocator reservations.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import re
import resource
import struct
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.lax import cond
from functools import partial

jax.config.update("jax_enable_x64", True)

STRUCT_RE = re.compile(r"^#\d+\s*$")
PIVOT_RE = re.compile(
    r"^#pivot\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)
ENERGY_RE = re.compile(r"^\s*Energy:\s*([-+0-9.eE]+)\s*$")
GRAD_RE = re.compile(r"^\s*Gradients:\s*(.*?)\s*$")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dat", help="input ATTRACT .dat with poses")
    ap.add_argument("legacy_score", help="legacy ATTRACT --score text output")
    ap.add_argument("receptor_ens_list", help="ensemble list file (1-based order)")
    ap.add_argument("ligand_pdb", help="reduced ligand pdb (e.g. ligandr.pdb)")
    ap.add_argument(
        "--grid", required=True, help="ATTRACT grid file (e.g. receptorgrid.grid)"
    )
    ap.add_argument(
        "--attract-par-npz", default="attract-par.npz", help="ATTRACT forcefield npz"
    )
    ap.add_argument("--epsilon", type=float, default=15.0, help="dielectric constant")
    ap.add_argument(
        "--cdie", action="store_true", help="use distance-dependent electrostatics"
    )
    ap.add_argument("--batch", type=int, default=256, help="poses per JAX batch")
    ap.add_argument(
        "--max-poses", type=int, default=0, help="cap number of poses (0 = all)"
    )
    ap.add_argument(
        "--memory-gb",
        type=float,
        default=20.0,
        help="address-space memory cap in GB (0 disables)",
    )
    ap.add_argument("--disable-jit", action="store_true", help="disable JAX JIT")
    ap.add_argument("--out-prefix", help="optional output prefix for .npy files")
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
                vals = [
                    float(tok.replace("D", "E").replace("d", "e"))
                    for tok in FLOAT_RE.findall(g.group(1))
                ]
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


def parse_dat_for_grid_score(path: str, max_poses: int = 0):
    pivots = {}
    ens: List[int] = []
    dofs: List[Tuple[float, float, float, float, float, float]] = []
    current_lines: List[List[float]] = []

    def flush():
        if not current_lines:
            return
        if len(current_lines) < 2:
            raise ValueError("Expected at least 2 DOF lines per structure")
        first = current_lines[0]
        second = current_lines[-1]
        if len(first) != 7:
            raise ValueError(
                f"Expected first DOF line to have 7 fields, got {len(first)}"
            )
        if len(second) not in (6, 7):
            raise ValueError(
                f"Expected ligand DOF line to have 6 or 7 fields, got {len(second)}"
            )
        ens_id = int(round(first[0]))
        if len(second) == 7:
            second = second[1:]
        phi, ssi, rot, xa, ya, za = second
        ens.append(ens_id)
        dofs.append((phi, ssi, rot, xa, ya, za))

    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            pm = PIVOT_RE.match(line)
            if pm:
                pid = int(pm.group(1))
                pivots[pid] = np.asarray(
                    [float(pm.group(2)), float(pm.group(3)), float(pm.group(4))],
                    dtype=np.float64,
                )
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
        raise ValueError("Could not parse ligand pivot (#pivot 2 ...)")
    return pivots, np.asarray(ens, dtype=np.int32), np.asarray(dofs, dtype=np.float64)


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


def read_grid_with_electro(grid_data: bytes):
    pos = 0

    def _r(token, size):
        nonlocal pos
        result = struct.unpack_from(token, grid_data, pos)
        pos += size
        return result

    def _rr(token):
        size = struct.calcsize(token)

        def func():
            result = _r(token, size)
            return result[0]

        return func

    read_bool = _rr("?")
    read_short = _rr("h")
    read_int = _rr("i")
    read_float = _rr("f")
    read_double = _rr("d")
    read_long = _rr("l")

    is_torquegrid = read_bool()
    if is_torquegrid:
        raise ValueError("Torque grids are not supported by this utility")
    arch = read_short()
    if arch != 64:
        raise ValueError(f"Unexpected grid architecture: {arch}")

    d = {}
    d["gridspacing"] = read_double()
    d["gridextension"] = read_int()
    d["plateaudis"] = read_double()
    d["neighbourdis"] = read_double()
    alphabet = np.array(_r("?" * 99, 99), bool)
    d["alphabet"] = alphabet
    alphabet_atomtypes = (np.where(alphabet)[0] + 1).astype(np.int32)
    nr_vdw_channels = len(alphabet_atomtypes)

    d["origin"] = np.array((read_float(), read_float(), read_float()), dtype=np.float64)
    dx, dy, dz = read_int(), read_int(), read_int()
    d["dim"] = np.array((dx, dy, dz), dtype=np.int32)
    dx2, dy2, dz2 = read_int(), read_int(), read_int()
    d["dim2"] = np.array((dx2, dy2, dz2), dtype=np.int32)
    d["natoms"] = read_int()
    _pivot = np.array((read_double(), read_double(), read_double()), dtype=np.float64)

    nr_energrads = read_int()
    shm_energrads = read_int()
    if nr_energrads == 0:
        raise ValueError("Grid has no potentials/gradients")
    if shm_energrads != -1:
        raise ValueError("Can't read grid potentials from shared memory")
    energrads = np.frombuffer(
        grid_data, offset=pos, count=nr_energrads * 4, dtype=np.float32
    )
    energrads = energrads.reshape(nr_energrads, 4)
    pos += energrads.nbytes

    nr_neighbours = read_int()
    shm_neighbours = read_int()
    if shm_neighbours != -1:
        raise ValueError("Can't read neighbour grid from shared memory")
    nb_dtype = np.dtype([("type", np.uint8), ("index", np.uint32)], align=True)
    neighbours = np.frombuffer(
        grid_data, offset=pos, count=nr_neighbours, dtype=nb_dtype
    )
    pos += neighbours.nbytes
    neighbour_index = np.ascontiguousarray(neighbours["index"])

    innergridsize = read_long()
    if innergridsize != dx * dy * dz:
        raise ValueError(f"Inner grid size mismatch: {innergridsize} vs {dx*dy*dz}")

    innergrid_dtype = np.dtype(
        [
            ("potential", np.uint32, 100),
            ("neighbourlist", np.int32),
            ("nr_neighbours", np.int16),
        ],
        align=True,
    )
    innergrid = np.frombuffer(
        grid_data, offset=pos, count=innergridsize, dtype=innergrid_dtype
    )
    pos += innergrid.nbytes
    innergrid = innergrid.reshape((dz, dy, dx)).swapaxes(0, 2)
    innergrid = np.ascontiguousarray(innergrid)

    inner_potential_grid = np.zeros((nr_vdw_channels, dx, dy, dz, 4), dtype=np.float32)
    inner_elec_grid = np.zeros((dx, dy, dz, 4), dtype=np.float32)
    pot_ind = innergrid["potential"]
    pot_pos = 0
    for n in range(99):
        curr = pot_ind[:, :, :, n]
        if not alphabet[n]:
            if curr.max() != 0:
                raise ValueError(
                    f"Unexpected potentials for alphabet-disabled atomtype {n+1}"
                )
            continue
        if curr.min() < 1 or curr.max() > len(energrads):
            raise ValueError(f"Inner potential index out of range for atomtype {n+1}")
        inner_potential_grid[pot_pos] = energrads[curr - 1]
        pot_pos += 1
    elec_ind = pot_ind[:, :, :, 99]
    mask = elec_ind > 0
    if elec_ind.max() > len(energrads):
        raise ValueError("Inner electrostatic potential index out of range")
    inner_elec_grid[mask] = energrads[elec_ind[mask] - 1]
    del pot_ind

    nr_neighbours_inner = np.ascontiguousarray(innergrid["nr_neighbours"])
    d["nr_neighbours"] = nr_neighbours_inner
    neighbourlist = innergrid["neighbourlist"]
    max_nr_neighbours = int(nr_neighbours_inner.max())
    d["max_nr_neighbours"] = max_nr_neighbours
    neighbour_grid = np.full((dx, dy, dz, max_nr_neighbours), 2**16 - 1, np.uint16)
    for n in range(max_nr_neighbours):
        maskn = nr_neighbours_inner > n
        nb_ind = neighbourlist[maskn]
        nb = nb_ind - 1 + n
        neighbour_grid[maskn, n] = neighbour_index[nb]
    d["neighbour_grid"] = neighbour_grid

    biggridsize = read_long()
    if biggridsize != dx2 * dy2 * dz2:
        raise ValueError(f"Outer grid size mismatch: {biggridsize} vs {dx2*dy2*dz2}")
    biggrid = np.frombuffer(
        grid_data, offset=pos, count=biggridsize * 100, dtype=np.uint32
    )
    pos += biggrid.nbytes
    pot_ind = biggrid.reshape((dz2, dy2, dx2, 100)).swapaxes(0, 2)

    outer_potential_grid = np.zeros(
        (nr_vdw_channels, dx2, dy2, dz2, 4), dtype=np.float32
    )
    outer_elec_grid = np.zeros((dx2, dy2, dz2, 4), dtype=np.float32)
    pot_pos = 0
    for n in range(99):
        curr = pot_ind[:, :, :, n]
        if not alphabet[n]:
            if curr.max() != 0:
                raise ValueError(f"Unexpected outer potentials for atomtype {n+1}")
            continue
        if curr.max() > len(energrads):
            raise ValueError(f"Outer potential index out of range for atomtype {n+1}")
        maskn = curr > 0
        outer_potential_grid[pot_pos][maskn] = energrads[curr[maskn] - 1]
        pot_pos += 1
    elec_ind = pot_ind[:, :, :, 99]
    if elec_ind.max() > len(energrads):
        raise ValueError("Outer electrostatic potential index out of range")
    maske = elec_ind > 0
    outer_elec_grid[maske] = energrads[elec_ind[maske] - 1]

    # Legacy .grid stores force convention: channels 1:4 = -(dE/d_i).
    # Negate to gradient convention: channels 1:4 = +(dE/d_i).
    inner_potential_grid[:, :, :, :, 1:4] *= -1
    outer_potential_grid[:, :, :, :, 1:4] *= -1
    inner_elec_grid[:, :, :, 1:4] *= -1
    outer_elec_grid[:, :, :, 1:4] *= -1

    d["inner_potential_grid"] = inner_potential_grid
    d["outer_potential_grid"] = outer_potential_grid
    d["inner_elec_grid"] = inner_elec_grid
    d["outer_elec_grid"] = outer_elec_grid
    d["alphabet_atomtypes"] = alphabet_atomtypes

    grid_class = namedtuple("Grid", tuple(d.keys()) + ("neighbour_grid_ravel",))
    return grid_class(*d.values(), neighbour_grid_ravel=None)


# ---------------------------------------------------------------------------
# NPZ grid serialisation
# ---------------------------------------------------------------------------

_GRID_SCALAR_FIELDS = (
    "gridspacing",
    "gridextension",
    "plateaudis",
    "neighbourdis",
    "natoms",
    "max_nr_neighbours",
)
_GRID_ARRAY_FIELDS = (
    "alphabet",
    "origin",
    "dim",
    "dim2",
    "nr_neighbours",
    "neighbour_grid",
    "inner_potential_grid",
    "outer_potential_grid",
    "inner_elec_grid",
    "outer_elec_grid",
    "alphabet_atomtypes",
)


def write_grid_npz(grid, path: str) -> None:
    """Serialise a Grid namedtuple (as returned by read_grid_with_electro or
    grid_generator.generate_grid) to a compressed .npz file.

    All array fields are stored verbatim.  Scalar fields are wrapped in
    0-d numpy arrays so np.savez_compressed handles them uniformly.
    ``neighbour_grid_ravel`` is recomputed on load and is NOT stored.
    """
    arrays = {}
    for name in _GRID_SCALAR_FIELDS:
        arrays[name] = np.array(getattr(grid, name))
    for name in _GRID_ARRAY_FIELDS:
        arrays[name] = np.asarray(getattr(grid, name))
    np.savez_compressed(path, **arrays)


def read_grid_npz(path: str):
    """Load a Grid namedtuple previously written by write_grid_npz.

    Returns the same namedtuple layout as read_grid_with_electro, with
    ``neighbour_grid_ravel`` set to None (it will be built lazily by
    JaxScoreOracle if needed).
    """
    data = np.load(path)
    d = {}
    for name in _GRID_SCALAR_FIELDS:
        val = data[name]
        # Restore Python scalars for the fields that were ints/floats
        if name in ("gridextension", "natoms", "max_nr_neighbours"):
            d[name] = int(val)
        else:
            d[name] = float(val)
    for name in _GRID_ARRAY_FIELDS:
        d[name] = data[name]
    grid_class = namedtuple("Grid", tuple(d.keys()) + ("neighbour_grid_ravel",))
    return grid_class(*d.values(), neighbour_grid_ravel=None)


def _run_argsort(args) -> np.ndarray:
    arr, axis = args
    return np.argsort(arr, axis=axis).astype(np.int32)


def run_argsort(arr: jnp.ndarray, axis=None) -> jnp.ndarray:
    if jax.devices()[0].device_kind != "cpu":
        return jnp.argsort(arr, axis=axis)
    if axis is None:
        result_shape = arr.ravel().shape
    else:
        result_shape = arr.shape
    result_shape = jax.ShapeDtypeStruct(result_shape, np.int32)
    return jax.pure_callback(_run_argsort, result_shape, (arr, axis))


if jax.devices()[0].device_kind == "cpu":
    run_argsort = jax.custom_jvp(run_argsort)

    @run_argsort.defjvp
    def default_grad(primals, tangents):
        return run_argsort(*primals), run_argsort(*tangents)


@jit
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


@jit
def rotvec2rotmat(v0, v1, v2):
    """Rodrigues rotation-vector to 3x3 rotation matrix (standard scipy convention)."""
    theta = jnp.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    safe_t = jnp.where(theta > 1.0e-10, theta, 1.0)
    k0 = v0 / safe_t
    k1 = v1 / safe_t
    k2 = v2 / safe_t
    s = jnp.where(theta > 1.0e-10, jnp.sin(theta), theta)
    c = jnp.where(theta > 1.0e-10, jnp.cos(theta), 1.0 - 0.5 * theta * theta)
    omc = 1.0 - c
    return jnp.asarray(
        [
            [c + omc * k0 * k0, omc * k0 * k1 - s * k2, omc * k0 * k2 + s * k1],
            [omc * k1 * k0 + s * k2, c + omc * k1 * k1, omc * k1 * k2 - s * k0],
            [omc * k2 * k0 - s * k1, omc * k2 * k1 + s * k0, c + omc * k2 * k2],
        ],
        dtype=jnp.float64,
    )


@jit
def rotvec_dofs_to_mats(dofs, pivot):
    """Like dofs_to_mats but for rotvec DOFs: (v0, v1, v2, tx, ty, tz)."""
    rot_col = jax.vmap(rotvec2rotmat)(dofs[:, 0], dofs[:, 1], dofs[:, 2])
    rot_row = jnp.swapaxes(rot_col, 1, 2)
    pivot_rot = jnp.einsum("j,bji->bi", pivot, rot_row)
    trans = dofs[:, 3:6] + pivot[None, :] - pivot_rot
    mats = jnp.zeros((dofs.shape[0], 4, 4), dtype=jnp.float64)
    mats = mats.at[:, :3, :3].set(rot_row)
    mats = mats.at[:, 3, :3].set(trans)
    mats = mats.at[:, 3, 3].set(1.0)
    return mats


@jit
def dofs_to_mats(dofs, pivot):
    rot_col = jax.vmap(euler2rotmat)(dofs[:, 0], dofs[:, 1], dofs[:, 2])
    rot_row = jnp.swapaxes(rot_col, 1, 2)
    pivot_rot = jnp.einsum("j,bji->bi", pivot, rot_row)
    trans = dofs[:, 3:6] + pivot[None, :] - pivot_rot
    mats = jnp.zeros((dofs.shape[0], 4, 4), dtype=jnp.float64)
    mats = mats.at[:, :3, :3].set(rot_row)
    mats = mats.at[:, 3, :3].set(trans)
    mats = mats.at[:, 3, 3].set(1.0)
    return mats


@jit
def transform_ligand(mats, coor_lig):
    coor_lig2 = jnp.concatenate(
        (coor_lig, jnp.ones((coor_lig.shape[0], 1), dtype=coor_lig.dtype)), axis=1
    )
    all_coors_lig = jnp.einsum("jk,bkl->bjl", coor_lig2, mats)
    return all_coors_lig[:, :, :3]


@jit
def transform_ligand_pooled(mats, coor_lig_ens, conformers):
    coor_lig2 = jnp.concatenate(
        (
            coor_lig_ens,
            jnp.ones(
                (coor_lig_ens.shape[0], coor_lig_ens.shape[1], 1),
                dtype=coor_lig_ens.dtype,
            ),
        ),
        axis=2,
    )
    all_coors_lig = jnp.einsum("bij,bjk->bik", coor_lig2[conformers], mats)
    return all_coors_lig[:, :, :3]


def build_kernel(
    grid,
    ff,
    lig_atomtypes_ff,
    lig_vdw_channel_idx,
    lig_charge_raw,
    lig_charge_scaled,
    cdie: bool,
    padded_nb_size: int = 0,
    max_nb_cap: int = 0,
    use_precomputed_grid_gradients: bool = True,
    rotation: str = "euler",
):
    _dofs_to_mats = rotvec_dofs_to_mats if rotation == "rotvec" else dofs_to_mats
    plateaudissq = jnp.float64(grid.plateaudis**2)
    inv50sq = jnp.float64(1.0 / (50.0 * 50.0))
    inv50 = jnp.float64(1.0 / 50.0)
    lig_charge_raw_np = np.asarray(lig_charge_raw, dtype=np.float64)
    charged_idx_np = np.nonzero(np.abs(lig_charge_raw_np) > 1.0e-3)[0].astype(np.int32)
    n_charged = int(charged_idx_np.shape[0])
    charged_idx_j = jnp.array(charged_idx_np, dtype=np.int32)
    charged_elec_channel_idx_j = jnp.full(
        (n_charged,), np.int32(grid.elec_channel_index), dtype=np.int32
    )

    grad_nonbon = jax.grad(nonbon)
    grad_elec = jax.grad(elec_dsq_cdie if cdie else elec_dsq_const)

    _at1, _at2 = jnp.indices((ff.rc.shape[0], ff.rc.shape[1]))
    plateau_nonbon_grad = jnp.vectorize(
        grad_nonbon, excluded=(1,), signature="(),(),()->()"
    )(plateaudissq, ff, _at1, _at2)

    def pair_dif(d, ff0, at1, at2, charge):
        dsq = (d * d).sum()
        charge0 = jnp.where(jnp.abs(charge) > 1.0e-3, charge, 0.0)
        e_nb = nonbon(dsq, ff0, at1, at2) - nonbon(plateaudissq, ff0, at1, at2)
        if cdie:
            e_el = elec_dsq_cdie(dsq, charge0) - elec_dsq_cdie(plateaudissq, charge0)
        else:
            e_el = elec_dsq_const(dsq, charge0) - elec_dsq_const(plateaudissq, charge0)
        return e_nb + e_el

    pair_dif = jax.custom_jvp(pair_dif)

    @pair_dif.defjvp
    @jit
    def pair_dif_jvp(primals, tangents):
        d, ff0, at1, at2, charge = primals
        dsq0 = (d * d).sum()
        charge00 = jnp.where(jnp.abs(charge) > 1.0e-3, charge, 0.0)
        ans_nb = nonbon(dsq0, ff0, at1, at2) - nonbon(plateaudissq, ff0, at1, at2)
        if cdie:
            ans_el = elec_dsq_cdie(dsq0, charge00) - elec_dsq_cdie(
                plateaudissq, charge00
            )
        else:
            ans_el = elec_dsq_const(dsq0, charge00) - elec_dsq_const(
                plateaudissq, charge00
            )
        ans = ans_nb + ans_el
        d_dot = tangents[0]
        dsq = dsq0
        charge0 = charge00
        grad_nb_main = grad_nonbon(dsq, ff0, at1, at2)
        grad_nb_plateau = plateau_nonbon_grad[at1, at2]
        if cdie:
            grad_el_main = grad_elec(dsq, charge0)
            grad_el_plateau = grad_elec(plateaudissq, charge0)
        else:
            grad_el_main = grad_elec(dsq, charge0)
            grad_el_plateau = grad_elec(plateaudissq, charge0)
        grad_main = 2.0 * (grad_nb_main + grad_el_main) * d
        ratio = jnp.sqrt(plateaudissq / dsq)
        grad_plateau = 2.0 * (grad_nb_plateau + grad_el_plateau) * d * ratio
        tangent = ((grad_main - grad_plateau) * d_dot).sum()
        return ans, tangent

    @jit
    def interpolate_grid(channel_idx, voxel, potential_grid):
        voxel0 = jnp.floor(voxel).astype(np.int32)
        voxel1 = jnp.ceil(voxel).astype(np.int32)
        w0 = voxel1 - voxel
        w1 = voxel - voxel0
        bsz = voxel.shape[0]
        chan = jnp.broadcast_to(channel_idx[None, :], (bsz, channel_idx.shape[0]))
        result = jnp.zeros((voxel.shape[0], voxel.shape[1], 4), dtype=jnp.float64)
        for wx, x in ((w0[:, :, 0], voxel0[:, :, 0]), (w1[:, :, 0], voxel1[:, :, 0])):
            for wy, y in (
                (w0[:, :, 1], voxel0[:, :, 1]),
                (w1[:, :, 1], voxel1[:, :, 1]),
            ):
                for wz, z in (
                    (w0[:, :, 2], voxel0[:, :, 2]),
                    (w1[:, :, 2], voxel1[:, :, 2]),
                ):
                    corner = potential_grid[chan, x, y, z]
                    result = result + (wx * wy * wz)[:, :, None] * corner
        return result

    @jit
    def potential_atom_energrads(all_coors_lig, channel_idx, grid0):
        vox_innergrid = (all_coors_lig - grid0.origin) / grid0.gridspacing
        x, y, z = vox_innergrid[:, :, 0], vox_innergrid[:, :, 1], vox_innergrid[:, :, 2]
        in_innergrid = (
            (x >= 0)
            & (x < grid0.dim[0] - 1)
            & (y >= 0)
            & (y < grid0.dim[1] - 1)
            & (z >= 0)
            & (z < grid0.dim[2] - 1)
        )
        vox_outergrid = (vox_innergrid + grid0.gridextension) / 2.0
        x2, y2, z2 = (
            vox_outergrid[:, :, 0],
            vox_outergrid[:, :, 1],
            vox_outergrid[:, :, 2],
        )
        in_outergrid = (
            (x2 >= 0)
            & (x2 < grid0.dim2[0] - 1)
            & (y2 >= 0)
            & (y2 < grid0.dim2[1] - 1)
            & (z2 >= 0)
            & (z2 < grid0.dim2[2] - 1)
        )
        inner = interpolate_grid(
            channel_idx, vox_innergrid, grid0.inner_potential_grid_all
        )
        outer = interpolate_grid(
            channel_idx, vox_outergrid, grid0.outer_potential_grid_all
        )
        zeros = jnp.zeros_like(inner)
        return jnp.where(
            in_innergrid[:, :, None],
            inner,
            jnp.where(in_outergrid[:, :, None], outer, zeros),
        )

    if use_precomputed_grid_gradients:

        @jit
        def potential_atom_energies(
            all_coors_lig, lig_vdw_channel_idx0, lig_charge_raw0, grid0
        ):
            vdw_eg = potential_atom_energrads(
                all_coors_lig, lig_vdw_channel_idx0, grid0
            )
            ans = vdw_eg[:, :, 0]
            if n_charged > 0:
                # Evaluate electrostatic interpolation only for charged ligand atoms.
                all_coors_ch = all_coors_lig[:, charged_idx_j, :]
                elec_eg_ch = potential_atom_energrads(
                    all_coors_ch, charged_elec_channel_idx_j, grid0
                )
                qraw_ch = lig_charge_raw0[charged_idx_j]
                e_el = jnp.zeros_like(ans)
                e_el = e_el.at[:, charged_idx_j].set(
                    qraw_ch[None, :] * elec_eg_ch[:, :, 0]
                )
                ans = ans + e_el
            return ans

        potential_atom_energies = jax.custom_jvp(potential_atom_energies)

        @potential_atom_energies.defjvp
        @jit
        def potential_atom_energies_jvp(primals, tangents):
            all_coors_lig, lig_vdw_channel_idx0, lig_charge_raw0, grid0 = primals
            vdw_eg = potential_atom_energrads(
                all_coors_lig, lig_vdw_channel_idx0, grid0
            )
            ans = vdw_eg[:, :, 0]
            # Channels 1:4 now store +dE/d_i (gradient convention).
            atom_grad = vdw_eg[:, :, 1:4]
            if n_charged > 0:
                all_coors_ch = all_coors_lig[:, charged_idx_j, :]
                elec_eg_ch = potential_atom_energrads(
                    all_coors_ch, charged_elec_channel_idx_j, grid0
                )
                qraw_ch = lig_charge_raw0[charged_idx_j]
                e_el = jnp.zeros_like(ans)
                e_el = e_el.at[:, charged_idx_j].set(
                    qraw_ch[None, :] * elec_eg_ch[:, :, 0]
                )
                ans = ans + e_el

                g_el = jnp.zeros_like(atom_grad)
                g_el = g_el.at[:, charged_idx_j, :].set(
                    qraw_ch[None, :, None] * elec_eg_ch[:, :, 1:4]
                )
                atom_grad = atom_grad + g_el
            tangent = (atom_grad * tangents[0]).sum(axis=2)
            return ans, tangent

    else:

        @jit
        def potential_atom_energies(
            all_coors_lig, lig_vdw_channel_idx0, lig_charge_raw0, grid0
        ):
            vdw_e = potential_atom_energrads(
                all_coors_lig, lig_vdw_channel_idx0, grid0
            )[:, :, 0]
            ans = vdw_e
            if n_charged > 0:
                all_coors_ch = all_coors_lig[:, charged_idx_j, :]
                elec_e_ch = potential_atom_energrads(
                    all_coors_ch, charged_elec_channel_idx_j, grid0
                )[:, :, 0]
                qraw_ch = lig_charge_raw0[charged_idx_j]
                e_el = jnp.zeros_like(ans)
                e_el = e_el.at[:, charged_idx_j].set(qraw_ch[None, :] * elec_e_ch)
                ans = ans + e_el
            return ans

    @jit
    def potential_energy(mats, coor_lig, lig_vdw_channel_idx0, lig_charge_raw0, grid0):
        all_coors_lig = transform_ligand(mats, coor_lig)
        atom_energies = potential_atom_energies(
            all_coors_lig, lig_vdw_channel_idx0, lig_charge_raw0, grid0
        )
        return atom_energies.sum(axis=1)

    @jit
    def nb_energy_single(
        ind_innergrid,
        offset,
        lig_struc,
        lig_atom,
        all_coors_lig,
        coor_rec,
        rec_atomtypes,
        lig_atomtypes0,
        rec_charge_scaled0,
        lig_charge_scaled0,
        ff0,
    ):
        receptor_atom = grid.neighbour_grid_ravel[ind_innergrid, offset]
        nrec = coor_rec.shape[0]
        safe_atom = jnp.where(receptor_atom < nrec, receptor_atom, 0)
        rec_c = coor_rec[safe_atom]
        at1 = rec_atomtypes[safe_atom]
        lig_c = all_coors_lig[lig_struc, lig_atom]
        at2 = lig_atomtypes0[lig_atom]
        d = lig_c - rec_c
        dsq = (d * d).sum()
        charge = rec_charge_scaled0[safe_atom] * lig_charge_scaled0[lig_atom]
        valid = (receptor_atom < 2**16 - 1) & (dsq < plateaudissq)
        return cond(valid, pair_dif, lambda *_: 0.0, d, ff0, at1, at2, charge)

    nb_energy_vec = jnp.vectorize(
        nb_energy_single,
        excluded=(1, 4, 5, 6, 7, 8, 9, 10),
        signature="(),(),()->()",
    )

    if padded_nb_size > 0:
        # Padded mode: use fori_loop (no unrolling, no static ncontacts).
        # This compiles ONCE regardless of ncontacts value, eliminating
        # the recompilation storm from the chunked mode.
        @jit
        def nb_energy(
            ind_innergrid,
            ncontacts,
            lig_struc,
            lig_atom,
            all_coors_lig,
            coor_rec,
            rec_atomtypes,
            lig_atomtypes0,
            rec_charge_scaled0,
            lig_charge_scaled0,
            ff0,
        ):
            def body_fn(n, energies):
                return energies + nb_energy_vec(
                    ind_innergrid,
                    n,
                    lig_struc,
                    lig_atom,
                    all_coors_lig,
                    coor_rec,
                    rec_atomtypes,
                    lig_atomtypes0,
                    rec_charge_scaled0,
                    lig_charge_scaled0,
                    ff0,
                )

            return jax.lax.fori_loop(
                0,
                ncontacts,
                body_fn,
                jnp.zeros(ind_innergrid.shape[0], dtype=jnp.float64),
            )

    else:
        # Original mode: Python for-loop unrolled per ncontacts value.
        @partial(jit, static_argnames=("ncontacts",))
        def nb_energy(
            ind_innergrid,
            ncontacts,
            lig_struc,
            lig_atom,
            all_coors_lig,
            coor_rec,
            rec_atomtypes,
            lig_atomtypes0,
            rec_charge_scaled0,
            lig_charge_scaled0,
            ff0,
        ):
            energies = jnp.zeros(ind_innergrid.shape[0], dtype=jnp.float64)
            for n in range(ncontacts):
                energies = energies + nb_energy_vec(
                    ind_innergrid,
                    n,
                    lig_struc,
                    lig_atom,
                    all_coors_lig,
                    coor_rec,
                    rec_atomtypes,
                    lig_atomtypes0,
                    rec_charge_scaled0,
                    lig_charge_scaled0,
                    ff0,
                )
            return energies

    grid_dim_const = tuple(int(x) for x in grid.dim)

    @jit
    def generate_ind_innergrid(all_coors_lig, grid0):
        vox_innergrid = (all_coors_lig - grid0.origin) / grid0.gridspacing
        x, y, z = vox_innergrid[:, :, 0], vox_innergrid[:, :, 1], vox_innergrid[:, :, 2]
        in_innergrid = (
            (x >= 0)
            & (x < grid0.dim[0] - 1)
            & (y >= 0)
            & (y < grid0.dim[1] - 1)
            & (z >= 0)
            & (z < grid0.dim[2] - 1)
        )
        out_of_bounds = max(grid_dim_const) + 2
        pos = jnp.where(
            in_innergrid[:, :, None],
            jnp.floor(vox_innergrid + 0.5).astype(np.int32),
            out_of_bounds,
        )
        ind = jnp.ravel_multi_index(
            (pos[:, :, 0], pos[:, :, 1], pos[:, :, 2]),
            dims=grid_dim_const,
            mode="clip",
        )
        ind = ind + out_of_bounds**3 * (1 - in_innergrid.astype(np.uint8))
        nb_index = jnp.take(grid0.nr_neighbours, ind)
        nb_index = jnp.where(nb_index != -32768, nb_index, 0)
        return ind, nb_index, nb_index.max()

    @jit
    def generate_sort_index(nb_index):
        k = -nb_index
        sort_index = jnp.unravel_index(run_argsort(k, axis=None), nb_index.shape)
        nr_inner_atoms = jnp.searchsorted(k[sort_index], 0, side="left")
        return sort_index, nr_inner_atoms

    max_nr_neighbours_int = int(grid.max_nr_neighbours)
    if max_nb_cap > 0:
        max_nr_neighbours_int = min(max_nr_neighbours_int, max_nb_cap)
    if max_nr_neighbours_int < 1:
        max_nr_neighbours_int = 1

    if padded_nb_size > 0:
        # Precompute plateau energy lookup tables for the vectorized
        # neighbour energy.  Indexed by (rec_type, lig_type).
        _rt, _lt = jnp.indices((ff.rc.shape[0], ff.rc.shape[1]))
        _pdsq = plateaudissq
        _prr2 = 1.0 / _pdsq
        _prr23 = _prr2 * _prr2 * _prr2
        _prep = ff.rc[_rt, _lt] * _prr2
        _pvlj = (_prep - ff.ac[_rt, _lt]) * _prr23
        _plateau_lj = jnp.where(
            _pdsq < ff.rmin2[_rt, _lt],
            _pvlj + (ff.ivor[_rt, _lt] - 1.0) * ff.emin[_rt, _lt],
            ff.ivor[_rt, _lt] * _pvlj,
        )
        if cdie:
            _plateau_el_per_charge = jnp.maximum(1.0 / jnp.sqrt(_pdsq) - inv50, 0.0)
        else:
            _plateau_el_per_charge = jnp.maximum(_prr2 - inv50sq, 0.0)

        @partial(jit, static_argnames=("k",))
        def nb_energy_vectorized_k(
            ind_innergrid,
            lig_struc,
            lig_atom,
            all_coors_lig,
            coor_rec,
            rec_atomtypes,
            lig_atomtypes0,
            rec_charge_scaled0,
            lig_charge_scaled0,
            ff0,
            k,
        ):
            """Fully vectorized neighbour energy: no loop, all offsets at once.

            ind_innergrid : (N,) int32 — raveled inner grid index per atom
            Returns        : (N,) float64 — per-atom energies
            """
            K = k
            offsets = jnp.arange(K, dtype=jnp.int32)  # (K,)

            # Gather neighbour data: (N, K)
            receptor_atoms = grid.neighbour_grid_ravel[
                ind_innergrid[:, None], offsets[None, :]
            ]

            # Legacy non-rigid runtime includes both type-1 and type-2
            # neighbour entries; type-2 can contribute when they cross
            # inside plateau in the current pose.
            nrec = coor_rec.shape[0]
            valid_nb = receptor_atoms < (2**16 - 1)
            safe_atoms = jnp.where(receptor_atoms < nrec, receptor_atoms, 0)

            # Coordinates: (N, K, 3) and (N, 1, 3)
            rec_c = coor_rec[safe_atoms]  # (N, K, 3)
            lig_c = all_coors_lig[lig_struc, lig_atom][:, None, :]  # (N, 1, 3)

            d = lig_c - rec_c  # (N, K, 3)
            dsq = (d * d).sum(axis=-1)  # (N, K)

            # Full validity: real neighbour & within plateau distance
            valid = valid_nb & (dsq < plateaudissq)

            # Atom types: (N, K) for receptor, (N, 1) for ligand
            at1 = rec_atomtypes[safe_atoms]  # (N, K)
            at2 = lig_atomtypes0[lig_atom][:, None]  # (N, 1)

            # --- LJ energy ---
            safe_dsq = jnp.where(valid, dsq, 1.0)  # avoid div-by-zero
            rr2 = 1.0 / safe_dsq
            alen = ff0.ac[at1, at2]
            rlen = ff0.rc[at1, at2]
            rr23 = rr2 * rr2 * rr2
            rep = rlen * rr2
            vlj = (rep - alen) * rr23
            attraction = ff0.ivor[at1, at2]
            e_lj = jnp.where(
                safe_dsq < ff0.rmin2[at1, at2],
                vlj + (attraction - 1.0) * ff0.emin[at1, at2],
                attraction * vlj,
            )
            # Subtract plateau: (N, K)
            e_lj_plateau = _plateau_lj[at1, at2]
            e_lj_corr = e_lj - e_lj_plateau

            # --- Electrostatic energy ---
            charge = (
                rec_charge_scaled0[safe_atoms] * lig_charge_scaled0[lig_atom][:, None]
            )
            charge_safe = jnp.where(jnp.abs(charge) > 1e-3, charge, 0.0)
            if cdie:
                e_el = charge_safe * (
                    jnp.maximum(1.0 / jnp.sqrt(safe_dsq) - inv50, 0.0)
                    - _plateau_el_per_charge
                )
            else:
                e_el = charge_safe * (
                    jnp.maximum(rr2 - inv50sq, 0.0) - _plateau_el_per_charge
                )

            # Total per-pair energy, masked by validity
            e_pair = jnp.where(valid, e_lj_corr + e_el, 0.0)  # (N, K)
            return e_pair.sum(axis=1)  # (N,)

        @jit
        def nb_energy_vectorized(
            ind_innergrid,
            lig_struc,
            lig_atom,
            all_coors_lig,
            coor_rec,
            rec_atomtypes,
            lig_atomtypes0,
            rec_charge_scaled0,
            lig_charge_scaled0,
            ff0,
        ):
            return nb_energy_vectorized_k(
                ind_innergrid,
                lig_struc,
                lig_atom,
                all_coors_lig,
                coor_rec,
                rec_atomtypes,
                lig_atomtypes0,
                rec_charge_scaled0,
                lig_charge_scaled0,
                ff0,
                max_nr_neighbours_int,
            )

    def neighbour_energy_from_all_coors(
        all_coors_lig,
        coor_rec,
        rec_atomtypes,
        rec_charge_scaled0,
        lig_atomtypes0,
        lig_charge_scaled0,
        ff0,
        grid0,
        nb_chunk_thresholds,
        nb_chunk_size,
    ):
        ind_innergrid, nb_index, max_contacts = generate_ind_innergrid(all_coors_lig, grid0)
        if int(max_contacts) == 0:
            return jnp.zeros(all_coors_lig.shape[0], dtype=jnp.float64)

        sort_index, nr_inner_atoms = generate_sort_index(nb_index)
        sorted_ind_innergrid = ind_innergrid[sort_index[0], sort_index[1]]
        sorted_nb_index = nb_index[sort_index[0], sort_index[1]]
        nr_inner_atoms = int(nr_inner_atoms)
        tot_atoms = len(sort_index[0])

        if padded_nb_size > 0:
            # ---- Padded mode: vectorized nb_energy, fixed shape ----
            # Pad sorted arrays to padded_nb_size so the JIT function
            # always sees the same input shape.  Uses fully vectorized
            # nb_energy_vectorized (no loop over offsets).
            pad_n = padded_nb_size - nr_inner_atoms
            if pad_n < 0:
                raise ValueError(
                    f"padded_nb_size={padded_nb_size} too small for "
                    f"nr_inner_atoms={nr_inner_atoms}"
                )
            inner_ind = sorted_ind_innergrid[:nr_inner_atoms]
            inner_struc = sort_index[0][:nr_inner_atoms]
            inner_atom = sort_index[1][:nr_inner_atoms]

            padded_ind = jnp.pad(inner_ind, (0, pad_n), constant_values=0)
            padded_struc = jnp.pad(inner_struc, (0, pad_n), constant_values=0)
            padded_atom = jnp.pad(inner_atom, (0, pad_n), constant_values=0)

            chunk_e = nb_energy_vectorized(
                padded_ind,
                padded_struc,
                padded_atom,
                all_coors_lig,
                coor_rec,
                rec_atomtypes,
                lig_atomtypes0,
                rec_charge_scaled0,
                lig_charge_scaled0,
                ff0,
            )

            # Only scatter real atom energies (discard padding)
            atom_energies_real = chunk_e[:nr_inner_atoms]
            energies = jnp.zeros(nb_index.shape[0], dtype=jnp.float64)
            energies = energies.at[inner_struc].add(atom_energies_real)
            return energies

        # ---- Original chunked mode: variable-size chunks ----
        max_ncontacts_map = {}
        for n in range(0, len(nb_chunk_thresholds) - 1):
            p1 = nb_chunk_thresholds[n]
            p2 = nb_chunk_thresholds[n + 1]
            for p in range(p1 + 1, p2 + 1):
                max_ncontacts_map[p] = p2

        atom_energies = jnp.zeros(tot_atoms, dtype=jnp.float64)
        n = 0
        while n < nr_inner_atoms:
            start = n
            max_ncontacts0 = int(sorted_nb_index[start])
            max_ncontacts = max_ncontacts_map[max_ncontacts0]
            n += nb_chunk_size
            if max_ncontacts0 > 0:
                max_ncontacts0_next = int(
                    sorted_nb_index[min(n - 1, nr_inner_atoms - 1)]
                )
                if max_ncontacts - max_ncontacts0_next < 5:
                    n += 3 * nb_chunk_size
            end = min(n, nr_inner_atoms)
            chunk_energies = nb_energy(
                sorted_ind_innergrid[start:end],
                max_ncontacts,
                sort_index[0][start:end],
                sort_index[1][start:end],
                all_coors_lig,
                coor_rec,
                rec_atomtypes,
                lig_atomtypes0,
                rec_charge_scaled0,
                lig_charge_scaled0,
                ff0,
            )
            atom_energies = atom_energies.at[start:end].set(chunk_energies)

        energies = jnp.zeros(nb_index.shape[0], dtype=jnp.float64)
        energies = energies.at[sort_index[0]].add(atom_energies)
        return energies

    def neighbour_energy(
        mats,
        coor_rec,
        rec_atomtypes,
        rec_charge_scaled0,
        coor_lig,
        lig_atomtypes0,
        lig_charge_scaled0,
        ff0,
        grid0,
        nb_chunk_thresholds,
        nb_chunk_size,
    ):
        all_coors_lig = transform_ligand(mats, coor_lig)
        return neighbour_energy_from_all_coors(
            all_coors_lig,
            coor_rec,
            rec_atomtypes,
            rec_charge_scaled0,
            lig_atomtypes0,
            lig_charge_scaled0,
            ff0,
            grid0,
            nb_chunk_thresholds,
            nb_chunk_size,
        )

    def main(
        dofs,
        coor_rec,
        rec_atomtypes,
        rec_charge_scaled0,
        coor_lig,
        lig_atomtypes0,
        lig_vdw_channel_idx0,
        lig_charge_raw0,
        lig_charge_scaled0,
        ff0,
        grid0,
        nb_chunk_thresholds,
        grid_dim,
        lig_pivot,
    ):
        mats = _dofs_to_mats(dofs, lig_pivot)
        nb_energies = neighbour_energy(
            mats,
            coor_rec,
            rec_atomtypes,
            rec_charge_scaled0,
            coor_lig,
            lig_atomtypes0,
            lig_charge_scaled0,
            ff0,
            grid0,
            nb_chunk_thresholds,
            NB_CHUNK_SIZE,
        )
        pot_energies = potential_energy(
            mats, coor_lig, lig_vdw_channel_idx0, lig_charge_raw0, grid0
        )
        energies = pot_energies + nb_energies
        return energies.sum(), energies

    def main_pooled(
        dofs,
        coor_rec,
        rec_atomtypes,
        rec_charge_scaled0,
        coor_lig_ens,
        conformers,
        lig_atomtypes0,
        lig_vdw_channel_idx0,
        lig_charge_raw0,
        lig_charge_scaled0,
        ff0,
        grid0,
        nb_chunk_thresholds,
        grid_dim,
        lig_pivot,
    ):
        mats = _dofs_to_mats(dofs, lig_pivot)
        all_coors_lig = transform_ligand_pooled(mats, coor_lig_ens, conformers)
        nb_energies = neighbour_energy_from_all_coors(
            all_coors_lig,
            coor_rec,
            rec_atomtypes,
            rec_charge_scaled0,
            lig_atomtypes0,
            lig_charge_scaled0,
            ff0,
            grid0,
            nb_chunk_thresholds,
            NB_CHUNK_SIZE,
        )
        pot_energies = potential_atom_energies(
            all_coors_lig, lig_vdw_channel_idx0, lig_charge_raw0, grid0
        ).sum(axis=1)
        energies = pot_energies + nb_energies
        return energies.sum(), energies

    @jit
    def potential_ad(
        dofs,
        coor_rec,
        rec_atomtypes,
        rec_charge_scaled0,
        coor_lig0,
        lig_atomtypes0,
        lig_vdw_channel_idx0,
        lig_charge_raw0,
        lig_charge_scaled0,
        ff0,
        grid0,
        lig_pivot0,
    ):
        # Signature mirrors main_ad for easy reuse in jax_scorer.
        del (
            coor_rec,
            rec_atomtypes,
            rec_charge_scaled0,
            lig_atomtypes0,
            lig_charge_scaled0,
            ff0,
        )
        mats = _dofs_to_mats(dofs, lig_pivot0)
        all_coors_lig = transform_ligand(mats, coor_lig0)
        pot_e = potential_atom_energies(
            all_coors_lig, lig_vdw_channel_idx0, lig_charge_raw0, grid0
        ).sum(axis=1)
        return pot_e.sum(), pot_e

    if padded_nb_size > 0:
        # Fully JIT-able energy function for use with jax.value_and_grad.
        # Unlike `main` (which uses Python control flow in neighbour_energy),
        # this version processes ALL atom slots through the vectorized
        # nb_energy_vectorized, using a validity mask for out-of-grid atoms.
        # No sorting, no chunking, no Python `if` on traced values.
        grid_dim_tuple = tuple(int(x) for x in grid.dim)
        grid_dim_arr = jnp.array(list(grid_dim_tuple), dtype=jnp.int32)
        n_lig = lig_charge_raw.shape[0]

        @jit
        def main_ad(
            dofs,
            coor_rec,
            rec_atomtypes,
            rec_charge_scaled0,
            coor_lig0,
            lig_atomtypes0,
            lig_vdw_channel_idx0,
            lig_charge_raw0,
            lig_charge_scaled0,
            ff0,
            grid0,
            lig_pivot0,
        ):
            mats = _dofs_to_mats(dofs, lig_pivot0)
            all_coors_lig = transform_ligand(mats, coor_lig0)

            # --- Grid potential (differentiable via custom_jvp) ---
            pot_e = potential_atom_energies(
                all_coors_lig, lig_vdw_channel_idx0, lig_charge_raw0, grid0
            ).sum(axis=1)

            # --- Neighbour energy: all atoms, no sorting ---
            B = dofs.shape[0]
            A = coor_lig0.shape[0]
            vox = (all_coors_lig - grid0.origin) / grid0.gridspacing
            is_inner = (
                (vox[..., 0] >= 0)
                & (vox[..., 0] < grid0.dim[0] - 1)
                & (vox[..., 1] >= 0)
                & (vox[..., 1] < grid0.dim[1] - 1)
                & (vox[..., 2] >= 0)
                & (vox[..., 2] < grid0.dim[2] - 1)
            )
            # Clipped voxel positions for safe grid indexing
            pos = jnp.floor(vox + 0.5).astype(jnp.int32)
            pos = jnp.clip(pos, 0, grid_dim_arr[None, None, :] - 1)
            ind = jnp.ravel_multi_index(
                (pos[..., 0], pos[..., 1], pos[..., 2]),
                dims=grid_dim_tuple,
                mode="clip",
            )

            # Flatten (B, A) → (B*A,)
            flat_ind = ind.reshape(-1)
            flat_struc = jnp.repeat(jnp.arange(B, dtype=jnp.int32), A)
            flat_atom = jnp.tile(jnp.arange(A, dtype=jnp.int32), B)
            flat_is_inner = is_inner.reshape(-1).astype(jnp.float64)

            # Neighbour energy for all atom slots.
            nb_e_flat = nb_energy_vectorized(
                flat_ind,
                flat_struc,
                flat_atom,
                all_coors_lig,
                coor_rec,
                rec_atomtypes,
                lig_atomtypes0,
                rec_charge_scaled0,
                lig_charge_scaled0,
                ff0,
            )
            nb_e_flat = nb_e_flat * flat_is_inner  # zero out non-inner atoms
            nb_e = jnp.zeros(B, dtype=jnp.float64)
            nb_e = nb_e.at[flat_struc].add(nb_e_flat)

            energies = pot_e + nb_e
            return energies.sum(), energies

        main.ad = main_ad
    else:
        main.ad = None

    main.pooled = main_pooled
    main.pot_ad = potential_ad
    return main


@jit
def nonbon(dsq, ff, at1, at2):
    rr2 = 1.0 / dsq
    alen = ff.ac[at1, at2]
    rlen = ff.rc[at1, at2]
    rr23 = rr2 * rr2 * rr2
    rep = rlen * rr2
    vlj = (rep - alen) * rr23
    attraction = ff.ivor[at1, at2]
    return cond(
        dsq < ff.rmin2[at1, at2],
        lambda: vlj + (attraction - 1.0) * ff.emin[at1, at2],
        lambda: attraction * vlj,
    )


@jit
def elec_dsq_const(dsq, charge):
    rr2 = 1.0 / dsq
    rr2a = jnp.maximum(rr2 - 1.0 / (50.0 * 50.0), 0.0)
    return charge * rr2a


@jit
def elec_dsq_cdie(dsq, charge):
    rr1 = jnp.maximum(1.0 / jnp.sqrt(dsq) - 1.0 / 50.0, 0.0)
    return charge * rr1


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


NB_CHUNK_SIZE = 100_000


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

    pivots, ens_ids, dofs = parse_dat_for_grid_score(
        args.input_dat, max_poses=args.max_poses
    )
    ref_e, ref_g = parse_legacy_score(args.legacy_score, max_poses=args.max_poses)
    n = min(len(dofs), len(ref_e))
    if len(dofs) != len(ref_e):
        print(
            f"Warning: dat poses={len(dofs)}, legacy score={len(ref_e)}; truncating to {n}"
        )
    dofs = dofs[:n]
    ens_ids = ens_ids[:n]
    ref_e = ref_e[:n]
    ref_g = ref_g[:n]

    lig_pivot = pivots[2].astype(np.float64)

    with open(args.receptor_ens_list) as f:
        rec_files = [line.strip() for line in f if line.strip()]
    if not rec_files:
        raise ValueError("Empty receptor ensemble list")

    rec_coords_all = []
    rec_types_all = []
    rec_charge_all = []
    rec_weight_all = []
    list_dir = Path(args.receptor_ens_list).resolve().parent
    for rf in rec_files:
        p = Path(rf)
        if not p.is_absolute():
            p = list_dir / p
        c, a, q, w = parse_reduced_pdb(str(p))
        rec_coords_all.append(c)
        rec_types_all.append(a)
        rec_charge_all.append(q)
        rec_weight_all.append(w)

    lig_coords0, lig_types0, lig_charge0, lig_weight0 = parse_reduced_pdb(
        args.ligand_pdb
    )

    if not np.allclose(rec_weight_all[0], 1.0):
        print("Warning: receptor weights are not all 1.0")
    if not np.allclose(lig_weight0, 1.0):
        print("Warning: ligand weights are not all 1.0")

    rec_mask = rec_types_all[0] != 99
    lig_mask = lig_types0 != 99
    rec_types = rec_types_all[0][rec_mask]
    lig_types = lig_types0[lig_mask]

    for i in range(1, len(rec_types_all)):
        if not np.array_equal(rec_types_all[i][rec_mask], rec_types):
            raise ValueError(f"Receptor atom types differ in ensemble {i+1}")

    rec_coords_ens = np.asarray([c[rec_mask] for c in rec_coords_all], dtype=np.float64)
    rec_charge_ens_raw = np.asarray(
        [q[rec_mask] for q in rec_charge_all], dtype=np.float64
    )
    lig_coords = lig_coords0[lig_mask].astype(np.float64)
    lig_charge_raw = lig_charge0[lig_mask].astype(np.float64)

    felec = np.sqrt(332.053986 / args.epsilon)
    rec_charge_ens_scaled = rec_charge_ens_raw * felec
    lig_charge_scaled = lig_charge_raw * felec

    lig_alphabet, lig_atomtypes_ff = np.unique(lig_types, return_inverse=True)
    rec_alphabet, rec_atomtypes_ff = np.unique(rec_types, return_inverse=True)

    par = np.load(args.attract_par_npz)
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

    grid = read_grid_with_electro(Path(args.grid).read_bytes())
    rec_mapping = np.cumsum(rec_mask) - 1
    nb_flat = grid.neighbour_grid.reshape(-1)
    valid = nb_flat < 2**16 - 1
    nb_flat[valid] = rec_mapping[nb_flat[valid]]

    alpos = grid.alphabet_atomtypes.tolist()
    lig_vdw_channel_idx = np.array(
        [alpos.index(a) for a in lig_alphabet], dtype=np.int32
    )[lig_atomtypes_ff]

    inner_all = np.concatenate(
        (grid.inner_potential_grid, grid.inner_elec_grid[None, ...]), axis=0
    )
    outer_all = np.concatenate(
        (grid.outer_potential_grid, grid.outer_elec_grid[None, ...]), axis=0
    )
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
    dgrid["neighbour_grid_ravel"] = dgrid["neighbour_grid"].reshape(
        -1, grid.neighbour_grid.shape[-1]
    )
    GridJax = namedtuple(
        "GridJax",
        tuple(dgrid.keys()),
    )
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
        cdie=bool(args.cdie),
    )
    grad_main = jax.grad(lambda dof_batch, *rest: kernel_main(dof_batch, *rest)[0])

    dofs_j = jnp.array(dofs, dtype=jnp.float64)
    coor_lig_j = jnp.array(lig_coords, dtype=jnp.float64)
    lig_atomtypes_ff_j = jnp.array(lig_atomtypes_ff, dtype=np.int32)
    lig_vdw_channel_idx_j = jnp.array(lig_vdw_channel_idx, dtype=np.int32)
    lig_charge_raw_j = jnp.array(lig_charge_raw, dtype=jnp.float64)
    lig_charge_scaled_j = jnp.array(lig_charge_scaled, dtype=jnp.float64)
    rec_atomtypes_ff_j = jnp.array(rec_atomtypes_ff, dtype=np.int32)
    lig_pivot_j = jnp.array(lig_pivot, dtype=jnp.float64)

    cand_e = np.zeros(n, dtype=np.float64)
    cand_g = np.zeros((n, 6), dtype=np.float64)
    batch = max(1, int(args.batch))

    uniq_ens = np.unique(ens_ids)
    for ens in uniq_ens:
        ens0 = int(ens) - 1
        if ens0 < 0 or ens0 >= len(rec_files):
            raise ValueError(f"Ensemble index out of range: {ens}")
        idx = np.where(ens_ids == ens)[0]
        rec_coor_j = jnp.array(rec_coords_ens[ens0], dtype=jnp.float64)
        rec_charge_scaled_j = jnp.array(rec_charge_ens_scaled[ens0], dtype=jnp.float64)
        print(f"Processing ensemble {ens} ({len(idx)} poses)")
        for start0 in range(0, len(idx), batch):
            sub = idx[start0 : start0 + batch]
            dofs_b = dofs_j[sub]
            _, ene_b = kernel_main(
                dofs_b,
                rec_coor_j,
                rec_atomtypes_ff_j,
                rec_charge_scaled_j,
                coor_lig_j,
                lig_atomtypes_ff_j,
                lig_vdw_channel_idx_j,
                lig_charge_raw_j,
                lig_charge_scaled_j,
                ff,
                grid_j,
                nb_chunk_thresholds,
                grid_dim,
                lig_pivot_j,
            )
            grad_b = grad_main(
                dofs_b,
                rec_coor_j,
                rec_atomtypes_ff_j,
                rec_charge_scaled_j,
                coor_lig_j,
                lig_atomtypes_ff_j,
                lig_vdw_channel_idx_j,
                lig_charge_raw_j,
                lig_charge_scaled_j,
                ff,
                grid_j,
                nb_chunk_thresholds,
                grid_dim,
                lig_pivot_j,
            )
            cand_e[sub] = np.asarray(ene_b)
            cand_g[sub] = np.asarray(grad_b)
            if (start0 // batch) % 20 == 0:
                print(f"  {start0 + len(sub)}/{len(idx)}")

    mae_direct = float(np.mean(np.abs(cand_g - ref_g)))
    mae_flip = float(np.mean(np.abs(-cand_g - ref_g)))
    if mae_flip < mae_direct:
        cand_g = -cand_g
        print(
            f"Gradient sign flipped to match legacy output (mae {mae_direct:.6f} -> {mae_flip:.6f})"
        )
    else:
        print(f"Gradient sign kept (mae direct={mae_direct:.6f}, flip={mae_flip:.6f})")

    summarize("energy", ref_e, cand_e)
    summarize("grad", ref_g.reshape(-1), cand_g.reshape(-1))
    for i, name in enumerate(("phi", "ssi", "rot", "tx", "ty", "tz")):
        summarize(f"grad_{name}", ref_g[:, i], cand_g[:, i])

    if args.out_prefix:
        np.save(args.out_prefix + ".legacy_energy.npy", ref_e.astype(np.float32))
        np.save(args.out_prefix + ".legacy_grad.npy", ref_g.astype(np.float32))
        np.save(args.out_prefix + ".jax_energy.npy", cand_e.astype(np.float32))
        np.save(args.out_prefix + ".jax_grad.npy", cand_g.astype(np.float32))
        print(f"Saved outputs with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
