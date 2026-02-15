#!/usr/bin/env python3
"""ATTRACT-JAX minimizer using L-BFGS-B (scipy) with batched energy evaluation.

Replaces the minfor state machine with scipy.optimize.minimize(method='L-BFGS-B')
per pose. Energy and gradients are computed via jax.value_and_grad on the same
grid-based scoring kernel used by reproduce_grid_score.py.

Typical usage:
    python minimize_lbfgs.py test/systsearch-ens1-first200.dat \
        test/partner1-ensemble.list test/ligandr.pdb \
        --grid test/receptorgrid.grid \
        --attract-par-npz attract-jax/attract-par.npz \
        --maxfun 150 --out-prefix test/jax_lbfgs_first200
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import math
import re
import resource
import time
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize as scipy_minimize

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Import shared infrastructure from reproduce_grid_score
# ---------------------------------------------------------------------------
import sys

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
_jax_dir = _script_dir.parent
if str(_jax_dir) not in sys.path:
    sys.path.insert(0, str(_jax_dir))

from reproduce_grid_score import (
    build_kernel,
    parse_reduced_pdb,
    read_grid_with_electro,
    dofs_to_mats,
    NB_CHUNK_SIZE,
    summarize,
)

# ---------------------------------------------------------------------------
# DAT parser (from minfor_request_jax, slightly simplified)
# ---------------------------------------------------------------------------
STRUCT_RE = re.compile(r"^#\d+\s*$")
PIVOT_RE = re.compile(
    r"^#pivot\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)
ENERGY_RE = re.compile(r"^##\s*Energy:\s*([-+0-9.eE]+)\s*$")


def parse_dat_two_body(path: str, max_poses: int = 0):
    header: List[str] = []
    pivots: Dict[int, np.ndarray] = {}
    ens_list: List[int] = []
    dof_list: List[Tuple[float, ...]] = []
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
            raise ValueError(
                f"Expected first body line with 7 fields, got {len(first)}"
            )
        ens = int(round(first[0]))
        if len(second) == 7:
            second = second[1:]
        if len(second) != 6:
            raise ValueError(
                f"Expected ligand DOF line with 6 fields, got {len(second)}"
            )
        ens_list.append(ens)
        dof_list.append(tuple(float(v) for v in second))
        energy_list.append(
            float("nan") if current_energy is None else float(current_energy)
        )
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


def write_dat_two_body(
    path: str,
    header: List[str],
    ens: np.ndarray,
    dofs: np.ndarray,
    energies: Optional[np.ndarray] = None,
):
    with open(path, "w") as f:
        for line in header:
            f.write(line)
        for i in range(len(dofs)):
            if energies is not None:
                f.write(f"## Energy: {energies[i]:.15e}\n")
            f.write(f"#{i+1}\n")
            f.write(f"{int(ens[i]):12d}{0:12d}{0:12d}{0:12d}{0:12d}{0:12d}{0:12d}\n")
            phi, ssi, rot, xa, ya, za = dofs[i]
            f.write(
                f"{phi:24.16f} {ssi:24.16f} {rot:24.16f} "
                f"{xa:24.16f} {ya:24.16f} {za:24.16f}\n"
            )


def euler2rotmat_np(phi, ssi, rot):
    cs, cp = np.cos(ssi), np.cos(phi)
    ss, sp = np.sin(ssi), np.sin(phi)
    crot, srot = np.cos(rot), np.sin(rot)
    out = np.zeros((len(phi), 3, 3), dtype=np.float64)
    out[:, 0, 0] = crot * cs * cp + srot * sp
    out[:, 0, 1] = srot * cs * cp - crot * sp
    out[:, 0, 2] = ss * cp
    out[:, 1, 0] = crot * cs * sp - srot * cp
    out[:, 1, 1] = srot * cs * sp + crot * cp
    out[:, 1, 2] = ss * sp
    out[:, 2, 0] = -crot * ss
    out[:, 2, 1] = -srot * ss
    out[:, 2, 2] = cs
    return out


def dofs_to_mats_np(dofs, pivot):
    rot_col = euler2rotmat_np(dofs[:, 0], dofs[:, 1], dofs[:, 2])
    rot_row = np.swapaxes(rot_col, 1, 2)
    pivot_rot = np.einsum("j,bji->bi", pivot, rot_row)
    trans = dofs[:, 3:6] + pivot[None, :] - pivot_rot
    mats = np.zeros((len(dofs), 4, 4), dtype=np.float64)
    mats[:, :3, :3] = rot_row
    mats[:, 3, :3] = trans
    mats[:, 3, 3] = 1.0
    return mats


# ---------------------------------------------------------------------------
# GridEnergyModel — builds kernel and provides value_and_grad per ensemble
# ---------------------------------------------------------------------------
class GridEnergyModel:
    """Wraps the ATTRACT grid+neighbour energy with fused value_and_grad."""

    def __init__(
        self,
        receptor_ens_list: str,
        ligand_pdb: str,
        grid_file: str,
        attract_par_npz: str,
        epsilon: float,
        cdie: bool,
        lig_pivot: np.ndarray,
    ):
        with open(receptor_ens_list) as f:
            rec_files = [line.strip() for line in f if line.strip()]
        if not rec_files:
            raise ValueError("Empty receptor ensemble list")

        list_dir = Path(receptor_ens_list).resolve().parent
        rec_coords_all, rec_types_all, rec_charge_all, rec_weight_all = [], [], [], []
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
            ligand_pdb
        )

        rec_mask = rec_types_all[0] != 99
        lig_mask = lig_types0 != 99
        rec_types = rec_types_all[0][rec_mask]
        lig_types = lig_types0[lig_mask]

        rec_coords_ens = np.asarray(
            [c[rec_mask] for c in rec_coords_all], dtype=np.float64
        )
        rec_charge_ens_raw = np.asarray(
            [q[rec_mask] for q in rec_charge_all], dtype=np.float64
        )
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
        FF = namedtuple("FF", ("rc", "ac", "ivor", "emin", "rmin2"))
        ff = FF(
            jnp.array(rc),
            jnp.array(ac),
            jnp.array(ivor),
            jnp.array(emin),
            jnp.array(rmin2),
        )

        grid = read_grid_with_electro(Path(grid_file).read_bytes())
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
            lig_charge_raw=jnp.array(lig_charge_raw),
            lig_charge_scaled=jnp.array(lig_charge_scaled),
            cdie=bool(cdie),
        )

        # Fused value_and_grad — avoids double forward pass
        self._val_and_grad = jax.value_and_grad(
            lambda dof_batch, *rest: kernel_main(dof_batch, *rest)[0]
        )

        # Also keep the kernel for direct energy-only evaluation
        self._kernel_main = kernel_main

        self.rec_coor_ens = rec_coords_ens
        self.rec_charge_ens_scaled = rec_charge_ens_scaled
        self.rec_atomtypes_ff_j = jnp.array(rec_atomtypes_ff, dtype=np.int32)
        self.coor_lig_j = jnp.array(lig_coords)
        self.lig_atomtypes_ff_j = jnp.array(lig_atomtypes_ff, dtype=np.int32)
        self.lig_vdw_channel_idx_j = jnp.array(lig_vdw_channel_idx, dtype=np.int32)
        self.lig_charge_raw_j = jnp.array(lig_charge_raw)
        self.lig_charge_scaled_j = jnp.array(lig_charge_scaled)
        self.ff = ff
        self.grid_j = grid_j
        self.nb_chunk_thresholds = nb_chunk_thresholds
        self.grid_dim = grid_dim
        self.lig_pivot_j = jnp.array(lig_pivot)
        self.n_ensemble = len(rec_files)

    def _call_args(self, ens0: int):
        """Return (rec_coor, rec_charge, fixed_args...) for ensemble member ens0 (0-based)."""
        rec_coor_j = jnp.array(self.rec_coor_ens[ens0])
        rec_charge_j = jnp.array(self.rec_charge_ens_scaled[ens0])
        return (
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

    def value_and_grad_batch(self, ens0: int, dofs_batch: jnp.ndarray):
        """Compute (total_energy, per_pose_energies, per_pose_grads) for a batch.

        Args:
            ens0: 0-based ensemble index
            dofs_batch: (B, 6) array of DOFs

        Returns:
            energies: (B,) per-pose energies
            grads: (B, 6) per-pose gradients
        """
        args = self._call_args(ens0)
        total_e, grad_batch = self._val_and_grad(dofs_batch, *args)
        _, per_pose_e = self._kernel_main(dofs_batch, *args)
        return np.asarray(per_pose_e), np.asarray(grad_batch)

    def energy_and_grad_single(self, ens0: int, dof: np.ndarray):
        """Compute energy and gradient for a single pose.

        Args:
            ens0: 0-based ensemble index
            dof: (6,) DOF array

        Returns:
            energy: scalar
            grad: (6,) gradient
        """
        dof_batch = jnp.array(dof[None, :])
        args = self._call_args(ens0)
        total_e, grad_batch = self._val_and_grad(dof_batch, *args)
        return float(total_e), np.asarray(grad_batch[0])


# ---------------------------------------------------------------------------
# L-BFGS-B minimization
# ---------------------------------------------------------------------------
def minimize_poses(
    model: GridEnergyModel,
    ens: np.ndarray,
    dofs0: np.ndarray,
    maxfun: int = 150,
    grad_sign: float = -1.0,
    trace_every: int = 0,
):
    """Minimize all poses using scipy L-BFGS-B.

    Args:
        model: GridEnergyModel instance
        ens: (N,) ensemble ids (1-based)
        dofs0: (N, 6) starting DOFs
        maxfun: maximum number of function evaluations per pose
        grad_sign: sign correction for gradients (-1.0 to match legacy convention)
        trace_every: print progress every N poses (0 = silent)

    Returns:
        dofs_out: (N, 6) minimized DOFs
        energies_out: (N,) final energies
        nfev_out: (N,) number of function evaluations used
    """
    n = len(dofs0)
    dofs_out = np.zeros_like(dofs0)
    energies_out = np.zeros(n, dtype=np.float64)
    nfev_out = np.zeros(n, dtype=np.int32)

    # Group by ensemble for cache-friendliness (same rec coords + charge)
    uniq_ens = np.unique(ens)
    total_done = 0

    for ens_id in uniq_ens:
        ens0 = int(ens_id) - 1
        idx = np.where(ens == ens_id)[0]

        # Pre-compute and cache the JAX args for this ensemble
        args = model._call_args(ens0)

        # NOTE: We do NOT jit the outer value_and_grad call because
        # neighbour_energy contains data-dependent Python loops.
        # The inner kernels are already individually JIT-compiled.
        vg_fn = model._val_and_grad

        # Warm up inner JIT caches with first pose in this ensemble
        _warmup_dof = jnp.array(dofs0[idx[0]][None, :])
        _ = vg_fn(_warmup_dof, *args)

        for count, i in enumerate(idx):
            dof0 = dofs0[i].copy()

            # Closure for scipy: single-pose energy + gradient
            def func_and_grad(x):
                dof_j = jnp.array(x.reshape(1, 6))
                total_e, grad = vg_fn(dof_j, *args)
                e = float(total_e)
                g = np.asarray(grad[0]) * grad_sign
                return e, g.astype(np.float64)

            result = scipy_minimize(
                func_and_grad,
                dof0,
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxfun": maxfun,
                    "maxiter": maxfun,
                    "ftol": 1e-15,
                    "gtol": 1e-10,
                },
            )

            dofs_out[i] = result.x
            energies_out[i] = result.fun
            nfev_out[i] = result.nfev

            total_done += 1
            if trace_every and total_done % trace_every == 0:
                print(
                    f"  [{total_done}/{n}] ens={ens_id} nfev={result.nfev} "
                    f"energy={result.fun:.4f} success={result.success}"
                )

    return dofs_out, energies_out, nfev_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dat", help="starting .dat (e.g. systsearch-ens1.dat)")
    ap.add_argument("receptor_ens_list", help="ensemble list file")
    ap.add_argument("ligand_pdb", help="reduced ligand pdb")
    ap.add_argument("--grid", required=True, help="grid file (receptorgrid.grid)")
    ap.add_argument("--attract-par-npz", default="attract-par.npz")
    ap.add_argument("--epsilon", type=float, default=15.0)
    ap.add_argument("--cdie", action="store_true")
    ap.add_argument(
        "--maxfun", type=int, default=150, help="max energy evaluations per pose"
    )
    ap.add_argument(
        "--max-poses", type=int, default=0, help="cap number of poses (0 = all)"
    )
    ap.add_argument("--memory-gb", type=float, default=20.0)
    ap.add_argument("--disable-jit", action="store_true")
    ap.add_argument(
        "--trace-every", type=int, default=50, help="print progress every N poses"
    )
    ap.add_argument(
        "--grad-sign",
        type=float,
        default=1.0,
        help="multiply JAX gradient by this for optimizer (1.0 for scipy L-BFGS-B)",
    )
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--legacy-dat", help="optional legacy .dat for energy comparison")
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    if args.memory_gb > 0:
        mem_bytes = int(args.memory_gb * (1024**3))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        print(f"Applied RLIMIT_AS={args.memory_gb:.2f} GB")

    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
        print("JAX JIT disabled (--disable-jit)")

    print(f"JAX devices: {jax.devices()}")

    # Parse input
    header, pivots, ens, dofs0, _, centered_ligands = parse_dat_two_body(
        args.input_dat, max_poses=args.max_poses
    )

    # Resolve ligand pivot
    if 2 in pivots:
        lig_pivot = pivots[2]
    else:
        lig_coords0, _, _, _ = parse_reduced_pdb(args.ligand_pdb)
        lig_pivot = lig_coords0[lig_coords0[:, 0] != 99].mean(axis=0)
        print(f"No explicit #pivot 2; using ligand mean coordinate: {lig_pivot}")

    # Convert centered-ligand DOFs to internal convention
    input_centered = bool(centered_ligands) if centered_ligands is not None else False
    if input_centered:
        dofs0 = dofs0.copy()
        dofs0[:, 3:6] -= lig_pivot[None, :]
        print("Converted centered-ligand translations to internal convention")

    n = len(dofs0)
    print(f"Poses: {n}")

    # Build model
    par_npz = args.attract_par_npz
    if not Path(par_npz).exists():
        # Try relative to script
        par_npz2 = _jax_dir / "attract-par.npz"
        if par_npz2.exists():
            par_npz = str(par_npz2)
    model = GridEnergyModel(
        receptor_ens_list=args.receptor_ens_list,
        ligand_pdb=args.ligand_pdb,
        grid_file=args.grid,
        attract_par_npz=par_npz,
        epsilon=args.epsilon,
        cdie=bool(args.cdie),
        lig_pivot=lig_pivot,
    )

    # Minimize
    print(f"Starting L-BFGS-B minimization (maxfun={args.maxfun})...")
    t1 = time.time()
    dofs_min, energies, nfev = minimize_poses(
        model=model,
        ens=ens,
        dofs0=dofs0,
        maxfun=int(args.maxfun),
        grad_sign=float(args.grad_sign),
        trace_every=int(args.trace_every),
    )
    t2 = time.time()
    print(f"Minimization done in {t2 - t1:.1f}s ({n} poses, {t2 - t1:.3f}s total)")
    print(
        f"  nfev: mean={nfev.mean():.1f} median={np.median(nfev):.0f} "
        f"min={nfev.min()} max={nfev.max()}"
    )
    print(
        f"  energy: min={energies.min():.3f} mean={energies.mean():.3f} "
        f"p1={np.percentile(energies, 1):.3f} p10={np.percentile(energies, 10):.3f}"
    )

    # Convert DOFs back to centered-ligand convention for output
    dofs_out = dofs_min.copy()
    if input_centered:
        dofs_out[:, 3:6] += lig_pivot[None, :]

    # Compute 4x4 matrices
    mats = dofs_to_mats_np(dofs_min, lig_pivot)

    # Save outputs
    np.save(args.out_prefix + ".dofs_internal.npy", dofs_min.astype(np.float32))
    np.save(args.out_prefix + ".dofs.npy", dofs_out.astype(np.float32))
    np.save(args.out_prefix + ".mat4.npy", mats.astype(np.float32))
    np.save(args.out_prefix + ".energy.npy", energies.astype(np.float32))
    np.save(args.out_prefix + ".ens.npy", ens.astype(np.int32))
    np.save(args.out_prefix + ".nfev.npy", nfev.astype(np.int32))

    out_dat = args.out_prefix + ".dat"
    write_dat_two_body(out_dat, header, ens, dofs_out, energies=energies)
    print(f"Saved: {args.out_prefix}.[dofs|energy|mat4|ens|nfev].npy + {out_dat}")

    # Optional comparison with legacy
    if args.legacy_dat:
        _, piv_ref, ens_ref, dof_ref, e_ref, centered_ref = parse_dat_two_body(
            args.legacy_dat, max_poses=n
        )
        m = min(n, len(e_ref))
        if np.isfinite(e_ref[:m]).all():
            summarize("energy_vs_legacy", e_ref[:m], energies[:m])
            # Compare low-energy tails
            k = max(1, m // 100)
            leg_top = np.sort(e_ref[:m])[:k]
            jax_top = np.sort(energies[:m])[:k]
            print(
                f"Top 1% ({k} poses): legacy mean={leg_top.mean():.3f} jax mean={jax_top.mean():.3f}"
            )

    t3 = time.time()
    print(f"Total wall time: {t3 - t0:.1f}s")


if __name__ == "__main__":
    main()
