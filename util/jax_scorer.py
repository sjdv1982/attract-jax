#!/usr/bin/env python3
"""JAX-based energy oracle for the batched VA13 minimizer.

Drop-in replacement for LegacyScoreOracle.  Uses the ATTRACT-JAX grid
kernel (reproduce_grid_score.build_kernel) for the forward energy pass
only.  Gradients are computed via central finite differences (12 extra
forward evaluations per gradient for 6 DOFs).  This avoids the costly
jax.grad compilation through the neighbour-energy kernel.

Interface
---------
    oracle.score_batch(ens, dofs) -> (energies, gradients)
        ens  : (N,) int — 1-based ensemble indices
        dofs : (N, 6) float64 — (phi, ssi, rot, tx, ty, tz), non-centered
        Returns energies (N,) and gradients (N, 6), both float64.

    oracle.score_single(ens_id, dof) -> (float, ndarray(6,))
        Thin wrapper around score_batch for a single pose.
"""

import os

# Keep host memory bounded on CPU — set before importing jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Enable CPU multi-threading for XLA.  The --xla_cpu_multi_thread_eigen=false
# flag was historically used when running alongside other CPU-bound processes,
# but with the merged-ensemble oracle the per-call workload is large enough
# to benefit from thread-level parallelism.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_cpu_multi_thread_eigen=false" in _xla_flags:
    _xla_flags = _xla_flags.replace("--xla_cpu_multi_thread_eigen=false", "")
    if "--xla_cpu_multi_thread_eigen" not in _xla_flags:
        _xla_flags += " --xla_cpu_multi_thread_eigen=true"
    os.environ["XLA_FLAGS"] = _xla_flags.strip()

import math
import time
from collections import namedtuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from reproduce_grid_score import (
    build_kernel,
    parse_reduced_pdb,
    read_grid_with_electro,
)

# Finite-difference step sizes for each DOF (radians, radians, radians, Å, Å, Å)
FD_DELTA = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6], dtype=np.float64)

# Pre-defined batch sizes for JIT-shape caching.  Using a discrete set avoids
# recompilation for every unique active-pose count while keeping padding low.
_BATCH_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024)


def _round_up_batch(n: int) -> int:
    """Round *n* up to the next pre-defined batch size."""
    for s in _BATCH_SIZES:
        if s >= n:
            return s
    return ((n + 255) // 256) * 256


class JaxScoreOracle:
    """ATTRACT-JAX grid energy oracle with finite-difference gradients.

    Parameters
    ----------
    receptor_ens_list : str
        Path to text file listing receptor ensemble PDB paths (one per line).
    ligand_pdb : str
        Path to reduced ligand PDB.
    grid_file : str
        Path to binary .grid file (produced by make-grid).
    attract_par_npz : str
        Path to attract-par.npz (pre-computed from attract.par).
    lig_pivot : ndarray (3,)
        Ligand pivot point.
    epsilon : float
        Dielectric constant (default 15.0).
    cdie : bool
        Use distance-dependent dielectric (default False).
    energy_batch : int
        Max poses per JAX kernel call (controls peak memory).
    max_nb_cap : int
        Cap max_nr_neighbours to this value (0 = no cap).
        Reduces computation for high-neighbour voxels at the cost of
        accuracy (missing some pair corrections).
    """

    def __init__(
        self,
        receptor_ens_list: str,
        ligand_pdb: str,
        grid_file: str,
        attract_par_npz: str,
        lig_pivot: np.ndarray,
        epsilon: float = 15.0,
        cdie: bool = False,
        energy_batch: int = 256,
        max_nb_cap: int = 0,
    ):
        self.energy_batch = int(max(1, energy_batch))
        self._call_count = 0
        self._total_kernel_calls = 0
        self._total_kernel_time = 0.0

        # --- Load receptor ensemble ---
        with open(receptor_ens_list) as f:
            rec_files = [line.strip() for line in f if line.strip()]
        if not rec_files:
            raise ValueError("Empty receptor ensemble list")

        list_dir = Path(receptor_ens_list).resolve().parent
        rec_coords_all, rec_types_all, rec_charge_all = [], [], []
        for rf in rec_files:
            p = Path(rf)
            if not p.is_absolute():
                p = list_dir / p
            c, a, q, _w = parse_reduced_pdb(str(p))
            rec_coords_all.append(c)
            rec_types_all.append(a)
            rec_charge_all.append(q)

        # --- Load ligand ---
        lig_coords0, lig_types0, lig_charge0, _lig_w = parse_reduced_pdb(ligand_pdb)

        # Mask out dummy atoms (type 99)
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

        # --- Electrostatics scaling ---
        felec = math.sqrt(332.053986 / epsilon)
        rec_charge_ens_scaled = rec_charge_ens_raw * felec
        lig_charge_scaled = lig_charge_raw * felec

        # --- Force field ---
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

        # --- Grid ---
        grid = read_grid_with_electro(Path(grid_file).read_bytes())

        # Remap neighbour indices through the rec_mask mapping
        rec_mapping = np.cumsum(rec_mask) - 1
        nb_flat = grid.neighbour_grid.reshape(-1)
        valid = nb_flat < 2**16 - 1
        nb_flat[valid] = rec_mapping[nb_flat[valid]]

        # Map ligand atom types to grid VDW channels
        alpos = grid.alphabet_atomtypes.tolist()
        lig_vdw_channel_idx = np.array(
            [alpos.index(a) for a in lig_alphabet], dtype=np.int32
        )[lig_atomtypes_ff]

        # Combine potential + electrostatic grids
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

        # Neighbour chunk thresholds
        nb_chunk_thresholds = (0, 1, 2, 3, 4, 5, 10, 15, 20)
        for n0 in range(nb_chunk_thresholds[-1] + 10, int(grid.max_nr_neighbours), 10):
            nb_chunk_thresholds += (n0,)
        nb_chunk_thresholds += (int(grid.max_nr_neighbours),)
        grid_dim = tuple(int(x) for x in grid.dim)

        # --- Build JAX kernel (forward pass only, no jax.grad) ---
        n_lig_atoms = int(lig_coords.shape[0])
        padded_nb_size = energy_batch * n_lig_atoms

        kernel_main = build_kernel(
            grid=grid_j,
            ff=ff,
            lig_atomtypes_ff=jnp.array(lig_atomtypes_ff, dtype=np.int32),
            lig_vdw_channel_idx=jnp.array(lig_vdw_channel_idx, dtype=np.int32),
            lig_charge_raw=jnp.array(lig_charge_raw, dtype=np.float64),
            lig_charge_scaled=jnp.array(lig_charge_scaled, dtype=np.float64),
            cdie=bool(cdie),
            padded_nb_size=padded_nb_size,
            max_nb_cap=int(max_nb_cap),
        )

        # Store per-ensemble data
        self._kernel_main = kernel_main
        self._rec_coor_ens = rec_coords_ens
        self._rec_charge_ens_scaled = rec_charge_ens_scaled
        # Pre-convert receptor data to JAX arrays (avoid per-call conversion)
        self._rec_coor_ens_j = [
            jnp.array(rec_coords_ens[i], dtype=jnp.float64)
            for i in range(rec_coords_ens.shape[0])
        ]
        self._rec_charge_ens_scaled_j = [
            jnp.array(rec_charge_ens_scaled[i], dtype=jnp.float64)
            for i in range(rec_charge_ens_scaled.shape[0])
        ]
        # Stacked numpy arrays for the merged-ensemble score_batch path.
        # Indexing with ens-1 gives per-pose receptor data without a Python loop.
        self._rec_coor_ens_np = rec_coords_ens  # (n_ens, Nrec, 3)
        self._rec_charge_ens_np = rec_charge_ens_scaled  # (n_ens, Nrec)
        self._rec_atomtypes_ff_j = jnp.array(rec_atomtypes_ff, dtype=np.int32)
        self._coor_lig_j = jnp.array(lig_coords, dtype=jnp.float64)
        self._lig_atomtypes_ff_j = jnp.array(lig_atomtypes_ff, dtype=np.int32)
        self._lig_vdw_channel_idx_j = jnp.array(lig_vdw_channel_idx, dtype=np.int32)
        self._lig_charge_raw_j = jnp.array(lig_charge_raw, dtype=jnp.float64)
        self._lig_charge_scaled_j = jnp.array(lig_charge_scaled, dtype=jnp.float64)
        self._ff = ff
        self._grid_j = grid_j
        self._nb_chunk_thresholds = nb_chunk_thresholds
        self._grid_dim = grid_dim
        self._lig_pivot_j = jnp.array(lig_pivot, dtype=jnp.float64)
        self._pad_poses = int(energy_batch)
        self._n_lig_atoms = n_lig_atoms

        # --- Build value_and_grad function for analytical gradients ---
        # Uses main_ad (fully JIT-compilable, no Python control flow).
        kernel_ad = kernel_main.ad
        if kernel_ad is not None:

            def _single_energy(dof_1d, rec_coor, rec_charge_scaled):
                """Energy of a single pose (6,) → scalar."""
                dof_2d = dof_1d[None, :]  # (1, 6)
                _, per_pose = kernel_ad(
                    dof_2d,
                    rec_coor,
                    self._rec_atomtypes_ff_j,
                    rec_charge_scaled,
                    self._coor_lig_j,
                    self._lig_atomtypes_ff_j,
                    self._lig_vdw_channel_idx_j,
                    self._lig_charge_raw_j,
                    self._lig_charge_scaled_j,
                    self._ff,
                    self._grid_j,
                    self._lig_pivot_j,
                )
                return per_pose[0]

            _vg_single = jax.value_and_grad(_single_energy)
            # vmap over (dof, rec_coor, rec_charge): each pose carries its own
            # receptor data.  At large scale (165k poses) this avoids the
            # per-ensemble Python dispatch overhead (np.where + scatter-back).
            _vg_batch = jax.jit(jax.vmap(_vg_single, in_axes=(0, 0, 0)))
            self._vg_batch = _vg_batch
        else:
            self._vg_batch = None

    def _vg_ensemble(self, ens0: int, dofs_j: jnp.ndarray):
        """Compute per-pose energies AND gradients for one ensemble.

        Uses jax.value_and_grad (analytical) — no finite differences.
        Broadcasts shared receptor data to match the per-pose vmap signature.

        Parameters
        ----------
        ens0 : int  — 0-based ensemble index
        dofs_j : (M, 6) jnp.float64

        Returns
        -------
        energies  : (M,) float64
        gradients : (M, 6) float64
        """
        M = dofs_j.shape[0]
        pad_n = _round_up_batch(M)

        if M < pad_n:
            dofs_j = jnp.pad(dofs_j, ((0, pad_n - M), (0, 0)))

        # Broadcast shared rec data to (pad_n, ...) for the per-pose vmap
        rec_coor_j = jnp.broadcast_to(
            self._rec_coor_ens_j[ens0][None, :, :],
            (pad_n,) + self._rec_coor_ens_j[ens0].shape,
        )
        rec_charge_j = jnp.broadcast_to(
            self._rec_charge_ens_scaled_j[ens0][None, :],
            (pad_n,) + self._rec_charge_ens_scaled_j[ens0].shape,
        )

        t0 = time.monotonic()
        energies, grads = self._vg_batch(dofs_j, rec_coor_j, rec_charge_j)
        energies.block_until_ready()
        self._total_kernel_time += time.monotonic() - t0
        self._total_kernel_calls += 1

        return np.asarray(energies[:M]), np.asarray(grads[:M])

    def _energy_ensemble(self, ens0: int, dofs_j: jnp.ndarray) -> np.ndarray:
        """Compute per-pose energies for one ensemble (forward pass only).

        Parameters
        ----------
        ens0 : int  — 0-based ensemble index
        dofs_j : (M, 6) jnp.float64

        Returns
        -------
        energies : (M,) float64
        """
        M = dofs_j.shape[0]

        # Pad to energy_batch so all JIT functions see fixed shapes
        if M < self._pad_poses:
            dofs_j = jnp.pad(dofs_j, ((0, self._pad_poses - M), (0, 0)))

        rec_coor_j = self._rec_coor_ens_j[ens0]
        rec_charge_j = self._rec_charge_ens_scaled_j[ens0]

        t0 = time.monotonic()
        _, per_pose_e = self._kernel_main(
            dofs_j,
            rec_coor_j,
            self._rec_atomtypes_ff_j,
            rec_charge_j,
            self._coor_lig_j,
            self._lig_atomtypes_ff_j,
            self._lig_vdw_channel_idx_j,
            self._lig_charge_raw_j,
            self._lig_charge_scaled_j,
            self._ff,
            self._grid_j,
            self._nb_chunk_thresholds,
            self._grid_dim,
            self._lig_pivot_j,
        )
        per_pose_e.block_until_ready()
        self._total_kernel_time += time.monotonic() - t0
        self._total_kernel_calls += 1

        return np.asarray(per_pose_e[:M])

    def _energy_batch_raw(self, ens: np.ndarray, dofs: np.ndarray) -> np.ndarray:
        """Compute energies for a mixed-ensemble batch.

        Parameters
        ----------
        ens  : (N,) int — 1-based ensemble indices
        dofs : (N, 6) float64

        Returns
        -------
        energies : (N,) float64
        """
        n = len(dofs)
        energies = np.zeros(n, dtype=np.float64)

        for ens_id in np.unique(ens):
            ens0 = int(ens_id) - 1
            idx = np.where(ens == ens_id)[0]

            for start in range(0, len(idx), self.energy_batch):
                sub = idx[start : start + self.energy_batch]
                dof_b = jnp.array(dofs[sub], dtype=jnp.float64)
                energies[sub] = self._energy_ensemble(ens0, dof_b)

        return energies

    def score_batch(self, ens, dofs):
        """Score a batch of poses with analytical gradients (jax.value_and_grad).

        Merged-ensemble approach: per-pose receptor data is gathered from
        stacked numpy arrays per chunk, so all ensembles are processed in
        a single sequential loop without per-ensemble Python dispatch.

        Parameters
        ----------
        ens  : (N,) int — 1-based ensemble indices
        dofs : (N, 6) float64

        Returns
        -------
        energies  : (N,) float64
        gradients : (N, 6) float64
        """
        self._call_count += 1
        n = len(dofs)
        ens0 = np.asarray(ens, dtype=np.intp) - 1  # (N,) 0-based

        energies = np.zeros(n, dtype=np.float64)
        gradients = np.zeros((n, 6), dtype=np.float64)

        chunk = self.energy_batch
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            m = end - start
            pad_n = _round_up_batch(m)

            dofs_j = jnp.array(dofs[start:end], dtype=jnp.float64)
            rc_j = jnp.array(self._rec_coor_ens_np[ens0[start:end]], dtype=jnp.float64)
            rq_j = jnp.array(
                self._rec_charge_ens_np[ens0[start:end]], dtype=jnp.float64
            )

            if m < pad_n:
                dofs_j = jnp.pad(dofs_j, ((0, pad_n - m), (0, 0)))
                rc_j = jnp.pad(rc_j, ((0, pad_n - m), (0, 0), (0, 0)))
                rq_j = jnp.pad(rq_j, ((0, pad_n - m), (0, 0)))

            t0 = time.monotonic()
            e_b, g_b = self._vg_batch(dofs_j, rc_j, rq_j)
            e_b.block_until_ready()
            self._total_kernel_time += time.monotonic() - t0
            self._total_kernel_calls += 1

            energies[start:end] = np.asarray(e_b[:m])
            gradients[start:end] = np.asarray(g_b[:m])

        return energies, gradients

    def score_single(self, ens_id, dof):
        """Score a single pose."""
        e, g = self.score_batch(np.array([ens_id], dtype=np.int32), dof.reshape(1, 6))
        return float(e[0]), g[0]

    def print_stats(self):
        """Print kernel timing statistics."""
        if self._total_kernel_calls > 0:
            avg = self._total_kernel_time / self._total_kernel_calls * 1000
            print(
                f"JAX kernel stats: {self._total_kernel_calls} calls, "
                f"{self._total_kernel_time:.1f}s total, {avg:.1f}ms avg"
            )
