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
import ctypes
import subprocess
import sys

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
from typing import Optional

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


class NbFusedStepData(ctypes.Structure):
    _fields_ = [
        ("nposes", ctypes.c_int32),
        ("natoms", ctypes.c_int32),
        ("dofs", ctypes.POINTER(ctypes.c_double)),
        ("ens", ctypes.POINTER(ctypes.c_int16)),
        ("lig_coords", ctypes.POINTER(ctypes.c_double)),
        ("lig_pivot", ctypes.POINTER(ctypes.c_double)),
        ("lig_type", ctypes.POINTER(ctypes.c_int16)),
        ("lig_charge", ctypes.POINTER(ctypes.c_double)),
    ]


class NbFusedGridData(ctypes.Structure):
    _fields_ = [
        ("dim", ctypes.c_int32 * 3),
        ("origin", ctypes.c_double * 3),
        ("spacing", ctypes.c_double),
        ("nr_neigh", ctypes.POINTER(ctypes.c_int32)),
        ("nb_start", ctypes.POINTER(ctypes.c_int64)),
    ]


class NbGlobalData(ctypes.Structure):
    _fields_ = [
        ("nrec", ctypes.c_int32),
        ("nens", ctypes.c_int32),
        ("nb_concat_len", ctypes.c_int64),
        ("nb_concat", ctypes.POINTER(ctypes.c_int32)),
        ("rec_coord", ctypes.POINTER(ctypes.c_double)),
        ("rec_type", ctypes.POINTER(ctypes.c_int16)),
        ("rec_charge", ctypes.POINTER(ctypes.c_double)),
        ("nrec_types", ctypes.c_int32),
        ("nlig_types", ctypes.c_int32),
        ("rc", ctypes.POINTER(ctypes.c_double)),
        ("ac", ctypes.POINTER(ctypes.c_double)),
        ("emin", ctypes.POINTER(ctypes.c_double)),
        ("rmin2", ctypes.POINTER(ctypes.c_double)),
        ("ivor", ctypes.POINTER(ctypes.c_int8)),
        ("plateaudissq", ctypes.c_double),
    ]


class NbRunConfig(ctypes.Structure):
    _fields_ = [("num_threads", ctypes.c_int32), ("kernel_variant", ctypes.c_int32)]


def _as_ptr(a, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


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
    grid_file : str or None
        Path to binary .grid file (produced by make-grid).  Mutually exclusive
        with ``grid_object``; exactly one must be provided.
    attract_par_npz : str
        Path to attract-par.npz (pre-computed from attract.par).
    lig_pivot : ndarray (3,)
        Ligand pivot point.
    epsilon : float
        Dielectric constant (default 15.0).
    energy_batch : int
        Max poses per JAX kernel call (controls peak memory).
    grid_object : Grid namedtuple or None
        Pre-built Grid (e.g. from ``generate_grid()``).  If provided,
        ``grid_file`` is ignored.  Exactly one of ``grid_file`` or
        ``grid_object`` must be supplied.
    """

    def __init__(
        self,
        receptor_ens_list: str,
        ligand_pdb: str,
        grid_file: Optional[str] = None,
        attract_par_npz: str = "",
        lig_pivot: np.ndarray = None,
        epsilon: float = 15.0,
        cdie: bool = False,
        energy_batch: int = 256,
        nb_kernel: str = "jax",
        autodiff_potentials: bool = False,
        energy_only: bool = False,
        grid_object=None,
    ):
        _ = cdie  # Milestone 1: nonbon8 + rdie fixed, keep arg for compatibility.
        self.energy_batch = int(max(1, energy_batch))
        self._call_count = 0
        self._total_kernel_calls = 0
        self._total_kernel_time = 0.0
        self._energy_only = bool(energy_only)

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
        if grid_object is not None:
            # Use pre-built Grid (e.g. from generate_grid())
            grid = grid_object
        elif grid_file is not None:
            grid = read_grid_with_electro(Path(grid_file).read_bytes())
        else:
            raise ValueError(
                "JaxScoreOracle: exactly one of 'grid_file' or 'grid_object' must be provided"
            )

        # Remap neighbour indices through the rec_mask mapping
        # (Only needed when grid was loaded from a binary file and neighbour
        # indices reference the original unreduced receptor atom ordering.
        # When grid_object is provided the indices already address the filtered array.)
        if grid_object is None:
            rec_mapping = np.cumsum(rec_mask) - 1
            nb_flat = grid.neighbour_grid.reshape(-1)
            valid = nb_flat < 2**16 - 1
            nb_flat[valid] = rec_mapping[nb_flat[valid]]

        # Map ligand atom types to grid VDW channels
        alpos = grid.alphabet_atomtypes.tolist()
        lig_vdw_channel_idx = np.array(
            [alpos.index(a) for a in lig_alphabet], dtype=np.int32
        )[lig_atomtypes_ff]

        self._nb_kernel = str(nb_kernel)
        if self._nb_kernel not in ("jax", "nonbon8"):
            raise ValueError(f"Unsupported nb_kernel={self._nb_kernel!r}")
        # Default behavior: use stored grid gradients via custom JVP.
        # Optional behavior: AD through energy-only potentials.
        self._use_precomputed_grid_gradients = not bool(autodiff_potentials)

        # Combine potential + electrostatic grids.
        # For nonbon8 mode we only keep energy channels; JAX AD handles grid gradients.
        inner_all = np.concatenate(
            (grid.inner_potential_grid, grid.inner_elec_grid[None, ...]), axis=0
        )
        outer_all = np.concatenate(
            (grid.outer_potential_grid, grid.outer_elec_grid[None, ...]), axis=0
        )
        if not self._use_precomputed_grid_gradients:
            inner_all = inner_all[..., :1]
            outer_all = outer_all[..., :1]
        elec_channel_index = inner_all.shape[0] - 1

        dgrid = {}
        for field in grid._fields:
            value = getattr(grid, field)
            if isinstance(value, np.ndarray) and value.shape != (3,):
                value = jnp.array(value, dtype=value.dtype)
            dgrid[field] = value
        # Handle zero-neighbour grids by inserting a 1-wide sentinel dimension.
        # This keeps reshape/indexing code valid while nr_neighbours==0 ensures
        # the NB contribution remains exactly zero.
        if int(grid.neighbour_grid.shape[-1]) == 0:
            ng_shape = tuple(int(x) for x in grid.neighbour_grid.shape[:3]) + (1,)
            dgrid["neighbour_grid"] = jnp.full(ng_shape, 2**16 - 1, dtype=jnp.uint16)
        dgrid["inner_potential_grid_all"] = jnp.array(inner_all, dtype=np.float32)
        dgrid["outer_potential_grid_all"] = jnp.array(outer_all, dtype=np.float32)
        dgrid["elec_channel_index"] = np.int32(elec_channel_index)
        ng_width = int(dgrid["neighbour_grid"].shape[-1])
        dgrid["neighbour_grid_ravel"] = dgrid["neighbour_grid"].reshape(-1, ng_width)
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
            cdie=False,
            padded_nb_size=padded_nb_size,
            use_precomputed_grid_gradients=self._use_precomputed_grid_gradients,
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
        self._nens = int(rec_coords_ens.shape[0])
        self._nrec = int(rec_coords_ens.shape[1])
        # --- Build value_and_grad function for analytical gradients ---
        # Uses main_ad (fully JIT-compilable, no Python control flow).
        kernel_ad = kernel_main.ad
        kernel_pot_ad = kernel_main.pot_ad
        self._kernel_pot_ad = kernel_pot_ad
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
            self._vg_single = jax.jit(_vg_single)

            def _single_potential_energy(dof_1d, rec_coor, rec_charge_scaled):
                dof_2d = dof_1d[None, :]
                _, per_pose = kernel_pot_ad(
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

            _pot_vg_single = jax.value_and_grad(_single_potential_energy)
            self._pot_vg_single = jax.jit(_pot_vg_single)
            self._pot_e_single = jax.jit(_single_potential_energy)
            # vmap over (dof, rec_coor, rec_charge): each pose carries its own
            # receptor data. At large scale this avoids per-ensemble Python dispatch.
            _vg_batch = jax.jit(jax.vmap(_vg_single, in_axes=(0, 0, 0)))
            self._vg_batch = _vg_batch
            _pot_vg_batch = jax.jit(jax.vmap(_pot_vg_single, in_axes=(0, 0, 0)))
            self._pot_vg_batch = _pot_vg_batch
            _pot_e_batch = jax.jit(jax.vmap(self._pot_e_single, in_axes=(0, 0, 0)))
            self._pot_e_batch = _pot_e_batch
        else:
            self._vg_single = None
            self._vg_batch = None
            self._pot_vg_single = None
            self._pot_vg_batch = None
            self._pot_e_single = None
            self._pot_e_batch = None

        if self._nb_kernel == "nonbon8":
            self._init_nonbon8_backend(
                grid=grid,
                rec_coords_ens=rec_coords_ens,
                rec_charge_ens_scaled=rec_charge_ens_scaled,
                rec_atomtypes_ff=rec_atomtypes_ff.astype(np.int16, copy=False),
                lig_coords=lig_coords,
                lig_atomtypes_ff=lig_atomtypes_ff.astype(np.int16, copy=False),
                lig_charge_scaled=lig_charge_scaled,
                rc=rc,
                ac=ac,
                emin=emin,
                rmin2=rmin2,
                ivor=ivor,
                plateaudissq=float(grid.plateaudis) ** 2,
            )

    def _init_nonbon8_backend(
        self,
        grid,
        rec_coords_ens,
        rec_charge_ens_scaled,
        rec_atomtypes_ff,
        lig_coords,
        lig_atomtypes_ff,
        lig_charge_scaled,
        rc,
        ac,
        emin,
        rmin2,
        ivor,
        plateaudissq,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        kernel_root = repo_root / "native" / "nb_kernel"
        if not kernel_root.exists():
            raise RuntimeError(f"Missing kernel root: {kernel_root}")

        if str(kernel_root) not in sys.path:
            sys.path.insert(0, str(kernel_root))
        from forcefields import bind_kernel_dispatch, find_kernel_so, load_forcefield

        ff_module = load_forcefield("forcefields.nonbon8")
        lib = find_kernel_so(ff_module)
        so = kernel_root / "nb_kernel_nonbon8.so"
        if lib is None:
            if not so.exists():
                subprocess.run(
                    ["make", "-C", str(kernel_root), "nb_kernel_nonbon8.so"],
                    check=True,
                )
            lib = ctypes.CDLL(str(so))

        dispatch, fn_nb_grad, fn_nb_energy = bind_kernel_dispatch(
            lib, rotation="euler", plateau_mode="correction"
        )
        if fn_nb_grad is not None:
            fn_nb_grad.argtypes = [
                ctypes.POINTER(NbFusedStepData),
                ctypes.POINTER(NbFusedGridData),
                ctypes.POINTER(NbGlobalData),
                ctypes.POINTER(NbRunConfig),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            fn_nb_grad.restype = ctypes.c_int
        fn_nb_energy.argtypes = [
            ctypes.POINTER(NbFusedStepData),
            ctypes.POINTER(NbFusedGridData),
            ctypes.POINTER(NbGlobalData),
            ctypes.POINTER(NbRunConfig),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        fn_nb_energy.restype = ctypes.c_int

        ng = np.asarray(grid.neighbour_grid, dtype=np.int32)
        nr_neigh = np.asarray(grid.nr_neighbours, dtype=np.int32).reshape(-1)
        nr_neigh = np.where(nr_neigh > 0, nr_neigh, 0).astype(np.int32, copy=False)
        ng_flat = ng.reshape(-1, ng.shape[-1])
        nb_start = np.zeros((nr_neigh.shape[0],), dtype=np.int64)
        if nr_neigh.shape[0] > 1:
            nb_start[1:] = np.cumsum(nr_neigh[:-1], dtype=np.int64)
        total_nb = int(nb_start[-1] + nr_neigh[-1]) if nr_neigh.size else 0
        nb_concat = np.empty((total_nb,), dtype=np.int32)
        for i in range(nr_neigh.shape[0]):
            k = int(nr_neigh[i])
            if k > 0:
                s = int(nb_start[i])
                nb_concat[s : s + k] = ng_flat[i, :k]

        self._nb = {
            "lib": lib,
            "dispatch": dispatch,
            "fn_grad": fn_nb_grad,
            "fn_energy": fn_nb_energy,
            "grid_nr_neigh": np.ascontiguousarray(nr_neigh, dtype=np.int32),
            "grid_nb_start": np.ascontiguousarray(nb_start, dtype=np.int64),
            "nb_concat": np.ascontiguousarray(nb_concat, dtype=np.int32),
            "rec_coord": np.ascontiguousarray(rec_coords_ens, dtype=np.float64),
            "rec_type": np.ascontiguousarray(rec_atomtypes_ff, dtype=np.int16),
            "rec_charge": np.ascontiguousarray(rec_charge_ens_scaled, dtype=np.float64),
            "rc": np.ascontiguousarray(rc, dtype=np.float64),
            "ac": np.ascontiguousarray(ac, dtype=np.float64),
            "emin": np.ascontiguousarray(emin, dtype=np.float64),
            "rmin2": np.ascontiguousarray(rmin2, dtype=np.float64),
            "ivor": np.ascontiguousarray(ivor, dtype=np.int8),
            "lig_coord": np.ascontiguousarray(lig_coords, dtype=np.float64),
            "lig_type": np.ascontiguousarray(lig_atomtypes_ff, dtype=np.int16),
            "lig_charge": np.ascontiguousarray(lig_charge_scaled, dtype=np.float64),
            "plateaudissq": float(plateaudissq),
        }

        dim = np.asarray(grid.dim, dtype=np.int32)
        origin = np.asarray(grid.origin, dtype=np.float64)
        self._nb_grid_struct = NbFusedGridData(
            dim=(ctypes.c_int32 * 3)(int(dim[0]), int(dim[1]), int(dim[2])),
            origin=(ctypes.c_double * 3)(
                float(origin[0]), float(origin[1]), float(origin[2])
            ),
            spacing=ctypes.c_double(float(grid.gridspacing)),
            nr_neigh=_as_ptr(self._nb["grid_nr_neigh"], ctypes.c_int32),
            nb_start=_as_ptr(self._nb["grid_nb_start"], ctypes.c_int64),
        )

        self._nb_global_struct = NbGlobalData(
            nrec=np.int32(self._nrec),
            nens=np.int32(self._nens),
            nb_concat_len=np.int64(self._nb["nb_concat"].shape[0]),
            nb_concat=_as_ptr(self._nb["nb_concat"], ctypes.c_int32),
            rec_coord=_as_ptr(self._nb["rec_coord"].reshape(-1), ctypes.c_double),
            rec_type=_as_ptr(self._nb["rec_type"], ctypes.c_int16),
            rec_charge=_as_ptr(self._nb["rec_charge"].reshape(-1), ctypes.c_double),
            nrec_types=np.int32(self._nb["rc"].shape[0]),
            nlig_types=np.int32(self._nb["rc"].shape[1]),
            rc=_as_ptr(self._nb["rc"].reshape(-1), ctypes.c_double),
            ac=_as_ptr(self._nb["ac"].reshape(-1), ctypes.c_double),
            emin=_as_ptr(self._nb["emin"].reshape(-1), ctypes.c_double),
            rmin2=_as_ptr(self._nb["rmin2"].reshape(-1), ctypes.c_double),
            ivor=_as_ptr(self._nb["ivor"].reshape(-1), ctypes.c_int8),
            plateaudissq=ctypes.c_double(self._nb["plateaudissq"]),
        )
        nthreads = int(os.environ.get("OMP_NUM_THREADS", "0")) or (os.cpu_count() or 1)
        self._nb_cfg = NbRunConfig(
            num_threads=np.int32(nthreads), kernel_variant=np.int32(1)
        )

    def _run_nb_energy_kernel(self, ens0: np.ndarray, dofs64: np.ndarray, out_e: np.ndarray):
        pivot = np.ascontiguousarray(np.asarray(self._lig_pivot_j), dtype=np.float64)
        n = int(dofs64.shape[0])
        step = NbFusedStepData(
            nposes=np.int32(n),
            natoms=np.int32(self._nb["lig_coord"].shape[0]),
            dofs=_as_ptr(dofs64.reshape(-1), ctypes.c_double),
            ens=_as_ptr(ens0, ctypes.c_int16),
            lig_coords=_as_ptr(self._nb["lig_coord"].reshape(-1), ctypes.c_double),
            lig_pivot=_as_ptr(pivot, ctypes.c_double),
            lig_type=_as_ptr(self._nb["lig_type"], ctypes.c_int16),
            lig_charge=_as_ptr(self._nb["lig_charge"], ctypes.c_double),
        )
        t0 = time.monotonic()
        rc = self._nb["fn_energy"](
            ctypes.byref(step),
            ctypes.byref(self._nb_grid_struct),
            ctypes.byref(self._nb_global_struct),
            ctypes.byref(self._nb_cfg),
            _as_ptr(out_e, ctypes.c_double),
            None,
        )
        self._total_kernel_time += time.monotonic() - t0
        self._total_kernel_calls += 1
        if rc != 0:
            energy_symbol = self._nb["dispatch"].energy_symbol
            raise RuntimeError(f"{energy_symbol} failed with error code {rc}")

    def _score_nb_nonbon8_batch(self, ens, dofs, compute_grad: bool = True):
        ens0 = np.ascontiguousarray(np.asarray(ens, dtype=np.int16) - 1, dtype=np.int16)
        dofs64 = np.ascontiguousarray(
            np.asarray(dofs, dtype=np.float64), dtype=np.float64
        )
        n = int(dofs64.shape[0])
        out_e = np.zeros((n,), dtype=np.float64)
        out_g = np.zeros((n, 6), dtype=np.float64)

        if not compute_grad:
            self._run_nb_energy_kernel(ens0, dofs64, out_e)
            return out_e, out_g

        pivot = np.ascontiguousarray(np.asarray(self._lig_pivot_j), dtype=np.float64)
        step = NbFusedStepData(
            nposes=np.int32(n),
            natoms=np.int32(self._nb["lig_coord"].shape[0]),
            dofs=_as_ptr(dofs64.reshape(-1), ctypes.c_double),
            ens=_as_ptr(ens0, ctypes.c_int16),
            lig_coords=_as_ptr(self._nb["lig_coord"].reshape(-1), ctypes.c_double),
            lig_pivot=_as_ptr(pivot, ctypes.c_double),
            lig_type=_as_ptr(self._nb["lig_type"], ctypes.c_int16),
            lig_charge=_as_ptr(self._nb["lig_charge"], ctypes.c_double),
        )
        family = self._nb["dispatch"].family

        if family == "grad_energy":
            t0 = time.monotonic()
            rc = self._nb["fn_grad"](
                ctypes.byref(step),
                ctypes.byref(self._nb_grid_struct),
                ctypes.byref(self._nb_global_struct),
                ctypes.byref(self._nb_cfg),
                _as_ptr(out_e, ctypes.c_double),
                _as_ptr(out_g.reshape(-1), ctypes.c_double),
            )
            self._total_kernel_time += time.monotonic() - t0
            self._total_kernel_calls += 1
            if rc != 0:
                grad_symbol = self._nb["dispatch"].grad_symbol or "nb_kernel_grad"
                raise RuntimeError(f"{grad_symbol} failed with error code {rc}")
            return out_e, out_g

        # Energy-only kernel fallback: approximate NB gradients by central FD.
        self._run_nb_energy_kernel(ens0, dofs64, out_e)
        for d in range(6):
            dofs_plus = dofs64.copy()
            dofs_minus = dofs64.copy()
            dofs_plus[:, d] += FD_DELTA[d]
            dofs_minus[:, d] -= FD_DELTA[d]

            e_plus = np.zeros((n,), dtype=np.float64)
            e_minus = np.zeros((n,), dtype=np.float64)
            self._run_nb_energy_kernel(ens0, dofs_plus, e_plus)
            self._run_nb_energy_kernel(ens0, dofs_minus, e_minus)
            out_g[:, d] = (e_plus - e_minus) / (2.0 * FD_DELTA[d])

        return out_e, out_g

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
        """Score a batch of poses using the selected NB backend.

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
        if self._energy_only:
            if self._nb_kernel == "nonbon8":
                e_pot = self.score_potential_energy_batch(ens, dofs)
                e_nb, _ = self._score_nb_nonbon8_batch(ens, dofs, compute_grad=False)
                gradients = np.zeros((len(dofs), 6), dtype=np.float64)
                return e_pot + e_nb, gradients

            energies = self._energy_batch_raw(np.asarray(ens), np.asarray(dofs))
            gradients = np.zeros((len(dofs), 6), dtype=np.float64)
            return energies, gradients

        if self._nb_kernel == "nonbon8":
            e_pot, g_pot = self.score_potential_batch(ens, dofs)
            e_nb, g_nb = self._score_nb_nonbon8_batch(ens, dofs)
            return e_pot + e_nb, g_pot + g_nb

        # Pure-JAX path (potential + NB in a single AD-evaluated kernel).
        n = len(dofs)
        ens0 = np.asarray(ens, dtype=np.intp) - 1
        energies = np.zeros(n, dtype=np.float64)
        gradients = np.zeros((n, 6), dtype=np.float64)
        chunk = self.energy_batch
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            m = end - start
            dofs_j = jnp.array(dofs[start:end], dtype=jnp.float64)
            rc_j = jnp.array(self._rec_coor_ens_np[ens0[start:end]], dtype=jnp.float64)
            rq_j = jnp.array(
                self._rec_charge_ens_np[ens0[start:end]], dtype=jnp.float64
            )
            t0 = time.monotonic()
            pad_n = _round_up_batch(m)
            if m < pad_n:
                dofs_j = jnp.pad(dofs_j, ((0, pad_n - m), (0, 0)))
                rc_j = jnp.pad(rc_j, ((0, pad_n - m), (0, 0), (0, 0)))
                rq_j = jnp.pad(rq_j, ((0, pad_n - m), (0, 0)))
            e_b, g_b = self._vg_batch(dofs_j, rc_j, rq_j)
            e_b.block_until_ready()
            energies[start:end] = np.asarray(e_b[:m])
            gradients[start:end] = np.asarray(g_b[:m])
            self._total_kernel_time += time.monotonic() - t0
            self._total_kernel_calls += 1
        return energies, gradients

    def score_potential_energy_batch(self, ens, dofs):
        """Score potential-grid contribution only (energy-only)."""
        n = len(dofs)
        ens0 = np.asarray(ens, dtype=np.intp) - 1
        energies = np.zeros(n, dtype=np.float64)

        chunk = self.energy_batch
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            m = end - start
            dofs_j = jnp.array(dofs[start:end], dtype=jnp.float64)
            rc_j = jnp.array(self._rec_coor_ens_np[ens0[start:end]], dtype=jnp.float64)
            rq_j = jnp.array(
                self._rec_charge_ens_np[ens0[start:end]], dtype=jnp.float64
            )
            pad_n = _round_up_batch(m)
            if m < pad_n:
                dofs_j = jnp.pad(dofs_j, ((0, pad_n - m), (0, 0)))
                rc_j = jnp.pad(rc_j, ((0, pad_n - m), (0, 0), (0, 0)))
                rq_j = jnp.pad(rq_j, ((0, pad_n - m), (0, 0)))
            t0 = time.monotonic()
            _, e_b = self._kernel_pot_ad(
                dofs_j,
                rc_j,
                self._rec_atomtypes_ff_j,
                rq_j,
                self._coor_lig_j,
                self._lig_atomtypes_ff_j,
                self._lig_vdw_channel_idx_j,
                self._lig_charge_raw_j,
                self._lig_charge_scaled_j,
                self._ff,
                self._grid_j,
                self._lig_pivot_j,
            )
            e_b.block_until_ready()
            self._total_kernel_time += time.monotonic() - t0
            self._total_kernel_calls += 1
            energies[start:end] = np.asarray(e_b[:m])

        return energies

    def score_potential_batch(self, ens, dofs):
        """Score potential-grid contribution only (no NB correction)."""
        n = len(dofs)
        ens0 = np.asarray(ens, dtype=np.intp) - 1
        energies = np.zeros(n, dtype=np.float64)
        gradients = np.zeros((n, 6), dtype=np.float64)

        chunk = self.energy_batch
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            m = end - start
            dofs_j = jnp.array(dofs[start:end], dtype=jnp.float64)
            rc_j = jnp.array(self._rec_coor_ens_np[ens0[start:end]], dtype=jnp.float64)
            rq_j = jnp.array(
                self._rec_charge_ens_np[ens0[start:end]], dtype=jnp.float64
            )
            pad_n = _round_up_batch(m)
            if m < pad_n:
                dofs_j = jnp.pad(dofs_j, ((0, pad_n - m), (0, 0)))
                rc_j = jnp.pad(rc_j, ((0, pad_n - m), (0, 0), (0, 0)))
                rq_j = jnp.pad(rq_j, ((0, pad_n - m), (0, 0)))
            e_b, g_b = self._pot_vg_batch(dofs_j, rc_j, rq_j)
            e_b.block_until_ready()
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
