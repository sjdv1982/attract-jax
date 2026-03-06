#!/usr/bin/env python3
"""
score_pairs.py — Python NB dispatch abstraction (Milestone 5).

``score_pairs()`` evaluates pairwise nonbonded energy (and optionally the
spatial gradient w.r.t. query position) over precomputed neighbour lists.
The backend (JAX or compiled C kernel) is selected automatically: if a
compiled ``nb_kernel_<forcefield>.so`` is found next to the force-field
package, the kernel backend is used; otherwise pure JAX is used.

Both the in-house grid generator (M5) and the NB correction path call through
this dispatch.  The difference is the plateau mode:

  plateau_mode="clamp"
      Grid precomputation: ``E(max(d, plateaudis))``.  Used at every voxel
      corner; adds the clamped potential of all receptor atoms in the
      neighbour list.

  plateau_mode="correction"
      Runtime NB correction: ``E(d) - E(plateaudis)`` for pairs inside the
      plateau.  Only pairs with ``d < plateaudis`` contribute.

Backend selection (automatic)
------------------------------
  kernel  (preferred)
      Used when ``nb_kernel_<forcefield>.so`` is found near the force-field
      package.  Calls the compiled C kernel via ctypes; each query position is
      wrapped as a pseudo-pose.  The correct plateau-mode symbol is selected
      automatically from the loaded shared library.

  jax  (fallback)
      Pure JAX via ``jnp.vmap``/``jit``.  Used when no compiled kernel is
      found for the requested force field.

Public API
----------
``score_pairs(query_coords, rec_coords, rec_atomtypes_idx, rec_charges_scaled,
              query_atomtype_idx, query_charge, ff_params, forcefield,
              nr_neigh, nb_start, nb_concat, origin, gridspacing, dim,
              plateaudis_sq, plateau_mode, return_gradients)``

All arrays must be NumPy.  JAX arrays are accepted and will be converted
internally where needed (at a copy cost); pass NumPy for best performance.
"""

import ctypes
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Force-field + kernel loader (cached per forcefield name)
# ---------------------------------------------------------------------------

_ff_lib_cache: dict = {}


def _load_ff_and_lib(forcefield: str):
    """Load the ff_module for *forcefield* and obtain its kernel library.

    Backend selection is based on the ``lj.h`` header in the force-field
    package directory:

    - If ``lj.h`` is absent or is a generated skeleton (contains ``// TODO``),
      the JAX backend is used and ``lib`` is ``None``.
    - Otherwise the kernel backend is required.  The compiled
      ``nb_kernel_<forcefield>.so`` is located by walking up the directory
      tree from the FF package.  If it is not found, ``make nb_kernel_<name>.so``
      is attempted in the ``native/nb_kernel`` build directory.  If compilation
      fails, a ``RuntimeError`` is raised.

    Results are cached so subsequent calls for the same name are free.

    Parameters
    ----------
    forcefield : str
        Short forcefield name, e.g. ``"nonbon8"``.

    Returns
    -------
    ff_module : module
    lib : ctypes.CDLL or None
    """
    if forcefield in _ff_lib_cache:
        return _ff_lib_cache[forcefield]

    kernel_root = Path(__file__).resolve().parents[1] / "native" / "nb_kernel"
    if str(kernel_root) not in sys.path:
        sys.path.insert(0, str(kernel_root))

    from forcefields import find_kernel_so, load_forcefield  # noqa: PLC0415

    ff_module = load_forcefield(f"forcefields.{forcefield}")

    # Determine whether the kernel backend is intended by inspecting lj.h.
    ff_dir = Path(ff_module.__file__).parent
    lj_h = ff_dir / "lj.h"
    kernel_intended = lj_h.exists() and "// TODO" not in lj_h.read_text()

    if kernel_intended:
        lib = find_kernel_so(ff_module)
        if lib is None:
            so_name = f"nb_kernel_{forcefield}.so"
            result = subprocess.run(
                ["make", so_name],
                cwd=str(kernel_root),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Kernel backend required for force field '{forcefield}' "
                    f"(lj.h is not a skeleton) but '{so_name}' is missing "
                    f"and compilation failed:\n{result.stderr}"
                )
            lib = find_kernel_so(ff_module)
            if lib is None:
                raise RuntimeError(
                    f"make succeeded but '{so_name}' still not found near "
                    f"{ff_dir}; check the Makefile output directory."
                )
    else:
        lib = None

    _ff_lib_cache[forcefield] = (ff_module, lib)
    return ff_module, lib


# ---------------------------------------------------------------------------
# ctypes structures (shared with jax_scorer.py; duplicated here to avoid
# circular imports)
# ---------------------------------------------------------------------------


class _NbFusedStepData(ctypes.Structure):
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


class _NbFusedGridData(ctypes.Structure):
    _fields_ = [
        ("dim", ctypes.c_int32 * 3),
        ("origin", ctypes.c_double * 3),
        ("spacing", ctypes.c_double),
        ("nr_neigh", ctypes.POINTER(ctypes.c_int32)),
        ("nb_start", ctypes.POINTER(ctypes.c_int64)),
    ]


class _NbGlobalData(ctypes.Structure):
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


class _NbRunConfig(ctypes.Structure):
    _fields_ = [("num_threads", ctypes.c_int32), ("kernel_variant", ctypes.c_int32)]


def _ptr(arr, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


# ---------------------------------------------------------------------------
# JAX pair-energy helpers
# ---------------------------------------------------------------------------


def _make_jax_kernel(ff_module, plateau_mode: str, plateaudis_sq: float):
    """Return a JAX function that computes (N,) energies from vectorised inputs.

    The returned function signature::

        energies = kernel(
            query_c,           # (N, 1, 3) float64
            rec_c,             # (N, K, 3) float64
            dsq,               # (N, K) float64
            valid,             # (N, K) bool
            rt,                # (N, K) int32  — receptor atom-type indices
            qt,                # (N, K) int32  — query atom-type index
            rec_charge,        # (N, K) float64
            query_charge_full, # (N, K) float64  — rec_charge * query_charge
            rc, ac, emin, rmin2, ivor,  # 2-D arrays (nrec_t, nlig_t)
        )
    """
    pdsq = float(plateaudis_sq)
    prr2 = 1.0 / pdsq

    def kernel(
        query_c,
        rec_c,
        dsq,
        valid,
        rt,
        qt,
        rec_charge,
        query_charge_full,
        rc,
        ac,
        emin,
        rmin2,
        ivor,
    ):
        if plateau_mode == "clamp":
            dsq_eff = jnp.where(dsq < pdsq, pdsq, dsq)
        else:
            # correction: only pairs inside plateau contribute
            valid = valid & (dsq <= pdsq)
            dsq_eff = jnp.where(valid, dsq, 1.0)  # avoid div-by-zero

        rr2 = 1.0 / dsq_eff

        # LJ energy (vectorised over (N, K))
        rlen = rc[rt, qt]
        alen = ac[rt, qt]
        emin_v = emin[rt, qt]
        rmin2_v = rmin2[rt, qt]
        iv = ivor[rt, qt]
        rr23 = rr2 * rr2 * rr2
        rep = rlen * rr2
        vlj = (rep - alen) * rr23
        e_lj = jnp.where(dsq_eff < rmin2_v, vlj + (iv - 1.0) * emin_v, iv * vlj)

        if plateau_mode == "correction":
            # Subtract E(plateaudis)
            vlj_p = (rlen * prr2 - alen) * (prr2**3)
            e_lj_p = jnp.where(pdsq < rmin2_v, vlj_p + (iv - 1.0) * emin_v, iv * vlj_p)
            e_lj = e_lj - e_lj_p

        # Electrostatics (rdie, nonbon8 convention)
        inv50sq = (1.0 / 50.0) ** 2
        charge = query_charge_full
        dd = jnp.where(rr2 > inv50sq, rr2 - inv50sq, 0.0)
        e_elec = charge * dd

        if plateau_mode == "correction":
            dd_p = jnp.where(prr2 > inv50sq, prr2 - inv50sq, 0.0)
            e_elec = e_elec - charge * dd_p

        e_pair = jnp.where(valid, e_lj + e_elec, 0.0)
        return e_pair.sum(axis=-1)  # (N,)

    return jax.jit(kernel)


# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------


def score_pairs_jax(
    query_coords: np.ndarray,  # (N, 3)
    rec_coords: np.ndarray,  # (M, 3)
    rec_atomtypes_idx: np.ndarray,  # (M,) int  — index into ff rec dimension
    rec_charges_scaled: np.ndarray,  # (M,) float64  — q_rec * felec (or * felec^2)
    query_atomtype_idx: int,  # index into ff lig dimension
    query_charge: float,  # scalar (q_lig * felec, or 1.0 for elec grid)
    ff_params,  # FFParams(rc, ac, ivor, emin, rmin2)
    ff_module,
    nr_neigh: np.ndarray,  # (N,) int32
    nb_start: np.ndarray,  # (N,) int64
    nb_concat: np.ndarray,  # (total_nb,) int32
    plateaudis_sq: float,
    plateau_mode: str = "clamp",
    return_gradients: bool = False,
    batch_size: int = 16384,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Evaluate pairwise NB energy over neighbour lists (JAX backend).

    Parameters
    ----------
    query_coords : (N, 3) float64
    rec_coords   : (M, 3) float64
    rec_atomtypes_idx : (M,) int  — indices into the rec dimension of ff_params
    rec_charges_scaled : (M,) float
        Receptor charges multiplied by felec (for combined charge product).
    query_atomtype_idx : int
        Ligand/query atom type index into the lig dimension of ff_params.
    query_charge : float
        Query atom charge (scaled by felec), or 1.0 for the electrostatic
        potential grid channel (divide by felec at scoring time).
    ff_params : FFParams
        Namedtuple with rc, ac, ivor, emin, rmin2 arrays of shape
        (nrec_types, nlig_types).
    ff_module : module
        Force-field module (used for API reference; physics inlined here for
        performance).
    nr_neigh, nb_start, nb_concat : numpy arrays
        Compact neighbour list representation.
    plateaudis_sq : float
    plateau_mode : "clamp" | "correction"
    return_gradients : bool
        If True, also return dE/d(query_pos) for each query (useful for
        building stored-gradient potential grids).
    batch_size : int
        Number of query positions to process per JAX JIT call (controls peak
        memory).

    Returns
    -------
    energies : (N,) float64
    gradients : (N, 3) float64 or None
    """
    N = int(query_coords.shape[0])
    if N == 0:
        empty_e = np.zeros(0, dtype=np.float64)
        empty_g = np.zeros((0, 3), dtype=np.float64) if return_gradients else None
        return empty_e, empty_g

    max_K = int(nr_neigh.max()) if nr_neigh.size else 0
    if max_K == 0:
        empty_e = np.zeros(N, dtype=np.float64)
        empty_g = np.zeros((N, 3), dtype=np.float64) if return_gradients else None
        return empty_e, empty_g

    # Convert to JAX arrays once
    rec_c_j = jnp.asarray(rec_coords, dtype=jnp.float64)
    rec_t_j = jnp.asarray(rec_atomtypes_idx, dtype=jnp.int32)
    rec_ch_j = jnp.asarray(rec_charges_scaled, dtype=jnp.float64)
    nb_c_j = jnp.asarray(nb_concat, dtype=jnp.int32)

    rc_j = jnp.asarray(ff_params.rc, dtype=jnp.float64)
    ac_j = jnp.asarray(ff_params.ac, dtype=jnp.float64)
    emin_j = jnp.asarray(ff_params.emin, dtype=jnp.float64)
    rmin2_j = jnp.asarray(ff_params.rmin2, dtype=jnp.float64)
    ivor_j = jnp.asarray(ff_params.ivor, dtype=jnp.float64)

    qt_scalar = int(query_atomtype_idx)
    qch = float(query_charge)

    offsets_j = jnp.arange(max_K, dtype=jnp.int32)
    M = int(rec_coords.shape[0])

    if ff_module is None:
        raise ValueError(
            "score_pairs_jax: ff_module is required for the JAX backend. "
            "Pass the force-field module (e.g. nonbon8) or use backend='kernel'."
        )

    if return_gradients:
        if not getattr(ff_module, "lj_grad", None) or not getattr(
            ff_module, "elec_grad", None
        ):
            missing = [
                n for n in ("lj_grad", "elec_grad") if not getattr(ff_module, n, None)
            ]
            raise AttributeError(
                f"score_pairs_jax: return_gradients=True requires {missing} in "
                f"ff_module ({ff_module.__name__ if hasattr(ff_module, '__name__') else ff_module}). "
                "Implement these functions or use autodiff potentials (--autodiff-potentials)."
            )

    pdsq_f = float(plateaudis_sq)
    prr2_f = 1.0 / pdsq_f

    @jax.jit
    def _batch_energy(qc_batch, nr_batch, nb_start_batch):
        # qc_batch : (B, 3)
        # nr_batch : (B,)
        # nb_start_batch : (B,)
        B = qc_batch.shape[0]
        # Flat indices into nb_concat: (B, max_K)
        flat_nb = nb_start_batch[:, None] + offsets_j[None, :]  # (B, K)
        flat_nb_clip = jnp.clip(flat_nb, 0, nb_c_j.shape[0] - 1)
        rec_idx = nb_c_j[flat_nb_clip]  # (B, K) receptor atom index
        valid = offsets_j[None, :] < nr_batch[:, None]  # (B, K)

        safe_rec = jnp.clip(rec_idx, 0, M - 1)  # (B, K)
        rec_c = rec_c_j[safe_rec]  # (B, K, 3)
        d = qc_batch[:, None, :] - rec_c  # (B, K, 3)
        dsq = (d * d).sum(axis=-1)  # (B, K)

        if plateau_mode == "clamp":
            dsq_eff = jnp.where(dsq < pdsq_f, pdsq_f, dsq)
        else:
            valid = valid & (dsq <= pdsq_f)
            dsq_eff = jnp.where(valid, dsq, 1.0)

        rr2 = 1.0 / dsq_eff

        rt = rec_t_j[safe_rec]  # (B, K)
        qt = jnp.full_like(rt, qt_scalar)

        rlen = rc_j[rt, qt]
        alen = ac_j[rt, qt]
        emin_v = emin_j[rt, qt]
        rmin2_v = rmin2_j[rt, qt]
        iv = ivor_j[rt, qt]

        e_lj = ff_module.lj_energy(rlen, alen, emin_v, rmin2_v, iv, dsq_eff, rr2)
        if plateau_mode == "correction":
            e_lj = e_lj - ff_module.lj_energy(
                rlen, alen, emin_v, rmin2_v, iv, pdsq_f, prr2_f
            )

        # Electrostatics
        charge = rec_ch_j[safe_rec] * qch  # (B, K)
        e_elec = ff_module.elec_energy(charge, rr2)
        if plateau_mode == "correction":
            e_elec = e_elec - ff_module.elec_energy(charge, prr2_f)

        e_pair = jnp.where(valid, e_lj + e_elec, 0.0)
        return e_pair.sum(axis=-1)  # (B,)

    @jax.jit
    def _batch_energy_and_grad(qc_batch, nr_batch, nb_start_batch):
        """Compute energy AND explicit gradients using the same clamp/correction
        formula as the C kernel.  ``jax.grad`` is NOT used here because in clamp
        mode the clamped energy is constant w.r.t. query position inside the
        plateau sphere, yielding incorrect zero gradients.  The explicit formula
        produces the "pseudo-gradient" stored in the ATTRACT grid channels 1-3,
        which represent the force direction on the probe atom.

        Returns
        -------
        energies : (B,)
        grads    : (B, 3)  — gradient convention: +(dE/dq)
        """
        B = qc_batch.shape[0]
        flat_nb = nb_start_batch[:, None] + offsets_j[None, :]  # (B, K)
        flat_nb_clip = jnp.clip(flat_nb, 0, nb_c_j.shape[0] - 1)
        rec_idx = nb_c_j[flat_nb_clip]  # (B, K)
        valid = offsets_j[None, :] < nr_batch[:, None]  # (B, K)

        safe_rec = jnp.clip(rec_idx, 0, M - 1)
        rec_c = rec_c_j[safe_rec]  # (B, K, 3)
        d = qc_batch[:, None, :] - rec_c  # (B, K, 3)  q - r
        dsq = (d * d).sum(axis=-1)  # (B, K)

        # ---- plateau mode ----
        if plateau_mode == "clamp":
            within = dsq < pdsq_f
            ratio = jnp.where(within, jnp.sqrt(dsq / pdsq_f), 1.0)
            eval_dsq = jnp.where(within, pdsq_f, dsq)
        else:  # correction
            within = dsq <= pdsq_f
            eval_dsq = dsq
            valid = valid & within

        # Avoid division by zero; invalid pairs will be masked out
        safe_dsq = jnp.where(valid, dsq, 1.0)
        eval_dsq_safe = jnp.where(valid, eval_dsq, 1.0)
        eval_rr2 = 1.0 / eval_dsq_safe

        # Displacement for gradient computation.
        #   Clamp mode: "pseudo-gradient" d/dsq*ratio — points toward the plateau sphere
        #               surface, matching the kernel's eval_dx = sdx * ratio convention.
        #   Correction mode: actual gradient at real distance (d/dsq).  The correction
        #               component V(pdsq) is constant w.r.t. query position, so its
        #               gradient contribution is exactly zero and must NOT be subtracted.
        if plateau_mode == "clamp":
            eval_d = d / safe_dsq[..., None] * ratio[..., None]
        else:
            eval_d = d / safe_dsq[..., None]

        rt = rec_t_j[safe_rec]
        qt = jnp.full_like(rt, qt_scalar)

        rlen = rc_j[rt, qt]
        alen = ac_j[rt, qt]
        emin_v = emin_j[rt, qt]
        rmin2_v = rmin2_j[rt, qt]
        iv = ivor_j[rt, qt]

        # LJ energy + gradient via ff_module (routes through force-field implementation).
        e_lj, gx_lj, gy_lj, gz_lj = ff_module.lj_grad(
            rlen,
            alen,
            emin_v,
            rmin2_v,
            iv,
            eval_dsq_safe,
            eval_rr2,
            eval_d[..., 0],
            eval_d[..., 1],
            eval_d[..., 2],
        )
        g_lj = jnp.stack([gx_lj, gy_lj, gz_lj], axis=-1)  # (B, K, 3)

        if plateau_mode == "correction":
            # Subtract energy at plateau distance (constant w.r.t. query → zero gradient).
            e_lj = e_lj - ff_module.lj_energy(
                rlen, alen, emin_v, rmin2_v, iv, pdsq_f, prr2_f
            )
            # g_lj is unchanged: gradient of the correction component is zero.

        # Electrostatics: energy + gradient via ff_module.
        charge = rec_ch_j[safe_rec] * qch
        e_elec, gx_el, gy_el, gz_el = ff_module.elec_grad(
            charge,
            eval_rr2,
            eval_d[..., 0],
            eval_d[..., 1],
            eval_d[..., 2],
        )
        g_elec = jnp.stack([gx_el, gy_el, gz_el], axis=-1)  # (B, K, 3)

        if plateau_mode == "correction":
            # Subtract elec energy at plateau distance (constant → zero gradient).
            e_elec = e_elec - ff_module.elec_energy(charge, prr2_f)
            # g_elec is unchanged.

        mask3 = valid[..., None]
        e_pair = jnp.where(valid, e_lj + e_elec, 0.0)
        g_pair = jnp.where(mask3, g_lj + g_elec, 0.0)  # (B, K, 3)

        energies = e_pair.sum(axis=-1)  # (B,)
        # Sum over neighbours; negate because lj_grad/elec_grad return force
        # direction (-dE/dq); negating gives gradient convention (+dE/dq).
        grads = -g_pair.sum(axis=-2)  # (B, 3)
        return energies, grads

    out_e = np.empty(N, dtype=np.float64)
    out_g = np.empty((N, 3), dtype=np.float64) if return_gradients else None

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        qc_b = jnp.asarray(query_coords[start:end], dtype=jnp.float64)
        nr_b = jnp.asarray(nr_neigh[start:end], dtype=jnp.int32)
        ns_b = jnp.asarray(nb_start[start:end], dtype=jnp.int32)

        if return_gradients:
            e_b, g_b = _batch_energy_and_grad(qc_b, nr_b, ns_b)
            out_e[start:end] = np.asarray(e_b)
            out_g[start:end] = np.asarray(g_b)
        else:
            e_b = _batch_energy(qc_b, nr_b, ns_b)
            out_e[start:end] = np.asarray(e_b)

    return out_e, out_g


# ---------------------------------------------------------------------------
# Kernel backend
# ---------------------------------------------------------------------------


def score_pairs_kernel(
    query_coords: np.ndarray,  # (N, 3) — voxel corner OR ligand positions
    rec_coords: np.ndarray,  # (M, 3)
    rec_atomtypes_idx: np.ndarray,  # (M,) int16
    rec_charges_scaled: np.ndarray,  # (M,) float64
    query_atomtype_idx: int,
    query_charge: float,
    ff_params,
    lib: "ctypes.CDLL",
    nr_neigh: np.ndarray,  # (N,) int32  — raveled over query positions
    nb_start: np.ndarray,  # (N,) int64
    nb_concat: np.ndarray,  # (total_nb,) int32
    origin: np.ndarray,  # (3,) float64
    gridspacing: float,
    dim: np.ndarray,  # (3,) int
    plateaudis_sq: float,
    plateau_mode: str = "clamp",
    return_gradients: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Evaluate pairwise NB energy (and optionally gradient) via the C kernel.

    Each query position is wrapped as a pseudo-pose: rotation = identity,
    translation = (qx, qy, qz), single probe atom at origin with
    ``query_atomtype_idx`` and ``query_charge``.  The kernel uses its own
    neighbour-grid lookup to find receptor atoms for each query pose; the
    ``nr_neigh``/``nb_start``/``nb_concat`` arrays override the grid data.

    Parameters
    ----------
    return_gradients : bool
        If True, also compute the spatial gradient dE/d(query_pos) for each
        query via the ``_grad`` kernel symbol.

    Returns
    -------
    energies : (N,) float64
    gradients : (N, 3) float64 or None
        ``out_grad[:, 3:6]`` from the kernel, which stores ``-dE/d(tx,ty,tz)``.
        None when ``return_gradients=False``.
    """
    from forcefields import bind_kernel_dispatch  # noqa: PLC0415

    dispatch, fn_grad, fn_energy = bind_kernel_dispatch(
        lib, rotation="euler", plateau_mode=plateau_mode
    )
    if return_gradients and fn_grad is None:
        raise RuntimeError(
            f"return_gradients=True but no grad symbol found for "
            f"plateau_mode={plateau_mode!r} in the kernel library"
        )
    _fn_to_use = fn_grad if return_gradients else fn_energy
    _fn_to_use.argtypes = [
        ctypes.POINTER(_NbFusedStepData),
        ctypes.POINTER(_NbFusedGridData),
        ctypes.POINTER(_NbGlobalData),
        ctypes.POINTER(_NbRunConfig),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _fn_to_use.restype = ctypes.c_int

    N = int(query_coords.shape[0])
    M = int(rec_coords.shape[0])

    # Probe atom: at origin, single atom
    lig_coords_np = np.zeros((1, 3), dtype=np.float64)
    lig_pivot_np = np.zeros(3, dtype=np.float64)
    lig_type_np = np.array([query_atomtype_idx], dtype=np.int16)
    lig_charge_np = np.array([float(query_charge)], dtype=np.float64)

    # DOFs: (phi=0, ssi=0, rot=0, tx, ty, tz) for each query position
    dofs_np = np.zeros((N, 6), dtype=np.float64)
    dofs_np[:, 3:6] = query_coords

    # Ensemble IDs: all 0 (single ensemble)
    ens_np = np.zeros(N, dtype=np.int16)

    nr_neigh_flat = np.ascontiguousarray(nr_neigh.reshape(-1), dtype=np.int32)
    nb_start_flat = np.ascontiguousarray(nb_start.reshape(-1), dtype=np.int64)
    nb_concat_c = np.ascontiguousarray(nb_concat, dtype=np.int32)
    rec_coord_c = np.ascontiguousarray(rec_coords, dtype=np.float64)
    rec_type_c = np.ascontiguousarray(rec_atomtypes_idx, dtype=np.int16)
    rec_charge_c = np.ascontiguousarray(rec_charges_scaled, dtype=np.float64)

    rc_c = np.ascontiguousarray(ff_params.rc.reshape(-1), dtype=np.float64)
    ac_c = np.ascontiguousarray(ff_params.ac.reshape(-1), dtype=np.float64)
    emin_c = np.ascontiguousarray(ff_params.emin.reshape(-1), dtype=np.float64)
    rmin2_c = np.ascontiguousarray(ff_params.rmin2.reshape(-1), dtype=np.float64)
    ivor_c = np.ascontiguousarray(
        ff_params.ivor.reshape(-1).astype(np.int8), dtype=np.int8
    )

    nrec_types = int(ff_params.rc.shape[0])
    nlig_types = int(ff_params.rc.shape[1])
    dim3 = tuple(int(x) for x in dim)

    step = _NbFusedStepData(
        nposes=np.int32(N),
        natoms=np.int32(1),
        dofs=_ptr(dofs_np.reshape(-1), ctypes.c_double),
        ens=_ptr(ens_np, ctypes.c_int16),
        lig_coords=_ptr(lig_coords_np.reshape(-1), ctypes.c_double),
        lig_pivot=_ptr(lig_pivot_np, ctypes.c_double),
        lig_type=_ptr(lig_type_np, ctypes.c_int16),
        lig_charge=_ptr(lig_charge_np, ctypes.c_double),
    )
    dim32 = np.array(dim3, dtype=np.int32)
    origin64 = np.asarray(origin, dtype=np.float64)
    grid_struct = _NbFusedGridData(
        dim=(ctypes.c_int32 * 3)(int(dim32[0]), int(dim32[1]), int(dim32[2])),
        origin=(ctypes.c_double * 3)(
            float(origin64[0]), float(origin64[1]), float(origin64[2])
        ),
        spacing=ctypes.c_double(float(gridspacing)),
        nr_neigh=_ptr(nr_neigh_flat, ctypes.c_int32),
        nb_start=_ptr(nb_start_flat, ctypes.c_int64),
    )
    global_struct = _NbGlobalData(
        nrec=np.int32(M),
        nens=np.int32(1),
        nb_concat_len=np.int64(len(nb_concat_c)),
        nb_concat=_ptr(nb_concat_c, ctypes.c_int32),
        rec_coord=_ptr(rec_coord_c.reshape(-1), ctypes.c_double),
        rec_type=_ptr(rec_type_c, ctypes.c_int16),
        rec_charge=_ptr(rec_charge_c.reshape(-1), ctypes.c_double),
        nrec_types=np.int32(nrec_types),
        nlig_types=np.int32(nlig_types),
        rc=_ptr(rc_c, ctypes.c_double),
        ac=_ptr(ac_c, ctypes.c_double),
        emin=_ptr(emin_c, ctypes.c_double),
        rmin2=_ptr(rmin2_c, ctypes.c_double),
        ivor=_ptr(ivor_c, ctypes.c_int8),
        plateaudissq=ctypes.c_double(float(plateaudis_sq)),
    )

    import os  # noqa: PLC0415

    nthreads = int(os.environ.get("OMP_NUM_THREADS", "0")) or (os.cpu_count() or 1)
    cfg = _NbRunConfig(num_threads=np.int32(nthreads), kernel_variant=np.int32(1))

    out_e = np.zeros(N, dtype=np.float64)
    out_grad_flat = np.zeros(N * 6, dtype=np.float64) if return_gradients else None
    grad_ptr = _ptr(out_grad_flat, ctypes.c_double) if return_gradients else None
    rc_ret = _fn_to_use(
        ctypes.byref(step),
        ctypes.byref(grid_struct),
        ctypes.byref(global_struct),
        ctypes.byref(cfg),
        _ptr(out_e, ctypes.c_double),
        grad_ptr,
    )
    sym_name = dispatch.grad_symbol if return_gradients else dispatch.energy_symbol
    if rc_ret != 0:
        raise RuntimeError(f"score_pairs_kernel: {sym_name} returned {rc_ret}")
    if return_gradients:
        # out_grad_flat[6*p + 3:6] = (+dE/dtx, +dE/dty, +dE/dtz) per pose.
        # lj_grad returns force direction (-dE/dx); pose_loop negates before
        # writing out_grad, so the stored values are gradient direction (+dE/dx).
        grads = out_grad_flat.reshape(N, 6)[:, 3:6]
        return out_e, np.ascontiguousarray(grads)
    return out_e, None


# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------


def score_pairs(
    query_coords: np.ndarray,
    rec_coords: np.ndarray,
    rec_atomtypes_idx: np.ndarray,
    rec_charges_scaled: np.ndarray,
    query_atomtype_idx: int,
    query_charge: float,
    ff_params,
    forcefield: str,
    nr_neigh: np.ndarray,
    nb_start: np.ndarray,
    nb_concat: np.ndarray,
    origin: np.ndarray,
    gridspacing: float,
    dim: np.ndarray,
    plateaudis_sq: float,
    plateau_mode: str = "clamp",
    return_gradients: bool = False,
    batch_size: int = 16384,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Evaluate pairwise NB energy over neighbour lists.

    The backend (JAX or compiled C kernel) is selected automatically: if
    ``nb_kernel_<forcefield>.so`` is found near the force-field package the
    kernel backend is used; otherwise pure JAX is used.

    Parameters
    ----------
    query_coords : (N, 3) float64
        Positions to evaluate (voxel corners for grid, ligand atoms at runtime).
    rec_coords : (M, 3) float64
    rec_atomtypes_idx : (M,) int
        Receptor atom-type indices into the *rec* dimension of ``ff_params``.
    rec_charges_scaled : (M,) float64
        Per-atom charges times ``felec``.  For the electrostatic grid channel,
        pass ``q_rec_raw * felec**2`` instead to absorb the second scaling.
    query_atomtype_idx : int
        Query atom-type index into the *lig* dimension of ``ff_params``.
    query_charge : float
        Query atom charge (scaled by ``felec``).  Pass 1.0 when computing the
        unit electrostatic potential grid channel.
    ff_params : FFParams
        Namedtuple with rc, ac, ivor, emin, rmin2 2-D arrays.
    forcefield : str
        Short force-field name (e.g. ``"nonbon8"``).  The corresponding
        ``forcefields.<forcefield>`` module is imported automatically.
        If ``nb_kernel_<forcefield>.so`` is found near the FF package the
        kernel backend is used; otherwise pure JAX is used.
    nr_neigh : (N,) int32
        Number of neighbours for each query position.
    nb_start : (N,) int64
        Start offset in ``nb_concat`` for each query position.
    nb_concat : (total_nb,) int32
        Flat concatenated neighbour indices (index into rec_coords).
    origin : (3,) float64
        Grid origin (Å).  Required by the kernel backend.
    gridspacing : float
        Grid spacing (Å).  Required by the kernel backend.
    dim : (3,) int
        Inner grid dimensions.  Required by the kernel backend.
    plateaudis_sq : float
        Plateau distance squared (Å²).
    plateau_mode : "clamp" | "correction"
    return_gradients : bool
        Return dE/d(query_pos) alongside energies (JAX backend) or from the
        kernel's grad symbol (kernel backend).
    batch_size : int
        JAX backend: queries per JIT call.

    Returns
    -------
    energies : (N,) float64
    gradients : (N, 3) float64 or None
        None unless ``return_gradients=True``.
    """
    query_coords = np.asarray(query_coords, dtype=np.float64)
    rec_coords = np.asarray(rec_coords, dtype=np.float64)
    rec_atomtypes_idx = np.asarray(rec_atomtypes_idx, dtype=np.int32)
    rec_charges_scaled = np.asarray(rec_charges_scaled, dtype=np.float64)
    nr_neigh = np.asarray(nr_neigh, dtype=np.int32)
    nb_start = np.asarray(nb_start, dtype=np.int64)
    nb_concat = np.asarray(nb_concat, dtype=np.int32)
    origin = np.asarray(origin, dtype=np.float64)
    dim = np.asarray(dim, dtype=np.int32)

    if plateau_mode not in ("clamp", "correction"):
        raise ValueError(f"Invalid plateau_mode={plateau_mode!r}")

    ff_module, lib = _load_ff_and_lib(forcefield)

    if lib is not None:
        return score_pairs_kernel(
            query_coords=query_coords,
            rec_coords=rec_coords,
            rec_atomtypes_idx=rec_atomtypes_idx,
            rec_charges_scaled=rec_charges_scaled,
            query_atomtype_idx=query_atomtype_idx,
            query_charge=query_charge,
            ff_params=ff_params,
            lib=lib,
            nr_neigh=nr_neigh,
            nb_start=nb_start,
            nb_concat=nb_concat,
            origin=origin,
            gridspacing=gridspacing,
            dim=dim,
            plateaudis_sq=plateaudis_sq,
            plateau_mode=plateau_mode,
            return_gradients=return_gradients,
        )
    else:
        return score_pairs_jax(
            query_coords=query_coords,
            rec_coords=rec_coords,
            rec_atomtypes_idx=rec_atomtypes_idx,
            rec_charges_scaled=rec_charges_scaled,
            query_atomtype_idx=query_atomtype_idx,
            query_charge=query_charge,
            ff_params=ff_params,
            ff_module=ff_module,
            nr_neigh=nr_neigh,
            nb_start=nb_start,
            nb_concat=nb_concat,
            plateaudis_sq=plateaudis_sq,
            plateau_mode=plateau_mode,
            return_gradients=return_gradients,
            batch_size=batch_size,
        )
