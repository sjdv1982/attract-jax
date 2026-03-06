#!/usr/bin/env python3
"""
grid_generator.py — In-house nonbonded potential grid precomputation (Milestone 5).

Generates a ``Grid`` namedtuple identical in format to the object returned by
``read_grid_with_electro()`` in ``reproduce_grid_score.py``.  All physics
evaluations go through ``score_pairs()`` so that the same force-field dispatch
code is used at grid-precomputation time and at scoring time.

Grid topology
-------------
Inner grid (dim[0] × dim[1] × dim[2])
    Potential + gradient channels at voxel corners
    origin + (ix, iy, iz) * gridspacing

Outer grid (dim2[0] × dim2[1] × dim2[2])
    Same channels but coarser (2× spacing) covering a wider area
    origin_outer + (ix2, iy2, iz2) * 2 * gridspacing
    where origin_outer = origin − gridextension * gridspacing

Correction neighbour grid (inner dims)
    Indices of receptor atoms within ``neighbourdis`` of each voxel *centre*
    Used at scoring time for the NB-correction soft-core path.

Sign conventions (gradient convention — matches ``read_grid_with_electro()``)
-----------------------------------------------------------------------------
Channel 0: raw potential energy E (kcal/mol)
Channels 1-3: +(dE/dx), +(dE/dy), +(dE/dz)  — gradient direction
              — OR zero when ``return_gradients=False``

Note: the raw binary ``.grid`` file uses the *force* convention (-(dE/dx)) and its
reader (``read_grid_with_electro()``) negates channels 1–3 after loading.  This
generator applies the same negation so the resulting ``Grid`` namedtuple uses the
same gradient convention expected by all JAX scorers.

When ``generate_grid(return_gradients=False)`` is used (e.g. for a force field that
has no Python gradient implementation), channels 1-3 are stored as zero.  Scoring
must then use ``--autodiff-potentials`` so that JAX autodiffs through channel 0.
"""

import math
from collections import namedtuple
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

# Cutoff used for the potential grid neighbour search.
# Matches ``distcutoff_vdw = distcutoff_elec = 50`` in grid_calculate.cpp.
_POTENTIAL_CUTOFF = 50.0

# Shared zero-LJ sentinel namedtuple type
_FFParamsType = namedtuple("FFParams", ("rc", "ac", "ivor", "emin", "rmin2"))


def _make_zero_lj(ff_params):
    """Return a copy of ff_params with all LJ parameters zeroed.

    Used for the electrostatic-only grid channel calculation.  With rc=ac=emin=0
    the LJ energy is identically zero for any distance, leaving only the
    electrostatic contribution from the charge product.
    """
    z = np.zeros_like(ff_params.rc)
    o = np.ones_like(ff_params.rmin2)  # rmin2 > 0 avoids any potential NaN
    return _FFParamsType(rc=z, ac=z, ivor=z, emin=z, rmin2=o)


def _build_kd_neighbour_list(
    query_pos: np.ndarray,
    rec_pos: np.ndarray,
    cutoff: float,
) -> tuple:
    """Build compact CSR-style neighbour lists via KDTree.

    Parameters
    ----------
    query_pos : (N, 3) float64
    rec_pos   : (M, 3) float64
    cutoff    : float — search radius (Å)

    Returns
    -------
    nr_neigh : (N,) int32
    nb_start : (N,) int64 — start offset in nb_concat
    nb_concat : (total_nb,) int32 — receptor atom indices (0-based into rec_pos)
    """
    if len(rec_pos) == 0:
        nr_neigh = np.zeros(len(query_pos), dtype=np.int32)
        nb_start = np.zeros(len(query_pos), dtype=np.int64)
        return nr_neigh, nb_start, np.empty(0, dtype=np.int32)

    tree = cKDTree(rec_pos)
    results = tree.query_ball_point(query_pos, r=cutoff, workers=-1)

    nr_neigh = np.array([len(r) for r in results], dtype=np.int32)
    nb_start = np.zeros(len(results), dtype=np.int64)
    if len(results) > 1:
        nb_start[1:] = np.cumsum(nr_neigh[:-1])
    total = int(nb_start[-1]) + int(nr_neigh[-1]) if len(results) else 0
    nb_concat = np.empty(total, dtype=np.int32)
    for i, nbrs in enumerate(results):
        k = int(nr_neigh[i])
        if k > 0:
            s = int(nb_start[i])
            nb_concat[s : s + k] = sorted(nbrs)
    return nr_neigh, nb_start, nb_concat


def _compute_potential_grid(
    corners: np.ndarray,  # (N, 3) float64 — all voxel corners
    shape: tuple,  # (dx, dy, dz)
    rec_c: np.ndarray,  # (M, 3) float64
    rec_atomtypes_idx: np.ndarray,  # (M,) int32 — 0-based
    rec_charges_scaled,  # (M,) float64 for LJ (zeros) or elec
    query_atomtype_idx: int,
    query_charge: float,
    ff_params,
    forcefield: str,
    origin: np.ndarray,  # (3,) for kernel lookup
    gridspacing: float,
    dim: np.ndarray,  # (3,) for kernel lookup
    plateaudis_sq: float,
    nr_neigh: np.ndarray,  # (N,)
    nb_start: np.ndarray,  # (N,)
    nb_concat: np.ndarray,  # (total,)
    return_gradients: bool = True,
) -> np.ndarray:
    """Evaluate potential (and optionally gradients) at ``corners``.

    Returns (dx, dy, dz, 4) float32.  When ``return_gradients=False``,
    channels 1–3 are stored as zero (energy-only grid, suitable for
    autodiff scoring).
    """
    from score_pairs import score_pairs  # noqa: PLC0415

    N = len(corners)
    dx, dy, dz = shape
    assert dx * dy * dz == N

    energies, grads = score_pairs(
        query_coords=corners,
        rec_coords=rec_c,
        rec_atomtypes_idx=rec_atomtypes_idx,
        rec_charges_scaled=rec_charges_scaled,
        query_atomtype_idx=query_atomtype_idx,
        query_charge=query_charge,
        ff_params=ff_params,
        forcefield=forcefield,
        nr_neigh=nr_neigh,
        nb_start=nb_start,
        nb_concat=nb_concat,
        origin=origin,
        gridspacing=gridspacing,
        dim=dim,
        plateaudis_sq=plateaudis_sq,
        plateau_mode="clamp",
        return_gradients=return_gradients,
    )
    # Channel layout: [E, +(dE/dx), +(dE/dy), +(dE/dz)]  — gradient convention.
    # score_pairs_kernel already returns +dE/dx (the pose loop negates lj_grad's
    # force-direction output before writing to out_grad, so what comes back is the
    # gradient, same convention as the Grid loaded by read_grid_with_electro).
    # When return_gradients=False, grads is None and channels 1-3 are zero.
    voxels = np.zeros((N, 4), dtype=np.float64)
    voxels[:, 0] = energies
    if return_gradients and grads is not None:
        voxels[:, 1:4] = grads
    return voxels.reshape(dx, dy, dz, 4).astype(np.float32)


def generate_grid(
    rec_coords: np.ndarray,
    rec_atomtypes: np.ndarray,
    rec_charges_raw: np.ndarray,
    ff_params,
    forcefield: str,
    ffelec: float,
    gridspacing: float = 0.9,
    gridextension: int = 32,
    plateaudis: float = 10.0,
    neighbourdis: float = 12.0,
    lig_atomtypes: Optional[np.ndarray] = None,
    return_gradients: bool = True,
):
    """Generate a nonbonded potential grid from receptor coordinates.

    Uses the first ensemble member only (matching ``make-grid`` behaviour).
    Returns a ``Grid`` namedtuple with the same field layout as produced by
    ``read_grid_with_electro()``, suitable as a drop-in replacement for a
    binary .grid file.

    Parameters
    ----------
    rec_coords : (M, 3) float64
        Receptor atom coordinates (first ensemble member).
    rec_atomtypes : (M,) int
        1-based ATTRACT atom type indices (type 99 = dummy, filtered out).
    rec_charges_raw : (M,) float64
        Raw per-atom charges (unscaled, from PDB column 60-66).
    ff_params : FFParams
        Full atom-type pair parameter table from ``load_params()``.
        Must cover all atomtypes present in ``rec_atomtypes``.
        Expected shape (MAXATOMTYPES, MAXATOMTYPES) with 1-based atomtype k
        at row/col index k-1.
    forcefield : str
        Short force-field name (e.g. ``"nonbon8"``).  The backend is chosen
        automatically: if the force field's ``lj.h`` is a real implementation
        (not a generated skeleton), the compiled kernel is used; otherwise
        pure JAX is used.  If the ``.so`` is missing when the kernel is
        required, compilation is attempted via ``make``; an error is raised
        if that fails.
    return_gradients : bool
        If True (default), compute and store gradient channels 1–3 in the
        grid (``+dE/d_i``).  When using the JAX backend this requires
        ``lj_grad`` / ``elec_grad`` in the force-field module.
        If False, channels 1–3 are stored as zero.  Use this when the
        force field has no Python gradient implementation and only autodiff
        scoring (``--autodiff-potentials``) is needed.
    ffelec : float
        Electrostatics scaling factor √(332.054 / ε).
    gridspacing : float
        Grid voxel size (Å).
    gridextension : int
        Half-width of extra outer grid extension (in inner-half-voxels).
        Outer grid has 2× inner spacing.
    plateaudis : float
        Soft-core plateau distance (Å).
    neighbourdis : float
        Correction-term neighbour cutoff (Å).

    Returns
    -------
    Grid namedtuple
        Same field layout as ``read_grid_with_electro()``.
    """
    rec_coords = np.asarray(rec_coords, dtype=np.float64)
    rec_atomtypes = np.asarray(rec_atomtypes, dtype=np.int32)
    rec_charges_raw = np.asarray(rec_charges_raw, dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Filter dummy atoms (atomtype 99)
    # ------------------------------------------------------------------
    valid_mask = rec_atomtypes != 99
    rec_c = np.ascontiguousarray(rec_coords[valid_mask])
    rec_t = rec_atomtypes[valid_mask]
    rec_q_raw = rec_charges_raw[valid_mask]
    M = len(rec_c)
    if M == 0:
        raise ValueError("No valid receptor atoms after filtering dummy type 99")

    # 0-based ff_params indices: atomtype k → index k-1
    rec_atomtypes_idx = (rec_t - 1).astype(np.int32)

    # ------------------------------------------------------------------
    # 2. Alphabet: VDW channels to precompute
    #    lig_atomtypes given → receptor ∪ ligand types
    #    lig_atomtypes=None  → all 98 non-dummy types (universal grid)
    # ------------------------------------------------------------------
    alphabet = np.zeros(99, dtype=bool)
    if lig_atomtypes is not None:
        alphabet[rec_t - 1] = True
        lig_t = np.asarray(lig_atomtypes, dtype=np.int32)
        lig_t_valid = lig_t[(lig_t >= 1) & (lig_t <= 98)]
        alphabet[lig_t_valid - 1] = True
    else:
        alphabet[:98] = True  # types 1-98; type 99 (dummy) excluded
    alphabet_atomtypes = (np.where(alphabet)[0] + 1).astype(np.int32)
    n_alpha = int(len(alphabet_atomtypes))

    # ------------------------------------------------------------------
    # 3. Grid dimensions
    # ------------------------------------------------------------------
    mn = rec_c.min(axis=0)
    mx = rec_c.max(axis=0)
    origin = mn - float(plateaudis)

    gs = float(gridspacing)
    ge = int(gridextension)
    plateaudis_sq = float(plateaudis) ** 2

    dx, dy, dz = tuple(
        int(math.ceil((mx[i] - mn[i] + 2.0 * float(plateaudis)) / gs)) for i in range(3)
    )
    dim = np.array([dx, dy, dz], dtype=np.int32)

    dx2 = int((dx + 2 * ge) / 2) + 1
    dy2 = int((dy + 2 * ge) / 2) + 1
    dz2 = int((dz + 2 * ge) / 2) + 1
    dim2 = np.array([dx2, dy2, dz2], dtype=np.int32)

    # Outer grid origin and spacing:
    #   corner (ix2, iy2, iz2) = origin_outer + (ix2, iy2, iz2) * outer_gs
    #   = origin + (2*ix2 - ge) * gs
    # → origin_outer = origin - ge * gs
    origin_outer = origin - ge * gs
    outer_gs = 2.0 * gs
    dim_outer = dim2.copy()

    print(
        f"Grid dims: inner={dx}×{dy}×{dz}, outer={dx2}×{dy2}×{dz2}, "
        f"n_alpha={n_alpha}, origin=({origin[0]:.3f},{origin[1]:.3f},{origin[2]:.3f})"
    )

    # ------------------------------------------------------------------
    # 4. Generate voxel corner positions (C-order: ix slowest, iz fastest)
    # ------------------------------------------------------------------
    def _make_corners(ox, oy, oz, ndx, ndy, ndz, spacing):
        """All ndx*ndy*ndz corners in C-order flat layout."""
        ix, iy, iz = np.mgrid[0:ndx, 0:ndy, 0:ndz]
        return np.stack(
            [
                ox + ix * spacing,
                oy + iy * spacing,
                oz + iz * spacing,
            ],
            axis=-1,
        ).reshape(-1, 3)

    inner_corners = _make_corners(origin[0], origin[1], origin[2], dx, dy, dz, gs)
    outer_corners = _make_corners(
        origin_outer[0], origin_outer[1], origin_outer[2], dx2, dy2, dz2, outer_gs
    )

    # ------------------------------------------------------------------
    # 5. Neighbour lists for potential grid (50 Å, ALL rec atoms for LJ)
    # ------------------------------------------------------------------
    print("Building KDTree neighbour lists for potential grid (50 Å)...")

    # LJ neighbour list: all rec atoms
    lj_nr_inner, lj_ns_inner, lj_nc_inner = _build_kd_neighbour_list(
        inner_corners, rec_c, _POTENTIAL_CUTOFF
    )
    lj_nr_outer, lj_ns_outer, lj_nc_outer = _build_kd_neighbour_list(
        outer_corners, rec_c, _POTENTIAL_CUTOFF
    )

    # Electrostatics neighbour list: charged atoms only, but indices into FULL rec array
    charged_mask = np.abs(rec_q_raw) > 1e-3
    charged_idx = np.where(charged_mask)[0].astype(np.int32)
    rec_c_charged = rec_c[charged_mask]

    if len(rec_c_charged) > 0:
        elec_nr_inner, elec_ns_inner, elec_nc_inner_local = _build_kd_neighbour_list(
            inner_corners, rec_c_charged, _POTENTIAL_CUTOFF
        )
        elec_nr_outer, elec_ns_outer, elec_nc_outer_local = _build_kd_neighbour_list(
            outer_corners, rec_c_charged, _POTENTIAL_CUTOFF
        )
        # Remap local charged-atom indices → full rec array indices
        elec_nc_inner = (
            charged_idx[elec_nc_inner_local]
            if len(elec_nc_inner_local)
            else np.empty(0, np.int32)
        )
        elec_nc_outer = (
            charged_idx[elec_nc_outer_local]
            if len(elec_nc_outer_local)
            else np.empty(0, np.int32)
        )
    else:
        elec_nr_inner = np.zeros(len(inner_corners), dtype=np.int32)
        elec_ns_inner = np.zeros(len(inner_corners), dtype=np.int64)
        elec_nc_inner = np.empty(0, dtype=np.int32)
        elec_nr_outer = np.zeros(len(outer_corners), dtype=np.int32)
        elec_ns_outer = np.zeros(len(outer_corners), dtype=np.int64)
        elec_nc_outer = np.empty(0, dtype=np.int32)

    # ------------------------------------------------------------------
    # 6. Zero-LJ params for electrostatics-only channel
    # ------------------------------------------------------------------
    ff_params_zero_lj = _make_zero_lj(ff_params)
    # Charges for elec call: q_raw * ffelec^2 (second ffelec absorbed here so
    # query_charge=1.0 at scoring time gives the correct product)
    rec_charges_elec = rec_q_raw * (ffelec**2)

    # ------------------------------------------------------------------
    # 7. Compute inner potential grid (n_alpha VDW channels)
    # ------------------------------------------------------------------
    print("Computing inner VDW potential grid...")
    inner_potential_grid = np.zeros((n_alpha, dx, dy, dz, 4), dtype=np.float32)
    for ch_i, alpha_t in enumerate(alphabet_atomtypes):
        if (ch_i % 4) == 0:
            print(f"  VDW channel {ch_i+1}/{n_alpha} (atomtype {alpha_t})...")
        inner_potential_grid[ch_i] = _compute_potential_grid(
            corners=inner_corners,
            shape=(dx, dy, dz),
            rec_c=rec_c,
            rec_atomtypes_idx=rec_atomtypes_idx,
            rec_charges_scaled=np.zeros(M, dtype=np.float64),
            query_atomtype_idx=int(alpha_t) - 1,
            query_charge=0.0,
            ff_params=ff_params,
            forcefield=forcefield,
            origin=origin,
            gridspacing=gs,
            dim=dim,
            plateaudis_sq=plateaudis_sq,
            nr_neigh=lj_nr_inner,
            nb_start=lj_ns_inner,
            nb_concat=lj_nc_inner,
            return_gradients=return_gradients,
        )

    # ------------------------------------------------------------------
    # 8. Compute inner electrostatic grid
    # ------------------------------------------------------------------
    print("Computing inner electrostatic potential grid...")
    inner_elec_grid = np.zeros((dx, dy, dz, 4), dtype=np.float32)
    if len(elec_nc_inner) > 0:
        inner_elec_grid = _compute_potential_grid(
            corners=inner_corners,
            shape=(dx, dy, dz),
            rec_c=rec_c,
            rec_atomtypes_idx=rec_atomtypes_idx,
            rec_charges_scaled=rec_charges_elec,
            query_atomtype_idx=0,  # LJ=0 for all types; any index works
            query_charge=1.0,
            ff_params=ff_params_zero_lj,
            forcefield=forcefield,
            origin=origin,
            gridspacing=gs,
            dim=dim,
            plateaudis_sq=plateaudis_sq,
            nr_neigh=elec_nr_inner,
            nb_start=elec_ns_inner,
            nb_concat=elec_nc_inner,
            return_gradients=return_gradients,
        )

    # ------------------------------------------------------------------
    # 9. Compute outer potential grid (VDW channels)
    # ------------------------------------------------------------------
    print("Computing outer VDW potential grid...")
    outer_potential_grid = np.zeros((n_alpha, dx2, dy2, dz2, 4), dtype=np.float32)
    for ch_i, alpha_t in enumerate(alphabet_atomtypes):
        if (ch_i % 4) == 0:
            print(f"  VDW channel {ch_i+1}/{n_alpha} (atomtype {alpha_t})...")
        outer_potential_grid[ch_i] = _compute_potential_grid(
            corners=outer_corners,
            shape=(dx2, dy2, dz2),
            rec_c=rec_c,
            rec_atomtypes_idx=rec_atomtypes_idx,
            rec_charges_scaled=np.zeros(M, dtype=np.float64),
            query_atomtype_idx=int(alpha_t) - 1,
            query_charge=0.0,
            ff_params=ff_params,
            forcefield=forcefield,
            origin=origin_outer,
            gridspacing=outer_gs,
            dim=dim_outer,
            plateaudis_sq=plateaudis_sq,
            nr_neigh=lj_nr_outer,
            nb_start=lj_ns_outer,
            nb_concat=lj_nc_outer,
            return_gradients=return_gradients,
        )

    # ------------------------------------------------------------------
    # 10. Compute outer electrostatic grid
    # ------------------------------------------------------------------
    print("Computing outer electrostatic potential grid...")
    outer_elec_grid = np.zeros((dx2, dy2, dz2, 4), dtype=np.float32)
    if len(elec_nc_outer) > 0:
        outer_elec_grid = _compute_potential_grid(
            corners=outer_corners,
            shape=(dx2, dy2, dz2),
            rec_c=rec_c,
            rec_atomtypes_idx=rec_atomtypes_idx,
            rec_charges_scaled=rec_charges_elec,
            query_atomtype_idx=0,
            query_charge=1.0,
            ff_params=ff_params_zero_lj,
            forcefield=forcefield,
            origin=origin_outer,
            gridspacing=outer_gs,
            dim=dim_outer,
            plateaudis_sq=plateaudis_sq,
            nr_neigh=elec_nr_outer,
            nb_start=elec_ns_outer,
            nb_concat=elec_nc_outer,
            return_gradients=return_gradients,
        )

    # ------------------------------------------------------------------
    # 11. Build correction neighbour grid (inner voxel CENTRES, neighbourdis)
    # ------------------------------------------------------------------
    print("Building correction neighbour grid...")
    voxel_centres = inner_corners + 0.5 * gs
    nb_nr, nb_ns, nb_nc = _build_kd_neighbour_list(
        voxel_centres, rec_c, float(neighbourdis)
    )
    max_k = int(nb_nr.max()) if nb_nr.size and nb_nr.max() > 0 else 0

    # Compact format: (dx, dy, dz, max_k) uint16; fill value = 0xFFFF
    neighbour_grid = np.full((dx, dy, dz, max(1, max_k)), 0xFFFF, dtype=np.uint16)
    if max_k > 0:
        flat_nb_start = np.zeros(dx * dy * dz, dtype=np.int64)
        flat_nb_start[1:] = np.cumsum(nb_nr[:-1])
        for fi in range(dx * dy * dz):
            k = int(nb_nr[fi])
            if k > 0:
                s = int(nb_ns[fi])
                neighbour_grid.reshape(-1, max(1, max_k))[fi, :k] = nb_nc[s : s + k]

    nr_neighbours = nb_nr.reshape(dx, dy, dz).astype(np.int16)

    # ------------------------------------------------------------------
    # 12. Assemble Grid namedtuple
    # ------------------------------------------------------------------
    print("Assembling Grid namedtuple...")
    d = {
        "gridspacing": float(gs),
        "gridextension": int(ge),
        "plateaudis": float(plateaudis),
        "neighbourdis": float(neighbourdis),
        "alphabet": alphabet,
        "alphabet_atomtypes": alphabet_atomtypes,
        "origin": np.asarray(origin, dtype=np.float64),
        "dim": dim,
        "dim2": dim2,
        "natoms": int(M),
        "nr_neighbours": nr_neighbours,
        "max_nr_neighbours": int(max_k),
        "neighbour_grid": neighbour_grid,
        "inner_potential_grid": inner_potential_grid,
        "outer_potential_grid": outer_potential_grid,
        "inner_elec_grid": inner_elec_grid,
        "outer_elec_grid": outer_elec_grid,
    }
    GridClass = namedtuple("Grid", tuple(d.keys()) + ("neighbour_grid_ravel",))
    grid = GridClass(*d.values(), neighbour_grid_ravel=None)
    print("Grid generation complete.")
    return grid
