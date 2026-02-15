#!/usr/bin/env python3
"""Precompute interior voxels and write a culled grid with their nb lists zeroed.

Strategy:
1. KD-tree pre-filter: for each active voxel, count receptor atoms within
   (1.5 Å + half_voxel_diagonal) across all ensemble members. Voxels with
   few nearby atoms cannot be interior → skip them (fast rejection).
2. For candidate voxels, compute the LJ nb correction energy at the voxel
   center for every (ensemble, alphabet ligand type) pair, with a conservative
   margin for ligand position uncertainty (± half voxel diagonal from center).
3. A voxel is "interior" if its energy lower bound > threshold for ALL combos.
   We zero its nb list in the output .grid file.

Output: a modified .grid file and statistics.

Usage:
    conda run -n jax python precompute_interior_voxels.py \\
        --grid test/receptorgrid.grid \\
        --receptor-ens-list test/partner1-ensemble.list \\
        --attract-par-npz attract-jax/attract-par.npz \\
        --ligand-pdb test/ligandr.pdb \\
        --threshold 100.0 \\
        --out-grid test/receptorgrid_culled.grid
"""

import argparse
import struct
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_pdb_coords(path: str):
    """Parse any PDB file, return just (N,3) coordinates."""
    coor = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coor.append((x, y, z))
    return np.asarray(coor, dtype=np.float64)


def parse_reduced_pdb(path: str):
    """Parse reduced PDB, return (coords, atomtypes, charges)."""
    coor, atomtype, charge = [], [], []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            at = int(float(line[54:59]))
            q = float(line[59:67])
            coor.append((x, y, z))
            atomtype.append(at)
            charge.append(q)
    return (
        np.asarray(coor, dtype=np.float64),
        np.asarray(atomtype, dtype=np.int32),
        np.asarray(charge, dtype=np.float64),
    )


def read_grid_binary(grid_path):
    """Read grid file, return raw bytes + parsed metadata."""
    grid_data = open(grid_path, "rb").read()
    pos = 0

    def _r(fmt):
        nonlocal pos
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, grid_data, pos)
        pos += size
        return vals[0] if len(vals) == 1 else vals

    read_bool = lambda: _r("?")
    read_short = lambda: _r("h")
    read_int = lambda: _r("i")
    read_float = lambda: _r("f")
    read_double = lambda: _r("d")
    read_long = lambda: _r("l")

    is_torquegrid = read_bool()
    arch = read_short()
    assert not is_torquegrid, "Torque grids not supported"
    assert arch == 64, f"Unexpected arch: {arch}"

    d = {}
    d["gridspacing"] = read_double()
    d["gridextension"] = read_int()
    d["plateaudis"] = read_double()
    d["neighbourdis"] = read_double()

    alphabet_vals = struct.unpack_from("?" * 99, grid_data, pos)
    pos += struct.calcsize("?" * 99)
    d["alphabet"] = np.array(alphabet_vals, bool)

    origin = np.array((read_float(), read_float(), read_float()), dtype=np.float64)
    d["origin"] = origin
    dx, dy, dz = read_int(), read_int(), read_int()
    d["dim"] = (dx, dy, dz)
    dx2, dy2, dz2 = read_int(), read_int(), read_int()
    d["dim2"] = (dx2, dy2, dz2)
    d["natoms"] = read_int()
    d["pivot"] = np.array(
        (read_double(), read_double(), read_double()), dtype=np.float64
    )

    # Energrads
    nr_energrads = read_int()
    shm_energrads = read_int()
    assert shm_energrads == -1
    pos += nr_energrads * 4 * 4

    # Neighbours
    nr_neighbours_total = read_int()
    shm_neighbours = read_int()
    assert shm_neighbours == -1
    nb_dtype = np.dtype([("type", np.uint8), ("index", np.uint32)], align=True)
    neighbours = np.frombuffer(
        grid_data, offset=pos, count=nr_neighbours_total, dtype=nb_dtype
    )
    pos += nr_neighbours_total * nb_dtype.itemsize
    d["neighbour_index"] = np.ascontiguousarray(neighbours["index"])
    d["neighbour_type"] = np.ascontiguousarray(neighbours["type"])

    # Inner grid
    innergridsize = read_long()
    assert innergridsize == dx * dy * dz

    innergrid_dtype = np.dtype(
        [
            ("potential", np.uint32, 100),
            ("neighbourlist", np.int32),
            ("nr_neighbours", np.int16),
        ],
        align=True,
    )
    innergrid_offset = pos
    innergrid = np.frombuffer(
        grid_data, offset=pos, count=innergridsize, dtype=innergrid_dtype
    )
    pos += innergrid.nbytes
    innergrid = innergrid.reshape((dz, dy, dx)).swapaxes(0, 2)
    innergrid = np.ascontiguousarray(innergrid)

    nr_neighbours_inner = np.ascontiguousarray(innergrid["nr_neighbours"])
    neighbourlist = np.ascontiguousarray(innergrid["neighbourlist"])
    d["nr_neighbours_inner"] = nr_neighbours_inner
    d["neighbourlist"] = neighbourlist
    d["innergrid_offset"] = innergrid_offset
    d["innergrid_dtype"] = innergrid_dtype

    max_nr_neighbours = int(nr_neighbours_inner.max())
    d["max_nr_neighbours"] = max_nr_neighbours

    neighbour_grid = np.full((dx, dy, dz, max_nr_neighbours), 2**16 - 1, np.uint16)
    neighbour_type_grid = np.zeros((dx, dy, dz, max_nr_neighbours), np.uint8)
    neighbour_index = d["neighbour_index"]
    neighbour_type = d["neighbour_type"]
    for n in range(max_nr_neighbours):
        maskn = nr_neighbours_inner > n
        nb_ind = neighbourlist[maskn]
        nb = nb_ind - 1 + n
        neighbour_grid[maskn, n] = neighbour_index[nb]
        neighbour_type_grid[maskn, n] = neighbour_type[nb]
    d["neighbour_grid"] = neighbour_grid
    d["neighbour_type_grid"] = neighbour_type_grid

    return grid_data, d


def compute_nb_energies_vectorized(
    voxel_coords,  # (V, 3)
    rec_coords,  # (natoms, 3)
    rec_atomtypes_0,  # (natoms,) 0-indexed
    nb_grid,  # (V, K) receptor atom indices
    nb_type_grid,  # (V, K) neighbour types
    nr_nb_arr,  # (V,)
    lig_type_0,  # scalar
    lig_charge_scaled,  # scalar
    rec_charge_scaled,  # (natoms,)
    rc,
    ac,
    ivor,
    emin,
    rmin2,
    plateaudissq,
    inv50sq,
    margin,
):
    """Vectorized nb energy lower bound for all candidate voxels.

    Returns (V,) array of lower-bound nb correction energies.
    """
    V, K = nb_grid.shape

    # Validity: real neighbour, type==1, within actual nb count
    valid = (nb_grid < 2**16 - 1) & (nb_type_grid == 1)
    offset_range = np.arange(K)[None, :]
    valid &= offset_range < nr_nb_arr[:, None]

    safe_atoms = np.where(valid, nb_grid.astype(np.int64), 0)

    # Distances
    rec_c = rec_coords[safe_atoms]  # (V, K, 3)
    d = voxel_coords[:, None, :] - rec_c  # (V, K, 3)
    dist = np.sqrt((d * d).sum(axis=-1))  # (V, K)
    dist_lower = np.maximum(dist - margin, 0.01)
    dsq = dist_lower * dist_lower

    within_plateau = dsq < plateaudissq
    active = valid & within_plateau

    at1 = rec_atomtypes_0[safe_atoms]
    at2 = lig_type_0

    # Clamp at1 to valid FF range (0..97). Type 98 (=type 99 1-indexed) is
    # a dummy that shouldn't appear in reduced PDBs, but guard against it.
    at1_safe = np.clip(at1, 0, rc.shape[0] - 1)
    at2_safe = min(at2, rc.shape[1] - 1)

    rc_vk = rc[at1_safe, at2_safe]
    ac_vk = ac[at1_safe, at2_safe]
    ivor_vk = ivor[at1_safe, at2_safe]
    emin_vk = emin[at1_safe, at2_safe]
    rmin2_vk = rmin2[at1_safe, at2_safe]

    safe_dsq = np.where(active, dsq, 1.0)
    rr2 = 1.0 / safe_dsq
    rr23 = rr2 * rr2 * rr2
    rep = rc_vk * rr2
    vlj = (rep - ac_vk) * rr23
    e_lj = np.where(
        safe_dsq < rmin2_vk,
        vlj + (ivor_vk - 1.0) * emin_vk,
        ivor_vk * vlj,
    )

    rr2_p = 1.0 / plateaudissq
    rr23_p = rr2_p**3
    vlj_p = (rc_vk * rr2_p - ac_vk) * rr23_p
    e_lj_p = np.where(
        plateaudissq < rmin2_vk,
        vlj_p + (ivor_vk - 1.0) * emin_vk,
        ivor_vk * vlj_p,
    )

    charge = rec_charge_scaled[safe_atoms] * lig_charge_scaled
    charge_safe = np.where(np.abs(charge) > 1e-3, charge, 0.0)
    plateau_el = max(1.0 / plateaudissq - inv50sq, 0.0)
    e_el = charge_safe * (np.maximum(rr2 - inv50sq, 0.0) - plateau_el)

    e_pair = np.where(active, (e_lj - e_lj_p) + e_el, 0.0)
    return e_pair.sum(axis=1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--grid", required=True)
    ap.add_argument("--receptor-ens-list", required=True)
    ap.add_argument("--attract-par-npz", required=True)
    ap.add_argument("--ligand-pdb", required=True)
    ap.add_argument("--epsilon", type=float, default=15.0)
    ap.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="nb energy lower-bound above which voxel is culled",
    )
    ap.add_argument(
        "--receptor-aa-ens-list",
        help="All-atom ensemble list for KD-tree pre-filter (denser atoms). "
        "If not given, uses --receptor-ens-list for KD-tree too.",
    )
    ap.add_argument(
        "--min-nearby-atoms",
        type=int,
        default=3,
        help="KD-tree pre-filter: voxels with fewer nearby atoms "
        "in ANY ensemble are definitely not interior",
    )
    ap.add_argument("--out-grid", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    # ---- Load grid first (need alphabet, spacing) ----
    print(f"Loading grid: {args.grid}")
    grid_data, gd = read_grid_binary(args.grid)
    dx, dy, dz = gd["dim"]
    origin = gd["origin"]
    gridspacing = gd["gridspacing"]
    plateaudis = gd["plateaudis"]
    plateaudissq = plateaudis**2
    inv50sq = 1.0 / (50.0 * 50.0)

    nr_nb = gd["nr_neighbours_inner"]
    nb_grid = gd["neighbour_grid"]
    nb_type_grid = gd["neighbour_type_grid"]
    max_nb = gd["max_nr_neighbours"]

    half_diag = np.sqrt(3) * gridspacing / 2.0

    print(f"Grid: {dx}×{dy}×{dz} = {dx*dy*dz} voxels, spacing={gridspacing:.3f}")
    print(f"Plateau: {plateaudis:.1f} Å, max_nb: {max_nb}")
    print(f"Half voxel diagonal (margin): {half_diag:.3f} Å")

    # ---- Alphabet types ----
    alphabet = gd["alphabet"]
    alphabet_types_1 = np.where(alphabet)[0] + 1
    # FF arrays are 98×98 (types 1..98, 0-indexed 0..97). Type 99 is a dummy.
    alphabet_types_1 = alphabet_types_1[alphabet_types_1 <= 98]
    print(f"Grid alphabet: {len(alphabet_types_1)} active types (excl. type 99)")

    # ---- Load receptor ensemble (reduced, for energy calculation) ----
    ens_dir = Path(args.receptor_ens_list).parent
    with open(args.receptor_ens_list) as f:
        ens_paths = [str(ens_dir / line.strip()) for line in f if line.strip()]
    print(f"Loading {len(ens_paths)} receptor ensemble members (reduced)...")
    rec_ensemble = []
    for p in ens_paths:
        coords, atypes, charges = parse_reduced_pdb(p)
        rec_ensemble.append((coords, atypes, charges))

    # ---- Load all-atom ensemble (for KD-tree pre-filter) ----
    if args.receptor_aa_ens_list:
        aa_dir = Path(args.receptor_aa_ens_list).parent
        with open(args.receptor_aa_ens_list) as f:
            aa_paths = [str(aa_dir / line.strip()) for line in f if line.strip()]
        print(f"Loading {len(aa_paths)} all-atom ensemble members for KD-tree...")
        aa_ensemble_coords = [parse_pdb_coords(p) for p in aa_paths]
        print(f"  All-atom atoms per member: {len(aa_ensemble_coords[0])}")
    else:
        aa_ensemble_coords = [c for c, _, _ in rec_ensemble]
        print("Using reduced ensemble for KD-tree (no --receptor-aa-ens-list).")

    # ---- Load ligand (for charges per alphabet type) ----
    _, lig_atypes, lig_charges = parse_reduced_pdb(args.ligand_pdb)
    lig_type_charges = {}
    for t1 in alphabet_types_1:
        mask = lig_atypes == t1
        if mask.any():
            lig_type_charges[t1] = lig_charges[mask][
                np.argmax(np.abs(lig_charges[mask]))
            ]
        else:
            lig_type_charges[t1] = 0.0

    # ---- Load force field ----
    felec = np.sqrt(332.053986 / args.epsilon)
    par = np.load(args.attract_par_npz)
    rc_full = par["rc"].astype(np.float64)
    ac_full = par["ac"].astype(np.float64)
    ivor_full = par["ivor"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        emin_full = np.where(
            rc_full != 0,
            -27.0 * ac_full**4 / (256.0 * rc_full**3),
            0.0,
        )
        rmin2_full = np.where(ac_full > 0, 4.0 * rc_full / (3.0 * ac_full), 0.0)

    # ---- Active voxels ----
    active_mask = nr_nb > 0
    active_indices = np.argwhere(active_mask)
    n_active = len(active_indices)
    print(f"\nActive voxels (nb > 0): {n_active}")

    voxel_coords = origin[None, :] + active_indices.astype(np.float64) * gridspacing

    # ================================================================
    # KD-tree pre-filter
    # A voxel can only be interior if it has many receptor atoms very close.
    # Radius = 1.5 Å + half_diag covers the closest a ligand atom can be
    # to any point within the voxel.
    # ================================================================
    kdtree_radius = 1.5 + half_diag
    min_nearby = args.min_nearby_atoms
    print(
        f"\nKD-tree pre-filter: radius={kdtree_radius:.2f} Å, "
        f"min_atoms={min_nearby}"
    )

    candidate_mask = np.ones(n_active, dtype=bool)
    for ens_idx, aa_coords in enumerate(aa_ensemble_coords):
        tree = cKDTree(aa_coords)
        counts = tree.query_ball_point(
            voxel_coords, r=kdtree_radius, return_length=True
        )
        ens_reject = counts < min_nearby
        candidate_mask &= ~ens_reject
        n_remaining = int(candidate_mask.sum())
        print(
            f"  ens {ens_idx+1}: {int(ens_reject.sum())} rejected "
            f"(aa atoms={len(aa_coords)}), "
            f"{n_remaining} candidates remain"
        )

    n_candidates = int(candidate_mask.sum())
    print(
        f"\nAfter KD-tree filter: {n_candidates}/{n_active} candidates "
        f"({100*n_candidates/n_active:.1f}%)"
    )

    if n_candidates == 0:
        print("No candidates — nothing to cull.")
        return

    # ---- Extract candidate voxel data ----
    cand_idx = np.where(candidate_mask)[0]
    cand_coords = voxel_coords[cand_idx]
    cand_grid_idx = active_indices[cand_idx]

    cand_nb = np.zeros((n_candidates, max_nb), dtype=np.uint16)
    cand_nb_type = np.zeros((n_candidates, max_nb), dtype=np.uint8)
    cand_nr = np.zeros(n_candidates, dtype=np.int32)
    for ci in range(n_candidates):
        ix, iy, iz = cand_grid_idx[ci]
        n = int(nr_nb[ix, iy, iz])
        cand_nr[ci] = n
        cand_nb[ci, :n] = nb_grid[ix, iy, iz, :n]
        cand_nb_type[ci, :n] = nb_type_grid[ix, iy, iz, :n]

    # ================================================================
    # Compute nb energy lower bound for candidates
    # ================================================================
    n_lig_types = len(alphabet_types_1)
    n_combos = len(ens_paths) * n_lig_types
    print(
        f"\nComputing nb energies: {n_candidates} candidates " f"× {n_combos} combos..."
    )

    min_energy = np.full(n_candidates, np.inf, dtype=np.float64)

    combo = 0
    for ens_idx, (rec_coords, rec_atypes, rec_charges) in enumerate(rec_ensemble):
        rec_charge_scaled = rec_charges * felec
        rec_atomtypes_0 = rec_atypes - 1

        for lig_type_1 in alphabet_types_1:
            lig_type_0 = lig_type_1 - 1
            lig_charge_scaled = lig_type_charges[lig_type_1] * felec

            combo += 1
            if combo == 1 or combo % 50 == 0 or combo == n_combos:
                print(
                    f"  combo {combo}/{n_combos} "
                    f"(ens {ens_idx+1}, type {lig_type_1}), "
                    f"current interior: "
                    f"{int((min_energy > args.threshold).sum())}",
                    flush=True,
                )

            e = compute_nb_energies_vectorized(
                cand_coords,
                rec_coords,
                rec_atomtypes_0,
                cand_nb,
                cand_nb_type,
                cand_nr,
                lig_type_0,
                lig_charge_scaled,
                rec_charge_scaled,
                rc_full,
                ac_full,
                ivor_full,
                emin_full,
                rmin2_full,
                plateaudissq,
                inv50sq,
                half_diag,
            )
            np.minimum(min_energy, e, out=min_energy)

    elapsed = time.time() - t0
    print(f"\nComputation took {elapsed:.1f}s")

    # ---- Results ----
    interior_cand = min_energy > args.threshold
    n_interior = int(interior_cand.sum())

    print(f"\n=== Results ===")
    print(f"Active voxels: {n_active}")
    print(f"KD-tree candidates: {n_candidates}")
    print(
        f"Interior (min energy > {args.threshold}): "
        f"{n_interior} ({100*n_interior/n_active:.1f}% of active)"
    )
    print(f"Remaining active after culling: {n_active - n_interior}")

    remaining_nb = nr_nb.copy()
    for ci in range(n_candidates):
        if interior_cand[ci]:
            ix, iy, iz = cand_grid_idx[ci]
            remaining_nb[ix, iy, iz] = 0
    remaining_max = int(remaining_nb.max())
    print(f"Max nb: {max_nb} → {remaining_max} after culling")

    print(f"\nMin energy distribution (candidates only):")
    for thresh in [0, 1, 10, 50, 100, 500, 1000, 10000]:
        n = int((min_energy > thresh).sum())
        print(f"  > {thresh:6d}: {n:6d} / {n_candidates}")

    if args.dry_run:
        print("\nDry run — not writing output.")
        return

    # ---- Write culled grid ----
    print(f"\nWriting culled grid to {args.out_grid}...")
    grid_out = bytearray(grid_data)
    innergrid_offset = gd["innergrid_offset"]
    innergrid_dtype = gd["innergrid_dtype"]
    item_size = innergrid_dtype.itemsize
    nb_field_offset = innergrid_dtype.fields["nr_neighbours"][1]

    n_culled = 0
    for ci in range(n_candidates):
        if not interior_cand[ci]:
            continue
        ix, iy, iz = cand_grid_idx[ci]
        file_idx = iz * dy * dx + iy * dx + ix
        byte_offset = innergrid_offset + file_idx * item_size + nb_field_offset
        struct.pack_into("h", grid_out, byte_offset, 0)
        n_culled += 1

    Path(args.out_grid).write_bytes(bytes(grid_out))
    print(f"Wrote {args.out_grid} ({n_culled} voxels culled)")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
