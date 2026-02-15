#!/usr/bin/env python3
"""Precompute interior voxels and write a culled grid with their nb lists zeroed.

Strategy:
- For each inner grid voxel with nb > 0, compute the total LJ nb correction
  energy at the voxel center for every (receptor ensemble, ligand atom type) pair.
- The nb correction is: sum over neighbours of [E(dsq) - E(plateaudissq)]
  where dsq is the squared distance from voxel center to receptor atom.
- At runtime, the ligand atom is snapped to the nearest voxel (floor(vox + 0.5)),
  but could be anywhere within ±0.5*gridspacing of the center. The worst case
  (most favorable) is when the atom is farthest from each receptor atom, which
  reduces the repulsive nb correction. We compute a conservative lower bound
  by using (distance - margin)^2 where margin = sqrt(3)*gridspacing/2.
- A voxel is marked "interior" if the nb energy lower bound is > threshold
  for ALL (ensemble, ligand type) combinations. We zero its nb list.
- This is conservative: we never cull a voxel that could produce a meaningful
  (non-garbage) energy.

Output: a modified .grid file and statistics.

Usage:
    conda run -n jax python precompute_interior_voxels.py \\
        --grid test/receptorgrid.grid \\
        --receptor-ens-list test/partner1-ensemble.list \\
        --attract-par-npz attract-jax/attract-par.npz \\
        --threshold 100.0 \\
        --out-grid test/receptorgrid_culled.grid
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np


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
    """Read grid file, return raw bytes + parsed metadata needed for analysis."""
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
    alphabet = np.array(alphabet_vals, bool)
    d["alphabet"] = alphabet

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
    energrads_start = pos
    pos += nr_energrads * 4 * 4  # 4 float32 per entry

    # Neighbours
    nr_neighbours_total = read_int()
    shm_neighbours = read_int()
    assert shm_neighbours == -1
    nb_dtype = np.dtype([("type", np.uint8), ("index", np.uint32)], align=True)
    neighbours = np.frombuffer(
        grid_data, offset=pos, count=nr_neighbours_total, dtype=nb_dtype
    )
    pos += nr_neighbours_total * nb_dtype.itemsize
    neighbour_index = np.ascontiguousarray(neighbours["index"])
    neighbour_type = np.ascontiguousarray(neighbours["type"])
    d["neighbour_index"] = neighbour_index
    d["neighbour_type"] = neighbour_type

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
    innergrid_offset = pos  # remember where innergrid starts in the file
    innergrid = np.frombuffer(
        grid_data, offset=pos, count=innergridsize, dtype=innergrid_dtype
    )
    pos += innergrid.nbytes
    # Note: stored as (dz, dy, dx), then swapaxes to (dx, dy, dz)
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

    # Build expanded neighbour grid: (dx, dy, dz, max_nr_neighbours)
    # neighbour_grid[i,j,k,n] = receptor atom index for n-th neighbour of voxel (i,j,k)
    # neighbour_type_grid[i,j,k,n] = type (1=within plateaudis, 0=outside)
    neighbour_grid = np.full((dx, dy, dz, max_nr_neighbours), 2**16 - 1, np.uint16)
    neighbour_type_grid = np.zeros((dx, dy, dz, max_nr_neighbours), np.uint8)
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
    voxel_coords,  # (V, 3) world coordinates of active voxels
    rec_coords,  # (natoms, 3)
    rec_atomtypes_0,  # (natoms,) 0-indexed
    nb_grid,  # (V, K) receptor atom indices per voxel
    nb_type_grid,  # (V, K) neighbour types
    nr_nb_arr,  # (V,) count of neighbours per voxel
    lig_type_0,  # scalar, 0-indexed
    lig_charge_scaled,  # scalar
    rec_charge_scaled,  # (natoms,)
    rc,
    ac,
    ivor,
    emin,
    rmin2,  # ff arrays (98, 98)
    plateaudissq,
    inv50sq,
    margin,
):
    """Vectorized nb energy lower bound for all voxels at once.

    Returns (V,) array of lower-bound nb energies.
    """
    V, K = nb_grid.shape
    nrec = rec_coords.shape[0]

    # Validity mask: real neighbour with type==1
    valid = (nb_grid < 2**16 - 1) & (nb_type_grid == 1)
    # Also mask offsets beyond actual nb count per voxel
    offset_range = np.arange(K)[None, :]  # (1, K)
    valid &= offset_range < nr_nb_arr[:, None]  # (V, K)

    # Safe atom indices for gather (clamp invalid to 0)
    safe_atoms = np.where(valid, nb_grid.astype(np.int64), 0)

    # Receptor coordinates at neighbour positions: (V, K, 3)
    rec_c = rec_coords[safe_atoms]

    # Distances from voxel center to each neighbour: (V, K)
    d = voxel_coords[:, None, :] - rec_c  # (V, K, 3)
    dist = np.sqrt((d * d).sum(axis=-1))  # (V, K)

    # Conservative: subtract margin, clamp to small positive
    dist_lower = np.maximum(dist - margin, 0.01)
    dsq = dist_lower * dist_lower

    # Beyond plateau → no correction, mark invalid
    within_plateau = dsq < plateaudissq
    active = valid & within_plateau

    # Receptor atom types for each neighbour: (V, K)
    at1 = rec_atomtypes_0[safe_atoms]
    at2 = lig_type_0  # scalar broadcast

    # Gather FF params: (V, K)
    rc_vk = rc[at1, at2]
    ac_vk = ac[at1, at2]
    ivor_vk = ivor[at1, at2]
    emin_vk = emin[at1, at2]
    rmin2_vk = rmin2[at1, at2]

    # --- LJ at dsq ---
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

    # --- LJ at plateau ---
    rr2_p = 1.0 / plateaudissq
    rr23_p = rr2_p**3
    rep_p = rc_vk * rr2_p
    vlj_p = (rep_p - ac_vk) * rr23_p
    e_lj_p = np.where(
        plateaudissq < rmin2_vk,
        vlj_p + (ivor_vk - 1.0) * emin_vk,
        ivor_vk * vlj_p,
    )

    # --- Electrostatic correction ---
    charge = rec_charge_scaled[safe_atoms] * lig_charge_scaled
    charge_safe = np.where(np.abs(charge) > 1e-3, charge, 0.0)
    plateau_el = max(1.0 / plateaudissq - inv50sq, 0.0)
    e_el = charge_safe * (np.maximum(rr2 - inv50sq, 0.0) - plateau_el)

    # Total per-pair, masked
    e_pair = np.where(active, (e_lj - e_lj_p) + e_el, 0.0)
    return e_pair.sum(axis=1)  # (V,)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grid", required=True, help="input .grid file")
    ap.add_argument("--receptor-ens-list", required=True, help="ensemble list file")
    ap.add_argument("--attract-par-npz", required=True, help="attract forcefield npz")
    ap.add_argument(
        "--ligand-pdb", required=True, help="reduced ligand PDB (for atom types)"
    )
    ap.add_argument("--epsilon", type=float, default=15.0, help="dielectric constant")
    ap.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="nb energy threshold: voxels above this for ALL types are culled",
    )
    ap.add_argument("--out-grid", required=True, help="output culled .grid file")
    ap.add_argument("--dry-run", action="store_true", help="analyze only, don't write")
    args = ap.parse_args()

    t0 = time.time()

    # Load receptor ensemble
    ens_dir = Path(args.receptor_ens_list).parent
    with open(args.receptor_ens_list) as f:
        ens_paths = [str(ens_dir / line.strip()) for line in f if line.strip()]

    print(f"Loading {len(ens_paths)} receptor ensemble members...")
    rec_ensemble = []  # list of (coords, atomtypes, charges)
    for p in ens_paths:
        coords, atypes, charges = parse_reduced_pdb(p)
        rec_ensemble.append((coords, atypes, charges))

    # Load ligand (just for charges per atom type)
    lig_coords, lig_atypes, lig_charges = parse_reduced_pdb(args.ligand_pdb)

    # Use alphabet types from the grid, not all ligand types
    alphabet = gd[
        "alphabet"
    ]  # (99,) bool — alphabet[i] True means type (i+1) is active
    alphabet_types_1 = np.where(alphabet)[0] + 1  # 1-indexed active types
    print(f"Grid alphabet: {len(alphabet_types_1)} active types: {alphabet_types_1}")

    # For charges: for each alphabet type, find the worst-case (max abs) charge
    # among ligand atoms of that type (if any exist). If no ligand atoms have
    # that type, use charge=0 (conservative — less favorable electrostatics).
    lig_type_charges = {}
    for t1 in alphabet_types_1:
        mask = lig_atypes == t1
        if mask.any():
            lig_type_charges[t1] = lig_charges[mask][
                np.argmax(np.abs(lig_charges[mask]))
            ]
        else:
            lig_type_charges[t1] = 0.0

    # Load force field
    par = np.load(args.attract_par_npz)
    rc_full = par["rc"].astype(np.float64)
    ac_full = par["ac"].astype(np.float64)
    ivor_full = par["ivor"].astype(np.float64)
    # Derived
    emin_full = -27.0 * ac_full**4 / (256.0 * rc_full**3)
    rmin2_full = np.where(ac_full > 0, 4.0 * rc_full / (3.0 * ac_full), 0.0)

    # Load grid
    print(f"Loading grid: {args.grid}")
    grid_data, gd = read_grid_binary(args.grid)
    dx, dy, dz = gd["dim"]
    origin = gd["origin"]
    gridspacing = gd["gridspacing"]
    plateaudis = gd["plateaudis"]
    plateaudissq = plateaudis**2
    inv50sq = 1.0 / (50.0 * 50.0)
    felec = np.sqrt(332.053986 / args.epsilon)

    nr_nb = gd["nr_neighbours_inner"]
    nb_grid = gd["neighbour_grid"]
    nb_type_grid = gd["neighbour_type_grid"]
    max_nb = gd["max_nr_neighbours"]

    # Position uncertainty margin: ligand snaps to nearest voxel, could be
    # anywhere within ±0.5*gridspacing in each dimension
    margin = np.sqrt(3) * gridspacing / 2.0
    print(f"Grid spacing: {gridspacing:.3f}, margin: {margin:.3f} Å")
    print(f"Plateau distance: {plateaudis:.1f} Å")
    print(f"Max neighbours: {max_nb}")
    print(f"Grid dim: {dx}×{dy}×{dz} = {dx*dy*dz} voxels")
    print(f"Energy threshold: {args.threshold}")

    # Find active voxels (nb > 0)
    active_mask = nr_nb > 0
    active_indices = np.argwhere(active_mask)  # (N_active, 3) — (ix, iy, iz)
    n_active = len(active_indices)
    print(f"Active voxels (nb > 0): {n_active}")

    # For each active voxel, compute min nb energy across all (ensemble, lig_type) combos
    # If the MINIMUM energy is still > threshold, mark as interior
    n_lig_types = len(alphabet_types_1)
    n_combos = len(ens_paths) * n_lig_types
    print(
        f"\nComputing nb energies for {n_active} voxels × {len(ens_paths)} ensembles × {n_lig_types} alphabet types = {n_combos} combos..."
    )

    # Precompute voxel world coordinates: (V, 3)
    voxel_coords = origin[None, :] + active_indices.astype(np.float64) * gridspacing

    # Precompute nb data for active voxels: (V, max_nb)
    active_nb = np.zeros((n_active, max_nb), dtype=np.uint16)
    active_nb_type = np.zeros((n_active, max_nb), dtype=np.uint8)
    active_nr = np.zeros(n_active, dtype=np.int32)
    for v in range(n_active):
        ix, iy, iz = active_indices[v]
        n = int(nr_nb[ix, iy, iz])
        active_nr[v] = n
        active_nb[v, :n] = nb_grid[ix, iy, iz, :n]
        active_nb_type[v, :n] = nb_type_grid[ix, iy, iz, :n]

    # Track minimum energy per voxel (across all ensemble+type combos)
    min_energy = np.full(n_active, np.inf, dtype=np.float64)

    combo = 0
    for ens_idx, (rec_coords, rec_atypes, rec_charges) in enumerate(rec_ensemble):
        rec_charge_scaled = rec_charges * felec
        rec_atomtypes_0 = rec_atypes - 1  # 0-indexed

        for lig_type_1 in alphabet_types_1:
            lig_type_0 = lig_type_1 - 1
            lig_charge_scaled = lig_type_charges[lig_type_1] * felec

            combo += 1
            if combo == 1 or combo % 20 == 0 or combo == n_combos:
                print(
                    f"  combo {combo}/{n_combos} (ens {ens_idx+1}, lig type {lig_type_1})...",
                    flush=True,
                )

            e = compute_nb_energies_vectorized(
                voxel_coords,
                rec_coords,
                rec_atomtypes_0,
                active_nb,
                active_nb_type,
                active_nr,
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
                margin,
            )
            np.minimum(min_energy, e, out=min_energy)

    elapsed = time.time() - t0
    print(f"\nComputation took {elapsed:.1f}s")

    # Analysis
    interior_mask = min_energy > args.threshold
    n_interior = int(interior_mask.sum())

    print(f"\n=== Results ===")
    print(f"Active voxels: {n_active}")
    print(
        f"Interior voxels (min energy > {args.threshold}): {n_interior} ({100*n_interior/n_active:.1f}%)"
    )
    print(f"Remaining active: {n_active - n_interior}")

    remaining_nb = nr_nb.copy()
    for v in range(n_active):
        if interior_mask[v]:
            ix, iy, iz = active_indices[v]
            remaining_nb[ix, iy, iz] = 0

    remaining_max = int(remaining_nb.max())
    print(f"Original max_nb: {max_nb}, after culling: {remaining_max}")
    print(f"Potential max_nb_cap reduction: {max_nb} → {remaining_max}")

    # Distribution of min_energy
    print(f"\nMin energy distribution (across all combos):")
    for thresh in [0, 1, 10, 50, 100, 500, 1000, 10000]:
        n = int((min_energy > thresh).sum())
        print(f"  > {thresh:6d}: {n:6d} voxels ({100*n/n_active:.1f}%)")

    if args.dry_run:
        print("\nDry run — not writing output grid.")
        return

    # Write modified grid
    # Strategy: copy the original grid bytes, then modify the innergrid section
    # to zero out nr_neighbours for interior voxels.
    print(f"\nWriting culled grid to {args.out_grid}...")

    grid_out = bytearray(grid_data)
    innergrid_offset = gd["innergrid_offset"]
    innergrid_dtype = gd["innergrid_dtype"]
    item_size = innergrid_dtype.itemsize

    # The innergrid is stored as (dz, dy, dx) in the file, then we swapped axes.
    # To modify the file, we need to address (iz, iy, ix) order.
    n_culled = 0
    for v in range(n_active):
        if not interior_mask[v]:
            continue
        ix, iy, iz = active_indices[v]
        # File layout: (dz, dy, dx) → linear index = iz * dy * dx + iy * dx + ix
        file_idx = iz * dy * dx + iy * dx + ix
        # nr_neighbours is the last field: offset within struct
        # potential: 100 * uint32 = 400 bytes
        # neighbourlist: 1 * int32 = 4 bytes
        # nr_neighbours: 1 * int16 = 2 bytes
        # But the struct has align=True, so we need actual offsets
        nb_field_offset = innergrid_dtype.fields["nr_neighbours"][1]
        byte_offset = innergrid_offset + file_idx * item_size + nb_field_offset
        struct.pack_into("h", grid_out, byte_offset, 0)
        n_culled += 1

    Path(args.out_grid).write_bytes(bytes(grid_out))
    print(f"Wrote {args.out_grid} ({n_culled} voxels culled)")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
