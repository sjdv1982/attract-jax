#!/usr/bin/env python3
"""Analyze grid voxel neighbour count distribution.

Loads the receptor grid and examines:
1. Distribution of nr_neighbours across inner grid voxels
2. What fraction of voxels are "interior" (high nb count → always repulsive)
3. Potential speedup from zeroing interior voxel neighbor lists

Usage:
    conda run -n jax python analyze_grid_nb.py /path/to/receptorgrid.grid
"""

import sys
import struct
import numpy as np


def read_grid_nb_info(grid_path):
    """Read grid and return nr_neighbours array + grid metadata."""
    grid_data = open(grid_path, "rb").read()
    pos = 0

    def read_int():
        nonlocal pos
        v = struct.unpack_from("<i", grid_data, pos)[0]
        pos += 4
        return v

    def read_double():
        nonlocal pos
        v = struct.unpack_from("<d", grid_data, pos)[0]
        pos += 8
        return v

    def read_float():
        nonlocal pos
        v = struct.unpack_from("<f", grid_data, pos)[0]
        pos += 4
        return v

    def read_long():
        nonlocal pos
        v = struct.unpack_from("<q", grid_data, pos)[0]
        pos += 8
        return v

    def read_bool():
        nonlocal pos
        v = struct.unpack_from("?", grid_data, pos)[0]
        pos += 1
        return v

    def read_short():
        nonlocal pos
        v = struct.unpack_from("h", grid_data, pos)[0]
        pos += 2
        return v

    def _r(fmt, count):
        nonlocal pos
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, grid_data, pos)
        pos += size
        return vals

    is_torquegrid = read_bool()
    arch = read_short()
    assert not is_torquegrid, "Torque grids not supported"
    assert arch == 64, f"Unexpected arch: {arch}"

    gridspacing = read_double()
    gridextension = read_int()
    plateaudis = read_double()
    neighbourdis = read_double()
    alphabet = np.array(_r("?" * 99, 99), bool)
    nr_vdw_channels = int(alphabet.sum())

    origin = np.array((read_float(), read_float(), read_float()), dtype=np.float64)
    dx, dy, dz = read_int(), read_int(), read_int()
    dx2, dy2, dz2 = read_int(), read_int(), read_int()
    natoms = read_int()
    pivot = np.array((read_double(), read_double(), read_double()), dtype=np.float64)

    # Skip energrads
    nr_energrads = read_int()
    shm_energrads = read_int()
    energrads_bytes = nr_energrads * 4 * 4  # 4 floats per entry
    pos += energrads_bytes

    # Skip neighbours
    nr_neighbours = read_int()
    shm_neighbours = read_int()
    nb_dtype = np.dtype([("type", np.uint8), ("index", np.uint32)], align=True)
    pos += nr_neighbours * nb_dtype.itemsize

    # Read inner grid
    innergridsize = read_long()
    assert (
        innergridsize == dx * dy * dz
    ), f"Inner grid size mismatch: {innergridsize} vs {dx*dy*dz}"

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
    innergrid = innergrid.reshape((dz, dy, dx)).swapaxes(0, 2)
    innergrid = np.ascontiguousarray(innergrid)

    nr_neighbours_inner = np.ascontiguousarray(innergrid["nr_neighbours"])

    # Also read the potential grid to check if high-nb voxels have repulsive potentials
    # The potential indices are in innergrid["potential"], 1-indexed into energrads
    # We already skipped past energrads, so let's re-read them
    # Actually, let's just return what we need

    # Re-read energrads for potential analysis
    pos2 = 0
    pos2 += 1  # is_torquegrid (bool)
    pos2 += 2  # arch (short)
    pos2 += 8  # gridspacing
    pos2 += 4  # gridextension
    pos2 += 8  # plateaudis
    pos2 += 8  # neighbourdis
    pos2 += struct.calcsize("?" * 99)  # alphabet
    pos2 += 4 * 3  # origin (3 floats)
    pos2 += 4 * 6  # dim + dim2 (6 ints)
    pos2 += 4  # natoms
    pos2 += 8 * 3  # pivot (3 doubles)
    pos2 += 4  # nr_energrads
    pos2 += 4  # shm_energrads

    energrads = np.frombuffer(
        grid_data, offset=pos2, count=nr_energrads * 4, dtype=np.float32
    ).reshape(
        nr_energrads, 4
    )  # (energy, gx, gy, gz)

    # Get the first active atomtype's potential grid values
    pot_indices = innergrid["potential"]  # (dx, dy, dz, 100), 1-indexed
    first_active = np.where(alphabet)[0][0]  # first active atomtype index
    pot_idx_first = pot_indices[:, :, :, first_active]  # 1-indexed

    # Get energy values for voxels with nonzero potential
    mask_has_pot = pot_idx_first > 0
    pot_energies = np.zeros((dx, dy, dz), dtype=np.float32)
    pot_energies[mask_has_pot] = energrads[pot_idx_first[mask_has_pot] - 1, 0]

    return {
        "nr_neighbours": nr_neighbours_inner,
        "pot_energies_first_type": pot_energies,
        "dim": (dx, dy, dz),
        "gridspacing": gridspacing,
        "plateaudis": plateaudis,
        "neighbourdis": neighbourdis,
        "natoms": natoms,
        "max_nr_neighbours": int(nr_neighbours_inner.max()),
    }


def analyze(grid_path):
    info = read_grid_nb_info(grid_path)
    nb = info["nr_neighbours"]
    pot = info["pot_energies_first_type"]

    print(f"Grid dimensions: {info['dim']}")
    print(f"Grid spacing: {info['gridspacing']}")
    print(f"Plateau distance: {info['plateaudis']}")
    print(f"Neighbour distance: {info['neighbourdis']}")
    print(f"Receptor atoms: {info['natoms']}")
    print(f"Total voxels: {nb.size}")
    print()

    print("=== Neighbour count distribution ===")
    print(f"Min: {nb.min()}, Max: {nb.max()}, Mean: {nb.mean():.1f}")
    print(
        f"Voxels with 0 neighbours: {(nb == 0).sum()} ({100*(nb==0).sum()/nb.size:.1f}%)"
    )
    print()

    # Histogram in bins
    thresholds = [0, 1, 5, 10, 20, 30, 40, 50, 80, 100, 150, 200]
    print(f"{'Range':>12s}  {'Count':>8s}  {'Pct':>6s}  {'Cum%':>6s}")
    cum = 0
    for i in range(len(thresholds)):
        lo = thresholds[i]
        hi = thresholds[i + 1] if i + 1 < len(thresholds) else nb.max() + 1
        count = int(((nb >= lo) & (nb < hi)).sum())
        cum += count
        pct = 100 * count / nb.size
        cum_pct = 100 * cum / nb.size
        print(f"  [{lo:3d}, {hi:3d})  {count:8d}  {pct:5.1f}%  {cum_pct:5.1f}%")

    print()
    print("=== Potential energy at high-nb voxels (first active atomtype) ===")
    for threshold in [20, 30, 40, 50, 80]:
        mask = nb >= threshold
        if mask.sum() == 0:
            continue
        pot_vals = pot[mask]
        print(
            f"  nb >= {threshold}: {mask.sum()} voxels, "
            f"pot energy: min={pot_vals.min():.1f}, mean={pot_vals.mean():.1f}, "
            f"max={pot_vals.max():.1f}"
        )
        # How many have strongly repulsive potential?
        repulsive = pot_vals > 10.0
        print(
            f"    pot > 10: {repulsive.sum()} ({100*repulsive.sum()/len(pot_vals):.1f}%)"
        )
        very_repulsive = pot_vals > 100.0
        print(
            f"    pot > 100: {very_repulsive.sum()} ({100*very_repulsive.sum()/len(pot_vals):.1f}%)"
        )

    print()
    print("=== Impact on JAX kernel ===")
    # In main_ad, every ligand atom in the inner grid gets nb_energy_vectorized
    # called with K=max_nb_cap offsets. The cost is proportional to K.
    # If we zero out interior voxels, those atoms still get called but
    # all K offsets are invalid (receptor_atom = 0xFFFF), so the masking
    # already handles it. But we avoid actual pair energy computation.
    #
    # Actually the bigger win: if interior voxels have nb=0, then in the
    # sorting-based path, they won't be "inner atoms" at all, reducing N.
    # In main_ad (no sorting), the nb_energy_vectorized still runs on all
    # atoms, but with all-invalid neighbours the result is 0.

    # What fraction of inner-grid atoms (nb > 0) are "interior" at various thresholds?
    active = nb > 0
    print(f"Active voxels (nb > 0): {active.sum()} ({100*active.sum()/nb.size:.1f}%)")
    for threshold in [20, 30, 40, 50]:
        high = nb >= threshold
        frac = high.sum() / max(active.sum(), 1)
        print(f"  nb >= {threshold}: {high.sum()} voxels = {100*frac:.1f}% of active")

    print()
    print("=== Recommended optimization ===")
    # For each voxel, if the grid potential energy is already very large
    # (say > 100), the pose is garbage regardless of nb correction.
    # We can set nr_neighbours=0 for those voxels.
    # This is safe because the nb correction subtracts energy at the
    # plateau distance — if the grid pot is >> 100, the nb correction
    # (a few units at most) doesn't matter.

    for pot_thresh in [10, 50, 100, 500]:
        cullable = (pot > pot_thresh) & (nb > 0)
        remaining_max_nb = (
            nb[~cullable & (nb > 0)].max() if (~cullable & (nb > 0)).any() else 0
        )
        print(
            f"  Cull voxels with pot > {pot_thresh}: "
            f"{cullable.sum()} voxels culled, "
            f"remaining max_nb = {remaining_max_nb}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <grid_file>")
        sys.exit(1)
    analyze(sys.argv[1])
