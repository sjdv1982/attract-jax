#!/usr/bin/env python3
"""Quick probe of receptor atom density around active grid voxels."""
import struct, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree


def parse_pdb(path):
    coor = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            coor.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.array(coor)


grid_path = "test/receptorgrid.grid"
grid_data = open(grid_path, "rb").read()
pos = 0
pos += 3
gridspacing = struct.unpack_from("d", grid_data, pos)[0]
pos += 8
pos += 4 + 8 + 8 + 99
ox, oy, oz = struct.unpack_from("fff", grid_data, pos)
pos += 12
dx, dy, dz = struct.unpack_from("iii", grid_data, pos)
pos += 12
pos += 12 + 4 + 24
nr_eg = struct.unpack_from("i", grid_data, pos)[0]
pos += 8
pos += nr_eg * 16
nr_nb_total = struct.unpack_from("i", grid_data, pos)[0]
pos += 8
nb_dtype = np.dtype([("type", np.uint8), ("index", np.uint32)], align=True)
pos += nr_nb_total * nb_dtype.itemsize
pos += 8
ig_dtype = np.dtype(
    [("pot", np.uint32, 100), ("nblist", np.int32), ("nr_nb", np.int16)], align=True
)
ig = np.frombuffer(grid_data, offset=pos, count=dx * dy * dz, dtype=ig_dtype)
ig = ig.reshape((dz, dy, dx)).swapaxes(0, 2)
nr_nb = np.ascontiguousarray(ig["nr_nb"])
origin = np.array([ox, oy, oz])

active = nr_nb > 0
aidx = np.argwhere(active)
vcoords = origin + aidx.astype(np.float64) * gridspacing
print(f"Grid: {dx}x{dy}x{dz}, spacing={gridspacing:.3f}")
print(f"Active voxels: {len(aidx)}")

ens_dir = Path("test")
with open("test/partner1-ensemble.list") as f:
    paths = [str(ens_dir / l.strip()) for l in f if l.strip()]

rec = parse_pdb(paths[0])
print(f"Receptor atoms (ens 1): {len(rec)}")

tree = cKDTree(rec)
for r in [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
    counts = tree.query_ball_point(vcoords, r=r, return_length=True)
    print(f"\nRadius {r:.1f} Å:")
    for thr in [1, 3, 5, 8, 15, 30, 50]:
        n = int((counts >= thr).sum())
        print(f"  >= {thr:2d} atoms: {n:6d} voxels ({100*n/len(aidx):.1f}%)")

# Also show what happens for high-nb voxels
print("\n--- High-nb voxels (nb >= 40) density check ---")
highnb = nr_nb[active] >= 40
highnb_coords = vcoords[highnb]
print(f"High-nb voxels: {highnb.sum()}")
for r in [3.0, 5.0, 7.0, 10.0]:
    counts = tree.query_ball_point(highnb_coords, r=r, return_length=True)
    print(
        f"  r={r:.1f}: min={counts.min()}, mean={counts.mean():.1f}, "
        f"max={counts.max()}, median={np.median(counts):.0f}"
    )
