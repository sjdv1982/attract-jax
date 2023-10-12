# Calculates the grid on the fly 
# Compare with reading it from disk

# from jax import config
# config.update("jax_enable_x64", True)
DEBUG = False

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree as KDTree
from jax import jit
from jax.lax import cond
from functools import partial

from collections import namedtuple
from read_grid import read_grid

if DEBUG:
    jnp = np
    jit = lambda func, **kwargs: func
    def cond(expr, func1, func2, *args):
        if expr:
            return func1(*args)
        else:
            return func2(*args)

mat = jnp.load("mono-leu/minim-sorted.npy")
mat = mat[:10] ###
lig = jnp.load("mono-leu/mono-leuc-pdb.npy")
# lig = lig[:1] ### one atom
rec = lig.copy()

potshape = 8

par = np.load("attract-par.npz")
rc, ac, ivor = par["rc"], par["ac"], par["ivor"]

rec_atomtypes00 = rec["occupancy"].astype(np.uint8)
mask = rec_atomtypes00 != 99
rec = rec[mask]
rec_atomtypes0 = rec_atomtypes00[mask]
rec_mapping = np.cumsum(mask)-1

lig_atomtypes00 = lig["occupancy"].astype(np.uint8)
mask = rec_atomtypes00 != 99
lig = lig[mask]
lig_atomtypes0 = lig_atomtypes00[mask]

lig_alphabet, lig_atomtypes = np.unique(lig_atomtypes0, return_inverse=True)
rec_alphabet, rec_atomtypes = np.unique(rec_atomtypes0, return_inverse=True)

coor_rec = np.stack([rec["x"], rec["y"], rec["z"]], axis=1)
coor_lig = np.stack([lig["x"], lig["y"], lig["z"]], axis=1)

tree_rec = KDTree(coor_rec)

rc = rc[rec_alphabet-1][:, lig_alphabet-1]
ac = ac[rec_alphabet-1][:, lig_alphabet-1]
ivor = ivor[rec_alphabet-1][:, lig_alphabet-1]

emin = -27.0 * ac**4 / (256.0 * rc**3)
rmin2 = 4.0 * rc / (3.0 * ac)
ff = {}
ff["rc"] = rc
ff["ac"] = ac
ff["ivor"] = ivor
ff["emin"] = emin
ff["rmin2"] = rmin2
ff = namedtuple("ff", field_names=ff.keys())(**ff)

grid = read_grid(open("mono-leu/mono-leuc.grid", "rb").read())
# grid = read_grid(open("mono-leu/mono-leuc-1atom.grid", "rb").read()) ### one atom
grid.neighbours[:] = rec_mapping[grid.neighbours]
alpos = (np.where(grid.alphabet)[0]+1).tolist()
lig_alphabet_pos = np.array([alpos.index(a) for a in lig_alphabet])
lig_atomtype_pos = jnp.array(lig_alphabet_pos[lig_atomtypes])

if not DEBUG:
    d = {}
    for field in grid._fields:
        value = getattr(grid, field)
        if isinstance(value, np.ndarray) and value.shape != (3,):
            value = jnp.array(value)
        d[field] = value
    grid = type(grid)(**d)

# Calculate neigbours and neighbour_grid
grid_ind0=np.transpose(np.mgrid[0:grid.dim[2], 0:grid.dim[1], 0:grid.dim[0]], (3,2,1,0))
grid_ind = grid_ind0.reshape(-1, 3)[:, ::-1]
grid_coor = (grid_ind + 0.5) * grid.gridspacing + grid.origin
tree_grid_coor = KDTree(grid_coor)
neighbours_lists = tree_grid_coor.query_ball_tree(tree_rec, grid.neighbourdis)
neighbours = np.concatenate(neighbours_lists).astype(np.uint32)

neighbour_grid_len = np.array([len(l) for l in neighbours_lists])
neighbour_grid0 = np.empty((len(grid_coor), 2), dtype=np.int32)
neighbour_grid0[:, 0] = neighbour_grid_len
neighbour_grid0[:, 1][0] = 0
neighbour_grid0[:, 1][1:] = neighbour_grid_len.cumsum()[:-1]
neighbour_grid = neighbour_grid0.reshape(list(grid.dim) + [2]) 

from scipy.stats import pearsonr
print("Number of neighbours:", len(neighbours), len(grid.neighbours))
print("Neighbour grid correlation:", pearsonr(
    neighbour_grid.reshape(-1,2)[:, 0], 
    grid.neighbour_grid.reshape(-1,2)[:, 0]
)[0])

# Calculate potential energy

@jit
def nonbon(dsq, ff, at1, at2):
    rr2 = 1 / dsq

    alen = ff.ac[at1, at2]
    rlen = ff.rc[at1, at2]
    rr23 = rr2 * rr2 * rr2
    rep = rlen * rr2
    vlj = (rep - alen) * rr23
    attraction = ff.ivor[at1, at2]

    energy = cond(
        dsq < ff.rmin2[at1, at2],
        lambda: vlj + (attraction - 1) * ff.emin[at1, at2],
        lambda: attraction * vlj,
   )
    return energy

@jit
def nonbon_pot2(dsq, ff, at1, at2, plateau_energy):
    energy = cond(
        dsq < plateaudissq,
        lambda: plateau_energy,
        lambda: nonbon(dsq, ff, at1, at2),
    )
    return energy

@jit
def nonbon_pot(dsq, ff, at1, at2, plateau_energy):
    energy = cond(
        dsq < 2500,
        lambda: nonbon_pot2(dsq, ff, at1, at2, plateau_energy),
        lambda: 0.0,
   )
    return energy

plateaudissq = grid.plateaudis**2

nonbon2 = jit(jnp.vectorize(nonbon_pot, excluded=(1,4), signature="(),(),()->()"))

def calc_grid():

    grid1_ind0 = jnp.transpose(jnp.mgrid[0:grid.dim[2], 0:grid.dim[1], 0:grid.dim[0]], (3,2,1,0))
    grid1_ind = grid1_ind0.reshape(-1, 3)[:, ::-1]
    grid1_coor = grid1_ind * grid.gridspacing + grid.origin

    grid2_ind0 = jnp.transpose(jnp.mgrid[0:grid.dim2[2], 0:grid.dim2[1], 0:grid.dim2[0]], (3,2,1,0))
    grid2_ind = grid2_ind0.reshape(-1, 3)[:, ::-1]
    grid2_coor = (grid2_ind - grid.gridextension/2.0 ) * (2 * grid.gridspacing) + grid.origin

    nr_atomtypes = len(lig_alphabet)
    inner_potential_grid = jnp.zeros([nr_atomtypes] + list(grid.dim), np.float32 )
    outer_potential_grid = jnp.zeros([nr_atomtypes] + list(grid.dim2), np.float32 )

    CHUNKSIZE=int(50000000/len(coor_rec))
    for rec_atomtype in range(len(rec_alphabet)):
        curr_coor_rec = coor_rec[rec_atomtypes==rec_atomtype]
        curr_coor_rec_sq = (curr_coor_rec**2).sum(-1)[None, :]
        for it in 1,2:
            if it == 1:
                grid_coor = grid1_coor
                potential_grid_shape = inner_potential_grid.shape
            else:
                grid_coor = grid2_coor
                potential_grid_shape = outer_potential_grid.shape
            for lig_atomtype in range(len(lig_alphabet)): # pylint: disable=consider-using-enumerate
                plateau_energy = nonbon(plateaudissq, ff, rec_atomtype, lig_atomtype)
                curr_potential_grid = jnp.zeros(potential_grid_shape[1:]).reshape(-1)
                print(rec_atomtype, lig_atomtype)

                for pos in range(0, len(grid_coor), CHUNKSIZE):
                    curr_grid_coor = grid_coor[pos:pos+CHUNKSIZE]
                    dsq = (curr_grid_coor**2).sum(-1)[:, None] \
                        + curr_coor_rec_sq \
                        - 2 * jnp.dot(curr_grid_coor , curr_coor_rec.T)
                    ene = nonbon2(dsq, ff, rec_atomtype, lig_atomtype, plateau_energy)
                    curr_potential_grid = curr_potential_grid.at[pos:pos+CHUNKSIZE].set(ene.sum(axis=1))
            curr_potential_grid = curr_potential_grid.reshape(potential_grid_shape[1:])
            if it == 1:
                inner_potential_grid = inner_potential_grid.at[lig_atomtype].set(curr_potential_grid)
            else:
                outer_potential_grid = outer_potential_grid.at[lig_atomtype].set(curr_potential_grid)
    return inner_potential_grid, outer_potential_grid
t = time.time()
inner_potential_grid, outer_potential_grid = calc_grid()
print("Time", time.time() - t)
print("Inner potential grid correlation:", pearsonr(
   inner_potential_grid[0].reshape(-1),
   grid.inner_potential_grid[lig_alphabet[0]-1].reshape(-1) 
)[0])
print("Outer potential grid correlation:", pearsonr(
   outer_potential_grid[0].reshape(-1),
   grid.outer_potential_grid[lig_alphabet[0]-1].reshape(-1) 
)[0])