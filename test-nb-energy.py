#from jax import config
#config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import cond
from functools import partial

from read_grid import read_grid

mat = np.load("mono-leu/minim-sorted.npy")
lig = np.load("mono-leu/mono-leuc-pdb.npy")
grid = read_grid(open("mono-leu/mono-leuc.nbgrid", "rb").read())

coor_lig = np.stack([lig["x"],lig["y"], lig["z"], np.ones(len(lig))], axis=1)
coor_rec = coor_lig[:, :3]

all_coors_lig = np.einsum("jk,ikl->ijl", coor_lig, mat) #diagonally broadcasted form of coor_lig.dot(mat)
all_coors_lig = all_coors_lig[:, :, :3]

# ATTRACT params for Leu CSE - Leu CSE (bead 15 - bead 15)

rbc = 3.88
abc = 13.81
ipon = 1
potshape = 8

rc =abc*rbc**potshape
ac =abc*rbc**6

emin=-27.0*ac**4/(256.0*rc**3)
rmin2=4.0*rc/(3.0*ac)
ivor = ipon

def nonbon(dsq, rc, ac, emin, rmin2, ivor):
    rr2 = 1/dsq

    alen = ac
    rlen = rc
    rr23 = rr2 * rr2 * rr2
    rep = rlen * rr2
    vlj = (rep-alen)*rr23
    
    energy = cond(dsq < rmin2, lambda: vlj + (ivor-1) * emin, lambda: ivor * vlj)
    return energy

nonbon2 = lambda dsq: nonbon(dsq, rc, ac, emin, rmin2, ivor)
nonbon2_vec = jnp.vectorize(nonbon2, signature='()->()')


@partial(jit, static_argnums=(2,))
def generate_grid_table(all_coors_lig, grid, grid_dim):
    vox_innergrid =  (all_coors_lig - grid.origin) / grid.gridspacing
    x, y, z = vox_innergrid[:, :, 0], vox_innergrid[:, :, 1], vox_innergrid[:, :, 2]
    in_innergrid = ((x >= 0) & (x < grid.dim[0]-1) & (y >= 0) & (y < grid.dim[1]-1) & (z >= 0) & (z < grid.dim[2]-1))

    vox_outergrid =  (vox_innergrid + grid.gridextension)/2
    x, y, z = vox_outergrid[:, :, 0], vox_outergrid[:, :, 1], vox_outergrid[:, :, 2]
    in_outergrid = ((x >= 0) & (x < grid.dim2[0]) & (y >= 0) & (y < grid.dim2[1]) & (z >= 0) & (z < grid.dim2[2]))

    '''
    print(in_innergrid.sum()).
    print(in_outergrid.sum())
    assert np.sum(in_innergrid & ~in_outergrid) == 0
    '''

    gridtype = in_innergrid.astype(np.uint8) + in_outergrid
    inner_mask = (gridtype == 2)[:,:,None]

    potential = jnp.where(in_outergrid[:,:,None], jnp.where(inner_mask, vox_innergrid, vox_outergrid), 0)
    pos_innergrid = jnp.where(inner_mask, jnp.floor(vox_innergrid+0.5), 0).astype(np.int32)
    
    '''
    for i in range(3):
        assert pos_innergrid[:, :, i].min() >= 0 and pos_innergrid[:, :, i].max() < grid.dim2[i], (i, pos_innergrid[:, :, i].min(), pos_innergrid[:, i].max(), grid.dim2[i])
    '''
    ind_innergrid = jnp.ravel_multi_index((pos_innergrid[:, :, 0], pos_innergrid[:, :, 1],pos_innergrid[:, :, 2]),dims=grid_dim, mode="clip")

    nb_index = jnp.take(grid.neighbour_grid.reshape(-1, 2), ind_innergrid,axis=0)
    return gridtype, nb_index, potential


def build_nb_list(nb_index, neighbours, neighbour_grid):
    pass

d = {}
for field in grid._fields:
    value = getattr(grid, field)
    if isinstance(value, np.ndarray) and value.shape != (3,):
        value = jnp.array(value)
    d[field] = value
grid = type(grid)(**d)

plateau_energy = nonbon(grid.plateaudis**2, rc, ac, emin, rmin2, ivor)
print(plateau_energy)

lig_atomtypes = jnp.ones(len(coor_lig), np.uint8)
rec_atomtypes = jnp.ones(len(coor_rec), np.uint8)

all_coors_lig = all_coors_lig[:10] ###

def nb_nonbon(struc, start, end, atom_index, nb, neighbours):
    atom_ind = atom_index[start:end]
    nb_range = nb[start:end]
    atom_type = lig_atomtypes[atom_ind]
    lig_coor = all_coors_lig[struc, atom_ind]
    print(struc+1, start, end, len(nb))
    energy = 0
    for i, (atp, ligc, (nb_length, nb_start)) in enumerate(zip(atom_type, lig_coor, nb_range)):
        def func():
            rec_atom_indices = neighbours[nb_start-1:nb_start+nb_length-1]
            rec_coor = coor_rec[rec_atom_indices]
            d = rec_coor - ligc
            dsq = (d * d).sum(axis=1)
            # if potentials:
            #ene = jnp.where(dsq<grid.plateaudis**2, nonbon2_vec(dsq) - plateau_energy, 0).sum()
            
            # if no potentials:
            ene = jnp.where(dsq<grid.plateaudis**2, jit(nonbon2_vec)(dsq), 0).sum()
            return ene
        ene = func()
        energy += ene
    return energy
#nb_nonbon_vec = jnp.vectorize(nb_nonbon, excluded=(3,4,5), signature='(),(),()->()')

grid_table = generate_grid_table(all_coors_lig, grid, tuple(grid.dim))

gridtype, nb_index, potential = grid_table

nb_index2 = jnp.where(nb_index[:, :, 0] > 0)
nb = nb_index[nb_index2]

nb_struc_offset = jnp.searchsorted(nb_index2[0], jnp.arange(len(all_coors_lig)+1))
nb_struc = jnp.arange(len(nb_struc_offset)-1)
nb_struc_start = nb_struc_offset[:-1]
nb_struc_end = nb_struc_offset[1:]
nb_atom_index = nb_index2[1]
#energies = nb_nonbon_vec(nb_struc, nb_struc_start, nb_struc_end, nb_atom_index, nb, grid.neighbours)
energies = []
for struc, start, end in zip(nb_struc, nb_struc_start, nb_struc_end):
    energy = nb_nonbon(struc, start, end, nb_atom_index, nb, grid.neighbours)
    print(energy)
    energies.append(energy)

