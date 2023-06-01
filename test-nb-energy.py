#from jax import config
#config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import cond
from functools import partial
import time

from read_grid import read_grid

mat = np.load("mono-leu/minim-sorted.npy")
lig = np.load("mono-leu/mono-leuc-pdb.npy")
grid = read_grid(open("mono-leu/mono-leuc.nbgrid", "rb").read())

coor_lig = np.stack([lig["x"],lig["y"], lig["z"], np.ones(len(lig))], axis=1)
coor_rec = coor_lig[:, :3]
coor_rec = jnp.array(coor_rec)

all_coors_lig = np.einsum("jk,ikl->ijl", coor_lig, mat) #diagonally broadcasted form of coor_lig.dot(mat)
all_coors_lig = all_coors_lig[:, :, :3]
all_coors_lig = jnp.array(all_coors_lig)

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
    
    # we are interested in gridtype=2 (inner grid), sorted by length (nb_index[:,:,0]), in descending order
    # TODO: make sure that max #contacts is lower than 1000
    key = (-1000.0 * gridtype - nb_index[:, :, 0]).astype(jnp.int16)
    # Slow! 2 secs for 10k structures... (https://github.com/google/jax/issues/10434)
    #   XLA argsort is slow on CPU. Should be fine on GPU. 
    sort_index = jnp.unravel_index(jnp.argsort(key,axis=None), key.shape)    

    return gridtype, nb_index, potential, sort_index


@jit
def nb_energy_single(nb_index, offset, ligand_struc, ligand_atom):

    receptor_atom = grid.neighbours[nb_index + offset - 1]
    rec_c = coor_rec[receptor_atom]
    lig_c = all_coors_lig[ligand_struc, ligand_atom]
    d = (rec_c - lig_c)
    dsq = (d*d).sum()
    # if potentials present, also subtract plateau energy
    return cond(dsq<plateaudissq, nonbon2, lambda dsq: 0.0, dsq)

nb_energy = jnp.vectorize(nb_energy_single, excluded=(1,))

d = {}
for field in grid._fields:
    value = getattr(grid, field)
    if isinstance(value, np.ndarray) and value.shape != (3,):
        value = jnp.array(value)
    d[field] = value
grid = type(grid)(**d)

plateaudissq = grid.plateaudis**2
plateau_energy = nonbon(plateaudissq, rc, ac, emin, rmin2, ivor)
print(plateau_energy)

lig_atomtypes = jnp.ones(len(coor_lig), np.uint8)
rec_atomtypes = jnp.ones(len(coor_rec), np.uint8)

#all_coors_lig = all_coors_lig[:20] ###

grid_table = generate_grid_table(all_coors_lig, grid, tuple(grid.dim))
t = time.time()
grid_table = jax.block_until_ready(generate_grid_table(all_coors_lig, grid, tuple(grid.dim)))
print(time.time() - t)

# Numpy: 0.36 secs instead of 2 secs
gridtype, nb_index, potential, sort_index = grid_table
gridtypeX = np.array(gridtype)
nb_indexX = np.array(nb_index)

t = time.time()
key = (-1000.0 * gridtypeX - nb_indexX[:, :, 0]).astype(np.int16)
sort_index = np.unravel_index(np.argsort(key,axis=None), key.shape)    
print(time.time() - t)

gridtype, nb_index, potential, sort_index = grid_table
nr_inner_atoms = jnp.searchsorted(-gridtype[sort_index], -2, side="right")
sort_index = sort_index[:nr_inner_atoms]

# we are interested in length > 0
sorted_nb_index = nb_index[sort_index]
max_contacts = sorted_nb_index[0, 0]

# The number of atoms with contacts c: c=max_contacts, c>=max_contacts-1, c>=max_contacts-2, ..., c>=1 
iter_pos = jnp.searchsorted(-sorted_nb_index[:, 0], jnp.arange(-max_contacts+1, 1))

MIN_CHUNK=2**21
MAX_CHUNK=2**26

# TODO: in jnp, so it is on the GPU
# anyway, probably unnecessary, always a single chunk should do on a GPU 
# TODO: more clever tiling afterwards...
def chunker(s, offset=0):
    if s <= MIN_CHUNK:
        result = [(offset, s, MIN_CHUNK)]
    elif s > MAX_CHUNK:
        result = [(offset, MAX_CHUNK, MAX_CHUNK)] + chunker(s-MAX_CHUNK, offset+MAX_CHUNK)
    else:
        c = MIN_CHUNK
        while s > c:
            c *= 2
        c //= 2
        result = [(offset, c, c)] + chunker(s-c, offset+c)
    return result

chunks = [chunker(int(ip)) for ip in iter_pos]

for runs in (1,2):
    # below runs in 1.8 secs on a CPU when JIT is warm
    chunk_energies = []
    for i, chs in enumerate(chunks[::-1]):
        for ch in chs:
            start, realsize, padded_size = ch
            chunk_nb_index = sorted_nb_index[start:start+padded_size, 1]
            chunk_ligand_struc = sort_index[0][start:start+padded_size]
            chunk_ligand_atom = sort_index[1][start:start+padded_size]
            offset = i
            energy = nb_energy(chunk_nb_index, offset, chunk_ligand_struc, chunk_ligand_atom)
            print(i, start, realsize)
            chunk_energies.append((i, start, realsize, energy))


# TODO: to jnp to run on GPU
nr_energies = iter_pos[-1]
atom_energies = np.zeros(nr_energies)
for i, start, realsize, energy in chunk_energies:
    atom_energies[start:start+realsize] += energy[:realsize]
    
energies = np.zeros(len(all_coors_lig))
ligand_struc = sort_index[0][:nr_energies]
np.add.at(energies, ligand_struc, atom_energies)    
