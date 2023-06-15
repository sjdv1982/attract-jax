# from jax import config
# config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import cond
from read_grid import read_grid
from functools import partial
from collections import namedtuple
import sys

def _run_argsort(args) -> jnp.ndarray:
    jax.debug.print("Run argsort using Numpy")
    arr, axis = args
    return np.argsort(arr, axis=axis).astype(np.int32)

def run_argsort(arr:jnp.ndarray, axis=None) -> jnp.ndarray:
    if jax.devices()[0].device_kind != "cpu":
        return jnp.argsort(arr,axis=axis)
    
    if axis is None:
        result_shape = arr.ravel().shape
    else:
        result_shape = arr.shape
    result_shape=jax.ShapeDtypeStruct(result_shape, np.int32)

    return jax.pure_callback(
        _run_argsort, result_shape, (arr, axis)
    )    

if jax.devices()[0].device_kind == "cpu":
    run_argsort = jax.custom_jvp(run_argsort)
    
    @run_argsort.defjvp
    def default_grad(primals, tangents):
        return run_argsort(*primals), run_argsort(*tangents)

mat = jnp.load("mono-leu/minim-sorted.npy")
#mat = mat[:10] ###
lig = jnp.load("mono-leu/mono-leuc-pdb.npy")
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

grid = read_grid(open("mono-leu/mono-leuc.grid", "rb").read(), read_gradients=True)
grid.neighbours[:] = rec_mapping[grid.neighbours]
alpos = (np.where(grid.alphabet)[0]+1).tolist()
lig_alphabet_pos = np.array([alpos.index(a) for a in lig_alphabet])
lig_atomtype_pos = jnp.array(lig_alphabet_pos[lig_atomtypes])

d = {}
for field in grid._fields:
    value = getattr(grid, field)
    if isinstance(value, np.ndarray) and value.shape != (3,):
        value = jnp.array(value)
    d[field] = value
grid = type(grid)(**d)

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

grad_nonbon=jax.grad(nonbon)
nonbon2 = jnp.vectorize(nonbon, excluded=(1,), signature="(),(),()->()")
grad_nonbon2 = jnp.vectorize(grad_nonbon, excluded=(1,), signature="(),(),()->()")

'''
def nonbon_d(d,ff, at1, at2):
    dsq = (d*d).sum()
    return nonbon(dsq,ff, at1, at2)
grad_nonbon_d = jax.grad(nonbon_d)
'''

plateaudissq = grid.plateaudis**2
_at1, _at2 = jnp.indices((len(rec_alphabet), len(lig_alphabet) ))
plateau_energies = nonbon2(plateaudissq,ff, _at1, _at2)
plateau_gradients = grad_nonbon2(plateaudissq,ff, _at1, _at2)

def nonbon_dif(d,ff, at1, at2):
    dsq = (d*d).sum()
    return nonbon(dsq,ff, at1, at2) - plateau_energies[at1, at2]

nonbon_dif = jax.custom_jvp(nonbon_dif)
@nonbon_dif.defjvp
@jit
def nonbon_dif_jvp(primals, tangents):    
    d,ff, at1, at2 = primals
    ans = nonbon_dif(*primals)
    d_dot = tangents[0]
    dsq = (d*d).sum()
    #grad_main = grad_nonbon_d(d,ff,at1,at2)
    grad_main=2*grad_nonbon(dsq,ff,at1,at2)*d
    ratio = jnp.sqrt(plateaudissq/dsq)
    #grad_plateau = grad_nonbon_d(d * ratio ,ff,at1,at2)
    grad_plateau = 2*plateau_gradients[at1, at2]*d*ratio
    gradient = ((grad_main - grad_plateau) * d_dot).sum()
    return ans, gradient

@jit
def nb_energy_single(nb_index, offset, lig_struc, lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff):

    receptor_atom = grid.neighbours[nb_index + offset - 1]
    rec_c = coor_rec[receptor_atom]
    at1 = rec_atomtypes[receptor_atom]
    lig_c = all_coors_lig[lig_struc, lig_atom]
    at2 = lig_atomtypes[lig_atom]
    d = (lig_c - rec_c)
    dsq = (d*d).sum()
    return cond(dsq<plateaudissq, nonbon_dif, lambda *args: 0.0, d,ff, at1, at2)


nb_energy = jnp.vectorize(nb_energy_single, excluded=(1,4,5,6,7,8))

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

def eval_potential_grid(lig_atomtype_pos, voxel, potential_grid):
    voxel0 = jnp.floor(voxel).astype(np.int32)
    voxel1 = jnp.ceil(voxel).astype(np.int32)
    w0 = voxel1 - voxel 
    w1 = voxel - voxel0
    result = jnp.zeros(4)
    for wx, x in ((w0[0], voxel0[0]), (w1[0], voxel1[0])):
        for wy, y in ((w0[1], voxel0[1]), (w1[1], voxel1[1])):
            for wz, z in ((w0[2], voxel0[2]), (w1[2], voxel1[2])):
                w = wx * wy * wz
                result += w * potential_grid[lig_atomtype_pos,x,y,z]
    return result

# NOTE: jit is essential here, else memory usage goes through the roof
# (inner/outer_potential_grid cloned for every vectorization!)
@jit
def potential_energrad_voxel(lig_atomtype_pos, vox_innergrid, in_innergrid, vox_outergrid, in_outergrid, inner_potential_grid, outer_potential_grid):
    def inner():
        return eval_potential_grid(lig_atomtype_pos, vox_innergrid, inner_potential_grid)
    def outer():
        return eval_potential_grid(lig_atomtype_pos, vox_outergrid, outer_potential_grid)
    def not_inner():
        return cond(in_outergrid, outer, lambda: jnp.zeros(4))
    return cond(in_innergrid, inner, not_inner)
        
potential_energrad_all = jnp.vectorize(
    potential_energrad_voxel, 
    excluded=(5,6),
    signature="(),(m),(),(m),()->(3)"
)

@jit
def potential_energrad(all_coors_lig, lig_atomtype_pos, grid):
    vox_innergrid =  (all_coors_lig - grid.origin) / grid.gridspacing
    x, y, z = vox_innergrid[:, :, 0], vox_innergrid[:, :, 1], vox_innergrid[:, :, 2]
    in_innergrid = ((x >= 0) & (x < grid.dim[0]-1) & (y >= 0) & (y < grid.dim[1]-1) & (z >= 0) & (z < grid.dim[2]-1))

    vox_outergrid =  (vox_innergrid + grid.gridextension)/2
    x, y, z = vox_outergrid[:, :, 0], vox_outergrid[:, :, 1], vox_outergrid[:, :, 2]
    in_outergrid = ((x >= 0) & (x < grid.dim2[0]-1) & (y >= 0) & (y < grid.dim2[1]-1) & (z >= 0) & (z < grid.dim2[2]-1))

    atom_energrads = potential_energrad_all(lig_atomtype_pos, vox_innergrid, in_innergrid, vox_outergrid, in_outergrid, grid.inner_potential_grid, grid.outer_potential_grid)
    return atom_energrads

@jit
def potential_energies(all_coors_lig, lig_atomtype_pos, grid):
    atom_energrads = potential_energrad(all_coors_lig, lig_atomtype_pos, grid)
    return atom_energrads[:, :, 0]

potential_energies = jax.custom_jvp(potential_energies)
@potential_energies.defjvp
@jit
def potential_energies_jvp(primals, tangents):
    atom_energrads = potential_energrad(*primals)
    ans = atom_energrads[:, :, 0]
    atom_gradients = -atom_energrads[:, :, 1:4] # sign difference with ATTRACT 
    return ans, (atom_gradients * tangents[0]).sum(axis=2)

@jit
def potential_energy(all_coors_lig, lig_atomtype_pos, grid):
    atom_energies = potential_energies(all_coors_lig, lig_atomtype_pos, grid)
    return atom_energies.sum(axis=1)

@partial(jit, static_argnames=("grid_dim",))
def generate_nb_table(all_coors_lig, grid, grid_dim):
    vox_innergrid =  (all_coors_lig - grid.origin) / grid.gridspacing
    x, y, z = vox_innergrid[:, :, 0], vox_innergrid[:, :, 1], vox_innergrid[:, :, 2]
    in_innergrid = ((x >= 0) & (x < grid.dim[0]-1) & (y >= 0) & (y < grid.dim[1]-1) & (z >= 0) & (z < grid.dim[2]-1))

    pos_innergrid = jnp.where(in_innergrid[:, :, None], jnp.floor(vox_innergrid+0.5), 0).astype(np.int32)
    ind_innergrid = jnp.ravel_multi_index((pos_innergrid[:, :, 0], pos_innergrid[:, :, 1],pos_innergrid[:, :, 2]),dims=grid_dim, mode="clip")
    nb_index = jnp.take(grid.neighbour_grid.reshape(-1, 2), ind_innergrid,axis=0)

    in_innergrid2 = in_innergrid.astype(jnp.int8)
    key = in_innergrid2 * -nb_index[:, :, 0]
    sort_index = jnp.unravel_index(run_argsort(key,axis=None), key.shape)
    nr_inner_atoms = jnp.searchsorted(key[sort_index], 0, side="left")
    return nb_index, sort_index, nr_inner_atoms

def neighbour_energy(coor_rec, rec_atomtypes, all_coors_lig, lig_atomtypes, ff, grid):
    nb_index, sort_index, nr_inner_atoms = generate_nb_table(all_coors_lig, grid, tuple(grid.dim))
    sort_index = (sort_index[0][:nr_inner_atoms], sort_index[1][:nr_inner_atoms])

    # we are interested in length > 0
    sorted_nb_index = nb_index[sort_index]
    max_contacts = sorted_nb_index[0, 0]

    # The number of atoms with contacts c: c=max_contacts, c>=max_contacts-1, c>=max_contacts-2, ..., c>=1 
    iter_pos = jnp.searchsorted(-sorted_nb_index[:, 0], jnp.arange(-max_contacts+1, 1))

    chunks = [chunker(int(ip)) for ip in iter_pos]

    chunk_energies = []
    for i, chs in enumerate(chunks[::-1]):
        for ch in chs:
            start, realsize, padded_size = ch
            chunk_nb_index = sorted_nb_index[start:start+padded_size, 1]
            chunk_lig_struc = sort_index[0][start:start+padded_size]
            chunk_lig_atom = sort_index[1][start:start+padded_size]
            offset = i
            energy = nb_energy(chunk_nb_index, offset, chunk_lig_struc, chunk_lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff)
            #print(i, start, realsize, file=sys.stderr)
            chunk_energies.append((i, start, realsize, energy))
        
    nr_energies = iter_pos[-1]
    atom_energies = jnp.zeros(nr_energies)
    for i, start, realsize, energy in chunk_energies:
        atom_energies = atom_energies.at[start:start+realsize].add(energy[:realsize])
        
    energies = jnp.zeros(len(all_coors_lig))
    lig_struc = sort_index[0][:nr_energies]
    energies = energies.at[lig_struc].add(atom_energies)

    return energies

def main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, lig_atomtype_pos, ff, grid):
    coor_lig2 = jnp.concatenate((coor_lig, jnp.ones((len(coor_lig),1))),axis=1)
    all_coors_lig = jnp.einsum("jk,ikl->ijl", coor_lig2, mat) #diagonally broadcasted form of coor_lig.dot(mat)
    all_coors_lig = all_coors_lig[:, :, :3]

    pot_energies = potential_energy(all_coors_lig, lig_atomtype_pos, grid)
    nb_energies = neighbour_energy(coor_rec, rec_atomtypes, all_coors_lig, lig_atomtypes, ff, grid)
    energies = pot_energies + nb_energies
    return energies.sum(), energies

coor_rec = jnp.array(coor_rec)
coor_lig = jnp.array(coor_lig)
rec_atomtypes = jnp.array(rec_atomtypes)
lig_atomtypes = jnp.array(lig_atomtypes)
_, energies = main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, lig_atomtype_pos, ff, grid)
    

print("Calculate energies and gradients...")
vgrad_main = jax.value_and_grad(main, has_aux=True)
(_, energies), gradients = vgrad_main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, lig_atomtype_pos, ff, grid)

print("Energies:")  
print(energies[:10])
print()

print("Gradients (translational):")
print(-gradients[:10, 3, :3])  # sign is opposite of ATTRACT...

print("Gradients (torque):")
print(-gradients[:10, :3, :3])  # sign is opposite of ATTRACT...
