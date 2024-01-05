# version of test4.py with gravity (hardcoded) and optimized learning rate/steps
# writes out poses and energies
# not cut 
# and no JIT warmup

# ***Performs about as well (slightly worse) as current ATTRACT***
# Minimizes 3 million structures/hour on Newton

# from jax import config
# config.update("jax_enable_x64", True)
BATCH = 200000   # the number of structures to score in at a time
NB_CHUNK_SIZE = 100000  # will be quadrupled for small ncontacts

SCHEMA = [
    # learning rate,  epochs,  gravity
    [5, 10, True],
    [10, 5, True],
    [10, 1, True],  
    [50, 0.5, True], 
    [100, 0.1, False],
    [100, 0.05, False],
    [100, 0.02, False],
]

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import cond
from jax.scipy.spatial.transform import Rotation
import optax
from read_grid2 import read_grid
from functools import partial
from collections import namedtuple
import time
import sys
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("poses", help="docking poses as 4x4 matrices")
parser.add_argument("receptor", help="receptor coordinates")
parser.add_argument("ligand", help="ligand coordinates (can be conformer library)")

parser.add_argument("--atrec", help="receptor atom types",required=True)
parser.add_argument("--atlig", help="ligand atom types",required=True)
parser.add_argument("--grid", help="ATTRACT docking grid",required=True)
parser.add_argument("--conformers", help="ligand conformer indices")
parser.add_argument("--out_poses", help="output poses",required=True)
parser.add_argument("--out_ene", help="output energies",required=True)

args = parser.parse_args()

rec = jnp.load(args.receptor)
assert rec.dtype in (np.float32, np.float64)
assert rec.ndim == 2 and rec.shape[-1] == 3, rec.shape

lig = jnp.load(args.ligand)
assert lig.dtype in (np.float32, np.float64)
assert lig.shape[-1] == 3, lig.shape
assert lig.ndim in (2,3), lig.shape
has_conformer = (lig.ndim == 3)

mat = np.load(args.poses)
scramble = np.arange(len(mat))
np.random.seed(0)
np.random.shuffle(scramble)
mat = mat[scramble]

if has_conformer:
    ###
    if args.conformers is None: 
        conformers = np.ones(len(mat)).astype(int)
    else:
        conformers = np.load(args.conformers).astype(int)  
    ###
    ###assert args.conformers is not None
    ###conformers = np.load(args.conformers).astype(int)
    assert len(conformers) == len(mat)
    assert conformers.min() >= 1
    assert conformers.max() <= len(lig)
    conformers -= 1
    conformers = conformers[scramble]
else:
    assert args.conformers is None
    conformers = None    
    lig = lig[None, :, :]

rec_atomtypes00 = jnp.load(args.atrec).astype(np.uint8)
assert rec_atomtypes00.ndim == 1 and len(rec_atomtypes00) == len(rec)
mask = rec_atomtypes00 != 99
rec = rec[mask]
rec_atomtypes0 = rec_atomtypes00[mask]
rec_mapping = np.cumsum(mask)-1
coor_rec = rec

lig_atomtypes00 = jnp.load(args.atlig).astype(np.uint8)
assert lig_atomtypes00.ndim == 1 and len(lig_atomtypes00) == lig.shape[-2], (len(lig_atomtypes00), lig.shape)
mask = lig_atomtypes00 != 99
lig = lig[:, mask]
lig_atomtypes0 = lig_atomtypes00[mask]
coor_lig = lig

lig_alphabet, lig_atomtypes = np.unique(lig_atomtypes0, return_inverse=True)
rec_alphabet, rec_atomtypes = np.unique(rec_atomtypes0, return_inverse=True)

##############################################

grid = read_grid(open(args.grid, "rb").read(), read_gradients=True)
nb_grid_flat0 = grid.neighbour_grid.reshape(-1)
mask = (nb_grid_flat0 < 2**16-1)
nb_grid_flat0[mask] = rec_mapping[nb_grid_flat0[mask]]

nb_chunk_thresholds = (0,1,2,3,4,5,10,15,20)
for n in range(nb_chunk_thresholds[-1]+10, grid.max_nr_neighbours, 10):
    nb_chunk_thresholds += (n,)
nb_chunk_thresholds += (grid.max_nr_neighbours,)
print("Neighbour list chunk thresholds:", nb_chunk_thresholds)

##############################################

potshape = 8

currdir = os.path.dirname(os.path.realpath(__file__))
par = np.load(f"{currdir}/attract-par.npz")
rc, ac, ivor = par["rc"], par["ac"], par["ivor"]

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

##############################################

alpos = (np.where(grid.alphabet)[0]+1).tolist()
lig_alphabet_pos = np.array([alpos.index(a) for a in lig_alphabet])
lig_atomtype_pos = jnp.array(lig_alphabet_pos[lig_atomtypes])

d = {}
for field in grid._fields:
    value = getattr(grid, field)
    if isinstance(value, np.ndarray) and value.shape != (3,):
        value = jnp.array(value, dtype=value.dtype)
    d[field] = value
d["neighbour_grid_ravel"] = d["neighbour_grid"].reshape(-1, grid.neighbour_grid.shape[-1])
grid = type(grid)(**d)


##############################################
def _run_argsort(args) -> jnp.ndarray:
    #jax.debug.print("Run argsort using Numpy")
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
    grad_main=2*grad_nonbon(dsq,ff,at1,at2)*d
    ratio = jnp.sqrt(plateaudissq/dsq)
    grad_plateau = 2*plateau_gradients[at1, at2]*d*ratio
    gradient = ((grad_main - grad_plateau) * d_dot).sum()
    return ans, gradient

@jit
def nb_energy_single(ind_innergrid, offset, lig_struc, lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff):
    receptor_atom = grid.neighbour_grid_ravel[ind_innergrid, offset]
    rec_c = coor_rec[receptor_atom]
    at1 = rec_atomtypes[receptor_atom]
    lig_c = all_coors_lig[lig_struc, lig_atom]
    at2 = lig_atomtypes[lig_atom]
    d = (lig_c - rec_c)
    dsq = (d*d).sum()
    return cond(((receptor_atom < 2**16-1) & (dsq<plateaudissq)), nonbon_dif, lambda *args: 0.0, d,ff, at1, at2)

nb_energy_vec = jnp.vectorize(nb_energy_single, excluded=(1,4,5,6,7,8), signature="(),(),()->()")

@partial(jit, static_argnames=("ncontacts",))
def nb_energy(ind_innergrid, ncontacts, lig_struc, lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff):
    for n in range(ncontacts):
        e = nb_energy_vec(ind_innergrid, n, lig_struc, lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff)
        assert len(e) == len(ind_innergrid)
        if n == 0:
            energies = e
        else:
            energies = energies + e
    return energies

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
def potential_energy(mat, coor_lig, conformers, lig_atomtype_pos, grid):
    coor_lig2 = jnp.concatenate((coor_lig, jnp.ones(coor_lig.shape[:2] + (1,))),axis=2)
    all_coors_lig = jnp.einsum("ijk,ikl->ijl", coor_lig2[conformers], mat) #diagonally broadcasted form of coor_lig.dot(mat)
    all_coors_lig = all_coors_lig[:, :, :3]
    
    atom_energies = potential_energies(all_coors_lig, lig_atomtype_pos, grid)
    return atom_energies.sum(axis=1)

@partial(jit, static_argnames=("grid_dim",))
def generate_ind_innergrid(all_coors_lig, grid, grid_dim):
    vox_innergrid =  (all_coors_lig - grid.origin) / grid.gridspacing
    x, y, z = vox_innergrid[:, :, 0], vox_innergrid[:, :, 1], vox_innergrid[:, :, 2]
    in_innergrid = ((x >= 0) & (x < grid.dim[0]-1) & (y >= 0) & (y < grid.dim[1]-1) & (z >= 0) & (z < grid.dim[2]-1))
    out_of_bounds=max(grid_dim)+2
    pos_innergrid = jnp.where(in_innergrid[:, :, None], jnp.floor(vox_innergrid+0.5).astype(np.int32), out_of_bounds)
    ind_innergrid = jnp.ravel_multi_index((pos_innergrid[:, :, 0], pos_innergrid[:, :, 1],pos_innergrid[:, :, 2]),dims=grid_dim, mode="clip")
    ind_innergrid = ind_innergrid + out_of_bounds**3 * (1 - in_innergrid.astype(np.uint8))

    #nb_index = jnp.take(grid.nr_neighbours, ind_innergrid, fill_value=0)
    nb_index = jnp.take(grid.nr_neighbours, ind_innergrid)
    nb_index = jnp.where(nb_index != -32768, nb_index, 0)
    max_contacts = nb_index.max()
    return ind_innergrid, nb_index, max_contacts 

@jit
def generate_sort_index(nb_index):
    k=-nb_index
    sort_index = jnp.unravel_index(run_argsort(k,axis=None), nb_index.shape)
    nr_inner_atoms = jnp.searchsorted(k[sort_index], 0, side="left")
    return sort_index, nr_inner_atoms

@jit
def neighbour_energy_accum(all_coors_lig, lig_struc, atom_energies):
    energies = jnp.zeros(len(all_coors_lig))    
    energies = energies.at[lig_struc].add(atom_energies)
    return energies

def neighbour_energy(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, ff, grid, nb_chunk_thresholds, nb_chunk_size, grid_dim):
    coor_lig2 = jnp.concatenate((coor_lig, jnp.ones(coor_lig.shape[:2] + (1,))),axis=2)
    all_coors_lig = jnp.einsum("ijk,ikl->ijl", coor_lig2[conformers], mat) #diagonally broadcasted form of coor_lig.dot(mat)
    all_coors_lig = all_coors_lig[:, :, :3]

    ind_innergrid, nb_index, max_contacts  = generate_ind_innergrid(all_coors_lig, grid, grid_dim)    
    if max_contacts == 0:
        return jnp.zeros(len(mat))

    sort_index, nr_inner_atoms = generate_sort_index(nb_index)
    sorted_ind_innergrid = ind_innergrid[sort_index[0], sort_index[1]]
    sorted_nb_index = nb_index[sort_index[0], sort_index[1]]

    nr_inner_atoms = int(nr_inner_atoms)
    
    tot_atoms = len(sort_index[0])

    max_ncontacts_map = {}
    for n in range(0, len(nb_chunk_thresholds)-1):
        p1 = nb_chunk_thresholds[n]
        p2 = nb_chunk_thresholds[n+1]
        for p in range(p1+1, p2+1):
            max_ncontacts_map[p] = p2

    atom_energies = jnp.zeros(tot_atoms)  #only :nr_inner_atoms will be used    

    n = 0
    while n < nr_inner_atoms:
        start = n
        max_ncontacts0 = int(sorted_nb_index[start])
        max_ncontacts = max_ncontacts_map[max_ncontacts0]

        n += nb_chunk_size

        if max_ncontacts0 > 0:
            max_ncontacts0_next = int(sorted_nb_index[n-1])
            if max_ncontacts - max_ncontacts0_next < 5:
                #print("QUAD", start, max_ncontacts0_next, max_ncontacts)
                n +=  3 * nb_chunk_size
        end = n
    
        sorted_ind_innergrid_chunk = sorted_ind_innergrid[start:end]
        chunk_lig_struc = sort_index[0][start:end]
        chunk_lig_atom = sort_index[1][start:end]            

        chunk_energies = nb_energy(sorted_ind_innergrid_chunk, max_ncontacts, chunk_lig_struc, chunk_lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff)
        
        atom_energies = atom_energies.at[start:end].set(chunk_energies)

    lig_struc = sort_index[0]
    energies = neighbour_energy_accum(all_coors_lig, lig_struc, atom_energies)
    return energies

def main0(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff, grid, nb_chunk_thresholds, grid_dim):

    nb_energies = neighbour_energy(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, ff, grid, nb_chunk_thresholds, NB_CHUNK_SIZE, grid_dim)
    pot_energies = potential_energy(mat, coor_lig, conformers, lig_atomtype_pos, grid)
    energies = pot_energies + nb_energies
    return energies.sum(), energies

def main(rotations: Rotation, translations, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff, grid, nb_chunk_thresholds, grid_dim, *, gravity):
    assert len(rotations) == len(translations) == len(conformers)
    mat = jnp.ones((len(rotations), 4, 4))
    mat = mat.at[:, :3, :3].set(rotations.as_matrix())
    mat = mat.at[:, 3, :3].set(translations)
    tot_energy, energies = main0(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff, grid, nb_chunk_thresholds, grid_dim)
    if not gravity:
        return tot_energy, energies
    gravity_energy = jnp.linalg.norm(translations, axis=1)**2 * 0.1
    return tot_energy + gravity_energy.sum(), energies + gravity_energy

coor_rec = jnp.array(coor_rec)
coor_lig = jnp.array(coor_lig)
rec_atomtypes = jnp.array(rec_atomtypes)
lig_atomtypes = jnp.array(lig_atomtypes)

grid_dim = tuple(grid.dim)

vgrad_main = jax.value_and_grad(main, argnums=(0,1), has_aux=True)

t = time.time()

assert len(mat) == len(conformers)
minimized_energies = np.zeros(len(mat))
minimized_poses = np.ones_like(mat)
for offset in range(0, len(mat), BATCH):
    print(time.time() - t, offset)
    start, end = offset, offset + BATCH
    if end > len(mat):
        end = len(mat)
        start = end - BATCH
        if start < 0:
            start = 0
    batch = end - start
    mat_batch = mat[start:end]
    rotations_batch = Rotation.from_matrix(mat_batch[:, :3, :3])
    translations_batch = mat_batch[:, 3, :3]
    conformers_batch = conformers[start:end]
    ### optimizer
    for epochs, learning_rate, gravity in SCHEMA:
        optimizer = optax.adabelief(learning_rate)    
        opt_state = optimizer.init((rotations_batch, translations_batch))
        for epoch in range(epochs):
            (total_energy_batch, energies_batch), gradients_batch = vgrad_main(rotations_batch, translations_batch, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers_batch, lig_atomtype_pos, ff,grid, nb_chunk_thresholds, grid_dim, gravity=gravity)
            print(f"Batch {start}-{end} epoch {epoch} median energy {np.sort(energies_batch)[len(energies_batch)//2]}")
            updates, opt_state = optimizer.update(gradients_batch, opt_state)
            rotations_batch, translations_batch = optax.apply_updates((rotations_batch, translations_batch), updates)
            rotations_batch = Rotation.from_quat(rotations_batch.as_quat())
    
    # Final scoring
    energies_minimized_batch = energies_batch ###
    ###total_minimized_energy_batch, energies_minimized_batch = main(rotations_batch, translations_batch, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers_batch, lig_atomtype_pos, ff,grid, nb_chunk_thresholds, grid_dim, gravity=False)
    ###print(f"Batch {start}-{end} Final median energy{np.sort(energies_batch)[len(energies_batch)//2]}")

    minimized_energies[start:end] = energies_minimized_batch
    minimized_poses[start:end, :3, :3] = rotations_batch.as_matrix()
    minimized_poses[start:end, 3, :3] = translations_batch
    

print("{:.3f} seconds".format((time.time() - t)), file=sys.stderr)
print(file=sys.stderr)

final_poses = np.zeros_like(mat)
final_poses[scramble] = minimized_poses
if args.out_poses:
    np.save(args.out_poses, final_poses)

final_energies = np.zeros(len(mat))
final_energies[scramble] = minimized_energies
if args.out_ene:
    np.save(args.out_ene, final_energies)
