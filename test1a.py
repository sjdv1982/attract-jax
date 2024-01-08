# version of test1.py where JIT is disabled for outer functions
# Loss of efficiency: about 10 %
# Allows pre-definition of neighbour chunks (test2.py)
#  allowing for more flexible energy evaluation
# For crocodile, where all poses are known in advance, better stick to jit to maximize speed


# test-grid-energrad-retile.py, taking command line parameters, and adding ensemble

# from jax import config
# config.update("jax_enable_x64", True)
CALC_GRADS = False
ITER = 50
ITER_GRAD = 50
TOP = 500000

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import cond
from read_grid import read_grid, pad_grid
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

args = parser.parse_args()

mat = jnp.load(args.poses)

rec = jnp.load(args.receptor)
assert rec.dtype in (np.float32, np.float64)
assert rec.ndim == 2 and rec.shape[-1] == 3, rec.shape

lig = jnp.load(args.ligand)
assert lig.dtype in (np.float32, np.float64)
assert lig.shape[-1] == 3, lig.shape
assert lig.ndim in (2,3), lig.shape
has_conformer = (lig.ndim == 3)
if has_conformer:
    assert args.conformers is not None
    conformers = jnp.load(args.conformers).astype(int)
    assert len(conformers) == len(mat)
    assert conformers.min() >= 1
    assert conformers.max() <= len(lig)
    conformers -= 1
else:
    assert args.conformers is None
    conformers = None
    lig = lig[None, :, :]

if TOP and TOP > 0:
    mat = mat[:TOP]
    conformers = conformers[:len(mat)]

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


grid = read_grid(open(args.grid, "rb").read(), read_gradients=True)
grid.neighbours[:] = rec_mapping[grid.neighbours]

print("Pad grid")
max_contacts = grid.neighbour_grid.reshape(-1,2)[:,0].max()
lp5 = list(range(5)[::-1])
lp = list(range(10)[::-1])
padding = [0] * 5 + lp5 + lp5 + lp5 + lp + lp
print(f"Maximum padding from {len(padding)+1} to {max_contacts}")
padding  += list(range(max_contacts-len(padding))[::-1])
grid2 = pad_grid(grid, padding=padding)
print("Padding increase:",  len(grid.neighbours), len(grid2.neighbours))
grid = grid2

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
def nb_energy_single(nb_index, offset, lig_struc, lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff):

    receptor_atom = grid.neighbours[nb_index + offset - 1]
    rec_c = coor_rec[receptor_atom]
    at1 = rec_atomtypes[receptor_atom]
    lig_c = all_coors_lig[lig_struc, lig_atom]
    at2 = lig_atomtypes[lig_atom]
    d = (lig_c - rec_c)
    dsq = (d*d).sum()
    return cond(((receptor_atom < 999999999) & (dsq<plateaudissq)), nonbon_dif, lambda *args: 0.0, d,ff, at1, at2)

nb_energy_vec = jnp.vectorize(nb_energy_single, excluded=(1,4,5,6,7,8), signature="(),(),()->()")

@partial(jit, static_argnames=("ncontacts",))
def nb_energy(nb_index, ncontacts, lig_struc, lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff):
    for n in range(ncontacts):
        e = nb_energy_vec(nb_index, n, lig_struc, lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff)
        assert len(e) == len(nb_index)
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


def neighbour_energy_accum(all_coors_lig, lig_struc, atom_energies):
    energies = jnp.zeros(len(all_coors_lig))    
    energies = energies.at[lig_struc].add(atom_energies)
    return energies

def neighbour_energy(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, ff, grid, chunks, grid_dim):
    coor_lig2 = jnp.concatenate((coor_lig, jnp.ones(coor_lig.shape[:2] + (1,))),axis=2)
    all_coors_lig = jnp.einsum("ijk,ikl->ijl", coor_lig2[conformers], mat) #diagonally broadcasted form of coor_lig.dot(mat)
    all_coors_lig = all_coors_lig[:, :, :3]

    nb_index, sort_index, nr_inner_atoms = generate_nb_table(all_coors_lig, grid, grid_dim)
    max_nr_inner_atoms = chunks[-1][-1]
    # assert nr_inner_atoms <= max_nr_inner_atoms
    sort_index = (sort_index[0][:max_nr_inner_atoms], sort_index[1][:max_nr_inner_atoms])

    # we are interested in length > 0
    sorted_nb_index = nb_index[sort_index]
    
    atom_energies = jnp.zeros(max_nr_inner_atoms)
    for ncontacts, start, length, end in chunks:
        if length == 0:
            continue
        sorted_nb_index_chunk = sorted_nb_index[start-1:end, 1]
        chunk_lig_struc = sort_index[0][start-1:end]
        chunk_lig_atom = sort_index[1][start-1:end]
        chunk_energies = nb_energy(sorted_nb_index_chunk, ncontacts, chunk_lig_struc, chunk_lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff)
        atom_energies = atom_energies.at[start-1:end].set(chunk_energies)

    lig_struc = sort_index[0][:max_nr_inner_atoms]
    energies = neighbour_energy_accum(all_coors_lig, lig_struc, atom_energies)
    return energies

def get_contact_lengths(mat, coor_lig,  conformers, grid):
    coor_lig2 = jnp.concatenate((coor_lig, jnp.ones(coor_lig.shape[:2] + (1,))),axis=2)
    all_coors_lig = jnp.einsum("ijk,ikl->ijl", coor_lig2[conformers], mat) #diagonally broadcasted form of coor_lig.dot(mat)
    all_coors_lig = all_coors_lig[:, :, :3]

    nb_index, sort_index, nr_inner_atoms = generate_nb_table(all_coors_lig, grid, tuple(grid.dim))
    sort_index = (sort_index[0][:nr_inner_atoms], sort_index[1][:nr_inner_atoms])

    # we are interested in length > 0
    sorted_nb_index = nb_index[sort_index]
    max_contacts = sorted_nb_index[0, 0]
    contact_lengths = jnp.searchsorted(-sorted_nb_index[:, 0], jnp.arange(-max_contacts+1, 1))
    return tuple(contact_lengths.tolist())
    
def main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff, grid, chunks, grid_dim):

    nb_energies = neighbour_energy(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, ff, grid, chunks, grid_dim)
    pot_energies = potential_energy(mat, coor_lig, conformers, lig_atomtype_pos, grid)
    energies = pot_energies + nb_energies
    return energies.sum(), energies

coor_rec = jnp.array(coor_rec)
coor_lig = jnp.array(coor_lig)
rec_atomtypes = jnp.array(rec_atomtypes)
lig_atomtypes = jnp.array(lig_atomtypes)


contact_lengths = get_contact_lengths(mat, coor_lig, conformers, grid)
data_max_contacts = len(contact_lengths)
print("Original contact lengths")
print(contact_lengths, data_max_contacts)
#print(list(zip(contact_lengths,range(data_max_contacts,0,-1))))
print("Contact chunks")
c = []
pos = 0
last = data_max_contacts
done = 0

#for n in range(data_max_contacts,-1,-1): ###   #disable padding requirement
for n in (0,1,2,3,4,5,10,15,20,30,40)[::-1]:
    if n >= data_max_contacts:
        continue
    done2 = contact_lengths[data_max_contacts-n-1]
    c.append((last, done+1, done2-done, done2))
    done = done2
    last = n    
chunks = tuple(c)
for cnr,c in enumerate(chunks):
    ncontacts = c[0]
    if cnr == len(chunks) - 1:
        minrealcontacts = 1
    else:
        minrealcontacts = chunks[cnr+1][0]+1
    for n in range(minrealcontacts, ncontacts):
        assert n+padding[n-1] >= ncontacts, (c, n, padding[n-1])


print(chunks)

grid_dim = tuple(grid.dim)


if CALC_GRADS:
    print("Calculate energies and gradients...")
    vgrad_main = jax.value_and_grad(main, has_aux=True)
    (_, energies), gradients = vgrad_main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff, grid, chunks, grid_dim)

    print("Energies:")  
    print(energies[:10])
    print()

    print("Gradients (translational):")
    print(-gradients[:10, 3, :3])  # sign is opposite of ATTRACT...

    print("Gradients (torque):")
    print(-gradients[:10, :3, :3])  # sign is opposite of ATTRACT...
else:
    print("Calculate energies...")
    _, energies = main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff, grid, chunks, grid_dim)

    print("Energies:")  
    print(energies[:10])
    print()

total_energy = energies.sum()

main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff, grid, chunks, grid_dim)

import gc
gc.collect()

print(f"Timing (energies, x10)...", file=sys.stderr)
t = time.time()

for n in range(10):
    total_energy0, energies = main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff, grid, chunks, grid_dim)
    print(n+1, time.time() - t)

    #assert abs(total_energy0 - total_energy) < 0.1, (total_energy0, total_energy)

print("{:.3f} seconds".format((time.time() - t)/10), file=sys.stderr)
print(file=sys.stderr)

print(f"Timing (energies, x{ITER})...", file=sys.stderr)
t = time.time()

for n in range(ITER):
    total_energy0, energies = main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff, grid, chunks, grid_dim)
    print(n+1, time.time() - t)

    #assert abs(total_energy0 - total_energy) < 0.1, (total_energy0, total_energy)

print("{:.3f} seconds".format((time.time() - t)/ITER), file=sys.stderr)
print(file=sys.stderr)

print(f"Timing (energies, scrambled, x{ITER})...", file=sys.stderr)

scrambles = []
scramble = np.arange(len(mat))
np.random.seed(0)

for n in range(ITER):
    np.random.shuffle(scramble) 
    scrambles.append(jnp.array(scramble))

t = time.time()
for n in range(ITER):
    mat_scramble, conformers_scramble = mat[scrambles[n]], conformers[scrambles[n]]
overhead = time.time() - t

import gc
gc.collect()

t = time.time()
for n in range(ITER):
    mat_scramble, conformers_scramble = mat[scrambles[n]], conformers[scrambles[n]]
    total_energy0, energies = main(mat_scramble, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers_scramble, lig_atomtype_pos, ff, grid, chunks, grid_dim)

    #assert abs(total_energy0 - total_energy) < 0.1, (total_energy0, total_energy)

print("{:.3f} seconds".format((time.time() - t - overhead)/ITER), file=sys.stderr)
print(file=sys.stderr)

if CALC_GRADS:
    (total_energy0, energies), gradients = vgrad_main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff,grid, chunks, grid_dim)

    #assert abs(total_energy0 - total_energy) < 0.1, (total_energy0, total_energy)

    print(f"Timing (energy + gradients, x10)...", file=sys.stderr)
    t = time.time()

    for n in range(10):

        (total_energy0, energies), gradients = vgrad_main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff, grid, chunks, grid_dim)
        print(n+1, time.time() - t)

        #assert abs(total_energy0 - total_energy) < 0.1, (total_energy0, total_energy)

    print("{:.3f} seconds".format((time.time() - t)/10), file=sys.stderr)


    print(f"Timing (energy + gradients, x{ITER_GRAD})...", file=sys.stderr)
    t = time.time()

    for n in range(ITER_GRAD):

        (total_energy0, energies), gradients = vgrad_main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, conformers, lig_atomtype_pos, ff, grid, chunks, grid_dim)
        print(n+1, time.time() - t)

        #assert abs(total_energy0 - total_energy) < 0.1, (total_energy0, total_energy)

    print("{:.3f} seconds".format((time.time() - t)/ITER_GRAD), file=sys.stderr)
