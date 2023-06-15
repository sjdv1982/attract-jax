# a version of test-nb-energrad where the plateau distance/gradients are subtracted,
#  as if potential grids where there.
# This is the best version to test subsequent tiling optimization

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
import time
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

grid = read_grid(open("mono-leu/mono-leuc.grid", "rb").read())
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
    return cond(dsq<plateaudissq, nonbon_dif, lambda *args: 0.0, d,ff, at1, at2)

nb_energy = jnp.vectorize(nb_energy_single, excluded=(1,4,5,6,7,8), signature="(),(),()->()")

@partial(jit, static_argnames=("padded_size",))
def nb_energy_dyn(padded_size, realsize, nb_index, offset, lig_struc, lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff):
    ind = jnp.arange(padded_size,dtype=np.int32)
    energies = nb_energy(nb_index, offset, lig_struc, lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff)
    return jnp.where(ind<realsize, energies, jnp.zeros_like(energies))

@partial(jit, static_argnames=("padded_size",))
def nb_energy_dynblock(padded_size, realsizes, nb_index, offset, lig_struc, lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff):    
    for n in range(len(realsizes)):
        e = nb_energy_dyn(padded_size, realsizes[n], nb_index, offset+n, lig_struc, lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff)
        assert len(e) == padded_size
        if n == 0:
            energies = e
        else:
            energies = energies + e
    return energies
2
MIN_CHUNK_FULL_WIDTH=2**14
MIN_CHUNK_FRAC = 0.6
MIN_CHUNK_WIDE_FRAC = 0.4
MIN_CHUNK=2**15
MAX_CHUNK=2**26

# TODO: in jnp, so it is on the GPU
# anyway, probably unnecessary, always a single chunk should do on a GPU 
# TODO: more clever tiling afterwards...
def chunker(s,offset=0,first=True, remaining_s=None):
    found = False
    if first:
        assert remaining_s is not None
        if s <= MIN_CHUNK_FULL_WIDTH:
            result = [(offset, remaining_s, MIN_CHUNK_FULL_WIDTH)]
            found = True
        elif s < MAX_CHUNK:
            c = MIN_CHUNK
            while s > c:
                c *= 2
            if s/c < MIN_CHUNK_WIDE_FRAC and MIN_CHUNK_WIDE_FRAC > 0.5:
                c //= 2
            slist = [s]
            for s_next in remaining_s[1:]:
                if s_next <= MIN_CHUNK_FULL_WIDTH or s_next/c < MIN_CHUNK_WIDE_FRAC:
                    break
                slist.append(s_next)
            result = [(offset, jnp.array(slist,int), c)]
            found = True
    if not found:
        if s == 0:
            return []
        if s <= MIN_CHUNK:
            result = [(offset, s, MIN_CHUNK)]
        elif s > MAX_CHUNK:
            result = [(offset, MAX_CHUNK, MAX_CHUNK)] + chunker(s-MAX_CHUNK, offset+MAX_CHUNK,first=False)
        else:
            c = MIN_CHUNK
            while s > c:
                c *= 2
            if s/c < MIN_CHUNK_FRAC:
                c //= 2
            result = [(offset, c, c)] + chunker(s-c, offset+c,first=False)    
    return result

def chunker_all(iter_pos):
    chunks = []
    iter_pos2 = [int(ip) for ip in iter_pos[::-1]]
    n = 0
    while n < len(iter_pos2):
        ip = iter_pos2[n]
        result = chunker(ip, remaining_s=jnp.array(iter_pos2[n:]))
        if len(result) and not isinstance(result[0][1], int):
            n += len(result[0][1])
        else:
            n += 1
        chunks.append(result)
    return chunks

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

@jit
def neighbour_energy_accum(all_coors_lig, lig_struc, atom_energies):
    energies = jnp.zeros(len(all_coors_lig))    
    energies = energies.at[lig_struc].add(atom_energies)
    return energies

def neighbour_energy(coor_rec, rec_atomtypes, all_coors_lig, lig_atomtypes, ff, grid):
    nb_index, sort_index, nr_inner_atoms = generate_nb_table(all_coors_lig, grid, tuple(grid.dim))
    sort_index = (sort_index[0][:nr_inner_atoms], sort_index[1][:nr_inner_atoms])

    # we are interested in length > 0
    sorted_nb_index = nb_index[sort_index]
    max_contacts = sorted_nb_index[0, 0]
    sorted_nb_index_p2 = sorted_nb_index[:, 1]
    sort_index_p1 = sort_index[0]
    sort_index_p2 = sort_index[1]
    
    sorted_nb_index_p2 = jnp.concatenate((sorted_nb_index_p2, jnp.zeros(MAX_CHUNK,sorted_nb_index.dtype)))
    sort_index_p1 = jnp.concatenate((sort_index_p1, jnp.zeros(MAX_CHUNK,sort_index[0].dtype)))
    sort_index_p2 = jnp.concatenate((sort_index_p2, jnp.zeros(MAX_CHUNK,sort_index[1].dtype)))

    # The number of atoms with contacts c: c=max_contacts, c>=max_contacts-1, c>=max_contacts-2, ..., c>=1 
    iter_pos = jnp.searchsorted(-sorted_nb_index[:, 0], jnp.arange(-max_contacts+1, 1))

    chunks = chunker_all(iter_pos)

    chunk_energies = []
    pos = 0
    for i, chs in enumerate(chunks):
        for ch in chs:
            start, realsizes, padded_size = ch
            assert start >= 0
            chunk_nb_index = sorted_nb_index_p2[start:start+padded_size]
            chunk_lig_struc = sort_index_p1[start:start+padded_size]
            chunk_lig_atom = sort_index_p2[start:start+padded_size]
            assert len(chunk_nb_index) == padded_size
            assert len(chunk_lig_struc) == padded_size
            assert len(chunk_lig_atom) == padded_size
            offset = pos
            if isinstance(realsizes, int):
                realsize = realsizes
                #energy = nb_energy(chunk_nb_index, offset, chunk_lig_struc, chunk_lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff)                                
                energy = nb_energy_dyn(padded_size, realsize, chunk_nb_index, offset, chunk_lig_struc, chunk_lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff)
                print(i, pos, start, realsize, energy.shape,  padded_size, file=sys.stderr)
                chunk_energies.append((i, start, realsize, energy))
                pos += 1
            else:
                energy = nb_energy_dynblock(padded_size, realsizes, chunk_nb_index, offset, chunk_lig_struc, chunk_lig_atom, all_coors_lig, coor_rec, rec_atomtypes, lig_atomtypes, ff)
                print(i, (pos, pos + len(realsizes)), jnp.max(realsizes), energy.shape, padded_size, file=sys.stderr)
                chunk_energies.append((i, start, jnp.max(realsizes), energy))
                pos += len(realsizes)

    nr_energies = iter_pos[-1]
    nr_energies_pad = jnp.exp2(jnp.ceil(jnp.log2(nr_energies))).astype(int)
    atom_energies = jnp.zeros(nr_energies_pad)
    for i, start, realsize, energy in chunk_energies:
        print(i, start, realsize, file=sys.stderr)
        atom_energies = atom_energies.at[start:start+realsize].add(energy[:realsize])

    if nr_energies_pad == nr_energies:
        lig_struc = sort_index[0][:nr_energies]
    else:
        lig_struc = jnp.concatenate((sort_index[0][:nr_energies], jnp.zeros(nr_energies_pad-nr_energies, dtype=sort_index[0].dtype)))
    
    energies = neighbour_energy_accum(all_coors_lig, lig_struc, atom_energies)
        
    return energies

def main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, lig_atomtype_pos, ff, grid):
    coor_lig2 = jnp.concatenate((coor_lig, jnp.ones((len(coor_lig),1))),axis=1)
    all_coors_lig = jnp.einsum("jk,ikl->ijl", coor_lig2, mat) #diagonally broadcasted form of coor_lig.dot(mat)
    all_coors_lig = all_coors_lig[:, :, :3]

    nb_energies = neighbour_energy(coor_rec, rec_atomtypes, all_coors_lig, lig_atomtypes, ff, grid)
    energies = nb_energies
    return energies.sum(), energies

coor_rec = jnp.array(coor_rec)
coor_lig = jnp.array(coor_lig)
rec_atomtypes = jnp.array(rec_atomtypes)
lig_atomtypes = jnp.array(lig_atomtypes)
    
print("Calculate energies and gradients...")
#_, energies = main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, lig_atomtype_pos, ff, grid)
vgrad_main = jax.value_and_grad(main, has_aux=True)
(_, energies), gradients = vgrad_main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, lig_atomtype_pos, ff, grid)

print("Energies:")  
print(energies[:10])
print()

print("Gradients (translational):")
print(-gradients[:10, 3, :3])  # sign is opposite of ATTRACT...

print("Gradients (torque):")
print(-gradients[:10, :3, :3])  # sign is opposite of ATTRACT...

total_energy = energies.sum()

main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, lig_atomtype_pos, ff, grid)

print("Timing (energies)...", file=sys.stderr)
t = time.time()

total_energy0, energies = main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, lig_atomtype_pos, ff, grid)

assert abs(total_energy0 - total_energy) < 0.00001

print("{:.2f} seconds".format(time.time() - t), file=sys.stderr)
print(file=sys.stderr)


(total_energy0, energies), gradients = vgrad_main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, lig_atomtype_pos, ff, grid)

assert abs(total_energy0 - total_energy) < 0.00001

print("Timing (gradients)...", file=sys.stderr)
t = time.time()

(total_energy0, energies), gradients = vgrad_main(mat, coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, lig_atomtype_pos, ff, grid)

assert abs(total_energy0 - total_energy) < 0.00001

print("{:.2f} seconds".format(time.time() - t), file=sys.stderr)
