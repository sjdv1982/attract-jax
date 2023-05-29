#from jax import config
#config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import cond

mat = np.load("mono-leu/minim-sorted.npy")
lig = np.load("mono-leu/mono-leuc-pdb.npy")


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
    
@jit
def main(all_coors_lig):
    nstruc = len(all_coors_lig)
    delta = all_coors_lig[:, :, None, :] - coor_rec[None, None, :, :]
    dsq = jnp.einsum("ijkl,ijkl->ijk", delta, delta)
    dsq = dsq.reshape(nstruc, -1)
    nb = lambda dsq: nonbon(dsq, rc, ac, emin, rmin2, ivor)
    #nb2 = lambda dsq: cond(dsq < 50*50, lambda: nb(dsq-(1.0/50.0)**2 ), lambda: 0.0)
    nb2 = nb
    nb2_vec = jnp.vectorize(nb2, signature='()->()')
    nb_energies = nb2_vec(dsq)
    energy = nb_energies.sum(axis=1)
    return energy

all_coors_lig = all_coors_lig[:2] ###
CHUNKSIZE = 100
for pos in range(0, len(all_coors_lig), CHUNKSIZE):
    arr = jnp.array(all_coors_lig[pos:pos+CHUNKSIZE])
    energy = main(arr)
    for e in energy:
        print(e)
