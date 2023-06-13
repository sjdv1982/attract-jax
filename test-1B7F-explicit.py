# from jax import config
# config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import cond
import sys

frag = int(sys.argv[1])
potshape = 8

par = np.load("attract-par.npz")
rc, ac, ivor = par["rc"], par["ac"], par["ivor"]

rec = np.load("1B7F-bound/receptorr-pdb.npy")
rec_atomtypes00 = rec["occupancy"].astype(np.uint8)
mask = rec_atomtypes00 != 99
rec = rec[mask]
rec_atomtypes0 = rec_atomtypes00[mask]

lig = np.load(f"1B7F-bound/frag{frag}r-pdb.npy")
lig_atomtypes0 = lig["occupancy"].astype(np.uint8)

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


@jit
def nonbon(dsq, ff, at1, at2):
    rr2 = 1 / dsq

    alen = ff["ac"][at1, at2]
    rlen = ff["rc"][at1, at2]
    rr23 = rr2 * rr2 * rr2
    rep = rlen * rr2
    vlj = (rep - alen) * rr23
    attraction = ff["ivor"][at1, at2]

    energy = cond(
        dsq < ff["rmin2"][at1, at2],
        lambda: vlj + (attraction - 1) * ff["emin"][at1, at2],
        lambda: attraction * vlj,
    )
    return energy

nonbon2 = jnp.vectorize(nonbon, excluded=(1,), signature="(),(),()->()")

@jit
def main(coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, ff):
    delta = coor_lig[:, None, :] - coor_rec[None, :, :]
    dsq = jnp.einsum("ijk,ijk->ij", delta, delta)
    i_lig_atomtypes, i_rec_atomtypes = jnp.indices((len(lig_atomtypes), len(rec_atomtypes) ))
    at_lig = lig_atomtypes[i_lig_atomtypes]
    at_rec = rec_atomtypes[i_rec_atomtypes]
    nb_energies = nonbon2(dsq, ff, at_rec, at_lig)
    energy = nb_energies.sum()
    return energy

coor_rec = jnp.array(coor_rec)
coor_lig = jnp.array(coor_lig)
rec_atomtypes = jnp.array(rec_atomtypes)
lig_atomtypes = jnp.array(lig_atomtypes)
energy = main(coor_rec, rec_atomtypes, coor_lig, lig_atomtypes, ff)
print(energy)
