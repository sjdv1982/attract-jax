import numpy as np

mat = np.load("mono-leu/minim-sorted-center.npy")
lig = np.load("mono-leu/mono-leu-centered-pdb.npy")


coor_lig = np.stack([lig["x"],lig["y"], lig["z"], np.ones(len(lig))], axis=1)
coor_rec = coor_lig[:, :3]

all_coors_lig = np.einsum("jk,ikl->ijl", coor_lig, mat) #diagonally broadcasted form of coor_lig.dot(mat)
all_coors_lig = all_coors_lig[:, :, :3]

struc = all_coors_lig[0]
delta = struc[:, None, :] - coor_rec[None, :, :]
dis = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))

dis2 = dis.flatten()
last = None
for d in np.arange(0.1, 30.1, 0.1):
    err = dis2 - d
    pos = np.argmin(err**2)
    dd = dis2[pos]
    if dd != last:
        print(dd)
        last = dd
