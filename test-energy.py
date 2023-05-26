import numpy as np

mat = np.load("mono-leu/minim-sorted-center.npy")
lig = np.load("mono-leu/mono-leu-centered-pdb.npy")


coor_lig = np.stack([lig["x"],lig["y"], lig["z"], np.ones(len(lig))], axis=1)
coor_rec = coor_lig[:, :3]

all_coors_lig = np.einsum("jk,ikl->ijl", coor_lig, mat) #diagonally broadcasted form of coor_lig.dot(mat)
all_coors_lig = all_coors_lig[:, :, :3]

for struc in all_coors_lig:
    delta = struc[:, None, :] - coor_rec[None, :, :]
    dis = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    break #
