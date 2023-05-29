import numpy as np

mat = np.load("mono-leu/minim-sorted.npy")
lig = np.load("mono-leu/mono-leuc-pdb.npy")
refe = np.load("mono-leu/minim-best-ligand-pdb.npy")
assert lig.shape == refe.shape
mask = (np.isin(lig["name"] , (b'CA', b'C', b'O', b'N')))
lig = lig[mask]
refe = refe[mask]
coor_lig = np.stack([lig["x"],lig["y"], lig["z"], np.ones(len(lig))], axis=1)
coor_refe = np.stack([refe["x"],refe["y"], refe["z"], np.ones(len(lig))], axis=1)

all_coors_lig = np.einsum("jk,ikl->ijl", coor_lig, mat) #diagonally broadcasted form of coor_lig.dot(mat)
all_coors_lig = all_coors_lig[:, :, :3]
all_d = all_coors_lig - coor_refe[:, :3]
all_sd = np.einsum("ijk,ijk->i", all_d, all_d)  # sum over last 2 axes
all_rmsd = np.sqrt(all_sd / len(coor_refe))
for r in all_rmsd:
    print("{:.3f}".format(r))