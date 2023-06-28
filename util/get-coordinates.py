import sys
import numpy as np
pdb = np.load(sys.argv[1])
assert 'x' in pdb.dtype.fields and 'y' in pdb.dtype.fields and 'z' in pdb.dtype.fields, pdb.dtype
assert pdb.ndim == 1
coor = np.stack((pdb["x"], pdb["y"], pdb["z"])).T
np.save(sys.argv[2], coor)