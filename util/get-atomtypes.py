import sys
import numpy as np
pdb = np.load(sys.argv[1])
assert 'x' in pdb.dtype.fields and 'y' in pdb.dtype.fields and 'z' in pdb.dtype.fields, pdb.dtype
assert pdb.ndim == 1
atomtypes = pdb["occupancy"].astype(int)
np.save(sys.argv[2], atomtypes)