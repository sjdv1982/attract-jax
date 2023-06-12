# attract.par taken from the ATTRACT project (see copyright)

import numpy as np
header = [int(ll) for ll in open("attract.par").readline().split()]
potshape = header[0]
assert potshape == 8 # do sth else for all-atom

ntypes = header[1]
par = np.loadtxt("attract.par", skiprows=1)
par = par.reshape(3, ntypes, ntypes)
rbc = par[0]
abc = par[1]
ipon = par[2].astype(int)
assert np.unique(ipon).tolist() == [-1, 1]

rc =abc*rbc**potshape
ac =abc*rbc**6

# emin=-27.0*ac**4/(256.0*rc**3) # at runtime
# rmin2=4.0*rc/(3.0*ac) # at runtime
ivor = ipon
np.savez("../attract-par.npz", rc=rc, ac=ac, ivor=ivor)
