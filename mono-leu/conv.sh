
rm -f *.npy
d=~/nefertiti/nefertiti/functions
python3 $d/read_dat.py minim-sorted.dat minim-sorted.npy --energies minim-sorted-energies.npy
python3 $d/parse_pdb.py mono-leuc.pdb mono-leuc-pdb.npy
python3 $d/parse_pdb.py minim-best-ligand.pdb minim-best-ligand-pdb.npy