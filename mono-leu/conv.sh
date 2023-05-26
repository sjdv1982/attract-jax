d=~/nefertiti/nefertiti/functions
python3 $d/read_dat.py minim-sorted-center.dat minim-sorted-center.npy --energies minim-sorted-center-energies.npy
python3 $d/parse_pdb.py mono-leu-centered.pdb mono-leu-centered-pdb.npy
python3 $d/parse_pdb.py minim-best-ligand.pdb minim-best-ligand-pdb.npy