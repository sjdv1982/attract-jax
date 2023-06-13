
rm -f *.npy
d=~/nefertiti/nefertiti/functions
python3 $d/parse_pdb.py receptorr.pdb receptorr-pdb.npy
for frag in 5 6 7 8 9 10; do
python3 $d/parse_pdb.py frag${frag}r.pdb frag${frag}r-pdb.npy
done