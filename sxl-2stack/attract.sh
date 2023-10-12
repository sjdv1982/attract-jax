# next command requires attract-shell
python2 $ATTRACTDIR/../allatom/aareduce.py 3sxl_dom2.pdb 3sxl_dom2-aa.pdb --pdb2pqr --heavy
python2 $ATTRACTTOOLS/reduce.py 3sxl_dom2-aa.pdb 3sxl_dom2-r.pdb

python2 $ATTRACTDIR/../allatom/aareduce.py frag3.pdb frag3-aa.pdb --rna --heavy --nalib
python2 $ATTRACTTOOLS/reduce.py frag3-aa.pdb frag3-r.pdb

python2 $ATTRACTTOOLS/fit-multi.py frag3-aa.pdb UGU-aa.list --rmsd --allatoms > UGU-aa.rmsd

python2 $ATTRACTTOOLS/split.py 2stack.dat 2stack-chunk 10

ln -s ~/ssrna-attract-fraglib/clean-pdb/UGU
ln -s ~/ssrna-attract-fraglib/clean-pdb/UGU-aa.list
for i in `seq 10`; do
  python2 $ATTRACTDIR/lrmsd.py 2stack-chunk-$i \
    `head -1 UGU-aa.list` frag3-aa.pdb --ens 2 UGU-aa.list --allatoms > 2stack-chunk-$i.lrmsd &
done
wait
cat /dev/null > 2stack.lrmsd
for i in `seq 10`; do
  cat 2stack-chunk-$i.lrmsd >> 2stack.lrmsd
done