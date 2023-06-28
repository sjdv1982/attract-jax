ln -s ~/ssrna-attract-fraglib/clean-pdb/UGU
ln -s ~/ssrna-attract-fraglib/clean-pdb/UGU.list
ln -s ~/ssrna-attract-fraglib/clean-pdb/UGU-aa.list

$ATTRACTDIR/shm-grid 3sxl_dom2-r.grid 3sxl_dom2-r.gridheader
listr=UGU.list
ligandr=`head -1 UGU.list`
Nconf=`cat $listr|wc -l` # Nb of conformers in the library for the $motif sequence
params="$ATTRACTDIR/../attract.par 3sxl_dom2-r.pdb $ligandr --vmax 20 --fix-receptor --ens 2 $listr --grid 1 3sxl_dom2-r.gridheader"

python2 $ATTRACTDIR/../protocols/attract.py 2stack-rcut-sorted.dat $params --np 10 --chunks 10 --output 2stack-rcut-sorted-reminim.dat 

$ATTRACTDIR/shm-clean


python2 $ATTRACTDIR/lrmsd.py 2stack-rcut-sorted-reminim.dat \
  `head -1 UGU-aa.list` frag3-aa.pdb --ens 2 UGU-aa.list --allatoms > 2stack-rcut-sorted-reminim.lrmsd
