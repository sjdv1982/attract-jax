ln -s ~/ssrna-attract-fraglib/clean-pdb/UGU
ln -s ~/ssrna-attract-fraglib/clean-pdb/UGU.list

awk '{print substr($0,58,2)}' frag3-r.pdb | sort -nu > receptorgrid.alphabet
$ATTRACTDIR/make-grid-omp 3sxl_dom2-r.pdb $ATTRACTDIR/../attract.par 5 7 3sxl_dom2-r.grid --alphabet receptorgrid.alphabet
$ATTRACTDIR/shm-grid 3sxl_dom2-r.grid 3sxl_dom2-r.gridheader
listr=UGU.list
ligandr=`head -1 UGU.list`
Nconf=`cat $listr|wc -l` # Nb of conformers in the library for the $motif sequence
scoreparams="$ATTRACTDIR/../attract.par 3sxl_dom2-r.pdb $ligandr --score --fix-receptor --ens 2 $listr"

for i in `seq 10`; do
  $ATTRACTDIR/attract 2stack-chunk-$i $scoreparams --rcut 50.0 > 2stack-chunk-$i-rcut.score &
  $ATTRACTDIR/attract 2stack-chunk-$i $scoreparams --grid 1  3sxl_dom2-r.gridheader > 2stack-chunk-$i-grid.score &
done
wait

$ATTRACTDIR/shm-clean

cat /dev/null > 2stack-grid.score
cat /dev/null > 2stack-rcut.score
for i in `seq 10`; do
  cat 2stack-chunk-$i-rcut.score >> 2stack-rcut.score
  cat 2stack-chunk-$i-grid.score >> 2stack-grid.score
done

python2 $ATTRACTTOOLS/fill-energies.py 2stack.dat 2stack-rcut.score > /tmp/z
python2 $ATTRACTTOOLS/filter-energy.py /tmp/z 200 > /tmp/zz
python2 $ATTRACTTOOLS/sort.py /tmp/zz > 2stack-rcut-sorted.dat

rm -f /tmp/z /tmp/zz

ln -s ~/ssrna-attract-fraglib/clean-pdb/UGU
ln -s ~/ssrna-attract-fraglib/clean-pdb/UGU-aa.list
python2 $ATTRACTDIR/lrmsd.py 2stack-rcut-sorted.dat \
  `head -1 UGU-aa.list` frag3-aa.pdb --ens 2 UGU-aa.list --allatoms > 2stack-rcut-sorted.lrmsd
