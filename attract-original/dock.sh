#!/bin/bash
set -u -e

# authors: Isaure Chauvot de Beauchene, Sjoerd de Vries, CNRS.

#trap "kill -- -$BASHPID; $ATTRACTDIR/shm-clean" ERR EXIT

#set -u -e

dat2npy(){
  dmotif=$1; dlistr=$2; dligandr=$3
  $ATTRACTTOOLS/top $dmotif-sorted-dr.dat 1000000 > $dmotif-sorted-dr-e6.dat
  $ATTRACTDIR/collect $dmotif-sorted-dr-e6.dat /dev/null $dligandr --ens 2 $dlistr > $dmotif-sorted-dr-e6.pdb
  ./pdb2npy.py $dmotif-sorted-dr-e6.pdb --outp pdb2$dmotif-e6.npy
  rm -f $dmotif-sorted-dr-e6.pdb
}

eval(){
  dmotif=$1; dlistr=$2; dligandr=$3
  echo '**************************************************************'
  echo 'calculate lRMSD'
  echo '**************************************************************'
  boundfrag=`awk -v m=$dmotif '$2==m{print $1}' boundfrag.list`
  x=$dmotif-sorted-dr-e6
  for i in $boundfrag; do
      refr=$refe/frag$i\r.pdb
      f=frag$i\r
      echo "compute $f"
      python $ATTRACTDIR/lrmsd.py $x.dat $dligandr $refr --ens 2 $dlistr --allatoms|awk '{print NR,$2}' > $f.rmsd
  done
}

dock(){
motif=$1 #3-nucl sequence
np=$2 # nb of CPU for parallelization

listr=${motif}.list
listaa=$motif-aa.list
ligandr=`head -n 1 $listr`
ligandaa=`head -n 1 $listaa`

Nstart=30000000         # Nb starting positions & orientations
Nconf=`cat $listr|wc -l` # Nb of conformers in the library for the $motif sequence
params="$ATTRACTDIR/../attract.par receptorr.pdb $ligandr --fix-receptor --ens 2 $listr"
scoreparams="$ATTRACTDIR/../attract.par receptorr.pdb $ligandr --score --fix-receptor --ens 2 $listr"
parals="--np $np --chunks 1000"
gridparams=" --grid 1 receptorgrid.gridheader"

$ATTRACTDIR/shm-clean
echo '**************************************************************'
echo 'calculate receptorgrid grid'
echo '**************************************************************'
$ATTRACTDIR/shm-grid receptorgrid.grid receptorgrid.gridheader

echo '**************************************************************'
echo 'Generate starting structures...'
echo '**************************************************************'
RAND=./
if [ ! -s $RAND/randsearch-$Nstart.dat ];then
    python $ATTRACTTOOLS/randsearch.py 2 $Nstart > $RAND/randsearch-$Nstart.dat
fi
if [ ! -s $RAND/randsearch-$Nstart-ens-$Nconf.dat ];then
    python $ATTRACTTOOLS/ensemblize.py $RAND/randsearch-$Nstart.dat $Nconf 2 random \
    > $RAND/randsearch-$Nstart-ens-$Nconf.dat
fi
start=$RAND/randsearch-$Nstart-ens-$Nconf.dat

echo '**************************************************************'
echo 'Docking'
echo '**************************************************************'
python $ATTRACTDIR/../protocols/attract.py $start $params $gridparams --vmax 500 $parals --output $motif.dat
python $ATTRACTDIR/../protocols/attract.py $motif.dat $scoreparams --rcut 50.0 $parals --output $motif.score

python $ATTRACTTOOLS/fill-energies.py $motif.dat $motif.score > tempdir/$motif-scored.dat
$ATTRACTTOOLS/top tempdir/$motif-scored.dat 10000 > tempdir/$motif-sample.dat
python $ATTRACTTOOLS/sort.py tempdir/$motif-sample.dat > tempdir/$motif-sample-sorted.dat
thresh=`grep Energy tempdir/$motif-sample-sorted.dat | awk 'NR == 1000{print $NF}'`
python $ATTRACTTOOLS/filter-energy.py tempdir/$motif-scored.dat $thresh > tempdir/$motif-score-filtered.dat
python $ATTRACTTOOLS/sort.py tempdir/$motif-score-filtered.dat > tempdir/$motif-sorted.dat
$ATTRACTDIR/fix_receptor tempdir/$motif-sorted.dat 2 --ens 0 $Nconf | python $ATTRACTTOOLS/fill.py /dev/stdin tempdir/$motif-sorted.dat > tempdir/$motif-sorted.dat-fixre
$ATTRACTDIR/deredundant tempdir/$motif-sorted.dat-fixre 2 --ens 0 $Nconf | python $ATTRACTTOOLS/fill-deredundant.py /dev/stdin tempdir/$motif-sorted.dat-fixre > $motif-sorted-dr.dat
rm -f $motif.dat $motif.score

dat2npy $motif $listr $ligandr &
eval $motif $listr $ligandr &
wait

}

refe=./

for m in `cat motif.list`; do
    dock $m 10
done
$ATTRACTDIR/shm-clean
rm -f *gridheader
wait