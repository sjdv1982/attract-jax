rm -f *.dat *.score

python2 $ATTRACTTOOLS/randsearch.py 2 10000 0 --fix-receptor --radius 40 > start.dat
$ATTRACTDIR/make-grid-omp mono-leuc.pdb $ATTRACTDIR/../attract.par 5 7 mono-leuc.grid
$ATTRACTDIR/make-grid-omp mono-leuc.pdb $ATTRACTDIR/../attract.par 5 7 mono-leuc.nbgrid --calc-potentials=0
$ATTRACTDIR/attract start.dat $ATTRACTDIR/../attract.par  mono-leuc.pdb mono-leuc.pdb --rcut 9999999 --score | grep Energy| awk '{print $NF}' > start.score
$ATTRACTDIR/attract start.dat $ATTRACTDIR/../attract.par  mono-leuc.pdb mono-leuc.pdb --grid 1 mono-leuc.grid --fix-receptor > minim.dat
$ATTRACTDIR/attract minim.dat $ATTRACTDIR/../attract.par  mono-leuc.pdb mono-leuc.pdb --rcut 9999999 --score | grep Energy| awk '{print $NF}' > minim.score

python2 $ATTRACTTOOLS/sort.py minim.dat > minim-sorted.dat
$ATTRACTDIR/attract minim-sorted.dat $ATTRACTDIR/../attract.par  mono-leuc.pdb mono-leuc.pdb --rcut 9999999 --score | grep Energy| awk '{print $NF}' > minim-sorted.score

$ATTRACTTOOLS/top minim-sorted.dat 1 > minim-best.dat
$ATTRACTDIR/collect minim-best.dat /dev/null mono-leuc.pdb | grep ATOM > minim-best-ligand.pdb
python2 $ATTRACTDIR/lrmsd.py minim-sorted.dat mono-leuc.pdb minim-best-ligand.pdb > minim-best.lrmsd


for grid in grid nbgrid; do
  for dat in start minim minim-sorted; do
    $ATTRACTDIR/attract $dat.dat $ATTRACTDIR/../attract.par  mono-leuc.pdb mono-leuc.pdb --rcut 9999999 --grid 1 mono-leuc.$grid --fix-receptor --score > $dat-$grid.score &
  done
done
wait


