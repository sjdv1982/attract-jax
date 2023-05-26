
python2 $ATTRACTTOOLS/randsearch.py 2 10000 0 --fix-receptor --radius 40 > start.dat
$ATTRACTDIR/make-grid-omp mono-leu.pdb $ATTRACTDIR/../attract.par 5 7 mono-leu.grid
$ATTRACTDIR/make-grid-omp mono-leu.pdb $ATTRACTDIR/../attract.par 5 7 --calc-potentials=0 mono-leu.nbgrid
$ATTRACTDIR/attract start.dat $ATTRACTDIR/../attract.par  mono-leu.pdb mono-leu.pdb --rcut 9999999 --score > start.score
$ATTRACTDIR/attract start.dat $ATTRACTDIR/../attract.par  mono-leu.pdb mono-leu.pdb --grid 1 mono-leu.grid --fix-receptor > minim.dat
$ATTRACTDIR/attract minim.dat $ATTRACTDIR/../attract.par  mono-leu.pdb mono-leu.pdb --rcut 9999999 --score > minim.score

python2 $ATTRACTTOOLS/sort.py minim.dat > minim-sorted.dat
python2 $ATTRACTTOOLS/depivotize.py minim-sorted.dat > minim-sorted-depiv.dat
$ATTRACTDIR/fix_receptor minim-sorted-depiv.dat 2 > x
python2 $ATTRACTTOOLS/fill-energies.py x minim-sorted.dat > minim-sorted-center.dat
rm -f x
$ATTRACTTOOLS/top minim-sorted-center.dat 1 > minim-best.dat
$ATTRACTDIR/center mono-leu.pdb > mono-leu-centered.pdb
$ATTRACTDIR/collect minim-best.dat /dev/null mono-leu-centered.pdb | grep ATOM > minim-best-ligand.pdb
python2 $ATTRACTDIR/lrmsd.py minim-sorted-center.dat mono-leu-centered.pdb minim-best-ligand.pdb > minim-best.lrmsd


for grid in grid nbgrid; do
  for dat in start minim; do
    $ATTRACTDIR/attract $dat.dat $ATTRACTDIR/../attract.par  mono-leu.pdb mono-leu.pdb --rcut 9999999 --grid 1 mono-leu.$grid --fix-receptor --score > $dat-$grid.score &
  done
done
wait


