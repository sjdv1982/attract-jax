#!/bin/bash

ATTRACTDIR=/attract/bin
awk '{print substr($0,58,2)}' frag5r.pdb | sort -nu > receptorgrid.alphabet
$ATTRACTDIR/make-grid-omp receptorr.pdb $ATTRACTDIR/../attract.par 5 7 receptorgrid.grid --alphabet receptorgrid.alphabet
$ATTRACTDIR/make-grid-omp receptorr.pdb $ATTRACTDIR/../attract.par 5 7 receptorgrid.nbgrid --alphabet receptorgrid.alphabet --calc-potentials=0
for frag in 5 6 7 8 9 10; do
  echo $frag
  $ATTRACTDIR/attract-infinite $ATTRACTDIR/../structure-single.dat $ATTRACTDIR/../attract.par receptorr.pdb frag${frag}r.pdb --score > bound-frag${frag}-infinite.score
  $ATTRACTDIR/attract $ATTRACTDIR/../structure-single.dat $ATTRACTDIR/../attract.par receptorr.pdb frag${frag}r.pdb --grid 1 receptorgrid.grid --fix-receptor --score > bound-frag${frag}-grid.score
  $ATTRACTDIR/attract $ATTRACTDIR/../structure-single.dat $ATTRACTDIR/../attract.par receptorr.pdb frag${frag}r.pdb --grid 1 receptorgrid.nbgrid --fix-receptor --score > bound-frag${frag}-nbgrid.score
done