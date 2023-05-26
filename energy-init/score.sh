for dis in `cat make-energy-init.out`; do
    sed 's/XXXXXX/'$dis'/' template.dat > struc.dat
    $ATTRACTDIR/attract struc.dat $ATTRACTDIR/../attract.par  leu-sphere.pdb leu-sphere.pdb --rcut 9999999 --score | grep Energy
done