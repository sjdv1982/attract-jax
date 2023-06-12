#!/usr/bin/env python3
## Copyright (C) Isaure Chauvot de Beauchene (CNRS)
## Modified by Sjoerd de Vries

import sys, argparse
import numpy as np

#######################################
parser =argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('pdbfile', help="(list of) pdb to convert")
parser.add_argument('--outp', help="output: uniq npy file")
parser.add_argument('--backbone', help="only select backbone (CA, C, O, N) atoms", action="store_true")
parser.add_argument("--list",help="input is a list of files", action="store_true")

args = parser.parse_args()
#######################################

if not args.outp:
    outp = args.pdbfile.split('.')[0] + '.npy'
else:
    outp = args.outp

def pdb2npy(pdb, backbone):
    coord = []
    for l in pdb:
        if l.startswith("MODEL") or len(coord) == 0:
            coord.append([])
        if not l.startswith("ATOM"):
            continue
        if backbone:
            if l[13:15] not in ("CA","C ","O ","N "):
                continue
        coord[-1].append([ float(i) for i in [l[30:38], l[38:46], l[46:54]] ] )
    coord = [c for c in coord if len(c)]
    for c in coord:
        assert(len(c) == len(coord[0]))
    return(np.array(coord,dtype=np.float32))

if args.list:
    coord = []
    for l in open(args.pdbfile).readlines():
        c =  pdb2npy( open(l.split()[0]).readlines(), args.backbone )
        coord.append(c[0])
        assert(c[0].shape == coord[0].shape), (l, c[0].shape, coord[0].shape)
    print(len(coord), file=sys.stderr)
    coord = np.array(coord)
else:
    assert  sys.argv[1].split(".")[-1] == "pdb"
    coord = pdb2npy(open(args.pdbfile).readlines(), args.backbone)

if len(coord) == 1:
    np.save(outp, coord[0])

else:
    np.save(outp, coord)
