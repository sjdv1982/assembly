#!/usr/bin/env python3

import sys, os, argparse
import numpy as np

############
parser =argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('chains', type=str)
parser.add_argument('nfrag', type=int)
parser.add_argument('lrmsd', nargs='+', help="list of frag lrmsd")
parser.add_argument('--average', action="store_true")
args = parser.parse_args()
############

nfrag = args.nfrag
cc =  [ l for l in open(args.chains).readlines()]
print(cc[0],end="")
cc = cc[1:]
cc = [l.split() for l in cc]
chains = [ [int(i)-1 for i in l[-nfrag:]] for l in cc]
print(args.lrmsd, file=sys.stderr)
lrmsds = [ [float(l.split()[-1]) for l in open(f).readlines()] for f in args.lrmsd]

if args.average:
    for cnr, c in enumerate(chains):
        for p in cc[cnr][:-nfrag]:
            print(p, end=" ")
        for p in c:
            print(p, end=" ")
        rms = []
        for np, p in enumerate(c):
            rms.append(lrmsds[np][p])
        a = [sum(rms**2)/nfrag]**0.5
        print(a)

else:
    for cnr, c in enumerate(chains):
        for p in cc[cnr][:-nfrag]:
            print(p, end=" ")
        for p in c:
            print(p, end=" ")
        for fr, p in enumerate(c):
            print(lrmsds[fr][p], end=" ")
        print("")
