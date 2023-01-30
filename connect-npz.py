#!/usr/bin/env python2

"""
connect.py
Calculates the connections between poses of consecutive fragments based on overlap RMSD
During the calculation, poses of half of the fragments are clustered hierarchically, to speed up the calculation
Prints out the connectivity tree in npz format

NOTE: First run get_msd_build.py to build the _get_msd Python extension module

Argument 1: nfrags, the number of fragments
Argument 2: the maximum RMSD
Argument 3: the maximum number of poses to consider (take the first poses in the .postatoms, .preatoms files)
Argument 4: the mimimum number of children clusters in 1st decomposition
Argument 5: output = connectivity graph in npz format
Argument 6 to (5 + nfrags -1): the "preatoms" portion of atom coordinates of the poses (the atoms that overlap with the next fragment)
Argument (6 + nfrags) to (6 + 2 * nfrags - 1): the "postatoms" portion of atom coordinates of the poses (the atoms that overlap with the previous fragment)
NOTE: the preatoms and postatoms are in .npy format, and must be sorted by ATTRACT rank! The first pose is rank 1, the 2nd is rank 2, etc.

Argument (5 + 2 * nfrags) - (5 + 3 * nfrags): optional: lists of pose indices to select for each fragment

Copyright 2015-2017 Sjoerd de Vries, Isaure Chauvot de Beauchene, TUM
"""
import sys, numpy as np, bisect
import connectlib
from connectlib import Cluster, decompose, MAX_CLUSTERING, CLUSTERING

###################################################
if __name__ == "__main__":
    if len(sys.argv) < 8:
        raise Exception("usage: connect.py nfrags rmsd maxstruc MINCHILD outp.npz [preatoms.npy] [postatoms.npy] [sel]")
    nfrags = int(sys.argv[1])
    max_rmsd = float(sys.argv[2]) #overlapping cutoff (<3.0A recommended)
    maxstruc = int(sys.argv[3]) #take only the maxstruc top-ranked poses.
    MINCHILD = int(sys.argv[4])
    outp = sys.argv[5]
    connectlib.MINCHILD = MINCHILD
    nargs = len(sys.argv) - 6

    # you can give selections of poses as arguments. see l38
    assert nargs == 2 * nfrags or nargs == 3 * nfrags, (nargs, nfrags)
    print("nfrags", nfrags, file=sys.stderr)
    assert nfrags >= 2
    max_msd = max_rmsd**2

    preatoms = sys.argv[6:nfrags+6]
    postatoms = sys.argv[nfrags+6:2*nfrags+6]
    print("PREATOMS", preatoms, file=sys.stderr)
    print("POSTATOMS", postatoms, file=sys.stderr)
    postatoms = [np.load(f) for f in postatoms]
    preatoms = [np.load(f) for f in preatoms]

    # lists of ranks to consider for each pose pool, counting from 1
    selections = [[] for n in range(nfrags)]
    if nargs == 3 * nfrags:
        selections = sys.argv[2*nfrags+6:3*nfrags+6]
        print("SELECTIONS", selections, file=sys.stderr)
        selections = [np.array(sorted([int(l.split()[0]) for l in open(f) if len(l.strip())])) for f in selections]

    # dimensions = (Nb poses, Nb atoms * 3 coordinates)
    for arr in (preatoms, postatoms):
        for anr, a in enumerate(arr):
            if len(a.shape) == 3 and a.shape[-1] == 3:
                a = a.reshape(len(a), -1)
                arr[anr] = a
            assert len(a.shape) == 2 and a.shape[1] % 3 == 0, a.shape

    for arr in (preatoms, postatoms):
        for anr, a in enumerate(arr):
            ncoor = a.shape[1] // 3
            arr[anr] = a.reshape(a.shape[0], ncoor, 3)

    #If you use both maxstruc and selection,
    #remove from selection what is beyond rank maxstruc
    if maxstruc > 0:
        for arr in (preatoms, postatoms):
            for anr, a in enumerate(arr):
                arr[anr] = arr[anr][:maxstruc]
        for selnr, sel in enumerate(selections):
            if not len(sel): continue
            pos = bisect.bisect_right(sel, maxstruc)
            selections[selnr] = sel[:pos]
            assert len(selections[selnr])

    # preatoms_frag(i) and postatoms_frag(i) must have
    # the same number of poses (nstruc)
    nstruc = []
    for n in range(nfrags):
        a1 = preatoms[n]
        a2 = postatoms[n]
        assert a1.shape[0] == a2.shape[0], (n, a1.shape, a2.shape)
        nstruc.append(a1.shape[0])

    # postatoms_frag(i) and preatoms_frag(i-1) must have
    # the same number of atoms, as they overlap in sequence.
    for n in range(1, nfrags):
        a1 = preatoms[n-1]
        a2 = postatoms[n]
        assert a1.shape[1] == a2.shape[1], (n, a1.shape, n+1, a2.shape)

    # Check that the pose exists for each rank in selection.
    for selnr, sel in enumerate(selections):
        for conf in sel:
            assert conf > 0 and conf <= nstruc[selnr], (conf, nstruc[selnr])

    ranks = [np.arange(s)+1 for s in nstruc]
    for n in range(nfrags):
        if len(selections[n]):
            preatoms[n] = preatoms[n][selections[n]-1]
            postatoms[n] = postatoms[n][selections[n]-1]
            ranks[n] = selections[n]
            nstruc[n] = len(selections[n])

    #Build cluster tree
    clusters = []
    for n in range(nfrags):
        for atoms in (postatoms, preatoms):
            i = n + 1
            if atoms is preatoms: i += 1000 # see clustid scheme (line 112)
            c = Cluster(clusters, (i,), None, atoms[n], ranks[n])
            if not (n % 2):
                a = atoms[n]
                r = ranks[n]
                for nn in range(len(a)):
                    cc = Cluster(clusters, (i,nn), MAX_CLUSTERING, a[nn:nn+1], np.array(r[nn:nn+1]))
                    cc.parent = c
                    c.children.append(cc)
                c.nodes = len(a)
                clusters.append([c])
                continue

            clusterlevel = 0
            c.cluster(clusterlevel)
            for clusterlevel in range(1, len(CLUSTERING)-1):
                if len(c.children) >= MINCHILD: break
                c.dissolve(clusterlevel)
            count = 0
            assert c.clusterlevel is None or c.clusterlevel == MAX_CLUSTERING

            def split_all(c):
                """Split c and all of its children, all the way down"""
                global count
                if not c._splittable: return
                if not len(c.children):
                    ok = c.split()
                    count += 1
                    if not ok: return
                for cc in c.children:
                    split_all(cc)

            split_all(c)
            c.reorganize()
            #print >> sys.stderr, n+1, nstruc[n], CLUSTERING[clusterlevel], len(c.children), c.nodes
            assert c.clusterlevel is None or c.clusterlevel == MAX_CLUSTERING
            clusters.append([c])

    #Initialize tree connections, intra-fragment, flat
    for n in range(0, 2 * nfrags, 4):
        c1, c2 = clusters[n][0], clusters[n+1][0]
        #print >> sys.stderr,  n, n+1, len(c1.children), len(c2.children)
        for nn in range(len(c1.children)):
            cc1, cc2 = c1.children[nn], c2.children[nn]
            cc1.connections = [cc2]
            cc2.back_connections = [cc1]

    #Initialize tree connections, intra-fragment, hierarchical
    for n in range(2, 2 * nfrags, 4):
        c1, c2 = clusters[n][0], clusters[n+1][0]
        c1.connections.append(c2)
        c2.back_connections.append(c1)

    #Initialize tree connections, inter-fragment, flat to hierarchical
    for n in range(2, 2 * nfrags, 4):
        c1, c2 = clusters[n-1][0], clusters[n][0]
        for cc in c1.children:
            cc.connections = [c2]
            c2.back_connections.append(cc)
        clusters[n-1] = c1.children

    #Initialize tree connections, inter-fragment, hierarchical to flat
    for n in range(4, 2 * nfrags, 4):
        c1, c2 = clusters[n-1][0], clusters[n][0]
        for cc in c2.children:
            cc.back_connections = [c1]
            c1.connections.append(cc)
        clusters[n] = c2.children

    clusters[0] = clusters[0][0].children
    if len(clusters[-2]) > 1:
        clusters[-1] = clusters[-1][0].children

    clusters[:] = [set(c) for c in clusters]
    for c in clusters:
        for cc in c:
            cc.parent = None
            cc.check_parentage(None)

    #Decompose tree (divide clusters in sub-clusters)
    # Decompose first the clusters of poses for the extrem fragments
    # (fisrt and last in chain, then 2nd first and 2nd last...)
    # to gain efficiency, as less poses will be connected for those fragments
    # (at least in the exemples tested so far)
    step = 0
    to_decompose = []
    to_decompose0 = list(range(2, len(clusters), 4))
    while len(to_decompose0):
        to_decompose.append(to_decompose0.pop(0))
        if not len(to_decompose0): break
        to_decompose.append(to_decompose0.pop(-1))

    # Decompose preatoms and postatoms alternately at each clustering level,
    # so that you eliminate clusters that have no connections
    # e.g. from the postatoms to the preatoms, because the corresponding preatoms
    # had been eliminated as they did not connect to any pose of the downstream fragment
    for clusnr in to_decompose:
        done1, done2 = False, False
        while 1:
            if not (step % 5):
                print([len(c) for c in clusters], file=sys.stderr)
                print([sum([cc.nodes for cc in c]) for c in clusters], file=sys.stderr)
            step += 1
            if not done1:
                ok1 = decompose(clusters, clusnr, max_rmsd)
                if not ok1:
                    if done2: break
                    done1 = True
            if not done2:
                ok2 = decompose(clusters, clusnr+1, max_rmsd)
                if not ok2:
                    if done1: break
                    done2 = True

    print([len(c) for c in clusters[::2]], file=sys.stderr)
    print([sum([cc.nodes for cc in c]) for c in clusters[::2]], file=sys.stderr)

    #Verification
    for c in clusters[1::2]:
        for cc in c:
            cc.verify(max_rmsd)

    #Sort clusters
    clusters[:] = [list(c) for c in clusters]
    for c in clusters:
        c.sort(key=lambda clus: clus.ranks[0])

    #write out tree
    tree = {"nfrags":nfrags, "max_rmsd": max_rmsd}
    count = 0
    for cnr in range(1,len(clusters), 2):
        c = clusters[cnr]
        clus = []
        inter = []
        for ccnr, cc in enumerate(c):
            cclus = {"radius":CLUSTERING[cc.clusterlevel], "ranks": cc.ranks.tolist()}
            clus.append(cclus)
            if cnr < len(clusters)-1:
                for other in cc.connections:
                    inter.append((cc.ranks[0] - 1, other.ranks[0] - 1))
                    #print >> sys.stderr, (cc.ranks[0] - 1, other.ranks[0] - 1)
        inter.sort(key=lambda i: 100000*i[0]+i[1])
        if cnr < len(clusters)-1:
            tree["interactions-%i"%count] = inter
            count+=1

    np.savez(outp, **tree)
