import sys
import random
import numpy as np
cimport numpy as np
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef int find_neighbours(np.ndarray[np.int_t, ndim=2] interaction, int p, np.ndarray[np.int_t, ndim=1] neighbours):
    cdef int Nb_neighbours
    cdef int n_inter
    with nogil:
        Nb_neighbours = 0
        for n_inter in range(interaction.shape[0]):
            if interaction[n_inter,0] == p:
                neighbours[Nb_neighbours] = interaction[n_inter,1]
                Nb_neighbours += 1
    return Nb_neighbours
#
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void stochastic_backtrack_rec(np.ndarray[np.float_t,ndim=1] energies, int frag, \
  np.ndarray[np.float_t,ndim=2] z, interactions,np.ndarray[np.int_t,ndim=1] neighbours, \
  np.ndarray[np.int_t,ndim=2] chains, int curr_chain, np.ndarray[np.float_t,ndim=2] rand
):
    cdef int nfrags, Nb_neighbours, nextelem, nr, previouselem, selected
    cdef float count
    cdef double threshold
    nfrags = z.shape[0]
    threshold = rand[curr_chain, frag+1]
    count = 0
    previouselem = chains[curr_chain, frag]
    Nb_neighbours = find_neighbours(interactions[frag], previouselem, neighbours)
    selected = neighbours[Nb_neighbours-1]
    #assert abs(sum([energies[i] * z[frag+1,i] / z[frag,p] for i in neighbours[:Nb_neighbours] ] ) - 1) < 0.0001
    for nr in range(Nb_neighbours):
        nextelem = neighbours[nr]
        count += energies[nextelem] * z[frag+1,nextelem] / z[frag,previouselem]
        if count > threshold:
            selected = nextelem
            break
    chains[curr_chain,frag+1] = selected
    if frag < nfrags - 2:
        stochastic_backtrack_rec(energies, frag+1, z, interactions, neighbours, chains, curr_chain, rand)
#####
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def stochastic_backtrack(np.ndarray[np.float_t,ndim=1] energies, double Z, np.ndarray[np.float_t,ndim=2] z, interactions, int num_samples):
    cdef int firstelem, selected, nfrags, nposes, curr_chain
    cdef double threshold, count
    cdef np.ndarray[np.float_t,ndim=2] rand
    cdef np.ndarray[np.int_t,ndim=2] chains
    nfrags, nposes = z.shape[0], z.shape[1]
    rand = np.random.random((num_samples, nfrags))
    neighbours = np.zeros(nposes, dtype=int)
    chains = np.zeros((num_samples, nfrags),dtype=int)
    for curr_chain in range(num_samples):
        if not (curr_chain+1)%10000:
            print(curr_chain+1,file=sys.stderr)
        threshold = rand[curr_chain, 0]
        count = 0
        selected = nposes - 1
        for firstelem in range(nposes):
            count += energies[firstelem] * z[0,firstelem] / Z
            if count > threshold:
                selected = firstelem
                break
        chains[curr_chain,0] = selected
        stochastic_backtrack_rec(energies, 0, z, interactions, neighbours, chains, curr_chain, rand)
    uniq_chains, occurencies = np.unique(chains, return_counts=True, axis=0)
    return uniq_chains, occurencies
#    return chains, chains[:,0] #####
