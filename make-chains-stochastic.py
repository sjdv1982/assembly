import sys
import numpy as np
import pyximport
pyximport.install(
    language_level=3, 
    setup_args={"include_dirs":[np.get_include()]}
)
import lib_backtrack

boltzmann = float(3.2976230  * 6.022140857 * 10**(-4))

def map_npz(npz_file):
    sys.stderr.flush()
    npz = np.load(npz_file)
    nfrags = npz["nfrags"]
    poses, interactions =  [], []
    for n in range(nfrags-1):
        inter = npz["interactions-%d"%n]
        interactions.append(inter)
        poses.append(np.unique(inter[:,0]))
    poses.append(np.unique(inter[:,1]))
    npz = []
    #interactions = [ np.array(i, dtype=int) for i in inter]
    return interactions, poses

def store_energies(scores_file,RT):
    with open(scores_file,"r") as f:
        scores = np.array([l for l in f.readlines()], float)
    energies = np.exp(-scores/RT)
    return energies

def fwd(energies, interactions):
    nposes = len(energies)
    nfrags = len(interactions) + 1
    zbar = np.zeros((nfrags,nposes))
    zbar[-1] = 1.0
    for frag in range(nfrags-2, -1,-1):
        for inter in interactions[frag]:
            previouselem = inter[0]
            nextelem = inter[1]
            zbar[frag, previouselem] += energies[nextelem] * zbar[frag+1, nextelem]
    return zbar

def bwd(energies, interactions):
    nposes = len(energies)
    nfrags = len(interactions) + 1
    ybar = np.zeros((nfrags,nposes))
    ybar[0] = 1.0
    for frag in range(1, nfrags):
        for inter in interactions[frag-1]:
            previouselem = inter[0]
            nextelem = inter[1]
            ybar[frag, nextelem] += energies[previouselem]  * ybar[frag-1, previouselem]
    return ybar

def fwd_bwd(npz_file, scores_file, temperature):
    RT = boltzmann * temperature

    interactions = map_npz(npz_file)[0]
    print("Interactions mapped",file=sys.stderr)
    nfrags = len(interactions) + 1

    #Storing the energy values
    energies = np.array(store_energies(scores_file, RT))
    print("Scores stored",file=sys.stderr)
    nposes = len(energies)   # total Nb poses

    #The forward-backward algorithm
    z = fwd(energies, interactions)
    y = bwd(energies, interactions)

    #print(z)
    #print(y)
    #The Big "Z"
    Z = sum(energies * z[0,:])
    print("Big Z2 calculated : %s"%Z,file=sys.stderr)
    #Y = sum(y[:,-1])
    #print("Big Y2 calculated : %s"%Y)

    #Calculating the value of E(p*) per fragment
    E = y*energies*z # E[pose, frag]
    #print("E %s"%E)

    Bprob = np.sum(E, axis = 0)/np.sum(E, axis = (0,1))
    ###print("Bprob %s"%Bprob)

    result = {
        "energies": energies,
        "z": z,
        "interactions": interactions,
        "Bprob": Bprob
    }
    return energies, z, interactions, Bprob

def stochastic_backtrack(energies, z, interactions, num_samples, seed):
    Z = (energies * z[0,:]).sum()

    np.random.seed(seed)
    chains, occurrencies = lib_backtrack.stochastic_backtrack(energies, Z, z, interactions, num_samples)

    counted = sorted(list(zip(occurrencies, chains)), key=lambda k: -k[0])
    chains_sorted = np.array([c[1] for c in counted])
    occurrencies_sorted =  np.array([c[0] for c in counted])
    return chains_sorted, occurrencies_sorted

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("npz_file")
    p.add_argument("scores_file")
    p.add_argument("num_samples",type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--temperature", default=293, type=float)
    args = p.parse_args()
    npz_file = args.npz_file
    scores_file = args.scores_file
    num_samples = args.num_samples
    seed = args.seed
    temperature = args.temperature

    energies, z, interactions, Bprob = fwd_bwd(npz_file, scores_file, temperature)
    chains, occurrencies = stochastic_backtrack(energies, z, interactions, num_samples, seed)
    print("#header <occurrency> <ranks>")
    for chain, occ in zip(chains, occurrencies):
        print(occ, end=' ')
        for j in chain:
            print(j+1, end=' ')
        print()
