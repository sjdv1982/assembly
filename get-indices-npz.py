import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("graph", type=argparse.FileType(), help="npz graph to analyze")
parser.add_argument("position", type=int, help="Fragment position to get indices(ranks) from")
args = parser.parse_args()
graph = np.load(args.graph.name)
nfrags = graph["nfrags"]
if args.position < 1 or args.position > nfrags:
    print("position must be between 1 and {}".format(nfrags))
    exit(1)
indices = None
if args.position > 1:
    interactions = graph[f"interactions-{args.position-2}"]
    indices = interactions[:, 1] + 1
if args.position < nfrags:
    interactions = graph[f"interactions-{args.position-1}"]
    new_indices = interactions[:, 0] + 1
    if indices is not None:
        indices = np.concatenate((indices, new_indices))
    else:
        indices = new_indices
indices = np.unique(indices)
for ind in indices:
    print(ind)        