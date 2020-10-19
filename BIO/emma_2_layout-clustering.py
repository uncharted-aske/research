# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Idea: 
# * Assume an undirected graph
# * Use the graph Laplacian vectors as initial node embeddings
# * Create 2D/3D layout by dimensional reduction (UMAP)
# * Learn an ontology by applying hierarchy clustering (HDBSCAN)


# %%
import json
import time
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import pickle
import umap
import sklearn as skl

# %%
np.random.seed(0)

# %%[markdown]
# ## Load node and edge data from Dario

nodes = {}
with open('./data/covid19-snapshot_sep18-2020/processed/nodes.jsonl', 'r') as x:
    nodes = [json.loads(i) for i in x]

edges = {}
with open('./data/covid19-snapshot_sep18-2020/processed/edges.jsonl', 'r') as x:
    edges = [json.loads(i) for i in x]

edges_ = {}
with open('./data/covid19-snapshot_sep18-2020/processed/collapsedEdges.jsonl', 'r') as x:
    edges_ = [json.loads(i) for i in x]

x = None
del x

# %%
%%time

# Collate CollapseEdges with edges data
for edge in edges:
    i = edge['collapsed_id']
    edge['source'] = edges_[i]['source']
    edge['target'] = edges_[i]['target']

edges_ = None
del edges_

# %%[markdown]
# ## Calculate the graph Laplacian vectors

# %%
%%time

# Count node degree
z = np.array([[edge['source'], edge['target']] for edge in edges])
nodeDegreeCounts = np.array([[np.sum(z[:, j] == i) for j in range(2)] for i in range(len(nodes))])

# Construct undirected Laplacian in sparse format

# Off-diagonal Laplacian elements
x = np.array([-1.0 / np.sqrt(nodeDegreeCounts[edge['source'], :].sum() * nodeDegreeCounts[edge['target'], :].sum()) for edge in edges])
L = sp.sparse.csr_matrix((x, (z[:, 0], z[:, 1])), shape = (len(nodes), len(nodes)))

# Diagonal Laplacian ielements for all connected nodes
i = np.flatnonzero(nodeDegreeCounts.sum(axis = 1))
L[i, i] = 1.0

# %%[markdown]
# ## Calculate pair distance distribution (PDF)

%%time

def reduce_func(D_chunk, start):
    # iNeigh = [np.flatnonzero((np.abs(d) < 0.001)) for d in D_chunk]
    # dNeigh = [np.sort(np.abs(d))[1] for d in D_chunk]
    histDist = [np.histogram(np.abs(d), bins = 5000, range = (0, 2.5))[0] for d in D_chunk]
    return histDist

gen = skl.metrics.pairwise_distances_chunked(L, reduce_func = reduce_func, metric = 'euclidean', n_jobs = 3)

histDist = np.zeros((5000, ))
for z in gen:
    histDist += np.sum(z, axis = 0)

gen = None
del gen

# Time: 70 s

# %%
# Plot PDF
i = np.flatnonzero(histDist)
x = np.linspace(0, 2.5, 5000)[i]
y = histDist[i] / (len(nodes) ** 2)

fig, ax = plt.subplots(1, 1, figsize = (8, 6))
ax.scatter(x, y, s = 5, alpha = 0.5)
__ = plt.setp(ax, xlabel = 'Pairwise Distance (L2)', ylabel = 'Normalized Histogram', yscale = 'log', title = 'Pair Distance Function of Laplacian-Embedded Nodes')

fig.savefig('./figures/nodeLaplacianPDF.png', dpi = 150)

i = x = y = None
del i, x, y

# %%[markdown]
# ## Apply dimensional reduction with UMAP
%%time

numDimEmb = 2
# modelUMAP = umap.UMAP(n_components = numDimEmb, n_neighbors = 10, min_dist = 0.05, metric = 'euclidean', random_state = 0)
modelUMAP = umap.UMAP(n_components = numDimEmb, n_neighbors = 10, min_dist = 0.05, metric = 'minkowski', metric_kwds = {'p': 0.5}, random_state = 0)
posNodes = modelUMAP.fit_transform(L.tolil(copy = False))

# Time: 303 s

# %%[markdown]
# ## Clustering

# %%
# Try with just node types:
# ['chemical', 'protein', 'not-grounded', 'general', 'tissue', 'bioprocess']

z = [list(set([y[i] for y in nodes])) if not isinstance(nodes[0][i], dict) else [] for i in nodes[0].keys()]
labelNodes = np.asarray([np.sum([i if node['type'] == nodeType else 0 for i, nodeType in enumerate(z[3])]) for node in nodes])

# %%[markdown]
# ## Plot results

k = np.median(posNodes, axis = 0)
col = np.asarray([plt.cm.get_cmap('tab10')(i) for i in range(10)])

fig, ax = plt.subplots(1, 1, figsize = (12, 12))
for i, c in enumerate(z[3]):
    j = labelNodes == i
    __ = ax.scatter(posNodes[j, 0] - k[0], posNodes[j, 1] - k[1], s = 2, marker = 'o', alpha = 0.2, facecolor = col[i % 10, :3], label = f'{c}')

__ = plt.setp(ax, xlabel = 'x', ylabel = 'y', aspect = 1, title = 'Dimensionally Reduced Laplacian Node Embeddings')

l = [mpl.lines.Line2D([0], [0], marker = 'o', markersize = 2 ** 2, color = 'none', markeredgecolor = 'none', markerfacecolor = col[i, :3], alpha = 1.0, label = f'{z[3][i]}') for i in range(len(z[3]))]
__ = ax.legend(handles = l, loc = 'lower right')

fig.savefig('./figures/nodeLaplacianDimRed.png', dpi = 150)

# %%[markdown]
# ## Again but in 3D
%%time

numDimEmb = 3
modelUMAP = umap.UMAP(n_components = numDimEmb, n_neighbors = 10, min_dist = 0.05, metric = 'minkowski', metric_kwds = {'p': 0.5}, random_state = 0)
posNodes = modelUMAP.fit_transform(L.tolil(copy = False))

# %%[markdown]
# ## Output results
# 
# ```
# {
#   'id': <int>,
#   'x': <float>,
#   'y': <float>,
#   'z': <float>,
#   'group': [<int>, ...];
#   'score': [<float>, ...];
# }
# ```

output = [
    {
        'id': int(node['id']),
        'x': float(posNodes[i, 0]),
        'y': float(posNodes[i, 1]),
        'z': float(posNodes[i, 2]), 
        'group': [int(labelNodes[i])], 
        'score': [float(0.0)]
    }
    for i, node in enumerate(nodes)]

with open('./dist/nodeLayoutClustering.jsonl', 'w') as x:
    for i in output:
        json.dump(i, x)
        x.write('\n')


# %%
