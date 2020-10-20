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

import emmaa_lib as emlib
import importlib
# `importlib.reload(emlib)`

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

numDimEmb = 3
# modelUMAP = umap.UMAP(n_components = numDimEmb, n_neighbors = 10, min_dist = 0.05, metric = 'euclidean', random_state = 0)
modelUMAP = umap.UMAP(n_components = numDimEmb, n_neighbors = 10, min_dist = 0.05, metric = 'minkowski', metric_kwds = {'p': 2.0/3.0}, random_state = 0)
posNodes = modelUMAP.fit_transform(L.tolil(copy = False))

posCentroid = np.median(posNodes, axis = 0)
posNodes = posNodes - posCentroid

# Time: 277 s

# %%[markdown]
# ## Clustering

# %%
# Try with just node types:
# ['chemical', 'protein', 'not-grounded', 'general', 'tissue', 'bioprocess']

clusterLabels = [list(set([y[i] for y in nodes])) if not isinstance(nodes[0][i], dict) else [] for i in nodes[0].keys()][3]
clusterIDs = np.asarray([np.sum([i if node['type'] == nodeType else 0 for i, nodeType in enumerate(clusterLabels)]) for node in nodes])

# %%[markdown]
# ## Plot results

markerSize = np.log10(nodeDegreeCounts.sum(axis = 1) + 2) ** 4

emlib.plot_emb(coor = posNodes, labels = clusterIDs, cmap_name = 'qual', colorbar = True, str_title = 'Dimensionally Reduced Laplacian Node Embeddings')

# %%[markdown]
# ## Plot results

# %%
# 2D projection
fig, ax = emlib.plot_emb(coor = posNodes[:, :2], labels = clusterIDs, marker_size = markerSize, marker_alpha = 0.5, cmap_name = 'qual', colorbar = True, str_title = 'Dimensionally Reduced Laplacian Node Embeddings')

fig.savefig('./figures/nodeLaplacianDimRed_2D.png', dpi = 150)

# %%
# Full 3D
fig, ax = emlib.plot_emb(coor = posNodes, labels = clusterIDs, marker_size = markerSize, marker_alpha = 0.1, cmap_name = 'qual', colorbar = True, str_title = 'Dimensionally Reduced Laplacian Node Embeddings')

fig.savefig('./figures/nodeLaplacianDimRed_3D.png', dpi = 150)

# %%[markdown]
# ## Output results

# %%[markdown]
# ```
# {
#   'id': <int>,
#   'x': <float>,
#   'y': <float>,
#   'z': <float>,
#   'size': <float>,
#   'group': [<int>, ...];
#   'score': [<float>, ...];
# }
# ```

outputNodes = [
    {
        'id': int(node['id']),
        'x': float(posNodes[i, 0]),
        'y': float(posNodes[i, 1]),
        'z': float(posNodes[i, 2]), 
        'size': markerSize[i],
        'clusterID': [int(clusterIDs[i])], 
        'clusterScore': [float(0.0)]
    }
    for i, node in enumerate(nodes)]

with open('./dist/nodeLayoutClustering.jsonl', 'w') as x:
    for i in outputNodes:
        json.dump(i, x)
        x.write('\n')

with open('./dist/nodeLayoutClustering.pkl', 'wb') as x:
    pickle.dump(outputNodes, x)

# %%
# Cluster meta-data

outputClusters = [
    {
        'clusterID': int(i),
        'clusterLabel': c
    }
    for i, c in enumerate(clusterLabels)
]

with open('./dist/nodeClusters.jsonl', 'w') as x:
    for i in outputClusters:
        json.dump(i, x)
        x.write('\n')

with open('./dist/nodeClusters.pkl', 'wb') as x:
    pickle.dump(outputClusters, x)


# %%[markdown]
# ## Experiment: Dimensional Reduction on a Sphere
%%time

numDimEmb = 2
# modelUMAP = umap.UMAP(n_components = numDimEmb, n_neighbors = 10, min_dist = 0.05, metric = 'euclidean', random_state = 0)
modelUMAP_sph = umap.UMAP(n_components = numDimEmb, n_neighbors = 10, min_dist = 0.05, metric = 'minkowski', metric_kwds = {'p': 2.0/3.0}, random_state = 0, output_metric = 'haversine')
posNodes_sph = modelUMAP_sph.fit_transform(L.tolil(copy = False))

# %%
# Transform to Cartesian coordinates
posNodes_sphCart = np.empty((posNodes_sph.shape[0], 3))
posNodes_sphCart[:, 0] = np.sin(posNodes_sph[:, 0]) * np.cos(posNodes_sph[:, 1])
posNodes_sphCart[:, 1] = np.sin(posNodes_sph[:, 0]) * np.sin(posNodes_sph[:, 1])
posNodes_sphCart[:, 2] = np.cos(posNodes_sph[:, 0])

# %%
# Plot result

fig, ax = emlib.plot_emb(coor = posNodes_sphCart, labels = clusterIDs, marker_size = np.log10(nodeDegreeCounts.sum(axis = 1) + 2) ** 4, marker_alpha = 0.5, cmap_name = 'qual', colorbar = True, str_title = 'Dimensionally Reduced Laplacian Node Embeddings')

fig.savefig('./figures/nodeLaplacianDimRed_sph.png', dpi = 150)

# %%

outputNodes = [
    {
        'id': int(node['id']),
        'x': float(posNodes_sphCart[i, 0]),
        'y': float(posNodes_sphCart[i, 1]),
        'z': float(posNodes_sphCart[i, 2]),
        'size': markerSize[i],
        'clusterID': [int(clusterIDs[i])], 
        'clusterScore': [float(0.0)]
    }
    for i, node in enumerate(nodes)]

with open('./dist/nodeLayoutClustering_sphCart.jsonl', 'w') as x:
    for i in outputNodes:
        json.dump(i, x)
        x.write('\n')

with open('./dist/nodeLayoutClustering_sphCart.pkl', 'wb') as x:
    pickle.dump(outputNodes, x)

# %%[markdown]
# ## Output Workspace Variables

with open('./dist/emmaa_2_layout-clustering.pkl', 'wb') as x:
    pickle.dump([nodes, edges, nodeDegreeCounts, posNodes, posNodes_sphCart, clusterIDs, clusterLabels], x)
    