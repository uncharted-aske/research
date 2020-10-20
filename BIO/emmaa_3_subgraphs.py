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
# ## Load previous workspace variables

with open('./dist/emmaa_2_layout-clustering.pkl', 'rb') as x:
    [nodes, edges, nodeDegreeCounts, posNodes, posNodes_sphCart, clusterIDs, clusterLabels] = pickle.load(x)

# %%
# ## Facet by belief scores
# %%
# Aggregate belief scores by node
z = [[] for node in nodes]
for edge in edges:
    i = edge['source']
    j = edge['target']
    z[i].append(edge['belief'])
    z[j].append(edge['belief'])

nodeBeliefScores = np.array([max(i) if len(i) else 0.0 for i in z])
edgeBeliefScores = np.array([edge['belief'] for edge in edges])

# Plot histogram of belief scores
k = 25
y, x = np.histogram(edgeBeliefScores, bins = k, range = (0, 1))
w, __ = np.histogram(nodeBeliefScores, bins = k, range = (0, 1))

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6))
__ = ax.bar(x[:-1], y / len(edges), width = 0.75 * (1.0 / k) * 0.5, align = 'edge', label = 'Per Edges')
__ = ax.bar(x[:-1] + 0.75 * (1.0 / k) * 0.5, w / len(nodes), width = 0.75 * (1.0 / k) * 0.5, align = 'edge', label = 'Max per Node')
__ = plt.setp(ax, xlabel = 'Belief Scores', ylabel = 'Fraction of Edges or Nodes', title = 'Distribution of Belief Scores')
__ = ax.legend()

fig.savefig('./figures/nodeBeliefScoresHistogram.png', dpi = 150)

# %%

k = 0.95
j = nodeBeliefScores > k
# markerSize = 100 * nodeBeliefScores ** 2 + 0.1
markerSize = np.log10(nodeDegreeCounts.sum(axis = 1) + 2) ** 4

fig, ax = emlib.plot_emb(coor = posNodes[j, :2], labels = clusterIDs[j], marker_size = markerSize[j], marker_alpha = 0.5, cmap_name = 'qual', colorbar = True, str_title = f'Belief Score > {k} ({len(np.flatnonzero(j))} Nodes Shown)')
fig.savefig(f'./figures/nodesBeliefScore.png', dpi = 150)


# Output belief score filtering results
outputNodes = [
    {
        'id': int(nodes[i]['id']),
        'x': float(posNodes[i, 0]),
        'y': float(posNodes[i, 1]),
        'z': float(posNodes[i, 2]), 
        'size': markerSize[i],
        'clusterID': [int(clusterIDs[i])], 
        'clusterScore': [float(0.0)]
    }
    for i in np.flatnonzero(j)]

with open(f'./dist/nodeLayoutClustering_belief{100*k:2.0f}.jsonl', 'w') as x:
    for i in outputNodes:
        json.dump(i, x)
        x.write('\n')

with open(f'./dist/nodeLayoutClustering_belief{100*k:2.0f}.pkl', 'wb') as x:
    pickle.dump(outputNodes, x)

# Same but for spherical layout
# Output belief score filtering results
outputNodes = [
    {
        'id': int(nodes[i]['id']),
        'x': float(posNodes_sphCart[i, 0]),
        'y': float(posNodes_sphCart[i, 1]),
        'z': float(posNodes_sphCart[i, 2]), 
        'size': markerSize[i],
        'clusterID': [int(clusterIDs[i])], 
        'clusterScore': [float(0.0)]
    }
    for i in np.flatnonzero(j)]

with open(f'./dist/nodeLayoutClustering_sphCart_belief{100*k:2.0f}.jsonl', 'w') as x:
    for i in outputNodes:
        json.dump(i, x)
        x.write('\n')

with open(f'./dist/nodeLayoutClustering_sphCart_belief{100*k:2.0f}.pkl', 'wb') as x:
    pickle.dump(outputNodes, x)


# %%
%%time

# texts = ['SARS-Cov-2', 'chloroquine', 'remdesivir']
# texts = ['chloroquine']
# texts = ['personal protective equipment']
texts = ['BRCA']

markerSize = np.log10(nodeDegreeCounts.sum(axis = 1) + 2) ** 4

for numHops in [1, 2]:

    # Get node/edge indices within the directed hop neighbourhood of given text
    textsIndex, textsNodeIndex, textsEdgeIndex, nodeFlags, edgeFlags = emlib.getTextNodeEdgeIndices(nodes = nodes, edges = edges, texts = texts, numHops = numHops)

    print(f'{len(textsNodeIndex)} nodes and {len(textsEdgeIndex)} edges within the directed neighbourhood of the given node(s) {textsIndex}\n')

    # Plot results
    i = nodeFlags
    fig, ax = emlib.plot_emb(coor = posNodes[i, :], labels = clusterIDs[i], 
        marker_size = markerSize[i], marker_alpha = 1.0, cmap_name = 'qual', colorbar = True, 
        str_title = f'{len(textsNodeIndex)} Nodes and {len(textsEdgeIndex)} Edges Shown')
    
    # fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
    __ = [ax.plot(
        posNodes[[edges[j]['source'], edges[j]['target']], 0], 
        posNodes[[edges[j]['source'], edges[j]['target']], 1], 
        posNodes[[edges[j]['source'], edges[j]['target']], 2],
        color = 'k', linewidth = 0.5, alpha = 0.5, zorder = 0) for j in textsEdgeIndex]

    x = (-1.75, 1.75)
    __ = plt.setp(ax, xlim = x, ylim = x, zlim = x)
    fig.savefig(f'./figures/nodes_{texts[0]}_{numHops}hops.png', dpi = 150)

    # Output graph data
    with open(f'./dist/nodes_{texts[0]}_{numHops}hops.jsonl', 'w') as x:
        for i in textsNodeIndex:
            json.dump(nodes[i], x)
            x.write('\n')

    with open(f'./dist/edges_{texts[0]}_{numHops}hops.jsonl', 'w') as x:
        for i in textsEdgeIndex:
            json.dump(edges[i], x)
            x.write('\n')


    # Output layout and clustering data
    outputNodes = [
        {
            'id': int(nodes[i]['id']),
            'x': float(posNodes[i, 0]),
            'y': float(posNodes[i, 1]),
            'z': float(posNodes[i, 2]), 
            'size': markerSize[i],
            'clusterID': [int(clusterIDs[i])], 
            'clusterScore': [float(0.0)]
        }
        for i in textsNodeIndex
    ]

    with open(f'./dist/nodeLayoutClustering_{texts[0]}_{numHops}hops.jsonl', 'w') as x:
        for i in outputNodes:
            json.dump(i, x)
            x.write('\n')

    # Output layout and clustering data (spherical)
    outputNodes = [
        {
            'id': int(nodes[i]['id']),
            'x': float(posNodes_sphCart[i, 0]),
            'y': float(posNodes_sphCart[i, 1]),
            'z': float(posNodes_sphCart[i, 2]), 
            'size': markerSize[i],
            'clusterID': [int(clusterIDs[i])], 
            'clusterScore': [float(0.0)]
        }
        for i in textsNodeIndex
    ]

    with open(f'./dist/nodeLayoutClustering_sphCart_{texts[0]}_{numHops}hops.jsonl', 'w') as x:
        for i in outputNodes:
            json.dump(i, x)
            x.write('\n')


# %%[markdown]
# ## Generate subgraphs 'clustered' by tested/untested paths

# %%
# Load paths

paths_curated = {}
with open('./data/covid19-snapshot_sep18-2020/processed/curated_tests.jsonl', 'r') as x:
    paths_curated = [json.loads(i) for i in x]

paths_mitre = {}
with open('./data/covid19-snapshot_sep18-2020/processed/mitre_tests.jsonl', 'r') as x:
    paths_mitre = [json.loads(i) for i in x]


x = list(set([i for n in [path['nodes'] for path in paths_curated] for i in n]))
y = list(set([i for n in [path['edges'] for path in paths_curated] for m in n for i in m]))
print(f'{len(paths_curated)} curated paths with {len(x)} unique nodes and {len(y)} edges.')

x = list(set([i for n in [path['nodes'] for path in paths_mitre] for i in n]))
y = list(set([i for n in [path['edges'] for path in paths_mitre] for m in n for i in m]))
print(f'{len(paths_mitre)} mitre paths with {len(x)} unique nodes and {len(y)} edges.')

# 247 curated paths with 183 unique nodes and 305 edges.
# 4818 mitre paths with 1834 unique nodes and 4554 edges.

# %%
# Plot and output curated paths

# Belief score threshold
b = 0.0
# b = 0.95

# for paths, z in [[paths_curated, 'Curated']]:
for paths, z in [[paths_curated, 'Curated'], [paths_mitre, 'Mitre']]:

    # Conditional indices for the nodes and edges in the tested paths
    nodeFlags = np.full((len(nodes), ), False)
    edgeFlags = np.full((len(edges), ), False)
    x = list(set([i for n in [path['nodes'] for path in paths] for i in n]))
    y = list(set([i for n in [path['edges'] for path in paths] for m in n for i in m]))
    k = []
    l = []
    for i in x:
        if i in range(len(nodes)):
            nodeFlags[i] = True
        else:
            k.append(i)
    for i in y:
        if i in range(len(edges)):
            edgeFlags[i] = True
        else:
            l.append(i)

    # Conditional index for belief score filtering
    j = edgeFlags * (edgeBeliefScores > b)
    x = list(set([m for n in [[edges[n]['source'], edges[n]['target']] for n in np.flatnonzero(j)] for m in n]))
    y = np.full((len(nodes), ), False)
    for n in x:
        y[n] = True
    i = nodeFlags * y

    # Plot
    title = f'{z} Paths (Belief Score > {b:.2f}) - {sum(i)} Nodes and {sum(j)} Edges Shown'
    fig, ax = emlib.plot_emb(
        coor = posNodes_sphCart[i, :], labels = clusterIDs[i], 
        marker_size = markerSize[i], marker_alpha = 1.0, 
        cmap_name = 'qual', colorbar = True, str_title = title)
    __ = [ax.plot(
        posNodes_sphCart[[edges[n]['source'], edges[n]['target']], 0], 
        posNodes_sphCart[[edges[n]['source'], edges[n]['target']], 1], 
        posNodes_sphCart[[edges[n]['source'], edges[n]['target']], 2],
        color = 'k', linewidth = 0.5, alpha = 0.1, zorder = 0) for n in np.flatnonzero(j)]

    fig.savefig(f'./figures/nodes_sphCart_{z.lower()}_belief{100*b:.0f}.png', dpi = 150)


    # Output data
    with open(f'./dist/nodes_{z.lower()}_belief{100*b:.0f}.jsonl', 'w') as x:
        for n in np.flatnonzero(i):
            json.dump(nodes[n], x)
            x.write('\n')

    with open(f'./dist/edges_{z.lower()}_belief{100*b:.0f}.jsonl', 'w') as x:
        for n in np.flatnonzero(j):
            json.dump(edges[n], x)
            x.write('\n')

    outputNodes = [
        {
            'id': int(nodes[n]['id']),
            'x': float(posNodes_sphCart[n, 0]),
            'y': float(posNodes_sphCart[n, 1]),
            'z': float(posNodes_sphCart[n, 2]), 
            'size': float(markerSize[n]),
            'clusterID': [int(clusterIDs[n])], 
            'clusterScore': [float(0.0)]
        }
        for n in np.flatnonzero(i)
    ]

    with open(f'./dist/nodeLayoutClustering_sphCart_{z.lower()}_belief{100*b:.0f}.jsonl', 'w') as x:
        for n in outputNodes:
            json.dump(n, x)
            x.write('\n')

# Note: 
# Mitre tested paths reference edgeID = 281075 which doesn't exist in `edges.jsonl`

# %%





# %%

