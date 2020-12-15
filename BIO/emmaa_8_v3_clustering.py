# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Idea: 
# * Read `node2vec` node embeddings from Luis
# * Cluster with HDSCAN
# * Reduce to 3D with UMAP


# %%
import json
import pickle
import time
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import numba
import umap
import hdbscan

import emmaa_lib as emlib
import importlib
# importlib.reload(emlib)

# %%
np.random.seed(0)

# %%[markdown]
# # Load Data

nodes_mitre = emlib.load_jsonl('./dist/v3/nodes_mitre.jsonl', remove_preamble = True)
edges_mitre = emlib.load_jsonl('./dist/v3/edges_mitre.jsonl', remove_preamble = True)

nodeEmb_mitre = []
nodeEmb_mitre.append(emlib.load_jsonl('./G_mitre_p1_q05_n10len80_undirected.jsonl', remove_preamble = False))
nodeEmb_mitre.append(emlib.load_jsonl('./G_mitre_p1_q1_n10len80_undirected.jsonl', remove_preamble = False))
nodeEmb_mitre.append(emlib.load_jsonl('./G_mitre_p1_q2_n10len80_undirected.jsonl', remove_preamble = False))

num_nodes = len(nodes_mitre)
num_embs = len(nodeEmb_mitre)
num_emb_dim = len(nodeEmb_mitre[0][0]['embedding'])

print(f"Node embeddings: {num_nodes} nodes x {num_emb_dim} dimensions")


# Check if node IDs match between the embeddings and `nodes`
print(f"Embedding and Node ID Match: {len([False for x, y in zip(nodes_mitre, nodeEmb_mitre[0]) if x['id'] != y['id']]) == 0}")

# %%[markdown]
# # Apply dimensional reduction with UMAP

# %%
emb_nodes = [np.array([node['embedding'] for node in emb]) for emb in nodeEmb_mitre]


# %%
%%time

num_emb_dim_red = 2
model_umap = umap.UMAP(n_components = num_emb_dim_red, n_neighbors = 10, min_dist = 0.05, metric = 'minkowski', metric_kwds = {'p': 2.0/3.0}, random_state = 0)

emb_nodes_red = [model_umap.fit_transform(emb) for emb in emb_nodes]
emb_nodes_red = [emb - np.mean(emb, axis = 0) for emb in emb_nodes_red]

# Time: 40.1 s


# %%
%%time

def pairwise_distance(u, v, p = 1):
    return np.sum(np.abs(u - v) ** p) ** (1.0 / p)


pdist = [{(i, j): -1.0 for i in range(num_nodes) for j in range(0, i) if i != j} for emb in nodeEmb_mitre]
for p, emb in zip(pdist, emb_nodes):
    for u, v in p:
        p[(u, v)] = pairwise_distance(emb[u, :], emb[v, :], p = 2.0/3.0)


pdist_red = [{(i, j): -1.0 for i in range(num_nodes) for j in range(0, i) if i != j} for emb in nodeEmb_mitre]
for p, emb in zip(pdist_red, emb_nodes_red):
    for u, v in p:
        p[(u, v)] = pairwise_distance(emb[u, :], emb[v, :], p = 2.0/3.0)


p = u = v = emb = None
del p, u, v, emb

# time: 1 m 54 s

# %%

i = 0
m = min([v for __, v in pdist[i].items()] + [v for __, v in pdist_red[i].items()])
n = max([v for __, v in pdist[i].items()] + [v for __, v in pdist_red[i].items()])
x = 10 ** np.linspace(np.log10(m), np.log10(n), 100)

fig, ax = plt.subplots(nrows = num_embs, ncols = 1, figsize = (12, 12))

for i, (ax_, s) in enumerate(zip(ax, ['0.5', '1', '2'])):

    h, __ = np.histogram([v for k, v in pdist[i].items()], bins = x, density = False)
    h_red, __ = np.histogram([v for k, v in pdist_red[i].items()], bins = x, density = False)

    __ = ax_.bar(x[:-1], h, width = 1.0 * np.diff(x), align = 'edge', alpha = 0.5, label = f"Native {num_emb_dim} Dimensions")
    __ = ax_.bar(x[:-1], h_red, width = 1.0 * np.diff(x), align = 'edge', alpha = 0.5, label = f"{num_emb_dim_red} Dimensions")


    __ = ax_.text(0.5, 0.9, f"node2vec p = 1, q = {s}", transform = ax_.transAxes, horizontalAlignment = 'center')
    __ = plt.setp(ax_, ylabel = 'Counts', xscale = 'log', yscale = 'log')
    if i == 0:
        __ = plt.setp(ax_, title = 'Un-Normalized Pairwise Distribution - Before/After UMAP Dimensional Reduction')
        __ = ax_.legend()
    if i == (num_embs - 1):
        __ = plt.setp(ax_, xlabel = 'Minkowski Distance (p = 2/3)')
    

fig.savefig('./figures/v3/mitre_subgraph_node2vec_pdist.png', dpi = 150)

m = n = h = h_red = x = fig = ax = ax_ = None
del m, n, h, h_red, x, fig, ax, ax_

# %%
##################################

# %%
# # Compare Dimensionally Reduced Node Embeddings with Reference Layout

# %%
# Generate reference force-directed graph layout with NetworkX
coors_nx, __, __, __ = emlib.generate_nx_layout(nodes = nodes_mitre, edges = edges_mitre, layout = 'spring', layout_atts = {'k': 0.08})

# %%
# Generate edge list for plotting
map_ids_nodes = {node['id']: i for i, node in enumerate(nodes_mitre)}
edge_list = {(map_ids_nodes[edge['source_id']], map_ids_nodes[edge['target_id']]): {'weight': edge['belief']} for edge in edges_mitre}


# Label by the two obvious clusters in node-embedding space
labels = np.array([0 if x < 5.0 else 1 for x in emb_nodes_red[0][:, 0]])

labels_ = np.array([node['out_degree'] + node['in_degree'] for node in nodes_mitre])


# %%
fig, ax = plt.subplots(nrows = num_embs, ncols = 2, figsize = (18, 9 * num_embs))

for i, (emb, s) in enumerate(zip(emb_nodes_red, ['0.5', '1', '2'])):

    if i == 0:
        __ = emlib.plot_emb(coor = emb, labels = labels, marker_size = labels_, edge_list = edge_list, ax = ax[i, 0], str_title = ' ')
    else:
        __ = emlib.plot_emb(coor = -emb, labels = labels, marker_size = labels_, edge_list = edge_list, ax = ax[i, 0], str_title = ' ')

    __ = emlib.plot_emb(coor = np.array([v for __, v in coors_nx.items()]), labels = labels, marker_size = labels_, edge_list = edge_list, ax = ax[i, 1], str_title = ' ')

    __ = ax[i, 0].text(0.5, 0.9, f"node2vec p = 1, q = {s}", transform = ax[i, 0].transAxes, horizontalAlignment = 'center')
    __ = ax[i, 0].axis('off')
    __ = ax[i, 1].axis('off')

i = 0
__ = plt.setp(ax[i, 0], title = '`node2vec` Embeddings (Dim. Reduced and Clustered)')
__ = plt.setp(ax[i, 1], title = 'Reference Forced-Directed Layout (k = 0.08)')

# %%
fig.savefig('./figures/v3/mitre_subgraph_node2vec_umap.png', dpi = 150)

i = s = fig = ax = emb = None
del i, s, fig, ax, emb

# %%



