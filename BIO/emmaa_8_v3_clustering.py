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

import emmaa_lib as emlib
import importlib
# importlib.reload(emlib)

# %%
np.random.seed(0)

# %%[markdown]
# # Load Data

nodes_mitre = emlib.load_jsonl('./dist/v3/nodes_mitre.jsonl', remove_preamble = True)
edges_mitre = emlib.load_jsonl('./dist/v3/edges_mitre.jsonl', remove_preamble = True)


nodeEmb_mitre = emlib.load_jsonl('./G_mitre_p1_q1_n10len80_undirected.jsonl', remove_preamble = False)
num_nodes = len(nodeEmb_mitre)
num_dim_emb = len(nodeEmb_mitre[0]['embedding'])

print(f"Node embeddings: {num_nodes} nodes x {num_dim_emb} dimensions")


# Check if node IDs match between the embeddings and `nodes`
print(f"Embedding and Node ID Match: {len([False for x, y in zip(nodes_mitre, nodeEmb_mitre) if x['id'] != y['id']]) == 0}")

# %%[markdown]
# # Apply dimensional reduction with UMAP

# %%
emb_nodes = np.array([node['embedding'] for node in nodeEmb_mitre])


# %%
%%time

num_dim_emb_red = 2
model_umap = umap.UMAP(n_components = num_dim_emb_red, n_neighbors = 10, min_dist = 0.05, metric = 'minkowski', metric_kwds = {'p': 2.0/3.0}, random_state = 0)
emb_nodes_red = model_umap.fit_transform(emb_nodes)
emb_nodes_red = emb_nodes_red - np.mean(emb_nodes_red, axis = 0)

# Time: 16.7 s


# %%
# Generate reference force-directed graph layout with NetworkX

coors_nx, __, __, __ = emlib.generate_nx_layout(nodes = nodes_mitre, edges = edges_mitre, layout = 'spring')


# %%
# Compare dim.-reduced embeddings and reference layout

map_ids_nodes = {node['id']: i for i, node in enumerate(nodes_mitre)}
edge_list = {(map_ids_nodes[edge['source_id']], map_ids_nodes[edge['target_id']]): {'weight': edge['belief']} for edge in edges_mitre}

# Label by the two obvious clusters in node-embedding space
labels = [0 if x < -2.5 else 1 for x in emb_nodes_red[:, 0]]

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 9))

__ = emlib.plot_emb(coor = emb_nodes_red, labels = labels, edge_list = edge_list, ax = ax[0], str_title = 'Dimensionally Reduced `node2vec` Node Embeddings (p = 1, q = 1)')
__ = emlib.plot_emb(coor = np.array([v for __, v in coors_nx.items()]), labels = labels, edge_list = edge_list, ax = ax[1], str_title = 'Reference Force-Directed Layout')

fig.savefig('./figures/v3/mitre_subgraph_node2vec_umap.png', dpi = 150)

fig = ax = None
del fig, ax

# %%
%%time

def pairwise_distance(u, v, p = 1):
    # return np.sum([(u_i - v_i) ** p for u_i, v_i in zip(u, v)]) ** (1.0 / p)
    return np.sum(np.abs(u - v) ** p) ** (1.0 / p)


pdist = {(i, j): -1.0 for i in range(num_nodes) for j in range(0, i) if i != j}
for u, v in pdist:
    pdist[(u, v)] = pairwise_distance(emb_nodes[u, :], emb_nodes[v, :], p = 2.0/3.0)

pdist_red = {(i, j): -1.0 for i in range(num_nodes) for j in range(0, i) if i != j}
for u, v in pdist_red:
    pdist_red[(u, v)] = pairwise_distance(emb_nodes_red[u, :], emb_nodes_red[v, :], p = 2.0/3.0)


# time: 35 s

# %%

m = min([v for __, v in pdist.items()] + [v for __, v in pdist_red.items()])
n = max([v for __, v in pdist.items()] + [v for __, v in pdist_red.items()])
x = 10 ** np.linspace(np.log10(m), np.log10(n), 100)
h, __ = np.histogram([v for k, v in pdist.items()], bins = x, density = False)
h_red, __ = np.histogram([v for k, v in pdist_red.items()], bins = x, density = False)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
__ = ax.bar(x[:-1], h, width = 1.0 * np.diff(x), align = 'edge', alpha = 0.5, label = f"Native {num_dim_emb} Dimensions")
__ = ax.bar(x[:-1], h_red, width = 1.0 * np.diff(x), align = 'edge', alpha = 0.5, label = f"{num_dim_emb_red} Dimensions")
__ = plt.setp(ax, xlabel = 'Minkowski Distance (p = 2/3)', ylabel = 'Counts', xscale = 'log', yscale = 'log', title = 'Un-Normalized Pairwise Distribution - Before/After Dimensional Reduction')
__ = ax.legend()

fig.savefig('./figures/v3/mitre_subgraph_node2vec_pdist.png', dpi = 150)

m = n = h = h_red = x = fig = ax = None
del m, n, h, h_red, x, fig, ax

# %%







