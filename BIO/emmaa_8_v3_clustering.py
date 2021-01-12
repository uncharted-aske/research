# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content: 
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
import matplotlib as mpl
import matplotlib.pyplot as plt
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
nodeEmb_mitre.append(emlib.load_jsonl('./dist/v3/node2vec/G_mitre_p1000000_q1_n10len80_directed.jsonl', remove_preamble = False))
# nodeEmb_mitre.append(emlib.load_jsonl('./dist/v3/node2vec/G_mitre_p4_q1_n10len80_undirected.jsonl', remove_preamble = False))
# nodeEmb_mitre.append(emlib.load_jsonl('./dist/v3/node2vec/G_mitre_p1_q05_n10len80_undirected.jsonl', remove_preamble = False))
# nodeEmb_mitre.append(emlib.load_jsonl('./dist/v3/node2vec/G_mitre_p1_q1_n10len80_undirected.jsonl', remove_preamble = False))
# nodeEmb_mitre.append(emlib.load_jsonl('./dist/v3/node2vec/G_mitre_p1_q1_n10len80_undirected_w2.jsonl', remove_preamble = False))
nodeEmb_mitre.append(emlib.load_jsonl('./dist/v3/node2vec/G_mitre_p1_q1_n10len80_directed.jsonl', remove_preamble = False))
# nodeEmb_mitre.append(emlib.load_jsonl('./dist/v3/node2vec/G_mitre_p1_q2_n10len80_undirected.jsonl', remove_preamble = False))
# nodeEmb_mitre.append(emlib.load_jsonl('./dist/v3/node2vec/G_mitre_p1_q4_n10len80_undirected.jsonl', remove_preamble = False))
nodeEmb_mitre.append(emlib.load_jsonl('./dist/v3/node2vec/G_mitre_p1_q1000000_n10len80_directed.jsonl', remove_preamble = False))


emb_names = [
    'node2vec p = 1e6, q = 1, directed, w = 10', 
    # 'node2vec p = 4, q = 1, undirected, w = 10', 
    # 'node2vec p = 1, q = 0.5, undirected, w = 10', 
    # 'node2vec p = 1, q = 1, undirected, w = 10', 
    # 'node2vec p = 1, q = 1, undirected, w = 2', 
    'node2vec p = 1, q = 1, directed, w = 10', 
    # 'node2vec p = 1, q = 2, undirected, w = 10', 
    # 'node2vec p = 1, q = 4, undirected, w = 10',
    'node2vec p = 1, q = 1e6, directed, w = 10', 
    ]

# 'node2vec p = 1e6, q = 1, directed, w = 10'

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

# Time: 1 m 14 s

# %%
%%time

def pairwise_distance(u, v, p = 1):
    return np.sum(np.abs(u - v) ** p) ** (1.0 / p)


pdist = [{(i, j): -1.0 for i in range(num_nodes) for j in range(0, i) if i != j} for __ in nodeEmb_mitre]
for p, emb in zip(pdist, emb_nodes):
    for u, v in p:
        p[(u, v)] = pairwise_distance(emb[u, :], emb[v, :], p = 2.0/3.0)


pdist_red = [{(i, j): -1.0 for i in range(num_nodes) for j in range(0, i) if i != j} for __ in nodeEmb_mitre]
for p, emb in zip(pdist_red, emb_nodes_red):
    for u, v in p:
        p[(u, v)] = pairwise_distance(emb[u, :], emb[v, :], p = 2.0/3.0)


p = u = v = emb = None
del p, u, v, emb

# time: 3 m 2 s

# %%

i = 0
m = min([v for __, v in pdist[i].items()] + [v for __, v in pdist_red[i].items()])
n = max([v for __, v in pdist[i].items()] + [v for __, v in pdist_red[i].items()])
x = 10 ** np.linspace(np.log10(m), np.log10(n), 100)

fig, ax = plt.subplots(nrows = num_embs, ncols = 1, figsize = (12, 30))

for i, (ax_, s) in enumerate(zip(ax, emb_names)):

    h, __ = np.histogram([v for k, v in pdist[i].items()], bins = x, density = False)
    h_red, __ = np.histogram([v for k, v in pdist_red[i].items()], bins = x, density = False)

    __ = ax_.bar(x[:-1], h, width = 1.0 * np.diff(x), align = 'edge', alpha = 0.5, label = f"Native {num_emb_dim} Dimensions")
    __ = ax_.bar(x[:-1], h_red, width = 1.0 * np.diff(x), align = 'edge', alpha = 0.5, label = f"{num_emb_dim_red} Dimensions")


    __ = ax_.text(0.5, 0.9, f"{s}", transform = ax_.transAxes, horizontalAlignment = 'center')
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
# 
# ## Change Centrality Measures

# %%
%%time

# Generate reference force-directed graph layout with NetworkX
coors_nx, __, __, __ = emlib.generate_nx_layout(nodes = nodes_mitre, edges = edges_mitre, layout = 'spring', layout_atts = {'k': 0.08})

# %%
# Generate edge list for plotting
map_ids_nodes = {node['id']: i for i, node in enumerate(nodes_mitre)}
edge_list = {(map_ids_nodes[edge['source_id']], map_ids_nodes[edge['target_id']]): {'weight': edge['belief']} for edge in edges_mitre}

# %%
%%time

# Load graph object to get graph measures
with open('./dist/v3/G_mitre.pkl', 'rb') as x:
    G_mitre = pickle.load(x)

x = None
del x


# Labels from centrality measures
labels = [dict(G_mitre.degree()),
    nx.algorithms.centrality.closeness_centrality(G_mitre), 
    # nx.algorithms.centrality.eigenvector_centrality(G_mitre),
    # nx.algorithms.centrality.katz_centrality(G_mitre),
    # nx.algorithms.centrality.current_flow_closeness_centrality(G_mitre),
    # nx.algorithms.centrality.betweenness_centrality(G_mitre),
    # nx.algorithms.centrality.communicability_betweenness_centrality(G_mitre),
    # nx.algorithms.centrality.load_centrality(G_mitre),
    # nx.algorithms.centrality.subgraph_centrality(G_mitre),
    nx.algorithms.centrality.harmonic_centrality(G_mitre),
    # nx.algorithms.centrality.percolation_centrality(G_mitre),
    # nx.algorithms.centrality.second_order_centrality(G_mitre),
    # nx.algorithms.centrality.trophic_levels(G_mitre),
    # nx.algorithms.centrality.voterank(G_mitre),
]

labels = [np.array([v for __, v in l.items()]) if isinstance(l, dict) else l for l in labels]

for i in [0]:
    labels[i] = np.log10(labels[i])

labels_name = ['Log Degree', 'Closeness', 'Harmonic']


# time: 2.99 s

# %%
fig, ax = plt.subplots(nrows = len(labels), ncols = 2, figsize = (6 * 2, 6 * len(labels)))

k = 1
emb = emb_nodes_red[k]
for i, (labels_, s) in enumerate(zip(labels, labels_name)):

    __ = emlib.plot_emb(coor = emb, labels = labels_, cmap_name = 'cool', legend_kwargs = {}, colorbar = False, marker_size = 2.0, edge_list = edge_list, ax = ax[i, 0], str_title = ' ')

    __ = emlib.plot_emb(coor = np.array([v for __, v in coors_nx.items()]), labels = labels_, cmap_name = 'cool', legend_kwargs = {}, colorbar = False, marker_size = 2.0, edge_list = edge_list, ax = ax[i, 1], str_title = ' ')

    __ = ax[i, 0].text(0.5, 0.9, f"{emb_names[k]}", transform = ax[i, 0].transAxes, horizontalAlignment = 'center')
    # __ = ax[i, 0].text(0, 0.5, f"{s} Centrality", transform = ax[i, 0].transAxes, horizontalAlignment = 'center', rotation = 'vertical')
    __ = plt.setp(ax[i, 0], xlabel = '', ylabel = f"{s} Centrality")
    __ = plt.setp(ax[i, 1], xlabel = '', ylabel = '')
    ax[i, 0].tick_params(length = 0, labelbottom = False, labelleft = False)
    ax[i, 1].tick_params(length = 0, labelbottom = False, labelleft = False)


i = 0
__ = plt.setp(ax[i, 0], title = 'node2vec Embeddings (p = 1, q = 1, UMAP)')
__ = plt.setp(ax[i, 1], title = 'Reference Forced-Directed Layout (k = 0.08)')


fig.savefig('./figures/v3/mitre_subgraph_node2vec_umap_centralities.png', dpi = 150)

i = s = fig = ax = emb = labels_ = None
del i, s, fig, ax, emb, labels_


# %%[markdown]
# ## Change in Embedding Parameters

# %%
fig, ax = plt.subplots(nrows = num_embs, ncols = 2, figsize = (6 * 2, 6 * num_embs))

for i, (emb, s) in enumerate(zip(emb_nodes_red, emb_names)):

    if i == 0:
        __ = emlib.plot_emb(coor = emb, labels = labels[1], cmap_name = 'cool', legend_kwargs = {}, colorbar = False, marker_size = 2.0, edge_list = edge_list, ax = ax[i, 0], str_title = ' ')

        # ax[i, 0].add_patch(mpl.patches.Rectangle([-6, -5], 10, 12, fill = False, edgecolor = plt.cm.get_cmap('tab10')(0), zorder = 0))
        # ax[i, 0].add_patch(mpl.patches.Rectangle([6, -5], 5, 12, fill = False, edgecolor = plt.cm.get_cmap('tab10')(1), zorder = 0))

    else:
        __ = emlib.plot_emb(coor = emb, labels = labels[1], cmap_name = 'cool', legend_kwargs = {}, colorbar = False, edge_list = edge_list, ax = ax[i, 0], str_title = ' ')

    __ = emlib.plot_emb(coor = np.array([v for __, v in coors_nx.items()]), labels = labels[1], cmap_name = 'cool', legend_kwargs = {}, colorbar = False, edge_list = edge_list, ax = ax[i, 1], str_title = ' ')

    __ = ax[i, 0].text(0.5, 0.9, f"{s}", transform = ax[i, 0].transAxes, horizontalAlignment = 'center')
    __ = plt.setp(ax[i, 0], xlabel = '', ylabel = '')
    __ = plt.setp(ax[i, 1], xlabel = '', ylabel = '')
    ax[i, 0].tick_params(length = 0, labelbottom = False, labelleft = False)
    ax[i, 1].tick_params(length = 0, labelbottom = False, labelleft = False)

i = 0
__ = plt.setp(ax[i, 0], title = 'node2vec Embeddings (UMAP)')
__ = plt.setp(ax[i, 1], title = 'Reference Forced-Directed Layout (k = 0.08)')


fig.savefig('./figures/v3/mitre_subgraph_node2vec_umap_pq.png', dpi = 150)

i = s = fig = ax = emb = None
del i, s, fig, ax, emb

# %%[markdown]
# ## Track by Clusters (UMAP Dim)

# %%
%%time

# Generate cluster labels using the (p = 1, q = 4) embeddings

# kwargs = {'metric': 'minkowski', 'p': 2.0 / 3.0, 'min_cluster_size': 2, 'min_samples': 3, 'cluster_selection_epsilon': 0.2}
kwargs = {'metric': 'euclidean', 'min_cluster_size': 2, 'min_samples': 3, 'cluster_selection_epsilon': 0.45}
clusterer = hdbscan.HDBSCAN(**kwargs)
clusterer.fit(emb_nodes_red[-1])
labels_ = clusterer.labels_
# cluster_probs = clusterer.probabilities_
# outlier_scores = clusterer.outlier_scores_
# cluster_persist = clusterer.cluster_persistence_


print(f'Number of clusters: {len(np.unique(labels_)):d}')
print(f'Number of unclustered points: {sum(labels_ == -1):d} (of {num_nodes:d})')


kwargs = clusterer = None
del kwargs, clusterer

# Time: 58.3 ms


# %%
fig, ax = plt.subplots(nrows = num_embs, ncols = 2, figsize = (6 * 2, 6 * num_embs))

for i, (emb, s) in enumerate(zip(emb_nodes_red, emb_names)):

    if i == 2:
        __ = emlib.plot_emb(coor = emb, labels = labels_, cmap_name = 'qual', legend_kwargs = {'loc': 'lower left', 'ncol': 3}, colorbar = False, marker_size = 2.0, edge_list = edge_list, ax = ax[i, 0], str_title = ' ')

    else:
        __ = emlib.plot_emb(coor = emb, labels = labels_, cmap_name = 'qual', legend_kwargs = {}, colorbar = False, edge_list = edge_list, ax = ax[i, 0], str_title = ' ')

    __ = emlib.plot_emb(coor = np.array([v for __, v in coors_nx.items()]), labels = labels_, cmap_name = 'qual', legend_kwargs = {}, colorbar = False, edge_list = edge_list, ax = ax[i, 1], str_title = ' ')

    __ = ax[i, 0].text(0.5, 0.95, f"{s}", transform = ax[i, 0].transAxes, horizontalAlignment = 'center')
    __ = plt.setp(ax[i, 0], xlabel = '', ylabel = '')
    __ = plt.setp(ax[i, 1], xlabel = '', ylabel = '')
    ax[i, 0].tick_params(length = 0, labelbottom = False, labelleft = False)
    ax[i, 1].tick_params(length = 0, labelbottom = False, labelleft = False)

i = 0
__ = plt.setp(ax[i, 0], title = 'node2vec Embeddings (UMAP & HDBSCAN)')
__ = plt.setp(ax[i, 1], title = 'Reference Forced-Directed Layout (k = 0.08)')


fig.savefig('./figures/v3/mitre_subgraph_node2vec_umap_hdbscan_pq.png', dpi = 150)

i = s = fig = ax = emb = None
del i, s, fig, ax, emb

# %%

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))

ax[0].scatter(emb_nodes_red[0][:, 0], labels[1], marker = 'o', s = 20, alpha = 0.15, label = 'p/q = 1e6')
ax[0].scatter(emb_nodes_red[2][:, 0], labels[1], marker = 'o', s = 20, alpha = 0.15, label = 'p/q = 1e-6')
__ = plt.setp(ax[0], xlabel = '1st Embedding Dimension', ylabel = 'Closeness Centrality')
__ = ax[0].legend()

ax[1].scatter(labels_, labels[1], marker = 'o', s = 50, alpha = 0.2)
__ = plt.setp(ax[1], xlabel = "Cluster Labels, Derived from 'p/q = 1e-6' Case", ylabel = 'Closeness Centrality')


fig.savefig('./figures/v3/mitre_subgraph_node2vec_umap_hdbscan_pq_comp.png', dpi = 150)

fig = ax = None
del fig, ax


# %%












