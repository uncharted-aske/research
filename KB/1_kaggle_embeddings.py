# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Load Kaggle CORD document embeddings
# * Check overlap with EMMAA `text_refs`
# * Explore

# %%
import sys
import csv
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
import importlib
import emmaa_lib as emlib


# %%
np.random.seed(0)

# %%[markdown]
# # Load Kaggle Data

docs = []
with open('./data/kaggle/metadata.csv') as f:
    docs.extend([row for row in csv.DictReader(f)])

num_docs = len(docs)
map_uids_docs = {doc['cord_uid']: i for i, doc in enumerate(docs)}

ind = [num_docs for i in range(num_docs)]
embs = [[] for i in range(num_docs)]
with open('./data/kaggle/cord_19_embeddings_2020-12-13.csv') as f:
    for i, row in enumerate(csv.reader(f)):
        ind[i] = map_uids_docs[row[0]]
        embs[i] = list(map(float, row[1:]))

embs = np.array(embs)
num_dims = embs.shape[1]


f = i = ind = row = doc = None
del f, i, ind, row, doc

# %%
print(f"Number of Docs: {num_docs}")
print(f"Number of Embedding Dimensions: {num_dims}")
print(f"Document Metadata Keys:")
__ = [print(f"\t{k}") for k in list(docs[0].keys())]

# Number of Docs: 381817
# Number of Embedding Dimensions: 768
# Document Metadata Keys:
# 	cord_uid
# 	sha
# 	source_x
# 	title
# 	doi
# 	pmcid
# 	pubmed_id
# 	license
# 	abstract
# 	publish_time
# 	authors
# 	journal
# 	mag_id
# 	who_covidence_id
# 	arxiv_id
# 	pdf_json_files
# 	pmc_json_files
# 	url
# 	s2_id

# %%[markdown]
# # Apply Dimensional Reduction

# %%
%%time

num_dims_red = 2
model_umap = umap.UMAP(n_components = num_dims_red, n_neighbors = 10, min_dist = 0.05, metric = 'minkowski', metric_kwds = {'p': 2.0/3.0}, random_state = 0)

embs_red = model_umap.fit_transform(embs)
embs_red = embs_red - np.mean(embs_red, axis = 0)

# Time: 9 m 22 s

# %%
with open('./dist/kaggle/embeddings_umap.pkl', 'wb') as f:
    pickle.dump(embs_red, f)

# %%
if False:
    with open('./dist/kaggle/embeddings_umap.pkl', 'rb') as f:
        embs_red = pickle.load(f)

# %%
# Plot result

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = emlib.plot_emb(coor = embs_red, cmap_name = 'qual', legend_kwargs = {}, colorbar = False, str_title = 'Dimensionally Reduced Document Embeddings (SPECTER) of the Kaggle Covid-19 Dateset', ax = ax)
__ = plt.setp(ax, xlabel = 'x', ylabel = 'y')

fig.savefig('./figures/kaggle/embeddings_umap.png', dpi = 150)

# %%[markdown]
# # Apply Hierarchical Clustering

# %%
%%time

# Generate cluster labels
kwargs = {'metric': 'euclidean', 'min_cluster_size': 2, 'min_samples': 3, 'cluster_selection_epsilon': 0.2}
clusterer = hdbscan.HDBSCAN(**kwargs)
clusterer.fit(embs_red)
labels = clusterer.labels_
cluster_probs = clusterer.probabilities_
outlier_scores = clusterer.outlier_scores_
cluster_persist = clusterer.cluster_persistence_

print(f'Number of clusters: {len(np.unique(labels)):d}')
print(f'Number of unclustered points: {sum(labels == -1):d} (of {len(labels):d})')

kwargs = clusterer = None
del kwargs, clusterer

# Time: 1 m 10 s

# %%[markdown]
# Plot result

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = emlib.plot_emb(coor = embs_red, labels = labels, cmap_name = 'qual', legend_kwargs = {}, colorbar = False, str_title = 'Dimensionally Reduced Document Embeddings (SPECTER) of the Kaggle Covid-19 Dateset', ax = ax)
__ = plt.setp(ax, xlabel = 'x', ylabel = 'y')

fig.savefig('./figures/kaggle/embeddings_umap_hdbscan.png', dpi = 150)

fig = ax = None
del fig, ax

# %%[markdown]
# # Generate Node/Edge List

# %%
%%time

nodes, edges, G = emlib.generate_nn_graph(coors = embs[:2000, :], metadata = docs[:2000], model_id = 0)

# time: 1 m 3 s

# %%[markdown]
# # Generate kNN layout

# %%
%%time

embs_knn, __, __, __ = emlib.generate_nx_layout(G = G, layout = 'spring', layout_atts = {'k': 0.01}, plot = False)

# time: 16.5 s

# %%
with open('./dist/kaggle/embeddings_knn.pkl', 'wb') as f:
    pickle.dump(embs_knn, f)

# %%
if False:
    with open('./dist/kaggle/embeddings_knn.pkl', 'rb') as f:
        embs_knn = pickle.load(f)

# %%[markdown]
# Plot result

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = emlib.plot_emb(coor = np.array([v for __, v in embs_knn.items()]), labels = labels[:2000], cmap_name = 'qual', edge_list = G.edges(data = False), legend_kwargs = {}, colorbar = False, marker_size = 2.0, ax = ax, str_title = 'kNN Graph of the Document Embeddings of the Kaggle Covid-19 Dataset')
__ = plt.setp(ax, xlabel = 'x', ylabel = 'y')

fig.savefig('./figures/kaggle/embeddings_knn.png', dpi = 150)

fig = ax = None
del fig, ax


# %%
%%time

# Save `nodes`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique node ID that is referenced by other files',
    'name': '<str> unique human-interpretable name of this node (from the `title` attribute in `metadata.csv`)',
    'db_refs': '<dict> database references of this node (`doi`, `pmcid`, `pubmed_id` attributes in `metadata.csv`)',
    'grounded': '<bool> whether this node is grounded to any database (`True` for all)',
    'edge_ids_source': '<list of int> ID of edges that have this node as a source',
    'edge_ids_target': '<list of int> ID of edges that have this node as a target',
    'out_degree': '<int> out-degree of this node',
    'in_degree': '<int> in-degree of this node', 
}
emlib.save_jsonl(nodes, './dist/kaggle/nodes.jsonl', preamble = preamble)


# Save `edges`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique edge ID that is referenced by other files',
    'type': '<str> type of this edge (`knn` = k nearest neighbour)',
    'belief': '<float> belief score of this edge (= `1 / d` where `d` is the KNN distance)',
    'statement_id': '<str> unique statement id (`None` for all)',
    'source_id': '<int> ID of the source node (as defined in `nodes.jsonl`)' ,
    'target_id': '<int> ID of the target node (as defined in `nodes.jsonl`)',
    'tested': '<bool> whether this edge is tested (`True` for all)'
}
emlib.save_jsonl(edges, './dist/kaggle/edges.jsonl', preamble = preamble)

# time: 

# %%

