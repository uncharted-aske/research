# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Load Wisconsin's `doc2vec` document embeddings (from `https://xdd.wisc.edu/app_output/xdd-covid-19-8Dec-doc2vec.zip`)
# * Check overlap with EMMAA Covid-19 corpus
# * 


# %%
import sys
import csv
import json
import pickle
import time
from tqdm import tqdm
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import umap
import hdbscan
import importlib
import emmaa_lib as emlib

import gensim as gs

# %%
np.random.seed(0)

# %%[markdown]
# # Load Data

# %%
# Embeddings
embs_wisconsin = np.load('/home/nliu/projects/aske/research/KB/data/wisconsin/xdd-covid-19-8Dec-doc2vec/model_streamed_doc2vec.docvecs.vectors_docs.npy')

# Doc metadata
with open('/home/nliu/projects/aske/research/KB/data/wisconsin/xdd-covid-19-8Dec-doc2vec/xdd-covid-19-8Dec.bibjson', 'r') as f:
    docs_wisconsin = json.load(f)

# Model
# model_wisconsin = gs.models.doc2vec.Doc2Vec.load('/home/nliu/projects/aske/research/KB/data/wisconsin/xdd-covid-19-8Dec-doc2vec/model_streamed_doc2vec')
# docs_wisconsin_xddids = list(model_wisconsin.docvecs.doctags.keys())


print(f"There are {len(docs_wisconsin)} documents in the Wisconsin corpus.")
# There are 104997 documents in the Wisconsin corpus.

f = None
del f

# %%[markdown]
# Note: 
# > The 105k documents is the set of fulltext holdings within xDD that are covid-related. 
# This means they mention related terms (covid-19, coronavirus, MERS-CoV, etc) or they appear in the CORD-19 set. 
# It is not one-to-one with the most recent EMMAA corpus. 
# At the moment we don't include abstract-only content in this set, 
# but we're actively working on expanding to include them"

# %%[markdown]
# # Load Kaggle and EMMAA Covid-19 Corpora

docs_kaggle = []
with open('/home/nliu/projects/aske/research/KB/data/kaggle/metadata.csv') as f:
    docs_kaggle.extend([row for row in csv.DictReader(f)])


docs_emmaa = emlib.load_jsonl('/home/nliu/projects/aske/research/BIO/dist/v3.1/full/documents.jsonl', remove_preamble = True)

# %%[markdown]
# # Check Overlap between the Corpora

# %%
dois_kaggle = set([doc['doi'] for doc in docs_kaggle])
dois_emmaa = set([doc['DOI'] for doc in docs_emmaa])
dois_wisconsin = set([doc['identifier'][1]['id'].upper() if len(doc['identifier']) > 1 else None for doc in docs_wisconsin])


print(f"There are {len(dois_kaggle)}, {len(dois_wisconsin)}, {len(dois_emmaa)} unique docs in the Kaggle CORD-19, Wisconsin Covid-19, EMMAA Covid-19 corpora respectively.")
print(f"{len(dois_emmaa - dois_kaggle) / len(dois_emmaa) * 100:.0f}% of the EMMAA corpus is missing from the Kaggle one.")
print(f"{len(dois_emmaa - dois_wisconsin) / len(dois_emmaa) * 100:.0f}% of the EMMAA corpus is missing from the Wisconsin one.")

# There are 220839, 104940, 85959 unique docs in the Kaggle CORD-19, Wisconsin Covid-19, EMMAA Covid-19 respectively.
# 87% of the EMMAA corpus is missing from the Kaggle one.
# 72% of the EMMAA corpus is missing from the Wisconsin one.

# %%[markdown]
# # Apply Dimensional Reduction

# %%
%%time

num_dims_red = 2
model_umap = umap.UMAP(n_components = num_dims_red, n_neighbors = 4, min_dist = 0.05, metric = 'minkowski', metric_kwds = {'p': 2.0/3.0}, random_state = 0)

embs_wisconsin_red = model_umap.fit_transform(embs_wisconsin)
embs_wisconsin_red = embs_wisconsin_red - np.mean(embs_wisconsin_red, axis = 0)


embs_wisconsin = None
del embs_wisconsin

# Time: 2 m 19 s

# %%
with open('./dist/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d.pkl', 'wb') as f:
    pickle.dump(embs_wisconsin_red, f)

# %%
if False:
    with open('./dist/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d.pkl', 'rb') as f:
        embs_wisconsin_red = pickle.load(f)

# %% 
# Label by overlap with EMMAA Covid-19 corpus 
x = [doc['identifier'][1]['id'].upper() if len(doc['identifier']) > 1 else None for doc in docs_wisconsin]
labels_emmaa = np.array([True if i in dois_emmaa else False for i in x])

# Plot result
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = emlib.plot_emb(coor = embs_wisconsin_red, labels = labels_emmaa, cmap_name = 'qual', legend_kwargs = {'loc': 'lower left'}, colorbar = False, str_title = 'Dimensionally Reduced Document Embeddings (Doc2Vec) of the Wisconsin Covid-19 Corpus', ax = ax)
__ = plt.setp(ax, xlabel = 'x', ylabel = 'y')
fig.savefig('./figures/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_emmaa.png', dpi = 150)


fig = ax = x = None
del fig, ax, x

# %%
# Label by overlap with Mitre doc
doi_doc = '10.1016/j.immuni.2020.04.003'.upper()
x = [doc['identifier'][1]['id'].upper() if len(doc['identifier']) > 1 else None for doc in docs_wisconsin]
labels_doc = np.array([True if i == doi_doc else False for i in x])

# Plot result
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = emlib.plot_emb(coor = embs_wisconsin_red, labels = labels_doc, cmap_name = 'qual', legend_kwargs = {'loc': 'lower left'}, colorbar = False, str_title = 'Dimensionally Reduced Document Embeddings (Doc2Vec) of the Wisconsin Covid-19 Corpus', ax = ax)
__ = plt.setp(ax, xlabel = 'x', ylabel = 'y')

fig.savefig('./figures/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_doc.png', dpi = 150)

fig = ax = x = None
del fig, ax, x


# %%[markdown]
# # Apply Hierarchical Clustering

# %%
%%time

# Generate cluster labels
cluster_eps = [0.05, 0.1, 0.15, 0.20]
labels_clusters = []

for i, eps in enumerate(tqdm(cluster_eps)):

    kwargs = {'metric': 'euclidean', 'min_cluster_size': 2, 'min_samples': 3, 'cluster_selection_epsilon': eps}
    clusterer = hdbscan.HDBSCAN(**kwargs)
    clusterer.fit(embs_wisconsin_red)
    labels_clusters.append(clusterer.labels_)
    # cluster_probs = clusterer.probabilities_
    # outlier_scores = clusterer.outlier_scores_
    # cluster_persist = clusterer.cluster_persistence_

for i, __ in enumerate(cluster_eps):

    print(f'\nNumber of clusters: {len(np.unique(labels_clusters[i])):d}')
    print(f'Number of unclustered points: {sum(labels_clusters[i] == -1) / len(labels_clusters[i]) * 100:.3f} %')

# Number of clusters: 1108
# Number of unclustered points: 4.206 %
#  
# Number of clusters: 458
# Number of unclustered points: 0.479 %
#  
# Number of clusters: 378
# Number of unclustered points: 0.118 %
#
# Number of clusters: 341
# Number of unclustered points: 0.047 %


kwargs = clusterer = eps = i = None
del kwargs, clusterer, eps, i

# Time: 31.7 s

# %%
with open('./dist/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_clusters.pkl', 'wb') as f:
    pickle.dump(labels_clusters, f)

# %%
if False:
    with open('./dist/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_clusters.pkl', 'rb') as f:
        labels_clusters = pickle.load(f)

# %%[markdown]
# Generate centroid list from cluster labels
knn_ind = [emlib.generate_nn_cluster_centroid_list(coors = embs_wisconsin_red, labels = l, p = 2)[0] for l in labels_clusters]

# %%[markdown]
# Plot result

fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12))

for i, x in enumerate(fig.axes):
    __ = emlib.plot_emb(coor = embs_wisconsin_red, labels = labels_clusters[i], cmap_name = 'qual', legend_kwargs = {}, colorbar = False, str_title = f"Cluster Epsilon = {i:.2f}", ax = x)
    __ = x.scatter(embs_wisconsin_red[knn_ind[i], :], marker = '+', color = 'k')
    __ = plt.setp(x, xlabel = 'x', ylabel = 'y')

fig.savefig('./figures/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_clusters.png', dpi = 150)

fig = ax = x = None
del fig, ax, x


# %%[markdown]
# # Generate Node and Cluster Lists

# %%
%%time

model_id = 0
x = ['doc', 'emmaa', 'clusters']

for labels, name in zip([labels_doc, labels_emmaa, labels_clusters[0]], x):

    # Ensure integer type in case of boolean labels
    labels = labels.astype('int')

    # Generate node lists
    nodes, nodeLayout, nodeAtts = emlib.generate_nodelist_bibjson(model_id = model_id, node_metadata = docs_wisconsin, node_coors = embs_wisconsin_red, node_labels = labels)

    # Save node lists
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
    emlib.save_jsonl(nodes, './dist/wisconsin/xdd-covid-19-8Dec-doc2vec/' + name + '/nodes.jsonl', preamble = preamble)

    preamble = {
        'model_id': '<int> unique model ID that is present in all related distribution files',
        'id': '<int> unique node ID that is defined in `nodes.jsonl`',
        'x': '<float> position of the node in the graph layout',
        'y': '<float> position of the node in the graph layout',
        'z': '<float> position of the node in the graph layout',
    }
    emlib.save_jsonl(nodeLayout, './dist/wisconsin/xdd-covid-19-8Dec-doc2vec/' + name + '/nodeLayout.jsonl', preamble = preamble)

    preamble = {
        'model_id': '<int> unique model ID that is present in all related distribution files',
        'id': '<int> unique node ID that is defined in `nodes.jsonl`',
        'db_ref_priority': '<str> database reference from `db_refs` of `nodes.jsonl`, that is used by the INDRA ontology v1.5', 
        'grounded_onto': '<bool> whether this model node is grounded to something that exists within the ontology', 
        'ontocat_level': '<int> the level of the most fine-grained ontology node/category to which this model node was mapped (`-1` if not mappable, `0` if root)', 
        'ontocat_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)', 
        'grounded_cluster': '<bool> whether this model node is grounded to any cluster', 
        'cluster_level': '<int> the level of the most fine-grained cluster at which this model node was mapped (`-1` if not mappable, `0` if root)', 
        'cluster_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)', 
    }
    emlib.save_jsonl(nodeAtts, './dist/wisconsin/xdd-covid-19-8Dec-doc2vec/' + name + '/nodeAtts.jsonl', preamble = preamble)


    # Generate cluster list
    labels_unique, labels_unique_counts = np.unique(labels, return_counts = True)
    labels_unique = labels_unique.astype('int')
    labels_unique_counts = labels_unique_counts.astype('int')

    clusters = [{
        'model_id': model_id,
        'id': int(cluster_id),
        'ref': None,
        'name': None,
        'size': int(cluster_size),
        'level': 0,
        'parent_id': None,
        'children_ids': None,
        'node_ids': [i for i, l in enumerate(labels) if l == cluster_id],
        'node_ids_direct': [i for i, l in enumerate(labels) if l == cluster_id],
        'hyperedge_ids': None,
    } for cluster_id, cluster_size in zip(labels_unique, labels_unique_counts)]


    # Get centroid name
    knn_ind = emlib.generate_nn_cluster_centroid_list(coors = embs_wisconsin_red, labels = labels, p = 2)[0]
    for i, __  in enumerate(labels_unique):
        clusters[i]['name'] = nodes[knn_ind[i]]['name']


    preamble = {
        'model_id': '<int> unique model ID that is present in all related distribution files',
        'id': '<int> unique ID for this cluster that is referenced by other files',
        'ref': '<str> None',
        'name': '<str> name of this cluster (taken as the name of the centroid node)',
        'size': '<int> number of model nodes that were mapped to this cluster and its children',
        'level': '<int> number of hops to reach the local root (`0` if root)',
        'parent_id': '<int> ID of the parent of this cluster in the ontology',
        'children_ids': '<array of int> unordered list of the child cluster IDs',
        'node_ids': '<array of int> unordered list of IDs from model nodes in the membership of this cluster',
        'node_ids_direct': '<array of int> node_ids but only model nodes which were directly mapped to this cluster and not any of the child categories',
        'hyperedge_ids': '<array of int> unordered list of hyperedge IDs (see `hyperedges.jsonl`) that are within this cluster',
    }
    emlib.save_jsonl(clusters, './dist/wisconsin/xdd-covid-19-8Dec-doc2vec/' + name + '/clusters.jsonl', preamble = preamble)



i = x = node = nodeLayout = nodeAtts = name = labels = labels_unique = clusters = knn_ind = None
del i, x, node, nodeLayout, nodeAtts, name, labels, labels_unique, clusters, knn_ind


# time: 1 m 36 s


# %%
