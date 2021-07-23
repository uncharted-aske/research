# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Load Wisconsin's `doc2vec` document embeddings (from `https://xdd.wisc.edu/app_output/xdd-covid-19-8Dec-doc2vec.zip`)
# * Check overlap with EMMAA Covid-19 corpus
# * Reduce dimensionality and cluster
# * Output node and edge lists


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
import sklearn as skl
import importlib
import pathlib
import requests

import emmaa_lib as emlib

import gensim as gs

# %%
np.random.seed(0)

# %%[markdown]
# # Load Data

# %%
# Embeddings
embs_wisconsin = np.load('/home/nliu/projects/aske/research/KB/data/wisconsin/xdd-covid-19-8Dec-doc2vec/model_streamed_doc2vec.docvecs.vectors_docs.npy')

# %%
# Doc metadata
with open('/home/nliu/projects/aske/research/KB/data/wisconsin/xdd-covid-19-8Dec-doc2vec/xdd-covid-19-8Dec.bibjson', 'r') as f:
    docs_wisconsin = json.load(f)


# Map xDD IDs to DOIs
# (Not all xDD docs have DOIs)
with open('/home/nliu/projects/aske/research/KB/data/wisconsin/xdd-covid-19-8Dec-doc2vec/xdd-covid-19-8Dec-xddid-doi', 'r') as f:
    map_xddids_dois = {line[0]: line[1].upper() for line in csv.reader(f, delimiter = '\t')}


# `gensim` model
model_wisconsin = gs.models.doc2vec.Doc2Vec.load('/home/nliu/projects/aske/research/KB/data/wisconsin/xdd-covid-19-8Dec-doc2vec/model_streamed_doc2vec')
model_wisconsin_xddids = list(model_wisconsin.docvecs.doctags.keys())
model_wisconsin_dois = [map_xddids_dois[xddid] if xddid in map_xddids_dois.keys() else None for xddid in model_wisconsin_xddids]


# Re-order the doc metadata according to the model tags
map_xddids_inds = {xddid: i for i, xddid in enumerate(model_wisconsin_xddids)}
docs_wisconsin_reordered = sorted(docs_wisconsin, key = lambda doc: map_xddids_inds[doc['identifier'][0]['id']])


print(f"There are {len(docs_wisconsin_reordered)} documents in the Wisconsin corpus.")
print(f"There are {len(docs_wisconsin_reordered) - len(map_xddids_dois)} documents that do not have a DOI.")
# There are 104997 documents in the Wisconsin corpus.
# There are 26 documents that do not have a DOI.


f = docs_wisconsin = map_xddids_inds = None
del f, docs_wisconsin, map_xddids_inds

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
dois_wisconsin = set(model_wisconsin_dois) - {None}


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
# model_umap = umap.UMAP(n_components = num_dims_red, n_neighbors = 4, min_dist = 0.05, metric = 'minkowski', metric_kwds = {'p': 2.0/3.0}, random_state = 0)
model_umap = umap.UMAP(n_components = num_dims_red, n_neighbors = 10, min_dist = 0.05, metric = 'cosine', random_state = 0)
# model_umap = umap.UMAP(densmap = True, n_components = num_dims_red, n_neighbors = 10, min_dist = 0.1, metric = 'cosine', random_state = 0)

embs_wisconsin_2d = model_umap.fit_transform(embs_wisconsin)
embs_wisconsin_2d = embs_wisconsin_2d - np.mean(embs_wisconsin_2d, axis = 0)

model_umap = None
del model_umap

# Time: 2 m 19 s

# %%
with open('./dist/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d.pkl', 'wb') as f:
    pickle.dump(embs_wisconsin_2d, f)

# %%
if False:
    with open('./dist/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d.pkl', 'rb') as f:
        embs_wisconsin_2d = pickle.load(f)


# %% 
# Label by overlap with EMMAA Covid-19 corpus 
labels_emmaa = np.array([True if doi in dois_emmaa else False for doi in model_wisconsin_dois])

# %%
# Plot result
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = emlib.plot_emb(
    coor = embs_wisconsin_2d, 
    # labels = labels_emmaa, 
    marker_size = 0.5, marker_alpha = 0.1, 
    cmap_name = 'qual', legend_kwargs = {'loc': 'lower left'}, colorbar = False, 
    str_title = 'Dimensionally Reduced Document Embeddings (Doc2Vec) of the Wisconsin Covid-19 Corpus', ax = ax)
__ = plt.setp(ax, xlabel = 'x', ylabel = 'y')

fig.savefig('./figures/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_emmaa.png', dpi = 150)


fig = ax = x = None
del fig, ax, x

# %%
# Label by overlap with Mitre doc
doi_doc = '10.1016/j.immuni.2020.04.003'.upper()
labels_doc = np.array([True if doi == doi_doc else False for doi in model_wisconsin_dois])

# %%
# Plot result
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = emlib.plot_emb(coor = embs_wisconsin_2d, labels = labels_doc, cmap_name = 'qual', legend_kwargs = {'loc': 'lower left'}, colorbar = False, str_title = 'Dimensionally Reduced Document Embeddings (Doc2Vec) of the Wisconsin Covid-19 Corpus', marker_size = 10, ax = ax)
__ = plt.setp(ax, xlabel = 'x', ylabel = 'y')

fig.savefig('./figures/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_doc.png', dpi = 150)

fig = ax = x = None
del fig, ax, x

# %%
# Label by overlap with EPI doc and its neighbourhood
doi_epi = '10.1101/2020.07.06.20147868'.upper()
response = requests.get(f"https://xdd.wisc.edu/sets/xdd-covid-19/doc2vec/api/similar?doi={doi_epi}").json()
dois_epi = [id['id'].upper() for doc in response['data'] for id in doc['bibjson']['identifier'] if id['type'] == 'doi']

print(f"{len(dois_epi)} documents found by Cosmos/xDD to be in the neighbourhood of {doi_epi}.")
# 10 documents found by Cosmos/xDD to be in the neighbourhood of 10.1101/2020.07.06.20147868.

labels_epi = np.array([1 if doi in set(dois_epi) else 2 if doi == doi_epi else 0 for doi in model_wisconsin_dois])


response = x = None
del response, x

# %%
# Plot results
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = emlib.plot_emb(coor = embs_wisconsin_2d, labels = labels_epi, cmap_name = 'qual', legend_kwargs = {'loc': 'lower left'}, colorbar = False, str_title = 'Dimensionally Reduced Document Embeddings (Doc2Vec) of the Wisconsin Covid-19 Corpus', marker_size = 10, ax = ax)
__ = plt.setp(ax, xlabel = 'x', ylabel = 'y')

fig.savefig('./figures/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_epi.png', dpi = 150)


fig = ax = x = None
del fig, ax, x


# %%[markdown]
# # Apply Hierarchical Clustering (2D)

# %%
%%time

epsilons = [0.1, 0.065, 0.05, 0.01]
labels_clusters_2d = []
for eps in epsilons:

    # Generate cluster labels
    cluster_eps = eps
    kwargs = {'prediction_data': True, 'metric': 'euclidean', 'min_cluster_size': 10, 'min_samples': 10, 'cluster_selection_epsilon': cluster_eps}
    clusterer = hdbscan.HDBSCAN(**kwargs)
    clusterer.fit(embs_wisconsin_2d)
    labels_clusters_2d.append(clusterer.labels_)
    # memberships = hdbscan.all_points_membership_vectors(clusterer)
    # cluster_probs = clusterer.probabilities_
    # outlier_scores = clusterer.outlier_scores_
    # cluster_persist = clusterer.cluster_persistence_


    print(f'\nNumber of clusters: {len(np.unique(labels_clusters_2d[-1])):d}')
    print(f'Number of unclustered points: {sum(labels_clusters_2d[-1] == -1) / len(labels_clusters_2d[-1]) * 100:.3f} %')
    # Number of clusters: 1902
    # Number of unclustered points: 4.770%


kwargs = clusterer = None
del kwargs, clusterer

# Time: 31.7 s

# %%
with open('./dist/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_clusters.pkl', 'wb') as f:
    pickle.dump(labels_clusters_2d, f)

# %%
if False:
    with open('./dist/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_clusters.pkl', 'rb') as f:
        labels_clusters_2d = pickle.load(f)

# %%[markdown]
# Generate centroid list from cluster labels

knn_ind = emlib.generate_nn_cluster_centroid_list(coors = embs_wisconsin_2d, labels = labels_clusters_2d, p = 2)[0]

# %%[markdown]
# Plot result

N = 50000
num_docs = len(docs_wisconsin_reordered)
mask = np.random.randint(0, high = num_docs, size = N)

fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (16, 16))

for i, (eps, x) in tqdm(enumerate(zip(epsilons, fig.axes)), total = len(epsilons)):

    mask = labels_clusters_2d[i] != -1

    __ = emlib.plot_emb(
        coor = embs_wisconsin_2d[mask , :], 
        labels = labels_clusters_2d[i][mask], 
        cmap_name = 'qual', legend_kwargs = {}, colorbar = False, 
        str_title = f"Epsilon = {eps}, Number of Clusters = {len(np.unique(labels_clusters_2d[i]))}", 
        marker_size = 0.5, marker_alpha = 0.1, 
        ax = x)
    # __ = ax.scatter(embs_wisconsin_2d[knn_ind, 0], embs_wisconsin_2d[knn_ind, 1], marker = '+', alpha = 0.5, color = 'k', zorder = 101)
    __ = plt.setp(x, xlabel = 'x', ylabel = 'y')

# fig.savefig('./figures/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_clusters.png', dpi = 150)
fig.savefig('./figures/wisconsin/xdd-covid-19-8Dec-doc2vec/embeddings_2d_clusters_noNoise.png', dpi = 150)

fig = ax = None
del fig, ax

# %%[markdown]
# # Generate Node and Cluster Lists (2D)

# %%
%%time

labels_ = np.array(labels_clusters_2d).transpose()

for doc in docs_wisconsin_reordered:
    if isinstance(doc['journal'], str) == False:
        doc['journal'] = doc['journal']['name']

nodes, nodeLayout, nodeAtts, groups = emlib.generate_top2vec_nodelist(
    docs = docs_wisconsin_reordered, 
    embs = embs_wisconsin_2d, 
    labels = labels_, 
    model_id = None
)

# Time: 43.9 s

# %%
print(f"Number of nodes: {len(nodes)}")
print(f"Number of groups: {len(groups)}")

# Number of nodes: 104997
# Number of groups: 1907

# %%

%%time

dist_dir = './dist/wisconsin/xdd-covid-19-8Dec-doc2vec/v4.0_nonTop2Vec/'
for x, y in zip(('nodes', 'nodeLayout', 'nodeAtts', 'groups'), (nodes, nodeLayout, nodeAtts, groups)):

    emlib.save_jsonl(y, f'{dist_dir}{x}.jsonl', preamble = emlib.get_obj_preamble(obj_type = x))

# %%
%%time

data = {
    'nodes': nodes,
    'nodeLayout': nodeLayout,
    'nodeAtts': nodeAtts,
    'groups': groups,
}

s3_url = 'http://10.64.18.171:9000'
s3_bucket = 'aske'
s3_path = 'research/KB' + dist_dir[1:(len(dist_dir) - 1)]

for x in tqdm(('nodes', 'nodeLayout', 'nodeAtts', 'groups')):

    emlib.load_obj_to_s3(
        obj = data, 
        s3_url = s3_url, 
        s3_bucket = s3_bucket, 
        s3_path = f"{s3_path}/{x}.jsonl", 
        preamble = emlib.get_obj_preamble(obj_type = x),
        obj_key = x
    )



# %%
# %%time

# model_id = None
# x = ['doc', 'epi', 'emmaa', 'clusters']

# # for labels, name in zip([labels_doc, labels_emmaa, labels_clusters[0]], x):
# for labels, name in zip([labels_doc, labels_epi, labels_emmaa, labels_clusters_2d], x):

#     # Ensure integer type in case of boolean labels
#     labels = labels.astype('int')

#     # Generate node lists
#     nodes, nodeLayout, nodeAtts = emlib.generate_nodelist_bibjson(model_id = model_id, node_metadata = docs_wisconsin_reordered, node_coors = embs_wisconsin_2d, node_labels = labels)

#     # Save node lists
#     preamble = {
#         'model_id': '<int> unique model ID that is present in all related distribution files',
#         'id': '<int> unique node ID that is referenced by other files',
#         'name': '<str> unique human-interpretable name of this node (from the `title` attribute in `metadata.csv`)',
#         'db_refs': '<dict> database references of this node (`doi`, `pmcid`, `pubmed_id` attributes in `metadata.csv`)',
#         'grounded': '<bool> whether this node is grounded to any database (`True` for all)',
#         'edge_ids_source': '<list of int> ID of edges that have this node as a source',
#         'edge_ids_target': '<list of int> ID of edges that have this node as a target',
#         'out_degree': '<int> out-degree of this node',
#         'in_degree': '<int> in-degree of this node' 
#     }
#     emlib.save_jsonl(nodes, './dist/wisconsin/xdd-covid-19-8Dec-doc2vec/' + name + '/nodes.jsonl', preamble = preamble)

#     preamble = {
#         'model_id': '<int> unique model ID that is present in all related distribution files',
#         'id': '<int> unique node ID that is defined in `nodes.jsonl`',
#         'x': '<float> position of the node in the graph layout',
#         'y': '<float> position of the node in the graph layout',
#         'z': '<float> position of the node in the graph layout'
#     }
#     emlib.save_jsonl(nodeLayout, './dist/wisconsin/xdd-covid-19-8Dec-doc2vec/' + name + '/nodeLayout.jsonl', preamble = preamble)

#     preamble = {
#         'model_id': '<int> unique model ID that is present in all related distribution files',
#         'id': '<int> unique node ID that is defined in `nodes.jsonl`',
#         'db_ref_priority': '<str> database reference from `db_refs` of `nodes.jsonl`, that is used by the INDRA ontology v1.5', 
#         'grounded_onto': '<bool> whether this model node is grounded to something that exists within the ontology', 
#         'ontocat_level': '<int> the level of the most fine-grained ontology node/category to which this model node was mapped (`-1` if not mappable, `0` if root)', 
#         'ontocat_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)', 
#         'grounded_cluster': '<bool> whether this model node is grounded to any cluster', 
#         'cluster_level': '<int> the level of the most fine-grained cluster at which this model node was mapped (`-1` if not mappable, `0` if root)', 
#         'cluster_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)' 
#     }
#     emlib.save_jsonl(nodeAtts, './dist/wisconsin/xdd-covid-19-8Dec-doc2vec/' + name + '/nodeAtts.jsonl', preamble = preamble)


#     # Generate cluster list
#     labels_unique, labels_unique_counts = np.unique(labels, return_counts = True)
#     labels_unique = labels_unique.astype('int')
#     labels_unique_counts = labels_unique_counts.astype('int')

#     clusters = [{
#         'model_id': model_id,
#         'id': int(cluster_id),
#         'ref': None,
#         'name': None,
#         'size': int(cluster_size),
#         'level': 0,
#         'parent_id': None,
#         'children_ids': None,
#         'node_ids': [i for i, l in enumerate(labels) if l == cluster_id],
#         'node_ids_direct': [i for i, l in enumerate(labels) if l == cluster_id],
#         'hyperedge_ids': []
#     } for cluster_id, cluster_size in zip(labels_unique, labels_unique_counts)]


#     # Get centroids
#     knn_ind, __, coors_centroid = emlib.generate_nn_cluster_centroid_list(coors = embs_wisconsin_2d, labels = labels, p = 2)

#     # Get centroid name
#     for i, __  in enumerate(labels_unique):
#         clusters[i]['name'] = nodes[knn_ind[i]]['name']

#     # Save cluster metadata
#     preamble = {
#         'model_id': '<int> unique model ID that is present in all related distribution files',
#         'id': '<int> unique ID for this cluster that is referenced by other files',
#         'ref': '<str> None',
#         'name': '<str> name of this cluster (taken as the name of the centroid node)',
#         'size': '<int> number of model nodes that were mapped to this cluster and its children',
#         'level': '<int> number of hops to reach the local root (`0` if root)',
#         'parent_id': '<int> ID of the parent of this cluster in the ontology',
#         'children_ids': '<array of int> unordered list of the child cluster IDs',
#         'node_ids': '<array of int> unordered list of IDs from model nodes in the membership of this cluster',
#         'node_ids_direct': '<array of int> node_ids but only model nodes which were directly mapped to this cluster and not any of the child categories',
#         'hyperedge_ids': '<array of int> unordered list of hyperedge IDs (see `hyperedges.jsonl`) that are within this cluster',
#     }
#     emlib.save_jsonl(clusters, './dist/wisconsin/xdd-covid-19-8Dec-doc2vec/' + name + '/clusters.jsonl', preamble = preamble)


#     # Generate cluster layout
#     clusterLayout = [{
#         'model_id': model_id,
#         'id': int(cluster_id),
#         'x': coors_centroid[k, 0],
#         'y': coors_centroid[k, 1],
#         'z': 0.0,

#     } for k, cluster_id in enumerate(labels_unique)]

#     # Save cluster layout
#     preamble = {
#         "model_id": "<int> unique model ID that is present in all related distribution files", 
#         "id": "<int> unique node ID that is defined in `nodes.jsonl`", 
#         "x": "<float> position of the node in the graph layout", 
#         "y": "<float> position of the node in the graph layout", 
#         "z": "<float> position of the node in the graph layout"
#     }
#     emlib.save_jsonl(clusterLayout, './dist/wisconsin/xdd-covid-19-8Dec-doc2vec/' + name + '/clusterLayout.jsonl', preamble = preamble)



# i = x = node = nodeLayout = nodeAtts = name = labels = labels_unique = clusters = clusterLayout = knn_ind = None
# del i, x, node, nodeLayout, nodeAtts, name, labels, labels_unique, clusters, clusterLayout, knn_ind


# # time: 1 m 36 s


# %%
# Sanity check
#
# Check if every cluster name is a name of a member node

map_ids_nodes = {node['id']: i for i, node in enumerate(nodes)}

x = [None for cluster in clusters]

for i, cluster in enumerate(clusters):

    x[i] = cluster['name'] in [nodes[map_ids_nodes[id]]['name'] for id in cluster['node_ids']]

# %%


[(i, d) for i, d in enumerate(docs_wisconsin) if 'Selling Luxury Brands Online' in d['title']]

[(i, d) for i, d in enumerate(docs_wisconsin) if 'uxury' in d['title']]


l = np.array([1 if 'Selling Luxury Brands Online' in d['title'] else 0 for d in docs_wisconsin])

__ = emlib.plot_emb(coor = embs_wisconsin_2d, labels = l, cmap_name = 'qual', legend_kwargs = {}, colorbar = False, xlim = (-1, 1), ylim = (-1, 1), marker_size = 20)

# %%
l = np.array([1 if 'Selling Luxury Brands Online' in d['title'] else 0 for d in docs_wisconsin])
l_ = np.flatnonzero(l)
knn = skl.neighbors.NearestNeighbors(n_neighbors = 10, metric = 'minkowski', p = 2)
knn.fit(embs_wisconsin_2d)
k = knn.kneighbors(embs_wisconsin_2d[l_, :], return_distance = False)
k = k.squeeze()


__ = emlib.plot_emb(coor = embs_wisconsin_2d, labels = np.array([1 if i in k else 0 for i, d in enumerate(docs_wisconsin)])
, cmap_name = 'qual', legend_kwargs = {}, colorbar = False, marker_size = 20)


# %%
# Check if these xddids match with those from `docs_wisconsin`
[i for i, j in zip(docs_wisconsin_xddids, x) if i != j]

# %%[markdown]
# # Test AlignedUMAP

# %%
%%time

N = 9
num_docs = embs_wisconsin.shape[0]
min_dist = np.linspace(0.01, 0.5, N)
constant_relations = [{i:i for i in range(num_docs)} for i in range(N - 1)]

M = 5000
mask = np.random.randint(0, num_docs, M)

X = umap.AlignedUMAP(
    n_neighbors = 10,
    min_dist = min_dist,
    alignment_window_size = 2,
    alignment_regularisation = 1e-3
).fit([embs_wisconsin[mask, :] for __ in range(N)], relations = constant_relations)

# %%
from IPython.display import display, Image, HTML
from matplotlib import animation
import scipy as sp
import pandas as pd

def axis_bounds(embedding):
    left, right = embedding.T[0].min(), embedding.T[0].max()
    bottom, top = embedding.T[1].min(), embedding.T[1].max()
    adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
    return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]

# %%


ax_bound = axis_bounds(np.vstack(X.embeddings_))

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
h = ax.scatter([], [], s = 2)
text = ax.text(ax_bound[0] + 0.5, ax_bound[2] + 0.5, '')
__ = plt.setp(h, array = labels_[:, 0], cmap = 'Spectral', )
__ = plt.setp(ax, xticks = [], yticks = [])


n_embeddings = len(X.embeddings_)
es = X.embeddings_
embedding_df = pd.DataFrame(np.vstack(es), columns=('x', 'y'))
embedding_df['z'] = np.repeat(np.linspace(0, 1.0, n_embeddings), es[0].shape[0])
embedding_df['id'] = np.tile(np.arange(es[0].shape[0]), n_embeddings)
embedding_df['digit'] = np.tile(labels_[:, 0], n_embeddings)
fx = sp.interpolate.interp1d(
    embedding_df.z[embedding_df.id == 0], embedding_df.x.values.reshape(n_embeddings, labels_[:, 0].shape[0]).T, kind="cubic"
)
fy = sp.interpolate.interp1d(
    embedding_df.z[embedding_df.id == 0], embedding_df.y.values.reshape(n_embeddings, labels_[:, 0].shape[0]).T, kind="cubic"
)
z = np.linspace(0, 1.0, 100)
interpolated_traces = [fx(z), fy(z)]

offsets = np.array(interpolated_traces).T
num_frames = offsets.shape[0]

def animate(i):
    h.set_offsets(offsets[i])
    text.set_text(f'Frame {i}')
    return scat

anim = animation.FuncAnimation(
    fig,
    init_func = None,
    func = animate,
    frames = num_frames,
    interval = 40)

anim.save('aligned_umap_anim.gif', writer = 'pillow')
plt.close(anim._fig)

with open('aligned_umap_anim.gif', 'rb') as f:
    display(Image(f.read()))
