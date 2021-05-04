# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Load Kaggle CORD document embeddings
# * Dimensionally reduce
# * Hierarchically cluster over multiple epsilon values
# * Generate v4.0 `docs`, `docAtts`, `docGroups` files


# %%
from tqdm import tqdm
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

import numba
import sklearn as skl

import emmaa_lib as emlib

from typing import Dict, List, Tuple, Set, Union, Optional, NoReturn, Any


# %%
np.random.seed(0)

# %%[markdown]
# # Load Kaggle Data

docs = []
with open('./data/kaggle/metadata.csv') as f:
    docs.extend([row for row in csv.DictReader(f)])


# %%
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

# %%
with open('./dist/kaggle/embeddings.pkl', 'wb') as f:
    pickle.dump(embs, f)

# %%
if False:
    with open('./dist/kaggle/embeddings.pkl', 'rb') as f:
        embs = pickle.load(f)

# %%[markdown]
# # Apply Dimensional Reduction

# %%
%%time

model_umap = umap.UMAP(n_components = 2, n_neighbors = 7, min_dist = 0.5, metric = 'minkowski', metric_kwds = {'p': 2.0/3.0}, random_state = 0)

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

i = np.random.randint(0, high = embs.shape[0], size = 10000)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
# __ = emlib.plot_emb(coor = embs_red[i, :], cmap_name = 'qual', legend_kwargs = {}, colorbar = False, str_title = 'Dimensionally Reduced SPECTER Embeddings of the Kaggle CORD-19 Dataset', ax = ax)
__ = emlib.plot_emb(coor = embs_red, cmap_name = 'qual', marker_size = 0.5, marker_alpha = 0.01, legend_kwargs = {}, colorbar = False, str_title = 'Dimensionally Reduced SPECTER Embeddings of the Kaggle CORD-19 Dataset', ax = ax)
__ = plt.setp(ax, xlabel = 'x', ylabel = 'y', )

fig.savefig('./figures/kaggle/embeddings_umap.png', dpi = 150)

# %%[markdown]
# # Apply Hierarchical Clustering

# %%

# epsilons = np.arange(0.01, 0.041, 0.01)
epsilons = np.array([0.03, 0.04, 0.05])

labels = []
for eps in tqdm(epsilons):

    # Generate cluster labels
    kwargs = {'metric': 'euclidean', 'min_cluster_size': 50, 'min_samples': 3, 'cluster_selection_epsilon': float(eps)}
    clusterer = hdbscan.HDBSCAN(**kwargs)
    clusterer.fit(embs_red)
    l = clusterer.labels_
    labels.append(l)
    # labels = clusterer.labels_
    # cluster_probs = clusterer.probabilities_
    # outlier_scores = clusterer.outlier_scores_
    # cluster_persist = clusterer.cluster_persistence_


# Transpose to align with `embs` and reverse order to bring up ancestor group labels
labels = np.array(labels).transpose()[:, ::-1]
epsilons = epsilons[::-1]


for i, l in enumerate(labels.T):
    print(f'\nEpsilon: {float(epsilons[i]): .2f}')
    print(f'Number of clusters: {len(np.unique(l)):d}')
    print(f'Unclustered Fraction: {sum(l == -1) / len(l) * 100:.2f} %')


eps = i = l = kwargs = clusterer = None
del eps, i, l, kwargs, clusterer

# Time: 1 m 10 s


# %%
with open('./dist/kaggle/embeddings_umap_hdbscan.pkl', 'wb') as f:
    pickle.dump(labels, f)

# %%
if False:
    with open('./dist/kaggle/embeddings_umap_hdbscan.pkl', 'rb') as f:
        labels = pickle.load(f)

    # epsilons = np.arange(0.01, 0.041, 0.01)[::-1]
    epsilons = np.array([0.03, 0.04, 0.05])[::-1]

# %%[markdown]
# Plot cluster distribution

# fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 12))

# # Number of clusters/groups as a function of epsilon
# c = 'tab:blue'
# x = [np.unique(l, return_counts = True) for l in labels.T]
# __ = ax[0].plot(epsilons, [len(i[0]) for i in x], marker = 'o', color = c)
# __ = ax[0].set_ylabel('Number of Groups', color = c)
# __ = ax[0].tick_params(axis = 'y', labelcolor = c)
# # __ = ax[0].tick_params(axis = 'x', labelbottom = False)
# __ = plt.setp(ax[0], yscale = 'linear')


# # Size of clusters/groups as a function of epsilon
# c = 'tab:blue'
# y = [i[1] for i in x]
# h = ax[1].violinplot(dataset = y, positions = epsilons, widths = 0.005)
# __ = plt.setp(ax[1], xlabel = 'Epsilon, \u03B5', yscale = 'log')
# __ = ax[1].set_ylabel('Size of Groups', color = c)
# __ = ax[1].tick_params(axis = 'y', labelcolor = c)
# for i in h['bodies']:
#     i.set_facecolor(c)

# # Number of noise points
# c = 'tab:red'
# ax_ = ax[1].twinx()
# z = [sum(labels[:, i] == -1) for i in range(labels.shape[1])]
# __ = ax_.plot(epsilons, z, marker = 'o', color = c)
# __ = plt.setp(ax_, yscale = 'log')
# __ = ax_.set_ylabel('Number of Noise Points', color = c)
# __ = ax_.tick_params(axis = 'y', labelcolor = c)

# __ = plt.setp(ax_, ylim = plt.getp(ax[1], 'ylim'))
# __ = plt.setp(ax[0], xlim = plt.getp(ax[1], 'xlim'))


# fig.savefig('./figures/kaggle/embeddings_umap_hdbscan_distribution.png', dpi = 150)

# fig = ax = x = y = z = h = None
# del fig, ax, x, y, z, h


# %%
# X% quantile of cluster size
n = 0.98
labels_ = []

for i in range(len(epsilons)):

    x, y = np.unique(labels[:, i], return_counts = True)
    j = np.argsort(y)[::-1]
    x = x[j]
    y = y[j]

    m = np.quantile(y, n)
    p = sum(y >= m)

    print(f"eps = {epsilons[i]}: {n}-quantile cluster size = {m}, top {p} clusters")
    
    labels_.append([l if l in x[:p] else -1 for l in labels[:, i]])
    
labels_ = np.array(labels_).T

# eps = 0.05: 0.98-quantile cluster size = 52.0, top 66 clusters
# eps = 0.04: 0.98-quantile cluster size = 59.0, top 118 clusters
# eps = 0.03: 0.98-quantile cluster size = 54.0, top 317 clusters


# %%[markdown]
# Plot result

%%time

j = np.random.randint(0, high = embs_red.shape[0], size = 50000)

fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 15))

for i, x in enumerate(fig.axes):

    if i < labels.shape[1]:
        __ = emlib.plot_emb(coor = embs_red[j, :], labels = labels_[j, i], cmap_name = 'qual', marker_size = 1.0, marker_alpha = 0.1, legend_kwargs = {}, colorbar = False, str_title = f'\u03B5 = {epsilons[i]:.3f}', ax = x)
        __ = plt.setp(x, xlabel = '', ylabel = '')
        __ = x.tick_params(axis = 'both', bottom = False, left = False, labelbottom = False, labelleft = False)
    
    else:
        plt.axis('off')


fig.savefig('./figures/kaggle/embeddings_umap_hdbscan.png', dpi = 150)


fig = ax = x = i = j = None
del fig, ax, x, i, j


# %%
# Transform object IDs (local <-> global)
def transform_obj_ids(obj: Union[Dict, int], obj_type: str, obj_id_key: str = 'id', num_bits_global: int = 32, num_bits_namespace: int = 4, reverse: bool = False) -> Union[Dict, int]:

    # Number of bits for the local ID
    num_bits_local = num_bits_global - num_bits_namespace


    # Local ID -> globally unique ID
    if reverse == False:

        # Object type -> namespace ID
        obj_types = ('models', 'tests', 'paths', 'edges', 'evidences', 'docs', 'nodes', 'groups')
        map_type_namespace = {t: i for i, t in enumerate(obj_types)}
        id_namespace = map_type_namespace[obj_type]


        if isinstance(obj, dict):
            id_local = obj[obj_id_key]
            id_global = (id_namespace << num_bits_local) | id_local
            obj[obj_id_key] = id_global
        
        elif isinstance(obj, int):
            id_local = obj
            id_global = (id_namespace << num_bits_local) | id_local
            obj = id_global


    # globally unique ID -> local ID
    else:

        if isinstance(obj, dict):
            id_global = obj[obj_id_key]

        elif isinstance(obj, int):
            id_global = obj


        # a = 0xf0000000
        a = int((1.0 - 2.0 ** num_bits_namespace) / (1.0 - 2.0)) << num_bits_local
        id_namespace = (id_global & a) >> num_bits_local


        # b = 0x0fffffff
        b = int(2.0 ** num_bits_local - 1.0)
        id_local = (id_global & b)


        if isinstance(obj, dict):
            obj[obj_id_key] = id_local
        
        elif isinstance(obj, int):
            obj = id_local


    return obj


# Get the preamble of a given object type
def get_obj_preamble(obj_type: str) -> Dict:

    preamble = {}
    obj_types_valid = ('models', 'tests', 'paths', 'edges', 'evidences', 'docs', 'docLayout', 'nodes', 'nodeLayout', 'nodeAtts', 'groups', 'groupLayout')
    if obj_type not in obj_types_valid:
        raise ValueError(f'`{obj_type}` is not a valid object type.')


    if obj_type == 'models':
    
        preamble = {
            'id': '<int> ID of this model',
            'id_emmaa': '<str> EMMAA ID of this model, necessary for making requests on the EMMAA API',
            'name': '<str> Human-readable name of this model',
            'description': '<str> Human-readable description of this model',
            'test_ids': '<list of ints> list ofIDs of the tests against which this model has been tested by EMMAA',
            'snapshot_time': '<str> Date and UTC time (ISO 8601 format) at which the model data (statements, etc.) is requested on the EMMAA API'
        }


    if obj_type == 'tests':

        preamble = {
            'id': '<int> ID of this test (corpus)',
            'id_emmaa': '<str> EMMAA ID of this test, necessary for making requests on the EMMAA API',
            'name': '<str> Human-readable name of this test',
            'model_ids': '<list of ints> list ofIDs of the models that have been tested against this test by EMMAA',
            'snapshot_time': '<str> Date and UTC time (ISO 8601 format) at which the model data (statements, etc.) is requested on the EMMAA API'
        }


    if obj_type == 'paths':

        preamble = {
            'id': '<int> ID of this (test or explanatory) path',
            'model_id': '<int> ID of the associated model',
            'test_id': '<int> ID of the associated test (corpus)',
            'test_statement_id': '<str> EMMAA ID of the test statement that this path explains',
            'type': '<str> type of this path (`unsigned_graph`, `signed_graph`, `pybel`, etc.)',
            'edge_ids': '<int> List of IDs of the edges in this path',
            'node_ids': '<int> List of IDs of the nodes in this path'
        }


    if obj_type == 'edges':

        preamble = {
            'id': '<int> ID of this edge',
            'model_id': '<int> ID of the associated model',
            'statement_id': '<str> EMMAA ID or `matches_hash` of the INDRA statement from which this edge is derived',
            'statement_type': '<str> type of the source statement',
            'belief': '<float> belief score of the source statement',
            'evidence_ids': '<list of ints> list of IDs of the evidences that support the source statement',
            'doc_ids': '<list of ints> list of IDs of the docs that support the source statement',
            'source_node_id': '<int> ID of the source node',
            'target_node_id': '<int> ID of the target node',
            'tested': '<bool> test status of the source statement according to `paths`',
            'test_path_ids': '<list of ints> list of IDs of (test/explanatory) paths that reference this edge',
            'curated': '<int> curation status of the source statement',
            'directed': '<bool> whether this edge is directed or not (`True` = directed, `False` = undirected)',
            'polarity': '<bool> prescribed polarity of this edge (`True` = positive, `False` = negative, `None` = undefined)'
        }


    if obj_type == 'evidences':

        preamble = {
            'id': '<int> ID of this evidence',
            'model_id': '<int> ID of the associated model',
            'text': '<str> plain text of this evidence',
        }


    if obj_type == 'docs':

        preamble = {
            'id': '<int> ID of this doc',
            'model_id': '<int> ID of the associated model',
            'evidence_ids': '<list of ints> list ofIDs of the evidences that references this doc',
            'edge_ids': '<list of ints> list ofIDs of the edges that references this doc',
            'identifier': '<list of dicts> list ofexternal doc IDs (dict keys = `type`, `id`)',
        }


    if obj_type == 'docLayout':

        preamble = {
            'doc_id': '<int> ID of the associated doc',
            'coor_sys_name': '<str> name of the coordinate system (`cartesian`, `spherical`)',
            'coors': '<list of floats> list ofcoordinate values of this doc layout',
        }


    if obj_type == 'nodes':

        preamble = {
            'id': '<int> ID of this node',
            'model_id': '<int> ID of the associated model',
            'name': '<str> human-readable name of this node',
            'grounded_db': '<bool> whether this node is grounded to any database',
            'db_ids': '<list of strs> list of database `namespace:id` strings, sorted by priority in descending order',
            'edge_ids_source': '<list of ints> list of IDs of the edges to which this node is the source',
            'edge_ids_target': '<list of ints> list of IDs of the edges to which this node is the target',
            'out_degree': '<int> out-degree of this node (length of `edge_ids_source`)', 
            'in_degree': '<int> in-degree of this node (length of `edge_ids_target`)'
        }


    if obj_type == 'nodeLayout':

        preamble = {
            'node_id': '<int> ID of the node',
            'coor_sys_name': '<str> name of the coordinate system (`cartesian`, `spherical`)',
            'coors': '<list of floats> list of coordinate values of this node layout',
        }

    if obj_type == 'nodeAtts':

        preamble = {
            'node_id': '<int> ID of the node',
            # 'db_ref_priority': '<str>',
            'grounded_group': '<bool> whether this node is grounded to the given ontology',
            'type': '<str> `name` of the ancestor ontological group of this node',
            'group_ids': 'ordered list of IDs of the ontological groups to which database grounding of the node is mapped (order = ancestor-to-parent)',
            # 'group_refs': '<list of strs>',
            'node_group_level': '<int> length of the shortest path from the parent ontological group of the node to the ancestor group, plus one',
            'extras': '<dict> extra attributes' 
        }


    if obj_type == 'groups':

        preamble = {
            'id': '<int> ID of this group',
            'id_onto': '<str> ID of this group within the given ontology (format = `namespace:id`)',
            'name': '<str> human-readable name of this group',
            'level': '<int> length of the shortest path from this group to the ancestor group',
            'parent_id': '<int> ID of other groups in the ontology that are the immediate parent of this group',
            'children_ids': '<list of ints> List of IDs of other groups in the ontology that are the immediate children of this group',
            'model_id': '<int> ID of the associated model',
            'node_ids_all': '<list of ints> List of IDs of the model nodes that are grounded to this group and all its children',
            'node_ids_direct': '<list of ints> List of IDs of the model nodes that are directly grounded to this group (i.e. excluding its children)',
            'node_id_centroid': '<int> ID of the model node that is nearest to the median-centroid of this group'
        }


    if obj_type == 'groupLayout':

        preamble = {
            'node_id': '<int> ID of the group',
            'coor_sys_name': '<str> name of the coordinate system (`cartesian`, `spherical`)',
            'coors': '<list of floats> list of coordinate values of this group layout',
        }


    return preamble


# Generate nearest-neighbour median-centroid list from a set of coordinates and cluster labels
def generate_nn_cluster_centroid_list(coors: Any, labels: Any, p: Union[int, float] = 2) -> Tuple:

    # Error handling
    if not isinstance(coors, np.ndarray):
        raise TypeError("'coors' must be an numpy ndarray.")

    # if (not isinstance(labels, np.ndarray)) | (len(labels) not in [0, coors.shape[0]]): 
    #     raise TypeError("'labels' must be a N x 1 numpy ndarrray.")


    # Dimensions
    num_coors, num_dim = coors.shape


    # Assume no label = identically zeros
    if len(labels) == 0:
        labels = np.zeros((num_coors, ))

    labels_unique = np.unique(labels)
    num_unique = len(labels_unique)


    # Calculate centroid coordinates
    coors_centroid = np.empty((num_unique, num_dim))
    for i in range(num_unique):
        coors_centroid[i, :] = np.nanmedian(coors[labels == labels_unique[i], :], axis = 0)


    # Choose kNN metric
    if isinstance(p, int) & (p >= 1):
        knn = skl.neighbors.NearestNeighbors(n_neighbors = 1, metric = 'minkowski', p = p)

    else:

        # Define custom Minkowski distance function to enable non-integer `p`
        @numba.njit
        def minkowski_distance(u, v, p):
            return (np.abs(u - v) ** p).sum() ** (1.0 / p)

        knn = skl.neighbors.NearestNeighbors(n_neighbors = 2, metric = lambda u, v: minkowski_distance(u, v, p = p))
    
    
    # Find index of k-nearest neighbour to the cluster centroids
    knn_ind = np.empty((num_unique, ), dtype = np.int)
    for i in range(num_unique):

        knn.fit(coors[labels == labels_unique[i], :])
        k = knn.kneighbors(coors_centroid[i, :].reshape(1, -1), return_distance = False)
        k = k.item()

        # Convert to global index
        knn_ind[i] = np.flatnonzero(labels == labels_unique[i])[k].astype('int')


    return (knn_ind, labels_unique, coors_centroid)


# Calculate minimum spanning tree and distances from given clusters
def calc_mstree_dist(X: Any, labels: Any = [], metric: str = 'euclidean', plot_opt: bool = False, plot_opt_hist: bool = False, ax: Any = []) -> Any:

    # Error handling
    if not isinstance(X, np.ndarray):
        raise TypeError("'coor' must be an numpy ndarray.")
    # if (len(labels) > 1) and (not isinstance(labels, np.ndarray)):
    #     raise TypeError("'labels' must be an numpy ndarrray.")
    
    # Label clusters if not given
    if len(labels) < 1:
        labels = np.zeros((X.shape[0], ), dtype = np.int8)
    

    # Unique cluster labels
    labels_uniq = np.unique(labels)
    n_labels_uniq = labels_uniq.size


    # If boolean labels, only keep `True` labels
    if isinstance(labels_uniq[0], np.bool_):
        labels_uniq = np.array([True])
        n_labels_uniq = 1


    # Get pairwise distances from cluster-specific minimum spanning tree
    mstree = []
    mstree_dist_uniq = []
    for i, l in enumerate(labels_uniq):

        # Filter by cluster
        ind = (labels == l)

        # Run HDBSCAN to generate the minimum spanning tree
        mstree.append(hdbscan.HDBSCAN(metric = metric, gen_min_span_tree = True).fit(X[ind, :]).minimum_spanning_tree_.to_numpy())

        # Get the pairwise distances
        mstree_dist_uniq.append(mstree[i][:, 2])


    # Plot for debugging
    if plot_opt == True:

        # Colormap
        col = np.asarray([plt.cm.get_cmap('tab10')(i) for i in range(10)])
        col[:, 3] = 1.0

        # Plot figure
        if type(ax).__name__ != 'AxesSubplot':
            if plot_opt_hist == True:
                fig, ax = plt.subplots(figsize = (12, 6), nrows = 1, ncols = 2)
                ax_ = ax[0]
            else:
                fig, ax = plt.subplots(figsize = (6, 6), nrows = 1, ncols = 1)
                ax_ = ax
        else:
            fig = plt.getp(ax, 'figure')
        

        # Plot mstree for each labelled set of vertices
        for i, l in enumerate(labels_uniq):

            # Filter by cluster
            ind = (labels == l)

            for j, k, __ in mstree[i]:

                x = [X[ind, :][int(j), 0], X[ind, :][int(k), 0]]
                y = [X[ind, :][int(j), 1], X[ind, :][int(k), 1]]

                __ = ax_.plot(x, y, color = col[i % 10, :3])
                    
        plt.setp(ax_, title = 'Minimum Spanning Tree', xlabel = '$x$', ylabel = '$y$', aspect = 1.0)

        # Square axes
        xlim = plt.getp(ax_, 'xlim')
        ylim = plt.getp(ax_, 'ylim')
        dx = 0.5 * (xlim[1] - xlim[0])
        dy = 0.5 * (ylim[1] - ylim[0])
        if dy > dx:
            xlim = tuple(np.mean(xlim) + (-dy, dy))
            plt.setp(ax_, xlim = xlim)
        elif dx > dy:
            ylim = tuple(np.mean(ylim) + (-dx, dx))
            plt.setp(ax_, ylim = ylim)


        # Plot histogram
        if plot_opt_hist == True:
            
            m = max([max(l) for l in mstree_dist_uniq])
            z = np.linspace(0, m, 101)

            for i, l in enumerate(mstree_dist_uniq):
                
                # Calculate and plot histogram
                y = np.histogram(l, bins = z, density = True)[0] * (z[1] - z[0])
                h = ax[1].plot(z[:-1], y, color = col[i % 10, :3], label = f'{labels_uniq[i]}')

                # min, median, max
                min_dist = min(l)
                median_dist = np.median(l)
                max_dist = max(l)
                n = 0.5 * max(y)
                __ = ax[1].plot([min_dist, median_dist, max_dist], [n, n, n], color = plt.getp(h[0], 'color'), marker = 'o', markerfacecolor = 'w')

            __ = plt.setp(ax[1], xlabel = 'Pairwise Distance', ylabel = 'PMF', title = f'Histogram of Pairwise Distances')

            if n_labels_uniq > 1:
                __ = ax[1].legend()

    return mstree_dist_uniq, mstree


# Generate node, node-attribute, node-layout lists from a set of metadata, coordinates, labels
def generate_kaggle_nodelist(docs: List, embs: Any, labels: Any, model_id: int = 0, print_opt: bool = False) -> Tuple:

    # Check sizes
    if (not isinstance(embs, np.ndarray)) | (not isinstance(labels, np.ndarray)):
        raise TypeError("'embs' and 'labels' must be numpy ndarrays.")

    if (len(docs) != embs.shape[0]) | (len(docs) != labels.shape[0]):
        raise ValueError("'docs', 'embs', and 'labels' must have the same number of elements.")


    # Get number of nodes and epsilon values
    num_nodes = len(docs)
    num_eps = labels.shape[1]


    if num_nodes == 0:

        nodes = []
        nodeLayout = []
        nodeAtts = []
        groups = []


    else:

        # Initialize the node list
        nodes = [{
            'id': i,
            'model_id': model_id,
            'name': doc['title'],
            'grounded_db': True,
            'db_ids': [
                {k: doc[k]}
            for k in ('cord_uid', 'doi', 'pmcid', 'pubmed_id', 'mag_id', 'who_covidence_id', 'arxiv_id') if k in doc.keys() if doc[k] != ''],
            'edge_ids_source': [],
            'edge_ids_target': [],
            'out_degree': 0,
            'in_degree:': 0
        } for i, doc in enumerate(docs)]


        # Create global node IDs
        nodes = [transform_obj_ids(node, obj_type = 'nodes') for node in nodes]


        # Generate node-layout list
        nodeLayout = [{
            'node_id': node['id'],
            'coor_sys_name': 'cartesian',
            'coors': [float(x) for x in embs[i, :3]]
        } for i, node in enumerate(nodes)]


        # Generate node-attribute list
        nodeAtts = [{
            'node_id': node['id'],
            'grounded_group': None,
            'type': None,
            'group_ids': None,
            'node_group_level': None,
            'extras': {}
        } for __, node in enumerate(nodes)]


        # Get node bibjson data
        for node, atts, doc in zip(nodes, nodeAtts, docs):

            atts['extras']['bibjson'] = {
                'title': doc['title'],
                'author': [{'name': name} for name in doc['authors'].split(';')],
                'type': 'article',
                'year': doc['publish_time'].split('-')[0],
                # 'month': doc['publish_time'].split('-')[1],
                # 'day': doc['publish_time'].split('-')[2],
                'journal': doc['journal'],
                'link': [{'url': doc['url']}],
                'identifier': [{'type': k, 'id': v} for d in node['db_ids'] for k, v in d.items()],
                'abstract': doc['abstract']
            }


        # Create global group IDs from labels
        m = np.cumsum(np.insert(np.max(labels, axis = 0)[:-1], 0, 0))
        labels_ = labels + m
        labels_[labels == -1] = -1
        labels_ids = np.array([[transform_obj_ids(int(i), obj_type = 'groups') if i != -1 else -1 for i in l] for l in labels_])


        for i, node in enumerate(nodeAtts):

            j = np.argwhere(labels[i, :] != -1).flatten()

            if len(j) > 0:
                node['grounded_group'] = True
                node['group_ids'] = [int(l) for l in labels_ids[i, labels_ids[i, :] != -1]]
                node['node_group_level'] = int(j[-1] + 1)
      
            else:
                node['grounded_group'] = False
                node['groupd_ids'] = None
                node['node_group_level'] = None


        # Generate group membership list
        map_nodes_ids = {i: node['id'] for i, node in enumerate(nodes)}
        map_groups_nodes = {group_id: [] for row in labels_ids for group_id in row if group_id != -1}
        for i in range(num_nodes):
            for j in range(num_eps):
                group_id = labels_ids[i, j]
                if group_id != -1:
                    map_groups_nodes[group_id].append(map_nodes_ids[i])


        # Generatet group-level list
        map_groups_levels = {group_id: j for row in labels_ids for j, group_id in enumerate(row) if group_id != -1}


        # Generate group-parent list
        map_groups_parent = {group_id: row[j - 1] if j > 0 else None for row in labels_ids for j, group_id in enumerate(row) if group_id != -1}
        map_groups_children = {group_id: [] for row in labels_ids for j, group_id in enumerate(row) if group_id != -1}
        for i in range(num_nodes):
            for j in range(num_eps - 1):
                group_id = labels_ids[i, j]
                if group_id != -1:
                    map_groups_children[group_id].extend(labels_ids[i, (j + 1):])


        # Generate group list
        groups = [{
            'id': group_id,
            'id_onto': None,
            'name': None,
            'level': (lambda x: int(x) if x != None else None)(map_groups_levels[group_id]),
            'parent_id': (lambda x: int(x) if x != None else None)(map_groups_parent[group_id]),
            'children_ids': [int(i) for i in sorted(list(set(map_groups_children[group_id]))) if i != -1],
            'model_id': model_id,
            'node_ids_all': None,
            'node_ids_direct': [int(i) for i in sorted(map_groups_nodes[group_id])],
            'node_id_centroid': None
        } for group_id in sorted(map_groups_nodes.keys())]


        # # Node IDs of children groups
        # for group in groups:

        #     # group['node_ids_all'] = [int(i) for group_id in group['children_ids'] if group_id != -1 for i in sorted(map_groups_nodes[group_id])]

        #     group['node_ids_all'] = [int(i) for i in sorted(map_groups_nodes[group_id])]

        #     group['node_ids_all'].extend([int(i) for group_id in group['children_ids'] if group_id != -1 for i in map_groups_nodes[group_id]])


        # Calculate kNN median centroid of each group
        map_ids_nodes = {node['id']: i for i, node in enumerate(nodes)}
        for group in groups:
            
            coors = embs[np.array([map_ids_nodes[i] for i in group['node_ids_direct']]), :]

            i, __, __ = generate_nn_cluster_centroid_list(coors = coors, labels = [])

            group['node_id_centroid'] = group['node_ids_direct'][i.item()]
            group['name'] = nodes[map_ids_nodes[group['node_ids_direct'][i.item()]]]['name']

    
    if print_opt == True:

        print(f"{len(nodes)} nodes and {len(groups)} groups.")


    return nodes, nodeLayout, nodeAtts, groups


# %%
# Calculate kNN median centroid list for each epsilon

i = 0
j = np.random.randint(0, high = embs_red.shape[0], size = 5000)
knn_ind, l_unique, coors_centroid = generate_nn_cluster_centroid_list(coors = embs_red, labels = labels[:, i])


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 15))
__ = emlib.plot_emb(coor = embs_red[j, :], labels = labels[j, i], cmap_name = 'qual', marker_size = 0.5, marker_alpha = 0.1, legend_kwargs = {}, colorbar = False, str_title = f'\u03B5 = {epsilons[i]:.2f}', ax = ax)
ax.scatter(coors_centroid[:, 0], coors_centroid[:, 1], marker = '+')


# %%


i, j = np.unique(labels[:, 0], return_counts = True)
k = np.argsort(j)
l = i[k][-5]

i = 0
j = labels[:, i] == l
knn_ind, l_unique, coors_centroid = generate_nn_cluster_centroid_list(coors = embs_red[j, :], labels = [])

d, m = calc_mstree_dist(X = embs_red[j, :], labels = [], plot_opt = False, plot_opt_hist = False)


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 15))
__ = emlib.plot_emb(coor = embs_red[j, :], labels = labels[j, i], cmap_name = 'qual', marker_size = 0.5, marker_alpha = 0.1, legend_kwargs = {}, colorbar = False, str_title = f'\u03B5 = {epsilons[i]:.2f}', ax = ax)
ax.scatter(coors_centroid[:, 0], coors_centroid[:, 1], marker = '+')


# %%
# Generate lists
nodes, nodeLayout, nodeAtts, groups = generate_kaggle_nodelist(docs = docs, embs = embs_red, labels = labels_, model_id = None)

# %%

s = [len(group['node_ids_direct']) for group in groups]
i = np.argsort(s)[-3]
x = [node['coors'] for node in nodeLayout]
y = {node['id']: i for i, node in enumerate(nodes)}
z = np.array([x[y[k]] for k in groups[i]['node_ids_direct']])

x[y[groups[i]['node_id_centroid']]]


plt.scatter(z[:, 0], z[:, 1])
plt.scatter(x[y[groups[i]['node_id_centroid']]][0], x[y[groups[i]['node_id_centroid']]][1], color = 'r')


# %%
# Export data

for x, y in zip(('nodes', 'nodeLayout', 'nodeAtts', 'groups'), (nodes, nodeLayout, nodeAtts, groups)):

    emlib.save_jsonl(y, f'./dist/kaggle/{x}.jsonl', preamble = get_obj_preamble(obj_type = x))


# %%


