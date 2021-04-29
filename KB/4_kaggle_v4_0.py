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
epsilons = np.arange(0.05, 0.6, 0.1)
labels = []
for eps in tqdm(epsilons):

    # Generate cluster labels
    kwargs = {'metric': 'euclidean', 'min_cluster_size': 2, 'min_samples': 3, 'cluster_selection_epsilon': float(eps)}
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

for l in labels.T:
    print(f'\nNumber of clusters: {len(np.unique(l)):d}')
    print(f'Number of unclustered points: {sum(l == -1):d} (of {len(l):d})')


eps = l = kwargs = clusterer = None
del eps, l, kwargs, clusterer

# Time: 1 m 10 s


# %%
with open('./dist/kaggle/embeddings_umap_hdscan.pkl', 'wb') as f:
    pickle.dump(labels, f)

# %%
if False:
    with open('./dist/kaggle/embeddings_umap_hdbscan.pkl', 'rb') as f:
        labels = pickle.load(f)


# %%[markdown]
# Plot result

fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 10))

for i, x in enumerate(fig.axes):

    if i < len(labels):
        __ = emlib.plot_emb(coor = embs_red[:5000], labels = labels[:, i], cmap_name = 'qual', legend_kwargs = {}, colorbar = False, str_title = f'\u03B5 = {epsilons[i]:.2f}', ax = x)
        __ = plt.setp(x, xlabel = '', ylabel = '')
    
    else:
        plt.axis('off')

fig.savefig('./figures/kaggle/embeddings_umap_hdbscan.png', dpi = 150)

fig = ax = x = i = None
del fig, ax, x, i


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
            'children_ids': [int(i) for i in sorted(list(set(map_groups_children[group_id])))],
            'model_id': model_id,
            'node_ids_all': [int(i) for i in sorted(map_groups_nodes[group_id])],
            'node_ids_direct': [int(i) for i in sorted(map_groups_nodes[group_id])]

        } for group_id in sorted(map_groups_nodes.keys())]

    
    if print_opt == True:

        print(f"{len(nodes)} nodes and {len(groups)} groups.")


    return nodes, nodeLayout, nodeAtts, groups


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
            'node_ids_direct': '<list of ints> List of IDs of the model nodes that are directly grounded to this group (i.e. excluding its children)'
        }


    if obj_type == 'groupLayout':

        preamble = {
            'node_id': '<int> ID of the group',
            'coor_sys_name': '<str> name of the coordinate system (`cartesian`, `spherical`)',
            'coors': '<list of floats> list of coordinate values of this group layout',
        }


    return preamble


# %%
# Generate lists
nodes, nodeLayout, nodeAtts, groups = generate_kaggle_nodelist(docs = docs, embs = embs_red, labels = labels, model_id = None)


# %%
# Export data

for x, y in zip(('nodes', 'nodeLayout', 'nodeAtts', 'groups'), (nodes, nodeLayout, nodeAtts, groups)):

    emlib.save_jsonl(y, f'./dist/kaggle/{x}.jsonl', preamble = get_obj_preamble(obj_type = x))


# %%

