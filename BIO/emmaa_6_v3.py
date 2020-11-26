# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Idea: 
# * Reset in v3
# * Start from the MITRE test set
# * 
# 

# %%
import json
import pickle
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import numba

import emmaa_lib as emlib
import importlib
# importlib.reload(emlib)

# %%
np.random.seed(0)

# %%
nodes = {}
with open('./data/covid19-snapshot_sep18-2020/processed/nodes.jsonl', 'r') as x:
    nodes = [json.loads(i) for i in x]

edges = {}
with open('./data/covid19-snapshot_sep18-2020/processed/edges.jsonl', 'r') as x:
    edges = [json.loads(i) for i in x]

edges_ = {}
with open('./data/covid19-snapshot_sep18-2020/processed/collapsedEdges.jsonl', 'r') as x:
    edges_ = [json.loads(i) for i in x]

paths_mitre = {}
with open('./data/covid19-snapshot_sep18-2020/processed/mitre_tests.jsonl', 'r') as x:
    paths_mitre = [json.loads(i) for i in x]


# %%
# Uncollapse the model edge data
for edge in edges:
    i = edge['collapsed_id']
    edge['source'] = edges_[i]['source']
    edge['target'] = edges_[i]['target']

i = x = edge = edges_ = None
del i, x, edge, edges_


# %%
# Filter model graph by MITRE paths
nodes_mitre, edges_mitre, __ = emlib.intersect_graph_paths(nodes, edges, paths_mitre)


# #####<<< RESET NODE ID FOR GRAFER OPTIMIZATION


nodes = edges = None
del nodes, edges

# %%
# Load the INDRA ontology

with open('./data/indra_ontology_v1.3.json', 'r') as x:
    ontoJSON = json.load(x)

# Remove 'xref' links
ontoJSON['links'] = [link for link in ontoJSON['links'] if link['type'] != 'xref']


# %%
# Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces = emlib.generate_ordered_namespace_list(nodes_mitre, ontoJSON, namespaces_priority)


# %%
# Compute model-ontology mapping
nodeData_ontoRefs = []
for node in nodes_mitre:

    if len(node['info']['links']) < 1:
        names = ['not-grounded']
        k = names[0]
    else:
        names = [link[0] for link in node['info']['links']]

        # Use first matching namespace in the ordered common list
        i = np.flatnonzero([True if name in names else False for name in namespaces])[0]
        j = np.flatnonzero(np.asarray(names) == namespaces[i])[0]
        k = f"{node['info']['links'][j][0]}:{node['info']['links'][j][1]}"

    nodeData_ontoRefs.append(k)


names = node = i = j = k = None
del names, node, i, j, k


# %%
# Load the ontology graph as a `networkx` object
ontoG = nx.readwrite.json_graph.node_link_graph(ontoJSON)


# Generate components, sorted by size
ontoSubs = sorted(nx.weakly_connected_components(ontoG), key = len, reverse = True)

# Find the root nodes of each component (degree = 0 or out-degree = 0)
z = [np.flatnonzero([True if ontoG.out_degree(node) < 1 else False for node in sub]) for sub in ontoSubs]
ontoSubRoots = [[list(ontoSubs[i])[j] for j in indices] for i, indices in enumerate(z)]
ontoSubRoots_num = np.sum([True if len(indices) > 1 else False for indices in z])


# # List onto node names/refs
# ontoRefs = nx.nodes(ontoG)

# %%
%%time

# Index all model nodes that can be mapped to the ontology graph


# Initialize and set the ontological level of the unmappable nodes to -1
x = np.flatnonzero([True if i in nx.nodes(ontoG) else False for i in nodeData_ontoRefs])
nodeData_ontoLevels = np.zeros((len(nodes_mitre), ), dtype = np.int64)
nodeData_ontoPaths = list(np.zeros((len(nodes_mitre), ), dtype = np.int64))
for i in range(len(nodes_mitre)):
    if i not in x:
        nodeData_ontoLevels[i] = -1
        nodeData_ontoPaths[i] = [nodeData_ontoRefs[i]]


# Find subgraph index of each mapped model node
# * Limited to non-trivial subgraphs
# * Set to -1 if a node is mapped to a trivial subgraph
y = np.empty(x.shape, dtype = np.int64)
for i, k in enumerate(x):
    j = np.flatnonzero([True if nodeData_ontoRefs[k] in sub else False for sub in ontoSubs[:ontoSubRoots_num]])
    if len(j) == 1:
        y[i] = j[0]
    else:
        y[i] = -1


# Find shortest path between each onto-mapped model node and any target root node amongst the ontology subgraphs
for i, j in zip(x, y):

    source = nodeData_ontoRefs[i]

    # Trivial ontology subgraphs
    if j == -1:
        nodeData_ontoLevels[i] = 0
        nodeData_ontoPaths[i] = [source]

    # All other subgraphs
    else:

        z = []
        for target in ontoSubRoots[j]:
            try:
                p = nx.algorithms.shortest_paths.generic.shortest_path(ontoG.subgraph(ontoSubs[j]), source = source, target = target)
                z.append(p)
            except:
                pass
        
        # Find shortest path and reverse such that [target, ..., source]
        z = sorted(z, key = len, reverse = False)
        nodeData_ontoPaths[i] = z[0][::-1]
        nodeData_ontoLevels[i] = len(z[0]) - 1


i = j = p = x = y = z = source = target = None
del i, j, p, x, y, z, source, target

# time: 5 m 6 s

# %%

# Ensure that identical onto nodes share the same lineage (i.e. path to their ancestor) for hierarchical uniqueness
nodeData_ontoPaths_reduce = nodeData_ontoPaths[:]
m = max([len(path) for path in nodeData_ontoPaths])
n = len(nodes_mitre)
for i in range(1, m):

    # All nodes
    x = [path[i] if len(path) > i else '' for path in nodeData_ontoPaths]

    # All unique nodes
    y = list(set(x) - set(['']))

    # Mapping from all nodes to unique nodes
    xy = [y.index(node) if node is not '' else '' for node in x]

    # Choose the path segment of the first matching node for each unique node
    z = [nodeData_ontoPaths[x.index(node)][:i] for node in y]
    
    # Substitute path segments
    for j in range(n):
        if xy[j] is not '':
            nodeData_ontoPaths_reduce[j][:i] = z[xy[j]]
        else:
            nodeData_ontoPaths_reduce[j][:i] = nodeData_ontoPaths[j][:i]


x = y = z = xy = i = j = m = n = None
del x, y, z, xy, i, j, m, n


# %%
%%time

# Generate list of mapped ontology categories, sorted by size
ontoCats = {}
ontoCats['ref'], ontoCats['size'] = np.unique([node for path in nodeData_ontoPaths_reduce for node in path], return_counts = True)

num_ontoCats = len(ontoCats['ref'])
i = np.argsort(ontoCats['size'])[::-1]
ontoCats['ref'] = ontoCats['ref'][i]
ontoCats['size'] = ontoCats['size'][i]
ontoCats['id'] = list(range(num_ontoCats))


# Get the mapped onto category names
x = dict(ontoG.nodes(data = 'name', default = None))
ontoCats['name'] = list(np.empty((num_ontoCats, )))
for i, ontoRef in enumerate(ontoCats['ref']):
    try:
        ontoCats['name'][i] = x[ontoRef]
    except:
        ontoCats['name'][i] = ''


# Get onto level of each category
i = max([len(path) for path in nodeData_ontoPaths_reduce])
x = [np.unique([path[j] if len(path) > j else '' for path in nodeData_ontoPaths_reduce]) for j in range(i)]
ontoCats['ontoLevel'] = [np.flatnonzero([ontoRef in y for y in x])[0] for ontoRef in ontoCats['ref']]


# Get numeric id version of nodeData_ontoPaths_reduce
x = {k: v for k, v in zip(ontoCats['ref'], ontoCats['id'])}
nodeData_ontoPaths_id = [[x[node] for node in path] for path in nodeData_ontoPaths_reduce]


# Get parent category id for each category (for root nodes, parentID = None)
y = [np.flatnonzero([True if ref in path else False for path in nodeData_ontoPaths_reduce])[0] for ref in ontoCats['ref']]
ontoCats['parent'] = [nodeData_ontoPaths_reduce[y[i]][nodeData_ontoPaths_reduce[y[i]].index(ontoRef) - 1] if nodeData_ontoPaths_reduce[y[i]].index(ontoRef) > 0 else None for i, ontoRef in enumerate(ontoCats['ref'])]
ontoCats['parentID'] = [x[parent] if parent is not None else None for parent in ontoCats['parent']]


# Find membership of onto categories
ontoCats['nodeIDs'] = [[node['id'] for node, path in zip(nodes_mitre, nodeData_ontoPaths_reduce) if ontoRef in path] for ontoRef in ontoCats['ref']]


i = x = y = None
del i, x, y


# %%

# Output intersected model nodes and edges
# with open(f'./dist/v3/nodes_mitre.jsonl', 'w') as x:


# with open(f'./dist/v3/edges_mitre.jsonl', 'w') as x:


# Output model node data
# with open(f'./dist/v1/nodeData.jsonl', 'w') as x:

#     # Description
#     y = {
#         'id': '<int> unique ID for the node in the KB graph as specified in `nodes.jsonl`',
#         'x': '<float> position of the node in the graph layout (symmetric Laplacian + UMAP 3D)',
#         'y': '<float> position of the node in the graph layout (symmetric Laplacian + UMAP 3D)',
#         'z': '<float> position of the node in the graph layout (symmetric Laplacian + UMAP 3D)',
#         'degreeIn': '<int> in-degree in the KB graph',
#         'degreeOut': '<int> out-degree in the KB graph',
#         'belief': '<float> max of the belief scores of all adjacent edges in the KB graph',
#         'ontoID': '<str> unique ref ID of the INDRA ontology (v1.3) node to which this KB node is mapped', 
#         'ontoLevel': '<int> hierarchy level of the ontology node (`-1` if not mappable)',
#         'clusterIDs': '<array of int> ordered list of cluster IDs (see `clusters.jsonl`) to which this node is mapped (cluster hierarchy = INDRA ontology v1.3, order = root-to-leaf)'
#     }
#     json.dump(y, x)
#     x.write('\n')

#     # Data
#     for i in range(len(nodesKB)):
#         z = {
#             'id': int(nodesKB[i]['id']),
#             'x': float(nodesKB_pos[i, 0]), 
#             'y': float(nodesKB_pos[i, 1]), 
#             'z': float(nodesKB_pos[i, 2]), 
#             'degreeOut': int(nodesKB_degrees[i, 0]),
#             'degreeIn': int(nodesKB_degrees[i, 1]),
#             'belief': float(nodesKB_belief[i]),
#             'ontoID': nodesKB_ontoIDs[i], 
#             'ontoLevel': int(nodesKB_ontoLevels[i]),
#             'clusterIDs': nodesKB_ontoPaths_id[i]
#         }

#         json.dump(z, x)
#         x.write('\n')


# i = x = y = z = None
# del i, x, y, z













# %%


