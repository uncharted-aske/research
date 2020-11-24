# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Idea: 
# * Aggregate KB edges by onto clusters into 'hyperedges'
# * Output results to `clusterEdges.jsonl`
# 
# 
# Aggregations:
# 1. source = node, target = ancestors
# 2. source = node, target = parents
# 3. source = node, target = siblings
# 4. source = node, target = descendants
# 4. source = node, target = all else


# %%
import json
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle

from numba import njit


import emmaa_lib as emlib
import importlib
# `importlib.reload(emlib)`

# %%
np.random.seed(0)

# %%
# Load KB graph
nodesKB = {}
with open('./data/covid19-snapshot_sep18-2020/processed/nodes.jsonl', 'r') as x:
    nodesKB = [json.loads(i) for i in x]

nodesKB_data = {}
with open('./dist/v1/nodeData.jsonl', 'r') as x:
    nodesKB_data = [json.loads(i) for i in x]
nodesKB_data = nodesKB_data[1:]

edgesKB = {}
with open('./data/covid19-snapshot_sep18-2020/processed/edges.jsonl', 'r') as x:
    edgesKB = [json.loads(i) for i in x]

edgesKB_ = {}
with open('./data/covid19-snapshot_sep18-2020/processed/collapsedEdges.jsonl', 'r') as x:
    edgesKB_ = [json.loads(i) for i in x]

# Un-collapse edges
for edge in edgesKB:
    i = edge['collapsed_id']
    edge['source'] = edgesKB_[i]['source']
    edge['target'] = edgesKB_[i]['target']


# Load ontology graph
ontoG = {}
with open('./dist/v1/emmaa_4_indraOntology_onto.pkl', 'rb') as x:
    ontoG = pickle.load(x)


# Load ontology clusters
ontoClusters = {}
with open('./dist/v1/clusters.jsonl', 'rb') as x:
    ontoClusters = [json.loads(i) for i in x]
ontoClusters = ontoClusters[1:]

# ontoClusters_nested = {}
# with open('./dist/v1/clusters_nested.jsonl', 'rb') as x:
#     ontoClusters_nested = [json.loads(i) for i in x]
# ontoClusters_nested = ontoClusters_nested[1:]


i = x = edge = edgesKB_ = None
del i, x, edge, edgesKB_

#%%
# Generate onto ref list and onto subgraph 
ontoClusters_ref = {cluster['ref']: cluster for cluster in ontoClusters}
ontoG_sub = ontoG.subgraph(ontoClusters_ref.keys())

# KB edge vertices
edgesKB_nodes = np.array([[edge['source'], edge['target']] for edge in edgesKB])

# %%
%%time

# KB nodes mapped to the sibling clusters
# * Get the `ref` of all onto clusters with same `parentID` (i.e. siblings)
# * Get the node ids of their KB membership
# * Do not flatten the list
siblings_ref = [set([c['ref'] if c['parentID'] == cluster['parentID'] else '' for c in ontoClusters]) - {'', cluster['ref']} for cluster in ontoClusters]
siblings_id = [[ontoClusters_ref[c]['id'] for c in siblings_ref[i]] for i, __ in enumerate(ontoClusters)]
siblings_nodeIDs = [[ontoClusters_ref[c]['nodeIDs'] for c in siblings_ref[i]] for i, __ in enumerate(ontoClusters)]


# time: 4 m 31 s

# %%
%%time

# KB nodes mapped to the parent cluster (without given cluster's member nodes)
parent_nodeIDs = [set([node for node in ontoClusters[cluster['parentID']]['nodeIDs']]) - set(cluster['nodeIDs']) - set([node for c in siblings_nodeIDs[i] for node in c]) if cluster['parentID'] is not None else [] for i, cluster in enumerate(ontoClusters)]


# time: 1.16 s

# %%




# %%
n = 200

# KB edges with the given cluster member as their source
X = [match_arrays(edgesKB_nodes[:, 0], cluster['nodeIDs']) for cluster in ontoClusters[:1]]

# ... and a sibling cluster member as their target
Y = [[np.flatnonzero(X[i] & match_arrays(edgesKB_nodes[:, 1], nodes)) for nodes in siblings_nodeIDs[i][:n]] for i, __ in enumerate(ontoClusters[:1])]
Y_ = [[list(k) for k in y if len(k) > 0] for y in Y]

# KB edge attributes
Z = [[{'level': cluster['level'], 'source': {'clusterID': cluster['id']}, 'target': {'clusterID': nodes}} for nodes in siblings_id[i][:n]] for i, cluster in enumerate(ontoClusters[:1])]
Z_ = [[l for k, l in zip(y, z) if len(k) > 0] for y, z in zip(Y, Z)]


# %%


# KB edges with the given cluster member as their source
# ... and a parent cluster member as their target 
Y = [[np.flatnonzero(X[i] & match_arrays(edgesKB_nodes[:, 1], node)) for node in parent_nodeIDs[i][:n]] for i, __ in enumerate(ontoClusters[:1])]
Y_ = [[list(k) for k in y if len(k) > 0] for y in Y]


# # KB edge attributes
# Z = [[{'level': cluster['level'], 'source': {'clusterID': cluster['id'], 'nodeID': X[i]}, 'target': {'clusterID': None, 'nodeID': node}} for node in parent_nodeIDs[i][:n]] for i, cluster in enumerate(ontoClusters[:1])]
# Z_ = [[l for k, l in zip(y, z) if len(k) > 0] for y, z in zip(Y, Z)]



# %%


        for targetNode in parent_nodeIDs:

            # KB edges with a parent cluster member as their target 
            y = set(np.flatnonzero(edgesKB_nodes[:, 1] == targetNode))

            z = list(x & y)
            if len(z) > 0:
                hyperedge = {
                    'id': k, 
                    'level': cluster['level'],
                    'source': {'clusterID': cluster['id'], 'nodeID': sourceNode},
                    # 'target': {'clusterID': cluster['parentID'], 'nodeIDs': targetNodes}, 
                    'target': {'clusterID': None, 'nodeID': targetNode},                     
                    'size': len(z),
                    'edgeIDs': [int(j) for j in z]
                }
                clusterEdges.append(hyperedge)
                k = k + 1




#         hyperedge = {
#             'id': k, 
#             'level': cluster['level'],
#             'source': {'clusterID': cluster['id'], 'nodeID': None},
#             'target': {'clusterID': siblings_id[i][j], 'nodeID': None},                     
#             'size': int(sum(z)),
#             'edgeIDs': list(map(lambda l: int(l), np.flatnonzero(z)))
#         }



# %%
    # x = np.zeros((len(edgesKB), ), dtype = bool)
    # for node in cluster['nodeIDs']:
    #     x = x | (edgesKB_nodes[:, 0] == node)

        # y = np.zeros((len(edgesKB), ), dtype = bool)
        # for node in siblings_nodeIDs[i][j]:
        #     y = y | (edgesKB_nodes[:, 1] == node)


# %%
%%time

k = 0
clusterEdges = []
for cluster in ontoClusters:

    # KB nodes of the sibling clusters
    # * Get the `ref` of all onto clusters with same `parentID` (i.e. siblings)
    # * Get the node ids of their KB membership
    # * Do not flatten the list
    siblings_ref = set([c['ref'] if c['parentID'] == cluster['parentID'] else '' for c in ontoClusters]) - {'', cluster['ref']}
    siblings_id = [ontoClusters_ref[c]['id'] for c in siblings_ref]
    siblings_nodeIDs = [ontoClusters_ref[c]['nodeIDs'] for c in siblings_ref]


    # KB nodes of the parent cluster (without given cluster's member nodes)
    # * Get the node ids of KB nodes mapped to the parent cluster
    # x = nx.algorithms.dag.ancestors(ontoG_sub.reverse(copy = False), cluster['ref'])
    # y = set([c['ref'] if c['level'] == (cluster['level'] - 1) else '' for c in ontoClusters]) - {''}
    # parent_ref = x & y
    if cluster['parentID'] != None:
        parent_nodeIDs = set([node for node in ontoClusters[cluster['parentID']]['nodeIDs']]) - set(cluster['nodeIDs']) - set([node for c in siblings_nodeIDs for node in c])
    else:
        parent_nodeIDs = []


    # Aggregate KB edges into hyperedges
    for sourceNode in cluster['nodeIDs']:

        # KB edges with the given cluster member as their source
        x = set(np.flatnonzero(edgesKB_nodes[:, 0] == sourceNode))

        for i, targetNodes in zip(siblings_id, siblings_nodeIDs):
            
            # KB edges with a sibling cluster member as their target 
            y = set(np.flatnonzero(np.asarray([edgesKB_nodes[:, 1] == node for node in targetNodes]).sum(axis = 0)))

            z = list(x & y)
            if len(z) > 0:
                hyperedge = {
                    'id': k, 
                    'level': cluster['level'],
                    'source': {'clusterID': cluster['id'], 'nodeID': sourceNode},
                    # 'target': {'clusterID': i, 'nodeIDs': targetNodes}, 
                    'target': {'clusterID': i, 'nodeID': None},                     
                    'size': len(z),
                    'edgeIDs': [int(j) for j in z]
                }
                clusterEdges.append(hyperedge)
                k = k + 1

        
        for targetNode in parent_nodeIDs:

            # KB edges with a parent cluster member as their target 
            y = set(np.flatnonzero(edgesKB_nodes[:, 1] == targetNode))

            z = list(x & y)
            if len(z) > 0:
                hyperedge = {
                    'id': k, 
                    'level': cluster['level'],
                    'source': {'clusterID': cluster['id'], 'nodeID': sourceNode},
                    # 'target': {'clusterID': cluster['parentID'], 'nodeIDs': targetNodes}, 
                    'target': {'clusterID': None, 'nodeID': targetNode},                     
                    'size': len(z),
                    'edgeIDs': [int(j) for j in z]
                }
                clusterEdges.append(hyperedge)
                k = k + 1


i = k = x = y = z = cluster = parent_nodeIDs = siblings_ref = siblings_id = siblings_nodeIDs = targetNodes = targetNode = hyperedge = None
del i, k, x, y, z, cluster, parent_nodeIDs, siblings_ref, siblings_id, siblings_nodeIDs, targetNodes, targetNode, hyperedge

# %%

# Output cluster edges data
with open(f'./dist/v1/clusterEdges.jsonl', 'w') as x:

    # Description
    y = {
        'id': '<int> unique ID for the cluster hyperedge to which `edgeIDs` in `clusters.jsonl` refers',
        'level': '<int> ontological level of this hyperedge (number of hops to the local root node of the ontology)',
        'source': '<dict with 2 keys> source of this hyperedge',
        'target': '<dict with 2 keys> target of this hyperedge (if `clusterID` = None, then this is a onto-cluster-to-KB-node edge)',
        'size': '<int> size of the hyperedge (number of KB edges that is aggregated)',
        'edgeIDs': '<array of int> unordered list of KB edge IDs (`id` in `edges.jsonl`) that are aggregated to this hyperedge'
    }
    json.dump(y, x)
    x.write('\n')


    # Data
    for y in clusterEdges:
        json.dump(y, x)
        x.write('\n')


x = y = None
del x, y

