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
import networkx as nx
import pickle
import umap

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

# %%

# Recall that onto edges point from specific concepts to more general ones (i.e. towards the root).


edgesKB_nodes = np.array([[edge['source'], edge['target']] for edge in edgesKB])


cluster = ontoClusters[15000]

k = 0
hyperedges = []
for cluster in ontoClusters[15000:15010]:

    # KB nodes of the sibling clusters
    # * Get the `ref` of all onto clusters with same `parentID` (i.e. siblings)
    # * Get the node ids of their KB membership
    # * Do not flatten the list
    siblings_ref = set([c['ref'] if c['parentID'] == cluster['parentID'] else '' for c in ontoClusters]) - {'', cluster['ref']}
    siblings_id = [ontoClusters_ref[c]['id'] for c in siblings_ref]
    siblings_nodeIDs = [ontoClusters_ref[c]['nodeIDs'] for c in siblings_ref]

    # Aggregate KB edges (to sibling cluster members) into hyperedges
    # One hyperedge per sibling
    for sourceNode in cluster['nodeIDs']:
        for i, targetNodes in zip(siblings_id, siblings_nodeIDs):
            
            x = set(np.flatnonzero(edgesKB_nodes[:, 0] == sourceNode))
            y = set(np.flatnonzero(np.asarray([edgesKB_nodes[:, 1] == node for node in targetNodes]).sum(axis = 0)))
            z = list(x & y)

            if len(z) > 0:
                hyperedge = {
                    'id': k, 
                    'level': cluster['level'],
                    # 'source': {'clusterID': cluster['id'], 'nodeIDs': sourceNode},
                    # 'target': {'clusterID': i, 'nodeIDs': targetNodes}, 
                    'source': {'clusterID': cluster['id'], 'nodeID': None},
                    'target': {'clusterID': i, 'nodeID': None},                     
                    'size': len(z),
                    'edgeIDs': z
                }
                hyperedges.append(hyperedge)
                k = k + 1




x = y = z = siblings_ref = siblings_ids = siblings_nodeIDs = hyperedge = None
del x, y, z, siblings_ref, siblings_ids, siblings_nodeIDs, hyperedge

# %%





    # KB nodes of the parent cluster
    # * Get the node ids of KB nodes mapped to the parent cluster
    # x = nx.algorithms.dag.ancestors(ontoG_sub.reverse(copy = False), cluster['ref'])
    # y = set([c['ref'] if c['level'] == (cluster['level'] - 1) else '' for c in ontoClusters]) - {''}
    # parent_ref = x & y
    if cluster['parentID'] != None:
        parent_nodeIDs = [[node] for node in ontoClusters[cluster['parentID']]['nodeIDs']]
    else:
        parent_nodeIDs = []


    # Aggregate KB edges (that is targeted at a KB member of the parent cluster) into hyperedges
    # One hyperedge per 
    sources = cluster['nodeIDs']
    targets = parent_nodeIDs



    # # Switch to <int> cluster id
    # lineage = {k: [ontoClusters_ref[c]['id'] for c in v] for k, v in lineage.items()}

    # # Lineage in terms of KB nodes (by <int> id)

    










# %%

# x = {node: True if node in ontoClusters_ref else False for node in ontoG.nodes()}
# ontoG_sub = nx.subgraph_view(ontoG, filter_node = (lambda n: n in ontoClusters_ref))

# y = ontoClusters[2]['ref']
# print(nx.get_node_attributes(ontoG, 'name')[x])
# z = nx.algorithms.dag.ancestors(ontoG.reverse(copy = False), x)








