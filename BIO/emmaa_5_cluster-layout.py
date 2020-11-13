# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Idea: 
# * Aggregate KB edges by ontological clusters
# * Output results to `clusterEdges.jsonl`

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


# Load ontology clusters
ontoClusters = {}
with open('./dist/v1/clusters.jsonl', 'rb') as x:
    ontoClusters = [json.loads(i) for i in x]
ontoClusters = ontoClusters[1:]


x = edge = edgesKB_ = None
del x, edge, edgesKB_


#%%

ontoClusters_nodesKB = [np.flatnonzero([True if cluster['id'] in node['clusterIDs'] else False for node in nodesKB_data]) for cluster in ontoClusters]






# %%
