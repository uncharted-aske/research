# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Idea: 
# * Assume an undirected graph
# * Use the graph Laplacian vectors as initial node embeddings
# * Create 2D/3D layout by dimensional reduction (UMAP)
# * Learn an ontology by applying hierarchy clustering (HDBSCAN)

# %%
import json
import time
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import pickle
import umap
import sklearn as skl

import emmaa_lib as emlib
import importlib
# `importlib.reload(emlib)`

# %%
np.random.seed(0)

# %%[markdown]
# ## Load node and edge data from Dario

nodes = {}
with open('./data/covid19-snapshot_sep18-2020/processed/nodes.jsonl', 'r') as x:
    nodes = [json.loads(i) for i in x]

edges = {}
with open('./data/covid19-snapshot_sep18-2020/processed/edges.jsonl', 'r') as x:
    edges = [json.loads(i) for i in x]

edges_ = {}
with open('./data/covid19-snapshot_sep18-2020/processed/collapsedEdges.jsonl', 'r') as x:
    edges_ = [json.loads(i) for i in x]

x = None
del x

# %%
%%time

# Collate CollapseEdges with edges data
for edge in edges:
    i = edge['collapsed_id']
    edge['source'] = edges_[i]['source']
    edge['target'] = edges_[i]['target']

edges_ = None
del edges_

# %%


