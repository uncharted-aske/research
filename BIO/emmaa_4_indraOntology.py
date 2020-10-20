# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Idea: 
# * Explore the INDRA Ontology
# * Cluster nodes 

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
# ## Load INDRA Ontology v1.3 from Ben Gyori

with open('./data/indra_ontology_v1.3.json', 'r') as x:
    ontoData = json.load(x)


ontoG = nx.readwrite.json_graph.node_link_graph(ontoData, directed = ontoData['directed'], multigraph = ontoData['multigraph'])



