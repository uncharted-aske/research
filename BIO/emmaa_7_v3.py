# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Idea: 
# * New ontology (`v1.5`) from Ben Gyori
# * New COVID-19 model graph (`dec8-2020`)
# * Handle better the multi-member statements
# * Re-generate distribution files


# %%
import json
import pickle
import time
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
# # Extract Nodes and Edges from Statements

# %%

statements_all = {}
with open('./data/covid19-snapshot_dec8-2020/source/latest_statements_covid19.json', 'r') as x:
    statements_all = json.load(x)


nodes, edges, statements = emlib.parse_statements(statements_all)


# %%
