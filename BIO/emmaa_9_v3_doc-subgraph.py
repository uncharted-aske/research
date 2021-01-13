# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content: 
# * Load data of the full EMMAA Covid-19 graph
# * Filter statements by a given document DOI/PMID/PMCID
# * Get the nodes and edges associated with these statements
# * Generate the associated output files (`nodes.jsonl, edges.jsonl, G.pkl, ...`)


# %%
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

import emmaa_lib as emlib
import importlib
# importlib.reload(emlib)

# %%
np.random.seed(0)

# %%[markdown]
# # Load Statement Data
statements_all = {}
with open('./data/covid19-snapshot_dec8-2020/source/latest_statements_covid19.json', 'r') as x:
    statements_all = json.load(x)

# %%[markdown]
# # Define the document of interest
doc = {'id_type': 'DOI', 'id': '10.1016/j.immuni.2020.04.003'.upper()}

# %%[markdown]
# # Filter Statements by Reference
statement_ids = [statements_all[j]['matches_hash'] for j in set([i for i, s in enumerate(statements_all) for ev in s['evidence'] if ('text_refs' in ev.keys()) if ('DOI' in ev['text_refs'].keys()) if ev['text_refs'][doc['id_type']] == doc['id']])]


# %%
# Nodes and edges of the model graph
nodes_model = emlib.load_jsonl('./dist/v3/nodes_model.jsonl', remove_preamble = True)
edges_model = emlib.load_jsonl('./dist/v3/edges_model.jsonl', remove_preamble = True)








# Load statements
statements_all = {}
with open('./data/covid19-snapshot_dec8-2020/source/latest_statements_covid19.json', 'r') as x:
    statements_all = json.load(x)


statement_ids = [statements_all[j]['matches_hash'] for j in set([i for i, s in enumerate(statements_all) for ev in s['evidence'] if ('text_refs' in ev.keys()) if ('DOI' in ev['text_refs'].keys()) if ev['text_refs']['DOI'] == '10.1016/j.immuni.2020.04.003'.upper()])]
# len(statement_ids) = 7


nodes_model = emlib.load_jsonl('./dist/v3/nodes_model.jsonl', remove_preamble = True)
edges_model = emlib.load_jsonl('./dist/v3/edges_model.jsonl', remove_preamble = True)


x = [edge for edge in edges_model if edge['statement_id'] in y]
node_names = [nodes_model[j]['name'] for j in set([edge['source_id'] for edge in z] + [edge['target_id'] for edge in x])]
# len(node_names) = 10
