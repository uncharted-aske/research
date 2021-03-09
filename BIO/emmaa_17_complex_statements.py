# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content: 
# * Calculate overlap of statements of type `Complex`

# %%
import json
import pickle
import time
import datetime
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm

import emmaa_lib as emlib
import importlib
# importlib.reload(emlib)

# %%
np.random.seed(0)

# %%[markdown]
# # Load Statements of Full Model Graph

statements_full = emlib.load_jsonl(f"/home/nliu/projects/aske/research/BIO/data/models/covid19/2021-02-09/latest_statements.jsonl")


# %%
n = {m['name']: set() for s in statements_full if s['type'] == 'Complex' for m in s['members']}

for s in statements_full:
    if s['type'] == 'Complex':
        for m in set([i['name'] for i in s['members']]):
            n[m] = n[m] | set([s['matches_hash']])

x = [len(i) for __, i in n.items()]
i = np.argsort(x)[::-1]
x = np.array(x)[i]
y = np.array(list(n.keys()))[i]


# Histogram
fig, ax = plt.subplots(figsize = (12, 6), nrows = 1, ncols = 1)
ax.hist(x, bins = 100)
__ = plt.setp(ax, yscale = 'log', ylabel = 'Number of Agents/Nodes', xlabel = 'Number of Complexes that a Given Agent/Node is a Member')

# %%[markdown]
# * The agent/node 'YP_009227196' is a member of 703 statements of type `Complex`
# * The mean is 5.2 statements over 80,2223 statements and 15,299 agents/nodes
# 
#  
# Example: 'STK10'
# * Member of 3 complex statements
# * Agent of 6 non-complex statements
#    

# %%
%%time
z = 'STK10'
x = [s for s in statements_full if len([True for k in s.keys() if isinstance(s[k], dict) if 'name' in s[k].keys() if s[k]['name'] == z]) > 0]
y = [s for s in statements_full if len([True for k in s.keys() if isinstance(s[k], list) for l in s[k] if isinstance(l, dict) if 'name' in l.keys() if l['name'] == z]) > 0]
statements_complex = x + y

emlib.save_jsonl(statements_complex, './dist/temp/statements_complex.jsonl', preamble = None)


x = y = z = None
del x, y, z

# %%
# # Generate Nodes and Edges

# %%
%%time

nodes, edges, statements_complex_, __, evidences, documents = emlib.process_statements(statements_complex, model_id = -1)


# %%

# %%
# `nodes`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique node ID that is referenced by other files',
    'name': '<str> unique human-interpretable name of this node (from the `name` attribute in `latest_statements.jsonl`)',
    'db_refs': '<dict> database references of this node (from the `db_refs` attribute in `latest_statements.jsonl`)',
    'grounded': '<bool> whether this node is grounded to any database',
    'edge_ids_source': '<list of int> ID of edges that have this node as a source',
    'edge_ids_target': '<list of int> ID of edges that have this node as a target',
    'out_degree': '<int> out-degree of this node',
    'in_degree': '<int> in-degree of this node', 
}
emlib.save_jsonl(nodes, './dist/temp/nodes.jsonl', preamble = preamble)


# `edges`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique edge ID that is referenced by other files',
    'type': '<str> type of this edge (from `type` attribute in `latest_statements_covid19.jsonl`)',
    'belief': '<float> belief score of this edge (from `belief` attribute in `latest_statements.jsonl`)',
    'statement_id': '<str> unique statement id (from `matches_hash` in `latest_statements.jsonl`)', 
    'evidence_ids': '<list of int> IDs of the supporting evidences (as defined in `evidences.jsonl`)',
    'doc_ids': '<list of int> IDs of the supporting documents (as defined in `documents.jsonl`)',
    'source_id': '<int> ID of the source node (as defined in `nodes.jsonl`)' ,
    'target_id': '<int> ID of the target node (as defined in `nodes.jsonl`)',
    'tested': '<bool> whether this edge is tested'
}
emlib.save_jsonl(edges, './dist/temp/edges.jsonl', preamble = preamble)

# %%