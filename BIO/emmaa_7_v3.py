# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Idea: 
# * New ontology (`v1.5`) from Ben Gyori
# * New COVID-19 model graph (`dec8-2020`)
# * Keep 2-member complex and 3-member conversion statements
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
# # Extract Nodes and Edges from Statements and Tested Paths

# %%
%%time

# Load statements
statements_all = {}
with open('./data/covid19-snapshot_dec8-2020/source/latest_statements_covid19.json', 'r') as x:
    statements_all = json.load(x)

# Load tested paths
paths_mitre_all = emlib.load_jsonl('./data/covid19-snapshot_dec8-2020/source/covid19_mitre_tests_latest_paths.jsonl')

model_id = 2
nodes_model, edges_model, statements_model, paths_mitre = emlib.process_statements(statements_all, paths = paths_mitre_all, model_id = model_id)
# 379717 statements -> 375575 processed statements.
# Found 42198 nodes and 434006 edges.
# 3714 paths -> 3714 processed paths.
# Found 5112 tested edges.

x = statements_all = paths_mitre_all = None
del x, statements_all, paths_mitre_all

# time: 2 m 6 s

# %%

print(f"{len({node['id']: 0 for node in nodes_model})} == {len(nodes_model)}")
print(f"{len({edge['id']: 0 for edge in edges_model})} == {len(edges_model)}")

# 42198 == 42198
# 429976 == 429976

# %%
# ## Generate NetworkX MultiDiGraph from `nodes` and `edges`

# %%
%%time

G_model = emlib.generate_nx_object(nodes_model, edges_model)

# Save as `.pkl`
with open('./dist/v3/G_model.pkl', 'wb') as x:
    pickle.dump(G_model, x)

G_model = None
del G_model

# time: 3.19 s

# %%
# ## Reduce Model Graph to Tested Paths

# %%
%%time

# Find intersection between the 'paths' subgraph and the model graph
nodes_mitre, edges_mitre, __ = emlib.intersect_graph_paths(nodes_model, edges_model, paths_mitre)
num_nodes = len(nodes_mitre)
num_edges = len(edges_mitre)
# 3714 paths, 42198 nodes, and 429976 edges in total.
# 3714 paths, 1687 nodes, and 5034 edges in the intersection.


print(f"{len({node['id']: 0 for node in nodes_model})} == {len(nodes_model)}")
print(f"{len({edge['id']: 0 for edge in edges_model})} == {len(edges_model)}")

print(f"{len({node['id']: 0 for node in nodes_mitre})} == {len(nodes_mitre)}")
print(f"{len({edge['id']: 0 for edge in edges_mitre})} == {len(edges_mitre)}")


# 42198 == 42198
# 429976 == 429976
# 1687 == 1687
# 5034 == 5034


# Reset node id in `nodes` and `edges` to maintain Grafer optimization
__, __ = emlib.reset_node_edge_ids(nodes_mitre, edges_mitre)


print(f"{len({node['id']: 0 for node in nodes_model})} == {len(nodes_model)}")
print(f"{len({edge['id']: 0 for edge in edges_model})} == {len(edges_model)}")

print(f"{len({node['id']: 0 for node in nodes_mitre})} == {len(nodes_mitre)}")
print(f"{len({edge['id']: 0 for edge in edges_mitre})} == {len(edges_mitre)}")


# 40590 == 42198
# 424971 == 429976
# 1687 == 1687
# 5034 == 5034


# time: 115 ms

# %%
%%time

# Save `nodes`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique node ID that is referenced by other files',
    'name': '<str> unique human-interpretable name of this node (from the `name` attribute in `latest_statements_covid19.jsonl`)',
    'db_refs': '<dict> database references of this node (from the `db_refs` attribute in `latest_statements_covid19.jsonl`)',
    'grounded': '<bool> whether this node is grounded to any database',
    'edge_ids_source': '<list of int> ID of edges that have this node as a source',
    'edge_ids_target': '<list of int> ID of edges that have this node as a target',
    'out_degree': '<int> out-degree of this node',
    'in_degree': '<int> in-degree of this node', 
}
emlib.save_jsonl(nodes_model, './dist/v3/nodes_model.jsonl', preamble = preamble)
emlib.save_jsonl(nodes_mitre, './dist/v3/nodes_mitre.jsonl', preamble = preamble)

# Save `edges`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique edge ID that is referenced by other files',
    'type': '<str> type of this edge (from `type` attribute in `latest_statements_covid19.jsonl`)',
    'belief': '<float> belief score of this edge (from `belief` attribute in `latest_statements_covid19.jsonl`)',
    'statement_id': '<str> unique statement id (from `matches_hash` in `latest_statements_covid19.jsonl`)',
    'source_id': '<int> ID of the source node (as defined in `nodes.jsonl`)' ,
    'target_id': '<int> ID of the target node (as defined in `nodes.jsonl`)',
    'tested': '<bool> whether this edge is tested'
}
emlib.save_jsonl(edges_model, './dist/v3/edges_model.jsonl', preamble = preamble)
emlib.save_jsonl(edges_mitre, './dist/v3/edges_mitre.jsonl', preamble = preamble)


# time: 14.8 s

# %%
# Reload `nodes` and `edges` if necessary
if False:
    nodes_model = emlib.load_jsonl('./dist/v3/nodes_model.jsonl', remove_preamble = True)
    edges_model = emlib.load_jsonl('./dist/v3/edges_model.jsonl', remove_preamble = True)

if False:
    nodes_mitre = emlib.load_jsonl('./dist/v3/nodes_mitre.jsonl', remove_preamble = True)
    edges_mitre = emlib.load_jsonl('./dist/v3/edges_mitre.jsonl', remove_preamble = True)


# %%


