# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content: 
# * Load full EMMAA Covid-19 model graph
# * Define objects (`documents`, `evidences`, `paths`, `nodes`, `edges`, `G`, `ontocats`)
# * Create subgraphs ("grounded_onto", "tested_mitre", "belief095", "doc")
# * Generate the associated output files

# %%
import json
import pickle
import time
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
statements_full = {}
with open('./data/covid19-snapshot_dec8-2020/source/latest_statements_covid19.json', 'r') as x:
    statements_full = json.load(x)

x = None
del x

# %%[markdown]
# Load Mitre-Tested Paths
paths_mitre = emlib.load_jsonl('./data/covid19-snapshot_dec8-2020/source/covid19_mitre_tests_latest_paths.jsonl')


# %%[markdown]
# # Full Graph

# %%
%%time

model_id = 0
nodes_full, edges_full, statements_full_, paths_mitre_, evidences_full, documents_full = emlib.process_statements(statements_full, paths = paths_mitre, model_id = model_id)

# 379717 statements -> 375575 processed statements.
# Found 478598 evidences and 85959 documents.
# Found 42198 nodes and 429976 edges.

# time: 2 m 2 s

# %%[markdown]
# # Onto-Grounded Subgraph

nodes_grounded = [node for node in nodes_full if node['grounded_onto']]

x = [node['id'] for node in nodes_grounded]
edges_grounded = [edge for edge in edges_full if (edge['source_id'] in x) | (edge['target_id'] in x)]

print(f"{len(nodes_grounded)} nodes and {len(edges_grounded)} edges are in the onto-grounded subgraph.")

# 35202 nodes and 428155 edges are in the onto-grounded subgraph.

# %%[markdown]
# # High-Belief Subgraph

# Filter `edges` by belief score
edges_belief = [edge for edge in edges_full if edge['belief'] >= 0.95]

# Filter `nodes` by `edges`
x = set([edge['id'] for edge in edges_belief])
nodes_belief = [node for node in nodes_full if (len(set(node['edge_ids_source']) & x) > 0) | (len(set(node['edge_ids_target']) & x) > 0)]

print(f"{len(nodes_belief)} nodes and {len(edges_belief)} edges are in the high-belief subgraph.")

# 6289 nodes and 33380 edges are in the high-belief subgraph.

# %%[markdown]
# # Mitre-Tested Subgraph 

nodes_mitre, edges_mitre, __ = emlib.intersect_graph_paths(nodes_full, edges_full, paths_mitre_)

print(f"{len(nodes_mitre)} nodes and {len(edges_mitre)} edges are in the Mitre-tested subgraph.")

# 1687 nodes and 5034 edges are in the Mitre-tested subgraph.

# %%[markdown]
# # Document Subgraph

# Define document of interest
doc_doi = '10.1016/j.immuni.2020.04.003'.upper()
map_dois_doc_ids = {doc['DOI']: doc['id'] for doc in documents_full}
doc_id = map_dois_doc_ids[doc_doi]

# Filter `edges` by `doc_id`
edges_doc = [edge for edge in edges_full if doc_id in edge['doc_ids']]

# Filter `nodes` by `edges`
x = set([edge['source_id'] for edge in edges_doc] + [edge['target_id'] for edge in edges_doc])
nodes_doc = [node for node in nodes_full if node['id'] in x]

print(f"{len(nodes_doc)} nodes and {len(edges_doc)} edges are in the document subgraph.")

# 4 nodes and 8 edges are in the document subgraph.

# %%%%

# PROBLEM

# This is empty
[edge for edge in edges_doc if edge['id'] in nodes_doc[0]['edge_ids_source']] + [edge for edge in edges_doc if edge['id'] in nodes_doc[0]['edge_ids_target']]

# while nodes_doc[0] is somehow included in edges_doc['source_id'], edges_doc['target_id'], i = [64288, 198299, 198300]




# %%

# `evidences`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique evidence ID that is referenced by other files',
    'text': '<str>  (from the `text` attribute in `latest_statements_covid19.jsonl`)',
    'text_refs': '<dict where key = <str> identifier type and value = <str> document identifier> reference of source document (from the `text_refs` attribute in `latest_statements_covid19.jsonl`)',
    'doc_id': '<int> ID of the source document (see `documents.jsonl`)',
    'statement_ids': '<str> ID of supported statement (from `matches_hash` in `latest_statements_covid19.jsonl`)',
    'edge_ids': '<list of int> IDs of supported edges',
}
emlib.save_jsonl(evidences_full, './dist/v3.1/full/evidences.jsonl', preamble = preamble)


# `documents`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique document ID that is referenced by other files',
    'DOI': '<str> DOI identifier of this document (all caps)'
}
emlib.save_jsonl(documents_full, './dist/v3.1/full/documents.jsonl', preamble = preamble)


# `nodes`
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
emlib.save_jsonl(nodes_full, './dist/v3.1/full/nodes.jsonl', preamble = preamble)
emlib.save_jsonl(nodes_grounded, './dist/v3.1/grounded/nodes.jsonl', preamble = preamble)
emlib.save_jsonl(nodes_belief, './dist/v3.1/belief/nodes.jsonl', preamble = preamble)
emlib.save_jsonl(nodes_mitre, './dist/v3.1/mitre/nodes.jsonl', preamble = preamble)
emlib.save_jsonl(nodes_doc, './dist/v3.1/doc/nodes.jsonl', preamble = preamble)


# `edges`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique edge ID that is referenced by other files',
    'type': '<str> type of this edge (from `type` attribute in `latest_statements_covid19.jsonl`)',
    'belief': '<float> belief score of this edge (from `belief` attribute in `latest_statements_covid19.jsonl`)',
    'statement_id': '<str> unique statement id (from `matches_hash` in `latest_statements_covid19.jsonl`)', 
    'evidence_ids': '<list of int> IDs of the supporting evidences (as defined in `evidences.jsonl`)',
    'doc_ids': '<list of int> IDs of the supporting documents (as defined in `documents.jsonl`)',
    'source_id': '<int> ID of the source node (as defined in `nodes.jsonl`)' ,
    'target_id': '<int> ID of the target node (as defined in `nodes.jsonl`)',
    'tested': '<bool> whether this edge is tested'
}
emlib.save_jsonl(edges_full, './dist/v3.1/full/edges.jsonl', preamble = preamble)
emlib.save_jsonl(edges_grounded, './dist/v3.1/grounded/edges.jsonl', preamble = preamble)
emlib.save_jsonl(edges_belief, './dist/v3.1/belief/edges.jsonl', preamble = preamble)
emlib.save_jsonl(edges_mitre, './dist/v3.1/mitre/edges.jsonl', preamble = preamble)
emlib.save_jsonl(edges_doc, './dist/v3.1/doc/edges.jsonl', preamble = preamble)


# %%

