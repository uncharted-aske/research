# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content: 
# * Same process as v3.3
# * Latest EMMAA Covid-19 model statements (2021-02-08)
# * Added Ontology v1.7 export v3

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


# %%[markdown]
# Load Paths from Mitre Test
paths_mitre = emlib.load_jsonl('/home/nliu/projects/aske/research/BIO/data/models/covid19/2021-02-09/covid19_mitre_tests_latest_paths.jsonl')


# %%[markdown]
# # Full Graph

# %%
%%time

model_id = 2
nodes_full, edges_full, statements_full_, paths_mitre_, evidences_full, documents_full = emlib.process_statements(statements_full, paths = paths_mitre, model_id = model_id)


# 395394 statements -> 391070 processed statements.
# Found 501203 evidences and 92456 documents.
# Found 44104 nodes and 448723 edges.
# 4289 paths -> 4289 processed paths.
# Found 5521 tested edges.

# time: 2 m 30 s

# %%[markdown]
# # Grounded Subgraph

# Filter nodes by 'groundedness'
nodes_grounded = [node for node in nodes_full if node['grounded']]

x = [node['id'] for node in nodes_grounded]
edges_grounded = [edge for edge in edges_full if (edge['source_id'] in x) & (edge['target_id'] in x)]

print(f"{len(nodes_grounded)} nodes and {len(edges_grounded)} edges are in the grounded subgraph.")

# 36038 nodes and 427735 edges are in the grounded subgraph.


# %%[markdown]
# Save outputs

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
emlib.save_jsonl(evidences_full, './dist/v3.4/evidences.jsonl', preamble = preamble)


# `documents`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique document ID that is referenced by other files',
    'DOI': '<str> DOI identifier of this document (all caps)'
}
emlib.save_jsonl(documents_full, './dist/v3.4/documents.jsonl', preamble = preamble)

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
emlib.save_jsonl(nodes_full, './dist/v3.4/nodes.jsonl', preamble = preamble)


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
emlib.save_jsonl(edges_full, './dist/v3.4/edges.jsonl', preamble = preamble)


# %%[markdown]
# # Generate Ontocats

# %%
with open('./data/ontologies/bio_ontology_v1.7_export_v3.json', 'r') as x:
    G_onto_JSON = json.load(x)

# Remove 'xref' links
G_onto_JSON['links'] = [link for link in G_onto_JSON['links'] if link['type'] != 'xref']


# %%
%%time

# Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, G_onto_JSON, nodes_full)

# Reduce 'db_refs' of each model node to a single entry by namespace priority
nodes_full, __ = emlib.reduce_nodes_db_refs(nodes_full, namespaces)

# Calculate in-ontology paths
emlib.calculate_onto_root_path(nodes_full, G_onto_JSON)

# Extract Ontological Categories
ontocats_full = emlib.extract_ontocats(nodes_full, G_onto_JSON)

# time: 3 h 1 s

# %%

# Save new attributes of `nodes` as `nodeAtts`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique node ID that is defined in `nodes.jsonl`',
    'db_ref_priority': '<str> database reference from `db_refs` of `nodes.jsonl`, that is used by the INDRA ontology', 
    'grounded_onto': '<bool> whether this model node is grounded to something that exists within the ontology', 
    'ontocat_level': '<int> the level of the most fine-grained ontology node/category to which this model node was mapped (`-1` if not mappable, `0` if root)', 
    'ontocat_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)', 
    'grounded_cluster': '<bool> whether this model node is grounded to any cluster', 
    'cluster_level': '<int> the level of the most fine-grained cluster at which this model node was mapped (`-1` if not mappable, `0` if root)', 
    'cluster_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)', 
}
emlib.save_jsonl(nodes_full, './dist/v3.4/nodeAtts.jsonl', preamble = preamble)


# Save `ontocats`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique ID for the ontological category that is referenced by other files',
    'ref': '<str> unique reference ID of this category (as given by the INDRA Ontology)',
    'name': '<str> name of this category (as given by the INDRA Ontology)',
    'size': '<int> number of model nodes that were mapped to this category and its children',
    'level': '<int> number of hops to reach the local root (`0` if root)',
    'parent_id': '<int> ID of the parent of this category in the ontology',
    'children_ids': '<array of int> unordered list of the child category IDs',
    'node_ids': '<array of int> unordered list of IDs from model nodes in the membership of this category',
    'node_ids_direct': '<array of int> node_ids but only model nodes which were directly mapped to this category and not any of the child categories',
    'hyperedge_ids': '<array of int> unordered list of hyperedge IDs (see `hyperedges.jsonl`) that are within this category',
}
emlib.save_jsonl(ontocats_full, './dist/v3.4/ontocats.jsonl', preamble = preamble)


# %%


