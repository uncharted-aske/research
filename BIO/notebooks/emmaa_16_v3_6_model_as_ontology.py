# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content: 
# * Use BIO Ontology v1.8 export v1
# * Create a mock model (one node per ontocat, one edge per ontolink)

# %%
import re
import json
import pickle
import time
import datetime
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

import emmaa_lib as emlib
import importlib
# importlib.reload(emlib)

# %%
np.random.seed(0)

# %%[markdown]
# # Load Ontology

i = '/home/nliu/projects/aske/research/BIO/data'
with open(f"{i}/ontologies/bio_ontology_v1.8_export_v1.json", "r") as f:
    G_onto_JSON = json.load(f)

# %%
# Remove 'xref' links
G_onto_JSON['links'] = [link for link in G_onto_JSON['links'] if link['type'] != 'xref']

# %%

# %%[markdown]
# # Generate `nodes`

nodes = [{
    'model_id': -1, 
    'id': i, 
    'name': f"{node['id'] if 'name' not in node.keys() else node['id'] if node['name'] == None else node['id'] if len(node['name']) == 0 else node['name']}",
    'db_refs': None,
    'db_ref_priority': node['id'],
    'grounded': True,
    'edge_ids_source': [],
    'edge_ids_target': [],
    'out_degree': 0,
    'in_degree': 0,
    'grounded_onto': True,
    'ontocat_level': 0,
    'ontocat_ids': [0]
    } for i, node in enumerate(G_onto_JSON['nodes'])]

# %%[markdown]
# # Generate `ontocats`

ontocats = [{
    'model_id': -1,
    'id': 0,
    'ref': 'all',
    'name': 'all',
    'size': len(nodes),
    'level': 0,
    'parent_id': None,
    'children_ids': None,
    'node_ids': [i for i in range(len(nodes))],
    'node_ids_direct': [i for i in range(len(nodes))]
}]

# %%[markdown]
# # Generate `edges`

map_nodes_ids = {node['db_ref_priority']: i for i, node in enumerate(nodes)}

edges = [{
    'model_id': -1,
    'id': i,
    'type': edge['type'],
    'belief': 1.0,
    'statement_id': None, 
    'evidence_ids': [],
    'doc_ids': [],
    'source_id': map_nodes_ids[edge['source']],
    'target_id': map_nodes_ids[edge['target']],
    'tested': False,
    'curated': 3
    } for i, edge in enumerate(G_onto_JSON['links'])]

# %%
# * 2,246,201 nodes
# * 2,148,015 edges
# * 1 ontocat

# %%[markdown]
# # Save Files


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
emlib.save_jsonl(nodes, './dist/v3.6/nodes.jsonl', preamble = preamble)



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
    'tested': '<bool> whether the underlying statement of this edge is pass the Mitre test',
    'curated': '<int> curation status of the underlying statement of this edge (`incorrect` = `0`, `correct` = `1`, `partial` = `2`, `uncurated` = `3`)'
}
emlib.save_jsonl(edges, './dist/v3.6/edges.jsonl', preamble = preamble)



preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique node ID that is defined in `nodes.jsonl`',
    'type': '<str> node type (currently set to the `name` of the ancestor ontocat, `None` if ungrounded)',
    'db_ref_priority': '<str> database reference from `db_refs` of `nodes.jsonl`, that is used by the INDRA ontology', 
    'grounded_onto': '<bool> whether this model node is grounded to something that exists within the ontology', 
    'ontocat_level': '<int> the level of the most fine-grained ontology node/category to which this model node was mapped (`-1` if not mappable, `0` if root)', 
    'ontocat_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)' 
}
emlib.save_jsonl(nodes, './dist/v3.6/nodeAtts.jsonl', preamble = preamble)



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
    'node_ids_direct': '<array of int> node_ids but only model nodes which were directly mapped to this category and not any of the child categories'
}
emlib.save_jsonl(ontocats, './dist/v3.6/ontocats.jsonl', preamble = preamble)

# %%





