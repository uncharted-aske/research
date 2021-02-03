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
# # Grounded Subgraph

nodes_grounded = [node for node in nodes_full if node['grounded']]

x = [node['id'] for node in nodes_grounded]
edges_grounded = [edge for edge in edges_full if (edge['source_id'] in x) & (edge['target_id'] in x)]

print(f"{len(nodes_grounded)} nodes and {len(edges_grounded)} edges are in the onto-grounded subgraph.")

# 35202 nodes and 411850 edges are in the grounded subgraph.

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

# 10 nodes and 8 edges are in the document subgraph.

# %%

x = (nodes_doc, edges_doc)


y = []
z = []
map_edge = {edge['id']: i for i, edge in enumerate(x[1])}
edge_ids_x = [edge['id'] for edge in x[1]]
for node in x[0]:

    # Check for each subgraph node if its referenced edges are contained amongst the subgraph edges
    edge_ids = node['edge_ids_source'] + node['edge_ids_target']
    edge_ids_inter = set(edge_ids_x) & set(edge_ids)
    if len(edge_ids_inter) < 1:
        y.append(node['id'])

    # Check for each subgraph node if its referenced edges (contained in the subgraph) actually contain itself
    node_ids = [x[1][map_edge[edge_id]]['source_id'] for edge_id in edge_ids_inter] + [x[1][map_edge[edge_id]]['target_id'] for edge_id in edge_ids_inter]
    if node['id'] not in node_ids:
        z.append(node['id'])

print(f"{len(y)} nodes have edges not present amongst the subgraph edges.")
print(f"{len(z)} nodes not present in the source/target IDs of its referenced edges in the subgraph.")


y = []
z = []
map_node = {node['id']: i for i, node in enumerate(x[0])}
node_ids_x = [node['id'] for node in x[0]]
for edge in x[1]:

    # Check for each subgraph edge if its referenced nodes are contained amongst the subgraph nodes
    node_ids = [edge['source_id']] + [edge['target_id']]
    node_ids_inter = set(node_ids_x) & set(node_ids)
    if len(node_ids_inter) < 1:
        y.append(edge['id'])

    # Check for each subgraph edge if its referenced nodes (in the subraph) actually contain itself
    edge_ids = [i for node_id in node_ids_inter for i in x[0][map_node[node_id]]['edge_ids_source']] + [i for node_id in node_ids_inter for i in x[0][map_node[node_id]]['edge_ids_target']]
    if edge['id'] not in edge_ids:
        z.append(edge['id'])

print(f"{len(y)} edges have nodes not present amongst the subgraph nodes.")
print(f"{len(z)} edges not present in the source/target IDs of its referenced nodes in the subgraph.")



node = edge = edge_ids = edge_ids_inter = edge_ids_x = node_ids = node_ids_inter = node_ids_x = map_edge = map_node = x = y = z = None
del node, edge, edge_ids, edge_ids_inter, edge_ids_x, x, y, z, map_node, map_edge, node_ids, node_ids_x, node_ids_inter


# %%[markdown]
# Save outputs

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
# emlib.save_jsonl(nodes_grounded, './dist/v3.1/grounded/nodes.jsonl', preamble = preamble)
# emlib.save_jsonl(nodes_belief, './dist/v3.1/belief/nodes.jsonl', preamble = preamble)
# emlib.save_jsonl(nodes_mitre, './dist/v3.1/mitre/nodes.jsonl', preamble = preamble)
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
# emlib.save_jsonl(edges_full, './dist/v3.1/full/edges.jsonl', preamble = preamble)
emlib.save_jsonl(edges_grounded, './dist/v3.1/grounded/edges.jsonl', preamble = preamble)
# emlib.save_jsonl(edges_belief, './dist/v3.1/belief/edges.jsonl', preamble = preamble)
# emlib.save_jsonl(edges_mitre, './dist/v3.1/mitre/edges.jsonl', preamble = preamble)
# emlib.save_jsonl(edges_doc, './dist/v3.1/doc/edges.jsonl', preamble = preamble)


# %%[markdown]
# # Generate Ontocats

# %%
with open('./data/bio_ontology_v1.5.json', 'r') as x:
    G_onto_JSON = json.load(x)

# Remove 'xref' links
G_onto_JSON['links'] = [link for link in G_onto_JSON['links'] if link['type'] != 'xref']

# %%
%%time

# Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, G_onto_JSON, nodes_doc)

# Reduce 'db_refs' of each model node to a single entry by namespace priority
nodes_doc, __ = emlib.reduce_nodes_db_refs(nodes_doc, namespaces)

# Calculate in-ontology paths
emlib.calculate_onto_root_path(nodes_doc, G_onto_JSON)

# Extract Ontological Categories
ontocats_doc = emlib.extract_ontocats(nodes_doc, G_onto_JSON)

# time: 1 m 17 s

# %%
%%time

# Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, G_onto_JSON, nodes_belief)

# Reduce 'db_refs' of each model node to a single entry by namespace priority
nodes_belief, __ = emlib.reduce_nodes_db_refs(nodes_belief, namespaces)

# Calculate in-ontology paths
emlib.calculate_onto_root_path(nodes_belief, G_onto_JSON)

# Extract Ontological Categories
ontocats_belief = emlib.extract_ontocats(nodes_belief, G_onto_JSON)

# time: 32 m 39 s

# %%
%%time

# Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, G_onto_JSON, nodes_grounded)

# Reduce 'db_refs' of each model node to a single entry by namespace priority
nodes_grounded, __ = emlib.reduce_nodes_db_refs(nodes_grounded, namespaces)

# Calculate in-ontology paths
emlib.calculate_onto_root_path(nodes_grounded, G_onto_JSON)

# Extract Ontological Categories
ontocats_grounded = emlib.extract_ontocats(nodes_grounded, G_onto_JSON)

# time: 3 h 12 m


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
%%time

# Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, G_onto_JSON, nodes_mitre)

# Reduce 'db_refs' of each model node to a single entry by namespace priority
nodes_mitre, __ = emlib.reduce_nodes_db_refs(nodes_mitre, namespaces)

# Calculate in-ontology paths
emlib.calculate_onto_root_path(nodes_mitre, G_onto_JSON)

# Extract Ontological Categories
ontocats_mitre = emlib.extract_ontocats(nodes_mitre, G_onto_JSON)

# time: 5 m 42 s

# %%

# Save new attributes of `nodes` as `nodeAtts`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique node ID that is defined in `nodes.jsonl`',
    'db_ref_priority': '<str> database reference from `db_refs` of `nodes.jsonl`, that is used by the INDRA ontology v1.5', 
    'grounded_onto': '<bool> whether this model node is grounded to something that exists within the ontology', 
    'ontocat_level': '<int> the level of the most fine-grained ontology node/category to which this model node was mapped (`-1` if not mappable, `0` if root)', 
    'ontocat_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)', 
    'grounded_cluster': '<bool> whether this model node is grounded to any cluster', 
    'cluster_level': '<int> the level of the most fine-grained cluster at which this model node was mapped (`-1` if not mappable, `0` if root)', 
    'cluster_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)', 
}
# emlib.save_jsonl(nodes_doc, './dist/v3.1/doc/nodeAtts.jsonl', preamble = preamble)
# emlib.save_jsonl(nodes_belief, './dist/v3.1/belief/nodeAtts.jsonl', preamble = preamble)
# emlib.save_jsonl(nodes_mitre, './dist/v3.1/mitre/nodeAtts.jsonl', preamble = preamble)
# emlib.save_jsonl(nodes_full, './dist/v3.1/full/nodeAtts.jsonl', preamble = preamble)
emlib.save_jsonl(nodes_grounded, './dist/v3.1/grounded/nodeAtts.jsonl', preamble = preamble)


# Save `ontocats`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique ID for the ontological category that is referenced by other files',
    'ref': '<str> unique reference ID of this category (as given by the INDRA Ontology v1.5)',
    'name': '<str> name of this category (as given by the INDRA Ontology v1.5)',
    'size': '<int> number of model nodes that were mapped to this category and its children',
    'level': '<int> number of hops to reach the local root (`0` if root)',
    'parent_id': '<int> ID of the parent of this category in the ontology',
    'children_ids': '<array of int> unordered list of the child category IDs',
    'node_ids': '<array of int> unordered list of IDs from model nodes in the membership of this category',
    'node_ids_direct': '<array of int> node_ids but only model nodes which were directly mapped to this category and not any of the child categories',
    'hyperedge_ids': '<array of int> unordered list of hyperedge IDs (see `hyperedges.jsonl`) that are within this category',
}
# emlib.save_jsonl(ontocats_doc, './dist/v3.1/doc/ontocats.jsonl', preamble = preamble)
# emlib.save_jsonl(ontocats_belief, './dist/v3.1/belief/ontocats.jsonl', preamble = preamble)
# emlib.save_jsonl(ontocats_mitre, './dist/v3.1/mitre/ontocats.jsonl', preamble = preamble)
# emlib.save_jsonl(ontocats_full, './dist/v3.1/full/ontocats.jsonl', preamble = preamble)
emlib.save_jsonl(ontocats_grounded, './dist/v3.1/grounded/ontocats.jsonl', preamble = preamble)

# %%


