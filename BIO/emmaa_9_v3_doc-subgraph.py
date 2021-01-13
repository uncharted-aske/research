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
# # Define the document of interest
# doc = {'id_type': 'DOI', 'id': '10.1016/j.immuni.2020.04.003'.upper()}
doc = {'id_type': 'PMID', 'id': '32325025'}

# %%[markdown]
# # Load Statement Data
statements_all = {}
with open('./data/covid19-snapshot_dec8-2020/source/latest_statements_covid19.json', 'r') as x:
    statements_all = json.load(x)

x = None
del x

# %%[markdown]
# # Filter Statements by Reference
statements_doc = [statements_all[j] for j in set([i for i, s in enumerate(statements_all) for ev in s['evidence'] if ('text_refs' in ev.keys()) if (doc['id_type'] in ev['text_refs'].keys()) if ev['text_refs'][doc['id_type']] == doc['id']])]

statements_all = None
del statements_all

# %%
print(f"{len(statements_doc)} statements found to reference the document with {doc['id_type']} = {doc['id']}")
# 7 statements found to reference the document with DOI = 10.1016/J.IMMUNI.2020.04.003

# %%[markdown]
# # Load the Nodes and Edges Data of the Model Graph

nodes_model = emlib.load_jsonl('./dist/v3/nodes_model.jsonl', remove_preamble = True)
edges_model = emlib.load_jsonl('./dist/v3/edges_model.jsonl', remove_preamble = True)

# %%[markdown]
# # Filter the Node and Edge List

x = [s['matches_hash'] for s in statements_doc]
edges_doc = [edge for edge in edges_model if edge['statement_id'] in x]
nodes_doc = [nodes_model[j] for j in set([edge['source_id'] for edge in edges_doc] + [edge['target_id'] for edge in edges_doc])]

nodes_model = edges_model = x = None
del nodes_model, edges_model, x

# %%
print(f"{len(nodes_doc)} nodes and {len(edges_doc)} edges found to be associated with the document with {doc['id_type']} = {doc['id']}")
# 10 nodes and 8 edges found to be associated with the document with DOI = 10.1016/J.IMMUNI.2020.04.003

# %%
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
emlib.save_jsonl(nodes_doc, './dist/v3/doc/nodes_doc.jsonl', preamble = preamble)

# %%
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
emlib.save_jsonl(edges_doc, './dist/v3/doc/edges_doc.jsonl', preamble = preamble)

preamble = None
del preamble

# %%
# Generate a `G.pkl`

G_doc = emlib.generate_nx_object(nodes_doc, edges_doc)
with open('./dist/v3/doc/G_doc.pkl', 'wb') as x:
    pickle.dump(G_doc, x)


map_ids_edges = {edge['id']: i for i, edge in enumerate(edges_doc)}
# x = [max([edges_doc[map_ids_edges[edge_id]]['belief'] for edge_id in node[1]]) if len(node[1]) > 0 else 0.0 for node in G_doc.nodes.data('edge_ids_target')]
# __, __, fig, __ = emlib.generate_nx_layout(G = G_doc, layout = 'spring', layout_atts = {'k': 0.1}, plot = True, plot_atts = {'node_color': x, 'vmin': 0.0, 'vmax': 1.0, 'cmap': 'cool'})
__, __, fig, __ = emlib.generate_nx_layout(G = G_doc, layout = 'spring', layout_atts = {'k': 0.1}, plot = True, plot_atts = {'width': 1, 'node_size': 100, 'labels': {node: G_doc.nodes[node]['name'] for node in G_doc.nodes()}, 'cmap': 'cool'})
fig.savefig('./figures/v3/doc_subgraph_layout_degree.png', dpi = 150)


G_doc = x = fig = None
del x, fig, G_doc


# %%

with open('./data/bio_ontology_v1.5.json', 'r') as x:
    G_onto_JSON = json.load(x)

# Remove 'xref' links
G_onto_JSON['links'] = [link for link in G_onto_JSON['links'] if link['type'] != 'xref']

# Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, G_onto_JSON, nodes_doc)

# Reduce 'db_refs' of each model node to a single entry by namespace priority
nodes_doc, __ = emlib.reduce_nodes_db_refs(nodes_doc, namespaces)

# Calculate in-ontology paths
emlib.calculate_onto_root_path(nodes_doc, G_onto_JSON)

# Extract Ontological Categories
ontocats_doc = emlib.extract_ontocats(nodes_doc, G_onto_JSON)


# Generate Hyperedges
hyperedges_doc = emlib.generate_hyperedges(nodes_doc, edges_doc, ontocats_doc)


# Compute Layout using Ontological Categories and Hyperedges
__ = emlib.generate_onto_layout(nodes_doc, ontocats_doc, hyperedges_doc, plot = True)

# %%
# # Save Outputs

# %%
# Save layout of `nodes` as `nodeLayout`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique node ID that is defined in `nodes.jsonl`',
    'x': '<float> position of the node in the graph layout',
    'y': '<float> position of the node in the graph layout',
    'z': '<float> position of the node in the graph layout',
}
emlib.save_jsonl(nodes_doc, './dist/v3/doc/nodeLayout_doc.jsonl', preamble = preamble)


# Save new attributes of `nodes` as `nodeAtts`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique node ID that is defined in `nodes.jsonl`',
    'db_ref_priority': '<str> database reference from `db_refs` of `nodes.jsonl`, that is used by the INDRA ontology v1.5', 
    'grounded_onto': '<bool> whether this model node is grounded to something that exists within the ontology', 
    'ontocat_level': '<int> level of the ontology node/category to which this model node was mapped (`-1` if not mappable, `0` if root)', 
    'ontocat_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)', 
    'cluster_ids': '<array of int> (placeholder for cluster)'
}
emlib.save_jsonl(nodes_doc, './dist/v3/doc/nodeAtts_doc.jsonl', preamble = preamble)


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
emlib.save_jsonl(ontocats_doc, './dist/v3/doc/ontocats_doc.jsonl', preamble = preamble)


# Save layout of ontocats as `ontocatLayout`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique ID for the ontological category that is referenced by other files',
    'x': '<float> position of the node in the graph layout',
    'y': '<float> position of the node in the graph layout',
    'z': '<float> position of the node in the graph layout'
}
emlib.save_jsonl(ontocats_doc, './dist/v3/doc/ontocatLayout_doc.jsonl', preamble = preamble)


# Save `hyperedges`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique hyperedge ID that is referenced by other files',
    'level': '<float> ontological level of this hyperedge (taken from the source ontocat)',
    'size': '<int> number of model edges that is aggregated here',
    'edge_ids': '<array of int> unordered list of edge ID of the underlying model edges (see `edges.jsonl`)',
    'source_type': '<str> object type of the source (only `ontocat`)',
    'source_id': '<int> ID of the source (defined in `ontocats.jsonl`)' ,
    'target_type': '<str> object type of the target (either `ontocat` or `node`)',
    'target_id': '<int> ID of the target (defined in either `ontocats.jsonl` or `nodes.jsonl`)',
}
emlib.save_jsonl(hyperedges_doc, './dist/v3/doc/hyperedges_doc.jsonl', preamble = preamble)



# %%
