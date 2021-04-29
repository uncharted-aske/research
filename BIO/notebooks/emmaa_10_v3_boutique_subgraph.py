# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content: 
# * Load text sentences to be used to build the boutique model graph
# * Use INDRA API to generate compatible statement objects
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
import requests

import emmaa_lib as emlib
import importlib
# importlib.reload(emlib)

# %%
np.random.seed(0)

# %%[markdown]
# # Load Boutique Text Sentences
# 
# These text sentences were the result of querying a paper with `PMID = 32325025` on the [INDRA Database](https://db.indra.bio/search).

texts_boutique = []
with open('./data/covid19-boutique_jan14-2021.txt', 'r') as f:
    # texts_boutique = [line.rstrip('\n') for line in f]
    texts_boutique = f.read().splitlines()

f = None
del f

# %%[markdown]
# # Generate INDRA Statement Objects

# %%
%%time

statements_boutique = []
for text in texts_boutique:

    # Get raw statements
    request_body = requests.post('http://api.indra.bio:8000/reach/process_text', json = {'text': text})
    s_raw = request_body.json()['statements']

    if len(s_raw) > 0:

        # Assemble
        assembly_pipeline = [{'function': 'map_grounding'}, {'function': 'map_sequence'}, {'function': 'run_preassembly'}]
        request_body = requests.post('http://api.indra.bio:8000/preassembly/pipeline', json = {'statements': s_raw, 'pipeline': assembly_pipeline})
        s = request_body.json()['statements']
        
        # Flag to distinguish from machine-read statements
        s[0]['evidence'][0]['text_refs'] = {}

        statements_boutique = statements_boutique + s


print(f"{len(statements_boutique)} INDRA statements are generated from the {len(texts_boutique)} boutique text sentences.")
# 49 INDRA statements are generated from the 68 boutique text sentences.


text = request_body = s_raw = assembly_pipeline = None
del text, request_body, s_raw, assembly_pipeline

# time: 11.1 s

# %%[markdown]
# # Append to Full Model Graph

# %%
# Load statements from the full model graph
statements_all = {}
with open('./data/covid19-snapshot_dec8-2020/source/latest_statements_covid19.json', 'r') as x:
    statements_all = json.load(x)

x = None
del x


# Extend
statements_all.extend(statements_boutique)

print(f"There are now {len(statements_all)} INDRA statements in total in the full model graph.")
# There are now 379766 INDRA statements in total in the full model graph.

# %%[markdown]
# # Extract Node and Edge Lists

# %%
# Generate new node and edge list for the extended model graph
model_id = 3
nodes_model, edges_model, __, __ = emlib.process_statements(statements_all, paths = [], model_id = model_id)

# Extract the node and edge list of the boutique subgraph
x = [s['matches_hash'] for s in statements_boutique]
edges_boutique = [edge for edge in edges_model if edge['statement_id'] in x]
nodes_boutique = [nodes_model[j] for j in set([edge['source_id'] for edge in edges_boutique] + [edge['target_id'] for edge in edges_boutique])]


print(f"{len(nodes_model)} nodes and {len(edges_model)} edges are extracted from the {len(statements_all)} extended INDRA statements.")
# 42206 nodes and 430025 edges are extracted from the 379766 extended INDRA statements.

print(f"{len(nodes_boutique)} nodes and {len(edges_boutique)} edges are extracted from the {len(statements_boutique)} boutique INDRA statements.")
# 36 nodes and 59 edges are extracted from the 49 boutique INDRA statements.


# %% [markdown]
# # Save Data

# %%
# Save `statements`
emlib.save_jsonl(statements_all, './dist/v3/boutique/statements_model.jsonl', preamble = {})
emlib.save_jsonl(statements_boutique, './dist/v3/boutique/statements_boutique.jsonl', preamble = {})


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
emlib.save_jsonl(nodes_model, './dist/v3/boutique/nodes_model.jsonl', preamble = preamble)
emlib.save_jsonl(nodes_boutique, './dist/v3/boutique/nodes_boutique.jsonl', preamble = preamble)



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
emlib.save_jsonl(edges_model, './dist/v3/boutique/edges_model.jsonl', preamble = preamble)
emlib.save_jsonl(edges_boutique, './dist/v3/boutique/edges_boutique.jsonl', preamble = preamble)

preamble = None
del preamble

# %%
# Generate a `G.pkl`

G_boutique = emlib.generate_nx_object(nodes_boutique, edges_boutique)
with open('./dist/v3/boutique/G_boutique.pkl', 'wb') as x:
    pickle.dump(G_boutique, x)


map_ids_edges = {edge['id']: i for i, edge in enumerate(edges_boutique)}
# __, __, fig, __ = emlib.generate_nx_layout(G = G_boutique, layout = 'spring', layout_atts = {'k': 0.1}, plot = True, plot_atts = {'node_color': x, 'vmin': 0.0, 'vmax': 1.0, 'cmap': 'cool'})
__, __, fig, __ = emlib.generate_nx_layout(G = G_boutique, layout = 'spring', layout_atts = {'k': 0.5}, plot = True, 
    plot_atts = {'verticalalignment': 'top','font_size': 10, 'with_labels': True, 'arrows': True, 'node_size': [50 * node[1] for node in G_boutique.degree()], 'width': 1, 'labels': {node: G_boutique.nodes[node]['name'] for node in G_boutique.nodes()}, 'cmap': 'cool'})
fig.savefig('./figures/v3/boutique_subgraph_layout_degree.png', dpi = 150)


G_boutique = x = fig = None
del x, fig, G_boutique

# %%[markdown]
# # Impose Ontology and Generate Hyperedges

# %%
%%time

with open('./data/bio_ontology_v1.5.json', 'r') as x:
    G_onto_JSON = json.load(x)

# Remove 'xref' links
G_onto_JSON['links'] = [link for link in G_onto_JSON['links'] if link['type'] != 'xref']

# Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, G_onto_JSON, nodes_boutique)

# Reduce 'db_refs' of each model node to a single entry by namespace priority
nodes_boutique, __ = emlib.reduce_nodes_db_refs(nodes_boutique, namespaces)

# Calculate in-ontology paths
emlib.calculate_onto_root_path(nodes_boutique, G_onto_JSON)

# Extract Ontological Categories
ontocats_boutique = emlib.extract_ontocats(nodes_boutique, G_onto_JSON)


# Generate Hyperedges
hyperedges_boutique = emlib.generate_hyperedges(nodes_boutique, edges_boutique, ontocats_boutique)


# Compute Layout using Ontological Categories and Hyperedges
__ = emlib.generate_onto_layout(nodes_boutique, ontocats_boutique, hyperedges_boutique, plot = True)


G_onto_JSON = None
del G_onto_JSON

# time: 1 m 30 s


# %%[markdown]
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
emlib.save_jsonl(nodes_boutique, './dist/v3/boutique/nodeLayout_boutique.jsonl', preamble = preamble)


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
emlib.save_jsonl(nodes_boutique, './dist/v3/boutique/nodeAtts_boutique.jsonl', preamble = preamble)


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
    'hyperedge_ids': '<array of int> unordered list of hyperedge IDs (see `hyperedges.jsonl`) that are within this category (i.e. the source is a child of this ontocat)',
}
emlib.save_jsonl(ontocats_boutique, './dist/v3/boutique/ontocats_boutique.jsonl', preamble = preamble)


# Save layout of ontocats as `ontocatLayout`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique ID for the ontological category that is referenced by other files',
    'x': '<float> position of the node in the graph layout',
    'y': '<float> position of the node in the graph layout',
    'z': '<float> position of the node in the graph layout'
}
emlib.save_jsonl(ontocats_boutique, './dist/v3/boutique/ontocatLayout_boutique.jsonl', preamble = preamble)


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
emlib.save_jsonl(hyperedges_boutique, './dist/v3/boutique/hyperedges_boutique.jsonl', preamble = preamble)


preamble = None
del preamble

# %%
