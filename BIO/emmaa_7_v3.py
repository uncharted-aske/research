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
# ## Reduce Model Graph to Tested Paths

# %%
%%time

# Find intersection between the 'paths' subgraph and the model graph
nodes_mitre, edges_mitre, __ = emlib.intersect_graph_paths(nodes_model, edges_model, paths_mitre)
num_nodes = len(nodes_mitre)
num_edges = len(edges_mitre)
# 3714 paths, 42198 nodes, and 429976 edges in total.
# 3714 paths, 1687 nodes, and 5034 edges in the intersection.


# print(f"{len({node['id']: 0 for node in nodes_model})} == {len(nodes_model)}")
# print(f"{len({edge['id']: 0 for edge in edges_model})} == {len(edges_model)}")
# print(f"{len({node['id']: 0 for node in nodes_mitre})} == {len(nodes_mitre)}")
# print(f"{len({edge['id']: 0 for edge in edges_mitre})} == {len(edges_mitre)}")
# 42198 == 42198
# 429976 == 429976
# 1687 == 1687
# 5034 == 5034


# Reset node id in `nodes` and `edges` to maintain Grafer optimization
# Not applied due to some Python pointer BS
# __, __ = emlib.reset_node_edge_ids(nodes_mitre, edges_mitre)

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


# Save `paths`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'node_ids': '<list of int> node IDs from `nodes.jsonl` (`null` = out-of-range nodes)',
    'edge_ids': '<list of int> edge IDs from `edges.jsonl` (`null` = out-of-range edges)',
    'graph_type': '<str> `graph_type` attribute from `covid19_mitre_tests_latest_paths.jsonl`'
}
emlib.save_jsonl(paths_mitre, './dist/v3/paths_mitre.jsonl', preamble = preamble)


# time: 14.8 s


# %%
# ## Generate NetworkX MultiDiGraph from `nodes` and `edges`

# %%
G_model = emlib.generate_nx_object(nodes_model, edges_model)
with open('./dist/v3/G_model.pkl', 'wb') as x:
    pickle.dump(G_model, x)

G_model = x = None
del x, G_model

# %%
G_mitre = emlib.generate_nx_object(nodes_mitre, edges_mitre)
with open('./dist/v3/G_mitre.pkl', 'wb') as x:
    pickle.dump(G_mitre, x)


map_ids_edges = {edge['id']: i for i, edge in enumerate(edges_mitre)}
x = [max([edges_mitre[map_ids_edges[edge_id]]['belief'] for edge_id in node[1]]) if len(node[1]) > 0 else 0.0 for node in G_mitre.nodes.data('edge_ids_target')]
__, __, fig, __ = emlib.generate_nx_layout(G = G_mitre, layout = 'spring', layout_atts = {'k': 0.1}, plot = True, plot_atts = {'node_color': x, 'vmin': 0.0, 'vmax': 1.0, 'cmap': 'cool'})
fig.savefig('./figures/v3/mitre_subgraph_layout_degree_belief.png', dpi = 150)

G_mitre = x = fig = None
del x, fig, G_mitre

# %%
# Reload `nodes` and `edges` if necessary
if False:
    nodes_model = emlib.load_jsonl('./dist/v3/nodes_model.jsonl', remove_preamble = True)
    edges_model = emlib.load_jsonl('./dist/v3/edges_model.jsonl', remove_preamble = True)

if False:
    nodes_mitre = emlib.load_jsonl('./dist/v3/nodes_mitre.jsonl', remove_preamble = True)
    edges_mitre = emlib.load_jsonl('./dist/v3/edges_mitre.jsonl', remove_preamble = True)


# %%
# #####################################################################

# %%[markdown]
# # Categorize model nodes by ontology categories

# %%[markdown]
# ## Load the INDRA ontology

# %%
with open('./data/bio_ontology_v1.5.json', 'r') as x:
    G_onto_JSON = json.load(x)

# Remove 'xref' links
G_onto_JSON['links'] = [link for link in G_onto_JSON['links'] if link['type'] != 'xref']

# %%[markdown]
# ## Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, G_onto_JSON, nodes_mitre)

# %%
__ = [print(f"{k:<20} | {v[0]:>6} | {v[1]:>6}") for k, v in namespaces_count.items()]
# FPLX                 |     79 |    703
# UPPRO                |      1 | 602521
# HGNC                 |    490 |  46893
# UP                   |    551 | 736039
# CHEBI                |    930 | 117779
# GO                   |     11 |  44523
# MESH                 |   1058 | 299121
# MIRBASE              |      0 |   7410
# DOID                 |      2 |  10049
# HP                   |      2 |  14967
# EFO                  |     20 |   9577
# CAS                  |    876 |  22313
# CHEMBL               |    835 |  61107
# DRUGBANK             |    845 |  13609
# DRUGBANKV4.TARGET    |      8 |      0
# ECCODE               |     11 |     41
# EGID                 |     64 |  41726
# HGNC_GROUP           |     11 |    130
# HMDB                 |    565 |   7819
# HMS-LINCS            |     69 |      0
# IP                   |     27 |    217
# LINCS                |     39 |   3007
# LSPCI                |     13 |      0
# NCIT                 |     43 |  18414
# NXPFA                |      1 |      0
# PF                   |     12 |     59
# PUBCHEM              |    850 |  93857
# TAXONOMY             |      5 |     21
# TEXT                 |   1106 |      0
# TEXT_NORM            |    161 |      0
# INDRA                |      0 |      2
# INDRA_ACTIVITIES     |      0 |      8
# INDRA_MODS           |      0 |     13


# %%
# Reduce 'db_refs' of each model node to a single entry by namespace priority
# * Find the first model node namespace that is the sorted namespace list
# * 'db_ref_priority' = namespace`:`ref`
# * `grounded = False` -> 'not-grounded' 
nodes_mitre, __ = emlib.reduce_nodes_db_refs(nodes_mitre, namespaces)

# %%[markdown]
%%time

# ## Calculate In-Ontology Paths
emlib.calculate_onto_root_path(nodes_mitre, G_onto_JSON)

# time: 14 m 5 s

# %%[markdown]
%%time

# ## Extract Ontological Categories
ontocats_mitre = emlib.extract_ontocats(nodes_mitre, G_onto_JSON)

# time: 21.9 s

# %%[markdown]
%%time

# ## Generate Hyperedges
hyperedges_mitre = emlib.generate_hyperedges(nodes_mitre, edges_mitre, ontocats_mitre)

# time: 

# %%[markdown]
# # Compute Layout using Ontological Categories and Hyperedges
%%time

__ = emlib.generate_onto_layout(nodes_mitre, ontocats_mitre, hyperedges_mitre, plot = True)

# time: 7.56 s

# %%[markdown]

%%time

# Save layout of `nodes` as `nodeLayout`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique node ID that is defined in `nodes.jsonl`',
    'x': '<float> position of the node in the graph layout',
    'y': '<float> position of the node in the graph layout',
    'z': '<float> position of the node in the graph layout',
}
emlib.save_jsonl(nodes_mitre, './dist/v3/nodeLayout_mitre.jsonl', preamble = preamble)


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
emlib.save_jsonl(nodes_mitre, './dist/v3/nodeAtts_mitre.jsonl', preamble = preamble)


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
emlib.save_jsonl(ontocats_mitre, './dist/v3/ontocats_mitre.jsonl', preamble = preamble)


# Save layout of ontocats as `ontocatLayout`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique ID for the ontological category that is referenced by other files',
    'x': '<float> position of the node in the graph layout',
    'y': '<float> position of the node in the graph layout',
    'z': '<float> position of the node in the graph layout'
}
emlib.save_jsonl(ontocats_mitre, './dist/v3/ontocatLayout_mitre.jsonl', preamble = preamble)


# time: 167 ms

# %%[markdown]
%%time

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
emlib.save_jsonl(hyperedges_mitre, './dist/v3/hyperedges_mitre.jsonl', preamble = preamble)


# time: 29.3 ms

# %%







