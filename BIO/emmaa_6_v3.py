# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Idea: 
# * Reset in v3
# * Start from the MITRE test set
# * 
# 

# %%
import json
import pickle
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

# %%[markdown] 
# # Extract Nodes and Edges from Source Data

# %%[markdown]
# Generate `nodes`, `edges`, and `paths` from the source data since
# data from Dario appear to be internally inconsistent (paths references `edges` 
# which do not have the same source and target node ids as in `nodes`).

# %%
statements = {}
with open('./data/covid19-snapshot_sep18-2020/source/latest_statements_covid19.jsonl', 'r') as x:
    statements = [json.loads(i) for i in x]


# Example:
# {'type': 'Acetylation',
#  'enz': {'name': 'BHB', 'db_refs': {'TEXT': 'BHB'}},
#  'sub': {'name': 'Histone',
#   'db_refs': {'FPLX': 'Histone',
#    'NCIT': 'C16687',
#    'MESH': 'D006657',
#    'CHEBI': 'CHEBI:15358',
#    'CAS': '37244-51-2',
#    'TEXT': 'histone'}},
#  'belief': 0.65,
#  'evidence': [{'source_api': 'reach',
#    'pmid': '32838361',
#    'text': 'BHB activates the transcription factor Nrf2 to induce antioxidant response element (ARE) gene expression XREF_BIBR, XREF_BIBR, XREF_BIBR and induces local histone acetylation at the promoter of oxidative stress resistance genes (Foxo3a and Mt2) by inhibiting activity of histone deacetylases histone deacetylase 1 (HDAC1) and HDAC2.',
#    'annotations': {'found_by': 'Acetylation_syntax_4_noun',
#     'agents': {'coords': [[0, 3], [160, 167]],
#      'raw_text': ['BHB', 'histone'],
#      'raw_grounding': [{'TEXT': 'BHB'},
#       {'FPLX': 'Histone',
#        'NCIT': 'C16687',
#        'MESH': 'D006657',
#        'CHEBI': 'CHEBI:15358',
#        'CAS': '37244-51-2',
#        'TEXT': 'histone'}]},
#     'prior_uuids': ['7785273c-cd77-4f97-beb5-5d1f570a232f']},
#    'epistemics': {'direct': False, 'section_type': None},
#    'text_refs': {'PMID': '32838361',
#     'TRID': 31859624,
#     'PMCID': 'PMC7362813',
#     'DOI': '10.1016/J.MEDJ.2020.06.008'},
#    'source_hash': -1138439492266259709}],
#  'id': '7785273c-cd77-4f97-beb5-5d1f570a232f',
#  'matches_hash': '9432360253009950',
#  'supported_by': ['89c84fa4-3d99-48e9-a64c-e336a551886c']}


# Only keep statements that are clearly directed (i.e. having either sub/obj or enz/sub attributes)
x = {s['type']: set(s.keys()) - {'type', 'belief', 'evidence', 'id', 'matches_hash'} for s in statements}
y = ['Activation', 'Inhibition', 'Dehydroxylation', 'Demethylation', 'Deribosylation']
statements = [s for s in statements if s['type'] in y]
num_statements = len(statements)


# %%
# One edge per statement

edges = [{
    'id': int(i), 
    'type': str(s['type']), 
    'belief': float(s['belief']), 
    'statement_id': str(s['matches_hash']), 
    'source': None, 
    'target': None
    } 
    for i, s in enumerate(statements)]


# %%
%%time

# Generate list of unique nodes referenced by the statements
# Note that node type only come in the pairs ('subj', 'obj'), ('enz', 'sub')
x = ['subj', 'obj', 'enz', 'sub']
for i, t in enumerate(x):
    x[i] = {s[t]['name']: t for s in statements if t in s.keys()}

y = {**x[0], **x[1], **x[2], **x[3]}

nodes = [{
    'id': int(i),
    'name': str(name),
    'type': [str(y[name])], 
    'db_refs': {},
    'grounded': False,
    'edge_ids': []
    } 
    for i, name in enumerate(y.keys())]

num_nodes = len(nodes)


# Fetch one 'db_refs' for each unique node and check whether grounded
# Grounded <- if there is a namespace that is neither 'TEXT' or 'TEXT_NORM'
x = ['subj', 'obj', 'enz', 'sub']
for i, t in enumerate(x):
    x[i] = {s[t]['name']: s[t]['db_refs'] for s in statements if t in s.keys()}

y = {**x[0], **x[1], **x[2], **x[3]}

for i in range(num_nodes):
    try:
        nodes[i]['db_refs'] = y[nodes[i]['name']]

        if len(set(nodes[i]['db_refs'].keys()) - {'TEXT'} - {'TEXT_NORM'}) > 0:
            nodes[i]['grounded'] = True

    except:
        pass


# Find all statements that reference each node
x = ['subj', 'enz']
sources = [[s[t]['name'] for t in x if t in s.keys()][0] for s in statements]

x = ['obj', 'sub']
targets = [[s[t]['name'] for t in x if t in s.keys()][0] for s in statements]

map_nodes_ids = {node['name']: i for i, node in enumerate(nodes)}
for i, source in enumerate(sources):
    nodes[map_nodes_ids[source]]['edge_ids'].append(int(i))

for i, target in enumerate(targets):
    nodes[map_nodes_ids[target]]['edge_ids'].append(int(i))


# Find all nodes attached to each edge
for i, j in enumerate(zip(sources, targets)):
    edges[i]['source'] = int(map_nodes_ids[j[0]])
    edges[i]['target'] = int(map_nodes_ids[j[1]])


i = j = x = y = source = sources = target = targets = map_nodes_ids = None
del i, j, t, x, y, source, sources, target, targets, map_nodes_ids

# time: 1.99 s 


# %%
# Need to remove duplicate nodes such as '3A' and '3a'

# %%
# Output `nodes` and `edges`

# `nodes`
with open(f'./dist/v2/nodes.jsonl', 'w') as x:

    # Description
    y = {
        'id': '<int> unique node ID',
        'name': '<str> node name from the `name` attribute in `latest_statements_covid19.jsonl`',
        'db_refs': '<dict> database references from the `db_refs` attribute in `latest_statements_covid19.jsonl`',
        'grounded': '<bool> whether the node is grounded to a database',
        'edge_ids': '<list of int> ID of edges that have the node as a source and/or a target' 
    }
    json.dump(y, x)
    x.write('\n')

    # Data
    for node in nodes:
        json.dump(node, x)
        x.write('\n')


# `edges`
with open(f'./dist/v2/edges.jsonl', 'w') as x:

    # Description
    y = {
        'id': '<int> unique edge ID',
        'type': '<str> edge type, taken from `type` attribute in `latest_statements_covid19.jsonl`',
        'belief': '<float> edge belief score, taken from `belief` attribute in `latest_statements_covid19.jsonl`',
        'statement_id': '<str> unique statement id, taken from `matches_hash` from `latest_statements_covid19.jsonl`',
        'source': '<int> ID of the source node (see `nodes.jsonl`)' ,
        'target': '<int> ID of the target node (see `nodes.jsonl`)'
    }
    json.dump(y, x)
    x.write('\n')

    # Data
    for edge in edges:
        json.dump(edge, x)
        x.write('\n')


x = y = node = edge = None
del x, y, node, edge


# %%
# Reload if necessary
nodes = []
with open(f'./dist/v2/nodes.jsonl', 'r') as x:
    for i in x:
        nodes.append(json.loads(i))

nodes = nodes[1:]

edges = []
with open(f'./dist/v2/edges.jsonl', 'r') as x:
    for i in x:
        edges.append(json.loads(i))

edges = edges[1:]

# %%[markdown]
# # Generate Model Subgraph from Tested Paths

# %%[markdown]
# ## Map path references from names to numeric ids from `nodes` and `edges`

# %%
# Load tested paths (MITRE)
paths_mitre = {}
with open('./data/covid19-snapshot_sep18-2020/source/covid19_mitre_tests_latest_paths.jsonl', 'r') as x:
    paths_mitre = [json.loads(i) for i in x]

num_paths = len(paths_mitre)


# Generate name-to-id maps
map_nodes_ids = {node['name']: node['id'] for node in nodes}
map_edges_ids = {edge['statement_id']: edge['id'] for edge in edges}


# Convert node names and statement ids to node ids and edge ids
paths_mitre_ids = [{} for path in paths_mitre]
for i in range(num_paths):
    paths_mitre_ids[i]['node_ids'] = [map_nodes_ids[node] if node in map_nodes_ids.keys() else None for node in paths_mitre[i]['nodes']]
    paths_mitre_ids[i]['edge_ids'] = [map_edges_ids[str(edge)] if str(edge) in map_edges_ids.keys() else None for l in paths_mitre[i]['edges'] for edge in l]
    paths_mitre_ids[i]['graph_type'] = paths_mitre[i]['graph_type']



# Output numeric-id paths
with open(f'./dist/v2/paths_mitre.jsonl', 'w') as x:

    # Description
    y = {
        'node_ids': '<list of int> node IDs from `nodes.jsonl` (`null` = out-of-range nodes)',
        'edge_ids': '<list of int> edge IDs from `edges.jsonl` (`null` = out-of-range edges)',
        'graph_type': '<float> `graph_type` attribute from `covid19_mitre_tests_latest_paths.jsonl`'
    }
    json.dump(y, x)
    x.write('\n')

    # Data
    for path in paths_mitre_ids:
        json.dump(path, x)
        x.write('\n')


x = y = path = map_nodes_ids = map_edges_ids = None
del x, y, path, map_nodes_ids, map_edges_ids


# %%[markdown]
# Reload if necessary

paths_mitre_ids = []
with open(f'./dist/v2/paths_mitre.jsonl', 'r') as x:
    for i in x:
        paths_mitre_ids.append(json.loads(i))

paths_mitre_ids = paths_mitre_ids[1:]


# %%[markdown]
# ## Carve out subgraph from model graph using given paths

# Filter model graph by MITRE paths
nodes_mitre, edges_mitre, __ = emlib.intersect_graph_paths(nodes, edges, paths_mitre_ids)
num_nodes = len(nodes_mitre)
num_edges = len(edges_mitre)

# Reset node id in `nodes` and `edges` to maintain Grafer optimization
nodes_mitre, edges_mitre, __ = emlib.reset_node_ids(nodes_mitre, edges_mitre)


# %%
# Output MITRE `nodes` and `edges`

# `nodes`
with open(f'./dist/v2/nodes_mitre.jsonl', 'w') as x:

    # Description
    y = {
        'id': '<int> unique node ID',
        'name': '<str> node name from the `name` attribute in `latest_statements_covid19.jsonl`',
        'db_refs': '<dict> database references from the `db_refs` attribute in `latest_statements_covid19.jsonl`',
        'grounded': '<bool> whether the node is grounded to a database',
        'edge_ids': '<list of int> ID of edges that have the node as a source and/or a target' 
    }
    json.dump(y, x)
    x.write('\n')

    # Data
    for node in nodes_mitre:
        json.dump({k: node[k] for k in nodes[0].keys()}, x)
        x.write('\n')


# `edges`
with open(f'./dist/v2/edges_mitre.jsonl', 'w') as x:

    # Description
    y = {
        'id': '<int> unique edge ID',
        'type': '<str> edge type, taken from `type` attribute in `latest_statements_covid19.jsonl`',
        'belief': '<float> edge belief score, taken from `belief` attribute in `latest_statements_covid19.jsonl`',
        'statement_id': '<str> unique statement id, taken from `matches_hash` from `latest_statements_covid19.jsonl`',
        'source': '<int> ID of the source node (see `nodes.jsonl`)' ,
        'target': '<int> ID of the target node (see `nodes.jsonl`)'
    }
    json.dump(y, x)
    x.write('\n')

    # Data
    for edge in edges_mitre:
        json.dump({k: edge[k] for k in edges[0].keys()}, x)
        x.write('\n')


x = y = node = edge = None
del x, y, node, edge


# %%
# Reload if necessary
nodes_mitre = []
with open(f'./dist/v2/nodes_mitre.jsonl', 'r') as x:
    for i in x:
        nodes_mitre.append(json.loads(i))

nodes_mitre = nodes_mitre[1:]

edges_mitre = []
with open(f'./dist/v2/edges_mitre.jsonl', 'r') as x:
    for i in x:
        edges_mitre.append(json.loads(i))

edges_mitre = edges_mitre[1:]


# %%
# # Additional Model Node Attributes

# %%
# ## Calculate node degrees
nodes_mitre, __, __ = emlib.calculate_node_degrees(nodes_mitre, edges_mitre)

# %%
# ## Get node-wise belief score from `edges`
nodes_mitre, __ = emlib.calculate_node_belief(nodes_mitre, edges_mitre, mode = 'max')

# %%
# ## Placeholder clustering attributes
for node in nodes_mitre:
    node['cluster_ids'] = []

# %%
# ## Generate placeholder layout of the MITRE subgraph

# %%
# Generate placeholder layout
coors, G_mitre = emlib.generate_nx_layout(edges_mitre, layout = 'spring', plot = True)

x = {node['id']: i for i, node in enumerate(nodes_mitre)}
for i in coors:
    j = x[i]
    nodes_mitre[j]['x'] = float(coors[i][0])
    nodes_mitre[j]['y'] = float(coors[i][1])
    nodes_mitre[j]['z'] = float(0.0)

# %%
# Plot layout

plt.style.use('dark_background')
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = plt.setp(ax, aspect = 1.0, title = 'MITRE Subgraph (node color = belief score, node size: total node degree)')

options = {
    'ax': ax,
    'arrows': False, 
    'with_labels': False,
    'node_size': 0.5 * np.asarray([nodes_mitre[x[node_id]]['in_degree'] + nodes_mitre[x[node_id]]['out_degree'] for node_id in coors]),
    'width': 0.05,
    'alpha': 0.8,
    'node_color': np.asarray([nodes_mitre[x[node_id]]['belief'] > 0.90 for node_id in coors]), 
    'cmap': 'cividis',
    'edge_color': 'w'
}
nx.draw_networkx(G_mitre, pos = coors, **options)

fig.savefig('./figures/v2/mitre_subgraph_layout.png', dpi = 150)


x = fig = ax = options = None
del x, fig, ax, options


# %%[markdown]
# # Categorize model nodes by ontology categories


# %%[markdown]
# ## Load the INDRA ontology

# %%
with open('./data/indra_ontology_v1.3.json', 'r') as x:
    G_onto_JSON = json.load(x)

# Remove 'xref' links
G_onto_JSON['links'] = [link for link in G_onto_JSON['links'] if link['type'] != 'xref']


# %%[markdown]
# ## Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, G_onto_JSON, nodes_mitre)


# %%
# Reduce 'db_refs' of each model node to a single entry by namespace priority
# * Find the first model node namespace that is the sorted namespace list
# * 'db_ref_priority' = namespace`:`ref`
# * `grounded = False` -> 'not-grounded' 
nodes_mitre, __ = emlib.reduce_nodes_db_refs(nodes_mitre, namespaces)


# %%[markdown]
# ## Extract components and find their roots

# %%
# Load the ontology graph as a `networkx` object
G_onto = nx.readwrite.json_graph.node_link_graph(G_onto_JSON)


# Extract components, sorted by size
ontoSubs = sorted(nx.weakly_connected_components(G_onto), key = len, reverse = True)


# Find the root nodes of each component (degree = 0 or out-degree = 0)
# z = [np.flatnonzero([True if G_onto.out_degree(node) < 1 else False for node in sub]) for sub in ontoSubs]
# ontoSubRoots = [[list(ontoSubs[i])[j] for j in indices] for i, indices in enumerate(z)]
# ontoSubRoots_num = np.sum([True if len(indices) > 1 else False for indices in z])
ontoSubRoots = [[node for node in sub if G_onto.out_degree(node) < 1] for sub in ontoSubs]


# List onto node names/refs
# ontoRefs = nx.nodes(G_onto)


# %%[markdown]
# ## Index all model nodes that can be mapped to the ontology graph

# %%
%%time

# Initialize the ontological attributes
# Unmappable nodes: level = `-1` and to-root list = [`not-grounded-onto`]
for node in nodes_mitre:
    if node['db_ref_priority'] in nx.nodes(G_onto):
        node['grounded_onto'] = True
        node['ontocat_level'] = -1
        node['ontocat_refs'] = []
    else:
        node['grounded_onto'] = False
        node['ontocat_level'] = -1
        node['ontocat_refs'] = ['not-grounded-onto']


# Index of mappable model nodes
node_indices = [i for i, node in enumerate(nodes_mitre) if node['grounded_onto']]

# Index of the onto subgraph to which the model nodes are mapped
# (if in nontrivial subgraph -> -1)
num_ontoSub_nontrivial = sum([1 for sub in ontoSubs if len(sub) > 1])
x = [{(nodes_mitre[i]['db_ref_priority'] in sub): j for j, sub in enumerate(ontoSubs[:num_ontoSub_nontrivial])} for i in node_indices]
ontoSub_indices = [d[True] if True in d.keys() else -1 for d in x]


for i, j in zip(node_indices, ontoSub_indices):

    source = nodes_mitre[i]['db_ref_priority']

    # Case: model node was mapped to either a trivial subgraph or the root of a non-trivial subgraph
    if (j == -1) or (source in ontoSubRoots[j]):
        nodes_mitre[i]['ontocat_level'] = 0
        nodes_mitre[i]['ontocat_refs'] = [source]

    else:

        z = []
        for target in ontoSubRoots[j]:
            try:
                p = nx.algorithms.shortest_paths.generic.shortest_path(G_onto.subgraph(ontoSubs[j]), source = source, target = target)
                z.append(p)
            except:
                pass
        
        # Find shortest path and reverse such that [target, ..., source]
        z = sorted(z, key = len, reverse = False)
        nodes_mitre[i]['ontocat_level'] = len(z[0]) - 1
        nodes_mitre[i]['ontocat_refs'] = z[0][::-1]



i = j = p = z = source = target = num_ontoSub_nontrivial = node_indices = ontoSub_indices = None
del i, j, p, z, source, target, num_ontoSub_nontrivial, node_indices, ontoSub_indices

# time: 7 m 35 s

# %%
# Ensure that identical onto nodes share the same lineage (path to their ancestor) for hierarchical uniqueness

ontocat_refs = [node['ontocat_refs'] for node in nodes_mitre]
m = max([len(path) for path in ontocat_refs])
for i in range(1, m):

    # All nodes
    x = [path[i] if len(path) > i else '' for path in ontocat_refs]

    # All unique nodes
    y = list(set(x) - set(['']))

    # Mapping from all nodes to unique nodes
    xy = [y.index(node) if node is not '' else '' for node in x]

    # Choose the path segment of the first matching node for each unique node
    z = [ontocat_refs[x.index(node)][:i] for node in y]
    
    # Substitute path segments
    for j in range(num_nodes):
        if xy[j] is not '':
            nodes_mitre[j]['ontocat_refs'][:i] = z[xy[j]]
        else:
            nodes_mitre[j]['ontocat_refs'][:i] = ontocat_refs[j][:i]


# Copy results
for j in range(num_nodes):
    nodes_mitre[j]['ontocat_refs'][:i] = ontocat_refs[j][:i].copy()


i = j = m = x = y = z = xy = None
del i, j, m, x, y, z, xy


# %%
%%time

# Generate list of mapped ontology categories, sorted by size
ontocats_ = {}
ontocats_['ref'], ontocats_['size'] = np.unique([node for path in ontocat_refs for node in path], return_counts = True)

num_ontocats = len(ontocats_['ref'])
i = np.argsort(ontocats_['size'])[::-1]
ontocats_['ref'] = list(ontocats_['ref'][i])
ontocats_['size'] = [int(k) for k in ontocats_['size']]
ontocats_['id'] = list(range(num_ontocats))


# Get the mapped onto category names
x = dict(G_onto.nodes(data = 'name', default = None))
ontocats_['name'] = list(np.empty((num_ontocats, )))
for i, ref in enumerate(ontocats_['ref']):
    try:
        ontocats_['name'][i] = x[ref]
    except:
        ontocats_['name'][i] = ''


# Get onto level of each category
i = max([len(path) for path in ontocat_refs])
x = [np.unique([path[j] if len(path) > j else '' for path in ontocat_refs]) for j in range(i)]
ontocats_['level'] = [int(np.flatnonzero([ref in y for y in x])[0]) for ref in ontocats_['ref']]


# Get numeric id version of ontocat_refs
x = {k: v for k, v in zip(ontocats_['ref'], ontocats_['id'])}
ontocat_ids = [[x[node] for node in path] for path in ontocat_refs]
for node in nodes_mitre:
    node['ontocat_ids'] = [x[ontocat] for ontocat in node['ontocat_refs']]


# Get parent category id for each category (for root nodes, parentID = None)
y = [np.flatnonzero([True if ref in path else False for path in ontocat_refs])[0] for ref in ontocats_['ref']]
ontocats_['parent_ref'] = [ontocat_refs[y[i]][ontocat_refs[y[i]].index(ref) - 1] if ontocat_refs[y[i]].index(ref) > 0 else None for i, ref in enumerate(ontocats_['ref'])]
ontocats_['parent_id'] = [x[parent] if parent is not None else None for parent in ontocats_['parent_ref']]


# Find membership of onto categories
ontocats_['node_ids'] = [[node['id'] for node, path in zip(nodes_mitre, ontocat_refs) if ref in path] for ref in ontocats_['ref']]


# Placeholder for hyperedges
ontocats_['hyperedge_ids'] = [[] for i in range(num_ontocats)]


# Switch to row-wise structure
ontocats = [{k: ontocats_[k][i] for k in ontocats_.keys()} for i in range(num_ontocats)]


# Placeholder for layout coordinates
# (use median of the membership)
x = {node['id']: i for i, node in enumerate(nodes_mitre)}
for ontocat in ontocats:
    for i in ['x', 'y', 'z']:
        ontocat[i] = float(np.median([nodes_mitre[x[j]][i] for j in ontocat['node_ids']]))


i = x = y = None
del i, x, y

# time: 3.89 s


# %%

# Output model node data
with open(f'./dist/v2/nodeData_mitre.jsonl', 'w') as x:

    # Description
    y = {
        'id': '<int> unique ID for the node in the KB graph as specified in `nodes.jsonl`',
        'x': '<float> position of the node in the graph layout',
        'y': '<float> position of the node in the graph layout',
        'z': '<float> position of the node in the graph layout',
        'in_degree': '<int> in-degree in the KB graph',
        'out_degree': '<int> out-degree in the KB graph',
        'belief': '<float> max of the belief scores of all adjacent edges in the KB graph',
        'db_ref_priority': '<str> database reference from `db_refs` of `nodes.jsonl`, that is used by the INDRA ontology (v1.3)', 
        'grounded_onto': '<bool> whether this model node is grounded to something that exists within the ontology', 
        'ontocat_level': '<int> level of the ontology node/category to which this model node was mapped (`-1` if not mappable, `0` if root)', 
        'ontocat_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)', 
        'cluster_ids': '<array of int> (placeholder for cluster)'
    }
    json.dump(y, x)
    x.write('\n')

    # Data
    for node in nodes_mitre:
        z = {k: node[k] for k in y.keys()}

        json.dump(z, x)
        x.write('\n')


# %%
# Output onto category data
with open(f'./dist/v2/ontocats_mitre.jsonl', 'w') as x:

    # Description
    y = {
        'id': '<int> unique ID for the ontological category (referenced by `ontocat_ids` in `nodeData.jsonl`)',
        'ref': '<str> unique reference ID of this category (as given by the INDRA Ontology v1.3)',
        'name': '<str> name of this category (as given by the INDRA Ontology v1.3)',
        'size': '<int> number of model nodes that were mapped to this category and its children',
        'level': '<int> number of hops to reach the local root (`0` if root)',
        'parent_id': '<int> ID of the parent of this category in the ontology',
        'node_ids': '<array of int> unordered list of IDs from model nodes in the membership of this category',
        'hyperedge_ids': '<array of int> (placeholder for hyperedge IDs)',
        'x': '<float> position of the node in the graph layout',
        'y': '<float> position of the node in the graph layout',
        'z': '<float> position of the node in the graph layout'
    }
    json.dump(y, x)
    x.write('\n')

    # Data
    for ontocat in ontocats:
        z = {k: ontocat[k] for k in y.keys()}

        json.dump(z, x)
        x.write('\n')


node = ontocat = x = y = z = None
del node, ontocat, x, y, z


# %%
###########################################

# %% 
# Reload onto category data

nodeData = []
with open(f'./dist/v2/nodeData.jsonl', 'r') as x:
    for i in x:
        nodeData.append(json.loads(i))

nodeData = nodeData[1:]
for x, y in zip(nodes_mitre, nodeData):
    for j in y.keys():
        if isinstance(y[j], list):
            x[j] = y[j].copy()
        else:
            x[j] = y[j]


ontocats = []
with open(f'./dist/v2/ontocats.jsonl', 'r') as x:
    for i in x:
        ontocats.append(json.loads(i))

ontocats = ontocats[1:]
num_ontocats = len(ontocats)

# %%
# # Hyperedge Generation

# %%
%%time

# Find model edges that have the given ontocat member as their source
x = [edge['source'] for edge in edges_mitre]
ontocats_edges_source = [emlib.match_arrays(x, ontocat['node_ids']) for ontocat in ontocats]

x = [edge['target'] for edge in edges_mitre]
ontocats_edges_target = [emlib.match_arrays(x, ontocat['node_ids']) for ontocat in ontocats]


# time: 17.5 s

# %%
%%time

# Find the onto-category siblings of each onto-category
# * Generate list of parent IDs
# * Make lists of onto-categories with matching parent IDs for each onto-category
# * Remove self from each list
ontocats_parent = [ontocat['parent_id'] if ontocat['parent_id'] != None else -1 for ontocat in ontocats]
ontocats_siblings = [[ontocats[j]['id'] for j in np.flatnonzero(emlib.match_arrays(ontocats_parent, [ontocat_parent])) if ontocats[j]['id'] != ontocats[i]['id']] for i, ontocat_parent in enumerate(ontocats_parent)]


# Find each set of edges (hyperedges) that has: 
# * a given onto-category members as the edges' source 
# * the sibling onto-category members as the edges target
hyperedges_siblings_ = {}
hyperedges_siblings_['source'] = [ontocat['id'] for i, ontocat in enumerate(ontocats) for sibling_id in ontocats_siblings[i]]
hyperedges_siblings_['target'] = [sibling_id for i, ontocat in enumerate(ontocats) for sibling_id in ontocats_siblings[i]]
hyperedges_siblings_['edge_indices'] = [np.flatnonzero([ontocats_edges_source[ontocat['id']] & ontocats_edges_target[sibling_id]]) for i, ontocat in enumerate(ontocats) for sibling_id in ontocats_siblings[i]]
num_hyperedges_siblings_ = len(hyperedges_siblings_['source'])

# Map between list indices and edge ID
map_edges_ids = {i: edge['id'] for i, edge in enumerate(edges_mitre)}
hyperedges_siblings_['edge_ids'] = [[map_edges_ids[i] for i in edges] for edges in hyperedges_siblings_['edge_indices']]

hyperedges_siblings_['level'] = [ontocat['level'] for i, ontocat in enumerate(ontocats) for sibling_id in ontocats_siblings[i]]
hyperedges_siblings_['size'] = [len(edges) for edges in hyperedges_siblings_['edge_indices']]
hyperedges_siblings_['id'] = list(range(num_hyperedges_siblings_))


# Specify source and target types
hyperedges_siblings_['source_type'] = ['ontocat' for i in range(num_hyperedges_siblings_)]
hyperedges_siblings_['target_type'] = ['ontocat' for i in range(num_hyperedges_siblings_)]


# Trim empty hyperedges and change to row-wise
hyperedges_siblings = [{k: hyperedges_siblings_[k][i] for k in hyperedges_siblings_.keys()} for i in range(num_hyperedges_siblings_) if hyperedges_siblings_['size'][i] > 0]
num_hyperedges_siblings = len(hyperedges_siblings)


x = map_edges_ids = hyperedges_siblings_ = num_hyperedges_siblings_ = None
del x, map_edges_ids, hyperedges_siblings_, num_hyperedges_siblings_

# time: 13.7 s


# %%
%%time

# Find the node membership of the parent of each onto-category that is not in the node membership of the siblings
x = [[node for sibling_id in ontocats_siblings[i] for node in ontocats[sibling_id]['node_ids']] + ontocat['node_ids'] for i, ontocat in enumerate(ontocats)]
ontocats_parent_nodes = [list(set(ontocats[ontocat['parent_id']]['node_ids']) - set(x[i])) if ontocat['parent_id'] != None else [] for i, ontocat in enumerate(ontocats)]


# Find hyperedges that has:
# * a given onto-category members as the edges' source 
# * a member of the onto-category parent that is not in any sibling onto-category as the edges' target
hyperedges_parent_nodes_ = {}
hyperedges_parent_nodes_['source'] = [ontocat['id'] for i, ontocat in enumerate(ontocats) for node_id in ontocats_parent_nodes[i]]
hyperedges_parent_nodes_['target'] = [node_id for i, ontocat in enumerate(ontocats) for node_id in ontocats_parent_nodes[i]]
num_hyperedges_parent_nodes_ = len(hyperedges_parent_nodes_['source'])

x = [edge['target'] for edge in edges_mitre]
hyperedges_parent_nodes_['edge_indices'] = [np.flatnonzero(ontocats_edges_source[ontocat['id']] & emlib.match_arrays(x, [node_id])) for i, ontocat in enumerate(ontocats) for node_id in ontocats_parent_nodes[i]]

# Map between list indices and edge ID
map_edges_ids = {i: edge['id'] for i, edge in enumerate(edges_mitre)}
hyperedges_parent_nodes_['edge_ids'] = [[map_edges_ids[i] for i in edges] for edges in hyperedges_parent_nodes_['edge_indices']]

hyperedges_parent_nodes_['level'] = [ontocat['id'] for i, ontocat in enumerate(ontocats) for node_id in ontocats_parent_nodes[i]]
hyperedges_parent_nodes_['size'] = [len(edges) for edges in hyperedges_parent_nodes_['edge_ids']]
hyperedges_parent_nodes_['id'] = list(range(len(hyperedges_parent_nodes_['source'])))


# Specify source and target types
hyperedges_parent_nodes_['source_type'] = ['ontocat' for i in range(num_hyperedges_parent_nodes_)]
hyperedges_parent_nodes_['target_type'] = ['node' for i in range(num_hyperedges_parent_nodes_)]


# Trim empty hyperedges and change to row-wise
hyperedges_parent_nodes = [{k: hyperedges_parent_nodes_[k][i] for k in hyperedges_parent_nodes_.keys()} for i in range(num_hyperedges_parent_nodes_) if hyperedges_parent_nodes_['size'][i] > 0]
num_hyperedges_parent_nodes = len(hyperedges_parent_nodes)


x = ontocats_parent_nodes = map_edges_ids = hyperedges_parent_nodes_ = num_hyperedges_parent_nodes_ = None
del x, ontocats_parent_nodes, map_edges_ids, hyperedges_parent_nodes_, num_hyperedges_parent_nodes_ 

ontocats_edges_source = ontocats_edges_target = ontocats_parent = ontocats_siblings = None
del ontocats_edges_source, ontocats_edges_target, ontocats_parent, ontocats_siblings

# time: 1.43 s


# %%[markdown]
# # ## Concatenate the hyperedges together
hyperedges_mitre = hyperedges_parent_nodes + hyperedges_siblings
num_hyperedges = len(hyperedges_mitre)
for i, hyperedge in enumerate(hyperedges_mitre):
    hyperedge['id'] = i


hyperedge = hyperedges_siblings = hyperedges_parent_nodes = None
del hyperedge, hyperedges_siblings, hyperedges_parent_nodes


# %%[markdown]
# ## Output hyperedge data
with open(f'./dist/v2/hyperedges_mitre.jsonl', 'w') as x:

    y = {
        'id': '<int> unique hyperedge ID',
        'level': '<float> ontological level of this hyperedge (taken from the source)',
        'size': '<int> number of model edges that is aggregated here',
        'edge_ids': '<array of int> unordered list of edge ID of the underlying model edges (see `edges.jsonl`)',
        'source_type': '<str> object type of the source (only `ontocat`)',
        'target_type': '<str> object type of the target (either `ontocat` or `node`)',
        'source': '<int> unique ID of the source (see `ontocats.jsonl`)' ,
        'target': '<int> unique ID of the target (see `ontocats.jsonl` or `nodes.jsonl`)',
    }
    json.dump(y, x)
    x.write('\n')


    for edge in hyperedges_mitre:
        json.dump({k: edge[k] for k in y.keys()}, x)
        x.write('\n')


x = y = node = edge = None
del x, y, node, edge

# %%





# %%

