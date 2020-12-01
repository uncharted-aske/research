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


# %%[markdown]
# # Categorize model nodes by ontology categories


# %%[markdown]
# ## Load the INDRA ontology

# %%
with open('./data/indra_ontology_v1.3.json', 'r') as x:
    ontoJSON = json.load(x)

# Remove 'xref' links
ontoJSON['links'] = [link for link in ontoJSON['links'] if link['type'] != 'xref']


# %%[markdown]
# ## Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, ontoJSON, nodes_mitre)


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
ontoG = nx.readwrite.json_graph.node_link_graph(ontoJSON)


# Extract components, sorted by size
ontoSubs = sorted(nx.weakly_connected_components(ontoG), key = len, reverse = True)


# Find the root nodes of each component (degree = 0 or out-degree = 0)
# z = [np.flatnonzero([True if ontoG.out_degree(node) < 1 else False for node in sub]) for sub in ontoSubs]
# ontoSubRoots = [[list(ontoSubs[i])[j] for j in indices] for i, indices in enumerate(z)]
# ontoSubRoots_num = np.sum([True if len(indices) > 1 else False for indices in z])
ontoSubRoots = [[node for node in sub if ontoG.out_degree(node) < 1] for sub in ontoSubs]


# List onto node names/refs
# ontoRefs = nx.nodes(ontoG)


# %%[markdown]
# ## Index all model nodes that can be mapped to the ontology graph

# %%
%%time

# Initialize the ontological attributes
# Unmappable nodes: level = `-1` and to-root list = [`not-grounded-onto`]
for node in nodes_mitre:
    if node['db_ref_priority'] in nx.nodes(ontoG):
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
                p = nx.algorithms.shortest_paths.generic.shortest_path(ontoG.subgraph(ontoSubs[j]), source = source, target = target)
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
ontocats = {}
ontocats['ref'], ontocats['size'] = np.unique([node for path in ontocat_refs for node in path], return_counts = True)

num_ontocats = len(ontocats['ref'])
i = np.argsort(ontocats['size'])[::-1]
ontocats['ref'] = list(ontocats['ref'][i])
ontocats['size'] = [int(k) for k in ontocats['size']]
ontocats['id'] = list(range(num_ontocats))


# Get the mapped onto category names
x = dict(ontoG.nodes(data = 'name', default = None))
ontocats['name'] = list(np.empty((num_ontocats, )))
for i, ref in enumerate(ontocats['ref']):
    try:
        ontocats['name'][i] = x[ref]
    except:
        ontocats['name'][i] = ''


# Get onto level of each category
i = max([len(path) for path in ontocat_refs])
x = [np.unique([path[j] if len(path) > j else '' for path in ontocat_refs]) for j in range(i)]
ontocats['level'] = [int(np.flatnonzero([ref in y for y in x])[0]) for ref in ontocats['ref']]


# Get numeric id version of ontocat_refs
x = {k: v for k, v in zip(ontocats['ref'], ontocats['id'])}
ontocat_ids = [[x[node] for node in path] for path in ontocat_refs]
for node in nodes_mitre:
    node['ontocat_ids'] = [x[ontocat] for ontocat in node['ontocat_refs']]


# Get parent category id for each category (for root nodes, parentID = None)
y = [np.flatnonzero([True if ref in path else False for path in ontocat_refs])[0] for ref in ontocats['ref']]
ontocats['parent_ref'] = [ontocat_refs[y[i]][ontocat_refs[y[i]].index(ref) - 1] if ontocat_refs[y[i]].index(ref) > 0 else None for i, ref in enumerate(ontocats['ref'])]
ontocats['parent_id'] = [x[parent] if parent is not None else None for parent in ontocats['parent_ref']]


# Find membership of onto categories
ontocats['node_ids'] = [[node['id'] for node, path in zip(nodes_mitre, ontocat_refs) if ref in path] for ref in ontocats['ref']]


# Placeholder for hyperedges
ontocats['hyperedge_ids'] = [[] for i in range(num_ontocats)]

# Switch to row-wise structure
ontocats_ = [{k: ontocats[k][i] for k in ontocats.keys()} for i in range(num_ontocats)]


i = x = y = None
del i, x, y

# time: 3.89 s


# %%
# ## Additional node attributes

# Calculate node degrees
nodes_mitre, __, __ = emlib.calculate_node_degrees(nodes_mitre, edges_mitre)

# Get node-wise belief score from `edges`
nodes_mitre, __ = emlib.calculate_node_belief(nodes_mitre, edges_mitre, mode = 'max')


# Placeholder values
for node in nodes_mitre:
    node['x'] = float(0.0)
    node['y'] = float(0.0)
    node['z'] = float(0.0)
    node['cluster_level'] = 0
    node['cluster_ids'] = []


# %%

# Output model node data
with open(f'./dist/v2/nodeData.jsonl', 'w') as x:

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
        'cluster_level': '<int> (placeholder for clustering)',
        'cluster_ids': '<array of int> (placeholder for cluster)'
    }
    json.dump(y, x)
    x.write('\n')

    # Data
    for node in nodes_mitre:
        z = {k: node[k] for k in ['id', 'x', 'y', 'z', 'in_degree', 'out_degree', 'belief', 'db_ref_priority', 'grounded_onto', 'ontocat_level', 'ontocat_ids', 'cluster_level', 'cluster_ids']}

        json.dump(z, x)
        x.write('\n')


# %%
# Output onto category data
with open(f'./dist/v2/ontocats.jsonl', 'w') as x:

    # Description
    y = {
        'id': '<int> unique ID for the node in the KB graph as specified in `nodes.jsonl`',
        'ref': '<float> position of the node in the graph layout',
        'name': '<float> position of the node in the graph layout',
        'size': '<float> position of the node in the graph layout',
        'level': '<int> in-degree in the KB graph',
        'parent_id': '<int> out-degree in the KB graph',
        'node_ids': '<float> max of the belief scores of all adjacent edges in the KB graph',
        'hyperedge_ids': '<array of int> (placeholder for hyperedges)'
    }
    json.dump(y, x)
    x.write('\n')

    # Data
    for ontocat in ontocats_:
        z = {k: ontocat[k] for k in ['id', 'ref', 'name', 'size', 'level', 'parent_id', 'node_ids', 'hyperedge_ids']}

        json.dump(z, x)
        x.write('\n')


node = ontocat = x = y = z = None
del node, ontocat, x, y, z


# %%


