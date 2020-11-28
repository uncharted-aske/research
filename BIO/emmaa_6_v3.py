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
x = ['subj', 'obj', 'enz', 'sub']
for i, t in enumerate(x):
    x[i] = {s[t]['name']: s[t]['db_refs'] for s in statements if t in s.keys()}

y = {**x[0], **x[1], **x[2], **x[3]}

for i in range(num_nodes):
    try:
        nodes[i]['db_refs'] = y[nodes[i]['name']]

        if len(set(nodes[i]['db_refs'].keys()) - {'TEXT'}) > 0:
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

# Calculate node degrees
nodes_mitre, __, __ = emlib.calculate_node_degrees(nodes_mitre, edges_mitre)

# Get node-wise belief score from `edges`
nodes_mitre, __ = emlib.calculate_node_belief(nodes_mitre, edges_mitre, mode = 'max')


# %%[markdown]
# # Categorize model nodes by ontology categories

# %%[markdown]
# ## Load the INDRA ontology

with open('./data/indra_ontology_v1.3.json', 'r') as x:
    ontoJSON = json.load(x)

# Remove 'xref' links
ontoJSON['links'] = [link for link in ontoJSON['links'] if link['type'] != 'xref']


# %%[markdown]
# ## Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces = emlib.generate_ordered_namespace_list(nodes_mitre, ontoJSON, namespaces_priority)


# %%
# Compute model-ontology mapping
nodeData_ontoRefs = []
for node in nodes_mitre:

    if len(node['info']['links']) < 1:
        names = ['not-grounded']
        k = names[0]
    else:
        names = [link[0] for link in node['info']['links']]

        # Use first matching namespace in the ordered common list
        i = np.flatnonzero([True if name in names else False for name in namespaces])[0]
        j = np.flatnonzero(np.asarray(names) == namespaces[i])[0]
        k = f"{node['info']['links'][j][0]}:{node['info']['links'][j][1]}"

    nodeData_ontoRefs.append(k)


names = node = i = j = k = None
del names, node, i, j, k


# %%
# Load the ontology graph as a `networkx` object
ontoG = nx.readwrite.json_graph.node_link_graph(ontoJSON)

# %%[markdown]
# ## Extract components and find their roots

# %%
# Extract components, sorted by size
ontoSubs = sorted(nx.weakly_connected_components(ontoG), key = len, reverse = True)

# Find the root nodes of each component (degree = 0 or out-degree = 0)
z = [np.flatnonzero([True if ontoG.out_degree(node) < 1 else False for node in sub]) for sub in ontoSubs]
ontoSubRoots = [[list(ontoSubs[i])[j] for j in indices] for i, indices in enumerate(z)]
ontoSubRoots_num = np.sum([True if len(indices) > 1 else False for indices in z])

# List onto node names/refs
# ontoRefs = nx.nodes(ontoG)

# %%[markdown]
# ## Index all model nodes that can be mapped to the ontology graph

# %%
%%time

# Initialize and set the ontological level of the unmappable nodes to -1
x = np.flatnonzero([True if i in nx.nodes(ontoG) else False for i in nodeData_ontoRefs])
nodeData_ontoLevels = np.zeros((len(nodes_mitre), ), dtype = np.int64)
nodeData_ontoPaths = list(np.zeros((len(nodes_mitre), ), dtype = np.int64))
for i in range(len(nodes_mitre)):
    if i not in x:
        nodeData_ontoLevels[i] = -1
        nodeData_ontoPaths[i] = [nodeData_ontoRefs[i]]


# Find subgraph index of each mapped model node
# * Limited to non-trivial subgraphs
# * Set to -1 if a node is mapped to a trivial subgraph
y = np.empty(x.shape, dtype = np.int64)
for i, k in enumerate(x):
    j = np.flatnonzero([True if nodeData_ontoRefs[k] in sub else False for sub in ontoSubs[:ontoSubRoots_num]])
    if len(j) == 1:
        y[i] = j[0]
    else:
        y[i] = -1


# Find shortest path between each onto-mapped model node and any target root node amongst the ontology subgraphs
for i, j in zip(x, y):

    source = nodeData_ontoRefs[i]

    # Trivial ontology subgraphs
    if j == -1:
        nodeData_ontoLevels[i] = 0
        nodeData_ontoPaths[i] = [source]

    # All other subgraphs
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
        nodeData_ontoPaths[i] = z[0][::-1]
        nodeData_ontoLevels[i] = len(z[0]) - 1


i = j = p = x = y = z = source = target = None
del i, j, p, x, y, z, source, target

# time: 10 m 21 s

# %%

# Ensure that identical onto nodes share the same lineage (i.e. path to their ancestor) for hierarchical uniqueness
nodeData_ontoPaths_reduce = nodeData_ontoPaths[:]
m = max([len(path) for path in nodeData_ontoPaths])
n = len(nodes_mitre)
for i in range(1, m):

    # All nodes
    x = [path[i] if len(path) > i else '' for path in nodeData_ontoPaths]

    # All unique nodes
    y = list(set(x) - set(['']))

    # Mapping from all nodes to unique nodes
    xy = [y.index(node) if node is not '' else '' for node in x]

    # Choose the path segment of the first matching node for each unique node
    z = [nodeData_ontoPaths[x.index(node)][:i] for node in y]
    
    # Substitute path segments
    for j in range(n):
        if xy[j] is not '':
            nodeData_ontoPaths_reduce[j][:i] = z[xy[j]]
        else:
            nodeData_ontoPaths_reduce[j][:i] = nodeData_ontoPaths[j][:i]


x = y = z = xy = i = j = m = n = None
del x, y, z, xy, i, j, m, n


# %%
%%time

# Generate list of mapped ontology categories, sorted by size
ontoCats = {}
ontoCats['ref'], ontoCats['size'] = np.unique([node for path in nodeData_ontoPaths_reduce for node in path], return_counts = True)

num_ontoCats = len(ontoCats['ref'])
i = np.argsort(ontoCats['size'])[::-1]
ontoCats['ref'] = ontoCats['ref'][i]
ontoCats['size'] = ontoCats['size'][i]
ontoCats['id'] = list(range(num_ontoCats))


# Get the mapped onto category names
x = dict(ontoG.nodes(data = 'name', default = None))
ontoCats['name'] = list(np.empty((num_ontoCats, )))
for i, ontoRef in enumerate(ontoCats['ref']):
    try:
        ontoCats['name'][i] = x[ontoRef]
    except:
        ontoCats['name'][i] = ''


# Get onto level of each category
i = max([len(path) for path in nodeData_ontoPaths_reduce])
x = [np.unique([path[j] if len(path) > j else '' for path in nodeData_ontoPaths_reduce]) for j in range(i)]
ontoCats['ontoLevel'] = [np.flatnonzero([ontoRef in y for y in x])[0] for ontoRef in ontoCats['ref']]


# Get numeric id version of nodeData_ontoPaths_reduce
x = {k: v for k, v in zip(ontoCats['ref'], ontoCats['id'])}
nodeData_ontoPaths_id = [[x[node] for node in path] for path in nodeData_ontoPaths_reduce]


# Get parent category id for each category (for root nodes, parentID = None)
y = [np.flatnonzero([True if ref in path else False for path in nodeData_ontoPaths_reduce])[0] for ref in ontoCats['ref']]
ontoCats['parent'] = [nodeData_ontoPaths_reduce[y[i]][nodeData_ontoPaths_reduce[y[i]].index(ontoRef) - 1] if nodeData_ontoPaths_reduce[y[i]].index(ontoRef) > 0 else None for i, ontoRef in enumerate(ontoCats['ref'])]
ontoCats['parentID'] = [x[parent] if parent is not None else None for parent in ontoCats['parent']]


# Find membership of onto categories
ontoCats['nodeIDs'] = [[node['id'] for node, path in zip(nodes_mitre, nodeData_ontoPaths_reduce) if ontoRef in path] for ontoRef in ontoCats['ref']]


i = x = y = None
del i, x, y

# time: 3.89 s

# %%

# Output intersected model nodes and edges
with open(f'./dist/v3/nodes_mitre.jsonl', 'w') as x:
    for i in range(num_nodes):
        json.dump(nodes_mitre[i], x)

with open(f'./dist/v3/edges_mitre.jsonl', 'w') as x:
    for i in range(num_edges):
        json.dump(edges_mitre[i], x)


# Output model node data
with open(f'./dist/v3/nodeData.jsonl', 'w') as x:

    # Description
    y = {
        'id': '<int> unique ID for the node in the KB graph as specified in `nodes.jsonl`',
        'x': '<float> position of the node in the graph layout (symmetric Laplacian + UMAP 3D)',
        'y': '<float> position of the node in the graph layout (symmetric Laplacian + UMAP 3D)',
        'z': '<float> position of the node in the graph layout (symmetric Laplacian + UMAP 3D)',
        'degreeIn': '<int> in-degree in the KB graph',
        'degreeOut': '<int> out-degree in the KB graph',
        'belief': '<float> max of the belief scores of all adjacent edges in the KB graph',
        'ontoID': '<str> unique ref ID of the INDRA ontology (v1.3) node to which this KB node is mapped', 
        'ontoLevel': '<int> hierarchy level of the ontology node (`-1` if not mappable)',
        'clusterIDs': '<array of int> ordered list of cluster IDs (see `clusters.jsonl`) to which this node is mapped (cluster hierarchy = INDRA ontology v1.3, order = root-to-leaf)'
    }
    json.dump(y, x)
    x.write('\n')

    # Data
    for i in range(len(nodesKB)):
        z = {
            'id': int(nodesKB[i]['id']),
            'x': float(nodesKB_pos[i, 0]), 
            'y': float(nodesKB_pos[i, 1]), 
            'z': float(nodesKB_pos[i, 2]), 
            'degreeOut': int(nodesKB_degrees[i, 0]),
            'degreeIn': int(nodesKB_degrees[i, 1]),
            'belief': float(nodesKB_belief[i]),
            'ontoID': nodesKB_ontoIDs[i], 
            'ontoLevel': int(nodesKB_ontoLevels[i]),
            'clusterIDs': nodesKB_ontoPaths_id[i]
        }

        json.dump(z, x)
        x.write('\n')


# i = x = y = z = None
# del i, x, y, z













# %%


