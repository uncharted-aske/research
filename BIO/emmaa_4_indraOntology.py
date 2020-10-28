# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Idea: 
# * Explore the INDRA Ontology
# * Cluster nodes 

# %%
import json
import time
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import pickle
import umap
import sklearn as skl
import re

import emmaa_lib as emlib
import importlib
# `importlib.reload(emlib)`

# %%
np.random.seed(0)

# %%[markdown]
%%time

# ## Load data
# 
# INDRA Ontology v1.3 from Ben Gyori
with open('./data/indra_ontology_v1.3.json', 'r') as x:
    ontoJSON = json.load(x)

# `nodes` data from the Covid-19 knowledge base
nodesKB = {}
with open('./data/covid19-snapshot_sep18-2020/processed/nodes.jsonl', 'r') as x:
    nodesKB = [json.loads(i) for i in x]

# `nodes` data from the Covid-19 knowledge base (curated)
nodesKB_curated = {}
with open('./dist/nodes_curated_belief0.jsonl', 'r') as x:
    nodesKB_curated = [json.loads(i) for i in x]

# `nodes` data from the Covid-19 knowledge base (belief score > 0.95)
nodesKB_belief95 = {}
with open('./dist/nodes_belief95_curatedTested.jsonl', 'r') as x:
    nodesKB_belief95 = [json.loads(i) for i in x]
with open('./dist/nodes_belief95_curatedTested.jsonl', 'r') as x:
    nodesKB_belief95.extend([json.loads(i) for i in x])


x = None
del x

# %%
# Data structure of the ontology JSON

print(f"'ontoJSON': {type(ontoJSON)}")
for k in ontoJSON.keys():
    if isinstance(ontoJSON[k], list):
        print(f"{'':<3}'{k}': {type(ontoJSON[k])} ({len(ontoJSON[k])})")
        print(f"{'':<6}'[0]': {type(ontoJSON[k][0])}")
        for l in ontoJSON[k][0].keys():
            print(f"{'':<9}'{l}': {type(ontoJSON[k][0][l])}")
    else:
        print(f"{'':<3}'{k}': {type(ontoJSON[k])}")

# 'ontoJSON': <class 'dict'>
#    'directed': <class 'bool'>
#    'multigraph': <class 'bool'>
#    'graph': <class 'dict'>
#    'nodes': <class 'list'> (2100558)
#       '[0]': <class 'dict'>
#          'name': <class 'str'>
#          'id': <class 'str'>
#    'links': <class 'list'> (1437172)
#       '[0]': <class 'dict'>
#          'type': <class 'str'>
#          'source': <class 'str'>
#          'target': <class 'str'>

# Note: 
# * not all ontology nodes have a 'name' attribute
# * ontology node 'id' is the form of 'NAMESPACE:ID'

# %%
# Data structure of the KB node data
print(f"'nodesKB': {type(nodesKB)} ({len(nodesKB)})")
for k in nodesKB[0].keys():
    print(f"{'':<3}'{k}': {type(nodesKB[0][k])}")
    if isinstance(nodesKB[0][k], dict):
        for l in nodesKB[0][k].keys():
            print(f"{'':<6}'{k}': {type(nodesKB[0][k][l])}")

# %%
# Count namespace usage in ontology graph
x = [re.findall('\w{1,}(?=:)', node['id'])[0] for node in ontoJSON['nodes']]
y, z = np.unique(x, return_counts = True)
i = np.argsort(z)[::-1]
namespacesOnto = [[name, count / np.sum(z) * 100] for name, count in zip(y[i], z[i])]


# Ben Gyori remarks that the INDRA ontology has an explicit priority order amongst the namespaces. 
# This order is currently: 
namespacesPriority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']


# Count namespace usage in knowledge graph (one per node)
# * empty list -> ''
# * not in priority list -> just append
x = []
for node in nodesKB:
    if len(node['info']['links']) > 0:
        names = [link[0] for link in node['info']['links']]
    else:
        names = ['not-grounded']
    
    # Check against priority list
    y = np.flatnonzero([True if name in names else False for name in namespacesPriority])
    if len(y) > 0:
        i = y[0]
        x.append(namespacesPriority[i])
    else:

        # Check against ontology list
        z = np.flatnonzero([True if name[0] in names else False for name in namespacesOnto])
        if len(z) > 0:
            i = z[0]
            x.append(namespacesOnto[i][0])
        else:
            x.append(names[0])

y, z = np.unique(x, return_counts = True)
i = np.argsort(z)[::-1]
namespacesKB = [[name, count / np.sum(z) * 100] for name, count in zip(y, z)]


# Create combined namespace list
x = list(set([name for name, __ in namespacesKB]) - set(namespacesPriority))
y = list(set([name for name, __ in namespacesOnto]) - set(namespacesPriority) - set([name for name, __ in namespacesKB]))
namespaces = namespacesPriority + x + y

# Collate data
x = [name for name, __ in namespacesKB]
y = [name for name, __ in namespacesOnto]
z = np.array([[next((percent for n, percent in N if n == name), 0.0) for N in [namespacesKB, namespacesOnto]] for i, name in enumerate(namespaces)])

# %%[markdown]
# Print result
print(f"| {'Namespaces':<20} | {'KB [%]':>10} | {'Onto [%]':>10} |")
print(f"|:{'-' * 20} | {'-' * 10}:| {'-' * 10}:|")
__ = [print(f"| {name:<20} | {z[i][0]:>10.2f} | {z[i][1]:>10.2f} |") for i, name in enumerate(namespaces)]

name = names = node = x = y = z = None
del x, y, z, name, names, node 


# | Namespaces           |     KB [%] |   Onto [%] |
# |:-------------------- | ----------:| ----------:|
# | FPLX                 |       1.15 |       0.03 |
# | UPPRO                |       0.05 |      28.53 |
# | HGNC                 |      24.40 |       2.23 |
# | UP                   |      12.39 |      34.75 |
# | CHEBI                |      16.56 |       5.61 |
# | GO                   |       5.71 |       2.12 |
# | MESH                 |      19.20 |      14.24 |
# | MIRBASE              |       0.00 |       0.35 |
# | DOID                 |       0.55 |       0.48 |
# | HP                   |       1.11 |       0.71 |
# | EFO                  |       1.15 |       0.46 |
# | not-grounded         |      13.78 |       0.00 |
# | NXPFA                |       0.16 |       0.00 |
# | PUBCHEM              |       0.22 |       4.47 |
# | GENBANK              |       0.02 |       0.00 |
# | CVCL                 |       0.00 |       0.00 |
# | RGD                  |       0.00 |       0.00 |
# | CO                   |       0.00 |       0.00 |
# | PR                   |       0.01 |       0.00 |
# | NCIT                 |       0.19 |       0.86 |
# | BTO                  |       0.01 |       0.00 |
# | REFSEQ_PROT          |       0.06 |       0.00 |
# | CHEMBL               |       0.00 |       2.91 |
# | IP                   |       0.51 |       0.01 |
# | PF                   |       2.75 |       0.00 |
# | INDRA_MODS           |       0.00 |       0.00 |
# | ECCODE               |       0.00 |       0.00 |
# | DRUGBANK             |       0.00 |       0.65 |
# | INDRA_ACTIVITIES     |       0.00 |       0.00 |
# | TAXONOMY             |       0.00 |       0.00 |
# | CAS                  |       0.00 |       1.06 |
# | LINCS                |       0.00 |       0.14 |
# | HMDB                 |       0.00 |       0.37 |
# | HGNC_GROUP           |       0.00 |       0.00 |

# %%
# Generate the id of KB nodes
nodesKB_id = []
for node in nodesKB:
    if len(node['info']['links']) > 0:
        names = [link[0] for link in node['info']['links']]

        # Check against namespace list (priority + KB + Onto)
        i = np.flatnonzero([True if name in names else False for name in namespaces])[0]
        j = np.flatnonzero(np.asarray(names) == namespaces[i])[0]
        k = f"{node['info']['links'][j][0]}:{node['info']['links'][j][1]}"

    else:
        names = ['not-grounded']
        k = names[0]

    nodesKB_id.append(k)


x, y = np.unique(nodesKB_id, return_counts = True)
i = np.argsort(y)[::-1]
x = x[i]
y = y[i]

j = len(np.flatnonzero(y > 1))
k = np.sum(y[:j]) - y[0]

# 5161 KB nodes (13.78%) are ungrounded.
# 1148 KB nodes (3.07%) share the same 453 node ids.

# %%[markdown]

i = 20
print(f"| {'Node ID in KB':<16} | {'KB Count':>15} |")
print(f"|:{'-' * 16} | {'-' * 15}:|")
__ = [print(f"| {nodeKB_id:<16} | {count:>15} |") for nodeKB_id, count in zip(x[:i], y[:i])]
print(f"| {'...':<16} | {' ' * 15} |")
print(f"|:{'-' * 16} | {'-' * 15}:|")
print(f"| {'':<16} | {np.sum(y[:j]):>15} |")


# | Node ID in KB    |        KB Count |
# |:---------------- | ---------------:|
# | not-grounded     |            5161 |
# | UP:P0C6V4        |              11 |
# | UP:Q89613        |              10 |
# | UP:P81460        |              10 |
# | UP:P69208        |              10 |
# | UP:P19711        |               9 |
# | UP:P0C5A6        |               8 |
# | UP:P19194        |               7 |
# | UP:P03595        |               7 |
# | UP:B3VML1        |               6 |
# | UP:P35849        |               6 |
# | UP:Q1HVG1        |               6 |
# | UP:Q18LE5        |               6 |
# | PF:PF06460       |               6 |
# | PF:PF01661       |               5 |
# | UP:Q86199        |               5 |
# | UP:Q65943        |               5 |
# | UP:Q9G050        |               5 |
# | PUBCHEM:86583374 |               5 |
# | UP:P42540        |               5 |
# | ...              |                 |
# |:---------------- | ---------------:|
# |                  |            6309 |


name = names = node = x = y = z = i = j = k = None
del i, j, k, x, y, z, name, names, node 

# %%[markdown]
# List of link types
x = [link['type'] for link in ontoJSON['links']]
y, z = np.unique(x, return_counts = True)
i = np.argsort(z)[::-1]

print(f"| {'Link Types':<20} | {'Counts':>10} | {'Percent [%]':>15} |")
print(f"| {'-' * 20} | {'-' * 10}:| {'-' * 15}:|")
__ = [print(f'| {name:<20} | {z[i][j]:>10} | {z[i][j] / len(x) * 100:>15.0f} |') for j, name in enumerate(y[i])]


# | Link Types           |     Counts |    Percent [%] |
# | -------------------- | ----------:| --------------:|
# | partof               |     608706 |          42.35 |
# | xref                 |     420348 |          29.25 |
# | isa                  |     408118 |          28.40 |

# %%[markdown]
# ## Filter out `xref` links

x = np.flatnonzero([True if link['type'] != 'xref' else False for link in ontoJSON['links']])
links = [ontoJSON['links'][i] for i in x]

ontoJSON['links'] = links.copy()

i = x = y = z = links = None
del i, x, y, z, links

# %%[markdown]
# ## Load the ontology graph as a `networkx` object
ontoG = nx.readwrite.json_graph.node_link_graph(ontoJSON)

# %%
print(nx.info(ontoG))

# Name: 
# Type: DiGraph
# Number of nodes: 2100558
# Number of edges: 1016824
# Average in degree:   0.4841
# Average out degree:   0.4841

# %%
# Test graph properties

# Check uniqueness of ontology node ids
ontoNames = nx.get_node_attributes(ontoG, 'name')
y = list(ontoNames.keys())
i, j = np.unique(y, return_counts = True)
print(np.sum(j > 2) == 0)
# True

x = np.sum([True if i in ontoNames.keys() else False for i in nodesKB_id]) / len(nodesKB_id) * 100
print(f"{x:10.2f}%")
y = len(set(ontoNames.keys()) & set(nodesKB_id)) / len(set(nodesKB_id)) * 100
print(f"{y:10.2f}%")
# 82.01% (96.19%, excluding duplicates) of KB node ids are found in the ontology graph.


# x = np.sum([True if i in nodesKB_id else False for i in ontoNames.keys()]) / len(ontoNames.keys()) * 100
x = len(set(ontoNames.keys()) & set(nodesKB_id)) / len(ontoNames.keys()) * 100
print(f"{x:10.2f}%")
# 1.57% of ontology node ids are found in the KB graph (excluding duplicates).

print(nx.is_directed_acyclic_graph(ontoG))
# True

print(nx.is_strongly_connected(ontoG))
# False

print(nx.number_strongly_connected_components(ontoG))
# 2100558

print(nx.is_weakly_connected(ontoG))
# False

print(nx.number_weakly_connected_components(ontoG))
# 1272734

# Generate components, sorted by size
ontoSub = sorted(nx.weakly_connected_components(ontoG), key = len, reverse = True)

# Check overlap between ontology component nodes and KB nodes
i = 5000
x = [len(sub) for sub in ontoSub]
y = [len(set(nodesKB_id) & set(sub)) for sub in ontoSub[:i]]
z = np.asarray(y) / len(nodesKB_id) * 100

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
__ = ax[0].plot(x, label = 'Ontology Component Size')
__ = ax[0].plot(y, label = 'Intersection with KB Graph')
__ = plt.setp(ax[0], xlabel = 'Ontology Component Index', ylabel = 'Number of Nodes', xscale = 'log', yscale = 'log', title = 'Size of Ontology Components')
__ = ax[1].plot(np.cumsum(z))
__ = ax[1].plot([0, len(ontoNames.keys())], [96.19, 96.19], linestyle = '--', label = 'Total')
__ = plt.setp(ax[1], xlim = plt.getp(ax[0], 'xlim'), ylim = (0, 100), xlabel = 'Ontology Component Index', ylabel = 'Cumulative Fraction of the KB Graph [%]', xscale = 'log', yscale = 'linear', title = '')
__ = plt.setp(ax[1], title = 'Set Intersection between Ontology Components and KB Graphs')
__ = ax[0].legend()
fig.savefig('./figures/ontoComponentSize.png', dpi = 150)




# %% 
# Re-do for the other two `nodes` datasets

# Generate the id of KB nodes
nodesKB_id_curated = []
for node in nodesKB_curated:
    if len(node['info']['links']) > 0:
        names = [link[0] for link in node['info']['links']]

        # Check against namespace list (priority + KB + Onto)
        i = np.flatnonzero([True if name in names else False for name in namespaces])[0]
        j = np.flatnonzero(np.asarray(names) == namespaces[i])[0]
        k = f"{node['info']['links'][j][0]}:{node['info']['links'][j][1]}"

    else:
        names = ['not-grounded']
        k = names[0]

    nodesKB_id_curated.append(k)

nodesKB_id_belief95 = []
for node in nodesKB_belief95:
    if len(node['info']['links']) > 0:
        names = [link[0] for link in node['info']['links']]

        # Check against namespace list (priority + KB + Onto)
        i = np.flatnonzero([True if name in names else False for name in namespaces])[0]
        j = np.flatnonzero(np.asarray(names) == namespaces[i])[0]
        k = f"{node['info']['links'][j][0]}:{node['info']['links'][j][1]}"

    else:
        names = ['not-grounded']
        k = names[0]

    nodesKB_id_belief95.append(k)



# Check overlap between ontology component nodes and KB nodes
i = 5000
x = [len(sub) for sub in ontoSub]
y = [len(set(nodesKB_id_curated) & set(sub)) for sub in ontoSub[:i]]
z = np.asarray(y) / len(nodesKB_id_curated) * 100

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
__ = ax[0].plot(x, label = 'Ontology Component Size')
__ = ax[0].plot(y, label = 'Intersection with KB Graph')
__ = plt.setp(ax[0], xlabel = 'Ontology Component Index', ylabel = 'Number of Nodes', xscale = 'log', yscale = 'log', title = 'Size of Ontology Components')
__ = ax[1].plot(np.cumsum(z))

j = len(set(ontoNames.keys()) & set(nodesKB_id_curated)) / len(set(nodesKB_id_curated)) * 100
__ = ax[1].plot([0, len(ontoNames.keys())], [j, j], linestyle = '--', label = 'Total')

__ = plt.setp(ax[1], xlim = plt.getp(ax[0], 'xlim'), ylim = (0, 100), xlabel = 'Ontology Component Index', ylabel = 'Cumulative Fraction of the KB Graph [%]', xscale = 'log', yscale = 'linear', title = '')
__ = plt.setp(ax[1], title = 'Set Intersection between Ontology Components and KB Graphs')
__ = ax[0].legend()
fig.savefig('./figures/ontoComponentSize_curated.png', dpi = 150)


i = 5000
x = [len(sub) for sub in ontoSub]
y = [len(set(nodesKB_id_belief95) & set(sub)) for sub in ontoSub[:i]]
z = np.asarray(y) / len(nodesKB_id_belief95) * 100

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
__ = ax[0].plot(x, label = 'Ontology Component Size')
__ = ax[0].plot(y, label = 'Intersection with KB Graph')
__ = plt.setp(ax[0], xlabel = 'Ontology Component Index', ylabel = 'Number of Nodes', xscale = 'log', yscale = 'log', title = 'Size of Ontology Components')
__ = ax[1].plot(np.cumsum(z))

j = len(set(ontoNames.keys()) & set(nodesKB_id_belief95)) / len(set(nodesKB_id_belief95)) * 100
__ = ax[1].plot([0, len(ontoNames.keys())], [j, j], linestyle = '--', label = 'Total')

__ = plt.setp(ax[1], xlim = plt.getp(ax[0], 'xlim'), ylim = (0, 100), xlabel = 'Ontology Component Index', ylabel = 'Cumulative Fraction of the KB Graph [%]', xscale = 'log', yscale = 'linear', title = '')
__ = plt.setp(ax[1], title = 'Set Intersection between Ontology Components and KB Graphs')
__ = plt.setp(ax[1], xlim = plt.getp(ax[0], 'xlim'), ylim = (0, 100), xlabel = 'Ontology Component Index', ylabel = 'Cumulative Fraction of the KB Graph [%]', xscale = 'log', yscale = 'linear', title = '')
__ = plt.setp(ax[1], title = 'Set Intersection between Ontology Components and KB Graphs')
__ = ax[0].legend()
fig.savefig('./figures/ontoComponentSize_belief95.png', dpi = 150)


# %%
# Table of properties for the largest weakly connected components of the ontology graph

k = 30
y = [[  
        i, 
        nx.is_arborescence(nx.reverse_view(ontoG.subgraph(nodes))),
        len(nodes), 
        len(nodes) / nx.number_of_nodes(ontoG) * 100, 
        nx.algorithms.dag.dag_longest_path_length(ontoG.subgraph(nodes)),
        np.sum([True if out_degree == 0 else False for __, out_degree in list(ontoG.subgraph(nodes).out_degree())]), 
        np.sum([True if in_degree == 0 else False for __, in_degree in list(ontoG.subgraph(nodes).in_degree())]),
        max([out_degree for __, out_degree in list(ontoG.subgraph(nodes).out_degree())])
    ] 
    for i, nodes in enumerate(ontoSub[:k])]

print(f"| {'Index':<5} | {'Rooted DAG?':<12} | {'# Nodes':<10} | {'Relative # [%]':<15} | {'Max Depth':<10} | {'# Roots':<10} | {'# Leafs':<10} | {'Max # Parents':<15} |")
print(f"|:{'-' * 5}:|:{'-' * 12}:|:{'-' * 10}:|:{'-' * 15}:|:{'-' * 10}:|:{'-' * 10}:|:{'-' * 10}:|:{'-' * 15}:|")
__ = [print(f'| {i:>5} | {j!s:>12} | {k:>10} | {l:>15.2f} | {m:>10} | {n:>10} | {o:>10} | {p:>15} |') for i, j, k, l, m, n, o, p in y]
print(f"| {'...':>5} | {' ' * 12} | {' ' * 10} | {' ' * 15} | {' ' * 10} | {' ' * 10} | {' ' * 10} | {' ' * 15} |")

# %%[markdown]
# | Index | Rooted DAG?  | # Nodes    | Relative # [%]  | Max Depth  | # Roots    | # Leafs    | Max # Parents   |
# |:-----:|:------------:|:----------:|:---------------:|:----------:|:----------:|:----------:|:---------------:|
# |     0 |        False |     117180 |            5.58 |         37 |          3 |      93377 |              30 |
# |     1 |        False |      40324 |            1.92 |         18 |          2 |      22215 |              10 |
# |     2 |        False |      29390 |            1.40 |         15 |        106 |      19758 |               6 |
# |     3 |        False |      14965 |            0.71 |         15 |          1 |       9912 |               5 |
# |     4 |        False |      10041 |            0.48 |         12 |          1 |       7832 |               7 |
# |     5 |        False |       5537 |            0.26 |          9 |         69 |       4293 |               5 |
# |     6 |        False |       4184 |            0.20 |         14 |          1 |       2783 |               5 |
# |     7 |        False |       1348 |            0.06 |          4 |         20 |       1299 |               5 |
# |     8 |        False |        656 |            0.03 |          6 |          6 |        594 |               3 |
# |     9 |         True |        427 |            0.02 |          2 |          1 |        409 |               1 |
# |    10 |        False |        355 |            0.02 |          4 |          5 |        305 |               2 |
# |    11 |        False |        236 |            0.01 |          2 |          2 |        210 |               2 |
# |    12 |        False |        179 |            0.01 |          5 |          3 |        144 |               3 |
# |    13 |        False |         93 |            0.00 |          2 |          5 |         84 |               2 |
# |    14 |         True |         81 |            0.00 |          2 |          1 |         78 |               1 |
# |    15 |         True |         74 |            0.00 |          2 |          1 |         70 |               1 |
# |    16 |         True |         70 |            0.00 |          3 |          1 |         55 |               1 |
# |    17 |         True |         69 |            0.00 |          1 |          1 |         68 |               1 |
# |    18 |        False |         67 |            0.00 |          3 |          1 |         53 |               2 |
# |    19 |        False |         66 |            0.00 |          1 |          4 |         62 |               2 |
# |    20 |         True |         65 |            0.00 |          3 |          1 |         49 |               1 |
# |    21 |        False |         63 |            0.00 |          4 |          2 |         53 |               2 |
# |    22 |         True |         61 |            0.00 |          5 |          1 |         31 |               1 |
# |    23 |         True |         60 |            0.00 |          2 |          1 |         35 |               1 |
# |    24 |         True |         55 |            0.00 |          3 |          1 |         49 |               1 |
# |    25 |        False |         51 |            0.00 |          2 |          3 |         45 |               2 |
# |    26 |        False |         48 |            0.00 |          2 |          2 |         44 |               2 |
# |    27 |         True |         48 |            0.00 |          1 |          1 |         47 |               1 |
# |    28 |         True |         45 |            0.00 |          1 |          1 |         44 |               1 |
# |    29 |         True |         45 |            0.00 |          1 |          1 |         44 |               1 |
# |   ... |              |            |                 |            |            |            |                 |


# %%







# %%
%%time

# Trophic levels vs. length of shortest path to root

i = 10

# Trophic levels
y = nx.algorithms.centrality.trophic_levels(ontoG.subgraph(ontoSub[i]))

# %%
%%time

# Length of shortest path to a root
roots = [list(ontoG.subgraph(x[i]))[j] for j in np.flatnonzero([True if out_degree == 0 else False for name, out_degree in list(ontoG.subgraph(x[i]).out_degree())])]

z = []
for source in y.keys():

    w = []
    for root in roots:
        try:
            w.append(nx.algorithms.shortest_paths.generic.shortest_path_length(ontoG.subgraph(x[i]), source = source, target = root))
        except:
            w.extend([])

    z.append(min(w))

# %%
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
ax.scatter(np.asarray(list(y.values())) - 1, z, s = 100, alpha = 0.5, marker = '.')
__ = plt.setp(ax, xlabel = 'Trophic Level - 1', ylabel = 'Length of Shortest Path to a Root Node', aspect = 1.0)


# %%



i = 9
print(nx.is_weakly_connected(ontoG.subgraph(x[i])))
# True

print(nx.is_branching(nx.reverse_view(ontoG.subgraph(x[i]))))
# True

print(nx.is_arborescence(nx.reverse_view(ontoG.subgraph(x[i]))))
# True



print(f'{nx.is_weakly_connected(ontoG.subgraph(x[i]))} {nx.is_branching(nx.reverse_view(ontoG.subgraph(x[i])))} {nx.is_arborescence(nx.reverse_view(ontoG.subgraph(x[i])))}')




i = 9
y = list((ontoG.subgraph(x[i])).out_degree)
k = np.argsort([j for __, j in y])[0]
# z = [nx.algorithms.shortest_paths.generic.shortest_path_length(nx.reverse_view(ontoG.subgraph(x[i])), source = y[k][0], target = t) for t in nx.nodes(ontoG.subgraph(x[i]))]

# w = nx.spring_layout(ontoG.subgraph(x[i]))
# l = [list(np.flatnonzero(np.asarray(z) == j)) for j in np.unique(z)]
# n = [[list(nx.nodes(ontoG.subgraph(x[i])))[m] for m in j] for j in l]

# w = nx.shell_layout(ontoG.subgraph(x[i]), nlist = n)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
# __ = nx.draw(ontoG.subgraph(x[i]), pos = w, ax = ax, arrows = False, with_labels = False, node_size = 5, width = 0.1)
__ = nx.draw_kamada_kawai(ontoG.subgraph(x[i]), ax = ax, arrows = False, with_labels = False, node_size = 5, width = 0.1)

i = 9
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 8))
__ = nx.draw_planar(ontoG.subgraph(x[i]), ax = ax, arrows = False, with_labels = True, node_size = 5, width = 0.1)


# %%
__ = [print(f'{i:<5} {nx.is_weakly_connected(ontoG.subgraph(x[i]))} {nx.is_branching(nx.reverse_view(ontoG.subgraph(x[i])))} {nx.is_arborescence(nx.reverse_view(ontoG.subgraph(x[i])))}') for i in range(20)]

# Only works when there is a single root
i = 9
y = list((ontoG.subgraph(x[i])).out_degree)
np.unique([j for __, j in y], return_counts = True)
k = np.argsort([j for __, j in y])[0]

# w = hierarchy_pos(nx.reverse_view(ontoG.subgraph(x[i])), root = y[k][0])
w = hierarchy_pos(nx.reverse_view(ontoG.subgraph(x[i])), root = y[k][0], width = 2 * np.pi)
w = {u: (r * np.cos(t), r * np.sin(t)) for u, (t, r) in w.items()}

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = nx.draw(ontoG.subgraph(x[i]), pos = w, ax = ax, arrows = False, with_labels = True, node_size = 5, width = 0.1)



# %%
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, leaf_vs_root_factor = 0.5):

    '''
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.  

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).
    
    There are two basic approaches we think of to allocate the horizontal 
    location of a node.  
    
    - Top down: we allocate horizontal space to a node.  Then its ``k`` 
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a 
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.  
      
    We use use both of these approaches simultaneously with ``leaf_vs_root_factor`` 
    determining how much of the horizontal space is based on the bottom up 
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.   
    
    
    :Arguments: 
    
    **G** the graph (must be a tree)

    **root** the root node of the tree 
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be 
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root
    
    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    '''
    # if not nx.is_tree(G):
    #     raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, leftmost, width, leafdx = 0.2, vert_gap = 0.2, vert_loc = 0, 
                    xcenter = 0.5, rootpos = None, 
                    leafpos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root:(xcenter,vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            rootdx = width/len(children)
            nextx = xcenter - width/2 - rootdx/2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(G,child, leftmost+leaf_count*leafdx, 
                                    width=rootdx, leafdx=leafdx,
                                    vert_gap = vert_gap, vert_loc = vert_loc-vert_gap, 
                                    xcenter=nextx, rootpos=rootpos, leafpos=leafpos, parent = root)
                leaf_count += newleaves

            leftmostchild = min((x for x,y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x,y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild+rightmostchild)/2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root]  = (leftmost, vert_loc)
#        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
#        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width/2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node)==0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node)==1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width, 
                                                    leafdx=width*1./leafcount, 
                                                    vert_gap=vert_gap, 
                                                    vert_loc = vert_loc, 
                                                    xcenter = xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (leaf_vs_root_factor*leafpos[node][0] + (1-leaf_vs_root_factor)*rootpos[node][0], leafpos[node][1]) 
#    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x,y in pos.values())
    for node in pos:
        pos[node]= (pos[node][0]*width/xmax, pos[node][1])
    return pos




# %%
# https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"

    # if (root is None) & isinstance(G, nx.DiGraph):
    #     root = next(iter(nx.topological_sort(G)))
    # elif not isinstance(G, nx.DiGraph):
    #     raise TypeError('graph must be a networkx digraph')

    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = list(G.neighbors(node))
        neighbors.sort()
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})




# %%
# x = nx.algorithms.centrality.trophic_levels(ontoG)


# False




nodes = [n for n in ontoG.nodes()]

x = [d for n, d in ontoG.in_degree()]
y = [d for n, d in ontoG.out_degree()]
z = np.array([x, y]).transpose()
w = np.sum(z, axis = 1)


print(nx.is_strongly_connected(ontoG))
print(nx.number_strongly_connected_components(ontoG))
x = [c for c in sorted(nx.strongly_connected_components(ontoG), key = len, reverse = True)]
y = [len(c) for c in x]

fig, ax = plt.subplots(1, 1, figsize = (6, 6))
ax.hist(y, range = (0, 20), bins = 20)
__ = plt.setp(ax, yscale = 'log')


print(nx.is_weakly_connected(ontoG))
print(nx.number_weakly_connected_components(ontoG))
x = [c for c in sorted(nx.weakly_connected_components(ontoG), key = len, reverse = True)]
y = [len(c) for c in x]

fig, ax = plt.subplots(1, 1, figsize = (6, 6))
ax.hist(y, range = (0, max(y)), bins = max(y))
__ = plt.setp(ax, xscale = 'log')




x = ontoJSON['nodes'][0]['id']


y = nx.algorithms.dag.topological_sort(ontoG)
z = list(reversed(list(y)))


# Leaf nodes?
z = [True if d == 0 else False for n, d in ontoG.in_degree()]

# Root node?
z = [True if d == 0 else False for n, d in ontoG.out_degree()]

z = [[n, d] for n, d in ontoG.in_degree()]



# %%


