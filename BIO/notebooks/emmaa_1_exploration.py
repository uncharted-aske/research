# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

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

# %%
np.random.seed(0)

# %%[markdown]
# ## Load node and edge data from Dario

nodes = {}
with open('./data/covid19-snapshot_sep18-2020/processed/nodes.jsonl', 'r') as x:
    nodes = [json.loads(i) for i in x]

edges = {}
with open('./data/covid19-snapshot_sep18-2020/processed/edges.jsonl', 'r') as x:
    edges = [json.loads(i) for i in x]

edges_ = {}
with open('./data/covid19-snapshot_sep18-2020/processed/collapsedEdges.jsonl', 'r') as x:
    edges_ = [json.loads(i) for i in x]

x = None
del x

# %%[markdown]
# ## Summary of node-edge data
for x in [nodes, edges]:
    
    z = [list(set([y[i] for y in x])) if not isinstance(x[0][i], dict) else [] for i in x[0].keys()]
    
    print(f"Number of elements: {len(x)}")
    print(f"Attributes: {list(x[0].keys())}")
    print("Unique attribute values:")
    [print(f"  * {list(x[0].keys())[i]} ({len(y)}): {y}") if ((len(y) > 0) and (len(y) < 10)) else print(f"  * {list(x[0].keys())[i]} ({len(y)}): {y[:5]}...") for i, y in enumerate(z)]
    print("\n")

x = None
del x


# %%
# Number of elements: 37450
# Attributes: ['id', 'label', 'name', 'type', 'grounded', 'edges', 'loops', 'complex', 'conversion', 'info']
# Unique attribute values:
#   * id (37450): [0, 1, 2, 3, 4]...
#   * label (36933): ['Diplopia', 'histamine binding', 'BHV-1 UL41', 'salivation', 'synaptic membrane']...
#   * name (37450): ['amphotericin B methyl ester', 'Diplopia', 'histamine binding', 'BHV-1 UL41', 'synaptic membrane']...
#   * type (6): ['protein', 'chemical', 'tissue', 'bioprocess', 'general', 'not-grounded']
#   * grounded (2): [False, True]
#   * edges (493): [0, 1, 2, 3, 4]...
#   * loops (8): [0, 1, 2, 3, 4, 5, 6, 7]
#   * complex (149): [0, 1, 2, 3, 4]...
#   * conversion (13): [0, 1, 2, 3, 4]...
#   * info (0): []...


# Number of elements: 281075
# Attributes: ['id', 'collapsed_id', 'type', 'belief', 'statement_id']
# Unique attribute values:
#   * id (281075): [0, 1, 2, 3, 4]...
#   * collapsed_id (249472): [0, 1, 2, 3, 4]...
#   * type (24): ['IncreaseAmount', 'Activation', 'Inhibition', 'Palmitoylation', 'Glycosylation']...
#   * belief (614): [0.923, 0.9615, 1, 0.0, 0.9499999999999993]...
#   * statement_id (281075): ['165812769913812', '-12706043031082655', '4902333648079931', '-33691498061767493', '-34536499731221590']...


# %%[markdown]
# ## Agent Ontologies

z = sorted(list(set([link[0] if len(link) > 0 else '' for node in nodes for link in node['info']['links']])))
k = np.array([np.sum([1 if link[0] == i else 0 for node in nodes for link in node['info']['links']]) for i in z])

i = np.argsort(k)[::-1]
z = [z[j] for j in i]
k = k[i]

__ = [print(f"  * {i} ({j}, {j / len(nodes) * 100:3.1f}%)") for i, j in zip(z, k)]

# %%
#   * UP (13524, 36.1%)
#   * MESH (11125, 29.7%)
#   * HGNC (9148, 24.4%)
#   * CHEBI (6220, 16.6%)
#   * PUBCHEM (4656, 12.4%)
#   * CAS (4022, 10.7%)
#   * CHEMBL (3470, 9.3%)
#   * GO (2213, 5.9%)
#   * DRUGBANK (2143, 5.7%)
#   * HMDB (1881, 5.0%)
#   * PF (1138, 3.0%)
#   * EFO (823, 2.2%)
#   * HP (680, 1.8%)
#   * EGID (485, 1.3%)
#   * DOID (472, 1.3%)
#   * FPLX (430, 1.1%)
#   * IP (322, 0.9%)
#   * NCIT (250, 0.7%)
#   * HMS-LINCS (206, 0.6%)
#   * LINCS (138, 0.4%)
#   * MIRBASE (96, 0.3%)
#   * NXPFA (68, 0.2%)
#   * ECCODE (39, 0.1%)
#   * HGNC_GROUP (38, 0.1%)
#   * REFSEQ_PROT (24, 0.1%)
#   * UPPRO (20, 0.1%)
#   * TAXONOMY (13, 0.0%)
#   * GENBANK (9, 0.0%)
#   * PR (3, 0.0%)
#   * BTO (3, 0.0%)
#   * LNCRNADB (1, 0.0%)
#   * NONCODE (1, 0.0%)
#   * CVCL (1, 0.0%)
#   * CO (1, 0.0%)
#   * RGD (1, 0.0%)

# %%[markdown]
# Note from Ben Gyori: 
# The Indra BioOntology has an explicit priority order amongst the namespaces. 
# This order is currently `[‘FPLX’, ‘UPPRO’, ‘HGNC’, ‘UP’, ‘CHEBI’, ‘GO’, ‘MESH’, ‘MIRBASE’, ‘DOID’, ‘HP’, ‘EFO’]`.

# %%
# Filter by ontology

i = 'TAXONOMY'
z = np.flatnonzero(np.asarray([np.sum([1 if link[0] == i else 0  for link in node['info']['links']]) for node in nodes]))


# %%[markdown]
# ## Node Degree Distribution

# %%
# Collate CollapseEdges with edges data
for edge in edges:
    i = edge['collapsed_id']
    edge['source'] = edges_[i]['source']
    edge['target'] = edges_[i]['target']

# Count node degree
z = np.array([[edge['source'], edge['target']] for edge in edges])
nodeDegreeCounts = np.array([[np.sum(z[:, j] == i) for j in range(2)] for i in range(len(nodes))])

print(f"\nTop 10 by out-degree...\nIndex: Name (In-degree, Out-degree)")
for i in np.argsort(nodeDegreeCounts[:, 0])[::-1][:10]:
    print(f"{i}: {nodes[i]['name']} ({nodeDegreeCounts[i, 0]}, {nodeDegreeCounts[i, 1]})")

print(f"\nTop 10 by in-degree...\nIndex: Name (In-degree, Out-degree)")
for i in np.argsort(nodeDegreeCounts[:, 1])[::-1][:10]:
    print(f"{i}: {nodes[i]['name']} ({nodeDegreeCounts[i, 0]}, {nodeDegreeCounts[i, 1]})")

print(f"\nTop 10 by total degree...\nIndex: Name (In-degree, Out-degree, Total Degree)")
for i in np.argsort(np.sum(nodeDegreeCounts, axis = 1))[::-1][:10]:
    print(f"{i}: {nodes[i]['name']} ({nodeDegreeCounts[i, 0]}, {nodeDegreeCounts[i, 1]}, {np.sum(nodeDegreeCounts[i, :])})")

# %%
# Top 10 by out-degree...
# Index: Name (In-degree, Out-degree)
# 103: cyclosporin A (2135, 55)
# 236: Infections (1915, 1774)
# 8508: cycloheximide (1326, 6)
# 243: Casp14 (1308, 1460)
# 45: lipopolysaccharide (1149, 373)
# 475: Interferon (1143, 1520)
# 118: SGCG (1030, 1055)
# 37: TNF (997, 1644)
# 7225: chloroquine (981, 41)
# 444: Disease (953, 2112)

# Top 10 by in-degree...
# Index: Name (In-degree, Out-degree)
# 377: apoptotic process (173, 2605)
# 411: inflammatory response (392, 2484)
# 444: Disease (953, 2112)
# 236: Infections (1915, 1774)
# 412: cell population proliferation (61, 1698)
# 37: TNF (997, 1644)
# 42: IL6 (773, 1550)
# 475: Interferon (1143, 1520)
# 243: Casp14 (1308, 1460)
# 233: Death (224, 1402)

# Top 10 by total degree...
# Index: Name (In-degree, Out-degree, Total Degree)
# 236: Infections (1915, 1774, 3689)
# 444: Disease (953, 2112, 3065)
# 411: inflammatory response (392, 2484, 2876)
# 377: apoptotic process (173, 2605, 2778)
# 243: Casp14 (1308, 1460, 2768)
# 475: Interferon (1143, 1520, 2663)
# 37: TNF (997, 1644, 2641)
# 42: IL6 (773, 1550, 2323)
# 103: cyclosporin A (2135, 55, 2190)
# 118: SGCG (1030, 1055, 2085)

# %%[markdown]
# ## Node degree distribution by node

k = np.max(nodeDegreeCounts)
z = 10 ** np.linspace(0, np.log10(k), 100)
H, __, __ = np.histogram2d(nodeDegreeCounts[:, 0], nodeDegreeCounts[:, 1], bins = z, density = True)


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
fig.suptitle('Node Degree Distribution by Node')

__ = ax[0].pcolormesh(z[:-1], z[:-1], H, cmap = 'PuBu', norm = mpl.colors.LogNorm(vmin = np.min(H.flat[np.flatnonzero(H > 0)]), vmax = np.max(H)), shading = 'auto')
__ = plt.setp(ax[0], xlabel = 'Out Degree', ylabel = 'In Degree', xlim = (1, z[-1]), ylim = (1, z[-1]), xscale = 'log', yscale = 'log', aspect = 1.0)


for i, x in enumerate(['Out', 'In']):
    H, __ = np.histogram(nodeDegreeCounts[:, i], bins = z, density = True)
    ax[1].plot(z[:-1], H, label = x)

__ = plt.setp(ax[1], xlim = (1, z[-1]), xlabel = 'Node Degree', xscale = 'log', ylabel = 'PDF (Fraction of Nodes per Node Degree)', yscale = 'log')
__ = ax[1].legend()

fig.savefig('./figures/nodeDegreeDistByNode.png', dpi = 150)

i = j = k = x = y = z = H = fig = ax = None
del i, j, k, x, y, z, H, fig, ax

# %%[markdown]
# ## Node degree distribution by edge

k = np.max(nodeDegreeCounts)
z = 10 ** np.linspace(0, np.log10(k), 100)
x = np.array([[nodes[edge[y]]['edges'] for edge in edges] for y in ['source', 'target']])
H, __, __ = np.histogram2d(x[0, :], x[1, :], bins = z, density = True)


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
fig.suptitle('Node Degree Distribution by Edge')

__ = ax[0].pcolormesh(z[:-1], z[:-1], H, cmap = 'PuBu', norm = mpl.colors.LogNorm(vmin = np.min(H.flat[np.flatnonzero(H > 0)]), vmax = np.max(H)), shading = 'auto')
__ = plt.setp(ax[0], xlabel = 'Source Degree', ylabel = 'Target Degree', xlim = (1, z[-1]), ylim = (1, z[-1]), xscale = 'log', yscale = 'log', aspect = 1.0)


for x in ['Source', 'Target']:
    y = np.array([nodes[edge[x[0].lower() + x[1:]]]['edges'] for edge in edges])
    H, __ = np.histogram(y, bins = z, density = True)
    ax[1].plot(z[:-1], H, label = x)

__ = plt.setp(ax[1], xlim = (1, z[-1]), xlabel = 'Node Degree', xscale = 'log', ylabel = 'PDF (Fraction of Edges per Node Degree)', yscale = 'log')
__ = ax[1].legend()

fig.savefig('./figures/nodeDegreeDistByEdge.png', dpi = 150)

i = j = k = x = y = z = H = fig = ax = None
del i, j, k, x, y, z, H, fig, ax

# %%[markdown]
# ## Sample Node-Link Diagram 

i = np.random.randint(0, len(edges), size = (500, ))
z = [[edges[j]['source'], edges[j]['target']] for j in i]

G = nx.DiGraph(z)
x = nx.spring_layout(G)
# y = dict(zip(range(len(listVerticesAll)), iter(listVerticesAll)))

fig, ax = plt.subplots(figsize = (12, 12), nrows = 1, ncols = 1)
__ = plt.setp(ax, title = "Node-Link Diagram of a Random Subgraph")
nx.draw_networkx(G, pos = x, with_labels = False, node_size = 5, width = 0.1, arrows = False, arrowsize = 1, ax = ax)

