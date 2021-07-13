# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Parse the GroMEt of the MARM model using Dario's parser
# * Apply aggregation as before to this object

# %%
import os
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# %%[markdown]
# # Run Parser on MARM-Model GroMEt

deno_command = 'deno run --allow-write --allow-read'
parser_path = '/home/nliu/projects/aske/research/gromet/tools/parse.ts'
data_dir = '/home/nliu/projects/aske/research/gromet/data/august_2021_demo_repo'
dist_dir = '/home/nliu/projects/aske/research/gromet/dist/august_2021_demo_repo'

gromet_path = data_dir + '/' + 'emmaa_models/marm_model_gromet_2021-06-28-17-07-14.json'
graph_path = dist_dir + '/' + 'emmaa_models/marm_model_gromet_2021-06-28-17-07-14_graph.json'

__ = os.system(deno_command + ' ' + parser_path + ' ' + gromet_path + ' ' + graph_path)


# %%
# ## Load the Parsed GroMEt Graph

with open(graph_path, 'r') as f:
    graph = json.load(f)

f = None
del f

# %%
print(f"{'Name:':<25} {graph['nodes'][0]['label']:<}")
print(f"{'Type:':<25} {graph['nodes'][0]['dataType']:<}")

print(f"Number of ")
print(f"   {'edges:':<25} {len(graph['edges']):>5d}")
print(f"   {'nodes:':<25} {len(graph['nodes']):>5d}\n")

print(f"      {'variables:':<25} {len(graph['metadata'][0]['variables']):>5d}")
print(f"      {'parameters:':<25} {len(graph['metadata'][0]['parameters']):>5d}")
print(f"      {'initial conditions:':<25} {len(graph['metadata'][0]['initial_conditions']):>5d}\n")

print(f"      {'junctions:':<25} {len([None for node in graph['nodes'] if node['nodeType'] == 'Junction']):>5d}")
__ = [print(f"         {x.lower() + ':':<22} {len([None for node in graph['nodes'] if (node['nodeType'] == 'Junction') & (x in node['nodeSubType'])]):>5d}") for x in ('State', 'Rate', 'FluxState', 'Tangent')]


# Name:                     marm_model
# Type:                     PetriNetClassic
# Number of 
#    edges:                     1207
#    nodes:                      474
#
#       variables:                  473
#       parameters:                 399
#       initial conditions:          74
#
#       junctions:                  473
#          state:                    74
#          rate:                    399
#          fluxstate:                 0
#          tangent:                   0


# %%[markdown]
# List of all rate junctions
dict_rates = {node['id']: node for node in graph['nodes'] if node['nodeType'] == 'Junction' if 'Rate' in node['nodeSubType']}


# List of all state junctions
dict_states = {node['id']: {agent['name']: agent for agent in node['metadata'][0]['indra_agent_references']} for node in graph['nodes'] if node['nodeType'] == 'Junction' if 'State' in node['nodeSubType']}


# List of all INDRA agents referenced by the state junctions
dict_agents = {agent: metadata for __, agents in dict_states.items() for agent, metadata in agents.items()}


# List of all INDRA agents (and combinations thereof) referenced by the state junctions
dict_groups = {tuple(sorted(agents.keys())): [] for __, agents in dict_states.items()}
for s, agents in dict_states.items():
    i = tuple(sorted(agents.keys()))
    dict_groups[i].append(s)


# Mapping between a state junction and the group to which it belongs
map_states_groups = {s: tuple(sorted(agents.keys())) for s, agents in dict_states.items()}


i = s = agents = None
del i, s, agents

# %%
print(f"Number of ")
print(f"   {'INRDA agents:':<25} {len(dict_agents.keys()):>5d}")
print(f"   {'groups:':<25} {len(dict_groups.keys()):>5d}")

# Number of 
#    INDRA agents:                12
#    groups:                      28

# %%[markdown]
# # Generate NetworkX Object

# %%[markdown]
# ## Full Graph

# %%
G = nx.MultiDiGraph()

# State junctions
G.add_nodes_from(list(dict_states.keys()), bipartite = 0, type = 'State')

# Rate junctions
G.add_nodes_from(list(dict_rates.keys()), bipartite = 1, type = 'Rate')

# Graph edges as directed edges
__ = [G.add_edge(edge['source'], edge['target'], weight = 1.0) for edge in graph['edges']]


# %%
# ## Collapsed Graph
# 
#  Collapse all state-junction nodes to their INDRA-agent groups and hide all rate nodes

# %%
# Aggregated nodes = INDRA-agent groups
list_nodes = list(dict_groups.keys())


# Draw undirected edges between groups only if there are connecting GroMEt edges
dict_edges = {(src, tgt): [] for src in list_nodes for tgt in list_nodes if src != tgt}
g = G.to_undirected()
for s1 in dict_states:
    for s2 in dict_states:

        src = map_states_groups[s1]
        tgt = map_states_groups[s2]

        if src != tgt:
            p = nx.shortest_path(g, source = s1, target = s2)

            if len(p) == 3:
                dict_edges[(src, tgt)].append(None)


# Remove empty edges
x = [k for k, v in dict_edges.items() if len(v) == 0]
__ = [dict_edges.pop(k, None) for k in x]


s1 = s2 = src = tgt = x = g = p = None
del s1, s2, src, tgt, x, g, p


# %%
G_ = nx.Graph()

# State nodes
G_.add_nodes_from(list_nodes, bipartite = 0, type = 'State')


# Edges
__ = [G_.add_edge(edge[0], edge[1], weight = len(l)) for edge, l in dict_edges.items()]

# %%[markdown]
# ## Draw Graphs

# %%

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 12))

titles = ("Full", "Collapsed")
for (x, g, t) in zip(fig.axes, (G, G_), titles):

    p = nx.kamada_kawai_layout(g, weight = 'weight')
    c = ['r' if g.nodes[n]['type'] == 'State' else 'b' for n in g.nodes]
    s = [50 if n in dict_rates else 100 * len(dict_groups[n]) if n in dict_groups else 100 for n in g.nodes]
    w = 0.5 * np.array(list(nx.get_edge_attributes(g, 'weight').values()))

    nx.draw(g, ax = x, pos = p, with_labels = False, node_color = c, node_size = s, width = w, alpha = 0.5, arrows = True)
    x.title.set_text(f"{t} ({len(g.nodes)} nodes, {len(g.edges)} edges)")


fig.savefig('../figures/marm_model_gromet_collapse_parsed.png', dpi = 150)


p = c = s = g = x = t = w = fig = ax = titles = None
del p, c, s, g, x, t, w, fig, ax, titles

# %%[markdown]
# ## Generate Collapsed Graph Object

# %%

dict_nodes = {node['id']: node for node in graph['nodes']}

graph_ = {
    'nodes': [graph['nodes'][0]] + [{
        'id': group.__repr__(), 
        'concept': 'Junction', 
        'role': None,
        'label': group.__repr__(),
        'nodeType': 'Junction',
        'dataType': None,
        'parent': '0',
        'nodeSubType': ['State'],
        'metadata': [dict_nodes[n]['metadata'][0] for n in l]
    } for group, l in dict_groups.items()],
    'edges': [{
        'source': edge[0], 
        'target': edge[1], 
        'weight': len(l)
    } for edge, l in dict_edges.items()],
    'metadata': graph['metadata']
}


for x in ('variables', 'initial_conditions', 'parameters'):
    graph_['metadata'][0][x] = None


# %%
# ## Save Collapsed Graph Object

p = graph_path.split('.')
p.insert(1, '_collapsed.')
p = ''.join(p)
with open(p, 'w') as f:
    json.dump(graph_, f, indent = 2)

f = p = None
del f, p


# %%
