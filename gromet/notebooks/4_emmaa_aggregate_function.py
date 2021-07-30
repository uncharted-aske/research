# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Run the Dario parser over the GroMEt of a model (MARM here)
# * Load the output graph object
# * Apply aggregation using the node agent-metadata

# %%

from typing import Dict, Tuple
import networkx as nx

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# %%[markdown]
# # Run the Dario parser over the GroMEt

deno_command = 'deno run --allow-write --allow-read'
parser_path = '/home/nliu/projects/aske/research/gromet/tools/parse.ts'
data_dir = '/home/nliu/projects/aske/research/gromet/data/august_2021_demo_repo/'
dist_dir = '/home/nliu/projects/aske/research/gromet/dist/august_2021_demo_repo/'

gromet_path = data_dir + 'emmaa_models/marm_model_gromet_2021-06-28-17-07-14.json'
graph_path = dist_dir + 'emmaa_models/marm_model_gromet_2021-06-28-17-07-14_graph.json'

__ = os.system(deno_command + ' ' + parser_path + ' ' + gromet_path + ' ' + graph_path)


with open(gromet_path, 'r') as f:
    gromet = json.load(f)


with open(graph_path, 'r') as f:
    graph = json.load(f)


deno_command = parser_path = data_dir = dist_dir = gromet_path = graph_path = f = None
del deno_command, parser_path, data_dir, dist_dir, gromet_path, graph_path, f


# %%
# Aggregate a given GroMEt graph that has EMMAA agent metadata for all nodes
def aggregate_emmaa_graph(graph: Dict) -> Tuple[Dict, Dict, Dict, Tuple[Dict, Dict]]:

    # Dict of all graph nodes by UID
    dict_nodes = {node['id']: node for node in graph['nodes']}

    # Dict of all rate junctions by UID
    dict_rates = {node['id']: node for node in graph['nodes'] if node['nodeType'] == 'Junction' if 'Rate' in node['nodeSubType']}
    dict_states = {node['id']: {agent['name']: agent for agent in node['metadata'][0][0]['indra_agent_references']} for node in graph['nodes'] if node['nodeType'] == 'Junction' if 'State' in node['nodeSubType']}

    # Dict of all INDRA agents referenced by the metadata of the state junctions
    # dict_agents = {agent: metadata for __, agents in dict_states.items() for agent, metadata in agents.items()}

    # Dict of all INDRA agents (and combinations thereof) referenced by the state junctions
    # == 'groups'
    dict_groups = {tuple(sorted(agents.keys())): [] for __, agents in dict_states.items()}
    for s, agents in dict_states.items():
        i = tuple(sorted(agents.keys()))
        dict_groups[i].append(s)

    # Mapping between a given state junction UID and the group tuple to which it belongs
    map_states_groups = {state: tuple(sorted(agents.keys())) for state, agents in dict_states.items()}
    
    # Build full graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(list(dict_states.keys()), bipartite = 0, type = 'State') # state-junction nodes
    G.add_nodes_from(list(dict_rates.keys()), bipartite = 1, type = 'Rate') # rate-junction nodes
    G.add_edges_from([(edge['source'], edge['target'], {'weight': 1.0}) for edge in graph['edges']]) # directed edges

    # Node list of aggregated graph == INDRA-agent groups
    list_nodes = list(dict_groups.keys())

    # Draw undirected edges (of aggregated graph) between groups only if there are connecting edges (of the full graph)
    dict_edges = {(src, tgt): [] for src in list_nodes for tgt in list_nodes if src != tgt}
    G_ = G.to_undirected()
    for s1 in dict_states.keys():
        for s2 in dict_states.keys():

            src = map_states_groups[s1]
            tgt = map_states_groups[s2]

            if src != tgt:
                p = nx.shortest_path(G_, source = s1, target = s2)

                if len(p) == 3:
                    dict_edges[(src, tgt)].append(None)

    # Remove unused keys in edge list
    x = [k for k, v in dict_edges.items() if len(v) == 0]
    __ = [dict_edges.pop(k, None) for k in x]

    # Build aggregated graph
    G_agg = nx.Graph()
    G_agg.add_nodes_from(list_nodes, bipartite = 0, type = 'State') # INDRA-agent nodes
    G_agg.add_edges_from([(edge[0], edge[1], {'weight': len(l)}) for edge, l in dict_edges.items()]) # directed edges between INDRA-agent nodes, weighted by the number of full-graph edges

    # Generate JSON of the aggregated graph
    # (same schema as the input graph)
    # group tuples -> string representation
    graph_agg = {
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
            'source': edge[0].__repr__(), 
            'target': edge[1].__repr__(), 
            'weight': len(l)
        } for edge, l in dict_edges.items()],
        'metadata': graph['metadata']
    }

    # Placeholder for parameter metadata
    for x in ('variables', 'initial_conditions', 'parameters'):
        graph_agg['metadata'][0][x] = None


    return (G, G_agg, graph_agg, (dict_rates, dict_groups))

# %%

G, G_agg, graph_agg, (dict_rates, dict_groups) = aggregate_emmaa_graph(graph = graph)


# %%
# Plot outputs

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 12))
titles = ("Full", "Collapsed")

for (x, g, t) in zip(fig.axes, (G, G_agg), titles):

    p = nx.kamada_kawai_layout(g, weight = 'weight')
    c = ['r' if g.nodes[n]['type'] == 'State' else 'b' for n in g.nodes]
    s = [50 if n in dict_rates.keys() else 100 * len(dict_groups[n]) if n in dict_groups.keys() else 100 for n in g.nodes]
    w = 0.5 * np.array(list(nx.get_edge_attributes(g, 'weight').values()))

    nx.draw(g, ax = x, pos = p, with_labels = False, node_color = c, node_size = s, width = w, alpha = 0.5, arrows = True)
    x.title.set_text(f"{t} ({len(g.nodes)} nodes, {len(g.edges)} edges)")


fig.savefig('../figures/marm_model_gromet_collapse_parsed.png', dpi = 150)

p = c = s = g = x = t = w = fig = ax = titles = None
del p, c, s, g, x, t, w, fig, ax, titles

# %%