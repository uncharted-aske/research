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
import copy
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# %%[markdown]
# # Run the Dario parser over the GroMEt

deno_command = 'deno run --allow-write --allow-read'
parser_path = '/home/nliu/projects/aske/research/gromet/tools/parse.ts'
data_dir = '/home/nliu/projects/aske/research/gromet/data/august_2021_demo_repo/emmaa_models/'
dist_dir = '/home/nliu/projects/aske/research/gromet/dist/august_2021_demo_repo/emmaa_models/'

files = os.listdir(data_dir)
gromet = []
graph = []
for i, file in enumerate(files):

    gromet_path = data_dir + file
    graph_path = dist_dir + file.split('.')[0] + '_graph.json'

    __ = os.system(deno_command + ' ' + parser_path + ' ' + gromet_path + ' ' + graph_path)

    with open(gromet_path, 'r') as f:
        gromet.append(json.load(f))

    with open(graph_path, 'r') as f:
        graph.append(json.load(f))


    print(f"{file}: {len(graph[i]['nodes'])} nodes, {len(graph[i]['edges'])} edges")


deno_command = parser_path = data_dir = gromet_path = graph_path = file = f = None
del deno_command, parser_path, data_dir, gromet_path, graph_path, file, f


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

                try:
                    p = nx.shortest_path(G_, source = s1, target = s2)
                except:
                    p = []

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
for file, g in zip(files, graph):

    try:

        if file != files[1]:

            G, G_agg, graph_agg, (dict_rates, dict_groups) = aggregate_emmaa_graph(graph = g)

            p = dist_dir + file.split('.')[0] + '_graph_agg.json'
            with open(p, 'w') as f:
                json.dump(graph_agg, f, indent = 2)

            # Plot outputs
            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 12))
            titles = ("Full", "Collapsed")

            for (x, gg, t) in zip(fig.axes, (G, G_agg), titles):

                p = nx.kamada_kawai_layout(gg, weight = 'weight')
                c = ['r' if gg.nodes[n]['type'] == 'State' else 'b' for n in gg.nodes]
                s = [50 if n in dict_rates.keys() else 100 * len(dict_groups[n]) if n in dict_groups.keys() else 100 for n in gg.nodes]
                w = 0.5 * np.array(list(nx.get_edge_attributes(gg, 'weight').values()))

                nx.draw(gg, ax = x, pos = p, with_labels = False, node_color = c, node_size = s, width = w, alpha = 0.5, arrows = True)
                x.title.set_text(f"{t} ({len(gg.nodes)} nodes, {len(gg.edges)} edges)")


            fig.savefig(f"../figures/{file.split('.')[0]}.png", dpi = 150)

    except:
        print(f"Error: {file}")


    file = g = f = p = p = c = s = gg = x = t = w = fig = ax = titles = None
    del file, g, f, p, c, s, gg, x, t, w, fig, ax, titles


# %%
# Similar to `aggregate_emmaa_graph`
# - Preserves original graph structure
# - INDRA-agent groups are just assigned to the node `parent` attribute and added as extra, disconnected nodes
def aggregate_emmaa_graph_reversible(graph: Dict) -> Dict:

    def tuple2str(t: Tuple) -> str:
        return ' + '.join(list(t))


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

    # Generate output
    graph_agg = copy.deepcopy(graph)
    for node in graph_agg['nodes']:
        if 'State' in node['nodeSubType']:
            node['parent'] = tuple2str(map_states_groups[node['id']])

    # Add the group nodes (with `parent` = '0')
    graph_agg['nodes'] += [{
            'id': tuple2str(group), 
            'concept': 'Relation', 
            'role': None,
            'label': tuple2str(group),
            'nodeType': 'Box',
            'dataType': None,
            'parent': '0',
            'nodeSubType': ['Relation'],
            'metadata': [dict_nodes[n]['metadata'][0] for n in l]
        } for group, l in dict_groups.items()]


    # Get the 1-hop neighbours of every rate-junction node
    dict_rate_neighbours = {rate_id: [] for rate_id in dict_rates}
    for edge in graph['edges']:
        if edge['source'] in dict_rates.keys():
            dict_rate_neighbours[edge['source']].append(edge['target'])
        
        if edge['target'] in dict_rates.keys():
            dict_rate_neighbours[edge['target']].append(edge['source'])


    # Assigned parenthood of rate-junction nodes
    # If all neighbours have the same parent, assign this parent to it, otherwise `parent` = '0'
    for node in graph_agg['nodes']:

        if 'Rate' in node['nodeSubType']:

            d = {tuple2str(map_states_groups[n]): None for n in dict_rate_neighbours[node['id']]}

            if len(d) == 1:
                node['parent'] = list(d.keys())[0]
            else:
                node['parent'] = '0'


    return graph_agg

# %%

for file, g in zip(files, graph):

    try:

        graph_agg_rev = aggregate_emmaa_graph_reversible(graph = g)

        p = dist_dir + file.split('.')[0] + '_graph_agg_rev.json'
        with open(p, 'w') as f:
            json.dump(graph_agg_rev, f, indent = 2)

    except:
        print(f"Error: {file}")


    files = g = f = p = None
    del files, g, f, p

# %%
# Similar to `aggregate_emmaa_graph_reversible`
# - Preserves original graph structure
# - INDRA-agent groups are just assigned to the node `parent` attribute and added as extra, disconnected nodes
# - Group rate-junction nodes by what INDRA-agent group they connect
def aggregate_emmaa_graph_reversible_rategroups(graph: Dict) -> Dict:

    def tuple2str(t: Tuple) -> str:
        return ' + '.join(list(t))


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

    # Generate output
    graph_agg = copy.deepcopy(graph)
    for node in graph_agg['nodes']:
        if 'State' in node['nodeSubType']:
            node['parent'] = tuple2str(map_states_groups[node['id']])

    # Add the group nodes (with `parent` = '0')
    graph_agg['nodes'] += [{
            'id': tuple2str(group), 
            'concept': 'Relation', 
            'role': None,
            'label': tuple2str(group),
            'nodeType': 'Box',
            'dataType': None,
            'parent': '0',
            'nodeSubType': ['Relation'],
            'metadata': [dict_nodes[n]['metadata'][0] for n in l]
        } for group, l in dict_groups.items()]


    # Get the 1-hop neighbours of every rate-junction node
    dict_rate_neighbours = {rate_id: [] for rate_id in dict_rates}
    for edge in graph['edges']:
        if edge['source'] in dict_rates.keys():
            dict_rate_neighbours[edge['source']].append(edge['target'])
        
        if edge['target'] in dict_rates.keys():
            dict_rate_neighbours[edge['target']].append(edge['source'])


    # Assigned parenthood of rate-junction nodes
    # If all neighbours have the same parent, assign this parent to it, otherwise `parent` = '0'
    for node in graph_agg['nodes']:

        if 'Rate' in node['nodeSubType']:

            d = {tuple2str(map_states_groups[n]): None for n in dict_rate_neighbours[node['id']]}

            if len(d) == 1:
                node['parent'] = list(d.keys())[0]
            else:
                node['parent'] = '0'


    # Get the group to which the 1-hop neighbours of a given rate-junction node belong
    dict_rate_neighbours_groups = {rate_id: set([map_states_groups[node] for node in nodes]) for rate_id, nodes in dict_rate_neighbours.items()}

    dict_rategroups = {tuple(rategroup_id): [] for __, rategroup_id in dict_rate_neighbours_groups.items()}
    for rate_id, rategroup_id in dict_rate_neighbours_groups.items():
        dict_rategroups[tuple(rategroup_id)].append(rate_id)

    # Add the rate-group nodes
    for rategroup, l in dict_rategroups.items():

        if len(rategroup) > 2:
            i = '0'

        else:
            group = rategroup[0]
            i = tuple2str(group)

        graph_agg['nodes'].append({
            'id': rategroup.__repr__(), 
            'concept': 'Relation', 
            'role': None,
            'label': rategroup.__repr__(), 
            'nodeType': 'Box',
            'dataType': None,
            'parent': i,
            'nodeSubType': ['Relation'],
            'metadata': None
        })

    
    # Assign parenthood of rate-junctions (rate-group nodes)
    for node in graph_agg['nodes']:
        if 'Rate' in node['nodeSubType']:
            node['parent'] = tuple(dict_rate_neighbours_groups[node['id']]).__repr__()



    print(f"Number of State Nodes: {len(dict_states)}")
    print(f"Number of Rate Nodes: {len(dict_rates)}")
    print(f"Number of State Groups: {len(dict_groups)}")
    print(f"Number of Rate Groups: {len(dict_rategroups)}")



    return graph_agg

# %%

for file, g in zip(files, graph):

    graph_agg_rev_rategroups = aggregate_emmaa_graph_reversible_rategroups(graph = g)

    p = dist_dir + file.split('.')[0] + '_agg_rev_rategroups.json'
    p = ''.join(p)
    with open(p, 'w') as f:
        json.dump(graph_agg_rev_rategroups, f, indent = 2)

    f = p = None
    del f, p

# %%
# Number of State Nodes: 74
# Number of Rate Nodes: 399
# Number of State Groups: 28
# Number of Rate Groups: 121

# Number of State Nodes: 849
# Number of Rate Nodes: 21592
# Number of State Groups: 167
# Number of Rate Groups: 177

# Number of State Nodes: 45
# Number of Rate Nodes: 44
# Number of State Groups: 31
# Number of Rate Groups: 43
