# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Parse directly the "loop" and CHIME GroMEt examples from UAZ/Clay
# * 

# %%
import json
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, Optional

# %%
# # Load the Loop GroMEt

uaz_path = "/home/nliu/projects/aske/research/gromet/data/uaz/"

with open(uaz_path + 'loop/loop_ex2_gromet_FunctionNetwork.json', 'r') as f:
    gromet_loop = json.load(f)

with open(uaz_path + 'conditional/cond_ex1_gromet_FunctionNetwork.json', 'r') as f:
    gromet_cond = json.load(f)

f = None
del f

# %%
# Box-type nodes
nodes = {node['uid']: {
    'id': node['uid'],
    'concept': node['syntax'],
    'role': [],
    'label': node['uid'],
    'nodeType': 'Box',
    'dataType': None if 'value_type' not in node.keys() else node['value_type'],
    'parent': None,
    # 'children': [],
    'nodeSubType': [node['type']],
    'metadata': node['metadata']
} for node in gromet_loop['boxes']}

# Other node types
for k in ('ports', 'junctions'):

    for node in gromet_loop[k]:

        nodes[node['uid']] = {
            'id': node['uid'],
            'concept': node['syntax'],
            'role': [],
            'label': node['uid'],
            'nodeType': node['syntax'],
            'dataType': None if 'value_type' not in node.keys() else node['value_type'],
            'parent': None,
            # 'children': [],
            'nodeSubType': [node['type']],
            'metadata': node['metadata']
        }

# %%
# Parenthood
for node in gromet_loop['boxes']:
    for k in ('ports', 'junctions', 'exit_condition'):
        if k in node.keys():
            if node[k] != None:
                if isinstance(node[k], list):
                    for m in node[k]:
                        nodes[m]['parent'] = node['uid']
                else:
                    nodes[m]['parent'] = node['uid']

# Role
for k in ('variables', 'parameters', 'initial_conditions'):
    for m in gromet_loop['metadata'][0][k]:
        nodes[m]['role'] += k


# %%
# Edges
edges = [{
    'source': wire['src'],
    'target': wire['tgt']
} for wire in gromet_loop['wires']]


# edges += {'source': 'P:loop_1_in.i', 'target': 'B:loop_1'}

# %%


# "W:loop_ex2.loop_1.e"
# "W:loop_1.loop_1_cond.e"
# "wires": [
#         "W:loop_ex2.loop_1.e",
#         "W:loop_ex2.loop_1.k",
#         "W:loop_1.loop_ex2.k"
#       ],


# gromet_loop['wires'][0]['tgt'] = 'P:loop_1.in.e'
# gromet_loop['wires'][1]['src'] = 'P:loop_1.in.e'

# %%
# Parsed graph
graph_loop = {'nodes': [node for __, node in nodes.items()], 'edges': edges, 'metadata': gromet_loop['metadata']}


# %%
# Generate NX Object from a parsed GroMEt
def generate_nx_obj(graph: Dict) -> Any:

    # Missing graph metadata (lost during Dario parsing)
    G = nx.MultiDiGraph(
        uid = None, 
        type = None, 
        name = None,
        metadata = graph['metadata']
    )

    # Add nodes
    G.add_nodes_from([(node['id'], {**node}) for node in graph['nodes']])

    # Add edges
    G.add_edges_from([(edge['source'], edge['target'], {'weight': 1}) for edge in graph['edges']])

    return G

# Get all descendants of every node in NX graph object
def get_node_descendants_nx(G: Dict) -> Dict:

    node_descendants = {node: [] for node in G.nodes}
    for node in G.nodes:
        k = node.split('::')[:-1]
        for i in range(1, len(k) + 1):
            node_parent = '::'.join(k[0:i])
            node_descendants[node_parent].append(node)

    return node_descendants

# %%
# Generate linear layout from a NetworkX graph using condensation & topological sorting
def generate_linear_layout(G: Any, offset: Optional[Dict] = None, draw: Optional[bool] = False, ax: Optional[Any] = None) -> Any:

    # Condense graph to a DAG and topologically sort it
    G_cond = nx.condensation(G)
    S = nx.topological_sort(G_cond)

    # Generate x-coordinates by offset or at least 1
    k = 0
    pos = {}
    for i in S:
        for node_id in G_cond.nodes[i]['members']:
            pos[node_id] = np.array([k, 0])

            if offset != None:
                k += max([offset[node_id], 1])
            else:
                k += 1
    if draw == True:

        if ax == None:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20, 5))

        __ = draw_graph(G, pos = pos, ax = ax, G_full = G)

    return pos

# %%
# Generate linear layout that is layered by the parent-child hierarchy of the graph
def generate_linear_layout_with_hierarchy(G: Any, draw: Optional[bool] = False, ax: Optional[Any] = None) -> Dict:

    if (draw == True) & (ax == None):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20, 5))

    pos_full = {None: np.array([0, -1])}
    max_node_level = max([calculate_node_level(node) for node in G.nodes])

    node_descendants = get_node_descendants_nx(G)
    num_descendants = {node: len(l) for node, l in node_descendants.items()}


    for l in range(max_node_level + 1):

        nodes = [node for node in G.nodes if calculate_node_level(node) == l]
        parents = set([G.nodes[node]['parent'] for node in nodes])

        for parent in parents:

            nodes_ = [node for node in G.nodes if G.nodes[node]['parent'] == parent]
            G_sub = G.subgraph(nodes_)
            pos = generate_linear_layout(G = G_sub, offset = num_descendants)
            pos = {k: v + pos_full[parent] + np.array([0, 1]) for k, v in pos.items()}
            pos_full = {**pos_full, **pos}

            if draw == True:
                __ = draw_graph(G_sub, pos = pos, ax = ax, G_full = G)

    if draw == True:
        __ = plt.setp(ax, ylim = (-1, max_node_level + 1))
    
    return pos_full



# %%
# Draw NX Object
def draw_graph(G: Any, pos: Dict, ax: Optional[Any] = None, node_args: Optional[Dict] = None, edge_args: Optional[Dict] = None, label_args: Optional[Dict] = None, legend_args: Optional[Dict] = None, G_full: Optional[Any] = None, label_key: Optional[str] = 'label') -> Any:

    if ax == None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20, 5))

    if node_args == None:

        if G_full == None:
            node_types = {t: i for i, t in enumerate(np.unique([G.nodes[node]['nodeType'] for node in G.nodes]))}
        else:
            node_types = {t: i for i, t in enumerate(np.unique([G_full.nodes[node]['nodeType'] for node in G_full.nodes]))}
        
        node_args = {
            # 'node_size': [len(get_node_children_nx(G, node_id = node)) for node in G.nodes],
            'node_size': [100 for node in G.nodes],
            'node_color': [node_types[G.nodes[node]['nodeType']] for node in G.nodes],
            'cmap': 'tab10',
            'vmin': 0,
            'vmax': 9,
        }
        node_norm = mpl.colors.Normalize(vmin = 0, vmax = 9)
        node_cmap = mpl.cm.get_cmap(node_args['cmap'])

    if edge_args == None:
        edge_args = {
            'edge_color': 'tab:gray',
            'alpha': 0.5,
            'connectionstyle': 'arc3,rad=0.15',
        }

    if label_args == None:
        label_args = {
            'font_color': 'black',
            'font_size': 10,
            'horizontalalignment': 'left', 
            'verticalalignment': 'bottom',
            'labels': {node: G.nodes[node][label_key] for node in G.nodes},
        }

    if legend_args == None:

        if len(node_args) > 0:

            legend_elements = [
                mpl.lines.Line2D(
                    [0], [0], marker = 'o', color = 'w', label = t, markersize = 20,
                    markerfacecolor = node_cmap(node_norm(val))
                ) for t, val in node_types.items()
            ]

            legend_args = {
                'loc': 'lower right',
                'ncol': len(legend_elements)
            }

    if len(node_args) > 0:
        h_nodes = nx.draw_networkx_nodes(G, pos = pos, ax = ax, **node_args)

    if len(edge_args) > 0:
        h_edges = nx.draw_networkx_edges(G, pos = pos, ax = ax, **edge_args)

    if len(label_args) > 0:
        h_labels = nx.draw_networkx_labels(G, pos = pos, ax = ax, **label_args)
        __ = [t.set_rotation(15) for __, t in h_labels.items()]

    __ = plt.setp(ax, ylim = (-3, 3))

    if len(legend_args) > 0:
        # ax.legend(handles = legend_elements, loc = 'lower right', ncol = len(legend_elements))
        ax.legend(handles = legend_elements, **legend_args)


    return None

# %%


G = generate_nx_obj(graph = graph_loop)
# __ = add_missing_edges(G = G)
# __ = promote_edges(G = G)
__ = generate_linear_layout(G, draw = True)
# __ = generate_linear_layout_with_hierarchy(G, draw = True)



# %%
G_loop = generate_nx_obj(gromet = gromet_loop)

G_map, fig = compare_graphs(G1 = G_loop, G2 = G_loop, map = None, leg_id = None, rename = ('Loop1 ', 'Loop2 '), plot = True, plot_layout = 'linear')
# fig.savefig(f'../figures/map_sir_chime+_{i}.png', dpi = 150)
