# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Parse directly the "loop" and CHIME GroMEt examples from UAZ/Clay
# * 

# %%
import os
import json
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, Optional

# %%[markdown]
# # Run the Dario parser over the GroMEt

deno_command = 'deno run --allow-write --allow-read'
parser_path = '/home/nliu/projects/aske/research/gromet/tools/parse.ts'
data_dir = '/home/nliu/projects/aske/research/gromet/data/uaz/'
dist_dir = '/home/nliu/projects/aske/research/gromet/dist/uaz/'

gromet = []
graph = []
for p in (
    'loop/loop_ex2_gromet_FunctionNetwork_correction', 
    'conditional/cond_ex1_gromet_FunctionNetwork', 
    'CHIME/CHIME_SIR_Base_intermediate_versions/CHIME_SIR_v01_gromet_FunctionNetwork_by_hand', 
    'CHIME/CHIME_SIR_Base_variables_gromet_FunctionNetwork',
    'CHIME/CHIME_SIR_Base_variables_gromet_FunctionNetwork-with-metadata-CTM', 
    'CHIME/CHIME_SVIIvR_variables_gromet_FunctionNetwork-with-metadata-CTM'):

    gromet_path = data_dir + f'{p}.json'
    graph_path = dist_dir + f'{p}_graph.json'

    __ = os.system(deno_command + ' ' + parser_path + ' ' + gromet_path + ' ' + graph_path)

    try:
        with open(gromet_path, 'r') as f:
            gromet.append(json.load(f))
    except:
        pass

    try:
        with open(graph_path, 'r') as f:
            graph.append(json.load(f))
    except:
        pass


deno_command = parser_path = data_dir = dist_dir = gromet_path = f = p = None
del deno_command, parser_path, data_dir, dist_dir, gromet_path, f, p


# %%
# Calculate hierarchy level of a node from its ID (e.g. "0::12:11" = 0 -> 12 -> 11)
def calculate_node_level(node_id: str) -> int:

    node_level = len(node_id.split('::')) - 1

    return node_level

# Get children of all nodes in graph
def get_node_children(graph: Dict) -> Dict:

    node_children = {node['id']: [] for node in graph['nodes']}
    for node in graph['nodes']:
        if node['parent'] != None:
            node_children[node['parent']].append(node['id'])

    return node_children

# Get children of all nodes in NX graph object
def get_node_children_nx(G: Dict) -> Dict:

    node_children = {node['id']: [] for node in G.nodes}
    for node in G.nodes:
        if G.nodes[node]['parent'] != None:
            node_children[G.nodes[node]['parent']].append(node)

    return node_children

# Get all descendants of every node in NX graph object
def get_node_descendants_nx(G: Dict) -> Dict:

    node_descendants = {node: [] for node in G.nodes}
    for node in G.nodes:
        k = node.split('::')[:-1]
        for i in range(1, len(k) + 1):
            node_parent = '::'.join(k[0:i])
            if node_parent in node_descendants.keys():
                node_descendants[node_parent].append(node)

    return node_descendants

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

# Add equi-level edges
# If an edge exists between a level-k node and a level-(k + 1) one, 
# add an edge between the level-k node and the level-k parent of the other
def add_missing_edges(G: Any) -> Any:

    edges = []
    for edge in G.edges:

        m = calculate_node_level(edge[0])
        n = calculate_node_level(edge[1])
        
        if m != n:
            
            i = np.argmin([m, n])
            if i == 0:
                src = G.nodes[edge[0]]['id']
                tgt = G.nodes[edge[1]]['parent']
            else:
                src = G.nodes[edge[0]]['parent']
                tgt = G.nodes[edge[1]]['id']

            if src != tgt:
                edges.append((src, tgt, 0))


    G.add_edges_from(edges, weight = 0)

    return None

# Promote edges between nodes of different parent nodes to edges between the parent nodes
# e.g. "0::1::2"->"0::2::3" becomes "0::1"->"0::2"
# "0::3"->"0:28:27" becomes "0::3"->"0:28"
def promote_edges(G: Any) -> Any:

    edges_remove = []
    edges_add = []

    for edge in G.edges:

        src_parent = G.nodes[edge[0]]['parent']
        tgt_parent = G.nodes[edge[1]]['parent']

        if src_parent != tgt_parent:

            m = calculate_node_level(edge[0])
            n = calculate_node_level(edge[1])
            
            if m != n:
                k = min([m, n]) + 1
                src = '::'.join(edge[0].split('::')[:k])
                tgt = '::'.join(edge[1].split('::')[:k])
                edges_add.append((src, tgt))

            else:
                edges_add.append((src_parent, tgt_parent))

            edges_remove.append(edge)


    G.remove_edges_from(edges_remove)
    G.add_edges_from(edges_add, weight = -1)

    return None

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

G = generate_nx_obj(graph = graph[0])

__ = add_missing_edges(G = G)
__ = promote_edges(G = G)
# __ = generate_linear_layout(G, draw = True)

__ = generate_linear_layout_with_hierarchy(G, draw = True)

# %%

G = generate_nx_obj(graph = graph[1])

__ = add_missing_edges(G = G)
__ = promote_edges(G = G)

__ = generate_linear_layout_with_hierarchy(G, draw = True)


# %%
G = generate_nx_obj(graph = graph[2])

__ = add_missing_edges(G = G)
__ = promote_edges(G = G)

__ = generate_linear_layout_with_hierarchy(G, draw = True)

# %%
G = generate_nx_obj(graph = graph[3])

__ = add_missing_edges(G = G)
__ = promote_edges(G = G)

__ = generate_linear_layout_with_hierarchy(G, draw = True)


# %%
