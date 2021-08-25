# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Load SimpleSIR and (mini) CHIME 
# * 

# %%
import json
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

# %%[markdown]
# # Load Model Data

# %%

# path = "/home/nliu/projects/aske/research/gromet/data/august_2021_demo_repo/Simple_SIR/SimpleSIR_metadata_gromet_FunctionNetwork.json"
# with open(path, 'r') as f:
#     gromet_sir = json.load(f)

# path = "/home/nliu/projects/aske/research/gromet/dist/august_2021_demo_repo/Simple_SIR/SimpleSIR_metadata_gromet_FunctionNetwork_graph.json"
# with open(path, 'r') as f:
#     graph_sir = json.load(f)

# path = "/home/nliu/projects/aske/research/gromet/data/uaz/CHIME/CHIME_SIR_v01_gromet_FunctionNetwork_by_hand.json"
# with open(path, 'r') as f:
#     gromet_chime = json.load(f)

# path = "/home/nliu/projects/aske/research/gromet/dist/uaz/CHIME/CHIME_SIR_v01_gromet_FunctionNetwork_by_hand_graph.json"
# with open(path, 'r') as f:
#     graph_chime = json.load(f)

# path = "/home/nliu/projects/aske/research/gromet/data/uaz/gromet_intersection_graph/gig__SimpleSIR_metadata-CHIME_SIR_v01.json"
# with open(path, 'r') as f:
#     comparison_sir_chime = json.load(f)


# f = None
# del f

# %%

# path = "/home/nliu/projects/aske/research/gromet/data/uaz/CHIME/CHIME_SIR_Base_variables_gromet_FunctionNetwork-with-metadata-CTM.json"
# with open(path, 'r') as f:
#     gromet_sir = json.load(f)

# path = "/home/nliu/projects/aske/research/gromet/dist/uaz/CHIME/CHIME_SIR_Base_variables_gromet_FunctionNetwork-with-metadata-CTM_graph.json"
# with open(path, 'r') as f:
#     graph_sir = json.load(f)

# path = "/home/nliu/projects/aske/research/gromet/data/uaz/CHIME/CHIME_SVIIvR_variables_gromet_FunctionNetwork-with-metadata-CTM.json"
# with open(path, 'r') as f:
#     gromet_chime = json.load(f)

# path = "/home/nliu/projects/aske/research/gromet/dist/uaz/CHIME/CHIME_SVIIvR_variables_gromet_FunctionNetwork-with-metadata-CTM_graph.json"
# with open(path, 'r') as f:
#     graph_chime = json.load(f)

# path = "/home/nliu/projects/aske/research/gromet/data/uaz/gromet_intersection_graph/gig__CHIME_SIR_Base_v01-CHIME_SVIIvR_v01.json"
# with open(path, 'r') as f:
#     comparison_sir_chime = json.load(f)


# f = None
# del f

# %%




path = "/home/nliu/projects/aske/research/gromet/data/uaz/CHIME/CHIME_SIR_Dyn_gromet_FunctionNetwork-with-vars-with-metadata--GroMEt.json"
with open(path, 'r') as f:
    gromet_sir = json.load(f)

path = "/home/nliu/projects/aske/research/gromet/dist/uaz/CHIME/CHIME_SIR_Dyn_gromet_FunctionNetwork-with-vars-with-metadata--GroMEt_graph.json"
with open(path, 'r') as f:
    graph_sir = json.load(f)

path = "/home/nliu/projects/aske/research/gromet/data/uaz/CHIME/CHIME_SVIIvR_Dyn_gromet_FunctionNetwork-with-vars-with-metadata--GroMEt.json"
with open(path, 'r') as f:
    gromet_chime = json.load(f)

path = "/home/nliu/projects/aske/research/gromet/dist/uaz/CHIME/CHIME_SIR_Dyn_gromet_FunctionNetwork-with-vars-with-metadata--GroMEt_graph.json"
with open(path, 'r') as f:
    graph_chime = json.load(f)

path = "/home/nliu/projects/aske/research/gromet/data/uaz/gromet_intersection_graph/gig__CHIME_SIR_Base_v01-CHIME_SVIIvR_v01.json"
with open(path, 'r') as f:
    comparison_sir_chime = json.load(f)


f = None
del f




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
def draw_graph(G: Any, pos: Dict, ax: Optional[Any] = None, node_args: Optional[Dict] = None, edge_args: Optional[Dict] = None, label_args: Optional[Dict] = None, legend_args: Optional[Dict] = None, G_full: Optional[Any] = None, label_key: Optional[str] = 'label', label_angle: Optional[int] = 15) -> Any:

    if ax == None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20, 5))

    if node_args == None:

        if G_full == None:
            node_types = {t: i for i, t in enumerate(np.unique([G.nodes[node]['nodeType'] for node in G.nodes]))}
        else:
            node_types = {t: i for i, t in enumerate(np.unique([G_full.nodes[node]['nodeType'] for node in G_full.nodes]))}
        
        node_args = {
            # 'node_size': [len(get_node_children_nx(G, node_id = node)) for node in G.nodes],
            'node_size': [20 for node in G.nodes],
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
        __ = [t.set_rotation(label_angle) for __, t in h_labels.items()]

    # __ = plt.setp(ax, ylim = (-3, 3))

    if len(legend_args) > 0:
        # ax.legend(handles = legend_elements, loc = 'lower right', ncol = len(legend_elements))
        ax.legend(handles = legend_elements, **legend_args)


    return None

# Generate linear layout that is layered by the parent-child hierarchy of the graph
def generate_linear_layout_with_hierarchy(G: Any, draw: Optional[bool] = False, ax: Optional[Any] = None, draw_graph_args: Optional[Dict] = {}) -> Dict:

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
                __ = draw_graph(G_sub, pos = pos, ax = ax, G_full = G, **draw_graph_args)

    if draw == True:
        __ = plt.setp(ax, ylim = (-1, max_node_level + 1))
    
    return pos_full

# Calculate lineage of nodes of a given parsed graph
def calculate_node_lineage(graph: Any) -> Dict:

    map_ids_nodes = {node['id']: node for node in graph['nodes']}
    node_lineage = {node['id']: [] for node in graph['nodes']}

    for node_id in node_lineage.keys():

        node_temp = map_ids_nodes[node_id]

        while node_temp['parent'] != None:

            node_lineage[node_id].append(node_temp['parent'])

            node_temp = map_ids_nodes[node_temp['parent']]

        # Reverse order to ancestor-to-parent
        node_lineage[node_id] = node_lineage[node_id][::-1]

    return node_lineage


# %%[markdown]
# ## Draw FN SimpleSIR Model

# %%
G_sir = generate_nx_obj(graph = graph_sir)
__ = add_missing_edges(G = G_sir)
__ = promote_edges(G = G_sir)

# __ = generate_linear_layout(G_sir, draw = True)
pos_sir = generate_linear_layout_with_hierarchy(G_sir, draw = True)

# %%[markdown]
# ## Draw FN CHIME Model

# %%
G_chime = generate_nx_obj(graph = graph_chime)
__ = add_missing_edges(G = G_chime)
__ = promote_edges(G = G_chime)

# __ = generate_linear_layout(G_chime, draw = True)
pos_chime = generate_linear_layout_with_hierarchy(G_chime, draw = True, draw_graph_args = {'label_key': 'grometID', 'label_angle': 30})


# %%
# Comparison graph
G_comp = nx.union(G_sir, G_chime, rename = ('SIR-', 'CHIME-'))
edges = list(G_comp.edges)
G_comp.remove_edges_from(edges)


# Map GroMEt IDs to Dario-Parser IDs
map_grometID_id_sir = {node['grometID']: node['id'] for node in graph_sir['nodes']}
map_grometID_id_chime = {node['grometID']: node['id'] for node in graph_chime['nodes']}


if False:

    # Map variables to their states
    map_vars_states_sir = {var['uid']: var['states'] for var in gromet_sir['variables']}
    map_vars_states_chime = {var['uid']: var['states'] for var in gromet_chime['variables']}


    # Calculate node lineage to get hierarchical level
    node_lineage_sir = calculate_node_lineage(graph = graph_sir)
    node_lineage_chime = calculate_node_lineage(graph = graph_chime)

    # Truncate state lists to lowest hierarchical level only
    # Exclude wires since they are not nodes
    for var, states in map_vars_states_sir.items():
        x = {state: node_lineage_sir[map_grometID_id_sir[state]] for state in states if state in map_grometID_id_sir.keys()}
        i = np.argmin([len(v) for __, v in x.items()])
        map_vars_states_sir[var] = [list(x.keys())[i]]

    for var, states in map_vars_states_chime.items():
        x = {state: node_lineage_chime[map_grometID_id_chime[state]] for state in states if state in map_grometID_id_chime.keys()}
        i = np.argmin([len(v) for __, v in x.items()])
        map_vars_states_chime[var] = [list(x.keys())[i]]

    var = states = i = x = None
    del var, states, i, x


    # Comparison edges (many-to-many -> one-to-one)
    comparison_edges = {
        (map_grometID_id_sir[state_1], map_grometID_id_chime[state_2])
        for common_nodes in comparison_sir_chime['common_nodes'] for var_1 in common_nodes['g1_variable'] for state_1 in map_vars_states_sir[var_1] for var_2 in common_nodes['g2_variable'] for state_2 in map_vars_states_chime[var_2]
        if (state_1 in map_grometID_id_sir.keys()) & (state_2 in map_grometID_id_chime)
    }


if True:

    # Alternative way to create these one-to-one mappings using "proxy_state"
    map_vars_proxy_sir = {var['uid']: var['proxy_state'] for var in gromet_sir['variables']}
    map_vars_proxy_chime = {var['uid']: var['proxy_state'] for var in gromet_chime['variables']}

    comparison_edges = {
        (map_grometID_id_sir[map_vars_proxy_sir[var_1]], map_grometID_id_chime[map_vars_proxy_chime[var_2]]) 
        for common_nodes in comparison_sir_chime['common_nodes'] for var_1 in common_nodes['g1_variable'] for var_2 in common_nodes['g2_variable']
    }


# Problem: 
# * too many comparison edges since it was one-to-one in terms of variables
# * but each variable has many states
# * 8 hyperedges -> 55 simple edges

# Solution 1:
# Only retain the lowest-level pairing by truncate the state lists (^)
# 
# Solution 2:
# Use the "proxy_state" attribute of each variable


__ = G_comp.add_edges_from([
    (
        'SIR-' + src, 
        'CHIME-' + tgt, 
        0
    ) 
    for src, tgt in comparison_edges
])

# %%
# Parallel positions
p_chime = {node: p + 5 * np.array([0, 1]) for node, p in pos_chime.items()}
p_sir = {node: p - 5 * np.array([0, 1]) for node, p in pos_sir.items()}


# # Align mapped nodes
# for src, tgt, __ in G_comp.edges:

#     src = ''.join(src.split('-')[1:])
#     tgt = ''.join(tgt.split('-')[1:])

#     p_sir[src][0] = p_chime[tgt][0]

#     print(f"{src} ---> {tgt}")


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20, 5))
__ = draw_graph(G_chime, pos = p_chime, ax = ax, G_full = G_comp, label_key = 'label', label_args = {'font_size': 6, 'alpha': 0.1})
__ = draw_graph(G_sir, pos = p_sir, ax = ax, G_full = G_comp, label_key = 'label', label_args = {'font_size': 6, 'alpha': 0.1})

p_chime = {'CHIME-' + node: val for node, val in p_chime.items() if node != None}
p_sir = {'SIR-' + node: val for node, val in p_sir.items() if node != None}
p_comp = {**p_chime, **p_sir}

__ = draw_graph(G = G_comp, pos = p_comp, ax = ax, node_args = {}, label_args = {}, legend_args = {}, edge_args = {'edge_color': 'tab:cyan', 'alpha': 1.0})
# __ = plt.setp(ax, ylim = (-3, 5))
# __ = plt.setp(ax, ylim = (-6, 15))
__ = plt.setp(ax, ylim = (-8, 12))

# fig.savefig(f'../figures/comparison_SimpleSIR_CHIME_FN.png', dpi = 150)
# fig.savefig(f'../figures/comparison_SIRBase_SVIIR.png', dpi = 150)
fig.savefig(f'../figures/comparison_SIRBase_SVIIR_Dyn.png', dpi = 150)

# %%[markdown]
# # Generate Output for HMI

# %%
# Generate a map between the variables of a GroMEt and its states
# The first state is always the "proxy state"
def generate_map_vars_states(gromet: Any, filter_nodes: Optional[List] = None) -> Any:

    map_vars_states = {var['uid']: tuple({**{var['proxy_state']: None}, **{state: None for state in var['states']}}.keys()) for var in gromet['variables']}

    if isinstance(filter_nodes, List):
        for k, v in map_vars_states.items():
            map_vars_states[k] = tuple([element for element in v if element in filter_nodes])

    return map_vars_states

# %%
map_vars_states_sir = generate_map_vars_states(gromet = gromet_sir, filter_nodes = [node['grometID'] for node in graph_sir['nodes']])
map_vars_states_chime = generate_map_vars_states(gromet = gromet_chime, filter_nodes = [node['grometID'] for node in graph_chime['nodes']])


# %%

output = {g1['uid']: {g2['uid']: {}} for g1 in (gromet_sir, gromet_chime) for g2 in (gromet_sir, gromet_chime) if g1 != g2}

# %%
# ## One-to-Many Comparison Edges

# %%
# For each state of the referenced variables in G1, map to the list of states in the corresponding variable in G2
output[gromet_sir['uid']][gromet_chime['uid']] = { 
    state: list(map_vars_states_chime[common_node['g2_variable'][0]])
    for common_node in comparison_sir_chime['common_nodes'] 
    for state in map_vars_states_sir[common_node['g1_variable'][0]]
}

# Vice versa
output[gromet_chime['uid']][gromet_sir['uid']] = { 
    state: list(map_vars_states_sir[common_node['g1_variable'][0]])
    for common_node in comparison_sir_chime['common_nodes'] 
    for state in map_vars_states_chime[common_node['g2_variable'][0]]
}

# %%
# path = "../dist/uaz/gromet_intersection_graph/gig__SimpleSIR_metadata-CHIME_SIR_v01_HMI.json"
# path = "../dist/uaz/gromet_intersection_graph/gig__CHIME_SIR_Base_v01-CHIME_SVIIvR_v01_HMI.json"
path = "../dist/uaz/gromet_intersection_graph/gig__CHIME_SIR_Base_v01-CHIME_SVIIvR_v01_Dyn_HMI.json"
with open(path, 'w') as f:
    json.dump(output, f, indent = 2)

# %%
