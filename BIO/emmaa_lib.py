# %% [markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %% [markdown]
# ## Import required modules.

# import sys
# from time import time
from networkx.algorithms.centrality.degree_alg import out_degree_centrality
import numpy as np
# import scipy as sp
# import csv
import re
import numba
import networkx as nx

# import sklearn as skl
# import hdbscan

import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# %%

# Scatter plot of (un)labeled data points
def plot_emb(coor = np.array([]), labels = [], ax = [], figsize = (12, 12), marker_size = 2.0, marker_alpha = 0.5,  cmap_name = 'qual', colorbar = True, str_title = '', xlim = (), ylim = (), zlim = (), vlim = (), hull = []):

    # Error handling
    if not isinstance(coor, np.ndarray):
        raise TypeError("'coor' must be an numpy ndarray.")
    if not (coor.shape[1] <= 3):
        raise ValueError("'coor' must be a N x 2 or N x 3 array.")
    if not ((isinstance(labels, list) | isinstance(labels, np.ndarray)) and (len(labels) in [0, coor.shape[0]])): 
        raise TypeError("'labels' must be either [] or a N x 1 list or numpy ndarrray.")
    if not (isinstance(marker_size, int) | isinstance(marker_size, float) | isinstance(marker_size, list) | isinstance(marker_size, np.ndarray)):
        raise TypeError("'marker_size' must be either an int, a float, a list, or a numpy ndarray.")
    if not (isinstance(marker_alpha, int) | isinstance(marker_alpha, float)):
        raise TypeError("'marker_alpha' must be either an int or a float.")
    if not (isinstance(xlim, tuple) | isinstance(ylim, tuple) | isinstance(zlim, tuple) | isinstance(vlim, tuple)):
        raise TypeError("'xlim', 'ylim', 'zlim', 'vlim' must be tuples.")
    if not isinstance(hull, list):
        raise TypeError("'hull' must be a list.")

    # Dimensions
    n_row, n_dim_emb = coor.shape

    # Colormap
    n_uniq = 1
    if len(labels) < 1:
        n_uniq = 1
        labels = np.zeros((n_row, ), dtype = np.int8)

    # Qualitative cmap for cluster labels
    # tab10: Tableau 10
    if cmap_name == 'qual':
        labels_uniq = np.unique(labels)
        n_uniq = labels_uniq.size
        col = np.asarray([plt.cm.get_cmap('tab10')(i) for i in range(10)])
        col[:, 3] = 1.0

    # Sequential cmap
    # e.g. cividis, gray, RdBu
    # (+ '_r' for reverse)
    else:
        col = plt.cm.get_cmap(cmap_name)

        # n_uniq = len(labels)
        # labels_uniq = np.arange(n_uniq)
        labels_uniq = np.unique(labels)
        n_uniq = labels_uniq.size

        if len(vlim) < 2:
            vlim = (np.min(labels), np.max(labels))

    # # Recentre
    r0 = np.median(coor, axis = 0)
    # coor = coor - r0

    # Plot figure
    if type(ax).__name__ != 'AxesSubplot':
        fig = plt.figure(figsize = figsize)
    else:
        fig = plt.getp(ax, 'figure')

    # 2D plot
    if n_dim_emb == 2:

        if type(ax).__name__ != 'AxesSubplot':
            ax = fig.add_subplot(111)

        # Shapes
        if len(hull) == n_uniq:

            for i in range(n_uniq):
                
                # Skip noise
                if labels_uniq[i] == -1:
                    continue
                
                # Skip empty
                if len(hull[i]) < 1:
                    continue
                
                # Alpha hulls
                if len(hull[i][0].shape) == 1:

                    j = (labels == labels_uniq[i])
                    lines = hull[i]

                    for k in lines:
                        x = coor[j, 0][k]
                        y = coor[j, 1][k]
                        ax.fill(x, y, facecolor = col[i % 10, :], edgecolor = col[i % 10, :], linewidth = 1, label = f'{labels_uniq[i]}', zorder = 0)

                # Voronoi polygons
                elif len(hull[i][0].shape) == 2:
                    
                    polygons = hull[i]
                    for k in polygons:
                        k = k - r0
                        x = k[:, 0]
                        y = k[:, 1]
                        ax.fill(x, y, facecolor = col[i % 10, :], edgecolor = col[i % 10, :], linewidth = 1, label = f'{labels_uniq[i]}', zorder = 0)

        # Scatter plots
        else:

            # Qualitative colours
            if cmap_name == 'qual':
                for i in range(n_uniq):
                    j = (labels == labels_uniq[i])

                    if isinstance(marker_size, int) or isinstance(marker_size, float):
                        plt_obj = ax.scatter(coor[j, 0], coor[j, 1], marker = 'o', s = marker_size, facecolor = col[i % 10, :3], alpha = marker_alpha, label = f'{labels_uniq[i]}', zorder = 0)
                    else:
                        plt_obj = ax.scatter(coor[j, 0], coor[j, 1], marker = 'o', s = marker_size[j], facecolor = col[i % 10, :3], alpha = marker_alpha, label = f'{labels_uniq[i]}', zorder = 0)

            # Sequential colours
            elif cmap_name != '':

                if isinstance(marker_size, int) or isinstance(marker_size, float):
                    plt_obj = ax.scatter(coor[:, 0], coor[:, 1], c = labels, cmap = col, vmin = vlim[0], vmax = vlim[1], marker = 'o', s = marker_size, alpha = marker_alpha, label = f'', zorder = 0)
                else:
                    plt_obj = ax.scatter(coor[:, 0], coor[:, 1], c = labels, cmap = col, vmin = vlim[0], vmax = vlim[1], marker = 'o', s = marker_size[j], alpha = marker_alpha, label = f'', zorder = 0)
    # 3D plot
    elif n_dim_emb == 3:

        if type(ax).__name__ != 'AxesSubplot':
            ax = fig.add_subplot(111, projection = '3d')

        # Qualitative colours
        if cmap_name == 'qual':
            for i in range(n_uniq):
                j = (labels == labels_uniq[i])

                if isinstance(marker_size, int) or isinstance(marker_size, float):
                    plt_obj = ax.scatter(coor[j, 0], coor[j, 1], coor[j, 2], marker = 'o', s = marker_size, facecolor = col[i % 10, :3], alpha = marker_alpha, label = f'{labels_uniq[i]}')
                else:
                    plt_obj = ax.scatter(coor[j, 0], coor[j, 1], coor[j, 2], marker = 'o', s = marker_size[j], facecolor = col[i % 10, :3], alpha = marker_alpha, label = f'{labels_uniq[i]}')

        # Sequential colours
        else:
            if isinstance(marker_size, int) or isinstance(marker_size, float):
                plt_obj = ax.scatter(coor[:, 0], coor[:, 1], coor[:, 2], c = labels, cmap = col, marker = 'o', s = marker_size, alpha = marker_alpha, label = f'')
            else:
                plt_obj = ax.scatter(coor[:, 0], coor[:, 1], coor[:, 2], c = labels, cmap = col, marker = 'o', s = marker_size[j], alpha = marker_alpha, label = f'')

    # Default ranges
    ax_lim = np.asarray([plt.getp(ax, 'xlim'), plt.getp(ax, 'ylim'), (0, 0)])
    if n_dim_emb == 3:
        ax_lim[2] = plt.getp(ax, 'zlim')
    i = 0.5 * np.max(ax_lim[:, 1] - ax_lim[:, 0])
    x = np.mean(ax_lim, axis = 1)

    # Custom ranges
    if len(xlim) == 2:
        plt.setp(ax, xlim = xlim)
    else:
        plt.setp(ax, xlim = (x[0] - i, x[0] + i))
    
    if len(ylim) == 2:
        plt.setp(ax, ylim = ylim)
    else:
        plt.setp(ax, ylim = (x[1] - i, x[1] + i))

    if (len(zlim) == 2) & (n_dim_emb == 3):
        plt.setp(ax, zlim = zlim)
    elif (len(zlim) != 2) & (n_dim_emb == 3):
        plt.setp(ax, zlim = (x[2] - i, x[2] + i))

    # Axis labels and aspect ratio
    if n_dim_emb == 2:
        plt.setp(ax, xlabel = '$x$', ylabel = '$y$', aspect = 1.0)
    else:
        plt.setp(ax, xlabel = '$x$', ylabel = '$y$', zlabel = '$z$')
    
    # Legend
    if (cmap_name == 'qual') & (n_uniq <= 10) & (n_uniq > 1):

        # Custom
        legend_obj = [mpl.lines.Line2D([0], [0], marker = 'o', markersize = 2.0 ** 2, color = 'none', markeredgecolor = 'none', markerfacecolor = col[i, :3], alpha = 1.0, label = f'{labels_uniq[i]}') for i in range(n_uniq)]
        ax.legend(handles = legend_obj, loc = 'lower left')

    # Axis title
    if str_title == ' ':
        pass
    
    elif len(str_title) == 0:

        str_title = f'{n_uniq} Unique Labels'

        if 'lines' in locals():
            str_title += ' (Alpha)'
        elif 'polygons' in locals():
            str_title += ' (Voronoi)'
        else:
            str_title += ' (Scatter)'

        plt.setp(ax, title = str_title)

    else:
        plt.setp(ax, title = str_title)

    # Colorbar
    if (cmap_name != 'qual') and (colorbar == True): 
        plt.colorbar(plt_obj, ax = plt.getp(fig, 'axes'), extend = 'max', shrink = 0.85)


    return fig, ax

#%%
# Get the index of all connecting nodes and edges within N directed hops of the nodes with the given text names
def getTextNodeEdgeIndices(nodes, edges, texts, numHops = 1):

    # Get node indices with given text names
    textsIndex = np.array([np.flatnonzero(np.asarray([(text in node['info']['text']) and (node['grounded'] == True) for node in nodes])) for text in texts]).flatten()

    # List of edge sources and targets
    x = np.array([[edge['source'], edge['target']] for edge in edges])

    # Flag edges 
    edgeFlags = np.full((len(edges), ), False)
    z = textsIndex
    for hop in range(numHops):

        # Match source
        edgeFlags = edgeFlags + np.sum([x[:, 0] == i for i in z], axis = 0).astype(bool)

        # Get target
        z = x[edgeFlags, 1]

    # Get node/edge indices
    textsEdgeIndex = np.flatnonzero(edgeFlags)
    textsNodeIndex = np.array(list(set([edges[j]['source'] for j in textsEdgeIndex] + [edges[j]['target'] for j in textsEdgeIndex])))

    # Flag nodes
    nodeFlags = [True if j in textsNodeIndex else False for j, node in enumerate(nodes)]

    return textsIndex, textsNodeIndex, textsEdgeIndex, nodeFlags, edgeFlags






# %%
# Parse EMMAA statements and return a node/edge list
def parse_statements(statements):


    # Only keep statements from which direct edges can be clearly extracted
    source_target_pairs = [{'subj', 'obj'}, {'enz', 'sub'}, {'subj', 'obj_from', 'obj_to'}, {'members'}]
    bool_ind = np.sum(np.array([[True if x <= set(s.keys()) else False for s in statements] for x in source_target_pairs]), axis = 0)
    statements = [s for i, s in zip(bool_ind, statements) if i]


    # Extracted edge list
    edges = []
    for s in statements:

        # subj/obj statements
        if source_target_pairs[0] <= set(s.keys()):

            edge = [{
                'id': None, 
                'type': str(s['type']), 
                'belief': float(s['belief']), 
                'statement_id': str(s['matches_hash']), 
                'source_name': str(s['subj']['name']),
                'source_db_refs': s['subj']['db_refs'],
                'target_name': str(s['obj']['name']),
                'target_db_refs': s['obj']['db_refs']
            }]

        # enz/sub statements
        elif source_target_pairs[1] <= set(s.keys()):

            edge = [{
                'id': None, 
                'type': str(s['type']), 
                'belief': float(s['belief']), 
                'statement_id': str(s['matches_hash']), 
                'source_name': str(s['enz']['name']), 
                'source_db_refs': s['enz']['db_refs'],
                'target_name': str(s['sub']['name']),
                'target_db_refs': s['sub']['db_refs'],
            }]

        # subj/obj_from/obj_to statements
        # 1. subj -> obj_from
        # 2. obj_from -> obj_to
        elif source_target_pairs[2] <= set(s.keys()):

            edge = [{
                'id': None, 
                'type': str(s['type']), 
                'belief': float(s['belief']), 
                'statement_id': str(s['matches_hash']), 
                'source_name': str(s['subj']['name']), 
                'source_db_refs': s['subj']['db_refs'],
                'target_name': str(s['obj_from'][0]['name']),
                'target_db_refs': s['obj_from'][0]['db_refs'],
            }, 
            {
                'id': None, 
                'type': str(s['type']), 
                'belief': float(s['belief']), 
                'statement_id': str(s['matches_hash']), 
                'source_name': str(s['obj_from'][0]['name']), 
                'source_db_refs': s['obj_from'][0]['db_refs'],
                'target_name': str(s['obj_to'][0]['name']),
                'target_db_refs': s['obj_to'][0]['db_refs'],
            }]

        # many-member statements
        # * consider only two- and three-member statements
        # * assume bidirectionality
        elif (source_target_pairs[3] <= set(s.keys())) & (len(s['members']) <= 3):
            
            num_members = len(s['members'])
            perm = [(i, j) for i in range(num_members) for j in range(num_members) if i != j]

            edge = [{
                'id': None, 
                'type': str(s['type']), 
                'belief': float(s['belief']), 
                'statement_id': str(s['matches_hash']), 
                'source_name': str(s['members'][x[0]]['name']), 
                'source_db_refs': s['members'][x[0]]['db_refs'], 
                'target_name': str(s['members'][x[1]]['name']),
                'target_db_refs': s['members'][x[1]]['db_refs'],
            } for x in perm]

    
        edges.extend(edge)


    # Generate edge IDs
    for i, edge in enumerate(edges):
        edge['id'] = i


    # Extract node list
    nodes_name = {**{edge['source_name']: {'edge_ids_source': [], 'edge_ids_target': []} for edge in edges}, **{edge['target_name']: {'edge_ids_source': [], 'edge_ids_target': []} for edge in edges}}
    for edge in edges:
        nodes_name[edge['source_name']]['db_refs'] = edge['source_db_refs']
        nodes_name[edge['target_name']]['db_refs'] = edge['target_db_refs']
        nodes_name[edge['source_name']]['edge_ids_source'].append(edge['id'])
        nodes_name[edge['target_name']]['edge_ids_target'].append(edge['id'])

    nodes = [{
        'id': i,
        'name': name, 
        'db_refs': nodes_name[name]['db_refs'],
        'grounded': len(set(nodes_name[name]['db_refs'].keys()) - {'TEXT'}) > 0, 
        'edge_ids_source': nodes_name[name]['edge_ids_source'], 
        'edge_ids_target': nodes_name[name]['edge_ids_target'],
        'in_degree': len(nodes_name[name]['edge_ids_target']),
        'out_degree': len(nodes_name[name]['edge_ids_source'])
    } for i, name in enumerate(nodes_name)]


    return nodes, edges, statements

# %%







# %%
# Intersect a set of graph nodes/edges with a set of graph paths
def intersect_graph_paths(nodes, edges, paths):

    # Get the edge IDs
    edgeIDs_edges = set([edge['id'] for edge in edges]) - {None}
    edgeIDs_paths = set([edge for path in paths for edge in path['edge_ids']]) - {None}

    # Find intersection between the graph edges and the path edges
    edgeIDs_inter = edgeIDs_paths & edgeIDs_edges
    # print(f"{len(edgeIDs_inter)} {len(edgeIDs_paths)} {len(edgeIDs_edges)}")

    # Select the edges within the intersection
    nodes_inter = [node for node in nodes if len(set(node['edge_ids']) & edgeIDs_inter) > 0]
    edges_inter = [edge for edge in edges if edge['id'] in edgeIDs_inter]
    
    # Get the node IDs in the intersection
    nodeIDs_inter = set([node['id'] for node in nodes_inter]) - {None}

    # Remove `None` from `node_ids` and `edge_ids` of `paths`
    paths_inter = [path for path in paths if len(set(path['edge_ids']) & edgeIDs_inter) > 0]
    num_paths = len(paths_inter)
    for i in range(num_paths):
        paths_inter[i]['node_ids'] = list(set(paths_inter[i]['node_ids']) & nodeIDs_inter)
        paths_inter[i]['edge_ids'] = list(set(paths_inter[i]['edge_ids']) & edgeIDs_inter)

    # Restrict `edge_ids` in `nodes` to the subgraph
    num_nodes = len(nodes_inter)
    for i in range(num_nodes):
        nodes_inter[i]['edge_ids'] = list(set(nodes_inter[i]['edge_ids']) & edgeIDs_inter)

    return nodes_inter, edges_inter, paths_inter

# %%
# Reset node ids in `nodes` and `edges`
def reset_node_ids(nodes, edges):

    # Make new node-id map
    map_nodes_ids = {node['id']: i for i, node in enumerate(nodes)}

    num_nodes = len(nodes)
    for i in range(num_nodes):
        nodes[i]['id'] = i

    num_edges = len(edges)
    for i in range(num_edges):
        j = edges[i]['source']
        edges[i]['source'] = map_nodes_ids[j]

        k = edges[i]['target']
        edges[i]['target'] = map_nodes_ids[k]

    return nodes, edges, map_nodes_ids

# %%
# Calculate in- and out-degree of nodes in `nodes` using `edges` data
def calculate_node_degrees(nodes, edges):

    # Make node-id map
    map_nodes_ids = {node['id']: i for i, node in enumerate(nodes)}

    # Count edge sources and targets
    num_nodes = len(nodes)
    in_degree = [0 for i in range(num_nodes)]
    out_degree = [0 for i in range(num_nodes)]
    for edge in edges:
        i = map_nodes_ids[edge['source']]
        j = map_nodes_ids[edge['target']]
        in_degree[j] += 1
        out_degree[i] += 1

    # Insert into `nodes`
    for i in range(num_nodes):
        nodes[i]['in_degree'] = in_degree[i]
        nodes[i]['out_degree'] = out_degree[i]


    return nodes, in_degree, out_degree

# %%
# Get node-wise belief score from `edges`
def calculate_node_belief(nodes, edges, mode = 'max'):

    # Make node-id map
    map_nodes_ids = {node['id']: i for i, node in enumerate(nodes)}


    # Count edge sources and targets
    num_nodes = len(nodes)
    belief_edges = [[] for i in range(num_nodes)]
    for edge in edges:
        i = map_nodes_ids[edge['source']]
        j = map_nodes_ids[edge['target']]
        belief_edges[i].append(edge['belief'])
        belief_edges[j].append(edge['belief'])


    # Max
    if mode == 'max':
        belief = [float(max(b)) if len(b) > 0 else 0.0 for b in belief_edges]
    elif mode == 'median':
        belief = [float(np.median(b)) if len(b) > 0 else 0.0 for b in belief_edges]
    elif mode == 'mean':
        belief = [float(np.mean(b)) if len(b) > 0 else 0.0 for b in belief_edges]


    # Insert into `nodes`
    for i in range(num_nodes):
        nodes[i]['belief'] = belief[i]

    return nodes, belief

# %%
def generate_ordered_namespace_list(namespaces_priority, ontoJSON, nodes):

    # Namespaces referenced in the ontology node/category names
    namespaces_onto = set([re.findall('\w{1,}(?=:)', node['id'])[0] for node in ontoJSON['nodes']])


    # Namespaces referenced in the model nodes 'db_refs'
    namespaces_model = set([namespace for node in nodes for namespace in node['db_refs'].keys()])


    # Combine the namespace lists in order: 
    # * given priority
    # * 'not-grounded'
    # * from model
    # * from ontology
    x = sorted(list(namespaces_model - set(namespaces_priority)))
    y = sorted(list(namespaces_onto - set(namespaces_priority) - namespaces_model))
    namespaces = namespaces_priority + x + y


    # Count references
    namespaces_count = {namespace: [0, 0] for namespace in namespaces}
    for node in nodes:
        for name in node['db_refs'].keys():
            namespaces_count[name][0] += 1
    
    for node in ontoJSON['nodes']:
        name = re.findall('\w{1,}(?=:)', node['id'])[0]
        namespaces_count[name][1] += 1

    
    return namespaces, namespaces_count

# %%
# Reduce 'db_refs' of each model node to a single entry by namespace priority
# * `namespace`:`ref` -> 'ontocats_ref'
# * `grounded = False` -> 'not-grounded' 
def reduce_nodes_db_refs(nodes, namespaces):

    num_nodes = len(nodes)
    for i in range(num_nodes):

        if nodes[i]['grounded'] == True:
            n = [name for name in namespaces if name in nodes[i]['db_refs'].keys()][0]
            nodes[i]['db_ref_priority'] = f"{n}:{nodes[i]['db_refs'][n]}"
        else:
            nodes[i]['db_ref_priority'] = 'not-grounded'

    # Column-wise result
    nodes_db_ref_priority = [node['db_ref_priority'] for node in nodes]

    return nodes, nodes_db_ref_priority

# %%
# Generate NetworkX layout from given subgraph (as specified by `edges`)
def generate_nx_layout(nodes, edges, node_list = [], edge_list = [], layout = 'spring', layout_atts = {}, draw = False, draw_atts = {}, ax = None):
    
    # Generate node list if unavailable
    # node_list = <list of (node, attributes = {k: v})>
    if len(node_list) == 0:
        node_list = [(node['id'], {'name': node['name']}) for node in nodes]

    # Generate edge list if unavailable
    # edge_list = <list of (source_node, target_node, attributes = {k: v})>
    if len(edge_list) == 0:

        # Generate edge list from `edges`
        edge_dict = {(edge['source'], edge['target']): 0 for edge in edges}
        for edge in edges:
            edge_dict[(edge['source'], edge['target'])] += 1

        # weight = number of equivalent edges
        edge_list = [(edge[0], edge[1], {'weight': edge_dict[edge]}) for edge in edge_dict]


    # Generate NetworkX graph object from node and edge list
    G = nx.DiGraph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    # print(f"{G.number_of_nodes()} nodes and {G.number_of_edges()} edges has been added to the graph object.")
    # Note: self-loops are ignored


    # Generate layout coordinate
    if layout == 'kamada_kawai':
        coors = nx.kamada_kawai_layout(G, weight = 'weight', **layout_atts)
    elif layout == 'spring':
        coors = nx.spring_layout(G, weight = 'weight', seed = 0, **layout_atts)
    else:
        print(f"No layout selected!")
        draw = False
        coors = {}

    if draw == True:

        if ax == None:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))

        # print(plt.style.available)
        plt.style.use('fast')
        draw_atts = {
            'ax': ax,
            'arrows': False, 
            'with_labels': False,
            'node_size': 0.5,
            'width': 0.05,
            'alpha': 0.8,
            'cmap': 'cividis',
            'edge_color': 'k'
        }

        nx.draw_networkx(G, pos = coors, **draw_atts)
        __ = plt.setp(ax, aspect = 1.0)
    

    return coors, G

# %%
# Match arrays using a hash table
@numba.njit
def match_arrays(A, B):

    # hashTable = {b: True for b in B}
    hashTable = numba.typed.Dict.empty(key_type = numba.int32, value_type = numba.boolean)
    for b in B:
        hashTable[b] = True

    index = np.zeros((len(A), ), dtype = numba.boolean)
    for i, a in enumerate(A):
        try:
            index[i] = hashTable[a]
        except:
            pass

    return index

# %%