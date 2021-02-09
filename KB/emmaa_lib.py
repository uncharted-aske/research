# %% [markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %% [markdown]
# ## Import required modules.

# import sys
# from time import time
from networkx.algorithms.centrality.degree_alg import out_degree_centrality
from networkx.utils.decorators import nodes_or_number
import numpy as np
# import scipy as sp
# import csv
import copy
import json
import re
import numba
import networkx as nx

import sklearn as skl
# import hdbscan

import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# %%

# Scatter plot of (un)labeled data points
def plot_emb(coor = np.array([]), edge_list = [], labels = [], ax = [], figsize = (12, 12), marker_size = 2.0, marker_alpha = 0.5,  cmap_name = 'qual', legend_kwargs = {'loc': 'lower left', 'ncol': 1}, colorbar = True, str_title = '', xlim = (), ylim = (), zlim = (), vlim = (), hull = []):

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
                        plt_obj = ax.scatter(coor[j, 0], coor[j, 1], marker = 'o', s = marker_size, facecolor = col[i % 10, :3], alpha = marker_alpha, label = f'{labels_uniq[i]}', zorder = 100)
                    else:
                        plt_obj = ax.scatter(coor[j, 0], coor[j, 1], marker = 'o', s = marker_size[j], facecolor = col[i % 10, :3], alpha = marker_alpha, label = f'{labels_uniq[i]}', zorder = 100)

            # Sequential colours
            elif cmap_name != '':

                if isinstance(marker_size, int) or isinstance(marker_size, float):
                    plt_obj = ax.scatter(coor[:, 0], coor[:, 1], c = labels, cmap = col, vmin = vlim[0], vmax = vlim[1], marker = 'o', s = marker_size, alpha = marker_alpha, label = f'', zorder = 100)
                else:
                    plt_obj = ax.scatter(coor[:, 0], coor[:, 1], c = labels, cmap = col, vmin = vlim[0], vmax = vlim[1], marker = 'o', s = marker_size, alpha = marker_alpha, label = f'', zorder = 100)
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


    # Draw edges
    if len(edge_list) > 0:

        for edge in edge_list:
            # __ = mpl.lines.Line2D(coor[edge, 0], coor[edge, 1], linewidth = 1, marker = None, color = 'k', alpha = 0.5, zorder = 1)
            __ = ax.plot(coor[edge, 0], coor[edge, 1], linewidth = 0.05, marker = None, color = 'k', alpha = 0.5, zorder = 1)


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
    if (cmap_name == 'qual') & (n_uniq <= 10) & (n_uniq > 1) & (len(legend_kwargs) > 0):

        # Custom
        legend_obj = [mpl.lines.Line2D([0], [0], marker = 'o', markersize = 2.0 ** 2, color = 'none', markeredgecolor = 'none', markerfacecolor = col[i, :3], alpha = 1.0, label = f'{labels_uniq[i]}') for i in range(n_uniq)]
        
        # ax.legend(handles = legend_obj, loc = 'lower left')
        ax.legend(handles = legend_obj, **legend_kwargs)

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
    x = np.array([[edge['source_id'], edge['target_id']] for edge in edges])

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
    textsNodeIndex = np.array(list(set([edges[j]['source_id'] for j in textsEdgeIndex] + [edges[j]['target_id'] for j in textsEdgeIndex])))

    # Flag nodes
    nodeFlags = [True if j in textsNodeIndex else False for j, node in enumerate(nodes)]

    return textsIndex, textsNodeIndex, textsEdgeIndex, nodeFlags, edgeFlags







# %%
# Recursively sanitize the strings of a nested object
def sanitize_strings(obj):

    # Regex pattern for invalid JSON characters
    # * control characters (U+0000 to U+001F)
    # * reverse solidus/backslash (U+005C)
    # * double quotes (U+0022)
    pattern = re.compile(r"[\u0000-\u001F]|(\u005C)|(\u0022)")

    if isinstance(obj, str):

        # Sanitize
        obj = re.sub(pattern, '', obj)


    if isinstance(obj, list) | isinstance(obj, tuple) | isinstance(obj, set):

        # Sanitize the items
        for obj_item in obj:
            sanitize_strings(obj_item)

    if isinstance(obj, dict):

        # Sanitize the keys and values
        for key, value in obj.items():
            sanitize_strings(key)
            sanitize_strings(value)


# %%
# Save list of objects as a JSONL file
# (using `json.dumps` to ensure preservation of escape characters)
def save_jsonl(list_dicts, full_path, preamble = None):

    # Make directory if non-existent
    pathlib.Path(pathlib.PurePath(full_path).parents[0]).mkdir(parents = True, exist_ok = True)

    # Write file
    with open(f'{full_path}', 'w') as x:

        # Preamble
        if preamble != None:
            x.write(json.dumps(preamble) + '\n')

        # Data
        for obj in list_dicts:
            
            obj_ = obj

            if preamble != None:

                obj_ = {k: obj[k] if k in obj.keys() else None for k in preamble.keys()}
            
            x.write(json.dumps(obj_) + '\n')



# %%
# Load JSONL file
def load_jsonl(full_path, remove_preamble = False):

    # Check if the path points to an existing file
    if pathlib.Path(full_path).exists() == False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), full_path)

    list_objects = []
    with open(f'{full_path}', 'r') as file:
        for line in file:

            # Read line by line
            try: 
                list_objects.append(json.loads(line))
            except:
                print(line)

    if remove_preamble:
        list_objects = list_objects[1:]

    return list_objects


# %%
# Process EMMAA statements and return a node/edge list
def process_statements(statements, paths = [], model_id = None):

    statements_processed = []
    nodes = []
    edges = []
    num_statements = len(statements)
    if num_statements > 0:

        # Only keep statements from which direct edges can be clearly extracted
        source_target_pairs = [{'subj', 'obj'}, {'enz', 'sub'}, {'subj', 'obj_from', 'obj_to'}, {'members'}]
        bool_ind = np.sum(np.array([[True if x <= set(s.keys()) else False for s in statements] for x in source_target_pairs]), axis = 0)
        statements_processed = [s for i, s in zip(bool_ind, statements) if i]
        num_statements_processed = len(statements_processed)
        
        statements = None
        del statements


        # Extracted edge list
        edge = []
        for s in statements_processed:
            
            # subj/obj statements
            if source_target_pairs[0] <= set(s.keys()):

                edge = [{
                    'model_id': model_id,
                    'id': None, 
                    'type': str(s['type']), 
                    'belief': float(s['belief']), 
                    'statement_id': str(s['matches_hash']), 
                    'source_id': None, 
                    'source_name': str(s['subj']['name']),
                    'source_db_refs': s['subj']['db_refs'],
                    'target_id': None, 
                    'target_name': str(s['obj']['name']),
                    'target_db_refs': s['obj']['db_refs'], 
                    'tested': False
                }]

            # enz/sub statements
            if source_target_pairs[1] <= set(s.keys()):

                edge = [{
                    'model_id': model_id,
                    'id': None, 
                    'type': str(s['type']), 
                    'belief': float(s['belief']), 
                    'statement_id': str(s['matches_hash']), 
                    'source_id': None, 
                    'source_name': str(s['enz']['name']), 
                    'source_db_refs': s['enz']['db_refs'],
                    'target_id': None, 
                    'target_name': str(s['sub']['name']),
                    'target_db_refs': s['sub']['db_refs'],
                    'tested': False
                }]

            # subj/obj_from/obj_to statements
            # 1. subj -> obj_from
            # 2. obj_from -> obj_to
            if source_target_pairs[2] <= set(s.keys()):

                edge = [{
                    'model_id': model_id,
                    'id': None, 
                    'type': str(s['type']), 
                    'belief': float(s['belief']), 
                    'statement_id': str(s['matches_hash']), 
                    'source_id': None, 
                    'source_name': str(s['subj']['name']), 
                    'source_db_refs': s['subj']['db_refs'],
                    'target_id': None, 
                    'target_name': str(s['obj_from'][0]['name']),
                    'target_db_refs': s['obj_from'][0]['db_refs'],
                    'tested': False
                }, 
                {
                    'model_id': model_id,
                    'id': None, 
                    'type': str(s['type']), 
                    'belief': float(s['belief']), 
                    'statement_id': str(s['matches_hash']), 
                    'source_id': None, 
                    'source_name': str(s['obj_from'][0]['name']), 
                    'source_db_refs': s['obj_from'][0]['db_refs'],
                    'target_id': None, 
                    'target_name': str(s['obj_to'][0]['name']),
                    'target_db_refs': s['obj_to'][0]['db_refs'],
                    'tested': False
                }]

            # many-member statements
            # * consider only two- and three-member statements
            # * assume bidirectionality
            if source_target_pairs[3] <= set(s.keys()):
                
                if len(s['members']) <= 3:
                    num_members = len(s['members'])
                    perm = [(i, j) for i in range(num_members) for j in range(num_members) if i != j]

                    edge = [{
                        'model_id': model_id,
                        'id': None, 
                        'type': str(s['type']), 
                        'belief': float(s['belief']), 
                        'statement_id': str(s['matches_hash']), 
                        'source_id': None, 
                        'source_name': str(s['members'][x[0]]['name']), 
                        'source_db_refs': s['members'][x[0]]['db_refs'], 
                        'target_id': None, 
                        'target_name': str(s['members'][x[1]]['name']),
                        'target_db_refs': s['members'][x[1]]['db_refs'],
                        'tested': False
                    } for x in perm]

            edges.extend(edge)
            edge = []


        # Generate edge IDs
        num_edges = len(edges)
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
            'model_id': model_id,
            'id': i,
            'name': name, 
            'db_refs': nodes_name[name]['db_refs'],
            'grounded': len(set(nodes_name[name]['db_refs'].keys()) - {'TEXT'}) > 0, 
            'edge_ids_source': nodes_name[name]['edge_ids_source'], 
            'edge_ids_target': nodes_name[name]['edge_ids_target'],
            'out_degree': len(nodes_name[name]['edge_ids_source']),
            'in_degree': len(nodes_name[name]['edge_ids_target'])
        } for i, name in enumerate(nodes_name)]
        num_nodes = len(nodes)
    

        # Map node names to IDs
        for i, name in enumerate(nodes_name):
            nodes_name[name]['id'] = i
        
        for edge in edges:
            edge['source_id'] = nodes_name[edge['source_name']]['id']
            edge['target_id'] = nodes_name[edge['target_name']]['id']


        # Delete unnecessary `edges` keys
        for edge in edges:
            for x in ['source_name', 'target_name', 'source_db_refs', 'target_db_refs']:
                try:
                    del edge[x]
                except:
                    pass


        # Status
        print(f"{num_statements} statements -> {num_statements_processed} processed statements.")
        print(f"Found {num_nodes} nodes and {num_edges} edges.")


    # Process paths
    paths_processed = []
    num_paths = len(paths)
    if num_paths > 0:
        
        # Hash tables
        map_nodes_ids = {node['name']: node['id'] for node in nodes}
        map_edges_ids = {edge['statement_id']: [] for edge in edges}
        for edge in edges:
            map_edges_ids[edge['statement_id']] = map_edges_ids[edge['statement_id']] + [edge['id']]

        # Map node names and statement IDs to node IDs and edge IDs
        paths_processed = [{
            'model_id': model_id, 
            'node_ids': [map_nodes_ids[node_name] for node_name in path['nodes'] if node_name in map_nodes_ids],
            'edge_ids': [map_edges_ids[str(statement_id)] for k in path['edges'] for statement_id in k if str(statement_id) in map_edges_ids],
            'graph_type': path['graph_type']
        } for path in paths]

        # Flatten `edge_ids` in `paths_processed`
        for path in paths_processed:
            path['edge_ids'] = [edge_id  for k in path['edge_ids'] for edge_id in k]

        # Only keep paths with non-empty node and edge lists
        paths_processed = [path for path in paths_processed if (len(path['node_ids']) > 0) & (len(path['edge_ids']) > 0)]
        num_paths_processed = len(paths_processed)

        # Update 'tested' of `edges`
        paths_edge_ids = [edge_id for path in paths_processed for edge_id in path['edge_ids']]
        for edge in edges:
            if edge['id'] in paths_edge_ids:
                edge['tested'] = True

        num_edges_tested = len([True for edge in edges if edge['tested']])


        # Status
        print(f"{num_paths} paths -> {num_paths_processed} processed paths.")
        print(f"Found {num_edges_tested} tested edges.")


    return nodes, edges, statements_processed, paths_processed







# %%
# Generate NetworkX `MultiDiGraph` from `nodes` and `edges`
# Note that parallel edges and self-loops are allowed in this object type
def generate_nx_object(nodes, edges):

    # Generate node list from `nodes`
    node_list = [(node['id'], node) for node in nodes]

    # Generate edge list from `edges`
    edge_list = [(edge['source_id'], edge['target_id'], edge) for edge in edges]


    # Generate NetworkX graph object from node and edge list
    G = nx.MultiDiGraph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    # print(f"{G.number_of_nodes()} nodes and {G.number_of_edges()} edges has been added to the graph object.")

    return G

# %%
# Intersect a set of graph nodes/edges with a set of graph paths
def intersect_graph_paths(nodes, edges, paths):

    # Get the edge IDs
    edgeIDs_edges = set([edge['id'] for edge in edges])
    edgeIDs_paths = set([edge for path in paths for edge in path['edge_ids']])

    # Find intersection between the graph edges and the path edges
    edgeIDs_inter = edgeIDs_paths & edgeIDs_edges
    print(f"{len(edgeIDs_paths)} {len(edgeIDs_edges)} -> {len(edgeIDs_inter)}")

    # Select the edges within the intersection
    edges_inter = [edge for edge in edges if edge['id'] in edgeIDs_inter]
    num_edges_inter = len(edges_inter)

    # Select the paths within the intersection
    paths_inter = [path for path in paths if set(path['edge_ids']) <= edgeIDs_inter]
    num_paths_inter = len(paths_inter)

    # Select the nodes within the intersection
    nodeIDs_inter = set([edge['source_id'] for edge in edges_inter] + [edge['target_id'] for edge in edges_inter])
    nodes_inter = [node for node in nodes if node['id'] in nodeIDs_inter]
    num_nodes_inter = len(nodes_inter)

    # Restrict `edge_ids` in `nodes` to the subgraph
    # Update node degree values
    num_nodes_inter = len(nodes_inter)
    for node in nodes_inter:
        for k, l in zip(['edge_ids_source', 'edge_ids_target'], ['out_degree', 'in_degree']):
            node[k] = list(set(node[k]) & edgeIDs_inter)
            node[l] = len(node[k])

    # Status
    print(f"{len(paths)} paths, {len(nodes)} nodes, and {len(edges)} edges in total.")
    print(f"{num_paths_inter} paths, {num_nodes_inter} nodes, and {num_edges_inter} edges in the intersection.")

    return nodes_inter, edges_inter, paths_inter

# %%
# Reset the IDs in `nodes` and `edges`
def reset_node_edge_ids(nodes, edges):

    # Make new node/edge-to-ID map
    map_nodes_ids = {node['id']: i for i, node in enumerate(nodes)}
    map_edges_ids = {edge['id']: i for i, edge in enumerate(edges)}


    # Reset the node IDs in `nodes`
    for node in nodes:
        node['id'] = map_nodes_ids[node['id']]

    # Reset the node IDs in `edges`
    for edge in edges:
        for k in ['source_id', 'target_id']:
            try:
                edge[k] = map_nodes_ids[edge[k]]
            except:
                print(edge)


    # Reset the edge IDs in `nodes`
    for node in nodes:
        for k in ['edge_ids_source', 'edge_ids_target']:
            node[k] = [map_edges_ids[edge_id] for edge_id in node[k]]

    # Reset the edge IDs in `edges`
    for edge in edges:
        edge['id'] = map_edges_ids[edge['id']]


    return map_nodes_ids, map_edges_ids

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
        i = map_nodes_ids[edge['source_id']]
        j = map_nodes_ids[edge['target_id']]
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
        i = map_nodes_ids[edge['source_id']]
        j = map_nodes_ids[edge['target_id']]
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
def generate_nx_layout(G = None, nodes = None, edges = None, node_list = None, edge_list = None, layout = 'spring', layout_atts = {}, plot = False, plot_atts = {}, ax = None):
    
    if G == None:

        # Generate node list if unavailable
        # node_list = <list of (node, attributes = {k: v})>
        if node_list == None:
            node_list = [(node['id'], {'name': node['name']}) for node in nodes]
            
        # Generate edge list if unavailable
        # edge_list = <list of (source_node, target_node, attributes = {k: v})>
        if edge_list == None:

            # Generate edge list from `edges`
            edge_dict = {(edge['source_id'], edge['target_id']): 0 for edge in edges}
            for edge in edges:
                edge_dict[(edge['source_id'], edge['target_id'])] += 1

            # weight = number of equivalent edges
            edge_list = [(edge[0], edge[1], {'weight': edge_dict[edge]}) for edge in edge_dict]


        # Generate NetworkX graph object from node and edge list
        G = nx.MultiDiGraph()
        if len(node_list) > 0:
            G.add_nodes_from(node_list)

        if len(edge_list) > 0:
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
        plot = False
        coors = {}

    fig = None
    ax = None
    if plot == True:

        if ax == None:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))

        # print(plt.style.available)
        # plt.style.use('fast')
        # plt.style.use('dark_background')

        plot_atts_default = {
            'ax': ax,
            'arrows': G.number_of_nodes() < 20, 
            'with_labels': G.number_of_nodes() < 20,
            'labels': {node: str(node) for node in G.nodes()},
            'node_size': [node[1] + 0.1 for node in G.degree()],
            'width': 0.05,
            'alpha': 0.8,
            'cmap': 'cividis',
            'edge_color': 'k',
            'font_color': 'k'
        }

        nx.draw_networkx(G, pos = coors, **{**plot_atts_default, **plot_atts})
        __ = plt.setp(ax, aspect = 1.0)
    

    return coors, G, fig, ax

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
# Calculate the shortest root-to-leaf paths of model nodes that have been grounded to a given ontological graph
def calculate_onto_root_path(nodes, G_onto_JSON):

    # Load the ontology graph as a `networkx` object
    G_onto = nx.readwrite.json_graph.node_link_graph(G_onto_JSON)

    # Extract components, sorted by size
    ontoSubs = sorted(nx.weakly_connected_components(G_onto), key = len, reverse = True)

    # Find the root nodes of each component (degree = 0 or out-degree = 0)
    ontoSubRoots = [[node for node in sub if G_onto.out_degree(node) < 1] for sub in ontoSubs]


    # Initialize the ontological attributes
    # Unmappable nodes: level = `-1` and to-root list = [`not-grounded-onto`]
    num_nodes = len(nodes)
    for node in nodes:
        if node['db_ref_priority'] in nx.nodes(G_onto):
            node['grounded_onto'] = True
            node['ontocat_level'] = -1
            node['ontocat_refs'] = []
        else:
            node['grounded_onto'] = False
            node['ontocat_level'] = -1
            node['ontocat_refs'] = ['not-grounded-onto']


    # Index of mappable model nodes
    node_indices = [i for i, node in enumerate(nodes) if node['grounded_onto']]

    # Index of the onto subgraph to which the model nodes are mapped
    # (if in nontrivial subgraph -> -1)
    num_ontoSub_nontrivial = sum([1 for sub in ontoSubs if len(sub) > 1])
    x = [{(nodes[i]['db_ref_priority'] in sub): j for j, sub in enumerate(ontoSubs[:num_ontoSub_nontrivial])} for i in node_indices]
    ontoSub_indices = [d[True] if True in d.keys() else -1 for d in x]


    # Calculate shortest path to local root
    for i, j in zip(node_indices, ontoSub_indices):

        source = nodes[i]['db_ref_priority']

        # Case: model node was mapped to either a trivial subgraph or the root of a non-trivial subgraph
        if (j == -1) or (source in ontoSubRoots[j]):
            nodes[i]['ontocat_level'] = 0
            nodes[i]['ontocat_refs'] = [source]

        else:

            z = []
            for target in ontoSubRoots[j]:
                try:
                    p = nx.algorithms.shortest_paths.generic.shortest_path(G_onto.subgraph(ontoSubs[j]), source = source, target = target)
                    z.append(p)
                except:
                    pass
            
            # Find shortest path and reverse such that [target, ..., source]
            z = sorted(z, key = len, reverse = False)
            nodes[i]['ontocat_level'] = len(z[0]) - 1
            nodes[i]['ontocat_refs'] = z[0][::-1]


    # Ensure that identical onto nodes share the same lineage (path to their ancestor) for hierarchical uniqueness
    ontocat_refs = [node['ontocat_refs'] for node in nodes]
    m = max([len(path) for path in ontocat_refs])
    for i in range(1, m):

        # All nodes
        x = [path[i] if len(path) > i else '' for path in ontocat_refs]

        # All unique nodes
        y = list(set(x) - set(['']))

        # Mapping from all nodes to unique nodes
        xy = [y.index(node) if node is not '' else '' for node in x]

        # Choose the path segment of the first matching node for each unique node
        z = [ontocat_refs[x.index(node)][:i] for node in y]
        
        # Substitute path segments
        for j in range(num_nodes):
            if xy[j] is not '':
                nodes[j]['ontocat_refs'][:i] = z[xy[j]]
            else:
                nodes[j]['ontocat_refs'][:i] = ontocat_refs[j][:i]


    # Copy results
    for j in range(num_nodes):
        nodes[j]['ontocat_refs'][:i] = ontocat_refs[j][:i].copy()

# %%
# Extract list of ontological categories that are in the shortest paths of the model nodes
def extract_ontocats(nodes, G_onto_JSON):

    # Generate list of mapped ontology categories, sorted by size
    ontocat_refs = [node['ontocat_refs'] for node in nodes]
    ontocats_ = {}
    ontocats_['ref'], ontocats_['size'] = np.unique([node for path in ontocat_refs for node in path], return_counts = True)

    num_ontocats = len(ontocats_['ref'])
    i = np.argsort(ontocats_['size'])[::-1]
    ontocats_['ref'] = list(ontocats_['ref'][i])
    ontocats_['size'] = [int(k) for k in ontocats_['size']]
    ontocats_['id'] = list(range(num_ontocats))


    # Load the ontology graph as a `networkx` object
    G_onto = nx.readwrite.json_graph.node_link_graph(G_onto_JSON)

    # Get the mapped onto category names
    x = dict(G_onto.nodes(data = 'name', default = None))
    ontocats_['name'] = list(np.empty((num_ontocats, )))
    for i, ref in enumerate(ontocats_['ref']):
        try:
            ontocats_['name'][i] = x[ref]
        except:
            ontocats_['name'][i] = ''


    # Get onto level of each category
    i = max([len(path) for path in ontocat_refs])
    x = [np.unique([path[j] if len(path) > j else '' for path in ontocat_refs]) for j in range(i)]
    ontocats_['level'] = [int(np.flatnonzero([ref in y for y in x])[0]) for ref in ontocats_['ref']]


    # Get numeric id version of ontocat_refs
    x = {k: v for k, v in zip(ontocats_['ref'], ontocats_['id'])}
    ontocat_ids = [[x[node] for node in path] for path in ontocat_refs]
    for node in nodes:
        node['ontocat_ids'] = [x[ontocat] for ontocat in node['ontocat_refs']]


    # Get parent category id for each category (for root nodes, parentID = None)
    y = [np.flatnonzero([True if ref in path else False for path in ontocat_refs])[0] for ref in ontocats_['ref']]
    ontocats_['parent_ref'] = [ontocat_refs[y[i]][ontocat_refs[y[i]].index(ref) - 1] if ontocat_refs[y[i]].index(ref) > 0 else None for i, ref in enumerate(ontocats_['ref'])]
    ontocats_['parent_id'] = [x[parent] if parent is not None else None for parent in ontocats_['parent_ref']]


    # Find membership of onto categories
    ontocats_['node_ids'] = [[node['id'] for node, path in zip(nodes, ontocat_refs) if ref in path] for ref in ontocats_['ref']]


    # Placeholder for hyperedges
    ontocats_['hyperedge_ids'] = [[] for i in range(num_ontocats)]

    # Model ID
    ontocats_['model_id'] = [nodes[0]['model_id'] for i in range(num_ontocats)]

    # Switch to row-wise structure
    ontocats = [{k: ontocats_[k][i] for k in ontocats_.keys()} for i in range(num_ontocats)]


    # # Placeholder for layout coordinates
    # # (use median of the membership)
    # x = {node['id']: i for i, node in enumerate(nodes)}
    # for ontocat in ontocats:
    #     for i in ['x', 'y', 'z']:
    #         ontocat[i] = float(np.median([nodes[x[j]][i] for j in ontocat['node_ids']]))


    return ontocats

# %%
# Generate hyperedges by aggregate model edges that are between children of each ontological category
def generate_hyperedges(nodes, edges, ontocats):

    # Find model edges that have the given ontocat member as their source
    x = [edge['source_id'] for edge in edges]
    ontocats_edges_source = [match_arrays(x, ontocat['node_ids']) for ontocat in ontocats]

    x = [edge['target_id'] for edge in edges]
    ontocats_edges_target = [match_arrays(x, ontocat['node_ids']) for ontocat in ontocats]

    #####################################################

    # Find the onto-category siblings of each onto-category
    # * Generate list of parent IDs
    # * Make lists of onto-categories with matching parent IDs for each onto-category
    # * Remove self from each list
    ontocats_parent = [ontocat['parent_id'] if ontocat['parent_id'] != None else -1 for ontocat in ontocats]
    ontocats_siblings = [[ontocats[j]['id'] for j in np.flatnonzero(match_arrays(ontocats_parent, [ontocat_parent])) if ontocats[j]['id'] != ontocats[i]['id']] for i, ontocat_parent in enumerate(ontocats_parent)]

    #####################################################

    # Get Type-1 Hyperedges

    # Find each set of edges (hyperedges) that has: 
    # * a given onto-category members as the edges' source 
    # * the sibling onto-category members as the edges target
    hyperedges_siblings_ = {}
    hyperedges_siblings_['source_id'] = [ontocat['id'] for i, ontocat in enumerate(ontocats) for sibling_id in ontocats_siblings[i]]
    hyperedges_siblings_['target_id'] = [sibling_id for i, ontocat in enumerate(ontocats) for sibling_id in ontocats_siblings[i]]
    hyperedges_siblings_['edge_indices'] = [np.flatnonzero([ontocats_edges_source[ontocat['id']] & ontocats_edges_target[sibling_id]]) for i, ontocat in enumerate(ontocats) for sibling_id in ontocats_siblings[i]]
    num_hyperedges_siblings_ = len(hyperedges_siblings_['source_id'])

    # Map between list indices and edge ID
    map_edges_ids = {i: edge['id'] for i, edge in enumerate(edges)}
    hyperedges_siblings_['edge_ids'] = [[map_edges_ids[i] for i in edge_ids] for edge_ids in hyperedges_siblings_['edge_indices']]

    hyperedges_siblings_['level'] = [ontocat['level'] for i, ontocat in enumerate(ontocats) for sibling_id in ontocats_siblings[i]]
    hyperedges_siblings_['size'] = [len(edge_ids) for edge_ids in hyperedges_siblings_['edge_indices']]
    hyperedges_siblings_['id'] = list(range(num_hyperedges_siblings_))


    # Specify source and target types
    hyperedges_siblings_['source_type'] = ['ontocat' for i in range(num_hyperedges_siblings_)]
    hyperedges_siblings_['target_type'] = ['ontocat' for i in range(num_hyperedges_siblings_)]


    # Trim empty hyperedges and change to row-wise
    hyperedges_siblings = [{k: hyperedges_siblings_[k][i] for k in hyperedges_siblings_.keys()} for i in range(num_hyperedges_siblings_) if hyperedges_siblings_['size'][i] > 0]
    # num_hyperedges_siblings = len(hyperedges_siblings)

    #####################################################

    # Get Type-2 Hyperedges

    # Find the node membership of the parent of each onto-category that is not in the node membership of the siblings
    x = [[node for sibling_id in ontocats_siblings[i] for node in ontocats[sibling_id]['node_ids']] + ontocat['node_ids'] for i, ontocat in enumerate(ontocats)]
    ontocats_parent_nodes = [list(set(ontocats[ontocat['parent_id']]['node_ids']) - set(x[i])) if ontocat['parent_id'] != None else [] for i, ontocat in enumerate(ontocats)]


    # Find hyperedges that has:
    # * a given onto-category members as the edges' source 
    # * a member of the onto-category parent that is not in any sibling onto-category as the edges' target
    hyperedges_parent_nodes_ = {}
    hyperedges_parent_nodes_['source_id'] = [ontocat['id'] for i, ontocat in enumerate(ontocats) for node_id in ontocats_parent_nodes[i]]
    hyperedges_parent_nodes_['target_id'] = [node_id for i, ontocat in enumerate(ontocats) for node_id in ontocats_parent_nodes[i]]
    num_hyperedges_parent_nodes_ = len(hyperedges_parent_nodes_['source_id'])

    x = [edge['target_id'] for edge in edges]
    hyperedges_parent_nodes_['edge_indices'] = [np.flatnonzero(ontocats_edges_source[ontocat['id']] & match_arrays(x, [node_id])) for i, ontocat in enumerate(ontocats) for node_id in ontocats_parent_nodes[i]]

    # Map between list indices and edge ID
    map_edges_ids = {i: edge['id'] for i, edge in enumerate(edges)}
    hyperedges_parent_nodes_['edge_ids'] = [[map_edges_ids[i] for i in edge_ids] for edge_ids in hyperedges_parent_nodes_['edge_indices']]

    hyperedges_parent_nodes_['level'] = [ontocat['level'] for i, ontocat in enumerate(ontocats) for node_id in ontocats_parent_nodes[i]]
    hyperedges_parent_nodes_['size'] = [len(edge_ids) for edge_ids in hyperedges_parent_nodes_['edge_ids']]
    hyperedges_parent_nodes_['id'] = list(range(len(hyperedges_parent_nodes_['source_id'])))


    # Specify source and target types
    hyperedges_parent_nodes_['source_type'] = ['ontocat' for i in range(num_hyperedges_parent_nodes_)]
    hyperedges_parent_nodes_['target_type'] = ['node' for i in range(num_hyperedges_parent_nodes_)]


    # Trim empty hyperedges and change to row-wise
    hyperedges_parent_nodes = [{k: hyperedges_parent_nodes_[k][i] for k in hyperedges_parent_nodes_.keys()} for i in range(num_hyperedges_parent_nodes_) if hyperedges_parent_nodes_['size'][i] > 0]
    # num_hyperedges_parent_nodes = len(hyperedges_parent_nodes)

    #####################################################

    # Children that are onto-categories
    ontocats_children_ontocat_ids = {ontocat['id']: [] for ontocat in ontocats}
    for ontocat, siblings in zip(ontocats, ontocats_siblings):
        if ontocat['parent_id'] != None:
            if len(ontocats_children_ontocat_ids[ontocat['parent_id']]) < 1:
                ontocats_children_ontocat_ids[ontocat['parent_id']] = siblings + [ontocat['id']]


    # Children that are model nodes that are not in the membership of any children onto-category
    # Filled from children to parent
    ontocats_children_node_ids = {ontocat['id']: set() for ontocat in ontocats}
    for ontocat, parent_nodes in zip(ontocats, ontocats_parent_nodes):
        if ontocat['parent_id'] != None:
            ontocats_children_node_ids[ontocat['parent_id']] = ontocats_children_node_ids[ontocat['parent_id']] | set(parent_nodes)

    # Add children model nodes to onto-categories that have no children onto-categories from which to get node IDs
    for node in nodes:
        ontocats_children_node_ids[node['ontocat_ids'][-1]] = ontocats_children_node_ids[node['ontocat_ids'][-1]] | set([node['id']])


    # Flatten dict
    ontocats_children_ontocat_ids = [list(ontocats_children_ontocat_ids[ontocat['id']]) for ontocat in ontocats]
    ontocats_children_node_ids = [list(ontocats_children_node_ids[ontocat['id']]) for ontocat in ontocats]


    # Onto-category siblings of given model nodes
    x = set([j for i in [node_ids for node_ids in ontocats_children_node_ids] for j in i])
    nodes_siblings_ontocat_ids = {i: [] for i in x}
    for ontocat_ids, node_ids in zip(ontocats_children_ontocat_ids, ontocats_children_node_ids):
        for i in node_ids:
            nodes_siblings_ontocat_ids[i] = ontocat_ids

    #####################################################

    # Get Type-3 Hyperedges

    # Find hyperedges that has:
    # * a member of the onto-category parent that is not in any sibling onto-category as the edges' source
    # * a onto-category sibling of that model node as the edges' target
    hyperedges_nodes_ontocats_ = {}
    hyperedges_nodes_ontocats_['source_id'] = [node_id for node_id, ontocat_ids in nodes_siblings_ontocat_ids.items() for __ in ontocat_ids]
    hyperedges_nodes_ontocats_['target_id'] = [ontocat_id for __, ontocat_ids in nodes_siblings_ontocat_ids.items() for ontocat_id in ontocat_ids]
    num_hyperedges_nodes_ontocats_ = len(hyperedges_nodes_ontocats_['source_id'])

    x = [edge['source_id'] for edge in edges]
    z = {ontocat['id']: i for i, ontocat in enumerate(ontocats)}
    hyperedges_nodes_ontocats_['edge_indices'] = [np.flatnonzero(match_arrays(x, [node_id]) & ontocats_edges_target[z[ontocat_id]]) for node_id, ontocat_ids in nodes_siblings_ontocat_ids.items() for ontocat_id in ontocat_ids]

    # Map between list indices and edge ID
    map_edges_ids = {i: edge['id'] for i, edge in enumerate(edges)}
    hyperedges_nodes_ontocats_['edge_ids'] = [[map_edges_ids[i] for i in edge_ids] for edge_ids in hyperedges_nodes_ontocats_['edge_indices']]

    # Note: hyperedge_level = node_level + 1 because the level of the model node is that of the parent onto-category
    z = {node['id']: i for i, node in enumerate(nodes)}
    hyperedges_nodes_ontocats_['level'] = [nodes[z[node_id]]['ontocat_level'] + 1 for node_id, ontocat_ids in nodes_siblings_ontocat_ids.items() for __ in ontocat_ids]
    hyperedges_nodes_ontocats_['size'] = [len(edge_ids) for edge_ids in hyperedges_nodes_ontocats_['edge_ids']]
    hyperedges_nodes_ontocats_['id'] = list(range(len(hyperedges_nodes_ontocats_['source_id'])))


    # Specify source and target types
    hyperedges_nodes_ontocats_['source_type'] = ['node' for i in range(num_hyperedges_nodes_ontocats_)]
    hyperedges_nodes_ontocats_['target_type'] = ['ontocat' for i in range(num_hyperedges_nodes_ontocats_)]


    # Trim empty hyperedges and change to row-wise
    hyperedges_nodes_ontocats = [{k: hyperedges_nodes_ontocats_[k][i] for k in hyperedges_nodes_ontocats_.keys()} for i in range(num_hyperedges_nodes_ontocats_) if hyperedges_nodes_ontocats_['size'][i] > 0]
    # num_hyperedges_nodes_ontocats = len(hyperedges_nodes_ontocats)

    #####################################################

    # Get Type-4 Hyperedges

    # Find hyperedges that has:
    # * a model node that is a member of a onto-category that is not in any child onto-category as the edges' source
    # * another other such member as the edges' target

    hyperedges_nodes_ = {}
    hyperedges_nodes_['source_id'] = [node_id_source for ids in ontocats_children_node_ids for node_id_source in ids for node_id_target in ids if node_id_target != node_id_source]
    hyperedges_nodes_['target_id'] = [node_id_target for ids in ontocats_children_node_ids for node_id_source in ids for node_id_target in ids if node_id_target != node_id_source]
    num_hyperedges_nodes_ = len(hyperedges_nodes_['source_id'])

    # Find edges between two such nodes
    x = [edge['source_id'] for edge in edges]
    y = [edge['target_id'] for edge in edges]
    hyperedges_nodes_['edge_indices'] = [np.flatnonzero(match_arrays(x, [node_id_source]) & match_arrays(y, [node_id_target])) for ids in ontocats_children_node_ids for node_id_source in ids for node_id_target in ids if node_id_target != node_id_source]

    # Map between list indices and edge ID
    map_edges_ids = {i: edge['id'] for i, edge in enumerate(edges)}
    hyperedges_nodes_['edge_ids'] = [[map_edges_ids[i] for i in edge_ids] for edge_ids in hyperedges_nodes_['edge_indices']]

    # Note: hyperedge_level = node_level + 1 because the level of the model node is that of the parent onto-category
    z = {node['id']: i for i, node in enumerate(nodes)}
    hyperedges_nodes_['level'] = [nodes[z[node_id_source]]['ontocat_level'] + 1 for ids in ontocats_children_node_ids for node_id_source in ids for node_id_target in ids if node_id_target != node_id_source]
    hyperedges_nodes_['size'] = [len(edge_ids) for edge_ids in hyperedges_nodes_['edge_ids']]
    hyperedges_nodes_['id'] = list(range(num_hyperedges_nodes_))


    # Specify source and target types
    hyperedges_nodes_['source_type'] = ['node' for i in range(num_hyperedges_nodes_)]
    hyperedges_nodes_['target_type'] = ['node' for i in range(num_hyperedges_nodes_)]


    # Trim empty hyperedges and change to row-wise
    hyperedges_nodes = [{k: hyperedges_nodes_[k][i] for k in hyperedges_nodes_.keys()} for i in range(num_hyperedges_nodes_) if hyperedges_nodes_['size'][i] > 0]
    # num_hyperedges_nodes = len(hyperedges_nodes)

    #####################################################

    # Concatenate all hyperedges together
    hyperedges = hyperedges_siblings + hyperedges_parent_nodes + hyperedges_nodes_ontocats + hyperedges_nodes
    # num_hyperedges = len(hyperedges)
    for i, hyperedge in enumerate(hyperedges):
        hyperedge['id'] = i
        hyperedge['model_id'] = nodes[0]['model_id']

    #####################################################

    # Add child list to `ontocats`
    for ontocat in ontocats:
        ontocat['children_ids'] = ontocats_children_ontocat_ids[ontocat['id']]
        ontocat['node_ids_direct'] = ontocats_children_node_ids[ontocat['id']]

    #####################################################

    # Find all hyperedges that are within each given onto-category
    map_ids_nodes = {node['id']: i for i, node in enumerate(nodes)}
    x = {ontocat['id']: [] for ontocat in ontocats}
    for hyperedge in hyperedges:

        if (hyperedge['source_type'] == 'ontocat'):
            if (ontocats[hyperedge['source_id']]['parent_id'] != None):
                x[ontocats[hyperedge['source_id']]['parent_id']] = x[ontocats[hyperedge['source_id']]['parent_id']] + [hyperedge['id']]
        
        if hyperedge['source_type'] == 'node':
            x[nodes[map_ids_nodes[hyperedge['source_id']]]['ontocat_ids'][-1]] = x[nodes[map_ids_nodes[hyperedge['source_id']]]['ontocat_ids'][-1]] + [hyperedge['id']]


    # Add to `ontocats`
    for ontocat in ontocats:
        ontocat['hyperedge_ids'] = x[ontocat['id']]

    #####################################################

    return hyperedges

# %%
# Generate graph layout based on the ontological categories and hyperedges
def generate_onto_layout(nodes, ontocats, hyperedges, plot = False, ax = None):

    # Reset all coordinates to None
    for x in [nodes, ontocats]:
        for y in x:
            for k in ['x', 'y', 'z']:
                y[k] = None


    # Hash tables
    map_ids_nodes = {node['id']: i for i, node in enumerate(nodes)}
    map_ids_ontocats = {ontocat['id']: i for i, ontocat in enumerate(ontocats)}
    map_ids_hyperedges = {hyperedge['id']: i for i, hyperedge in enumerate(hyperedges)}


    # For each level and each parent onto-categories, generate the layout of their children
    max_level = max([ontocat['level'] for ontocat in ontocats])
    coors = [[] for l in range(max_level)]
    G = [[] for l in range(max_level)]

    for l in range(max_level):

        if l == 0:

            node_list = [('ontocat_' + str(ontocat['id']), {'name': ontocat['name']}) for ontocat in ontocats if (ontocat['level'] == l)]

            H = [hyperedge for hyperedge in hyperedges if hyperedge['level'] == l]
            edge_list = [(h['source_type'] + '_' + str(h['source_id']), h['target_type'] + '_' + str(h['target_id']), {'weight': h['size']}) for h in H]

            layout_atts = {
                'k': 1.0,
                'center': (0, 0),
                'scale': 1.0
            }
            coors_, G_, __, __ = generate_nx_layout(node_list = node_list, edge_list = edge_list, layout = 'spring', layout_atts = layout_atts, plot = False)


            # Put coordinates in `ontocats` and `nodes`
            for name in coors_:
                t = re.findall('[a-z]+', name)[0]
                i = int(re.findall('\d+', name)[0])

                if t == 'ontocat':
                    ontocats[map_ids_ontocats[i]]['x'] = float(coors_[name][0])
                    ontocats[map_ids_ontocats[i]]['y'] = float(coors_[name][1])
                    ontocats[map_ids_ontocats[i]]['z'] = float(0.0)
                
                if t == 'node':
                    nodes[map_ids_nodes[i]]['x'] = float(coors_[name][0])
                    nodes[map_ids_nodes[i]]['y'] = float(coors_[name][1])
                    nodes[map_ids_nodes[i]]['z'] = float(0.0)


            # Combine the coordinate dicts and graph objects
            coors[l].append(coors_)
            G[l].append(G_)

        else:

            ontocats_parent = [ontocat for ontocat in ontocats if (ontocat['level'] == l - 1)]

            for ontocat_parent in ontocats_parent:

                node_list = [('ontocat_' + str(ontocat_id), {'name': ontocats[map_ids_ontocats[ontocat_id]]['name']}) for ontocat_id in ontocat_parent['children_ids']]
                node_list = node_list + [('node_' + str(node_id), {'name': nodes[map_ids_nodes[node_id]]['name']}) for node_id in ontocat_parent['node_ids_direct']]
                
                H = [hyperedges[map_ids_hyperedges[h]] for h in ontocat_parent['hyperedge_ids']]
                edge_list = [(h['source_type'] + '_' + str(h['source_id']), h['target_type'] + '_' + str(h['target_id']), {'weight': h['size']}) for h in H]
                

                layout_atts = {
                    'k': 2.0,
                    'center': (0, 0),
                    'scale': 1.0
                }
                coors_, G_, __, __ = generate_nx_layout(node_list = node_list, edge_list = edge_list, layout = 'spring', layout_atts = layout_atts, plot = False)


                # Rescale to parent size and shift to parent centre
                radius = 0.01 ** l
                coor_parent = np.asarray([ontocat_parent['x'], ontocat_parent['y']])
                # coor_parent = np.asarray([ontocats[0]['x'], ontocats[0]['y']])
                coors_ = {name: radius * c + coor_parent for name, c in coors_.items()}

                # Put coordinates in `ontocats` and `nodes_`
                for name in coors_:
                    t = re.findall('[a-z]+', name)[0]
                    i = int(re.findall('\d+', name)[0])

                    if t == 'ontocat':
                        ontocats[map_ids_ontocats[i]]['x'] = float(coors_[name][0])
                        ontocats[map_ids_ontocats[i]]['y'] = float(coors_[name][1])
                        ontocats[map_ids_ontocats[i]]['z'] = float(0.0)
                    
                    if t == 'node':
                        nodes[map_ids_nodes[i]]['x'] = float(coors_[name][0])
                        nodes[map_ids_nodes[i]]['y'] = float(coors_[name][1])
                        nodes[map_ids_nodes[i]]['z'] = float(0.0)


                # Combine the coordinate dicts and graph objects
                coors[l].append(coors_)
                G[l].append(G_)


    # Plot results
    fig = None
    ax = None
    if plot == True:

        if ax == None:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
            __ = plt.setp(ax, aspect = 1.0, title = 'Ontology-Based Graph Layout')
            ax.grid(True)
            j = 1.05
            __ = plt.setp(ax, xlim = (-j, j), ylim = (-j, j), aspect = 1.0)


            # print(plt.style.available)
            plt.style.use('default')
            # plt.style.use('dark_background')


        # Plot nodes and ontocats
        for l in range(len(G)):

            G_all = nx.algorithms.operators.all.union_all(G[l])
            coors_all = {}
            for c in coors[l]:
                coors_all = {**coors_all, **c}

            # Get coordinates
            node = [[coors_all[name][k] for name in coors_all if re.findall('[a-z]+', name)[0] == 'node'] for k in [0, 1]]
            ontocat = [[coors_all[name][k] for name in coors_all if re.findall('[a-z]+', name)[0] == 'ontocat'] for k in [0, 1]]

            if l == 0:
                ax.scatter(node[0], node[1], s = 1, alpha = 1, c = 'k', zorder = 100, label = 'Model Nodes')

                name = [nx.get_node_attributes(G_all, 'name')[name] for name in coors_all if re.findall('[a-z]+', name)[0] == 'ontocat']
                __ = [ax.text(ontocat[0][i], ontocat[1][i], '   ' + j, fontsize = 7, va = 'center', alpha = 1.0, color = plt.get_cmap('tab10').colors[l % 10]) for i, j in enumerate(name) if j != None]
            else:
                ax.scatter(node[0], node[1], s = 1, alpha = 1, c = 'k', zorder = 100)

            if l < 4:
                ax.scatter(ontocat[0], ontocat[1], s = 500.0 / (l + 1.0) ** 3, alpha = 1.0 - 1.0 / (l + 1.0) + 0.2, color = plt.get_cmap('tab10').colors[l % 10], zorder = l, label = 'Onto-Category in Level ' + str(l))


        # Plot hyperedges
        # Levels 0 and 1 only
        for l in range(2):

            G_all = nx.algorithms.operators.all.union_all(G[l])
            coors_all = {}
            for c in coors[l]:
                coors_all = {**coors_all, **c}

            for hyperedge in G_all.edges():

                x = [coors_all[hyperedge[k]][0] for k in range(2)]
                y = [coors_all[hyperedge[k]][1] for k in range(2)]
                ax.plot(x, y, linewidth = 0.05, color = plt.get_cmap('tab10').colors[l % 10], zorder = l)

        __ = ax.legend()


    return G, coors, fig, ax

# %%
# Generate node, node-attribute, node-layout lists from a set of metadata, coordinates, labels
def generate_nodelist(model_id = -1, node_metadata = [], node_coors = [], node_labels = []):

    # Check types
    if (len(node_metadata) > 0) & (not isinstance(node_metadata[0], dict)):
        raise TypeError("'node_metadata' must be a list of dicts.")
    if (len(node_coors) > 0) & ((not isinstance(node_coors, np.ndarray)) or (len(node_coors.shape) != 2)):
        raise TypeError("'node_coors' must be a 2D numpy ndarray.")
    if (len(node_labels) > 0) & (not isinstance(node_labels, np.ndarray)):
        raise TypeError("'node_coors' must be a numpy ndarray.")

    # Get number of nodes
    if len(node_metadata) > 0:
        num_nodes = len(node_metadata)
    elif len(node_coors) > 0:
        num_nodes = len(node_coors)
    elif len(node_labels) > 0:
        num_nodes = len(node_labels)
    else:
        num_nodes = 0

    # Initialize
    nodes = []
    nodeLayout = []
    nodeAtts = []


    if num_nodes > 0:

        # Initialize the node list
        nodes = [{
            'model_id': model_id,
            'id': i,
            'name': None,
            'db_refs': {},
            'grounded': False,
            'edge_ids_source': [],
            'edge_ids_target': [],
            'out_degree': 0,
            'in_degree:': 0,
        } for i in range(num_nodes)]

        # Add metadata if available
        if len(node_metadata) > 0:

            for node, metadata in zip(nodes, node_metadata):

                node['grounded'] = True
                node['name'] = metadata['title']

                if 'doi' in metadata.keys(): 
                    node['db_refs']['DOI'] = metadata['doi'].upper()

                if 'pmcid' in metadata.keys():
                    node['db_refs']['PMCID'] = metadata['pmcid'].upper()

                if 'pubmed_id' in metadata.keys(): 
                    node['db_refs']['PMID'] = metadata['pubmed_id'].upper()

        # Generate node-layout list
        nodeLayout = [{
            'model_id': model_id,
            'id': i,
            'x': None,
            'y': None,
            'z': None,
        } for i in range(num_nodes)]

        # Get coordinate data if available
        if len(node_coors) > 0:

            # Map coordinate data
            num_dim_coors = node_coors.shape[1]
            c = {0: 'x', 1: 'y', 2: 'z'}
            for i, node in enumerate(nodeLayout):
                for j in range(3):
                    if j < num_dim_coors:
                        node[c[j]] = node_coors[i, j].item()
                    else:
                        node[c[j]] = 0.0

        # Generate node-attribute list
        nodeAtts = [{
            'model_id': model_id,
            'id': i,
            "db_ref_priority": None, 
            "grounded_onto": False, 
            "ontocat_level": None, 
            "ontocat_ids": None, 
            "grounded_cluster": False, 
            "cluster_level": None, 
            "cluster_ids": None
        } for i in range(num_nodes)]

        # Get priority DB reference (DOI)
        if len(node_metadata) > 0:
            for i in range(num_nodes):
                nodeAtts[i]['db_ref_priority'] = nodes[i]['db_refs']['DOI']

        # Get cluster label data if available
        if len(node_labels) > 0:

            # Check axes
            if len(node_labels.shape) < 2:
                node_labels = node_labels[:, np.newaxis]
            num_dim_labels = node_labels.shape[1]

            for i, node in enumerate(nodeAtts):
                
                node['cluster_ids'] = [l.item() for l in node_labels[i, :]]

                # Cluster level = last level at which the cluster label is not -1
                j = np.flatnonzero(node_labels[i, :] == -1)
                k = 0
                if len(j) != 0:
                    k = (j[0] - 1).item()
                node['cluster_level'] = k

                if k >= 0:
                    node['grounded_cluster'] = True

                
    return nodes, nodeLayout, nodeAtts


# %%
# Generate a node list, edge list, and nearest-neighbour graph from a set of coordinates
def generate_nn_graph(node_coors, node_metadata = [], model_id = -1):

    # Define custom Minkowski distance function to enable non-integer `p`
    @numba.njit
    def minkowski_distance(u, v, p = 2.0):
        return (np.abs(u - v) ** p).sum() ** (1.0 / p)
        # return np.sum(np.abs(u - v) ** p) ** (1.0 / p)



    # Find k-nearest neighbours
    # knn = skl.neighbors.NearestNeighbors(n_neighbors = 1, metric = 'minkowski', p = 2.0 / 3.0)
    knn = skl.neighbors.NearestNeighbors(n_neighbors = 2, metric = lambda u, v: minkowski_distance(u, v, p = 2.0 / 3.0))
    knn.fit(node_coors)
    knn_dist, knn_ind = knn.kneighbors(node_coors)


    # Define edge list
    num_coors = node_coors.shape[0]
    edges = [{
        'model_id': model_id,
        'id': i,
        'type': 'knn',
        'belief': float(1.0 / knn_dist[i][1]),
        'statement_id': None,
        'source_id': i,
        'target_id': int(knn_ind[i][1]),
        'tested': True
    } for i in range(num_coors)]


    # Define NetworkX graph object
    edge_list = [(edge['source_id'], edge['target_id'], edge) for edge in edges]
    G = nx.MultiDiGraph()
    G.add_edges_from(edge_list)


    # Define node list
    num_coors = node_coors.shape[0]
    nodes = []
    nodes = [{
        'model_id': model_id,
        'id': i,
        'name': None,
        'db_refs': None,
        'grounded': False,
        'edge_ids_source': [i],
        'edge_ids_target': [j for j, k in G.in_edges(i)],
        'out_degree': G.out_degree(i),
        'in_degree:': G.in_degree(i),
    } for i in range(num_coors)]


    # Add metadata if available
    if len(node_metadata) == num_coors:

        for i, node in enumerate(nodes):
            node['grounded'] = True
            node['name'] = node_metadata[i]['title']
            node['db_refs'] = {
                'DOI': node_metadata[i]['doi'].upper(), 
                'PMCID': node_metadata[i]['pmcid'].upper(), 
                'PMID': node_metadata[i]['pubmed_id'].upper()
            }


    return nodes, edges, G

# %%
# Generate nearest-neighbour centroid list from a set of coordinates and cluster labels
def generate_nn_cluster_centroid_list(coors, labels = [], p = 2):

    # Error handling
    if not isinstance(coors, np.ndarray):
        raise TypeError("'coor' must be an numpy ndarray.")
    if not ((isinstance(labels, list) | isinstance(labels, np.ndarray)) and (len(labels) in [0, coors.shape[0]])): 
        raise TypeError("'labels' must be either [] or a N x 1 list or numpy ndarrray.")


    # Dimensions
    num_coors, num_dim = coors.shape


    # Assume no label = identically zeros
    if len(labels) == 0:
        labels = np.zeros((num_coors, ))

    labels_unique = np.unique(labels)
    num_unique = len(labels_unique)


    # Calculate centroid coordinates
    coors_centroid = np.empty((num_unique, num_dim))
    for i in range(num_unique):
        coors_centroid[i, :] = np.nanmedian(coors[labels == labels_unique[i], :], axis = 0)


    # Choose kNN metric
    if isinstance(p, int) & (p >= 1):
        knn = skl.neighbors.NearestNeighbors(n_neighbors = 1, metric = 'minkowski', p = p)

    else:

        # Define custom Minkowski distance function to enable non-integer `p`
        @numba.njit
        def minkowski_distance(u, v, p):
            return (np.abs(u - v) ** p).sum() ** (1.0 / p)

        knn = skl.neighbors.NearestNeighbors(n_neighbors = 2, metric = lambda u, v: minkowski_distance(u, v, p = p))
    
    
    # Find index of k-nearest neighbour to the cluster centroids
    for i in range(num_unique):
        knn.fit(coors[labels == labels_unique[i], :])
        knn_ind = knn.kneighbors(coors_centroid, return_distance = False)
        knn_ind = np.squeeze(knn_ind)


    return knn_ind, labels_unique, coors_centroid


# %%
# Generate node, node-attribute, node-layout lists from a set of BibJSON metadata, coordinates, labels
def generate_nodelist_bibjson(model_id = -1, node_metadata = [], node_coors = [], node_labels = []):

    # Check types
    if (len(node_metadata) > 0) & (not isinstance(node_metadata[0], dict)):
        raise TypeError("'node_metadata' must be a list of dicts.")
    if (len(node_coors) > 0) & ((not isinstance(node_coors, np.ndarray)) or (len(node_coors.shape) != 2)):
        raise TypeError("'node_coors' must be a 2D numpy ndarray.")
    if (len(node_labels) > 0) & (not isinstance(node_labels, np.ndarray)):
        raise TypeError("'node_labels' must be a numpy ndarray.")

    # Get number of nodes
    if len(node_metadata) > 0:
        num_nodes = len(node_metadata)
    elif len(node_coors) > 0:
        num_nodes = len(node_coors)
    elif len(node_labels) > 0:
        num_nodes = len(node_labels)
    else:
        num_nodes = 0

    # Initialize
    nodes = []
    nodeLayout = []
    nodeAtts = []


    if num_nodes > 0:

        # Initialize the node list
        nodes = [{
            'model_id': model_id,
            'id': i,
            'name': None,
            'db_refs': {},
            'grounded': False,
            'edge_ids_source': [],
            'edge_ids_target': [],
            'out_degree': 0,
            'in_degree': 0,
        } for i in range(num_nodes)]

        # Add metadata if available
        if len(node_metadata) > 0:

            for node, metadata in zip(nodes, node_metadata):

                node['name'] = metadata['title']

                if (metadata['identifier'] != None) | ((isinstance(metadata['identifier'], list)) & (len(metadata['identifier']) > 0)):
                    
                    node['grounded'] = True
                    
                    for identifier in metadata['identifier']:

                        node['db_refs'][identifier['type']] = identifier['id']

        # Generate node-layout list
        nodeLayout = [{
            'model_id': model_id,
            'id': i,
            'x': None,
            'y': None,
            'z': None,
        } for i in range(num_nodes)]

        # Get coordinate data if available
        if len(node_coors) > 0:

            # Map coordinate data
            num_dim_coors = node_coors.shape[1]
            c = {0: 'x', 1: 'y', 2: 'z'}
            for i, node in enumerate(nodeLayout):
                for j in range(3):
                    if j < num_dim_coors:
                        node[c[j]] = node_coors[i, j].item()
                    else:
                        node[c[j]] = 0.0

        # Generate node-attribute list
        nodeAtts = [{
            'model_id': model_id,
            'id': i,
            "db_ref_priority": None, 
            "grounded_onto": False, 
            "ontocat_level": None, 
            "ontocat_ids": None, 
            "grounded_cluster": False, 
            "cluster_level": None, 
            "cluster_ids": None
        } for i in range(num_nodes)]


        # Get cluster label data if available
        if len(node_labels) > 0:

            # Check axes
            if len(node_labels.shape) < 2:
                node_labels = node_labels[:, np.newaxis]
            num_dim_labels = node_labels.shape[1]

            # Ensure type in case of boolean labels
            node_labels = node_labels.astype('int')


            for i, node in enumerate(nodeAtts):
                
                node['cluster_ids'] = [l.item() for l in node_labels[i, :]]

                # Cluster level = last level at which the cluster label is not -1
                j = np.flatnonzero(node_labels[i, :] == -1)
                k = 0
                if len(j) != 0:
                    k = (j[0] - 1).item()
                node['cluster_level'] = k

                if k >= 0:
                    node['grounded_cluster'] = True

                
    return nodes, nodeLayout, nodeAtts

