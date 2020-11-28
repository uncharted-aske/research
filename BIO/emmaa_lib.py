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
        in_degree[i] += 1
        out_degree[j] += 1

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
# Generate an ordered list of namespaces present in the priority list, the graph nodes, and the given ontology (JSON format)
def generate_ordered_namespace_list(nodes, ontoJSON, namespaces_priority):

    # Generate namespace list of the given ontology
    x = [re.findall('\w{1,}(?=:)', node['id'])[0] for node in ontoJSON['nodes']]
    y, z = np.unique(x, return_counts = True)
    i = np.argsort(z)[::-1]
    namespaces_onto = [[name, count / np.sum(z) * 100] for name, count in zip(y[i], z[i])]


    # Generate namespace list of the node list
    x = []
    for node in nodes:

        if len(node['info']['links']) > 0:
            names = [link[0] for link in node['info']['links']]
        else:
            names = ['not-grounded']
        
        # Check against priority list
        y = np.flatnonzero([True if name in names else False for name in namespaces_priority])
        if len(y) > 0:
            i = y[0]
            x.append(namespaces_priority[i])
        else:

            # Check against ontology list
            z = np.flatnonzero([True if name[0] in names else False for name in namespaces_onto])
            if len(z) > 0:
                i = z[0]
                x.append(namespaces_onto[i][0])
            else:
                x.append(names[0])


    # Count node namespace list
    y, z = np.unique(x, return_counts = True)
    i = np.argsort(z)[::-1]
    namespaces_nodes = [[name, count / np.sum(z) * 100] for name, count in zip(y, z)]


    # Combine namespace lists in order
    # 1. Given priority
    # 2. extra from node list
    # 3. extra from onto list
    x = list(set([name for name, __ in namespaces_nodes]) - set(namespaces_priority))
    y = list(set([name for name, __ in namespaces_onto]) - set(namespaces_priority) - set([name for name, __ in namespaces_nodes]))
    namespaces_combined = namespaces_priority + x + y


    return namespaces_combined

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

