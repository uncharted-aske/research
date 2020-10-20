# %% [markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %% [markdown]
# ## Import required modules.

import sys
from time import time
import numpy as np
import scipy as sp
import csv
import re

import spacy
import nltk
import squarify as sqf
import sklearn as skl
import hdbscan

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cmocean


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
    # r0 = np.median(coor, axis = 0)
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
