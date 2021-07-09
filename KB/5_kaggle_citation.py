# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Build citation network of the Kaggle CORD19 dataset

# %%
import time
from tqdm import tqdm
import json
import csv
import re
import pickle
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import umap
import sklearn as skl
import hdbscan
from sklearn.cluster import KMeans
# from sklearn.cluster import MiniBatchKMeans
import importlib

import emmaa_lib as emlib


# %%
np.random.seed(0)

# %%[markdown]
# # Load Metadata

# %%

data_dir = './data/kaggle/2021-06-29/'

metadata = []
with open(data_dir  + 'metadata.csv') as f:
    metadata.extend([row for row in csv.DictReader(f)])


# Drop 'abstract'
__ = [m.pop('abstract', None) for m in metadata]


f = None
del f


# %%[markdown]
# ## Check for duplication amongst titles

# %%
pattern = re.compile(r'\W+')
map_titles_uids = {pattern.sub(' ', m['title'].lower()): m['cord_uid'] for m in metadata}

titles = {t: [] for t in map_titles_uids.keys()}
titles_uids = {t: set() for t in map_titles_uids.keys()}

for i, m in enumerate(metadata):

    j = pattern.sub(' ', m['title'].lower())

    titles[j].append(i)
    titles_uids[j].add(m['cord_uid'])


i = j = m = None
del i, j, m

#%%

x = np.array([len(v) for __, v in titles.items()])
y = np.array([len(v) for __, v in titles_uids.items()])
s = np.array([len(k) for k in titles.keys()])


n = np.random.randint(0, len(titles), size = 10000)
m = max((max(x), max(y)))


fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12))

__ = ax[0, 0].scatter(x[n], y[n], s = s[n] + 1, marker = '.', alpha = 0.2)
__ = ax[0, 0].plot([1, m], [1, m], alpha = 0.5, color = 'k', linestyle = '--')
__ = plt.setp(ax[0, 0], xscale = 'log', yscale = 'log', xlabel = 'Number of Title Duplicate', ylabel = 'Number of CORD-UID Duplicate', )

__ = ax[0, 1].hist(x, bins = np.arange(0, m, 1), alpha = 0.2, label = 'Title')
__ = ax[0, 1].hist(y, bins = np.arange(0, m, 1), alpha = 0.2, label = 'CORD-UID')
__ = plt.setp(ax[0, 1], xscale = 'log', yscale = 'log', xlabel = 'Number of Duplicates', ylabel = 'Count', xlim = plt.getp(ax[0, 0], 'xlim'))
__ = ax[0, 1].legend()

__ = ax[1, 0].scatter(s[n], x[n], s = 50, marker = '.', alpha = 0.2, label = 'Title')
__ = ax[1, 0].scatter(s[n], y[n], s = 50, marker = '.', alpha = 0.2, label = 'CORD-UID')
__ = plt.setp(ax[1, 0], xscale = 'linear', yscale = 'log', xlabel = 'Length of Title Strings', ylabel = 'Number of Duplicates', ylim = plt.getp(ax[0, 0], 'ylim'))
__ = ax[1, 0].legend()
__ = ax[1, 0].plot([0, max(s)], [0.1, 0.1], alpha = 0.5, color = 'k', linestyle = '--')

__ = ax[1, 1].hist(s, bins = np.arange(0, max(s), 1), alpha = 0.2)
__ = plt.setp(ax[1, 1], xscale = 'linear', yscale = 'log', xlabel = 'Length of Title String', ylabel = 'Count', xlim = plt.getp(ax[1, 0], 'xlim'))


x = y = s = m = n = fig = ax = None
del x, y, s, m, n, fig, ax

# %%
z = sorted(titles.items(), key = lambda d: len(d[1]), reverse = True)

n = 30
print(f"{'Rank':>6}   {'Title':<50}   {'Number of Duplicates':>20}   {'Number of CORD UIDs':>20}")
__ = [print(f"{i:>6}   {k[0].__repr__()[:50]:<50}   {len(k[1]):>20}   {len(titles_uids[k[0]]):>20}") for i, k in enumerate(z[:n])]
print(f"{'...':>6}")
__ = [print(f"{len(z) - (5 - i):>6}   {(k[0].__repr__()[:50]):<50}   {len(k[1]):>20}   {len(titles_uids[k[0]]):>20}") for i, k in enumerate(z[-5:])]


n = None
del n

# %%
# Rank   Title                                                Number of Duplicates    Number of CORD UIDs
#      0   ''                                                                    308                    306
#      1   'reply'                                                               194                     42
#      2   'department of error'                                                 115                     51
#      3   'acr convergence 2020 abstract supplement'                             99                      1
#      4   'corrigendum'                                                          82                     11
#      5   'erratum'                                                              68                     20
#      6   'correction'                                                           66                     12
#      7   'the authors reply'                                                    60                      8
#      8   'guest editorial'                                                      57                      4
#      9   'response'                                                             56                     21
#     10   'news'                                                                 56                     42
#     11   'ueg week 2020 poster presentations'                                   55                      1
#     12   'news in brief'                                                        53                     50
#     13   'in response'                                                          47                     12
#     14   'reply '                                                               46                     46
#     15   'covid 19'                                                             43                     14
#     16   'authors response'                                                     43                     20
#     17   'conclusion'                                                           40                     37
#     18   'corrigendum '                                                         40                     40
#     19   'highlights from this issue'                                           38                      3
#     20   'european association of nuclear medicine october                      38                      2
#     21   'letter to the editor'                                                 37                     17
#     22   'panorama'                                                             35                     35
#     23   'editorial '                                                           35                     35
#     24   'editor s note'                                                        34                      6
#     25   'mitteilungen des bdi'                                                 32                     30
#     26   'authors reply'                                                        32                     10
#     27   'in reply'                                                             31                      6
#     28   'hiv glasgow virtual 5 8 october 2020'                                 31                      1
#     29   'public health round up'                                               30                     23
#    ...
# 466935   'the epidemiological investigation of co infection                      1                      1
# 466936   'perceived usefulness and ease of use of fundoscop                      1                      1
# 466937   'ritter reaction mediated syntheses of 2 oxaadaman                      1                      1
# 466938   'an entropy based approach for anomaly detection i                      1                      1
# 466939   'de novo design of high affinity antibody variable                      1                      1


# %%
z = [list(v)[0] for k, v in titles_uids.items() if len(v) == 1]
print(f"{100 - len(z) / len(titles_uids) * 100:.2f}% of titles have more than one CORD UID.")


z = set(z)
print(f"This represents {len([None for m in metadata if m['cord_uid'] not in z]) / len(metadata) * 100:.2f}% of the corpus.")


# %%
# 25.14% of titles have more than one CORD UID.
# This represents 42.21% of the corpus.


# %%[markdown]
# Ignore documents that have duplicated CORD UIDs and titles
# i.e. titles with > 1 CORD UID, 
# e.g. "reply", "erratum", "highlights from this issue".

# %%
map_uids_metadata = {m['cord_uid']: m for m in metadata if m['cord_uid'] in z}

map_titles_uids = {pattern.sub(' ', m['title'].lower()): k for k, m in map_uids_metadata.items()}


print(f"{len(map_uids_metadata)} CORD UIDs with {len(map_titles_uids)} unique titles out of the original {len(metadata)} documents.")


z = titles_uids = titles = metadata = None
del z, titles_uids, titles, metadata


# %%
# # Read Document Parses

# %%

for __, m in tqdm(map_uids_metadata.items()):

    # Read PMC
    try: 
        with open(data_dir + m['pmc_json_files']) as f:
            x = json.load(f)

    except:

        # Read PDF
        try:
            with open(data_dir + m['pdf_json_files']) as f:
                x = json.load(f)
        
        # No parse file
        except:
            x = {}
    
    finally:

        if 'bib_entries' in x.keys():
            m['bib_entries_titles'] = [pattern.sub(' ', v['title'].lower()) for __, v in x['bib_entries'].items()]
            m['bib_entries'] = [map_titles_uids[t] for t in m['bib_entries_titles'] if t in map_titles_uids.keys()]

        else:
            m['bib_entries_titles'] = []
            m['bib_entries'] = []


m = x = None
del m, x

# %%
dist_dir = './dist/kaggle/v4.0_citations/'

# %%
with open(dist_dir +  'map_uids_metadata.pkl', 'wb') as f:
    pickle.dump(map_uids_metadata, f)

# %%
if False:
    data_dir = './data/kaggle/2021-06-29/'
    dist_dir = './dist/kaggle/v4.0_citations/'
    with open(dist_dir +  'map_uids_metadata.pkl', 'rb') as f:
        map_uids_metadata = pickle.load(f)

    pattern = re.compile(r'\W+')
    map_titles_uids = {pattern.sub(' ', m['title'].lower()): k for k, m in map_uids_metadata.items()}


# %%[markdown]
# # Load CORD-19 Embeddings

# %%

# embs = [[] for i in range(num_docs)]
# map_uids_inds = {k: i for i, k in enumerate(map_uids_metadata.keys())}

list_uids = []
embs = []
with open(data_dir + 'cord_19_embeddings/cord_19_embeddings_2021-05-31.csv') as f:
    for row in tqdm(csv.reader(f)):
        if row[0] in map_uids_metadata.keys():

            list_uids.append(row[0])
            embs.append(list(map(float, row[1:])))

            # i = map_uids_inds[row[0]]
            # embs[i] = list(map(float, row[1:]))


list_uids = np.array(list_uids)
embs = np.array(embs)
num_docs, num_dims = embs.shape


f = i = row = doc = None
del f, i, row, doc


# %%
print(f"Number of Docs with Embeddings: {num_docs}")
print(f"Number of Embedding Dimensions: {num_dims}")

# %%
# Number of Docs with Embeddings: 333361
# Number of Embedding Dimensions: 768

# %%
with open(dist_dir +  'embs.pkl', 'wb') as f:
    pickle.dump((embs, list_uids), f)

# %%
if False:
    with open(dist_dir +  'embs.pkl', 'rb') as f:
        embs, list_uids = pickle.load(f)
        num_docs, num_dims = embs.shape

# %%[markdown]
# # Apply Dimensional Reduction

# %%
%%time

model_umap = umap.UMAP(n_components = 2, n_neighbors = 7, min_dist = 0.2, metric = 'minkowski', metric_kwds = {'p': 2.0/3.0}, random_state = 0)

embs_red = model_umap.fit_transform(embs)
embs_red = embs_red - np.mean(embs_red, axis = 0)

# Time: 6 m 2 s

# %%
with open(dist_dir + 'embs_red.pkl', 'wb') as f:
    pickle.dump((embs_red, list_uids), f)

# %%
if False:
    with open(dist_dir + 'embs_red.pkl', 'rb') as f:
        embs_red, list_uids = pickle.load(f)
        num_docs, __ = embs_red.shape

# %%[markdown]
# ## Plot result

mask = np.random.randint(0, high = num_docs, size = 50000)

# %%

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = emlib.plot_emb(
        coor = embs_red[mask, :], 
        cmap_name = 'qual', 
        marker_size = 1.0, marker_alpha = 0.2, 
        legend_kwargs = {}, colorbar = False, 
        str_title = 'Dimensionally Reduced SPECTER Embeddings of the Kaggle CORD-19 Dataset', ax = ax)
# __ = emlib.plot_emb(coor = embs_red, cmap_name = 'qual', marker_size = 0.5, marker_alpha = 0.01, legend_kwargs = {}, colorbar = False, str_title = 'Dimensionally Reduced SPECTER Embeddings of the Kaggle CORD-19 Dataset', ax = ax)
__ = plt.setp(ax, xlabel = 'x', ylabel = 'y', )

fig.savefig(dist_dir + 'embeddings_umap.png', dpi = 150)

fig = ax = None
del fig, ax

# %%[markdown]
# # Apply Hierarchical Clustering

# %%
# Ideas: 
# 1. Apply K-means recursively
# 2. Apply HDBSCAN recursively with decreasing epsilon and increasing min_samples

# %%
# ## Idea 1

# %%
%%time

kwargs = {'n_clusters': 2, 'random_state': 0}
model_kmeans = KMeans(**kwargs).fit(embs_red)
# model_kmeans = MiniBatchKMeans(**kwargs).fit(embs_red)

# Time: 55.9 s

labels = model_kmeans.labels_


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))
__ = emlib.plot_emb(
        coor = embs_red[mask, :], 
        labels = labels[mask], 
        cmap_name = 'qual', 
        marker_size = 1.0, marker_alpha = 0.2, 
        legend_kwargs = {}, colorbar = False, 
        str_title = 'Dimensionally Reduced SPECTER Embeddings of the Kaggle CORD-19 Dataset', 
        ax = ax)
__ = plt.setp(ax, xlabel = 'x', ylabel = 'y', )


# %%[markdown]
# ## Idea 2

# %%

# epsilons = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005])
# min_samples = np.array([10, 1000, 1500, 2000, 2500, 3000])

epsilons = np.array([0.1, 0.1, 0.03, 0.02, 0.01])
min_samples = np.array([10, 1000, 10, 10, 10])

labels = np.zeros((num_docs, len(epsilons)), dtype = int)

for k, (eps, m) in tqdm(enumerate(zip(epsilons, min_samples)), total = len(epsilons)):

    # Generate clusterer
    kwargs = {'metric': 'euclidean', 'min_cluster_size': 50, 'min_samples': int(m), 'cluster_selection_epsilon': float(eps)}
    clusterer = hdbscan.HDBSCAN(**kwargs)
    

    # Old
    if False:

        clusterer.fit(embs_red)
        l = clusterer.labels_
        # cluster_probs = clusterer.probabilities_
        # outlier_scores = clusterer.outlier_scores_
        # cluster_persist = clusterer.cluster_persistence_


    # New
    if True:

        # First level = all
        if k == 0:
            clusterer.fit(embs_red)
            l = clusterer.labels_

        # Other levels = each cluster of the previous level
        else:

            l = -2 * np.ones(labels[:, k - 1].shape, dtype = int)
            cluster_ids = np.unique(labels[:, k - 1])
            
            for i in cluster_ids:
                
                cluster_mask = (labels[:, k - 1] == i)
                
                if i != -1:
                    clusterer.fit(embs_red[cluster_mask, :])
                    l_ = clusterer.labels_
                else:
                    l_ = -1 * np.ones(embs_red[cluster_mask, 0].shape, dtype = int)


                # if all noise or noise & 1 cluster -> retain cluster membership
                if l_.any(-1) & (len(np.unique(l_)) <= 2):
                    l_ = 0 * l_


                # Make cluster IDs unique, except -1 (= noise)
                j = np.max(l) + 1
                l_[l_ != -1] += j

                l[cluster_mask] = np.copy(l_)


    labels[:, k] = np.copy(l)


for i, l in enumerate(labels.T):
    print(f'\nepsilon: {float(epsilons[i])}')
    print(f'min_samples: {float(min_samples[i])}')
    print(f'Number of clusters: {len(np.unique(l)):d}')
    print(f'Unclustered Fraction: {sum(l == -1) / len(l) * 100:.2f} %')


eps = i = l = l_ = m = kwargs = clusterer = cluster_ids = cluster_mask = None
del eps, i, l, l_, m, kwargs, clusterer, cluster_ids, cluster_mask

# epsilon: 0.1
# min_samples: 10.0
# Number of clusters: 90
# Unclustered Fraction: 1.34 %

# epsilon: 0.1
# min_samples: 1000.0
# Number of clusters: 92
# Unclustered Fraction: 1.41 %

# epsilon: 0.03
# min_samples: 10.0
# Number of clusters: 585
# Unclustered Fraction: 35.63 %

# epsilon: 0.02
# min_samples: 10.0
# Number of clusters: 714
# Unclustered Fraction: 43.05 %

# epsilon: 0.01
# min_samples: 10.0
# Number of clusters: 1002
# Unclustered Fraction: 57.60 %

# Time: 1 m 10 s

# %%

m = int(np.floor(np.sqrt(len(epsilons))))
if m * m != len(epsilons):
    n = m + 1
else:
    n = m

fig, ax = plt.subplots(nrows = n, ncols = m, figsize = (16, 16 * (n / m)))

for i, x in enumerate(fig.axes):

    if i < len(epsilons):
        __ = emlib.plot_emb(
            coor = embs_red[mask, :][labels[mask, i] != -1, :], 
            labels = labels[mask, :][labels[mask, i] != -1, i], 
            # coor = embs_red[mask, :], 
            # labels = labels[mask, :], 
            cmap_name = 'qual', 
            marker_size = 1.0, marker_alpha = 0.1, 
            legend_kwargs = {}, colorbar = False, 
            str_title = f"{epsilons[i]}, {min_samples[i]} -> {len(np.unique(labels[:, i]))}, {sum(labels[:, i] == -1) / num_docs * 100:.2f}%", 
            ax = x)
        __ = plt.setp(x, xlabel = 'x', ylabel = 'y')
        __ = x.tick_params(length = 0)
    
    else:
        x.axis('off')

fig.suptitle(f"epsilon, min_samples -> n_clusters, noise")
fig.savefig(dist_dir + 'embeddings_umap_clusters.png', dpi = 150)


i = m = n = x = fig = ax = None
del i, m, n, x, fig, ax

# %%[markdown]
# # Generate Lists

# %%
%%time

docs = [map_uids_metadata[u] for u in list_uids]

nodes, nodeLayout, nodeAtts, groups = emlib.generate_kaggle_nodelist(docs = docs, embs = embs_red, labels = labels, model_id = None)

edges, nodes = emlib.generate_kaggle_edgelist(docs = docs, nodes = nodes)

# Time: 43.9 s

# %%

print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {len(edges)}")

# Number of nodes: 347666
# Number of edges: 292077

# %%

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))

l = 0
m = 1
n = []
while m > 0:

    m = len([None for g in groups if g['level'] == l])
    
    if m > 0:

        n.append(m)

        x = 10 ** np.arange(np.log10(40), 5.6, 0.02)
        y, __ = np.histogram([len(g['node_ids_all']) for g in groups if g['level'] == l], bins = x)
        ax[1].plot(0.5 * (x[:-1] + x[1:]), y, label = f"l = {l}")
        
    l += 1

__ = ax[1].plot((m, m), (0, 40), color = 'k', linestyle = '--', linewidth = 1.0, label = 'Total')
__ = ax[1].legend()
__ = plt.setp(ax[1], xlabel = 'Size of Membership', ylabel = 'Number of Groups', xscale = 'log', title = 'Histogram of Group Membership Size')

m = len(np.array([j for i in [g['node_ids_all'] for g in groups if g['level'] == 0] for j in i]))


__ = ax[0].plot(range(l - 1), n, marker = 'o')
__ = [ax[0].annotate(f"{i:d}", (j - 0.2, i + 15)) for i, j in zip(n, list(range(l - 1)))]
__ = plt.setp(ax[0], xlabel = 'Hierarchical Level', ylabel = 'Number of Groups', xticks = range(l - 1), yticks = range(0, 1001, 200), 
    title = 'Group Generation Per Level')


fig.suptitle(f"Iterative Hierarchical Clustering")
fig.savefig(dist_dir + 'embeddings_umap_clusters_dist.png', dpi = 150)


l = m = n = x = y = fig = ax = None
del l, m, n, x, y, fig, ax


# %%[markdown]
# ## Export Data Locally

# %%
%%time

for x, y in zip(('nodes', 'nodeLayout', 'nodeAtts', 'groups', 'edges'), (nodes, nodeLayout, nodeAtts, groups, edges)):

    emlib.save_jsonl(y, f'{dist_dir}{x}.jsonl', preamble = emlib.get_obj_preamble(obj_type = x))


# %%[markdown]
# ## Export Data Remotely

# %%
%%time

data = {
    'nodes': nodes,
    'nodeLayout': nodeLayout,
    'nodeAtts': nodeAtts,
    'groups': groups,
    'edges': edges
}

s3_url = 'http://10.64.18.171:9000'
s3_bucket = 'aske'
s3_path = 'research/KB' + dist_dir[1:(len(dist_dir) - 2)]

for x in tqdm(('nodes', 'nodeLayout', 'nodeAtts', 'groups', 'edges')):

    emlib.load_obj_to_s3(
        obj = data, 
        s3_url = s3_url, 
        s3_bucket = s3_bucket, 
        s3_path = f"{s3_path}/{x}.jsonl", 
        preamble = emlib.get_obj_preamble(obj_type = x),
        obj_key = x
    )


# %%
# ## Reload

# %%
if False:
    data_dir = './data/kaggle/2021-06-29/'
    dist_dir = './dist/kaggle/v4.0_citations/'

    groups = emlib.load_jsonl(dist_dir + 'groups' + '.jsonl', remove_preamble = True)


# %%
