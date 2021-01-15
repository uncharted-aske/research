# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Load Kaggle CORD document embeddings
# * Check overlap with EMMAA `text_refs`
# * Explore

# %%
import sys
import csv
import json
import pickle
import time
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import umap
import hdbscan
import importlib
import emmaa_lib as emlib


# %%
np.random.seed(0)

# %%[markdown]
# # Load Kaggle Data

docs_cord = []
with open('./data/kaggle/metadata.csv') as f:
    docs_cord.extend([row for row in csv.DictReader(f)])

num_docs_cord = len(docs_cord)
map_uids_docs = {doc['cord_uid']: i for i, doc in enumerate(docs_cord)}


with open('./data/kaggle/cord_19_embeddings_2020-12-13.csv') as f:
    for row in csv.reader(f):
        docs_cord[map_uids_docs[row[0]]]['embedding'] = list(map(float, row[1:]))

embs_cord = np.array([doc['embedding'] for doc in docs_cord])
num_dims_cord = embs_cord.shape[1]


for doc in docs_cord:
    doc['embedding'] = None
    del doc['embedding']


f = row = doc = None
del f, row, doc

# %%
print(f"Number of Docs: {num_docs_cord}")
print(f"Number of Embedding Dimensions: {num_dims_cord}")
print(f"Document Metadata Keys:\n\t")
__ = [print(f"{k}") for k in list(docs_cord[0].keys())]

# Number of Docs: 381817
# Number of Embedding Dimensions: 768
# Document Metadata Keys: 
#    ['cord_uid', 'sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id', 'license', 'abstract', 'publish_time', 'authors', 'journal', 
#    'mag_id', 'who_covidence_id', 'arxiv_id', 'pdf_json_files', 'pmc_json_files', 'url', 's2_id', 'embedding']

# %%[markdown]
# # Apply Dimensional Reduction
#
# Build matrix of embeddings



# %%
%%time

num_dims_red = 2
model_umap = umap.UMAP(n_components = num_dims_red, n_neighbors = 10, min_dist = 0.05, metric = 'minkowski', metric_kwds = {'p': 2.0/3.0}, random_state = 0)

embs_red = model_umap.fit_transform(embs[:1000, :])
embs_red = embs_red - np.mean(embs_red, axis = 0)

# Time: 1 m 14 s

# %%
# Plot result

__ = emlib.plot_emb(coor = embs_red, cmap_name = 'qual', legend_kwargs = {}, colorbar = False)



# %%[markdown]
# # Apply Hierarchical Clustering



# %%