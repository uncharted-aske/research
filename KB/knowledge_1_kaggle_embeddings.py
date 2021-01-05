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
        docs_cord[map_uids_docs[row[0]]]['embedding'] = row[1:]

num_dims_cord = len(docs_cord[0]['embedding'])

print(f"Number of Docs: {num_docs_cord}")
print(f"Number of Embedding Dimensions: {num_dims_cord}")
print(f"Document Metadata Keys: {list(docs_cord[0].keys())}")


f = row = None
del f, row

# %%
# Load EMMAA edge data

edges_mitre = emlib.load_jsonl('./dist/v3/edges_mitre.jsonl', remove_preamble = True)




# %%




# %%