# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Load EMMAA corpus DOI list and Kaggle CORD document meta-data
# * Calculate overlap

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

# %%
# # Load EMMAA and Kaggle Corpus Meta-Data

docs_emmaa = emlib.load_jsonl('/home/nliu/projects/aske/research/BIO/dist/v3.1/full/documents.jsonl', remove_preamble = True)

docs_kaggle = []
with open('./data/kaggle/metadata.csv') as f:
    docs_kaggle.extend([row for row in csv.DictReader(f)])

f = None
del f

# %%
# # Calculate set intersection and differences

# %%
dois_emmaa = set([doc['DOI'] for doc in docs_emmaa])
dois_kaggle = set([doc['doi'] for doc in docs_kaggle])


print(f"There are {len(dois_emmaa)} and {len(dois_kaggle)} unique docs in the EMMAA and Kaggle corpuses respectively.")
print(f"* union (E & K): {len(dois_emmaa & dois_kaggle)}")
print(f"* intersection (E | K): {len(dois_emmaa | dois_kaggle)}")
print(f"* left difference (E - K): {len(dois_emmaa - dois_kaggle)}")
print(f"* right difference (K - E): {len(dois_kaggle - dois_emmaa)}")
print(f"* symmetric difference (E ^ K): {len(dois_emmaa ^ dois_kaggle)}")

print(f"{len(dois_emmaa - dois_kaggle) / len(dois_emmaa) * 100:.0f}% of the EMMAA corpus is missing from the Kaggle one.")

# %%

# There are 85959 and 220839 unique docs in the EMMAA and Kaggle corpuses respectively.
# * union (E & K): 10938
# * intersection (E | K): 295860
# * left difference (E - K): 75021
# * right difference (K - E): 209901
# * symmetric difference (E ^ K): 284922
# 
# 87% of the EMMAA corpus are missing from the Kaggle one.

# %%
