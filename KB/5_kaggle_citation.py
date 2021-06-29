# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Build citation network of the Kaggle CORD19 dataset

# %%
from tqdm import tqdm
import json
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# %%
np.random.seed(0)

# %%[markdown]
# # Load Metadata

# %%
metadata = []
with open('./data/kaggle/2021-06-29/metadata.csv') as f:
    metadata.extend([row for row in csv.DictReader(f)])


# Drop 'abstract'
__ = [m.pop('abstract', None) for m in metadata]


f = None
del f

# %%[markdown]
# ## Generate Maps for Uniqueness

map_uids_metadata = {m['cord_uid']: m for m in metadata}
map_titles_uids = {m['title']: m['cord_uid'] for m in metadata}

titles = {m['title']: [] for m in metadata}
for i, m in enumerate(metadata):
    titles[m['title']].append(i)


# %%
# # Read Document Parses

# %%

data_dir = './data/kaggle/2021-06-29/'

for m in tqdm(metadata):

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
            m['bib_entries_titles'] = [v['title'] for k, v in x['bib_entries'].items()]
            m['bib_entries'] = [map_titles_uids[t] for t in m['bib_entries_titles'] if t in map_titles_uids.keys()]

        else:
            m['bib_entries_titles'] = []
            m['bib_entries'] = []

m = x = None
del m, x


# %%

