# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content: 
# * Use BIO Ontology v1.7 export v3
# * Create a mock model wherein a node exists for each ontology category


# %%
import re
import json
import pickle
import time
import datetime
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

import emmaa_lib as emlib
import importlib
# importlib.reload(emlib)

# %%
np.random.seed(0)

# %%[markdown]
# # Load Ontology

# %%
with open('./data/ontologies/bio_ontology_v1.7_export_v3.json', 'r') as x:
    G_onto_JSON = json.load(x)

# Remove 'xref' links
G_onto_JSON['links'] = [link for link in G_onto_JSON['links'] if link['type'] != 'xref']


# Load the ontology graph as a `networkx` object
# G_onto = nx.readwrite.json_graph.node_link_graph(G_onto_JSON)


# %%[markdown]
# # Generate `nodes`
nodes = [{
    'model_id': -1, 
    'id': i, 
    'name': f"{node['id'] if 'name' not in node.keys() else node['id'] if node['name'] == None else node['id'] if len(node['name']) == 0 else node['name']}",
    'db_refs': None,
    'db_ref_priority': node['id'],
    'grounded': True,
    'edge_ids_source': [],
    'edge_ids_target': [],
    'out_degree': 0,
    'in_degree': 0
    } for i, node in enumerate(G_onto_JSON['nodes'])]

# %%
%%time

emlib.calculate_onto_root_path(nodes, G_onto_JSON)

# time: 

# %%
%%time

ontocats = emlib.extract_ontocats(nodes, G_onto_JSON)

# time:

# %%







# %%
# Empty `edges`
edges = []


# %%

namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, G_onto_JSON, nodes_full)




