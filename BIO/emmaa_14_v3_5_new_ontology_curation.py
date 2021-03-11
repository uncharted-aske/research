# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content: 
# * Same process as v3.3
# * Latest EMMAA Covid-19 model statements (2021-03-10)
# * Added Ontology v1.8 export v1
# * Latest list of curated statements (2021-03-10)

# %%
import json
import pickle
import time
import datetime
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
import requests

import emmaa_lib as emlib
import importlib
# importlib.reload(emlib)

# %%
np.random.seed(0)

# %%[markdown]
# # Download Latest Data

# %%
# ## Statements
try:
    url = f"https://emmaa.s3.amazonaws.com/assembled/{'covid19'}/latest_statements_{'covid19'}.json"
    statements = requests.get(url).json()
except:
    statements = None

emlib.save_jsonl(statements, f"/home/nliu/projects/aske/research/BIO/data/models/{'covid19'}/{str(datetime.date.today())}/latest_statements.jsonl")

# %%
# ## MITRE tested paths
try: 
    url = f"https://emmaa.s3.amazonaws.com/paths/{'covid19'}/{'covid19'}_mitre_tests_latest_paths.jsonl"
    r = requests.get(url)
    paths_mitre = [json.loads(line) for line in r.text.splitlines()]
except:
    paths_mitre = None

emlib.save_jsonl(paths_mitre, f"/home/nliu/projects/aske/research/BIO/data/models/{'covid19'}/{str(datetime.date.today())}/latest_paths_mitre.jsonl")

# %% 
# ## Curated statements
try: 
    url = f"https://emmaa.indra.bio/curated_statements/{'covid19'}"
    curated = requests.get(url).json()
except:
    curated = None

with open(f"/home/nliu/projects/aske/research/BIO/data/models/{'covid19'}/{str(datetime.date.today())}/curated_statements.jsonl", 'w') as x:
    json.dump(curated, x)

# %%
# ## Ontology

try:
    url = "https://emmaa.s3.amazonaws.com/integration/ontology/bio_ontology_v1.8_export_v1.json.gz"
    G_onto_JSON = json.loads(gzip.decompress(requests.get(url).content))
except:
    G_onto_JSON = None

# %%
# Patch name error in ontology
for node in G_onto_JSON['nodes']:
    if node['id'] in ['HP:HP:0000001', 'CHEBI:CHEBI:24431', 'CHEBI:CHEBI:50906']:
        node['name'] = node['name'][node['id']]


# %%
with open(f"/home/nliu/projects/aske/research/BIO/data/ontologies/bio_ontology_v1.8_export_v1.json", 'w') as f:
    json.dump(G_onto_JSON, f)


f = None
del f


# %%[markdown]
# ## Reload Data
if False:
    i = '/home/nliu/projects/aske/research/BIO/data'
    statements = emlib.load_jsonl(f"{i}/models/covid19/2021-03-10/latest_statements.jsonl")
    paths_mitre = emlib.load_jsonl(f"{i}/models/covid19/2021-03-10/latest_paths_mitre.jsonl")
    curated = emlib.load_jsonl(f"{i}/models/covid19/2021-03-10/curated_statements.jsonl")

    with open(f"{i}/ontologies/bio_ontology_v1.8_export_v1.json", "r") as f:
        G_onto_JSON = json.load(f)


f = None
del f

# %%[markdown]
# # Generate Node/Edge Lists

# %%
%%time

model_id = -1
nodes, edges, statements_, paths_mitre_, evidences, documents = emlib.process_statements(statements, paths = paths_mitre, model_id = model_id)


# 403405 statements -> 399049 processed statements.
# Found 514280 evidences and 96088 documents.
# Found 44497 nodes and 457872 edges.
# 4319 paths -> 4319 processed paths.
# Found 5126 tested edges.

# time: 2 m 27 s

# %%[markdown]
# ## Include Curation Status in Node List

# %%
x = {'incorrect': 0, 'correct': 1, 'partial': 2}
for edge in edges:

    edge['curated'] = 3

    for k in x:
        if edge['statement_id'] in curated[k]:
            edge['curated'] = x[k]


print("Curation status:")
__ = [print(f"{len([True for edge in edges if edge['curated'] == v])} {k} edges") for k, v in x.items()]

k = x = edge = None
del k, x, edge

# %%[markdown]
# Save outputs

# %%
# `evidences`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique evidence ID that is referenced by other files',
    'text': '<str>  (from the `text` attribute in `latest_statements_covid19.jsonl`)',
    'text_refs': '<dict where key = <str> identifier type and value = <str> document identifier> reference of source document (from the `text_refs` attribute in `latest_statements_covid19.jsonl`)',
    'doc_id': '<int> ID of the source document (see `documents.jsonl`)',
    'statement_ids': '<str> ID of supported statement (from `matches_hash` in `latest_statements_covid19.jsonl`)',
    'edge_ids': '<list of int> IDs of supported edges',
}
emlib.save_jsonl(evidences, './dist/v3.5/evidences.jsonl', preamble = preamble)


# `documents`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique document ID that is referenced by other files',
    'DOI': '<str> DOI identifier of this document (all caps)'
}
emlib.save_jsonl(documents, './dist/v3.5/documents.jsonl', preamble = preamble)

# %%
# `nodes`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique node ID that is referenced by other files',
    'name': '<str> unique human-interpretable name of this node (from the `name` attribute in `latest_statements.jsonl`)',
    'db_refs': '<dict> database references of this node (from the `db_refs` attribute in `latest_statements.jsonl`)',
    'grounded': '<bool> whether this node is grounded to any database',
    'edge_ids_source': '<list of int> ID of edges that have this node as a source',
    'edge_ids_target': '<list of int> ID of edges that have this node as a target',
    'out_degree': '<int> out-degree of this node',
    'in_degree': '<int> in-degree of this node', 
}
emlib.save_jsonl(nodes, './dist/v3.5/nodes.jsonl', preamble = preamble)


# `edges`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique edge ID that is referenced by other files',
    'type': '<str> type of this edge (from `type` attribute in `latest_statements_covid19.jsonl`)',
    'belief': '<float> belief score of this edge (from `belief` attribute in `latest_statements.jsonl`)',
    'statement_id': '<str> unique statement id (from `matches_hash` in `latest_statements.jsonl`)', 
    'evidence_ids': '<list of int> IDs of the supporting evidences (as defined in `evidences.jsonl`)',
    'doc_ids': '<list of int> IDs of the supporting documents (as defined in `documents.jsonl`)',
    'source_id': '<int> ID of the source node (as defined in `nodes.jsonl`)' ,
    'target_id': '<int> ID of the target node (as defined in `nodes.jsonl`)',
    'tested': '<bool> whether the underlying statement of this edge is pass the Mitre test',
    'curated': '<int> curation status of the underlying statement of this edge (`incorrect` = `0`, `correct` = `1`, `partial` = `2`, `uncurated` = `3`)'
}
emlib.save_jsonl(edges, './dist/v3.5/edges.jsonl', preamble = preamble)


# %%[markdown]
# # Generate Ontocats

# %%
# Remove 'xref' links
G_onto_JSON['links'] = [link for link in G_onto_JSON['links'] if link['type'] != 'xref']

# %%
%%time

# Generate a namespace list common to the model graph and the ontology
namespaces_priority = ['FPLX', 'UPPRO', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'MIRBASE', 'DOID', 'HP', 'EFO']
namespaces, namespaces_count = emlib.generate_ordered_namespace_list(namespaces_priority, G_onto_JSON, nodes)

# Reduce 'db_refs' of each model node to a single entry by namespace priority
nodes, __ = emlib.reduce_nodes_db_refs(nodes, namespaces)

# Calculate in-ontology paths
emlib.calculate_onto_root_path(nodes, G_onto_JSON)

# Extract Ontological Categories
ontocats = emlib.extract_ontocats(nodes, G_onto_JSON)

# time: 3 h

# %%[markdown]
# ## Create Node Type
# 
# Define 'node type' as the name of the ancestor ontocat/group of the node in the ontology

# %%
# Fill in missing names
for ontocat in ontocats:
    if (ontocat['name'] == '') | (ontocat['name'] == None):
        ontocat['name'] = ontocat['ref']

# %%
map_ids_ontocats = {ontocat['id']: i for i, ontocat in enumerate(ontocats)}

for node in nodes:
    if node['grounded_onto'] == True:
        i = map_ids_ontocats[node['ontocat_ids'][0]]
        node['type'] = ontocats[i]['name']
    else:
        node['type'] = None


# %%

# Save new attributes of `nodes` as `nodeAtts`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique node ID that is defined in `nodes.jsonl`',
    'type': '<str> node type (currently set to the `name` of the ancestor ontocat, `None` if ungrounded)',
    'db_ref_priority': '<str> database reference from `db_refs` of `nodes.jsonl`, that is used by the INDRA ontology', 
    'grounded_onto': '<bool> whether this model node is grounded to something that exists within the ontology', 
    'ontocat_level': '<int> the level of the most fine-grained ontology node/category to which this model node was mapped (`-1` if not mappable, `0` if root)', 
    'ontocat_ids': '<array of int> ordered list of ontological category IDs (see `ontocats.jsonl`) to which this node is mapped (order = root-to-leaf)' 
}
emlib.save_jsonl(nodes, './dist/v3.5/nodeAtts.jsonl', preamble = preamble)


# Save `ontocats`
preamble = {
    'model_id': '<int> unique model ID that is present in all related distribution files',
    'id': '<int> unique ID for the ontological category that is referenced by other files',
    'ref': '<str> unique reference ID of this category (as given by the INDRA Ontology)',
    'name': '<str> name of this category (as given by the INDRA Ontology)',
    'size': '<int> number of model nodes that were mapped to this category and its children',
    'level': '<int> number of hops to reach the local root (`0` if root)',
    'parent_id': '<int> ID of the parent of this category in the ontology',
    'children_ids': '<array of int> unordered list of the child category IDs',
    'node_ids': '<array of int> unordered list of IDs from model nodes in the membership of this category',
    'node_ids_direct': '<array of int> node_ids but only model nodes which were directly mapped to this category and not any of the child categories'
}
emlib.save_jsonl(ontocats, './dist/v3.5/ontocats.jsonl', preamble = preamble)

# %%

if False:
    nodes = [{**node, **att} for node, att in zip(emlib.load_jsonl('./dist/v3.5/nodes.jsonl', remove_preamble = True), emlib.load_jsonl('./dist/v3.5/nodeAtts.jsonl', remove_preamble = True))]
    edges = emlib.load_jsonl('./dist/v3.5/edges.jsonl', remove_preamble = True)
    ontocats = emlib.load_jsonl('./dist/v3.5/ontocats.jsonl', remove_preamble = True)
