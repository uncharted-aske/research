# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content: 
# * Load EMMAA models beyond Covid-19
# * Check compatiblity with existing pipeline

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
import pathlib

import emmaa_lib as emlib
import importlib
# importlib.reload(emlib)

# %%
np.random.seed(0)

# %%[markdown]
# # List of Available EMMAA Models
# 
# * Acute myeloid leukemia ('aml')
# * Breast cancer ('brca')
# * Covid-19 ('covid19')
# * Lung adenocarcinoma ('luad')
# * Marm model ('marm_model')
# * Multiple sclerosis ('ms')
# * Neurofibromatosis ('nf')
# * Pancreatic adenocarcinoma ('paad')
# * Pain Machine ('painmachine')
# * Prostate Adenocarcinomo ('prad')
# * Ras Machine 2.0 ('rasmachine')
# * Ras model ('rasmodel')
# * Melanoma ('skcm')
# * Vitiligo ('vitiligo')


# %%
# Create model list
emmaa_api = 'https://emmaa.indra.bio'
url = emmaa_api + '/models'
try:
    x = requests.get(url).json()
    models = [{'id': i, 'id_emmaa': name} for i, name in enumerate(x['models'])]
except:
    models = []


for model in models:

    # Get model metadata
    try:
        url = emmaa_api + '/model_info/' + model['id_emmaa']
        x = requests.get(url).json()
        model['name'] = x['human_readable_name']
        model['description']  = x['description']

    except: 
        model['name'] = None
        model['description'] = None


    # Get model tests
    try:
        url = emmaa_api + '/test_corpora/' + model['id_emmaa']
        x = requests.get(url).json()
        model['tests'] = x['test_corpora']

    except: 
        model['tests'] = None


# %%
# Create test list
x = sorted(set([test for model in models for test in model['tests']]))
tests = [{'id': i, 'id_emmaa': name} for i, name in enumerate(x)]

map_tests_ids = {test['id_emmaa']: test['id'] for test in tests}
for model in models:
    model['tests'] = [map_tests_ids[test_name] for test_name in model['tests']]


# Get test metadata
for test in tests:

    try:
        url = emmaa_api + '/tests_info/' + test['id_emmaa']
        x = requests.get(url).json()
        test['name'] = x['name']
    except:
        test['name'] = None


x = url = model_ids_emmaa = model = test = map_tests_ids = None
del x, url, model_ids_emmaa, model, test, map_tests_ids

# %%
# Save these lists
preamble = {
    'id': '<int> unique ID of the model that referenced by all related files',
    'id_emmaa': '<str> unique ID of the model used by EMMAA',
    'name': '<str> human-readable name of the model',
    'description': '<str> description of the model', 
    'test_ids': '<array of int> list of the IDs of the test corpora available for this model (see `tests.jsonl`)', 
}
emlib.save_jsonl(models, '/home/nliu/projects/aske/research/BIO/dist/v3.2/models.jsonl', preamble = preamble)


preamble = {
    'id': '<int> unique ID of the test that referenced by all related files (e.g. `models.jsonl`)',
    'id_emmaa': '<str> unique ID of the test used by EMMAA',
    'name': '<str> human-readable name of the test'
}
emlib.save_jsonl(tests, '/home/nliu/projects/aske/research/BIO/dist/v3.2/tests.jsonl', preamble = preamble)


preamble = None
del preamble

# %%[markdown]
# # Download and Process Models

# %%
%%time

for model in models:

    try:

        # Download model statements
        url = f"https://emmaa.s3.amazonaws.com/assembled/{model['id_emmaa']}/latest_statements_{model['id_emmaa']}.json"
        statements = requests.get(url).json()
        

        # Generate node and edge lists
        nodes, edges, __, __, evidences, documents = emlib.process_statements(statements, model_id = model['id'])


        # Save evidence list
        preamble = {
            'model_id': '<int> unique model ID that is present in all related files',
            'id': '<int> unique evidence ID that is referenced by other files',
            'text': '<str>  (from the `text` attribute in `latest_statements.jsonl`)',
            'text_refs': '<dict where key = <str> identifier type and value = <str> document identifier> reference of source document (from the `text_refs` attribute in `latest_statements.jsonl`)',
            'doc_id': '<int> ID of the source document (see `documents.jsonl`)',
            'statement_ids': '<str> ID of supported statement (from `matches_hash` in `latest_statements.jsonl`)',
            'edge_ids': '<list of int> IDs of supported edges',
        }
        emlib.save_jsonl(evidences, f"/home/nliu/projects/aske/research/BIO/dist/v3.2/models/{model['id_emmaa']}/evidences.jsonl", preamble = preamble)


        # Save document list
        preamble = {
            'model_id': '<int> unique model ID that is present in all related files',
            'id': '<int> unique document ID that is referenced by other files',
            'DOI': '<str> DOI identifier of this document (all caps)'
        }
        emlib.save_jsonl(documents, f"/home/nliu/projects/aske/research/BIO/dist/v3.2/models/{model['id_emmaa']}/documents.jsonl", preamble = preamble)


        # Save node list
        preamble = {
            'model_id': '<int> unique model ID that is present in all related files',
            'id': '<int> unique node ID that is referenced by other files',
            'name': '<str> unique human-interpretable name of this node (from the `name` attribute in `latest_statements_covid19.jsonl`)',
            'db_refs': '<dict> database references of this node (from the `db_refs` attribute in `latest_statements_covid19.jsonl`)',
            'grounded': '<bool> whether this node is grounded to any database',
            'edge_ids_source': '<list of int> ID of edges that have this node as a source',
            'edge_ids_target': '<list of int> ID of edges that have this node as a target',
            'out_degree': '<int> out-degree of this node',
            'in_degree': '<int> in-degree of this node', 
        }
        emlib.save_jsonl(nodes, f"/home/nliu/projects/aske/research/BIO/dist/v3.2/models/{model['id_emmaa']}/nodes.jsonl", preamble = preamble)


        # Save edge list
        preamble = {
            'model_id': '<int> unique model ID that is present in all related files',
            'id': '<int> unique edge ID that is referenced by other files',
            'type': '<str> type of this edge (from `type` attribute in `latest_statements.jsonl`)',
            'belief': '<float> belief score of this edge (from `belief` attribute in `latest_statements.jsonl`)',
            'statement_id': '<str> unique statement id (from `matches_hash` in `latest_statements.jsonl`)', 
            'evidence_ids': '<list of int> IDs of the supporting evidences (as defined in `evidences.jsonl`)',
            'doc_ids': '<list of int> IDs of the supporting documents (as defined in `documents.jsonl`)',
            'source_id': '<int> ID of the source node (as defined in `nodes.jsonl`)' ,
            'target_id': '<int> ID of the target node (as defined in `nodes.jsonl`)',
            'tested': '<bool> whether this edge is tested'
        }
        emlib.save_jsonl(edges, f"/home/nliu/projects/aske/research/BIO/dist/v3.2/models/{model['id_emmaa']}/edges.jsonl", preamble = preamble)

    except:
        statements = []


    emlib.save_jsonl(statements, f"/home/nliu/projects/aske/research/BIO/data/models/{model['id_emmaa']}/{str(datetime.date.today())}/latest_statements.jsonl")


model = statements = nodes = edges = evidences = documents = None
del model, statements, nodes, edges, evidences, documents


#  time:


# %%

# 2561 statements -> 1891 processed statements.
# Found 7497 evidences and 617 documents.
# Found 777 nodes and 2548 edges.

# 6371 statements -> 5269 processed statements.
# Found 27991 evidences and 1754 documents.
# Found 1367 nodes and 6669 edges.

# 395394 statements -> 391070 processed statements.
# Found 501203 evidences and 92456 documents.
# Found 44104 nodes and 448723 edges.

# 2802 statements -> 2310 processed statements.
# Found 10361 evidences and 1449 documents.
# Found 1000 nodes and 2867 edges.

# 25 statements -> 19 processed statements.
# Found 14 evidences and 0 documents.
# Found 12 nodes and 30 edges.

# 13883 statements -> 13441 processed statements.
# Found 39051 evidences and 8539 documents.
# Found 3652 nodes and 15373 edges.

# 9017 statements -> 8654 processed statements.
# Found 15223 evidences and 5621 documents.
# Found 2706 nodes and 10382 edges.

# 994 statements -> 954 processed statements.
# Found 2755 evidences and 615 documents.
# Found 605 nodes and 1246 edges.

# 8025 statements -> 7769 processed statements.
# Found 22502 evidences and 186 documents.
# Found 3167 nodes and 13239 edges.

# 1602 statements -> 1499 processed statements.
# Found 4723 evidences and 935 documents.
# Found 809 nodes and 1908 edges.

# 6508 statements -> 5602 processed statements.
# Found 30215 evidences and 1413 documents.
# Found 1356 nodes and 8868 edges.

# 257 statements -> 185 processed statements.
# Found 168 evidences and 0 documents.
# Found 111 nodes and 245 edges.

# 1534 statements -> 1110 processed statements.
# Found 4775 evidences and 639 documents.
# Found 526 nodes and 1495 edges.

# 2848 statements -> 2736 processed statements.
# Found 2618 evidences and 693 documents.
# Found 1352 nodes and 3132 edges.

# %%[markdown]
# # Download Test Data

# %%
%%time

for test in tests:

    try:
        url = f"https://emmaa.s3.amazonaws.com/tests/{test['id_emmaa']}.jsonl"
        test_statements = [json.loads(line) for line in requests.get(url).text.splitlines()]
        
    except:
        test_statements = []

    emlib.save_jsonl(test_statements, f"/home/nliu/projects/aske/research/BIO/data/tests/{test['id_emmaa']}/{str(datetime.date.today())}/latest_test_statements.jsonl")


url = test = test_statements = None
del url, test, test_statements


# time: 

# %%
# Download Tested Paths

# %%
# Paths from Mitre test
try: 
    url = 'https://emmaa.s3.amazonaws.com/paths/covid19/covid19_mitre_tests_latest_paths.jsonl'
    paths = [json.loads(line) for line in requests.get(url).text.splitlines()]
except:
    paths = []

emlib.save_jsonl(paths, f"/home/nliu/projects/aske/research/BIO/data/models/covid19/{str(datetime.date.today())}/covid19_mitre_tests_latest_paths.jsonl")

# %%
# Paths from curated test
try: 
    url = 'https://emmaa.s3.amazonaws.com/paths/covid19/covid19_curated_tests_latest_paths.jsonl'
    paths = [json.loads(line) for line in requests.get(url).text.splitlines()]

except:
    paths = []

emlib.save_jsonl(paths, f"/home/nliu/projects/aske/research/BIO/data/models/covid19/{str(datetime.date.today())}/covid19_curated_tests_latest_paths.jsonl")


url = paths = None
del url, paths


# %%
# Upload files
with open('/home/nliu/projects/aske/research/BIO/dist/v3.2/put_files.sh', 'w') as f:

    f.write('#!/bin/sh\n')


    # Raw Data
    f.write(f"python /home/nliu/projects/aske/research/put_files.py /home/nliu/projects/aske/research/BIO/data/ontologies/ aske/research/BIO/data/ontologies/\n")
    f.write(f"python /home/nliu/projects/aske/research/put_files.py /home/nliu/projects/aske/research/BIO/data/ aske/research/BIO/data/\n")
    
    for test in tests:
        f.write(f"python /home/nliu/projects/aske/research/put_files.py /home/nliu/projects/aske/research/BIO/data/tests/{test['id_emmaa']}/{str(datetime.date.today())}/ aske/research/BIO/data/tests/{test['id_emmaa']}/{str(datetime.date.today())}/\n")

    for model in models:
        f.write(f"python /home/nliu/projects/aske/research/put_files.py /home/nliu/projects/aske/research/BIO/models/{model['id_emmaa']}/{str(datetime.date.today())}/ aske/research/BIO/data/tests/{model['id_emmaa']}/{str(datetime.date.today())}/\n")


    # Processed Data
    f.write(f"python /home/nliu/projects/aske/research/put_files.py /home/nliu/projects/aske/research/BIO/dist/v3.2/ aske/research/BIO/dist/v3.2/\n")
    for model in models:
        f.write(f"python /home/nliu/projects/aske/research/put_files.py /home/nliu/projects/aske/research/BIO/dist/v3.2/models/{model['id_emmaa']}/ aske/research/BIO/dist/v3.2/models/{model['id_emmaa']}/\n")


# %%



# %%
