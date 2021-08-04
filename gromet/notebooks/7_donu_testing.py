# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Interact with Donu API
# * Run MARM PNC model

# %%
import requests
import json
import matplotlib.pyplot as plt

DONU_ENDPOINT = 'https://aske.galois.com/donu/'


# %%[markdown]
# # List Models

models = requests.post(DONU_ENDPOINT, json = {'command': 'list-models'}).json()['result']

# %%
# print(f"{json.dumps(response_body, indent = 2)}")

print(f"{'i':>3} | {'Model':<50} | {'Type':15}")
__ = [print(f"{i:>3} | {model['source']['model']:<50} | {model['type']:15}") for i, model in enumerate(models)]


#   i | Model                                              | Type                
#   0 | sir-meta.easel                                     | easel               
#   1 | sirsuper.easel                                     | easel               
#   2 | sir-vd.easel                                       | easel               
#   3 | sir.easel                                          | easel               
#   4 | sirs.easel                                         | easel               
#   5 | sir-no-parameters.easel                            | easel               
#   6 | seird_hosp.easel                                   | easel               
#   7 | sir.deq                                            | diff-eq             
#   8 | sir-meta.easel                                     | gromet-prt          
#   9 | sirsuper.easel                                     | gromet-prt          
#  10 | sir-vd.easel                                       | gromet-prt          
#  11 | sir.easel                                          | gromet-prt          
#  12 | sirs.easel                                         | gromet-prt          
#  13 | sir-no-parameters.easel                            | gromet-prt          
#  14 | seird_hosp.easel                                   | gromet-prt          
#  15 | seir.json                                          | gromet-pnc          
#  16 | marm_model_gromet_2021-06-28-17-07-14.json         | gromet-pnc          
#  17 | sird.json                                          | gromet-pnc          
#  18 | sir.gromet                                         | gromet-pnc          
#  19 | 3_city_seird.json                                  | gromet-pnc          
#  20 | rasmachine_gromet_2021-06-28-17-29-57.json         | gromet-pnc          
#  21 | seird.json                                         | gromet-pnc          
#  22 | SimpleSIR_metadata_gromet_PetriNetClassic.json     | gromet-pnc          
#  23 | SimpleSIR_metadata_gromet_FunctionNetwork.json     | gromet-fnet 

# %%[markdown]
# # Query Models

response_body = requests.post(DONU_ENDPOINT, json = {'command': 'query-models', 'text': '*marm*'}).json()

# %%
print(f"{json.dumps(response_body, indent = 2)}")

# {
#   "status": "success",
#   "result": []
# }

# %%[markdown]
# # Get Model Schematics

model_def = models[18]
response_body = requests.post(DONU_ENDPOINT, json = {'command': 'get-model-schematic', 'definition': model_def}).json()

# %%
print(f"{json.dumps(response_body, indent = 2)}")

# {
#   "status": "success",
#   "result": {
#     "nodes": [
#       {
#         "name": "J:beta",
#         "type": "event"
#       },
#       {
#         "name": "J:S",
#         "type": "state"
#       },
#       {
#         "name": "J:I",
#         "type": "state"
#       },
#       {
#         "name": "J:gamma",
#         "type": "event"
#       },
#       {
#         "name": "J:R",
#         "type": "state"
#       }
#     ],
#     "edges": [
#       [
#         {
#           "name": "J:beta",
#           "type": "event"
#         },
#         {
#           "name": "J:I",
#           "type": "state"
#         }
#       ],
#       [
#         {
#           "name": "J:S",
#           "type": "state"
#         },
#         {
#           "name": "J:beta",
#           "type": "event"
#         }
#       ],
#       [
#         {
#           "name": "J:I",
#           "type": "state"
#         },
#         {
#           "name": "J:gamma",
#           "type": "event"
#         }
#       ],
#       [
#         {
#           "name": "J:gamma",
#           "type": "event"
#         },
#         {
#           "name": "J:R",
#           "type": "state"
#         }
#       ]
#     ]
#   }
# }

# %%[markdown]
# # Get Model Source

model_def = models[18]
response_body = requests.post(DONU_ENDPOINT, json = {'command': 'get-model-source', 'definition': model_def}).json()

# %%
print(f"{json.dumps(response_body, indent = 2)}")

# {
#   "status": "success",
#   "result": {
#     "source": "{\n  \"syntax\": \"Gromet\",\n  \"type\": \"PetriNetClassic\",\n  \"name\": \"SimpleSIR\",\n  \"metadata\": null,\n  \"uid\": \"SimpleSIR_PetriNetClassic\",\n  \"root\": \"B:sir\",\n  \"types\": null,\n  \"literals\": null,\n  \"junctions\": [\n    {\n      \"syntax\": \"Junction\",\n      \"type\": \"State\",\n      \"name\": \"S\",\n      \"metadata\": null,\n      \"value\": {\n        \"type\": \"Integer\",\n        \"syntax\": \"Literal\",\n        \"value\": {\n          \"syntax\": \"Val\",\n          \"val\": \"997\"\n        },\n        \"metadata\": null\n      },\n      \"value_type\": \"Integer\",\n      \"uid\": \"J:S\"\n    },\n    {\n      \"syntax\": \"Junction\",\n      \"type\": \"State\",\n      \"name\": \"I\",\n      \"metadata\": null,\n      \"value\": {\n        \"type\": \"Integer\",\n        \"syntax\": \"Literal\",\n        \"value\": {\n          \"syntax\": \"Val\",\n          \"val\": \"3\"\n        },\n        \"metadata\": null\n      },\n      \"value_type\": \"Integer\",\n      \"uid\": \"J:I\"\n    },\n    {\n      \"syntax\": \"Junction\",\n      \"type\": \"State\",\n      \"name\": \"R\",\n      \"metadata\": null,\n      \"value\": {\n        \"type\": \"Integer\",\n        \"syntax\": \"Literal\",\n        \"value\": {\n          \"syntax\": \"Val\",\n          \"val\": \"0\"\n        },\n        \"metadata\": null\n      },\n      \"value_type\": \"Integer\",\n      \"uid\": \"J:R\"\n    },\n    {\n      \"syntax\": \"Junction\",\n      \"type\": \"Rate\",\n      \"name\": \"beta\",\n      \"metadata\": null,\n      \"value\": {\n        \"type\": \"Real\",\n        \"syntax\": \"Literal\",\n        \"value\": {\n          \"syntax\": \"Val\",\n          \"val\": \"0.0004\"\n        },\n        \"metadata\": null\n      },\n      \"value_type\": \"Real\",\n      \"uid\": \"J:beta\"\n    },\n    {\n      \"syntax\": \"Junction\",\n      \"type\": \"Rate\",\n      \"name\": \"gamma\",\n      \"metadata\": null,\n      \"value\": {\n        \"type\": \"Real\",\n        \"syntax\": \"Literal\",\n        \"value\": {\n          \"syntax\": \"Val\",\n          \"val\": \"0.04\"\n        },\n        \"metadata\": null\n      },\n      \"value_type\": \"Real\",\n      \"uid\": \"J:gamma\"\n    }\n  ],\n  \"ports\": null,\n  \"wires\": [\n    {\n      \"syntax\": \"Wire\",\n      \"type\": null,\n      \"name\": null,\n      \"metadata\": null,\n      \"value\": null,\n      \"value_type\": null,\n      \"uid\": \"W:S.beta\",\n      \"src\": \"J:S\",\n      \"tgt\": \"J:beta\"\n    },\n    {\n      \"syntax\": \"Wire\",\n      \"type\": null,\n      \"name\": null,\n      \"metadata\": null,\n      \"value\": null,\n      \"value_type\": null,\n      \"uid\": \"W:beta.I1\",\n      \"src\": \"J:beta\",\n      \"tgt\": \"J:I\"\n    },\n    {\n      \"syntax\": \"Wire\",\n      \"type\": null,\n      \"name\": null,\n      \"metadata\": null,\n      \"value\": null,\n      \"value_type\": null,\n      \"uid\": \"W:beta.I2\",\n      \"src\": \"J:beta\",\n      \"tgt\": \"J:I\"\n    },\n    {\n      \"syntax\": \"Wire\",\n      \"type\": null,\n      \"name\": null,\n      \"metadata\": null,\n      \"value\": null,\n      \"value_type\": null,\n      \"uid\": \"W:I.beta\",\n      \"src\": \"J:I\",\n      \"tgt\": \"J:beta\"\n    },\n    {\n      \"syntax\": \"Wire\",\n      \"type\": null,\n      \"name\": null,\n      \"metadata\": null,\n      \"value\": null,\n      \"value_type\": null,\n      \"uid\": \"W:I.gamma\",\n      \"src\": \"J:I\",\n      \"tgt\": \"J:gamma\"\n    },\n    {\n      \"syntax\": \"Wire\",\n      \"type\": null,\n      \"name\": null,\n      \"metadata\": null,\n      \"value\": null,\n      \"value_type\": null,\n      \"uid\": \"W:gamma.R\",\n      \"src\": \"J:gamma\",\n      \"tgt\": \"J:R\"\n    }\n  ],\n  \"boxes\": [\n    {\n      \"wires\": [\n        \"W:S.beta\",\n        \"W:beta.I1\",\n        \"W:beta.I2\",\n        \"W:I.beta\",\n        \"W:I.gamma\",\n        \"W:gamma.R\"\n      ],\n      \"boxes\": null,\n      \"junctions\": [\n        \"J:S\",\n        \"J:I\",\n        \"J:R\",\n        \"J:beta\",\n        \"J:gamma\"\n      ],\n      \"syntax\": \"Relation\",\n      \"type\": null,\n      \"name\": \"sir\",\n      \"metadata\": null,\n      \"uid\": \"B:sir\",\n      \"ports\": null\n    }\n  ],\n  \"variables\": null\n}",
#     "type": "gromet-pnc"
#   }
# }

# %%[markdown]
# # Convert Model

model_def = models[3]
del model_def['name'], model_def['description']

response_body = requests.post(DONU_ENDPOINT, json = {'command': 'convert-model', 'definition': model_def, 'dest-type': 'diff-eqs'}).json()

# %%
print(f"{json.dumps(response_body, indent = 2)}")

# {
#   "status": "success",
#   "result": "parameter beta = 0.4\nparameter gamma = 0.04\nparameter i_initial = 3.0\nparameter r_initial = 0.0\nparameter s_initial = 997.0\nlet total_population = S + I + R\nI(0) = i_initial\nR(0) = r_initial\nS(0) = s_initial\nd/dt I = beta * S * I / total_population + -gamma * I\nd/dt R = gamma * I\nd/dt S = -beta * S * I / total_population"
# }

# %%[markdown]
# # Describe Model Interface

model_def = models[18]
__ = [model_def.pop(k, None) for k in ('type', 'source') if k not in model_def.keys()]

response_body = requests.post(DONU_ENDPOINT, json = {'command': 'describe-model-interface', 'definition': model_def}).json()

# %%
# print(f"{json.dumps(response_body, indent = 2)}")

print(f"{'i':>3} | {'Role':<15} | {'Name':10} | {'Metadata':15}")
for k, v in response_body['result'].items():
    __ = [print(f"{i:>3} | {k:<15} | {obj['metadata']['name']:<10} | {obj['metadata'].__repr__()}") for i, obj in enumerate(v)]

#   i | Role            | Name       | Metadata       
#   0 | measures        | I          | {'name': 'I', 'type': 'Integer'}
#   1 | measures        | R          | {'name': 'R', 'type': 'Integer'}
#   2 | measures        | S          | {'name': 'S', 'type': 'Integer'}
#   0 | parameters      | I          | {'name': 'I', 'group': 'Initial State', 'type': 'Integer'}
#   1 | parameters      | R          | {'name': 'R', 'group': 'Initial State', 'type': 'Integer'}
#   2 | parameters      | S          | {'name': 'S', 'group': 'Initial State', 'type': 'Integer'}
#   3 | parameters      | beta       | {'name': 'beta', 'group': 'Rate', 'type': 'Real'}
#   4 | parameters      | gamma      | {'name': 'gamma', 'group': 'Rate', 'type': 'Real'}

# %%[markdown]
# # Simulate

# %%[markdown]
# ## Set Model Parameter Values

parameters = response_body['result']['parameters']

parameters_set = {p['uid']: p['default'] for p in parameters}
parameters_set['J:beta_rate'] = 1.0 / 10.0 / 100
parameters_set['J:gamma_rate'] = 1.0 / 14.0 / 100

print(f"{'i':>3} | {'UID':<15} | {'Name':<15} | {'Default':10} ")
__ = [print(f"{i:>3} | {p['uid']:<15} | {p['metadata']['name']:<15} | {p['default']:<10} ") for i, p in enumerate(parameters)]


# %%[markdown]
# ## Request Simulation

request_body = {
    'command': 'simulate',
    'sim-type': 'gsl',
    'definition': model_def,
    'start': 0,
    'end': 100,
    'step': 0.,
    'parameters': parameters_set
}

response_body = requests.post(DONU_ENDPOINT, json = request_body).json()


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6))
__ = [ax.plot(response_body['result'][0]['times'], v, label = k) for k, v in response_body['result'][0]['values'].items()]
__ = plt.setp(ax, xlabel = 'Times', ylabel = 'Measures')
__ = ax.legend()

# %%
