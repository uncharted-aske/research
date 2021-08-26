# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Interact with Donu API
# * Test SIR (i.e. CHIME) PNC and CHIME+ (i.e. CHIME SVIIvR) PNC models 
# * Find parameter set for baseline scenario (no vaccination)
# * Repeat for 1-2 other scenarios


# %%
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DONU_ENDPOINT = 'https://aske.galois.com/donu/'


# %%[markdown]
# # List Models

response_body = requests.post(DONU_ENDPOINT, json = {'command': 'list-models'}).json()
models = response_body['result']

# %%
print(f"{'i':>3} | {'Model':<70} | {'Type':15}")
__ = [print(f"{i:>3} | {model['source']['model']:<70} | {model['type']:15}") for i, model in enumerate(models)]

# %%[markdown]
# # Simulate SIR PNC model

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 6))

for x, scenario in zip(ax, ('baseline', 'vaccinating-slow', 'vaccinating-fast')):

    model_def = models[25]
    __ = [model_def.pop(k, None) for k in ('type', 'source') if k not in model_def.keys()]
    response_body = requests.post(DONU_ENDPOINT, json = {'command': 'describe-model-interface', 'definition': model_def}).json()


    # print(f"{'i':>3} | {'Role':<10} | {'UID':<23} | {'Name':<48} | {'Default':>7}")
    # for k, v in response_body['result'].items():
    #     for i, obj in enumerate(v):
    #         if 'default' in obj.keys():
    #             j = obj['default']
    #         else:
    #             j = 'None'

    #         print(f"{i:>3} | {k:<10} | {obj['uid'][:20]:<23} | {obj['metadata']['name']:<48} | {j:>7}")


    measures = response_body['result']['measures']
    measures_metadata = {p['uid']: p['metadata'] for p in measures}

    parameters = response_body['result']['parameters']
    parameters_set = {p['uid']: p['default'] for p in parameters}


    gamma = 1.0 / 14.0
    beta = 2.0 ** (1.0 / 5.0) - 1.0 + gamma
    parameters_set['J:I_init'] = 0.05
    parameters_set['J:R_init'] = 0.0
    parameters_set['J:S_init'] = 1.0 - (parameters_set['J:I_init'] + parameters_set['J:R_init'])
    parameters_set['J:gamma_rate'] = gamma

    if scenario != None:
        parameters_set['J:beta_rate'] = beta
        


    request_body = {
        'command': 'simulate',
        'sim-type': 'gsl',
        'definition': model_def,
        'start': 0,
        'end': 120,
        'step': 1,
        'parameters': parameters_set
    }
    response_body = requests.post(DONU_ENDPOINT, json = request_body).json()
    measures_baseline = response_body['result'][0]


    __ = [x.plot(measures_baseline['times'], v, label = measures_metadata[k]['name']) for k, v in measures_baseline['values'].items()]
    __ = plt.setp(x, xlabel = 'Time (days)', ylabel = 'Measures', title = f"{model_def['source']['model']}")

    if scenario == 'vaccinating-slow':
        __ = x.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.1), ncol = 3)


fig.savefig(f'../figures/scenarios_SIR_PNC.png', dpi = 150)


# %%[markdown]
# # Simulate SVIIVR PNC model

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 6))

for x, scenario in zip(ax, ('baseline', 'vaccinating-slow', 'vaccinating-fast')):

    model_def = models[17]
    __ = [model_def.pop(k, None) for k in ('type', 'source') if k not in model_def.keys()]
    response_body = requests.post(DONU_ENDPOINT, json = {'command': 'describe-model-interface', 'definition': model_def}).json()


    # print(f"{'i':>3} | {'Role':<10} | {'UID':<23} | {'Name':<48} | {'Default':>7}")
    # for k, v in response_body['result'].items():
    #     for i, obj in enumerate(v):
    #         if 'default' in obj.keys():
    #             j = obj['default']
    #         else:
    #             j = 'None'

    #         print(f"{i:>3} | {k:<10} | {obj['uid'][:20]:<23} | {obj['metadata']['name']:<48} | {j:>7}")


    measures = response_body['result']['measures']
    measures_metadata = {p['uid']: p['metadata'] for p in measures}

    parameters = response_body['result']['parameters']
    parameters_set = {p['uid']: p['default'] for p in parameters}

    if True:
        gamma = 1.0 / 14.0
        beta = 2.0 ** (1.0 / 5.0) - 1.0 + gamma
        parameters_set['J:I_U_init'] = 0.05
        parameters_set['J:I_V_init'] = 0.0
        parameters_set['J:R_init'] = 0.0
        parameters_set['J:V_init'] = 0.0
        parameters_set['J:rec_u_rate'] = gamma
        parameters_set['J:rec_v_rate'] = gamma
        parameters_set['J:inf_uu_rate'] = beta
        parameters_set['J:inf_uv_rate'] = 0.05
        parameters_set['J:inf_vu_rate'] = parameters_set['J:inf_uv_rate']
        parameters_set['J:inf_vv_rate'] = 0.0

    if scenario == 'baseline':
        parameters_set['J:vax_rate'] = 0.0

    elif scenario == 'vaccinating-slow':
        parameters_set['J:vax_rate'] = 0.01
        
    elif scenario == 'vaccinating-fast':
        parameters_set['J:vax_rate'] = 0.05

    parameters_set['J:S_init'] = 1.0 - (parameters_set['J:V_init'] + parameters_set['J:I_U_init'] + parameters_set['J:I_V_init'])


    request_body = {
        'command': 'simulate',
        'sim-type': 'aj',
        'definition': model_def,
        'start': 0,
        'end': 120,
        'step': 1,
        'parameters': parameters_set
    }
    response_body = requests.post(DONU_ENDPOINT, json = request_body).json()
    measures_baseline = response_body['result'][0]



    __ = [x.plot(measures_baseline['times'], v, label = measures_metadata[k]['name']) for k, v in measures_baseline['values'].items()]
    __ = plt.setp(x, xlabel = 'Time (days)', ylabel = 'Measures', title = f"{model_def['source']['model']}")

    if scenario == 'vaccinating-slow':
        __ = x.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.1), ncol = 3)


fig.savefig(f'../figures/scenarios_SVIIR_PNC.png', dpi = 150)

# %%[markdown]
# # Fit Scenarios from Florida

# %%
# Get dataset

datasets = requests.post(DONU_ENDPOINT, json = {'command': 'list-datasets'}).json()['result']
dataset = requests.post(DONU_ENDPOINT, json = {'command': 'get-dataset', 'source': datasets[9]['source']}).json()['result']

# %%
plt.plot(dataset['columns'][0]['values'], dataset['columns'][1]['values'])


# %%
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))

for x, scenario in zip(ax, ('sir', 'sviivr')):

    if scenario == 'sir':

        model_def = models[25]
        __ = [model_def.pop(k, None) for k in ('type', 'source') if k not in model_def.keys()]
        response_body = requests.post(DONU_ENDPOINT, json = {'command': 'describe-model-interface', 'definition': model_def}).json()

        measures = response_body['result']['measures']
        measures_metadata = {p['uid']: p['metadata'] for p in measures}

        parameters = response_body['result']['parameters']
        parameters_set = {p['uid']: p['default'] for p in parameters}

        parameters_set = {
            "J:beta_rate": 1.7221469058282074e-7,
            "J:gamma_rate": 0.04954677073300576,
            "J:S_init": 720130.5394121626,
            "J:I_init":6368.2763686775015,
            "J:R_init":9.999999999999996e-11
        }

        
    if scenario == 'sviivr':
        
        model_def = models[17]
        __ = [model_def.pop(k, None) for k in ('type', 'source') if k not in model_def.keys()]
        response_body = requests.post(DONU_ENDPOINT, json = {'command': 'describe-model-interface', 'definition': model_def}).json()

        measures = response_body['result']['measures']
        measures_metadata = {p['uid']: p['metadata'] for p in measures}

        parameters = response_body['result']['parameters']
        parameters_set = {p['uid']: p['default'] for p in parameters}

        parameters_set = {
            "J:S_init": 874270.0703645574,
            "J:I_U_init": 2657.7553778002,
            "J:V_init": 45.90865845686915,
            "J:R_init": 0.10000000000000002,
            "J:I_V_init": 1.5201678216286594,
            "J:rec_v_rate": 2.3526228492762642e-5,
            "J:inf_vv_rate": 2.6041225138605557e-8,
            "J:inf_vu_rate": 2.8956286100930364e-9,
            "J:inf_uu_rate": 1.466296766065254e-7,
            "J:inf_uv_rate": 1.1577435141227581e-8,
            "J:vax_rate": 0.011287926731402404,
            "J:rec_u_rate": 1.7081993411296042e-5
        }


    request_body = {
        'command': 'simulate',
        'sim-type': 'gsl',
        'definition': model_def,
        'start': 0,
        'end': 120,
        'step': 1,
        'parameters': parameters_set
    }
    response_body = requests.post(DONU_ENDPOINT, json = request_body).json()
    measures_baseline = response_body['result'][0]


    __ = [x.plot(measures_baseline['times'], v, label = measures_metadata[k]['name']) for k, v in measures_baseline['values'].items()]
    

    if scenario == 'sir':
        __ = plt.setp(x, ylabel = 'Measures')
    
    __ = plt.setp(x, xlabel = 'Time (days)', title = f"{model_def['source']['model']}")
    __ = x.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.1), ncol = 3)


    # Manifold
    if scenario == 'sir':
        manifold = parameters_set['J:beta_rate'] * np.array(measures_baseline['values']['J:S']) * np.array(measures_baseline['values']['J:I'])
    
    else:
        manifold = parameters_set['J:inf_uu_rate'] * np.array(measures_baseline['values']['J:S']) * np.array(measures_baseline['values']['J:I_U']) 
        manifold += parameters_set['J:inf_uv_rate'] * np.array(measures_baseline['values']['J:S']) * np.array(measures_baseline['values']['J:I_V'])
        manifold += parameters_set['J:inf_vu_rate'] * np.array(measures_baseline['values']['J:S']) * np.array(measures_baseline['values']['J:I_U'])
        manifold += parameters_set['J:inf_vv_rate'] * np.array(measures_baseline['values']['J:S']) * np.array(measures_baseline['values']['J:I_V'])
        # S*I_U*inf_uu + S*I_V*inf_uv + V*I_U*inf_vu + S*I_V*inf_vv

    y = x.twinx()
    y.plot(dataset['columns'][0]['values'][:-26], dataset['columns'][1]['values'][26:], color = 'k', label = f"{dataset['description']}")
    y.plot(measures_baseline['times'], manifold, color = 'k', linewidth = 2, linestyle = '--', label = 'd I_inflow / dt')
    __ = y.legend(loc = 'upper right')
    __ = plt.setp(y, ylabel = 'Data')
    


fig.savefig(f'../figures/scenarios_SIR_PNC_fitting.png', dpi = 150)


# %%
