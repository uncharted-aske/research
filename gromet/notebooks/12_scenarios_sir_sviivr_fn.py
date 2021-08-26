# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Interact with Donu API
# * Test SIR (i.e. CHIME) PNC and CHIME+ (i.e. CHIME SVIIvR) FN models 
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
# # Simulate the two CHIME FN models


model_inds = [i for i, model in enumerate(models) if model['source']['model'] in ('CHIME_SIR_Base_variables_gromet_FunctionNetwork-with-metadata--GroMEt.json', 'CHIME_SVIIvR_variables_gromet_FunctionNetwork-with-metadata--GroMEt.json')]


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))

for x, scenario in zip(ax, model_inds):

    model_def = models[scenario]
    __ = [model_def.pop(k, None) for k in ('type', 'source') if k not in model_def.keys()]
    response_body = requests.post(DONU_ENDPOINT, json = {'command': 'describe-model-interface', 'definition': model_def}).json()


    print(f"{'i':>3} | {'Role':<10} | {'UID':<23} | {'Name':<48} | {'Default':>7}")
    for k, v in response_body['result'].items():
        for i, obj in enumerate(v):
            if 'default' in obj.keys():
                j = obj['default']
            else:
                j = 'None'

            print(f"{i:>3} | {k:<10} | {obj['uid'][:20]:<23} | {obj['metadata']['name']:<48} | {j:>7}")


    measures = response_body['result']['measures']
    measures_metadata = {p['uid']: p['metadata'] for p in measures}

    parameters = response_body['result']['parameters']
    # parameters_set = {p['uid']: p['default'] if 'default' in p.keys() else None for p in parameters}


    # CHIME-Base
    if models[scenario]['source']['model'] == 'CHIME_SIR_Base_variables_gromet_FunctionNetwork-with-metadata--GroMEt.json':

        parameters_set = {
            'J:main.i_n': 0.05,
            'J:main.r_n': 0.0,
            'J:main.s_n': 1.0 - 0.05,
            'J:main.i_day': 17.0,
            'J:main.N_p': 3,
            'J:main.infections_days': 14.0,
            # 'J:main.relative_contact_rate': 0.05,
            'J:main.relative_contact_rate': 0.05,
        }

        outputs = [measure['uid'] for measure in measures if 'out' in measure['uid'].split('.')]

    # CHIME-SVIIvR
    if models[scenario]['source']['model'] == 'CHIME_SVIIvR_variables_gromet_FunctionNetwork-with-metadata--GroMEt.json':

        parameters_set = {
            'J:main.i_day': 17.0,
            'J:main.N_p': 2,
            'J:main.infectious_days_unvaccinated': 14,
            'J:main.infectious_days_vaccinated': 10,
            'J:main.vaccination_rate': 0.02,
            'J:main.vaccine_efficacy': 0.85,
            # 'J:main.relative_contact_rate': 0.45,
            'J:main.relative_contact_rate': 0.45,
            # 'J:main.s_n': 1000,
            'J:main.s_n': 1.0 - 0.05,
            'J:main.v_n': 0,
            # 'J:main.i_n': 1,
            'J:main.i_n': 0.05,
            'J:main.i_v_n': 0,
            'J:main.r_n': 0
        }

        outputs = ['P:main.out.S', 'P:main.out.I', 'P:main.out.R', 'P:main.out.E']


    request_body = {
        'command': 'simulate',
        'sim-type': 'automates',
        'definition': model_def,
        'start': 0,
        'end': 120,
        'step': 1,
        'domain_parameter': 'J:main.n_days',
        'parameters': parameters_set,
        'outputs': outputs,
    }
    response_body = requests.post(DONU_ENDPOINT, json = request_body).json()

    print(request_body)

    try:
        measures_baseline = response_body['result'][0]

        __ = [x.plot(measures_baseline['domain_parameter'], v, label = measures_metadata[k]['name']) for k, v in measures_baseline['values'].items()]
    
    except:
        pass
    
    finally:
        __ = plt.setp(x, xlabel = 'Time (days)', ylabel = 'Measures', title = f"{model_def['source']['model']}")
        __ = x.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.1), ncol = 3)


fig.savefig(f'../figures/scenarios_SIR_SVIIR_FN.png', dpi = 150)


# %%
