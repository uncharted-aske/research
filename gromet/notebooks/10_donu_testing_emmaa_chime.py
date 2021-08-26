# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Interact with Donu API
# * Test EMMAA Covid-19 PNC model
# * Test function-network CHIME model

# %%
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DONU_ENDPOINT = 'https://aske.galois.com/donu/'


# %%[markdown]
# # List Models

models = requests.post(DONU_ENDPOINT, json = {'command': 'list-models'}).json()['result']

# %%
print(f"{'i':>3} | {'Model':<70} | {'Type':15}")
__ = [print(f"{i:>3} | {model['source']['model']:<70} | {model['type']:15}") for i, model in enumerate(models)]

# %%[markdown]
# # Simulate EMMAA COVID-19 Inflammasome Model

# %%[markdown]
# ## Describe Model Interface

model_def = models[19]
__ = [model_def.pop(k, None) for k in ('type', 'source') if k not in model_def.keys()]

response_body = requests.post(DONU_ENDPOINT, json = {'command': 'describe-model-interface', 'definition': model_def}).json()

# %%
# print(f"{json.dumps(response_body, indent = 2)}")

print(f"{'i':>3} | {'Role':<10} | {'UID':<23} | {'Name':<48} | {'Default':>7}")
for k, v in response_body['result'].items():
    for i, obj in enumerate(v):
        if 'default' in obj.keys():
            j = obj['default']
        else:
            j = 'None'

        print(f"{i:>3} | {k:<10} | {obj['uid'][:20] + '...':<23} | {obj['metadata']['name']:<48} | {j:>7}")

# %%
#   i | Role       | UID                     | Name                                             | Default
#   0 | measures   | J:CASP1(activity='ac... | CASP1(activity='active')                         |    None
#   1 | measures   | J:CASP1(activity='in... | CASP1(activity='inactive')                       |    None
#   2 | measures   | J:E()...                | E()                                              |    None
#   3 | measures   | J:HMOX1()...            | HMOX1()                                          |    None
#   4 | measures   | J:IKBKB(activity='in... | IKBKB(activity='inactive')                       |    None
#   5 | measures   | J:IL18(activity='act... | IL18(activity='active')                          |    None
#   6 | measures   | J:IL18(activity='ina... | IL18(activity='inactive')                        |    None
#   7 | measures   | J:IL1B(activity='act... | IL1B(activity='active')                          |    None
#   8 | measures   | J:IL1B(activity='ina... | IL1B(activity='inactive')                        |    None
#   9 | measures   | J:NADPH()...            | NADPH()                                          |    None
#  10 | measures   | J:NADP___(activity='... | NADP___(activity='active')                       |    None
#  11 | measures   | J:NADP___(activity='... | NADP___(activity='inactive')                     |    None
#  12 | measures   | J:NFKB1(activity='ac... | NFKB1(activity='active')                         |    None
#  13 | measures   | J:NFKB1(activity='in... | NFKB1(activity='inactive')                       |    None
#  14 | measures   | J:NLRP3(activity='ac... | NLRP3(activity='active')                         |    None
#  15 | measures   | J:NLRP3(activity='in... | NLRP3(activity='inactive')                       |    None
#  16 | measures   | J:Orf9b()...            | Orf9b()                                          |    None
#  17 | measures   | J:PYCARD(activity='a... | PYCARD(activity='active')                        |    None
#  18 | measures   | J:PYCARD(activity='i... | PYCARD(activity='inactive')                      |    None
#  19 | measures   | J:Pralnacasan()...      | Pralnacasan()                                    |    None
#  20 | measures   | J:Proteasome()...       | Proteasome()                                     |    None
#  21 | measures   | J:SUGT1(activity='ac... | SUGT1(activity='active')                         |    None
#  22 | measures   | J:SUGT1(activity='in... | SUGT1(activity='inactive')                       |    None
#  23 | measures   | J:TRAF3(activity='ac... | TRAF3(activity='active')                         |    None
#  24 | measures   | J:TRAF3(activity='in... | TRAF3(activity='inactive')                       |    None
#  25 | measures   | J:TXN(activity='acti... | TXN(activity='active')                           |    None
#  26 | measures   | J:TXN(activity='inac... | TXN(activity='inactive')                         |    None
#  27 | measures   | J:TXNIP(activity='ac... | TXNIP(activity='active')                         |    None
#  28 | measures   | J:TXNIP(activity='in... | TXNIP(activity='inactive')                       |    None
#  29 | measures   | J:_E__3_tosylacrylon... | _E__3_tosylacrylonitrile()                       |    None
#  30 | measures   | J:biliverdin(activit... | biliverdin(activity='active')                    |    None
#  31 | measures   | J:biliverdin(activit... | biliverdin(activity='inactive')                  |    None
#  32 | measures   | J:calcium_2__(activi... | calcium_2__(activity='active')                   |    None
#  33 | measures   | J:calcium_2__(activi... | calcium_2__(activity='inactive')                 |    None
#  34 | measures   | J:carbon_monoxide(ac... | carbon_monoxide(activity='active')               |    None
#  35 | measures   | J:carbon_monoxide(ac... | carbon_monoxide(activity='inactive')             |    None
#  36 | measures   | J:dioxygen()...         | dioxygen()                                       |    None
#  37 | measures   | J:heme()...             | heme()                                           |    None
#  38 | measures   | J:hyaluronic_acid()...  | hyaluronic_acid()                                |    None
#  39 | measures   | J:iron_2__(activity=... | iron_2__(activity='active')                      |    None
#  40 | measures   | J:iron_2__(activity=... | iron_2__(activity='inactive')                    |    None
#  41 | measures   | J:mitochondrion()...    | mitochondrion()                                  |    None
#  42 | measures   | J:p3a()...              | p3a()                                            |    None
#  43 | measures   | J:parthenolide()...     | parthenolide()                                   |    None
#  44 | measures   | J:pattern_recognitio... | pattern_recognition_receptor_signaling_pathway() |    None
#  45 | measures   | J:reactive_oxygen_sp... | reactive_oxygen_species(activity='active')       |    None
#  46 | measures   | J:reactive_oxygen_sp... | reactive_oxygen_species(activity='inactive')     |    None
#  47 | measures   | J:water(activity='ac... | water(activity='active')                         |    None
#  48 | measures   | J:water(activity='in... | water(activity='inactive')                       |    None
#   0 | parameters | J:CASP1(activity='ac... | CASP1(activity='active')                         |       0
#   1 | parameters | J:CASP1(activity='in... | CASP1(activity='inactive')                       |   10000
#   2 | parameters | J:E()_init...           | E()                                              |      10
#   3 | parameters | J:HMOX1()_init...       | HMOX1()                                          |      10
#   4 | parameters | J:IKBKB(activity='in... | IKBKB(activity='inactive')                       |   10000
#   5 | parameters | J:IL18(activity='act... | IL18(activity='active')                          |       0
#   6 | parameters | J:IL18(activity='ina... | IL18(activity='inactive')                        |   10000
#   7 | parameters | J:IL1B(activity='act... | IL1B(activity='active')                          |       0
#   8 | parameters | J:IL1B(activity='ina... | IL1B(activity='inactive')                        |   10000
#   9 | parameters | J:NADPH()_init...       | NADPH()                                          |      10
#  10 | parameters | J:NADP___(activity='... | NADP___(activity='active')                       |       0
#  11 | parameters | J:NADP___(activity='... | NADP___(activity='inactive')                     |   10000
#  12 | parameters | J:NFKB1(activity='ac... | NFKB1(activity='active')                         |       0
#  13 | parameters | J:NFKB1(activity='in... | NFKB1(activity='inactive')                       |   10000
#  14 | parameters | J:NLRP3(activity='ac... | NLRP3(activity='active')                         |       0
#  15 | parameters | J:NLRP3(activity='in... | NLRP3(activity='inactive')                       |   10000
#  16 | parameters | J:Orf9b()_init...       | Orf9b()                                          |   10000
#  17 | parameters | J:PYCARD(activity='a... | PYCARD(activity='active')                        |       0
#  18 | parameters | J:PYCARD(activity='i... | PYCARD(activity='inactive')                      |   10000
#  19 | parameters | J:Pralnacasan()_init... | Pralnacasan()                                    |       0
#  20 | parameters | J:Proteasome()_init...  | Proteasome()                                     |   10000
#  21 | parameters | J:SUGT1(activity='ac... | SUGT1(activity='active')                         |       0
#  22 | parameters | J:SUGT1(activity='in... | SUGT1(activity='inactive')                       |   10000
#  23 | parameters | J:TRAF3(activity='ac... | TRAF3(activity='active')                         |       0
#  24 | parameters | J:TRAF3(activity='in... | TRAF3(activity='inactive')                       |   10000
#  25 | parameters | J:TXN(activity='acti... | TXN(activity='active')                           |       0
#  26 | parameters | J:TXN(activity='inac... | TXN(activity='inactive')                         |   10000
#  27 | parameters | J:TXNIP(activity='ac... | TXNIP(activity='active')                         |       0
#  28 | parameters | J:TXNIP(activity='in... | TXNIP(activity='inactive')                       |   10000
#  29 | parameters | J:_E__3_tosylacrylon... | _E__3_tosylacrylonitrile()                       |       0
#  30 | parameters | J:biliverdin(activit... | biliverdin(activity='active')                    |       0
#  31 | parameters | J:biliverdin(activit... | biliverdin(activity='inactive')                  |   10000
#  32 | parameters | J:calcium_2__(activi... | calcium_2__(activity='active')                   |       0
#  33 | parameters | J:calcium_2__(activi... | calcium_2__(activity='inactive')                 |   10000
#  34 | parameters | J:carbon_monoxide(ac... | carbon_monoxide(activity='active')               |       0
#  35 | parameters | J:carbon_monoxide(ac... | carbon_monoxide(activity='inactive')             |   10000
#  36 | parameters | J:dioxygen()_init...    | dioxygen()                                       |      10
#  37 | parameters | J:heme()_init...        | heme()                                           |      10
#  38 | parameters | J:hyaluronic_acid()_... | hyaluronic_acid()                                |      10
#  39 | parameters | J:iron_2__(activity=... | iron_2__(activity='active')                      |       0
#  40 | parameters | J:iron_2__(activity=... | iron_2__(activity='inactive')                    |   10000
#  41 | parameters | J:kf_3t_act_1:1_rate... | kf_3t_act_1                                      |   1e-06
#  42 | parameters | J:kf__n_act_1:1_rate... | kf__n_act_1                                      |   1e-06
#  43 | parameters | J:kf_ci_act_1:1_rate... | kf_ci_act_1                                      |   1e-06
#  44 | parameters | J:kf_ci_act_2:1_rate... | kf_ci_act_2                                      |   1e-06
#  45 | parameters | J:kf_cn_act_1:1_rate... | kf_cn_act_1                                      |   1e-06
#  46 | parameters | J:kf_cr_act_1:1_rate... | kf_cr_act_1                                      |   1e-06
#  47 | parameters | J:kf_cs_act_1:1_rate... | kf_cs_act_1                                      |   1e-06
#  48 | parameters | J:kf_db_act_1:1_rate... | kf_db_act_1                                      |   1e-06
#  49 | parameters | J:kf_dc_act_1:1_rate... | kf_dc_act_1                                      |   1e-06
#  50 | parameters | J:kf_di_act_1:1_rate... | kf_di_act_1                                      |   1e-06
#  51 | parameters | J:kf_dn_act_1:1_rate... | kf_dn_act_1                                      |   1e-06
#  52 | parameters | J:kf_dw_act_1:1_rate... | kf_dw_act_1                                      |   1e-06
#  53 | parameters | J:kf_ec_act_1:1_rate... | kf_ec_act_1                                      |   1e-06
#  54 | parameters | J:kf_hb_act_1:1_rate... | kf_hb_act_1                                      |   1e-06
#  55 | parameters | J:kf_hb_act_2:1_rate... | kf_hb_act_2                                      |   1e-06
#  56 | parameters | J:kf_hc_act_1:1_rate... | kf_hc_act_1                                      |   1e-06
#  57 | parameters | J:kf_hc_act_2:1_rate... | kf_hc_act_2                                      |   1e-06
#  58 | parameters | J:kf_hi_act_1:1_rate... | kf_hi_act_1                                      |   1e-06
#  59 | parameters | J:kf_hi_act_2:1_rate... | kf_hi_act_2                                      |   1e-06
#  60 | parameters | J:kf_hn_act_1:1_rate... | kf_hn_act_1                                      |   1e-06
#  61 | parameters | J:kf_hn_act_2:1_rate... | kf_hn_act_2                                      |   1e-06
#  62 | parameters | J:kf_hn_act_3:1_rate... | kf_hn_act_3                                      |   1e-06
#  63 | parameters | J:kf_hs_act_1:1_rate... | kf_hs_act_1                                      |   1e-06
#  64 | parameters | J:kf_hw_act_1:1_rate... | kf_hw_act_1                                      |   1e-06
#  65 | parameters | J:kf_hw_act_2:1_rate... | kf_hw_act_2                                      |   1e-06
#  66 | parameters | J:kf_mr_act_1:1_rate... | kf_mr_act_1                                      |   1e-06
#  67 | parameters | J:kf_nb_act_1:1_rate... | kf_nb_act_1                                      |   1e-06
#  68 | parameters | J:kf_nc_act_1:1_rate... | kf_nc_act_1                                      |   1e-06
#  69 | parameters | J:kf_ni_act_1:1_rate... | kf_ni_act_1                                      |   1e-06
#  70 | parameters | J:kf_ni_act_2:1_rate... | kf_ni_act_2                                      |   1e-06
#  71 | parameters | J:kf_ni_act_3:1_rate... | kf_ni_act_3                                      |   1e-06
#  72 | parameters | J:kf_nn_act_1:1_rate... | kf_nn_act_1                                      |   1e-06
#  73 | parameters | J:kf_np_act_1:1_rate... | kf_np_act_1                                      |   1e-06
#  74 | parameters | J:kf_nw_act_1:1_rate... | kf_nw_act_1                                      |   1e-06
#  75 | parameters | J:kf_ot_deg_1:1_rate... | kf_ot_deg_1                                      |   2e-09
#  76 | parameters | J:kf_ot_deg_1:2_rate... | kf_ot_deg_1                                      |   2e-09
#  77 | parameters | J:kf_pc_act_1:1_rate... | kf_pc_act_1                                      |   1e-06
#  78 | parameters | J:kf_pc_act_2:1_rate... | kf_pc_act_2                                      |   1e-06
#  79 | parameters | J:kf_pc_act_3:1_rate... | kf_pc_act_3                                      |   1e-06
#  80 | parameters | J:kf_pi_act_2:1_rate... | kf_pi_act_2                                      |   1e-06
#  81 | parameters | J:kf_pn_act_1:1_rate... | kf_pn_act_1                                      |   1e-06
#  82 | parameters | J:kf_pn_act_2:1_rate... | kf_pn_act_2                                      |   1e-06
#  83 | parameters | J:kf_ps_act_1:1_rate... | kf_ps_act_1                                      |   1e-06
#  84 | parameters | J:kf_pt_act_1:1_rate... | kf_pt_act_1                                      |   1e-06
#  85 | parameters | J:kf_rn_act_1:1_rate... | kf_rn_act_1                                      |   1e-06
#  86 | parameters | J:kf_rs_act_1:1_rate... | kf_rs_act_1                                      |   1e-06
#  87 | parameters | J:kf_rt_act_1:1_rate... | kf_rt_act_1                                      |   1e-06
#  88 | parameters | J:kf_sn_act_1:1_rate... | kf_sn_act_1                                      |   1e-06
#  89 | parameters | J:kf_tn_act_1:1_rate... | kf_tn_act_1                                      |   1e-06
#  90 | parameters | J:kf_tn_act_2:1_rate... | kf_tn_act_2                                      |   1e-06
#  91 | parameters | J:kf_ts_act_1:1_rate... | kf_ts_act_1                                      |   1e-06
#  92 | parameters | J:kf_tt_act_1:1_rate... | kf_tt_act_1                                      |   1e-06
#  93 | parameters | J:mitochondrion()_in... | mitochondrion()                                  |      10
#  94 | parameters | J:p3a()_init...         | p3a()                                            |      10
#  95 | parameters | J:parthenolide()_ini... | parthenolide()                                   |       0
#  96 | parameters | J:pattern_recognitio... | pattern_recognition_receptor_signaling_pathway() |      10
#  97 | parameters | J:reactive_oxygen_sp... | reactive_oxygen_species(activity='active')       |       0
#  98 | parameters | J:reactive_oxygen_sp... | reactive_oxygen_species(activity='inactive')     |   10000
#  99 | parameters | J:water(activity='ac... | water(activity='active')                         |       0
# 100 | parameters | J:water(activity='in... | water(activity='inactive')                       |   10000

# %%[markdown]
# ## Set Model Parameter Values

measures = response_body['result']['measures']
measures_metadata = {p['uid']: p['metadata'] for p in measures}

parameters = response_body['result']['parameters']
parameters_set = {p['uid']: p['default'] for p in parameters}

# %%[markdown]
# ## Request Simulation

request_body = {
    'command': 'simulate',
    'sim-type': 'gsl',
    'definition': model_def,
    'start': 0,
    'end': 1000,
    'step': 1,
    'parameters': parameters_set
}

response_body = requests.post(DONU_ENDPOINT, json = request_body).json()
measures_baseline = response_body['result'][0]

# %%
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
__ = [ax.plot(measures_baseline['times'], v, label = measures_metadata[k]['name']) for k, v in measures_baseline['values'].items()]
__ = plt.setp(ax, xlabel = 'Times (seconds)', ylabel = 'Measures', title = f"{model_def['source']['model']}")
__ = ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.1), ncol = 3)

fig.savefig(f'../figures/donu_testing_emmaa_baseline.png', dpi = 150)

fig = ax = None
del fig, ax

# %%
# ## Ben's Scenarios

# %%
drug_uids = ['J:Pralnacasan()', 'J:parthenolide()', 'J:_E__3_tosylacrylonitrile()']
drug_init_val = 1e6
parameters_set_scenarios = [{k: drug_init_val if k == (uid + '_init') else v for k, v in parameters_set.items()} for uid in drug_uids]

measures_scenarios = []
for scenario in tqdm(parameters_set_scenarios):

    request_body = {
        'command': 'simulate',
        'sim-type': 'gsl',
        'definition': model_def,
        'start': 0,
        'end': 1000,
        'step': 1,
        'parameters': scenario
    }

    measures_scenarios.append(requests.post(DONU_ENDPOINT, json = request_body).json()['result'][0])


# %%
# 
measure_uids = ["J:IL18(activity='active')", "J:CASP1(activity='active')", "J:NFKB1(activity='active')"]

fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (12, 12))

for i in range(3):
    for j in range(3):

        # Baseline
        __ = ax[i, j].plot(measures_baseline['times'], measures_baseline['values'][measure_uids[i]], label = f"Baseline")

        # Scenarios
        __ = ax[i, j].plot(measures_baseline['times'], measures_scenarios[j]['values'][measure_uids[i]], label = f"Scenario")


        if j == 0:
            __ = plt.setp(ax[i, j], ylabel = f"{measures_metadata[measure_uids[i]]['name']}")
        if i == 0:
            __ = plt.setp(ax[i, j], title = f"{measures_metadata[drug_uids[j]]['name']}")
        if i == 2:
            __ = plt.setp(ax[i, j], xlabel = 'Times (seconds)')


__ = ax[2, 1].legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.4), ncol = 2)

fig.savefig(f'../figures/donu_testing_emmaa_scenarios.png', dpi = 150)

fig = ax = None
del fig, ax

# %%[markdown]
# # Simulate FN CHIME Model

# %%[markdown]
# ## Describe Model Interface

model_def = models[29]
__ = [model_def.pop(k, None) for k in ('type', 'source') if k not in model_def.keys()]

response_body = requests.post(DONU_ENDPOINT, json = {'command': 'describe-model-interface', 'definition': model_def}).json()

# %%
# print(f"{json.dumps(response_body, indent = 2)}")

print(f"{'i':>3} | {'Role':<10} | {'UID':<23} | {'Default':>7} | {'Description':<30}")
for k, v in response_body['result'].items():
    for i, obj in enumerate(v):
        if 'default' in obj.keys():
            j = obj['default']
        else:
            j = 'None'

        if 'Description' in obj['metadata'].keys():
            l = obj['metadata']['Description']
        else:
            l = 'None'

        print(f"{i:>3} | {k:<10} | {obj['uid'][:20]:<23} | {j:>7} | {l:<30}")

# %%[markdown]
# ## Set Model Parameter Values

measures = response_body['result']['measures']
measures_metadata = {p['uid']: p['metadata'] for p in measures}

parameters = response_body['result']['parameters']
parameters_set = {p['uid']: p['default'] for p in parameters}


# %%[markdown]
# ## Request Simulation

request_body = {
    'command': 'simulate',
    'sim-type': 'gsl',
    'definition': model_def,
    'start': 0,
    'end': 1000,
    'step': 1,
    'parameters': parameters_set
}

response_body = requests.post(DONU_ENDPOINT, json = request_body).json()
measures_baseline = response_body['result'][0]

# %%
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 6))
__ = [ax.plot(measures_baseline['times'], v, label = measures_metadata[k]['name']) for k, v in measures_baseline['values'].items()]
__ = plt.setp(ax, xlabel = 'Times (seconds)', ylabel = 'Measures', title = f"{model_def['source']['model']}")
__ = ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.1), ncol = 3)
