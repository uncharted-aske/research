# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Load the model comparison examples from Algebraic Julia (James and Andrew)
# * Plot results

# %%
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, Optional

# %%[markdown]
# # Load Model Data

# %%

path = "../data/august_2021_demo_repo/Simple_SIR/SimpleSIR_metadata_gromet_PetriNetClassic.json"
with open(path, 'r') as f:
    model_sir = json.load(f)


path = "../data/august_2021_demo_repo/AlgebraicJulia_models/chime+.json"
with open(path, 'r') as f:
    model_chimep = json.load(f)


path = "../data/august_2021_demo_repo/AlgebraicJulia_models/model_comparisons/sir_chime+.json"
with open(path, 'r') as f:
    map_sir_chimep = json.load(f)


f = None
del f

# %%[markdown]
# # Parse GroMEts into NX Objects
def generate_nx_obj(gromet: Dict) -> Any:

    G = nx.MultiDiGraph(uid = gromet['uid'], type = gromet['type'], name = gromet['name'])

    G.add_nodes_from(
        [(j['uid'], {
            'model_uid': gromet['uid'], 
            'uid': j['uid'],
            'name': j['name'], 
            'type': j['type']
        }) for j in gromet['junctions']]
    )

    G.add_edges_from(
        [(w['src'], w['tgt'], {
            'model_uid': gromet['uid'], 'uid': w['uid']
        }) 
        for w in gromet['wires']], weight = 1
    )

    return G

# %%
def compare_graphs(G1: Any, G2: Any, map: Optional[Dict], leg_id: Optional[int] = 0, rename: Tuple = (None, None), plot: bool = False) -> Tuple:

    # Mapping graph
    G_map = nx.MultiDiGraph()
    for rn, g in zip(rename, (G1, G2)):
        G_map.add_nodes_from([(rn + n, g.nodes[n]) for n in g.nodes])
        G_map.add_edges_from([(rn + edge[0], rn + edge[1], g.edges[edge]) for edge in g.edges], weight = 1.0)

    # G1 -> G2 map edges
    if map != None:
        G_map.add_edges_from([(rename[0] + k, rename[1] + v) for k, v in map['legs'][G2.graph['uid']][leg_id].items()], weight = 5.0)

    # Plot bipartite graph
    fig = None
    if plot == True:

        # Colour
        node_col = [0 if G_map.nodes[node]['model_uid'] == G1.graph['uid'] else 1 for node in G_map.nodes]
        edge_col = [
            0 if (G_map.nodes[edge[0]]['model_uid'] == G1.graph['uid']) & (G_map.nodes[edge[1]]['model_uid'] == G1.graph['uid']) 
            else 1 if (G_map.nodes[edge[0]]['model_uid'] == G2.graph['uid']) & (G_map.nodes[edge[1]]['model_uid'] == G2.graph['uid']) 
            else 0.5 for edge in G_map.edges]

        # Size
        node_size = np.array([500 if G_map.nodes[node]['type'] == 'State' else 50 for node in G_map.nodes])

        # Layout
        pos_map = nx.kamada_kawai_layout(G_map, weight = 'weight', center = (0, 0))

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 12))

        nx.draw(
            G_map, pos = pos_map, 
            node_size = node_size,
            node_color = node_col, 
            #cmap = 'tab10', vmin = 0, vmax = 9, 
            edge_color = edge_col, 
            # edge_cmap = 'tab10', edge_vmin = 0, edge_vmax = 9, 
            with_labels = True, 
            ax = ax
        )

        __ = plt.setp(ax, title = f"{G1.graph['uid']} -> {G2.graph['uid']}")

    return (G_map, fig)

# %%

G_sir = generate_nx_obj(gromet = model_sir)
G_chimep = generate_nx_obj(gromet = model_chimep)
G_map, fig = compare_graphs(G1 = G_sir, G2 = G_chimep, map = map_sir_chimep, leg_id = 1, rename = ('SIR ', 'CHIME+ '), plot = True)

fig.savefig('../figures/map_sir_chime+.png', dpi = 150)

# %%
