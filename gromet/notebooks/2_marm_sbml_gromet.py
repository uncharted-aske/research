# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Explore the MARM model in SBML format
# * Compare it with the GroMEt generated by EMMAA and that by Clay

# %%
import os
import json
import requests
import xml.etree.ElementTree as ET

# %%
# https://github.com/indralab/emmaa/raw/master/models/marm_model/gromet_2021-06-07-17-20-49.json


# %%[markdown]
# # Get SBML file of MARM model

url = "https://emmaa.s3.amazonaws.com/exports/marm_model/sbml_2021-05-11-18-35-46.sbml"
local_dir = "../data/emmaa"

# %%
with requests.get(url, stream = True) as r:

    r.raise_for_status()

    with open(local_dir + "/" + url.split("/")[-1], "wb") as f:
        
        for chunk in r.iter_content(chunk_size = 1024):

            f.write(chunk)


f = r = None
del f, r

# %%

tree = ET.parse(local_dir + "/" + url.split("/")[-1])
root = tree.getroot()


model_sbml = {}
model_sbml["sbml"] = root.attrib
model_sbml = {**model_sbml, **root.getchildren()[0].attrib}

for l in root.getchildren()[0].getchildren():

    k = l.tag.split("}")[-1]
    model_sbml[k] = []

    for i, m in enumerate(l.getchildren()):

        model_sbml[k].append(m.attrib)

        if k == "listOfInitialAssignments":

            model_sbml[k][i]['ci'] = m.getchildren()[0].getchildren()[0].text.replace(" ", "")

        if k == "listOfReactions":

            for n in m.getchildren():

                kk = n.tag.split("}")[-1]

                if kk != "kineticlaw":
                    model_sbml[k][i][kk] = [nn.attrib for nn in n.getchildren()]
                
                else:
                    model_sbml[k][i][kk] = n


# %%


