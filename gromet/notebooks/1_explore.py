# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Download latest files from AutoMATES/Clay's repository

# %%
import os
import json
import requests


# %%[markdown]
# # Get Latest GroMEt scripts

# %%[markdown]
# ## From AutoMATES

# %%
url_repo = "https://github.com/ml4ai/automates/raw/claytonm/gromet/scripts/gromet"

filenames = [
    "gromet.py",
    "gromet_experiment.py", 
    "gromet_metadata.py",
    # "gromet_validate.py",
    # "example_SimpleSIR_Bilayer.py", 
    "example_SimpleSIR_Bilayer_metadata.py", 
    # "example_SimpleSIR_PetriNetClassic.py",
    "example_SimpleSIR_PetriNetClassic_metadata.py",
    # "example_SimpleSIR_FN.py",
    "example_SimpleSIR_FN_metadata.py",
    "example_SimpleSIR_PrTNet.py",
    "example_emmaaSBML_PetriNetClassic.py",
    # "example_call_ex1.py",
    # "example_cond_ex1.py",
    # "example_toy1.py"
]

local_dir = "../data/ml4ai_repo"

# %%
for filename in filenames:

    url = url_repo + "/" + filename

    with requests.get(url, stream = True) as r:

        r.raise_for_status()

        with open(local_dir + "/" + filename, "wb") as f:
            
            for chunk in r.iter_content(chunk_size = 1024):

                f.write(chunk)

url = url_repo = r = None
del url, url_repo, r


# %%[markdown]
# ## Generate GroMEt files 

# %%
# %run -i "../data/ml4ai_repo/example_SimpleSIR_Bilayer.py"

for filename in filenames:

    f = local_dir + '/' + filename

    try:
        exec("\n".join(open(f).read().split("\n")[1:]))

    except:
        print(f"Error: {f}")

# %%

for filename in os.listdir():

    if filename.split(".")[1] == "json":
        os.replace(filename, local_dir + "/" + filename)


filename = f = None
del filename, f


# %%[markdown]
# ## Read GroMEt Files

gromets = [json.load(open(local_dir + "/" + filename)) for filename in next(os.walk(local_dir))[2] if filename.split(".")[1] == "json"]


# %%[markdown]
# ## From EMMAA

# %%
# url = "https://github.com/indralab/emmaa/raw/master/models/marm_model/gromet_2021-06-07-17-20-49.json"
# url = "https://github.com/dianakolusheva/emmaa/raw/gromet_metadata/models/marm_model/gromet_2021-06-20-17-05-07.json"
url = "https://github.com/dianakolusheva/emmaa/blob/gromet_metadata/models/marm_model/gromet_2021-06-23-17-08-47.json"
filename = url.split("/")[-1]
local_dir = "../data/emmaa"

with requests.get(url, stream = True) as r:

    r.raise_for_status()

    with open(local_dir + "/" + filename, "wb") as f:
        
        for chunk in r.iter_content(chunk_size = 1024):

            f.write(chunk)

url = r = None
del url, r

# %%[markdown]
# ## Read EMMAA GroMEt Files

gromets = gromets + [json.load(open(local_dir + "/" + filename)) for filename in next(os.walk(local_dir))[2] if filename.split(".")[1] == "json"]

# %%[markdown]
# # Validate GroMEt Object

# %%

for gromet in gromets:

    # Generate lists
    objects = {}
    for k in ("variables", "boxes", "wires", "junctions", "ports"):
        objects[k] = {}
        if gromet[k] != None:
            objects[k] = {**objects[k], **{obj['uid']: obj for obj in gromet[k]}}

    objects["nodes"] = {**objects["ports"], **objects["junctions"]}


    # Check wire sources and targets
    for i, wire in objects["wires"].items():
        for k in ('src', 'tgt'):
            if wire[k] not in objects["nodes"]:
                print(f"Error: Wire '{i}' of GroMEt '{gromet['uid']}'' is missing its {k} node.")


    # PNC-specific test: wires cannot connect state/rate junctions to state/rate junctions
    if gromet['type'] == 'PetriNetClassic':
        for i, wire in objects["wires"].items():
            for k in ('State', 'Rate'):
                if (objects["nodes"][wire['src']]['type'] == k) & (objects["nodes"][wire['tgt']]['type'] == k):
                    print(f"Error: Wire '{i}'' of GroMEt '{gromet['uid']}'' is connecting two {k} junctions.")


    # Test if variable states exist in nodes or wires
    for i, variable in objects['variables'].items():
        if set(variable['states']) <= (set(objects["nodes"].keys()) | set(objects["wires"].keys())):
            pass
        else:
            print(f"Error: States of variable '{variable['uid']}' of GroMEt '{gromet['uid']}' are missing from the list of ports and junctions.")


    # Test if parent box of ports exists
    for i, port in objects['ports'].items():
        if port["box"] not in objects["boxes"]:
            print(f"Error: Parent box of port '{port['uid']}' of GroMEt '{gromet['uid']}' is missing from the list of boxes.")


    # Test if children of boxes exist
    for i, box in objects["boxes"].items():
        for k in ("wires", "boxes", "ports", "junctions"):
            if k in box.keys():
                if box[k] != None:
                    if set(box[k]) <= set(objects[k].keys()):
                        pass
                    else:
                        print(f"Error: {k} of box '{box['uid']}' of GroMEt '{gromet['uid']}' are missing from the list of {k}.")


    # Tests specific to the 'tree' attribute of 'Expression' boxes
    # WIP...
    # Need to be recursive...


b = i = k = box = port = wire = gromet = None
del b, i, k, box, port, wire, gromet

# %%
# Error: boxes of box 'B:sir' of GroMEt 'SimpleSIR_PrTNet' are missing from the list of boxes.
# Error: Wire 'W:c1_c1exp.y' of GroMEt 'cond_ex1'' missing its tgt node.
# Error: States of variable 'var3' of GroMEt 'toy1' are missing from the list of ports and junctions.

