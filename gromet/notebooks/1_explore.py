# %%[markdown]
# Author: Nelson Liu 
#
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Content:
# * Download latest files from AutoMATES/Clay's repository

# %%
import numpy as np
import requests

# %%
np.random.seed(0)

# %%
url_repo = "https://github.com/ml4ai/automates/raw/claytonm/gromet/scripts/gromet"

filenames = [
    "gromet.py", 
    "example_SimpleSIR_Bilayer.py", 
    "example_SimpleSIR_PetriNetClassic.py",
    "example_SimpleSIR_FN.py",
    "example_SimpleSIR_PrTNet.py",
    "example_emmaaSBML_PetriNetClassic.py",
    "example_call_ex1.py",
    "example_cond_ex1.py",
    "example_toy1.py",
    "examples_misc_small.py"
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

# %%

