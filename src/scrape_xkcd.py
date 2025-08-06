# %% HEADER
# Get explain xkcd data
# TODO: Code some general inspections
# TODO: Check fill rate for each explanation, title and title text

# %% IMPORTS
from xkcd import get_explain_xkcd
from utils import db_utils
from time import sleep
from tqdm import tqdm
import pandas as pd

# %% INPUTS
xkcd_id_start = 1
xkcd_id_end = 3124
db_path = '../data/relevant_xkcd.db'

# %% FUNCTIONS


# %% MAIN
for xkcd_id in tqdm(range(xkcd_id_start, xkcd_id_end+1), desc='Retrieving and storing xkcd data...'):
    # Get xkcd explanation
    xkcd_explained = get_explain_xkcd(xkcd_id)
    
    # Add to database
    db_utils.insert_xkcd_properties_into_db(db_path, xkcd_explained)
    db_utils.insert_xkcd_explained_into_db(db_path, xkcd_explained)
    
    # Don't hammer the server (too much)
    sleep(5)  
    
# %% CHECK IT OUT
# Get tables and inspect them
xkcd_properties = pd.DataFrame(db_utils.get_xkcd_properties(db_path))
xkcd_explanations = pd.DataFrame(db_utils.get_xkcd_explained(db_path))

# %%
