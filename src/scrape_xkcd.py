# %% HEADER
# Get explain xkcd data
# TODO: Code some general inspections

# %% IMPORTS
from xkcd import get_explain_xkcd
from utils import db_utils
from time import sleep
from tqdm import tqdm
import pandas as pd

# %% INPUTS
xkcd_id_start = 1
xkcd_id_end = 1756
db_path = '../data/relevant_xkcd.db'

# %% FUNCTIONS


# %% MAIN
xkcd_ids = [2011, 1488, 18, 1245, 29, 1048, 666]

for xkcd_id in tqdm(xkcd_ids, desc='Retrieving and storing xkcd data...'):
    # Get xkcd explanation
    xkcd_explained = get_explain_xkcd(xkcd_id)
    
    # Add to database
    db_utils.insert_xkcd_properties_into_db(db_path, xkcd_explained)
    db_utils.insert_xkcd_explained_into_db(db_path, xkcd_explained)
    
    # Don't hammer the server (too much)
    sleep(1)  
    
# %% CHECK IT OUT
# Get tables and inspect them
xkcd_properties = pd.DataFrame(db_utils.get_xkcd_properties(db_path))
xkcd_explanations = pd.DataFrame(db_utils.get_xkcd_explained(db_path))

# %%
