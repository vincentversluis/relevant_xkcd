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
from requests.exceptions import HTTPError

# %% INPUTS
xkcd_id_start = 1
xkcd_id_end = 3125
db_path = '../data/relevant_xkcd.db'
retries = 10
zzz = 10  # Seconds to sleep between requests
backoff = 60  # Seconds to sleep between retries

# %% MAIN
# Scrape. Be kind to the server with sleep and leave time after HTTPErrors
for retry in range(retries):
    # Get all xkcd ids that haven't been scraped yet
    xkcd_ids_to_scrape = [
        xkcd_id 
        for xkcd_id 
        in range(xkcd_id_start, xkcd_id_end+1) 
        if xkcd_id not in db_utils.get_scraped_xkcd_ids(db_path)
        ]
    
    # Scrape each xkcd
    for xkcd_id in tqdm(xkcd_ids_to_scrape, desc=f'Retrieving and storing xkcd data. Try {retry+1}/{retries}...'):
        # Get xkcd explanation
        try:
            xkcd_explained = get_explain_xkcd(xkcd_id)
            sleep(zzz)  
            # Don't hammer the server (too much)
            # Add to database
            db_utils.insert_xkcd_properties_into_db(db_path, xkcd_explained)
            db_utils.insert_xkcd_explained_into_db(db_path, xkcd_explained)
        except HTTPError as e:
            print(f"HTTPError occurred for xkcd {xkcd_id}: {e}")
            print(f"Backing off for {backoff} seconds...")
            sleep(backoff)
        except Exception as e:
            print(f"Some error occurred for xkcd {xkcd_id}: {e}")
    
# %% CHECK IT OUT
# Get tables and inspect them
xkcd_properties = pd.DataFrame(db_utils.get_xkcd_properties(db_path))
xkcd_explanations = pd.DataFrame(db_utils.get_xkcd_explained(db_path))

# %%
# TODO: Remove dupes if there are any
import arrow
import re

dates = [
    'Jan 11, 2008',
    'January 19th, 2011',
    '31 October 2012'
]
def parse_with_dateutil(date_str):
    try:
        # Clean up ordinal suffixes like '19th'
        dt = parser.parse(date_str)
        return arrow.get(dt)
    except Exception as e:
        print(f"Couldn't parse {date_str} because {e}")
        return None

parsed_dates = [parse_with_dateutil(d) for d in dates]
for parsed_date in parsed_dates:
    print(parsed_date)

# %%
