# %% HEADER
# Calculate TF-IDF vectors for each heading and store in database

# https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/#:~:text=Term%20Frequency%3A%20TF%20of%20a,of%20words%20in%20the%20document.&text=Inverse%20Document%20Frequency%3A%20IDF%20of,corpus%20that%20contain%20the%20term.

# TODO: Consider removing character names from strings
# TODO: Rework comments to nicer language
# TODO: What if semantics return nothing? Then first try destopworded search in titles and such, then literal words
# TODO: Consider a separate calculation for each length of ngram
# TODO: Strip upper case using NER
# TODO: Build in extra step of removing punctuation and stopwords
# TODO: Docstrings
# TODO: Titles Are In This Kind Of Illegible Format

# %% IMPORTS
import re
import string

from nltk.corpus import stopwords
import pandas as pd
from tqdm import tqdm

from utils import db_utils
import xkcd_tfidfing

# %% INPUTS
db_path = "../data/relevant_xkcd.db"
n_gram_max_length = 3

# Which headings to ignore for tf-idf calculation
headings_ignore = ["Trivia"]

# Build split pattern with punctuation and word boundaries for stopwords
stopwords_pattern = (
    r"\b(?:" + "|".join(map(re.escape, stopwords.words("english"))) + r")\b"
)
punctuation_pattern = r'[^\p{L}\p{M}\p{N}\- ]+'
split_pattern = f"({stopwords_pattern})|({punctuation_pattern})"


# %% FUNCTIONS


# %% GET DATA
xkcd_properties_df = db_utils.get_xkcd_properties(db_path)
xkcd_explanations_df = db_utils.get_xkcd_explained(db_path)

# %% CALCULATE TFIDFS AND ADD TO DATABASE
for n_gram_length in tqdm(
    range(1, n_gram_max_length + 1), desc="Calculating tf-idf scores..."
):
    # Get tfidf for titles
    tfidf_title_df = xkcd_tfidfing.get_tfidf(
        xkcd_properties_df, 
        "xkcd_id", 
        "title", 
        n_gram_length=n_gram_length,
        split_pattern=split_pattern
    )
    tfidf_title_df["heading"] = "title"
    print(tfidf_title_df.head())
    db_utils.insert_tfidfs_into_db(db_path, tfidf_title_df)
    
    # Get tfidf for title texts
    tfidf_title_text_df = xkcd_tfidfing.get_tfidf(
        xkcd_properties_df, 
        "xkcd_id", 
        "title_text", 
        n_gram_length=n_gram_length,
        split_pattern=split_pattern
    )
    tfidf_title_text_df["heading"] = "title_text"
    print(tfidf_title_text_df.head())
    db_utils.insert_tfidfs_into_db(db_path, tfidf_title_text_df)

    # Get tfidf for transcripts
    xkcd_transcripts_df = xkcd_explanations_df[
        xkcd_explanations_df["heading"] == "Transcript"
    ]
    xkcd_transcripts_df = xkcd_transcripts_df.groupby(
        ["xkcd_id", "heading"], as_index=False
    ).agg({"text": " ".join})
    tfidf_transcripts_df = xkcd_tfidfing.get_tfidf(
        xkcd_transcripts_df, 
        "xkcd_id", 
        "text", 
        n_gram_length=n_gram_length,
        split_pattern=split_pattern
    )
    tfidf_transcripts_df["heading"] = "transcript"
    print(tfidf_transcripts_df.head())
    db_utils.insert_tfidfs_into_db(db_path, tfidf_transcripts_df)

    # Get tfidf for rest of explanation, minus headings to ignore
    xkcd_rest_of_explanations_df = xkcd_explanations_df[
        ~xkcd_explanations_df["heading"].isin([headings_ignore, "Transcript"])
    ]
    xkcd_rest_of_explanations_df = xkcd_rest_of_explanations_df.groupby(
        ["xkcd_id"], as_index=False
    ).agg({"text": " ".join})
    tfidf_rest_of_explanations_df = xkcd_tfidfing.get_tfidf(
        xkcd_rest_of_explanations_df, 
        "xkcd_id", 
        "text", 
        n_gram_length=n_gram_length,
        split_pattern=split_pattern
    )
    print(tfidf_rest_of_explanations_df.head())
    db_utils.insert_tfidfs_into_db(db_path, tfidf_rest_of_explanations_df)

    # Report on tfidfs
    print(f"For n_gram_length = {n_gram_length}:")
    print(f"{len(tfidf_title_df)=}")
    print(f"{len(tfidf_title_text_df)=}")
    print(f"{len(tfidf_transcripts_df)=}")
    print(f"{len(tfidf_rest_of_explanations_df)=}")
    print()

# %% CHECK OUT THE RESULTS
tfidf_title_df

# %%
# n_gram_length
# Goes fucky on xkcd_id 2849, 2519, 2304, 3054, 1954, 1950, 1683, 1943, 1920
# %%
import sqlite3
conn = sqlite3.connect(db_path)
df_mess = pd.read_sql_query("""
    SELECT 
        * 
    FROM XKCD_PROPERTIES
    WHERE xkcd_id IN (2849, 2519, 2304, 3054, 1954, 1950, 1683, 1943, 1920)
    """, conn)
conn.close()
df_mess

# %%
df = pd.concat([
    df_calculated_tfidf, 
    tfidf_title_text_df])

duplicates = df_calculated_tfidf[
    df_calculated_tfidf.duplicated(
        subset=['xkcd_id', 'token', 'tfidf', 'heading'], 
        keep=False)]
print(duplicates)

# 2822
# %%
tfidf_df.columns
# %%
import sqlite3
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("DELETE FROM XKCD_EXPLAINED_TFIDF")
conn.commit()
cursor.execute("VACUUM")
conn.close()


# %%
import sqlite3
conn = sqlite3.connect(db_path)
df_calculated_tfidf = pd.read_sql_query("SELECT * FROM XKCD_EXPLAINED_TFIDF", conn)
conn.close()
df_calculated_tfidf

# %%
tfidf_title_text_df

# %%
db_utils.insert_tfidfs_into_db(db_path, tfidf_title_text_df)
# %%
stuff = xkcd_explanations_df[
        ~xkcd_explanations_df["heading"].isin([headings_ignore, "Transcript"])
    ]
stuff
# %%
n_gram_length = 2
# Get tfidf for title texts
tfidf_title_text_df = xkcd_tfidfing.get_tfidf(
    xkcd_properties_df, 
    "xkcd_id", 
    "title_text", 
    n_gram_length=n_gram_length,
    split_pattern=split_pattern
)
tfidf_title_text_df["heading"] = "title_text"
db_utils.insert_tfidfs_into_db(db_path, tfidf_title_text_df)

# %%
tfidf_title_text_df
# %%
