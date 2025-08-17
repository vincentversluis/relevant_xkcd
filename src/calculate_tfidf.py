# %% HEADER
# Calculate TF-IDF vectors for each heading and store in database

# https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/#:~:text=Term%20Frequency%3A%20TF%20of%20a,of%20words%20in%20the%20document.&text=Inverse%20Document%20Frequency%3A%20IDF%20of,corpus%20that%20contain%20the%20term.

# TODO: Consider removing character names from strings
# TODO: Rework comments to nicer language
# TODO: What if semantics return nothing? Then first try destopworded search in titles and such, then literal words
# TODO: Strip upper case using NER
# TODO: Docstrings
# TODO: Titles Are In This Kind Of Illegible Format - might want to prepreprocess this

# %% IMPORTS
import re

from nltk.corpus import stopwords
import pandas as pd
from tqdm import tqdm

from utils import db_utils
from xkcd_tfidfing import get_tfidf

# %% INPUTS
db_path = "../data/relevant_xkcd.db"
n_gram_max_length = 3

# Build split pattern with punctuation and word boundaries for stopwords
stopwords_pattern = (
    r"\b(?:" + "|".join(map(re.escape, stopwords.words("english"))) + r")\b"
)
punctuation_pattern = r'[^\p{L}\p{M}\p{N} ]+'
split_pattern = f"({stopwords_pattern})|({punctuation_pattern})"

# %% FUNCTIONS


# %% GET DATA
xkcd_properties_df = db_utils.get_xkcd_properties(db_path)
xkcd_explanations_df = db_utils.get_xkcd_explained(db_path)


# %% CALCULATE TFIDFS AND ADD TO DATABASE
for n_gram_length in tqdm(
    range(1, n_gram_max_length + 1), desc="Calculating tf-idf scores..."
):
    # Titles
    df = xkcd_properties_df
    text_col = "title"
    tfidf_df = get_tfidf(
        df, 
        doc_id_col="xkcd_id", 
        text_col=text_col, 
        n_gram_length=n_gram_length, 
        split_pattern=split_pattern)
    tfidf_df['heading'] = "title"
    print(f"\nFound {len(tfidf_df)} tf-idf scores for titles with n_gram_length {n_gram_length}")
    print(tfidf_df.head())
    db_utils.insert_tfidfs_into_db(db_path, tfidf_df)

    #  Title texts
    df = xkcd_properties_df
    text_col = "title_text"
    tfidf_df = get_tfidf(
        df, 
        doc_id_col="xkcd_id", 
        text_col=text_col, 
        n_gram_length=n_gram_length, 
        split_pattern=split_pattern)
    tfidf_df['heading'] = "title_text"
    print(f"\nFound {len(tfidf_df)} tf-idf scores for title text with n_gram_length {n_gram_length}")
    print(tfidf_df.head())
    db_utils.insert_tfidfs_into_db(db_path, tfidf_df)

    # Transcripts
    df = xkcd_explanations_df[
        xkcd_explanations_df["heading"] == "Transcript"
    ]
    text_col = "text"
    tfidf_df = get_tfidf(
        df, 
        doc_id_col="xkcd_id", 
        text_col=text_col, 
        n_gram_length=n_gram_length, 
        split_pattern=split_pattern)
    tfidf_df['heading'] = "transcript"
    tfidf_df = tfidf_df.drop_duplicates(subset=['xkcd_id', 'token', 'tfidf'], keep=False)
    print(f"\nFound {len(tfidf_df)} tf-idf scores for transcripts with n_gram_length {n_gram_length}")
    print(tfidf_df.head())
    db_utils.insert_tfidfs_into_db(db_path, tfidf_df)

    # Explanation
    df = xkcd_explanations_df[
        xkcd_explanations_df["heading"] == "Explanation"
    ]
    text_col = "text"
    tfidf_df = get_tfidf(
        df, 
        doc_id_col="xkcd_id", 
        text_col=text_col, 
        n_gram_length=n_gram_length, 
        split_pattern=split_pattern)
    tfidf_df['heading'] = "explanation"
    tfidf_df = tfidf_df.drop_duplicates(subset=['xkcd_id', 'token', 'tfidf'], keep=False)
    print(f"\nFound {len(tfidf_df)} tf-idf scores for rest of explanation with n_gram_length {n_gram_length}")
    print(tfidf_df.head())
    db_utils.insert_tfidfs_into_db(db_path, tfidf_df)


# %% CHECK OUT THE RESULTS
tfidf_df = tfidf_df.drop_duplicates(subset=['xkcd_id', 'token', 'tfidf', 'heading'], keep=False)
tfidf_df

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
# df = pd.concat([
#     df_calculated_tfidf, 
#     tfidf_title_text_df])

# df = tfidf_df

duplicates = df[
    df.duplicated(
        subset=['xkcd_id', 'token', 'heading'], 
        keep=False)]
print(duplicates)

# %%
# Dupes in source data
df = xkcd_explanations_df
duplicates = df[
    df.duplicated(
        subset=['xkcd_id', 'heading', 'tag_id', 'text'], 
        keep=False)]
print(duplicates)

# %%
import sqlite3
import unicodedata

def normalize_string(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    return unicodedata.normalize("NFC", s)

key_cols = ['xkcd_id', 'token', 'tfidf']

conn = sqlite3.connect(db_path)
existing = pd.read_sql(f"SELECT {','.join(key_cols)} FROM XKCD_EXPLAINED_TFIDF", conn)
conn.close()

df = tfidf_df.copy()

for c in key_cols:
    df[c] = df[c].apply(normalize_string)
    existing[c] = existing[c].apply(normalize_string)

overlap = df.merge(existing, on=key_cols, how="inner")

print("Conflicting rows (already exist in DB):")
print(overlap)


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
xkcd_exlanations_df_fucky = xkcd_explanations_df[xkcd_explanations_df['xkcd_id'].isin([1])]
xkcd_exlanations_df_fucky

# %%
# Transcripts
df = xkcd_exlanations_df_fucky[
    xkcd_exlanations_df_fucky["heading"] == "Transcript"
]
text_col = "text"
tfidf_df_fucky = get_tfidf(
    df, 
    doc_id_col="xkcd_id", 
    text_col=text_col, 
    n_gram_length=n_gram_length, 
    split_pattern=split_pattern)
tfidf_df_fucky['heading'] = "transcript"
tfidf_df_fucky = tfidf_df_fucky.drop_duplicates(subset=['xkcd_id', 'token', 'tfidf'], keep=False)
print(f"\nFound {len(tfidf_df_fucky)} tf-idf scores for transcripts with n_gram_length {n_gram_length}")
tfidf_df_fucky

# %%
