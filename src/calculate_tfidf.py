# %% HEADER
# Calculate TF-IDF vectors for each heading and store in database

# https://www.learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/#:~:text=Term%20Frequency%3A%20TF%20of%20a,of%20words%20in%20the%20document.&text=Inverse%20Document%20Frequency%3A%20IDF%20of,corpus%20that%20contain%20the%20term.

# This might take quite a while - bigrams took a full day to run

# TODO: Consider removing character names from strings
# TODO: Rework comments to nicer language
# TODO: What if semantics return nothing? Then first try destopworded search in titles and such, then literal words
# TODO: Strip upper case using NER
# TODO: Docstrings
# TODO: Titles Are In This Kind Of Illegible Format - might want to prepreprocess this
# TODO: Sort warnings
# TODO: Investigate resulting tf-idf scores

# %% IMPORTS
import re

from nltk.corpus import stopwords
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
punctuation_pattern = r"[^\p{L}\p{M}\p{N} ]+"
split_pattern = f"({stopwords_pattern})|({punctuation_pattern})"

# %% FUNCTIONS


# %% GET DATA
xkcd_properties_df = db_utils.get_xkcd_properties(db_path)
xkcd_explanations_df = db_utils.get_xkcd_explained(db_path)


# %% CLEAR TABLE
# Manually enable this code to clear the table
if 1 == 1 * 1:
    pass
    # import sqlite3
    # conn = sqlite3.connect(db_path)
    # cursor = conn.cursor()
    # cursor.execute("DELETE FROM XKCD_EXPLAINED_TFIDF")
    # conn.commit()
    # cursor.execute("VACUUM")
    # conn.close()

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
        split_pattern=split_pattern,
    )
    tfidf_df["heading"] = "title"
    print(
        f"\nFound {len(tfidf_df)} tf-idf scores for titles with n_gram_length {n_gram_length}"
    )
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
        split_pattern=split_pattern,
    )
    tfidf_df["heading"] = "title_text"
    print(
        f"\nFound {len(tfidf_df)} tf-idf scores for title text with n_gram_length {n_gram_length}"
    )
    print(tfidf_df.head())
    db_utils.insert_tfidfs_into_db(db_path, tfidf_df)

    # Transcripts
    df = xkcd_explanations_df[xkcd_explanations_df["heading"] == "Transcript"]
    text_col = "text"
    tfidf_df = get_tfidf(
        df,
        doc_id_col="xkcd_id",
        text_col=text_col,
        n_gram_length=n_gram_length,
        split_pattern=split_pattern,
    )
    tfidf_df["heading"] = "transcript"
    print(
        f"\nFound {len(tfidf_df)} tf-idf scores for transcripts with n_gram_length {n_gram_length}"
    )
    print(tfidf_df.head())
    db_utils.insert_tfidfs_into_db(db_path, tfidf_df)

    # Explanation
    df = xkcd_explanations_df[xkcd_explanations_df["heading"] == "Explanation"]
    text_col = "text"
    tfidf_df = get_tfidf(
        df,
        doc_id_col="xkcd_id",
        text_col=text_col,
        n_gram_length=n_gram_length,
        split_pattern=split_pattern,
    )
    tfidf_df["heading"] = "explanation"
    print(
        f"\nFound {len(tfidf_df)} tf-idf scores for rest of explanation with n_gram_length {n_gram_length}"
    )
    print(tfidf_df.head())
    db_utils.insert_tfidfs_into_db(db_path, tfidf_df)
