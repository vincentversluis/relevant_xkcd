# %% HEADER
# Calculate TF-IDF vectors for each heading and store in database.
# This might take quite a while - bigrams took a full day to run.


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


# %% GET DATA
xkcd_properties_df = db_utils.get_xkcd_properties(db_path)
xkcd_explanations_df = db_utils.get_xkcd_explained(db_path)


# %% CALCULATE TFIDFS AND ADD TO DATABASE
# Calculate tf-idf scores for each part of the explanation separately
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
