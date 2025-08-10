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

# %% IMPORTS
from nltk.corpus import stopwords
import re
import string
import pandas as pd
from utils import db_utils
from tqdm import tqdm
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
punctuation_pattern = "|".join(map(re.escape, string.punctuation))
split_pattern = f"({stopwords_pattern})|({punctuation_pattern})"

# %% GET DATA
xkcd_properties_df = db_utils.get_xkcd_properties(db_path)
xkcd_explanations_df = db_utils.get_xkcd_explained(db_path)

# %% ANALYSE DATA
for n_gram_length in tqdm(
    range(1, n_gram_max_length + 1), desc="Calculating tf-idf scores..."
):
    # Get tfidf for titles
    tfidf_title_df = xkcd_tfidfing.get_tfidf(
        xkcd_properties_df, "xkcd_id", "title", n_gram_length=n_gram_length
    )
    tfidf_title_df["heading"] = "title"

    # Get tfidf for title texts
    tfidf_title_text_df = xkcd_tfidfing.get_tfidf(
        xkcd_properties_df, "xkcd_id", "title_text", n_gram_length=n_gram_length
    )
    tfidf_title_text_df["heading"] = "title_text"

    # Get tfidf for transcripts
    xkcd_transcripts_df = xkcd_explanations_df[
        xkcd_explanations_df["heading"] == "Transcript"
    ]
    xkcd_transcripts_df = xkcd_transcripts_df.groupby(
        ["xkcd_id", "heading"], as_index=False
    ).agg({"text": lambda x: " ".join(x)})
    tfidf_transcripts_df = xkcd_tfidfing.get_tfidf(
        xkcd_transcripts_df, "xkcd_id", "text", n_gram_length=n_gram_length
    )
    tfidf_transcripts_df["heading"] = "transcript"

    # Get tfidf for rest of explanation, minus headings to ignore
    xkcd_rest_of_explanations_df = xkcd_explanations_df[
        ~xkcd_explanations_df["heading"].isin(headings_ignore)
    ]
    xkcd_rest_of_explanations_df = xkcd_rest_of_explanations_df.groupby(
        ["xkcd_id", "heading"], as_index=False
    ).agg({"text": lambda x: " ".join(x)})
    tfidf_rest_of_explanations_df = xkcd_tfidfing.get_tfidf(
        xkcd_rest_of_explanations_df, "xkcd_id", "text", n_gram_length=n_gram_length
    )

    tfidf_df = pd.concat([
        tfidf_title_df,
        tfidf_title_text_df,
        tfidf_transcripts_df,
        tfidf_rest_of_explanations_df,
    ])

    # Add tf-idf scores to the database
    # db_utils.insert_tfidfs_into_db(db_path, tfidf_df)
    print(tfidf_df.head())

# %% CHECK OUT THE RESULTS
# TODO: Check out some results
# Number of tf_idfs per heading
# Are all xkcd with tfidfs
