# %% HEADER
# Functions that recommend xkcds based on a given text, making the assumption that 'there is always a
# relevant xkcd'.

# %% IMPORTS
import gensim.downloader as api
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import re
import string
from itertools import pairwise
from time import sleep
import arrow
from functools import cache
import sqlite3
import pandas as pd
from typing import Optional
from xkcd_tfidfing import get_preprocessed_tokens, get_up_to_n_grams
from utils import db_utils
from utils import utils

# %% INPUTS
db_path = "../data/relevant_xkcd.db"

# n_gram_max_length = 3

# n_gram_weights = {1: 1, 2: 1.5, 3: 2}
# n_gram_weights = "length"

# # Build split pattern with punctuation and word boundaries for stopwords
# stopwords_pattern = (
#     r"\b(?:" + "|".join(map(re.escape, stopwords.words("english"))) + r")\b"
# )
# punctuation_pattern = r"[^\p{L}\p{M}\p{N} ]+"
# split_pattern = f"({stopwords_pattern})|({punctuation_pattern})"

# %% LOAD THINGS
# This is large (~1.5GB file), so download it once and use the cached version
# Loading it also takes a minute or so
word2vec = api.load("word2vec-google-news-300")


# %% FUNCTIONS
@cache
def get_similar_words(word: str, topn: int = 10) -> list[str]:
    """Get a list of similar words for a given word, including the similarity score.
    
    This uses the word2vec model to get the most similar words to the given word. This can handle more
    than just words, it does numbers, ngrams and even some punctuation.

    Args:
        word (str): The word to get similar words for.
        topn (int, optional): The number of similar words to return. Defaults to 10.

    Returns:
        list: The most similar words to the given word.
    """    
    try:
        similar_words = word2vec.most_similar(word, topn=topn)
    except KeyError:  # Word is not in the model
        similar_words = []
    return similar_words


def get_similar_words_for_list(words: list[str], topn: int = 10) -> list[str]:
    """Get a list of similar words for a list of words.

    Args:
        words (list[str]): The words to get similar words for.
        topn (int, optional): The number of similar words to return. Defaults to 10.

    Returns:
        list[str]: The most similar words to the given word.
    """    
    similar_words = []
    for word in words:
        similar_words.append({word: get_similar_words(word, topn=topn)})
    return similar_words


def recommend_xkcd(
    text: str,
    top_n: int = 10,
    n_gram_max_length: int = 3,
    n_gram_weights: dict | None = "length",
    split_pattern: str | None = None,
    heading_weights: dict | None = None,
    number_to_words: bool = True,
    number_to_words_threshold: int = 20,
) -> pd.DataFrame:
    """Recommend xkcds based on a given text.
    
    This uses the tf-idf scores of the text to recommend xkcds. It also takes into account the weights of
    the different parts of the explanation, such as title, transcript, explanation and title text. Several
    knobs and buttons are available to configure the recommendation process.
    
    

    Args:
        text (str): The text to recommend xkcds for.
        top_n (int, optional): The number of xkcds to recommend. Defaults to 10.
        n_gram_max_length (int, optional): The maximum length of n-grams to use. Defaults to 3.
        n_gram_weights (dict | None, optional): The weighting method to use for n-grams. Defaults to "length". Alternatively, a dictionary can be passed with the length of the n-gram as the key and the weight as the value.
        split_pattern (str | None, optional): The split pattern to use for the text. Defaults to English stopwords and punctuation.
        heading_weights (dict | None, optional): The weights of the different parts of the explanation. Defaults to None. Alternatively, a dictionary can be passed with the heading as the key and the weight as the value.
        number_to_words (bool, optional): If numbers should be interpreted as words. Defaults to True.
        number_to_words_threshold (int, optional): The highest number to convert to written out numbers. Defaults to 20.

    Returns:
        pd.DataFrame: The recommended xkcds.
    """    
    if split_pattern is None:
        stopwords_pattern = (
            r"\b(?:" + "|".join(map(re.escape, stopwords.words("english"))) + r")\b"
        )
        punctuation_pattern = r"[^\p{L}\p{M}\p{N} ]+"
        split_pattern = f"({stopwords_pattern})|({punctuation_pattern})"

    # Create heading weight df
    if heading_weights is None:
        heading_weights = {
            "title": 1,  # Usually a description in only a few words
            "title_text": 0.1,  # Often this does not really pertain to the rest
            "transcript": 1,  # What people actually read
            "explanation": 0.5,  # Contains things that touch on the topic
        }
    heading_weights_df = pd.DataFrame.from_dict(
        heading_weights, orient="index"
    ).reset_index()
    heading_weights_df.columns = ["heading", "weight"]

    # Turn numerical words into written out numbers
    if number_to_words:
        text = utils.sub_number_to_words(
            text, numerical_threshold=number_to_words_threshold
        )

    # Get all up to n-grams from text
    tokens = set(
        get_up_to_n_grams(
            text, split_pattern=split_pattern, n_gram_max_length=n_gram_max_length
        )
    )

    similar_words = get_similar_words_for_list(tokens)

    # Create a df with the tokens and similar words
    similarwords_df = pd.DataFrame()
    for similar_word in similar_words:
        # Get current token
        token = list(similar_word.keys())[0]

        # Add entry with similarity of 1 to df
        similarwords_df = pd.concat([
            similarwords_df,
            pd.DataFrame([[token, float(1)]], columns=["token", "similarity"]),
        ])

        # Get similar words from dict and add if not empty
        similarities = pd.DataFrame(
            similar_word[token], columns=["token", "similarity"]
        )
        if len(similarities) > 0:
            similarwords_df = pd.concat([similarwords_df, similarities])

    # Add n-gram weight
    similarwords_df["n_gram_weight"] = similarwords_df["token"].apply(
        lambda x: utils.get_ngram_weight(x, n_gram_weights=n_gram_weights)
    )

    # Get tf-idf scores for all similar words
    similar_words = similarwords_df["token"].unique()
    xkcd_tfidfs = db_utils.get_xkcd_tfidfs(db_path, similar_words)

    # Combine dfs and calculate total token_score per xkcd_id and sort by score
    df = pd.merge(similarwords_df, xkcd_tfidfs, on="token", how="left")
    df = pd.merge(df, heading_weights_df, on="heading", how="left")
    df["token_score"] = (
        df["tfidf"] * df["similarity"] * df["n_gram_weight"] * df["weight"]
    )
    df = df.groupby("xkcd_id")["token_score"].sum().sort_values(ascending=False)

    # Get top results and make presentable
    df = df.head(top_n).reset_index().rename(columns={"token_score": "score"})
    df["score"] = df["score"].round(2)

    # Add some properties
    xkcd_properties_df = db_utils.get_xkcd_properties_for_ids(
        db_path, df["xkcd_id"].unique()
    )
    xkcd_properties_df["link"] = xkcd_properties_df["xkcd_id"].apply(
        lambda x: f"https://www.explainxkcd.com/wiki/index.php/{x}"
    )
    xkcd_properties_df["id_title"] = (
        xkcd_properties_df["xkcd_id"].astype(str) + " - " + xkcd_properties_df["title"]
    )
    df = pd.merge(df, xkcd_properties_df, on="xkcd_id", how="left")

    return df[["score", "id_title", "link"]]
